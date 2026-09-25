"""Unit tests for the QSA MTP sparse-indices capture path — no server, no GPU.

Guards the Tier-0 sync-free rewrite of ``capture_mtp_sparse_indices``:
the fixed-shape scatter_reduce anchors must stay byte-identical to the
original ``nonzero``-based implementation, and the eager branch must not
reintroduce host-syncing ops (``nonzero``/``bincount``/``cumsum`` block on
in-flight GPU work; the whole point of the rewrite is that they are gone).
"""

import inspect
import random
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LAYER_IDS = [0, 3]
NUM_REQ = 8
TOPK = 6
TAIL = 4
NUM_CASES = 100


class _Mode:
    def is_draft_extend_v2(self):
        return False


class _FB:
    forward_mode = _Mode()

    def __init__(self, req_pool_indices=None, extend_seq_lens=None):
        if req_pool_indices is not None:
            self.req_pool_indices = req_pool_indices
        if extend_seq_lens is not None:
            self.extend_seq_lens = extend_seq_lens


class _Meta:
    def __init__(self, token_to_batch_idx, req_pool_indices, seqlens):
        self.token_to_batch_idx = token_to_batch_idx
        self.req_pool_indices = req_pool_indices
        self._seqlens = seqlens
        self.is_cuda_graph = False

    def get_token_to_batch_idx(self):
        return self.token_to_batch_idx

    def get_seqlens_expanded(self):
        return self._seqlens.to(torch.int32).index_select(
            0, self.token_to_batch_idx.long()
        )


def _reference_capture(state, topk_indices, meta, layer_id):
    """Pre-rewrite implementation, verbatim: is_last mask + nonzero anchors."""
    row_to_req = meta.get_token_to_batch_idx().long()
    row_req_pool_indices = meta.req_pool_indices[
        row_to_req[: topk_indices.shape[0]]
    ]
    is_last = torch.ones_like(row_req_pool_indices, dtype=torch.bool)
    if row_req_pool_indices.numel() > 1:
        is_last[:-1] = row_req_pool_indices[:-1] != row_req_pool_indices[1:]
    anchor_rows = is_last.nonzero().flatten()
    req_rows = row_req_pool_indices[anchor_rows]
    captured_lens = meta.get_seqlens_expanded()[anchor_rows]
    state.capture(topk_indices[anchor_rows], req_rows, captured_lens, layer_id)


def _make_case(rng):
    bs = rng.randint(1, 6)
    lens = [rng.randint(1, 7) for _ in range(bs)]
    if rng.random() < 0.3 and bs >= 2:
        lens[rng.randrange(1, bs)] = 0  # request with no rows in this forward
    t2b = torch.repeat_interleave(
        torch.arange(bs), torch.tensor(lens, dtype=torch.long)
    )
    req_pool_indices = torch.tensor(
        rng.sample(range(1, NUM_REQ), bs), dtype=torch.int64
    )
    seqlens = torch.tensor(
        [rng.randint(1, 4096) for _ in range(bs)], dtype=torch.int64
    )
    rows = t2b.numel()
    topk_rows = (
        rows if rng.random() < 0.8 else max(1, rows - rng.randint(0, rows - 1))
    )  # truncated topk window
    topk = torch.randint(0, 2048, (max(topk_rows, 1), TOPK), dtype=torch.int32)
    return _Meta(t2b, req_pool_indices, seqlens), topk[:topk_rows]


class TestQSAMTPCaptureBitIdentity(CustomTestCase):
    def _states(self, backend_cls, state_cls):
        backend = backend_cls(runner=None)
        new_state = state_cls(
            layer_ids=LAYER_IDS,
            num_requests=NUM_REQ,
            token_topk=TOPK,
            tail_width=TAIL,
            device="cpu",
        )
        ref_state = state_cls(
            layer_ids=LAYER_IDS,
            num_requests=NUM_REQ,
            token_topk=TOPK,
            tail_width=TAIL,
            device="cpu",
        )
        backend._mtp_shared_sparse_indices = new_state
        return backend, new_state, ref_state

    def test_capture_matches_reference_over_random_cases(self):
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QSAMTPSharedSparseIndices,
            QwenSparseAttnBackend,
        )

        rng = random.Random(20260925)
        for case in range(NUM_CASES):
            backend, new_state, ref_state = self._states(
                QwenSparseAttnBackend, QSAMTPSharedSparseIndices
            )
            for layer_id in LAYER_IDS:
                meta, topk = _make_case(rng)
                _reference_capture(ref_state, topk, meta, layer_id)
                backend.capture_mtp_sparse_indices(
                    topk, _FB(), layer_id, metadata=meta
                )
            # Trash row (num_requests) may legitimately differ between the
            # implementations; every real request row must be byte-equal.
            self.assertTrue(
                torch.equal(
                    new_state.indices[:, :NUM_REQ], ref_state.indices[:, :NUM_REQ]
                ),
                f"case {case}: captured indices differ",
            )
            self.assertTrue(
                torch.equal(
                    new_state.captured_len[:, :NUM_REQ],
                    ref_state.captured_len[:, :NUM_REQ],
                ),
                f"case {case}: captured lens differ",
            )

    def test_eager_capture_path_stays_sync_free(self):
        from sglang.srt.layers.attention import qwen_sparse_attn_backend as m

        src = "\n".join(
            line.split("#", 1)[0]
            for line in inspect.getsource(
                m.QwenSparseAttnBackend.capture_mtp_sparse_indices
            ).splitlines()
        )
        for syncing_op in ("nonzero", "bincount", "cumsum", "tolist", ".item("):
            self.assertNotIn(
                syncing_op,
                src,
                f"capture path reintroduced a host-syncing op: {syncing_op}",
            )

    def test_uniform_row_to_request_mapping_and_cache(self):
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseAttnBackend,
        )

        for bs, rows_per_req in [(1, 4), (2, 4), (3, 5), (4, 1), (5, 4)]:
            fb = _FB(req_pool_indices=torch.arange(bs, dtype=torch.int64))
            got = QwenSparseAttnBackend._speculative_row_to_request(
                fb, bs * rows_per_req
            )
            want = torch.arange(bs, dtype=torch.long).repeat_interleave(
                rows_per_req
            )
            self.assertTrue(
                torch.equal(got, want),
                f"mapping mismatch for bs={bs} rows_per_req={rows_per_req}",
            )
            # Second identical lookup must hit the cache and stay equal.
            again = QwenSparseAttnBackend._speculative_row_to_request(
                fb, bs * rows_per_req
            )
            self.assertTrue(torch.equal(again, want))


if __name__ == "__main__":
    unittest.main(verbosity=3)
