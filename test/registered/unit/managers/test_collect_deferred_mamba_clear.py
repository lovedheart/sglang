"""Host-mirror ping-pong selection must equal the device mask-index result.

``_collect_deferred_mamba_cow_and_clear`` picks the freshly recycled ping-pong
slots from the host empty-mirror instead of the device buffer's ``buf != -1``
mask index (a data-dependent-shape read that synchronized once per request
per batch prep). The selected rows must be exactly what the mask index
produced -- same values, ascending slot order, per-request order -- and must
snapshot the values at collection time, since later slot writes must not
leak into a clear staged for an earlier forward.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(buf, needs_clear=True, clear_queue=None):
    return SimpleNamespace(
        kv=SimpleNamespace(
            mamba_ping_pong_needs_clear=needs_clear,
            mamba_ping_pong_track_buffer=buf,
            mamba_ping_pong_empty=[v == -1 for v in buf.tolist()],
            mamba_slot_clear_queue=clear_queue,
            mamba_cow_src_index=None,
            mamba_needs_clear=False,
            mamba_pool_idx=torch.tensor(3),
        )
    )


def _collect(reqs):
    self_like = SimpleNamespace(
        mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
        mamba_clear_indices=None,
    )
    ScheduleBatch._collect_deferred_mamba_cow_and_clear(self_like, reqs)
    return self_like


class TestCollectDeferredMambaClear(unittest.TestCase):
    def test_selection_matches_mask_index_all_patterns(self):
        # Every occupancy pattern of a size-2 pair, plus a cleared request
        # and a staged clear queue, in one batch: the concatenated result
        # must equal the per-request mask-index oracle.
        bufs = [
            torch.tensor([41, -1]),
            torch.tensor([-1, 42]),
            torch.tensor([43, 44]),
            torch.tensor([-1, -1]),
        ]
        reqs = [_req(b) for b in bufs]
        reqs[2].kv.mamba_ping_pong_needs_clear = False  # not collected
        staged = torch.tensor([77])
        reqs[3].kv.mamba_ping_pong_needs_clear = True
        reqs[3].kv.mamba_slot_clear_queue = [staged]
        out = _collect(reqs)
        oracle = torch.cat(
            [buf[buf != -1] for buf in [bufs[0], bufs[1], bufs[3]]] + [staged]
        )
        # reqs[2] never enters the ping-pong branch; its (already clear)
        # flag and untouched buffer prove the gate is the flag, not -1s.
        self.assertFalse(reqs[2].kv.mamba_ping_pong_needs_clear)
        for r in (reqs[0], reqs[1], reqs[3]):
            self.assertFalse(r.kv.mamba_ping_pong_needs_clear)
            self.assertIsNone(r.kv.mamba_slot_clear_queue)
        self.assertTrue(torch.equal(out.mamba_clear_indices, oracle))

    def test_staged_clear_snapshots_slot_values(self):
        # Mask-index materialized; views into the request's buffer row must
        # not escape the call either. A slot written after collection (e.g.
        # the next prepare_for_decode on the overlap path) must not change
        # an already-staged clear.
        buf = torch.tensor([51, -1])
        req = _req(buf)
        out = _collect([req])
        # Simulate set_mamba_ping_pong_slot(req, 0, new_slot) afterwards.
        buf[0] = 99
        req.kv.mamba_ping_pong_empty[0] = False
        self.assertTrue(torch.equal(out.mamba_clear_indices, torch.tensor([51])))


if __name__ == "__main__":
    unittest.main()
