"""QSA K/V host offload: layout, pool sizing and the reference resolve.

The resolve test drives ``QSAKVHostOffload`` the way the QSA backend does
(extend, then decode or chain-verify forwards that store their new K/V and
resolve the selected positions) and checks after every forward that each
selected position's device row holds exactly the K/V of the logical slot
``req_to_token`` maps it to, with the hot cache far smaller than the contexts,
radix remaps of resident blocks and rejected speculative tokens.
"""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.qsa_kv_offload_hook import check_qsa_kv_offload_args
from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.kv_offload_attention import (
    _gather_prefill_context_slab,
)
from sglang.srt.mem_cache.qsa_kv_offload import (
    QSAKVHostOffload,
    QSAKVOffloadBudget,
    QSAKVOffloadLayout,
    compute_offload_layout,
    offload_table_bytes_per_slot,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

CR = 4


class TestOffloadLayout(CustomTestCase):
    def test_regions_scratch_and_page_alignment(self):
        layout = compute_offload_layout(
            compress_ratio=CR,
            token_topk=2048,
            hot_blocks=4096,
            num_request_slots=65,
            rows_per_request=1,
            max_extend_tokens=8192,
            page_size=64,
        )
        self.assertEqual(layout.region_rows, (4096 + 2) * CR)
        self.assertEqual(layout.scratch_base, 65 * (4096 + 2) * CR)
        self.assertGreaterEqual(layout.scratch_rows, 8192 + 2 * 64)
        self.assertEqual(layout.device_rows % 64, 0)

    def test_refuses_hot_cache_below_one_selection(self):
        with self.assertRaisesRegex(ValueError, "below 1536"):
            compute_offload_layout(
                compress_ratio=CR,
                token_topk=2048,
                hot_blocks=1535,
                num_request_slots=9,
                rows_per_request=3,
                max_extend_tokens=1024,
                page_size=64,
            )

    def test_refuses_more_rows_than_compress_ratio(self):
        with self.assertRaisesRegex(ValueError, "query rows per request"):
            compute_offload_layout(
                compress_ratio=CR,
                token_topk=2048,
                hot_blocks=4096,
                num_request_slots=9,
                rows_per_request=CR + 1,
                max_extend_tokens=1024,
                page_size=64,
            )


def _budget(**overrides):
    values = dict(
        offloaded_bytes_per_token=1000,
        region_rows=100,
        fixed_rows=64,
        table_bytes_per_slot=5000,
        host_size_gb=1000,
        page_size=64,
    )
    values.update(overrides)
    return QSAKVOffloadBudget(**values)


# Per request slot: 100 rows x 1000 B + 5000 B of tables.
_SLOT_BYTES = 100 * 1000 + 5000


class TestOffloadSizing(CustomTestCase):
    def test_gpu_bound(self):
        # 1100 - 1000 = 100 B per logical token stay on the GPU.
        tokens = _budget().max_total_tokens(
            available_bytes=10**9, cell_bytes=1100, num_request_slots=9
        )
        fixed = 9 * _SLOT_BYTES + 64 * 1000
        self.assertEqual(tokens, (10**9 - fixed) // 100 // 64 * 64)

    def test_host_bound(self):
        tokens = _budget(host_size_gb=1).max_total_tokens(
            available_bytes=10**12, cell_bytes=1100, num_request_slots=9
        )
        self.assertEqual(tokens, (10**9 // 1000 - 64) // 64 * 64)

    def test_hot_cache_larger_than_budget(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            _budget().max_total_tokens(
                available_bytes=10**6, cell_bytes=1100, num_request_slots=9
            )

    def test_verify_mode_keeps_the_full_cell(self):
        tokens = _budget(verify=True).max_total_tokens(
            available_bytes=10**9, cell_bytes=1100, num_request_slots=9
        )
        fixed = 9 * _SLOT_BYTES + 64 * 1000
        self.assertEqual(tokens, (10**9 - fixed) // 1100 // 64 * 64)

    def test_unused_slots_hold_tokens(self):
        # Sized for 33 request slots, but the request pool only gets 12.
        budget = _budget()
        sized = budget.max_total_tokens(
            available_bytes=10**9, cell_bytes=1100, num_request_slots=33
        )
        grown = budget.tokens_with_unused_slots(
            max_total_tokens=sized, cell_bytes=1100, unused_slots=21
        )
        direct = budget.max_total_tokens(
            available_bytes=10**9, cell_bytes=1100, num_request_slots=12
        )
        self.assertGreater(grown, sized)
        self.assertLessEqual(abs(grown - direct), 64)
        capped = _budget(host_size_gb=0.1).tokens_with_unused_slots(
            max_total_tokens=sized, cell_bytes=1100, unused_slots=21
        )
        self.assertEqual(capped, (10**8 // 1000 - 64) // 64 * 64)

    def test_table_bytes_grow_with_context_and_hot_blocks(self):
        def table(hot, ctx):
            return offload_table_bytes_per_slot(
                num_layers=12, hot_blocks=hot, compress_ratio=CR, context_len=ctx
            )

        self.assertGreater(table(512, 262144), table(512, 8192))
        self.assertGreater(table(4096, 8192), table(512, 8192))


_QSA_HF_CONFIG = SimpleNamespace(
    indexer_n_heads=4,
    indexer_kv_heads=1,
    indexer_head_dim=128,
    indexer_budget=2048,
    indexer_compress_ratio=CR,
)


def _offload_args(**overrides):
    graphs = SimpleNamespace(
        decode=SimpleNamespace(backend=Backend.DISABLED),
        prefill=SimpleNamespace(backend=Backend.DISABLED),
    )
    values = dict(
        qsa_kv_offload_host_size=40,
        qsa_kv_offload_hot_blocks=1024,
        max_running_requests=8,
        chunked_prefill_size=4096,
        max_prefill_tokens=16384,
        page_size=64,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_lmcache=False,
        enable_mixed_chunk=False,
        enable_dp_attention=False,
        enable_prefill_cp=False,
        attn_cp_size=1,
        dcp_size=1,
        disaggregation_mode="null",
        pp_size=1,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
        cuda_graph_config=graphs,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestOffloadArgs(CustomTestCase):
    def test_accepts_a_supported_config(self):
        check_qsa_kv_offload_args(_offload_args(), _QSA_HF_CONFIG)
        decode_graphs = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL),
            prefill=SimpleNamespace(backend=Backend.DISABLED),
        )
        check_qsa_kv_offload_args(
            _offload_args(cuda_graph_config=decode_graphs), _QSA_HF_CONFIG
        )
        with envs.SGLANG_USE_QSA_KV_OFFLOAD_TORCH_RESOLVE.override(True):
            with self.assertRaisesRegex(ValueError, "disable-cuda-graph"):
                check_qsa_kv_offload_args(
                    _offload_args(cuda_graph_config=decode_graphs), _QSA_HF_CONFIG
                )
        check_qsa_kv_offload_args(
            _offload_args(
                speculative_algorithm="NEXTN",
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=4,
                qsa_kv_offload_hot_blocks=2048,
            ),
            _QSA_HF_CONFIG,
        )

    def test_rocm_needs_eager_decode(self):
        # ROCm runs the eager torch resolve, so decode CUDA graphs are refused
        # before the weights load.
        decode_graphs = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL),
            prefill=SimpleNamespace(backend=Backend.DISABLED),
        )
        with patch("sglang.srt.utils.is_hip", return_value=True):
            check_qsa_kv_offload_args(_offload_args(), _QSA_HF_CONFIG)
            with self.assertRaisesRegex(ValueError, "ROCm.*disable-cuda-graph"):
                check_qsa_kv_offload_args(
                    _offload_args(cuda_graph_config=decode_graphs), _QSA_HF_CONFIG
                )

    def test_refusals(self):
        cases = {
            "compressed QSA": (dict(), SimpleNamespace()),
            "host-size": (dict(qsa_kv_offload_host_size=0), _QSA_HF_CONFIG),
            "max-running-requests": (dict(max_running_requests=None), _QSA_HF_CONFIG),
            "chunked prefill": (dict(chunked_prefill_size=-1), _QSA_HF_CONFIG),
            "hierarchical-cache": (
                dict(enable_hierarchical_cache=True),
                _QSA_HF_CONFIG,
            ),
            "hisparse": (dict(enable_hisparse=True), _QSA_HF_CONFIG),
            "lmcache": (dict(enable_lmcache=True), _QSA_HF_CONFIG),
            "mixed-chunk": (dict(enable_mixed_chunk=True), _QSA_HF_CONFIG),
            "dp-attention": (dict(enable_dp_attention=True), _QSA_HF_CONFIG),
            "prefill-cp": (dict(enable_prefill_cp=True), _QSA_HF_CONFIG),
            "attn-cp-size": (dict(attn_cp_size=2), _QSA_HF_CONFIG),
            "dcp-size": (dict(dcp_size=2), _QSA_HF_CONFIG),
            "disaggregation": (dict(disaggregation_mode="decode"), _QSA_HF_CONFIG),
            "pp-size": (dict(pp_size=2), _QSA_HF_CONFIG),
            "NGRAM": (dict(speculative_algorithm="NGRAM"), _QSA_HF_CONFIG),
            "eagle-topk": (
                dict(speculative_algorithm="EAGLE", speculative_eagle_topk=4),
                _QSA_HF_CONFIG,
            ),
            "disable-prefill-cuda-graph": (
                dict(
                    cuda_graph_config=SimpleNamespace(
                        decode=SimpleNamespace(backend=Backend.FULL),
                        prefill=SimpleNamespace(backend=Backend.FULL),
                    )
                ),
                _QSA_HF_CONFIG,
            ),
            "is below 2048": (
                dict(
                    speculative_algorithm="NEXTN",
                    speculative_eagle_topk=1,
                    speculative_num_draft_tokens=4,
                ),
                _QSA_HF_CONFIG,
            ),
        }
        for message, (overrides, hf_config) in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                check_qsa_kv_offload_args(_offload_args(**overrides), hf_config)


class _OffloadSim:
    """Logical K/V pool plus an offload, driven like the QSA backend drives it."""

    def __init__(
        self,
        *,
        rows_per_request,
        seed,
        device="cpu",
        pin=False,
        hot_blocks=None,
        quantized=False,
    ):
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.device = device
        self.layers, self.heads, self.dim = 2, 2, 8
        self.sf_dim = self.dim // 4  # per-block scale plane, like FP4's /16
        self.block_topk = 16
        self.q = rows_per_request
        self.slots = 6  # request slot 0 is padding
        self.width = 1024
        self.pool_size = 8192
        self.layout = compute_offload_layout(
            compress_ratio=CR,
            token_topk=self.block_topk * CR,
            hot_blocks=self.q * self.block_topk + 3,
            num_request_slots=self.slots,
            rows_per_request=self.q,
            max_extend_tokens=self.width,
            page_size=64,
        )
        if hot_blocks is not None:  # below the minimum compute_offload_layout allows
            self.layout = QSAKVOffloadLayout(
                compress_ratio=CR,
                hot_blocks=hot_blocks,
                num_request_slots=self.slots,
                scratch_rows=self.layout.scratch_rows,
                token_topk=self.block_topk * CR,
                rows_per_request=self.q,
            )
        shape = (self.layout.device_rows + 64, self.heads, self.dim)
        self.dev_k = [torch.zeros(shape, device=device) for _ in range(self.layers)]
        self.dev_v = [torch.zeros(shape, device=device) for _ in range(self.layers)]
        sf_shape = (self.layout.device_rows + 64, self.heads, self.sf_dim)
        sf = (
            {
                "k_sf_buffers": [
                    torch.zeros(sf_shape, device=device) for _ in range(self.layers)
                ],
                "v_sf_buffers": [
                    torch.zeros(sf_shape, device=device) for _ in range(self.layers)
                ],
            }
            if quantized
            else {}
        )
        self.offload = QSAKVHostOffload(
            layout=self.layout,
            k_buffers=self.dev_k,
            v_buffers=self.dev_v,
            host_rows=self.pool_size + 64,
            pin_memory=pin,
            **sf,
        )
        ref_shape = (self.layers, self.pool_size + 64, self.heads, self.dim)
        self.ref_k = torch.zeros(ref_shape, device=device)
        self.ref_v = torch.zeros(ref_shape, device=device)
        if quantized:
            ref_sf_shape = (
                self.layers,
                self.pool_size + 64,
                self.heads,
                self.sf_dim,
            )
            self.ref_ksf = torch.zeros(ref_sf_shape, device=device)
            self.ref_vsf = torch.zeros(ref_sf_shape, device=device)
        self.req_to_token = torch.zeros(
            (self.slots, self.width), dtype=torch.int32, device=device
        )
        self.free = list(range(64, self.pool_size + 64))
        self.rng.shuffle(self.free)
        self.lengths = {}

    def _alloc(self, n):
        out, self.free = self.free[:n], self.free[n:]
        return torch.tensor(out, dtype=torch.int64, device=self.device)

    def _write(self, slots, rows):
        """New K/V of ``slots``: device rows, write-through, reference pool."""
        for layer in range(self.layers):
            k = torch.randn(slots.numel(), self.heads, self.dim, device=self.device)
            v = torch.randn_like(k)
            self.dev_k[layer][rows] = k
            self.dev_v[layer][rows] = v
            if self.offload.quantized:
                ksf = torch.randn(
                    slots.numel(), self.heads, self.sf_dim, device=self.device
                )
                vsf = torch.randn_like(ksf)
                self.offload.dev_ksf[layer][rows] = ksf
                self.offload.dev_vsf[layer][rows] = vsf
                self.ref_ksf[layer][slots] = ksf
                self.ref_vsf[layer][slots] = vsf
            self.offload.write_through(layer, rows, slots)
            self.ref_k[layer][slots] = k
            self.ref_v[layer][slots] = v

    def extend(self, req, length):
        slots = self._alloc(length)
        self.req_to_token[req, :length] = slots.to(torch.int32)
        self.lengths[req] = length
        self.offload.begin_forward(
            self.req_to_token,
            reset_req_pool_indices=torch.tensor([req], device=self.device),
        )
        self._write(slots, self.offload.extend_rows(length))

    def remap(self, req):
        """Radix dedup: map complete blocks before the ring to other slots."""
        first_block = self.lengths[req] // CR
        if first_block < 2:
            return
        b = self.rng.randrange(first_block - 1)
        cols = torch.arange(b * CR, (b + 1) * CR, device=self.device)
        slots = self._alloc(CR)
        self.req_to_token[req, cols] = slots.to(torch.int32)
        for layer in range(self.layers):
            self.ref_k[layer][slots] = torch.randn_like(self.ref_k[layer][slots])
            self.ref_v[layer][slots] = torch.randn_like(self.ref_v[layer][slots])
            self.offload.host_k[layer][slots.cpu()] = self.ref_k[layer][slots].cpu()
            self.offload.host_v[layer][slots.cpu()] = self.ref_v[layer][slots].cpu()
            if self.offload.quantized:
                self.ref_ksf[layer][slots] = torch.randn_like(
                    self.ref_ksf[layer][slots]
                )
                self.ref_vsf[layer][slots] = torch.randn_like(
                    self.ref_vsf[layer][slots]
                )
                self.offload.host_ksf[layer][slots.cpu()] = self.ref_ksf[layer][
                    slots
                ].cpu()
                self.offload.host_vsf[layer][slots.cpu()] = self.ref_vsf[layer][
                    slots
                ].cpu()

    def positions(self, qpos):
        """Expanded QSA selection of one query row (valid positions first)."""
        out = [-1] * (self.block_topk * CR + CR - 1)
        tail_start = (qpos + 1) // CR * CR
        blocks = self.rng.sample(
            range(tail_start // CR), min(self.block_topk, tail_start // CR)
        )
        chosen = [b * CR + i for b in blocks for i in range(CR)]
        chosen += list(range(tail_start, qpos + 1))
        out[: len(chosen)] = chosen
        return out

    def step(self):
        """One decode / chain-verify forward over every request."""
        reqs = sorted(self.lengths)
        row_req, row_seq, positions = [], [], []
        new_slots = []
        for req in reqs:
            first = self.lengths[req]
            slots = self._alloc(self.q)
            self.req_to_token[req, first : first + self.q] = slots.to(torch.int32)
            new_slots.append(slots)
            for i in range(self.q):
                row_req.append(req)
                row_seq.append(first + i + 1)
                positions.append(self.positions(first + i))
        row_req = torch.tensor(row_req, device=self.device)
        row_seq = torch.tensor(row_seq, dtype=torch.int32, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int32, device=self.device)
        self.offload.begin_forward(self.req_to_token)
        self._write(torch.cat(new_slots), self.offload.new_token_rows(row_seq, row_req))
        for layer in range(self.layers):
            rows = self.offload.resolve_rows(layer, positions, row_seq, row_req, self.q)
            self._check(layer, rows, positions, row_seq, row_req)
        for req in reqs:  # accept a prefix of the verify rows; free the rest
            accepted = self.rng.randrange(1, self.q + 1)
            first = self.lengths[req]
            rejected = self.req_to_token[req, first + accepted : first + self.q]
            self.free += rejected.tolist()
            self.lengths[req] = first + accepted

    def _check(self, layer, rows, positions, row_seq, row_req):
        pos = positions.long()
        valid = (pos >= 0) & (pos < row_seq[:, None].long())
        assert torch.equal(rows >= 0, valid)
        slots = self.req_to_token[row_req[:, None], pos.clamp(min=0)].long()
        got_k = self.dev_k[layer][rows.long()[valid]]
        got_v = self.dev_v[layer][rows.long()[valid]]
        assert torch.equal(got_k, self.ref_k[layer][slots[valid]])
        assert torch.equal(got_v, self.ref_v[layer][slots[valid]])
        if self.offload.quantized:
            # The scale planes must have travelled with the data rows.
            got_ksf = self.offload.dev_ksf[layer][rows.long()[valid]]
            got_vsf = self.offload.dev_vsf[layer][rows.long()[valid]]
            assert torch.equal(got_ksf, self.ref_ksf[layer][slots[valid]])
            assert torch.equal(got_vsf, self.ref_vsf[layer][slots[valid]])


class TestOffloadResolve(CustomTestCase):
    def _run(self, rows_per_request, seed, quantized=False):
        sim = _OffloadSim(
            rows_per_request=rows_per_request, seed=seed, quantized=quantized
        )
        for req in range(1, sim.slots):
            sim.extend(req, sim.rng.randrange(200, 400))
        for step in range(60):
            if step % 7 == 3:
                sim.remap(sim.rng.randrange(1, sim.slots))
            if step == 30:  # a request slot is reused by a new request
                sim.extend(2, sim.rng.randrange(200, 400))
            sim.step()

    def test_decode(self):
        self._run(rows_per_request=1, seed=1)

    def test_quantized_packed_rows_travel_with_scale_planes(self):
        self._run(rows_per_request=1, seed=7, quantized=True)

    def test_chain_verify_with_rejections(self):
        for q in (2, 3, CR):
            with self.subTest(rows_per_request=q):
                self._run(rows_per_request=q, seed=10 + q)

    def test_verify_check(self):
        sim = _OffloadSim(rows_per_request=1, seed=5)
        ref = torch.zeros(4)
        sim.offload.verify = True
        sim.offload._verify_counts = torch.zeros(2, dtype=torch.int64)
        sim.offload.check("gather", 0, (ref.clone(),), (ref,))
        sim.offload.poll_verify()
        bad = ref.clone()
        bad[1] = 1
        sim.offload.check("gather", 0, (bad,), (ref,))
        with self.assertRaisesRegex(RuntimeError, "1 of 2 checks differ"):
            sim.offload.poll_verify()

    def test_verify_reads_one_mapping_snapshot_per_forward(self):
        sim = _OffloadSim(rows_per_request=1, seed=6)
        layout, live = sim.layout, torch.arange(sim.slots * 8).reshape(sim.slots, 8)
        offload = QSAKVHostOffload(
            layout=layout,
            k_buffers=sim.dev_k,
            v_buffers=sim.dev_v,
            host_rows=64,
            pin_memory=False,
            verify=True,
        )
        offload.bind(live)
        offload.snapshot_mapping()
        live[1, 2] = -5  # the scheduler rewrites a mapping mid-forward
        self.assertEqual(int(offload.req_to_token[1, 2]), 10)
        offload.snapshot_mapping()  # the next forward sees it
        self.assertEqual(int(offload.req_to_token[1, 2]), -5)

    def test_selection_larger_than_hot_cache_raises(self):
        sim = _OffloadSim(rows_per_request=1, seed=3, hot_blocks=4)
        sim.extend(1, 300)
        with self.assertRaisesRegex(RuntimeError, "cannot hold"):
            sim.step()


def _extend_more(sim, req, extra):
    """One more prefill chunk for ``req``: fresh slots stored into the
    scratch window, without touching the earlier chunks' rows."""
    prev = sim.lengths[req]
    slots = sim._alloc(extra)
    sim.req_to_token[req, prev : prev + extra] = slots.to(torch.int32)
    sim.offload.begin_forward(
        sim.req_to_token, reset_req_pool_indices=torch.tensor([req], device=sim.device)
    )
    sim._write(slots, sim.offload.extend_rows(extra))
    sim.lengths[req] = prev + extra


def _fb(reqs, seqs, exts, device):
    return SimpleNamespace(
        req_pool_indices=torch.tensor(reqs, dtype=torch.int64, device=device),
        seq_lens_cpu=seqs,
        extend_seq_lens_cpu=exts,
    )


class TestPrefillSlab(CustomTestCase):
    """The chunked-prefill slab gather serves the same bytes as the all-host
    gather while the cached prefix comes from the device without the host."""

    def _gather_both(self, sim, fb, slots, layer=0):
        offload = sim.offload
        k, v, ksf, vsf = _gather_prefill_context_slab(
            offload=offload, layer_idx=layer, slots=slots, forward_batch=fb
        )
        hk, hv, hksf, hvsf = offload.gather_context(layer_idx=layer, slots=slots)
        self.assertTrue(torch.equal(k, hk))
        self.assertTrue(torch.equal(v, hv))
        self.assertEqual(ksf is None, hksf is None)
        if ksf is not None:
            self.assertTrue(torch.equal(ksf, hksf))
            self.assertTrue(torch.equal(vsf, hvsf))
        return k, v, ksf, vsf

    def test_chunks_match_host_gather_and_serve_from_device(self):
        for quantized in (False, True):
            with (
                self.subTest(quantized=quantized),
                envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.override(8192),
            ):
                sim = _OffloadSim(rows_per_request=1, seed=11, quantized=quantized)
                sim.extend(1, 64)
                seq = 64
                slots = sim.req_to_token[1, :seq].long()
                self._gather_both(
                    sim, _fb([1], [seq], [seq], sim.device), slots
                )
                # Chunk 2 with the host store zeroed: the cache in slabsd prefix
                # must still come out right, proving it was served on the
                # device. The chunk's own rows come from the scratch.
                _extend_more(sim, 1, 48)
                for plane in (sim.offload.host_k, sim.offload.host_v):
                    plane.zero_()
                if quantized:
                    sim.offload.host_ksf.zero_()
                    sim.offload.host_vsf.zero_()
                seq = 112
                slots = sim.req_to_token[1, :seq].long()
                k, v, ksf, vsf = _gather_prefill_context_slab(
                    offload=sim.offload,
                    layer_idx=0,
                    slots=slots,
                    forward_batch=_fb([1], [seq], [48], sim.device),
                )
                # The whole context (slab-cached prefix + scratch chunk)
                # matches the reference pool although the host rows are
                # zeroed: the prefix was served from the device.
                self.assertTrue(torch.equal(k, sim.ref_k[0][slots]))
                self.assertTrue(torch.equal(v, sim.ref_v[0][slots]))
                if quantized:
                    self.assertTrue(
                        torch.equal(ksf, sim.ref_ksf[0][slots])
                    )
                    self.assertTrue(
                        torch.equal(vsf, sim.ref_vsf[0][slots])
                    )
                slab = sim.offload._slabs[1]
                # Only layer 0 was gathered; layers advance independently.
                self.assertEqual(slab.cursors[0], 112)
                # Chunk 3: chunk 2's append must have staged the *right*
                # rows (an append off by `cached` only shows up one chunk
                # later), still with the host store zeroed.
                _extend_more(sim, 1, 32)
                seq = 144
                slots = sim.req_to_token[1, :seq].long()
                k, v, ksf, vsf = _gather_prefill_context_slab(
                    offload=sim.offload,
                    layer_idx=0,
                    slots=slots,
                    forward_batch=_fb([1], [seq], [32], sim.device),
                )
                self.assertTrue(torch.equal(k, sim.ref_k[0][slots]))
                self.assertTrue(torch.equal(v, sim.ref_v[0][slots]))
                if quantized:
                    self.assertTrue(
                        torch.equal(ksf, sim.ref_ksf[0][slots])
                    )
                    self.assertTrue(
                        torch.equal(vsf, sim.ref_vsf[0][slots])
                    )

    def test_layers_share_the_slab_and_chunks_ride(self):
        # Layers advance independently: a layer whose own cursor still
        # fits the prior must ride the slab even while another layer's
        # cursor (ahead, from the same chunk) exceeds it.
        with envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.override(8192):
            sim = _OffloadSim(rows_per_request=1, seed=14)
            sim.extend(1, 64)
            slots = sim.req_to_token[1, :64].long()
            self._gather_both(sim, _fb([1], [64], [64], sim.device), slots)
            self._gather_both(sim, _fb([1], [64], [64], sim.device), slots, layer=1)
            slab = sim.offload._slabs[1]
            self.assertEqual(slab.cursors[:2], [64, 64])
            _extend_more(sim, 1, 32)
            seq = 96
            slots = sim.req_to_token[1, :seq].long()
            # Layer 1 first: its cursor hits 96. Layer 0 (cursor 64 <= the
            # new prior 64) must still ride the SAME slab, not re-stage.
            self._gather_both(
                sim, _fb([1], [seq], [32], sim.device), slots, layer=1
            )
            self._gather_both(sim, _fb([1], [seq], [32], sim.device), slots)
            self.assertIs(sim.offload._slabs[1], slab)
            self.assertEqual(slab.cursors[:2], [96, 96])

    def test_two_requests_offsets(self):
        with envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.override(8192):
            sim = _OffloadSim(rows_per_request=1, seed=12)
            slots_a = sim._alloc(40)
            slots_b = sim._alloc(24)
            sim.req_to_token[1, :40] = slots_a.to(torch.int32)
            sim.req_to_token[2, :24] = slots_b.to(torch.int32)
            sim.offload.begin_forward(
                sim.req_to_token,
                reset_req_pool_indices=torch.tensor([1, 2], device=sim.device),
            )
            sim._write(
                torch.cat([slots_a, slots_b]), sim.offload.extend_rows(64)
            )
            slots = torch.cat([slots_a, slots_b])
            self._gather_both(
                sim, _fb([1, 2], [40, 24], [40, 24], sim.device), slots, layer=1
            )
            self.assertEqual(sim.offload._slabs[2].cursors[1], 24)

    def test_full_pool_gathers_from_host_and_release_returns_rows(self):
        # Grow demand (4096 rows) exceeds the pool: every reservation fails
        # and the gather must equal the all-host gather.
        with envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.override(64):
            sim = _OffloadSim(rows_per_request=1, seed=13)
            sim.extend(1, 64)
            slots = sim.req_to_token[1, :64].long()
            self._gather_both(sim, _fb([1], [64], [64], sim.device), slots)
            self.assertEqual(sim.offload._slabs, {})
        with envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.override(64):
            sim2 = _OffloadSim(rows_per_request=1, seed=13)
            sim2.extend(1, 8)
            sim2.offload.slab_release(1)
            self.assertEqual(sim2.offload._slabs, {})
            self.assertEqual(sim2.offload._slab_spans, [(0, 64)])


if __name__ == "__main__":
    unittest.main()
