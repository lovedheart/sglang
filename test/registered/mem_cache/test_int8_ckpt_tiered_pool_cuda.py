"""CUDA legs of the tiered int8 checkpoint pool tests.

Split out from test_int8_ckpt_tiered_pool.py so CI runs them on a GPU runner
(the CPU suite skips these structurally: the pool copy stream and the
cross-device VALUE moves of the side-state mirrors need CUDA, and a CPU-host
row copied straight into a CUDA index_copy_ raises).

    python -m pytest test/registered/mem_cache/test_int8_ckpt_tiered_pool_cuda.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.mamba_checkpoint_pool import MambaCheckpointPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

H, V, K = 8, 16, 16
L = 2
CONV_SHAPE = (4, K)


def _pool_kwargs(**over):
    kw = dict(
        num_layers=L,
        num_slots=8,
        num_heads=H,
        head_v_dim=V,
        head_k_dim=K,
        conv_shapes=[CONV_SHAPE],
        conv_dtype=torch.bfloat16,
        temporal_dtype=torch.bfloat16,
    )
    kw.update(over)
    return kw


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-only legs")
class TestTieredPoolOnCuda(unittest.TestCase):
    """Covers what the CPU suite structurally cannot: the pool copy stream
    and the cross-device VALUE moves of the side-state mirrors (a CPU-host
    row copied straight into a CUDA index_copy_ raises)."""

    def setUp(self):
        # Bypass _alloc_host's cudaHostRegister (a leaked registration range
        # poisons later HiCache host pools on shared GPU boxes) but keep the
        # memory PINNED via torch's pooled pinning: the host-tier D2Hs stay
        # genuinely async, so a missing copy-stream guard cannot hide behind
        # a synchronous pageable copy.
        patcher = mock.patch.object(
            MambaCheckpointPool,
            "_alloc_host",
            lambda self, shape, dtype: torch.empty(
                shape, dtype=dtype, device="cpu", pin_memory=True
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_spill_and_promote_move_side_states(self):
        side_src = torch.zeros(10, L, 3, dtype=torch.float32).cuda()
        pool = MambaCheckpointPool(
            device="cuda",
            **_pool_kwargs(num_slots=2, host_slots=2, side_states=[(side_src, 0)]),
        )
        temporal = torch.zeros(L, 10, H, V, K, dtype=torch.bfloat16, device="cuda")
        conv = [torch.zeros(L, 10, *CONV_SHAPE, dtype=torch.bfloat16, device="cuda")]
        active = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=temporal, conv=conv)
        )
        state = torch.randn(L, H, V, K, dtype=torch.bfloat16, device="cuda") * 6e-2
        side = torch.randn(3, device="cuda")
        active.mamba_cache.temporal[:, 1] = state
        side_src[1] = side
        cuda_i = lambda *v: torch.tensor(list(v), dtype=torch.int64, device="cuda")

        # Spill: active slot 1 -> RAM id 3 (GPU tier has 2 slots).
        pool.store_from_active(active, cuda_i(1), cuda_i(3))
        gpu = pool.alloc_device_slot()
        self.assertIsNotNone(gpu)
        pool.promote(3, int(gpu.item()))

        out = torch.zeros(L, 5, H, V, K, dtype=torch.bfloat16, device="cuda")
        out_conv = [torch.zeros(L, 5, *CONV_SHAPE, dtype=torch.bfloat16, device="cuda")]
        dst_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=out, conv=out_conv)
        )
        pool.load_to_active(dst_pool, cuda_i(int(gpu.item())), cuda_i(2))
        torch.cuda.synchronize()
        rel = ((out[:, 2] - state).norm() / state.norm()).item()
        self.assertLess(rel, 1e-2)
        # The side mirror survived both tier crossings (D2H then H2D).
        self.assertTrue(torch.allclose(side_src[2], side, atol=1e-6))
        # Consumed: the promoted GPU slot is free again, and the RAM row the
        # promotion READ may not have been overwritten by a later spill.
        self.assertEqual(pool.num_pending_promotions(), 0)

    def test_deferred_sides_publish_after_many_spills(self):
        # Spill MORE times than the sides scratch ring depth, never syncing:
        # the ring must wrap (forcing a blocking publish) and every host row
        # must end up holding its own spill's side values, not a neighbor's.
        n_spill = 6
        side_src = torch.zeros(10, L, 3, dtype=torch.float32).cuda()
        pool = MambaCheckpointPool(
            device="cuda",
            **_pool_kwargs(
                num_slots=2, host_slots=n_spill, side_states=[(side_src, 0)]
            ),
        )
        temporal = torch.zeros(L, 10, H, V, K, dtype=torch.bfloat16, device="cuda")
        conv = [torch.zeros(L, 10, *CONV_SHAPE, dtype=torch.bfloat16, device="cuda")]
        active = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=temporal, conv=conv)
        )
        cuda_i = lambda *v: torch.tensor(list(v), dtype=torch.int64, device="cuda")
        expected = {}
        for i in range(1, n_spill + 1):
            state = torch.randn(L, H, V, K, dtype=torch.bfloat16, device="cuda") * 6e-2
            temporal[:, i] = state
            side = torch.randn(L, 3, dtype=torch.float32, device="cuda")
            side_src[i] = side
            expected[i] = side.cpu()
            # id 3..: all spill straight to the RAM tier (GPU tier has 2 slots).
            pool.store_from_active(active, cuda_i(i), cuda_i(2 + i))
        torch.cuda.synchronize()
        pool._publish_sides(blocking=True)
        hm = pool.host_sides[0][0]
        for i in range(1, n_spill + 1):
            row = pool.host_row(2 + i)
            self.assertTrue(
                torch.allclose(hm[row].cpu(), expected[i], atol=1e-6),
                f"host row {row} (spill {i}) published the wrong side state",
            )

    def test_respill_same_row_across_sides_ring_wrap(self):
        # Regression: a RAM row re-spilled while the sides scratch ring is
        # full must end up holding the NEWER side state. The wrap-time
        # blocking flush must run while the cursor still points at the
        # OLDEST pending spill (sweep is oldest-first); advancing the cursor
        # first visits the oldest LAST and lets it overwrite the newer spill
        # of the same row -- a stale side row a later promote would restore.
        side_src = torch.zeros(10, L, 3, dtype=torch.float32).cuda()
        pool = MambaCheckpointPool(
            device="cuda",
            **_pool_kwargs(num_slots=2, host_slots=5, side_states=[(side_src, 0)]),
        )
        temporal = torch.zeros(L, 10, H, V, K, dtype=torch.bfloat16, device="cuda")
        conv = [torch.zeros(L, 10, *CONV_SHAPE, dtype=torch.bfloat16, device="cuda")]
        active = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=temporal, conv=conv)
        )
        cuda_i = lambda *v: torch.tensor(list(v), dtype=torch.int64, device="cuda")

        # Warm up first (no spin): four spills retire through the ring so the
        # lazy pinned side scratches, the Triton quantize kernels and the
        # caching-allocator blocks all exist BEFORE arming. A lazy
        # cudaHostAlloc or JIT inside the armed loop would host-block long
        # enough to drain the spin, every event would retire between spills,
        # and the wrap flush (the tested line) would never fire.
        spill_seq = [3, 4, 5, 3, 6]
        idts = [cuda_i(c) for c in spill_seq]
        one = cuda_i(1)
        for cid in (3, 4, 5, 6):
            # Same allocations as the armed loop: the first randn/clone pays a
            # caching-allocator cudaMalloc that host-blocks for the spin.
            state = torch.randn(L, H, V, K, dtype=torch.bfloat16, device="cuda") * 6e-2
            side = torch.randn(L, 3, dtype=torch.float32, device="cuda")
            temporal[:, 1].copy_(state)
            side_src[1].copy_(side)
            _ = side.clone()
            pool.store_from_active(active, one, cuda_i(cid), ckpt_id=cid)
        torch.cuda.synchronize()
        pool.clear()

        # spill_seq[3] re-spills row 1 (id 3) with a fresh value; spill_seq[4]
        # finds the depth-4 ring full and fires the wrap flush. ckpt_id is
        # passed explicitly so nothing syncs the backlog (a .item() would
        # drain it and hide the ordering bug). Everything the loop touches --
        # device tensors, index tensors, scratches, kernels -- exists before
        # the arm: a pageable H2D (torch.tensor(..., device=cuda)) would SYNC
        # the stream, retire the events, and make the wrap a no-op.
        newest_row1 = None
        torch.cuda._sleep(5 * 10**8)  # ~0.3s spin: no sides event can retire
        for n, (cid, idt) in enumerate(zip(spill_seq, idts)):
            # Device-side writes only (see above).
            state = torch.randn(L, H, V, K, dtype=torch.bfloat16, device="cuda") * 6e-2
            side = torch.randn(L, 3, dtype=torch.float32, device="cuda")
            temporal[:, 1].copy_(state)
            side_src[1].copy_(side)
            if n == 3:
                newest_row1 = side.clone()
            pool.store_from_active(active, one, idt, ckpt_id=cid)
        torch.cuda.synchronize()
        pool._publish_sides(blocking=True)

        hm = pool.host_sides[0][0]
        self.assertTrue(
            torch.equal(hm[pool.host_row(3)], newest_row1.cpu()),
            "wrap flush published the OLDER spill of row 1 after the newer "
            "one (stale host side state; see _spill_sides_to_host cursor order)",
        )
        # The other spilled rows must be intact too (flush visited, not lost).
        self.assertGreater(hm[pool.host_row(4)].abs().sum().item(), 0.0)
        self.assertGreater(hm[pool.host_row(5)].abs().sum().item(), 0.0)
        self.assertGreater(hm[pool.host_row(6)].abs().sum().item(), 0.0)

    def test_ring_wrap_and_pack_reuse_keep_values(self):
        # More spills than the ring depth, WITHOUT syncing between them: each
        # wrap reuses a staging row and each spill reuses the pack buffers on
        # the copy stream. Every RAM row must still decode to the state that
        # was spilled into it.
        pool = MambaCheckpointPool(device="cuda", **_pool_kwargs(host_slots=8))
        temporal = torch.randn(L, 4, H, V, K, dtype=torch.bfloat16, device="cuda")
        conv = [torch.randn(L, 4, *CONV_SHAPE, dtype=torch.bfloat16, device="cuda")]
        active = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=temporal, conv=conv)
        )
        refs = {}
        # 6 spills into RAM ids 9..14 through a depth-4 staging ring.
        for i, ckpt in enumerate(range(9, 15)):
            state = torch.randn(L, H, V, K, dtype=torch.bfloat16, device="cuda") * 6e-2
            refs[ckpt] = state
            temporal[:, 1] = state
            pool.store_from_active(
                active,
                torch.tensor([1], dtype=torch.int64, device="cuda"),
                torch.tensor([ckpt], dtype=torch.int64, device="cuda"),
            )
        torch.cuda.synchronize()
        for ckpt, ref in refs.items():
            deq = (
                pool.host_temporal.load(
                    torch.tensor([pool.host_row(ckpt)]), torch.bfloat16
                )
                .squeeze(1)
                .float()
                .cuda()
            )
            self.assertLess(
                (deq - ref.float()).norm().item() / ref.float().norm().item(),
                1e-2,
                f"RAM row for {ckpt} corrupted across the ring wrap",
            )
        # And the pack round-trips back: promote every spilled checkpoint.
        for ckpt in refs:
            slot = pool.alloc_device_slot()
            self.assertIsNotNone(slot)
            pool.promote(ckpt, int(slot.item()))
            pool.release_promotion(int(slot.item()))
        pool.clear()

    def test_clear_waits_inflight_copies(self):
        pool = MambaCheckpointPool(
            device="cuda", **_pool_kwargs(num_slots=2, host_slots=2)
        )
        temporal = torch.zeros(L, 4, H, V, K, dtype=torch.bfloat16, device="cuda")
        conv = [torch.zeros(L, 4, *CONV_SHAPE, dtype=torch.bfloat16, device="cuda")]
        active = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=temporal, conv=conv)
        )
        active.mamba_cache.temporal[:, 1] = (
            torch.randn(L, H, V, K, dtype=torch.bfloat16, device="cuda") * 6e-2
        )
        pool.store_from_active(
            active,
            torch.tensor([1], dtype=torch.int64, device="cuda"),
            torch.tensor([3], dtype=torch.int64, device="cuda"),
        )
        pool.clear()  # must not return until the queued D2H retired
        self.assertEqual(pool.num_pending_promotions(), 0)
        self.assertEqual(pool.device_available_size(), 2)


if __name__ == "__main__":
    unittest.main()
