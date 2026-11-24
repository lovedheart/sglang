"""Tests for the tiered int8 checkpoint pool (GPU tier + pinned-RAM tier).

Covers MambaSlotAllocator's two-tier id space (device ids [1, size], RAM ids
(size, size+host_slots], device-first alloc, tier-aware free) and
MambaCheckpointPool's RAM tier: spill-to-RAM stores through the private GPU
staging row, RAM->GPU promotions with event-completion draining, and the
estimate that sizes the RAM tier. CPU legs only here (the pool's host
allocator falls back to pageable memory without CUDA); the CUDA legs live in
test_int8_ckpt_tiered_pool_cuda.py, registered on a GPU runner.

    python -m pytest test/registered/mem_cache/test_int8_ckpt_tiered_pool.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.mamba_checkpoint_pool import MambaCheckpointPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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


def _active_pool(n=10):
    temporal = torch.zeros(L, n, H, V, K, dtype=torch.bfloat16)
    conv = [torch.zeros(L, n, *CONV_SHAPE, dtype=torch.bfloat16)]
    return SimpleNamespace(mamba_cache=SimpleNamespace(temporal=temporal, conv=conv))


class TestMambaSlotAllocatorTiers(unittest.TestCase):
    def test_default_is_untouched_single_tier(self):
        # The active mamba pool keeps using host_slots=0: identity translate,
        # ids in [1, size], no host free list ever holds anything.
        alloc = MambaSlotAllocator(4, "cpu")
        self.assertEqual(alloc.available_size(), 4)
        ids = alloc.alloc(4)
        self.assertEqual(sorted(ids.tolist()), [1, 2, 3, 4])
        self.assertIsNone(alloc.alloc(1))
        self.assertTrue(torch.equal(alloc.translate(ids), ids))
        alloc.free(ids)
        self.assertEqual(alloc.available_size(), 4)
        self.assertEqual(alloc.free_host_slots.numel(), 0)

    def test_alloc_is_device_first(self):
        alloc = MambaSlotAllocator(3, "cpu", host_slots=2)
        self.assertEqual(alloc.available_size(), 5)
        ids = alloc.alloc(5)
        self.assertEqual(ids.tolist(), [1, 2, 3, 4, 5])
        self.assertIsNone(alloc.alloc(1))
        self.assertEqual(
            [alloc.tier_of(i) for i in (3, 4)], [("device", 3), ("host", 1)]
        )

    def test_free_routes_ids_to_their_tier(self):
        # If a freed RAM id ever landed on the device free list, a later alloc
        # would hand out an id past the GPU rows -> out-of-range row writes.
        alloc = MambaSlotAllocator(3, "cpu", host_slots=2)
        alloc.alloc(5)
        alloc.free(torch.tensor([3, 5], dtype=torch.int64))
        self.assertEqual(alloc.free_slots.tolist(), [3])
        self.assertEqual(alloc.free_host_slots.tolist(), [5])
        again = alloc.alloc(2)
        self.assertEqual(again.tolist(), [3, 5])

    def test_clear_refills_both_tiers(self):
        alloc = MambaSlotAllocator(3, "cpu", host_slots=2)
        alloc.alloc(5)
        alloc.clear()
        self.assertEqual(alloc.available_size(), 5)
        self.assertEqual(sorted(alloc.free_slots.tolist()), [1, 2, 3])
        self.assertEqual(sorted(alloc.free_host_slots.tolist()), [4, 5])


class TestTieredCheckpointPool(unittest.TestCase):
    def setUp(self):
        # Deterministic host tier on CUDA dev boxes too: no pinned registration.
        patcher = mock.patch.object(
            MambaCheckpointPool,
            "_alloc_host",
            lambda self, shape, dtype: torch.empty(shape, dtype=dtype, device="cpu"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _pool(self, **over):
        return MambaCheckpointPool(device="cpu", **_pool_kwargs(**over))

    def test_sizing_staging_row_and_estimate(self):
        pool = self._pool(host_slots=4)
        self.assertEqual(pool.available_size(), 12)
        self.assertEqual(pool.device_available_size(), 8)
        self.assertEqual(pool.host_available_size(), 4)
        # The staging row is the extra GPU row past the device tier, addressed
        # by neither tier's free list.
        self.assertEqual(pool.staging_row, 9)
        self.assertEqual(pool.temporal.qdata.shape[1], 10)
        est = MambaCheckpointPool.estimate_mem_usage_bytes(
            **_pool_kwargs(), staging_row=True
        )
        self.assertEqual(est["total"], pool.mem_usage_bytes())
        # RAM-tier sizing in maybe_init divides this into a byte budget.
        self.assertEqual(est["per_row"], pool.ckpt_slot_bytes())

    def test_store_to_ram_keeps_gpu_rows_untouched(self):
        pool = self._pool(host_slots=4)
        active = _active_pool()
        active.mamba_cache.temporal[:, 2] = torch.randn(
            L, H, V, K, dtype=torch.bfloat16
        )
        ids = pool.alloc(9)
        self.assertEqual(ids[-1].item(), 9)  # 8 device ids, then the RAM spill
        self.assertTrue(pool.is_host_slot(9))
        pool.store_from_active(active, torch.tensor([2]), ids[8:9])
        self.assertGreater(pool.host_temporal.qdata[:, pool.host_row(9)].abs().sum(), 0)
        # No device-tier row was consumed: the spill only touched host row 1
        # (via the staging row, verified by the promote roundtrip below).
        self.assertEqual(pool.device_available_size(), 0)
        self.assertEqual(pool.host_available_size(), 3)

    def test_promote_roundtrip_and_drain(self):
        pool = self._pool(host_slots=4)
        active = _active_pool()
        src = torch.randn(L, H, V, K, dtype=torch.bfloat16) * 6e-2
        active.mamba_cache.temporal[:, 2] = src
        ids = pool.alloc(9)
        host_id = ids[8:9]
        pool.store_from_active(active, torch.tensor([2]), host_id)

        self.assertIsNone(pool.alloc_device_slot())  # GPU tier full
        pool.free(torch.tensor([3], dtype=torch.int64))
        gpu_slot = pool.alloc_device_slot()
        self.assertEqual(gpu_slot.item(), 3)
        pool.promote(9, 3)
        self.assertEqual(pool.num_pending_promotions(), 1)
        # The promoted slot is pool-owned while a consumer may read it: it is
        # NOT on the free list even though its transfer is complete on CPU.
        self.assertNotIn(3, pool.allocator.free_slots.tolist())
        with self.assertRaises(AssertionError):
            pool.promote(3, 9)  # wrong tiers
        with self.assertRaises(ValueError):
            pool.load_to_active(active, host_id, torch.tensor([4]))

        out = _active_pool()
        pool.load_to_active(out, gpu_slot, torch.tensor([5]))
        rel = ((out.mamba_cache.temporal[:, 5] - src).norm() / src.norm()).item()
        self.assertLess(rel, 1e-2)

        # Consumption (load_to_active) returns the promoted slot.
        self.assertEqual(pool.num_pending_promotions(), 0)
        self.assertIn(3, pool.allocator.free_slots.tolist())

    def test_zero_alloc_returns_empty(self):
        alloc = MambaSlotAllocator(2, "cpu", host_slots=1)
        self.assertEqual(alloc.alloc(0).numel(), 0)

    def test_abandoned_promotion_releases_the_slot(self):
        pool = self._pool(host_slots=4)
        gpu_slot = pool.alloc_device_slot()
        pool.promote(9, int(gpu_slot.item()))
        pool.release_promotion(int(gpu_slot.item()))
        self.assertEqual(pool.num_pending_promotions(), 0)
        self.assertIn(int(gpu_slot.item()), pool.allocator.free_slots.tolist())
        # No-op for ids the pool never promoted.
        pool.release_promotion(12345)

    def test_two_ram_stores_do_not_clobber_the_staging_row(self):
        pool = self._pool(host_slots=4)
        active = _active_pool()
        a = torch.randn(L, H, V, K, dtype=torch.bfloat16) * 6e-2
        b = torch.randn(L, H, V, K, dtype=torch.bfloat16) * 6e-2
        for ref, ckpt in ((a, 9), (b, 10)):
            active.mamba_cache.temporal[:, 1] = ref
            pool.store_from_active(active, torch.tensor([1]), torch.tensor([ckpt]))
        # Each RAM row holds the quantized state stored into it, not the last
        # staging-row write.
        for slot_id, ref in ((9, a), (10, b)):
            deq = (
                pool.host_temporal.load(
                    torch.tensor([pool.host_row(slot_id)]), torch.bfloat16
                )
                .squeeze(1)
                .float()
            )
            self.assertLess(
                (deq - ref.float()).norm().item() / ref.float().norm().item(), 1e-2
            )

    def test_clear_releases_both_tiers(self):
        pool = self._pool(host_slots=4)
        active = _active_pool()
        ids = pool.alloc(10)
        pool.store_from_active(active, torch.tensor([0]), ids[9:])
        pool.free(ids[:2])
        pool.clear()
        self.assertEqual(pool.available_size(), 12)
        self.assertEqual(pool.num_pending_promotions(), 0)

    def test_no_host_tier_never_spills(self):
        # Default host_slots=0 keeps the pool GPU-only: ids stop at num_slots.
        pool = self._pool()
        self.assertFalse(pool.is_host_slot(8))
        ids = pool.alloc(8)
        self.assertEqual(int(ids.max()), 8)
        self.assertIsNone(pool.alloc(1))

    def test_release_promotion_is_noop_for_foreign_ids(self):
        # release_promotion is safe to call with an id the pool does not own;
        # it must not disturb genuinely pending promotions (the per-request
        # ownership flag leans on this).
        pool = self._pool(host_slots=4)
        gpu_slot = int(pool.alloc_device_slot()[0])
        pool.promote(pool.num_slots + 1, gpu_slot)
        pool.release_promotion(12345)
        self.assertEqual(pool.num_pending_promotions(), 1)
        pool.release_promotion(gpu_slot)
        self.assertEqual(pool.num_pending_promotions(), 0)


class TestStaleCowSourceOwnership(unittest.TestCase):
    """Round-6 B1: a stale TREE-OWNED CoW source must never release a
    promoted slot a DIFFERENT request staged into the same recycled slot id.
    Only the req whose flag says "I got this slot from promote()" may
    release it (components/mamba.py _stage_cow_source)."""

    def _component(self, pool):
        from types import MethodType
        from types import SimpleNamespace as NS

        from sglang.srt.mem_cache.unified_cache.components.mamba import (
            MambaComponent,
        )

        comp = MambaComponent.__new__(MambaComponent)
        storage = NS(checkpoint_pool=lambda _sub: pool)
        comp._state = MethodType(lambda self: (storage, None), comp)
        return comp

    def _req(self, src_index=None, promoted=False):
        from types import SimpleNamespace

        kv = SimpleNamespace(
            mamba_cow_src_index=(
                None if src_index is None else torch.tensor([src_index])
            ),
            mamba_cow_src_promoted=promoted,
        )
        return SimpleNamespace(kv=kv)

    def test_stale_tree_owned_source_does_not_free_foreign_promotion(self):
        pool = MambaCheckpointPool(device="cpu", **_pool_kwargs(host_slots=4))
        gpu_id = int(pool.alloc_device_slot()[0])
        pool.promote(pool.num_slots + 1, gpu_id)
        comp = self._component(pool)

        # Req A holds a stale tree-owned source colliding with B's freshly
        # promoted slot id (LIFO alloc reuses freed ids exactly).
        req_a = self._req(src_index=gpu_id, promoted=False)
        comp._stage_cow_source(req_a, None, False)
        self.assertEqual(
            pool.num_pending_promotions(),
            1,
            "stale tree-owned source released another request's promotion",
        )
        # Req B, the actual owner, releases it and only it.
        req_b = self._req(src_index=gpu_id, promoted=True)
        comp._stage_cow_source(req_b, None, False)
        self.assertEqual(pool.num_pending_promotions(), 0)
        self.assertFalse(req_b.kv.mamba_cow_src_promoted)

    def test_stage_sets_and_clears_ownership_flag(self):
        pool = MambaCheckpointPool(device="cpu", **_pool_kwargs(host_slots=4))
        comp = self._component(pool)
        req = self._req()
        comp._stage_cow_source(req, pool.alloc_device_slot(), True)
        self.assertTrue(req.kv.mamba_cow_src_promoted)
        comp._stage_cow_source(req, pool.alloc_device_slot(), False)
        self.assertFalse(req.kv.mamba_cow_src_promoted)
        req.kv.mamba_cow_src_promoted = True
        comp._stage_cow_source(req, None, False)
        self.assertFalse(req.kv.mamba_cow_src_promoted)


if __name__ == "__main__":
    unittest.main()
