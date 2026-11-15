"""Tests for the tiered int8 checkpoint pool (GPU tier + pinned-RAM tier).

Covers MambaSlotAllocator's two-tier id space (device ids [1, size], RAM ids
(size, size+host_slots], device-first alloc, tier-aware free) and
MambaCheckpointPool's RAM tier: spill-to-RAM stores through the private GPU
staging row, RAM->GPU promotions with event-completion draining, and the
estimate that sizes the RAM tier. CPU-only (the pool's host allocator falls
back to pageable memory without CUDA).

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
        with self.assertRaises(AssertionError):
            pool.promote(3, 9)  # wrong tiers
        with self.assertRaises(AssertionError):
            pool.load_to_active(active, host_id, torch.tensor([4]))

        out = _active_pool()
        pool.load_to_active(out, gpu_slot, torch.tensor([5]))
        rel = ((out.mamba_cache.temporal[:, 5] - src).norm() / src.norm()).item()
        self.assertLess(rel, 1e-2)

        # The promoted slot returns to the allocator once the transfer drains.
        pool._drain_promotions()
        self.assertEqual(pool.num_pending_promotions(), 0)
        self.assertIn(3, pool.allocator.free_slots.tolist())

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


if __name__ == "__main__":
    unittest.main()
