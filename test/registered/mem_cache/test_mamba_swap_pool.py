"""CPU tests for the pinned-RAM mamba state-swap pool (park/wake/drop).

The pool mirrors the active MambaPool's bf16 planes into pinned host rows
(one row per parked request) so a parked request can surrender its active
slot and resume later without re-prefill. CPU legs here use pageable host
rows (_alloc_host patched away); the CUDA legs in
test_mamba_swap_pool_cuda.py cover the copy-stream ordering.

    python -m pytest test/registered/mem_cache/test_mamba_swap_pool.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.mamba_swap_pool import MambaSwapPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

L, H, V, K = 4, 8, 16, 16
CONV_SHAPE = (4, K)
ACTIVE_SLOTS = 5  # rows in the fake active pool (slot ids are 1..4)


class _FakeSibling:
    """Slot-indexed side state (slot axis 0) with the retraction copy path."""

    def __init__(self, n):
        self.rows = torch.zeros(n, 6, dtype=torch.float32)
        self.cpu_rows = torch.zeros(n, 6, dtype=torch.float32)

    def get_cpu_slots(self, indices):
        return self.rows[indices.to(torch.long)].clone()

    def load_cpu_slots(self, data, indices):
        self.rows[indices.to(torch.long)] = data

    def iter_transfer_state_entries(self):
        yield "fake", self.rows, None, 0


def _fake_mamba_pool(siblings=()):
    temporal = torch.zeros(L, ACTIVE_SLOTS, H, V, K, dtype=torch.bfloat16)
    conv = [torch.zeros(L, ACTIVE_SLOTS, 2, *CONV_SHAPE, dtype=torch.bfloat16)]
    cache = SimpleNamespace(
        temporal=temporal,
        conv=conv,
        replayssm_d=None,
        replayssm_k=None,
        replayssm_g=None,
        replayssm_rawv=None,
        replayssm_rawk=None,
        replayssm_beta=None,
    )
    return SimpleNamespace(
        mamba_cache=cache,
        enable_linear_replayssm=False,
        enable_linear_replayssm_spec=False,
        replayssm_request_ring=False,
        replayssm_write_pos=None,
        _slot_siblings=list(siblings),
    )


def _write_state(pool, slot, value):
    pool.mamba_cache.temporal[:, slot] = value
    for c in pool.mamba_cache.conv:
        c[:, slot] = -value


class TestMambaSwapPool(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            MambaSwapPool,
            "_alloc_host",
            lambda self, shape, dtype: torch.zeros(shape, dtype=dtype, device="cpu"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _swap(self, *, swap_slots=3, siblings=(), allocator="auto"):
        mamba_pool = _fake_mamba_pool(siblings)
        alloc = (
            MambaSlotAllocator(ACTIVE_SLOTS - 1, "cpu")
            if allocator == "auto"
            else allocator
        )
        return MambaSwapPool(
            mamba_pool=mamba_pool,
            mamba_allocator=alloc,
            swap_slots=swap_slots,
            device="cpu",
        )

    def test_park_wake_is_bit_exact(self):
        swap = self._swap()
        pool = swap.mamba_pool
        _write_state(pool, 2, 0.75)
        swap_id = swap.park(2)
        self.assertIsNotNone(swap_id)
        _write_state(pool, 2, -3.0)  # slot recycled by someone else
        swap.wake(swap_id, 4)
        self.assertTrue((pool.mamba_cache.temporal[:, 4] == 0.75).all())
        for c in pool.mamba_cache.conv:
            self.assertTrue((c[:, 4] == -0.75).all())

    def _swap_rpr(self, rows, swap_slots=3):
        return MambaSwapPool(
            mamba_pool=_fake_mamba_pool(),
            mamba_allocator=MambaSlotAllocator(ACTIVE_SLOTS - 1, "cpu"),
            swap_slots=swap_slots,
            device="cpu",
            rows_per_record=rows,
        )

    def test_multirow_record_round_trips_atomically(self):
        # V2-P1 record v2: one id parks a slot SET (e.g. main + tracking).
        swap = self._swap_rpr(2)
        pool = swap.mamba_pool
        _write_state(pool, 1, 0.5)
        _write_state(pool, 2, 0.25)
        alloc_before = swap.mamba_allocator.available_size()
        swap_id = swap.park([1, 2])
        self.assertIsNotNone(swap_id)
        self.assertEqual(swap.mamba_allocator.available_size(), alloc_before + 2)
        _write_state(pool, 1, -9.0)  # slots recycled by someone else
        _write_state(pool, 2, -9.0)
        swap.wake(swap_id, [3, 4])
        self.assertTrue((pool.mamba_cache.temporal[:, 3] == 0.5).all())
        self.assertTrue((pool.mamba_cache.temporal[:, 4] == 0.25).all())
        for c in pool.mamba_cache.conv:
            self.assertTrue((c[:, 3] == -0.5).all())
            self.assertTrue((c[:, 4] == -0.25).all())
        # One record consumed one swap id regardless of row count.
        self.assertEqual(swap.num_parked(), 0)
        self.assertEqual(swap.available_size(), 3)

    def test_row_count_must_match_the_record(self):
        swap = self._swap_rpr(2)
        with self.assertRaises(ValueError):
            swap.park(1)
        with self.assertRaises(ValueError):
            swap.park([1, 2, 3])

    def test_park_frees_the_active_slot(self):
        swap = self._swap()
        alloc = swap.mamba_allocator
        before = alloc.available_size()
        swap.park(1)
        self.assertEqual(alloc.available_size(), before + 1)

    def test_park_refuses_when_full_and_keeps_slot(self):
        swap = self._swap(swap_slots=1)
        alloc = swap.mamba_allocator
        self.assertIsNotNone(swap.park(1))
        before = alloc.available_size()
        self.assertIsNone(swap.park(2))
        self.assertEqual(alloc.available_size(), before)
        self.assertEqual(swap.num_parked(), 1)

    def test_drop_is_idempotent_and_restores_capacity(self):
        swap = self._swap(swap_slots=2)
        swap_id = swap.park(1)
        self.assertTrue(swap.drop(swap_id))
        self.assertFalse(swap.drop(swap_id))
        self.assertEqual(swap.available_size(), 2)
        self.assertEqual(swap.num_parked(), 0)

    def test_wake_consumes_the_parked_record(self):
        swap = self._swap()
        swap_id = swap.park(1)
        swap.wake(swap_id, 2)
        self.assertEqual(swap.num_parked(), 0)
        # The active slot is re-allocated by the caller, so the swap id is
        # recycled by wake itself.
        self.assertEqual(swap.available_size(), 3)
        with self.assertRaises(KeyError):
            swap.wake(swap_id, 3)

    def test_sibling_state_round_trips(self):
        sib = _FakeSibling(ACTIVE_SLOTS)
        sib.rows[3] = torch.arange(6.0)
        swap = self._swap(siblings=[sib])
        swap_id = swap.park(3)
        sib.rows[3] = -1.0
        swap.wake(swap_id, 1)
        self.assertTrue(torch.equal(sib.rows[1], torch.arange(6.0)))

    def test_clear_releases_everything(self):
        swap = self._swap(swap_slots=3)
        swap.park(1)
        swap.park(2)
        swap.clear()
        self.assertEqual(swap.num_parked(), 0)
        self.assertEqual(swap.available_size(), 3)

    def test_rejects_replayssm_pool(self):
        pool = _fake_mamba_pool()
        pool.enable_linear_replayssm = True
        with self.assertRaisesRegex(ValueError, "ReplaySSM"):
            MambaSwapPool(
                mamba_pool=pool,
                mamba_allocator=None,
                swap_slots=2,
                device="cpu",
            )

    def test_plane_bytes_match_active_pool_density(self):
        swap = self._swap(swap_slots=3)
        per_slot = (
            swap.mamba_pool.mamba_cache.temporal[:, 0].numel()
            + sum(c[:, 0].numel() for c in swap.mamba_pool.mamba_cache.conv)
        ) * torch.bfloat16.itemsize
        # +1 for the index-0 pad row each plane carries, like the active pool.
        self.assertEqual(swap.total_bytes(), (3 + 1) * per_slot)


if __name__ == "__main__":
    unittest.main()
