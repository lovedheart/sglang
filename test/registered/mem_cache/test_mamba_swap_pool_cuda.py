"""CUDA tests for the pinned-RAM mamba state-swap pool.

Exercises the copy-stream paths the CPU leg cannot: park/wake copy on a
private stream with events, wake-then-park row reuse ordering, and clear()
draining in-flight copies. Host rows are forced pageable so CI never runs
cudaHostRegister on a shared runner (the pinned path is exercised in local
acceptance runs).

    python -m pytest test/registered/mem_cache/test_mamba_swap_pool_cuda.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.mamba_swap_pool import MambaSwapPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

L, H, V, K = 2, 4, 32, 32
CONV_SHAPE = (8, K)
ACTIVE_SLOTS = 6


def _fake_mamba_pool(device):
    temporal = torch.zeros(
        L, ACTIVE_SLOTS, H, V, K, dtype=torch.bfloat16, device=device
    )
    conv = [
        torch.zeros(
            L, ACTIVE_SLOTS, 2, *CONV_SHAPE, dtype=torch.bfloat16, device=device
        )
    ]
    return SimpleNamespace(
        mamba_cache=SimpleNamespace(
            temporal=temporal,
            conv=conv,
            replayssm_d=None,
            replayssm_k=None,
            replayssm_g=None,
            replayssm_rawv=None,
            replayssm_rawk=None,
            replayssm_beta=None,
        ),
        enable_linear_replayssm=False,
        enable_linear_replayssm_spec=False,
        replayssm_request_ring=False,
        replayssm_write_pos=None,
        _slot_siblings=[],
    )


class TestMambaSwapPoolCuda(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def setUp(self):
        # Pageable host rows in CI: never cudaHostRegister on a shared GPU.
        patcher = mock.patch.object(
            MambaSwapPool,
            "_alloc_host",
            lambda self, shape, dtype: torch.zeros(shape, dtype=dtype, device="cpu"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _swap(self, swap_slots=3):
        return MambaSwapPool(
            mamba_pool=_fake_mamba_pool(self.device),
            mamba_allocator=MambaSlotAllocator(ACTIVE_SLOTS - 1, self.device),
            swap_slots=swap_slots,
            device=self.device,
        )

    def _fill(self, pool, slot, gen):
        g = torch.Generator(device=self.device).manual_seed(gen)
        pool.mamba_cache.temporal[:, slot].copy_(
            torch.rand(L, H, V, K, generator=g, device=self.device, dtype=torch.float32)
        )
        for c in pool.mamba_cache.conv:
            c[:, slot].copy_(
                torch.rand(
                    L,
                    2,
                    *CONV_SHAPE,
                    generator=g,
                    device=self.device,
                    dtype=torch.float32,
                )
            )

    def test_roundtrip_bit_exact_on_gpu(self):
        swap = self._swap()
        pool = swap.mamba_pool
        self._fill(pool, 2, 11)
        swap_id = swap.park(2)
        self._fill(pool, 2, 12)  # slot recycled with different contents
        swap.wake(swap_id, 4)
        torch.cuda.synchronize()
        g = torch.Generator(device=self.device).manual_seed(11)
        ref_t = torch.rand(
            L, H, V, K, generator=g, device=self.device, dtype=torch.float32
        ).to(torch.bfloat16)
        ref_c = torch.rand(
            L, 2, *CONV_SHAPE, generator=g, device=self.device, dtype=torch.float32
        ).to(torch.bfloat16)
        self.assertTrue(torch.equal(pool.mamba_cache.temporal[:, 4], ref_t))
        self.assertTrue(torch.equal(pool.mamba_cache.conv[0][:, 4], ref_c))

    def test_multirow_record_roundtrip_on_gpu(self):
        # V2-P1 record v2: two slot rows, one id, one fence pair.
        swap = MambaSwapPool(
            mamba_pool=_fake_mamba_pool(self.device),
            mamba_allocator=MambaSlotAllocator(ACTIVE_SLOTS - 1, self.device),
            swap_slots=2,
            device=self.device,
            rows_per_record=2,
        )
        pool = swap.mamba_pool
        self._fill(pool, 1, 21)
        self._fill(pool, 2, 22)
        sid = swap.park([1, 2])
        self._fill(pool, 1, 23)  # both rows recycled with new contents
        self._fill(pool, 2, 24)
        swap.wake(sid, [4, 5])
        torch.cuda.synchronize()
        for slot, seed in ((4, 21), (5, 22)):
            g = torch.Generator(device=self.device).manual_seed(seed)
            ref_t = torch.rand(
                L, H, V, K, generator=g, device=self.device, dtype=torch.float32
            ).to(torch.bfloat16)
            ref_c = torch.rand(
                L,
                2,
                *CONV_SHAPE,
                generator=g,
                device=self.device,
                dtype=torch.float32,
            ).to(torch.bfloat16)
            self.assertTrue(torch.equal(pool.mamba_cache.temporal[:, slot], ref_t))
            self.assertTrue(torch.equal(pool.mamba_cache.conv[0][:, slot], ref_c))

    def test_wake_before_park_reuse_is_ordered(self):
        # Wake frees its swap row; the next park reuses that row on the
        # caller stream. The caller-stream wait recorded in wake must order
        # the new D2H after the previous H2D read.
        swap = self._swap(swap_slots=1)
        pool = swap.mamba_pool
        self._fill(pool, 1, 3)
        sid = swap.park(1)
        swap.wake(sid, 2)
        self._fill(pool, 3, 5)
        sid2 = swap.park(3)
        swap.wake(sid2, 5)
        torch.cuda.synchronize()
        g = torch.Generator(device=self.device).manual_seed(5)
        _ = torch.rand(L, H, V, K, generator=g, device=self.device, dtype=torch.float32)
        ref_conv = torch.rand(
            L, 2, *CONV_SHAPE, generator=g, device=self.device, dtype=torch.float32
        ).to(torch.bfloat16)
        self.assertTrue(torch.equal(pool.mamba_cache.conv[0][:, 5], ref_conv))

    def test_clear_drains_copy_stream(self):
        swap = self._swap(swap_slots=2)
        pool = swap.mamba_pool
        for slot, seed in ((1, 21), (2, 22)):
            self._fill(pool, slot, seed)
            swap.park(slot)
        swap.clear()
        torch.cuda.synchronize()
        self.assertEqual(swap.num_parked(), 0)
        self.assertEqual(swap.available_size(), 2)


if __name__ == "__main__":
    unittest.main()
