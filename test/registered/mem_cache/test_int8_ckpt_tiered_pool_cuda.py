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
        # No pinned registration in CI: a leaked cudaHostRegister range
        # poisons later HiCache host pools on shared GPU boxes.
        patcher = mock.patch.object(
            MambaCheckpointPool,
            "_alloc_host",
            lambda self, shape, dtype: torch.empty(shape, dtype=dtype, device="cpu"),
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
