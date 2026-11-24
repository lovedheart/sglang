"""Bit-parity tests for the fused mamba-checkpoint int8 quant/dequant kernels.

The Triton kernels replace the pool's eager quantize/dequantize (the reference
oracle imported here) on the checkpoint store and CoW paths. The replacement
must be bit exact -- a drifting scale or rounding boundary would silently
change every resumed prefix's first decode step, which no downstream test can
attribute. Covers multiple slots, non-square V/K, all three state dtypes, and
the slot-indirect gather the eager path materializes as a copy.

    python -m pytest test/registered/kernels/ops/quantization/test_int8_mamba_ckpt_quant.py -v
"""

import unittest

import torch

from sglang.kernels.ops.quantization.int8_mamba_ckpt_quant_triton import (
    int8_mamba_ckpt_dequant,
    int8_mamba_ckpt_quant,
)
from sglang.srt.mem_cache.mamba_checkpoint_pool import Int8CheckpointStore
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _eager_reference(src, src_slots, n_dst, scale_dtype):
    """Materialize the gathered rows, run the classmethod oracle, and return
    (q, scale) in checkpoint-plane layout (one dst row per src slot)."""
    state = src[:, src_slots.cpu()]
    q, scale = Int8CheckpointStore.quantize(state)
    return q, scale.to(scale_dtype)


class TestInt8MambaCkptQuantBitParity(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def _run_quant(self, L, N, H, V, K, dtype, slots):
        src = torch.randn(L, N, H, V, K, dtype=dtype, device="cuda").contiguous()
        src_slots = torch.tensor(slots, dtype=torch.int64, device="cuda")
        n_dst = 4
        dst_q = torch.zeros(L, n_dst, H, V, K, dtype=torch.int8, device="cuda")
        dst_scale = torch.zeros(L, n_dst, H, 1, K, dtype=dtype, device="cuda")
        dst_slots = torch.arange(n_dst, dtype=torch.int64, device="cuda")
        self.assertTrue(
            int8_mamba_ckpt_quant(src, src_slots, dst_q, dst_scale, dst_slots)
        )
        eq, es = _eager_reference(src, src_slots, n_dst, dtype)
        torch.testing.assert_close(dst_q, eq, rtol=0, atol=0, msg="int8 bits differ")
        torch.testing.assert_close(dst_scale, es, rtol=0, atol=0, msg="scales differ")

    def test_quant_bit_exact_shapes(self):
        for dtype in _DTYPES:
            with self.subTest(dtype=dtype):
                self._run_quant(4, 10, 4, 64, 128, dtype, [9, 0, 5, 7])
                self._run_quant(2, 6, 8, 128, 64, dtype, [3, 1, 4, 2])
                # non power-of-two dims exercise the masks
                self._run_quant(3, 5, 3, 40, 72, dtype, [4, 2, 1, 0])

    def test_dequant_bit_exact(self):
        L, N, H, V, K = 4, 10, 4, 64, 128
        for dtype in _DTYPES:
            with self.subTest(dtype=dtype):
                ckpt_q = torch.randint(
                    -127, 128, (L, 6, H, V, K), dtype=torch.int8, device="cuda"
                )
                ckpt_s = (
                    torch.rand(L, 6, H, 1, K, dtype=dtype, device="cuda") * 1e-2 + 1e-6
                )
                q_slots = torch.tensor([5, 0, 2], dtype=torch.int64, device="cuda")
                dst = torch.zeros(L, 8, H, V, K, dtype=dtype, device="cuda")
                dst_slots = torch.tensor([7, 1, 6], dtype=torch.int64, device="cuda")
                self.assertTrue(
                    int8_mamba_ckpt_dequant(ckpt_q, q_slots, ckpt_s, dst, dst_slots)
                )
                ref = Int8CheckpointStore.dequantize(
                    ckpt_q[:, q_slots.cpu()], ckpt_s[:, q_slots.cpu()], dtype
                )
                torch.testing.assert_close(dst[:, dst_slots.cpu()], ref, rtol=0, atol=0)

    def test_unsupported_shape_falls_back(self):
        # K beyond the kernel's coverage must report unsupported, not launch.
        L, H, V, K = 1, 2, 8, 4096
        src = torch.randn(L, 2, H, V, K, dtype=torch.bfloat16, device="cuda")
        q = torch.zeros(L, 2, H, V, K, dtype=torch.int8, device="cuda")
        s = torch.zeros(L, 2, H, 1, K, dtype=torch.bfloat16, device="cuda")
        slots = torch.arange(2, dtype=torch.int64, device="cuda")
        self.assertFalse(int8_mamba_ckpt_quant(src, slots, q, s, slots))


if __name__ == "__main__":
    unittest.main()
