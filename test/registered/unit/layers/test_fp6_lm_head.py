"""FP6-e2m3 blockwise-32 packed LM head: numerics and guard-semantics checks.

Same contract as the fp8 head test: quantize -> apply must stay close to the
bf16 reference matmul (the packed 6-bit layout and the 32x32 scale tiling are
easy to corrupt in "looks equivalent" rewrites), the bf16 weight must be
released, and the skip guards (already packed, no dense weight, fp8 head,
non-bf16 dtype, shape outside the kernel band) are what the model load hook
relies on. The M>8 chunking in apply is exercised on the fp8-head-shaped
input path.
"""

import unittest

import torch
from sglang.srt.layers.fp6_lm_head import (
    fp6_lm_head_apply,
    quantize_lm_head_to_fp6_b32,
)
from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


class _Head(torch.nn.Module):
    def __init__(self, n: int, k: int, dtype: torch.dtype):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.weight = torch.nn.Parameter(
            torch.randn(n, k, generator=g, dtype=torch.float32).to(dtype).cuda(),
            requires_grad=False,
        )


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm120_supported(), "SM120-only path"
)
class TestFp6LmHead(CustomTestCase):
    def test_quantize_apply_matches_bf16_reference(self):
        n, k, m = 96, 512, 1
        head = _Head(n, k, torch.bfloat16)
        weight = head.weight.detach().clone()
        self.assertTrue(quantize_lm_head_to_fp6_b32(head))
        self.assertIsNone(head.weight)
        self.assertTrue(head.fp6_b32_packed)
        self.assertEqual(head.weight_fp6_packed.dtype, torch.uint8)
        self.assertEqual(head.weight_fp6_packed.numel(), n * k * 6 // 8)
        self.assertEqual(tuple(head.weight_fp6_scale.shape), (n // 32, k // 32))

        torch.manual_seed(1)
        hidden = torch.randn(m, k, dtype=torch.bfloat16).cuda()
        got = fp6_lm_head_apply(head, hidden)
        ref = hidden.float() @ weight.float().T
        self.assertEqual(got.dtype, torch.bfloat16)
        rel_err = (got.float() - ref).norm() / ref.norm()
        self.assertLess(rel_err.item(), 0.06)
        self.assertTrue(torch.isfinite(got).all())

    def test_apply_chunks_rows_beyond_max_m(self):
        # m > 8 must fall back to multiple launches, not hit the kernel's
        # M <= 8 runtime check.
        n, k, m = 96, 512, 10
        head = _Head(n, k, torch.bfloat16)
        weight = head.weight.detach().clone()
        self.assertTrue(quantize_lm_head_to_fp6_b32(head))
        torch.manual_seed(2)
        hidden = torch.randn(m, k, dtype=torch.bfloat16).cuda()
        got = fp6_lm_head_apply(head, hidden)
        ref = hidden.float() @ weight.float().T
        self.assertEqual(tuple(got.shape), (m, n))
        rel_err = (got.float() - ref).norm() / ref.norm()
        self.assertLess(rel_err.item(), 0.06)

    def test_apply_adds_bias(self):
        n, k = 96, 512
        head = _Head(n, k, torch.bfloat16)
        weight = head.weight.detach().clone()
        self.assertTrue(quantize_lm_head_to_fp6_b32(head))
        torch.manual_seed(3)
        hidden = torch.randn(1, k, dtype=torch.bfloat16).cuda()
        bias = torch.randn(n, dtype=torch.bfloat16).cuda()
        got = fp6_lm_head_apply(head, hidden, bias=bias)
        ref = hidden.float() @ weight.float().T + bias.float()
        rel_err = (got.float() - ref).norm() / ref.norm()
        self.assertLess(rel_err.item(), 0.06)

    def test_quantize_guards(self):
        # Shape outside the kernel band (K below the 512-row SIMT floor).
        head = _Head(96, 256, torch.bfloat16)
        self.assertFalse(quantize_lm_head_to_fp6_b32(head))
        self.assertEqual(head.weight.dtype, torch.bfloat16)

        # Shape outside the kernel band (N not a multiple of 32).
        head = _Head(97, 512, torch.bfloat16)
        self.assertFalse(quantize_lm_head_to_fp6_b32(head))

        # Non-bf16 dtype is refused (fp16 would need a bf16 round-trip).
        head = _Head(96, 512, torch.float16)
        self.assertFalse(quantize_lm_head_to_fp6_b32(head))

        # A head already quantized to fp8 keeps its fp8 form.
        head = _Head(96, 512, torch.bfloat16)
        head.weight = torch.nn.Parameter(
            head.weight.detach().to(torch.float8_e4m3fn), requires_grad=False
        )
        self.assertFalse(quantize_lm_head_to_fp6_b32(head))
        self.assertEqual(head.weight.dtype, torch.float8_e4m3fn)

    def test_quantize_idempotent_and_none_weight(self):
        head = _Head(96, 512, torch.bfloat16)
        self.assertTrue(quantize_lm_head_to_fp6_b32(head))
        self.assertFalse(quantize_lm_head_to_fp6_b32(head))

        head = _Head(96, 512, torch.bfloat16)
        head.weight = None
        self.assertFalse(quantize_lm_head_to_fp6_b32(head))


if __name__ == "__main__":
    unittest.main()
