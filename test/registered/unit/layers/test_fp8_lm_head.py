"""FP8 blockwise-128 LM head: scale-layout and guard-semantics checks.

Derived-property test: the block scale tensor's [N//128, K//128] orientation
and the 128x128 tiling are easy to transpose in "looks equivalent" rewrites
(quantize -> apply must stay close to the bf16 reference matmul), and the
skip guards (non-divisible shape, non-bf16 dtype, repeat quantization) are
the contract the model load hook relies on.
"""

import unittest

import torch

from sglang.srt.layers.fp8_lm_head import (
    fp8_blockwise_lm_head_apply,
    quantize_lm_head_to_fp8_block128,
)
from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")


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
class TestFp8LmHead(CustomTestCase):
    def test_quantize_apply_matches_bf16_reference(self):
        n, k, m = 256, 256, 8
        head = _Head(n, k, torch.bfloat16)
        weight = head.weight.detach().clone()
        self.assertTrue(quantize_lm_head_to_fp8_block128(head))
        self.assertEqual(head.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(head.weight_block_scale.shape), (n // 128, k // 128))

        torch.manual_seed(1)
        hidden = torch.randn(m, k, dtype=torch.bfloat16).cuda()
        got = fp8_blockwise_lm_head_apply(head, hidden)
        ref = hidden.float() @ weight.float().T
        self.assertEqual(got.dtype, torch.bfloat16)
        rel_err = (got.float() - ref).norm() / ref.norm()
        self.assertLess(rel_err.item(), 0.06)
        self.assertTrue(torch.isfinite(got).all())

    def test_quantize_guards_and_idempotency(self):
        head = _Head(192, 256, torch.bfloat16)
        self.assertFalse(quantize_lm_head_to_fp8_block128(head))
        self.assertEqual(head.weight.dtype, torch.bfloat16)

        head = _Head(256, 256, torch.float32)
        self.assertFalse(quantize_lm_head_to_fp8_block128(head))

        head = _Head(256, 256, torch.bfloat16)
        self.assertTrue(quantize_lm_head_to_fp8_block128(head))
        self.assertFalse(quantize_lm_head_to_fp8_block128(head))


if __name__ == "__main__":
    unittest.main()
