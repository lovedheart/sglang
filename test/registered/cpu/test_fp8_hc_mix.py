"""MXFP8 HC-mix quantization: layout, guard, and forward-equivalence checks.

Derived-property tests for --enable-fp8-hc-mix (fp8_hc_mix.py): the row-major
[N, K//32] ue8m0 scale orientation is easy to transpose or swizzle in "looks
equivalent" rewrites, the skip guards (non-divisible K, non-bf16 dtype, the
last-layer / final-mixer exclusion) are the contract the qwen4_exp load hook
relies on, and quantize-then-mix must stay close to the BF16 mix reference.
CPU-only by design: the Triton decode kernel path is covered on GPU by the
regular HC mix tests.
"""

import unittest

import torch

from sglang.srt.layers.fp8_hc_mix import (
    mxfp8_dequantize_to_bf16,
    mxfp8_quantize_rowmajor,
    quantize_hc_mix_to_mxfp8,
    quantize_linear_to_mxfp8,
)
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="stage-a-test-cpu-intel")

HC, HS, LOWRANK = 4, 512, 320  # K = HC*HS = 2048: %32 == 0 and %2048 == 0


class TestMxfp8RoundTrip(CustomTestCase):
    def test_roundtrip_error_is_e4m3_bounded(self):
        torch.manual_seed(0)
        w = (torch.randn(320, 2048) * 0.02).bfloat16().contiguous()
        q, s = mxfp8_quantize_rowmajor(w)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.uint8)
        self.assertEqual(tuple(s.shape), (320, 2048 // 32))
        self.assertGreater(int(s.min()), 0, "ue8m0 bytes must be biased exponents")
        deq = mxfp8_dequantize_to_bf16(q, s)
        rel = ((deq.float() - w.float()).norm() / w.float().norm()).item()
        self.assertLess(rel, 0.03)

    def test_scale_orientation(self):
        # A single group along K scaled far up must survive only in that
        # group's slice: catches a transposed or K-major scale layout.
        w = torch.zeros(64, 128, dtype=torch.bfloat16)
        w[:, 32:64] = 500.0  # > 448 so this group's biased exponent beats the
        # zero groups' e=0 (s=127); a transposed layout would not separate.
        q, s = mxfp8_quantize_rowmajor(w)
        self.assertEqual(tuple(s.shape), (64, 4))
        self.assertEqual(int(s[0, 0]), 127)
        self.assertGreater(int(s[0, 1]), int(s[0, 0]))
        deq = mxfp8_dequantize_to_bf16(q, s)
        self.assertTrue(torch.all(deq[:, :32] == 0))
        self.assertTrue(torch.all(deq[:, 32:64].abs() > 250))


class TestQuantizeGuards(CustomTestCase):
    def test_ineligible_linears_stay_bf16(self):
        lin = torch.nn.Linear(100, 64, bias=False, dtype=torch.bfloat16)
        self.assertFalse(quantize_linear_to_mxfp8(lin))  # K % 32 != 0
        self.assertEqual(lin.weight.dtype, torch.bfloat16)
        lin = torch.nn.Linear(128, 64, bias=False, dtype=torch.float32)
        self.assertFalse(quantize_linear_to_mxfp8(lin))  # not bf16
        self.assertEqual(lin.weight.dtype, torch.float32)


def _cfg():
    return HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=torch.bfloat16,
        hc_lowrank=LOWRANK,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )


class _Block(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_hyper_connection = GatedResidual(cfg)
        self.mlp_hyper_connection = GatedResidual(cfg)


class _Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(cfg) for _ in range(3)])
        self.hyper_connection_mixer = GatedResidual(cfg, use_combine=False)


class TestHcMixSwap(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _Model(_cfg())
        self.x = (torch.randn(8, HC * HS) * 3.0).bfloat16()
        self.ref = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, GatedResidual) and hasattr(mod, "input_mix_weight_down"):
                normed = mod.hc_norm(self.x)
                t = torch.nn.functional.silu(
                    torch.nn.functional.linear(
                        normed, mod.input_mix_weight_down.weight
                    )
                    / HC
                )
                gate = torch.sigmoid(
                    torch.nn.functional.linear(t, mod.input_mix_weight_up.weight)
                ).unflatten(-1, (HC, HS))
                self.ref[name] = (
                    gate * normed.unflatten(-1, (HC, HS))
                ).mean(-2)

    def test_exclusions_and_forward_equivalence(self):
        n = quantize_hc_mix_to_mxfp8(self.model, last_layer=2)
        self.assertEqual(n, 4)  # layers 0-1 x {attn, mlp}
        last = self.model.layers[2].attn_hyper_connection
        self.assertEqual(last.input_mix_weight_down.weight.dtype, torch.bfloat16)
        mixer = self.model.hyper_connection_mixer
        self.assertEqual(mixer.input_mix_weight_up.weight.dtype, torch.bfloat16)
        converted = self.model.layers[0].mlp_hyper_connection
        self.assertEqual(
            converted.input_mix_weight_down.weight.dtype, torch.float8_e4m3fn
        )
        self.assertFalse(converted._norm_fold_ok)  # fold would dequantize eagerly

        # Idempotent: a second call converts nothing.
        self.assertEqual(quantize_hc_mix_to_mxfp8(self.model, last_layer=2), 0)

        for name, mod in self.model.named_modules():
            if not (isinstance(mod, GatedResidual) and hasattr(mod, "hc_norm")):
                continue
            out, _ = mod.mix(self.x)
            rel = ((out.float() - self.ref[name]).norm() / self.ref[name].norm()).item()
            self.assertLess(rel, 0.05, f"{name} rel_err {rel:.2e}")
            self.assertTrue(torch.isfinite(out).all())


class TestW8A8PrefillGate(CustomTestCase):
    def test_cpu_tensors_never_take_the_w8a8_path(self):
        # The W8A8 prefill GEMMs are Triton/cuda-only; a CPU (or too-small)
        # batch must fall back to the dequant chain, not reach the kernels.
        from sglang.kernels.ops.gemm.hc_mix_fp8 import (
            _W8A8_MIN_ROWS,
            fused_hc_mix_fp8_w8a8_supported,
        )

        torch.manual_seed(0)
        down = torch.randn(LOWRANK, HC * HS, dtype=torch.bfloat16)
        up = torch.randn(HC * HS, LOWRANK, dtype=torch.bfloat16)
        d8, ds = mxfp8_quantize_rowmajor(down)
        u8, us = mxfp8_quantize_rowmajor(up)
        big = torch.randn(_W8A8_MIN_ROWS, HC * HS, dtype=torch.bfloat16)
        self.assertFalse(fused_hc_mix_fp8_w8a8_supported(big, d8, u8, ds, us))
        small = torch.randn(17, HC * HS, dtype=torch.bfloat16)
        self.assertFalse(fused_hc_mix_fp8_w8a8_supported(small, d8, u8, ds, us))


if __name__ == "__main__":
    unittest.main()
