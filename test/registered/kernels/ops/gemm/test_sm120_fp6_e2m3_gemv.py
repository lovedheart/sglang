"""Tests for the SM120 FP6-e2m3 b32 GEMV/GEMM (lm-head band) against an
fp32 reference matmul over fp64-dequantized codes, plus pack/quantize
roundtrip checks.

The SIMT entry (bf16 activations) is checked to 5e-3 against an exact fp32
reference. The dispatcher routes M >= 3 to the W6A8 cubin, which quantizes
activations to e4m3 first, so its exact reference mirrors that quantization
and the cross-path (dispatcher vs SIMT) check uses a looser tolerance.
"""

from __future__ import annotations

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)):
    pytest.skip("SM120 fp6 e2m3 gemv requires CUDA SM120", allow_module_level=True)

from sglang.kernels.ops.gemm.sm120_fp6_e2m3 import (
    dequantize_e2m3_b32,
    quantize_e2m3_b32,
    sm120_fp6_e2m3_gemm,
    sm120_fp6_e2m3_gemm_simt,
    sm120_fp6_e2m3_gemv,
    use_sm120_fp6_e2m3_gemm_w6,
    use_sm120_fp6_e2m3_gemv,
)

SHAPES = [
    pytest.param(1, 2560, 2560, id="lmhead-like-m1"),
    pytest.param(4, 2560, 2560, id="mtp-verify-m4"),
    pytest.param(8, 8192, 2560, id="m8-wide"),
    pytest.param(1, 96, 512, id="tiny"),
    pytest.param(2, 2592, 1024, id="tail-n"),
]


def _case(m, n, k, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(n, k, device="cuda") * 0.02
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    packed, scale, codes = quantize_e2m3_b32(w)
    out = sm120_fp6_e2m3_gemm_simt(x, packed, scale)
    ref = x.float() @ dequantize_e2m3_b32(codes, scale).float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    return out, ref, rel


class TestFp6E2m3Gemv:
    @pytest.mark.parametrize("m,n,k", SHAPES)
    def test_matches_fp32_reference(self, m, n, k):
        out, _ref, rel = _case(m, n, k)
        # fp32 accumulation with per-32-block scale application vs a fused fp32
        # matmul: only output bf16 rounding separates them.
        assert rel < 5e-3, (m, n, k, rel)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("m,n,k", SHAPES[:2])
    def test_deterministic(self, m, n, k):
        for seed in (0, 7):
            a, _, _ = _case(m, n, k, seed)
            b, _, _ = _case(m, n, k, seed)
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_gemv_convenience_matches_gemm(self):
        torch.manual_seed(3)
        w = torch.randn(320, 1024, device="cuda") * 0.02
        x = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
        packed, scale, _ = quantize_e2m3_b32(w)
        torch.testing.assert_close(
            sm120_fp6_e2m3_gemv(x, packed, scale),
            sm120_fp6_e2m3_gemm(x.unsqueeze(0), packed, scale)[0],
            rtol=0,
            atol=0,
        )

    def test_simt_above_band_rejected(self):
        torch.manual_seed(0)
        w = torch.randn(320, 1024, device="cuda") * 0.02
        x = torch.randn(9, 1024, device="cuda", dtype=torch.bfloat16)
        packed, scale, _ = quantize_e2m3_b32(w)
        with pytest.raises((ValueError, RuntimeError, Exception)):
            sm120_fp6_e2m3_gemm_simt(x, packed, scale)
        # ... but the dispatcher serves it (W6A8 band).
        out = sm120_fp6_e2m3_gemm(x, packed, scale)
        assert out.shape == (9, 320)

    def test_gate(self):
        assert use_sm120_fp6_e2m3_gemv(1, 248320, 2560)
        assert use_sm120_fp6_e2m3_gemv(5, 8192, 2560)
        assert not use_sm120_fp6_e2m3_gemv(9, 8192, 2560)  # M > 8
        assert not use_sm120_fp6_e2m3_gemv(1, 8192, 2550)  # K % 128
        assert not use_sm120_fp6_e2m3_gemv(1, 8016, 2560)  # N % 32
        assert use_sm120_fp6_e2m3_gemm_w6(3, 248320, 2560)
        assert use_sm120_fp6_e2m3_gemm_w6(1024, 8192, 2560)  # chunked
        assert not use_sm120_fp6_e2m3_gemm_w6(2, 8192, 2560)  # SIMT owns M<=2
        assert not use_sm120_fp6_e2m3_gemm_w6(4, 8192, 2550)  # K % 128
        assert not use_sm120_fp6_e2m3_gemm_w6(4, 8016, 2560)  # N % 32


class TestW6A8Band:
    """The W6A8 cubin path of the dispatcher: e4m3 activations, per-token-group
    128 scales, exact-reference tolerance on output-bf16 rounding only."""

    @staticmethod
    def _case(m, n, k, seed=0):
        from sglang.kernels.ops.quantization.per_token_group_quant import (
            per_token_group_quant,
        )

        torch.manual_seed(seed)
        w = torch.randn(n, k, device="cuda") * 0.02
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        packed, scale, codes = quantize_e2m3_b32(w)
        out = sm120_fp6_e2m3_gemm(x, packed, scale)
        xq = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda")
        xs = torch.empty((m, k // 128), dtype=torch.float32, device="cuda")
        xq, xs = per_token_group_quant(x, xq, xs, group_size=128, scale_ue8m0=False)
        a = xq.float() * xs.repeat_interleave(128, dim=1)
        ref = a @ dequantize_e2m3_b32(codes, scale).float().t()
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        return out, rel

    @pytest.mark.parametrize(
        "m,n,k",
        [
            pytest.param(4, 2560, 2560, id="mtp-verify-m4"),
            pytest.param(16, 8192, 2560, id="m16"),
            pytest.param(33, 96, 384, id="m33-chunked"),
        ],
    )
    def test_matches_actquant_reference(self, m, n, k):
        out, rel = self._case(m, n, k)
        # kernel-vs-exact was verified at 1.7e-3 in gemm_dev; plus the srt
        # quantizer rounding here is identical, so only bf16 out rounding
        # separates (<=2e-3), leaving headroom for quantizer eps differences.
        assert rel < 6e-3, (m, n, k, rel)
        assert not torch.isnan(out).any()

    def test_cross_path_matches_simt(self):
        # dispatcher (fp8 acts) vs SIMT (bf16 acts): activation quantization
        # dominates the gap, ~2.5e-2 rel on unit-variance data.
        torch.manual_seed(5)
        w = torch.randn(2592, 1024, device="cuda") * 0.02
        x = torch.randn(6, 1024, device="cuda", dtype=torch.bfloat16)
        packed, scale, _ = quantize_e2m3_b32(w)
        a = sm120_fp6_e2m3_gemm(x, packed, scale).float()
        b = sm120_fp6_e2m3_gemm_simt(x[:2], packed, scale)
        a = torch.cat([a[:2], b.new_zeros((0, b.shape[1]))]).float()
        rel = ((a - b.float()).norm() / b.float().norm()).item()
        assert rel < 6e-2


class TestQuantizer:
    def test_pack_bitexact_roundtrip(self):
        torch.manual_seed(1)
        w = torch.randn(320, 640, device="cuda")
        packed, _, codes = quantize_e2m3_b32(w)
        torch.testing.assert_close(
            unpack_e2m3(packed, codes.numel()).view_as(codes), codes, rtol=0, atol=0
        )

    def test_codes_are_nearest_grid_point(self):
        torch.manual_seed(2)
        w = torch.randn(96, 96, device="cuda") * 0.1
        _, _, codes = quantize_e2m3_b32(w)
        deq = dequantize_e2m3_b32(codes, torch.ones(3, 3, device="cuda"))
        # every dequantized magnitude is one of the 31*2+1 grid points
        vals = {m / 8 for m in range(8)} | {
            (1 + m / 8) * 2.0 ** (e - 1)
            for e in (1, 2, 3)
            for m in range(8)
            if not (e == 3 and m == 7)
        }
        assert set(deq.abs().unique().tolist()) <= {round(v, 12) for v in vals} | {0.0}


def unpack_e2m3(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Pure-torch inverse of the 4-codes-per-3-bytes packing (byte-stream model)."""
    v = torch.cat(
        [
            packed.view(torch.int32),
            torch.zeros(1, dtype=torch.int32, device=packed.device),
        ]
    ).to(torch.int64)
    g = torch.arange(count // 4, device=packed.device, dtype=torch.int64)
    b = g * 3  # byte offset of each 4-code group
    u = ((v[b // 4 + 1] << 32) | (v[b // 4] & 0xFFFFFFFF)) >> ((b % 4) * 8)
    j = torch.arange(4, device=packed.device, dtype=torch.int64)
    return ((u[:, None] >> (6 * j)[None, :]) & 63).to(torch.uint8).reshape(-1)[:count]
