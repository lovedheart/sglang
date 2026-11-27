"""Bit-identity pin: grouped_gemma_rmsnorm_mxfp8 == grouped_gemma_rmsnorm
followed by hc_mix_fp8._mxfp8_quant_act_kernel, and the mix consumes the
fused (q, sf) with no output change.

A rewrite of the norm reduction, of the Triton quantizer, or of the CUDA
epilogue that quietly diverges (different log2 path, different rounding of
the scale byte, different fp8x2 pair packing) turns this red; a "tolerance"
comparison would not catch a single flipped exponent in one group.
"""

import pytest
import torch
import triton

from sglang.kernels.ops.gemm.hc_mix_fp8 import (
    _mxfp8_quant_act_kernel,
    fused_hc_mix_fp8_w8a8,
)
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
    grouped_gemma_rmsnorm,
)
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm_mxfp8 import (
    grouped_gemma_rmsnorm_mxfp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 quant epilogue needs SM100+",
)

EPS = 1e-6


def _quant_reference(normed: torch.Tensor):
    rows, k = normed.shape
    q = torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=normed.device)
    sf = torch.empty(rows, k // 32, dtype=torch.uint8, device=normed.device)
    _mxfp8_quant_act_kernel[(rows, triton.cdiv(k, 1024))](
        normed, q, sf, rows, K=k, BLOCK=1024, num_warps=4
    )
    return q, sf


def _adversarial(rows: int, hidden: int, device, dtype) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(rows, hidden, generator=g, device=device).to(dtype)
    if rows >= 6:
        x[0] = 0.0
        x[1] = 448.0  # exact e4m3 top-of-range group
        x[2] = -448.0
        x[3] = 1e-38  # subnormal-ish magnitudes
        x[4] = torch.nan
        x[5] = torch.inf
    return x


@requires_blackwell
@pytest.mark.parametrize(
    "rows,hidden,group",
    [(512, 10240, 2560), (777, 10240, 2560), (256, 5120, 5120), (33, 1024, 512)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_matches_unfused_bitwise(rows, hidden, group, dtype):
    device = "cuda"
    x = _adversarial(rows, hidden, device, dtype)
    weight = (torch.randn(hidden, device=device) * 0.1).to(dtype)

    normed_ref = grouped_gemma_rmsnorm(x, weight, group, EPS)
    q_ref, sf_ref = _quant_reference(normed_ref)

    normed, q, sf = grouped_gemma_rmsnorm_mxfp8(x, weight, group, EPS)

    # Bit-level compare: torch.equal says False for any NaN, even bit-identical.
    assert torch.equal(normed.view(torch.uint8), normed_ref.view(torch.uint8)), (
        "norm bits diverged"
    )
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), (
        "e4m3 bits diverged"
    )
    assert torch.equal(sf, sf_ref), "scale bytes diverged"


@requires_blackwell
def test_mix_output_unchanged_with_fused_quant():
    # Production mix shapes: hc=4, hs=2560, lowrank=320, 512 rows.
    torch.manual_seed(3)
    hc, hs, lowrank = 4, 2560, 320
    rows, k = 512, 4 * 2560
    device = "cuda"
    x = _adversarial(rows, k, device, torch.bfloat16)
    weight = (torch.randn(k, device=device) * 0.1).to(torch.bfloat16)
    w_down = torch.randn(lowrank, k, device=device).to(torch.float8_e4m3fn)
    w_down_scale = torch.randint(
        120, 132, (lowrank, k // 32), device=device, dtype=torch.uint8
    )
    w_up = torch.randn(hc * hs, lowrank, device=device).to(torch.float8_e4m3fn)
    w_up_scale = torch.randint(
        120, 132, (hc * hs, lowrank // 32), device=device, dtype=torch.uint8
    )

    normed, a8, asf = grouped_gemma_rmsnorm_mxfp8(x, weight, k, EPS)
    base = fused_hc_mix_fp8_w8a8(normed, w_down, w_down_scale, w_up, w_up_scale, hc, hs)
    folded = fused_hc_mix_fp8_w8a8(
        normed,
        w_down,
        w_down_scale,
        w_up,
        w_up_scale,
        hc,
        hs,
        act_quant=(a8, asf),
    )
    assert torch.equal(base.view(torch.uint8), folded.view(torch.uint8)), (
        "mix output changed with fused quant"
    )
    # The quant epilogue must match the standalone quantizer on normed.
    q_ref, sf_ref = _quant_reference(normed)
    assert torch.equal(a8.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(asf, sf_ref)


@requires_blackwell
def test_rejects_non_contiguous_input():
    x = torch.randn(512, 5120 + 64, device="cuda", dtype=torch.bfloat16)[:, :5120]
    weight = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        grouped_gemma_rmsnorm_mxfp8(x, weight, 5120)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
