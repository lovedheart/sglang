import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.hc_mix import (
    _FUSED_MIX_MAX_ROWS,
    fused_hc_mix,
    fused_hc_mix_fold,
    fused_hc_mix_fold_supported,
    fused_hc_mix_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320
FOLD_EPS = 1e-6


def _reference_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Mirrors GatedResidual._mix_compute in hyperconnection.py."""
    x = hyper_input_normed.to(compute_dtype)
    t = F.silu(F.linear(x, w_down.to(compute_dtype)) / hc)
    u = torch.sigmoid(F.linear(t, w_up.to(compute_dtype)))
    return (u.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def _make_inputs(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda")
    w_down = (
        torch.randn(LOWRANK, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.02
    )
    w_up = (
        torch.randn(HC_COUNT * HIDDEN_SIZE, LOWRANK, dtype=dtype, device="cuda") * 0.02
    )
    return x, w_down, w_up


def _make_fold_inputs(num_tokens: int, dtype: torch.dtype):
    """Raw x plus the fold-side tensors GatedResidual builds: wn = 1 + w_norm
    and w_down_fold = w_down * wn (bf16-rounded at "load", like production).
    w_down is kept unfolded so the same case can also drive the unfused path.
    """
    torch.manual_seed(0)
    x, w_down, w_up = _make_inputs(num_tokens, dtype)
    w_norm = torch.randn(HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.1
    wn = (1.0 + w_norm.float()).contiguous()
    w_down_fold = (w_down.float() * wn).to(dtype).contiguous()
    return x, w_down, w_down_fold, w_up, wn


def _reference_mix_fold(
    x: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    wn: torch.Tensor,
    hc: int,
    hs: int,
    eps: float,
    compute_dtype: torch.dtype = torch.float64,
):
    """fp64 reference for the norm-folded mix, taken from the raw residual:
    per-branch rsqrt factors g and the Gemma-style (1+w) applied inline. The
    projections use the *unfolded* fp64 weights, so the bf16 rounding of the
    folded weight (production stores w_down * wn in bf16) is part of the
    error being measured on the fold side. Returns (mixed, g) matching
    fused_hc_mix_fold's outputs."""
    x3 = x.to(compute_dtype).unflatten(-1, (hc, hs))
    g = torch.rsqrt(x3.square().mean(-1) + eps)
    wn3 = wn.to(compute_dtype).unflatten(0, (hc, hs))
    xn3 = x3 * wn3 * g.unsqueeze(-1)
    wd3 = w_down.to(compute_dtype).unflatten(-1, (hc, hs))
    t = F.silu(torch.einsum("rbk,lbk->rl", xn3, wd3) / hc)
    u = torch.sigmoid(F.linear(t, w_up.to(compute_dtype)))
    mixed = (u.unflatten(-1, (hc, hs)) * xn3).mean(dim=-2)
    return mixed, g


_TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=2e-3, atol=1e-3),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 4, 7, _FUSED_MIX_MAX_ROWS])
def test_fused_hc_mix_matches_reference(dtype, num_tokens):
    x, w_down, w_up = _make_inputs(num_tokens, dtype)
    assert fused_hc_mix_supported(x, w_down, w_up)
    out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.testing.assert_close(out.to(torch.float64), ref, **_TOLERANCES[dtype])


def test_fused_hc_mix_no_less_accurate_than_eager():
    """The fused kernel (fp32 accumulation throughout) must not be farther
    from the fp64 reference than the eager bf16 chain it replaces."""
    x, w_down, w_up = _make_inputs(8, torch.bfloat16)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    fused = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    eager = _reference_mix(
        x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, compute_dtype=torch.bfloat16
    )
    fused_err = (fused.to(torch.float64) - ref).abs().max()
    eager_err = (eager.to(torch.float64) - ref).abs().max()
    assert fused_err <= eager_err * 1.5 + 1e-6


def test_fused_hc_mix_gate_rejects_prefill_rows():
    x, w_down, w_up = _make_inputs(_FUSED_MIX_MAX_ROWS + 1, torch.bfloat16)
    assert not fused_hc_mix_supported(x, w_down, w_up)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 4, 7, _FUSED_MIX_MAX_ROWS])
def test_fused_hc_mix_fold_matches_reference(dtype, num_tokens):
    x, w_down, w_down_fold, w_up, wn = _make_fold_inputs(num_tokens, dtype)
    assert fused_hc_mix_fold_supported(x, w_down_fold, w_up)
    out, g = fused_hc_mix_fold(
        x, w_down_fold, w_up, wn, HC_COUNT, HIDDEN_SIZE, FOLD_EPS
    )
    ref, g_ref = _reference_mix_fold(
        x, w_down, w_up, wn, HC_COUNT, HIDDEN_SIZE, FOLD_EPS
    )
    torch.testing.assert_close(out.to(torch.float64), ref, **_TOLERANCES[dtype])
    # g is stored by a dedicated kernel pass and consumed by the fold combine
    # path only, so the mix-output check above would not catch a wrong store.
    torch.testing.assert_close(g.to(torch.float64), g_ref, rtol=1e-4, atol=1e-6)


def test_fused_hc_mix_fold_not_less_accurate_than_unfold():
    """Folding must not drift away from the fp64 reference: it removes the
    bf16 rounding of the normed residual, so its error vs the exact fold
    reference must stay below the unfused chain's (rmsnorm output materialized
    in bf16 before the GEMMs). Guards the fold algebra end to end: the sqrsum
    pre-pass, the g-scaled down-projection atomics, and the inline
    (1+w) * g gate application."""
    x, w_down, w_down_fold, w_up, wn = _make_fold_inputs(8, torch.bfloat16)
    ref, _ = _reference_mix_fold(x, w_down, w_up, wn, HC_COUNT, HIDDEN_SIZE, FOLD_EPS)
    fold, _ = fused_hc_mix_fold(
        x, w_down_fold, w_up, wn, HC_COUNT, HIDDEN_SIZE, FOLD_EPS
    )
    x3 = x.unflatten(-1, (HC_COUNT, HIDDEN_SIZE)).float()
    g = torch.rsqrt(x3.square().mean(-1) + FOLD_EPS)
    normed = (
        (x3 * g.unsqueeze(-1) * wn.float().unflatten(0, (HC_COUNT, HIDDEN_SIZE)))
        .flatten(-2)
        .to(torch.bfloat16)
    )
    unfold = fused_hc_mix(normed, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    fold_err = (fold.to(torch.float64) - ref).abs().max()
    unfold_err = (unfold.to(torch.float64) - ref).abs().max()
    assert fold_err <= unfold_err * 1.5 + 1e-6


def test_fused_hc_mix_fold_gate_rejects_prefill_rows():
    x, _, w_down_fold, w_up, _ = _make_fold_inputs(
        _FUSED_MIX_MAX_ROWS + 1, torch.bfloat16
    )
    assert not fused_hc_mix_fold_supported(x, w_down_fold, w_up)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
