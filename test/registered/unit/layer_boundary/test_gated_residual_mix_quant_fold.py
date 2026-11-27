"""Dispatch pins for GatedResidual.mix + the fused norm/quant fold.

Guards the wiring the kernel-level bit-identity tests cannot see:
- the per-branch norm must still be applied whenever the fold does NOT
  run (kill switch off, pre-Blackwell, decode-sized rows); a regression
  here feeds the mix an UN-normalized residual -- silently wrong output;
- with the fold on, mix() must produce bit-identical mixed inputs to the
  unfused norm -> quant -> mix pipeline, for both norm layouts
  (per-branch 2-D and the unflattened 3-D one);
- the fold must actually be attempted when it is eligible (otherwise the
  equivalence tests above would pass vacuously).
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm.hc_mix_fp8 import (
    _mxfp8_quant_act_kernel,
    fused_hc_mix_fp8_w8a8,
)
from sglang.srt.environ import envs
from sglang.srt.layers.fp8_hc_mix import quantize_linear_to_mxfp8
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HC, HS, LOWRANK = 4, 512, 64
K = HC * HS
ROWS = 512  # >= _W8A8_MIN_ROWS and K % 2048 == 0: w8a8 branch eligible


def _build(per_branch: bool, device: str) -> GatedResidual:
    torch.manual_seed(0)
    cfg = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=torch.bfloat16,
        hc_lowrank=LOWRANK,
        hc_per_branch_norm=per_branch,
    )
    gr = GatedResidual(cfg).to(device)
    # Model load runs under the bf16 default dtype; the module defaults its
    # norm weight to fp32, so mirror the real loaded dtype here.
    gr.hc_norm.weight.data = gr.hc_norm.weight.data.to(torch.bfloat16)
    # The real MXFP8 weight swap (same call the model loader makes).
    assert quantize_linear_to_mxfp8(gr.input_mix_weight_down)
    assert quantize_linear_to_mxfp8(gr.input_mix_weight_up)
    gr._mxfp8_mix = True
    gr._mix_up_weight_padded = None
    return gr


def _reference_mixed(gr: GatedResidual, x: torch.Tensor, per_branch: bool):
    """The unfused pipeline: plain norm, standalone quant launch, mix."""
    norm_in = x if per_branch else x.unflatten(-1, (HC, HS))
    normed = gr.hc_norm(norm_in)
    if not per_branch:
        normed = normed.flatten(-2)
    return fused_hc_mix_fp8_w8a8(
        normed,
        gr.input_mix_weight_down.weight,
        gr.input_mix_weight_down._mxfp8_scale,
        gr.input_mix_weight_up.weight,
        gr.input_mix_weight_up._mxfp8_scale,
        HC,
        HS,
    )


class TestGatedResidualMixQuantFold(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("MXFP8 quant epilogue needs SM100+")
        self.device = "cuda"

    def _probe(self, state):
        """Patch forward_quant_mxfp8 on this instance's class to record attempts."""
        cls = type(self.gr.hc_norm)
        orig = cls.forward_quant_mxfp8

        def probe(inner_self, t):
            out = orig(inner_self, t)
            if out is not None:
                state["attempted"] = True
            return out

        return patch.object(cls, "forward_quant_mxfp8", probe)

    def _run_case(self, per_branch: bool):
        self.gr = _build(per_branch, self.device)
        g = torch.Generator(device=self.device).manual_seed(2)
        x = torch.randn(ROWS, K, generator=g, device=self.device).to(torch.bfloat16)
        ref = _reference_mixed(self.gr, x, per_branch)

        state = {"attempted": False}
        with envs.SGLANG_HC_NORM_QUANT_FOLD.override(True), self._probe(state):
            mixed_fold, _ = self.gr.mix(x)
        self.assertTrue(
            state["attempted"],
            "fold was eligible but never attempted (per_branch="
            f"{per_branch}); the checks below would pass vacuously",
        )
        self.assertTrue(
            torch.equal(mixed_fold.view(torch.uint8), ref.view(torch.uint8)),
            f"mixed output diverged with the fold on (per_branch={per_branch})",
        )

        # Kill switch off: the plain norm path must still run, agree bit for
        # bit, and the fold must not be attempted.
        state_off = {"attempted": False}
        with envs.SGLANG_HC_NORM_QUANT_FOLD.override(False), self._probe(state_off):
            mixed_off, _ = self.gr.mix(x)
        self.assertFalse(state_off["attempted"], "fold ran despite the kill switch")
        self.assertTrue(
            torch.equal(mixed_off.view(torch.uint8), ref.view(torch.uint8)),
            f"norm path broken with the fold off (per_branch={per_branch})",
        )

    def test_per_branch_norm(self):
        self._run_case(per_branch=True)

    def test_flat_norm(self):
        # Flat norm only takes the rmsnorm+quant fold now; the decode/e3m4
        # fold machinery is gated to per-branch configs (its folded weight
        # prep assumes a K-sized gain vector).
        self._run_case(per_branch=False)

    def test_fold_quant_pin(self):
        # The fold's (q, sf) must equal the standalone quantizer on normed,
        # reached through the real norm module (eps/group_size taken from it).
        import triton

        self.gr = _build(True, self.device)
        g = torch.Generator(device=self.device).manual_seed(3)
        x = torch.randn(ROWS, K, generator=g, device=self.device).to(torch.bfloat16)
        normed, a8, asf = self.gr.hc_norm.forward_quant_mxfp8(x)
        normed_ref = self.gr.hc_norm(x)
        rows, k = normed_ref.shape
        q_ref = torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=self.device)
        sf_ref = torch.empty(rows, k // 32, dtype=torch.uint8, device=self.device)
        _mxfp8_quant_act_kernel[(rows, triton.cdiv(k, 1024))](
            normed_ref, q_ref, sf_ref, rows, K=k, BLOCK=1024, num_warps=4
        )
        self.assertTrue(
            torch.equal(normed.view(torch.uint8), normed_ref.view(torch.uint8))
        )
        self.assertTrue(torch.equal(a8.view(torch.uint8), q_ref.view(torch.uint8)))
        self.assertTrue(torch.equal(asf, sf_ref))


if __name__ == "__main__":
    unittest.main()
