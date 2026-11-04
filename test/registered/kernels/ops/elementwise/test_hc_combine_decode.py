"""Correctness for the HC decode gate/apply fusion (side-stream path).

Mirrors what GatedResidual does on sm_100/sm_120 decode: mix() launches the
inject-gate dot on a side stream, and combine() finishes with the apply-only
kernel. The fused result must be bit-identical to the stock combine, and
combine_and_normalize must equal combine + the next per-branch norm.
"""

import torch

from sglang.kernels.ops.elementwise.hc_combine_decode import (
    hc_combine_apply,
    hc_combine_apply_norm,
    hc_combine_gate,
)
from sglang.srt.layers.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
    _hc_decode_fusion_enabled,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

HC_COUNT = 4
HIDDEN_SIZE = 2560


def _make_layer():
    config = HyperConnectionConfig(
        hc_count=HC_COUNT,
        hidden_size=HIDDEN_SIZE,
        params_dtype=torch.bfloat16,
        hc_lowrank=320,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )
    layer = GatedResidual(config, use_mix=True, use_combine=True).cuda()
    layer = layer.to(torch.bfloat16)
    # The fusion needs a materialized normed residual; disable norm folding so
    # mix() carries one (the fold path rides g factors instead of partials).
    layer._norm_fold_ok = False
    layer._mxfp8_mix = False
    return layer


class TestHcCombineDecode(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        if not _hc_decode_fusion_enabled():
            self.skipTest("HC decode fusion disabled on this device")
        torch.manual_seed(0)
        self.layer = _make_layer()
        self.assertIsNotNone(
            self.layer._gate_stream,
            "gate stream must exist once the fusion is enabled",
        )

    def _inputs(self, rows):
        hyper = torch.randn(rows, HC_COUNT * HIDDEN_SIZE, device="cuda").to(
            torch.bfloat16
        )
        block = torch.randn(rows, HIDDEN_SIZE, device="cuda").to(torch.bfloat16)
        return hyper, block

    def test_gate_partials_are_emitted_by_mix(self):
        hyper, _ = self._inputs(1)
        _, residuals = self.layer.mix(hyper)
        state = residuals[2]
        self.assertIsNotNone(state)
        self.assertEqual(tuple(state.shape), (1, 8, HC_COUNT))

    def test_mix_emits_no_partials_above_one_row(self):
        # The side-stream gate is decode-only: above a single row the stock
        # combine path must stay in charge.
        hyper, _ = self._inputs(2)
        _, residuals = self.layer.mix(hyper)
        self.assertIsNone(residuals[2])

    def test_apply_matches_stock_combine(self):
        # The fusion applies at decode sizes: rows 1..4 are the CUDA-graph
        # decode tiers; partials come from the gate kernel directly.
        for rows in (1, 2, 3, 4):
            hyper, block = self._inputs(rows)
            _, residuals = self.layer.mix(hyper)
            _, normed, _ = residuals
            partials = hc_combine_gate(normed, self.layer.block_inject_weight.weight)
            stock = self.layer.combine(block, (hyper, normed, None))
            fused = hc_combine_apply(block, hyper, partials)
            torch.testing.assert_close(fused, stock, rtol=0, atol=0)

    def test_apply_norm_matches_combine_plus_norm(self):
        for rows in (1, 4):
            hyper, block = self._inputs(rows)
            _, residuals = self.layer.mix(hyper)
            _, normed, _ = residuals
            partials = hc_combine_gate(normed, self.layer.block_inject_weight.weight)
            combined, normalized = hc_combine_apply_norm(
                block,
                hyper,
                partials,
                self.layer.hc_norm.weight,
                self.layer.hc_norm.variance_epsilon,
            )
            stock = self.layer.combine(block, (hyper, normed, None))
            torch.testing.assert_close(combined, stock, rtol=0, atol=0)
            torch.testing.assert_close(
                normalized, self.layer.hc_norm(stock), rtol=0, atol=0
            )

    def test_combine_and_normalize_uses_the_fused_path(self):
        hyper, block = self._inputs(1)
        _, residuals = self.layer.mix(hyper)
        out, normalized = self.layer.combine_and_normalize(
            block, residuals, self.layer.hc_norm
        )
        _, normed, _ = residuals
        stock = self.layer.combine(block, (hyper, normed, None))
        torch.testing.assert_close(out, stock, rtol=0, atol=0)
        torch.testing.assert_close(
            normalized, self.layer.hc_norm(stock), rtol=0, atol=0
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
