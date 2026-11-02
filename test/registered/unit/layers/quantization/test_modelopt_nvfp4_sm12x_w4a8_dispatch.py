"""CPU unit tests for the sm12x_nvfp4_w4a8 MoE backend selection in ModelOpt NVFP4.

Mirrors test_modelopt_nvfp4_moe_dispatch.py for the SM12x W4A8 arm: the
backend must resolve from the CLI/enum, the quant method must prepare the
canonical (unswizzled) NVFP4 payload with a [E, 2] gate/up global scale, and
apply() must hand the runner an Sm12xNvfp4W4A8MoeQuantInfo over the layer's own
storage. The kernel numerics live on-device in
test/registered/kernels/ops/moe/test_sm12x_nvfp4_w4a8.py.

The platform check and the runner are stubbed and the layer is a bag of small
tensors, so the tests stay on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

# Import modelopt_quant before the runner modules (see
# test_modelopt_nvfp4_moe_scales.py for the circular-import reason).
# isort: off
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.moe.moe_runner.sm12x_nvfp4_w4a8 import (
    Sm12xNvfp4W4A8MoeQuantInfo,
    fused_experts_none_to_sm12x_nvfp4_w4a8,
)

# isort: on
from sglang.srt.arg_groups.choices import MOE_RUNNER_BACKEND_CHOICES
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.test_utils import CustomTestCase

BACKEND = MoeRunnerBackend.SM12X_NVFP4_W4A8
NUM_EXPERTS = 4
HIDDEN = 64
INTERMEDIATE = 32


class _Runner:
    """Records the quant_info apply() hands to the runner."""

    def __init__(self):
        self.calls = []

    def run(self, dispatch_output, quant_info):
        self.calls.append((dispatch_output, quant_info))
        return "combine-input"


def _p(t: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(t, requires_grad=False)


def _raw_layer(gate_scale: torch.Tensor = None) -> SimpleNamespace:
    """A layer as the checkpoint loader leaves it: packed FP4 weights, raw
    (unswizzled) row-major FP8 block scales, [E, 2] gate/up global scales, and
    the swizzled placeholders create_weights pre-allocated for FlashInfer."""
    gate = (
        torch.full((NUM_EXPERTS,), 0.25, dtype=torch.float32)
        if gate_scale is None
        else gate_scale
    )
    up = torch.full((NUM_EXPERTS,), 0.5, dtype=torch.float32)
    return SimpleNamespace(
        w13_weight=_p(
            torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8)
        ),
        w2_weight=_p(
            torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8)
        ),
        w13_weight_scale=_p(
            torch.zeros(
                NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16, dtype=torch.float8_e4m3fn
            )
        ),
        w2_weight_scale=_p(
            torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 16, dtype=torch.float8_e4m3fn)
        ),
        w13_weight_scale_2=_p(torch.stack([gate, up], dim=1)),
        w2_weight_scale_2=_p(torch.full((NUM_EXPERTS,), 0.125, dtype=torch.float32)),
        w13_blockscale_swizzled=_p(torch.zeros(1)),
        w2_blockscale_swizzled=_p(torch.zeros(1)),
        expert_map=None,
        _mega_moe_nvfp4=False,
        moe_runner_config=MoeRunnerConfig(
            num_experts=NUM_EXPERTS,
            num_local_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            activation="silu",
            is_gated=True,
        ),
    )


def _method(backend: MoeRunnerBackend = BACKEND) -> ModelOptNvFp4FusedMoEMethod:
    """The method as create_moe_runner leaves it, with a recording runner."""
    with (
        mock.patch.object(
            modelopt_quant, "get_moe_runner_backend", return_value=backend
        ),
        mock.patch.object(
            modelopt_quant, "get_platform", return_value=SimpleNamespace(is_blackwell=True)
        ),
        mock.patch.object(modelopt_quant, "is_cuda", return_value=False),
        mock.patch.object(modelopt_quant, "get_device_capability", return_value=(12, 0)),
    ):
        method = ModelOptNvFp4FusedMoEMethod(
            SimpleNamespace(use_per_token_activation=False)
        )
        method.create_moe_runner(
            SimpleNamespace(),
            MoeRunnerConfig(
                num_experts=NUM_EXPERTS,
                num_local_experts=NUM_EXPERTS,
                hidden_size=HIDDEN,
                intermediate_size_per_partition=INTERMEDIATE,
                activation="silu",
                is_gated=True,
            ),
        )
    method.runner = _Runner()
    return method


class TestSm12xW4A8Selection(CustomTestCase):
    def test_backend_name_parses_and_is_a_cli_choice(self):
        self.assertEqual(MoeRunnerBackend("sm12x_nvfp4_w4a8"), BACKEND)
        self.assertTrue(BACKEND.is_sm12x_nvfp4_w4a8())
        self.assertFalse(BACKEND.is_sm12x_nvfp4_bf16())
        self.assertIn("sm12x_nvfp4_w4a8", MOE_RUNNER_BACKEND_CHOICES)

    def test_runner_selects_the_registered_fused_func(self):
        runner = MoeRunner(
            BACKEND,
            MoeRunnerConfig(
                num_experts=NUM_EXPERTS,
                num_local_experts=NUM_EXPERTS,
                hidden_size=HIDDEN,
                intermediate_size_per_partition=INTERMEDIATE,
                activation="silu",
                is_gated=True,
            ),
        )
        self.assertIsNone(runner.runner_core)  # fused-path only
        self.assertIs(runner.fused_func, fused_experts_none_to_sm12x_nvfp4_w4a8)

    def test_apply_builds_w4a8_quant_info_from_layer_storage(self):
        method = _method()
        layer = _raw_layer()
        dispatch_output = object()

        out = method.apply(layer, dispatch_output)

        self.assertEqual(out, "combine-input")
        ((seen_dispatch, quant_info),) = method.runner.calls
        self.assertIs(seen_dispatch, dispatch_output)
        self.assertIsInstance(quant_info, Sm12xNvfp4W4A8MoeQuantInfo)
        for seen, param in (
            (quant_info.w13, layer.w13_weight),
            (quant_info.w13_sf, layer.w13_weight_scale),
            (quant_info.w13_gs, layer.w13_weight_scale_2),
            (quant_info.w2, layer.w2_weight),
            (quant_info.w2_sf, layer.w2_weight_scale),
            (quant_info.w2_gs, layer.w2_weight_scale_2),
        ):
            self.assertEqual(seen.data_ptr(), param.data_ptr())

    def test_apply_rejects_expert_map(self):
        method = _method()
        layer = _raw_layer()
        layer.expert_map = torch.arange(NUM_EXPERTS)
        with self.assertRaisesRegex(NotImplementedError, "expert_map"):
            method.apply(layer, object())

    def test_create_moe_runner_requires_sm12x(self):
        with (
            mock.patch.object(
                modelopt_quant, "get_moe_runner_backend", return_value=BACKEND
            ),
            mock.patch.object(
                modelopt_quant,
                "get_platform",
                return_value=SimpleNamespace(is_blackwell=True),
            ),
            mock.patch.object(modelopt_quant, "is_cuda", return_value=False),
            mock.patch.object(
                modelopt_quant, "get_device_capability", return_value=(10, 0)
            ),
        ):
            method = ModelOptNvFp4FusedMoEMethod(
                SimpleNamespace(use_per_token_activation=False)
            )
            with self.assertRaisesRegex(NotImplementedError, "SM120/SM121"):
                method.create_moe_runner(
                    SimpleNamespace(),
                    MoeRunnerConfig(
                        num_experts=NUM_EXPERTS,
                        num_local_experts=NUM_EXPERTS,
                        hidden_size=HIDDEN,
                        intermediate_size_per_partition=INTERMEDIATE,
                        activation="silu",
                        is_gated=True,
                    ),
                )


class TestSm12xW4A8WeightPrep(CustomTestCase):
    def _prepared(self, **layer_kwargs):
        method = _method()
        layer = _raw_layer(**layer_kwargs)
        method.moe_runner_config = layer.moe_runner_config
        method._moe_runner_backend = BACKEND
        method.process_weights_after_loading(layer)
        return layer

    def test_scales_stay_raw_and_swizzled_placeholders_are_freed(self):
        layer = self._prepared()
        # The Triton kernel indexes block scales in the raw row-major layout.
        self.assertEqual(tuple(layer.w13_weight_scale.shape), (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16))
        self.assertEqual(tuple(layer.w2_weight_scale.shape), (NUM_EXPERTS, HIDDEN, INTERMEDIATE // 16))
        self.assertIsNone(layer.w13_blockscale_swizzled)
        self.assertIsNone(layer.w2_blockscale_swizzled)

    def test_gate_up_global_scales_stay_separate(self):
        layer = self._prepared()
        self.assertEqual(tuple(layer.w13_weight_scale_2.shape), (NUM_EXPERTS, 2))
        torch.testing.assert_close(
            layer.w13_weight_scale_2,
            torch.stack(
                [
                    torch.full((NUM_EXPERTS,), 0.25),
                    torch.full((NUM_EXPERTS,), 0.5),
                ],
                dim=1,
            ),
        )
        # No gate/up ratio fold: the kernel applies both inside the activation.
        torch.testing.assert_close(
            layer.w2_weight_scale_2,
            torch.full((NUM_EXPERTS,), 0.125),
        )

    def test_zero_gate_scale_falls_back_to_the_up_scale(self):
        layer = self._prepared(
            gate_scale=torch.zeros(NUM_EXPERTS, dtype=torch.float32)
        )
        torch.testing.assert_close(
            layer.w13_weight_scale_2[:, 0], torch.full((NUM_EXPERTS,), 0.5)
        )

    def test_single_global_scale_broadcasts_to_both_halves(self):
        method = _method()
        layer = _raw_layer()
        layer.w13_weight_scale_2 = _p(torch.full((NUM_EXPERTS,), 0.25, dtype=torch.float32))
        method.moe_runner_config = layer.moe_runner_config
        method._moe_runner_backend = BACKEND
        method.process_weights_after_loading(layer)

        self.assertEqual(tuple(layer.w13_weight_scale_2.shape), (NUM_EXPERTS, 2))
        torch.testing.assert_close(
            layer.w13_weight_scale_2,
            torch.full((NUM_EXPERTS, 2), 0.25),
        )


if __name__ == "__main__":
    unittest.main()
