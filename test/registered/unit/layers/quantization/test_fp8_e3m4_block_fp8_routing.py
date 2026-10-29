"""Routing contract for checkpoint-declared fp8_e3m4 block-FP8 weights.

FP8_PB_WO_E3M4 stores e3m4 codes (1+3+4, bias 3, max 30) inside the
float8_e4m3fn container. Every e4m3 kernel accepts those bytes and returns
plausible-looking garbage, so the only safety net is the declaration chain:
checkpoint algo -> ``Fp8Config.weight_dtype`` -> per-tensor ``e4m3_only`` tag
-> the backend-independent gate in ``Fp8LinearMethod.apply`` and
``cutlass_w8a8_block_fp8_linear_with_fallback``. These tests pin each link, plus
an SM120-only numeric check that the e3m4 kernels match a dequantized reference.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8_row_padded,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    e3m4_block_fp8_linear,
    uses_e3m4_weight,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    E3M4_BLOCK_FP8_ALGO,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

BLOCK = 128
# (mantissa bits, exponent bias) of the two formats sharing fp8_e4m3fn bytes.
E3M4 = (4, 3)
E4M3 = (3, 7)


def _codes(shape, generator):
    """Random fp8 code bytes, with the NaN/reserved patterns replaced."""
    c = torch.randint(0, 256, shape, generator=generator, dtype=torch.int64).to(
        torch.uint8
    )
    for nan_code in (0x7F, 0xFF):
        c[c == nan_code] = nan_code - 1
    return c


def _decode(codes, fmt):
    """fp8 code bytes -> float64 values under ``fmt`` = (mantissa bits, bias)."""
    mbits, bias = fmt
    c = codes.to(torch.int64)
    sign = torch.where((c >> 7) == 1, -1.0, 1.0)
    exp = (c >> mbits) & ((1 << (7 - mbits)) - 1)
    man = c & ((1 << mbits) - 1)
    normal = sign * (1.0 + man / (1 << mbits)) * 2.0 ** (exp - bias)
    subnormal = sign * man * 2.0 ** (1 - bias - mbits)
    return torch.where(exp == 0, subnormal, normal).to(torch.float64)


def _reference(x_codes, x_scale, w_codes, w_scale, w_fmt=E3M4):
    """Block-128 scaled dot product: e4m3 activations against ``w_fmt`` weights."""
    m, k = x_codes.shape
    n = w_codes.shape[0]
    x = _decode(x_codes, E4M3)
    w = _decode(w_codes, w_fmt)
    # [N/128, K/128] -> [K/128, N]: column nn carries w_scale[nn // 128, kb].
    w_full = w_scale.T.repeat_interleave(BLOCK, dim=1).to(torch.float64)
    x_scale = x_scale.to(torch.float64)
    y = torch.zeros(m, n, dtype=torch.float64)
    for kb in range(k // BLOCK):
        slab = (
            x[:, kb * BLOCK : (kb + 1) * BLOCK] @ w[:, kb * BLOCK : (kb + 1) * BLOCK].T
        )
        y += slab * x_scale[:, kb].unsqueeze(1) * w_full[kb].unsqueeze(0)
    return y


def _rel_l2(actual, expected):
    a = actual.double().cpu().reshape(-1)
    e = expected.double().cpu().reshape(-1)
    return float(((a - e).norm() / e.norm()).item())


def _tagged_weight(shape, tag):
    w = torch.nn.Parameter(torch.zeros(shape))
    if tag is not None:
        w.e4m3_only = tag
    return w


class TestE3m4Tag(CustomTestCase):
    """Only an explicit ``e4m3_only=False`` declaration means e3m4."""

    def test_untagged_and_e4m3_weights_are_e4m3(self):
        self.assertFalse(uses_e3m4_weight(_tagged_weight((2, 2), None)))
        self.assertFalse(uses_e3m4_weight(_tagged_weight((2, 2), True)))

    def test_declared_weight_is_e3m4(self):
        self.assertTrue(uses_e3m4_weight(_tagged_weight((2, 2), False)))


class TestE3m4Gate(CustomTestCase):
    """The gate returns None only for e4m3 weights; a declared weight never
    falls through to an e4m3 kernel."""

    def test_e4m3_weights_are_left_to_the_runner(self):
        for tag in (None, True):
            with self.subTest(tag=tag):
                self.assertIsNone(
                    e3m4_block_fp8_linear(
                        torch.zeros(2, 512),
                        _tagged_weight((256, 512), tag),
                        [BLOCK, BLOCK],
                        torch.zeros(2, 4),
                    )
                )

    def test_declared_weight_with_unsupported_layout_raises(self):
        w_scale = torch.zeros(2, 4)
        with self.assertRaisesRegex(RuntimeError, "128x128 blocks"):
            e3m4_block_fp8_linear(
                torch.zeros(2, 512),
                _tagged_weight((256, 512), False),
                [BLOCK // 2, BLOCK // 2],
                w_scale,
            )
        with self.assertRaisesRegex(RuntimeError, "128x128 blocks"):
            e3m4_block_fp8_linear(
                torch.zeros(2, 512),
                _tagged_weight((2, 256, 512), False),
                [BLOCK, BLOCK],
                w_scale,
            )

    def test_declared_weight_without_a_usable_kernel_raises(self):
        # Off SM120, or with an n/k the kernels do not implement: a hard error
        # instead of a silent e4m3 fallback.
        with (
            patch.object(fp8_utils, "_e3m4_block_linear", return_value=None),
            self.assertRaisesRegex(RuntimeError, "sm120_fp8_e3m4 kernels"),
        ):
            e3m4_block_fp8_linear(
                torch.zeros(2, 512),
                _tagged_weight((200, 512), False),
                [BLOCK, BLOCK],
                torch.zeros(2, 2),
            )


def _mixed_precision_config(algos, group_sizes=None):
    group_sizes = group_sizes or [BLOCK] * len(algos)
    return {
        "quant_algo": "MIXED_PRECISION",
        "packed_modules_mapping": {},
        "quantized_layers": {
            f"model.layers.{i}.mlp.fc1": {
                "quant_algo": algo,
                "group_size": group_sizes[i],
            }
            for i, algo in enumerate(algos)
        },
    }


def _resolve_block_weight_dtype(algos, env_flag=False):
    with patch.object(envs.SGLANG_USE_E3M4_KERNELS, "get", return_value=env_flag):
        config = ModelOptMixedPrecisionConfig.from_config(
            _mixed_precision_config(algos)
        )
    return config.fp8_block_config.weight_dtype


class TestE3m4ConfigDeclaration(CustomTestCase):
    """The checkpoint algo, not the environment, decides the code format."""

    def test_declared_algo_selects_e3m4(self):
        self.assertEqual(_resolve_block_weight_dtype([E3M4_BLOCK_FP8_ALGO]), "e3m4")

    def test_plain_block_fp8_stays_e4m3(self):
        for algo in ("FP8_PB_WO", "FP8_BLOCK_SCALES"):
            with self.subTest(algo=algo):
                self.assertEqual(_resolve_block_weight_dtype([algo]), "e4m3")

    def test_legacy_env_declares_plain_block_fp8_as_e3m4(self):
        self.assertEqual(
            _resolve_block_weight_dtype(["FP8_PB_WO"], env_flag=True), "e3m4"
        )

    def test_env_cannot_promote_a_checkpoint_without_block_layers(self):
        self.assertEqual(
            _resolve_block_weight_dtype(["W4A16_NVFP4"], env_flag=True), "e4m3"
        )

    def test_mixing_e4m3_and_e3m4_block_layers_raises(self):
        with self.assertRaisesRegex(ValueError, "mixes e4m3 and e3m4"):
            _resolve_block_weight_dtype([E3M4_BLOCK_FP8_ALGO, "FP8_PB_WO"])

    def test_block_layers_must_share_one_group_size(self):
        config = _mixed_precision_config(
            [E3M4_BLOCK_FP8_ALGO, E3M4_BLOCK_FP8_ALGO], group_sizes=[BLOCK, 64]
        )
        with self.assertRaisesRegex(ValueError, "one group_size"):
            ModelOptMixedPrecisionConfig.from_config(config)

    def test_e3m4_layers_route_to_the_block_fp8_linear_method(self):
        prefix = "model.layers.0.mlp.fc1"

        def layer_for(algos):
            with patch.object(envs.SGLANG_USE_E3M4_KERNELS, "get", return_value=False):
                config = ModelOptMixedPrecisionConfig.from_config(
                    _mixed_precision_config(algos)
                )
            from sglang.srt.layers.linear import ReplicatedLinear

            return ReplicatedLinear(
                8, 8, bias=False, quant_config=config, prefix=prefix
            )

        e3m4_layer = layer_for([E3M4_BLOCK_FP8_ALGO])
        self.assertIsInstance(e3m4_layer.quant_method, Fp8LinearMethod)
        self.assertEqual(e3m4_layer.quant_method.quant_config.weight_dtype, "e3m4")
        # The tag is applied by process_weights_after_loading, not create_weights.
        self.assertFalse(hasattr(e3m4_layer.weight, "e4m3_only"))

        e4m3_layer = layer_for(["FP8_PB_WO"])
        self.assertIsInstance(e4m3_layer.quant_method, Fp8LinearMethod)
        self.assertEqual(e4m3_layer.quant_method.quant_config.weight_dtype, "e4m3")

    def test_fp8_config_rejects_unknown_weight_dtype(self):
        with self.assertRaisesRegex(ValueError, "Unsupported weight_dtype"):
            Fp8Config(is_checkpoint_fp8_serialized=True, weight_dtype="e5m2")


class _RunnerCalled(Exception):
    """Raised by the stand-in e4m3 runner to prove apply reached it."""


def _block_linear_method(runner, weight_dtype="e4m3"):
    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.use_marlin = False
    method.use_mxfp8 = False
    method.block_fp8_as_mxfp8 = False
    method.mxfp8_dense_backend = None
    method.block_quant = True
    method.weight_block_size = [BLOCK, BLOCK]
    method.convert_mxfp8_to_block = False
    method.is_checkpoint_fp8_serialized = True
    method.w8a8_block_fp8_linear = runner
    method.quant_config = SimpleNamespace(
        weight_dtype=weight_dtype, weight_block_size=[BLOCK, BLOCK]
    )
    return method


class TestE3m4ApplyRouting(CustomTestCase):
    """A declared-e3m4 weight is intercepted before any e4m3 runner, whatever
    backend that runner would have been (CUTLASS, DeepGEMM, Triton, AMX)."""

    N, K = 256, 512

    def setUp(self):
        self.calls = []

        def runner(**kwargs):
            self.calls.append(kwargs)
            raise _RunnerCalled

        self.method = _block_linear_method(runner)
        self.x = torch.zeros(2, self.K, dtype=torch.bfloat16)

    def _layer(self, tag):
        return SimpleNamespace(
            weight=_tagged_weight((self.N, self.K), tag),
            weight_scale_inv=torch.zeros(self.N // BLOCK, self.K // BLOCK),
            block_fp8_mxfp8_ready=False,
        )

    def _fake_e3m4_path(self, record):
        def fake(input, weight, weight_scale, *, input_scale, bias, out_shape):
            record.append((input, weight, weight_scale, input_scale, bias))
            return torch.full(out_shape, 7.0, dtype=torch.bfloat16)

        return fake

    def test_declared_weight_never_reaches_the_e4m3_runner(self):
        record = []
        with (
            patch.object(
                fp8_utils,
                "_e3m4_block_linear",
                side_effect=self._fake_e3m4_path(record),
            ),
            patch.object(fp8_module, "use_intel_amx_backend", return_value=False),
        ):
            out = self.method.apply(self._layer(False), self.x)
        self.assertTrue(torch.all(out == 7.0).item())
        self.assertEqual(record[0][1].shape, (self.N, self.K))
        self.assertEqual(self.calls, [])

    def test_tuple_input_forwards_the_activation_scale(self):
        record = []
        x_fp8 = torch.zeros(2, self.K, dtype=torch.float8_e4m3fn)
        x_scale = torch.zeros(2, self.K // BLOCK)
        with (
            patch.object(
                fp8_utils,
                "_e3m4_block_linear",
                side_effect=self._fake_e3m4_path(record),
            ),
            patch.object(fp8_module, "use_intel_amx_backend", return_value=False),
        ):
            self.method.apply(self._layer(False), (x_fp8, x_scale))
        self.assertIs(record[0][0], x_fp8)
        self.assertIs(record[0][3], x_scale)

    def _e3m4_must_not_be_reached(self):
        return patch.object(
            fp8_utils,
            "_e3m4_block_linear",
            side_effect=AssertionError("e3m4 path taken for an e4m3 weight"),
        )

    def test_e4m3_weights_reach_the_runner_unchanged(self):
        for tag in (None, True):
            with (
                self.subTest(tag=tag),
                self._e3m4_must_not_be_reached(),
                patch.object(fp8_module, "use_intel_amx_backend", return_value=False),
                self.assertRaises(_RunnerCalled),
            ):
                self.method.apply(self._layer(tag), self.x)
            self.assertEqual(len(self.calls), 1)
            self.calls.clear()

    def test_env_flag_cannot_override_an_e4m3_tag(self):
        # The regression behind the garbled lm_head and shared expert: the legacy
        # flag must not re-decode weights that tagged themselves e4m3.
        with (
            patch.object(envs.SGLANG_USE_E3M4_KERNELS, "get", return_value=True),
            self._e3m4_must_not_be_reached(),
            patch.object(fp8_module, "use_intel_amx_backend", return_value=False),
            self.assertRaises(_RunnerCalled),
        ):
            self.method.apply(self._layer(True), self.x)
        self.assertTrue(self.calls[0]["weight"].e4m3_only)


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm120_supported(),
    "sm120_fp8_e3m4 kernels need an SM120/SM121 device",
)
class TestE3m4KernelNumerics(CustomTestCase):
    """End-to-end: a declared checkpoint tags its weights, ``apply`` routes them
    through the e3m4 kernels, and the result matches a dequantized reference."""

    N, K = 256, 512

    def setUp(self):
        generator = torch.Generator(device="cpu").manual_seed(1234)
        self.w_codes = _codes((self.N, self.K), generator).cuda()
        self.x_codes = _codes((5, self.K), generator).cuda()
        self.w_scale = (
            torch.rand(self.N // BLOCK, self.K // BLOCK, generator=generator) + 0.5
        ).cuda()
        self.x_scale = (
            torch.rand(5, self.K // BLOCK, generator=generator) + 0.5
        ).cuda()
        self.expected = _reference(
            self.x_codes.cpu(),
            self.x_scale.cpu(),
            self.w_codes.cpu(),
            self.w_scale.cpu(),
        )

        self.layer = SimpleNamespace(
            weight=torch.nn.Parameter(self.w_codes.view(torch.float8_e4m3fn)),
            weight_scale_inv=torch.nn.Parameter(self.w_scale),
            block_fp8_mxfp8_ready=False,
        )
        # The tag is what process_weights_after_loading_block_quant leaves behind
        # for an FP8_PB_WO_E3M4 checkpoint; the runner is a tripwire so nothing
        # but the e3m4 kernels can produce the result.
        self.method = _block_linear_method(self._tripwire, weight_dtype="e3m4")
        self.method.process_weights_after_loading_block_quant(self.layer)

    @staticmethod
    def _tripwire(**kwargs):
        raise AssertionError("e4m3 runner reached for a declared-e3m4 weight")

    def _apply(self, rows):
        x = (
            self.x_codes[:rows].view(torch.float8_e4m3fn),
            self.x_scale[:rows].contiguous(),
        )
        return self.method.apply(self.layer, x)

    def test_process_weights_tags_the_declared_weight(self):
        self.assertFalse(self.layer.weight.e4m3_only)
        self.assertTrue(uses_e3m4_weight(self.layer.weight))

    def test_gemv_path_m1(self):
        self.assertLess(_rel_l2(self._apply(1), self.expected[:1]), 1e-2)

    def test_gemm_path_m5(self):
        self.assertLess(_rel_l2(self._apply(5), self.expected), 1e-2)

    def test_bf16_input_is_quantized_then_decoded(self):
        # Server path: the gate quantizes the bf16 activations itself. Rebuilding
        # the reference from that same quantizer's output isolates the e3m4
        # decode from quantization rounding.
        x = (
            (
                _decode(self.x_codes.cpu(), E4M3)
                * self.x_scale.cpu().to(torch.float64).repeat_interleave(BLOCK, dim=1)
            )
            .to(torch.bfloat16)
            .cuda()
        )
        out = self.method.apply(self.layer, x)
        q, s = sglang_per_token_group_quant_fp8_row_padded(x, BLOCK)
        expected = _reference(
            q[: x.shape[0]].view(torch.uint8).cpu(),
            s[: x.shape[0]].cpu(),
            self.w_codes.cpu(),
            self.w_scale.cpu(),
        )
        self.assertLess(_rel_l2(out, expected), 1e-2)

    def test_e4m3_decoding_would_be_detected(self):
        # Fixture guard: the same bytes read as e4m3 must land far from the e3m4
        # reference, or the numeric tests could pass without e3m4 decode.
        misread = _reference(
            self.x_codes.cpu(),
            self.x_scale.cpu(),
            self.w_codes.cpu(),
            self.w_scale.cpu(),
            w_fmt=E4M3,
        )
        self.assertGreater(_rel_l2(misread, self.expected), 0.5)


if __name__ == "__main__":
    unittest.main()
