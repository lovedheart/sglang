"""--enable-fp8-weights: e3m4 b128 online quantization semantics.

Covers the three contracts the model load hook relies on: the encoder is
exact on representable codes and ~37 dB on real-magnitude noise, the apply
path (block-FP8 linear intercepting an e4m3_only=False weight) runs the
sm120 e3m4 kernels and stays close to the bf16 reference, and the name/shape
guards never touch excluded or non-128-multiple layers.
"""

import unittest

import torch
from sglang.srt.layers.fp8_e3m4_weights import (
    _is_eligible,
    dequant_e3m4,
    quantize_linear_to_fp8_e3m4_block128,
)
from sglang.srt.layers.fp8_lm_head import _e3m4_tables, _encode_e3m4
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.utils.device import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class _Lin(torch.nn.Module):
    def __init__(self, n: int, k: int, dtype=torch.bfloat16):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.weight = torch.nn.Parameter(
            torch.randn(n, k, generator=g, dtype=torch.float32).to(dtype).cuda(),
            requires_grad=False,
        )


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm120_supported(), "SM120-only path"
)
class TestFp8E3M4WeightsGpu(CustomTestCase):
    def test_encoder_exact_on_representable_codes(self):
        pos, _ = _e3m4_tables()
        vals = pos.cuda()
        codes = _encode_e3m4(vals)
        self.assertTrue(torch.equal(pos[codes.cpu().to(torch.long)].cuda(), vals))

    def test_quantize_dequant_roundtrip_sqnr(self):
        lin = _Lin(1024, 512)
        weight = lin.weight.detach().clone()
        self.assertTrue(quantize_linear_to_fp8_e3m4_block128(lin))
        self.assertEqual(lin.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(lin.weight_block_scale.shape), (8, 4))
        self.assertFalse(lin.weight.e4m3_only)
        wq = dequant_e3m4(lin.weight.view(torch.uint8).cuda(), lin.weight_block_scale)
        rel = ((wq - weight.float()).norm() / weight.float().norm()).item()
        # 4-bit mantissa: ~37.6 dB, far from e4m3's ~31.6 dB.
        self.assertLess(rel, 0.016)

    def test_apply_matches_bf16_reference(self):

        n, k, m = 1024, 512, 4
        lin = _Lin(n, k)
        weight = lin.weight.detach().clone()
        self.assertTrue(quantize_linear_to_fp8_e3m4_block128(lin))
        torch.manual_seed(1)
        x = torch.randn(m, k, dtype=torch.bfloat16).cuda()
        got = lin.quant_method.apply(lin, x)
        ref = x.float() @ weight.float().T
        self.assertEqual(got.dtype, torch.bfloat16)
        rel_err = ((got.float() - ref).norm() / ref.norm()).item()
        self.assertLess(rel_err, 0.03)  # bf16 act-quant + e3m4 weights
        self.assertTrue(torch.isfinite(got).all())


class TestFp8E3M4WeightsGuards(CustomTestCase):
    def test_shape_and_dtype_guards(self):
        for n, k in [(192, 256), (256, 200), (576, 3072)]:
            lin = torch.nn.Linear(k, n, dtype=torch.bfloat16)
            self.assertFalse(quantize_linear_to_fp8_e3m4_block128(lin))
            self.assertEqual(lin.weight.dtype, torch.bfloat16)
        lin = torch.nn.Linear(256, 256, dtype=torch.float32)
        self.assertFalse(quantize_linear_to_fp8_e3m4_block128(lin))

    def test_name_selection(self):
        cases = {
            "model.layers.0.self_attn.q_b_proj": True,
            "model.layers.0.self_attn.o_proj": True,
            "model.layers.0.self_attn.kv_b_proj": False,  # w_kc absorb trap
            "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa": False,
            "model.layers.0.mlp.gate_up_proj": True,
            "model.layers.0.mlp.down_proj": True,
            "model.layers.0.mlp.shared_experts.gate_up_proj": True,
            "model.layers.0.mlp.experts.gate_up_proj": False,  # W3A16 owns it
            "model.dfly_layers.0.self_attn.q_b_proj": False,
            "model.layers.0.self_attn.indexer.q_proj": False,
            "visual.blocks.0.mlp.gate_up_proj": False,
            "lm_head": False,
        }
        owners = {
            "q_b_proj": ColumnParallelLinear,
            "gate_up_proj": MergedColumnParallelLinear,
            "down_proj": RowParallelLinear,
            "o_proj": RowParallelLinear,
        }
        for name, expected in cases.items():
            leaf = name.rsplit(".", 1)[-1]
            cls = owners.get(leaf, ColumnParallelLinear)
            # __new__ without __init__: a typed handle for the isinstance gate,
            # no dist/process-group needed.
            mod = cls.__new__(cls)
            self.assertEqual(
                _is_eligible(name, mod), expected, f"eligibility of {name}"
            )

    def test_pass_skips_non_bf16_and_reports_count(self):
        from unittest.mock import patch

        from sglang.srt.layers.fp8_e3m4_weights import (
            quantize_hy4_dense_weights_fp8_e3m4,
        )

        root = torch.nn.Module()
        root.layers = torch.nn.ModuleList()
        layer = torch.nn.Module()
        g = torch.Generator().manual_seed(0)

        class _Col(torch.nn.Module):
            pass

        # A bf16 "eligible" linear with real parameter but unquantizable
        # dtype path: plain bf16 weights quantize; an fp8 container skips.
        col = _Col()
        col.weight = torch.nn.Parameter(
            torch.randn(256, 256, generator=g, dtype=torch.float32).to(torch.bfloat16),
            requires_grad=False,
        )
        import sglang.srt.layers.fp8_e3m4_weights as m

        with (
            patch.object(m, "is_sm120_supported", return_value=True),
            patch.object(m, "_LINEAR_CLASSES", (_Col,)),
        ):
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_b_proj = col
            root.layers.append(layer)
            n = quantize_hy4_dense_weights_fp8_e3m4(root)
        self.assertEqual(n, 1)
        self.assertEqual(
            col.weight.dtype, torch.float8_e3m4n if False else torch.float8_e4m3fn
        )
        # Second pass: no bf16 weights left, nothing more to do.
        with (
            patch.object(m, "is_sm120_supported", return_value=True),
            patch.object(m, "_LINEAR_CLASSES", (_Col,)),
        ):
            self.assertEqual(quantize_hy4_dense_weights_fp8_e3m4(root), 0)


if __name__ == "__main__":
    unittest.main()
