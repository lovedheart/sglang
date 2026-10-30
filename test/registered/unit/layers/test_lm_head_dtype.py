"""--lm-head-dtype: packer dispatch, the removed legacy boolean flags, and
the fp8_e3m4 online packer (declaration + numerics through the serving apply
path, including the M=1 padded b16 tile)."""

import argparse
import unittest

import torch

from sglang.srt.layers.fp8_lm_head import (
    _e3m4_tables,
    fp8_blockwise_lm_head_apply,
    quantize_lm_head_to_e3m4_block128,
    quantize_lm_head_to_fp8_block128,
)
from sglang.srt.layers.lm_head_quant import quantize_lm_head
from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestFlagsRemoved(CustomTestCase):
    """The legacy booleans are gone from both the CLI and the struct."""

    def test_cli_flags_unrecognized(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for flag in ("--enable-fp8-lm-head", "--enable-fp6-lm-head"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--model-path", "x", flag])
        self.assertEqual(
            parser.parse_args(
                ["--model-path", "x", "--lm-head-dtype", "fp8_e3m4"]
            ).lm_head_dtype,
            "fp8_e3m4",
        )

    def test_struct_fields_gone(self):
        import msgspec

        from sglang.srt.arg_groups.fields.exec_ import ExecFeatures

        names = {f.name for f in msgspec.structs.fields(ExecFeatures)}
        for name in ("enable_fp8_lm_head", "enable_fp6_lm_head"):
            self.assertNotIn(name, names)


class TestQuantizeDispatcher(CustomTestCase):
    def test_unknown_dtype_raises(self):
        with self.assertRaises(ValueError):
            quantize_lm_head(object(), "fp5_nope")

    def test_bf16_is_noop(self):
        class H:
            weight = None

        self.assertFalse(quantize_lm_head(H(), "bf16"))


class _Head(torch.nn.Module):
    def __init__(self, n: int, k: int, dtype: torch.dtype):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.weight = torch.nn.Parameter(
            torch.randn(n, k, generator=g, dtype=torch.float32).to(dtype).cuda(),
            requires_grad=False,
        )


def _dequant_e3m4(codes_u8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    pos, _ = _e3m4_tables()
    return (
        pos.float()[codes_u8.long() & 127]
        * torch.where(codes_u8 >= 128, -1.0, 1.0)
        * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    ).float()


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm120_supported(), "SM120-only path"
)
class TestE3m4LmHead(CustomTestCase):
    def test_pack_declaration_and_grid(self):
        n, k = 256, 256
        head = _Head(n, k, torch.bfloat16)
        self.assertTrue(quantize_lm_head_to_e3m4_block128(head))
        self.assertEqual(head.weight.dtype, torch.float8_e4m3fn)
        # the declaration that routes the weight to the e3m4 kernels
        self.assertTrue(hasattr(head.weight, "e4m3_only"))
        self.assertFalse(head.weight.e4m3_only)
        self.assertTrue(head.weight_is_e3m4)
        self.assertEqual(tuple(head.weight_block_scale.shape), (n // 128, k // 128))
        # every dequantized magnitude is an e3m4 grid point
        pos, _ = _e3m4_tables()
        grid = {round(v, 12) for v in pos.tolist()} | {0.0}
        codes = head.weight.view(torch.uint8).flatten().long().cpu()
        mags = {round(v, 12) for v in pos[codes & 127].tolist()}
        self.assertTrue(mags <= grid)

    def test_apply_matches_reference_at_decode_m(self):
        # M=1 decode pads to the 4-row tile the b16 kernel needs: the exact
        # serving dispatch of --lm-head-dtype fp8_e3m4 in greedy decode.
        n, k = 256, 256
        head = _Head(n, k, torch.bfloat16)
        self.assertTrue(quantize_lm_head_to_e3m4_block128(head))
        deq = _dequant_e3m4(
            head.weight.view(torch.uint8).cpu(), head.weight_block_scale.cpu()
        ).cuda()
        torch.manual_seed(2)
        for m in (1, 4):
            hidden = torch.randn(m, k, dtype=torch.bfloat16).cuda()
            got = fp8_blockwise_lm_head_apply(head, hidden)
            ref = hidden.float() @ deq.T
            # apply e4m3-quantizes the activations (per-128 group); that, not
            # the e3m4 codes, dominates the gap to this exact-dequant ref.
            rel = ((got.float() - ref).norm() / ref.norm()).item()
            self.assertLess(rel, 6e-2, (m, rel))
            self.assertTrue(torch.isfinite(got).all())

    def test_packer_guards(self):
        head = _Head(192, 256, torch.bfloat16)  # N % 128 fails
        self.assertFalse(quantize_lm_head_to_e3m4_block128(head))
        head = _Head(256, 256, torch.float32)  # dtype
        self.assertFalse(quantize_lm_head_to_e3m4_block128(head))

        head = _Head(256, 256, torch.bfloat16)
        self.assertTrue(quantize_lm_head_to_e3m4_block128(head))
        self.assertFalse(quantize_lm_head_to_e3m4_block128(head))  # idempotent
        # re-quantizing e3m4 bytes as e4m3 would corrupt them: refused
        self.assertFalse(quantize_lm_head_to_fp8_block128(head))

        head = _Head(256, 256, torch.bfloat16)
        self.assertTrue(quantize_lm_head_to_fp8_block128(head))
        self.assertFalse(quantize_lm_head_to_e3m4_block128(head))  # e4m3 wins


if __name__ == "__main__":
    unittest.main()
