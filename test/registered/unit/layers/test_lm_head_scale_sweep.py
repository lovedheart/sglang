"""MSE-optimal block scales for the fp8 lm-head packers (the default; the
scale_sweep argument's "off" restores the plain amax recipe). Pins what makes
the feature safe: "off" reproduces the legacy amax scales exactly, a sweep
never increases the weight error, the stored scales stay consistent with the
stored codes (the kernel reads one and dequantizes the other), and the format
declarations the serving dispatch routes on are unchanged."""

import unittest

import torch

from sglang.srt.layers.fp8_lm_head import (
    _e3m4_tables,
    quantize_lm_head_to_e3m4_block128,
    quantize_lm_head_to_fp8_block128,
)
from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

N, K = 512, 256


def _heterogeneous_head(seed: int = 0) -> torch.nn.Module:
    """Rows with wildly different magnitudes, the case where a per-block scale
    search has something to win against (uniform rows make every recipe equal)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=g, dtype=torch.float32)
    gain = torch.logspace(-2, 2, N).unsqueeze(1)
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter((w * gain).to(torch.bfloat16).cuda(), requires_grad=False)
    return head


def _rel_err(head: torch.nn.Module, ref: torch.Tensor) -> float:
    """Weight error of the packed head vs the bf16 source, through the stored
    scales (exactly the values the kernel multiplies back)."""
    codes = head.weight.view(torch.uint8)
    scale = head.weight_block_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    if getattr(head, "weight_is_e3m4", False):
        pos, _ = _e3m4_tables()
        deq = pos.cuda()[codes.long() & 127] * torch.where(codes >= 128, -1.0, 1.0)
    else:
        deq = codes.view(torch.float8_e4m3fn).float()
    return ((deq * scale - ref.cuda()).norm() / ref.cuda().norm()).item()


def _packed_with(scale_sweep: str):
    head = _heterogeneous_head()
    ref = head.weight.detach().float().clone()
    assert quantize_lm_head_to_e3m4_block128(head, scale_sweep=scale_sweep)
    return head, ref


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm120_supported(), "SM120-only path"
)
class TestLmHeadScaleSweep(CustomTestCase):
    def test_off_matches_legacy_amax_recipe(self):
        """Default path is the legacy recipe exactly: scale = block amax / 30."""
        src = _heterogeneous_head().weight.detach()
        head, _ = _packed_with("off")
        amax = src.float().view(N // 128, 128, K // 128, 128).abs().amax(dim=(1, 3))
        torch.testing.assert_close(head.weight_block_scale, (amax / 30.0).float())

    def test_sweep_reduces_weight_error(self):
        base, ref = _packed_with("off")
        err_off = _rel_err(base, ref)
        for mode in ("block", "global"):
            head, ref = _packed_with(mode)
            err = _rel_err(head, ref)
            self.assertLess(
                err, err_off, f"{mode} sweep did not improve on amax scales ({err} vs {err_off})"
            )

    def test_scales_consistent_with_codes(self):
        """A swept scale must be the one the codes were actually encoded with."""
        head, _ = _packed_with("block")
        codes = head.weight.view(torch.uint8)
        scale = head.weight_block_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        pos, _ = _e3m4_tables()
        mag = pos.cuda()[codes.long() & 127] * scale
        block_max = mag.view(N // 128, 128, K // 128, 128).abs().amax(dim=(1, 3))
        # With mul > 1 the block max sits below the format max; it must never
        # exceed it (that would mean a value was clipped away).
        self.assertTrue(bool((block_max <= 30.001 * head.weight_block_scale + 1e-9).all()))

    def test_declarations_unchanged_by_sweep(self):
        head, _ = _packed_with("block")
        self.assertEqual(head.weight.dtype, torch.float8_e4m3fn)
        self.assertFalse(head.weight.e4m3_only)
        self.assertTrue(head.weight_is_e3m4)
        self.assertEqual(tuple(head.weight_block_scale.shape), (N // 128, K // 128))

    def test_e4m3_sweep_too(self):
        head = _heterogeneous_head()
        ref = head.weight.detach().float().clone()
        self.assertTrue(quantize_lm_head_to_fp8_block128(head, scale_sweep="block"))
        err_swept = _rel_err(head, ref)
        head2 = _heterogeneous_head()
        ref2 = head2.weight.detach().float().clone()
        self.assertTrue(quantize_lm_head_to_fp8_block128(head2, scale_sweep="off"))
        self.assertLess(err_swept, _rel_err(head2, ref2))


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm120_supported(), "SM120-only path"
)
class TestSweepIsDefault(CustomTestCase):
    def test_no_cli_flag_anymore(self):
        """The sweep has no flag: it is the default behaviour of the packers."""
        import argparse

        from sglang.srt.server_args import ServerArgs

        p = argparse.ArgumentParser()
        ServerArgs.add_cli_args(p)
        with self.assertRaises(SystemExit):
            p.parse_args(["--model-path", "x", "--lm-head-scale-sweep", "block"])

    def test_dispatch_sweeps_by_default(self):
        """quantize_lm_head() with no scale_sweep arg must beat the amax scales."""
        from sglang.srt.layers.lm_head_quant import quantize_lm_head

        swept = _heterogeneous_head()
        ref = swept.weight.detach().float().clone()
        self.assertTrue(quantize_lm_head(swept, "fp8_e3m4"))
        plain = _heterogeneous_head()
        ref2 = plain.weight.detach().float().clone()
        self.assertTrue(quantize_lm_head(plain, "fp8_e3m4", scale_sweep="off"))
        self.assertLess(_rel_err(swept, ref), _rel_err(plain, ref2))


if __name__ == "__main__":
    unittest.main()
