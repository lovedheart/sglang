"""Unit tests for the TS4 (w2a16) quantization primitives and CPU kernel.

Run: python3 test/registered/moe/test_w2a16_ts4.py
"""

import unittest

import torch

from sglang.srt.layers.moe.w2a16 import ts4
from sglang.srt.layers.moe.w2a16.native import cpu_supports_kernel, get_native


def _reference_forward(x, ids, tw, w13, s13, w2, s2, H, I, K):
    """Numerical contract of ts4_a16_ffn: fp32 gate/up dots, h1 rounded to
    fp16 once, fp32 stage-B dots, silu clamped to +/-20."""
    T = x.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32)
    d13 = ts4.ts4_dequant(w13, s13, 1.0)
    d2 = ts4.ts4_dequant(w2, s2, 1.0)
    flat = ids.flatten().tolist()
    for pair in range(T * K):
        t = pair // K
        e = flat[pair]
        xe = x[t : t + 1].float()
        g = xe @ d13[e, :I].transpose(0, 1)
        u = xe @ d13[e, I:].transpose(0, 1)
        gc = g.clamp(-20.0, 20.0)
        h1 = (g / (1.0 + torch.exp(-gc)) * u).half()  # one fp16 rounding
        out[t] += tw[t, pair % K] * (h1.float() @ d2[e].transpose(0, 1))[0]
    return out


class TestTs4Quant(unittest.TestCase):
    def test_roundtrip_identity_levels(self):
        w = torch.randn(2, 32).clamp(-4, 4)
        codes, sc = ts4.quantize_ts4(w)
        deq = ts4.ts4_dequant(ts4.codes_to_plane(codes), sc, 1.0)
        rel = (deq - w).norm() / w.norm()
        self.assertLess(rel.item(), 0.5)

    def test_quantize_error_bound(self):
        torch.manual_seed(0)
        w = torch.randn(64, 128)
        codes, sc = ts4.quantize_ts4(w)
        deq = ts4.ts4_dequant(ts4.codes_to_plane(codes), sc, 1.0)
        rel = (deq - w).norm() / w.norm()
        self.assertLess(rel.item(), 0.5)  # 2-bit quintal: ~0.3 typical

    def test_plane_packing(self):
        codes = torch.randint(0, 4, (3, 8), dtype=torch.uint8)
        plane = ts4.codes_to_plane(codes)
        self.assertEqual(plane.shape, (3, 2))
        back = ts4.plane_to_codes(plane)
        self.assertTrue(torch.equal(back, codes))


@unittest.skipUnless(cpu_supports_kernel(), "host lacks AVX-512 / built kernel")
class TestCpuKernel(unittest.TestCase):
    def test_against_contract(self):
        lib, fn = get_native()
        torch.manual_seed(3)
        E, H, I, T, K = 4, 1536, 512, 2, 8
        w13 = torch.randn(E, 2 * I, H) * 0.05
        w2 = torch.randn(E, H, I) * 0.05
        c13, s13 = ts4.quantize_ts4(w13)
        p13 = ts4.codes_to_plane(c13).contiguous()
        c2, s2 = ts4.quantize_ts4(w2)
        p2 = ts4.codes_to_plane(c2).contiguous()
        x = torch.randn(T, H)
        ids = torch.randint(0, E, (T, K), dtype=torch.int32)
        tw = torch.rand(T, K)
        tw /= tw.sum(-1, keepdim=True)
        xh = x.half().contiguous()
        out = torch.zeros(T, H)
        fn(
            xh.data_ptr(), ids.data_ptr(), tw.data_ptr(),
            p13.data_ptr(), s13.data_ptr(), 1.0,
            p2.data_ptr(), s2.data_ptr(), 1.0,
            T, K, E, H, I, out.data_ptr(), 4,
        )
        ref = _reference_forward(x, ids, tw, p13, s13, p2, s2, H, I, K)
        rel = (out - ref).abs().max() / ref.abs().max()
        self.assertLess(rel.item(), 5e-4)

    def test_row_padding_sentinel(self):
        # ids beyond E-1 must not fault (kernel clamps/skips masked rows)
        lib, fn = get_native()
        E, H, I, T, K = 2, 64, 32, 1, 8
        c13, s13 = ts4.quantize_ts4(torch.randn(E, 2 * I, H) * 0.05)
        c2, s2 = ts4.quantize_ts4(torch.randn(E, H, I) * 0.05)
        p13 = ts4.codes_to_plane(c13).contiguous()
        p2 = ts4.codes_to_plane(c2).contiguous()
        xh = torch.randn(T, H).half().contiguous()
        ids = torch.full((T, K), E - 1, dtype=torch.int32)
        tw = torch.ones(T, K) / K
        out = torch.zeros(T, H)
        fn(
            xh.data_ptr(), ids.data_ptr(), tw.data_ptr(),
            p13.data_ptr(), s13.data_ptr(), 1.0,
            p2.data_ptr(), s2.data_ptr(), 1.0,
            T, K, E, H, I, out.data_ptr(), 2,
        )
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
