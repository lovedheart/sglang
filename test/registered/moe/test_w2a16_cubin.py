"""Unit tests for the SM12x cubin prefill arm (vLLM-Moet moe_w2_mm)."""

import unittest

import torch

from sglang.srt.layers.moe.w2a16.cubin import (
    cubin_forward,
    cubin_k_for,
    pack_planes_for_cubin,
    shapes_supported,
)

_MC4_KS = [384, 512, 768, 1024, 2048, 3584, 4096, 6144, 7168]


def _rowmajor(codes):
    b = torch.zeros(*codes.shape[:-1], codes.shape[-1] // 4,
                    dtype=torch.uint8, device=codes.device)
    for j in range(4):
        b |= codes[..., j::4] << (2 * j)
    return b


class TestCubinSelect(unittest.TestCase):
    def test_k_selection(self):
        self.assertEqual(cubin_k_for(1536, _MC4_KS), 2048)
        self.assertEqual(cubin_k_for(512, _MC4_KS), 512)
        self.assertEqual(cubin_k_for(1537, _MC4_KS), 2048)

    def test_shapes_supported(self):
        self.assertTrue(shapes_supported(1536, 512))   # Ling-3.0-tiny


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestCubinForward(unittest.TestCase):
    def test_forward_matches_exact_math(self):
        torch.manual_seed(9)
        dev = "cuda"
        e, h, i, t, topk = 8, 1536, 512, 20, 8
        k13, k2 = 2048, 512
        try:
            pack_planes_for_cubin(
                torch.zeros(1, 2 * i, h // 4, dtype=torch.uint8, device=dev),
                torch.zeros(1, 2 * i, h // 32, dtype=torch.uint8, device=dev),
                torch.zeros(1, h, i // 4, dtype=torch.uint8, device=dev),
                torch.zeros(1, h, i // 32, dtype=torch.uint8, device=dev),
                1, h, i, k13, k2, dev)
        except RuntimeError:
            self.skipTest("moe_w2_mm cubins need an sm_120 GPU")
        lut = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=dev)

        def deq(codes, sc, k):
            cp = torch.cat([codes, torch.zeros(*codes.shape[:-1],
                                               k - codes.shape[-1],
                                               dtype=torch.uint8, device=dev)], -1)
            sp = torch.cat([sc, torch.ones(*sc.shape[:-1],
                                           k // 32 - sc.shape[-1],
                                           dtype=torch.uint8, device=dev)], -1)
            return lut[cp.long()] * torch.exp2(
                sp.float() - 127).repeat_interleave(32, dim=-1)

        c13 = torch.randint(0, 4, (e, 2 * i, h), dtype=torch.uint8, device=dev)
        s13 = torch.randint(123, 126, (e, 2 * i, h // 32), dtype=torch.uint8,
                            device=dev)
        c2 = torch.randint(0, 4, (e, h, i), dtype=torch.uint8, device=dev)
        s2 = torch.randint(123, 126, (e, h, i // 32), dtype=torch.uint8,
                           device=dev)
        planes = pack_planes_for_cubin(_rowmajor(c13), s13, _rowmajor(c2), s2,
                                       e, h, i, k13, k2, dev)
        x = torch.randn(t, h, device=dev, dtype=torch.bfloat16)
        ids = torch.randint(0, e, (t, topk), device=dev, dtype=torch.int32)
        tw = torch.rand(t, topk, device=dev, dtype=torch.float32)
        got = cubin_forward(x, ids, tw, planes, e, h, i, k13, k2)

        w13, w2 = deq(c13, s13, k13), deq(c2, s2, k2)
        xr = torch.cat([x.float(), torch.zeros(t, k13 - h, device=dev)], 1)
        ref = torch.zeros(t, h, device=dev)
        for ex in range(e):
            m, j = (ids == ex).nonzero(as_tuple=True)
            if m.numel() == 0:
                continue
            g = xr[m] @ w13[ex, :i].T
            u = xr[m] @ w13[ex, i:].T
            act = (g.clamp(max=20) * torch.sigmoid(g.clamp(max=20))
                   * u.clamp(-20, 20))
            ref.index_add_(0, m, (act @ w2[ex].T) * tw[m, j].unsqueeze(1))
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 6e-2)  # dominated by fp8 e4m3 activation quant


if __name__ == "__main__":
    unittest.main()
