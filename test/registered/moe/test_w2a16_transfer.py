"""Regression tests for NVFP4 -> TS4 plane conversion (w2a16_offload).

Guards the row-layout bug: w13 [E,2I,H/2] rows are [gate; up] and must map
to plane rows [E,2I,H/4], not [E,3I,...] (the up-half is NOT extra columns).
"""

import unittest

import torch

from sglang.srt.layers.moe.w2a16 import ts4
from sglang.srt.layers.moe.w2a16.transfer import nvfp4_w13_to_ts4, nvfp4_w2_to_ts4


class TestW2A16Transfer(unittest.TestCase):
    def test_w13_plane_shape(self):
        E, I, H = 4, 512, 1536
        w13 = torch.randint(0, 255, (E, 2 * I, H // 2), dtype=torch.uint8)
        sf = torch.randint(100, 120, (E, 2 * I, H // 16), dtype=torch.uint8)
        gs = torch.ones(E, 1)
        plane, s = nvfp4_w13_to_ts4(w13, sf, gs, gs.clone())
        self.assertEqual(plane.shape, (E, 2 * I, H // 4))
        self.assertEqual(s.shape, (E, 2 * I, H // 32))

    def test_w13_row_order_gate_up(self):
        # Distinct gate/up weights: gate half must reconstruct from the
        # first I rows, up half from the second I rows.
        E, I, H = 2, 64, 256
        w13 = torch.randint(0, 255, (E, 2 * I, H // 2), dtype=torch.uint8)
        sf = torch.randint(110, 115, (E, 2 * I, H // 16), dtype=torch.uint8)
        gs = torch.ones(E, 1)
        plane, s = nvfp4_w13_to_ts4(w13, sf, gs, gs.clone())
        deq = ts4.ts4_dequant(plane, s, 1.0)
        ref = torch.cat(
            [
                _deq_half(w13[:, :I], sf[:, :I]),
                _deq_half(w13[:, I:], sf[:, I:]),
            ],
            dim=1,
        )
        # TS4 is lossy but monotone in magnitude: correlation must be high
        # per row-half, and the wrong (swapped) mapping must be worse.
        c_ok = _corr(deq, ref)
        c_swap = _corr(deq, torch.cat([ref[:, I:], ref[:, :I]], dim=1))
        self.assertGreater(c_ok, 0.9)
        self.assertLess(c_swap, c_ok - 0.2)

    def test_w2_plane_shape(self):
        E, H, I = 2, 1536, 512
        w2 = torch.randint(0, 255, (E, H, I // 2), dtype=torch.uint8)
        sf = torch.randint(100, 120, (E, H, I // 16), dtype=torch.uint8)
        plane, s = nvfp4_w2_to_ts4(w2, sf, torch.ones(E, 1))
        self.assertEqual(plane.shape, (E, H, I // 4))


def _deq_half(packed, sf):
    # Only used for scale patterns; exact values matter via correlation only.
    from sglang.srt.layers.moe.w2a16.transfer import _dequant_nvfp4

    return _dequant_nvfp4(packed, sf, torch.ones(packed.shape[0], 1))


def _corr(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return torch.corrcoef(torch.stack([a - a.mean(), b - b.mean()]))[0, 1].item()


if __name__ == "__main__":
    unittest.main()
