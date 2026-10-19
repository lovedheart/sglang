"""Registry-vs-checkpoint reassembly test for the SM12x disk MoE backend.

The disk runner never loads expert payloads through torch; it preads byte
ranges that the registry derived from safetensors headers. This pins that
derivation: random (layer, expert) segments must return exactly the bytes
``safe_open().get_tensor()`` returns, and segment-driven slot assembly must
land gate rows in w13[0:I] and up rows in w13[I:2I] (the layout the CuTe
kernel reads). Also pins the page-rounded fadvise range math, which has to
widen unaligned safetensors offsets without crossing file ends.

Run:
    python test/manual/test_sm12x_disk_registry.py
"""

import os
import random
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

CKPT = os.environ.get(
    "SGLANG_TEST_SM12X_CKPT",
    "/media/lovedheart/models/Ling-3.0-tiny-NVFP4-w4a4-4o6",
)


def _registry():
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_registry import (
        SafetensorsExpertRegistry,
    )

    return SafetensorsExpertRegistry(CKPT)


@unittest.skipUnless(
    os.path.exists(os.path.join(CKPT, "model.safetensors.index.json")),
    f"checkpoint not present at {CKPT}",
)
class TestSm12xDiskRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        from safetensors import safe_open

        cls.reg = _registry()
        with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
            cls.weight_map = json.load(f)["weight_map"]
        cls._handles = {}

        def tensor(name):
            shard = cls.weight_map[name]
            if shard not in cls._handles:
                cls._handles[shard] = safe_open(
                    os.path.join(CKPT, shard), framework="pt", device="cpu"
                )
            return cls._handles[shard].get_tensor(name)

        cls.tensor = staticmethod(tensor)
        cls.rng = random.Random(0)
        # Checkpoints differ in module path (model.layers vs
        # model.language_model.layers); derive it from the index itself.
        import re

        any_name = next(
            k for k in cls.weight_map if ".mlp.experts.0.gate_proj.weight" in k
        )
        cls.prefix = re.sub(r"\.layers\.\d+\..*$", ".layers", any_name)

    @classmethod
    def tearDownClass(cls):
        cls.reg.close()

    def _proj_names(self, layer, expert):
        p = f"{self.prefix}.{layer}.mlp.experts.{expert}"
        return (
            f"{p}.gate_proj.weight",
            f"{p}.up_proj.weight",
            f"{p}.down_proj.weight",
        )

    def test_random_segments_match_safe_open(self):
        """Every derived (shard, offset, length) must read exactly the bytes
        safetensors reports for that tensor -- the registry's core claim."""
        picks = [
            (layer, self.rng.choice(range(self.reg.num_experts)))
            for layer in self.reg.layer_ids
        ]
        for _ in range(500 - len(picks)):
            picks.append(
                (
                    self.rng.choice(self.reg.layer_ids),
                    self.rng.randrange(self.reg.num_experts),
                )
            )
        for layer, expert in picks[:500]:
            payload = self.reg.payload(layer, expert)
            for seg, name in zip(payload.segments, self._proj_names(layer, expert)):
                got = torch.frombuffer(
                    bytearray(self.reg.pread(seg.shard, seg.src_offset, seg.nbytes)),
                    dtype=torch.uint8,
                )
                want = self.tensor(name).reshape(-1)
                self.assertTrue(
                    torch.equal(got, want),
                    f"mismatch for layer {layer} expert {expert} {name}",
                )

    def test_slot_assembly_lands_gate_then_up(self):
        """Assembling a w13 slot via segment dst offsets must reproduce the
        resident loader's [gate; up] row packing exactly."""
        for layer in (self.reg.layer_ids[0], self.reg.layer_ids[-1]):
            for expert in (0, self.reg.num_experts - 1):
                payload = self.reg.payload(layer, expert)
                gate, up, down = (
                    self.tensor(n).reshape(-1) for n in self._proj_names(layer, expert)
                )
                w13 = torch.zeros(gate.numel() + up.numel(), dtype=torch.uint8)
                w2 = torch.zeros(down.numel(), dtype=torch.uint8)
                for seg in payload.w13:
                    w13[seg.dst_offset : seg.dst_offset + seg.nbytes] = (
                        torch.frombuffer(
                            bytearray(
                                self.reg.pread(seg.shard, seg.src_offset, seg.nbytes)
                            ),
                            dtype=torch.uint8,
                        )
                    )
                for seg in payload.w2:
                    w2[seg.dst_offset : seg.dst_offset + seg.nbytes] = torch.frombuffer(
                        bytearray(
                            self.reg.pread(seg.shard, seg.src_offset, seg.nbytes)
                        ),
                        dtype=torch.uint8,
                    )
                self.assertTrue(torch.equal(w13[: gate.numel()], gate))
                self.assertTrue(torch.equal(w13[gate.numel() :], up))
                self.assertTrue(torch.equal(w2, down))

    def test_page_range_rounding(self):
        from sglang.srt.layers.moe.moe_runner.sm12x_disk_registry import Segment

        cases = [
            # offset, nbytes, file_size -> expected rounded (lo, hi)
            (0, 4096, 1 << 20, 0, 4096),
            (1, 10, 1 << 20, 0, 4096),
            (4095, 2, 1 << 20, 0, 8192),
            (4096, 4096, 1 << 20, 4096, 8192),
            ((1 << 20) - 10, 100, 1 << 20, (1 << 20) - 4096, 1 << 20),
        ]
        for off, n, size, lo, hi in cases:
            self.assertEqual(Segment(0, 0, off, n).page_range(size), (lo, hi))

    def test_fadvise_runs_on_real_ranges(self):
        """Smoke: coalesced fadvise over a few layers' experts must not raise
        (offsets beyond EOF are clamped, fds opened lazily)."""
        for layer in self.reg.layer_ids[:2]:
            self.reg.fadvise(layer, range(self.reg.num_experts), willneed=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
