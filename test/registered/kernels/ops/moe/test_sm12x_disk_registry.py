# SPDX-License-Identifier: Apache-2.0
"""Owner tests for the SM12x disk MoE safetensors payload registry
(sm12x_disk_registry.py), on a synthetic checkpoint: segment math vs the
shard bytes, TP-split rejection, fadvise page rounding, and the meta-weight
payload recorder's guards. No GPU required."""

import json
import os
import struct

import pytest
import torch
from sglang.srt.layers.moe.moe_runner.sm12x_disk_registry import (
    SafetensorsExpertRegistry,
    make_payload_recorder,
    parse_layer_ranges,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

E, H, I = 8, 256, 128  # slot 48 KiB; FP4-packed [I, H/2] payloads
LAYERS = (0, 1)


def _expert_tensors(seed):
    g = torch.Generator().manual_seed(seed)
    gate = torch.randint(0, 256, (I, H // 2), dtype=torch.uint8, generator=g)
    up = torch.randint(0, 256, (I, H // 2), dtype=torch.uint8, generator=g)
    down = torch.randint(0, 256, (H, I // 2), dtype=torch.uint8, generator=g)
    return gate, up, down


def _write_ckpt(d, shards=1, lm_prefix=False):
    from safetensors.torch import save_file

    tensors = {}
    for L in LAYERS:
        for e in range(E):
            pre = (
                f"model.language_model.layers.{L}" if lm_prefix else f"model.layers.{L}"
            )
            gate, up, down = _expert_tensors(1000 * L + e)
            for nm, t in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
                tensors[f"{pre}.mlp.experts.{e}.{nm}.weight"] = t
                tensors[f"{pre}.mlp.experts.{e}.{nm}.weight_scale"] = torch.full(
                    (t.shape[0], t.shape[1] // 8), 0.5, dtype=torch.float32
                )
    names = sorted(tensors)
    wm = {n: f"model-{i % shards}.safetensors" for i, n in enumerate(names)}
    for s in range(shards):
        save_file(
            {n: tensors[n] for n in names if wm[n] == f"model-{s}.safetensors"},
            os.path.join(d, f"model-{s}.safetensors"),
        )
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)
    return tensors


def _registry(d):
    return SafetensorsExpertRegistry(str(d))


def test_registry_segments_match_shard_bytes(tmp_path):
    _write_ckpt(tmp_path, shards=2)
    reg = _registry(tmp_path)
    assert reg.layer_ids == list(LAYERS) and reg.num_experts == E
    assert reg.slot_shapes == ((2 * I, H // 2), (H, I // 2))
    g = torch.Generator().manual_seed(0)
    for _ in range(40):
        L = LAYERS[torch.randint(0, len(LAYERS), (1,), generator=g).item()]
        e = torch.randint(0, E, (1,), generator=g).item()
        payload = reg.payload(L, e)
        gate, up, down = _expert_tensors(1000 * L + e)
        for seg, want in (
            (payload.w13[0], gate),
            (payload.w13[1], up),
            (payload.w2[0], down),
        ):
            got = torch.frombuffer(
                bytearray(reg.pread(seg.shard, seg.src_offset, seg.nbytes)),
                dtype=torch.uint8,
            ).view(want.shape)
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    # dst offsets stitch [gate; up] into one w13 slot, row-major
    assert reg.payload(LAYERS[0], 0).w13[1].dst_offset == I * (H // 2)


def test_registry_language_model_prefix(tmp_path):
    _write_ckpt(tmp_path, lm_prefix=True)
    assert _registry(tmp_path).layer_ids == list(LAYERS)


def test_registry_rejects_tp_split(tmp_path):
    from safetensors.torch import save_file

    tensors = {}
    for e in range(E):
        for nm in ("gate_proj", "up_proj"):
            tensors[f"model.layers.0.mlp.experts.{e}.{nm}.weight"] = torch.zeros(
                I, H // 4, dtype=torch.uint8
            )  # hidden dim split in the payload
        tensors[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.zeros(
            H, I // 2, dtype=torch.uint8
        )
    save_file(tensors, os.path.join(tmp_path, "model.safetensors"))
    with pytest.raises(AssertionError, match="tensor-parallel"):
        _registry(tmp_path)


def test_page_rounding_and_fadvise_smoke(tmp_path):
    _write_ckpt(tmp_path)
    reg = _registry(tmp_path)
    seg = reg.payload(LAYERS[0], 3).w13[0]
    lo, hi = seg.page_range(reg._sizes[seg.shard])
    assert lo % 4096 == 0 and hi % 4096 == 0
    assert lo <= seg.src_offset and hi >= seg.src_offset + seg.nbytes
    reg.fadvise(LAYERS[0], range(E))  # must not raise


def test_payload_recorder_guards(tmp_path):
    tensors = _write_ckpt(tmp_path)
    reg = _registry(tmp_path)
    rec = make_payload_recorder(reg, layer_id=0)
    gate = tensors["model.layers.0.mlp.experts.0.gate_proj.weight"]

    class P:
        device = torch.device("cpu")

    rec(P(), gate, "w13_weight", shard_id="w1", expert_id=0)  # matches
    assert rec.stats["checked"] == 1
    with pytest.raises(RuntimeError, match="disagree"):
        rec(P(), _expert_tensors(4242)[0], "w13_weight", shard_id="w1", expert_id=0)
    with pytest.raises(AssertionError, match="bytes"):
        rec(P(), gate[:4], "w13_weight", shard_id="w1", expert_id=0)


def test_parse_layer_ranges():
    assert parse_layer_ranges("1-4,7") == frozenset({1, 2, 3, 4, 7})
    assert parse_layer_ranges("") == frozenset()


def test_header_offsets(tmp_path):
    _write_ckpt(tmp_path)
    reg = _registry(tmp_path)
    seg = reg.payload(LAYERS[0], 0).w13[0]
    path = os.path.join(str(tmp_path), reg.shard_paths[seg.shard])
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    base = 8 + n + header[name]["data_offsets"][0]
    assert seg.src_offset == base and seg.nbytes == I * (H // 2)


def _write_ckpt_b(d):
    """Layout B: DeepSeek-style layers.N.ffn.experts.E.{w1,w2,w3} naming
    with linear-layout e4m3 scale tensors."""
    from safetensors.torch import save_file

    tensors = {}
    for L in LAYERS:
        for e in range(E):
            pre = f"layers.{L}.ffn.experts.{e}"
            gate, up, down = _expert_tensors(1000 * L + e)
            for nm, w in (("w1", gate), ("w3", up), ("w2", down)):
                tensors[f"{pre}.{nm}.weight"] = w
                tensors[f"{pre}.{nm}.weight_scale"] = torch.rand(
                    w.shape[0],
                    w.shape[1] // 8,
                    generator=torch.Generator().manual_seed(e),
                ).to(torch.float8_e4m3fn)
    save_file(tensors, os.path.join(str(d), "model.safetensors"))


def test_registry_layout_b(tmp_path):
    _write_ckpt_b(tmp_path)
    reg = _registry(tmp_path)
    assert reg.sf_streamed and reg.layer_ids == list(LAYERS)
    assert reg.sf_raw == ((2 * I, H // 16), (H, I // 16))
    p = reg.payload(LAYERS[0], 0)
    assert [s.nbytes for s in p.w13] == [I * H // 2] * 2
    assert [s.nbytes for s in p.w13_sf] == [I * (H // 16)] * 2
    assert [s.nbytes for s in p.w2_sf] == [H * (I // 16)]
    assert p.segments == p.w13 + p.w2 + p.w13_sf + p.w2_sf
    # The role map sends w1 -> w13 rows [0, I) and w3 -> [I, 2I); check
    # the first scale bytes of each half (both were drawn from a fresh
    # seed-0 generator in _write_ckpt_b for expert 0).
    want = bytes(
        torch.rand(I, H // 16, generator=torch.Generator().manual_seed(0))
        .to(torch.float8_e4m3fn)[0, :8]
        .view(torch.uint8)
        .tolist()
    )
    for seg in p.w13_sf:
        assert reg.pread(seg.shard, seg.src_offset, 8) == want


def test_swizzle_idx_matches_production():
    import numpy as np
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_source import _swizzle_idx
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    for m, k in ((256, 16), (4608, 8)):
        lin = torch.randint(
            0,
            256,
            (m, k),
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(3),
        )
        want = swizzle_blockscale(lin.view(torch.float8_e4m3fn)).view(torch.uint8)
        got = lin.reshape(-1)[torch.from_numpy(_swizzle_idx(m, k).astype(np.int64))]
        assert torch.equal(got.view(m, k), want), f"swizzle mismatch at {m}x{k}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
