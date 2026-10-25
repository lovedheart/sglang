"""NVFP4 checkpoint -> TS4 plane transfer + on-disk archive.

One TS4 archive per checkpoint (``<model>/.ts4-cache`` or
``SGLANG_W2A16_TS4_CACHE``), built once by the first process that loads a
layer with ``--moe-runner-backend w2a16_offload`` and mmap'd by every later
launch:

    layer{L}.bin  w13 plane u8 [E,2I,H/4] | s13 u8 [E,2I,H/32] |
                  w2  plane u8 [E,H,I/4]  | s2  u8 [E,H,I/32]   (4K-aligned)
    manifest.json fingerprint + per-layer shapes (content is covered by the
                  fingerprint, which hashes config.json and the safetensors
                  index, so any checkpoint edit forces a rebuild).

Plane semantics (ig_moe CONTRACT v1 / vLLM-Moet w2): code k in bits
2*(k%4) of each byte, codebook LUT = {-4,-1,+1,+4}, one UE8M0 byte
(value 2^(byte-127)) per 32 codes. The NVFP4 per-tensor global scales are
dequantized *before* TS4 quantization, so they fold into the UE8M0 block
scales (rounded to a power of two — an inherent part of the W2 loss) and the
engines run with gs13 = gs2 = 1.0.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import time

import torch

from sglang.srt.layers.moe.w2a16.ts4 import quantize_ts4

_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

ALIGN = 4096


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def fingerprint_of(model_path: str) -> str:
    h = hashlib.sha256()
    cfg = os.path.join(model_path, "config.json")
    if os.path.exists(cfg):
        with open(cfg, "rb") as f:
            h.update(b"cfg:")
            h.update(hashlib.sha256(f.read()).digest())
    idx = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, "rb") as f:
            h.update(b"idx:")
            h.update(hashlib.sha256(f.read()).digest())
    h.update(b"abi:w2a16-ts4-v2")
    return h.hexdigest()[:16]


class Ts4Archive:
    """Per-checkpoint TS4 plane store (write-once files, mmap reads)."""

    def __init__(self, cache_dir: str, fingerprint: str):
        self.dir = cache_dir
        self.fingerprint = fingerprint
        self.manifest_path = os.path.join(cache_dir, "manifest.json")
        os.makedirs(cache_dir, exist_ok=True)
        manifest = {}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                manifest = json.load(f)
        if manifest.get("fingerprint") != fingerprint:
            if manifest:
                for name in os.listdir(cache_dir):
                    if name.startswith("layer") and name.endswith(".bin"):
                        os.remove(os.path.join(cache_dir, name))
            manifest = {"fingerprint": fingerprint, "layers": {}}
            self._write_manifest(manifest)
        self._manifest = manifest

    def _write_manifest(self, manifest: dict) -> None:
        tmp = self.manifest_path + f".{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(tmp, self.manifest_path)
        self._manifest = manifest

    def _key(self, layer_id: int, E: int, H: int, I: int) -> str:
        return str(layer_id)

    def has_layer(self, layer_id: int, E: int, H: int, I: int) -> bool:
        entry = self._manifest["layers"].get(str(layer_id))
        return entry is not None and (entry["E"], entry["H"], entry["I"]) == (E, H, I)

    def _path(self, layer_id: int) -> str:
        return os.path.join(self.dir, f"layer{layer_id}.bin")

    def write_layer(
        self,
        layer_id: int,
        E: int,
        H: int,
        I: int,
        w13_plane: torch.Tensor,
        s13: torch.Tensor,
        w2_plane: torch.Tensor,
        s2: torch.Tensor,
    ) -> None:
        path = self._path(layer_id) + f".{os.getpid()}.tmp"
        tensors = [w13_plane, s13, w2_plane, s2]
        offsets, off = [], 0
        for t in tensors:
            offsets.append(off)
            off += _align(t.numel())
        with open(path, "wb") as f:
            for t in tensors:
                buf = memoryview(t.detach().to(torch.uint8).flatten().numpy())
                f.write(buf)
                pad = _align(t.numel()) - t.numel()
                if pad:
                    f.write(b"\0" * pad)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path, self._path(layer_id))
        manifest = dict(self._manifest)
        manifest["layers"] = dict(self._manifest["layers"])
        manifest["layers"][str(layer_id)] = {
            "E": E, "H": H, "I": I,
            "shapes": [list(t.shape) for t in tensors],
            "offsets": offsets,
        }
        self._write_manifest(manifest)

    def load_layer(self, layer_id: int, E: int, H: int, I: int):
        """Return (mm, w13_plane, s13, w2_plane, s2) mmap-backed CPU uint8
        tensors with the recorded shapes. The caller must keep ``mm`` alive
        exactly as long as the views (the engines do)."""
        entry = self._manifest["layers"][str(layer_id)]
        assert (entry["E"], entry["H"], entry["I"]) == (E, H, I)
        f = open(self._path(layer_id), "rb")
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        f.close()
        base = torch.frombuffer(mm, dtype=torch.uint8)
        out = []
        for shape, off in zip(entry["shapes"], entry["offsets"]):
            n = 1
            for s in shape:
                n *= s
            out.append(base[off : off + n].view(*shape))
        return (mm, *out)


def _dequant_nvfp4(packed_u8: torch.Tensor, sf_e4m3: torch.Tensor,
                   gs: torch.Tensor) -> torch.Tensor:
    """[E,R,K/2] u8 + [E,R,K/16] e4m3 + [E] gs -> dense fp32 [E,R,K]."""
    lut = _E2M1_LUT.to(packed_u8.device)
    lo = packed_u8 & 0x0F
    hi = packed_u8 >> 4
    vals = torch.empty(
        packed_u8.shape[0], packed_u8.shape[1], packed_u8.shape[2] * 2,
        dtype=torch.float32, device=packed_u8.device,
    )
    vals[..., 0::2] = lut[lo.long()]
    vals[..., 1::2] = lut[hi.long()]
    sf = sf_e4m3.view(torch.float8_e4m3fn).to(torch.float32)
    sf = torch.nan_to_num(sf, nan=0.0)
    vals.mul_(sf.repeat_interleave(16, dim=-1))
    vals.mul_(gs.view(-1, 1, 1))
    return vals


def nvfp4_w13_to_ts4(w13_u8, w13_sf, w13_gs_gate, w13_gs_up, chunk=16):
    """[E,2I,H/2] packed (gate rows;up rows) -> plane [E,2I,H/4] + s [E,2I,H/32]."""
    E, N, Kh = w13_u8.shape  # N = 2I rows: [gate; up]; Kh = H/2 packed
    # Rows are [gate; up]; each row-half carries its own global scale.
    dense = _dequant_nvfp4(w13_u8, w13_sf, torch.ones_like(w13_gs_gate))
    gate = dense[:, : N // 2] * w13_gs_gate.view(-1, 1, 1)
    up = dense[:, N // 2 :] * w13_gs_up.view(-1, 1, 1)
    del dense
    codes, sc = quantize_ts4(torch.cat([gate, up], dim=1).float())
    return _codes_to_plane(codes), sc


def nvfp4_w2_to_ts4(w2_u8, w2_sf, w2_gs, chunk=16):
    E, H, Ih = w2_u8.shape
    planes, scales = [], []
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        dense = _dequant_nvfp4(w2_u8[e0:e1], w2_sf[e0:e1], w2_gs[e0:e1])
        codes, sc = quantize_ts4(dense)
        del dense
        planes.append(_codes_to_plane(codes))
        scales.append(sc)
        del codes, sc
    return torch.cat(planes), torch.cat(scales)


def _codes_to_plane(codes: torch.Tensor) -> torch.Tensor:
    c = codes.to(torch.int32)
    g = c.view(*c.shape[:-1], c.shape[-1] // 4, 4)
    packed = g[..., 0] | (g[..., 1] << 2) | (g[..., 2] << 4) | (g[..., 3] << 6)
    return packed.to(torch.uint8)


def build_layer_planes(layer, device) -> tuple[torch.Tensor, ...]:
    """TS4 planes for one loaded FusedMoE layer (uint8 CPU tensors:
    w13 [E,2I,H/4], s13 [E,2I,H/32], w2 [E,H,I/4], s2 [E,H,I/32])."""
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_sf = layer.w13_weight_scale.data
    w2_sf = layer.w2_weight_scale.data
    gs13 = layer.w13_weight_scale_2.data
    gs2 = layer.w2_weight_scale_2.data
    E, N2, Kh = w13.shape
    H = Kh * 2
    I = w2.shape[2] * 2
    if gs13.dim() > 1 and gs13.shape[-1] >= 2:
        g_gate, g_up = gs13[:, 0].float(), gs13[:, 1].float()
        # Rows are [gate; up]; the up-half ratio rides as a per-half gs so
        # the dequant math matches the checkpoint (see _sm12x_global_scales).
    else:
        g_gate = g_up = gs13.reshape(-1).float()
    w13_plane, s13 = nvfp4_w13_to_ts4(
        w13.to(device), w13_sf.to(device), g_gate.to(device), g_up.to(device)
    )
    w2_plane, s2 = nvfp4_w2_to_ts4(
        w2.to(device), w2_sf.to(device), gs2.reshape(-1).float().to(device)
    )
    return (
        w13_plane.cpu(), s13.cpu(), w2_plane.cpu(), s2.cpu(),
    )
