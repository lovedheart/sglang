"""Safetensors expert-payload registry for the SM12x disk MoE runner.

Maps every (MoE layer, expert) to the byte segments that hold its FP4-packed
payload inside the original checkpoint shards -- no repacking, no rewrite.
Each segment says "these ``nbytes`` file bytes at ``src_offset`` land at
element offset ``dst_offset`` of the layer's slot payload" (uint8 view).

Two checkpoint layouts are understood:

  * layout A (modelopt fused MoE, e.g. the 30B/35B NVFP4 ports):
    ``model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight``; scale
    factors are loaded VRAM-resident by the quant method and are NOT
    mapped here.
  * layout B (per-expert DeepSeek-style naming, e.g. the V4.1-Flash
    NVFP4 ports): ``layers.N.ffn.experts.E.{w1,w2,w3}.weight`` plus a
    LINEAR-layout e4m3 ``.weight_scale`` per weight. Those scale tensors
    are huge (GBs of VRAM if resident), so they are registered as extra
    streamed segments (``w13_sf`` / ``w2_sf``) and the source swizzles
    them into slot layout at fill time. ``weight_scale_2`` stays a
    resident fp32 scalar on the model side.

Safetensors data offsets are not 4 KiB aligned, so ``pread`` uses the exact
byte ranges (alignment-free) while ``fadvise`` uses page-rounded supersets.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
from dataclasses import dataclass
from typing import NamedTuple

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_PAGE = 4096

_EXPERT_RE = re.compile(
    r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)"
    r"\.(gate|up|down)_proj\.weight$"
)
# Layout B: DeepSeek-style per-expert naming, payload AND scale tensors.
_EXPERT_RE_B = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)"
    r"\.(weight|weight_scale)$"
)
_B_TO_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}


@dataclass(frozen=True)
class Segment:
    """A payload run: ``nbytes`` file bytes at ``src_offset`` in shard
    ``shard`` -> element ``dst_offset`` of the w13/w2 slot payload (uint8)."""

    dst_offset: int
    shard: int
    src_offset: int
    nbytes: int

    def page_range(self, file_size: int) -> tuple[int, int]:
        # Page-rounded superset for posix_fadvise only; pread needs none.
        lo = self.src_offset & -_PAGE
        hi = min(self.src_offset + self.nbytes, file_size)
        return lo, (hi + _PAGE - 1) & -_PAGE


class ExpertPayload(NamedTuple):
    # w13 slot rows are [gate; up], gate starting at element 0.
    w13: tuple[Segment, ...]
    w2: tuple[Segment, ...]
    # Layout B only: segments for the LINEAR-layout e4m3 scale factors of
    # w13 ([w1; w3] rows) and w2; the source swizzles them into slot shape.
    w13_sf: tuple[Segment, ...] = ()
    w2_sf: tuple[Segment, ...] = ()

    @property
    def segments(self) -> tuple[Segment, ...]:
        return self.w13 + self.w2 + self.w13_sf + self.w2_sf


@dataclass(frozen=True)
class _TensorLoc:
    shard: int
    begin: int
    end: int
    dtype: str
    shape: tuple[int, ...]


def _read_shard_header(path: str) -> dict[str, _TensorLoc]:
    """Parse a safetensors header (u64 length + JSON) into tensor locations;
    begin/end are file-byte offsets (data region base folded in)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    base = 8 + n
    return {
        name: _TensorLoc(
            shard=-1,
            begin=base + spec["data_offsets"][0],
            end=base + spec["data_offsets"][1],
            dtype=spec["dtype"],
            shape=tuple(spec["shape"]),
        )
        for name, spec in header.items()
        if name != "__metadata__"
    }


class SafetensorsExpertRegistry:
    """Per-(layer, expert) payload map over one local checkpoint dir."""

    def __init__(self, model_dir: str):
        weight_map = self._load_weight_map(model_dir)
        # Sorted order keeps Segment.shard indices stable across runs.
        self.model_dir = model_dir
        self.shard_paths = sorted(set(weight_map.values()))
        headers = {
            rel: _read_shard_header(os.path.join(model_dir, rel))
            for rel in self.shard_paths
        }
        for shard, rel in enumerate(self.shard_paths):
            headers[rel] = {
                name: _TensorLoc(shard, t.begin, t.end, t.dtype, t.shape)
                for name, t in headers[rel].items()
            }

        found: dict[tuple[int, int], dict[str, _TensorLoc]] = {}
        scales: dict[tuple[int, int], dict[str, _TensorLoc]] = {}
        for name, rel in weight_map.items():
            m = _EXPERT_RE.match(name)
            if m:
                loc = headers[rel][name]
                slot = found.setdefault((int(m[1]), int(m[2])), {})
                assert m[3] not in slot, f"duplicate tensor for {name}"
                slot[m[3]] = loc
                continue
            m = _EXPERT_RE_B.match(name)
            if m:
                role = _B_TO_ROLE[m[3]] + ("" if m[4] == "weight" else "_sf")
                slot = scales.setdefault((int(m[1]), int(m[2])), {})
                assert role not in slot, f"duplicate tensor for {name}"
                slot[role] = headers[rel][name]
        if not found and scales:
            found = {
                k: {r: t for r, t in v.items() if not r.endswith("_sf")}
                for k, v in scales.items()
            }
        if not found:
            raise ValueError(
                f"no model.layers.N.mlp.experts.E.{{gate,up,down}}_proj.weight "
                f"or layers.N.ffn.experts.E.{{w1,w2,w3}}.weight tensors under "
                f"{model_dir}"
            )
        self.sf_streamed = bool(scales) and not any(
            _EXPERT_RE.match(n) for n in weight_map
        )
        self.layer_ids, self.num_experts, payloads, self.slot_shapes = _assemble(
            found, scales if self.sf_streamed else None
        )
        self._payloads = payloads
        self.sf_raw = None
        if self.sf_streamed:
            (r13, c13), (r2, c2) = self.slot_shapes
            assert c13 % 8 == 0 and c2 % 8 == 0, "scale cols must be /16 logical"
            self.sf_raw = ((r13, c13 // 8), (r2, c2 // 8))

        self._fds: dict[int, int] = {}
        self._sizes = [
            os.path.getsize(os.path.join(model_dir, p)) for p in self.shard_paths
        ]

    @staticmethod
    def _load_weight_map(model_dir: str) -> dict[str, str]:
        index = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                return json.load(f)["weight_map"]
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(f"no safetensors under {model_dir}")
        return {n: "model.safetensors" for n in _read_shard_header(single)}

    # ---- queries ------------------------------------------------------

    def payload(self, layer: int, expert: int) -> ExpertPayload:
        return self._payloads[(layer, expert)]

    def pread(self, shard: int, offset: int, nbytes: int) -> bytes:
        return os.pread(self.fd(shard), nbytes, offset)

    def fd(self, shard: int) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            path = os.path.join(self.model_dir, self.shard_paths[shard])
            fd = self._fds[shard] = os.open(path, os.O_RDONLY)
        return fd

    def file_size(self, shard: int) -> int:
        return self._sizes[shard]

    def fadvise(self, layer: int, experts, *, willneed: bool = True) -> None:
        advice = os.POSIX_FADV_WILLNEED if willneed else os.POSIX_FADV_DONTNEED
        # Coalesce per shard so overlapping page ranges issue one syscall.
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for e in experts:
            for seg in self.payload(layer, e).segments:
                lo, hi = seg.page_range(self.file_size(seg.shard))
                by_shard.setdefault(seg.shard, []).append((lo, hi))
        for shard, ranges in by_shard.items():
            ranges.sort()
            merged = []
            for lo, hi in ranges:
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            fd = self.fd(shard)
            for lo, hi in merged:
                os.posix_fadvise(fd, lo, hi - lo, advice)

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def _assemble(
    found: dict[tuple[int, int], dict[str, _TensorLoc]],
    scales: dict[tuple[int, int], dict[str, _TensorLoc]] | None = None,
) -> tuple[list[int], int, dict[tuple[int, int], ExpertPayload], tuple[tuple, tuple]]:
    """Validate shapes/dtypes are uniform and pack the segment tables."""
    payloads: dict[tuple[int, int], ExpertPayload] = {}
    ref: tuple | None = None
    by_layer: dict[int, set] = {}
    for (layer, expert), projs in found.items():
        missing = {"gate", "up", "down"} - projs.keys()
        assert not missing, f"layer {layer} expert {expert} missing {missing}"
        gate, up, down = projs["gate"], projs["up"], projs["down"]
        for t in (gate, up, down):
            assert t.dtype == "U8", f"expected U8 FP4-packed payload, got {t.dtype}"
        assert gate.shape == up.shape, "gate/up payload shapes differ"
        assert all(t % 2 == 0 for t in gate.shape), "FP4 rows must be byte-packed"
        assert len(down.shape) == 2
        geom = (*gate.shape, *down.shape)
        if ref is None:
            ref = geom
        assert geom == ref, f"geometry {geom} differs from {ref}"
        i, h2 = gate.shape
        h, i2 = down.shape
        # gate rows cover the intermediate, gate cols half the hidden dim;
        # down is [H, I/2]. A mismatch means the shard stores an MP/TP-split
        # payload, which the disk backend rejects.
        assert i2 * 2 == i and h == 2 * h2, "payload looks tensor-parallel split"
        sfs = scales[(layer, expert)] if scales else None
        seg13 = (
            Segment(0, gate.shard, gate.begin, gate.end - gate.begin),
            Segment(i * h2, up.shard, up.begin, up.end - up.begin),
        )
        seg2 = (Segment(0, down.shard, down.begin, down.end - down.begin),)
        if sfs is None:
            sf13: tuple[Segment, ...] = ()
            sf2: tuple[Segment, ...] = ()
        else:
            for want in ("gate", "up", "gate_sf", "up_sf", "down_sf"):
                assert want in sfs, f"layer {layer} expert {expert} missing {want}"
            g_sf, u_sf, d_sf = sfs["gate_sf"], sfs["up_sf"], sfs["down_sf"]
            # Linear e4m3 block-16 scales: [rows, logical_cols / 16].
            for p, s in ((gate, g_sf), (up, u_sf), (down, d_sf)):
                assert s.dtype == "F8_E4M3", f"expected F8_E4M3 scale, got {s.dtype}"
                assert s.shape[0] == p.shape[0] and s.shape[1] * 8 == p.shape[1], (
                    f"scale shape {s.shape} disagrees with payload {p.shape}"
                )
            k16 = h2 // 8
            assert h2 % 8 == 0 and i2 % 8 == 0, "scale cols must be /16 of logical"
            sf13 = (
                Segment(0, g_sf.shard, g_sf.begin, g_sf.end - g_sf.begin),
                Segment(i * k16, u_sf.shard, u_sf.begin, u_sf.end - u_sf.begin),
            )
            sf2 = (Segment(0, d_sf.shard, d_sf.begin, d_sf.end - d_sf.begin),)
        payloads[(layer, expert)] = ExpertPayload(seg13, seg2, sf13, sf2)
        by_layer.setdefault(layer, set()).add(expert)
    n_experts = len(next(iter(by_layer.values())))
    for layer, experts in by_layer.items():
        assert len(experts) == n_experts, f"layer {layer} has partial experts"
    layer_ids = sorted(by_layer)
    i, h2 = ref[:2]
    h, i2 = ref[2:]
    slot_shapes = ((2 * i, h2), (h, i2))  # per-expert w13 [gate; up], w2
    return layer_ids, n_experts, payloads, slot_shapes


def parse_layer_ranges(spec: str) -> frozenset[int]:
    """Parse "1-4,7" into a frozenset; "" -> empty."""
    out: set[int] = set()
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return frozenset(out)


def _registry_factory() -> SafetensorsExpertRegistry:
    from sglang.srt.runtime_context import get_model

    return SafetensorsExpertRegistry(str(get_model().model_path))


def get_expert_registry() -> SafetensorsExpertRegistry:
    from sglang.srt.runtime_context import get_buffer

    return get_buffer("sm12x_disk_registry", _registry_factory)


def make_payload_recorder(registry: SafetensorsExpertRegistry, layer_id: int):
    """Weight-loader replacement for meta-resident expert payloads.

    Keeps load-time semantics (every expert tensor is still read by the
    loader) but validates instead of copying: byte counts against the
    registry segments plus a head/tail byte-for-byte sample against the
    shard file, which catches a (layer, expert, shard) mix-up that a
    shape-only check misses. Returns the loader with a ``stats`` dict.
    """
    stats = {"checked": 0, "bytes": 0}

    def load(param, loaded_weight, weight_name, *, shard_id, expert_id):
        payload = registry.payload(layer_id, expert_id)
        seg = {
            "w1": payload.w13[0],
            "w3": payload.w13[1] if len(payload.w13) > 1 else payload.w13[0],
            "w2": payload.w2[0],
        }[shard_id]
        nbytes = seg.nbytes
        assert loaded_weight.numel() == nbytes, (
            f"{weight_name}: layer {layer_id} expert {expert_id} shard "
            f"{shard_id} has {loaded_weight.numel()} bytes, registry says {nbytes}"
        )
        cpu = (
            loaded_weight if loaded_weight.device.type == "cpu" else loaded_weight.cpu()
        )
        flat = cpu.reshape(-1)
        k = min(256, nbytes)
        if bytes(registry.pread(seg.shard, seg.src_offset, k)) != bytes(
            flat[:k].numpy().tobytes()
        ) or bytes(registry.pread(seg.shard, seg.src_offset + nbytes - k, k)) != bytes(
            flat[-k:].numpy().tobytes()
        ):
            raise RuntimeError(
                f"{weight_name}: layer {layer_id} expert {expert_id} shard "
                f"{shard_id} bytes disagree with the safetensors registry"
            )
        stats["checked"] += 1
        stats["bytes"] += nbytes
        if envs.SGLANG_SM12X_DISK_DEBUG.get() and stats["checked"] % 256 == 0:
            logger.info(
                "sm12x disk recorder: %d tensors / %d MiB checked",
                stats["checked"],
                stats["bytes"] >> 20,
            )

    load.stats = stats
    return load
