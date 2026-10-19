"""Expert payload sources and placement for the sm12x disk MoE runner (P6).

The seam between ``ExpertSlotCache`` policy (which expert owns which slot)
and the bytes themselves. v1 has one implementation, ``SafetensorsSource``
(original checkpoint files -> pinned host slot or slab -> H2D); the
ig_moe host-arena tier (plan.md P6b, for models whose routed bytes exceed
~3x VRAM) plugs in as another ``ExpertSource`` + a ``PlacementPolicy``
answer without touching the cache, the fused func or the kernel.

Contract:
  - ``submit_fill(layer, expert, slot)`` writes the expert's FP4 payload
    (and, for scale-streamed checkpoints, its swizzled scale factors)
    into ``slot_w13[slot]`` / ``slot_w2[slot]``, issuing copies on the
    CURRENT stream (callers wrap the copy stream); returns the completion
    event the caller pairs with ``wait_event`` before compute. For layout A
    checkpoints, scale copies are NOT the source's concern -- those scales
    are VRAM-resident; for layout B they ride the slot as extra regions.
  - ``advise(layer, experts)`` is a hint only (fadvise); never blocking.
  - ``PlacementPolicy.place`` answers ``(slot, miss)``; a future host tier
    would extend it (never silently -- unknown placements raise).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_registry import (
        SafetensorsExpertRegistry,
    )

_PREAD_WORKERS = 4
_SLABS_MAX = 256


class _SlabRing:
    """Pinned staging slabs, recycled once both the pread and the H2D that
    consumed a slab have completed (busy list kept in issue order)."""

    def __init__(self, nbytes: int, device: torch.device, n_slabs: int):
        self.slabs = [
            torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(n_slabs)
        ]
        self.free = list(range(n_slabs))
        self.busy: list[tuple[int, object, torch.cuda.Event]] = []

    def acquire(self) -> tuple[int, torch.Tensor]:
        while not self.free:
            idx, fut, ev = self.busy.pop(0)
            fut.result()
            ev.synchronize()
            self.free.append(idx)
        idx = self.free.pop(0)
        return idx, self.slabs[idx]

    def release(self, idx: int, fut, ev: torch.cuda.Event) -> None:
        self.busy.append((idx, fut, ev))


class _HostArena:
    """Pinned host-DRAM expert slots for ONE MoE layer
    (``SGLANG_SM12X_DISK_HOST_SLOTS``): a cache in front of NVMe, counted
    in experts. Misses are pread DIRECTLY into a slot (the cache slot
    doubles as the staging buffer -- no slab ring on this path), cached
    experts fill VRAM with a bare H2D. A slot is never written over
    while the last H2D that read it, or an older pread that filled it,
    is still in flight: eviction synchronizes the completion event and
    never picks a slot whose own pread has not landed."""

    def __init__(self, n_slots: int, nbytes: int):
        self.n_slots = n_slots
        self.nbytes = nbytes
        self.buf = None  # pinned lazily: never-touched layers cost nothing
        self.tag: OrderedDict[int, int] = OrderedDict()  # expert -> slot
        self.free: list[int] = []
        self.filling: set[int] = set()  # slots with an unfinished pread
        self.last_ev: dict[int, torch.cuda.Event] = {}
        self.hits = 0

    def lookup(self, expert: int):
        idx = self.tag.get(expert)
        if idx is None:
            return None
        self.tag.move_to_end(expert)  # LRU order
        self.hits += 1
        return idx

    def acquire(self, expert: int):
        """Reserve ``expert``'s cache slot (pinned lazily), or return
        None when every slot is mid-fill -- the caller then falls back
        to the slab ring."""
        if self.buf is None:
            self.buf = torch.empty(
                self.n_slots, self.nbytes, dtype=torch.uint8, pin_memory=True
            )
            self.free = list(range(self.n_slots))
        idx = self.tag.pop(expert, None)
        if idx is None:
            if self.free:
                idx = self.free.pop(0)
            else:
                # LRU-evict the oldest cached expert whose slot is not
                # mid-fill (a filled slot's bytes must not be raced).
                victim = next(
                    (k for k, j in self.tag.items() if j not in self.filling),
                    None,
                )
                if victim is None:
                    return None
                idx = self.tag.pop(victim)
                ev = self.last_ev.pop(idx, None)
                if ev is not None:
                    ev.synchronize()  # do not clobber an in-flight H2D
        self.tag[expert] = idx
        self.filling.add(idx)
        return idx

    def finish(self, idx: int, ev: torch.cuda.Event) -> None:
        """Mark a slot's pread+H2D sequence complete; ``ev`` gates the
        slot's next overwrite."""
        self.filling.discard(idx)
        self.last_ev[idx] = ev


def _parse_host_slots(spec: str, layer_ids: list[int]) -> list[int]:
    """Per-layer arena sizes, in the order of ``layer_ids``. Accepted
    spellings:

      ``"32"``               uniform 32 slots on every MoE layer
      ``"0;2;3;4;5"``        per-layer counts, positionally (i-th entry =
                             i-th MoE layer)
      ``"0-15:8,16-39:16"``  ``layer_range:count`` terms, comma-separated,
                             keyed on the model LAYER IDS (same
                             coordinate as RESIDENT_LAYERS); a bare ``K``
                             term sets the default for layers no range
                             covers (later terms win on overlap). Layers
                             left at 0 get no arena at all.
    """
    spec = spec.strip()
    n = len(layer_ids)
    usage = (
        "SGLANG_SM12X_DISK_HOST_SLOTS must be a non-negative int, a "
        f"semicolon list of {n} counts, or comma-separated "
        "'layer-range:count' terms (e.g. '0-15:8,16-39:16'); got "
        f"{spec!r}"
    )
    try:
        if ";" not in spec and ":" not in spec:
            value = int(spec)
            if value < 0:
                raise ValueError(spec)
            return [value] * n
        if ";" in spec:
            per = [int(p) for p in spec.split(";")]
            if len(per) != n or any(v < 0 for v in per):
                raise ValueError(spec)
            return per
        per = [0] * n
        for term in spec.split(","):
            bounds, sep, value = term.partition(":")
            if not sep:  # bare int: default for every layer
                default = int(term)
                if default < 0:
                    raise ValueError(spec)
                per = [v if v else default for v in per]
                continue
            count = int(value)
            if count < 0:
                raise ValueError(spec)
            bounds = bounds.strip()
            lo, _, hi = bounds.partition("-")
            lo = int(lo) if lo else 0
            hi = int(hi) if hi else lo
            if hi < lo:
                raise ValueError(spec)
            touched = False
            for i, lid in enumerate(layer_ids):
                if lo <= lid <= hi:
                    per[i] = count
                    touched = True
            if not touched:  # typo guard: a range over no MoE layer
                raise ValueError(spec)
        return per
    except ValueError:
        raise ValueError(usage) from None


class _ArenaSet:
    """Per-layer view over ``HOST_SLOTS``: layers with 0 slots allocate no
    memory and fall back to the slab ring (or plain paged reads)."""

    def __init__(self, spec: str, layer_ids: list[int], nbytes: int):
        # Positionally over the MoE layers in ascending-id order: the
        # spec's i-th count applies to layer_ids[i] (the i-th MoE layer).
        self.bylayer: dict[int, _HostArena] = {}
        for lid, n in zip(layer_ids, _parse_host_slots(spec, layer_ids)):
            if n > 0:
                # Pinned memory is allocated on the layer's first fill,
                # so layers excluded by RESIDENT_LAYERS cost nothing.
                self.bylayer[lid] = _HostArena(min(n, 4096), nbytes)

    def arena(self, layer: int):
        return self.bylayer.get(layer)

    @property
    def hits(self) -> int:
        return sum(a.hits for a in self.bylayer.values())


_SWZ: dict = {}
_PREAD_TL = __import__("threading").local()


def _swizzle_idx(rows, cols):
    """Flat gather indices of the NVFP4 128x4 scale swizzle for an exact
    (rows % 128 == 0, cols % 4 == 0) linear matrix. Matches
    ``swizzle_blockscale``'s reshape(r/128, 4, 32, c/4, 4) ->
    permute(0, 3, 2, 1, 4) reading order. Cached; a few MB at V4.1 size."""
    idx = _SWZ.get((rows, cols))
    if idx is None:
        import numpy as np

        # Output iterates (r128, c4, b32, a4, c1); linear row r128*128 +
        # a4*32 + b32, col c4*4 + c1 (see swizzle_blockscale's permute).
        r128, c4, b32, a4, c1 = np.indices((rows // 128, cols // 4, 32, 4, 4))
        idx = _SWZ[(rows, cols)] = (
            (r128 * 128 + a4 * 32 + b32) * cols + (c4 * 4 + c1)
        ).reshape(-1)
    return idx


def _pread_into(
    registry,
    layer: int,
    expert: int,
    slab: torch.Tensor,
    w2_base: int,
    meter=None,
    sfplan=None,
) -> None:
    import numpy as np

    t0 = time.perf_counter()
    view = slab.numpy()
    payload = registry.payload(layer, expert)
    nbytes = 0
    for base, segs in ((0, payload.w13), (w2_base, payload.w2)):
        for seg in segs:
            lo = base + seg.dst_offset
            view[lo : lo + seg.nbytes] = np.frombuffer(
                registry.pread(seg.shard, seg.src_offset, seg.nbytes), np.uint8
            )
            nbytes += seg.nbytes
    if sfplan is not None:
        # Layout B: the linear e4m3 scale tensors ride the slot too, so
        # gather each region's segments into a zeroed linear scratch and
        # permute it swizzled into the slot's sf region.
        scratch = getattr(_PREAD_TL, "scratch", None)
        if scratch is None or scratch.shape[0] < sfplan["max"]:
            scratch = _PREAD_TL.scratch = np.empty(sfplan["max"], np.uint8)
        for base, rows, cols, attr in sfplan["regions"]:
            scratch[: rows * cols] = 0
            for seg in getattr(payload, attr):
                scratch[seg.dst_offset : seg.dst_offset + seg.nbytes] = np.frombuffer(
                    registry.pread(seg.shard, seg.src_offset, seg.nbytes), np.uint8
                )
                nbytes += seg.nbytes
            view[base : base + rows * cols] = scratch[: rows * cols][
                _swizzle_idx(rows, cols)
            ]
    if meter is not None:
        meter(nbytes / max(time.perf_counter() - t0, 1e-6))


class ExpertSource(ABC):
    """Where per-expert FP4 payloads come from and how they reach a slot."""

    #: MoE layer ids this source can serve (prefetch scheduling order).
    layer_ids: list[int]

    @abstractmethod
    def submit_fill(self, layer: int, expert: int, slot: int) -> torch.cuda.Event:
        """Fill ``slot`` with ``layer``'s expert ``expert``; work is issued
        on the calling stream. Returns an event that completes when the
        slot is safe to read."""

    def start_fill(self, layer: int, expert: int):
        """Optional pipelining hook: begin fetching ``layer``/``expert``
        (host-side, off the critical path) and return an opaque handle
        for :meth:`finish_fill`. ``None`` means no staging capacity is
        free right now; callers then fall back to :meth:`submit_fill` at
        copy time."""
        return

    def finish_fill(self, handle, slot: int) -> None:
        """Issue the payload copies for a :meth:`start_fill` handle on the
        current stream. Must be called before the slot is read."""
        raise AssertionError("finish_fill called without a handle")

    @abstractmethod
    def advise(self, layer: int, experts) -> None:
        """Hint that ``experts`` of ``layer`` will be needed soon."""

    def close(self) -> None:
        pass


class SafetensorsSource(ExpertSource):
    """v1 source: FP4 payloads paged from the original checkpoint files."""

    def __init__(
        self,
        registry: SafetensorsExpertRegistry,
        slot_w13: torch.Tensor,
        slot_w2: torch.Tensor,
        meter,
        slot_sf: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        self.registry = registry
        self.slot_w13 = slot_w13
        self.slot_w2 = slot_w2
        # Slab layout: [w13 payload | w2 payload (| w13 sf | w2 sf)];
        # segment dst offsets are per-tensor, so w2 runs are based past
        # the w13 region. With slot_sf (layout B checkpoints) the scale
        # factors are streamed too and swizzled by the pread workers.
        self.w2_base = slot_w13[0].numel()
        self.sfplan = None
        if slot_sf is not None:
            assert registry.sf_streamed, (
                "scale-streamed slot cache given a checkpoint whose scales "
                "are modelopt-resident"
            )
            sf13, sf2 = slot_sf
            (r13, c13), (r2, c2) = registry.sf_raw
            assert sf13[0].shape == (r13, c13) and sf2[0].shape == (r2, c2), (
                "streamed scale tensors must be exact (no 128x4 padding): "
                f"slot {tuple(sf13[0].shape)}, {tuple(sf2[0].shape)} vs ckpt "
                f"{r13}x{c13}, {r2}x{c2}"
            )
            b13 = self.w2_base + slot_w2[0].numel()
            b2 = b13 + r13 * c13
            self.sfplan = {
                "max": max(r13 * c13, r2 * c2),
                "regions": [
                    (b13, r13, c13, "w13_sf"),
                    (b2, r2, c2, "w2_sf"),
                ],
                "bytes": r13 * c13 + r2 * c2,
            }
            self.slot_sf = (
                sf13.view(torch.uint8),
                sf2.view(torch.uint8),
            )
        self.meter = meter
        total = (
            self.w2_base
            + slot_w2[0].numel()
            + (self.sfplan["bytes"] if self.sfplan else 0)
        )
        n_slabs = min(max(envs.SGLANG_SM12X_DISK_SLABS.get(), 1), _SLABS_MAX)
        self.ring = _SlabRing(total, slot_w13.device, n_slabs)
        self.executor = ThreadPoolExecutor(
            max_workers=_PREAD_WORKERS, thread_name_prefix="sm12x-disk"
        )
        self.layer_ids = list(registry.layer_ids)
        spec = envs.SGLANG_SM12X_DISK_HOST_SLOTS.get()
        self.arena = (
            _ArenaSet(spec, self.layer_ids, total)
            if spec.strip() not in ("", "0")
            else None
        )

    def start_fill(self, layer: int, expert: int):
        # Host arena first: a hit means no disk I/O at all; a miss
        # preads straight into the cache slot (the slot is itself the
        # pinned staging buffer -- NVMe -> host slot -> VRAM, no ring).
        if self.arena is not None:
            arena = self.arena.arena(layer)
            if arena is not None:
                idx = arena.lookup(expert)
                if idx is not None:
                    return ("h", arena, idx)
                idx = arena.acquire(expert)
                if idx is not None:
                    fut = self.executor.submit(
                        _pread_into,
                        self.registry,
                        layer,
                        expert,
                        arena.buf[idx],
                        self.w2_base,
                        self.meter,
                        self.sfplan,
                    )
                    return ("a", arena, idx, fut)
        if not self.ring.free:
            return None
        idx, slab = self.ring.acquire()
        fut = self.executor.submit(
            _pread_into,
            self.registry,
            layer,
            expert,
            slab,
            self.w2_base,
            self.meter,
            self.sfplan,
        )
        return ("s", idx, slab, fut)

    def finish_fill(self, handle, slot: int) -> None:
        if handle[0] in ("h", "a"):
            if handle[0] == "a":
                _, arena, idx, fut = handle
                fut.result()  # payload already sits in the cache slot
            else:
                _, arena, idx = handle
            self._copy_flat(arena.buf[idx], slot)
            arena.finish(idx, self._new_ev())  # gates the slot's next use
            return
        _, idx, slab, fut = handle
        fut.result()
        self._copy_flat(slab.view(torch.uint8), slot)
        ev = torch.cuda.Event()
        ev.record()  # caller's (copy) stream; slab is free once this lands
        self.ring.release(idx, fut, ev)

    def _new_ev(self) -> torch.cuda.Event:
        ev = torch.cuda.Event()
        ev.record()  # caller's (copy) stream
        return ev

    def _copy_flat(self, flat: torch.Tensor, slot: int) -> None:
        n13_half = self.w2_base
        self.slot_w13[slot].copy_(
            flat[:n13_half].view(self.slot_w13[slot].shape), non_blocking=True
        )
        if self.sfplan is None:
            self.slot_w2[slot].copy_(
                flat[n13_half:].view(self.slot_w2[slot].shape), non_blocking=True
            )
            return
        (b13, _, _, _), (b2, _, _, _) = self.sfplan["regions"]
        self.slot_w2[slot].copy_(
            flat[n13_half:b13].view(self.slot_w2[slot].shape), non_blocking=True
        )
        r13, r2 = self.slot_sf
        r13[slot].copy_(flat[b13:b2].view(r13[slot].shape), non_blocking=True)
        r2[slot].copy_(flat[b2:].view(r2[slot].shape), non_blocking=True)

    def submit_fill(self, layer: int, expert: int, slot: int) -> torch.cuda.Event:
        handle = self.start_fill(layer, expert)
        if handle is None:
            # No arena slot and no slab left: synchronous pread -> H2D
            # at copy time (the ring blocks until one recycles).
            idx, slab = self.ring.acquire()
            fut = self.executor.submit(
                _pread_into,
                self.registry,
                layer,
                expert,
                slab,
                self.w2_base,
                self.meter,
                self.sfplan,
            )
            fut.result()
            self._copy_flat(slab.view(torch.uint8), slot)
            ev = torch.cuda.Event()
            ev.record()
            self.ring.release(idx, fut, ev)
        else:
            self.finish_fill(handle, slot)
        ev = torch.cuda.Event()
        ev.record()
        return ev

    def advise(self, layer: int, experts) -> None:
        self.registry.fadvise(layer, experts)

    def pread_layer(self, layer: int, dst: torch.Tensor):
        """Pread every expert payload of ``layer`` into rows 0..E-1 of the
        pinned ``dst`` ([E, payload_bytes] uint8) on the pool; returns one
        future per row. For the layer ping-pong loader only."""
        return [
            self.executor.submit(
                _pread_into, self.registry, layer, i, dst[i], self.w2_base, self.meter
            )
            for i in range(dst.shape[0])
        ]

    def close(self) -> None:
        self.executor.shutdown(wait=False)


class PlacementPolicy:
    """LRU placement of (layer, expert) -> VRAM slot. v1: everything the
    cache can address is a ``vram_slot``; the host-arena answer of the
    ig_moe tier (P6b) would branch here by miss fraction/size, never
    silently, and the cache would never learn the difference."""

    def __init__(self, k: int):
        self.k = k
        self.slot_of: dict[tuple[int, int], int] = {}
        self.lru: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.free = list(range(k))
        self.hits = self.misses = 0

    def place(self, layer: int, expert: int) -> tuple[int, bool]:
        """Return ``(slot, miss)`` -- always a VRAM slot in v1."""
        key = (layer, expert)
        slot = self.slot_of.get(key)
        if slot is not None:
            self.lru.move_to_end(key)
            self.hits += 1
            return slot, False
        if self.free:
            slot = self.free.pop()
        else:
            old, slot = next(iter(self.lru.items()))
            self.lru.pop(old)
            self.slot_of.pop(old)
        self.slot_of[key] = slot
        self.lru[key] = slot
        self.misses += 1
        return slot, True

    def reset(self) -> None:
        """Drop all placements (harness cold-start hook)."""
        self.slot_of.clear()
        self.lru = OrderedDict()
        self.free = list(range(self.k))
