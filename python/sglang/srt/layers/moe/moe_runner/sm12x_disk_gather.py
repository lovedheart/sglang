"""VRAM slot cache for the sm12x_nvfp4_bf16_disk runner.

One ``ExpertSlotCache`` per (geometry, device) is shared by every streamed
MoE layer: slot ``k`` holds the payload of whichever expert currently maps
to it, so the CuTe-DSL kernel runs unmodified against ``[K, ...]`` weight
buffers with expert ids remapped through ``map[expert] -> slot``. Out-of-
range ids (padding, or experts evicted mid-call — impossible while
``K >= num_experts``) use the sentinel slot ``K`` and contribute nothing.

Scale handling has two regimes. For modelopt-style checkpoints the scale
factors are resident in VRAM (they are ~10x smaller), so the only per-call
traffic is the FP4 payload of *missing* experts. For checkpoints whose
scale tensors are too large for that (the disk backend detects
meta-device scales on the quant info), the scales ride the slot: pread
workers fetch them from the shards and swizzle them on the CPU, so a slot
fill moves payload AND scales together through the same staging tiers.
In both cases fills go pread -> pinned staging slab -> H2D on a side
stream into the victim slot.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.sm12x_disk_registry import (
    SafetensorsExpertRegistry,
)
from sglang.srt.layers.moe.moe_runner.sm12x_disk_source import (
    PlacementPolicy,
    SafetensorsSource,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.sm12x_nvfp4_bf16 import (
        Sm12xNvfp4W4A16MoeQuantInfo,
    )

logger = logging.getLogger(__name__)


class _DiskPrefetcher:
    """Per-layer popularity EMA -> ``posix_fadvise(WILLNEED)`` of the
    predicted experts of the next ``PREFETCH`` streamed layers, issued from
    one daemon-ish worker thread so the syscalls stay off the compute path.
    Opportunistic: jobs are dropped while the worker is busy. Talks to the
    source's ``advise`` hint, never to the registry directly."""

    def __init__(self, source, num_experts: int, layers_ahead: int, cold, decay=0.9):
        self.source = source
        self.num_experts = num_experts
        self.decay = decay
        # Callable reporting whether recent payload reads actually hit
        # storage (vs page cache); hints are pointless once reads are
        # RAM-speed.
        self.cold = cold
        ids = source.layer_ids
        self.ahead = {
            layer: ids[i + 1 : i + 1 + layers_ahead] for i, layer in enumerate(ids)
        }
        self.counts: dict[int, object] = {}
        self.missed: set[int] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sm12x-prefetch"
        )
        from threading import Lock

        self._lock = Lock()
        self._busy = False
        self.issued = self.dropped = 0

    def note_access(self, layer: int, uniq, topk: int, had_miss, resident) -> None:
        import numpy as np

        counts = self.counts.get(layer)
        if counts is None:
            counts = self.counts[layer] = np.zeros(self.num_experts)
        counts *= self.decay
        counts[uniq] += 1.0
        if had_miss:
            self.missed.add(layer)
        else:
            self.missed.discard(layer)
        if not self.cold():
            # Recent payload reads ran at page-cache speed: hints would only
            # cost syscalls. Dormant.
            return
        for nxt in self.ahead.get(layer, ()):
            nxt_counts = self.counts.get(nxt)
            if nxt_counts is None or nxt in self.missed:
                # Unobserved or I/O-pressured: warm the whole layer.
                predicted = np.arange(self.num_experts)
            else:
                p = min(4 * topk, self.num_experts)
                predicted = np.argpartition(nxt_counts, -p)[-p:]
            predicted = [e for e in predicted.tolist() if (nxt, e) not in resident]
            if predicted:
                self._submit(nxt, predicted)

    def _submit(self, layer: int, experts) -> None:
        with self._lock:
            if self._busy:
                self.dropped += 1
                return
            self._busy = True
        self._executor.submit(self._job, layer, experts)

    def _job(self, layer: int, experts) -> None:
        try:
            self.source.advise(layer, experts)
            self.issued += 1
        finally:
            with self._lock:
                self._busy = False

    def prewarm(self) -> None:
        """One-shot full-checkpoint warm (SGLANG_SM12X_DISK_PREWARM)."""
        import numpy as np

        all_e = np.arange(self.num_experts).tolist()
        for layer in self.source.layer_ids:
            # Queued unconditionally (not opportunistic like note_access).
            self._executor.submit(self._job, layer, all_e)

    def close(self) -> None:
        self._executor.shutdown(wait=False)


class _LayerPong:
    """Whole-layer ping-pong for the eager (prefill-sized) path.

    ``N`` VRAM buffers of ``[e + 1, ...]`` expert payloads (row ``e`` is the
    zeroed padding sentinel) with a pinned RAM mirror each. While layer L's
    GEMM runs, layer L+1's payload is pread into its buffer's RAM mirror and
    H2D'd on the copy stream; the launch then reads the layer tensors
    directly (identity expert->slot map) and merely ``wait_event``s on the
    load, so the GEMM never waits on a per-expert fill. Loads are kicked one
    layer ahead; a cold layer is still correct (the wait covers the full
    NVMe -> RAM -> VRAM chain, just without overlap).
    """

    MIN_TOKENS = 256  # below this an eager launch is decode-like, not prefill

    def __init__(self, cache: ExpertSlotCache, quant_info, n_bufs: int):
        e, dev = cache.e, cache.device
        self.e = e
        self.source = cache.source
        self.slot_bytes = cache.slot_bytes
        self.n2b = cache.slot_w2[0].numel()  # bytes of one expert's w2 payload
        self.n13b = cache.slot_bytes - self.n2b  # ... and of the w13 payload
        self.ident = torch.arange(e + 1, dtype=torch.int32, device=dev)
        self.bufs = [
            {
                "w13": torch.zeros(
                    (e + 1, *quant_info.w13.shape[1:]), dtype=torch.uint8, device=dev
                ),
                "w2": torch.zeros(
                    (e + 1, *quant_info.w2.shape[1:]), dtype=torch.uint8, device=dev
                ),
                "w13_sf": torch.zeros(
                    (e + 1, *quant_info.w13_sf.shape[1:]),
                    dtype=quant_info.w13_sf.dtype,
                    device=dev,
                ),
                "w2_sf": torch.zeros(
                    (e + 1, *quant_info.w2_sf.shape[1:]),
                    dtype=quant_info.w2_sf.dtype,
                    device=dev,
                ),
                "w13_gs": torch.zeros(
                    (e + 1, *quant_info.w13_gs.shape[1:]),
                    dtype=quant_info.w13_gs.dtype,
                    device=dev,
                ),
                "w2_gs": torch.zeros(
                    (e + 1, *quant_info.w2_gs.shape[1:]),
                    dtype=quant_info.w13_gs.dtype,
                    device=dev,
                ),
                "ram": torch.zeros(
                    (e, cache.slot_bytes), dtype=torch.uint8, pin_memory=True
                ),
                "holder": None,  # layer whose payloads are staged, fully read
                "ready": None,  # event closing the load on the copy stream
                "fut": None,  # the in-flight _load future
                "loading": None,  # layer the in-flight future is loading
                "busy": None,  # event closing the last launch that used it
            }
            for _ in range(n_bufs)
        ]
        ids = cache.source.layer_ids
        # Kick order over the STREAMED layers only, learned as prefill
        # calls reveal them (resident layers never reach prep): pass 1 is
        # cold-chain, from pass 2 the chain is complete and wraps.
        self.all_ids = ids
        self.stream: set[int] = set()
        self.nxt: dict[int, int] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=len(self.bufs), thread_name_prefix="sm12x-pong"
        )

    def usable(self, layer_id: int, t: int, capturing: bool) -> bool:
        return not capturing and t >= self.MIN_TOKENS

    def prep(self, layer_id: int, quant_info):
        """Make ``layer_id``'s buffer current; returns the launch dict."""
        self._note(layer_id)
        b = self.bufs[layer_id % len(self.bufs)]
        nxt = self.nxt.get(layer_id)
        if nxt is not None:
            self._kick(nxt)  # first: never stall the pipeline
        for _ in range(2):  # the in-flight load may be ours, or in our way
            if b["holder"] == layer_id:
                break
            if b["loading"] is None:
                self._kick(layer_id)  # cold: kick, then wait
            b["fut"].result()  # cold: host waits (GPU queue drains)
        torch.cuda.current_stream().wait_event(b["ready"])
        # Scales travel with the launch (the kick-ahead loader has no
        # quant_info for its target layer): four small D2D copies, ordered
        # on the compute stream behind the buffer's H2D and any earlier
        # launch that used it.
        copy13, copy2 = b["w13_sf"][: self.e], b["w2_sf"][: self.e]
        copy13.copy_(quant_info.w13_sf, non_blocking=True)
        copy2.copy_(quant_info.w2_sf, non_blocking=True)
        b["w13_gs"][: self.e].copy_(quant_info.w13_gs, non_blocking=True)
        b["w2_gs"][: self.e].copy_(quant_info.w2_gs, non_blocking=True)
        return {
            "w13": b["w13"],
            "w13_sf": b["w13_sf"],
            "w13_gs": b["w13_gs"],
            "w2": b["w2"],
            "w2_sf": b["w2_sf"],
            "w2_gs": b["w2_gs"],
            "k": self.e + 1,
            "buf": b,
        }

    def mark(self, layer_id: int) -> None:
        """Record that a launch on layer's buffer was issued (gates reload)."""
        b = self.bufs[layer_id % len(self.bufs)]
        ev = torch.cuda.Event()
        ev.record()
        b["busy"] = ev

    def _note(self, layer_id: int) -> None:
        if layer_id in self.stream:
            return
        self.stream.add(layer_id)
        order = sorted(self.stream)
        self.nxt = {x: y for x, y in zip(order, order[1:] + order[:1])}

    def _kick(self, layer_id: int) -> None:
        b = self.bufs[layer_id % len(self.bufs)]
        if b["holder"] == layer_id or b["loading"] is not None:
            return  # current, or the buffer is mid-load of something else
        b["loading"] = layer_id
        b["fut"] = self.executor.submit(self._load, b, layer_id)

    def _load(self, b, layer_id: int) -> None:
        from sglang.srt.runtime_context import get_stream

        if b["ready"] is not None:
            # The previous load's H2D may still be draining from the pinned
            # mirror on the GPU; preading now would clobber rows mid-copy.
            b["ready"].synchronize()
        futs = self.source.pread_layer(layer_id, b["ram"])
        copy = get_stream("sm12x_disk_copy")
        with torch.cuda.stream(copy):
            if b["busy"] is not None:
                copy.wait_event(b["busy"])  # previous holder's GEMM
            for f in futs:
                f.result()  # payloads now sit in the pinned mirror
            e, n13b = self.e, self.n13b
            for i in range(e):
                row = b["ram"][i]
                b["w13"][i].copy_(row[:n13b].view(b["w13"][i].shape), non_blocking=True)
                b["w2"][i].copy_(
                    row[n13b : self.slot_bytes].view(b["w2"][i].shape),
                    non_blocking=True,
                )
            ev = torch.cuda.Event()
            ev.record(copy)
        b["ready"] = ev
        b["holder"] = layer_id
        b["loading"] = None

    def close(self) -> None:
        self.executor.shutdown(wait=False)


class ExpertSlotCache:
    """LRU slot pool over the expert payloads of one geometry.

    ``K`` (slots == kernel ``num_experts``) is a power of two from the
    ``SGLANG_SM12X_DISK_SLOTS_MB`` budget, floored at the real expert count
    so the experts of any single launch always fit simultaneously.
    """

    def __init__(
        self,
        registry: SafetensorsExpertRegistry,
        quant_info: Sm12xNvfp4W4A16MoeQuantInfo,
        num_experts: int,
        device: torch.device,
    ):
        self.device = device
        e = num_experts
        i = quant_info.w2.shape[2] * 2
        h = quant_info.w13.shape[2] * 2
        # Meta scale tensors == the checkpoint's scale factors stream from
        # disk with the payload instead of living in quant_info.
        self.sf_streamed = quant_info.w13_sf.device.type == "meta"
        self.slot_bytes = 3 * i * h // 2
        if self.sf_streamed:
            for t in (quant_info.w13_sf, quant_info.w2_sf):
                assert t.dtype in (torch.float8_e4m3fn, torch.uint8)
            self.slot_bytes += (
                quant_info.w13_sf[0].numel() + quant_info.w2_sf[0].numel()
            )
        budget = int(envs.SGLANG_SM12X_DISK_SLOTS_MB.get()) << 20
        n = max(budget // self.slot_bytes, 1)
        k = min(1 << (n - 1).bit_length(), 4096)
        # Floor at the expert count: every expert a single launch can
        # simultaneously address must have a slot (K doubles as the
        # kernel's num_experts).
        self.k = max(k, e)
        self.debug = envs.SGLANG_SM12X_DISK_DEBUG.get()

        self.slot_w13 = torch.empty(
            (self.k, *quant_info.w13.shape[1:]), dtype=torch.uint8, device=device
        )
        self.slot_w2 = torch.empty(
            (self.k, *quant_info.w2.shape[1:]), dtype=torch.uint8, device=device
        )
        self.slot_w13_sf = torch.empty(
            (self.k, *quant_info.w13_sf.shape[1:]),
            dtype=quant_info.w13_sf.dtype,
            device=device,
        )
        self.slot_w2_sf = torch.empty(
            (self.k, *quant_info.w2_sf.shape[1:]),
            dtype=quant_info.w2_sf.dtype,
            device=device,
        )
        self.slot_w13_gs = torch.zeros(
            (self.k, *quant_info.w13_gs.shape[1:]),
            dtype=quant_info.w13_gs.dtype,
            device=device,
        )
        self.slot_w2_gs = torch.zeros(
            (self.k, *quant_info.w2_gs.shape[1:]),
            dtype=quant_info.w2_gs.dtype,
            device=device,
        )
        # Placement policy (v1: LRU into VRAM slots). slot_of aliases the
        # policy's live map so harnesses can inspect it.
        self.placement = PlacementPolicy(self.k)
        self.slot_of = self.placement.slot_of

        # Per-layer expert->slot tables ([e] is the padding index); the
        # change signature lets an unchanged call skip the blocking upload.
        self.e = e
        self._map: dict[int, dict] = {}

        # Byte-weighted EMA of observed pread throughput (B/s). Starts
        # "cold" so the first requests get prefetch hints.
        self.io_speed = 0.0
        self._io_gate = 4 << 30  # below this == reads are on storage
        self.source = SafetensorsSource(
            registry,
            self.slot_w13,
            self.slot_w2,
            self._note_io,
            slot_sf=(self.slot_w13_sf, self.slot_w2_sf) if self.sf_streamed else None,
        )
        # In-graph mode owns the pool exclusively (device-side LRU); the
        # host prefetch pipeline and the prefill ping-pong stay off.
        self.ingraph = None
        in_graph = bool(envs.SGLANG_SM12X_DISK_INGRAPH.get())
        layers_ahead = 0 if in_graph else int(envs.SGLANG_SM12X_DISK_PREFETCH.get())
        prewarm = False if in_graph else bool(envs.SGLANG_SM12X_DISK_PREWARM.get())
        self.prefetcher = (
            _DiskPrefetcher(
                self.source, e, layers_ahead, cold=lambda: self.io_speed < self._io_gate
            )
            if (layers_ahead > 0 or prewarm)
            else None
        )
        if prewarm and self.prefetcher is not None:
            self.prefetcher.prewarm()
        n_pp = 0 if in_graph else min(
            max(int(envs.SGLANG_SM12X_DISK_PREFILL_LAYERS.get()), 0), 4
        )
        if n_pp and self.sf_streamed:
            # pread_layer stages payloads only; the ping-pong path would
            # also need per-layer scale staging. Not wired; not needed
            # (checkpoints that stream scales are GB-per-layer models for
            # which the whole-layer ring is unaffordable anyway).
            logger.info(
                "sm12x disk: PREFILL_LAYERS disabled (scale-streamed checkpoint layout)"
            )
            n_pp = 0
        self.pong = _LayerPong(self, quant_info, n_pp) if n_pp else None
        if in_graph:
            from sglang.srt.layers.moe.moe_runner.sm12x_disk_ingraph import (
                InGraphPager,
            )

            self.ingraph = InGraphPager(self, registry, quant_info, e, device)

    # -- policy ---------------------------------------------------------

    _dbg_calls = 0

    def _dbg(self, layer_id, quant_info):
        if (
            self.ingraph is not None
            or self._dbg_calls >= 1000000
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        self._dbg_calls += 1
        import hashlib

        dig = []
        for ex in range(self.e):
            sl = self.slot_of.get((layer_id, ex))
            if sl is None:
                continue
            h = hashlib.md5(
                b"".join(
                    bytes(t[sl].view(torch.uint8).cpu().numpy())
                    for t in (
                        self.slot_w13,
                        self.slot_w2,
                        self.slot_w13_sf,
                        self.slot_w2_sf,
                    )
                )
                + bytes(
                    self.slot_w13_gs[sl : sl + 1].view(torch.uint8).cpu().numpy()
                )
                + bytes(
                    self.slot_w2_gs[sl : sl + 1].view(torch.uint8).cpu().numpy()
                )
            ).hexdigest()[:8]
            dig.append(f"{ex}:{sl}:{h}")
        logger.info("IGDBG L%d %s", layer_id, " ".join(dig[:6]))

    def reset_state(self) -> None:
        """Drop all placements (harness cold-start hook), both LRUs."""
        self.placement.reset()
        if self.ingraph is not None:
            self.ingraph.reset()

    @property
    def hits(self) -> int:
        return self.placement.hits

    @property
    def misses(self) -> int:
        return self.placement.misses

    def _assign(self, layer: int, expert: int) -> tuple[int, bool]:
        return self.placement.place(layer, expert)

    # -- per-call streaming ----------------------------------------------

    def _note_io(self, speed: float) -> None:
        self.io_speed = 0.95 * self.io_speed + 0.05 * speed

    @torch.inference_mode(False)
    def stream_weights(
        self, layer_id: int, ids_cpu: torch.Tensor, quant_info
    ) -> torch.Tensor:
        """Resolve ``ids_cpu`` (int32 [T, topk] on host) to slot ids, filling
        misses; returns the int32 device-side ``[e + 1]`` expert->slot map
        (index ``e`` is the padding/sentinel row holding ``K``).

        Runs outside InferenceMode: a graph break re-executes this on
        replay (outside the capture-time inference mode), so every buffer
        it touches must be a plain tensor, writable in either regime.
        """
        import numpy as np

        uniq = np.unique(ids_cpu.numpy())
        uniq = uniq[uniq < self.e]

        needed = []
        for expert in uniq.tolist():
            slot, miss = self._assign(layer_id, expert)
            needed.append((expert, slot, miss))

        # Start the disk reads early: one pinned slab per miss (bounded by
        # the staging-ring depth) with preads on the pool, so the I/O
        # overlaps the bookkeeping here and the H2D stream below.
        pending = [
            (expert, slot, self.source.start_fill(layer_id, expert))
            for expert, slot, miss in needed
            if miss
        ]
        had_miss = bool(pending)

        compute = torch.cuda.current_stream()
        if pending:
            ev_compute = torch.cuda.Event()
            ev_compute.record(compute)
            from sglang.srt.runtime_context import get_stream

            copy = get_stream("sm12x_disk_copy")
            with torch.cuda.stream(copy):
                copy.wait_event(ev_compute)
                for expert, slot, handle in pending:
                    # Payload bytes via the source (pread -> pinned slab ->
                    # H2D); for layout A scales are VRAM-resident and get
                    # copied slot-side, for layout B the source streams
                    # (and swizzles) them with the payload.
                    if handle is not None:
                        self.source.finish_fill(handle, slot)
                    else:
                        self.source.submit_fill(layer_id, expert, slot)
                    if not self.sf_streamed:
                        self.slot_w13_sf[slot].copy_(
                            quant_info.w13_sf[expert], non_blocking=True
                        )
                        self.slot_w2_sf[slot].copy_(
                            quant_info.w2_sf[expert], non_blocking=True
                        )
                    self.slot_w13_gs[slot].copy_(
                        quant_info.w13_gs[expert], non_blocking=True
                    )
                    self.slot_w2_gs[slot].copy_(
                        quant_info.w2_gs[expert], non_blocking=True
                    )
                ev_fill = torch.cuda.Event()
                ev_fill.record(copy)
            compute.wait_event(ev_fill)

        m = self._map.get(layer_id)
        if m is None:
            m = self._map[layer_id] = {
                "gpu": torch.full(
                    (self.e + 1,), self.k, dtype=torch.int32, device=self.device
                ),
                "cpu": torch.full(
                    (self.e + 1,), self.k, dtype=torch.int32, pin_memory=True
                ),
                "sig": None,
            }
        sig = (uniq.data.tobytes(), bytes(np.asarray([s for _, s, _ in needed], "i4")))
        if m["sig"] != sig:
            map_np = m["cpu"].numpy()
            map_np[:] = self.k
            for expert, slot, _ in needed:
                map_np[expert] = slot
            m["gpu"].copy_(m["cpu"])  # blocking: the host rewrites it next call
            m["sig"] = sig

        if self.debug and (self.hits + self.misses) % 1024 == 0:
            logger.info(
                "sm12x disk slots: %d hits / %d misses, %d/%d slots live",
                self.hits,
                self.misses,
                len(self.slot_of),
                self.k,
            )
        if self.prefetcher is not None:
            self.prefetcher.note_access(
                layer_id, uniq, ids_cpu.shape[1], had_miss, resident=self.slot_of
            )
        if self.debug:
            torch.cuda.synchronize()
            self._dbg(layer_id, quant_info)
        return m["gpu"]

    def close(self) -> None:
        self.source.close()
        if self.prefetcher is not None:
            self.prefetcher.close()
        if self.pong is not None:
            self.pong.close()


def get_slot_cache(
    registry: SafetensorsExpertRegistry,
    quant_info: Sm12xNvfp4W4A16MoeQuantInfo,
    num_experts: int,
) -> ExpertSlotCache:
    from sglang.srt.runtime_context import get_buffer

    h = quant_info.w13.shape[2] * 2
    i = quant_info.w2.shape[2] * 2
    # The payload and (for scale-streamed checkpoints) the scales may be
    # meta; the per-expert global scales are always resident and carry the
    # real device.
    device = quant_info.w13_gs.device
    if device.type == "meta":
        raise NotImplementedError(
            "sm12x disk layers need resident fp32 weight_scale_2 tensors "
            "(w13_gs); the checkpoint's per-expert scalars must be loaded."
        )
    if device.type != "cuda":
        raise RuntimeError(f"sm12x disk cache on a {device.type} device?")
    name = (
        f"sm12x_disk_cache:{device}:{h}:{i}:{num_experts}:"
        f"{envs.SGLANG_SM12X_DISK_SLOTS_MB.get()}:"
        f"{'sf' if quant_info.w13_sf.device.type == 'meta' else 'res'}:"
        f"{'ig' if envs.SGLANG_SM12X_DISK_INGRAPH.get() else 'host'}"
    )
    # Allocate outside InferenceMode: the cache is created under whichever
    # mode the first forward happens to use (often inference), but must be
    # writable from graph-replay breaks, which run without it.
    with torch.inference_mode(False):
        return get_buffer(
            name, lambda: ExpertSlotCache(registry, quant_info, num_experts, device)
        )
