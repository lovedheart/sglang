"""
Copyright 2023-2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

MambaCheckpointPool — the radix prefix cache's int8-compressed store for cached
linear-attention (KDA / GDN / Mamba2 gated-delta-rule) recurrent states.

It decouples the *cached* states (radix-owned, idle, compressed) from the *active*
``MambaPool`` (running requests, full precision, kernel-facing). The radix stores
one cached state per node HERE; on a prefix-cache hit it is dequantized back into
a fresh active slot (copy-on-write).

Per cached slot it holds:
  * the SSM temporal state in **int8** (per-(head,k-channel) symmetric), via the
    embedded ``Int8CheckpointStore`` — ~2x more cached states than bf16,
    quality-safe (quantized once on store, dequantized once on a hit; never
    re-enters the recurrence as a quant->dequant loop).
  * the conv1d window state at its native dtype (tiny, W-1 tokens; not worth
    quantizing).

Why int8 (not fp8): a cached checkpoint is loaded ONCE on a cache hit, then
decoding continues at full precision, so the only error is a single rounding of
S. The temporal state is roughly uniformly distributed, so int8-per-(head,
k-channel) beats fp8-e4m3 at the same 1 byte (fp8 wastes bits on the exponent).
The scale axis (reduces over d_v) matches the per-k-channel decay diag(alpha), so
the large state entries keep ~bf16 precision and the error concentrates on small
entries that barely affect the readout. Storing cached states int8 gives ~2x the
cached-prefix capacity at fixed memory, and composes with host-offload
(--enable-hierarchical-cache) which it also halves.

This is strategy-agnostic: whether the active slot to be cached was produced by
the ``no_buffer`` donate (copy_from) or the ``extra_buffer`` ping-pong track
buffer (spec path), both converge on "an active slot becomes the cached
``mamba_value``" — which is exactly the (store_from_active) hook here. Slot
lifecycle is owned by the caller via the embedded ``MambaSlotAllocator``.

With ``host_slots > 0`` the pool spans two tiers as ONE logical pool: GPU
checkpoint slots (allocator ids ``[1, num_slots]``) and pinned-RAM checkpoint
slots (ids ``(num_slots, num_slots + host_slots]``, the allocator's host tier).
Both tiers store the identical int8 format; a RAM store quantizes into a GPU
staging row and D2H-copies the planes unchanged, and a match on a RAM
checkpoint is served by ``promote`` (async H2D of one RAM slot into a GPU
slot), which the regular ``load_to_active`` dequantization then consumes.
``host_slots`` capacity is headroom too; the ACTIVE bf16 state pool is never
counted against it.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.runtime_context import get_exec

_FUSED_CKPT_KERNELS: Optional[tuple] = False  # None: probed, unavailable


def _fused_ckpt_kernels():
    """The fused Triton quant/dequant pair, or None to use the eager path.
    Resolved once (import is lazy and cached); unsupported shapes/devices
    fall back per call via the kernels' bool return."""
    global _FUSED_CKPT_KERNELS
    if _FUSED_CKPT_KERNELS is False:
        if not torch.cuda.is_available():
            _FUSED_CKPT_KERNELS = None
        else:
            try:
                from sglang.kernels.selector import get_kernel

                _FUSED_CKPT_KERNELS = (
                    get_kernel("quantization.int8_mamba_ckpt_quant"),
                    get_kernel("quantization.int8_mamba_ckpt_dequant"),
                )
            except Exception:
                logger.warning(
                    "fused int8 checkpoint kernels unavailable; using the "
                    "eager quantize path",
                    exc_info=True,
                )
                _FUSED_CKPT_KERNELS = None
    return _FUSED_CKPT_KERNELS


logger = logging.getLogger(__name__)

# Private staging rows a RAM spill cycles through (see MambaCheckpointPool.
# __init__). 4 rows overlap up to 3 queued D2Hs with decode compute at a cost
# of ~3 extra int8 rows (~15 MB on Qwen3.8-Flash-Next).
_STAGING_RING_DEPTH = 4

# Pending side-state spills cycle through this many sets of pinned scratches
# (see MambaCheckpointPool._spill_sides_to_host).
_SIDES_RING_DEPTH = 4


class Int8CheckpointStore:
    """int8 store for cached multi-layer linear-attn states.

    Tensors (slot index handed out by the caller's allocator):
        qdata : [L, num_slots, H, d_v, d_k]  int8        (the quantized state)
        scale : [L, num_slots, H, 1,   d_k]  scale_dtype  (per layer,slot,head,k-chan)

    A "state" spans all L mamba layers for one cached point (matching how the
    radix caches one full state per node). The reduction axis for the scale is
    d_v (dim=-2), so each (head, k-channel) gets its own scale — aligned with the
    per-k-channel decay diag(alpha).

    ``scale_dtype`` should match the source state's dtype (bf16 / fp16 / fp32) so
    that quantize and dequantize use the identical scale — it is NOT required to
    be bf16.

    ``buffer_factory(shape, dtype)``, when given, allocates the planes (the RAM
    tier passes a pinned-memory allocator; nothing else should need it).

    ``row_major`` (RAM tier): store each slot's full state CONTIGUOUS on the
    host ([slots, L, H, d_v, d_k] base) so a tier crossing is one contiguous
    cudaMemcpy per plane. Scattered per-layer slices cost ~4x the link
    bandwidth (measured) for the same bytes. The public ``qdata``/``scale``
    stay the logical [L, slots, ...] shape via permuted views, so every
    stride-walking reader is unchanged; the row-major bases live on
    ``qdata_base``/``scale_base``.
    """

    QMAX = 127

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        device: str,
        scale_dtype: torch.dtype = torch.bfloat16,
        buffer_factory=None,
        row_major: bool = False,
    ):
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.H = num_heads
        self.d_v = head_v_dim
        self.d_k = head_k_dim
        self.device = device
        alloc = buffer_factory or (
            lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
        )
        if row_major:
            self.qdata_base = alloc(
                (num_slots, num_layers, num_heads, head_v_dim, head_k_dim), torch.int8
            )
            self.scale_base = alloc(
                (num_slots, num_layers, num_heads, 1, head_k_dim), scale_dtype
            )
            self.qdata = self.qdata_base.permute(1, 0, 2, 3, 4)
            self.scale = self.scale_base.permute(1, 0, 2, 3, 4)
        else:
            self.qdata_base = self.qdata = alloc(
                (num_layers, num_slots, num_heads, head_v_dim, head_k_dim), torch.int8
            )
            self.scale_base = self.scale = alloc(
                (num_layers, num_slots, num_heads, 1, head_k_dim), scale_dtype
            )

    # ---- (de)quant math (also usable standalone for probes/tests) ----

    @classmethod
    def quantize(cls, state: torch.Tensor):
        """state [..., H, d_v, d_k] -> (qint8, scale[..., H, 1, d_k]).

        amax / scale / round are computed in float32 so quantizing a low-precision
        state doesn't lose precision in the intermediate (symmetric with
        ``dequantize``, which is already float32). The scale is rounded to the
        state dtype (its storage precision) BEFORE the division, so quantize and
        dequantize use the identical scale."""
        state_fp32 = state.to(torch.float32)
        amax = state_fp32.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
        scale = (amax / cls.QMAX).to(state.dtype)
        q = (
            torch.round(state_fp32 / scale.to(torch.float32))
            .clamp(-cls.QMAX, cls.QMAX)
            .to(torch.int8)
        )
        return q, scale

    @staticmethod
    def dequantize(q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype):
        return (q.to(torch.float32) * scale.to(torch.float32)).to(out_dtype)

    # ---- store / load (caller supplies slot indices) ----

    def store(self, slots: torch.Tensor, state: torch.Tensor) -> None:
        """Quantize and write states. state: [L, N, H, d_v, d_k] for the N slots
        (or [L, H, d_v, d_k] when slots is a scalar/len-1)."""
        if state.dim() == 4:
            state = state.unsqueeze(1)
        q, scale = self.quantize(state)
        self.qdata[:, slots] = q
        self.scale[:, slots] = scale.to(self.scale.dtype)

    def load(self, slots: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """Dequantize states at slots -> [L, N, H, d_v, d_k] in out_dtype."""
        return self.dequantize(self.qdata[:, slots], self.scale[:, slots], out_dtype)

    def copy_to_pool(
        self,
        dst_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        """Dequantize checkpoints at ``src_slots`` directly into the active pool
        tensor ``dst_temporal`` [L, pool_slots, H, d_v, d_k] at ``dst_slots`` (the
        copy-on-write on a cache hit). Output dtype follows ``dst_temporal``."""
        kernels = _fused_ckpt_kernels()
        if kernels is not None and kernels[1](
            self.qdata, src_slots, self.scale, dst_temporal, dst_slots
        ):
            return
        dst_temporal[:, dst_slots] = self.load(src_slots, dst_temporal.dtype)

    def store_from_pool(
        self,
        src_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        """Quantize states from an active pool tensor into checkpoint slots (cache
        store / donate)."""
        kernels = _fused_ckpt_kernels()
        if kernels is not None and kernels[0](
            src_temporal, src_slots, self.qdata, self.scale, dst_slots
        ):
            return
        self.store(dst_slots, src_temporal[:, src_slots])

    def mem_usage_bytes(self) -> int:
        return (
            self.qdata.numel() * self.qdata.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    def bytes_per_slot(self) -> int:
        return self.mem_usage_bytes() // max(1, self.num_slots)


class MambaCheckpointPool:
    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_shapes: List[tuple],
        conv_dtype: torch.dtype,
        device: str,
        temporal_dtype: Optional[torch.dtype] = None,
        side_states: Optional[List[tuple]] = None,
        host_slots: int = 0,
    ):
        self.num_slots = num_slots
        self.host_slots = host_slots
        self.device = device
        # A small ring of private staging rows for RAM stores (used only with
        # a host tier): counted in neither tier's free list, so a logical id
        # never maps to them. A single row serialized every spill (the next
        # store's quantize had to wait the previous D2H on the decode
        # stream); a ring of K overlaps K-1 D2Hs with compute.
        self.staging_row = num_slots + 1
        self.num_staging = _STAGING_RING_DEPTH if host_slots > 0 else 0
        gpu_rows = num_slots + 1 + self.num_staging
        self.temporal = Int8CheckpointStore(
            num_layers=num_layers,
            # the store's slot dim IS the row count: slot 0 + one per slot (+ staging)
            num_slots=gpu_rows,
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            device=device,
            # store the scale in the temporal state's own dtype so quantize and
            # dequantize use the identical scale (not hard-coded to bf16)
            scale_dtype=(
                temporal_dtype if temporal_dtype is not None else torch.bfloat16
            ),
        )
        # conv windows stay at their native dtype (small); one buffer per conv
        # tensor in the State
        self.conv = [
            torch.empty(
                (num_layers, gpu_rows) + tuple(shape),
                dtype=conv_dtype,
                device=device,
            )
            for shape in conv_shapes
        ]
        # PLE side states (short-conv window rows, n-gram context rows) are
        # slot-indexed in the active pool; donating frees that slot, so mirror
        # their rows here, slot-for-slot, verbatim (already fp8/int/whatever
        # the side pool stores them in -- no requantization).
        # side_states entries: (active_tensor, slot_dim); mirrors are built
        # with this pool's slot count on the same device.
        self.sides = []
        for src, slot_dim in side_states or []:
            if src is None:
                continue
            shape = list(src.shape)
            shape[slot_dim] = gpu_rows
            mirror = torch.zeros(shape, dtype=src.dtype, device=src.device)
            self.sides.append((mirror, src, slot_dim))
        self.allocator = MambaSlotAllocator(
            size=num_slots, device=device, host_slots=host_slots
        )
        # RAM tier: the identical planes in pinned host memory, one row per RAM
        # slot plus a dummy row 0 (RAM logical id r maps to host row r-num_slots).
        self.host_temporal = None
        self.host_conv = []
        self.host_sides = []
        # Pool-owned copy stream for the tier crossings; H2D events recorded
        # here are waited on by load_to_active before the dequantize.
        self._copy_stream = None
        self._sides_stream = None
        self._staging_idx = None
        self._sides_dev_scratch: list = []
        self._sides_promote_evt: list = []
        self._sides_spill_scratch: list = []
        self._sides_spill_evt: list = []
        self._sides_spill_next = 0
        self._pack_q = self._pack_s = None
        self._pack_c: list = []
        self._host_conv_base: list = []
        # One busy event per staging ring row: a spill must wait only for the
        # copy that last read ITS ring row, not for the whole chain.
        self._staging_busy_events: list = [None] * self.num_staging
        # Second guard per ring row: the SIDES stream also reads it (async),
        # so reuse must wait that stream's event too, not just the copy
        # stream's plane D2H.
        self._staging_busy_sides: list = [None] * self.num_staging
        self._staging_next = 0
        # GPU slots owned by RAM->GPU promotions: (slot, H2D event). A slot
        # stays owned until load_to_active consumes it (or a drop site calls
        # release_promotion); an event retiring does NOT free it, because the
        # consumer may still reference it as its COW source.
        self._pending_promotions: list[tuple[int, object]] = []
        # Host tier buffers registered (pinned) by _alloc_host; unregistered
        # when the pool dies, or the caching CPU allocator handing these
        # addresses back to a later pool trips cudaHostRegister (rc=712).
        self._registered_host_buffers: list[torch.Tensor] = []
        if host_slots > 0:
            self._build_host_tier(
                num_layers=num_layers,
                num_heads=num_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
                scale_dtype=(
                    temporal_dtype if temporal_dtype is not None else torch.bfloat16
                ),
            )

    def _alloc_host(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        if not torch.cuda.is_available():
            # CPU/mock pools (unit tests): pinning needs a CUDA context.
            return torch.empty(shape, dtype=dtype, device="cpu")
        from sglang.srt.mem_cache.pool_host.common import (
            HostTensorAllocator,
            alloc_with_host_register,
        )

        buf = alloc_with_host_register(
            dims=tuple(shape),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
            allocator=HostTensorAllocator(),
        )
        # Tracked so the finalizer can unregister: the caching CPU allocator
        # hands a freed-but-registered address range to a later pool, whose
        # cudaHostRegister then fails with rc=712 (HostMemoryAlreadyRegistered).
        self._registered_host_buffers.append(buf)
        return buf

    def _build_host_tier(
        self, *, num_layers, num_heads, head_v_dim, head_k_dim, scale_dtype
    ):
        rows = self.host_slots + 1
        self.host_temporal = Int8CheckpointStore(
            num_layers=num_layers,
            num_slots=rows,
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            device="cpu",
            scale_dtype=scale_dtype,
            buffer_factory=self._alloc_host,
            row_major=True,
        )
        # Host rows are per RAM slot (+ dummy row 0), not per GPU slot, and
        # stored ROW-major (see Int8CheckpointStore.row_major): base [rows, L,
        # ...] with the logical [L, rows, ...] shape as a permuted view.
        self._host_conv_base = [
            self._alloc_host((rows, c.shape[0]) + tuple(c.shape[2:]), c.dtype)
            for c in self.conv
        ]
        self.host_conv = [
            b.permute(1, 0, *range(2, b.dim())) for b in self._host_conv_base
        ]
        self.host_sides = []
        for m, _, dim in self.sides:
            shape = list(m.shape)
            shape[dim] = rows
            self.host_sides.append((self._alloc_host(tuple(shape), m.dtype), dim))
        self._sides_stream = None
        self._sides_host_scratch = [None] * len(self.sides)
        self._sides_dev_scratch = [None] * len(self.sides)
        self._sides_promote_evt = [None] * len(self.sides)
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            self._copy_stream = torch.cuda.Stream()
            # Side-state spills run on their own stream: their 150KB D2Hs
            # must not queue behind the multi-MB plane D2Hs, and publishing
            # the host rows must never make a spill commit wait on queued
            # compute (a host sync would drain the decode stream; see
            # _spill_sides_to_host). Spill scratches live in a ring (one set
            # per in-flight spill) and the promote pin is a separate buffer,
            # so no pinned buffer is host-read while another stream writes it.
            self._sides_stream = torch.cuda.Stream()
            self._sides_spill_scratch = [[None] * _SIDES_RING_DEPTH for _ in self.sides]
            self._sides_spill_evt = [None] * _SIDES_RING_DEPTH
            self._sides_spill_next = 0
            self._sides_host_scratch = [None] * len(self.sides)
            # Pre-baked index tensors: torch.tensor([...], device="cuda")
            # does a pageable H2D that SYNCS the current stream -- in the
            # spill path that sync waited on the queued D2Hs.
            self._staging_idx = torch.arange(
                self.staging_row,
                self.staging_row + self.num_staging,
                dtype=torch.int64,
                device=self.temporal.qdata.device,
            )
            # One contiguous device row per plane: tier crossings pack into /
            # scatter from these so the PCIe leg is a single cudaMemcpy per
            # plane instead of L strided slices (~4x the bytes on the wire
            # otherwise, measured). Only ever touched on _copy_stream, which
            # serializes every crossing, so one set is safe.
            dev = self.temporal.qdata.device
            self._pack_q = torch.empty(
                (num_layers, num_heads, head_v_dim, head_k_dim),
                dtype=torch.int8,
                device=dev,
            )
            self._pack_s = torch.empty(
                (num_layers, num_heads, 1, head_k_dim),
                dtype=self.temporal.scale.dtype,
                device=dev,
            )
            self._pack_c = [
                torch.empty(c.shape[:1] + tuple(c.shape[2:]), dtype=c.dtype, device=dev)
                for c in self.conv
            ]

    # ---- tier routing ----

    def is_host_slot(self, slot_id: int) -> bool:
        return self.host_slots > 0 and slot_id > self.num_slots

    # RAM logical id r indexes host row r - num_slots (host row 0 is a dummy).
    def host_row(self, logical_id: int) -> int:
        return logical_id - self.num_slots

    def ckpt_slot_bytes(self) -> int:
        """Bytes of one checkpoint (all planes), for sizing the RAM tier."""
        rows = self.temporal.qdata.shape[1]
        return self._gpu_bytes() // rows

    def _gpu_bytes(self) -> int:
        conv_bytes = sum(c.numel() * c.element_size() for c in self.conv)
        side_bytes = sum(m.numel() * m.element_size() for m, _, _ in self.sides)
        return self.temporal.mem_usage_bytes() + conv_bytes + side_bytes

    # ---- lifecycle (delegates to the embedded allocator) ----

    def alloc(self, n: int = 1):
        return self.allocator.alloc(n)

    def free(self, slots: torch.Tensor):
        self.allocator.free(slots)

    def available_size(self) -> int:
        return self.allocator.available_size()

    def device_available_size(self) -> int:
        return len(self.allocator.free_slots)

    def host_available_size(self) -> int:
        return len(self.allocator.free_host_slots)

    def num_pending_promotions(self) -> int:
        """GPU slots owned by RAM->GPU promotions not yet consumed by
        load_to_active (the pool's uncached occupancy)."""
        return len(self._pending_promotions)

    def release_promotion(self, slot_id: int) -> None:
        """Give back a promoted GPU slot the requester abandoned before its
        load (no-op for ids the pool does not own)."""
        self._release_promoted_slots([slot_id])

    def _release_promoted_slots(self, slot_ids) -> None:
        if not self._pending_promotions:
            return
        wanted = set(slot_ids)
        still_pending = []
        for slot, event in self._pending_promotions:
            if slot in wanted:
                if event is not None:
                    # The promotion's H2D on the copy stream may still be in
                    # flight. Order the consumer (current) stream after it
                    # BEFORE the row re-enters the sync-free free list: the
                    # next alloc hands this row to a quantize/H2D that must
                    # not race the still-running copy (same reason clear()
                    # drains the copy stream).
                    torch.cuda.current_stream().wait_event(event)
                self.allocator.free(
                    torch.tensor(
                        [slot], dtype=torch.int64, device=self.allocator.device
                    )
                )
            else:
                still_pending.append((slot, event))
        self._pending_promotions = still_pending

    def clear(self) -> None:
        """Release every checkpoint slot of both tiers (radix flush/reset) and
        drop pending-transfer bookkeeping. The int8 qdata is left as-is; slots
        are reused/overwritten on the next store."""
        if self._copy_stream is not None:
            # In-flight D2H/H2D may still read/write the staging and promoted
            # rows; drain the copy stream before those rows re-enter the free
            # lists. Flush runs at request idle, so this wait is cheap.
            self._copy_stream.synchronize()
        self._publish_sides(blocking=True)
        if self._sides_spill_evt:
            self._sides_spill_evt = [None] * _SIDES_RING_DEPTH
            self._sides_spill_next = 0
        self._pending_promotions = []
        self._staging_busy_events = [None] * self.num_staging
        self._staging_busy_sides = [None] * self.num_staging
        self._staging_next = 0
        self.allocator.clear()

    def _release_host_registrations(self) -> None:
        """Undo every cudaHostRegister done by _alloc_host. The host tier
        mirrors are pinned so D2H/H2D stays async; leaving them registered
        past teardown lets the caching CPU allocator recycle the addresses
        into a later pool's registration (rc=712)."""
        buffers, self._registered_host_buffers = self._registered_host_buffers, []
        if not buffers:
            return
        try:
            # A D2H/H2D still queued on the copy stream may be mid-DMA over
            # these pages; unregistering then is the stale-data hazard the
            # registration warns about. The refcycle means __del__ fires at
            # an arbitrary gc point, so drain first.
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

            for buf in buffers:
                _cuda_host_unregister(buf)
        except Exception:
            logger.warning(
                "int8 checkpoint pool: host unregister failed", exc_info=True
            )

    def __del__(self):
        # The store's buffer_factory closure keeps pool->store->pool in a
        # reference cycle, so this runs from gc, not refcounting.
        try:
            if torch.cuda.is_initialized():
                self._release_host_registrations()
        except Exception:
            pass

    # ---- state transfer between the active MambaPool and this store ----

    def store_from_active(
        self,
        active_mamba_pool,
        active_slots,
        ckpt_slots,
        ckpt_id=None,
        assume_device=False,
    ) -> None:
        """Quantize temporal + copy conv from the active pool into checkpoint slots
        (the radix donate / cache-store). A RAM-tier destination quantizes into the
        GPU staging row and D2H-copies the int8 planes unchanged; the RAM slot is
        safe to publish because every later reader chains on the pool copy
        stream. ``ckpt_id`` lets a caller that already read the slot id skip the
        D2H sync; ``assume_device`` asserts a GPU-tier destination (the pool is
        device-first, so an alloc while the GPU tier has free rows qualifies),
        keeping every GPU-slot store — and the GPU-only config — free of the
        per-store .item() pipeline sync."""
        assert ckpt_slots.numel() == 1, "one checkpoint per store"
        if self.host_slots > 0 and not assume_device:
            if ckpt_id is None:
                ckpt_id = int(ckpt_slots.reshape(-1)[0].item())
            if ckpt_id > self.num_slots:
                self._store_to_host(active_mamba_pool, active_slots, ckpt_id)
                return
        self._quantize_to_rows(active_mamba_pool, active_slots, ckpt_slots)

    def _quantize_to_rows(self, active_mamba_pool, active_slots, ckpt_slots) -> None:
        cache = active_mamba_pool.mamba_cache
        self.temporal.store_from_pool(cache.temporal, active_slots, ckpt_slots)
        for i, c in enumerate(self.conv):
            c[:, ckpt_slots] = cache.conv[i][:, active_slots]
        for mirror, src, dim in self.sides:
            mirror.index_copy_(dim, ckpt_slots, src.index_select(dim, active_slots))

    def _store_to_host(self, active_mamba_pool, active_slots, host_logical) -> None:
        # Round-robin the staging ring: this spill waits only the D2H that
        # last read THIS ring row (K spills ago), so up to K-1 earlier D2Hs
        # stay in flight while decode keeps running.
        ring_i = self._staging_next
        self._staging_next = (ring_i + 1) % self.num_staging
        row = self.staging_row + ring_i
        # View into the pre-baked device index: building torch.tensor([row],
        # device="cuda") here would pageable-H2D and SYNC the decode stream.
        staging = (
            self._staging_idx[ring_i : ring_i + 1]
            if self._staging_idx is not None
            else torch.tensor([row], dtype=torch.int64)
        )
        for busy in (
            self._staging_busy_events[ring_i],
            self._staging_busy_sides[ring_i],
        ):
            if busy is not None:
                # That D2H still reads this ring row; do not overwrite it
                # before the copy retires (stream-ordered, no host sync).
                torch.cuda.current_stream().wait_event(busy)
        # Quantize into the ring row (a GPU row addressed by no logical id, so
        # this cannot recurse back into _store_to_host).
        self._quantize_to_rows(active_mamba_pool, active_slots, staging)
        if self._copy_stream is None:
            # CPU/mock pools: plain copies, already complete.
            self._copy_staging_to_host(self.host_row(host_logical), row, staging, None)
            return
        # Capture the producing (compute) stream OUTSIDE the context below:
        # inside it, current_stream() is the copy stream and the wait would
        # self-order (no-op), letting the D2H race the quantize kernels.
        producer = torch.cuda.current_stream()
        event = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(producer)
            sides_ev = self._copy_staging_to_host(
                self.host_row(host_logical), row, staging, producer
            )
            # Guards this ring row against the next spill that reuses it
            # (the D2H above still reads it until the copy retires).
            event.record(self._copy_stream)
        self._staging_busy_events[ring_i] = event
        self._staging_busy_sides[ring_i] = sides_ev
        # The D2H (and any sides wait) is enqueued; nothing above read host
        # memory, so the decode stream stayed unblocked throughout.

    def _copy_staging_to_host(self, h_row: int, row: int, staging, producer) -> None:
        st = self.temporal
        ht = self.host_temporal
        if self._pack_q is not None:
            # Gather the GPU row into a contiguous pack, then ONE pinned D2H
            # per plane (host row-major bases). Strided slices would pay ~4x
            # the link time for the same bytes. Runs on the caller's
            # (_copy_stream) context, where every crossing is serialized.
            self._pack_q.copy_(st.qdata[:, row])
            ht.qdata_base[h_row].copy_(self._pack_q, non_blocking=True)
            self._pack_s.copy_(st.scale[:, row])
            ht.scale_base[h_row].copy_(self._pack_s, non_blocking=True)
            for pack_c, hc_base, c in zip(
                self._pack_c, self._host_conv_base, self.conv
            ):
                pack_c.copy_(c[:, row])
                hc_base[h_row].copy_(pack_c, non_blocking=True)
            return self._spill_sides_to_host(h_row, staging, producer)
        else:
            # CPU/mock pools: logical views, plain complete copies.
            ht.qdata[:, h_row].copy_(st.qdata[:, row], non_blocking=True)
            ht.scale[:, h_row].copy_(st.scale[:, row], non_blocking=True)
            for hc, c in zip(self.host_conv, self.conv):
                hc[:, h_row].copy_(c[:, row], non_blocking=True)
            host_idx = torch.tensor([h_row], dtype=torch.int64)
            for (hm, dim), (m, _, _) in zip(self.host_sides, self.sides):
                rows = m.index_select(dim, staging.to(m.device))
                hm.index_copy_(dim, host_idx, rows.to(hm.device))

    def _spill_sides_to_host(self, h_row: int, staging, producer) -> None:
        # Side rows are tiny (~150KB); their D2Hs run on the SIDES stream into
        # a ring of pinned scratches, and the host-row publish is DEFERRED to
        # _publish_sides. A host sync here would wait on the sides stream's
        # wait_event(producer) -- i.e. drain every decode kernel already
        # queued on the compute stream -- which is the stall this whole path
        # exists to avoid. So a spill commit only enqueues; completed scratch
        # contents land in the host mirrors when their events retire.
        if not self.host_sides:
            return None
        self._publish_sides()
        k = self._sides_spill_next
        if self._sides_spill_evt[k] is not None:
            # The ring is full: slot k holds the OLDEST pending spill and its
            # scratch is about to be overwritten. Flush now, while the
            # _publish_sides sweep still starts at k (oldest first). Advancing
            # the cursor first would start the sweep at k + 1 and visit slot k
            # LAST -- and a RAM row re-spilled since (its newer side spill in
            # a later slot) would then be republished under the older spill's
            # state, so a later promote would restore stale side state.
            self._publish_sides(blocking=True)
        self._sides_spill_next = (k + 1) % _SIDES_RING_DEPTH
        ev = torch.cuda.Event()
        self._sides_stream.wait_stream(producer)
        with torch.cuda.stream(self._sides_stream):
            for j, ((hm, dim), (m, _, _)) in enumerate(
                zip(self.host_sides, self.sides)
            ):
                rows = m.index_select(dim, staging.to(m.device))
                scratch = self._sides_spill_scratch[j][k]
                if scratch is None or scratch.shape != rows.shape:
                    scratch = self._sides_spill_scratch[j][k] = torch.empty_like(
                        rows, device="cpu", pin_memory=True
                    )
                scratch.copy_(rows, non_blocking=True)
            ev.record(self._sides_stream)
        self._sides_spill_evt[k] = (ev, h_row)
        # Also returned to _store_to_host: the event covers this spill's READ
        # of the staging ring row, which guards the row's next quantize.
        return ev

    def _publish_sides(self, blocking: bool = False, upto_h_row=None) -> None:
        """Copy finished side-row spills from their pinned scratches into the
        host mirrors (oldest pending first). Non-blocking skips spills whose
        D2H event has not retired. A reader of host_sides (promote) passes
        blocking=True and its h_row: only spills OF THAT ROW are awaited --
        pending spills of other rows would otherwise add an unrelated spill's
        whole backlog to the promote's host latency. Rows are independent,
        so their publish order does not matter."""
        if not self._sides_spill_evt:
            return
        for off in range(_SIDES_RING_DEPTH):
            # Ring order starting at _sides_spill_next is chronological
            # (oldest pending first), so a host row is never published stale
            # over a newer spill of the same row.
            k = (self._sides_spill_next + off) % _SIDES_RING_DEPTH
            entry = self._sides_spill_evt[k]
            if entry is None:
                continue
            ev, h_row = entry
            mine = upto_h_row is None or h_row == upto_h_row
            if not ev.query() and not (blocking and mine):
                continue
            ev.synchronize()
            host_idx = torch.tensor([h_row], dtype=torch.int64)
            for j, ((hm, dim), _) in enumerate(zip(self.host_sides, self.sides)):
                hm.index_copy_(dim, host_idx, self._sides_spill_scratch[j][k])
            self._sides_spill_evt[k] = None

    def alloc_device_slot(self) -> Optional[torch.Tensor]:
        """One GPU-tier checkpoint id, or None: promotions must land on the
        GPU tier, so they never fall through to the RAM tier."""
        if self.allocator.free_slots.numel() == 0:
            return None
        slot = self.allocator.free_slots[-1:]
        self.allocator.free_slots = self.allocator.free_slots[:-1]
        return slot

    def promote(self, host_logical: int, device_logical: int) -> None:
        """Copy one RAM checkpoint into a GPU checkpoint slot (async H2D on the
        pool copy stream). The caller owns ``device_logical`` (taken from the
        GPU tier); ``load_to_active`` waits the recorded event before
        dequantizing it and then releases the slot. Until that consumption the
        pool owns the slot: it is NOT on any free list, so neither a later
        promote nor a checkpoint commit can overwrite a row a request still
        references as its COW source."""
        assert self.is_host_slot(host_logical), (
            f"promote() wants a RAM checkpoint id, got {host_logical}"
        )
        assert not self.is_host_slot(device_logical), (
            f"promote() destination must be a GPU checkpoint id, got {device_logical}"
        )
        h_row = self.host_row(host_logical)
        # The host mirror row must reflect every spill of THIS row before it
        # is read out (other rows' pending spills are left alone).
        self._publish_sides(blocking=True, upto_h_row=h_row)
        if self._copy_stream is None:
            self._copy_host_to_device(h_row, device_logical)
            self._pending_promotions.append((device_logical, None))
            return
        # Order the H2D after every compute-stream read of the destination row
        # enqueued so far; capture outside the context (inside it the current
        # stream is the copy stream itself).
        compute = torch.cuda.current_stream()
        event = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(compute)
            self._copy_host_to_device(h_row, device_logical)
            event.record(self._copy_stream)
        self._pending_promotions.append((device_logical, event))

    def _copy_host_to_device(self, h_row: int, gpu_id: int) -> None:
        st = self.temporal
        ht = self.host_temporal
        if self._pack_q is not None:
            # Mirror of _copy_staging_to_host: ONE pinned H2D->pack per plane,
            # then a cheap device-side scatter into the logical slot row.
            self._pack_q.copy_(ht.qdata_base[h_row], non_blocking=True)
            st.qdata[:, gpu_id].copy_(self._pack_q)
            self._pack_s.copy_(ht.scale_base[h_row], non_blocking=True)
            st.scale[:, gpu_id].copy_(self._pack_s)
            for pack_c, hc_base, c in zip(
                self._pack_c, self._host_conv_base, self.conv
            ):
                pack_c.copy_(hc_base[h_row], non_blocking=True)
                c[:, gpu_id].copy_(pack_c)
        else:
            # CPU/mock pools: logical views, plain copies.
            st.qdata[:, gpu_id].copy_(ht.qdata[:, h_row], non_blocking=True)
            st.scale[:, gpu_id].copy_(ht.scale[:, h_row], non_blocking=True)
            for c, hc in zip(self.conv, self.host_conv):
                c[:, gpu_id].copy_(hc[:, h_row], non_blocking=True)
        for j, ((hm, dim), (m, _, _)) in enumerate(zip(self.host_sides, self.sides)):
            # Pinned + device scratches and a narrow() destination: a fresh
            # torch.tensor([gpu_id], device=...) or pageable rows.to(cuda)
            # would SYNC the copy stream (waiting the plane H2Ds and every
            # queued compute kernel). Dev-scratch reuse is stream-ordered on
            # the copy stream; the PINNED scratch is host-written, so its
            # overwrite waits the previous promote's event instead.
            rows = hm.index_select(dim, torch.tensor([h_row], dtype=torch.int64))
            if m.device.type != "cuda":
                # CPU/mock pools: plain same-device copy, no events.
                m.index_copy_(dim, torch.tensor([gpu_id], dtype=torch.int64), rows)
                continue
            prev = self._sides_promote_evt[j]
            if prev is not None:
                prev.synchronize()
            pin = self._sides_host_scratch[j]
            if pin is None or pin.shape != rows.shape:
                pin = self._sides_host_scratch[j] = torch.empty_like(
                    rows, device="cpu", pin_memory=True
                )
            pin.copy_(rows)
            dev = self._sides_dev_scratch[j]
            if dev is None or dev.shape != rows.shape:
                dev = self._sides_dev_scratch[j] = torch.empty_like(
                    rows, device=m.device
                )
            dev.copy_(pin, non_blocking=True)
            m.narrow(dim, gpu_id, 1).copy_(dev)
            evt = torch.cuda.Event()
            evt.record(torch.cuda.current_stream())
            self._sides_promote_evt[j] = evt

    def load_to_active(self, active_mamba_pool, ckpt_slots, active_slots) -> None:
        """Dequantize temporal + copy conv from GPU checkpoint slots into the
        active pool (the cache-hit copy-on-write). RAM checkpoints must be
        promoted first; promoted slots wait their H2D event (stream-ordered)."""
        ids = ckpt_slots.reshape(-1)
        id_list = None
        if self.host_slots or self._pending_promotions:
            # Only tiered pools can hand out a RAM id, and only a tiered pool
            # has promotions to wait/release; pay the one value read there.
            id_list = ids.tolist()
            if self.host_slots and any(i > self.num_slots for i in id_list):
                raise ValueError(
                    "load_to_active got a RAM checkpoint id; promote it to a "
                    f"GPU checkpoint first (ids={id_list}, GPU tier={self.num_slots})"
                )
        self._wait_promotions(id_list)
        cache = active_mamba_pool.mamba_cache
        self.temporal.copy_to_pool(cache.temporal, ckpt_slots, active_slots)
        for i, c in enumerate(self.conv):
            cache.conv[i][:, active_slots] = c[:, ckpt_slots].to(cache.conv[i].dtype)
        for mirror, src, dim in self.sides:
            src.index_copy_(dim, active_slots, mirror.index_select(dim, ckpt_slots))
        if id_list is not None:
            # The reads above are now enqueued on this stream: return any
            # promoted source slots to the allocator. Future writers order
            # after these reads (same stream, or the copy stream's captured
            # wait at the next promote/store).
            self._release_promoted_slots(id_list)

    def _wait_promotions(self, id_list: Optional[list]) -> None:
        """Order the forward stream after in-flight H2Ds of the requested slots."""
        if not self._pending_promotions or self._copy_stream is None or not id_list:
            return
        wanted = set(id_list)
        stream = torch.cuda.current_stream()
        for slot, event in self._pending_promotions:
            if slot in wanted and event is not None:
                stream.wait_event(event)

    @staticmethod
    def estimate_mem_usage_bytes(
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_shapes: List[tuple],
        conv_dtype: torch.dtype,
        temporal_dtype: torch.dtype,
        side_states: Optional[List[tuple]] = None,
        staging_row: bool = False,
    ) -> dict:
        """Estimate the pool's HBM footprint (bytes) WITHOUT allocating, so a
        caller can check it against free memory before construction. Mirrors the
        real layout: int8 qdata + per-(head,k) scale + bf16 conv windows +
        mirrored side-state rows, including the reserved slot 0 (plus the RAM
        tier's private staging ring when ``staging_row``)."""
        # Slot 0 is reserved (matches MambaSlotAllocator); the staging rows are
        # the RAM tier's private quantize ring.
        slots = num_slots + 1 + (_STAGING_RING_DEPTH if staging_row else 0)
        scale_isz = torch.empty((), dtype=temporal_dtype).element_size()
        conv_isz = torch.empty((), dtype=conv_dtype).element_size()
        qdata = num_layers * slots * num_heads * head_v_dim * head_k_dim  # int8 = 1B
        scale = num_layers * slots * num_heads * head_k_dim * scale_isz
        conv = 0
        for shape in conv_shapes:
            n = 1
            for s in shape:
                n *= int(s)
            conv += num_layers * slots * n * conv_isz
        # mirror rows match the source tensors' dtype verbatim (no requantization)
        sides = 0
        for src, slot_dim in side_states or []:
            if src is None:
                continue
            shape = list(src.shape)
            shape[slot_dim] = slots
            n = 1
            for s in shape:
                n *= int(s)
            sides += n * src.element_size()
        # The RAM tier's contiguous-pack buffers (one logical row of each
        # plane) are allocated alongside it on CUDA; the pre-flight guard
        # must charge for them too.
        pack = (qdata + scale + conv) // slots if staging_row else 0
        return {
            "qdata": qdata,
            "scale": scale,
            "conv": conv,
            "sides": sides,
            "pack": pack,
            "total": qdata + scale + conv + sides + pack,
            "per_row": qdata // slots + scale // slots + conv // slots + sides // slots,
        }

    def mem_usage_bytes(self) -> int:
        return self._gpu_bytes()


def maybe_init_int8_mamba_checkpoint_pool(
    *,
    mamba_size: int,
    cache_params,
    mamba_layer_ids: List[int],
    device: str,
    side_states: Optional[List[Optional[tuple]]] = None,
) -> Optional[MambaCheckpointPool]:
    """Build the optional int8 ``MambaCheckpointPool`` when
    ``--enable-int8-mamba-checkpoint`` is set (and a global server-args context
    exists), else return ``None``. The radix caches states here (int8) instead of
    in the active bf16 pool -> ~2x cached-prefix capacity at fixed memory.

    Estimates the pool's HBM footprint and checks it against free memory BEFORE
    allocating, so an oversized ``--int8-mamba-ckpt-size`` fails with an actionable
    message instead of a cryptic mid-allocation CUDA OOM.
    """

    try:
        mamba = get_exec().mamba
    except ValueError:
        # Some unit-test / mock runners construct HybridReqToTokenPool directly
        # without a global server-args context. The int8 checkpoint pool is opt-in
        # via a CLI flag, so an unset context unambiguously means it is off.
        mamba = None
    if mamba is None or not mamba.enable_int8_mamba_checkpoint:
        return None

    GB = 1 << 30
    H, d_v, d_k = cache_params.shape.temporal
    ckpt_size = mamba.int8_mamba_ckpt_size or (2 * mamba_size)
    offload = bool(mamba.enable_int8_mamba_checkpoint_offload)
    kwargs = dict(
        num_layers=len(mamba_layer_ids),
        num_slots=ckpt_size,
        num_heads=H,
        head_v_dim=d_v,
        head_k_dim=d_k,
        conv_shapes=list(cache_params.shape.conv),
        conv_dtype=cache_params.dtype.conv,
        temporal_dtype=cache_params.dtype.temporal,
        side_states=[t for t in side_states or [] if t is not None],
    )

    est = MambaCheckpointPool.estimate_mem_usage_bytes(**kwargs, staging_row=offload)
    host_slots = 0
    host_bytes = 0
    if offload:
        raw_host = mamba.int8_mamba_ckpt_host_size
        if not raw_host or int(raw_host) <= 0:
            raise RuntimeError(
                "--int8-mamba-ckpt-host-size must be a positive number of RAM "
                f"tier slots (got {raw_host})."
            )
        host_slots = int(raw_host)
        host_bytes = host_slots * est["per_row"]
    free_bytes = None
    if isinstance(device, str) and device.startswith("cuda"):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except Exception:
            free_bytes = None
    logger.info(
        f"int8 mamba checkpoint pool: {ckpt_size} slots, "
        f"{est['total'] / GB:.2f}GB (qdata {est['qdata'] / GB:.2f} + scale "
        f"{est['scale'] / GB:.2f} + conv {est['conv'] / GB:.2f}"
        + (f" + sides {est['sides'] / GB:.2f}" if est["sides"] else "")
        + f"); active mamba "
        f"pool {mamba_size} slots"
        + (f"; free HBM {free_bytes / GB:.2f}GB" if free_bytes is not None else "")
        + (
            f"; RAM tier {host_slots} slots ({host_bytes / GB:.2f}GB pinned)"
            if offload
            else ""
        )
    )
    if free_bytes is not None and est["total"] >= free_bytes:
        raise RuntimeError(
            f"int8 mamba checkpoint pool needs ~{est['total'] / GB:.2f}GB but only "
            f"{free_bytes / GB:.2f}GB HBM is free. Lower --int8-mamba-ckpt-size "
            f"(currently {ckpt_size}) or --mem-fraction-static."
        )
    if offload:
        from sglang.srt.mem_cache.host_memory import host_memory_headroom_bytes
        from sglang.srt.mem_cache.pool_host.base import ranks_per_host

        # Cgroup-aware allocatable RAM, divided by the ranks sharing this
        # host: every rank sizes its own tier from the same machine budget.
        free_host = host_memory_headroom_bytes() // max(1, ranks_per_host())
        if host_bytes >= free_host:
            raise RuntimeError(
                f"int8 mamba checkpoint RAM tier needs ~{host_bytes / GB:.2f}GB "
                f"pinned but only {free_host / GB:.2f}GB RAM is allocatable "
                f"per rank ({ranks_per_host()} rank(s) per host). Lower "
                f"--int8-mamba-ckpt-host-size (currently "
                f"{mamba.int8_mamba_ckpt_host_size})."
            )

    pool = MambaCheckpointPool(device=device, host_slots=host_slots, **kwargs)
    # NOTE: this pool's HBM is NOT subtracted from the KV-cache budget
    # (max_total_num_tokens); it is allocated from --mem-fraction-static headroom.
    # The estimate check above guards against an oversized pool; accounting it in
    # the KV budget is a follow-up. The RAM tier is headroom of the same kind:
    # pinned at boot, never counted as active-state capacity.
    logger.warning(
        f"int8 mamba checkpoint pool ({est['total'] / GB:.2f}GB"
        + (f" + RAM tier {host_bytes / GB:.2f}GB" if offload else "")
        + f") is allocated from "
        f"--mem-fraction-static headroom and is not reflected in "
        f"max_total_num_tokens; ensure headroom covers it."
    )
    return pool
