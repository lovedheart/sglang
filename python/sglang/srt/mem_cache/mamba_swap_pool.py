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

MambaSwapPool -- pinned-RAM backing store for PARKED linear-attention states.

The active ``MambaPool`` holds one full-precision recurrent state per
resident request; decode kernels read and write those rows every step, so a
request being decoded must own an active slot. This pool lets logical
concurrency exceed the active pool: a parked request writes its exact bf16
state (conv windows, temporal, and every registered slot sibling) into one
pinned-RAM swap slot, gives its active slot back, and a later ``wake``
copies the state into a freshly allocated active slot so decoding resumes
without re-prefill.

Ownership contract: a swap slot is owned by exactly one LIVE request while
parked. Swap slots are never radix-tree values, never LRU-evictable, and a
radix-cache flush must not touch them (only ``clear()`` releases the set).

Exactness is the contract: parked states are dirty (mutated every decode
step), so the backing store is a lossless bf16 copy at native dtypes -- no
quantization path exists in this file. The int8 checkpoint pool remains the
compression tier for cold, clean cached prefixes only.

Stream model (cloned from the tiered checkpoint pool, whose stream-ordering
patterns came out of two review rounds):
  * park: the copy stream waits the caller (compute) stream, reads the
    active row into the pinned swap row, records an event; the active
    slot's free is ordered AFTER that event (the slot allocator is
    deliberately sync-free, so the wait happens here before handing the
    row back to the free list).
  * wake: the copy stream waits the caller stream, H2Ds the swap row into
    the fresh active slot, records an event; the caller's next forward is
    ordered after it. Slot siblings ride the synchronous
    ``get_cpu_slots``/``load_cpu_slots`` path the retraction backup uses.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec

logger = logging.getLogger(__name__)


class _ParkedSlot:
    """One parked state: the D2H completion event plus sibling payloads."""

    __slots__ = ("event", "sibling_data")

    def __init__(self, event: Optional[Any], sibling_data: List[Any]):
        self.event = event
        self.sibling_data = sibling_data


class MambaSwapPool:
    """Fixed-capacity pinned-RAM store of exact active states.

    Swap ids are ``[1, swap_slots]`` and double as host-plane row indices
    (row 0 is the cuda-graph pad row, mirroring the active pool).
    """

    def __init__(
        self,
        *,
        mamba_pool,
        mamba_allocator,
        swap_slots: int,
        device: str,
        allow_replayssm: bool = False,
    ):
        # ReplaySSM adds slot-indexed ring planes (and, in plain decode mode,
        # a slot-indexed write cursor) to the parked state. Capture is cheap
        # and exact; the -spec row-keyed cursors live in req rows that park
        # keeps, so they need no record. ``allow_replayssm`` is the V2-P0b
        # spike switch — production gating stays in the mamba arg hook until
        # the swap-v2 P4 gate passes.
        if (
            mamba_pool.enable_linear_replayssm
            or mamba_pool.enable_linear_replayssm_spec
        ) and not allow_replayssm:
            raise ValueError(
                "--enable-mamba-state-swap does not support the ReplaySSM "
                "ring buffers yet; disable one of them."
            )
        self.mamba_pool = mamba_pool
        self.mamba_allocator = mamba_allocator
        self.swap_slots = swap_slots
        self.device = device
        cache = mamba_pool.mamba_cache

        # Host planes mirror the device layout (slot axis in place), pinned
        # so the bulk D2H/H2D below are genuinely async.
        self.host_temporal = self._alloc_host(
            (cache.temporal.shape[0], swap_slots + 1) + tuple(cache.temporal.shape[2:]),
            cache.temporal.dtype,
        )
        self.host_conv = [
            self._alloc_host((c.shape[0], swap_slots + 1) + tuple(c.shape[2:]), c.dtype)
            for c in cache.conv
        ]
        # ReplaySSM ring planes: [num_layers, num_slots, ...], slot-indexed
        # like temporal/conv, so a parked row is exactly complete state.
        # Under a request ring (memory_pool.py:808-815: GDN -spec always,
        # KDA -spec with host-only checkpoints) the rows are request-keyed
        # and park with the request row instead — copying them by slot would
        # park and wake a DIFFERENT request's rows, so skip them there.
        self._ring_planes = (
            []
            if mamba_pool.replayssm_request_ring
            else [
                plane
                for plane in (
                    cache.replayssm_d,
                    cache.replayssm_k,
                    cache.replayssm_g,
                    cache.replayssm_rawv,
                    cache.replayssm_rawk,
                    cache.replayssm_beta,
                )
                if plane is not None
            ]
        )
        self.host_ring = [
            self._alloc_host((p.shape[0], swap_slots + 1) + tuple(p.shape[2:]), p.dtype)
            for p in self._ring_planes
        ]
        # Plain-decode write cursor is slot-indexed; the -spec cursor triple
        # is req-row-indexed and survives parking with the request row.
        self.host_write_pos = (
            self._alloc_host((swap_slots + 1,), torch.int32)
            if mamba_pool.replayssm_write_pos is not None
            else None
        )
        # Slot siblings ride the retraction copy path verbatim.
        self._siblings = list(mamba_pool._slot_siblings)

        self._copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.free_ids = list(range(1, swap_slots + 1))
        self._parked = {}

    @staticmethod
    def _alloc_host(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        if not torch.cuda.is_available():
            # CPU/mock pools (unit tests): pinning needs a CUDA context.
            return torch.zeros(shape, dtype=dtype, device="cpu")
        from sglang.srt.mem_cache.pool_host.common import (
            HostTensorAllocator,
            alloc_with_host_register,
        )

        return alloc_with_host_register(
            dims=tuple(shape),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
            allocator=HostTensorAllocator(),
        )

    # ===== Capacity =====

    def available_size(self) -> int:
        return len(self.free_ids)

    def num_parked(self) -> int:
        return len(self._parked)

    def total_bytes(self) -> int:
        planes = [self.host_temporal, *self.host_conv, *self.host_ring]
        if self.host_write_pos is not None:
            planes.append(self.host_write_pos)
        return sum(p.numel() * p.element_size() for p in planes)

    def is_parked(self, swap_id: int) -> bool:
        return swap_id in self._parked

    # ===== Park / wake =====

    def park(self, active_slot: int, *, order_streams=()) -> Optional[int]:
        """Copy the active slot's state into a pinned swap slot and return
        the swap id, freeing the active slot. Returns None (leaving the
        active slot untouched) when the swap tier is full. ``order_streams``
        are the streams that may still hold in-flight reads/writes of the
        active row (the scheduler's forward and schedule streams); the D2H
        is fenced behind them and they are fenced behind it, so the row can
        be freed to the sync-free allocator immediately."""
        if not self.free_ids:
            return None
        swap_id = self.free_ids.pop(0)
        cache = self.mamba_pool.mamba_cache
        idx = torch.tensor([active_slot], dtype=torch.int64, device=self.device)

        if self._copy_stream is None:
            self._copy_planes(cache, active_slot, swap_id, to_device=False)
            sibling_data = [
                (sibling, sibling.get_cpu_slots(idx)) for sibling in self._siblings
            ]
        else:
            # Captured before entering the copy stream: the *caller's* current
            # stream is a producer, not the copy stream itself.
            producers = self._producers(order_streams)
            with torch.cuda.stream(self._copy_stream):
                # Fence behind every stream that may still touch the row.
                for stream in producers:
                    self._copy_stream.wait_stream(stream)
                self._copy_planes(cache, active_slot, swap_id, to_device=False)
                # Sibling rows ride the retraction copy path, on the copy
                # stream so the fence above orders them too.
                sibling_data = [
                    (sibling, sibling.get_cpu_slots(idx)) for sibling in self._siblings
                ]
                event = torch.cuda.Event()
                event.record(self._copy_stream)

        self._parked[swap_id] = _ParkedSlot(
            event if self._copy_stream is not None else None, sibling_data
        )
        if self._copy_stream is not None:
            # Order future writes to the row (a new owner's prefill) after
            # the D2H that still reads it (the promotion-release lesson).
            for stream in producers:
                stream.wait_event(self._parked[swap_id].event)

        if self.mamba_allocator is not None:
            idx_free = torch.tensor(
                [active_slot], dtype=torch.int64, device=self.device
            )
            if self._copy_stream is not None:
                torch.cuda.current_stream().wait_event(self._parked[swap_id].event)
            self.mamba_allocator.free(idx_free)
        return swap_id

    def wake(self, swap_id: int, active_slot: int, *, order_streams=()) -> None:
        """Copy a parked state into a freshly allocated active slot; every
        stream in ``order_streams`` (the forward's candidates) is fenced
        after the H2D before this returns. The swap slot is recycled: a
        later park reusing the row fences behind this H2D the same way."""
        record = self._parked.pop(swap_id)
        cache = self.mamba_pool.mamba_cache
        idx = torch.tensor([active_slot], dtype=torch.int64, device=self.device)

        if self._copy_stream is None:
            self._copy_planes(cache, active_slot, swap_id, to_device=True)
            for sibling, data in record.sibling_data:
                sibling.load_cpu_slots(data, idx)
            self.free_ids.append(swap_id)
            return
        event = torch.cuda.Event()
        producers = self._producers(order_streams)
        with torch.cuda.stream(self._copy_stream):
            for stream in producers:
                self._copy_stream.wait_stream(stream)
            self._copy_planes(cache, active_slot, swap_id, to_device=True)
            for sibling, data in record.sibling_data:
                sibling.load_cpu_slots(data, idx)
            event.record(self._copy_stream)
        for stream in producers:
            stream.wait_event(event)
        self.free_ids.append(swap_id)

    def drop(self, swap_id: int) -> bool:
        """Discard a parked state (request finished/aborted without waking).
        Idempotent for ids this pool does not own."""
        if self._parked.pop(swap_id, None) is None:
            return False
        self.free_ids.append(swap_id)
        return True

    def clear(self) -> None:
        """Full reset (server shutdown): drain in-flight copies, free all."""
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
        self._parked = {}
        self.free_ids = list(range(1, self.swap_slots + 1))

    # ===== Stream helpers =====

    @staticmethod
    def _producers(extra):
        """The calling stream plus the caller-named producer/consumer streams,
        de-duplicated by underlying CUDA stream."""
        streams = [torch.cuda.current_stream()] + [s for s in extra if s is not None]
        seen = set()
        out = []
        for stream in streams:
            key = stream.cuda_stream
            if key not in seen:
                seen.add(key)
                out.append(stream)
        return out

    # ===== Copy internals =====

    def _copy_planes(
        self, cache, active_slot: int, row: int, *, to_device: bool
    ) -> None:
        if to_device:
            cache.temporal[:, active_slot].copy_(
                self.host_temporal[:, row], non_blocking=True
            )
            for dst, src in zip(cache.conv, self.host_conv):
                dst[:, active_slot].copy_(src[:, row], non_blocking=True)
            for dst, src in zip(self._ring_planes, self.host_ring):
                dst[:, active_slot].copy_(src[:, row], non_blocking=True)
            if self.host_write_pos is not None:
                self.mamba_pool.replayssm_write_pos[active_slot].copy_(
                    self.host_write_pos[row], non_blocking=True
                )
        else:
            self.host_temporal[:, row].copy_(
                cache.temporal[:, active_slot], non_blocking=True
            )
            for dst, src in zip(self.host_conv, cache.conv):
                dst[:, row].copy_(src[:, active_slot], non_blocking=True)
            for dst, src in zip(self.host_ring, self._ring_planes):
                dst[:, row].copy_(src[:, active_slot], non_blocking=True)
            if self.host_write_pos is not None:
                self.host_write_pos[row].copy_(
                    self.mamba_pool.replayssm_write_pos[active_slot],
                    non_blocking=True,
                )


def maybe_init_mamba_state_swap_pool(
    *,
    mamba_pool,
    mamba_allocator,
    device: str,
):
    """Build the swap pool from the flags, or None when disabled. The flag
    pair ``--enable-mamba-state-swap`` / ``--mamba-swap-size`` is presence-
    validated in the arg_groups mamba hook."""
    mamba = None
    try:
        mamba = get_exec().mamba
    except ValueError:
        # Some unit-test / mock runners build HybridReqToTokenPool without a
        # global server-args context. The swap pool is CLI opt-in, so an unset
        # context unambiguously means it is off.
        pass
    if mamba is None or not mamba.enable_mamba_state_swap:
        return None
    pool = MambaSwapPool(
        mamba_pool=mamba_pool,
        mamba_allocator=mamba_allocator,
        swap_slots=mamba.mamba_swap_size,
        device=device,
        allow_replayssm=envs.SGLANG_TEST_MAMBA_SWAP_ALLOW_REPLAYSSM.get(),
    )
    logger.info(
        "mamba swap pool: %d parked slots (%.2fGB pinned)",
        mamba.mamba_swap_size,
        pool.total_bytes() / (1 << 30),
    )
    return pool
