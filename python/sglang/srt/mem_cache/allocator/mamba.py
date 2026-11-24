"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Slot allocator for the Mamba state pool.

Mamba caches one whole state tensor per request, so the allocator hands out
fixed-size slots (1 per request) rather than paged token KV indices.  The
underlying tensor storage lives in ``MambaPool``; this class owns only the
free-slot bookkeeping.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch


class MambaSlotAllocator:
    """Free-list of Mamba pool slot indices. Deliberately not a subclass of
    ``BaseTokenToKVPoolAllocator``: slots are per request, not per token.

    With ``host_slots > 0`` (the int8 checkpoint pool's RAM tier) it is one
    allocator over two tiers: logical slot ids [1, size] are device slots
    (physical row = id), ids [size+1, size+host_slots] are host slots
    (physical row = id - size). Slot 0 is reserved on the device tier only.
    ``alloc`` is device-first; with the default ``host_slots=0`` every
    behavior (ids, ordering, identity translate) is unchanged."""

    def __init__(self, size: int, device: str, host_slots: int = 0):
        self.size = size
        self.device = device
        self.host_slots = host_slots
        # Set by alloc_group_begin(); alloc(1) drains it until alloc_group_end().
        self._alloc_iter: Optional[Iterator] = None
        self.clear()

    def translate(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Static slot ids already index physical storage.

        Valid for the DEVICE tier only (row == id there). RAM-tier ids
        address a different plane at row ``id - size``; callers holding
        host values resolve them through the checkpoint pool, never here.
        """
        return slot_ids

    def tier_of(self, slot_id: int) -> tuple[str, int]:
        """``("device"|"host", physical_row)`` for a logical slot id."""
        if slot_id <= self.size:
            return ("device", slot_id)
        return ("host", slot_id - self.size)

    def available_size(self) -> int:
        return len(self.free_slots) + len(self.free_host_slots)

    def schedulable_available_size(self) -> int:
        """Planner-facing free count. Same as ``available_size`` for a static pool;
        byte-coordinated allocators return their byte-limited view instead."""
        return self.available_size()

    def slots_fit(self, count: int) -> bool:
        """Whether ``count`` slots can be taken now."""
        return self.schedulable_available_size() >= count

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch of slots for match_prefix to amortize overhead."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._do_alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    @property
    def alloc_group_open(self) -> bool:
        """True between ``alloc_group_begin``/``end``: ``alloc(1)`` then drains
        a pre-allocated batch whose tier composition is no longer decided by
        the current free counts."""
        return self._alloc_iter is not None

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self.free(torch.cat(remaining))
        self._alloc_iter = None

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._do_alloc(need_size)

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > self.available_size():
            return None
        if need_size == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        from_device = min(need_size, len(self.free_slots))
        parts = []
        if from_device > 0:
            parts.append(self.free_slots[:from_device])
            self.free_slots = self.free_slots[from_device:]
        from_host = need_size - from_device
        if from_host > 0:
            parts.append(self.free_host_slots[:from_host])
            self.free_host_slots = self.free_host_slots[from_host:]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def free(self, free_index: torch.Tensor, *, owned: bool = False):
        # ``owned`` matters only to a deferred free; this one frees at once.
        if free_index.numel() == 0:
            return
        if self.host_slots == 0:
            self.free_slots = torch.cat((self.free_slots, free_index))
            return
        # Split by tier without ANY host sync (no bool()/any()): both cats run
        # and a possibly-empty selection is a no-op.
        is_host = free_index > self.size
        self.free_slots = torch.cat(
            (self.free_slots, free_index.masked_select(~is_host))
        )
        self.free_host_slots = torch.cat(
            (self.free_host_slots, free_index.masked_select(is_host))
        )

    def clear(self):
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.free_host_slots = torch.arange(
            self.size + 1,
            self.size + self.host_slots + 1,
            dtype=torch.int64,
            device=self.device,
        )
