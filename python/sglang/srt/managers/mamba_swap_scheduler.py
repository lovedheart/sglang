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

MambaSwapScheduler -- park/wake policy for virtual SSM slots.

The active mamba state pool caps how many requests can decode at once. This
policy lifts that cap without buying VRAM: when an admission wants an active
slot and the pool is exhausted, it PARKS one running request (its exact bf16
state moves to a pinned swap slot via MambaSwapPool; the active slot goes
back to the allocator), and when a slot frees up it WAKES the longest-parked
request into it, resuming decode with no re-prefill.

Calls land only between steps (the scheduler's batch boundary), so no
forward is mid-capture here. The policy never touches KV: a parked request
keeps its token pages and its req row; only the active state slot moves.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence


from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.request_list import move

logger = logging.getLogger(__name__)


class MambaSwapScheduler:
    """Rotating park/wake over the running set (module docstring).

    ``wake_quantum`` protects a just-woken request from being re-parked
    before it has decoded a fair share of steps, which is what keeps the
    rotation from livelocking into park/wake churn.
    """

    # Park/wake is one-per-step: prefills admit one pass at a time anyway,
    # and a slow rotation bounds the state in flight through the copy stream.
    def __init__(
        self,
        *,
        swap_pool,
        state,
        table,
        running_reqs,
        parked_reqs,
        order_streams,
        wake_quantum: int = 32,
    ):
        self.swap_pool = swap_pool
        self.state = state  # StateStorage: the slot allocator + active pool
        self.table = table  # hybrid req-to-token table (slot<->row mapping)
        self.running_reqs = running_reqs
        self.parked_reqs = parked_reqs
        # Streams that may host the next forward; every copy is fenced
        # behind/against them (see MambaSwapPool's stream model).
        self.order_streams = tuple(s for s in order_streams if s is not None)
        self.wake_quantum = wake_quantum
        self.step = 0
        self._rotor = 0
        self.num_parks = 0
        self.num_wakes = 0

    def on_step_boundary(
        self, running_batch: ScheduleBatch, *, waiting_count: int
    ) -> List[Req]:
        """Run one rotation tick; return woken reqs for the caller to merge
        back into the decode batch. Called only at a step boundary."""
        self.step += 1
        free = self.state.slots.available_size()
        logger.debug(
            "mamba swap tick: step=%d parked=%d free=%d waiting=%d bs=%s swapfree=%d",
            self.step,
            len(self.parked_reqs),
            free,
            waiting_count,
            running_batch.batch_size(),
            self.swap_pool.available_size(),
        )
        if len(self.parked_reqs) > 0 and free > 0:
            return [self._wake_next()]
        if (
            waiting_count > 0
            and free == 0
            and running_batch.batch_size() >= 2
            and self.swap_pool.available_size() > 0
        ):
            victim = self._select_victim(running_batch.reqs)
            if victim is not None:
                self._park(running_batch, victim)
        return []

    # ===== Park =====

    def _select_victim(self, reqs: Sequence[Req]) -> Optional[Req]:
        """Round-robin over the running set; skip requests that are not
        parkable (freshly woken, streaming-session state, beam groups)."""
        n = len(reqs)
        for offset in range(n):
            req = reqs[(self._rotor + offset) % n]
            if not self._parkable(req):
                continue
            self._rotor = (self._rotor + offset + 1) % n
            return req
        return None

    def _parkable(self, req: Req) -> bool:
        if req.kv.mamba_pool_idx is None or req.kv.mamba_swap_id is not None:
            return False
        if req.session is not None or req.beam_group is not None:
            return False
        # Park/wake round-trips the main row only. Anything with state beyond
        # that row (radix CoW source, deferred clear, ping-pong tracking) is
        # not movable by this path; skip rather than silently discard it.
        if (
            req.kv.mamba_cow_src_index is not None
            or req.kv.mamba_slot_clear_queue is not None
            or req.kv.mamba_ping_pong_track_buffer is not None
        ):
            return False
        woken = req.kv.mamba_swap_woken_iter
        return woken < 0 or self.step - woken >= self.wake_quantum

    def _park(self, running_batch: ScheduleBatch, req: Req) -> None:
        active_slot = int(req.kv.mamba_pool_idx.reshape(-1)[0])
        swap_id = self.swap_pool.park(active_slot, order_streams=self.order_streams)
        if swap_id is None:
            return
        req.kv.mamba_swap_id = swap_id
        # The state now lives in the swap tier; the active bookkeeping goes
        # with it. The request keeps its token pages and its req row.
        req.kv.mamba_pool_idx = None
        req.kv.mamba_needs_clear = False
        req.kv.mamba_cow_src_index = None
        req.kv.mamba_slot_clear_queue = None
        move([req], self.parked_reqs)
        running_batch.filter_batch(reqs_to_exclude=[req])
        running_batch.batch_is_full = False
        self.num_parks += 1
        logger.debug(
            "mamba swap: parked %s (swap slot %d, freed active slot %d)",
            req.rid,
            swap_id,
            active_slot,
        )

    # ===== Wake =====

    def _wake_next(self) -> Req:
        """Take the longest-parked request, give it a fresh active slot, and
        move it back to the running list; the caller merges it into decode."""
        req = next(iter(self.parked_reqs))
        slot = self.state.slots.alloc(1)
        assert slot is not None, "wake ran with no free active slot"
        swap_id = req.kv.mamba_swap_id
        self.swap_pool.wake(swap_id, int(slot[0]), order_streams=self.order_streams)
        req.kv.mamba_pool_idx = slot[0]
        req.kv.mamba_swap_id = None
        req.kv.mamba_swap_woken_iter = self.step
        # The table maps rows to slots; the woken request gets a new slot.
        self.table.req_index_to_mamba_index_mapping[req.kv.req_pool_idx] = int(slot[0])
        move([req], self.running_reqs)
        self.num_wakes += 1
        logger.debug("mamba swap: woke %s into active slot %d", req.rid, int(slot[0]))
        return req
