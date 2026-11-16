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

import torch


from sglang.srt.environ import envs
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
        # Read once: the env is set before boot, and the tick also runs in
        # the idle spin between steps.
        self._debug_scan = envs.SGLANG_DEBUG_MEMORY_POOL.get()

    @property
    def predicted_tick_step(self) -> int:
        """The `step` value the next on_step_boundary will judge at (it
        increments before rotating). The overlap handshake evaluates
        wants_park at this step, so its prediction and the tick's own
        _parkable share one step base without duplicating the +1."""
        return self.step + 1

    def wants_park(
        self, running_batch: ScheduleBatch, waiting_count: int, *, at_step: int
    ) -> bool:
        """Whether a tick running at `at_step` would want to park AND could:
        the park-branch conditions plus a durably eligible victim. The
        overlap loop calls this (at ``predicted_tick_step``) to decide
        early result processing — parking a victim whose step result is
        still unprocessed would drop that step's sampled token from its
        output (see _parkable) — so the commit must land first; the victim
        check keeps permanently infeasible rotations (session/beam sets,
        wake-quantum windows) from silently disabling overlap for nothing.
        Must not mutate; unlike _select_victim it does not advance the
        rotor. A real tick always increments once before judging, so the
        prediction is exact; a cheap-idle early return (which increments
        nothing) only makes the next prediction match again, and a stale
        lower `at_step` could only over-restrict (extra wake-quantum
        protection), never park a protected victim. ``at_step`` is
        keyword-only: the wrong step base silently re-opens the
        uncommitted-victim park."""
        return (
            waiting_count > 0
            and self.state.slots.available_size() == 0
            and running_batch.batch_size() >= 2
            and self.swap_pool.available_size() > 0
            and any(self._victim_eligible(r, step=at_step) for r in running_batch.reqs)
        )

    def on_step_boundary(
        self, running_batch: ScheduleBatch, *, waiting_count: int
    ) -> List[Req]:
        """Run one rotation tick; return woken reqs for the caller to merge
        back into the decode batch. Called only at a step boundary."""
        if self._debug_scan:
            # Census the state carried in from previous ticks, before wake
            # or park mutate it.
            self._check_parked_ownership()
        if not self.parked_reqs and (
            waiting_count == 0 or running_batch.batch_size() < 2
        ):
            # Idle cheap path: neither branch could act, and this runs every
            # scheduler pass including the spin between steps.
            return []
        self.step += 1
        free = self.state.slots.available_size()
        logger.debug(
            "mamba swap tick: step=%d parked=%d free=%d waiting=%d bs=%s "
            "swapfree=%d parks=%d wakes=%d",
            self.step,
            len(self.parked_reqs),
            free,
            waiting_count,
            running_batch.batch_size(),
            self.swap_pool.available_size(),
            self.num_parks,
            self.num_wakes,
        )
        if len(self.parked_reqs) > 0 and free > 0:
            woken = self._wake_next()
            return [woken] if woken is not None else []
        if self.wants_park(running_batch, waiting_count, at_step=self.step):
            victim = self._select_victim(running_batch.reqs)
            if victim is not None:
                self._park(running_batch, victim)
        return []

    # ===== Invariant scan =====

    def _check_parked_ownership(self) -> None:
        """Invariant 1 (debug builds): a parked request holds no active
        row — working row, CoW source or deferred clear — and owns exactly
        one live, distinct swap record whose row count matches its ping-pong
        buffer (parked ping-pong entries keep their freed ids; wake rebinds
        them, so staleness is fine but a count drift is not).

        Runs at the step boundary, where the scheduler is the single
        writer; the idle memory-pool checker cannot host this scan
        because a non-empty parked list keeps the scheduler non-idle.
        """
        seen_ids = set()
        for req in self.parked_reqs:
            kv = req.kv
            problems = []
            if kv.mamba_pool_idx is not None:
                problems.append(f"row={int(kv.mamba_pool_idx.reshape(-1)[0])}")
            if kv.mamba_cow_src_index is not None:
                problems.append("cow_src")
            if kv.mamba_slot_clear_queue:
                problems.append("clear_queue")
            swap_id = kv.mamba_swap_id
            record_rows = (
                None if swap_id is None else self.swap_pool.parked_num_rows(swap_id)
            )
            if record_rows is None:
                problems.append(f"record={swap_id}")
            elif swap_id in seen_ids:
                problems.append(f"dup_record={swap_id}")
            else:
                seen_ids.add(swap_id)
                buf = kv.mamba_ping_pong_track_buffer
                # Parked buffer entries are exactly the record's track rows
                # (the working row is record row 0, never in the buffer).
                n_track = 0 if buf is None else int((buf != -1).sum().item())
                positions = kv.mamba_swap_track_positions
                if record_rows != 1 + n_track:
                    problems.append(f"record_rows={record_rows} vs track={n_track}")
                if n_track > 0 and not positions:
                    problems.append("no_positions")
                elif positions is not None and len(positions) != n_track:
                    problems.append("positions")
            if problems:
                raise RuntimeError(
                    f"mamba swap invariant 1 violated: rid={req.rid} "
                    f"{','.join(problems)}"
                )

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

    def _victim_eligible(self, req: Req, *, step: int) -> bool:
        """Durable park eligibility: nothing here is fixed by committing
        the in-flight step, so the overlap handshake (wants_park) and the
        tick (_parkable) share it. Quantum expiry is judged at `step`,
        which for wants_park is the step the consuming tick will run at."""
        if req.kv.mamba_pool_idx is None or req.kv.mamba_swap_id is not None:
            return False
        if req.session is not None or req.beam_group is not None:
            return False
        # The radix CoW source names a tree row this request's working copy
        # is still pending a copy from: dropping it would strand that row.
        # Ping-pong rows and deferred clears ARE movable now (V2.5): the
        # park drains the clears and copies every allocated row.
        if req.kv.mamba_cow_src_index is not None:
            return False
        # Defensive: tracking rows beyond record capacity (a config drift)
        # would spill into the neighbour record's host rows.
        buf = req.kv.mamba_ping_pong_track_buffer
        if (
            buf is not None
            and 1 + int((buf != -1).sum().item()) > self.swap_pool.rows_per_record
        ):
            return False
        woken = req.kv.mamba_swap_woken_iter
        return woken < 0 or step - woken >= self.wake_quantum

    def _parkable(self, req: Req) -> bool:
        # Transient conditions the overlap handshake commits away: a
        # merged-not-yet-finished prefill has no output yet, and a launched
        # step with no committed result would lose that result if the row
        # left the running list (the parked record would hold the post-step
        # state, so wake would re-feed a token it already consumed).
        transient_ok = req.inflight_steps == 0 and len(req.output_ids) > 0
        return transient_ok and self._victim_eligible(req, step=self.step)

    def _park(self, running_batch: ScheduleBatch, req: Req) -> None:
        self._drain_pending_clears(req)
        slots, positions = self._park_rows(req)
        swap_id = self.swap_pool.park(slots, order_streams=self.order_streams)
        if swap_id is None:
            return
        req.kv.mamba_swap_id = swap_id
        req.kv.mamba_swap_track_positions = positions
        # The state now lives in the swap tier; the active rows' bookkeeping
        # goes with it. The request keeps its token pages and its req row;
        # the ping-pong buffer entries keep the freed ids until wake rebinds
        # them. _victim_eligible kept the CoW source out of this set, so the
        # only obligations discharged here are the parked rows themselves.
        req.kv.mamba_pool_idx = None
        req.kv.mamba_needs_clear = False
        move([req], self.parked_reqs)
        running_batch.filter_batch(reqs_to_exclude=[req])
        running_batch.batch_is_full = False
        self.num_parks += 1
        logger.debug(
            "mamba swap: parked %s (swap slot %d, freed active slots %s)",
            req.rid,
            swap_id,
            slots,
        )

    def _park_rows(self, req: Req):
        """The request's active row set: the working row first, then each
        allocated ping-pong row with its buffer position. -1 entries are
        lazy's unallocated second slot; they park with the request, not as
        rows."""
        slots = [int(req.kv.mamba_pool_idx.reshape(-1)[0])]
        positions: List[int] = []
        buf = req.kv.mamba_ping_pong_track_buffer
        if buf is not None:
            for pos, slot in enumerate(buf.tolist()):
                if slot != -1:
                    slots.append(int(slot))
                    positions.append(pos)
        return slots, positions

    def _drain_pending_clears(self, req: Req) -> None:
        """Run a request's deferred mamba clears now, before the D2H, so the
        parked record holds zeroed scratch rows and no clear bookkeeping
        outlives the park (which would require wake to remap slot ids).
        Queued donate-swaps and freshly recycled ping-pong rows all park
        inside the record; the working row needs no clear because wake
        overwrites it whole from the record (v1 dropped that flag the same
        way). Safe at a step boundary: no forward is in flight to race."""
        kv = req.kv
        rows = []
        if kv.mamba_ping_pong_needs_clear:
            buf = kv.mamba_ping_pong_track_buffer
            if buf is not None:
                fresh = buf[buf != -1]
                if fresh.numel() > 0:
                    rows.append(fresh)
            kv.mamba_ping_pong_needs_clear = False
        if kv.mamba_slot_clear_queue:
            rows.extend(kv.mamba_slot_clear_queue)
            kv.mamba_slot_clear_queue = None
        if rows:
            self.state.pool.clear_slots(torch.cat(rows))

    def admission_slots(self, req: Req) -> int:
        """Active slots an admission will take from the state pool: its
        working row plus initial tracking rows (extra-buffer configs).
        Matches StateStorage.acquire; lazy mid-life growth beyond the
        initial rows is best-effort (alloc failure degrades the track)."""
        return self.state.pending_request_slots(self.table, req)

    # ===== Wake =====

    def _wake_next(self) -> Optional[Req]:
        """Take the longest-parked request, give it fresh active rows for
        its whole record, and move it back to the running list; the caller
        merges it into decode. Returns None (staying parked) if the rows
        vanished between the free check and the alloc — same-tick retire or
        an admit racing ahead."""
        req = next(iter(self.parked_reqs))
        swap_id = req.kv.mamba_swap_id
        n = self.swap_pool.parked_num_rows(swap_id)
        slot = self.state.slots.alloc(n)
        if slot is None:
            # Graceful defer: the free check only proved ≥1 row, and a V2.5
            # record may own more rows than any single release reopened.
            # Reaching this every tick would starve the rotation — keep it
            # visible.
            logger.warning(
                "mamba swap: wake deferred for %s (%d free active slot(s), "
                "record wants %d)",
                req.rid,
                self.state.slots.available_size(),
                n,
            )
            return None
        self.swap_pool.wake(
            swap_id, [int(x) for x in slot], order_streams=self.order_streams
        )
        req.kv.mamba_pool_idx = slot[0]
        req.kv.mamba_swap_id = None
        req.kv.mamba_swap_woken_iter = self.step
        # The table maps rows to slots; the woken request gets new slots.
        self.table.req_index_to_mamba_index_mapping[req.kv.req_pool_idx] = int(slot[0])
        # Rebind the ping-pong positions (sets the device-side mapping and
        # empties flags alongside the request-held buffer).
        for i, pos in enumerate(req.kv.mamba_swap_track_positions or ()):
            self.table.set_mamba_ping_pong_slot(req, pos, slot[i + 1])
        req.kv.mamba_swap_track_positions = None
        move([req], self.running_reqs)
        self.num_wakes += 1
        logger.debug("mamba swap: woke %s into active slots %s", req.rid, slot.tolist())
        return req
