"""CPU policy tests for MambaSwapScheduler (virtual SSM slots).

Drives the rotation tick directly against real pools and real request lists
with minimal fake requests: park-on-starvation, wake-into-capacity, the wake
quantum's re-park protection, and the non-parkable exclusions. The running
batch is stand-in for the three members the tick touches (reqs, batch_size,
filter_batch).

    python -m pytest test/registered/unit/managers/test_mamba_swap_scheduler.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.environ import envs
from sglang.srt.managers.mamba_swap_scheduler import MambaSwapScheduler
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.mamba_swap_pool import MambaSwapPool, _ParkedSlot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

L = 2


class _FakeReq:
    """Identity-hashable stand-in: the real Req is hashable by identity."""

    # Class attribute: only move()/the retirer set instance ownership, and the
    # list-edit audit forbids owner_list assignments outside those modules.
    owner_list = None
    # Committed-by-default: a launched-but-unprocessed step sets this on the
    # real Req; tests raise it to model the overlap in-flight case.
    inflight_steps = 0

    def __init__(self, rid):
        self.rid = rid
        self.kv = ReqKvInfo()
        self.session = None
        self.beam_group = None
        # Decoded at least one token; a merged-not-yet-finished prefill has [].
        self.output_ids = [1]


class _FakeRunningBatch:
    """The three running-batch members the rotation tick touches."""

    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.batch_is_full = False

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self, reqs_to_exclude=None):
        excl = set(reqs_to_exclude or [])
        self.reqs = [r for r in self.reqs if r not in excl]


class TestMambaSwapRotation(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            MambaSwapPool,
            "_alloc_host",
            lambda self, shape, dtype: torch.zeros(shape, dtype=dtype, device="cpu"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        temporal = torch.zeros(L, 5, 4, 8, 8, dtype=torch.bfloat16)
        conv = [torch.zeros(L, 5, 2, 4, 8, dtype=torch.bfloat16)]
        mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=temporal,
                conv=conv,
                replayssm_d=None,
                replayssm_k=None,
                replayssm_g=None,
                replayssm_rawv=None,
                replayssm_rawk=None,
                replayssm_beta=None,
            ),
            enable_linear_replayssm=False,
            enable_linear_replayssm_spec=False,
            replayssm_request_ring=False,
            replayssm_write_pos=None,
            _slot_siblings=[],
        )
        self.slots = MambaSlotAllocator(2, "cpu")
        self.swap_pool = MambaSwapPool(
            mamba_pool=mamba_pool,
            mamba_allocator=self.slots,
            swap_slots=3,
            device="cpu",
        )
        self.table = SimpleNamespace(
            req_index_to_mamba_index_mapping=torch.zeros(8, dtype=torch.int32)
        )
        self.running_reqs = KeyedReqList(ListKind.DECODING)
        self.parked = IndexedReqList(ListKind.MAMBA_PARKED)
        self.swap = MambaSwapScheduler(
            swap_pool=self.swap_pool,
            state=SimpleNamespace(slots=self.slots, pool=mamba_pool),
            table=self.table,
            running_reqs=self.running_reqs,
            parked_reqs=self.parked,
            order_streams=(),
            wake_quantum=4,
        )

    def _req(self, rid):
        req = _FakeReq(rid)
        req.kv.req_pool_idx = len(self.running_reqs) + 1
        req.kv.mamba_pool_idx = self.slots.alloc(1)[0]
        move([req], self.running_reqs)
        return req

    def _running_batch(self, reqs):
        return _FakeRunningBatch(reqs)

    def test_starved_admission_parks_one_victim(self):
        batch = self._running_batch([self._req("r0"), self._req("r1")])
        woken = self.swap.on_step_boundary(batch, waiting_count=1)
        self.assertEqual(woken, [])
        self.assertEqual(len(self.parked), 1)
        self.assertEqual(batch.batch_size(), 1)
        self.assertEqual(self.slots.available_size(), 1)
        victim = next(iter(self.parked))
        self.assertIsNotNone(victim.kv.mamba_swap_id)
        self.assertIsNone(victim.kv.mamba_pool_idx)
        self.assertEqual(self.swap_pool.num_parked(), len(list(self.parked)))

    def test_parked_wakes_into_freed_capacity(self):
        batch = self._running_batch([self._req("r0"), self._req("r1")])
        self.swap.on_step_boundary(batch, waiting_count=1)
        victim = next(iter(self.parked))
        woken = self.swap.on_step_boundary(batch, waiting_count=0)
        self.assertEqual(woken, [victim])
        self.assertEqual(len(self.parked), 0)
        self.assertEqual(self.swap_pool.num_parked(), 0)
        self.assertIsNone(victim.kv.mamba_swap_id)
        self.assertEqual(
            int(victim.kv.mamba_pool_idx),
            int(self.table.req_index_to_mamba_index_mapping[victim.kv.req_pool_idx]),
        )
        self.assertIn(victim, self.running_reqs)
        # The woken req reclaimed the slot freed by its park.
        self.assertEqual(self.slots.available_size(), 0)

    def test_wake_quantum_protects_freshly_woken(self):
        r0, r1 = self._req("r0"), self._req("r1")
        batch = self._running_batch([r0, r1])
        self.swap.on_step_boundary(batch, waiting_count=1)
        first = next(iter(self.parked))
        self.swap.on_step_boundary(batch, waiting_count=0)  # wakes `first`
        batch.reqs.append(first)  # the merge would add it back
        # Starve admission again: the woken request is inside its quantum, so
        # the round-robin must park the other request.
        self.swap.on_step_boundary(batch, waiting_count=1)
        self.assertNotIn(first, self.parked)
        self.assertEqual(len(self.parked), 1)
        self.assertNotEqual(next(iter(self.parked)), first)

    def test_quantum_expiry_frees_the_rotation(self):
        r0, r1 = self._req("r0"), self._req("r1")
        batch = self._running_batch([r0, r1])
        self.swap.on_step_boundary(batch, waiting_count=1)
        self.swap.on_step_boundary(batch, waiting_count=0)
        # Step past the quantum with no-op boundaries (no waiting, no parked).
        for _ in range(4):
            self.swap.on_step_boundary(batch, waiting_count=0)
        batch2 = self._running_batch(list(self.running_reqs))
        self.swap.on_step_boundary(batch2, waiting_count=1)
        self.assertEqual(len(self.parked), 1)

    def test_no_park_when_swap_tier_full(self):
        batch = self._running_batch([self._req("r0"), self._req("r1")])
        self.swap_pool.free_ids = []
        self.swap.on_step_boundary(batch, waiting_count=1)
        self.assertEqual(len(self.parked), 0)
        self.assertEqual(batch.batch_size(), 2)

    def test_min_running_and_no_demand_never_park(self):
        r0, r1 = self._req("r0"), self._req("r1")
        # Only one request running: never park (something must decode).
        self.swap.on_step_boundary(self._running_batch([r0]), waiting_count=1)
        self.assertEqual(len(self.parked), 0)
        # Demand-free step with two running: nothing to park for.
        self.swap.on_step_boundary(self._running_batch([r0, r1]), waiting_count=0)
        self.assertEqual(len(self.parked), 0)

    def test_session_and_beam_requests_are_not_parkable(self):
        reqs = [self._req("r0"), self._req("r1")]
        for req in reqs:
            req.session = object()
        self.swap.on_step_boundary(self._running_batch(reqs), waiting_count=1)
        self.assertEqual(len(self.parked), 0)
        for req in reqs:
            req.session = None
            req.beam_group = object()
        self.swap.on_step_boundary(self._running_batch(reqs), waiting_count=1)
        self.assertEqual(len(self.parked), 0)

    def test_prefill_not_yet_finished_is_not_parkable(self):
        # Overlap merges a prefill's reqs before its result appends output;
        # parking one leaves wake reading output_ids[-1] of an empty array.
        reqs = [self._req("r0"), self._req("r1")]
        for req in reqs:
            req.output_ids = []
        self.swap.on_step_boundary(self._running_batch(reqs), waiting_count=1)
        self.assertEqual(len(self.parked), 0)
        reqs[0].output_ids = [7]
        self.swap.on_step_boundary(self._running_batch(reqs), waiting_count=1)
        self.assertEqual(len(self.parked), 1)

    def test_inflight_step_is_not_parkable(self):
        # Overlap may leave a victim's last decode step launched but
        # uncommitted: parking then drops that row's sampled token (the
        # result skips rows that left the running list) and captures a
        # post-step state, so wake would re-feed a consumed token.
        reqs = [self._req("r0"), self._req("r1")]
        for req in reqs:
            req.inflight_steps = 1
        # The handshake must still fire (it commits the step away), but
        # the tick itself must not park until the results land.
        batch = self._running_batch(reqs)
        self.assertTrue(
            self.swap.wants_park(batch, waiting_count=1, at_step=self.swap.step)
        )
        self.swap.on_step_boundary(batch, waiting_count=1)
        self.assertEqual(len(self.parked), 0)
        # The early-process handshake commits r0's step: it parks next tick.
        reqs[0].inflight_steps = 0
        self.swap.on_step_boundary(self._running_batch(reqs), waiting_count=1)
        self.assertEqual(len(self.parked), 1)
        self.assertIs(next(iter(self.parked)), reqs[0])

    def test_wants_park_tracks_the_park_branch(self):
        # The overlap loop calls wants_park to decide early result
        # processing; it must agree with the tick's park precondition,
        # including whether any victim is parkable at all.
        batch = self._running_batch([self._req("r0"), self._req("r1")])
        self.assertFalse(
            self.swap.wants_park(batch, waiting_count=0, at_step=self.swap.step)
        )
        self.assertTrue(
            self.swap.wants_park(batch, waiting_count=1, at_step=self.swap.step)
        )
        # Nothing parkable (session states): overlap must stay enabled.
        for r in batch.reqs:
            r.session = object()
        self.assertFalse(
            self.swap.wants_park(batch, waiting_count=1, at_step=self.swap.step)
        )
        for r in batch.reqs:
            r.session = None
        self.swap.on_step_boundary(batch, waiting_count=1)
        # A slot freed by the park: nothing more to park this tick.
        self.assertFalse(
            self.swap.wants_park(batch, waiting_count=1, at_step=self.swap.step)
        )

    def test_predicted_tick_step_matches_the_tick(self):
        # wants_park(at_step=predicted_tick_step) must denote exactly the
        # step the next on_step_boundary judges at: one increment per tick.
        predicted = self.swap.predicted_tick_step
        batch = self._running_batch([self._req("r0"), self._req("r1")])
        self.swap.on_step_boundary(batch, waiting_count=1)
        self.assertEqual(self.swap.step, predicted)

    def test_wake_is_graceful_when_the_slot_vanishes(self):
        # F13: a same-tick retire may consume the free slot between the
        # gate and the alloc; the wake must defer, not assert.
        batch = self._running_batch([self._req("r0"), self._req("r1")])
        self.swap.on_step_boundary(batch, waiting_count=1)
        victim = next(iter(self.parked))
        original = self.slots.alloc
        self.slots.alloc = lambda n: None
        try:
            woken = self.swap.on_step_boundary(batch, waiting_count=0)
        finally:
            self.slots.alloc = original
        self.assertEqual(woken, [])
        self.assertIn(victim, self.parked)
        self.assertIsNotNone(victim.kv.mamba_swap_id)
        # The retry lands: the record was untouched by the failed tick.
        self.assertEqual(self.swap.on_step_boundary(batch, waiting_count=0), [victim])

    def test_debug_scan_accepts_healthy_park_and_flags_stolen_row(self):
        batch = self._running_batch([self._req("r0"), self._req("r1")])
        with envs.SGLANG_DEBUG_MEMORY_POOL.override(True):
            # Parking runs the scan inline: a healthy park passes.
            self.swap.on_step_boundary(batch, waiting_count=1)
            victim = next(iter(self.parked))
            # Someone else reclaims the freed row while it is parked.
            victim.kv.mamba_pool_idx = self.slots.alloc(1)[0]
            with self.assertRaisesRegex(RuntimeError, "row="):
                self.swap._check_parked_ownership()

    def test_debug_scan_flags_lost_or_duplicate_record(self):
        batch = self._running_batch([self._req("r0"), self._req("r1")])
        self.swap.on_step_boundary(batch, waiting_count=1)
        victim = next(iter(self.parked))
        victim.kv.mamba_swap_id = 999
        with self.assertRaisesRegex(RuntimeError, "record=999"):
            self.swap._check_parked_ownership()
        # Back to a live id, then a second parked req claims the same one.
        other = self._req("r2")
        other.kv.mamba_pool_idx = None
        move([other], self.parked)
        # A minimal real record shape: one row, CPU fence-free.
        self.swap_pool._parked[42] = _ParkedSlot(None, [], 1)
        victim.kv.mamba_swap_id = 42
        other.kv.mamba_swap_id = 42
        with self.assertRaisesRegex(RuntimeError, "dup_record=42"):
            self.swap._check_parked_ownership()


class _FakeTableV25:
    """The extra-buffer slice of HybridReqToTokenPool the rotation touches,
    mirroring set_mamba_ping_pong_slot's device-mapping maintenance."""

    def __init__(self):
        self.req_index_to_mamba_index_mapping = torch.zeros(8, dtype=torch.int32)
        self.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
            8, 2, dtype=torch.int32
        )
        self.enable_mamba_extra_buffer = True
        self.mamba_initial_tracking_slots = 1
        self.mamba_ping_pong_track_buffer_size = 2

    def set_mamba_ping_pong_slot(self, req, idx, value):
        if torch.is_tensor(value):
            req.kv.mamba_ping_pong_track_buffer[idx] = value
        else:
            req.kv.mamba_ping_pong_track_buffer[idx].fill_(-1)
        self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.kv.req_pool_idx] = (
            req.kv.mamba_ping_pong_track_buffer
        )
        req.kv.mamba_ping_pong_empty[idx] = not torch.is_tensor(value)


class TestMambaSwapTrackedRotation(unittest.TestCase):
    """V2.5: park/wake with ping-pong tracking rows (extra_buffer configs)."""

    def setUp(self):
        patcher = mock.patch.object(
            MambaSwapPool,
            "_alloc_host",
            lambda self, shape, dtype: torch.zeros(shape, dtype=dtype, device="cpu"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        temporal = torch.zeros(L, 7, 4, 8, 8, dtype=torch.bfloat16)
        conv = [torch.zeros(L, 7, 2, 4, 8, dtype=torch.bfloat16)]
        self.cleared = []
        self.mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=temporal,
                conv=conv,
                replayssm_d=None,
                replayssm_k=None,
                replayssm_g=None,
                replayssm_rawv=None,
                replayssm_rawk=None,
                replayssm_beta=None,
            ),
            enable_linear_replayssm=False,
            enable_linear_replayssm_spec=False,
            replayssm_request_ring=False,
            replayssm_write_pos=None,
            _slot_siblings=[],
            # Records the drain (the real impl zeroes the rows on the
            # forward stream; the rows leave via park either way).
            clear_slots=self.cleared.append,
        )
        self.slots = MambaSlotAllocator(4, "cpu")
        self.table = _FakeTableV25()
        self.swap_pool = MambaSwapPool(
            mamba_pool=self.mamba_pool,
            mamba_allocator=self.slots,
            swap_slots=3,
            device="cpu",
            rows_per_record=2,
        )
        self.running_reqs = KeyedReqList(ListKind.DECODING)
        self.parked = IndexedReqList(ListKind.MAMBA_PARKED)
        self.swap = MambaSwapScheduler(
            swap_pool=self.swap_pool,
            state=SimpleNamespace(
                slots=self.slots,
                pool=self.mamba_pool,
                # Mirror of StateStorage.pending_request_slots.
                pending_request_slots=lambda table, req: (
                    int(not req.kv.holds_mamba)
                    + (
                        table.mamba_initial_tracking_slots
                        if table.enable_mamba_extra_buffer
                        and req.kv.mamba_ping_pong_track_buffer is None
                        else 0
                    )
                ),
            ),
            table=self.table,
            running_reqs=self.running_reqs,
            parked_reqs=self.parked,
            order_streams=(),
            wake_quantum=4,
        )

    def _tracked_req(self, rid):
        req = _FakeReq(rid)
        req.kv.req_pool_idx = len(self.running_reqs) + 1
        # acquire(): working row first, then the initial tracking row; the
        # buffer is the lazy shape [row, -1] until the second row exists.
        pair = self.slots.alloc(2)
        req.kv.mamba_pool_idx = pair[0]
        buf = pair.new_full((2,), -1)
        buf[0] = pair[1]
        req.kv.mamba_ping_pong_track_buffer = buf
        req.kv.mamba_ping_pong_empty = [False, True]
        move([req], self.running_reqs)
        return req

    def test_tracked_park_wake_rebinds_the_track_row(self):
        r0, r1 = self._tracked_req("r0"), self._tracked_req("r1")
        old_track = int(r0.kv.mamba_ping_pong_track_buffer[0])
        batch = _FakeRunningBatch([r0, r1])
        self.swap.on_step_boundary(batch, waiting_count=1)
        victim = next(iter(self.parked))
        self.assertIs(victim, r0)
        self.assertEqual(self.swap_pool.parked_num_rows(victim.kv.mamba_swap_id), 2)
        self.assertIsNone(victim.kv.mamba_pool_idx)
        self.assertEqual(victim.kv.mamba_swap_track_positions, [0])
        # The parked buffer carries the lazy shape unchanged.
        self.assertEqual(
            victim.kv.mamba_ping_pong_track_buffer.tolist(), [old_track, -1]
        )
        # A record-complete parked req passes the invariant scan.
        self.swap._check_parked_ownership()

        woke = self.swap.on_step_boundary(batch, waiting_count=0)
        self.assertEqual(woke, [victim])
        self.assertIsNone(victim.kv.mamba_swap_id)
        self.assertIsNone(victim.kv.mamba_swap_track_positions)
        new_main = int(victim.kv.mamba_pool_idx)
        new_track = victim.kv.mamba_ping_pong_track_buffer.tolist()
        self.assertEqual(
            new_main,
            int(self.table.req_index_to_mamba_index_mapping[victim.kv.req_pool_idx]),
        )
        self.assertEqual(
            new_track,
            self.table.req_index_to_mamba_ping_pong_track_buffer_mapping[
                victim.kv.req_pool_idx
            ].tolist(),
        )
        # Main and track are distinct rows (an atomic two-row alloc).
        self.assertNotIn(new_main, new_track)
        self.assertEqual(victim.kv.mamba_ping_pong_empty, [False, True])
        self.assertEqual(self.slots.available_size(), 0)

    def test_park_drains_queued_clears(self):
        req = self._tracked_req("r0")
        other = self._tracked_req("r1")
        # A donate swap staged a clear on the swapped-in track row, plus the
        # hand-out flag over the whole buffer.
        track_row = int(req.kv.mamba_ping_pong_track_buffer[0])
        req.kv.mamba_slot_clear_queue = [torch.tensor([track_row], dtype=torch.int32)]
        req.kv.mamba_ping_pong_needs_clear = True
        batch = _FakeRunningBatch([req, other])
        self.swap.on_step_boundary(batch, waiting_count=1)
        # Whichever victim was picked, no clear bookkeeping survives a park.
        for parked_req in self.parked:
            self.assertIsNone(parked_req.kv.mamba_slot_clear_queue)
            self.assertFalse(parked_req.kv.mamba_ping_pong_needs_clear)
        self.assertEqual(len(self.cleared), 1)
        # Both the needs_clear buffer row and the queued donate row name the
        # same track row; the forward-path drain dedups the same way.
        self.assertEqual(sorted(self.cleared[0].tolist()), [track_row, track_row])

    def test_park_triggers_on_unaffordable_residue(self):
        # V2.5: rotation must engage when an admission cannot be priced,
        # not only when the pool reads empty — a sub-footprint residue
        # (1 free row vs a 2-row record) still starves the queue.
        r0 = self._tracked_req("p0")  # 2 rows
        r1 = _FakeReq("p1")
        r1.kv.req_pool_idx = 5
        r1.kv.mamba_pool_idx = self.slots.alloc(1)[0]  # leaves free == 1
        move([r1], self.running_reqs)
        batch = _FakeRunningBatch([r0, r1])
        self.assertEqual(self.slots.available_size(), 1)
        self.assertTrue(
            self.swap.wants_park(batch, 1, at_step=self.swap.predicted_tick_step)
        )
        self.swap.on_step_boundary(batch, waiting_count=1)
        self.assertEqual(len(self.parked), 1)

    def test_wake_first_fit_skips_a_too_wide_record(self):
        # M3 (ocr): with variable-width records the longest-parked record
        # may not fit the currently free rows; wake must reach past it to
        # the first record that fits instead of stalling.
        pool6 = MambaSlotAllocator(6, "cpu")
        big_pool = MambaSwapPool(
            mamba_pool=self.mamba_pool,
            mamba_allocator=pool6,
            swap_slots=3,
            device="cpu",
            rows_per_record=3,
        )
        running = KeyedReqList(ListKind.DECODING)
        parked = IndexedReqList(ListKind.MAMBA_PARKED)
        sched = MambaSwapScheduler(
            swap_pool=big_pool,
            state=SimpleNamespace(
                slots=pool6,
                pool=self.mamba_pool,
                pending_request_slots=self.swap.state.pending_request_slots,
            ),
            table=self.table,
            running_reqs=running,
            parked_reqs=parked,
            order_streams=(),
            wake_quantum=4,
        )
        wide, narrow = _FakeReq("wide"), _FakeReq("narrow")
        wide.kv.req_pool_idx, narrow.kv.req_pool_idx = 6, 7
        w_rows = pool6.alloc(3)
        wide.kv.mamba_pool_idx = w_rows[0]
        wide.kv.mamba_ping_pong_track_buffer = w_rows[1:]
        wide.kv.mamba_ping_pong_empty = [False, False]
        n_rows = pool6.alloc(2)
        narrow.kv.mamba_pool_idx = n_rows[0]
        narrow.kv.mamba_ping_pong_track_buffer = n_rows[1:]
        narrow.kv.mamba_ping_pong_empty = [False, True]
        move([wide, narrow], running)
        batch = _FakeRunningBatch([wide, narrow])
        sched._park(batch, wide)  # 3 rows
        sched._park(batch, narrow)  # 2 rows; all six rows free now
        held = pool6.alloc(4)  # an admit spends four rows, two stay free
        woke = sched.on_step_boundary(_FakeRunningBatch([]), waiting_count=0)
        self.assertEqual(woke, [narrow])  # wide (3) cannot fit the 2 free
        self.assertIn(wide, parked)
        self.assertIsNotNone(wide.kv.mamba_swap_id)
        # Release the admit's rows and the wide record wakes too.
        pool6.free(held)
        woke = sched.on_step_boundary(_FakeRunningBatch([]), waiting_count=0)
        self.assertEqual(woke, [wide])
        self.assertEqual(len(parked), 0)

    def test_cow_src_still_excludes_a_tracked_victim(self):
        # A CoW source remains a hard exclusion (transient clears are not).
        r0, r1 = self._tracked_req("e0"), self._tracked_req("e1")
        r0.kv.mamba_cow_src_index = torch.tensor([0])
        batch = _FakeRunningBatch([r0, r1])
        self.swap.on_step_boundary(batch, waiting_count=1)
        self.assertIs(next(iter(self.parked)), r1)

    def test_admission_slots_price_the_footprint(self):
        fresh = _FakeReq("q0")
        self.assertEqual(self.swap.admission_slots(fresh), 2)
        # A tracked request holds both rows already: nothing pending.
        held = self._tracked_req("q1")
        self.assertEqual(self.swap.admission_slots(held), 0)

    def test_wants_park_accepts_tracked_running_set(self):
        # V2.5 lifted v1's ping-pong exclusion: a fully tracked running set
        # is parkable, so the overlap handshake stays honest for it.
        r0, r1 = self._tracked_req("t0"), self._tracked_req("t1")
        batch = _FakeRunningBatch([r0, r1])
        self.assertTrue(
            self.swap.wants_park(batch, 1, at_step=self.swap.predicted_tick_step)
        )


if __name__ == "__main__":
    unittest.main()
