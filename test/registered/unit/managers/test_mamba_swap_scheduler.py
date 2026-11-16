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
from sglang.srt.mem_cache.mamba_swap_pool import MambaSwapPool
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
        self.swap.on_step_boundary(self._running_batch(reqs), waiting_count=1)
        self.assertEqual(len(self.parked), 0)
        # The early-process handshake commits r0's step: it parks next tick.
        reqs[0].inflight_steps = 0
        self.swap.on_step_boundary(self._running_batch(reqs), waiting_count=1)
        self.assertEqual(len(self.parked), 1)
        self.assertIs(next(iter(self.parked)), reqs[0])

    def test_wants_park_tracks_the_park_branch(self):
        # The overlap loop calls wants_park to decide early result
        # processing; it must agree with the tick's park precondition.
        batch = self._running_batch([self._req("r0"), self._req("r1")])
        self.assertFalse(self.swap.wants_park(batch, waiting_count=0))
        self.assertTrue(self.swap.wants_park(batch, waiting_count=1))
        self.swap.on_step_boundary(batch, waiting_count=1)
        # A slot freed by the park: nothing more to park this tick.
        self.assertFalse(self.swap.wants_park(batch, waiting_count=1))

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
        self.swap_pool._parked[42] = True
        victim.kv.mamba_swap_id = 42
        other.kv.mamba_swap_id = 42
        with self.assertRaisesRegex(RuntimeError, "dup_record=42"):
            self.swap._check_parked_ownership()


if __name__ == "__main__":
    unittest.main()
