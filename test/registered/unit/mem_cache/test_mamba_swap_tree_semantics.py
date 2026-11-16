"""Tree-semantics fault tests for mamba state swap under radix ON (V2-P2).

Real UnifiedRadixCache + real int8 checkpoint pool (no tree mocks), per the
swap-v2 plan: a parked request must survive full tree eviction and wake
bit-exact, abort-while-parked must drop its record without a tree insert,
and a checkpoint release of a parked request must fail loudly instead of
reading the null active row.

    python -m pytest test/registered/unit/mem_cache/test_mamba_swap_tree_semantics.py -v
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.managers.mamba_swap_scheduler import MambaSwapScheduler
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.mamba_swap_pool import MambaSwapPool
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.mem_cache_utils import finish_req
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CFG = CacheConfig(
    components=(ComponentType.FULL, ComponentType.MAMBA),
    enable_int8_mamba_checkpoint=True,
    mamba_cache_size=2,
    kv_size=64,
    max_context_len=64,
    page_size=1,
)


class _FakeRunningBatch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.batch_is_full = False

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self, reqs_to_exclude=None):
        excl = set(reqs_to_exclude or [])
        self.reqs = [r for r in self.reqs if r not in excl]


class TestMambaSwapTreeSemantics(CustomTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(
            MambaSwapPool,
            "_alloc_host",
            lambda self, shape, dtype: torch.zeros(shape, dtype=dtype, device="cpu"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cache, self.allocator, self.pool = build_fixture(CFG)
        self.mamba_pool = self.pool.mamba_pool
        self.swap = MambaSwapPool(
            mamba_pool=self.mamba_pool,
            mamba_allocator=self.pool.mamba_allocator,
            swap_slots=2,
            device=get_device(),
        )
        self.running = KeyedReqList(ListKind.DECODING)
        self.parked = IndexedReqList(ListKind.MAMBA_PARKED)
        self.sched = MambaSwapScheduler(
            swap_pool=self.swap,
            state=SimpleNamespace(
                slots=self.pool.mamba_allocator, pool=self.mamba_pool
            ),
            table=self.pool,
            running_reqs=self.running,
            parked_reqs=self.parked,
            order_streams=(),
            wake_quantum=4,
        )
        # Production wiring: releases find the swap pool through the
        # allocator's state (kv_cache_configurator does this at boot).
        self.allocator.state.swap_pool = self.swap

    def _req(self, rid, tokens):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        acquire_rows(self.pool, self.pool.state, [req])
        req.output_ids = array("q", [1])
        req.kv.kv_committed_len = len(tokens)
        req.kv.kv_allocated_len = len(tokens)
        req.kv.cache_protected_len = 0
        req.kv.mamba_last_track_seqlen = len(tokens)
        req.lock = None
        req.extra_key = None
        req.last_node = self.cache.root_node_handle()
        return req

    def _tag(self, req, value):
        slot = int(req.kv.mamba_pool_idx.reshape(-1)[0])
        self.mamba_pool.mamba_cache.temporal[:, slot] = value
        for c in self.mamba_pool.mamba_cache.conv:
            c[:, slot] = -value
        return slot

    def _park_one(self, a, b):
        batch = _FakeRunningBatch([a, b])
        woken = self.sched.on_step_boundary(batch, waiting_count=1)
        self.assertEqual(woken, [])
        self.assertEqual(len(self.parked), 1)
        return next(iter(self.parked))

    def test_evict_all_between_park_and_wake_keeps_state_exact(self):
        # A donated int8 checkpoint in the tree + FULL eviction of every
        # cached state in between: the parked record is untouched by the
        # tree, and the wake is bit-exact on whatever slot it lands in.
        donor = self._req("donor", [9, 10, 11, 12])
        finish_req(self.cache, donor, donor.owned_kv_len())
        self.assertEqual(self.cache.mamba_evictable_size(), 1)

        a, b = self._req("a", [1, 2, 3, 4]), self._req("b", [5, 6, 7, 8])
        tag_a, tag_b = self._tag(a, 0.5), self._tag(b, 0.25)
        victim = self._park_one(a, b)

        ev = self.cache.evict(EvictParams(num_tokens=64, mamba_num=8))
        self.assertEqual(ev.mamba_num_evicted, 1)
        self.assertEqual(self.cache.mamba_evictable_size(), 0)

        # The freed row is re-owned and written before the wake lands.
        old_slot = tag_a if victim is a else tag_b
        self.mamba_pool.mamba_cache.temporal[:, old_slot] = -1.0
        for c in self.mamba_pool.mamba_cache.conv:
            c[:, old_slot] = 1.0

        self.sched.on_step_boundary(_FakeRunningBatch([]), waiting_count=0)
        self.assertEqual(len(self.parked), 0)
        self.assertIn(victim, self.running)
        new_slot = int(victim.kv.mamba_pool_idx.reshape(-1)[0])
        want = 0.5 if victim is a else 0.25
        self.assertTrue(
            (self.mamba_pool.mamba_cache.temporal[:, new_slot] == want).all()
        )
        for c in self.mamba_pool.mamba_cache.conv:
            self.assertTrue((c[:, new_slot] == -want).all())
        self.assertIsNone(victim.kv.mamba_swap_id)

    def test_parked_abort_drops_record_without_tree_insert(self):
        a, b = self._req("a", [1, 2, 3, 4]), self._req("b", [5, 6, 7, 8])
        victim = self._park_one(a, b)
        self.assertEqual(self.cache.mamba_evictable_size(), 0)

        victim.finish_outside_step(FINISH_ABORT())
        release_kv_cache(victim, self.cache, checkpoint=False)

        self.assertIsNone(victim.kv.mamba_swap_id)
        self.assertEqual(self.swap.num_parked(), 0)
        self.assertEqual(self.cache.mamba_evictable_size(), 0)  # no insert
        self.assertEqual(self.swap.available_size(), 2)  # record row recycled

    def test_checkpoint_release_of_parked_req_raises_loudly(self):
        a, b = self._req("a", [1, 2, 3, 4]), self._req("b", [5, 6, 7, 8])
        victim = self._park_one(a, b)
        victim.finish_outside_step(FINISH_ABORT())
        with self.assertRaisesRegex(RuntimeError, "cannot be checkpointed"):
            release_kv_cache(victim, self.cache, checkpoint=True)
        # The record survived the refused release: state is not lost.
        self.assertIsNotNone(victim.kv.mamba_swap_id)

    def test_cow_source_defers_park_to_a_later_tick(self):
        a, b = self._req("a", [1, 2, 3, 4]), self._req("b", [5, 6, 7, 8])
        # Both carry an unmaterialized CoW source: no parkable candidate
        # this tick (require-owned); the tree stays untouched, state intact.
        a.kv.mamba_cow_src_index = 1
        b.kv.mamba_cow_src_index = 1
        self.sched.on_step_boundary(_FakeRunningBatch([a, b]), waiting_count=1)
        self.assertEqual(len(self.parked), 0)
        # The source materializes (one step later): park proceeds.
        b.kv.mamba_cow_src_index = None
        self.sched.on_step_boundary(_FakeRunningBatch([a, b]), waiting_count=1)
        self.assertEqual(len(self.parked), 1)
        self.assertIs(next(iter(self.parked)), b)


if __name__ == "__main__":
    unittest.main()
