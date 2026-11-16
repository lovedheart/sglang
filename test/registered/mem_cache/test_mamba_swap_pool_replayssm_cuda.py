"""CUDA parity tests: park/wake round-trips full ReplaySSM state (V2-P0b).

A parked row must be *exactly* complete: besides temporal/conv, the ReplaySSM
ring planes are slot-indexed and a slot can be re-owned (and refilled by a
chunked prefill) while the row is parked, so they must ride the swap record.
Plain decode adds the slot-indexed ``replayssm_write_pos``; ``-spec`` keeps
its cursor triple in request rows, which park keeps, so it must come back
untouched while the slot rows come back bit-identical.

    python -m pytest test/registered/mem_cache/test_mamba_swap_pool_replayssm_cuda.py -v
"""

import unittest
from unittest import mock

import torch
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.mamba_swap_pool import MambaSwapPool
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")

SLOTS = 6
# -spec requires the cache len to be at least 16 and the draft tokens.
CACHE_LEN = 16
DRAFT_TOKENS = 4


def _cache_params():
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=2,
        num_heads=4,
        head_dim=16,
        state_size=16,
        conv_kernel=4,
    )
    dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.bfloat16)
    return Mamba2CacheParams(shape=shape, dtype=dtype, layers=[0, 1])


def _make_pool(*, spec, device="cuda"):
    return MambaPool(
        size=SLOTS,
        spec_state_size=SLOTS,
        cache_params=_cache_params(),
        mamba_layer_ids=[0, 1],
        device=device,
        enable_linear_replayssm=not spec,
        enable_linear_replayssm_spec=spec,
        linear_replayssm_cache_len=CACHE_LEN,
        speculative_num_draft_tokens=DRAFT_TOKENS if spec else None,
    )


def _ring_planes(cache):
    return [
        p
        for p in (
            cache.replayssm_d,
            cache.replayssm_k,
            cache.replayssm_g,
            cache.replayssm_rawv,
            cache.replayssm_rawk,
            cache.replayssm_beta,
        )
        if p is not None
    ]


def _slot_rows(pool):
    cache = pool.mamba_cache
    return [cache.temporal, *cache.conv, *_ring_planes(cache)]


def _fill_state(pool, seed):
    g = torch.Generator().manual_seed(seed)
    for p in _slot_rows(pool):
        flat = (torch.randn(tuple(p.shape), generator=g) * 4).to(
            dtype=p.dtype, device=p.device
        )
        p.copy_(flat)
    if pool.replayssm_write_pos is not None:
        pool.replayssm_write_pos.copy_(
            torch.randint(
                0, CACHE_LEN + 1, pool.replayssm_write_pos.shape, generator=g
            ).to(pool.replayssm_write_pos.device)
        )


class _Base(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        # Pageable host rows in CI: never cudaHostRegister on a shared GPU.
        patcher = mock.patch.object(
            MambaSwapPool,
            "_alloc_host",
            lambda self, shape, dtype: torch.zeros(shape, dtype=dtype, device="cpu"),
        )
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        torch.manual_seed(0)

    def _park_clobber_wake(self, pool):
        swap = MambaSwapPool(
            mamba_pool=pool,
            mamba_allocator=None,
            swap_slots=2,
            device="cuda",
            allow_replayssm=True,
        )
        before = [p[:, 1].clone() for p in _slot_rows(pool)]
        wpos = (
            pool.replayssm_write_pos[1].clone()
            if pool.replayssm_write_pos is not None
            else None
        )
        swap_id = swap.park(1)
        self.assertIsNotNone(swap_id)
        # A new owner reuses slot 1: clobber every slot-1 row and its cursor.
        for p in _slot_rows(pool):
            p[:, 1].fill_(-7.0)
        if pool.replayssm_write_pos is not None:
            pool.replayssm_write_pos[1] = 99
        swap.wake(swap_id, 3)
        torch.cuda.synchronize()
        return pool, before, wpos


class TestSwapPlainReplayssm(_Base):
    def test_park_wake_round_trips_ring_and_write_pos(self):
        pool = _make_pool(spec=False)
        _fill_state(pool, seed=1)
        pool, before, wpos = self._park_clobber_wake(pool)
        cache = pool.mamba_cache
        self.assertTrue(torch.equal(cache.temporal[:, 3], before[0]))
        for p_now, p_was in zip(_slot_rows(pool)[1:], before[1:]):
            self.assertTrue(torch.equal(p_now[:, 3], p_was))
        self.assertTrue(torch.equal(pool.replayssm_write_pos[3], wpos))
        self.assertIsNotNone(cache.replayssm_d)  # plain decode ring is live


class TestSwapSpecReplayssm(_Base):
    def test_park_wake_round_trips_compact_ring(self):
        pool = _make_pool(spec=True)
        _fill_state(pool, seed=2)
        pool, before, _ = self._park_clobber_wake(pool)
        self.assertTrue(torch.equal(pool.mamba_cache.temporal[:, 3], before[0]))
        for p_now, p_was in zip([p for p in _slot_rows(pool)[1:]], before[1:]):
            self.assertTrue(torch.equal(p_now[:, 3], p_was))

    def test_spec_cursors_are_row_keyed_and_survive_park(self):
        pool = _make_pool(spec=True)
        _fill_state(pool, seed=3)
        # The cursor triple is indexed by request row, not by slot.
        self.assertIsNotNone(pool.replayssm_spec_write_pos)
        pool.replayssm_spec_write_pos[2] = 3
        pool.replayssm_cache_base[2] = 1
        pool.replayssm_is_flush[2] = 1
        pool, _, _ = self._park_clobber_wake(pool)
        self.assertEqual(int(pool.replayssm_spec_write_pos[2]), 3)
        self.assertEqual(int(pool.replayssm_cache_base[2]), 1)
        self.assertEqual(int(pool.replayssm_is_flush[2]), 1)


class TestV1GatePreserved(_Base):
    def test_replayssm_still_rejected_without_spike_switch(self):
        pool = _make_pool(spec=False)
        with self.assertRaises(ValueError):
            MambaSwapPool(
                mamba_pool=pool, mamba_allocator=None, swap_slots=1, device="cuda"
            )


if __name__ == "__main__":
    unittest.main()
