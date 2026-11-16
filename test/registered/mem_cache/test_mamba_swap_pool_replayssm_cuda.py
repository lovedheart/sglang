"""CUDA parity tests: park/wake round-trips full ReplaySSM state (V2-P0b).

A parked row must be *exactly* complete: besides temporal/conv, the
slot-indexed ReplaySSM ring planes and the plain-decode write cursor must
ride the swap record — the slot can be re-owned (and refilled by a chunked
prefill) while the row is parked. GDN ``-spec`` instead keeps its ring as
request-row scratch (memory_pool.py:808-815): those rows park with the
request, so the swap pool must NOT copy them by slot, and the cursor triple
must come back untouched with the request row.

    python -m pytest test/registered/mem_cache/test_mamba_swap_pool_replayssm_cuda.py -v
"""

import unittest
from unittest import mock

import torch
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
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
REQ_ROWS = 9  # distinct from SLOTS so row- vs slot-keying cannot coincide
# -spec requires the cache len to be at least 16 and the draft tokens.
CACHE_LEN = 16
DRAFT_TOKENS = 4


def _gdn_params():
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


def _kda_params():
    shape = KimiLinearStateShape.create(
        tp_world_size=1,
        num_heads=4,
        head_dim=16,
        num_k_heads=2,
        head_k_dim=16,
        conv_kernel_size=4,
        num_spec=DRAFT_TOKENS,
    )
    dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
    return KimiLinearCacheParams(shape=shape, dtype=dtype, layers=[0, 1])


def _make_pool(*, spec, kda=False, device="cuda"):
    return MambaPool(
        size=SLOTS,
        spec_state_size=REQ_ROWS,
        cache_params=_kda_params() if kda else _gdn_params(),
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
        p.copy_(
            (torch.randn(tuple(p.shape), generator=g) * 4).to(
                dtype=p.dtype, device=p.device
            )
        )
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

    def _park_clobber_wake(self, pool, rows=None):
        """Park slot 1, clobber it (as a new owner's prefill would), wake
        into slot 3, and return (swap, before, wpos) for the given rows
        (default: every slot-indexed row)."""
        swap = MambaSwapPool(
            mamba_pool=pool,
            mamba_allocator=None,
            swap_slots=2,
            device="cuda",
        )
        rows = _slot_rows(pool) if rows is None else rows
        before = [p[:, 1].clone() for p in rows]
        wpos = (
            pool.replayssm_write_pos[1].clone()
            if pool.replayssm_write_pos is not None
            else None
        )
        swap_id = swap.park(1)
        self.assertIsNotNone(swap_id)
        for p in _slot_rows(pool):
            p[:, 1].fill_(-7.0)
        if pool.replayssm_write_pos is not None:
            pool.replayssm_write_pos[1] = 99
        swap.wake(swap_id, 3)
        torch.cuda.synchronize()
        return swap, before, wpos


class TestSwapPlainReplayssm(_Base):
    def test_park_wake_round_trips_ring_and_write_pos(self):
        pool = _make_pool(spec=False)
        _fill_state(pool, seed=1)
        swap, before, wpos = self._park_clobber_wake(pool)
        self.assertFalse(pool.replayssm_request_ring)
        self.assertTrue(torch.equal(pool.mamba_cache.temporal[:, 3], before[0]))
        for p_now, p_was in zip(_slot_rows(pool)[1:], before[1:]):
            self.assertTrue(torch.equal(p_now[:, 3], p_was))
        self.assertTrue(torch.equal(pool.replayssm_write_pos[3], wpos))
        self.assertIsNotNone(pool.mamba_cache.replayssm_d)  # ring is live


class TestSwapKdaSpec(_Base):
    """KDA -spec (no host-only checkpoints) keeps a SLOT-KEYED ring."""

    def test_park_wake_round_trips_compact_ring(self):
        pool = _make_pool(spec=True, kda=True)
        _fill_state(pool, seed=2)
        self.assertFalse(pool.replayssm_request_ring)
        self.assertIsNotNone(pool.mamba_cache.replayssm_rawv)
        swap, before, _ = self._park_clobber_wake(pool)
        self.assertTrue(torch.equal(pool.mamba_cache.temporal[:, 3], before[0]))
        for p_now, p_was in zip(_slot_rows(pool)[1:], before[1:]):
            self.assertTrue(torch.equal(p_now[:, 3], p_was))


class TestSwapGdnSpecRequestRing(_Base):
    """GDN -spec keeps the ring as REQUEST-ROW scratch: it rides with the
    request row and must never be copied by slot index."""

    def test_request_ring_rows_are_not_touched_by_swap(self):
        pool = _make_pool(spec=True)
        _fill_state(pool, seed=3)
        self.assertTrue(pool.replayssm_request_ring)
        # A live request's scratch on ring ROW 2 (a request row, not a slot).
        for p in _ring_planes(pool.mamba_cache):
            p[0, 2].fill_(11.0)
        swap, before, _ = self._park_clobber_wake(
            pool, rows=[pool.mamba_cache.temporal, *pool.mamba_cache.conv]
        )
        self.assertEqual(swap._ring_planes, [])  # nothing captured by slot
        self.assertTrue(torch.equal(pool.mamba_cache.temporal[:, 3], before[0]))
        for p in _ring_planes(pool.mamba_cache):
            self.assertTrue(torch.equal(p[0, 2], torch.full_like(p[0, 2], 11.0)))


class TestSpecCursorsSurvive(_Base):
    def test_spec_cursors_are_row_keyed_and_survive_park(self):
        pool = _make_pool(spec=True)
        _fill_state(pool, seed=4)
        self.assertEqual(pool.replayssm_spec_write_pos.shape[0], REQ_ROWS + 1)
        pool.replayssm_spec_write_pos[2] = 3
        pool.replayssm_cache_base[2] = 1
        pool.replayssm_is_flush[2] = 1
        self._park_clobber_wake(
            pool, rows=[pool.mamba_cache.temporal, *pool.mamba_cache.conv]
        )
        self.assertEqual(int(pool.replayssm_spec_write_pos[2]), 3)
        self.assertEqual(int(pool.replayssm_cache_base[2]), 1)
        self.assertEqual(int(pool.replayssm_is_flush[2]), 1)


if __name__ == "__main__":
    unittest.main()
