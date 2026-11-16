"""Ownership accounting for mamba state-swap slots.

Swap slots must be released exactly once per parked request: the retire
funnel (`release_kv_cache`) drops the id, and the ledger invariant the
watchdog checks compares the pool's parked set with the parked list.

    python -m pytest test/registered/unit/mem_cache/test_mamba_swap_accounting.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.mamba_swap_pool import MambaSwapPool, _ParkedSlot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

L, H, V, K = 2, 4, 8, 8


def _swap_pool(swap_slots=2):
    temporal = torch.zeros(L, 4, H, V, K, dtype=torch.bfloat16)
    conv = [torch.zeros(L, 4, 2, 4, 8, dtype=torch.bfloat16)]
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
        replayssm_write_pos=None,
        _slot_siblings=[],
    )
    with mock.patch.object(
        MambaSwapPool,
        "_alloc_host",
        lambda self, shape, dtype: torch.zeros(shape, dtype=dtype, device="cpu"),
    ):
        return MambaSwapPool(
            mamba_pool=mamba_pool,
            mamba_allocator=None,
            swap_slots=swap_slots,
            device="cpu",
        )


class TestSwapReleaseExactlyOnce(unittest.TestCase):
    def _retire(self, req, swap):
        tree_cache = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                state=SimpleNamespace(swap_pool=swap)
            ),
            req_to_token_pool=None,
            supports_mamba=lambda: True,
        )
        release_kv_cache(req, tree_cache, checkpoint=False)

    def test_release_drops_parked_swap_id(self):
        swap = _swap_pool()
        swap_id = 1  # take the id out of circulation like park() would
        swap.free_ids = [2]
        swap._parked[swap_id] = _ParkedSlot(None, [])
        kv = ReqKvInfo(kv_allocated_len=0)
        kv.mamba_swap_id = swap_id
        req = SimpleNamespace(kv=kv, rid="r0")

        self._retire(req, swap)

        self.assertIsNone(kv.mamba_swap_id)
        self.assertEqual(swap.num_parked(), 0)
        self.assertEqual(swap.available_size(), 2)
        # A second retire of the same request must not touch the pool.
        self._retire(req, swap)
        self.assertEqual(swap.available_size(), 2)

    def test_release_of_unparked_req_is_untouched(self):
        swap = _swap_pool()
        swap_id = 1
        swap._parked[swap_id] = _ParkedSlot(None, [])
        kv = ReqKvInfo(kv_allocated_len=0)
        req = SimpleNamespace(kv=kv, rid="r1")

        self._retire(req, swap)

        self.assertEqual(swap.num_parked(), 1)
        self.assertTrue(swap.drop(1))


if __name__ == "__main__":
    unittest.main()
