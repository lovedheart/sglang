"""CPU contracts for the QSA compressed-K HiCache sidecar.

Covers the three seams the feature touches:
  1. ``QSATokenToKVPool.qsa_hicache_regions`` page-group view math.
  2. ``DeepSeekV4PagedHostPool._to_page_indices`` composition with that view:
     a KV token index must land on the compressed-slot block of its page.
  3. ``_MambaStrategy`` wiring: sidecar spec registration + fail-loud guards.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _MambaStrategy
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PAGES = 6
PAGE = 8
RATIO = 4
LAYERS = 3
HEAD_DIM = 4
GROUPS_PER_PAGE = PAGE // RATIO
SLOTS = PAGES * PAGE
CAP = SLOTS // RATIO


def _make_pool():
    pool = object.__new__(QSATokenToKVPool)
    pool.qsa_compress_ratio = RATIO
    pool.qsa_index_kv_heads = 1
    pool.qsa_index_head_dim = HEAD_DIM
    pool.qsa_compressed_capacity = CAP
    flat = torch.zeros(LAYERS, CAP * HEAD_DIM, dtype=torch.bfloat16)
    # Compressed slot j of layer l encodes value 1000*l + j in every head.
    for lyr in range(LAYERS):
        flat[lyr] = torch.repeat_interleave(
            torch.arange(CAP, dtype=torch.bfloat16) + 1000 * lyr, HEAD_DIM
        )
    pool.qsa_compressed_flat = flat
    return pool


class TestQSAHiCacheRegions(CustomTestCase):
    def test_rows_map_to_page_contiguous_group_blocks(self):
        pool = _make_pool()
        buffers, row_bytes = pool.qsa_hicache_regions(page_size=PAGE)
        self.assertEqual(len(buffers), LAYERS)
        self.assertEqual(
            row_bytes, GROUPS_PER_PAGE * HEAD_DIM * torch.bfloat16.itemsize
        )
        self.assertEqual(tuple(buffers[0].shape), (SLOTS // PAGE, row_bytes))
        for lyr in range(LAYERS):
            rows = buffers[lyr].view(torch.bfloat16).view(-1, GROUPS_PER_PAGE, HEAD_DIM)
            for r in range(SLOTS // PAGE):
                expect = (
                    torch.arange(GROUPS_PER_PAGE, dtype=torch.float32)
                    + r * GROUPS_PER_PAGE
                    + 1000 * lyr
                )
                got = rows[r][:, 0].float()
                self.assertTrue(
                    torch.equal(got, expect.to(torch.bfloat16).float()),
                    msg=f"layer {lyr} page {r}: expected group ids {expect.tolist()}",
                )

    def test_rejects_page_not_multiple_of_ratio(self):
        pool = _make_pool()
        with self.assertRaises(ValueError):
            pool.qsa_hicache_regions(page_size=RATIO + 1)


class TestSidecarRowMath(CustomTestCase):
    def _host_pool_view(self):
        return SimpleNamespace(slot_page_size=PAGE)

    def test_kv_index_to_row_lands_on_group_block(self):
        pool = _make_pool()
        buffers, _ = pool.qsa_hicache_regions(page_size=PAGE)
        # The executor always passes full page-aligned token vectors.
        kv_indices = torch.arange(0, SLOTS)
        rows = DeepSeekV4PagedHostPool._to_page_indices(
            self._host_pool_view(), kv_indices
        )
        # Layer 0 values are group ids (bf16-exact); each page's row and each
        # token's intra-page offset must address its group block.
        values = buffers[0].view(torch.bfloat16).view(-1, GROUPS_PER_PAGE, HEAD_DIM)
        for p in range(SLOTS // PAGE):
            r = int(rows[p])
            for g in range(GROUPS_PER_PAGE):
                t = p * PAGE + g * RATIO
                self.assertEqual(int(values[r, g, 0]), t // RATIO)

    def test_backup_then_restore_into_different_pages(self):
        """Emulate the executor's index algebra end to end (pure torch)."""
        pool = _make_pool()
        buffers, _ = pool.qsa_hicache_regions(page_size=PAGE)
        dev = buffers[0].view(torch.bfloat16).view(-1, GROUPS_PER_PAGE, HEAD_DIM)
        host = torch.zeros(
            SLOTS // PAGE, GROUPS_PER_PAGE, HEAD_DIM, dtype=torch.bfloat16
        )
        # Pages 0 and 2 backed up to host pages 4 and 5.
        kv_indices = torch.cat(
            [torch.arange(0 * PAGE, 1 * PAGE), torch.arange(2 * PAGE, 3 * PAGE)]
        )
        host_indices = torch.cat(
            [torch.arange(4 * PAGE, 5 * PAGE), torch.arange(5 * PAGE, 6 * PAGE)]
        )
        src = DeepSeekV4PagedHostPool._to_page_indices(
            self._host_pool_view(), kv_indices
        )
        dst = DeepSeekV4PagedHostPool._to_page_indices(
            self._host_pool_view(), host_indices
        )
        host[dst] = dev[src]
        # Restore into device pages 3 and 5 (different slots than the backup).
        new_indices = torch.cat(
            [torch.arange(3 * PAGE, 4 * PAGE), torch.arange(5 * PAGE, 6 * PAGE)]
        )
        d2 = DeepSeekV4PagedHostPool._to_page_indices(
            self._host_pool_view(), new_indices
        )
        dev[d2] = host[dst]
        for new_page, old_page in zip((3, 5), (0, 2)):
            for g in range(GROUPS_PER_PAGE):
                slot = new_page * PAGE + g * RATIO
                self.assertEqual(
                    int(dev[slot // PAGE, slot % PAGE // RATIO, 0]),
                    old_page * GROUPS_PER_PAGE + g,
                )


class TestMambaStrategyWiring(CustomTestCase):
    def _kvcache(self):
        kv = object.__new__(QSATokenToKVPool)
        kv.qsa_compress_ratio = RATIO
        kv.full_attention_layer_id_mapping = {3: 0, 7: 1, 11: 2}
        kv.start_layer = 0
        kv.use_mla = False
        kv.full_kv_pool = MagicMock()
        kv.full_kv_pool.size = SLOTS
        return kv

    def _run_build(self, *, host_mode, page_size):
        params = SimpleNamespace(
            page_size=page_size,
            req_to_token_pool=SimpleNamespace(
                mamba_map={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 8: 6, 9: 7, 10: 8},
                mamba_pool=MagicMock(),
            ),
        )
        group = MagicMock()
        group.get_pool.side_effect = lambda name: f"host[{name}]"
        with (
            patch.object(
                assembler,
                "build_hybrid_mamba_stack",
                return_value=(group, MagicMock()),
            ) as built,
            patch.object(
                assembler,
                "get_memory",
                lambda: SimpleNamespace(hicache_host_memory_mode=host_mode),
            ),
        ):
            result = _MambaStrategy().build(
                cache=MagicMock(),
                kvcache=self._kvcache(),
                params=params,
                server_args=MagicMock(),
                load_cache_event=MagicMock(),
            )
        return result, built

    def test_qsa_pool_registers_kv_derived_sidecar(self):
        result, built = self._run_build(host_mode="cache", page_size=PAGE)
        self.assertIs(built.call_args.kwargs["qsa_kvcache"].__class__, QSATokenToKVPool)
        self.assertEqual(len(result.sidecars), 1)
        spec = result.sidecars[0]
        self.assertIs(spec.pool_name, PoolName.QSA_COMPRESSED_K)
        self.assertIs(spec.indices_from_pool, PoolName.KV)
        self.assertIn("QSA", result.pools_desc)

    def test_buffer_only_mode_fails_loud(self):
        with self.assertRaises(NotImplementedError):
            self._run_build(host_mode="buffer_only", page_size=PAGE)

    def test_page_ratio_mismatch_fails_loud(self):
        with self.assertRaises(ValueError):
            self._run_build(host_mode="cache", page_size=RATIO + 1)


if __name__ == "__main__":
    unittest.main()
