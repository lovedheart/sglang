"""File-backed host storage for the offloaded Qwen4-Exp PLE table.

CPU part: the allocator builds a sparse file of exactly the table's size, hands
back a tensor with the requested shape/dtype whose writes land in the file and
survive a re-open, reuses the file across calls, replaces one of the wrong size,
and the prefetcher computes the right page set and honours its size floor. The
resident-set trimmer measures only its own mapping, drops its pages once over
budget without losing what was written through them, and is off when the budget
is zero or the mapping is pinned.

The device check asks for cudaDevAttrPageableMemoryAccess, which GB10 (host
page tables) and a discrete GPU with HMM both report.

GPU part (skipped unless the device reads pageable host memory): the production
Triton gather kernel reading from the file-backed table matches a torch gather.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.models.qwen4_exp_ple_table import (
    PleFilePrefetcher,
    PleFileRssTrimmer,
    _mapping_rss_bytes,
    allocate_ple_host_table,
    check_file_backend_supported,
    default_ple_table_dir,
    device_reads_pageable_memory,
    make_ple_file_prefetcher,
    make_ple_file_rss_trimmer,
    ple_table_file_name,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")


class TestPleFileTableAllocator(CustomTestCase):
    def test_file_is_sparse_and_sized_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((1000, 160), torch.float8_e4m3fn, "file", d)
            path = os.path.join(
                d, ple_table_file_name((1000, 160), torch.float8_e4m3fn)
            )
            self.assertTrue(os.path.exists(path))
            self.assertEqual(os.path.getsize(path), 1000 * 160)
            self.assertEqual(tuple(table.shape), (1000, 160))
            self.assertEqual(table.dtype, torch.float8_e4m3fn)
            # Sparse: nothing written yet, so (almost) no blocks allocated.
            self.assertLess(os.stat(path).st_blocks * 512, 64 * 1024)

    def test_writes_persist_and_file_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            shape, dtype = (64, 32), torch.bfloat16
            table = allocate_ple_host_table(shape, dtype, "file", d)
            row = torch.arange(32, dtype=torch.float32).to(dtype)
            table[7].copy_(row)  # what the weight loader does, row by row
            del table
            again = allocate_ple_host_table(shape, dtype, "file", d)
            self.assertTrue(torch.equal(again[7].float(), row.float()))
            self.assertEqual(len(os.listdir(d)), 1)

    def test_wrong_sized_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ple_table_file_name((8, 8), torch.bfloat16))
            with open(path, "wb") as f:
                f.write(b"\x01" * 10)
            table = allocate_ple_host_table((8, 8), torch.bfloat16, "file", d)
            self.assertEqual(os.path.getsize(path), 8 * 8 * 2)
            self.assertEqual(tuple(table.shape), (8, 8))

    def test_tag_separates_tensor_parallel_shards(self):
        with tempfile.TemporaryDirectory() as d:
            a = allocate_ple_host_table(
                (8, 8), torch.bfloat16, "file", d, tag="rows0-8"
            )
            b = allocate_ple_host_table(
                (8, 8), torch.bfloat16, "file", d, tag="rows8-16"
            )
            a.fill_(1.0)
            self.assertEqual(len(os.listdir(d)), 2)
            self.assertTrue(torch.all(b.float() == 0.0))
            self.assertIn(
                "rows8-16", ple_table_file_name((8, 8), torch.bfloat16, "rows8-16")
            )

    def test_default_dir_is_per_checkpoint(self):
        with mock.patch.dict(os.environ, {"SGLANG_QWEN4_PLE_FILE_DIR": "/cache/ple"}):
            a = default_ple_table_dir("RadixArk/Qwen3.8-Flash-Next-NVFP4")
            b = default_ple_table_dir("/root/.cache/huggingface/flashnext-fp8/")
            self.assertEqual(a, "/cache/ple/RadixArk_Qwen3.8-Flash-Next-NVFP4")
            self.assertEqual(b, "/cache/ple/root_.cache_huggingface_flashnext-fp8")
            self.assertNotEqual(a, b)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            allocate_ple_host_table((4, 4), torch.bfloat16, "nvme", None)

    def test_pinned_backend_unchanged(self):
        if not torch.cuda.is_available():
            self.skipTest("pinned memory needs a CUDA runtime")
        table = allocate_ple_host_table((4, 4), torch.bfloat16, "pinned", None)
        self.assertTrue(table.is_pinned())
        self.assertIsNone(make_ple_file_prefetcher(table))

    def test_pinned_table_locks_exactly_its_size(self):
        # The caching host allocator would round 2.5 GiB up to 4 GiB.
        if not torch.cuda.is_available():
            self.skipTest("pinned memory needs a CUDA runtime")
        torch.zeros(1, device="cuda")
        shape, dtype = (5 * 2**27, 2), torch.bfloat16  # 2.5 GiB
        before = _process_rss_bytes()
        table = allocate_ple_host_table(shape, dtype, "pinned", None)
        grown = _process_rss_bytes() - before
        self.assertEqual(tuple(table.shape), shape)
        self.assertEqual(table.dtype, dtype)
        self.assertTrue(table.is_pinned())
        self.assertLess(grown, 3 * 2**30)
        self.assertGreater(grown, 2 * 2**30)

    def test_triton_gather_reads_pinned_table(self):
        # The production gather kernel dereferences the host pointer directly.
        if not torch.cuda.is_available():
            self.skipTest("pinned memory needs a CUDA runtime")
        import triton

        from sglang.srt.models.qwen4_exp import (
            _gather_ple_embedding_from_pinned_kernel,
        )

        rows, dim = 4096, 160
        table = allocate_ple_host_table((rows, dim), torch.bfloat16, "pinned", None)
        table.copy_(torch.randn(rows, dim).to(torch.bfloat16))
        ids = torch.randint(0, rows, (2048,), device="cuda")
        out = torch.empty(2048, dim, dtype=torch.bfloat16, device="cuda")
        _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
            table.data_ptr(),
            0,
            ids,
            out,
            embedding_dim=dim,
            tp_vocab_start=0,
            tp_vocab_end=rows,
            is_fp8=False,
            is_fp6=False,
            BLOCK_D=triton.next_power_of_2(dim),
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, table[ids.cpu()].to("cuda")))


def _process_rss_bytes() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS not found")


class TestPleFilePrefetcher(CustomTestCase):
    def test_page_set_covers_row_start_and_end(self):
        # 160-byte rows: row 25 spans bytes 4000-4159, i.e. pages 0 and 1.
        pages = PleFilePrefetcher.pages_for_rows(torch.tensor([25, 0]), 160)
        self.assertEqual(pages, [0, 1])
        pages = PleFilePrefetcher.pages_for_rows(torch.tensor([1000, 1000]), 160)
        self.assertEqual(pages, [39])  # dedup, single page

    def test_enqueue_uses_local_tp_offsets_and_ignores_other_shards(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.bin")
            with open(path, "wb") as f:
                f.truncate(8192)
            pf = PleFilePrefetcher(path, row_bytes=160, min_rows=1)
            try:
                with mock.patch("os.posix_fadvise") as fadvise:
                    self.assertFalse(
                        pf.enqueue(
                            torch.tensor([999, 1032]), vocab_start=1000, vocab_end=1032
                        )
                    )
                    self.assertTrue(
                        pf.enqueue(
                            torch.tensor([999, 1000, 1025, 1032]),
                            vocab_start=1000,
                            vocab_end=1032,
                        )
                    )
                    pf._pool.shutdown(wait=True)
                    self.assertEqual(
                        sorted(c.args[1] for c in fadvise.call_args_list), [0, 4096]
                    )
            finally:
                pf.close()

    def test_enqueue_respects_min_rows_and_advises_pages(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.bin")
            with open(path, "wb") as f:
                f.truncate(1 << 20)
            pf = PleFilePrefetcher(path, row_bytes=160, min_rows=4)
            try:
                self.assertFalse(pf.enqueue(torch.tensor([1, 2, 3])))
                with mock.patch("os.posix_fadvise") as fadvise:
                    self.assertTrue(pf.enqueue(torch.tensor([0, 1, 2, 30])))
                    pf._pool.shutdown(wait=True)
                    offsets = sorted(c.args[1] for c in fadvise.call_args_list)
                    # rows 0-2 live in page 0; row 30 (bytes 4800-4959) in page 1
                    self.assertEqual(offsets, [0, 4096])
            finally:
                pf.close()


class TestSmapsParsing(CustomTestCase):
    """The parser runs anywhere; only the live mapping needs Linux."""

    SMAPS = """00400000-00401000 r--p 00000000 08:01 1  /usr/bin/x
Size:                  4 kB
Rss:                   4 kB
7f0000000000-7f0004000000 rw-s 00000000 08:01 2  /cache/ple/table.bin
Size:              65536 kB
Rss:               32768 kB
7f0004000000-7f0008000000 rw-s 00000000 08:01 2  /cache/ple/table.bin
Size:              65536 kB
Rss:                1024 kB
7f0100000000-7f0100001000 rw-p 00000000 00:00 0
Size:                  4 kB
Rss:                   4 kB
"""

    def _rss(self, addr, nbytes):
        with tempfile.NamedTemporaryFile("w", suffix=".smaps", delete=False) as f:
            f.write(self.SMAPS)
            path = f.name
        try:
            return _mapping_rss_bytes(addr, nbytes, smaps_path=path)
        finally:
            os.unlink(path)

    def test_sums_every_vma_of_the_table_and_nothing_else(self):
        # The table spans both of its VMAs; the unrelated ones must not count.
        self.assertEqual(self._rss(0x7F0000000000, 0x8000000), (32768 + 1024) * 1024)

    def test_counts_a_partially_overlapping_vma(self):
        # A range ending inside the first VMA still needs that VMA's pages.
        self.assertEqual(self._rss(0x7F0000000000, 0x1000), 32768 * 1024)

    def test_ignores_unrelated_mappings(self):
        self.assertEqual(self._rss(0x7F0200000000, 0x1000), 0)

    def test_missing_smaps_is_reported_as_unknown(self):
        self.assertIsNone(
            _mapping_rss_bytes(0x1000, 0x1000, smaps_path="/nonexistent/smaps")
        )


class TestPleFileRssTrimmerConfig(CustomTestCase):
    def test_budget_zero_disables_the_trimmer(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((64, 32), torch.bfloat16, "file", d)
            with mock.patch.dict(
                os.environ, {"SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB": "0"}
            ):
                self.assertIsNone(make_ple_file_rss_trimmer(table))

    def test_pinned_table_has_no_trimmer(self):
        if not torch.cuda.is_available():
            self.skipTest("pinned memory needs a CUDA runtime")
        table = allocate_ple_host_table((4, 4), torch.bfloat16, "pinned", None)
        self.assertIsNone(make_ple_file_rss_trimmer(table))


@unittest.skipUnless(
    os.path.exists("/proc/self/smaps"),
    "the resident set of a mapping is only readable on Linux",
)
class TestPleFileRssTrimmer(CustomTestCase):
    # 64 MiB: large enough that the Rss of the mapping stands out, small
    # enough to write in a CPU test.
    SHAPE = (32768, 1024)
    NBYTES = 32768 * 1024 * 2

    def _trimmer(self, table, budget_bytes):
        return PleFileRssTrimmer(
            addr=table.data_ptr(),
            nbytes=self.NBYTES,
            budget_bytes=budget_bytes,
            interval_s=3600.0,
            chunk_bytes=16 << 20,  # several chunks, as in production
        )

    def test_measures_its_own_mapping_only(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table(self.SHAPE, torch.bfloat16, "file", d)
            trimmer = self._trimmer(table, 0)
            empty = trimmer.mapping_rss_bytes()
            table.fill_(1.0)  # touches every page
            touched = trimmer.mapping_rss_bytes()
            self.assertIsNotNone(touched)
            self.assertGreater(touched, empty)
            self.assertGreater(touched, self.NBYTES // 2)
            # Never the whole process: only the VMAs backing this table.
            self.assertLessEqual(touched, self.NBYTES + (16 << 20))

    def test_over_budget_drops_pages_and_keeps_the_data(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table(self.SHAPE, torch.bfloat16, "file", d)
            table.fill_(1.0)
            table[7][3] = 2.0
            trimmer = self._trimmer(table, budget_bytes=1 << 20)
            before = trimmer.mapping_rss_bytes()
            freed = trimmer.trim_once()
            after = trimmer.mapping_rss_bytes()
            self.assertGreater(freed, 0)
            self.assertLess(after, before // 2)
            # MADV_DONTNEED on a shared file mapping drops the page-table
            # entries, not the page cache: the writes are still there.
            self.assertEqual(table[7][3].item(), 2.0)
            self.assertEqual(table[9][3].item(), 1.0)

    def test_under_budget_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table(self.SHAPE, torch.bfloat16, "file", d)
            table.fill_(1.0)
            trimmer = self._trimmer(table, budget_bytes=self.NBYTES * 4)
            before = trimmer.mapping_rss_bytes()
            self.assertEqual(trimmer.trim_once(), 0)
            self.assertEqual(trimmer.mapping_rss_bytes(), before)

    def test_factory_starts_and_stops_a_thread(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((64, 32), torch.bfloat16, "file", d)
            with mock.patch.dict(
                os.environ,
                {
                    "SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB": "1",
                    "SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S": "3600",
                },
            ):
                trimmer = make_ple_file_rss_trimmer(table)
            self.assertIsNotNone(trimmer)
            try:
                self.assertTrue(trimmer._thread.is_alive())
            finally:
                trimmer.close()
                trimmer._thread.join(timeout=5)
            self.assertFalse(trimmer._thread.is_alive())


class _FakeCudart:
    """cudaDeviceGetAttribute answering from ``attributes`` (cudaDeviceAttr -> value)."""

    def __init__(self, attributes):
        self.attributes = attributes

    def cudaDeviceGetAttribute(self, value, attribute, device):
        value._obj.value = self.attributes[attribute.value]
        return 0


class TestFileBackendDeviceCheck(CustomTestCase):
    PAGEABLE, USES_HOST_PAGE_TABLES = 88, 100  # cudaDeviceAttr

    def _check(self, attributes):
        with (
            mock.patch("ctypes.util.find_library", return_value="libcudart.so"),
            mock.patch("ctypes.CDLL", return_value=_FakeCudart(attributes)),
        ):
            check_file_backend_supported(0)

    def test_pageable_access_accepted_with_or_without_host_page_tables(self):
        for uses_host_page_tables in (1, 0):  # GB10, HMM
            with self.subTest(uses_host_page_tables=uses_host_page_tables):
                self._check(
                    {
                        self.PAGEABLE: 1,
                        self.USES_HOST_PAGE_TABLES: uses_host_page_tables,
                    }
                )

    def test_no_pageable_access_refused(self):
        with self.assertRaisesRegex(ValueError, "--ple-offload-backend pinned"):
            self._check({self.PAGEABLE: 0, self.USES_HOST_PAGE_TABLES: 0})


@unittest.skipUnless(
    torch.cuda.is_available() and device_reads_pageable_memory(0) is True,
    "needs a device that reads pageable host memory",
)
class TestPleFileTableGatherOnDevice(CustomTestCase):
    def test_triton_gather_reads_file_backed_table(self):
        import triton

        from sglang.srt.models.qwen4_exp import (
            _gather_ple_embedding_from_pinned_kernel,
        )

        rows, dim = 4096, 160
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((rows, dim), torch.bfloat16, "file", d)
            table.copy_(torch.randn(rows, dim).to(torch.bfloat16))
            ids = torch.randint(0, rows, (2048,), device="cuda")
            out = torch.empty(2048, dim, dtype=torch.bfloat16, device="cuda")
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                table.data_ptr(),
                0,
                ids,
                out,
                embedding_dim=dim,
                tp_vocab_start=0,
                tp_vocab_end=rows,
                is_fp8=False,
                is_fp6=False,
                BLOCK_D=triton.next_power_of_2(dim),
            )
            torch.cuda.synchronize()
            expected = table[ids.cpu()].to("cuda")
            self.assertTrue(torch.equal(out, expected))


def _write_fake_ple_checkpoint(d, shards=3, rows=64, row_bytes=120, dtype="U8"):
    """Mini checkpoint: uint8 (packed fp6, + fp32 row scales) or F8_E4M3."""
    import json

    from safetensors.torch import save_file

    torch.manual_seed(0)
    weight_map = {}
    for i in range(shards):
        key = (
            f"model.language_model.layers.1.ple.ple_embedding"
            f".ngram_embedding.shard_{i}.weight"
        )
        tensors = {}
        if dtype == "U8":
            w = torch.randint(0, 256, (rows, row_bytes), dtype=torch.uint8)
            sc = torch.linspace(0.5, 2.0, rows, dtype=torch.float32).reshape(rows, 1)
            tensors[key[: -len(".weight")] + ".scale"] = sc
        elif dtype == "F8_E4M3":
            w = (
                torch.randn(rows, row_bytes)
                .clamp(-4, 4)
                .to(torch.float8_e4m3fn)
            )
        else:
            raise ValueError(dtype)
        tensors[key] = w
        fname = f"ple-{i:05d}.safetensors"
        save_file(tensors, os.path.join(d, fname))
        for k in tensors:
            weight_map[k] = fname
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


class TestPleSafetensorsAllocator(CustomTestCase):
    PREFIX = "model.layers.2.ple.ple_embedding.ngram_embedding"

    def _alloc(self, d, shards, rows, row_bytes=120):
        from sglang.srt.models.qwen4_exp_ple_table import (
            allocate_ple_safetensors_table,
        )

        return allocate_ple_safetensors_table(
            model_dir=d,
            prefix=self.PREFIX,
            dtype=torch.uint8,
            row_bytes=row_bytes,
            shard_size=rows,
            vocab_start=0,
            vocab_end=shards * rows,
        )

    def test_maps_checkpoint_shards_with_local_rows(self):
        from sglang.srt.models.qwen4_exp_ple_table import (
            allocate_ple_safetensors_table,
        )

        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(d, shards=3, rows=64)
            tbl = allocate_ple_safetensors_table(
                model_dir=d,
                prefix=self.PREFIX,
                dtype=torch.uint8,
                row_bytes=120,
                shard_size=64,
                vocab_start=0,
                vocab_end=192,
            )
            self.assertEqual(tbl.row_start, 0)
            self.assertEqual(tbl.row_end, 192)
            self.assertEqual(tbl.segment_bounds(), [0, 64, 128])
            self.assertEqual(tbl.total_bytes(), 3 * 64 * 120)

    def test_tp_window_maps_only_overlapping_shards(self):
        # TP rank covering rows [64, 128) touches shard 1 only, and its
        # segment rows restart at 0.
        from sglang.srt.models.qwen4_exp_ple_table import (
            allocate_ple_safetensors_table,
        )

        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(d, shards=3, rows=64)
            tbl = allocate_ple_safetensors_table(
                model_dir=d,
                prefix=self.PREFIX,
                dtype=torch.uint8,
                row_bytes=120,
                shard_size=64,
                vocab_start=64,
                vocab_end=128,
            )
            self.assertEqual(len(tbl.segments), 1)
            self.assertEqual((tbl.row_start, tbl.row_end), (0, 64))

    def test_mapped_bytes_equal_the_saved_tensors(self):
        import ctypes
        import json

        import numpy as np

        from sglang.srt.models.qwen4_exp_ple_table import (
            allocate_ple_safetensors_table,
        )

        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(d, shards=2, rows=32)
            tbl = allocate_ple_safetensors_table(
                model_dir=d,
                prefix=self.PREFIX,
                dtype=torch.uint8,
                row_bytes=120,
                shard_size=32,
                vocab_start=0,
                vocab_end=64,
            )
            with open(os.path.join(d, "model.safetensors.index.json")) as f:
                wmap = json.load(f)["weight_map"]
            key = (
                "model.language_model.layers.1.ple.ple_embedding"
                ".ngram_embedding.shard_1.weight"
            )
            import safetensors.numpy

            with safetensors.numpy.safe_open(
                os.path.join(d, wmap[key]), framework="np"
            ) as f:
                want = f.get_tensor(key)[5:20].reshape(-1)
            seg = tbl.segments[tbl._segment_of(37)]  # row 37 -> shard 1 local 5
            got = np.frombuffer(
                ctypes.string_at(seg.addr + 5 * 120, 15 * 120), np.uint8
            )
            np.testing.assert_array_equal(got, want)

    def test_stored_dtype_mismatch_is_refused(self):
        # bf16 is a supported table dtype, but not against a uint8 checkpoint:
        # reinterpreting the bytes would silently read garbage.
        from sglang.srt.models.qwen4_exp_ple_table import (
            allocate_ple_safetensors_table,
        )

        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(d, shards=1, rows=8)
            with self.assertRaisesRegex(ValueError, "stored as U8"):
                allocate_ple_safetensors_table(
                    model_dir=d,
                    prefix=self.PREFIX,
                    dtype=torch.bfloat16,
                    row_bytes=320,
                    shard_size=8,
                    vocab_start=0,
                    vocab_end=8,
                )

    def test_trimmer_drops_pages_of_every_segment(self):
        # madvise(2) wants page-aligned addresses: the segment's tensor base
        # carries an intra-page pad the kernel needs but madvise rejects with
        # EINVAL (the trim then silently stops on the first segment).
        import ctypes

        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(d, shards=3, rows=40000)
            tbl = self._alloc(d, shards=3, rows=40000)
            for seg in tbl.segments:
                ctypes.string_at(seg.buffer._addr, seg.buffer._nbytes)
            trimmer = PleFileRssTrimmer(
                addr=0,
                nbytes=0,
                segments=tbl,
                budget_bytes=1 << 20,
                interval_s=999,
            )
            try:
                self.assertGreater(trimmer.mapping_rss_bytes(), 1 << 20)
                self.assertGreater(trimmer.trim_once(), 0)
                self.assertLessEqual(trimmer.mapping_rss_bytes(), 1 << 20)
            finally:
                trimmer.close()

    def test_lru_trim_keeps_hot_segments(self):
        # Budget under the table size: drop-all would re-cold the warm cache
        # on a timer; LRU must evict only the coldest mappings (by the
        # gather's own row stream) and leave the touched segment resident.
        import ctypes

        from sglang.srt.models.qwen4_exp_ple_table import (
            _ranges_rss_bytes,
        )

        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(d, shards=4, rows=40000)
            tbl = self._alloc(d, shards=4, rows=40000)
            self.assertEqual(len(tbl.segments), 4)
            prefetcher = PleFilePrefetcher(
                row_bytes=120, segments=tbl, min_rows=1
            )
            for seg in tbl.segments:
                ctypes.string_at(seg.buffer._addr, seg.buffer._nbytes)
            # Only rows of the last segment count as hot.
            seg3 = tbl.segments[3]
            ids = torch.arange(seg3.row_start, seg3.row_start + 5000)
            self.assertTrue(prefetcher.enqueue(ids))
            # Budget: the hot segment plus one more mapping's worth, so the
            # trim must evict at least the two coldest to get under it.
            warm = sum(_ranges_rss_bytes(tbl.mappings()))
            trimmer = PleFileRssTrimmer(
                segments=tbl,
                prefetcher=prefetcher,
                budget_bytes=int(warm * 0.6),
                interval_s=999,
            )
            try:
                self.assertGreater(trimmer.trim_once(), 0)
                rss = _ranges_rss_bytes(tbl.mappings())
            finally:
                trimmer.close()
            self.assertLess(sum(rss), warm)
            self.assertGreater(rss[3], 1 << 20, "hot segment was evicted")

    def test_prefetcher_advises_the_covered_segment_pages(self):
        # The kernel crashed on launch until this path existed: the
        # prefetcher must resolve rows through the segment table, not the
        # single-file offsets the "file" backend uses.
        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(d, shards=3, rows=256)
            tbl = self._alloc(d, shards=3, rows=256)
            advised = []

            class Probe(PleFilePrefetcher):
                def _advise(self, pages):
                    advised.extend(pages)

            p = Probe.__new__(Probe)
            PleFilePrefetcher.__init__(
                p, path=None, row_bytes=120, min_rows=0, segments=tbl
            )
            p.enqueue(torch.tensor([0, 255, 256, 511, 512, 767, 999999]))
            p.close()
            self.assertTrue(advised)
            self.assertEqual(len({fd for fd, _ in advised}), 3)  # one per shard

    def test_fp8_checkpoint_maps_rows(self):
        # fp8 rows are embedding_dim 1-byte elements; scales do not exist.
        from sglang.srt.models.qwen4_exp_ple_table import (
            allocate_ple_safetensors_table,
        )

        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(
                d, shards=2, rows=32, row_bytes=160, dtype="F8_E4M3"
            )
            tbl = allocate_ple_safetensors_table(
                model_dir=d,
                prefix=self.PREFIX,
                dtype=torch.float8_e4m3fn,
                row_bytes=160,
                shard_size=32,
                vocab_start=0,
                vocab_end=64,
            )
            self.assertEqual((tbl.row_start, tbl.row_end), (0, 64))
            self.assertEqual(tbl.total_bytes(), 2 * 32 * 160)

    def test_indirection_tensors_pad_to_power_of_two(self):
        import torch

        from sglang.srt.models.qwen4_exp_ple_table import (
            allocate_ple_safetensors_table,
        )

        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(d, shards=3, rows=8)
            tbl = allocate_ple_safetensors_table(
                model_dir=d,
                prefix=self.PREFIX,
                dtype=torch.uint8,
                row_bytes=120,
                shard_size=8,
                vocab_start=0,
                vocab_end=24,
            )
            if not torch.cuda.is_available():
                self.skipTest("needs CUDA to build device tensors")
            starts, ends, bases, block = tbl.indirection_tensors("cuda")
            self.assertEqual(block, 4)  # padded from 3
            self.assertEqual(starts.tolist()[:3], [0, 8, 16])
            self.assertTrue((starts[3:] > 2**60).all())


@unittest.skipUnless(
    torch.cuda.is_available() and device_reads_pageable_memory(0) is True,
    "needs a device that reads pageable host memory",
)
class TestPleSafetensorsGatherOnDevice(CustomTestCase):
    def test_segment_gather_matches_direct_decode(self):
        import json

        import triton
        from safetensors import safe_open

        from sglang.srt.models.qwen4_exp import (
            _gather_ple_embedding_from_pinned_kernel,
        )
        from sglang.srt.models.qwen4_exp_ple_table import (
            allocate_ple_safetensors_table,
        )

        dim = 160
        row_bytes = (dim // 4) * 3
        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(d, shards=3, rows=256, row_bytes=row_bytes)
            tbl = allocate_ple_safetensors_table(
                model_dir=d,
                prefix="model.layers.2.ple.ple_embedding.ngram_embedding",
                dtype=torch.uint8,
                row_bytes=row_bytes,
                shard_size=256,
                vocab_start=0,
                vocab_end=768,
            )
            starts, ends, bases, block = tbl.indirection_tensors("cuda")
            scales = torch.rand(768, 1, dtype=torch.float32, device="cuda")
            picks = [tbl.row_start, 255, 256, 511, 512, tbl.row_end - 1]
            ids = torch.tensor(picks, device="cuda")
            out = torch.empty(len(picks), dim, dtype=torch.bfloat16, device="cuda")
            _gather_ple_embedding_from_pinned_kernel[(len(picks),)](
                0,
                scales.data_ptr(),
                ids,
                out,
                embedding_dim=dim,
                tp_vocab_start=0,
                tp_vocab_end=768,
                is_fp8=False,
                is_fp6=True,
                BLOCK_D=triton.next_power_of_2(dim),
                seg_bases=bases,
                seg_starts=starts,
                seg_ends=ends,
                SEG_COUNT=len(tbl.segments),
                SEG_BLOCK=block,
            )
            torch.cuda.synchronize()

            with open(os.path.join(d, "model.safetensors.index.json")) as f:
                wmap = json.load(f)["weight_map"]

            def golden(row):
                shard = row // 256
                key = (
                    f"model.language_model.layers.1.ple.ple_embedding"
                    f".ngram_embedding.shard_{shard}.weight"
                )
                with safe_open(os.path.join(d, wmap[key]), framework="pt") as f:
                    packed = f.get_tensor(key)[row % 256].to(torch.int32)
                u24 = packed.view(-1, 3) @ torch.tensor(
                    [1, 256, 65536], dtype=torch.int32
                )
                codes = torch.stack(
                    [(u24 >> (l * 6)) & 63 for l in range(4)], dim=1
                ).reshape(-1)
                m = codes & 31
                v = torch.where(
                    m < 16,
                    m.float() * 0.125,
                    torch.where(m < 24, 2 + (m - 16) * 0.25, 4 + (m - 24) * 0.5),
                )
                v = torch.where((codes & 32) != 0, -v, v)
                # fp32 decode, fp32 scale, cast last -- kernel order.
                return (v.cuda() * scales[row]).to(torch.bfloat16)

            for i, row in enumerate(picks):
                torch.testing.assert_close(
                    out[i], golden(row), atol=0, rtol=0
                )

    def test_fp8_segment_gather_matches_tensor_reads(self):
        import json

        import triton
        from safetensors import safe_open

        from sglang.srt.models.qwen4_exp import (
            _gather_ple_embedding_from_pinned_kernel,
        )
        from sglang.srt.models.qwen4_exp_ple_table import (
            allocate_ple_safetensors_table,
        )

        dim = 160
        with tempfile.TemporaryDirectory() as d:
            _write_fake_ple_checkpoint(
                d, shards=3, rows=256, row_bytes=dim, dtype="F8_E4M3"
            )
            tbl = allocate_ple_safetensors_table(
                model_dir=d,
                prefix="model.layers.2.ple.ple_embedding.ngram_embedding",
                dtype=torch.float8_e4m3fn,
                row_bytes=dim,
                shard_size=256,
                vocab_start=0,
                vocab_end=768,
            )
            starts, ends, bases, block = tbl.indirection_tensors("cuda")
            picks = [tbl.row_start, 255, 256, 511, 512, tbl.row_end - 1]
            ids = torch.tensor(picks, device="cuda")
            out = torch.empty(len(picks), dim, dtype=torch.bfloat16, device="cuda")
            _gather_ple_embedding_from_pinned_kernel[(len(picks),)](
                0,
                0,
                ids,
                out,
                embedding_dim=dim,
                tp_vocab_start=0,
                tp_vocab_end=768,
                is_fp8=True,
                is_fp6=False,
                BLOCK_D=triton.next_power_of_2(dim),
                seg_bases=bases,
                seg_starts=starts,
                seg_ends=ends,
                SEG_COUNT=len(tbl.segments),
                SEG_BLOCK=block,
            )
            torch.cuda.synchronize()
            with open(os.path.join(d, "model.safetensors.index.json")) as f:
                wmap = json.load(f)["weight_map"]
            for i, row in enumerate(picks):
                key = (
                    f"model.language_model.layers.1.ple.ple_embedding"
                    f".ngram_embedding.shard_{row // 256}.weight"
                )
                with safe_open(os.path.join(d, wmap[key]), framework="pt") as f:
                    want = f.get_tensor(key)[row % 256].to(torch.bfloat16).cuda()
                torch.testing.assert_close(out[i], want, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
