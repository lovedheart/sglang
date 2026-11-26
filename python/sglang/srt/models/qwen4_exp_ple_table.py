"""Host-side storage for the offloaded Qwen4-Exp PLE n-gram table.

``--ple-offload-embedding`` keeps the PLE table (47.7 GiB in fp8 for
Qwen3.8-Flash-Next) out of device memory and lets the Triton gather kernel read
rows straight from a host pointer. Two backends provide that pointer:

``pinned`` (default)
    Page-locked host memory of exactly the table's size (``cudaHostAlloc``;
    ``torch.empty(..., pin_memory=True)`` on non-CUDA builds). On a discrete
    GPU this frees VRAM.

``file``
    A file-backed, shared ``mmap`` of a sparse file under
    ``--ple-offload-dir``. Meant for unified-memory parts (GB10 / DGX Spark and
    similar), where pinned host memory comes out of the *same* pool as the
    model weights and ``pinned`` therefore frees nothing: Qwen3.8-Flash-Next is
    126.0 GiB of weights on a 121.63 GiB box and does not boot with ``pinned``.
    The kernel dereferences the pageable pointer directly, which only works on
    devices that report ``cudaDevAttrPageableMemoryAccess`` (GB10 through the
    host page tables, a discrete GPU on the open kernel modules through HMM);
    rows are paged in from storage on demand, the file is sparse, deterministic
    in name and reused across restarts, and gathers of prefill size hint the
    page cache (``posix_fadvise(WILLNEED)``) so page faults are served
    concurrently instead of one at a time. A background trimmer keeps the
    mapping's resident set under a budget, because faulting rows in maps whole
    page-cache folios and the table would otherwise creep towards full
    residency (see ``PleFileRssTrimmer``).

This module has no Triton or CUDA-kernel imports so that its allocator and
prefetcher can be unit-tested on CPU.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Optional, Sequence

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LIBC: Optional[ctypes.CDLL] = None
_WILLNEED_FAILED = False
_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ")
_SMAPS_RSS = re.compile(r"^Rss:\s+(\d+) kB")

PLE_OFFLOAD_BACKENDS = ("pinned", "file", "safetensors")

# cudaDeviceAttr enum values (cuda_runtime_api.h).
_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS = 88
_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
_MADV_RANDOM = 1
_MADV_DONTNEED = 4
_PAGE_SHIFT = 12
# One MADV_DONTNEED call takes mmap_lock for its whole range; over the full
# 47.7 GiB table that is ~3.5 s during which every fault in the process --
# including the ones the gather kernel takes -- stalls. Trim in slices.
PLE_FILE_RSS_TRIM_CHUNK_BYTES = 1 << 30
# Below this many rows a gather is decode-sized (16 rows per token): the page
# faults are cheap and the host-side hint would cost more than it saves.
PLE_FILE_PREFETCH_MIN_ROWS = 2048
# Default unit of the resident-set trim and of the gather's recency tracking.
# The trimmer's own pacing sets the floor: a slice smaller than
# PLE_FILE_RSS_TRIM_CHUNK_BYTES means _evict_range madvises it in one call and
# then waits 5 ms, so every victim slice costs 5 ms whatever its size. Dropping
# the whole 47.7 GiB table down to an 8 GiB budget is then ~1 270 waits, about
# 6 s -- the first trim of a fully resident table; between checks only the
# overshoot goes, which is a handful of slices. A 2 MiB unit would make that
# same first trim ~20 300 waits, near two minutes, which is why this is not
# smaller; it sleeps between calls rather than holding mmap_lock, so the gather
# is not stalled, but the trim takes seconds to finish. Bigger than a working
# set, smaller than a shard mapping (hundreds of MiB), so one hot slice no
# longer keeps an entire cold shard resident.
PLE_FILE_RSS_RANGE_BYTES = 32 << 20
# Ceiling on the fraction of the *resident set* one trim cycle may drop. An
# eviction is only free until the workload reads the slice again, so the cost of
# a cycle is paid as a device fault per row by whatever reads it next -- a decode
# step reads 256 rows per verify, so a cycle that empties most of the resident
# set costs the next seconds of serving, not the 30 s until the next check. As a
# fraction of what is resident it still converges (geometrically) while a
# mapping is far over budget, and holds at the budget once there.
PLE_FILE_RSS_TRIM_MAX_DROP = 0.25


class PleSegment(NamedTuple):
    """One read-only mapping of a checkpoint shard's rows, in TP-local coordinates.

    ``row_start``/``row_end`` are rows of *this rank's* table (so the gather
    kernel indexes them with the same ``local_idx`` it already computes);
    ``addr`` is the host address of ``row_start`` and ``file_off`` its offset in
    ``fd``. Both are affine in the local row, which is what lets a multi-file
    table be read by one kernel launch: the kernel picks the segment's
    ``addr`` and keeps the uniform ``row * row_bytes`` stride it used before.
    """

    fd: int
    file_off: int
    addr: int
    nbytes: int
    row_start: int
    row_end: int
    buffer: object  # keeps the mmap alive


class PleRangeGrid(NamedTuple):
    """The resident-set trim's eviction grid: fixed-size slices of the mappings.

    A slice is what the trim drops at once and what the gather's recency is
    tracked per, so a hot slice keeps its rows while its colder neighbours go
    (a shard mapping is hundreds of MiB; the working set is tens of them).
    ``first[i]`` is where segment ``i``'s slices start, so binning a row is
    ``first[segment] + local_row // range_rows``, and ``owner[j]`` is the segment
    (mapping) slice ``j`` lives in, which is what the trim credits RSS against.
    """

    range_bytes: int  # nominal slice size; <= 0 = one slice per mapping
    range_rows: int  # rows per slice, what a row is divided by
    ranges: list  # (addr, nbytes) per slice, page-aligned inside its mapping
    first: list  # first slice of each segment
    owner: list  # segment index of each slice


class PleSegmentTable:
    """Shard-indirection descriptor for a table split across checkpoint files.

    The gather kernel cannot take 128 base pointers, so the segments are
    flattened into two tiny device tensors -- ascending row bounds and one
    adjusted base per segment -- and the row's segment becomes a lookup.
    """

    def __init__(self, segments: Sequence[PleSegment], row_bytes: int) -> None:
        segments = sorted(segments, key=lambda s: s.row_start)
        for prev, cur in zip(segments, segments[1:]):
            if cur.row_start < prev.row_end:
                raise ValueError("overlapping PLE table segments")
        self.segments = segments
        self.row_bytes = int(row_bytes)
        # Row reads (the host gather's staging copy) want a tensor per mapping, and
        # the split wants row bounds; all three are fixed at mapping time. Device
        # is pinned to CPU: the model is built under ``with torch.device(device)``
        # (model_loader/loader.py), which would otherwise put these on the GPU,
        # where they can neither index the host row views nor be read by the
        # host-side staging copy.
        self._bounds = torch.tensor(
            self.segment_bounds(), dtype=torch.int64, device="cpu"
        )
        self._starts = torch.tensor(
            [s.row_start for s in segments], dtype=torch.int64, device="cpu"
        )
        self._ends = torch.tensor(
            [s.row_end for s in segments], dtype=torch.int64, device="cpu"
        )
        self._row_views = self._build_row_views()

    @property
    def row_start(self) -> int:
        return self.segments[0].row_start if self.segments else 0

    @property
    def row_end(self) -> int:
        return self.segments[-1].row_end if self.segments else 0

    def segment_bounds(self) -> list[int]:
        """Row bound per segment, ascending; the kernel's segment index is the
        count of bounds <= ``local_idx``, minus one."""
        return [s.row_start for s in self.segments]

    def segment_bases(self) -> list[int]:
        """Host address of row 0 *within each segment*, so the kernel's row
        offset is relative: ``base + (local_idx - bound) * row_bytes``."""
        return [s.addr for s in self.segments]

    def _segment_runs(self, rows: torch.Tensor) -> list[tuple[int, int, int]]:
        """(segment index, first row, count) triples for ascending unique ``rows``.

        Ascending rows meet ascending, non-overlapping bounds, so one segment's
        rows are always one contiguous run, and the kernel's segment index -- the
        count of bounds at or below the row, minus one -- is also the run split.
        """
        if int(rows[0]) < self.row_start or int(rows[-1]) >= self.row_end:
            raise ValueError(
                f"PLE table: staged rows {int(rows[0])}..{int(rows[-1])} are "
                f"outside the mapped rows [{self.row_start}, {self.row_end})"
            )
        segment = torch.searchsorted(self._bounds, rows, right=True) - 1
        if bool((rows >= self._ends[segment]).any()):
            raise ValueError(
                "PLE table: staged rows fall in a gap between mapped segments "
                f"{[s.row_start for s in self.segments]}"
            )
        starts = torch.ones(rows.numel(), dtype=torch.bool, device=rows.device)
        starts[1:] = segment[1:] != segment[:-1]
        starts = torch.nonzero(starts).flatten()
        lengths = torch.diff(starts, append=starts.new_tensor([int(rows.numel())]))
        return list(zip(segment[starts].tolist(), starts.tolist(), lengths.tolist()))

    def read_rows(self, rows: torch.Tensor, out: torch.Tensor) -> int:
        """Copy the ascending unique int64 TP-local ``rows`` into ``out``'s rows.

        One ``index_select`` per segment rather than one per row: a staging copy
        of a prefill chunk reads tens of thousands of rows, and the segment's
        rows land in one contiguous slice of ``out``.
        """
        for index, start, count in self._segment_runs(rows):
            local = rows[start : start + count] - self.segments[index].row_start
            torch.index_select(
                self._row_views[index], 0, local, out=out[start : start + count]
            )
        return int(rows.numel())

    def range_grid(self, range_bytes: int) -> PleRangeGrid:
        """Slice every mapping into the trim's eviction units (see ``PleRangeGrid``).

        ``range_bytes <= 0`` yields the mappings themselves -- the pre-range LRU
        unit, so a whole checkpoint shard is still one victim. Otherwise slices are
        cut on row boundaries but widened out to whole pages, which ``madvise(2)``
        needs (an address inside a page is EINVAL); widening only ever pulls in a
        neighbour's page, never drops one of the slice's own rows.
        """
        mappings = self.mappings()
        if range_bytes <= 0:
            return PleRangeGrid(
                0,
                0,
                list(mappings),
                list(range(len(mappings))),
                list(range(len(mappings))),
            )
        page = 1 << _PAGE_SHIFT
        rows = max(1, range_bytes // self.row_bytes)
        out: list = []
        first: list = []
        owner: list = []
        for index, (base, size) in enumerate(mappings):
            segment = self.segments[index]
            rows_in_segment = segment.nbytes // self.row_bytes
            first.append(len(out))
            for lo in range(0, rows_in_segment, rows):
                aligned = max(base, (segment.addr + lo * self.row_bytes) & ~(page - 1))
                end = segment.addr + min(lo + rows, rows_in_segment) * self.row_bytes
                out.append(
                    (aligned, min(base + size, -(-end // page) * page) - aligned)
                )
                owner.append(index)
        return PleRangeGrid(range_bytes, rows, out, first, owner)

    def range_ids_for_rows(
        self, rows: torch.Tensor, grid: PleRangeGrid, first: torch.Tensor
    ) -> torch.Tensor:
        """Ascending slice ids (see ``range_grid``) holding any of the ``rows``.

        A row is binned by where its *first* byte falls, so the rare row whose tail
        crosses a slice boundary names only the first: that row's slice stays warm,
        and the neighbouring slice at worst loses one row's page. With the legacy
        whole-mapping unit the slice id is the segment id, so recency keeps working
        there too.
        """
        if rows.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device="cpu")
        rows = rows.to(torch.int64)
        segment = torch.searchsorted(self._bounds, rows, right=True) - 1
        keep = (
            (rows >= self.row_start)
            & (rows < self.row_end)
            & (rows < self._ends[segment])
        )
        rows, segment = rows[keep], segment[keep]
        if rows.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device="cpu")
        if grid.range_bytes <= 0:
            # One slice per mapping: the slice id is the segment id.
            return segment.unique()
        local = rows - self._starts[segment]
        return (first[segment] + local // grid.range_rows).unique()

    def advise_rows(self, rows: torch.Tensor) -> int:
        """Queue the page-cache reads of the mapped ``rows``, one ``posix_fadvise``
        per run of adjacent pages within each segment (see ``advise_rows_willneed``
        for why the runs matter).

        Rows outside this rank's window are skipped rather than refused: a gather's
        ids routinely leave the shard, and this path is advisory. A row inside the
        window but in no segment still raises -- that means the checkpoint is
        incomplete, which ``read_rows`` would misread as its neighbour's bytes.
        """
        rows = rows.to(torch.int64).unique()  # sorted, as _segment_runs wants
        rows = rows[(rows >= self.row_start) & (rows < self.row_end)]
        if rows.numel() == 0:
            return 0
        advised = 0
        for index, start, count in self._segment_runs(rows):
            segment = self.segments[index]
            local = rows[start : start + count] - segment.row_start
            advised += advise_rows_willneed(
                segment.fd, local, self.row_bytes, base_offset=segment.file_off
            )
        return advised

    def _build_row_views(self) -> list:
        """One uint8 ``[rows, row_bytes]`` tensor per segment, over its mapped bytes.

        torch has no ``from_address``, so the bridge is ctypes -> numpy -> torch and
        every hop shares the mapping. It is ``PROT_READ`` and numpy cannot see that,
        so it reports the result writable: nothing may write through these views.
        """
        import numpy as np

        views = []
        for segment in self.segments:
            rows = segment.nbytes // self.row_bytes
            memory = (ctypes.c_ubyte * segment.nbytes).from_address(segment.addr)
            array = np.frombuffer(memory, dtype=np.uint8)
            views.append(torch.from_numpy(array).view(rows, self.row_bytes))
        return views

    def mappings(self) -> list[tuple[int, int]]:
        """(addr, nbytes) per mapping, for madvise and RSS accounting.

        These are the mapping's own page-aligned bounds, not the segment's
        tensor base: ``addr`` carries the intra-page pad that lines the tensor
        up inside the mapping, and ``madvise(2)`` rejects any address with an
        offset inside a page (EINVAL).
        """
        out: list[tuple[int, int]] = []
        for s in self.segments:
            if getattr(s.buffer, "_addr", None) is not None:  # _MmapHolder
                out.append((s.buffer._addr, s.buffer._nbytes))
            else:
                out.append((s.addr, s.nbytes))
        return out

    def total_bytes(self) -> int:
        return sum(s.nbytes for s in self.segments)

    def indirection_tensors(self, device):
        """(starts, ends, bases, block) device tensors for the gather kernel.

        The kernel finds a row's segment by counting how many bounds are at or
        below it, so the arrays are padded to a power of two with a bound no
        row can reach.
        """
        import torch

        n = len(self.segments)
        block = 1
        while block < n:
            block *= 2
        far = 2**62
        starts = self.segment_bounds() + [far] * (block - n)
        ends = [s.row_end for s in self.segments] + [far] * (block - n)
        bases = self.segment_bases() + [0] * (block - n)
        return (
            torch.tensor(starts, dtype=torch.int64, device=device),
            torch.tensor(ends, dtype=torch.int64, device=device),
            torch.tensor(bases, dtype=torch.int64, device=device),
            block,
        )

    def advise(self, advice: int) -> None:
        for s in self.segments:
            try:
                os.posix_fadvise(s.fd, s.file_off, s.nbytes, advice)
            except OSError:
                pass


def _read_safetensors_header(path: str) -> tuple[dict, int]:
    """Parsed header and the offset where tensor data begins."""
    import struct

    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode("utf-8"))
    return header, 8 + hlen


_HEADER_CACHE: dict = {}


def _safetensors_header_cached(path: str) -> tuple[dict, int]:
    got = _HEADER_CACHE.get(path)
    if got is None:
        got = _read_safetensors_header(path)
        _HEADER_CACHE[path] = got
    return got


def find_safetensors_ple_shards(
    model_dir: str, key_filter
) -> list[tuple[int, str, int, int]]:
    """Locate PLE shard tensors in a checkpoint as
    ``(key_filter_result, path, data_start, tensor_begin)`` tuples.

    ``key_filter(name)`` maps a tensor name to a sortable key (usually the
    shard index) or returns None to skip. Reads
    ``model.safetensors.index.json`` (or scans the directory) plus the file
    headers; the offsets are what a read-only ``mmap`` needs to aim at the
    tensor's bytes without copying or repacking them.
    """
    index = os.path.join(model_dir, "model.safetensors.index.json")
    names_to_files: dict[str, str] = {}
    if os.path.exists(index):
        with open(index) as f:
            names_to_files = json.load(f)["weight_map"]
    else:
        for fname in sorted(os.listdir(model_dir)):
            if fname.endswith(".safetensors"):
                header, _ = _safetensors_header_cached(os.path.join(model_dir, fname))
                names_to_files.update({k: fname for k in header if k != "__metadata__"})
    out = []
    for name, fname in names_to_files.items():
        key = key_filter(name)
        if key is None:
            continue
        path = os.path.join(model_dir, fname)
        header, data_start = _safetensors_header_cached(path)
        begin, end = header[name]["data_offsets"]
        out.append((key, path, data_start, begin))
    out.sort(key=lambda t: t[0])
    return out


_PROT_READ = 1
_MAP_SHARED = 1


class _MmapHolder:
    """Keeps a raw mmap(2) region alive and unmaps it on release."""

    def __init__(self, libc, addr: int, nbytes: int):
        self._libc = libc
        self._addr = addr
        self._nbytes = nbytes

    def __del__(self):  # pragma: no cover - interpreter shutdown races
        try:
            self._libc.munmap(ctypes.c_void_p(self._addr), self._nbytes)
        except Exception:
            pass


def _mmap_readonly(path: str, offset: int, length: int):
    """Read-only ``MAP_SHARED`` mapping of ``[offset, offset+length)``.

    Returns ``(fd, mmap, addr)`` where ``addr`` is the host address of the
    tensor's first byte. ``mmap`` wants a page-aligned offset, so the mapping
    starts below ``offset`` and ``addr`` adds the remainder back.
    """
    page = 1 << _PAGE_SHIFT
    aligned = offset & ~(page - 1)
    pad = offset - aligned
    fd = os.open(path, os.O_RDONLY)
    try:
        # Raw mmap(2): Python's mmap object only reveals its address through
        # from_buffer()/buffer_info(), neither of which works on a PROT_READ
        # mapping on this build. munmap happens when the holder dies.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
        ]
        addr = libc.mmap(None, pad + length, _PROT_READ, _MAP_SHARED, fd, aligned)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap", path)
    except Exception:
        os.close(fd)
        raise
    buf = _MmapHolder(libc, addr, pad + length)
    return fd, buf, addr + pad


class PleFilePrefetcher:
    """Hint the page cache about the rows a prefill-sized gather is about to read.

    With the table on storage, a cold prefill chunk faults tens of thousands of
    4 KiB pages one at a time from inside the gather kernel. Advising them
    first (``posix_fadvise(WILLNEED)`` over each run of adjacent pages, on one
    background thread) lets the block layer serve them concurrently. Measured on
    a GB10 / NVMe: cold prefill 650-750 tok/s -> 1,000-2,100 tok/s (warm:
    ~2,200-2,600). Decode-sized gathers are skipped; nothing runs during
    CUDA-graph capture.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        row_bytes: int = 0,
        min_rows: int = PLE_FILE_PREFETCH_MIN_ROWS,
        segments: Optional[PleSegmentTable] = None,
        range_bytes: int = PLE_FILE_RSS_RANGE_BYTES,
    ) -> None:
        # A checkpoint-backed table spans many files; then PleSegmentTable maps
        # a row to (fd, file offset) and the page math happens there.
        self._segments = segments
        if segments is None:
            self._fd = os.open(path, os.O_RDONLY)
        self._row_bytes = int(row_bytes)
        self._min_rows = int(min_rows)
        self._pool = ThreadPoolExecutor(max_workers=1)
        # The grid both records recency against and (through ``grid``) that the
        # resident-set trim evicts by, so the two cannot disagree about which
        # slice a row fell in. Monotonic counters, one per slice, written only
        # from the gather thread and read by the trim thread; int stores are
        # atomic under the GIL and a racy cycle is harmless.
        self.grid = segments.range_grid(range_bytes) if segments else None
        self._hit_cycle = [0] * (len(self.grid.ranges) if self.grid else 0)
        self._slice_first = (
            torch.tensor(self.grid.first, dtype=torch.int64, device="cpu")
            if self.grid
            else None
        )
        self._cycle = 0

    def _advise(self, rows: torch.Tensor) -> None:
        """Queue the reads of TP-local ``rows``, one ``posix_fadvise`` per run of
        adjacent pages of one file.

        A prefill chunk touches a few thousand pages and rows are ~34 per page
        (120 B each), so adjacent ids -- the common case inside a batch -- would
        otherwise each get their own 4 KiB read issued one at a time. The row
        stream goes to ``advise_rows_willneed``, the same run merging the host
        gather uses for its own copy; for a checkpoint-backed table each segment
        has its own descriptor.
        """
        if self._segments is not None:
            self._segments.advise_rows(rows)
        else:
            advise_rows_willneed(self._fd, rows, self._row_bytes)

    def enqueue(
        self,
        flat_ids: torch.Tensor,
        *,
        vocab_start: int = 0,
        vocab_end: Optional[int] = None,
    ) -> bool:
        """Queue the hint for ``flat_ids``. Returns whether anything was queued."""
        if flat_ids.numel() < self._min_rows:
            return False
        if flat_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            return False
        # The .cpu() syncs the stream; acceptable for prefill chunks (~1 s) and
        # it is what lets the page set be computed without touching the kernel.
        row_ids = flat_ids.detach().cpu()
        if vocab_end is not None:
            # The file contains only this rank's vocabulary shard.
            row_ids = row_ids[(row_ids >= vocab_start) & (row_ids < vocab_end)]
        row_ids = row_ids - vocab_start
        if row_ids.numel() == 0:
            return False
        if self._segments is not None:
            self.note_rows(row_ids)
        self._pool.submit(self._advise, row_ids)
        return True

    def note_rows(self, rows: torch.Tensor) -> None:
        """Record that the gather touched these TP-local rows.

        The trim can only rank what the reads tell it about, and a gather that
        stages its rows on the host never reaches ``enqueue`` -- so the model
        calls this from that path too. Without it the resident set would freeze
        at whatever was warm when staging switched on.
        """
        if not self._hit_cycle:
            return
        hits = self._segments.range_ids_for_rows(rows, self.grid, self._slice_first)
        if hits.numel() == 0:
            return
        self._cycle += 1
        for index in hits.tolist():
            self._hit_cycle[index] = self._cycle

    def hit_cycles(self) -> list:
        """Per-slice last-touch counter (0 = never seen by a gather).

        Indexed by ``PleRangeGrid.ranges`` position: the trim may not build its
        own grid, or the two would rank different bytes.
        """
        return self._hit_cycle

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        if self._segments is None:
            try:
                os.close(self._fd)
            except OSError:
                pass


class PleFileRssTrimmer:
    """Keep the mapped table's resident set under a budget.

    Every random row fault maps in a whole page-cache folio, so with large
    folios (Linux 6.x) the mapping's Rss climbs towards the table's full size
    while a generated token only reads a few KB of it: measured ~45 KB of Rss
    growth per token on a GB10. On a unified-memory part that is not a slow
    leak, it is a countdown -- the free-memory readings that size the KV pool
    come from the same pool the folios are accumulating in.

    ``MADV_RANDOM`` does not prevent it (it limits readahead I/O, not the
    mapping-in of folios already in cache) and ``posix_fadvise(DONTNEED)`` does
    not release them either. ``MADV_DONTNEED`` over the mapping does: the page
    table entries go, the pages stay in the page cache, and hot rows come back
    at minor-fault cost.

    Dropping entries under a running gather is the state this backend already
    handles: the file starts out entirely unfaulted and every cold row is
    faulted in from inside the kernel through the same host page tables. What
    must not happen is one ``madvise`` call over the whole table, so the trim
    is chunked (see ``PLE_FILE_RSS_TRIM_CHUNK_BYTES``) and runs on its own
    daemon thread -- decode replays a CUDA graph and executes no Python, so a
    hook in the gather would never fire in the phase that grows the table.
    """

    def __init__(
        self,
        addr: int = 0,
        nbytes: int = 0,
        budget_bytes: int = 0,
        interval_s: float = 30.0,
        chunk_bytes: int = PLE_FILE_RSS_TRIM_CHUNK_BYTES,
        segments: Optional[PleSegmentTable] = None,
        prefetcher: Optional[PleFilePrefetcher] = None,
        lru: bool = True,
    ) -> None:
        # A checkpoint-backed table is many small mappings; the trim logic is
        # the same per mapping, so the ranges are kept as a list either way.
        self._ranges = (
            segments.mappings() if segments is not None else [(int(addr), int(nbytes))]
        )
        self._addr = int(addr)
        self._nbytes = int(nbytes) if segments is None else segments.total_bytes()
        self._budget = int(budget_bytes)
        self._interval = float(interval_s)
        self._chunk = int(chunk_bytes)
        # LRU mode needs the gather's per-slice recency, which only the row
        # stream knows; a single-mapping file table has no grid, so it keeps
        # drop-all.
        self._lru = bool(lru) and prefetcher is not None and prefetcher.grid is not None
        self._prefetcher = prefetcher
        # One report per trimmer: a budget the workload keeps exceeding is a
        # configuration fact, not a per-cycle event.
        self._held_warned = False
        # Slices this trim has already dropped, and the resident bytes it saw
        # per mapping last time. Resident bytes are only readable per mapping, so
        # without them the coldest-first sweep re-picks the same already-evicted
        # low-index slices: the nominal credit says there is more to free, the
        # madvise is a no-op, and the resident set plateaus above the budget. A
        # mapping whose residency grew had rows read back, so its slices are
        # candidates again.
        self._evicted: set = set()
        # Slices already evicted once that were then read back: the workload has
        # proved they are hot, whatever the row stream says. A captured decode
        # gather runs no Python, so decode never reaches ``note_rows`` and its
        # rows look cold forever; re-read-after-eviction is the one signal the
        # trim thread itself can see, and without it the trim and the workload
        # churn against each other at the cost of a device fault per row read.
        self._proven_hot: set = set()
        self._slices_of: list = []
        self._rss_seen: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="ple-file-rss-trim", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def mapping_rss_bytes(self) -> Optional[int]:
        """Resident bytes of the VMAs backing the table, or None off Linux."""
        totals = _ranges_rss_bytes(self._ranges)
        return None if totals is None else sum(totals)

    def _evict_range(self, base: int, nbytes: int) -> None:
        for offset in range(0, nbytes, self._chunk):
            if self._stop.is_set():
                return
            length = min(self._chunk, nbytes - offset)
            if not _madvise(base + offset, length, _MADV_DONTNEED):
                return
            # Let the faults that queued behind mmap_lock through.
            self._stop.wait(0.005)

    def _trim_lru(self, per_rss) -> int:
        """Evict grid slices, coldest first, until the resident set meets the budget.

        Recency is per slice (from the gather's row stream, see ``PleRangeGrid``),
        so a hot slice inside an otherwise cold shard survives: the whole-mapping
        unit this replaces re-colded hundreds of MiB of rows a gather had just
        read. Resident bytes are only readable per mapping (smaps), so an evicted
        slice is credited against its own mapping's residency and capped by it --
        a slice no gather ever reached costs a no-op madvise, not phantom freed
        bytes. Slices touched since the trim started are left alone, so at most
        the cold part of the table goes per cycle and the warm set migrates
        instead of collapsing.

        Victims are ranked by evidence, coldest first: a recorded touch that has
        aged out, then a slice no gather has been seen to read, then one this trim
        evicted and the workload read back. "No recorded touch" is not "cold" --
        the row stream sees only the gathers that run Python, while startup reads
        every segment through the warm-up ``posix_fadvise`` and a captured decode
        gather runs no Python at all -- so ranking unobserved slices first dropped
        what decode re-reads seconds later, at a device fault per row (~9 us). A
        slice already evicted and not read back is skipped entirely: it is already
        absent, and madvising it again frees nothing while the sweep never reaches
        the slices that are resident.
        """
        grid = self._prefetcher.grid
        hits = self._prefetcher.hit_cycles()
        cycle = self._prefetcher._cycle
        if not self._slices_of:
            self._slices_of = [[] for _ in self._ranges]
            for index, mapping in enumerate(grid.owner):
                self._slices_of[mapping].append(index)
        if len(self._rss_seen) == len(per_rss):
            for mapping, (now, seen) in enumerate(zip(per_rss, self._rss_seen)):
                if now > seen:  # rows of this mapping were read back
                    for index in self._slices_of[mapping]:
                        if index in self._evicted:
                            self._proven_hot.add(index)
                    self._evicted.difference_update(self._slices_of[mapping])
        left = list(per_rss)  # per mapping; a slice cannot credit more than it holds
        total = sum(left)
        victims: list[int] = []
        freed = 0
        # Coldest by evidence: an aged-out recorded touch, then a slice nobody
        # has been seen to read, then one evicted and read back (hot, but the
        # budget stays a hard bound). A slice evicted and not read back is out of
        # the running: it is already absent, so madvising it again frees nothing
        # and the sweep would never reach the slices that are resident.
        # Coldest by evidence: an aged-out recorded touch, then a slice nobody has
        # been seen to read. Two classes are out of the running. One evicted and
        # not read back is already absent, so madvising it again frees nothing and
        # the sweep never reaches the slices that are resident. One evicted and
        # then read back has been proved in use -- by a reader the row stream
        # cannot see, which is the only way a slice can be hot without a recorded
        # touch -- and re-evicting it starts a fight with that reader: the trim
        # drops it, the workload faults it back, at a device fault per row. What
        # is then left over budget has to be reported, not thrashed over; a leak
        # grows by touching slices nobody recorded, and those stay evictable.
        order = sorted(
            (
                i
                for i in range(len(grid.ranges))
                if i not in self._evicted and i not in self._proven_hot
            ),
            key=lambda i: (hits[i] == 0, min(hits[i], cycle), i),
        )
        cap = int(total * PLE_FILE_RSS_TRIM_MAX_DROP)
        for index in order:
            if total <= self._budget or freed >= cap or self._stop.is_set():
                break
            if hits[index] and hits[index] >= cycle:
                continue  # touched since the snapshot
            mapping = grid.owner[index]
            if left[mapping] == 0:
                continue
            # Credit the capped amount: a slice can only free what its mapping
            # actually has resident, and the same bound is what the log reports.
            freed_here = min(grid.ranges[index][1], left[mapping])
            left[mapping] -= freed_here
            total -= freed_here
            freed += freed_here
            victims.append(index)
        # A cycle that filled its cap and still sits well over budget is the
        # case where the trimmer cannot converge without re-colding the hot
        # set: either the budget is under the working set or the reads come
        # from a phase the row stream does not see (a captured decode gather
        # runs no Python, so its rows never reach note_rows).
        stuck = total > self._budget * 1.5 and (freed >= cap or not victims)
        if stuck and not self._held_warned:
            self._held_warned = True
            logger.warning(
                "PLE table: %.1f GiB resident against a %.1f GiB budget with "
                "%.0f%% of the table proved in use by reads it cannot observe "
                "(a captured decode gather records no recency). The budget is "
                "below the working set: raise "
                "SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB to bound it harder.",
                total / 2**30,
                self._budget / 2**30,
                100 * len(self._proven_hot) / max(len(grid.ranges), 1),
            )
        for index in victims:
            base, nbytes = grid.ranges[index]
            self._evict_range(base, nbytes)
            self._evicted.add(index)
        # What residency should now be: the next cycle compares against this, so
        # bytes above it were read back by something the row stream never saw.
        self._rss_seen = list(left)
        if freed:
            logger.info(
                "PLE table: LRU-trimmed %d cold slices (~%.1f GiB, budget %.1f GiB)",
                len(victims),
                freed / 2**30,
                self._budget / 2**30,
            )
        return freed

    def trim_once(self) -> int:
        """Cut the mapping's resident set to the budget. Returns bytes freed.

        LRU mode (a segment table with a live gather stream) evicts the coldest
        grid slices only; otherwise every mapping is dropped whole -- the
        pre-LRU behaviour, still right for single-mapping file tables where
        recency has nothing to rank.
        """
        if self._lru:
            per = _ranges_rss_bytes(self._ranges)
            if per is None:
                return 0
            before = sum(per)
            if before <= self._budget:
                return 0
            freed = self._trim_lru(per)
        else:
            before = self.mapping_rss_bytes()
            if before is None or before <= self._budget:
                return 0
            for base, nbytes in self._ranges:
                self._evict_range(base, nbytes)
            freed = before - (self.mapping_rss_bytes() or 0)
            if freed:
                logger.info(
                    "PLE table: trimmed resident set %.1f -> %.1f GiB "
                    "(budget %.1f GiB)",
                    before / 2**30,
                    max(before - freed, 0) / 2**30,
                    self._budget / 2**30,
                )
        return max(freed, 0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.trim_once()
            except Exception as exc:  # advisory only; never fail a request
                logger.warning("PLE table: resident-set trim skipped (%s)", exc)

    def close(self) -> None:
        self._stop.set()


def allocate_ple_host_table(
    shape: Sequence[int],
    dtype: torch.dtype,
    backend: str = "pinned",
    table_dir: Optional[str] = None,
    tag: Optional[str] = None,
) -> torch.Tensor:
    """Return a host tensor of ``shape``/``dtype`` for the PLE table.

    For the file backend, ``table_dir`` should be private to one checkpoint
    (the server defaults it to ``$SGLANG_CACHE_DIR/ple/<model path>``): the
    file name only encodes shape, dtype and ``tag``, and every boot rewrites
    the whole table through the weight loader.
    """
    if backend not in PLE_OFFLOAD_BACKENDS:
        raise ValueError(
            f"unknown PLE offload backend {backend!r}; choose from {PLE_OFFLOAD_BACKENDS}"
        )
    if backend == "pinned":
        if torch.version.cuda is not None:
            return _alloc_cuda_host_table(tuple(shape), dtype)
        return torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=True)

    numel = 1
    for d in shape:
        numel *= int(d)
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    table_dir = os.path.expanduser(table_dir or envs.SGLANG_QWEN4_PLE_FILE_DIR.get())
    os.makedirs(table_dir, exist_ok=True)
    path = os.path.join(table_dir, ple_table_file_name(shape, dtype, tag))
    if not os.path.exists(path) or os.path.getsize(path) != nbytes:
        # Sparse: only pages that get written take disk space.
        with open(path, "wb") as f:
            f.truncate(nbytes)
    logger.info(
        "PLE table: file-backed mmap %s (%.1f GiB, %s)", path, nbytes / 2**30, dtype
    )
    storage = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    _madvise_random(storage, nbytes)
    table = storage.view(dtype).view(*[int(d) for d in shape])
    table._sglang_ple_file_path = path  # consumed by PleFilePrefetcher
    return table


def _ple_shard_key_filter(prefix: str):
    """``key_filter`` for find_safetensors_ple_shards: this embedding's uint8
    ``shard_<i>.weight`` tensors -> shard index, anything else -> None.

    The prefix is matched loosely (everything after ``.layers.<n>.``, with the
    layer index wildcarded) because the checkpoint renumbers PLE layers and
    the loader rewrites ``model.language_model.`` to ``model.``; a checkpoint
    with two PLE layers makes both filters match and the segment overlap
    check below fails loudly.
    """
    tail = prefix.split(".layers.", 1)[-1] or prefix
    head = ""
    if tail[:1].isdigit():
        head = r"\d+\."
        tail = tail.split(".", 1)[-1]
    pattern = re.compile(head + re.escape(tail) + r"\.shard_(\d+)\.weight$")

    def key_filter(name: str):
        if ".ngram_embedding.shard_" not in name:
            return None
        match = pattern.search(name)
        return int(match.group(1)) if match else None

    return key_filter


# torch dtype -> safetensors header dtype string for the table tensors.
_ST_DTYPE = {
    torch.uint8: "U8",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.bfloat16: "BF16",
}


def allocate_ple_safetensors_table(
    model_dir: str,
    prefix: str,
    dtype: torch.dtype,
    row_bytes: int,
    shard_size: int,
    vocab_start: int,
    vocab_end: int,
) -> PleSegmentTable:
    """Map this rank's rows of the checkpoint's own PLE tensors, read-only.

    The checkpoint stores the table as ``<prefix>.shard_<i>.weight`` tensors
    whose rows are exactly the global table rows ``i * shard_size + r`` in the
    packed layout the gather kernel decodes, so the offload needs no storage of
    its own: mmap each shard's bytes ``MAP_SHARED`` read-only and let the
    kernel pick a per-segment base pointer.
    """
    want = _ST_DTYPE.get(dtype)
    if want is None:
        raise ValueError(
            f"--ple-offload-backend safetensors cannot serve {dtype} tables; "
            "supported: bfloat16, float8_e4m3fn, packed fp6 (uint8)"
        )
    shards = find_safetensors_ple_shards(model_dir, _ple_shard_key_filter(prefix))
    if not shards:
        raise ValueError(
            f"--ple-offload-backend safetensors: no PLE shard weights matching "
            f"{prefix!r} found under {model_dir}"
        )
    segments: list[PleSegment] = []
    for shard_idx, path, data_start, begin in shards:
        global_start = shard_idx * shard_size
        header, _ = _safetensors_header_cached(path)
        info = next(
            h
            for k, h in header.items()
            if k.endswith(f".ngram_embedding.shard_{shard_idx}.weight")
        )
        # The kernel reinterprets the mapped bytes as ``dtype``: a checkpoint
        # of another dtype would read back garbage silently, and the header's
        # element size is what sizes rows vs bytes.
        if info["dtype"] != want:
            raise ValueError(
                f"--ple-offload-backend safetensors: shard {shard_idx} of "
                f"{prefix!r} is stored as {info['dtype']} but the embedding "
                f"storage is {want}; set ple_embedding_dtype to match the "
                "checkpoint."
            )
        rows = int(info["shape"][0])
        ov_start = max(global_start, vocab_start)
        # The tensor's own row count can be short (the last shard absorbs
        # vocab padding); never map past its bytes.
        ov_end = min(global_start + rows, vocab_end)
        if ov_start >= ov_end:
            continue
        off = data_start + begin + (ov_start - global_start) * row_bytes
        nbytes = (ov_end - ov_start) * row_bytes
        fd, buf, addr = _mmap_readonly(path, off, nbytes)
        # madvise wants a page-aligned start; addr carries the intra-page pad
        # back, so advise from the mapping base over the padded length.
        _madvise(buf._addr, buf._nbytes, _MADV_RANDOM)
        segments.append(
            PleSegment(
                fd=fd,
                file_off=off,
                addr=addr,
                nbytes=nbytes,
                # TP-local rows: the kernel and the prefetcher both index the
                # table by global_idx - vocab_start.
                row_start=ov_start - vocab_start,
                row_end=ov_end - vocab_start,
                buffer=buf,
            )
        )
    if not segments:
        raise ValueError(
            f"--ple-offload-backend safetensors: rows [{vocab_start}, "
            f"{vocab_end}) of {prefix!r} are not covered by {model_dir}"
        )
    table = PleSegmentTable(segments, row_bytes)
    # Cold row faults cost ~0.5 ms each inside the gather kernel and all land
    # in the first seconds of serving; advise the whole range in once at
    # startup instead (measured 1.9-3.6 GiB/s: ~10-20 s for a 37 GiB table),
    # so the kernel faults through to pages the cache already holds.
    for seg in table.segments:
        try:
            os.posix_fadvise(seg.fd, seg.file_off, seg.nbytes, os.POSIX_FADV_WILLNEED)
        except OSError:
            break
    logger.info(
        "PLE table: checkpoint-backed read-only mmap, %.1f GiB in %d segments "
        "over local rows [%d, %d) from %s",
        table.total_bytes() / 2**30,
        len(table.segments),
        table.row_start,
        table.row_end,
        model_dir,
    )
    return table


def make_ple_file_prefetcher(table: torch.Tensor) -> Optional[PleFilePrefetcher]:
    """A prefetcher for a table returned by ``allocate_ple_host_table(..., "file")``."""
    segments = getattr(table, "_sglang_ple_segments", None)
    path = getattr(table, "_sglang_ple_file_path", None)
    if (
        path is None and segments is None
    ) or not envs.SGLANG_QWEN4_PLE_FILE_PREFETCH.get():
        return None
    if segments is not None:
        row_bytes = segments.row_bytes
    else:
        row_bytes = (
            int(table.shape[-1]) * table.element_size()
            if table.dim() >= 2
            else table.element_size()
        )
    if segments is not None:
        # The trimmer ranks and evicts by this prefetcher's grid, so the slice
        # size is read here and nowhere else.
        prefetcher = PleFilePrefetcher(
            row_bytes=row_bytes,
            segments=segments,
            range_bytes=int(envs.SGLANG_QWEN4_PLE_FILE_RSS_RANGE_MB.get()) << 20,
        )
    else:
        prefetcher = PleFilePrefetcher(path=path, row_bytes=row_bytes)
    logger.info(
        "PLE table: WILLNEED prefetch on for gathers of >= %d rows (row = %d B)",
        PLE_FILE_PREFETCH_MIN_ROWS,
        row_bytes,
    )
    return prefetcher


def make_ple_file_rss_trimmer(
    table: torch.Tensor,
    prefetcher: Optional[PleFilePrefetcher] = None,
) -> Optional[PleFileRssTrimmer]:
    """A started trimmer for a table from ``allocate_ple_host_table(..., "file")``.

    ``SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=0`` turns it off; it is also absent
    where the resident set cannot be read (no ``/proc/self/smaps``).
    """
    segments = getattr(table, "_sglang_ple_segments", None)
    path = getattr(table, "_sglang_ple_file_path", None)
    if path is None and segments is None:
        return None
    budget_gb = float(envs.SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB.get())
    if budget_gb <= 0:
        return None
    # A segment table (checkpoint-backed) has no single data_ptr; its own
    # mappings are what is measured and trimmed.
    if segments is not None:
        probe = segments.mappings()[0]
    else:
        nbytes = table.numel() * table.element_size()
        probe = (table.data_ptr(), nbytes)
    if _mapping_rss_bytes(*probe) is None:
        logger.warning(
            "PLE table: resident-set trim off, /proc/self/smaps is not readable; "
            "the mapping will creep towards %.1f GiB resident",
            (
                segments.total_bytes()
                if segments is not None
                else table.numel() * table.element_size()
            )
            / 2**30,
        )
        return None
    lru = bool(envs.SGLANG_QWEN4_PLE_FILE_RSS_LRU.get()) and prefetcher is not None
    trimmer = PleFileRssTrimmer(
        addr=0 if segments is not None else table.data_ptr(),
        nbytes=0 if segments is not None else table.numel() * table.element_size(),
        segments=segments,
        prefetcher=prefetcher if lru else None,
        budget_bytes=int(budget_gb * 2**30),
        interval_s=float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    trimmer.start()
    logger.info(
        "PLE table: resident set capped at %.1f GiB (%s), checked every %.0f s",
        budget_gb,
        "coldest-slice LRU" if trimmer._lru else "drop-all",
        float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    return trimmer


def check_file_backend_supported(device_index: int = 0, backend: str = "file") -> None:
    """Fail fast at load time instead of silently reading garbage in the kernel."""
    if envs.SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK.get():
        logger.warning(
            "PLE table: %s backend device check skipped by "
            "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK",
            backend,
        )
        return
    supported = device_reads_pageable_memory(device_index)
    if supported is None:
        raise RuntimeError(
            f"--ple-offload-backend {backend}: could not query "
            "cudaDevAttrPageableMemoryAccess. Set "
            "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 only if you know the "
            "device reads pageable host memory."
        )
    if not supported:
        raise ValueError(
            f"--ple-offload-backend {backend} needs a device that reads "
            "pageable host memory (GB10, or a discrete GPU with HMM on the "
            "open kernel modules). This device reports it does not; use "
            "--ple-offload-backend pinned."
        )


def default_ple_table_dir(model_path: str) -> str:
    """``$SGLANG_QWEN4_PLE_FILE_DIR/<model path>``, one directory per checkpoint."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_path).rstrip("/")).strip("_")
    return os.path.join(envs.SGLANG_QWEN4_PLE_FILE_DIR.get(), safe or "model")


def ple_table_file_name(
    shape: Sequence[int], dtype: torch.dtype, tag: Optional[str] = None
) -> str:
    """Deterministic file name so the sparse table is reused across restarts.

    ``tag`` distinguishes tables of the same shape that must not share a file,
    e.g. the vocabulary shards of different tensor-parallel ranks.
    """
    numel = 1
    for d in shape:
        numel *= int(d)
    elem = torch.empty(0, dtype=dtype).element_size()
    dims = "x".join(str(int(d)) for d in shape)
    suffix = f"_{tag}" if tag else ""
    return f"ple_table_{dims}_{str(dtype).replace('torch.', '')}_{numel * elem}B{suffix}.bin"


def device_reads_pageable_memory(device_index: int = 0) -> Optional[bool]:
    """Whether pageable host memory is directly addressable by the GPU.

    Returns None when the CUDA runtime library cannot be queried.
    """
    value = _device_attribute(_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS, device_index)
    return None if value is None else bool(value)


def device_faults_pageable_pages(device_index: int = 0) -> bool:
    """Whether the GPU reaches pageable host memory through page faults the driver
    serves (HMM on a discrete GPU), rather than through the host's page tables (ATS, as
    on GB10). False when either attribute cannot be queried."""
    reads = _device_attribute(_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS, device_index)
    host_tables = _device_attribute(
        _CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device_index
    )
    return reads == 1 and host_tables == 0


def _device_attribute(attribute: int, device_index: int) -> Optional[int]:
    """``cudaDeviceGetAttribute`` through the CUDA runtime library; None when it
    cannot be queried."""
    candidates = [ctypes.util.find_library("cudart")]
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        candidates += sorted(
            os.path.join(torch_lib, f)
            for f in os.listdir(torch_lib)
            if f.startswith("libcudart.so")
        )
    try:
        import nvidia.cuda_runtime  # type: ignore

        nv_lib = os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib")
        if os.path.isdir(nv_lib):
            candidates += sorted(
                os.path.join(nv_lib, f)
                for f in os.listdir(nv_lib)
                if f.startswith("libcudart.so")
            )
    except Exception:
        pass
    for name in [c for c in candidates if c]:
        try:
            cudart = ctypes.CDLL(name)
            value = ctypes.c_int()
            rc = cudart.cudaDeviceGetAttribute(
                ctypes.byref(value),
                ctypes.c_int(attribute),
                ctypes.c_int(device_index),
            )
            if rc == 0:
                return value.value
        except OSError:
            continue
    return None


def advise_rows_willneed(
    fd: int, rows: torch.Tensor, row_bytes: int, base_offset: int = 0
) -> int:
    """Queue the page-cache reads of ``rows`` of the file-backed table open as ``fd``
    and return once they are queued: ``posix_fadvise(WILLNEED)``, one call per run
    of adjacent pages. ``base_offset`` is where row 0 sits in ``fd``, which is
    nonzero for one shard of a checkpoint-backed table.

    The host gather copies its rows on one thread (the server runs torch with one
    thread) from a ``MADV_RANDOM`` mapping, so without this each cold row is its own
    4 KiB read from storage, one after another. Advised first, the reads are in flight
    together and the copy finds the rows in the page cache. Advisory: returns the number
    of pages advised, and stops at the first call that fails.
    """
    global _WILLNEED_FAILED
    if rows.numel() == 0:
        return 0
    first = rows.to(torch.int64) * row_bytes + base_offset
    pages = torch.cat(
        [first >> _PAGE_SHIFT, (first + row_bytes - 1) >> _PAGE_SHIFT]
    ).unique(sorted=True)
    run_starts = torch.ones_like(pages, dtype=torch.bool)
    run_starts[1:] = pages[1:] != pages[:-1] + 1
    starts = torch.nonzero(run_starts).flatten()
    lengths = torch.diff(starts, append=starts.new_tensor([pages.numel()]))
    advised = 0
    for page, count in zip(pages[starts].tolist(), lengths.tolist()):
        try:
            os.posix_fadvise(
                fd, page << _PAGE_SHIFT, count << _PAGE_SHIFT, os.POSIX_FADV_WILLNEED
            )
        except OSError as exc:
            if not _WILLNEED_FAILED:
                _WILLNEED_FAILED = True
                logger.warning(
                    "PLE table: posix_fadvise(WILLNEED) failed (%s); host gathers "
                    "read cold rows one at a time",
                    exc,
                )
            break
        advised += count
    return advised


def _madvise_random(storage: torch.Tensor, nbytes: int) -> None:
    """The table is pure random access (16 rows of 160 B per token). Without
    this the kernel's readahead pulls its whole window: measured 1.4 MB of disk
    per token, ~560x the bytes actually used.

    It bounds readahead I/O only. Folios that are already in the page cache are
    still mapped in whole on a fault, which is what ``PleFileRssTrimmer``
    exists for."""
    if not _madvise(storage.data_ptr(), nbytes, _MADV_RANDOM):
        logger.warning("PLE table: madvise(MADV_RANDOM) not applied")


def _alloc_cuda_host_table(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    # torch's pinned allocator rounds each block up to a power of two, pinning up
    # to ~2x the requested size; cudaHostAlloc pins exactly nbytes and is freed
    # with the table. Same call and flags as torch's (cudaHostAllocDefault).
    from sglang.srt.distributed.device_communicators.cuda_wrapper import (
        CudaRTLibrary,
    )
    from sglang.srt.mem_cache.pool_host.common import alloc_owned_host_tensor

    cudart = CudaRTLibrary()
    return alloc_owned_host_tensor(
        dims=shape,
        dtype=dtype,
        alloc=lambda nbytes: cudart.cudaHostAlloc(nbytes).value,
        free=lambda ptr: cudart.cudaFreeHost(ctypes.c_void_p(ptr)),
    )


def _libc() -> Optional[ctypes.CDLL]:
    global _LIBC
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL(
                ctypes.util.find_library("c") or "libc.so.6", use_errno=True
            )
        except OSError:
            return None
    return _LIBC


def _madvise(addr: int, length: int, advice: int) -> bool:
    """``madvise(2)`` on our own mapping. Advisory: never affects correctness."""
    libc = _libc()
    if libc is None:
        return False
    try:
        rc = libc.madvise(
            ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(advice)
        )
    except Exception:
        return False
    if rc != 0:
        logger.warning(
            "PLE table: madvise(advice=%d) failed (errno %d)",
            advice,
            ctypes.get_errno(),
        )
        return False
    return True


def _ranges_rss_bytes(
    ranges: Sequence[tuple[int, int]], smaps_path: str = "/proc/self/smaps"
) -> Optional[list]:
    """Resident bytes of each ``[addr, addr + nbytes)`` range, one smaps pass.

    Summed per mapping rather than taken from ``statm``/``smaps_rollup``: only
    the table's own residency should drive the trim, and on a unified-memory
    box the process RSS is dominated by everything else. One pass matters
    because the LRU trim wants per-segment residency for ~128 ranges.
    """
    spans = [(int(a), int(a) + int(n)) for a, n in ranges]
    totals = [0] * len(spans)
    try:
        with open(smaps_path, "r") as f:
            current: Optional[int] = None
            for line in f:
                header = _SMAPS_HEADER.match(line)
                if header is not None:
                    start = int(header.group(1), 16)
                    end = int(header.group(2), 16)
                    current = next(
                        (
                            i
                            for i, (lo, hi) in enumerate(spans)
                            if start < hi and end > lo
                        ),
                        None,
                    )
                elif current is not None:
                    rss = _SMAPS_RSS.match(line)
                    if rss is not None:
                        totals[current] += int(rss.group(1)) * 1024
    except OSError:
        return None
    return totals


def _mapping_rss_bytes(
    addr: int, nbytes: int, smaps_path: str = "/proc/self/smaps"
) -> Optional[int]:
    total = _ranges_rss_bytes([(addr, nbytes)], smaps_path)
    return None if total is None else total[0]
