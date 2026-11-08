"""Host-side storage for the offloaded Qwen4-Exp PLE n-gram table.

``--ple-offload-embedding`` keeps the PLE table (47.7 GiB in fp8 for
Qwen3.8-Flash-Next) out of device memory and lets the Triton gather kernel read
rows straight from a host pointer. Two backends provide that pointer:

``pinned`` (default)
    Page-locked host memory of exactly the table's size. On a discrete GPU this
    frees VRAM. The buffer is allocated with ``mmap`` and registered with
    ``cudaHostRegister`` rather than ``torch.empty(..., pin_memory=True)``:
    PyTorch's caching host allocator rounds every pinned allocation up to a
    power of two, so the 47.7 GiB fp8 table would lock 64 GiB (and a 95.4 GiB
    bf16 table 128 GiB) of host memory.

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
import mmap as _mmap
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Optional, Sequence

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LIBC: Optional[ctypes.CDLL] = None
_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ")
_SMAPS_RSS = re.compile(r"^Rss:\s+(\d+) kB")

PLE_OFFLOAD_BACKENDS = ("pinned", "file", "safetensors")

# cudaDeviceAttr enum values (cuda_runtime_api.h).
_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS = 88
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

    def _segment_of(self, row: int) -> int:
        """Last segment whose bound is <= ``row`` (caller bounds-checks)."""
        segs, lo, hi = self.segments, 0, len(self.segments)
        while lo < hi:
            mid = (lo + hi) // 2
            if segs[mid].row_start <= row:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    def pages_for_rows(self, row_ids: Sequence[int]) -> list[tuple[int, int]]:
        """(fd, page-aligned file offset) pairs covering the given rows."""
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        if not self.segments:
            return out
        first, last = self.segments[0].row_start, self.segments[-1].row_end
        for row in sorted(int(r) for r in row_ids):
            if not first <= row < last:
                continue
            seg = self.segments[self._segment_of(row)]
            start = seg.file_off + (row - seg.row_start) * self.row_bytes
            end = start + (self.row_bytes - 1)
            for page in (start >> _PAGE_SHIFT, end >> _PAGE_SHIFT):
                key = (seg.fd, page << _PAGE_SHIFT)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def segments_for_rows(self, row_ids: Sequence[int]) -> set:
        """Indices of the segments containing any of the given rows.

        Segments (one mapping per checkpoint shard, hundreds of MiB) are the
        unit the LRU trim evicts by: a row's segment is exactly the kernel's
        ``count(bounds <= row) - 1``, so this mirrors the gather.
        """
        out: set = set()
        if not self.segments:
            return out
        first, last = self.segments[0].row_start, self.segments[-1].row_end
        for row in row_ids:
            row = int(row)
            if first <= row < last:
                out.add(self._segment_of(row))
        return out

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
                header, _ = _safetensors_header_cached(
                    os.path.join(model_dir, fname)
                )
                names_to_files.update(
                    {k: fname for k in header if k != "__metadata__"}
                )
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
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
        ]
        addr = libc.mmap(
            None, pad + length, _PROT_READ, _MAP_SHARED, fd, aligned
        )
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
    first (``posix_fadvise(WILLNEED)`` per distinct page, on one background
    thread) lets the block layer serve them concurrently. Measured on a GB10 /
    NVMe: cold prefill 650-750 tok/s -> 1,000-2,100 tok/s (warm: ~2,200-2,600).
    Decode-sized gathers are skipped; nothing runs during CUDA-graph capture.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        row_bytes: int = 0,
        min_rows: int = PLE_FILE_PREFETCH_MIN_ROWS,
        segments: Optional["PleSegmentTable"] = None,
    ) -> None:
        # A checkpoint-backed table spans many files; then PleSegmentTable maps
        # a row to (fd, file offset) and the page math happens there.
        self._segments = segments
        if segments is None:
            self._fd = os.open(path, os.O_RDONLY)
        self._row_bytes = int(row_bytes)
        self._min_rows = int(min_rows)
        self._pool = ThreadPoolExecutor(max_workers=1)
        # Monotonic recency per segment, consumed by PleFileRssTrimmer's LRU
        # mode. Written only from the gather thread, read by the trim thread;
        # int stores are atomic under the GIL and a racy cycle is harmless.
        self._hit_cycle = [0] * (len(segments.segments) if segments else 0)
        self._cycle = 0

    @staticmethod
    def pages_for_rows(row_ids: torch.Tensor, row_bytes: int) -> list[int]:
        start = row_ids.to(torch.int64) * row_bytes
        end = start + (row_bytes - 1)
        return (
            torch.cat([start >> _PAGE_SHIFT, end >> _PAGE_SHIFT])
            .unique(sorted=True)
            .tolist()
        )

    def _advise(self, pages) -> None:
        if self._segments is not None:
            for fd, off in pages:
                try:
                    os.posix_fadvise(fd, off, 1 << _PAGE_SHIFT, os.POSIX_FADV_WILLNEED)
                except OSError:
                    return
            return
        for p in pages:
            try:
                os.posix_fadvise(
                    self._fd, p << _PAGE_SHIFT, 1 << _PAGE_SHIFT, os.POSIX_FADV_WILLNEED
                )
            except OSError:
                return

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
            rows = row_ids.tolist()
            if self._hit_cycle:
                hits = self._segments.segments_for_rows(rows)
                if hits:
                    self._cycle += 1
                    for i in hits:
                        self._hit_cycle[i] = self._cycle
            pages = self._segments.pages_for_rows(rows)
        else:
            pages = self.pages_for_rows(row_ids, self._row_bytes)
        self._pool.submit(self._advise, pages)
        return True

    def hit_cycles(self) -> list:
        """Per-segment last-touch counter (0 = never seen by a gather)."""
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
        segments: Optional["PleSegmentTable"] = None,
        prefetcher: Optional["PleFilePrefetcher"] = None,
        lru: bool = True,
    ) -> None:
        # A checkpoint-backed table is many small mappings; the trim logic is
        # the same per mapping, so the ranges are kept as a list either way.
        self._ranges = (
            segments.mappings()
            if segments is not None
            else [(int(addr), int(nbytes))]
        )
        self._addr = int(addr)
        self._nbytes = int(nbytes) if segments is None else segments.total_bytes()
        self._budget = int(budget_bytes)
        self._interval = float(interval_s)
        self._chunk = int(chunk_bytes)
        # LRU mode needs per-segment recency (only the row stream knows it);
        # single-mapping file tables have one segment, so they keep drop-all.
        self._lru = bool(lru) and segments is not None and prefetcher is not None
        self._prefetcher = prefetcher
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

    def _trim_lru(self, ranges, per_rss, before: int) -> int:
        """Evict whole mappings, coldest first, until under budget.

        Whole mappings are the unit: recency is only known per segment (the
        row stream), and evicting a slice of one would drop warm rows too.
        Segments touched since the last trim are left alone, so at most the
        coldest half goes per cycle and the resident working set migrates
        instead of collapsing. A never-touched mapping counts as coldest.
        """
        hits = self._prefetcher.hit_cycles()
        cycle = self._prefetcher._cycle
        # Snapshot totals and copy what we evict: total can only shrink past
        # the budget, so a stale per-segment value is never re-evicted.
        evictions = [0] * len(ranges)
        total = sum(per_rss)
        for i in sorted(range(len(ranges)), key=lambda i: hits[i]):
            if total <= self._budget or self._stop.is_set():
                break
            if per_rss[i] == 0:
                continue
            if hits[i] and hits[i] >= cycle:
                # touched during the very trim that measured it -- hot
                continue
            evictions[i] = per_rss[i]
            total -= per_rss[i]
        for i, victim in enumerate(evictions):
            if victim:
                base, nbytes = ranges[i]
                self._evict_range(base, nbytes)
        freed = sum(evictions)
        if freed:
            logger.info(
                "PLE table: LRU-trimmed %.1f GiB of %d cold segments "
                "(%.1f GiB left of %.1f GiB, budget %.1f GiB)",
                freed / 2**30,
                sum(1 for v in evictions if v),
                (before - freed) / 2**30,
                before / 2**30,
                self._budget / 2**30,
            )
        return freed

    def trim_once(self) -> int:
        """Cut the mapping's resident set to the budget. Returns bytes freed.

        LRU mode (segment tables with a live gather stream) evicts the
        coldest mappings only; otherwise every mapping is dropped whole --
        the pre-LRU behaviour, still right for single-mapping file tables
        where recency has nothing to rank.
        """
        if self._lru:
            per = _ranges_rss_bytes(self._ranges)
            if per is None:
                return 0
            before = sum(per)
            if before <= self._budget:
                return 0
            freed = self._trim_lru(self._ranges, per, before)
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


# Pinned tables live as long as the process (like the model weights), so their
# mappings are kept alive here and never unregistered.
_PINNED_TABLE_BUFFERS: list = []
_CUDA_HOST_REGISTER_PORTABLE_MAPPED = 0x01 | 0x02


def _table_nbytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    numel = 1
    for d in shape:
        numel *= int(d)
    return numel * torch.empty(0, dtype=dtype).element_size()


def _allocate_pinned_table(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """Page-locked host tensor that locks exactly ``shape``/``dtype`` bytes.

    ``torch.empty(..., pin_memory=True)`` goes through the caching host allocator,
    which rounds the request up to the next power of two; for a multi-GiB table
    that locks tens of GiB for nothing and can push a memory-limited container
    over its limit. Map anonymous memory of the exact size and register it with
    CUDA instead; the result is pinned and device-accessible the same way.
    """
    import mmap

    shape = tuple(int(d) for d in shape)
    nbytes = _table_nbytes(shape, dtype)
    if nbytes == 0:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    buf = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    raw = torch.frombuffer(buf, dtype=torch.uint8)
    err = torch.cuda.cudart().cudaHostRegister(
        raw.data_ptr(), nbytes, _CUDA_HOST_REGISTER_PORTABLE_MAPPED
    )
    if int(err) != 0:
        buf.close()
        raise RuntimeError(
            f"cudaHostRegister of the PLE table ({nbytes} bytes) failed: {err}"
        )
    _PINNED_TABLE_BUFFERS.append(buf)
    logger.info("PLE table: pinned host memory (%.1f GiB, %s)", nbytes / 2**30, dtype)
    return raw.view(dtype).view(shape)


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
        return _allocate_pinned_table(shape, dtype)

    nbytes = _table_nbytes(shape, dtype)
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
) -> "PleSegmentTable":
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
    shards = find_safetensors_ple_shards(
        model_dir, _ple_shard_key_filter(prefix)
    )
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
    if (path is None and segments is None) or not envs.SGLANG_QWEN4_PLE_FILE_PREFETCH.get():
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
        prefetcher = PleFilePrefetcher(row_bytes=row_bytes, segments=segments)
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
    prefetcher: Optional["PleFilePrefetcher"] = None,
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
        "coldest-segment LRU" if trimmer._lru else "drop-all",
        float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    return trimmer


def check_file_backend_supported(
    device_index: int = 0, backend: str = "file"
) -> None:
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
                ctypes.c_int(_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS),
                ctypes.c_int(device_index),
            )
            if rc == 0:
                return bool(value.value)
        except OSError:
            continue
    return None


def _madvise_random(storage: torch.Tensor, nbytes: int) -> None:
    """The table is pure random access (16 rows of 160 B per token). Without
    this the kernel's readahead pulls its whole window: measured 1.4 MB of disk
    per token, ~560x the bytes actually used.

    It bounds readahead I/O only. Folios that are already in the page cache are
    still mapped in whole on a fault, which is what ``PleFileRssTrimmer``
    exists for."""
    if not _madvise(storage.data_ptr(), nbytes, _MADV_RANDOM):
        logger.warning("PLE table: madvise(MADV_RANDOM) not applied")


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
                        (i for i, (lo, hi) in enumerate(spans)
                         if start < hi and end > lo),
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
