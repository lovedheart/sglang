"""ctypes bindings for the PLE ``nvmecache`` NVMe row-lookup engine.

The engine (``qwen4_exp_ple_nvme.c``, compiled at first use and cached)
serves PLE table rows straight from the checkpoint's safetensors shards: a
DRAM page cache first, ``O_DIRECT`` preads behind it, into pinned staging
the GPU copies back. It is the read side of the xtier-style
``cudaLaunchHostFunc`` gather; see ``GraphNvmeLookup`` in
``qwen4_exp_ple_table.py`` for the CUDA side.

Nothing in this module touches CUDA.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import subprocess
import tempfile
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

_C_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qwen4_exp_ple_nvme.c"
)
_LIB_DIR_ENV = "SGLANG_PLE_NVME_LIB_DIR"
_CC_ENV = "CC"


def _cache_dir() -> str:
    override = os.environ.get(_LIB_DIR_ENV)
    if override:
        return os.path.expanduser(override)
    from sglang.srt.environ import envs

    return os.path.join(envs.SGLANG_CACHE_DIR.get(), "ple_nvme")


def build_ple_nvme_library() -> str:
    """Compile (or reuse) the engine shared library; return its path.

    The cache key is the C source's hash, so a forked build is always
    consistent with the source it shipped with. TP ranks may race to build
    the same file; ``os.replace`` is atomic and the bytes are identical.
    """
    with open(_C_SOURCE, "rb") as f:
        source = f.read()
    tag = hashlib.sha256(source).hexdigest()[:16]
    out_dir = os.path.expanduser(_cache_dir())
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"libple_nvme_{tag}.so")
    if os.path.exists(path):
        return path
    cc = os.environ.get(_CC_ENV) or "cc"
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".so", delete=False) as tmp:
        tmp_path = tmp.name
    cmd = [cc, "-O2", "-fPIC", "-shared", "-pthread", _C_SOURCE, "-o", tmp_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        os.unlink(tmp_path)
        raise RuntimeError(
            f"building the PLE nvmecache engine with {cc!r} failed: {detail}; "
            f"set $_CC_ENV or install a C compiler"
        ) from exc
    os.replace(tmp_path, path)
    logger.info("PLE nvmecache: engine built at %s", path)
    return path


class PleNvmeStore:
    """Handle on the opened row store (segment fds + page cache + workers).

    Kept alive for the process lifetime, like the weights it reads. Row ids
    are global vocabulary rows; the caller's ``vocab_start/vocab_end`` pick
    this rank's shard window and the engine zeroes rows outside it.
    """

    def __init__(
        self,
        segments: Sequence[tuple[str, int, int, int]],
        row_bytes: int,
        cache_bytes: int,
        qd: int,
    ) -> None:
        self._lib = _load_engine()
        self.row_bytes = int(row_bytes)
        n = len(segments)
        paths_t = ctypes.c_char_p * n
        u64 = ctypes.c_uint64
        paths = paths_t(*[os.fsencode(p) for p, _, _, _ in segments])
        begins = (u64 * n)(*[b for _, b, _, _ in segments])
        row_starts = (u64 * n)(*[rs for _, _, rs, _ in segments])
        row_ends = (u64 * n)(*[re for _, _, _, re in segments])
        err = ctypes.create_string_buffer(256)
        self._store = self._lib.ple_nvme_open(
            paths,
            begins,
            row_starts,
            row_ends,
            ctypes.c_uint32(n),
            ctypes.c_uint32(row_bytes),
            u64(cache_bytes),
            ctypes.c_uint32(qd),
            err,
            ctypes.c_size_t(256),
        )
        if not self._store:
            raise RuntimeError(
                f"PLE nvmecache: opening the row store failed: "
                f"{err.value.decode('utf-8', 'replace')}"
            )
        # Held for as long as the lookups that embed the pointers.
        self._keepalive: list = []

    def create_lookup(
        self,
        ids_ptr: int,
        count_ptr: int,
        rows_ptr: int,
        capacity: int,
        vocab_start: int,
        vocab_end: int,
    ) -> PleNvmeLookup:
        return PleNvmeLookup(
            self, ids_ptr, count_ptr, rows_ptr, capacity, vocab_start, vocab_end
        )

    def stats(self) -> tuple[int, int, int]:
        """(cache hits, misses, storage pages read) since open."""
        hits = ctypes.c_uint64()
        misses = ctypes.c_uint64()
        pages = ctypes.c_uint64()
        self._lib.ple_store_stats(
            self._store,
            ctypes.byref(hits),
            ctypes.byref(misses),
            ctypes.byref(pages),
        )
        return hits.value, misses.value, pages.value

    def close(self) -> None:
        if self._store:
            self._lib.ple_nvme_close(self._store)
            self._store = None
            self._keepalive.clear()

    def __del__(self):  # pragma: no cover - interpreter shutdown races
        try:
            self.close()
        except Exception:
            pass


class PleNvmeLookup:
    """One preallocated staging run: fixed pointers a graph can capture.

    ``run_func`` is the ``cudaLaunchHostFunc`` target and ``run_arg`` its
    argument. The run serves ``count_ptr`` ids -- pinned like ``ids_ptr``, so
    a captured graph's D2H copies of both make replays self-contained.
    """

    def __init__(
        self,
        store: PleNvmeStore,
        ids_ptr: int,
        count_ptr: int,
        rows_ptr: int,
        capacity: int,
        vocab_start: int,
        vocab_end: int,
    ) -> None:
        self._lib = store._lib
        self._store = store
        self.capacity = int(capacity)
        err = ctypes.create_string_buffer(256)
        self._lookup = self._lib.ple_lookup_create(
            store._store,
            ctypes.c_void_p(ids_ptr),
            ctypes.c_void_p(count_ptr),
            ctypes.c_void_p(rows_ptr),
            ctypes.c_uint32(capacity),
            ctypes.c_uint64(vocab_start),
            ctypes.c_uint64(vocab_end),
            err,
            ctypes.c_size_t(256),
        )
        if not self._lookup:
            raise RuntimeError(
                "PLE nvmecache: creating a lookup failed: "
                f"{err.value.decode('utf-8', 'replace')}"
            )
        self.run_func = ctypes.cast(self._lib.ple_lookup_run, ctypes.c_void_p).value
        self.run_arg = ctypes.cast(self._lookup, ctypes.c_void_p).value

    def status(self) -> int:
        return int(self._lib.ple_lookup_status(self._lookup))

    def close(self) -> None:
        if self._lookup:
            self._lib.ple_lookup_destroy(self._lookup)
            self._lookup = None


_ENGINE: Optional[ctypes.CDLL] = None


def _load_engine() -> ctypes.CDLL:
    global _ENGINE
    if _ENGINE is None:
        lib = ctypes.CDLL(build_ple_nvme_library())
        u64p = ctypes.POINTER(ctypes.c_uint64)
        lib.ple_nvme_open.restype = ctypes.c_void_p
        lib.ple_nvme_open.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            u64p,
            u64p,
            u64p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.ple_nvme_close.argtypes = [ctypes.c_void_p]
        lib.ple_lookup_create.restype = ctypes.c_void_p
        lib.ple_lookup_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.ple_lookup_destroy.argtypes = [ctypes.c_void_p]
        lib.ple_lookup_status.restype = ctypes.c_uint32
        lib.ple_lookup_status.argtypes = [ctypes.c_void_p]
        lib.ple_lookup_run.argtypes = [ctypes.c_void_p]
        lib.ple_store_stats.argtypes = [ctypes.c_void_p, u64p, u64p, u64p]
        _ENGINE = lib
    return _ENGINE
