"""ctypes bindings for the w2a16 native bits.

Two lazily-built shared objects, cached under ~/.cache/sglang/w2a16 keyed by
source hash:

* ``_ts4cpu_native.so`` — the AVX-512 W2A16 FFN kernel
  (``csrc/cpu_ts4_a16.cpp``):

      void ts4_a16_ffn(const void* x16, const int32_t* ids, const float* tw,
                       const uint8_t* w13p, const uint8_t* s13, float gs13,
                       const uint8_t* w2p,  const uint8_t* s2,  float gs2,
                       int T, int K, int E, int H, int I, float* out32,
                       int nthreads);

  All pointers are HOST addresses (the engine passes mmap'd expert planes and
  pinned staging rows), x is fp16, out fp32, fully synchronous, GIL held by
  the caller (the kernel itself is OpenMP-parallel).

* ``_w2a16_host_thunk.so`` — the exported trampoline that lets
  ``cuLaunchHostFunc`` call back into Python inside a CUDA graph (a raw C
  symbol, never a Python object, is embeddable in a graph node).

The kernel requires AVX-512 F/BW/DQ/VL + F16C; ``cpu_supports_kernel()``
probes it through the compiler so deployment mistakes fail at startup with a
clear message instead of SIGILL-ing mid-serve.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import os
import subprocess
import sysconfig
import tempfile

logger = logging.getLogger(__name__)

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_KERNEL_SRC = os.path.join(_CSRC, "cpu_ts4_a16.cpp")
_THUNK_SRC = os.path.join(_CSRC, "w2a16_host_thunk.c")


def _cache_dir() -> str:
    d = os.environ.get("SGLANG_W2A16_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "sglang", "w2a16"
    )
    os.makedirs(d, exist_ok=True)
    return d


def _build(name: str, src: str, extra_flags: list[str]) -> str:
    with open(src, "rb") as f:
        tag = hashlib.sha256(f.read()).hexdigest()[:12]
    out = os.path.join(_cache_dir(), f"{name}_{tag}.so")
    if os.path.exists(out):
        return out
    cxx = os.environ.get("CC", "cc") if src.endswith(".c") else os.environ.get("CXX", "g++")
    tmp = out + f".{os.getpid()}.tmp.so"
    cmd = [
        cxx,
        "-O3",
        "-shared",
        "-fPIC",
        "-std=c++17",
        f"-I{sysconfig.get_paths()['include']}",
        *extra_flags,
        src,
        "-o",
        tmp,
    ]
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if rc.returncode != 0:
        with open(log.name) as f:
            raise RuntimeError(f"w2a16: failed to build {name} with {cmd!r}:\n{f.read()}")
    os.replace(tmp, out)
    logger.info("w2a16: built %s -> %s", name, out)
    return out


def cpu_supports_kernel() -> bool:
    """True if this CPU runs the ts4_a16_ffn kernel (AVX-512 F/BW/DQ/VL+F16C).

    Kept separate from ``get_native()`` so startup guards can probe without
    paying for the optimized build.
    """
    try:
        cc = os.environ.get("CC", "cc")
        probe = (
            '#include <immintrin.h>\n'
            'int main(){return __builtin_cpu_supports("avx512f")'
            '&&__builtin_cpu_supports("avx512bw")'
            '&&__builtin_cpu_supports("avx512dq")'
            '&&__builtin_cpu_supports("avx512vl")'
            '&&__builtin_cpu_supports("avx512vl")?0:1;}\n'
        )
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "probe.c")
            exe = os.path.join(d, "probe")
            with open(src, "w") as f:
                f.write(probe)
            if subprocess.run([cc, "-O0", src, "-o", exe]).returncode != 0:
                return False
            return subprocess.run([exe]).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@functools.lru_cache(maxsize=1)
def get_native():
    """Return (lib, fn) for ts4_a16_ffn, building it on first use."""
    path = _build(
        "_ts4cpu_native",
        _KERNEL_SRC,
        ["-march=x86-64-v4", "-fopenmp"],
    )
    lib = ctypes.CDLL(path)
    fn = lib.ts4_a16_ffn
    fn.restype = None
    fn.argtypes = [
        ctypes.c_void_p,  # x16  fp16 [T,H]
        ctypes.c_void_p,  # ids  int32 [T*K]
        ctypes.c_void_p,  # tw   fp32 [T*K]
        ctypes.c_void_p,  # w13 plane u8 [E,2I,H/4]
        ctypes.c_void_p,  # s13 u8 [E,2I,H/32]
        ctypes.c_float,  # gs13
        ctypes.c_void_p,  # w2 plane u8 [E,H,I/4]
        ctypes.c_void_p,  # s2 u8 [E,H,I/32]
        ctypes.c_float,  # gs2
        ctypes.c_int,  # T
        ctypes.c_int,  # K
        ctypes.c_int,  # E
        ctypes.c_int,  # H
        ctypes.c_int,  # I
        ctypes.c_void_p,  # out fp32 [T,H]
        ctypes.c_int,  # nthreads
    ]
    return lib, fn


@functools.lru_cache(maxsize=1)
def get_host_thunk():
    """Return (lib, fn_addr) of the graph-safe host-func trampoline."""
    path = _build("_w2a16_host_thunk", _THUNK_SRC, [])
    lib = ctypes.CDLL(path)
    lib.ig_set_cb.argtypes = [ctypes.c_void_p]
    lib.ig_set_cb.restype = None
    lib.ig_host_thunk.argtypes = [ctypes.c_void_p]
    lib.ig_host_thunk.restype = None
    return lib, ctypes.cast(lib.ig_host_thunk, ctypes.c_void_p).value


def host_thunk_keepalive():
    """The thunk lib must outlive every graph that references it."""
    return get_host_thunk()[0]
