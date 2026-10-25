"""CPU decode engine for w2a16_offload.

Wraps the mmap'd TS4 planes of one FusedMoE layer and runs
``ts4_a16_ffn`` over pinned staging rows.  Two entry points:

* ``run_eager(x, ids, tw, out)`` — D2H copies then a synchronous kernel
  call, used outside CUDA-graph capture;
* ``decode_capture(...)`` — the same job expressed as
  D2H memcpy + ``cuLaunchHostFunc`` + H2D memcpy against fixed-address
  pinned buffers, so it records into a decode CUDA graph and replays
  without any Python on the driver thread except the exported thunk.

The 2-bit planes are ~3.4x smaller than a w4 layout, so they stay host-side
mmap (page-cache shared across TP ranks); only activations move.
"""

from __future__ import annotations

import ctypes
import itertools
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.w2a16 import native

logger = logging.getLogger(__name__)

CAP_MAX = 512
MAX_CHUNK = 64  # rows per kernel call (bounds the fp32 h32 mirror)

_LIBCUDA = None


def _libcuda():
    global _LIBCUDA
    if _LIBCUDA is None:
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                _LIBCUDA = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("w2a16: libcuda.so not found")
    return _LIBCUDA


class W2A16CpuEngine:
    """One layer's TS4 planes + pinned decode staging."""

    def __init__(
        self,
        w13_plane: torch.Tensor,
        s13: torch.Tensor,
        w2_plane: torch.Tensor,
        s2: torch.Tensor,
        keepalive,
        E: int,
        H: int,
        I: int,
        K: int = 8,
        max_rows: int = 512,
    ):
        assert H % 32 == 0 and I % 32 == 0, "TS4 kernel needs H,I % 32 == 0"
        self.E, self.H, self.I = E, H, I
        self.K = K  # top_k (staging rows are exactly [cap, K])
        self._planes = (w13_plane, s13, w2_plane, s2)
        self._keepalive = keepalive
        self.cap = max(64, min(CAP_MAX, int(max_rows)))
        self.nthreads = envs.SGLANG_W2A16_CPU_THREADS.get()

        # The engine is built lazily inside the model forward, which runs
        # under InferenceMode; allocate staging as normal tensors so the
        # host-fn callback (no thread-local inference state) can copy_ them.
        with torch.inference_mode(False):
            self._xb = torch.empty((self.cap, H, 2), dtype=torch.uint8, pin_memory=True)
            self._xh = torch.empty((MAX_CHUNK, H), dtype=torch.float16, pin_memory=True)
            self._ids = torch.empty((self.cap, 8), dtype=torch.int32, pin_memory=True)
            self._tw = torch.empty((self.cap, 8), dtype=torch.float32, pin_memory=True)
            self._out = torch.empty((self.cap, H), dtype=torch.float32, pin_memory=True)
        self._xbv_cache = {}
        self._xh_p = self._xh.data_ptr()
        self._ids_p = self._ids.data_ptr()
        self._tw_p = self._tw.data_ptr()
        self._out_p = self._out.data_ptr()
        self._p = (
            w13_plane.data_ptr(), s13.data_ptr(),
            w2_plane.data_ptr(), s2.data_ptr(),
        )
        self._xdt = torch.bfloat16
        # graph jobs: slot -> (engine, bs); kept alive for replay
        self._jobs = {}
        self._slot = 0

        _, self._fn = native.get_native()
        self._thunk_lib, self._thunk_fn = native.get_host_thunk()
        self._cb = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self._host_slot)
        self._thunk_lib.ig_set_cb(ctypes.cast(self._cb, ctypes.c_void_p).value)
        # CFUNCTYPE object must outlive every graph node referencing it.
        self._cb_keepalive = self._cb

    # -- kernel over staged rows -------------------------------------

    def _run(self, m, ids_p, tw_p, out_p):
        self._fn(
            self._xh_p, ids_p, tw_p,
            self._p[0], self._p[1], 1.0,
            self._p[2], self._p[3], 1.0,
            m, self.K, self.E, self.H, self.I, out_p, self.nthreads,
        )

    def _xbv(self, dt):
        v = self._xbv_cache.get(dt)
        if v is None:
            v = self._xb.view(-1)[: self.cap * self.H * 2].view(dt).view(
                self.cap, self.H
            )
            self._xbv_cache[dt] = v
        return v

    def _hostfn(self, bs):
        xbv = self._xbv(self._xdt)
        K4, H4 = self.K * 4, self.H * 4
        for t0 in range(0, bs, MAX_CHUNK):
            m = min(MAX_CHUNK, bs - t0)
            self._xh[:m].copy_(xbv[t0 : t0 + m])
            self._run(
                m,
                self._ids_p + t0 * K4,
                self._tw_p + t0 * K4,
                self._out_p + t0 * H4,
            )

    def _host_slot(self, slot):
        eng, bs = _JOBS.get(int(slot), (None, 0))
        if eng is not None:
            eng._hostfn(bs)

    # -- eager path ----------------------------------------------------

    def run_eager(self, x, ids, tw, out, stream=None):
        """x cuda bf16/fp16 [bs,H]; ids int32/cuda [bs,K]; tw fp32/cuda
        [bs,K]; out cuda fp32 [bs,H]. Blocking, stream-ordered."""
        self._xdt = x.dtype
        for t0 in range(0, x.size(0), self.cap):
            m = min(self.cap, x.size(0) - t0)
            self._xbv(x.dtype)[:m].copy_(x[t0 : t0 + m])
            s = ids[t0 : t0 + m]
            self._ids[:m].copy_(s if s.dtype == torch.int32 else s.to(torch.int32))
            t = tw[t0 : t0 + m]
            self._tw[:m].copy_(t if t.dtype == torch.float32 else t.to(torch.float32))
            torch.cuda.synchronize(x.device)
            self._hostfn(m)
            out[t0 : t0 + m].copy_(self._out[:m], non_blocking=True)

    # -- graph-capture path --------------------------------------------

    def decode_capture(self, bs, x_ptr, esz, ids_ptr, tw_ptr, out_ptr, stream,
                       dtype=torch.bfloat16):
        if bs > self.cap:
            raise RuntimeError(f"w2a16: capture batch {bs} > staging cap {self.cap}")
        self._xdt = dtype
        rt = _cuda_rt()
        D2H = rt.cudaMemcpyKind.cudaMemcpyDeviceToHost
        H2D = rt.cudaMemcpyKind.cudaMemcpyHostToDevice
        _rt_check(rt.cudaMemcpyAsync(self._xb.data_ptr(), x_ptr,
                                     bs * self.H * esz, D2H, stream))
        _rt_check(rt.cudaMemcpyAsync(self._ids.data_ptr(), ids_ptr,
                                     bs * self.K * 4, D2H, stream))
        _rt_check(rt.cudaMemcpyAsync(self._tw.data_ptr(), tw_ptr,
                                     bs * self.K * 4, D2H, stream))
        self._slot = _next_slot()
        _JOBS[self._slot] = (self, bs)
        # raw driver call: cuda.bindings' host-func path installs a python
        # trampoline that segfaults on graph replay (ig_moe CONTRACT v1).
        err = _libcuda().cuLaunchHostFunc(
            ctypes.c_void_p(stream), ctypes.c_void_p(self._thunk_fn),
            ctypes.c_void_p(self._slot),
        )
        _rt_check(err)
        _rt_check(rt.cudaMemcpyAsync(out_ptr, self._out.data_ptr(),
                                     bs * self.H * 4, H2D, stream))


_JOBS: dict[int, tuple] = {}
_slot_counter = itertools.count(1)


def _next_slot() -> int:
    return next(_slot_counter)


def _cuda_rt():
    from cuda.bindings import runtime  # lazy: optional dependency

    return runtime


def _rt_check(err):
    if isinstance(err, tuple):  # cuda.bindings returns (err, ...) tuples
        err = err[0]
    if err is not None:
        code = getattr(err, "value", err)
        if code != 0:
            raise RuntimeError(f"w2a16: CUDA error {err}")
