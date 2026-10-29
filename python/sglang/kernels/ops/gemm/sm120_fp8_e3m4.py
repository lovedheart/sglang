from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# SM120 FP8-e3m4 block-128 kernels.
#
# The weight side stores e3m4 codes (1+3+4 bit, bias 3, max 30, 16 levels per
# octave) instead of e4m3, gaining ~6 dB per-tensor SNR for the same byte
# count. The activations stay e4m3. Decoding e3m4 on the tensor core is not
# expressible in PTX; see csrc/gemm/sm120_fp8_e3m4_gemm.cuh for the patched
# cubin and csrc/gemm/sm120_fp8_e3m4_gemv.cuh for the SIMT decode path.


@cache_once
def _jit_gemv_module(k: int) -> Module:
    return load_jit(
        "sm120_fp8_e3m4_gemv",
        k,
        cuda_files=["gemm/sm120_fp8_e3m4_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Fp8E3m4GemvKernel<{k}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_gemm_module() -> Module:
    if not is_sm120_supported():
        raise RuntimeError("sm120_fp8_e3m4_gemm requires SM120 (Blackwell).")
    return load_jit(
        "sm120_fp8_e3m4_gemm",
        cuda_files=["gemm/sm120_fp8_e3m4_gemm.cuh"],
        cuda_wrappers=[("run", "sglang::Sm120Fp8E3m4Gemm::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_gemm_cutlass_module() -> "Module | None":
    """CUTLASS-ported wide-band kernel; None (cached) keeps the old path."""
    if not is_sm120_supported():
        return None
    try:
        return load_jit(
            "sm120_fp8_e3m4_gemm_cutlass",
            cuda_files=["gemm/sm120_fp8_e3m4_gemm_cutlass.cuh"],
            cuda_wrappers=[("run", "sglang::Sm120Fp8E3m4GemmCutlass::run")],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=[
                "-DNDEBUG",
                "-DCUTE_USE_PACKED_TUPLE=1",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                "-DCUTLASS_TEST_LEVEL=0",
                "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
                "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
        )
    except Exception as e:  # build/link failure: silent fallback, like the cubin
        print(f"sm120_fp8_e3m4_gemm_cutlass unavailable, using handwritten: {e}")
        return None


@cache_once
def _jit_gemm_cute_ws_module() -> "Module | None":
    """Self-owned CuTe warp-specialized kernel; None (cached) keeps old paths."""
    if not is_sm120_supported():
        return None
    try:
        return load_jit(
            "sm120_fp8_e3m4_gemm_cute_ws",
            cuda_files=["gemm/sm120_fp8_e3m4_gemm_cute_ws.cuh"],
            cuda_wrappers=[("run", "sglang::Sm120Fp8E3m4GemmCuteWs::run")],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=[
                "-DNDEBUG",
                "-DCUTE_USE_PACKED_TUPLE=1",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
        )
    except Exception as e:  # build failure: silent fallback, like the cutlass path
        print(f"sm120_fp8_e3m4_gemm_cute_ws unavailable, using cutlass: {e}")
        return None


def use_sm120_fp8_e3m4_gemv(m: int, n: int, k: int) -> bool:
    # One warp per output row, four 128-blocks per pass.
    return (
        m == 1
        and is_sm120_supported()
        and k % 512 == 0
        and 512 <= k <= 32768
        and n % 8 == 0
        and 256 <= n <= 32768
    )


def use_sm120_fp8_e3m4_gemm(m: int, n: int, k: int) -> bool:
    # Tiled patched-QMMA kernels. The prefill band (M > 64) runs on cute_ws,
    # which stores C through TMA with a predicated boundary path and needs
    # only n % 8; the decode band stays on the handwritten 16x32/128x64 tiles
    # and requires n % 64.
    if not (m >= 2 and is_sm120_supported() and k % 128 == 0):
        return False
    return n % 8 == 0 if m > 64 else n % 64 == 0


@debug_kernel_api
def sm120_fp8_e3m4_gemv(
    x_fp8: torch.Tensor,
    w_e3m4: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """y[N] = sum_k dec34(w) * dec43(x) * x_scale[k/128] * w_scale[n/128, k/128].

    x_fp8: [K] e4m3 codes; w_e3m4: [N, K] e3m4 codes (F8_E4M3 container);
    x_scale: [K/128] fp32; w_scale: [N/128, K/128] fp32. bf16 out.
    """
    n = w_e3m4.shape[0]
    out = torch.empty((n,), dtype=torch.bfloat16, device=x_fp8.device)
    _jit_gemv_module(w_e3m4.shape[1]).run(
        x_fp8, w_e3m4, x_scale, w_scale, out
    )
    return out


def _gemm_route(m: int, n: int, k: int) -> str:
    """'b16' handwritten decode tile; 'cute_ws' CuTe kernel; 'cutlass'
    collective; 'hand' handwritten plain/split-K.

    Cold NC=8 medians (interleaved, 256MB L2 flush, rotating B): cute_ws beats
    the cutlass collective 1.10-1.14x on the seven model shapes at prefill M,
    and beats every handwritten entry for M >= 64 (128x16384x2560: 34.8us vs
    wide 61.4; 4096: 414.7 vs 735.2; narrow 1024x2560x6144: 49.2 vs sk 66.6).
    Below that the 16x32 b16 tile wins (16 rows: 32.8 vs 50.2; 32: 36.9 vs
    45.1; 48: 41.0 vs 43.0) — tile quantization. cute_ws needs only N % 8, so
    it also covers the N % 128 != 0 shapes the collective asserts away.
    """
    if m <= 64:
        return "b16"
    return "cute_ws"


def _cutlass_sched(m: int, n: int, k: int) -> int:
    """0: cutlass void (tile-per-CTA); 1: cutlass StreamK.

    Measured cold NC=8 (bench3.py), the cubin's void scheduler beats the
    handwritten plain/split-K across every mid band by 1.21-1.76x, so 0 is
    selected whenever the kernel can run at all. Only reachable now if the
    cute_ws build is unavailable. n % 128 required: the collective indexes
    w_scale as exactly [N/128, K/128]; the caller only routes such shapes.
    """
    return 0


@debug_kernel_api
def sm120_fp8_e3m4_gemm(
    a_fp8: torch.Tensor,
    b_e3m4: torch.Tensor,
    a_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """out[M, N] = A[M,K](e4m3) @ B[N,K](e3m4)^T with block-128 scales, bf16."""
    m, n, k = a_fp8.shape[0], b_e3m4.shape[0], a_fp8.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a_fp8.device)
    route = _gemm_route(m, n, k)
    if route == "cute_ws":
        cute_ws = _jit_gemm_cute_ws_module()
        if cute_ws is not None:
            # Params mirror the production layouts: a_scale is [M, K/128]
            # row-major (Major::K), w_scale [N/128, K/128] — no transposes.
            cute_ws.run(out, a_fp8, b_e3m4, a_scale, w_scale)
            return out
        route = "cutlass" if (n % 128 == 0 and k % 128 == 0) else "hand"
    if route == "cutlass":
        cutlass = _jit_gemm_cutlass_module()
        if cutlass is not None:
            cutlass.run(out, a_fp8, b_e3m4, a_scale, w_scale, _cutlass_sched(m, n, k))
            return out
    _jit_gemm_module().run(a_fp8, b_e3m4, a_scale, w_scale, out)
    return out
