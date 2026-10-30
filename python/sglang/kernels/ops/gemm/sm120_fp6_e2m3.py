from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# SM120 FP6-e2m3 blockwise-32 weight-only GEMV/GEMM (M <= 8).
#
# Weights are e2m3 codes (1+2+3 bit, bias 1, denorm m/8, max finite 7.0) packed
# 4 codes per 3 bytes, row-major, with fp32 scales per 32x32 block
# (s = amax / 7.0). Activations are plain bf16. This is the lm-head serving
# format: sm_120 has no fp6 tensor-core mode, so the kernel is a SIMT
# unpacking warp-per-row reduction; at M=1 it measures 254us on
# (248320 x 2560) vs 708us for a bf16 torch gemv (at its own 6-bit stream
# load floor). Each extra row adds one packed 64-bit x-load + fma per 4
# codes, so the band degrades gently with M: M=2 290us, M=4 364us, M=8
# 591us on the lm-head shape. It still beats the bf16 torch mm at M=4
# (672us); for M >= 8 at wide shapes the e3m4 b16 QMMA band (343us flat for
# M <= 64) is the better route.
# Accuracy (offline lm-head sim): logit rel-L2 2.45e-2,
# KL/row 4.1e-3, top-1 flip 0.001% -- between bf16 and fp8-e4m3-b128? no:
# it ties fp8-e4m3-b128 accuracy at 2.7x the speed, and fp8-e3m4-b128
# (rel-L2 1.29e-2) is 1.3x slower.
#
# Two backends, dispatched by :func:`sm120_fp6_e2m3_gemm`:
#   SIMT  (this file + gemv cuh):  M <= 2, bf16 activations, 254/290 us.
#   W6A8  (embedded QMMA cubin):   3 <= M <= 64, e4m3 activations with
#        per-128-group scales, flat 290-310 us (M <= 16); the cubin decodes
#        e2m3-in-byte natively -- no SASS patch (see the cuh header).


@cache_once
def _jit_gemv_module(k: int) -> Module:
    return load_jit(
        "sm120_fp6_e2m3_gemv",
        k,
        cuda_files=["gemm/sm120_fp6_e2m3_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Fp6E2m3GemvKernel<{k}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_sm120_fp6_e2m3_gemv(m: int, n: int, k: int) -> bool:
    """Whether the fp6-e2m3 b32 decode band applies (lm-head sized shapes)."""
    return (
        1 <= m <= 8
        and is_sm120_supported()
        and k % 128 == 0
        and 512 <= k <= 32768
        and n % 32 == 0
    )


@cache_once
def _jit_w6_module():
    """W6A8 QMMA cubin module (None if the JIT/driver path is unavailable)."""
    try:
        return load_jit(
            "sm120_fp6_e2m3_gemm_w6",
            cuda_files=["gemm/sm120_fp6_e2m3_gemm_w6.cuh"],
            cuda_wrappers=[("run", "sglang::Sm120Fp6E2m3W6Gemm::run")],
            extra_cuda_cflags=["-O3"],
        )
    except Exception:  # noqa: BLE001 - build failure: silent SIMT fallback
        return None


def use_sm120_fp6_e2m3_gemm_w6(m: int, n: int, k: int) -> bool:
    """Whether the W6A8 QMMA band applies (packed 6-bit B stream, e4m3 A).

    The SIMT kernel is faster at M=1 (254us vs 291) and keeps bf16
    activations, so it owns M <= 2; W6A8 is flat 291-590us for all larger
    M and the only band that scales past M=8 without launch-per-8 rows.
    """
    return (
        m >= 3
        and is_sm120_supported()
        and k % 128 == 0
        and n % 32 == 0
        and _jit_w6_module() is not None
    )


@debug_kernel_api
def sm120_fp6_e2m3_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """out[M, N] = x[M, K](bf16) @ dequant6(w_packed)[N, K]^T with b32 scales.

    w_packed: uint8 [N*K*6//8] tight 6-bit e2m3 codes (row-major);
    w_scale: fp32 [N//32, K//32] (amax / 7.0); bf16 out, any M.
    Routes M <= 2 (or no cubin) to the SIMT kernel and 3 <= M <= 64 to the
    W6A8 QMMA cubin; larger batches are chunked.
    """
    m, k = x.shape
    n = w_scale.shape[0] * 32
    simt_ok = m <= 8 and use_sm120_fp6_e2m3_gemv(m, n, k)
    # W6A8 owns every M once the SIMT band is out of range; SIMT (bf16
    # activations) stays preferred at M <= 2 where both bands apply.
    if (m > 2 or not simt_ok) and use_sm120_fp6_e2m3_gemm_w6(m, n, k):
        if m <= 32:
            return _w6a8_gemm(x, w_packed, w_scale, m, n, k)
        return torch.cat(
            [
                _w6a8_gemm(x[i : i + 32], w_packed, w_scale, min(32, m - i), n, k)
                for i in range(0, m, 32)
            ],
            dim=0,
        )
    if simt_ok:
        return sm120_fp6_e2m3_gemm_simt(x, w_packed, w_scale)
    if m > 8:
        return torch.cat(
            [
                sm120_fp6_e2m3_gemm(x[i : i + 8], w_packed, w_scale)
                for i in range(0, m, 8)
            ],
            dim=0,
        )
    return sm120_fp6_e2m3_gemm_simt(x, w_packed, w_scale)


@debug_kernel_api
def sm120_fp6_e2m3_gemm_simt(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """bf16-activation SIMT path (M <= 8) -- exact activations, lowest latency
    at M=1..2 and the fallback whenever the W6A8 cubin is unavailable."""
    m, k = x.shape
    n = w_scale.shape[0] * 32
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _jit_gemv_module(k).run(x, w_packed, w_scale, out)
    return out


def _w6a8_gemm(x, w_packed, w_scale, m, n, k) -> torch.Tensor:
    from sglang.kernels.ops.quantization.per_token_group_quant import (
        per_token_group_quant,
    )

    xq = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
    xq, xs = per_token_group_quant(
        x if x.is_contiguous() else x.contiguous(),
        xq,
        xs,
        group_size=128,
        scale_ue8m0=False,
    )
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _jit_w6_module().run(xq, w_packed, xs, w_scale, out)
    return out


@debug_kernel_api
def sm120_fp6_e2m3_gemv(
    x_1d: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """M=1 convenience over :func:`sm120_fp6_e2m3_gemm`; returns [N] bf16."""
    return sm120_fp6_e2m3_gemm(x_1d.unsqueeze(0), w_packed, w_scale)[0]


# ---------------------------------------------------------------------------
# Offline loader/test helper: bf16|fp32 [N, K] weights -> packed fp6-e2m3 b32.
# (Serving never calls this; checkpoint-build tools use it to emit the
# w_packed / w_scale tensors this kernel consumes.)
# ---------------------------------------------------------------------------

_E2M3_VALUES = torch.tensor(
    [m / 8 for m in range(8)]
    + [
        (1 + m / 8) * 2.0 ** (e - 1)
        for e in (1, 2, 3)
        for m in range(8)
        if not (e == 3 and m == 7)
    ],
    dtype=torch.float64,
)  # 31 finite positive magnitudes; index i <-> code i, max 7.0


def quantize_e2m3_b32(w: torch.Tensor):
    """[N, K] float weights -> (packed uint8 [N*K*6//8], scale fp32 [N//32, K//32])."""
    assert w.dim() == 2 and w.shape[0] % 32 == 0 and w.shape[1] % 32 == 0
    n, k = w.shape
    dev = w.device
    vals = _E2M3_VALUES.to(dev)
    mid = ((vals[1:] + vals[:-1]) / 2).contiguous()
    codes16 = torch.arange(vals.numel(), dtype=torch.uint8, device=dev)
    wf = w.double()
    blk = wf.reshape(n // 32, 32, k // 32, 32)
    s = (blk.abs().amax(dim=(1, 3), keepdim=True) / 7.0).clamp_min(1e-12)
    q = (blk / s).clamp(-7.0, 7.0)
    mag = q.abs().reshape(-1)
    idx = torch.searchsorted(mid, mag.contiguous())
    codes = (
        codes16[idx] | torch.where(blk.reshape(-1) < 0, 32, 0).to(torch.uint8)
    ).reshape(n, k)
    c4 = codes.reshape(-1, 4).to(torch.int32)
    packed = (
        (c4[:, 0] | (c4[:, 1] << 6) | (c4[:, 2] << 12) | (c4[:, 3] << 18))
        .view(torch.uint8)
        .reshape(-1, 4)[:, :3]
        .contiguous()
        .view(-1)
    )
    return packed, s.reshape(n // 32, k // 32).float(), codes


def dequantize_e2m3_b32(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reference inverse: codes [N, K] (0..63) + scale -> float64 [N, K] weights."""
    vals = _E2M3_VALUES.to(codes.device)
    c = codes.long()
    mag = vals[c & 31]
    return torch.where((c & 32) > 0, -mag, mag).reshape(
        codes.shape
    ) * scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
