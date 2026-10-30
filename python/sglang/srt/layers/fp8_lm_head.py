"""FP8 blockwise-quantized LM head (SM120).

The Qwen3.8-Flash-Next head is [248320, 2560]; at bf16 it is weight-streaming
bound (~660 us on a PRO 6000 Blackwell), and NEXTN verify calls it once per
draft step plus once for the target verify. Storing the weight as FP8 e4m3
with 128x128 block scales halves the traffic (~330 us, DRAM roofline).

An e3m4 twin (--lm-head-dtype fp8_e3m4) stores e3m4 codes (1+3+4, bias 3,
max 30) in the same float8_e3m4 container and block-128 layout, declared via
``weight.e4m3_only = False`` so ``cutlass_w8a8_block_fp8_linear_with_fallback``
routes it to the sm120 patched-QMMA kernels (flat ~340 us for any decode
batch). Both formats quantize once at weight-load time (never lazily) so the
swap always precedes CUDA-graph capture, and the steady-state forward uses
only the static-shape custom ops of the fp8 blockwise GEMM path. Measured on
real head weights: e4m3 logit sigma 0.025 abs / KL 2.9e-4 nats vs bf16,
GSM8K-neutral within rerun noise.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from sglang.kernels.jit.utils import cache_once
from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

logger = logging.getLogger(__name__)

FP8_E4M3_MAX = 448.0
FP8_E3M4_MAX = 30.0  # e3m4: 1+3+4 bits, bias 3, max finite (2 - 1/16) * 2^4
FP8_LM_HEAD_BLOCK = 128


@cache_once
def _e3m4_tables() -> tuple:
    """Positive e3m4 value grid and the 127 midpoints used for RTNE encoding.

    e3m4 is 1+3+4 bits (sign, 3-bit exponent bias 3, 4-bit mantissa), max
    finite 30.0 at code 126; code 127 is the format's only NaN.
    """
    vals = torch.empty(128, dtype=torch.float64)
    for byte in range(128):
        e, m = (byte >> 4) & 7, byte & 15
        if byte == 127:
            vals[byte] = float("nan")  # e=7, m=15: the single NaN
        else:
            vals[byte] = (m / 16) * 0.25 if e == 0 else (2.0 ** (e - 3)) * (1 + m / 16)
    pos = vals[:127].contiguous()  # finite positive branch, code 0..126 (0 = +0)
    mids = (pos[1:] + pos[:-1]) / 2  # all exactly representable in fp64
    return pos, mids


def _encode_e3m4(x: torch.Tensor) -> torch.Tensor:
    """RTNE-quantize to e3m4 codes (uint8, sign bit 7). Ties go to the even
    code, matching the hardware decode tables of the sm120 e3m4 kernels."""
    _, mids = _e3m4_tables()
    mids_d = mids.to(x.device)
    x64 = x.double().clamp(-FP8_E3M4_MAX, FP8_E3M4_MAX)
    sign = torch.where(x64.signbit(), 128, 0).to(torch.uint8)
    flat = x64.abs().reshape(-1)
    # Insertion point into the midpoint ladder = nearest code; clamping to
    # 30.0 keeps the NaN code 127 out of reach.
    idx = torch.searchsorted(mids_d, flat.contiguous()).clamp_(max=126)
    # searchsorted hands back the lower candidate on an exact midpoint; a
    # tie must go to the even code, and byte parity IS mantissa parity
    # (adjacent codes alternate), so bump odd lower bounds up one.
    tie = flat == mids_d.gather(0, idx.clamp(max=mids_d.numel() - 1))
    idx = torch.where(tie & (idx & 1 == 1), idx + 1, idx)
    return idx.to(torch.uint8).reshape(x.shape) | sign


@torch.no_grad()
def quantize_lm_head_to_fp8_block128(lm_head: ParallelLMHead) -> bool:
    """Swap ``lm_head.weight`` in place for an fp8 blockwise-128 copy.

    Scales are stored on the module as ``weight_block_scale`` with layout
    [N // 128, K // 128], the layout ``cutlass_w8a8_block_fp8_linear_with_fallback``
    expects. Returns True when the head was quantized, False when it was
    left untouched (already quantized, unsupported dtype, non-divisible
    shape, or non-SM120 platform). Weight reload after a True return is not
    supported: the bf16 source is gone.
    """
    weight = lm_head.weight
    if weight.dtype == torch.float8_e4m3fn:
        return False
    if getattr(lm_head, "weight_is_e3m4", False):
        logger.warning(
            "lm head is already packed as fp8_e3m4; refusing to re-quantize "
            "it as e4m3 (the e3m4 bytes would be read as e4m3)"
        )
        return False
    ok = _check_block128_head(lm_head, "e4m3")
    if not ok:
        return False
    n, k = weight.shape
    blocks = weight.float().view(
        n // FP8_LM_HEAD_BLOCK,
        FP8_LM_HEAD_BLOCK,
        k // FP8_LM_HEAD_BLOCK,
        FP8_LM_HEAD_BLOCK,
    )
    scales = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / FP8_E4M3_MAX
    quantized = (blocks / scales).to(torch.float8_e4m3fn).reshape(n, k)
    lm_head.weight = torch.nn.Parameter(quantized, requires_grad=False)
    lm_head.weight.e4m3_only = True  # online-quantized e4m3, not an e3m4 checkpoint
    lm_head.register_buffer(
        "weight_block_scale",
        scales.view(n // FP8_LM_HEAD_BLOCK, k // FP8_LM_HEAD_BLOCK).contiguous(),
        persistent=False,
    )
    return True


def _check_block128_head(lm_head: ParallelLMHead, fmt: str) -> bool:
    """Shared guards for the block-128 head packers (True = pack it)."""
    weight = lm_head.weight
    if weight.dtype not in (torch.bfloat16, torch.float16):
        logger.warning(
            "Skipping fp8-%s lm_head quantization: unexpected dtype %s",
            fmt,
            weight.dtype,
        )
        return False
    n, k = weight.shape
    if n % FP8_LM_HEAD_BLOCK or k % FP8_LM_HEAD_BLOCK:
        logger.warning(
            "Skipping fp8-%s lm_head quantization: shape (%d, %d) not divisible by %d",
            fmt,
            n,
            k,
            FP8_LM_HEAD_BLOCK,
        )
        return False
    if not is_sm120_supported():
        logger.warning(
            "--lm-head-dtype fp8_%s is SM120-only; leaving the lm head in %s",
            fmt,
            weight.dtype,
        )
        return False
    return True


@torch.no_grad()
def quantize_lm_head_to_e3m4_block128(
    lm_head: ParallelLMHead, n_chunk_rows: int = 8192
) -> bool:
    """Swap ``lm_head.weight`` in place for e3m4 codes + block-128 scales.

    Same [N // 128, K // 128] layout and float8 container as the e4m3 packer,
    but the weight declares itself e3m4 (``e4m3_only = False``,
    ``weight_is_e3m4 = True``) so the fp8 blockwise linear routes it to the
    sm120 patched-QMMA e3m4 kernels. Codes are RTNE to the e3m4 grid with
    scale = amax / 30 (fp32), mirroring the FP8_PB_WO_E3M4 checkpoint format.
    """
    weight = lm_head.weight
    if weight.dtype == torch.float8_e4m3fn:
        if getattr(lm_head, "weight_is_e3m4", False):
            return False  # already e3m4-packed
        logger.warning(
            "lm head is already e4m3 block-quantized; leaving it (--lm-head-"
            "dtype fp8_e3m4 ignored)"
        )
        return False
    if not _check_block128_head(lm_head, "e3m4"):
        return False
    n, k = weight.shape
    codes = torch.empty(n, k, dtype=torch.uint8, device=weight.device)
    scales = torch.empty(
        n // FP8_LM_HEAD_BLOCK,
        k // FP8_LM_HEAD_BLOCK,
        dtype=torch.float32,
        device=weight.device,
    )
    # Chunk the rows: the fp64 temporaries otherwise peak at 2x the head size
    # next to an already-loaded model.
    for n0 in range(0, n, n_chunk_rows):
        rows = min(n_chunk_rows, n - n0)
        blocks = (
            weight[n0 : n0 + rows]
            .double()
            .view(rows // FP8_LM_HEAD_BLOCK, FP8_LM_HEAD_BLOCK, k // 128, 128)
        )
        amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
        s = amax / FP8_E3M4_MAX
        codes[n0 : n0 + rows] = _encode_e3m4(blocks / s).reshape(rows, k)
        scales[n0 // 128 : (n0 + rows) // 128] = s.squeeze((1, 3)).float()
    quantized = codes.view(
        torch.float8_e4m3fn
    )  # container view; kernel reads raw bytes
    lm_head.weight = torch.nn.Parameter(quantized, requires_grad=False)
    lm_head.weight.e4m3_only = False  # declaration: these bytes are e3m4 codes
    lm_head.weight_is_e3m4 = True
    lm_head.register_buffer("weight_block_scale", scales, persistent=False)
    return True


def fp8_blockwise_lm_head_apply(
    lm_head: ParallelLMHead,
    hidden_states: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from sglang.srt.layers.quantization.fp8_utils import (
        cutlass_w8a8_block_fp8_linear_with_fallback,
    )

    return cutlass_w8a8_block_fp8_linear_with_fallback(
        input=hidden_states,
        weight=lm_head.weight,
        block_size=[FP8_LM_HEAD_BLOCK, FP8_LM_HEAD_BLOCK],
        weight_scale=lm_head.weight_block_scale,
        bias=bias,
    )
