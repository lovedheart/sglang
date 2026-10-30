"""FP6-e2m3 blockwise-32 packed LM head (SM120).

The Qwen3.8-Flash-Next head is [248320, 2560]; at bf16 it is weight-streaming
bound (~708 us on a PRO 6000 Blackwell). Packing it as fp6 e2m3 codes (4 per
3 bytes, 477 MB vs 1.27 GB) with fp32 32x32 block scales lets the SIMT
unpacking kernels of ``sglang.kernels.ops.gemm.sm120_fp6_e2m3`` serve decode
at ~250 us (the 6-bit stream load floor) and stay flat at ~280 us through
M=16 via the W6A8 QMMA cubin. Accuracy
on real head weights: logit rel-L2 2.45e-2 vs the bf16 head, KL/row 4.1e-3
nats, top-1 flip 0.001% -- ties fp8-e4m3-b128 at 2.7x the speed.

Like the fp8 head, quantization is applied once at weight-load time (never
lazily) so the swap always precedes CUDA-graph capture. Unlike it, there is
no fp6 torch dtype: the bf16 ``lm_head.weight`` is released and replaced by
uint8 packed codes + fp32 scales, so ``lm_head.weight`` is ``None``
afterwards. Dense-weight consumers (e.g. DFLASH draft folding) see no weight
and fall back to their eager paths. Weight reload after quantization is not
supported: the bf16 source is gone.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

logger = logging.getLogger(__name__)

FP6_LM_HEAD_BLOCK = 32
_QUANT_N_CHUNK = 8192  # bounds peak fp64 quantizer memory beside a loaded model


@torch.no_grad()
def quantize_lm_head_to_fp6_b32(lm_head: ParallelLMHead) -> bool:
    """Swap ``lm_head.weight`` in place for packed fp6-e2m3 b32 tensors.

    Stores ``weight_fp6_packed`` (uint8 ``[N*K*6//8]`` e2m3 codes, row-major,
    4 per 3 bytes) and ``weight_fp6_scale`` (fp32 ``[N//32, K//32]``,
    amax / 7.0) buffers and flips ``fp6_b32_packed``. Returns True when the
    head was packed, False when it was left untouched (already packed, no
    dense bf16 weight, shape outside the sm120 fp6 band, or non-SM120
    platform).
    """
    if getattr(lm_head, "fp6_b32_packed", False):
        return False
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return False
    if weight.dtype == torch.float8_e4m3fn:
        logger.warning(
            "--lm-head-dtype fp6_e2m3 ignored: the lm head is already FP8 "
            "blockwise-quantized (the fp8 formats win)"
        )
        return False
    if weight.dtype != torch.bfloat16:
        logger.warning(
            "Skipping fp6 lm_head quantization: unexpected dtype %s", weight.dtype
        )
        return False
    n, k = weight.shape
    if not is_sm120_supported():
        logger.warning(
            "--lm-head-dtype fp6_e2m3 is SM120-only; leaving the lm head in %s",
            weight.dtype,
        )
        return False
    from sglang.kernels.ops.gemm.sm120_fp6_e2m3 import (
        quantize_e2m3_b32,
        use_sm120_fp6_e2m3_gemv,
    )

    # Packing requires the SIMT band so M<=2 decode keeps exact (bf16)
    # activations; larger M then rides the W6A8 cubin.
    if not use_sm120_fp6_e2m3_gemv(1, n, k):
        logger.warning(
            "Skipping fp6 lm_head quantization: shape (%d, %d) outside the "
            "sm120 fp6 band (N %% 32, K %% 128 in [512, 32768])",
            n,
            k,
        )
        return False

    packed, scales = [], []
    for n0 in range(0, n, _QUANT_N_CHUNK):
        p_, s_, _ = quantize_e2m3_b32(weight[n0 : n0 + _QUANT_N_CHUNK])
        packed.append(p_)
        scales.append(s_)
    lm_head.register_buffer(
        "weight_fp6_packed", torch.cat(packed).contiguous(), persistent=False
    )
    lm_head.register_buffer(
        "weight_fp6_scale", torch.cat(scales).contiguous(), persistent=False
    )
    del packed, scales, weight
    lm_head.weight = None  # releases the 1.27 GB bf16 copy
    lm_head.fp6_b32_packed = True
    return True


def fp6_lm_head_apply(
    lm_head: ParallelLMHead,
    hidden_states: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """logits[M, N] (bf16) from the packed head, any M.

    CUDA-graph safe: the only tensors created per call are kernel inputs and
    outputs (torch allocator), same contract as the fp8 head path. The ops
    dispatcher picks the backend per M (SIMT <= 2, W6A8 QMMA above).
    """
    from sglang.kernels.ops.gemm.sm120_fp6_e2m3 import sm120_fp6_e2m3_gemm

    x = hidden_states
    if x.dtype != torch.bfloat16:
        x = x.bfloat16()
    x = x.contiguous()
    logits = sm120_fp6_e2m3_gemm(x, lm_head.weight_fp6_packed, lm_head.weight_fp6_scale)
    if bias is not None:
        logits = logits + bias.to(logits.dtype)
    return logits
