"""FP8 blockwise-quantized LM head (SM120).

The Qwen3.8-Flash-Next head is [248320, 2560]; at bf16 it is weight-streaming
bound (~660 us on a PRO 6000 Blackwell), and NEXTN verify calls it once per
draft step plus once for the target verify. Storing the weight as FP8 e4m3
with 128x128 block scales halves the traffic (~330 us, DRAM roofline).

Quantization is applied once at weight-load time (never lazily) so the swap
always precedes CUDA-graph capture, and the steady-state forward uses only
the static-shape custom ops of the fp8 blockwise GEMM path. Measured on real
head weights: logit sigma 0.025 abs / KL 2.9e-4 nats vs bf16, GSM8K-neutral
within rerun noise.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

logger = logging.getLogger(__name__)

FP8_E4M3_MAX = 448.0
FP8_LM_HEAD_BLOCK = 128


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
    if weight.dtype not in (torch.bfloat16, torch.float16):
        logger.warning(
            "Skipping fp8 lm_head quantization: unexpected dtype %s", weight.dtype
        )
        return False
    n, k = weight.shape
    if n % FP8_LM_HEAD_BLOCK or k % FP8_LM_HEAD_BLOCK:
        logger.warning(
            "Skipping fp8 lm_head quantization: shape (%d, %d) not divisible by %d",
            n,
            k,
            FP8_LM_HEAD_BLOCK,
        )
        return False
    if not is_sm120_supported():
        logger.warning(
            "--enable-fp8-lm-head is SM120-only; leaving the lm head in %s",
            weight.dtype,
        )
        return False

    blocks = weight.float().view(
        n // FP8_LM_HEAD_BLOCK,
        FP8_LM_HEAD_BLOCK,
        k // FP8_LM_HEAD_BLOCK,
        FP8_LM_HEAD_BLOCK,
    )
    scales = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / FP8_E4M3_MAX
    quantized = (blocks / scales).to(torch.float8_e4m3fn).reshape(n, k)
    lm_head.weight = torch.nn.Parameter(quantized, requires_grad=False)
    lm_head.register_buffer(
        "weight_block_scale",
        scales.view(n // FP8_LM_HEAD_BLOCK, k // FP8_LM_HEAD_BLOCK).contiguous(),
        persistent=False,
    )
    return True


def fp8_blockwise_lm_head_apply(
    lm_head: ParallelLMHead,
    hidden_states: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
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
