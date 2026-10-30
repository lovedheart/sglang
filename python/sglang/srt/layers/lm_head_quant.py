"""Unified quantized LM head selection (--lm-head-dtype).

One place maps the requested dtype to its load-time packer; serving needs no
knowledge of the flag. The fp8 variants (e4m3/e3m4) keep the float8 container
and are applied by ``fp8_blockwise_lm_head_apply`` (which routes
e3m4-declared weights to the sm120 patched-QMMA kernels by weight
declaration). The fp6 packer releases the weight tensor and is applied by
``fp6_lm_head_apply`` behind its ``fp6_b32_packed`` attribute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

LM_HEAD_DTYPES = ("bf16", "fp8_e4m3", "fp8_e3m4", "fp6_e2m3")

# Scale-block geometry per packed format (NxK blocks of the weight).
LM_HEAD_BLOCK = {"fp8_e4m3": 128, "fp8_e3m4": 128, "fp6_e2m3": 32}


@torch.no_grad()
def quantize_lm_head(lm_head: ParallelLMHead, dtype: str) -> bool:
    """Pack ``lm_head`` for serving in ``dtype`` ("bf16" = do nothing).

    Returns True when the head was repacked. All packers are one-shot at
    weight-load time (CUDA-graph safe) and no-ops on already-packed heads.
    fp8_e4m3/fp8_e3m4 use 128x128 block scales (LM_HEAD_BLOCK), fp6_e2m3
    uses 32x32 FP32 block scales.
    """
    from sglang.srt.layers.fp6_lm_head import quantize_lm_head_to_fp6_b32
    from sglang.srt.layers.fp8_lm_head import (
        quantize_lm_head_to_e3m4_block128,
        quantize_lm_head_to_fp8_block128,
    )

    if dtype == "fp8_e4m3":
        return quantize_lm_head_to_fp8_block128(lm_head)
    if dtype == "fp8_e3m4":
        return quantize_lm_head_to_e3m4_block128(lm_head)
    if dtype == "fp6_e2m3":
        return quantize_lm_head_to_fp6_b32(lm_head)
    if dtype != "bf16":
        raise ValueError(
            f"unknown lm_head_dtype {dtype!r}, want one of {LM_HEAD_DTYPES}"
        )
    return False


def lm_head_block_size(dtype: str) -> int:
    """Scale-block edge (square blocks) of a packed lm-head dtype."""
    return LM_HEAD_BLOCK[dtype]
