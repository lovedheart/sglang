"""NVFP4 W4A16 LM head copy for speculative draft runners (SM120).

NEXTN draft steps call the LM head once per draft step and use the logits
only to pick the next proposal (greedy top-1). The head GEMM is pure
weight-streaming, so the draft cost scales with head bytes: the target's
fp8 blockwise head reads ~640 MB per call. This module builds a second,
draft-only head copy quantized to NVFP4 with Marlin W4A16 weights
(~320 MB + scales), which measures ~1.7x faster per draft-step call.

The target's head is never touched, so emitted logits and greedy outputs
are unchanged; only proposals (and hence accept length) can shift. Built
once when the draft runner adopts the target head, always before CUDA
graph capture.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.srt.utils.common import is_sm120_supported
from torch import nn

logger = logging.getLogger(__name__)

_FP4_MAX = 6.0
_E4M3_MAX = 448.0
# Rows processed per pass; the head is [248320, K], built in row chunks to cap
# transient memory at a few hundred MB next to a nearly-full KV pool.
_HEAD_BUILD_CHUNK = 30720


class NvFp4DraftHeadMethod:
    """Stand-in for a LinearMethod so LogitsProcessor._compute_lm_head routes
    through quant_method.apply, like any quantized head."""

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )


def _block_source_block_size(source_head: nn.Module) -> tuple[int, int]:
    n, k = source_head.weight.shape
    blocks = source_head.weight_block_scale.shape
    return n // blocks[0], k // blocks[1]


@torch.no_grad()
def _dequant_rows(source_head: nn.Module, row_start: int, row_end: int) -> torch.Tensor:
    weight = source_head.weight
    if weight.dtype in (torch.bfloat16, torch.float16):
        return weight[row_start:row_end].to(torch.bfloat16)
    block_n, block_k = _block_source_block_size(source_head)
    w8 = weight[row_start:row_end].float()
    sc = source_head.weight_block_scale[
        row_start // block_n : row_end // block_n, :
    ].float()
    rows = row_end - row_start
    w8 = w8.view(rows // block_n, block_n, weight.shape[1] // block_k, block_k)
    sc = sc.view(rows // block_n, 1, weight.shape[1] // block_k, 1)
    # (row_blocks, rows_in_block, col_blocks, cols_in_block) is already the
    # row-major [rows, k] layout, so reshape (no permute) is exact.
    return (w8 * sc).to(torch.bfloat16).reshape(rows, weight.shape[1])


@torch.no_grad()
def build_nvfp4_draft_head(source_head: nn.Module) -> nn.Module | None:
    """Build a draft-only NVFP4 W4A16 copy of ``source_head``.

    Returns the head module, or None when unsupported (non-SM120, unknown
    weight form, or a shape the block-quantized source or Marlin cannot
    tile); callers keep the shared head.
    """
    weight = getattr(source_head, "weight", None)
    if weight is None:
        return None
    is_fp8 = weight.dtype == torch.float8_e4m3fn
    if not is_fp8 and weight.dtype not in (torch.bfloat16, torch.float16):
        logger.warning(
            "nvfp4 draft head: unsupported source dtype %s; keeping shared head",
            weight.dtype,
        )
        return None
    if is_fp8 and not hasattr(source_head, "weight_block_scale"):
        logger.warning("nvfp4 draft head: fp8 source without block scales")
        return None
    if not is_sm120_supported():
        logger.warning("--enable-nvfp4-draft-lm-head is SM120-only; ignored")
        return None

    n, k = weight.shape
    if n % 128 != 0 or k % 128 != 0:
        logger.warning(
            "nvfp4 draft head: shape (%d, %d) not 128-tiled; keeping shared head",
            n,
            k,
        )
        return None

    from flashinfer import SfLayout, nvfp4_quantize

    # Pass 1: global amax of the dequantized weights (sets the fp4 encodings).
    amax = torch.zeros((), dtype=torch.float32, device=weight.device)
    for i in range(0, n, _HEAD_BUILD_CHUNK):
        chunk = _dequant_rows(source_head, i, min(i + _HEAD_BUILD_CHUNK, n))
        torch.maximum(amax, chunk.abs().amax().float(), out=amax)
        del chunk
    if amax <= 0:
        return None
    weight_scale_2 = (amax / (_E4M3_MAX * _FP4_MAX)).to(torch.float32)
    inv_scale = (1.0 / weight_scale_2).float()

    # Pass 2: encode. nvfp4_quantize packs [rows, k//2] codes plus per-16
    # e4m3 block scales in linear layout.
    weight_packed = torch.empty(n, k // 2, dtype=torch.uint8, device=weight.device)
    weight_sf = torch.empty(n, k // 16, dtype=torch.uint8, device=weight.device)
    for i in range(0, n, _HEAD_BUILD_CHUNK):
        rows = min(i + _HEAD_BUILD_CHUNK, n) - i
        chunk = _dequant_rows(source_head, i, i + rows)
        codes, sf = nvfp4_quantize(
            chunk.contiguous(), inv_scale, sfLayout=SfLayout.layout_linear
        )
        weight_packed[i : i + rows] = codes.reshape(rows, k // 2)
        weight_sf[i : i + rows] = sf.reshape(rows, k // 16)
        del chunk, codes, sf

    head = nn.Module()
    head.weight = nn.Parameter(weight_packed, requires_grad=False)
    head.weight_scale = nn.Parameter(
        weight_sf.view(torch.float8_e4m3fn), requires_grad=False
    )
    head.weight_global_scale = nn.Parameter(
        weight_scale_2.clone().to(torch.float32), requires_grad=False
    )
    head.output_size_per_partition = n
    head.input_size_per_partition = k
    head.params_dtype = torch.bfloat16
    head.num_embeddings = getattr(source_head, "num_embeddings", n)
    head.embedding_dim = getattr(source_head, "embedding_dim", k)

    class _Group16:
        group_size = 16

    head.quant_config = _Group16()
    head.quant_method = NvFp4DraftHeadMethod()
    prepare_nvfp4_layer_for_marlin(head)
    return head
