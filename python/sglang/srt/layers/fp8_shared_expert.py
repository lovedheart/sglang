"""FP8 blockwise-quantized shared experts (SM120).

The Qwen3.8-Flash-Next shared experts run once per layer per token outside the
grouped MoE GEMM. In BF16 their skinny projections (gate_up [1280, 2560],
down [2560, 640]) make cuBLAS pick latency-bound wmma/split-K configs:
measured 21.6 us for the 1.6 MB down_proj inside the decode graph, ~12x the
DRAM floor. Storing them as FP8 e4m3 with 128x128 block scales routes both
projections to the sm120 decode-warp / CUTLASS blockwise path already used by
the attention projections and the fp8 lm_head.

Same recipe and load-time contract as ``fp8_lm_head``: quantize once at
weight-load (never lazily) so the swap always precedes CUDA-graph capture.
Measured weight error: rel-L2 0.026 on every one of the 147 tensors (see
digest/tasks/shared-expert-quant-probe-1021.md), the same band as the
already-shipped attention FP8.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.utils.common import is_sm120_supported

logger = logging.getLogger(__name__)

FP8_E4M3_MAX = 448.0
FP8_BLOCK = 128


class Fp8Block128LinearMethod(QuantizeMethodBase):
    """Drop-in ``quant_method`` for a Linear holding fp8 block-128 weights.

    The weight (fp8) and ``weight_block_scale`` (fp32, [N/128, K/128]) live on
    the module, mirroring ``fp8_lm_head``; the apply path is the shared
    ``cutlass_w8a8_block_fp8_linear_with_fallback`` (decode-warp for M <= 16,
    CUTLASS otherwise).

    Loaders call ``process_weights_after_loading`` on every module that exposes
    a ``quant_method``, so this inherits ``QuantizeMethodBase`` and keeps those
    hooks as their no-ops: the repack already happened in
    ``quantize_linear_to_fp8_block128``.
    """

    def apply(self, module: torch.nn.Module, x: torch.Tensor, bias=None):
        from sglang.srt.layers.quantization.fp8_utils import (
            cutlass_w8a8_block_fp8_linear_with_fallback,
        )

        return cutlass_w8a8_block_fp8_linear_with_fallback(
            input=x,
            weight=module.weight,
            block_size=[FP8_BLOCK, FP8_BLOCK],
            weight_scale=module.weight_block_scale,
            bias=bias,
        )


@torch.no_grad()
def quantize_linear_to_fp8_block128(linear: torch.nn.Module) -> bool:
    """Swap ``linear.weight`` in place for an fp8 blockwise-128 copy.

    Returns False (module untouched) when already quantized, an unsupported
    dtype, a shape not divisible by 128 (this also covers TP shards that cut
    a 128-block), or a non-SM120 platform. Weight reload after a True return
    is not supported: the bf16 source is gone (same caveat as fp8_lm_head).
    """
    weight = getattr(linear, "weight", None)
    if weight is None or weight.dtype == torch.float8_e4m3fn:
        return False
    if weight.dtype not in (torch.bfloat16, torch.float16):
        logger.warning(
            "Skipping fp8 shared-expert quantization: unexpected dtype %s",
            weight.dtype,
        )
        return False
    n, k = weight.shape
    if n % FP8_BLOCK or k % FP8_BLOCK:
        logger.warning(
            "Skipping fp8 shared-expert quantization: shape (%d, %d) not "
            "divisible by %d",
            n,
            k,
            FP8_BLOCK,
        )
        return False
    if not is_sm120_supported():
        logger.warning(
            "--enable-fp8-shared-expert is SM120-only; leaving the weight in %s",
            weight.dtype,
        )
        return False

    blocks = (
        weight.detach()
        .float()
        .view(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
    )
    scales = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / FP8_E4M3_MAX
    quantized = (blocks / scales).to(torch.float8_e4m3fn).reshape(n, k)
    linear.weight = torch.nn.Parameter(quantized, requires_grad=False)
    linear.weight.e4m3_only = True  # online-quantized e4m3, not an e3m4 checkpoint
    linear.register_buffer(
        "weight_block_scale",
        scales.view(n // FP8_BLOCK, k // FP8_BLOCK).contiguous(),
        persistent=False,
    )
    linear.quant_method = Fp8Block128LinearMethod()
    return True


@torch.no_grad()
def quantize_shared_experts_fp8_block128(model: torch.nn.Module) -> int:
    """Quantize every ``*.mlp.shared_expert`` Qwen2MoeMLP under ``model``.

    Returns the number of projections swapped; the caller logs it, so the
    target and MTP draft paths can word their own lines. Modules whose MLP is
    fused into the grouped MoE (``shared_expert is None``) or whose down_proj
    is on the nvfp4 act+requant fusion are skipped with a warning.
    """
    count = 0
    for name, mlp in model.named_modules():
        if not name.endswith(".mlp.shared_expert"):
            continue
        if getattr(mlp, "_enable_silu_fp4_quant_fusion", False):
            logger.warning(
                "Shared expert %s is on the nvfp4 act+requant fusion path; "
                "leaving it unquantized.",
                name,
            )
            continue
        if getattr(mlp, "enable_shared_expert_fusion", False):
            # Weights were remapped into the grouped MoE (expert slot); the
            # local copies are dead — quantizing them would be wasted.
            continue
        for attr in ("gate_up_proj", "down_proj"):
            linear = getattr(mlp, attr, None)
            if linear is not None and quantize_linear_to_fp8_block128(linear):
                count += 1
    return count
