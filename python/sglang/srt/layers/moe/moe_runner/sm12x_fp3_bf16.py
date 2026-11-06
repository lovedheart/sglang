"""SM12x FP3 (W3A16) fused MoE runner.

Runs the FP3 checkpoints produced by the QAT pipeline: 3-bit NVFP4-codebook
weight payloads with per-group-of-16 e4m3 block scales and per-expert FP32
global scales, against BF16 activations (no activation quantization). The
Triton GEMM dequantizes weight tiles inside the K-loop, so no expanded
BF16 copy of the weights is ever materialized.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

logger = logging.getLogger(__name__)


class Sm12xFp3W3A16MoeQuantInfo(MoeQuantInfo):
    """Packed FP3 expert weights for the W3A16 dequant experts (grouped SIMT
    GEMV at decode sizes, Triton tensor-core GEMM at prefill sizes).

    w13:   uint8 [E, 2I, K*3//8] LSB3-repacked payload (gate rows [0:I], up
    rows [I:2I]; bytes 3g..3g+2 hold codes 8g..8g+7 LSB-first, see
    sglang.kernels.ops.moe.fp3_w3a16_moe_kernels.repack_fp3_w3a16_lsb3)
    w13_sf: uint8 [E, 2I, K//16] raw sf bytes (bit0 = codebook mirror flag)
    w13_gs: fp32  [E, 2] per-expert (gate, up) global scales
    w2:    uint8 [E, K, I*3//8], w2_sf uint8 [E, K, I//16], w2_gs fp32 [E]
    """

    def __init__(
        self,
        w13: torch.Tensor,
        w13_sf: torch.Tensor,
        w13_gs: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        w2_gs: torch.Tensor,
    ) -> None:
        self.w13 = w13
        self.w13_sf = w13_sf
        self.w13_gs = w13_gs
        self.w2 = w2
        self.w2_sf = w2_sf
        self.w2_gs = w2_gs


@register_fused_func("none", "sm12x_fp3_bf16")
def fused_experts_none_to_sm12x_fp3_bf16(
    dispatch_output,
    quant_info: Sm12xFp3W3A16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
):
    from sglang.kernels.ops.moe.fp3_w3a16_moe_kernels import fused_experts_fp3_w3a16
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert isinstance(quant_info, Sm12xFp3W3A16MoeQuantInfo), (
        f"Unexpected quant_info type for sm12x_fp3_bf16: {type(quant_info)}"
    )
    cfg = runner_config
    if cfg.num_fused_shared_experts:
        raise NotImplementedError(
            "sm12x_fp3_bf16 expects shared experts to run outside the kernel."
        )
    if cfg.apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported by sm12x_fp3_bf16."
        )
    if cfg.no_combine:
        raise NotImplementedError("no_combine is not supported by sm12x_fp3_bf16.")
    if cfg.activation != "silu" or not cfg.is_gated:
        raise NotImplementedError(
            f"sm12x_fp3_bf16 supports only gated silu, got activation="
            f"{cfg.activation!r}, is_gated={cfg.is_gated}."
        )

    hidden = dispatch_output.hidden_states
    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError(
            "sm12x_fp3_bf16 expects unquantized BF16 activations."
        )
    if hidden.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"sm12x_fp3_bf16 requires BF16 activations, got {hidden.dtype}."
        )

    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids

    if quant_info.w13_gs.dim() == 1:
        # Checkpoints that share one global scale across gate/up: expand to
        # the [E, 2] layout the kernel selects per output half.
        w13_gs = quant_info.w13_gs.unsqueeze(1).expand(-1, 2).contiguous()
    else:
        w13_gs = quant_info.w13_gs

    output = fused_experts_fp3_w3a16(
        hidden_states=hidden,
        w13=quant_info.w13,
        w13_sf=quant_info.w13_sf,
        w13_gs=w13_gs,
        w2=quant_info.w2,
        w2_sf=quant_info.w2_sf,
        w2_gs=quant_info.w2_gs,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=cfg.activation,
        routed_scaling_factor=cfg.routed_scaling_factor,
    )
    return StandardCombineInput(hidden_states=output)
