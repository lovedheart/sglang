"""SM12x (Blackwell GeForce / RTX PRO) NVFP4 W4A16 fused MoE runner.

Wraps the vendored CuTe DSL kernel from flashinfer PR #5735
(``sglang.kernels.ops.moe.sm12x_nvfp4_w4a16_moe``): BF16 activations are fed
straight into fused NVFP4-weight-only grouped GEMMs (no activation
quantization), which is the numerics a W4A16 checkpoint is calibrated for.

Token counts are rounded up to a bounded bucket ladder (bounded
kernel-compile set); token counts above
``SGLANG_SM12X_W4A16_MOE_MAX_TOKENS`` run unpadded at their exact size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.moe.sm12x_nvfp4_w4a16_moe import (
    allocate_workspace,
    run_moe_w4a16,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class Sm12xNvfp4W4A16MoeQuantInfo(MoeQuantInfo):
    """Weight payload for the SM12x W4A16 fused MoE kernel.

    w13 rows are [gate; up] (gate first), FP4-packed along K. ``*_gs`` are the
    per-expert FP4 block-scale global factors (FP32 [num_experts]): raw weight
    ``weight_scale_2`` values, *not* folded with an activation scale
    (activations stay BF16, so any static input scale would cancel). When the
    checkpoint gives gate and up separate global scales, their ratio rides on
    ``w2_gs`` instead -- see ``_sm12x_global_scales``.
    """

    w13: torch.Tensor  # uint8 [E, 2I, H/2]
    w13_sf: torch.Tensor  # e4m3/uint8 [E, pad128(2I), pad4(H/16)], 128x4 swizzled
    w13_gs: torch.Tensor  # fp32 [E]
    w2: torch.Tensor  # uint8 [E, H, I/2]
    w2_sf: torch.Tensor  # e4m3/uint8 [E, pad128(H), pad4(I/16)]
    w2_gs: torch.Tensor  # fp32 [E]


# Scratch buffers are keyed by (bucket tokens, hidden, inter, N13, experts,
# top_k, device): every layer shares the same geometry, so one entry per
# bucket serves the whole model. The kernel restores its counters/combine
# slices to zero on exit, so a workspace is reusable across launches (and
# safe under CUDA-graph replay); it must not be shared by two *concurrent*
# launches, which per-layer sequential execution guarantees.
_WORKSPACE_CACHE: dict[tuple, list[torch.Tensor]] = {}


def _workspace(
    bucket: int, h: int, i: int, n13: int, e: int, topk: int, device: torch.device
) -> list[torch.Tensor]:
    key = (bucket, h, i, n13, e, topk, device)
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        ws = allocate_workspace(
            num_tokens=bucket,
            hidden_size=h,
            intermediate_size=i,
            num_experts=e,
            top_k=topk,
            is_gated=n13 == 2 * i,
            device=device,
        )
        _WORKSPACE_CACHE[key] = ws
    return ws


def _hybrid_buckets(max_num_tokens: int) -> tuple[int, ...]:
    """Bucket ladder mirroring flashinfer's hybrid policy: powers of 2 up to
    256, steps of 256 to 2048, steps of 512 to 4096, powers of 2 beyond."""
    buckets = []
    m = 1
    while m <= min(max_num_tokens, 256):
        buckets.append(m)
        m *= 2
    for start, end, step in ((256, 2048, 256), (2048, 4096, 512)):
        m = start + step
        while m <= min(max_num_tokens, end):
            buckets.append(m)
            m += step
    m = 8192
    while m <= max_num_tokens:
        buckets.append(m)
        m *= 2
    if not buckets or buckets[-1] != max_num_tokens:
        buckets.append(max_num_tokens)
    return tuple(sorted(set(buckets)))


def _map_to_bucket(num_tokens: int, max_num_tokens: int) -> int:
    for b in _hybrid_buckets(max_num_tokens):
        if num_tokens <= b:
            return b
    return max_num_tokens


def _pad_rows(x: torch.Tensor, rows: int, dtype: torch.dtype, pad_value: float):
    """Cast to ``dtype`` and pad to ``rows`` total rows along dim 0."""
    if x.shape[0] == rows:
        return x if x.dtype == dtype else x.to(dtype)
    out = torch.full((rows, *x.shape[1:]), pad_value, dtype=dtype, device=x.device)
    out[: x.shape[0]] = x.to(dtype)
    return out


@register_fused_func("none", "sm12x_nvfp4_bf16")
def fused_experts_none_to_sm12x_nvfp4_bf16(
    dispatch_output: StandardDispatchOutput,
    quant_info: Sm12xNvfp4W4A16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert isinstance(quant_info, Sm12xNvfp4W4A16MoeQuantInfo), (
        f"Unexpected quant_info type for sm12x_nvfp4_bf16: {type(quant_info)}"
    )
    cfg = runner_config
    if cfg.num_fused_shared_experts:
        raise NotImplementedError(
            "sm12x_nvfp4_bf16 expects shared experts to run outside the kernel."
        )
    if cfg.apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported by sm12x_nvfp4_bf16."
        )

    hidden = dispatch_output.hidden_states
    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError(
            "sm12x_nvfp4_bf16 expects unquantized BF16 activations."
        )
    if hidden.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"sm12x_nvfp4_bf16 requires BF16 activations, got {hidden.dtype}."
        )

    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids

    t, h = hidden.shape
    e, n13, _ = quant_info.w13.shape
    i = quant_info.w2.shape[2] * 2
    topk = topk_ids.shape[1]
    max_tokens = max(1, envs.SGLANG_SM12X_W4A16_MOE_MAX_TOKENS.get())

    if t > max_tokens:
        # Above the ladder cap (e.g. an oversized prefill chunk): run at the
        # exact token count rather than padding further.
        bucket = t
    else:
        bucket = _map_to_bucket(t, max_tokens)
    if bucket != t:
        ids = _pad_rows(topk_ids, bucket, torch.int32, float(e))
        weights = _pad_rows(topk_weights, bucket, torch.float32, 0.0)
        x = _pad_rows(hidden, bucket, torch.bfloat16, 0.0)
    else:
        ids = topk_ids.to(torch.int32)
        weights = topk_weights.to(torch.float32)
        x = hidden

    # Out rows past ``t`` (padding slots) are never written: routing slots
    # with an out-of-range expert id are skipped by the kernel.
    out = torch.empty((bucket, h), dtype=torch.bfloat16, device=hidden.device)
    workspace = _workspace(bucket, h, i, n13, e, topk, hidden.device)
    run_moe_w4a16(
        x,
        ids,
        weights,
        quant_info.w13,
        quant_info.w13_sf,
        quant_info.w13_gs,
        quant_info.w2,
        quant_info.w2_sf,
        quant_info.w2_gs,
        out,
        workspace=workspace,
    )
    return StandardCombineInput(hidden_states=out[:t])
