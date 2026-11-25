"""SM12x (Blackwell GeForce / RTX PRO) MXFP4 W4A16 fused MoE runner.

Wraps the CuTe DSL kernel ``sglang.kernels.ops.moe.sm12x_mxfp4_w4a16_moe``
(fork of the vendored NVFP4 W4A16 kernel, flashinfer PR #5735): BF16
activations feed fused MXFP4-weight-only grouped GEMMs directly (no
activation quantization), which is the numerics an MXFP4 (OCP g32, E8M0
scales) W4A16 checkpoint is calibrated for.

The kernel's swizzled-scale buffers keep the NVFP4 16-group granularity:
each E8M0 byte of the checkpoint's [.., K/32] grid is duplicated so it
covers both 16-halves of its 32-block (see ``_scale_shape`` in the kernel
and ``build_sm12x_mxfp4_scales`` in mxfp4.py, which performs the swizzle
and duplication once at load time).

Token counts are rounded up to the same bounded bucket ladder as
``sm12x_nvfp4_bf16`` (bounded kernel-compile set), reusing
``SGLANG_SM12X_W4A16_MOE_MAX_TOKENS``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.moe.sm12x_mxfp4_w4a16_moe import (
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

logger = logging.getLogger(__name__)


@dataclass
class Sm12xMxfp4W4A16MoeQuantInfo(MoeQuantInfo):
    """Weight payload for the SM12x MXFP4 W4A16 fused MoE kernel.

    w13 rows are [gate; up] (gate first), E2M1-packed along K (two nibbles
    per byte, low nibble = even k). ``*_sf`` are uint8 E8M0 (bias-127) block
    scales in the 128x4-swizzled NVFP4-granularity layout (each checkpoint
    K/32 byte duplicated over its two 16-halves). ``*_gs`` are the per-expert
    FP32 global scales (raw ``weight_scale`` / ``weight_scale_2`` values).
    """

    w13: torch.Tensor  # uint8 [E, 2I, H/2], [gate; up] rows
    w13_sf: torch.Tensor  # uint8 [E, pad128(2I), pad4(H/16)] swizzled, duplicated
    w13_gs: torch.Tensor  # fp32 [E]
    w2: torch.Tensor  # uint8 [E, H, I/2]
    w2_sf: torch.Tensor  # uint8 [E, pad128(H), pad4(I/16)]
    w2_gs: torch.Tensor  # fp32 [E]


# Workspace cache keyed exactly like sm12x_nvfp4_bf16's: one entry per
# (bucket, geometry, device) serves every layer; the kernel restores its
# counters/combine slices to zero on exit, so buffers are reusable across
# launches and safe under CUDA-graph replay.
_WORKSPACE_CACHE: dict[tuple, list[torch.Tensor]] = {}


def _hybrid_buckets(max_num_tokens: int) -> tuple[int, ...]:
    """Same bucket ladder as sm12x_nvfp4_bf16 (kept duplicated on purpose so
    the two backends never share a helper that one of them later tweaks)."""
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
    # Above the ladder cap: round up to a power of two to keep the set of
    # compiled (token-count-specialized) kernels bounded (see the nvfp4 notes).
    if num_tokens > 2 * 16384:
        return -(-num_tokens // 8192) * 8192
    return 1 << (num_tokens - 1).bit_length()


def _pad_rows(x: torch.Tensor, rows: int, dtype: torch.dtype, pad_value: float):
    if x.shape[0] == rows:
        return x if x.dtype == dtype else x.to(dtype)
    out = torch.full((rows, *x.shape[1:]), pad_value, dtype=dtype, device=x.device)
    out[: x.shape[0]] = x.to(dtype)
    return out


def _workspace(
    bucket: int,
    h: int,
    i: int,
    n13: int,
    e: int,
    topk: int,
    device: torch.device,
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


@register_fused_func("none", "sm12x_mxfp4")
def fused_experts_none_to_sm12x_mxfp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: Sm12xMxfp4W4A16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert isinstance(quant_info, Sm12xMxfp4W4A16MoeQuantInfo), (
        f"Unexpected quant_info type for sm12x_mxfp4: {type(quant_info)}"
    )
    cfg = runner_config
    if cfg.num_fused_shared_experts:
        raise NotImplementedError(
            "sm12x_mxfp4 expects shared experts to run outside the kernel."
        )
    if cfg.apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported by sm12x_mxfp4."
        )

    hidden = dispatch_output.hidden_states
    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError("sm12x_mxfp4 expects unquantized BF16 activations.")
    if hidden.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"sm12x_mxfp4 requires BF16 activations, got {hidden.dtype}."
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

    # Routing slots whose expert id is out of range (padding) are skipped by
    # the kernel, so rows past ``t`` are never written.
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
