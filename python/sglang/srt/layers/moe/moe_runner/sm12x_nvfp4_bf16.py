"""SM12x (Blackwell GeForce / RTX PRO) NVFP4 W4A16 fused MoE runner.

Wraps the vendored CuTe DSL kernel from flashinfer PR #5735
(``sglang.kernels.ops.moe.sm12x_nvfp4_w4a16_moe``): BF16 activations are fed
straight into fused NVFP4-weight-only grouped GEMMs (no activation
quantization), which is the numerics a W4A16 checkpoint is calibrated for.

Token counts are rounded up to a bounded bucket ladder (bounded
kernel-compile set); token counts above
``SGLANG_SM12X_W4A16_MOE_MAX_TOKENS`` run unpadded at their exact size.

With ``SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS`` set, batches at or above that
token count instead run flashinfer's ``moe_nvfp4_w4a4`` sibling kernel on the
same weights (NVFP4-quantized activations, faster for prefill).
"""

from __future__ import annotations

import logging
import math
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

logger = logging.getLogger(__name__)


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

    w13: torch.Tensor  # uint8 [E, 2I, H/2], [gate; up] rows
    w13_sf: torch.Tensor  # e4m3/uint8 [E, pad128(2I), pad4(H/16)], 128x4 swizzled
    w13_gs: torch.Tensor  # fp32 [E]
    w2: torch.Tensor  # uint8 [E, H, I/2]
    w2_sf: torch.Tensor  # e4m3/uint8 [E, pad128(H), pad4(I/16)]
    w2_gs: torch.Tensor  # fp32 [E]
    # Optional W4A4 prefill arm (flashinfer moe_nvfp4_w4a4, sibling of the
    # vendored W4A16 kernel reading the same canonical weights). When set,
    # batches with at least SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS tokens
    # quantize activations to NVFP4 using these quantization multipliers
    # (1 / input_scale); see _prepare_sm12x_w4a16_weights for their derivation.
    a13_gs: torch.Tensor | None = None
    a2_gs: torch.Tensor | None = None


# Scratch buffers are keyed by (bucket tokens, hidden, inter, N13, experts,
# top_k, device): every layer shares the same geometry, so one entry per
# bucket serves the whole model. The kernel restores its counters/combine
# slices to zero on exit, so a workspace is reusable across launches (and
# safe under CUDA-graph replay); it must not be shared by two *concurrent*
# launches, which per-layer sequential execution guarantees.
_WORKSPACE_CACHE: dict[tuple, list[torch.Tensor]] = {}
# One flat scratch arena for all W4A4 buckets (workspace bytes scale linearly
# with the token count, so buckets carved out of one max-size buffer replace
# the per-bucket ladder; views are 256B-aligned so TMA descriptors stay legal,
# and the arena is allocated once and never resized, so captured graphs keep
# valid addresses).
_ARENA_ALIGN = 256
_ARENA_SLACK = 1 << 20
_W4A4_ARENAS: dict[torch.device, list] = {}  # device -> [uint8 buffer, next offset]


def _ws_bytes(shapes) -> int:
    return sum(
        (math.prod(s) * d.itemsize + _ARENA_ALIGN - 1) // _ARENA_ALIGN * _ARENA_ALIGN
        for s, d in shapes
    )


def _w4a4_upper_tokens(bucket: int) -> int | None:
    """Upper bound on token counts the W4A4 arm can request (prefill batches
    dominate; above the bucket-ladder cap sizes round up to powers of two)."""
    try:
        from sglang.srt.server_args import get_server_args

        t = int(get_server_args().max_prefill_tokens or 0)
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None
    t = max(t, int(envs.SGLANG_SM12X_W4A16_MOE_MAX_TOKENS.get()))
    return max(t, 1 << (bucket - 1).bit_length())


def _w4a4_workspace(
    bucket: int,
    h: int,
    i: int,
    n13: int,
    e: int,
    topk: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Per-bucket W4A4 workspace views carved from the shared arena."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4 import (
        allocate_workspace,
    )

    shapes = None
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4 import (
            _entry,
        )

        _, shapes, _ = _entry(bucket, h, i, n13, e, topk, torch.device(device))
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        logger.debug("W4A4 arena disabled (%s); per-bucket workspace", exc)
    if shapes is not None:
        need = _ws_bytes(shapes)
        arena = _W4A4_ARENAS.get(device)
        if arena is None:
            upper = _w4a4_upper_tokens(bucket)
            if upper is not None and upper >= bucket:
                cap = _ws_bytes(_entry(upper, h, i, n13, e, topk, torch.device(device))[1])
                if cap >= need:
                    arena = [
                        torch.empty(cap + _ARENA_SLACK, dtype=torch.uint8, device=device),
                        0,
                    ]
                    _W4A4_ARENAS[device] = arena
        if arena is not None and arena[0].numel() - arena[1] >= need:
            off = arena[1]
            ws = []
            for shape, dtype in shapes:
                n = math.prod(shape)
                nbytes = (n * dtype.itemsize + _ARENA_ALIGN - 1) // _ARENA_ALIGN * _ARENA_ALIGN
                view = arena[0][off : off + n * dtype.itemsize].view(dtype)
                ws.append(view.view(tuple(shape)))
                off += nbytes
            arena[1] = off
            return ws
    return allocate_workspace(
        num_tokens=bucket,
        hidden_size=h,
        intermediate_size=i,
        num_experts=e,
        top_k=topk,
        is_gated=n13 == 2 * i,
        device=device,
    )


def _workspace(
    bucket: int,
    h: int,
    i: int,
    n13: int,
    e: int,
    topk: int,
    device: torch.device,
    w4a4: bool = False,
) -> list[torch.Tensor]:
    key = (bucket, h, i, n13, e, topk, device, w4a4)
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        if w4a4:
            ws = _w4a4_workspace(bucket, h, i, n13, e, topk, device)
        else:
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
    # Above the ladder cap the kernel compiles per *exact* token count, and
    # each new count costs a ~13 s cute.compile (prefill batches never repeat
    # a count, so an unpadded exact-size launch stalls every batch). Round up
    # to the next power of two instead: the compile set stays bounded (one
    # compile per octave) at the cost of <=2x padding on oversized batches.
    if num_tokens > 2 * 16384:
        # Power-of-two rounding would more than double the (already huge)
        # fp32 workspace; align to 8192 so padding stays bounded.
        return -(-num_tokens // 8192) * 8192
    return 1 << (num_tokens - 1).bit_length()


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

    w4a4_min = envs.SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS.get()
    # Large batches (typically prefill chunks) go through the W4A4 sibling
    # kernel, ~1.4x faster than W4A16 here, at the cost of quantizing
    # activations to NVFP4.
    use_w4a4 = w4a4_min > 0 and quant_info.a13_gs is not None and t >= w4a4_min

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
    if use_w4a4:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4 import (
            run_moe_w4a4,
        )

        workspace = _workspace(bucket, h, i, n13, e, topk, hidden.device, w4a4=True)
        run_moe_w4a4(
            x,
            ids,
            weights,
            quant_info.w13,
            quant_info.w13_sf,
            quant_info.w13_gs,
            quant_info.w2,
            quant_info.w2_sf,
            quant_info.w2_gs,
            quant_info.a13_gs,
            quant_info.a2_gs,
            out,
            workspace=workspace,
        )
    else:
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
