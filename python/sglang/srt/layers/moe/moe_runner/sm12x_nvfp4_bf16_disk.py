"""SM12x NVFP4 MoE runner backend that streams expert payloads from disk.

Interface- and numerics-identical to ``sm12x_nvfp4_bf16``: same dispatch/
combine contract, same CuTe-DSL ``run_moe_w4a16`` kernel, bit-comparable on
resident slots. What differs is where the FP4 weight payload comes from --
the original safetensors files, paged through a VRAM slot cache -- so only
the experts a batch actually routes to cost I/O (see plan.md phases for the
registry/slot-cache design).

Layers whose payload stayed VRAM-resident (``SGLANG_SM12X_DISK_RESIDENT_
LAYERS``) delegate to the resident fused func verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.moe.moe_runner.sm12x_nvfp4_bf16 import (
    Sm12xNvfp4W4A16MoeQuantInfo,
    fused_experts_none_to_sm12x_nvfp4_bf16,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@register_fused_func("none", "sm12x_nvfp4_bf16_disk")
def fused_experts_none_to_sm12x_nvfp4_bf16_disk(
    dispatch_output: StandardDispatchOutput,
    quant_info: Sm12xNvfp4W4A16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    if quant_info.w13.device.type != "meta":
        # Resident-tier layer (RESIDENT_LAYERS knob): verbatim resident path.
        return fused_experts_none_to_sm12x_nvfp4_bf16(
            dispatch_output, quant_info, runner_config
        )
    return _fused_streamed(dispatch_output, quant_info, runner_config)


@eager_on_graph
def _bcg_stream_fill(
    topk_ids: torch.Tensor, quant_info: Sm12xNvfp4W4A16MoeQuantInfo
) -> torch.Tensor:
    """Slot fill for one launch, as a graph break: on capture the host work
    runs between segments; on replay it re-runs eagerly, reading the ids
    buffer in place (its contents are per-step, its address is the graph's).
    """
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_gather import get_slot_cache
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_registry import (
        get_expert_registry,
    )

    cache = get_slot_cache(get_expert_registry(), quant_info, quant_info.w13.shape[0])
    return cache.stream_weights(quant_info.layer_id, topk_ids.to("cpu"), quant_info)


def _fused_streamed(
    dispatch_output: StandardDispatchOutput,
    quant_info: Sm12xNvfp4W4A16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.kernels.ops.moe.sm12x_nvfp4_w4a16_moe import run_moe_w4a16
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_gather import get_slot_cache
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_registry import (
        get_expert_registry,
    )
    from sglang.srt.layers.moe.moe_runner.sm12x_nvfp4_bf16 import (
        _map_to_bucket,
        _pad_rows,
        _workspace,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    cfg = runner_config
    if cfg.num_fused_shared_experts or cfg.apply_router_weight_on_input:
        raise NotImplementedError(
            "sm12x_nvfp4_bf16_disk expects shared experts and router weights "
            "to be applied outside the kernel."
        )
    hidden = dispatch_output.hidden_states
    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError("sm12x_nvfp4_bf16_disk expects BF16 activations.")
    if hidden.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"sm12x_nvfp4_bf16_disk requires BF16 activations, got {hidden.dtype}."
        )
    if (
        quant_info.a13_gs is not None
        and hidden.shape[0] >= envs.SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS.get()
    ):
        raise NotImplementedError(
            "the W4A4 prefill arm is not wired for the disk backend yet."
        )
    if quant_info.layer_id is None:
        raise RuntimeError("streamed sm12x disk layer is missing its layer_id.")

    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids

    t, h = hidden.shape
    e, n13, _ = quant_info.w13.shape
    topk = topk_ids.shape[1]

    cache = get_slot_cache(get_expert_registry(), quant_info, e)
    if (
        torch.cuda.is_current_stream_capturing()
        and not is_in_breakable_cuda_graph()
        and cache.ingraph is None
    ):
        raise NotImplementedError(
            "sm12x_nvfp4_bf16_disk needs an eager break for its host fill "
            "path: serve with --cuda-graph-backend-decode breakable (graph "
            "segments stay, the fill becomes an eager break), or disabled, "
            "or keep the streamed layers resident via "
            "SGLANG_SM12X_DISK_RESIDENT_LAYERS, or page from a pinned "
            "mirror with SGLANG_SM12X_DISK_INGRAPH=1 (full graph)."
        )
    pp = None
    if cache.ingraph is not None:
        # Device-side LRU + pinned-mirror fill: no host round trip, so the
        # whole step (fill included) is capturable into a full graph.
        id_map = cache.ingraph.fill(quant_info.layer_id, topk_ids, quant_info)
    else:
        if cache.pong is not None and cache.pong.usable(
            quant_info.layer_id,
            t,
            is_in_breakable_cuda_graph() or torch.cuda.is_current_stream_capturing(),
        ):
            # Prefill-sized eager launch: whole-layer ping-pong buffer, no
            # per-expert fill and no ids D2H (identity expert->slot map).
            pp = cache.pong.prep(quant_info.layer_id, quant_info)
        if pp is None:
            if is_in_breakable_cuda_graph():
                # Under a breakable graph capture this records an eager
                # break (and a replay fn that re-runs the fill against the
                # current ids buffer); outside capture it is a plain call.
                id_map = _bcg_stream_fill(topk_ids, quant_info)
            else:
                id_map = cache.stream_weights(
                    quant_info.layer_id, topk_ids.to("cpu"), quant_info
                )

    max_tokens = max(1, envs.SGLANG_SM12X_W4A16_MOE_MAX_TOKENS.get())
    bucket = t if t > max_tokens else _map_to_bucket(t, max_tokens)
    if bucket != t:
        ids = _pad_rows(topk_ids, bucket, torch.int32, float(e))
        weights = _pad_rows(topk_weights, bucket, torch.float32, 0.0)
        x = _pad_rows(hidden, bucket, torch.bfloat16, 0.0)
    else:
        ids = topk_ids.to(torch.int32)
        weights = topk_weights.to(torch.float32)
        x = hidden

    # expert id -> slot (the padding row e maps to sentinel slot K, which the
    # kernel skips exactly like the resident path skips out-of-range ids).
    # The ping-pong buffers hold experts at their own ids (row e == pad).
    remapped = ids.clamp(max=e) if pp is not None else id_map[ids.long().clamp(max=e)]

    out = torch.empty((bucket, h), dtype=torch.bfloat16, device=hidden.device)
    if pp is None:
        wts = (
            cache.slot_w13,
            cache.slot_w13_sf,
            cache.slot_w13_gs,
            cache.slot_w2,
            cache.slot_w2_sf,
            cache.slot_w2_gs,
        )
        slots = cache.k
    else:
        wts = (
            pp["w13"],
            pp["w13_sf"],
            pp["w13_gs"],
            pp["w2"],
            pp["w2_sf"],
            pp["w2_gs"],
        )
        slots = pp["k"]
    workspace = _workspace(bucket, h, n13 // 2, n13, slots, topk, hidden.device)
    run_moe_w4a16(x, remapped, weights, *wts, out, workspace=workspace)
    if pp is not None:
        cache.pong.mark(quant_info.layer_id)  # gates the buffer's next load
    return StandardCombineInput(hidden_states=out[:t])
