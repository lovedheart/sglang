"""w2a16_offload fused MoE runner: TS4 2-bit experts, CPU decode + GPU prefill.

One fused func routes by batch shape:

* ``t < SGLANG_W2A16_PREFILL_MIN_TOKENS`` (decode): the batch goes to the
  host through pinned staging + the AVX-512 ``ts4_a16_ffn`` kernel.  Under
  CUDA-graph capture the same job is expressed as D2H memcpy +
  ``cuLaunchHostFunc`` + H2D so decode graphs replay it natively.
* larger ``t`` (prefill): Triton grouped GEMMs over the same planes on the
  GPU (``w2a16.prefill``) — compute-bound there, bandwidth-bound on CPU.

Both arms share one weight archive (``w2a16.transfer``), so the numerics
match to fp32 accumulation order only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

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
class W2A16OffloadMoeQuantInfo(MoeQuantInfo):
    """Per-layer payload built by ModelOptNvFp4FusedMoEMethod.apply.

    ``cpu`` is the layer's W2A16CpuEngine (mmap'd planes + pinned staging);
    ``gpu_*`` are the resident device copies of the same planes for the
    prefill arm (None -> prefill also runs on CPU).
    """

    cpu: object
    gpu_w13: torch.Tensor | None = None
    gpu_s13: torch.Tensor | None = None
    gpu_w2: torch.Tensor | None = None
    gpu_s2: torch.Tensor | None = None
    E: int = 0
    H: int = 0
    I: int = 0
    apply_routed_scaling_factor: bool = False


@register_fused_func("none", "w2a16_offload")
def fused_experts_none_to_w2a16_offload(
    dispatch_output: StandardDispatchOutput,
    quant_info: W2A16OffloadMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert isinstance(quant_info, W2A16OffloadMoeQuantInfo), (
        f"Unexpected quant_info type for w2a16_offload: {type(quant_info)}"
    )
    if runner_config.num_fused_shared_experts:
        raise NotImplementedError(
            "w2a16_offload expects shared experts outside the kernel "
            "(use --disable-shared-experts-fusion)."
        )
    if runner_config.apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported by w2a16_offload."
        )

    hidden = dispatch_output.hidden_states
    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError(
            "w2a16_offload expects unquantized activations."
        )
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids
    if envs.SGLANG_W2A16_DEBUG.get():
        import sglang.srt.server_args as _sa
        if not hasattr(fused_experts_none_to_w2a16_offload, "_dbg0"):
            fused_experts_none_to_w2a16_offload._dbg0 = True
            logger.info("w2a16 dbg scaling: rsf=%s apply=%s",
                        runner_config.routed_scaling_factor,
                        quant_info.apply_routed_scaling_factor)
    if quant_info.apply_routed_scaling_factor and runner_config.routed_scaling_factor:
        topk_weights = topk_weights * runner_config.routed_scaling_factor

    t = hidden.shape[0]
    topk = topk_ids.shape[1]
    assert topk <= 8, "w2a16 staging rows are width 8"

    cpu_min = max(2, envs.SGLANG_W2A16_PREFILL_MIN_TOKENS.get())
    # prefill-sized batches go to the GPU arm; everything smaller (every
    # graph-captured decode step) goes to the CPU kernel.
    if quant_info.gpu_w13 is not None and t >= cpu_min:
        return _gpu_path(hidden, topk_ids, topk_weights, quant_info)
    return _cpu_path(hidden, topk_ids, topk_weights, quant_info)


def _cpu_path(hidden, topk_ids, topk_weights, quant_info):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    eng = quant_info.cpu
    t, h = hidden.shape
    ids = topk_ids.to(torch.int32)
    tw = topk_weights.to(torch.float32)
    if torch.cuda.is_current_stream_capturing() and hidden.is_cuda:
        out = torch.empty((t, h), dtype=torch.float32, device=hidden.device)
        esz = hidden.element_size()
        stream = torch.cuda.current_stream().cuda_stream
        eng.decode_capture(
            t, hidden.data_ptr(), esz, ids.data_ptr(), tw.data_ptr(),
            out.data_ptr(), stream, dtype=hidden.dtype,
        )
        return StandardCombineInput(hidden_states=out.to(hidden.dtype))
    out = torch.empty((t, h), dtype=torch.float32, device=hidden.device)
    eng.run_eager(hidden, ids, tw, out)
    if envs.SGLANG_W2A16_DEBUG.get():
        if not getattr(eng, "_dbg", False):
            eng._dbg = True
            from sglang.srt.layers.moe.w2a16.prefill import w2a16_gpu_forward
            g = w2a16_gpu_forward(
                hidden, ids, tw,
                *[q.contiguous().cuda() if q.device.type != "cuda" else q
                  for q in eng._planes],
                eng.E, eng.H, eng.I)
            logger.info("w2a16 dbg cpu-vs-gpu rel=%.3e xabs=%.3e ids=%s tw=%.3e",
                        ((g - out).abs().max() / out.abs().max().clamp(min=1e-30)).item(),
                        hidden.abs().mean().item(), ids.flatten()[:4].tolist(),
                        tw.mean().item())
    return StandardCombineInput(hidden_states=out.to(hidden.dtype))


def _gpu_path(hidden, topk_ids, topk_weights, quant_info):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.w2a16.prefill import w2a16_gpu_forward

    out = w2a16_gpu_forward(
        hidden, topk_ids.to(torch.int32), topk_weights.to(torch.float32),
        quant_info.gpu_w13, quant_info.gpu_s13,
        quant_info.gpu_w2, quant_info.gpu_s2,
        quant_info.E, quant_info.H, quant_info.I,
    )
    return StandardCombineInput(hidden_states=out.to(hidden.dtype))
