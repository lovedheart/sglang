# SPDX-License-Identifier: Apache-2.0
"""Latency benchmark for the SM12x NVFP4 W4A16 fused MoE runner backend.

Qwen3.8-Flash-Next-sized geometry (512 experts, H=2560, I=640, top-10) with
synthetic NVFP4 weights; sweeps token counts from decode to large prefill."""

import pytest
import torch

from sglang.kernels.jit.benchmark import marker
from sglang.srt.layers.moe.moe_runner.sm12x_nvfp4_bf16 import (
    Sm12xNvfp4W4A16MoeQuantInfo,
    fused_experts_none_to_sm12x_nvfp4_bf16,
)
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
if cc not in ((12, 0), (12, 1)):
    pytest.skip("sm12x_nvfp4_bf16 kernels require SM120/SM121", allow_module_level=True)

E, H, I, TOPK = 512, 2560, 640, 10


def _make_quant_info():
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda")
    w13_sf = (torch.rand(E, 2 * I, H // 16, device="cuda") * 0.2).to(
        torch.float8_e4m3fn
    )
    gs13 = (10 ** torch.empty(E, device="cuda").uniform_(-3, -2)).float()
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device="cuda")
    w2_sf = (torch.rand(E, H, I // 16, device="cuda") * 0.2).to(torch.float8_e4m3fn)
    gs2 = (10 ** torch.empty(E, device="cuda").uniform_(-3, -2)).float()
    return Sm12xNvfp4W4A16MoeQuantInfo(
        w13=w13,
        w13_sf=swizzle_blockscale(w13_sf),
        w13_gs=gs13,
        w2=w2,
        w2_sf=swizzle_blockscale(w2_sf),
        w2_gs=gs2,
    )


Q = _make_quant_info()


class _TopK:
    def __init__(self, ids, wts):
        self.topk_ids = ids
        self.topk_weights = wts
        self.router_logits = None


class _Dispatch:
    def __init__(self, x, ids, wts):
        self.hidden_states = x
        self.hidden_states_scale = None
        self.topk_output = _TopK(ids, wts)


class _Cfg:
    num_fused_shared_experts = 0
    apply_router_weight_on_input = False


def run_layer(x, ids, wts):
    return fused_experts_none_to_sm12x_nvfp4_bf16(
        _Dispatch(x, ids, wts), Q, _Cfg()
    ).hidden_states


@marker.parametrize("T", [1, 4, 16, 64, 256, 1024, 4096], [1, 16, 256])
@marker.benchmark("impl", ["sm12x_w4a16"], unit="us")
def benchmark(T: int, impl: str):
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device="cuda")
    wts = torch.softmax(torch.rand(T, TOPK, device="cuda"), dim=-1)
    return marker.do_bench(
        lambda: run_layer(x, ids, wts),
        memory_args=(x, Q.w13, Q.w2),
        memory_output="out",
        graph_clone_args="all",
    )


if __name__ == "__main__":
    benchmark.run()
