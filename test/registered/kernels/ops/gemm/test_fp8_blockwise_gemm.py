import sys
from typing import Optional, Type

import pytest
import torch

from sglang.kernels.ops.gemm.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b",
    runner_config="1-gpu-small",
)


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def baseline_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: Type[torch.dtype],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)
    output = torch.mm(
        (scale_a * a.to(dtype=torch.float32)), (scale_b * b.to(dtype=torch.float32))
    ).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output


def _test_accuracy_once(M, N, K, out_dtype, device):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn).t()
    scale_a_group_shape = (1, 128)
    scale_b_group_shape = (128, 128)
    scale_a_shape = scale_shape(a_fp8.shape, scale_a_group_shape)
    scale_b_shape = scale_shape(b_fp8.shape, scale_b_group_shape)
    scale_a = torch.randn(scale_a_shape, device=device, dtype=torch.float32) * 0.001
    scale_b = torch.randn(scale_b_shape, device=device, dtype=torch.float32) * 0.001
    scale_a = scale_a.t().contiguous().t()
    scale_b = scale_b.t().contiguous().t()
    o = baseline_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype)
    o1 = fp8_blockwise_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype)
    rtol = 0.02
    atol = 1
    torch.testing.assert_close(o, o1, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not is_sm120_supported(), reason="fp8_blockwise_scaled_mm requires SM120 (>= 12.0)"
)
# M=256 is the only case that reaches the 64-wide token-tile arm (128 < M <= 256):
# 127/128 take the 32-wide tile and 512 the non-swapAB one, so without it that arm
# would have no accuracy coverage at all.
@pytest.mark.parametrize("M", [1, 3, 5, 32, 48, 64, 127, 128, 256, 512, 1024, 4096])
@pytest.mark.parametrize("N", [128, 512, 1024, 4096, 8192])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_accuracy(M, N, K, out_dtype):
    _test_accuracy_once(M, N, K, out_dtype, "cuda")


# Decode shapes newly routed to the split-K warp arm (m <= 8 any width;
# m <= 16 narrow): the arm must stay bit-stable across CUDA-graph replays
# (ascending split-K reduction, no atomics) and exact against the fp32
# dequant reference.
@pytest.mark.skipif(
    not is_sm120_supported(), reason="fp8_blockwise_scaled_mm requires SM120 (>= 12.0)"
)
@pytest.mark.parametrize(
    "M, N, K",
    [(1, 2048, 2560), (4, 2560, 2048), (8, 512, 2560), (4, 16384, 2560), (16, 2560, 6144)],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_warp_arm_stability(M, N, K, out_dtype):
    device = "cuda"
    torch.manual_seed(0)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    a = ((torch.rand(M, K, device=device) - 0.5) * 2 * fp8_max).to(torch.float8_e4m3fn)
    b = ((torch.rand(N, K, device=device) - 0.5) * 2 * fp8_max).to(torch.float8_e4m3fn).t()
    scale_a = torch.randn(M, cdiv(K, 128), device=device, dtype=torch.float32) * 0.001
    scale_b = torch.randn(cdiv(K, 128), cdiv(N, 128), device=device, dtype=torch.float32) * 0.001
    # The blockwise kernels take col-major scale tensors (loader convention).
    scale_a = scale_a.t().contiguous().t()
    scale_b = scale_b.t().contiguous().t()
    ref = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype)
    out = fp8_blockwise_scaled_mm(a, b, scale_a, scale_b, out_dtype)
    # Tighter than the generic sweep (rtol 0.02, atol 1): the warp arm's
    # fp32 partials reduce in ascending k order, so deviation from the
    # reference is rounding-only.
    torch.testing.assert_close(out, ref, rtol=0.008, atol=0.25)
    first = fp8_blockwise_scaled_mm(a, b, scale_a, scale_b, out_dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fp8_blockwise_scaled_mm(a, b, scale_a, scale_b, out_dtype)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            out2 = fp8_blockwise_scaled_mm(a, b, scale_a, scale_b, out_dtype)
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out2.view(torch.uint16), first.view(torch.uint16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
