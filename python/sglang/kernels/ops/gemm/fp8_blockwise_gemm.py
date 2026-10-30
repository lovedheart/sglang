from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.common import is_sm120_supported
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _fp8_blockwise_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]


@cache_once
def _jit_fp8_blockwise_module() -> Module:
    """Compile and cache the SM120 fp8 blockwise GEMM module (handles fp16 + bf16)."""
    if not is_sm120_supported():
        raise RuntimeError(
            "fp8_blockwise_scaled_mm JIT kernel requires SM120 (Blackwell)."
        )
    return load_jit(
        "fp8_blockwise_scaled_mm",
        cuda_files=["gemm/fp8_blockwise/fp8_blockwise_scaled_mm_entry.cuh"],
        cuda_wrappers=[
            ("fp8_blockwise_scaled_mm", "fp8_blockwise_scaled_mm"),
            (
                "sm120_fp8_blockwise_warp_fused_act",
                "sm120_fp8_blockwise_warp_fused_act",
            ),
        ],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_fp8_blockwise_cuda_flags(),
    )


@register_custom_op(
    op_name="fp8_blockwise_scaled_mm",
    mutates_args=["out"],
)
def _fp8_blockwise_scaled_mm_custom_op(
    out: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
) -> None:
    module = _jit_fp8_blockwise_module()
    module.fp8_blockwise_scaled_mm(out, mat_a, mat_b, scales_a, scales_b)


@lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def sm120_fp8_blockwise_fused_act_supported(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> bool:
    """Gate for the fused activation-quantizing decode GEMM.

    True only when the plain pipeline (quant + ``sm120_fp8_decode_warp`` +
    reduce) would have taken the warp-kernel branch, so the fused kernel
    replaces exactly the shapes that measurement picked it on: the same
    ``warp_pr`` / ``warp_flashnext`` / ``warp_small`` arms from
    ``sm120_fp8_blockwise_dispatch_shape``, mirrored here because the fused
    path skips the activation quant and cannot fall back afterwards.
    Outputs are bit-identical to the two-kernel path.
    """
    from sglang.srt.environ import envs

    if not envs.SGLANG_ENABLE_SM120_FUSED_ACT_QUANT.get():
        return False
    if not is_sm120_supported():
        return False
    if x2d.dtype not in (torch.bfloat16, torch.float16):
        return False
    if not (x2d.is_cuda and weight.is_cuda and weight_scale.is_cuda):
        return False
    if (
        x2d.dim() != 2
        or weight.dim() != 2
        or not x2d.is_contiguous()
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
        or weight_scale.dtype != torch.float32
        or not weight_scale.is_contiguous()
    ):
        return False
    m, k = x2d.shape
    n = weight.shape[0]
    if m <= 0 or m > 16 or k % 128 != 0 or k > 8192 or n % 128 != 0:
        return False
    if tuple(weight_scale.shape) != (n // 128, k // 128):
        return False
    num_sms = _num_sms(x2d.device.index if x2d.device.index is not None else 0)
    warp_pr = (
        n <= 32 * num_sms and (m <= 8 or k >= 6144) and n >= 4096 and k >= 4096
    )
    warp_flashnext = n <= 32 * num_sms and (k >= 6144 or n <= 1024)
    warp_small = n <= 32768 and (m <= 8 or n <= 4096)
    return warp_pr or warp_flashnext or warp_small


@debug_kernel_api
def sm120_fp8_blockwise_warp_fused_act(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Blockwise-FP8 linear for decode shapes with in-kernel activation quant.

    Bit-identical to ``fp8_blockwise_scaled_mm(sglang_per_token_group_quant_
    fp8_row_padded(x), ...)`` on the shapes this is gated to; callers must
    pass the same ``(x, weight [n, k] row-major, weight_scale [n/128, k/128]
    row-major)`` the two-kernel path would have used.
    """
    out = torch.empty((x2d.shape[0], weight.shape[0]), dtype=out_dtype, device=x2d.device)
    _sm120_fp8_blockwise_warp_fused_act_custom_op(out, x2d, weight, weight_scale)
    return out


@register_custom_op(
    op_name="sm120_fp8_blockwise_warp_fused_act",
    mutates_args=["out"],
)
def _sm120_fp8_blockwise_warp_fused_act_custom_op(
    out: torch.Tensor,
    x2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> None:
    module = _jit_fp8_blockwise_module()
    module.sm120_fp8_blockwise_warp_fused_act(out, x2d, weight, weight_scale)


@debug_kernel_api
def fp8_blockwise_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """FP8 e4m3 block-wise scaled matmul on SM120."""
    assert out_dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"out_dtype must be Half or BFloat16, got {out_dtype}"

    out = torch.empty(
        (mat_a.shape[0], mat_b.shape[1]),
        dtype=out_dtype,
        device=mat_a.device,
    )
    _fp8_blockwise_scaled_mm_custom_op(out, mat_a, mat_b, scales_a, scales_b)
    return out
