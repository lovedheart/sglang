"""Grouped Gemma RMSNorm fused with the MXFP8 activation-quant epilogue.

Pair to ``ops.gemm.hc_mix_fp8.fused_hc_mix_fp8_w8a8``: it produces the e4m3
activations + ue8m0 scales that kernel consumes, saving one full read of the
normalized tensor per call. Output is bit-identical to running the plain
``grouped_gemma_rmsnorm`` followed by ``_mxfp8_quant_act_kernel`` (pinned by
test/registered/kernels/ops/layernorm/test_grouped_gemma_rmsnorm_mxfp8.py).

Coverage: contiguous CUDA input, bf16/fp16, ``group_size`` a multiple of 512,
SM100+ (Blackwell) only -- the quant epilogue's 32-element group mapping needs
the 32B-per-thread vector layout. Rejected cases must fall back to the two
unfused kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_grouped_gemma_rmsnorm_mxfp8_module(
    group_size: int, dtype: torch.dtype
) -> Module:
    """Compile and cache the JIT grouped Gemma RMSNorm + MXFP8-quant module."""
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"Unsupported dtype {dtype}. Supported: bfloat16, float16")
    if group_size <= 0 or group_size % 512 != 0:
        raise RuntimeError(
            f"Unsupported group_size {group_size}. Must be a multiple of 512."
        )
    args = make_cpp_args(group_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "grouped_gemma_rmsnorm_mxfp8",
        *args,
        cuda_files=["elementwise/grouped_gemma_rmsnorm_mxfp8.cuh"],
        cuda_wrappers=[
            (
                "grouped_gemma_rmsnorm_mxfp8",
                f"GroupedGemmaRMSNormMXFP8Kernel<{args}>::run",
            )
        ],
    )


def grouped_gemma_rmsnorm_mxfp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Grouped Gemma RMSNorm with an MXFP8 quant epilogue.

    Returns ``(normed, quant, scale)`` where ``normed`` is exactly what
    :func:`grouped_gemma_rmsnorm` would have returned (same bits), ``quant``
    is ``normed`` quantized per 32-element group to ``torch.float8_e4m3fn``
    with shape ``input.shape``, and ``scale`` is the ``uint8`` ue8m0
    exponents with shape ``(input.numel() // hidden_size, hidden_size // 32)``
    -- bit-identical to feeding ``normed`` through
    ``ops.gemm.hc_mix_fp8._mxfp8_quant_act_kernel`` on the flat
    ``[rows, hidden_size]`` view.
    """
    hidden_size = input.size(-1)
    # Validate the caller's tensors, then view (never reshape): a
    # non-contiguous input must fail loudly, not be copied into a temporary.
    if not input.is_contiguous():
        raise ValueError("grouped_gemma_rmsnorm_mxfp8: input must be contiguous")
    if out is not None and not out.is_contiguous():
        raise ValueError("grouped_gemma_rmsnorm_mxfp8: out must be contiguous")
    x = input.view(-1, hidden_size)
    if out is None:
        normed = torch.empty_like(x)
    else:
        normed = out.view(-1, hidden_size)
    rows = x.shape[0]
    quant_u8 = torch.empty((rows, hidden_size), dtype=torch.uint8, device=x.device)
    sf = torch.empty((rows, hidden_size // 32), dtype=torch.uint8, device=x.device)

    module = _jit_grouped_gemma_rmsnorm_mxfp8_module(group_size, input.dtype)
    module.grouped_gemma_rmsnorm_mxfp8(x, weight, normed, quant_u8, sf, eps)
    quant = quant_u8.view(torch.float8_e4m3fn).view(input.shape)
    return normed.view(input.shape), quant, sf
