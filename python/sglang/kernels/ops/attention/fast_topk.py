from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_FAST_TOPK_SUPPORTED_K = (512, 2048)

# Persistent zero row-start buffers keyed by (device, batch): decode rows
# start at zero, and reusing one buffer removes the zeros fill kernel from
# every fast_topk call. The kernel only reads it and CUDA graphs need the
# stable address, which the cache provides.
_ZERO_ROW_STARTS: dict = {}


@cache_once
def _jit_fast_topk_module(topk: int) -> Module:
    """Compile and cache the JIT fast top-k module for a given k."""
    # Checks on the compile key live here, not in `fast_topk`: `cache_once`
    # keys on `topk`, so this runs once per specialisation.
    if topk not in _FAST_TOPK_SUPPORTED_K:
        raise RuntimeError(
            f"Unsupported topk {topk}. Supported: {_FAST_TOPK_SUPPORTED_K}"
        )
    args = make_cpp_args(topk, is_arch_support_pdl())
    return load_jit(
        "fast_topk",
        *args,
        cuda_files=["elementwise/fast_topk.cuh"],
        cuda_wrappers=[("fast_topk", f"FastTopKKernel<{args}>::run")],
    )


@cache_once
def _jit_fast_topk_v2_module(topk: int) -> Module:
    """Compile and cache the split-row JIT fast top-k module for a given k."""
    if topk not in _FAST_TOPK_SUPPORTED_K:
        raise RuntimeError(
            f"Unsupported topk {topk}. Supported: {_FAST_TOPK_SUPPORTED_K}"
        )
    args = make_cpp_args(topk, is_arch_support_pdl())
    return load_jit(
        "fast_topk_v2",
        *args,
        cuda_files=["elementwise/fast_topk_v2.cuh"],
        cuda_wrappers=[
            ("fast_topk_v2", f"FastTopKSplitKernel<{args}>::run"),
            ("fast_topk_v2_refine", f"FastTopKSplitKernel<{args}>::run_refine"),
        ],
    )


def fast_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    *,
    split_row: bool = True,
) -> torch.Tensor:
    """
    Per-row top-k selection over a fp32 score matrix.

    Row b selects the `topk` largest values in
    ``score[b, row_starts[b] : row_starts[b] + lengths[b])`` and returns their
    indices relative to ``row_starts[b]``. Slots beyond ``lengths[b]`` are -1.
    Output order within a row is unspecified (atomic collection order).

    Parameters
    ----------
    score      : CUDA fp32 tensor [B, L]
    lengths    : CUDA int32 tensor [B]
    topk       : number of indices per row; 512 or 2048
    row_starts : optional CUDA int32 tensor [B]; defaults to zeros
    split_row  : use the split-row kernel (v2), which spreads each long row's
                 stage-1 histogram over several blocks at small batch. The
                 kernel falls back to the one-block path for short rows or
                 large batches; set False to always use the v1 kernel. Both
                 kernels select the same index set.

    Returns
    -------
    CUDA int32 tensor [B, topk]
    """
    batch = score.shape[0]
    if row_starts is None:
        key = (score.device, batch)
        row_starts = _ZERO_ROW_STARTS.get(key)
        if row_starts is None:
            row_starts = torch.zeros(batch, dtype=torch.int32, device=score.device)
            if not torch.cuda.is_current_stream_capturing():
                # Never cache a capture-pool tensor: a graph may later free
                # its pool while this cache still hands the address out.
                # Warmup runs eagerly first, so capture always finds a cached
                # eager-addressed (permanently live) buffer.
                _ZERO_ROW_STARTS[key] = row_starts
    indices = score.new_empty((batch, topk), dtype=torch.int32)

    if split_row:
        module = _jit_fast_topk_v2_module(topk)
        # The in-kernel bitonic (torch.topk-exact emission) costs 2-6us/layer
        # at decode shapes; the QSA decode caller folds the order into the
        # expand kernel instead, so this follows SGLANG_QSA_SORTED_TOPK.
        from sglang.srt.environ import envs

        module.fast_topk_v2(
            score, row_starts, indices, lengths, envs.SGLANG_QSA_SORTED_TOPK.get()
        )
    else:
        module = _jit_fast_topk_module(topk)
        module.fast_topk(score, row_starts, indices, lengths)
    return indices


def fast_topk_refine_from_hist(
    score: torch.Tensor,
    lengths: torch.Tensor,
    coarse_hist: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Refine-only fast_topk over a coarse histogram built by another kernel.

    ``coarse_hist`` is a [B, 256] uint32 tensor holding per-row counts of the
    fp16-ordered high byte of each score (exactly fast_topk's stage-1 binning,
    e.g. the QSA scorer's fused epilogue). The refine consumes it and zeroes
    it back, so the buffer must start zeroed and stays self-cleaning under
    CUDA-graph replay. Rows at or below the split floor emit the identity
    prefix (naive semantics), matching ``fast_topk`` for short rows. Output
    order matches ``torch.topk`` (descending score, ascending index on ties).
    """
    batch = score.shape[0]
    row_starts = _ZERO_ROW_STARTS.get((score.device, batch))
    if row_starts is None:
        row_starts = torch.zeros(batch, dtype=torch.int32, device=score.device)
        if not torch.cuda.is_current_stream_capturing():
            _ZERO_ROW_STARTS[(score.device, batch)] = row_starts
    indices = score.new_empty((batch, topk), dtype=torch.int32)
    module = _jit_fast_topk_v2_module(topk)
    module.fast_topk_v2_refine(score, row_starts, indices, lengths, coarse_hist)
    return indices
