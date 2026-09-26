"""Host-side entry for native NVFP4 paged decode on the SM120 FA4 kernel.

Self-contained (does not reuse the bf16 plan caches): compiles one
FlashAttentionForwardSm120 specialization with ``kv_fp4=True`` per static
argument signature. v1 scope: paged decode with pack_gqa, no split-KV,
causal=False. The packed KV tensors are uint8 (page_size, head_dim/2,
kv_heads, pages); SF tensors are uint8 e4m3 (page_size, head_dim/16,
kv_heads, pages). Global pool scales are folded by the caller: the K scale
into ``softmax_scale`` and the V scale as a post-multiply on the output.
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.ops.attention.fa4_sm120.flash_fwd import (
    FlashAttentionForwardSm120,
)
from sglang.kernels.ops.attention.flash_attn.cute.utils import AuxData

_compile_cache: dict[tuple, object] = {}


@cute.jit
def _launch_fp4_paged(
    kernel: cutlass.Constexpr,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mSFk: cute.Tensor,
    mSFv: cute.Tensor,
    mO: cute.Tensor,
    mSeqUsedK: cute.Tensor,
    mPageTable: cute.Tensor,
    softmax_scale: Float32,
    mDump: cute.Tensor,
    stream: cuda.CUstream,
):
    kernel(
        mQ,
        mK,
        mV,
        mO,
        None,
        softmax_scale,
        None,
        None,
        None,
        mSeqUsedK,
        mPageTable,
        None,
        None,
        None,
        None,
        AuxData(),
        None,
        Int32(0),
        mSFk,
        mSFv,
        mDump,
        stream=stream,
    )


def _build_kernel(qhead_per_kvhead: int, pack_gqa: bool, dump_k=None) -> FlashAttentionForwardSm120:
    head_dim = 256
    tile_m, tile_n = 16, 64
    num_threads = FlashAttentionForwardSm120.get_fwd_num_threads(
        head_dim, head_dim, tile_m, tile_n, paged_kv=True
    )
    num_stages = FlashAttentionForwardSm120.get_fwd_num_stages(
        head_dim, head_dim, tile_m, tile_n
    )
    kernel = FlashAttentionForwardSm120(
        cutlass.BFloat16,
        head_dim,
        head_dim,
        qhead_per_kvhead,
        is_causal=False,
        is_local=False,
        pack_gqa=pack_gqa,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=num_stages,
        num_threads=num_threads,
        Q_in_regs=False,
        is_split_kv=False,
        paged_kv=True,
        kv_fp4=True,
    )
    return kernel


def _signature(t: torch.Tensor) -> tuple:
    return (tuple(t.shape), tuple(t.stride()), t.dtype)



def _cute_kv_args(q, k_packed, v_packed, sf_k, sf_v, out, seqused_k, page_table, scale, dump_k, stream):
    """Raw torch mode orders ((pages, page, h, d...)); the kernel's __call__
    performs the (s, d, h, b) rearrangement itself, matching the bf16 path."""
    return (
        from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=3),
        from_dlpack(k_packed, assumed_align=16).mark_layout_dynamic(leading_dim=3),
        from_dlpack(v_packed, assumed_align=16).mark_layout_dynamic(leading_dim=3),
        from_dlpack(sf_k, assumed_align=16).mark_layout_dynamic(leading_dim=3),
        from_dlpack(sf_v, assumed_align=16).mark_layout_dynamic(leading_dim=3),
        from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=3),
        from_dlpack(seqused_k, assumed_align=4).mark_layout_dynamic(leading_dim=0),
        from_dlpack(page_table, assumed_align=4).mark_layout_dynamic(leading_dim=1),
        scale,
        (
            from_dlpack(dump_k, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            if dump_k is not None
            else None
        ),
        stream,
    )


def fp4_paged_decode(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    sf_k: torch.Tensor,
    sf_v: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    v_global_scale: Optional[torch.Tensor] = None,
    dump_k: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """q: (B, 1, Hq, D) bf16; k/v_packed: (pages, page, Hkv, D/2) uint8;
    sf_*: (pages, page, Hkv, D/16) uint8 e4m3; page_table: (B, blocks) int32;
    seqused_k: (B,) int32 tokens."""
    assert q.dtype == torch.bfloat16 and q.ndim == 4 and q.shape[1] == 1
    batch, _, h_q, dim = q.shape
    pages, page, h_kv, half = k_packed.shape
    assert k_packed.dtype == torch.uint8 and half * 2 == dim
    assert page % 64 == 0 and dim == 256
    pack_gqa = h_q // h_kv > 1
    if out is None:
        out = torch.empty_like(q)
    key = (
        "fa4-sm120-fp4-v1",
        _signature(q),
        _signature(k_packed),
        _signature(sf_k),
        _signature(page_table),
        _signature(seqused_k),
        pack_gqa,
        dump_k is not None,
    )
    compiled = _compile_cache.get(key)
    if compiled is None:
        kernel = _build_kernel(h_q // h_kv, pack_gqa, dump_k)
        compiled = cute.compile(
            _launch_fp4_paged,
            kernel,
            *_cute_kv_args(
                q, k_packed, v_packed, sf_k, sf_v, out, seqused_k, page_table,
                Float32(softmax_scale),
                dump_k,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            ),
        )
        _compile_cache[key] = compiled
    compiled(
        *_cute_kv_args(
            q, k_packed, v_packed, sf_k, sf_v, out, seqused_k, page_table,
            Float32(softmax_scale),
            dump_k,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
    )
    if v_global_scale is not None:
        out.mul_(v_global_scale)
    return out
