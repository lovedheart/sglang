"""Fused int8 quantize/dequantize for the mamba checkpoint pool.

Contract (pure format conversion, with slot indirection): symmetric int8
quantization of ``[L, N, H, V, K]`` state rows where the amax group is the
second-to-last dim (V), so one scale per ``(slot, layer, head, k)``. This is
the checkpoint pool's store/dequant path; the eager reference is
``Int8CheckpointStore.quantize/dequantize`` in
``sglang.srt.mem_cache.mamba_checkpoint_pool``, which these kernels match bit
for bit (round-half-even, scale rounded to the state dtype before the
division, amax clamped at 1e-8). No existing kernel in this repo covers this
contract: the per-token/group int8 and fp8 kernels reduce along the LAST dim
and quantize fresh activations, not slot-indexed cache rows.

NaN inputs diverge (torch propagates NaN through amax, Triton's max folds it
away); recurrent-state pools are finite by construction, so this is the
stated coverage rather than a bug. Kernels take the state slot ids as GPU
tensors, so the caller never materializes the gather/cast copies the eager
path pays (12 torch kernels and ~24x the bytes per store).

Exact IEEE arithmetic (div_rn, rint) keeps the result bit-identical to the
eager torch path; Triton's ``/`` would lower to reciprocal-multiply and flip
rounding on ties.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_QMAX = tl.constexpr(127.0)
_RECIP_QMAX = tl.constexpr(1.0 / 127.0)  # python double -> fp32 constant, as torch does


@triton.jit
def _int8_mamba_ckpt_quant_kernel(
    src_ptr,  # [L, N_src, H, V, K] state dtype, entry dims contiguous
    src_slots_ptr,  # [NS] int64
    dst_q_ptr,  # [L, N_dst, H, V, K] int8
    dst_s_ptr,  # [L, N_dst, H, 1, K] scale dtype
    dst_slots_ptr,  # [NS] int64
    H,
    V,
    K,
    stride_l,
    stride_slot,
    stride_h,
    stride_v,
    d_stride_l,
    d_stride_slot,
    d_stride_h,
    d_stride_v,
    s_stride_l,
    s_stride_slot,
    s_stride_h,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_lh = tl.program_id(1)
    l = pid_lh // H
    h = pid_lh % H
    src_slot = tl.load(src_slots_ptr + pid_s).to(tl.int64)
    dst_slot = tl.load(dst_slots_ptr + pid_s).to(tl.int64)

    src_base = (
        src_ptr
        + l.to(tl.int64) * stride_l
        + src_slot * stride_slot
        + h.to(tl.int64) * stride_h
    )
    k_off = tl.arange(0, BLOCK_K)
    k_mask = k_off < K

    amax = tl.zeros([BLOCK_K], dtype=tl.float32)
    for v0 in range(0, V, BLOCK_V):
        v = v0 + tl.arange(0, BLOCK_V)
        tile = tl.load(
            src_base + v[:, None].to(tl.int64) * stride_v + k_off[None, :],
            mask=(v[:, None] < V) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(tile), axis=0))

    # torch lowers ``amax / 127.0`` to a multiply by the fp32 reciprocal,
    # so bit parity requires the same reciprocal-multiply, not IEEE division.
    scale_f = tl.maximum(amax, 1e-8) * _RECIP_QMAX
    # The division uses the scale rounded to the state's storage dtype, and
    # the plane keeps a second rounding to its own dtype (the eager path's
    # two casts, kept bit-exact when the dtypes differ).
    scale_state = scale_f.to(src_ptr.dtype.element_ty)
    scale_div = scale_state.to(tl.float32)
    dst_s_base = (
        dst_s_ptr
        + l.to(tl.int64) * s_stride_l
        + dst_slot * s_stride_slot
        + h.to(tl.int64) * s_stride_h
    )
    tl.store(dst_s_base + k_off, scale_state.to(dst_s_ptr.dtype.element_ty), k_mask)

    dst_q_base = (
        dst_q_ptr
        + l.to(tl.int64) * d_stride_l
        + dst_slot * d_stride_slot
        + h.to(tl.int64) * d_stride_h
    )
    for v0 in range(0, V, BLOCK_V):
        v = v0 + tl.arange(0, BLOCK_V)
        mask = (v[:, None] < V) & k_mask[None, :]
        tile = tl.load(
            src_base + v[:, None].to(tl.int64) * stride_v + k_off[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.extra.libdevice.rint(tl.math.div_rn(tile, scale_div[None, :]))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(
            dst_q_base + v[:, None].to(tl.int64) * d_stride_v + k_off[None, :],
            q.to(tl.int8),
            mask,
        )


@triton.jit
def _int8_mamba_ckpt_dequant_kernel(
    q_ptr,  # [L, N_q, H, V, K] int8
    q_slots_ptr,  # [NS] int64
    s_ptr,  # [L, N_q, H, 1, K] scale dtype
    dst_ptr,  # [L, N_dst, H, V, K] state dtype, entry dims contiguous
    dst_slots_ptr,  # [NS] int64
    H,
    V,
    K,
    q_stride_l,
    q_stride_slot,
    q_stride_h,
    q_stride_v,
    s_stride_l,
    s_stride_slot,
    s_stride_h,
    d_stride_l,
    d_stride_slot,
    d_stride_h,
    d_stride_v,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_lh = tl.program_id(1)
    l = pid_lh // H
    h = pid_lh % H
    q_slot = tl.load(q_slots_ptr + pid_s).to(tl.int64)
    dst_slot = tl.load(dst_slots_ptr + pid_s).to(tl.int64)

    k_off = tl.arange(0, BLOCK_K)
    k_mask = k_off < K
    s_base = (
        s_ptr
        + l.to(tl.int64) * s_stride_l
        + q_slot * s_stride_slot
        + h.to(tl.int64) * s_stride_h
    )
    scale = tl.load(s_base + k_off, mask=k_mask, other=0.0).to(tl.float32)
    q_base = (
        q_ptr
        + l.to(tl.int64) * q_stride_l
        + q_slot * q_stride_slot
        + h.to(tl.int64) * q_stride_h
    )
    d_base = (
        dst_ptr
        + l.to(tl.int64) * d_stride_l
        + dst_slot * d_stride_slot
        + h.to(tl.int64) * d_stride_h
    )
    for v0 in range(0, V, BLOCK_V):
        v = v0 + tl.arange(0, BLOCK_V)
        mask = (v[:, None] < V) & k_mask[None, :]
        q = tl.load(
            q_base + v[:, None].to(tl.int64) * q_stride_v + k_off[None, :],
            mask=mask,
            other=0,
        )
        out = (q.to(tl.float32) * scale[None, :]).to(dst_ptr.dtype.element_ty)
        tl.store(
            d_base + v[:, None].to(tl.int64) * d_stride_v + k_off[None, :],
            out,
            mask,
        )


def _entry_contiguous(t: torch.Tensor) -> bool:
    # entry dims (H, V, K tail) contiguous: the kernels walk them by stride
    return t.stride(4) == 1 and t.stride(3) == t.shape[4]


def _blocks(V: int, K: int):
    block_k = triton.next_power_of_2(K)
    if K > 1024 or V > 4096:
        return None
    return 32, block_k


def int8_mamba_ckpt_quant(src, src_slots, dst_q, dst_scale, dst_slots) -> bool:
    """Fused quantize of ``src[:, src_slots]`` rows into the checkpoint
    planes at ``dst_slots`` (pairwise slots). Returns False (caller falls
    back to the eager path) when the shapes or devices are unsupported."""
    if not src.is_cuda or not _entry_contiguous(src) or not _entry_contiguous(dst_q):
        return False
    if dst_scale.stride(4) != 1:
        return False
    if src_slots.numel() != dst_slots.numel():
        # The eager path fails loudly on unpaired slots; keep False => eager.
        return False
    _, _, H, V, K = src.shape
    L = src.shape[0]
    blocks = _blocks(V, K)
    if blocks is None:
        return False
    BLOCK_V, BLOCK_K = blocks
    n = src_slots.numel()
    _int8_mamba_ckpt_quant_kernel[(n, L * H)](
        src,
        src_slots,
        dst_q,
        dst_scale,
        dst_slots,
        H,
        V,
        K,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        src.stride(3),
        dst_q.stride(0),
        dst_q.stride(1),
        dst_q.stride(2),
        dst_q.stride(3),
        dst_scale.stride(0),
        dst_scale.stride(1),
        dst_scale.stride(2),
        BLOCK_V=BLOCK_V,
        BLOCK_K=BLOCK_K,
    )
    return True


def int8_mamba_ckpt_dequant(q, q_slots, scale, dst, dst_slots) -> bool:
    """Fused dequantize of checkpoint rows at ``q_slots`` into
    ``dst[:, dst_slots]`` (pairwise slots). False => caller falls back."""
    if not q.is_cuda or not _entry_contiguous(q) or not _entry_contiguous(dst):
        return False
    if scale.stride(4) != 1:
        return False
    if q_slots.numel() != dst_slots.numel():
        return False  # unpaired slots: let the eager path raise
    L, _, H, V, K = q.shape
    blocks = _blocks(V, K)
    if blocks is None:
        return False
    BLOCK_V, BLOCK_K = blocks
    n = q_slots.numel()
    _int8_mamba_ckpt_dequant_kernel[(n, L * H)](
        q,
        q_slots,
        scale,
        dst,
        dst_slots,
        H,
        V,
        K,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        dst.stride(3),
        BLOCK_V=BLOCK_V,
        BLOCK_K=BLOCK_K,
    )
    return True
