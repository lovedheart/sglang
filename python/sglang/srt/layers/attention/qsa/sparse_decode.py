"""Fused gather + dequant + sparse-decode attention for the QSA decode path.

Replaces the valid_counts + strided gather + paged-decode kernel triple at
small batch with two Triton kernels:

* ``_sparse_decode_split`` -- grid (rows, kv_heads, splits).  Each program
  owns a 64-token slice of one row's top-k list, gathers KV straight out of
  the pool (topk index -> slot via ``req_to_token``), dequantizes NVFP4 in
  registers (or loads BF16), and reduces a GQA-packed [q_per_kv, head_dim]
  tile with online softmax.
* ``_sparse_decode_combine`` -- vectorized log-sum-exp merge of the splits.

The paged decode kernels (trtllm-gen / xqa / cudnn) all cost a fixed ~10.5 us
per call at bs<=4 on Blackwell (verified against a contiguous dense decode at
the same shape), while the memory floor of the sparse decode is <1 us.  This
kernel trades that fixed overhead for extra CTAs and runs roughly 2x faster
for small row counts; large-batch decode stays on the paged path, which is at
its bandwidth floor there.

Token semantics match ``_forward_trtllm_sparse``: a token participates iff
``0 <= idx < seq_len``; fully padded rows (CUDA graph filler rows) emit zeros.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl


# Tokens per split-program iteration; one 64-token page is the GEMM sub-tile.
_CHUNK = 64
_MAX_SPLITS = 33  # 33 * 64 = 2112 >= final_topk = token_topk + ratio - 1 (2051)


@triton.jit
def _dequant_nvfp4(
    packed_ptr,
    sf_ptr,
    slots,
    kvh,
    CHUNK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    SF_STRIDE: tl.constexpr,
):
    """codes * per-block SF (fp32, unscaled by the global scale)."""
    offs_p = tl.arange(0, HEAD_DIM // 2)
    packed = tl.load(
        packed_ptr
        + slots[:, None] * SLOT_STRIDE
        + kvh * (HEAD_DIM // 2)
        + offs_p[None, :]
    )
    # PTX cvt.rn.f16x2.e2m1x2 (sm_100a/sm_120a+): one instruction per packed
    # byte; bitwise-identical to the e2m1 LUT (verified).  Falls back would
    # be needed for pre-Blackwell, but the gate keeps this path Blackwell-only.
    lo, hi = tl.inline_asm_elementwise(
        "{ .reg .b8 q; .reg .b32 t; cvt.u8.u32 q, $2; cvt.rn.f16x2.e2m1x2 t, q; mov.b32 {$0, $1}, t; }",
        "=h,=h,r",
        [packed],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    vals = tl.interleave(lo, hi).to(tl.float32)  # low nibble = even column
    offs_g = tl.arange(0, HEAD_DIM // 16)
    sf = (
        tl.load(
            sf_ptr
            + slots[:, None] * SF_STRIDE
            + kvh * (HEAD_DIM // 16)
            + offs_g[None, :]
        )
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    for _ in tl.static_range(4):
        sf = tl.interleave(sf, sf)
    return vals * sf


@triton.jit
def _load_rows_bf16(
    buf_ptr,
    slots,
    kvh,
    CHUNK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_DIM)
    return tl.load(
        buf_ptr + slots[:, None] * SLOT_STRIDE + kvh * HEAD_DIM + offs_d[None, :]
    )


@triton.jit
def _sparse_decode_split(
    q_ptr,  # [rows, q_heads, head_dim] bf16
    k_ptr,  # pool K: bf16 [slots, kvh, D] or packed u8 [slots, kvh, D/2]
    v_ptr,
    ksf_ptr,  # fp8_e4m3 [slots, kvh, D/16] (nvfp4 only)
    vsf_ptr,
    kgs_ptr,  # fp32 [1] global K scale (nvfp4 only)
    idx_ptr,  # [rows, TOPK] int32 logical token positions, -1 pad
    r2t_ptr,  # [req_slots, R2T_STRIDE] int32
    req_ptr,  # [rows] row -> request slot
    seqlen_ptr,  # [rows] int32
    part_o_ptr,  # [rows, kvh, SPLITS, HEADS_P, head_dim] fp32
    part_ml_ptr,  # [rows, kvh, SPLITS, HEADS_P, 2] fp32
    sm_scale,
    TOPK: tl.constexpr,
    R2T_STRIDE: tl.constexpr,
    SLOT_STRIDE_K: tl.constexpr,
    SLOT_STRIDE_V: tl.constexpr,
    SF_STRIDE: tl.constexpr,
    PER_SPLIT: tl.constexpr,
    SPLITS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    H_PER_KV: tl.constexpr,
    HEADS_P: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK: tl.constexpr,
    NVFP4: tl.constexpr,
):
    row = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)

    offs_h = tl.arange(0, HEADS_P)
    offs_d = tl.arange(0, HEAD_DIM)
    hmask = offs_h < H_PER_KV
    q_ptrs = (
        q_ptr
        + row * (KV_HEADS * H_PER_KV * HEAD_DIM)
        + (kvh * H_PER_KV + offs_h)[:, None] * HEAD_DIM
        + offs_d[None, :]
    )
    q_tile = tl.load(q_ptrs, mask=hmask[:, None], other=0.0)

    req = tl.load(req_ptr + row).to(tl.int64)
    seqlen = tl.load(seqlen_ptr + row)
    r2t_base = req * R2T_STRIDE
    if NVFP4:
        k_gs = tl.load(kgs_ptr) * sm_scale
    else:
        k_gs = sm_scale

    m_i = tl.full([HEADS_P], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([HEADS_P], dtype=tl.float32)
    acc = tl.zeros([HEADS_P, HEAD_DIM], dtype=tl.float32)

    offs_t = tl.arange(0, CHUNK)
    tok0 = split * PER_SPLIT
    for c in range(0, PER_SPLIT, CHUNK):
        toks = tok0 + c + offs_t
        tmask = toks < TOPK
        idx = tl.load(idx_ptr + row * TOPK + toks, mask=tmask, other=-1)
        # Same valid mask as the paged path's qwen_sparse_valid_counts_triton.
        # It matters beyond -1 padding: the decode scorer writes only
        # [0, compressed_len) of its empty logits buffer, so fast_topk can
        # select an uninitialized token past the scored context; without the
        # seqlen term that stale token gathers a random KV row instead of
        # being dropped.
        valid = tmask & (idx >= 0) & (idx < seqlen)
        idx_c = tl.where(valid, idx, 0).to(tl.int64)
        slot = tl.load(r2t_ptr + r2t_base + idx_c, mask=valid, other=0).to(tl.int64)
        if NVFP4:
            k_tile = _dequant_nvfp4(
                k_ptr, ksf_ptr, slot, kvh, CHUNK, HEAD_DIM, SLOT_STRIDE_K, SF_STRIDE
            )
            v_tile = _dequant_nvfp4(
                v_ptr, vsf_ptr, slot, kvh, CHUNK, HEAD_DIM, SLOT_STRIDE_V, SF_STRIDE
            )
        else:
            k_tile = _load_rows_bf16(k_ptr, slot, kvh, CHUNK, HEAD_DIM, SLOT_STRIDE_K)
            v_tile = _load_rows_bf16(v_ptr, slot, kvh, CHUNK, HEAD_DIM, SLOT_STRIDE_V)
        s = tl.dot(q_tile, tl.trans(k_tile.to(tl.bfloat16))) * k_gs
        s = tl.where(valid[None, :] & tmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(tl.where(m_i == float("-inf"), 0.0, m_i) - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile.to(tl.bfloat16))
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    po = part_o_ptr + ((row * KV_HEADS + kvh) * SPLITS + split) * (HEADS_P * HEAD_DIM)
    tl.store(
        po + offs_h[:, None] * HEAD_DIM + offs_d[None, :],
        acc / l_safe[:, None],
        mask=hmask[:, None],
    )
    pm = part_ml_ptr + ((row * KV_HEADS + kvh) * SPLITS + split) * (HEADS_P * 2)
    tl.store(pm + offs_h * 2, m_i, mask=hmask)
    tl.store(pm + offs_h * 2 + 1, l_i, mask=hmask)


@triton.jit
def _sparse_decode_combine(
    part_o_ptr,
    part_ml_ptr,
    out_ptr,  # [rows, q_heads, head_dim] bf16
    vgs_ptr,  # fp32 [1] global V scale (nvfp4 only)
    SPLITS: tl.constexpr,
    SPLITS_P: tl.constexpr,
    KV_HEADS: tl.constexpr,
    H_PER_KV: tl.constexpr,
    HEADS_P: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NVFP4: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // H_PER_KV
    hloc = h % H_PER_KV
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SPLITS_P)
    smask = offs_s < SPLITS
    pm = (
        part_ml_ptr
        + ((row * KV_HEADS + kvh) * SPLITS + offs_s) * (HEADS_P * 2)
        + hloc * 2
    )
    m_s = tl.load(pm, mask=smask, other=float("-inf"))
    l_s = tl.load(pm + 1, mask=smask, other=0.0)
    m = tl.max(m_s, axis=0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    w = tl.where(m_s == float("-inf"), 0.0, tl.exp(m_s - m_safe)) * l_s
    denom = tl.sum(w, axis=0)
    d_safe = tl.where(denom == 0.0, 1.0, denom)
    po = (
        part_o_ptr
        + ((row * KV_HEADS + kvh) * SPLITS + offs_s[:, None]) * (HEADS_P * HEAD_DIM)
        + hloc * HEAD_DIM
        + offs_d[None, :]
    )
    o = tl.load(po, mask=smask[:, None], other=0.0)
    out = tl.sum(o * w[:, None], axis=0) / d_safe
    if NVFP4:
        out = out * tl.load(vgs_ptr)
    tl.store(
        out_ptr + row * (KV_HEADS * H_PER_KV * HEAD_DIM) + h * HEAD_DIM + offs_d,
        out.to(tl.bfloat16),
    )


def sparse_decode_attention(
    q: torch.Tensor,  # [rows, q_heads, head_dim]
    k: torch.Tensor,
    v: torch.Tensor,
    k_sf: Optional[torch.Tensor],
    v_sf: Optional[torch.Tensor],
    k_gs: Optional[torch.Tensor],
    v_gs: Optional[torch.Tensor],
    topk_idx: torch.Tensor,  # [rows, topk] int32, -1 pad
    req_to_token: torch.Tensor,
    req_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    nvfp4: bool,
    scratch: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Returns [rows, q_heads, head_dim] bf16."""
    rows, topk = topk_idx.shape
    q_heads, head_dim = q.shape[1], q.shape[2]
    kv_heads = k.shape[1]
    h_per_kv = q_heads // kv_heads
    heads_p = triton.next_power_of_2(h_per_kv)
    splits = max(1, min(_MAX_SPLITS, triton.cdiv(topk, _CHUNK)))
    splits_p = triton.next_power_of_2(splits)
    part_o, part_ml = scratch.get(rows, (None, None))
    if part_o is None or part_o.shape[2] < splits:
        part_o = torch.empty(
            (rows, kv_heads, splits, heads_p, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        part_ml = torch.empty(
            (rows, kv_heads, splits, heads_p, 2), dtype=torch.float32, device=q.device
        )
        scratch[rows] = (part_o, part_ml)
    out = torch.empty((rows, q_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    per_split = triton.cdiv(triton.cdiv(topk, splits), _CHUNK) * _CHUNK
    _sparse_decode_split[(rows, kv_heads, splits)](
        q,
        k,
        v,
        k_sf if nvfp4 else k,
        v_sf if nvfp4 else v,
        k_gs if nvfp4 else k,
        topk_idx,
        req_to_token,
        req_indices,
        seq_lens,
        part_o,
        part_ml,
        sm_scale,
        TOPK=topk,
        R2T_STRIDE=req_to_token.stride(0),
        SLOT_STRIDE_K=k.stride(0),
        SLOT_STRIDE_V=v.stride(0),
        SF_STRIDE=k_sf.stride(0) if nvfp4 else 0,
        PER_SPLIT=per_split,
        SPLITS=splits,
        KV_HEADS=kv_heads,
        H_PER_KV=h_per_kv,
        HEADS_P=heads_p,
        HEAD_DIM=head_dim,
        CHUNK=_CHUNK,
        NVFP4=nvfp4,
        num_warps=4,
        num_stages=1,
    )
    _sparse_decode_combine[(rows, q_heads)](
        part_o,
        part_ml,
        out,
        v_gs if nvfp4 else v,
        SPLITS=splits,
        SPLITS_P=splits_p,
        KV_HEADS=kv_heads,
        H_PER_KV=h_per_kv,
        HEADS_P=heads_p,
        HEAD_DIM=head_dim,
        NVFP4=nvfp4,
        num_warps=4,
    )
    return out
