# Fork note: this file mirrors gemm_dev/sm12x_mxfp4/mxfp4_w4a8_kernel.py
# byte-identical except this header block (source of truth stays in gemm_dev).
# MXFP4 W4A8 decode MoE: fork of sm12x_nvfp4_w4a8_moe.py (search "MXFP4:");
# e2m1 PRMT->fp8 decode, per-K32 E8M0 pow2 fold in fp32 between group dots,
# row-major [E,N,K/32] scales, no global scales.
"""SM12x MXFP4 W4A8 grouped MoE (decode path): FP4 weights decoded to FP8
in-kernel, per-K32-group UE8M0 (pow2, bias-127) scale folded per 32-column
group between fp8 tensor-core dots.

Fork of sglang.kernels.ops.moe.sm12x_nvfp4_w4a8_moe (search "MXFP4:"):
  1. scale bytes are E8M0 over K-groups of 32 at ROW-MAJOR [E, N, K/32]
     (no 16-group duplication, no swizzle, no MixFP4 bit-7 selector);
  2. weight dequant = e2m1 x 2^(b-127): the pow2 factor CANNOT fold into an
     fp8 operand (exponent range far outside e4m3), so each 32-column group's
     fp8 x fp8 dot is scaled in fp32 by exp2(b-127) per output column;
  3. no global scales (MXFP4 checkpoints are scale-free); gs plumbing kept
     with unit tensors.
Everything else (PRMT e2m1 decode, per-row fp8 activation quant, moe_align
tiles, silu*up + per-row requant, bf16 atomic finalize) matches the NVFP4
parent and its numerics docstring.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

try:
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
except ImportError:  # gemm_dev standalone
    moe_align_block_size = None

_FP8 = torch.float8_e4m3fn
_BLOCK = 64  # BM for moe_align and the GEMM tiles

# e2m1 magnitude bytes as e4m3: {0,.5,1,1.5,2,3,4,6} -> 00 30 38 3C 40 44 48 4C
# (bit-identical to the NVFP4 parent's table; flag path for e0m3 dropped).
_PRMT_ASM = tl.constexpr("""
{
.reg .b32 ta1, ta2, x, t1, t2, t3, sel, ma;
mov.b32 ta1, 0x3C383000;
mov.b32 ta2, 0x4C484440;
and.b32 x, $2, 0x07070707;
shr.b32 t1, x, 4;
shr.b32 t2, x, 8;
shr.b32 t3, x, 12;
or.b32 sel, x, t1;
or.b32 t2, t2, t3;
and.b32 sel, sel, 0x00FF;
and.b32 t2, t2, 0xFF00;
or.b32 sel, sel, t2;
prmt.b32 ma, ta1, ta2, sel;
shl.b32 t1, $2, 4;
and.b32 t1, t1, 0x80808080;
or.b32 $0, ma, t1;
shr.b32 x, $2, 4;
and.b32 x, x, 0x07070707;
shr.b32 t1, x, 4;
shr.b32 t2, x, 8;
shr.b32 t3, x, 12;
or.b32 sel, x, t1;
or.b32 t2, t2, t3;
and.b32 sel, sel, 0x00FF;
and.b32 t2, t2, 0xFF00;
or.b32 sel, sel, t2;
prmt.b32 ma, ta1, ta2, sel;
and.b32 t1, $2, 0x80808080;
or.b32 $1, ma, t1;
}""")


@triton.jit
def _decode_e2m1(packed):
    """(lo, hi) signed e4m3 magnitudes per e2m1 nibble pair (MXFP4: no flag)."""
    return tl.inline_asm_elementwise(
        asm=_PRMT_ASM, constraints="=r,=r,r", args=[packed],
        dtype=(tl.float8e4nv, tl.float8e4nv), is_pure=True, pack=4)


@triton.jit
def _row_quant(x_ptr, q_ptr, s_ptr, n_cols: tl.constexpr, BK: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, BK)
    m = offs < n_cols
    v = tl.load(x_ptr + r * n_cols + offs, mask=m, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-12)
    scale = amax / 448.0
    tl.store(q_ptr + r * n_cols + offs,
             tl.clamp(v / scale, -448.0, 448.0).to(q_ptr.dtype.element_ty), mask=m)
    tl.store(s_ptr + r, scale)


@triton.jit
def _w4a8_gemm_mx(
    a_ptr, a_s_ptr, w_ptr, sf_ptr, sid_ptr, ntp_ptr, eid_ptr, rw_ptr, out_ptr,
    npair,
    N: tl.constexpr, K: tl.constexpr, TOPK: tl.constexpr,
    OUT_MODE: tl.constexpr,  # 0: store sorted rows (GEMM1); 1: atomic token rows
    BM: tl.constexpr, BN: tl.constexpr,
):
    """BK fixed to 32 = the MX SF group; pow2 scale folded between group dots."""
    pid = tl.program_id(0)
    tiles_n = N // BN
    pid_m = pid // tiles_n
    pid_n = pid % tiles_n
    if pid_m * BM >= tl.load(ntp_ptr):
        return
    e = tl.load(eid_ptr + pid_m)
    if e < 0:
        return
    r = pid_m * BM + tl.arange(0, BM)
    sid = tl.load(sid_ptr + r)
    valid = sid < npair
    tok = tl.minimum(sid, npair - 1) // TOPK
    ns = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    kg = tl.arange(0, 32)
    for g in range(0, K // 32):
        kc = g * 32 + kg
        if OUT_MODE == 0:
            a = tl.load(a_ptr + tok[:, None] * K + kc[None, :])
        else:
            a = tl.load(a_ptr + r[:, None] * K + kc[None, :])
        j = g * 16 + tl.arange(0, 16)
        packed = tl.load(w_ptr + (e * N + ns[:, None]) * (K // 2) + j[None, :])
        # MXFP4: row-major E8M0 byte over the 32-group; 2^(b-127) via fp32 bits
        b = tl.load(sf_ptr + (e * N + ns) * (K // 32) + g).to(tl.int32)
        sfv = (b << 23).to(tl.float32, bitcast=True)
        lo, hi = _decode_e2m1(packed)
        w8 = tl.interleave(lo, hi)  # (BN, 32) fp8, unscaled e2m1 values
        acc += tl.dot(a, tl.trans(w8)) * sfv[None, :]
    if OUT_MODE == 0:
        a_s = tl.load(a_s_ptr + tok, mask=valid, other=0.0)
        acc = acc * a_s[:, None]
        tl.store(out_ptr + r[:, None] * N + ns[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=valid[:, None])
    else:
        a_s = tl.load(a_s_ptr + r, mask=valid, other=0.0)
        acc = acc * a_s[:, None]
        rw = tl.load(rw_ptr + sid, mask=valid, other=0.0)
        row = tl.minimum(sid, npair - 1) // TOPK
        tl.atomic_add(out_ptr + row[:, None] * N + ns[None, :],
                      (acc * rw[:, None]).to(out_ptr.dtype.element_ty),
                      mask=rw[:, None] != 0)


@triton.jit
def _act_quant(g_ptr, u_ptr, sid_ptr, ntp_ptr, a2_ptr, a2s_ptr, npair,
               K: tl.constexpr, TOPK: tl.constexpr,
               BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    if pid * BM >= tl.load(ntp_ptr):
        return
    r = pid * BM + tl.arange(0, BM)
    valid = r < tl.load(ntp_ptr)
    c = tl.arange(0, BK)
    cm = c < K
    g = tl.load(g_ptr + r[:, None] * (2 * K) + c[None, :],
                mask=valid[:, None] & cm[None, :], other=0.0).to(tl.float32)
    u = tl.load(u_ptr + r[:, None] * (2 * K) + c[None, :],
                mask=valid[:, None] & cm[None, :], other=0.0).to(tl.float32)
    act = (g * tl.sigmoid(g)) * u
    amax = tl.maximum(tl.max(tl.abs(act), axis=1), 1e-12)
    s = amax / 448.0
    tl.store(a2_ptr + r[:, None] * K + c[None, :],
             tl.clamp(act / s[:, None], -448.0, 448.0).to(a2_ptr.dtype.element_ty),
             mask=valid[:, None] & cm[None, :])
    tl.store(a2s_ptr + r, s, mask=valid)


def _align(topk_ids, num_experts, block):
    if moe_align_block_size is not None:
        return moe_align_block_size(topk_ids.int(), block, num_experts)
    # standalone torch fallback (correctness-only path)
    flat = topk_ids.reshape(-1).int()
    order = torch.argsort(flat, stable=True)
    cnt = torch.bincount(flat, minlength=num_experts)
    pad = ((cnt + block - 1) // block) * block
    ntp = torch.tensor([int(pad.sum())], dtype=torch.int32, device=flat.device)
    blocks = []
    starts = torch.cumsum(pad, 0) - pad
    eids = []
    parts = []
    for e in range(num_experts):
        seg = torch.full((int(pad[e]),), flat.numel(), dtype=torch.int32,
                         device=flat.device)
        ids_e = order[flat == e]
        seg[: ids_e.numel()] = ids_e
        parts.append(seg)
        eids += [e] * (int(pad[e]) // block)
    sids = torch.cat(parts)
    eidt = torch.full((len(eids),), -1, dtype=torch.int32, device=flat.device)
    eidt[: len(eids)] = torch.tensor(eids, dtype=torch.int32, device=flat.device)
    return sids, eidt, ntp


def allocate_workspace(num_tokens, hidden_size, intermediate_size, num_experts,
                       top_k, device):
    device = torch.device(device)
    npair = num_tokens * top_k
    cap = npair + (num_experts + 1) * (_BLOCK - 1)
    cap = (cap + 3) // 4 * 4
    eblocks = triton.cdiv(cap, _BLOCK)
    z = torch.zeros
    return {
        "cap": cap,
        "a8": z(cap, hidden_size, dtype=_FP8, device=device),
        "a_s": z(cap, dtype=torch.float32, device=device),
        "inter": z(cap, 2 * intermediate_size, dtype=torch.bfloat16, device=device),
        "a2": z(cap, intermediate_size, dtype=_FP8, device=device),
        "a2_s": z(cap, dtype=torch.float32, device=device),
    }


def run_moe_w4a4m(hidden_states, topk_ids, topk_weights, w13, w13_sf, w2, w2_sf,
                  workspace=None):
    """MXFP4 W4A8: w13 uint8 [E,2I,H/2], w13_sf uint8 [E,2I,H/32] row-major
    E8M0; w2/w2_sf [E,H,I/2], [E,H,I/32]. Returns bf16 [T,H]."""
    t, h = hidden_states.shape
    e, two_i, _ = w13.shape
    i = w2.shape[2] * 2
    topk = topk_ids.shape[1]
    dev = hidden_states.device
    npair = t * topk
    ws = workspace if workspace is not None else allocate_workspace(
        t, h, i, e, topk, dev)

    x8, x_s = ws["a8"], ws["a_s"]
    _row_quant[(t,)](hidden_states, x8[:t], x_s[:t], n_cols=h,
                     BK=triton.next_power_of_2(h), num_warps=8)

    sids, eids, ntp = _align(topk_ids, e, _BLOCK)
    rp = sids.numel()
    inter, a2, a2_s = ws["inter"][:rp], ws["a2"][:rp], ws["a2_s"][:rp]

    _w4a8_gemm_mx[(rp // _BLOCK * (2 * i // 256),)](
        x8, x_s, w13.view(torch.uint8), w13_sf.view(torch.uint8),
        sids, ntp, eids, topk_weights, inter, npair,
        N=2 * i, K=h, TOPK=topk, OUT_MODE=0,
        BM=_BLOCK, BN=256, num_warps=8, num_stages=3)

    _act_quant[(triton.cdiv(rp, 8),)](
        inter, inter[:, i:], sids, ntp, a2, a2_s, npair,
        K=i, TOPK=topk, BM=8, BK=triton.next_power_of_2(i), num_warps=4)

    out = torch.zeros(t, h, dtype=torch.bfloat16, device=dev)
    _w4a8_gemm_mx[(rp // _BLOCK * (h // 128),)](
        a2, a2_s, w2.view(torch.uint8), w2_sf.view(torch.uint8),
        sids, ntp, eids, topk_weights, out, npair,
        N=h, K=i, TOPK=topk, OUT_MODE=1,
        BM=_BLOCK, BN=128, num_warps=4, num_stages=2)
    return out
