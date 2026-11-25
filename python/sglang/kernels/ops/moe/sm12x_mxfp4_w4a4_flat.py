# Fork note: this file mirrors gemm_dev/sm12x_mxfp4/mxfp4_w4a4_flat.py byte-
# identical except this header block (source of truth stays in gemm_dev).
# Flat-token grouped W4A4 (Triton tl.dot_scaled) — replaces the 3D dense_gemm
# bucketed driver once registered in the sglang runner.
# Flat-token grouped W4A4 MoE for MXFP4 (M2e): kills the 6.4x per-expert m-padding
# of the 3D dense_gemm driver. Marlin/vLLM grouped layout: moe_align_block_size
# sorts top-k slots into BM=64 tiles (mean ~20 slots/expert -> ~1.8x padding,
# vs 6.4x at m=128/expert), tl.dot_scaled does native e2m1 x e2m1 with per-K32
# UE8M0 scales (sm_120a mxf4 QMMA, probed 760-796 TFLOPS dense). All device-side,
# CUDA-graph capture-safe (fixed grid + ntp early-out, no host syncs).
# SF contract: quant_act_mxfp4 swizzled grid, first 128 rows == rows 0..127
# row-major => sliceable as logical [M, K/32] for M <= 128 (T<=12, topk=10).
# Numerics gate: fp64 requant-oracle relL2 <= 9.5e-2.
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
except ImportError:  # gemm_dev standalone
    moe_align_block_size = None

try:
    from mxfp4_act_quant import quant_act_mxfp4
except ImportError:
    from sglang.kernels.ops.moe.sm12x_mxfp4_act_quant import quant_act_mxfp4

_BM_MAX = 32  # max align tile; workspace sized for it


def _align_bm(npair: int, num_experts: int) -> int:
    """Low slots/expert -> small tile (padding dominates); high -> BM=32
    (tile efficiency dominates). Measured crossover ~16 slots/expert."""
    return 16 if npair <= num_experts * 16 else _BM_MAX


@triton.jit
def _e2m1_q(v):
    """v [.., 32]-shaped group values -> (packed-lo/hi-ready code per elem)."""
    a = tl.abs(v)
    idx = ((a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32)
           + (a > 1.25).to(tl.int32) + (a > 1.75).to(tl.int32)
           + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32)
           + (a > 5.0).to(tl.int32))
    idx = tl.where(a == 0.0, 0, idx)
    return tl.where(v < 0, idx | 8, idx).to(tl.uint8)


@triton.jit
def _quant_rowmajor_kernel(x_ptr, out_ptr, sf_ptr, K, KP, C, stride_x,
                           BLOCK_K: tl.constexpr):
    """x row -> packed fp4 out[row, :] + row-major E8M0 sf[row, :]."""
    row = tl.program_id(0)
    kb = tl.program_id(1)
    k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    km = k < K
    v = tl.load(x_ptr + row * stride_x + k, mask=km, other=0.0).to(tl.float32)
    grp = tl.reshape(v, (BLOCK_K // 32, 32))
    amax = tl.max(tl.abs(grp), axis=1, keep_dims=True)
    e = tl.extra.libdevice.rint(tl.extra.libdevice.log2(
        tl.maximum(amax, 1e-30) / 6.0))
    s = tl.extra.libdevice.exp2(e)
    qv = tl.clamp(grp / s, -6.0, 6.0)
    code = tl.reshape(_e2m1_q(tl.ravel(qv)), (BLOCK_K // 32, 16, 2))
    lo = tl.sum(tl.where(tl.arange(0, 2)[None, None, :] == 0,
                         code.to(tl.uint16), code.to(tl.uint16) << 4), axis=2)
    tl.store(out_ptr + row * KP + kb * (BLOCK_K // 2)
             + tl.arange(0, BLOCK_K // 2), tl.reshape(lo.to(tl.uint8),
                                                      (BLOCK_K // 2,)),
             mask=kb * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2) < KP)
    sf = tl.where(amax > 0, (e + 127).to(tl.uint8), 0)
    tl.store(sf_ptr + row * C + kb * (BLOCK_K // 32)
             + tl.arange(0, BLOCK_K // 32)[None, :],
             tl.reshape(sf, (BLOCK_K // 32,)),
             mask=kb * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32) < C)


@triton.jit
def _silu_mul_fp4_kernel(inter_ptr, a2_ptr, a2s_ptr, ntp_ptr, I,
                         BM: tl.constexpr, TC: tl.constexpr):
    """slot-major bf16 inter [rp, 2I] (gate [:I] | up [I:]) -> fp4 a2 + E8M0.
    2D tiles so I needs no power-of-2 (prod I=640); TC=64 = 2 K32 groups."""
    pid = tl.program_id(0)
    cb = tl.program_id(1)
    if pid * BM >= tl.load(ntp_ptr):
        return
    r = pid * BM + tl.arange(0, BM)
    c = cb * TC + tl.arange(0, TC)
    cm = c < I
    g = tl.load(inter_ptr + r[:, None] * (2 * I) + c[None, :],
                mask=cm[None, :], other=0.0).to(tl.float32)
    u = tl.load(inter_ptr + r[:, None] * (2 * I) + I + c[None, :],
                mask=cm[None, :], other=0.0).to(tl.float32)
    act = (g * tl.sigmoid(g)) * u  # [BM, TC]
    act3 = tl.reshape(act, (BM, TC // 32, 32))
    amax = tl.max(tl.abs(act3), axis=2)          # [BM, TC//32]
    e = tl.extra.libdevice.rint(tl.extra.libdevice.log2(
        tl.maximum(amax, 1e-30) / 6.0))
    s = tl.extra.libdevice.exp2(e)
    qv = act3 / s[:, :, None]
    code = tl.reshape(_e2m1_q(tl.ravel(qv)), (BM, TC // 32, 16, 2))
    packed = tl.sum(tl.where(tl.arange(0, 2)[None, None, None, :] == 0,
                             code.to(tl.uint16), code.to(tl.uint16) << 4), axis=3)
    packed = tl.reshape(packed.to(tl.uint8), (BM, TC // 2))
    tl.store(a2_ptr + r[:, None] * (I // 2) + cb * (TC // 2)
             + tl.arange(0, TC // 2)[None, :], packed,
             mask=(cb * (TC // 2) + tl.arange(0, TC // 2))[None, :] < I // 2)
    sf = tl.where(amax > 0, (e + 127).to(tl.uint8), 0)  # [BM, TC//32]
    tl.store(a2s_ptr + r[:, None] * (I // 32) + cb * (TC // 32)
             + tl.arange(0, TC // 32)[None, :], sf,
             mask=(cb * (TC // 32) + tl.arange(0, TC // 32))[None, :] < I // 32)


@triton.jit
def _gather_sorted_kernel(a_ptr, as_ptr, sid_ptr, sa_ptr, sas_ptr, npair,
                          KP: tl.constexpr, CP: tl.constexpr, CPP: tl.constexpr,
                          BP: tl.constexpr, TOPK: tl.constexpr):
    """A_sorted[pos] = a[tok(sid[pos])] (packed + SF). Padding slots (sid>=npair)
    read row 0; their outputs are discarded by the gated epilogue."""
    pos = tl.program_id(0)
    sid = tl.load(sid_ptr + pos)
    tok = tl.minimum(sid, npair - 1) // TOPK
    k = tl.program_id(1) * BP + tl.arange(0, BP)
    km = k < KP
    v = tl.load(a_ptr + tok * KP + k, mask=km, other=0)
    tl.store(sa_ptr + pos * KP + k, v, mask=km)
    if tl.program_id(1) == 0:
        c = tl.arange(0, CPP)
        sv = tl.load(as_ptr + tok * CP + c, mask=c < CP, other=0)
        tl.store(sas_ptr + pos * CP + c, sv, mask=c < CP)


@triton.jit
def _w4a4_gemm_flat(
    a_ptr, as_ptr, w_ptr, ws_ptr, sid_ptr, ntp_ptr, eid_ptr, rw_ptr, out_ptr,
    npair,
    N: tl.constexpr, K: tl.constexpr, TOPK: tl.constexpr,
    OUT_MODE: tl.constexpr,  # 0: sorted-bf16 store (B N-major); 1: atomic token add (B K-major)
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
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
    for kb in range(0, K // BK):
        ak = kb * (BK // 2) + tl.arange(0, BK // 2)
        a = tl.load(a_ptr + r[:, None] * (K // 2) + ak[None, :])
        ask = kb * (BK // 32) + tl.arange(0, BK // 32)
        asc = tl.load(as_ptr + r[:, None] * (K // 32) + ask[None, :])
        if OUT_MODE == 0:
            b = tl.load(w_ptr + (e * N + ns[None, :]) * (K // 2) + ak[:, None])
            bsc = tl.load(ws_ptr + (e * N + ns[:, None]) * (K // 32)
                          + ask[None, :])
            acc = tl.dot_scaled(a, asc, "e2m1", b, bsc, "e2m1", acc)
        else:
            b = tl.load(w_ptr + (e * N + ns[None, :]) * (K // 2) + ak[:, None])
            bsc = tl.load(ws_ptr + (e * N + ns[:, None]) * (K // 32)
                          + ask[None, :])
            acc = tl.dot_scaled(a, asc, "e2m1", b, bsc, "e2m1", acc)
    if OUT_MODE == 0:
        tl.store(out_ptr + r[:, None] * N + ns[None, :],
                 acc.to(out_ptr.dtype.element_ty))
    else:
        # OUT_MODE==1: store rw-weighted partials to sorted rows; _reduce gathers
        rw = tl.load(rw_ptr + sid, mask=valid, other=0.0)
        tl.store(out_ptr + r[:, None] * N + ns[None, :],
                 (acc * rw[:, None]).to(out_ptr.dtype.element_ty))


@triton.jit
def _invert_kernel(sid_ptr, inv_ptr, npair):
    pos = tl.program_id(0)
    sid = tl.load(sid_ptr + pos)
    if sid < npair:
        tl.store(inv_ptr + sid, pos)


@triton.jit
def _reduce_topk_kernel(part_ptr, sid_ptr, out_ptr, N: tl.constexpr,
                        TOPK: tl.constexpr, BN: tl.constexpr):
    """out[t] = sum_k part[inv[t*TOPK+k]], inv = inverse of sids (slot->pos)."""
    t = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), tl.float32)
    for k in tl.static_range(TOPK):
        pos = tl.load(sid_ptr + t * TOPK + k)  # sid_ptr = inv (slot->pos)
        acc += tl.load(part_ptr + pos.to(tl.int64) * N + col,
                       mask=col[None if False else 0] < N if False else col < N,
                       other=0.0).to(tl.float32)
    tl.store(out_ptr + t * N + col, acc.to(out_ptr.dtype.element_ty),
             mask=col < N)


def allocate_workspace(num_tokens, hidden_size, intermediate_size, num_experts,
                       top_k, device):
    device = torch.device(device)
    npair = num_tokens * top_k
    cap = npair + (num_experts + 1) * (_BM_MAX - 1)
    cap = (cap + _BM_MAX - 1) // _BM_MAX * _BM_MAX
    return {
        "cap": cap,
        "inter": torch.zeros(cap, 2 * intermediate_size, dtype=torch.bfloat16,
                             device=device),
        "a2": torch.zeros(cap, intermediate_size // 2, dtype=torch.uint8, device=device),
        "a2s": torch.zeros(cap, intermediate_size // 32, dtype=torch.uint8, device=device),
        "sa": torch.zeros(cap, hidden_size // 2, dtype=torch.uint8, device=device),
        "sas": torch.zeros(cap, hidden_size // 32, dtype=torch.uint8, device=device),
        "part": torch.zeros(cap, hidden_size, dtype=torch.bfloat16, device=device),
    }


def run_moe_w4a4_flat(hidden_states, topk_ids, topk_weights, w13, w13_sf, w2, w2_sf,
                      workspace=None):
    """w13 [E,2I,H/2] u8 + w13_sf [E,2I,H/32] u8 E8M0 row-major; w2 likewise."""
    t, h = hidden_states.shape
    e, two_i, _ = w13.shape
    i = w2.shape[2] * 2
    topk = topk_ids.shape[1]
    dev = hidden_states.device
    npair = t * topk
    assert t <= 65536, "flat path"
    ws = workspace if workspace is not None else allocate_workspace(
        t, h, i, e, topk, dev)
    a4 = torch.empty(t, h // 2, dtype=torch.uint8, device=dev)
    asf = torch.empty(t, h // 32, dtype=torch.uint8, device=dev)
    BKQ = min(triton.next_power_of_2(h), 4096) // 32 * 32
    _quant_rowmajor_kernel[(t, triton.cdiv(h, BKQ))](
        hidden_states, a4, asf, h, h // 2, h // 32, hidden_states.stride(0),
        BLOCK_K=BKQ, num_warps=4)
    bm = _align_bm(npair, e)
    sids, eids, ntp = moe_align_block_size(topk_ids.int(), bm, e)
    rp = sids.numel()
    # full cap arrays: pad rows up to tile boundary stay inside cap (EM <= cap)
    inter, a2, a2s = ws["inter"], ws["a2"], ws["a2s"]
    sa, sas = ws["sa"], ws["sas"]
    _gather_sorted_kernel[(rp, triton.cdiv(h // 2, 1024))](
        a4, asf, sids, sa, sas, npair, KP=h // 2, CP=h // 32,
        CPP=triton.next_power_of_2(h // 32), BP=1024, TOPK=topk, num_warps=4)
    import os
    _g1bn = int(os.environ.get("FLAT_G1BN", "256"))
    _g1bk = int(os.environ.get("FLAT_G1BK", "256"))
    _g2bn = int(os.environ.get("FLAT_G2BN", "256"))
    _g2bk = int(os.environ.get("FLAT_G2BK", "128"))
    _w4a4_gemm_flat[(rp // bm * (2 * i // _g1bn),)](
        sa, sas, w13, w13_sf, sids, ntp, eids, topk_weights, inter, npair,
        N=2 * i, K=h, TOPK=topk, OUT_MODE=0,
        BM=bm, BN=_g1bn, BK=_g1bk, num_warps=8, num_stages=3)
    _silu_mul_fp4_kernel[(triton.cdiv(rp, 16), i // 64)](
        inter, a2, a2s, ntp, i, BM=16, TC=64, num_warps=4)
    part = ws["part"]
    _w4a4_gemm_flat[(rp // bm * (h // _g2bn),)](
        a2, a2s, w2, w2_sf, sids, ntp, eids, topk_weights, part, npair,
        N=h, K=i, TOPK=topk, OUT_MODE=1,
        BM=bm, BN=_g2bn, BK=_g2bk, num_warps=4, num_stages=3)
    inv = torch.empty(npair, dtype=torch.int32, device=dev)
    _invert_kernel[(rp,)](sids, inv, npair, num_warps=4)
    out = torch.empty(t, h, dtype=torch.bfloat16, device=dev)
    _reduce_topk_kernel[(t, triton.cdiv(h, 1024))](
        part, inv, out, N=h, TOPK=topk, BN=1024, num_warps=4)
    return out
