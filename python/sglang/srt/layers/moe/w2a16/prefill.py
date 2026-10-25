"""GPU prefill arm: Triton grouped GEMMs over the TS4 planes (W2A16).

The same 2-bit planes the CPU decode kernel uses (row-major, 4 codes/byte,
e8m0 per-32 scales, {-4,-1,+1,+4} levels), dequantized in-register to fp32
and multiplied on tensor cores through an exact fp16 mirror of the weights
(levels are small integers times a power of two, always representable in
fp16).  Stage 1 mirrors the CPU stage A (fused gate+up, clamped silu, fp16
h1); stage 2 mirrors stage B with the topk weight folded in after the dot.
gs13 = gs2 = 1 is assumed: the checkpoint global scales are baked into the
UE8M0 scale plane by ``transfer.py``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)

BLOCK_M = 16


@triton.jit
def _levels(c):
    """code {0,1,2,3} -> level {-4,-1,+1,+4} as fp32."""
    return tl.where(c == 0, -4.0, tl.where(c == 1, -1.0,
                   tl.where(c == 2, 1.0, 4.0)))


@triton.jit
def _w2a16_stage1_kernel(
    x_ptr, plane_ptr, sf_ptr, h_ptr,
    sorted_ids_ptr, expert_ids_ptr, npad_ptr,
    numel, K,
    TOPK: tl.constexpr, I: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """h1[pair, n] = silu20(x[t] . Wg[e,n]) * (x[t] . Wu[e,n]), fp16.

    plane rows [0, I) are gate, [I, 2I) up; row stride K/4 bytes.
    """
    pid = tl.program_id(0)
    num_pid_n = I // BLOCK_N
    pid_m = pid // num_pid_n
    if pid_m * BLOCK_M >= tl.load(npad_ptr):
        return
    pid_n = pid % num_pid_n

    offs_pair_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ids_ptr + offs_pair_id)
    valid = pair < numel
    t = pair // TOPK
    e = tl.load(expert_ids_ptr + pid_m)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + t[:, None] * K + k[None, :],
                    mask=valid[:, None], other=0.0).to(tl.float16)
        # weights as [BK, BN] tiles: row k, col n
        pg = plane_ptr + e * (2 * I) * (K // 4) + offs_n[None, :] * (K // 4)
        sg = sf_ptr + e * (2 * I) * (K // 32) + offs_n[None, :] * (K // 32)
        for half in tl.static_range(2):
            w = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
            for j in tl.static_range(4):
                b = tl.load(pg + half * I * (K // 4)
                            + (k[:, None] // 4)).to(tl.int32) >> (2 * j)
                w += tl.where((k[:, None] & 3) == j, _levels(b & 3), 0.0)
            s = tl.load(sg + half * I * (K // 32) + (k[:, None] // 32))
            w *= tl.exp2((s.to(tl.int32) - 127).to(tl.float32))
            if half == 0:
                acc_g = tl.dot(x, w.to(tl.float16), acc_g)
            else:
                acc_u = tl.dot(x, w.to(tl.float16), acc_u)

    g = acc_g
    u = acc_u
    gc = tl.minimum(tl.maximum(g, -20.0), 20.0)
    hv = (g / (1.0 + tl.exp(-gc))) * u
    tl.store(h_ptr + pair[:, None] * I + offs_n[None, :],
             hv.to(tl.float16), mask=valid[:, None])


@triton.jit
def _w2a16_stage2_kernel(
    h_ptr, plane_ptr, sf_ptr, tw_ptr, out_ptr,
    sorted_ids_ptr, expert_ids_ptr, npad_ptr,
    numel, I, H,
    TOPK: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out[t, m] += tw[pair] * (h1[pair] . W2[e, m]) via fp32 atomics."""
    pid = tl.program_id(0)
    num_pid_n = H // BLOCK_N
    pid_m = pid // num_pid_n
    if pid_m * BLOCK_M >= tl.load(npad_ptr):
        return
    pid_n = pid % num_pid_n

    offs_pair_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ids_ptr + offs_pair_id)
    valid = pair < numel
    t = pair // TOPK
    e = tl.load(expert_ids_ptr + pid_m)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, I, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        h = tl.load(h_ptr + pair[:, None] * I + k[None, :],
                    mask=valid[:, None], other=0.0)
        p = plane_ptr + e * H * (I // 4) + offs_n[None, :] * (I // 4) \
            + (k[:, None] // 4)
        srow = sf_ptr + e * H * (I // 32) + offs_n[None, :] * (I // 32)
        w = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        for j in tl.static_range(4):
            b = tl.load(p).to(tl.int32) >> (2 * j)
            w += tl.where((k[:, None] & 3) == j, _levels(b & 3), 0.0)
        s = tl.load(srow + (k[:, None] // 32))
        w *= tl.exp2((s.to(tl.int32) - 127).to(tl.float32))
        acc = tl.dot(h, w.to(tl.float16), acc)

    tw = tl.load(tw_ptr + pair, mask=valid, other=0.0)
    acc = acc * tw[:, None]
    tl.atomic_add(out_ptr + t[:, None] * H + offs_n[None, :], acc,
                  mask=valid[:, None])


def w2a16_gpu_forward(x, topk_ids, topk_weights, w13_plane, s13, w2_plane, s2,
                      E, H, I):
    """TS4 grouped-GEMM forward on the GPU. x cuda fp16/bf16 [T,H],
    ids int32 [T,topk], tw fp32 [T,topk] -> fp32 out [T,H]."""
    T, topk = topk_ids.shape
    numel = T * topk
    sorted_ids, expert_ids, npad = moe_align_block_size(
        topk_ids.to(torch.int32), BLOCK_M, E
    )
    device = x.device
    h1 = torch.empty((numel, I), dtype=torch.float16, device=device)
    # Padded worst-case grid; blocks past the real padded count early-exit.
    mblocks = (numel + E * (BLOCK_M - 1) + BLOCK_M - 1) // BLOCK_M
    grid1 = (mblocks * (I // 64),)
    _w2a16_stage1_kernel[grid1](
        x, w13_plane, s13, h1, sorted_ids, expert_ids, npad,
        numel, H, TOPK=topk, I=I, BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )
    out = torch.zeros((T, H), dtype=torch.float32, device=device)
    grid2 = (mblocks * (H // 64),)
    _w2a16_stage2_kernel[grid2](
        h1, w2_plane, s2, topk_weights, out, sorted_ids, expert_ids, npad,
        numel, I, H, TOPK=topk, BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )
    return out
