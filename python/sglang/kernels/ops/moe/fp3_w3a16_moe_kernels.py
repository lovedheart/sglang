"""FP3 (W3A16, 3-bit NVFP4-codebook) fused MoE Triton kernels.

The weights are stored as 3-bit codebook indices packed MSB-first into
48-bit blocks (16 codes = 6 bytes, big-endian) with one uint8 e4m3-derived
block scale per group of 16 codes, plus a per-expert FP32 global scale.
This kernel dequantizes the weight tiles *inside* the GEMM K-loop (never a
one-time full dequant) and runs bf16 tensor-core dots with FP32 accumulation:

    w = CB[code] * E4M3[sf_byte & 0xFE] * (sf_byte & 1 ? -1 : 1) * gs
    CB = [-6, -2, -1, 0, 1, 2, 3, 6]

Layouts (E local experts, K hidden, N output, group_size 16):
    payload : uint8 [E, N, K * 3 // 8]   (3-bit codes, MSB-first 48-bit blocks)
    scales  : uint8 [E, N, K // 16]      (raw ModelOpt sf bytes, unswizzled)
    gscale  : fp32  [E] or [E, 2] for the w13 gate/up halves

The GEMM1 global scale is selected per output tile (gate rows vs up rows of
w13), so the gate/up halves keep their exact per-half checkpoint scales and
no gate/up ratio fold is needed.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fp3_e4m3_scale(sf):
    """Decode raw ModelOpt sf bytes to the (signed) FP8-E4M3 block scale.

    ``sf & 0xFE`` is the e4m3 byte (sign, 4-bit exponent, 3-bit mantissa);
    bit0 is the codebook mirror flag (the quantizer used the negated
    codebook for those blocks), folded in as a sign. Returns FP32.
    """
    b = sf & 0xFE
    e = (b >> 3) & 15
    m = b & 7
    bits = ((e + 120) << 23) | (m << 20)
    normal = bits.to(tl.uint32).to(tl.float32, bitcast=True)
    denormal = m.to(tl.float32) * 0.001953125  # m * 2**-9
    scale = tl.where(e == 0, denormal, normal)
    sign_neg = ((b >> 7) ^ (sf & 1)) & 1
    return tl.where(sign_neg == 0, scale, -scale)


@triton.jit
def _fp3_codes_to_value(codes):
    """Map 3-bit codes onto the signed codebook [-6,-2,-1,0,1,2,3,6]."""
    mag = codes - codes
    mag = tl.where(codes == 1, 2, mag)
    mag = tl.where(codes == 2, 1, mag)
    mag = tl.where(codes == 4, 1, mag)
    mag = tl.where(codes == 5, 2, mag)
    mag = tl.where(codes == 6, 3, mag)
    mag = tl.where((codes == 0) | (codes == 7), 6, mag)
    return tl.where(codes < 4, -mag.to(tl.float32), mag.to(tl.float32))


@triton.jit
def _fp3_w3a16_gemm_kernel(
    a_ptr,
    b_ptr,
    sf_ptr,
    gs_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    GS_GATE_UP,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,  # payload row stride, in bytes (= K * 3 // 8)
    stride_sfe,
    stride_sfn,  # scale row stride (= K // 16)
    stride_cm,
    stride_cn,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Grouped FP3-W3A16 GEMM: C[token-slot, n] = A[token, :] @ dequant(B[e])^T.

    A is [M, K] bf16 (rows addressed via sorted_token_ids // top_k), B is the
    packed payload [E, N, K*3//8] with unswizzled sf [E, N, K//16]. Blocks
    beyond ``num_tokens_post_padded`` exit early; rows whose sorted slot is a
    padding sentinel are masked out of A loads and C stores.
    """
    BLOCK_K_GROUPS: tl.constexpr = BLOCK_SIZE_K // 16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    # Per-K-block byte tile: BLOCK_SIZE_N rows x 6 bytes per 16-code group.
    group_off = tl.arange(0, BLOCK_K_GROUPS)[None, :]
    b_base = b_ptr + off_experts * stride_be + offs_bn[:, None] * stride_bn
    sf_base = sf_ptr + off_experts * stride_sfe + offs_bn[:, None] * stride_sfn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kb in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = kb * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)

        # 48-bit big-endian blocks: code j of group g = (v >> (45 - 3j)) & 7,
        # v = b0<<40 | b1<<32 | b2<<24 | b3<<16 | b4<<8 | b5 at byte 6*g.
        byte0 = (k_start * 3 // 8) + 6 * group_off
        b0 = tl.load(b_base + byte0).to(tl.int64)
        b1 = tl.load(b_base + byte0 + 1).to(tl.int64)
        b2 = tl.load(b_base + byte0 + 2).to(tl.int64)
        b3 = tl.load(b_base + byte0 + 3).to(tl.int64)
        b4 = tl.load(b_base + byte0 + 4).to(tl.int64)
        b5 = tl.load(b_base + byte0 + 5).to(tl.int64)
        v48 = (
            (b0 << 40) | (b1 << 32) | (b2 << 24) | (b3 << 16) | (b4 << 8) | b5
        )
        shifts = (45 - 3 * tl.arange(0, 16)).to(tl.int64)
        codes = ((v48[:, :, None] >> shifts[None, None, :]) & 7).to(tl.int32)
        codes = tl.reshape(codes, (BLOCK_SIZE_N, BLOCK_SIZE_K))

        sf = tl.load(sf_base + (k_start // 16) + group_off).to(tl.int32)
        scale = _fp3_e4m3_scale(sf)
        scale = tl.reshape(
            tl.broadcast_to(scale[:, :, None], (BLOCK_SIZE_N, BLOCK_K_GROUPS, 16)),
            (BLOCK_SIZE_N, BLOCK_SIZE_K),
        )

        b_val = (_fp3_codes_to_value(codes) * scale).to(compute_type)
        b = tl.trans(b_val)

        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak

    # Per-expert FP32 global scale. For the w13 GEMM the tile lies entirely in
    # the gate half or the up half (half-width divisible by BLOCK_SIZE_N), so
    # pick the matching scale_2 column.
    if GS_GATE_UP > 0:
        gs_col = tl.where(pid_n * BLOCK_SIZE_N >= N // 2, 1, 0)
        gs = tl.load(gs_ptr + off_experts * GS_GATE_UP + gs_col).to(tl.float32)
    else:
        gs = tl.load(gs_ptr + off_experts).to(tl.float32)
    accumulator *= gs

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _default_config(num_slots: int) -> Dict[str, int]:
    if num_slots <= 32:
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3}
    if num_slots <= 512:
        return {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 3}
    return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}


def _invoke_fp3_gemm(
    a: torch.Tensor,  # [rows_a, K] bf16
    b: torch.Tensor,  # [E, N, K*3//8] uint8 payload
    sf: torch.Tensor,  # [E, N, K//16] uint8
    gs: torch.Tensor,  # [E] or [E, 2] fp32
    c: torch.Tensor,  # [M*topk, N] bf16, addressed by sorted slot ids
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    top_k_a: int,  # slots per A row (topk for GEMM1, 1 for GEMM2)
    mul_routed_weight: bool,
    config: Dict[str, int],
):
    e, n, k_bytes = b.shape
    k = k_bytes * 8 // 3
    assert k % 16 == 0, "K must be a multiple of the fp3 group size (16)"
    assert k % config["BLOCK_SIZE_K"] == 0, "K must divide BLOCK_SIZE_K"
    grid = (
        triton.cdiv(sorted_token_ids.shape[0], config["BLOCK_SIZE_M"])
        * triton.cdiv(n, config["BLOCK_SIZE_N"]),
    )
    gs_gate_up = gs.shape[1] if gs.dim() == 2 else 0
    _fp3_w3a16_gemm_kernel[grid](
        a,
        b,
        sf,
        gs,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        sorted_token_ids.shape[0],
        num_valid_tokens,
        gs_gate_up,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        sf.stride(0),
        sf.stride(1),
        c.stride(0),
        c.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k_a,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        compute_type=tl.bfloat16,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def fused_experts_fp3_w3a16(
    hidden_states: torch.Tensor,  # [M, K] bf16
    w13: torch.Tensor,  # [E, 2I, K*3//8] uint8 (gate rows then up rows)
    w13_sf: torch.Tensor,  # [E, 2I, K//16] uint8
    w13_gs: torch.Tensor,  # [E, 2] fp32 (gate, up) or [E] shared
    w2: torch.Tensor,  # [E, K, I*3//8] uint8
    w2_sf: torch.Tensor,  # [E, K, I//16] uint8
    w2_gs: torch.Tensor,  # [E] fp32
    topk_weights: torch.Tensor,  # [M, topk] fp32
    topk_ids: torch.Tensor,  # [M, topk] int
    activation: str = "silu",
    routed_scaling_factor: Optional[float] = None,
    config: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """Full W3A16 fused-MoE forward: GEMM1 -> silu*up -> GEMM2 -> topk sum."""
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        act_and_mul_triton,
        moe_sum_reduce_triton,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    assert hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
    assert activation == "silu", f"fp3_w3a16 supports silu, got {activation}"
    m, k = hidden_states.shape
    e, n13, k_bytes = w13.shape
    assert k * 3 // 8 == k_bytes, "w13 payload width mismatch"
    topk = topk_ids.shape[1]
    inter = w2.shape[2] * 8 // 3
    config = dict(config or _default_config(m * topk))
    assert (n13 // 2) % config["BLOCK_SIZE_N"] == 0, (
        "gate/up halves must not straddle a GEMM1 output tile"
    )

    topk_ids_i32 = topk_ids.to(torch.int32)
    topk_weights_f32 = topk_weights.to(torch.float32).contiguous()
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids_i32, config["BLOCK_SIZE_M"], e
    )
    num_valid = m * topk
    dev = hidden_states.device

    cache1 = torch.empty((num_valid, n13), dtype=torch.bfloat16, device=dev)
    _invoke_fp3_gemm(
        hidden_states, w13, w13_sf, w13_gs, cache1, topk_weights_f32,
        sorted_token_ids, expert_ids, num_tokens_post_padded, num_valid,
        top_k_a=topk, mul_routed_weight=False, config=config,
    )

    cache2 = torch.empty((num_valid, inter), dtype=torch.bfloat16, device=dev)
    act_and_mul_triton(
        cache1, cache2, config, topk_ids=topk_ids_i32, activation=activation,
    )

    cache3 = torch.empty((num_valid, k), dtype=torch.bfloat16, device=dev)
    _invoke_fp3_gemm(
        cache2, w2, w2_sf, w2_gs, cache3, topk_weights_f32,
        sorted_token_ids, expert_ids, num_tokens_post_padded, num_valid,
        top_k_a=1, mul_routed_weight=True, config=config,
    )

    out = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
    moe_sum_reduce_triton(
        cache3.view(m, topk, k), out,
        routed_scaling_factor=routed_scaling_factor or 1.0,
    )
    return out
