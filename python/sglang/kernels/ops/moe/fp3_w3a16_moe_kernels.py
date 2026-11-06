"""FP3 (W3A16, 3-bit NVFP4-codebook) fused MoE kernels.

The checkpoint stores 3-bit codebook indices packed MSB-first into 48-bit
blocks (16 codes = 6 bytes, big-endian) with one uint8 e4m3-derived block
scale per group of 16 codes, plus a per-expert FP32 global scale. Serving
repacks the payload once at weight load into an LSB-first 8-codes-per-3-bytes
layout (see :func:`repack_fp3_w3a16_lsb3`) -- a pure permutation of the
checkpoint bytes, so the repacked model costs no extra VRAM. Two executors
share that payload:

* ``Fp3W3A16GemvG1/G2`` (JIT CUDA, ``csrc/moe/fp3_w3a16_gemv.cuh``): grouped
  warp-per-row SIMT GEMV, one launch per GEMM covering all active experts;
  fp32 end to end. Used for decode-size slot counts, where the tensor-core
  Triton path is launch- and SM-starved.
* ``_fp3_w3a16_gemm_kernel`` (Triton): grouped tensor-core GEMM with in-loop
  dequant, bf16 dots and FP32 accumulation; used for prefill-size batches.

    w = CB[code] * E4M3[sf_byte & 0xFE] * (sf_byte & 1 ? -1 : 1) * gs
    CB = [-6, -2, -1, 0, 1, 2, 3, 6]

Layouts (E local experts, K hidden, N output, group_size 16):
    payload : uint8 [E, N, K * 3 // 8]  (LSB-first: byte triple 3g..3g+2
          holds codes 8g..8g+7, code j at bits 3j of the 24-bit group)
    scales  : uint8 [E, N, K // 16]      (raw ModelOpt sf bytes, unswizzled)
    gscale  : fp32  [E] or [E, 2] for the w13 gate/up halves

The GEMM1 global scale is selected per output tile (gate rows vs up rows of
w13), so the gate/up halves keep their exact per-half checkpoint scales and
no gate/up ratio fold is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Fused MoE slot counts (tokens x topk) at or below which the grouped SIMT
# GEMV beats the Triton tensor-core path on sm_120. v6's per-block X staging
# adds traffic that grows with slots; measured crossover sits between 320 and
# 512 with the Qwen3.8-Flash-Next-FP3 expert shape.
_GEMV_MAX_SLOTS = 384

# Slot count at or below which the CuTe DSL LSB3 fork (3-bit codes expanded
# to NVFP4 nibbles in registers, sm12x W4A16 tile kernels) beats both other
# paths on sm_120; measured 1.5-3.8x over GEMV/Triton up to ~1536 tokens at
# top-k 10, Triton GEMM wins from ~2048 tokens up.
_CUTE_MAX_SLOTS = 15360
_cute_failures = 0
_cute_disabled = False
# swizzled-scale cache keyed by the raw scale tensor's storage; the entry
# keeps a reference so the data_ptr key cannot be recycled underneath it.
_cute_sf_cache: dict = {}


def _is_sm12x() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


def _cute_sf_swizzled(sf: torch.Tensor) -> torch.Tensor:
    """Fold the codebook mirror bit into the e4m3 sign and swizzle to the
    128x4 tile layout the CuTe fork consumes. Done once per weight tensor."""
    key = (sf.data_ptr(), tuple(sf.shape))
    hit = _cute_sf_cache.get(key)
    if hit is not None:
        return hit[1]
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    b = sf.to(torch.int64)
    folded = ((b & 0xFE) ^ ((b & 1) << 7)).to(torch.uint8)
    sw = swizzle_blockscale(folded.view(torch.float8_e4m3fn)).to(sf.device)
    _cute_sf_cache[key] = (sf, sw)
    return sw


def _cute_fork_experts(
    hidden_states,
    w13,
    w13_sf,
    w13_gs,
    w2,
    w2_sf,
    w2_gs,
    topk_weights,
    topk_ids,
    routed_scaling_factor,
):
    from sglang.kernels.ops.moe.sm12x_fp3_lsb3_w3a16_moe import run_moe_fp3_lsb3

    if w13_gs.dim() == 1:
        w13_gs = w13_gs.unsqueeze(1).expand(-1, 2).contiguous()
    wts = (
        topk_weights
        if routed_scaling_factor == 1.0
        else topk_weights.float() * routed_scaling_factor
    )
    out = torch.empty_like(hidden_states)
    run_moe_fp3_lsb3(
        hidden_states,
        topk_ids,
        wts,
        w13,
        _cute_sf_swizzled(w13_sf),
        w13_gs,
        w2,
        _cute_sf_swizzled(w2_sf),
        w2_gs.float(),
        out,
    )
    return out


def repack_fp3_w3a16_lsb3(payload: torch.Tensor) -> torch.Tensor:
    """Convert the checkpoint 48-bit-block payload to the LSB3 kernel layout.

    [E, N, K * 3 // 8] uint8 (16 codes per 6-byte big-endian block, code j =
    (v >> (45 - 3j)) & 7) -> same-shape uint8 where each 3-byte group holds 8
    codes LSB-first (code j of group g at bits 3j of bytes 3g..3g+2). Run once
    per weight at load time; a pure byte permutation, so the dequantized
    values are bit-identical and VRAM is unchanged.
    """
    e, n, kb3 = payload.shape
    k = kb3 * 8 // 3
    assert k % 16 == 0, "K must be a multiple of the fp3 group size (16)"
    out = torch.empty_like(payload)
    rows = e * n
    src = payload.reshape(rows, kb3)
    dst = out.reshape(rows, kb3)
    # Chunk rows so the int64 bit-unpacking intermediates stay ~64 MiB.
    chunk = max(1, (1 << 26) // (k * 8))
    for s in range(0, rows, chunk):
        b = src[s : s + chunk].to(torch.int64).reshape(-1, k // 16, 6)
        v = (
            (b[..., 0] << 40)
            | (b[..., 1] << 32)
            | (b[..., 2] << 24)
            | (b[..., 3] << 16)
            | (b[..., 4] << 8)
            | b[..., 5]
        )
        codes = torch.stack([(v >> (45 - 3 * j)) & 7 for j in range(16)], -1)
        c = codes.reshape(-1, k // 8, 8)
        w = sum(c[..., j] << (3 * j) for j in range(8))
        dst[s : s + chunk] = (
            torch.stack([(w >> (8 * t)) & 0xFF for t in range(3)], -1)
            .reshape(-1, kb3)
            .to(torch.uint8)
        )
    return out


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
    stride_be,  # payload row stride, in bytes (= K * 3 // 8)
    stride_bn,
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
    LSB3 payload [E, N, K*3//8] with unswizzled sf [E, N, K//16]. Blocks
    beyond ``num_tokens_post_padded`` exit early; rows whose sorted slot is a
    padding sentinel are masked out of A loads and C stores.
    """
    BLOCK_K_GROUPS: tl.constexpr = BLOCK_SIZE_K // 16
    BLOCK_K_CODE_GROUPS: tl.constexpr = BLOCK_SIZE_K // 8

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
    # 4-byte window per 8-code group (the 4th byte lane is masked off; the
    # payload stores 3 bytes per group).
    bgrp = (tl.arange(0, BLOCK_K_CODE_GROUPS)[:, None] * 3 + tl.arange(0, 4)[None, :])[
        None, :, :
    ]
    bsel = tl.arange(0, 4)[None, None, :] < 3
    group_off = tl.arange(0, BLOCK_K_GROUPS)[None, :]
    b_base = b_ptr + off_experts * stride_be + offs_bn[:, None] * stride_bn
    sf_base = sf_ptr + off_experts * stride_sfe + offs_bn[:, None] * stride_sfn
    shifts = (3 * tl.arange(0, 8))[None, None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kb in range(tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = kb * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)

        # Bytes 3g..3g+2 of the row window hold codes 8g..8g+7 LSB-first;
        # assemble each 24-bit group, then extract its 8 codes.
        p = tl.load(
            b_base[:, None, :] + (k_start * 3 // 8) + bgrp, mask=bsel, other=0
        ).to(tl.uint32)
        p3 = tl.sum(p << (8 * tl.arange(0, 4))[None, None, :], axis=2)  # [BN, BK/8]
        codes = tl.reshape(
            (p3.to(tl.int32)[:, :, None] >> shifts) & 7,
            (BLOCK_SIZE_N, BLOCK_SIZE_K),
        )

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


def _default_config(num_slots: int) -> dict[str, int]:
    # Tuned on sm_120 (RTX PRO 6000) with the Qwen3.8-Flash-Next-FP3 expert
    # shape (K=2560, I=640, E=512, topk=10); BLOCK_SIZE_N must divide the
    # gate/up half width of GEMM1. Decode-size slot counts go to the grouped
    # SIMT GEMV (see fused_experts_fp3_w3a16), so these tiers start above
    # that regime.
    if num_slots <= 8192:
        # Slots per expert stay small; BM16 minimizes padding waste.
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        }
    # Prefill-size batches: each expert owns enough slots to fill BM128.
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    }


def _invoke_fp3_gemm(
    a: torch.Tensor,  # [rows_a, K] bf16
    b: torch.Tensor,  # [E, N, K*3//8] uint8 LSB3 payload
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
    config: dict[str, int],
):
    _e, n, k3 = b.shape
    k = k3 * 8 // 3
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


# ---------------------------------------------------------------------------
# Grouped SIMT GEMV decode path (JIT CUDA, see csrc/moe/fp3_w3a16_gemv.cuh)
# ---------------------------------------------------------------------------

_gemv_unavailable = False


def _gemv_blocks(num_slots: int):
    # (rows_per_block, lanes_per_row) per GEMM, tuned on sm_120 with the
    # Qwen3.8-Flash-Next-FP3 expert shape: LN = 8 lanes/row doubles the
    # prefetch horizon vs LN = 16 (measured ~1.6x at 160 slots on GEMM1).
    if num_slots <= 48:
        return (16, 16), (16, 2)
    if num_slots <= 192:
        return (16, 8), (32, 4)
    return (32, 16), (32, 4)


@cache_once
def _jit_gemv_module(gemm: int, kt: int, r: int, ln: int) -> Module:
    cls = "Fp3W3A16GemvG1" if gemm == 1 else "Fp3W3A16GemvG2"
    return load_jit(
        "fp3_w3a16_gemv",
        gemm,
        kt,
        r,
        ln,
        cuda_files=["moe/fp3_w3a16_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::{cls}<{kt}, {r}, {ln}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def _gemv_fused_experts(
    hidden_states: torch.Tensor,  # [M, K] bf16
    w13: torch.Tensor,  # [E, 2I, K*3//8] uint8 LSB3 (gate rows then up rows)
    w13_sf: torch.Tensor,  # [E, 2I, K//16] uint8
    w13_gs: torch.Tensor,  # [E, 2] fp32
    w2: torch.Tensor,  # [E, K, I*3//8] uint8 LSB3
    w2_sf: torch.Tensor,  # [E, K, I//16] uint8
    w2_gs: torch.Tensor,  # [E] fp32
    topk_weights: torch.Tensor,  # [M, topk] fp32
    topk_ids: torch.Tensor,  # [M, topk] int
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Decode-size fused MoE via two grouped SIMT GEMV launches; fp32 math."""
    m, k = hidden_states.shape
    topk = topk_ids.shape[1]
    slots = m * topk
    inter = w2.shape[2] * 8 // 3
    dev = hidden_states.device
    # etab[slot] = local expert id (low 32 bits); slot is also the Y row and
    # (for GEMM1) resolves to token slot // topk inside the kernel.
    etab = topk_ids.reshape(-1).to(torch.int64, copy=False) | (
        torch.arange(slots, dtype=torch.int64, device=dev) << 32
    )
    (r1, ln1), (r2, ln2) = _gemv_blocks(slots)
    y1 = torch.empty((slots, 2 * inter), dtype=torch.float32, device=dev)
    _jit_gemv_module(1, k // 8, r1, ln1).run(
        hidden_states, w13, w13_sf, w13_gs, etab, y1, topk
    )
    act = torch.nn.functional.silu(y1[:, :inter]) * y1[:, inter:]
    y2 = torch.empty((slots, k), dtype=torch.float32, device=dev)
    _jit_gemv_module(2, inter // 8, r2, ln2).run(act, w2, w2_sf, w2_gs, etab, y2)
    out = (y2.view(m, topk, k) * topk_weights.reshape(m, topk, 1)).sum(1)
    if routed_scaling_factor != 1.0:
        out = out * routed_scaling_factor
    return out.to(hidden_states.dtype)


def fused_experts_fp3_w3a16(
    hidden_states: torch.Tensor,  # [M, K] bf16
    w13: torch.Tensor,  # [E, 2I, K*3//8] uint8 LSB3 (gate rows then up rows)
    w13_sf: torch.Tensor,  # [E, 2I, K//16] uint8
    w13_gs: torch.Tensor,  # [E, 2] fp32 (gate, up) or [E] shared
    w2: torch.Tensor,  # [E, K, I*3//8] uint8 LSB3
    w2_sf: torch.Tensor,  # [E, K, I//16] uint8
    w2_gs: torch.Tensor,  # [E] fp32
    topk_weights: torch.Tensor,  # [M, topk] fp32
    topk_ids: torch.Tensor,  # [M, topk] int
    activation: str = "silu",
    routed_scaling_factor: float | None = None,
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    """Full W3A16 fused-MoE forward: GEMM1 -> silu*up -> GEMM2 -> topk sum.

    Dispatches decode-size slot counts to the grouped SIMT GEMV (zero VRAM
    overhead, fp32 math) and larger ones to the Triton tensor-core GEMM.
    """
    global _gemv_unavailable, _cute_failures, _cute_disabled
    assert hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
    assert activation == "silu", f"fp3_w3a16 supports silu, got {activation}"
    m, k = hidden_states.shape
    e, n13, k3 = w13.shape
    assert k * 3 // 8 == k3, "w13 payload width mismatch"
    topk = topk_ids.shape[1]
    inter = w2.shape[2] * 8 // 3
    slots = m * topk

    # CuTe DSL LSB3 fork: expands the 3-bit codebook payload to NVFP4 nibbles
    # in registers and runs the sm12x W4A16 tile kernels. Fastest path on
    # sm_120/121 up to the measured crossover with the Triton GEMM; any
    # failure (shape unsupported, not compiled before graph capture) falls
    # back to GEMV/Triton for that call.
    if (
        not _cute_disabled
        and config is None
        and 0 < slots <= _CUTE_MAX_SLOTS
        and _is_sm12x()
    ):
        try:
            return _cute_fork_experts(
                hidden_states,
                w13,
                w13_sf,
                w13_gs,
                w2,
                w2_sf,
                w2_gs,
                topk_weights.float(),
                topk_ids.int(),
                routed_scaling_factor or 1.0,
            )
        except Exception as exc:  # noqa: BLE001  (fork optional: never fatal)
            _cute_failures += 1
            if _cute_failures >= 3:
                _cute_disabled = True
                print(
                    f"fp3_w3a16 CuTe DSL fork disabled after {_cute_failures} "
                    f"failures ({exc}); using GEMV/Triton"
                )

    # The quad-load GEMV needs 12-byte-aligned row slices (K/8 % 4 == 0).
    if (
        config is None
        and slots <= _GEMV_MAX_SLOTS
        and w13_gs.dim() == 2
        and (k // 8) % 4 == 0
        and (inter // 8) % 4 == 0
        and not _gemv_unavailable
    ):
        try:
            return _gemv_fused_experts(
                hidden_states,
                w13,
                w13_sf,
                w13_gs,
                w2,
                w2_sf,
                w2_gs,
                topk_weights,
                topk_ids,
                routed_scaling_factor or 1.0,
            )
        except Exception as exc:  # noqa: BLE001  (JIT build/host: stay on Triton)
            _gemv_unavailable = True
            print(f"fp3_w3a16 grouped GEMV unavailable, using Triton: {exc}")

    config = dict(config or _default_config(slots))
    assert (n13 // 2) % config["BLOCK_SIZE_N"] == 0, (
        "gate/up halves must not straddle a GEMM1 output tile"
    )

    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        act_and_mul_triton,
        moe_sum_reduce_triton,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    topk_ids_i32 = topk_ids.to(torch.int32)
    topk_weights_f32 = topk_weights.to(torch.float32).contiguous()
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids_i32, config["BLOCK_SIZE_M"], e
    )
    num_valid = slots
    dev = hidden_states.device

    cache1 = torch.empty((num_valid, n13), dtype=torch.bfloat16, device=dev)
    _invoke_fp3_gemm(
        hidden_states,
        w13,
        w13_sf,
        w13_gs,
        cache1,
        topk_weights_f32,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid,
        top_k_a=topk,
        mul_routed_weight=False,
        config=config,
    )

    cache2 = torch.empty((num_valid, inter), dtype=torch.bfloat16, device=dev)
    act_and_mul_triton(
        cache1,
        cache2,
        config,
        topk_ids=topk_ids_i32,
        activation=activation,
    )

    cache3 = torch.empty((num_valid, k), dtype=torch.bfloat16, device=dev)
    _invoke_fp3_gemm(
        cache2,
        w2,
        w2_sf,
        w2_gs,
        cache3,
        topk_weights_f32,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid,
        top_k_a=1,
        mul_routed_weight=True,
        config=config,
    )

    out = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
    moe_sum_reduce_triton(
        cache3.view(m, topk, k),
        out,
        routed_scaling_factor=routed_scaling_factor or 1.0,
    )
    return out
