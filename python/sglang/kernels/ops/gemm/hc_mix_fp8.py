"""MXFP8 (e4m3 + UE8M0) W8A16 variants of the fused HC-mix kernels.

Kept out of hc_mix.py on purpose: that file is the upstream hot path and
stays byte-identical to it. These kernels descale packed e4m3 tiles to bf16
in registers (power-of-2 scale, exact) and run bf16 tl.dot, so activations
are never quantized (W8A16). Used only when --enable-fp8-hc-mix swapped the
HC mix weights to packed MXFP8 at load time (see srt/layers/fp8_hc_mix.py).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.hc_mix import (
    _FUSED_MIX_MAX_ROWS,
    _get_counters,
    _get_ss,
    _grid_barrier,
)

# Rows a prefill chunk must reach before the W8A8 GEMMs beat the compiled
# BF16 dequant chain on this GPU. Measured on sm_120 (RTX PRO 6000,
# HC=4/HS=2560/LR=320, median of 50 timed calls) against the *full* fallback
# cost -- per-call MXFP8 dequant of both mix weights + the compiled chain:
# fallback wins below ~256 rows (0.83-0.95x for the small-launch path), the
# paths cross at ~512 (1.04x), and W8A8 pulls away to 1.1-1.5x through the
# 4k chunk band and 1.24x at 6144. The old 4096 floor timed the W8A8-only
# side of that comparison, so serving prefill chunks (17..4095 rows after a
# 4096-token chunk split) never actually reached it and re-dequantized the
# mix weights on every call.
_W8A8_MIN_ROWS = 512


@triton.jit
def _mxfp8_tile_to_bf16(
    w8, scale_u8, r: tl.constexpr, kb: tl.constexpr, out_dtype: tl.constexpr
):
    # w8: [R, kb*32] e4m3; scale_u8: [R, kb] ue8m0 (bias 127), groups along K.
    # Power-of-2 descale of an e4m3 mantissa is exact in bf16.
    descale = tl.exp2(scale_u8.to(tl.float32) - 127.0)
    w = tl.reshape(w8.to(tl.float32), (r, kb, 32)) * descale[:, :, None]
    return tl.reshape(w, (r, kb * 32)).to(out_dtype)


@triton.jit
def _hc_mix_fp8_persistent_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    w_ds_ptr,
    w_us_ptr,
    t_raw_ptr,
    out_ptr,
    counters_ptr,
    K,
    LOWRANK,
    HS,
    num_rows,
    num_ctas,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Unfolded variant: x is the RMSNormed (and gain-scaled) stream.
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    zero_span = ROWS * LOWRANK
    offs_z = tl.arange(0, 256)
    for z0 in range(pid * 256, zero_span, num_ctas * 256):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
    _grid_barrier(counters_ptr + 0, num_ctas)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in range(pid, n_blocks * k_chunks, num_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        k = kc * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        xt = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w8 = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        offs_kb = tl.arange(0, BLOCK_K // 32)
        kb = kc * (BLOCK_K // 32) + offs_kb
        ws = tl.load(
            w_ds_ptr + n[:, None] * (K // 32) + kb[None, :],
            mask=mask_n[:, None],
            other=127,
        )
        w = _mxfp8_tile_to_bf16(
            w8, ws, BLOCK_N, BLOCK_K // 32, x_ptr.dtype.element_ty
        )
        acc = tl.dot(xt, tl.trans(w))
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    _grid_barrier(counters_ptr + 1, num_ctas)

    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, BLOCK_R)
    offs_g = tl.arange(0, HC)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    for jb in range(pid, j_blocks, num_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        gj = offs_g[:, None] * HS + j[None, :]
        gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
        mask_gj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
        )
        acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in range(0, LOWRANK, BLOCK_R):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(
                t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
                mask=mask_r[None, :],
                other=0.0,
            )
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
            w8 = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            )
            offs_rb = tl.arange(0, BLOCK_R // 32)
            rb = r0 // 32 + offs_rb
            ws = tl.load(
                w_us_ptr + gj_flat[:, None] * (LOWRANK // 32) + rb[None, :],
                mask=mask_gj[:, None] & (rb < LOWRANK // 32)[None, :],
                other=127,
            )
            w = _mxfp8_tile_to_bf16(
                w8, ws, HC * BLOCK_J, BLOCK_R // 32, x_ptr.dtype.element_ty
            )
            acc = tl.dot(t, tl.trans(w), acc)
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        xg = tl.load(
            x_ptr
            + offs_m[:, None, None] * (HC * HS)
            + offs_g[None, :, None] * HS
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(
            out_ptr + offs_m[:, None] * HS + j[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_j[None, :],
        )

    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        tl.store(counters_ptr + 0, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


@triton.jit
def _hc_mix_fp8_persistent_fold_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    wn_ptr,
    w_ds_ptr,
    w_us_ptr,
    t_raw_ptr,
    ss_ptr,
    g_ptr,
    out_ptr,
    counters_ptr,
    K,
    LOWRANK,
    HS,
    num_rows,
    num_ctas,
    inv_hc,
    inv_hs,
    eps,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Norm-fold variant on RAW x: the RMSNorm gain (1+w) rides the (packed,
    # unfolded) down weights in-register right after the descale, and the
    # per-branch rsqrt factors from ss scale the down atomics so every later
    # phase is layout-identical to the unfused kernel.
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    offs_g = tl.arange(0, HC)

    zero_span = ROWS * LOWRANK
    offs_z = tl.arange(0, 256)
    for z0 in range(pid * 256, zero_span, num_ctas * 256):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
    # NB: ss is NOT zeroed at launch (a zero store before the first barrier
    # races peers' atomic_adds); the final-ticket CTA below keeps it zero
    # across launches.
    offs_k0 = tl.arange(0, BLOCK_K)
    k_chunks = tl.cdiv(K, BLOCK_K)
    k_per_branch = HS // BLOCK_K
    for kc0 in range(pid, k_chunks, num_ctas):
        b0 = kc0 // k_per_branch
        x0 = tl.load(
            x_ptr + offs_m[:, None] * K + (kc0 * BLOCK_K + offs_k0)[None, :],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        tl.atomic_add(
            ss_ptr + offs_m * HC + b0,
            tl.sum(x0 * x0, axis=1),
            mask=mask_m,
            sem="relaxed",
            scope="gpu",
        )
    _grid_barrier(counters_ptr + 0, num_ctas)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    for tile in range(pid, n_blocks * k_chunks, num_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        b = kc // k_per_branch
        n = nb * BLOCK_N + offs_n
        k = kc * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        xt = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w8 = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        offs_kb = tl.arange(0, BLOCK_K // 32)
        kb = kc * (BLOCK_K // 32) + offs_kb
        ws = tl.load(
            w_ds_ptr + n[:, None] * (K // 32) + kb[None, :],
            mask=mask_n[:, None],
            other=127,
        )
        wn_k = tl.load(wn_ptr + k, mask=k < K, other=0.0)
        w = (
            _mxfp8_tile_to_bf16(
                w8, ws, BLOCK_N, BLOCK_K // 32, x_ptr.dtype.element_ty
            )
            * wn_k[None, :]
        ).to(x_ptr.dtype.element_ty)
        ss_b = tl.load(ss_ptr + offs_m * HC + b, mask=mask_m, other=1.0)
        acc = tl.dot(xt, tl.trans(w)) * tl.math.rsqrt(ss_b * inv_hs + eps)[:, None]
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    _grid_barrier(counters_ptr + 1, num_ctas)

    if pid == 0:
        offs_gs = tl.arange(0, ROWS)
        for b in tl.static_range(HC):
            ss_b = tl.load(
                ss_ptr + offs_gs * HC + b, mask=offs_gs < num_rows, other=1.0
            )
            tl.store(
                g_ptr + offs_gs * HC + b,
                tl.math.rsqrt(ss_b * inv_hs + eps),
                mask=offs_gs < num_rows,
            )

    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, BLOCK_R)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    for jb in range(pid, j_blocks, num_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        gj = offs_g[:, None] * HS + j[None, :]
        gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
        mask_gj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
        )
        acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in range(0, LOWRANK, BLOCK_R):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(
                t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
                mask=mask_r[None, :],
                other=0.0,
            )
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
            w8 = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            )
            offs_rb = tl.arange(0, BLOCK_R // 32)
            rb = r0 // 32 + offs_rb
            ws = tl.load(
                w_us_ptr + gj_flat[:, None] * (LOWRANK // 32) + rb[None, :],
                mask=mask_gj[:, None] & (rb < LOWRANK // 32)[None, :],
                other=127,
            )
            w = _mxfp8_tile_to_bf16(
                w8, ws, HC * BLOCK_J, BLOCK_R // 32, x_ptr.dtype.element_ty
            )
            acc = tl.dot(t, tl.trans(w), acc)
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        xg = tl.load(
            x_ptr
            + offs_m[:, None, None] * (HC * HS)
            + offs_g[None, :, None] * HS
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        ss3 = tl.load(
            ss_ptr + offs_m[:, None] * HC + offs_g[None, :],
            mask=mask_m[:, None],
            other=1.0,
        )
        g3 = tl.math.rsqrt(ss3 * inv_hs + eps)
        wn = tl.load(
            wn_ptr + offs_g[:, None] * HS + j[None, :], mask=mask_j[None, :], other=0.0
        )
        out = (
            tl.sum(gate * xg * wn[None] * tl.reshape(g3, (ROWS, HC, 1)), axis=1)
            * inv_hc
        )
        tl.store(
            out_ptr + offs_m[:, None] * HS + j[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_j[None, :],
        )

    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        # All ss reads precede every CTA's barrier-1 ticket, so the final CTA
        # may reset the barriers and re-zero ss for the next launch.
        tl.store(counters_ptr + 0, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)
        offs_s = tl.arange(0, 256)
        for z0 in range(0, ROWS * HC, num_ctas * 256):
            idx_s = z0 + offs_s
            tl.store(ss_ptr + idx_s, 0.0, mask=idx_s < ROWS * HC)


def _weights_ok(w, scale, dtype) -> bool:
    if scale is None:
        return False
    return (
        w.dtype == torch.float8_e4m3fn
        and w.is_contiguous()
        and w.shape[1] % 32 == 0
        and scale.dtype == torch.uint8
        and tuple(scale.shape) == (w.shape[0], w.shape[1] // 32)
        and scale.is_contiguous()
        and dtype in (torch.bfloat16, torch.float16)
    )


def _inputs_ok(hyper_input: torch.Tensor) -> bool:
    from sglang.srt.environ import envs

    return (
        envs.SGLANG_HC_MIX_TRITON.get()
        and hyper_input.is_cuda
        and hyper_input.dtype in (torch.bfloat16, torch.float16)
        and hyper_input.shape[0] <= _FUSED_MIX_MAX_ROWS
        and hyper_input.dim() == 2
        and hyper_input.shape[1] % 2048 == 0
        and hyper_input.is_contiguous()
    )


def fused_hc_mix_fp8_supported(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_down_scale,
    w_up_scale,
) -> bool:
    return (
        _inputs_ok(hyper_input_normed)
        and _weights_ok(w_down, w_down_scale, hyper_input_normed.dtype)
        and _weights_ok(w_up, w_up_scale, hyper_input_normed.dtype)
    )


def fused_hc_mix_fold_fp8_supported(
    hyper_input: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_down_scale,
    w_up_scale,
) -> bool:
    return _inputs_ok(hyper_input) and _weights_ok(
        w_down, w_down_scale, hyper_input.dtype
    ) and _weights_ok(w_up, w_up_scale, hyper_input.dtype)


# ---------------------------------------------------------------------------
# Prefill W8A8 path: activations go to MXFP8 on the fly so the two mix GEMMs
# run on fp8 tensor cores (the mix is traffic/compute bound at chunk sizes,
# where the W8A16 persistent kernel cannot go). Kept separate from the
# decode kernels: different tiling (M-tile GEMMs, no grid barrier).
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp8_quant_groups(x, amax):
    # Shared activation-group quantizer: e = ceil(log2(amax/448)), zeros -> e=0.
    e = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0))
    e = tl.where(amax > 0, tl.minimum(tl.maximum(e, -126.0), 126.0), 0.0)
    return e


@triton.jit
def _mxfp8_quant_act_kernel(
    x_ptr, q_ptr, sf_ptr, T, K: tl.constexpr, BLOCK: tl.constexpr
):
    # x [T, K] bf16 -> q [T, K] e4m3 + sf [T, K//32] ue8m0 (row-major, 32-blocks).
    row = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK + tl.arange(0, BLOCK)
    offs_g32 = tl.arange(0, BLOCK // 32)
    x = tl.load(x_ptr + row.to(tl.int64) * K + c).to(tl.float32)
    xg = tl.reshape(x, (BLOCK // 32, 32))
    e = _mxfp8_quant_groups(xg, tl.max(tl.abs(xg), axis=1))
    q = (xg / tl.exp2(e)[:, None]).to(tl.float8e4nv)
    tl.store(q_ptr + row.to(tl.int64) * K + c, tl.reshape(q, (BLOCK,)))
    tl.store(
        sf_ptr + row.to(tl.int64) * (K // 32) + cb * (BLOCK // 32) + offs_g32,
        (e + 127).to(tl.uint8),
    )


@triton.jit
def _hc_mix_a8w8_down_kernel(
    a_ptr,
    asf_ptr,
    w_ptr,
    wsf_ptr,
    t_ptr,
    tsf_ptr,
    T,
    K: tl.constexpr,
    LOWRANK,
    inv_hc,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NKB: tl.constexpr,  # K // 32, groups per weight row
):
    # t = silu((q_a @ w_down^T) / hc) in fp8-mx GEMM tiles; epilogue quantizes
    # t straight to e4m3 + ue8m0 for the up GEMM.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < T
    mask_n = offs_n < LOWRANK
    offs_k = tl.arange(0, BK)
    offs_g = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(
            a_ptr + offs_m[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        asf = tl.load(
            asf_ptr + offs_m[:, None] * NKB + (k0 // 32) + offs_g[None, :],
            mask=mask_m[:, None],
            other=127,
        )
        b = tl.load(
            w_ptr + offs_n[None, :].to(tl.int64) * K + (k0 + offs_k)[:, None],
            mask=mask_n[None, :],
            other=0.0,
        )
        bsf = tl.load(
            wsf_ptr + offs_n[:, None] * NKB + (k0 // 32) + offs_g[None, :],
            mask=mask_n[:, None],
            other=127,
        )
        acc = tl.dot_scaled(a, asf, "e4m3", b, bsf, "e4m3", acc)
    # silu(x/hc) epilogue + per-32 mxfp8 quant of t for the up GEMM operand.
    tn = acc * inv_hc
    t = tn * tl.sigmoid(tn)
    tg = tl.reshape(t, (BM, BN // 32, 32))
    e = _mxfp8_quant_groups(tg, tl.max(tl.abs(tg), axis=2))
    q = (tg / tl.exp2(e)[:, :, None]).to(tl.float8e4nv)
    tl.store(
        t_ptr + offs_m[:, None].to(tl.int64) * LOWRANK + offs_n[None, :],
        tl.reshape(q, (BM, BN)),
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(
        tsf_ptr
        + offs_m[:, None] * (LOWRANK // 32)
        + pid_n * (BN // 32)
        + tl.arange(0, BN // 32)[None, :],
        (e + 127).to(tl.uint8),
        mask=mask_m[:, None],
    )


@triton.jit
def _hc_mix_a8w8_up_kernel(
    t_ptr,
    tsf_ptr,
    w_ptr,
    wsf_ptr,
    n_ptr,
    out_ptr,
    T,
    LOWRANK: tl.constexpr,
    HS: tl.constexpr,
    HC: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # gate = sigmoid(t @ w_up^T) per branch, out = mean_i(gate_i * n_i) inline.
    # Native mxfp8 dot_scaled: 1x32 e8m0 tiles keep the weight bit-exact vs
    # the W8A16 decode path (no per-row requant), and t rides the fp8 the
    # down epilogue already produced.
    pid_m = tl.program_id(0)
    pid_j = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_j = pid_j * BN + tl.arange(0, BN)
    mask_m = offs_m < T
    mask_j = offs_j < HS
    offs_k = tl.arange(0, BK)
    offs_g = tl.arange(0, BK // 32)
    out = tl.zeros((BM, BN), dtype=tl.float32)
    for i in tl.static_range(HC):
        rows = i * HS + offs_j
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for r0 in range(0, LOWRANK, BK):
            a = tl.load(
                t_ptr + offs_m[:, None].to(tl.int64) * LOWRANK + (r0 + offs_k)[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            asf = tl.load(
                tsf_ptr + offs_m[:, None] * (LOWRANK // 32) + r0 // 32 + offs_g[None, :],
                mask=mask_m[:, None],
                other=127,
            )
            b = tl.load(
                w_ptr + rows[None, :].to(tl.int64) * LOWRANK + (r0 + offs_k)[:, None],
                mask=mask_j[None, :],
                other=0.0,
            )
            bsf = tl.load(
                wsf_ptr + rows[:, None] * (LOWRANK // 32) + r0 // 32 + offs_g[None, :],
                mask=mask_j[:, None],
                other=127,
            )
            acc = tl.dot_scaled(a, asf, "e4m3", b, bsf, "e4m3", acc)
        gate = tl.sigmoid(acc)
        n_i = tl.load(
            n_ptr + offs_m[:, None].to(tl.int64) * (HC * HS) + (i * HS + offs_j)[None, :],
            mask=mask_m[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)
        out += gate * n_i
    tl.store(
        out_ptr + offs_m[:, None] * HS + offs_j[None, :],
        (out / HC).to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_j[None, :],
    )


def fused_hc_mix_fp8_w8a8_supported(
    x_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_down_scale,
    w_up_scale,
) -> bool:
    from sglang.srt.environ import envs

    return (
        envs.SGLANG_HC_MIX_W8A8_PREFILL.get()
        and envs.SGLANG_HC_MIX_TRITON.get()
        and x_normed.is_cuda
        and x_normed.dtype in (torch.bfloat16, torch.float16)
        and x_normed.dim() == 2
        # Crossover measured on sm_120: the compiled BF16 dequant chain is
        # launch/occupancy-bound below ~2k rows and beats this path there
        # (0.6-0.95x); only large prefill chunks gain (1.1x at 4k+).
        and x_normed.shape[0] >= _W8A8_MIN_ROWS
        and x_normed.shape[1] % 2048 == 0
        and x_normed.is_contiguous()
        and _weights_ok(w_down, w_down_scale, x_normed.dtype)
        and _weights_ok(w_up, w_up_scale, x_normed.dtype)
    )


def fused_hc_mix_fp8_w8a8(
    x_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_down_scale: torch.Tensor,
    w_up: torch.Tensor,
    w_up_scale: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    rows, k = x_normed.shape
    lowrank = w_down.shape[0]
    device = x_normed.device
    out = torch.empty((rows, hs), dtype=x_normed.dtype, device=device)
    if rows == 0:
        return out
    a8 = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=device)
    asf = torch.empty((rows, k // 32), dtype=torch.uint8, device=device)
    t8 = torch.empty((rows, lowrank), dtype=torch.float8_e4m3fn, device=device)
    tsf = torch.empty((rows, lowrank // 32), dtype=torch.uint8, device=device)
    _mxfp8_quant_act_kernel[(rows, triton.cdiv(k, 1024))](
        x_normed, a8, asf, rows, K=k, BLOCK=1024, num_warps=4
    )
    BM, BN, BK = 128, 64, 256
    _hc_mix_a8w8_down_kernel[(triton.cdiv(rows, BM), triton.cdiv(lowrank, BN))](
        a8,
        asf,
        w_down,
        w_down_scale,
        t8,
        tsf,
        rows,
        k,
        lowrank,
        1.0 / hc,
        BM=BM,
        BN=BN,
        BK=BK,
        NKB=k // 32,
        num_warps=8,
        num_stages=2,
    )
    _hc_mix_a8w8_up_kernel[(triton.cdiv(rows, 64), hs // 64)](
        t8,
        tsf,
        w_up,
        w_up_scale,
        x_normed,
        out,
        rows,
        LOWRANK=lowrank,
        HS=hs,
        HC=hc,
        BM=64,
        BN=64,
        BK=64,
        num_warps=4,
        num_stages=3,
    )
    return out


def fused_hc_mix_fp8(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    w_down_scale: torch.Tensor,
    w_up_scale: torch.Tensor,
) -> torch.Tensor:
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    device = hyper_input_normed.device
    t_raw = torch.empty((16, lowrank), dtype=torch.float32, device=device)
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    if rows == 0:
        return out
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    _hc_mix_fp8_persistent_kernel[(num_ctas,)](
        hyper_input_normed,
        w_down,
        w_up,
        w_down_scale,
        w_up_scale,
        t_raw,
        out,
        _get_counters(device),
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc,
        ROWS=16,
        HC=hc,
        BLOCK_N=32,
        BLOCK_K=256,
        BLOCK_J=32,
        BLOCK_R=64,
        num_warps=8,
    )
    return out


def fused_hc_mix_fold_fp8(
    hyper_input: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    wn: torch.Tensor,
    hc: int,
    hs: int,
    eps: float,
    w_down_scale: torch.Tensor,
    w_up_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns (mixed_input [rows, hs], per-branch norm factors g [rows, hc]).
    rows, k = hyper_input.shape
    lowrank = w_down.shape[0]
    device = hyper_input.device
    t_raw = torch.empty((16, lowrank), dtype=torch.float32, device=device)
    ss = _get_ss(device, 16 * hc)
    g = torch.empty((rows, hc), dtype=torch.float32, device=device)
    out = torch.empty((rows, hs), dtype=hyper_input.dtype, device=device)
    if rows == 0:
        return out, g
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    # sm_120-tuned (RTX PRO 6000, HC=4/HS=2560/LR=320): BLOCK_N=128 widens the
    # down tiles so fewer of them underfill the persistent grid, and BLOCK_J=16
    # doubles the up-phase's block count (hs/16 >= SMs) so it stops running on
    # a half-idle GPU. 10.7 -> 8.9 us/launch at decode shapes; numerics
    # unchanged except reduction order.
    _hc_mix_fp8_persistent_fold_kernel[(num_ctas,)](
        hyper_input,
        w_down,
        w_up,
        wn,
        w_down_scale,
        w_up_scale,
        t_raw,
        ss,
        g,
        out,
        _get_counters(device),
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc,
        1.0 / hs,
        eps,
        ROWS=16,
        HC=hc,
        BLOCK_N=128,
        BLOCK_K=256,
        BLOCK_J=16,
        BLOCK_R=64,
        num_warps=8,
    )
    return out, g
