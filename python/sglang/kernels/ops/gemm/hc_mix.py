"""Fused HC low-rank mix for decode-size batches.

One persistent kernel replaces the five-kernel `GatedResidual._mix_compute` chain.
One CTA per SM keeps every CTA resident, so the software grid barrier cannot deadlock;
the last CTA to finish resets the barrier counters,
so a captured CUDA graph replays with them in their initial state.
Row counts beyond ``_FUSED_MIX_MAX_ROWS`` stay on the torch.compile path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FUSED_MIX_MAX_ROWS = 16


def hc_norm_fold_enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_HC_NORM_FOLD.get()) and not _deterministic_inference()


@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


@triton.jit
def _hc_mix_persistent_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
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
        w = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
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
            w = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
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


_counters_cache = {}


def _get_counters(device: torch.device) -> torch.Tensor:
    buf = _counters_cache.get(device)
    if buf is None:
        buf = torch.zeros(3, dtype=torch.int32, device=device)
        _counters_cache[device] = buf
    return buf


def _deterministic_inference() -> bool:
    from sglang.srt.runtime_context import get_exec

    try:
        exec_cfg = get_exec()
    except ValueError:
        return False
    return bool(exec_cfg.deterministic.enable_deterministic_inference)


def fused_hc_mix_supported(
    hyper_input_normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> bool:
    # The persistent kernel accumulates the down projection with
    # device-scope atomics, so summation order varies across replays.
    if _deterministic_inference():
        return False
    from sglang.srt.environ import envs

    if not envs.SGLANG_HC_MIX_TRITON.get():
        return False
    return (
        hyper_input_normed.is_cuda
        and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
        and w_down.dtype == hyper_input_normed.dtype
        and w_up.dtype == hyper_input_normed.dtype
        and hyper_input_normed.shape[0] <= _FUSED_MIX_MAX_ROWS
        and hyper_input_normed.dim() == 2
        and hyper_input_normed.shape[1] % 2048 == 0
        and hyper_input_normed.is_contiguous()
        and w_down.is_contiguous()
        and w_up.is_contiguous()
    )


def fused_hc_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    rows_pad = 16
    device = hyper_input_normed.device
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    if rows == 0:
        return out
    _hc_mix_persistent_kernel[(num_ctas,)](
        hyper_input_normed,
        w_down,
        w_up,
        t_raw,
        out,
        _get_counters(device),
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_N=32,
        BLOCK_K=256,
        BLOCK_J=32,
        BLOCK_R=64,
        num_warps=8,
    )
    return out


@triton.jit
def _hc_mix_persistent_fold_kernel(
    x_ptr,
    w_down_fold_ptr,
    w_up_ptr,
    wn_ptr,
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
    # Norm-fold variant of _hc_mix_persistent_kernel: x is the raw residual,
    # w_down_fold has the RMSNorm weight baked in, and the per-branch rsqrt
    # factors (from ss) scale the down-projection atomics so every later
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
    if pid == 0:
        offs_s = tl.arange(0, 64)
        tl.store(ss_ptr + offs_s, 0.0, mask=offs_s < ROWS * HC)
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
        w = tl.load(
            w_down_fold_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
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
            w = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
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
        tl.store(counters_ptr + 0, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


def fused_hc_mix_fold_supported(
    hyper_input: torch.Tensor, w_down_fold: torch.Tensor, w_up: torch.Tensor
) -> bool:
    # The fold flag is latched once by GatedResidual at construction; callers
    # only reach this guard from inside that latched path, so no env re-check.
    from sglang.srt.environ import envs

    if not envs.SGLANG_HC_MIX_TRITON.get():
        return False
    return (
        hyper_input.is_cuda
        and hyper_input.dtype in (torch.bfloat16, torch.float16)
        and w_down_fold.dtype == hyper_input.dtype
        and w_up.dtype == hyper_input.dtype
        and hyper_input.shape[0] <= _FUSED_MIX_MAX_ROWS
        and hyper_input.dim() == 2
        and hyper_input.shape[1] % 2048 == 0
        and hyper_input.is_contiguous()
        and w_down_fold.is_contiguous()
        and w_up.is_contiguous()
    )


def fused_hc_mix_fold(
    hyper_input: torch.Tensor,
    w_down_fold: torch.Tensor,
    w_up: torch.Tensor,
    wn: torch.Tensor,
    hc: int,
    hs: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns (mixed_input [rows, hs], per-branch norm factors g [rows, hc]).
    rows, k = hyper_input.shape
    lowrank = w_down_fold.shape[0]
    rows_pad = 16
    device = hyper_input.device
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
    ss = torch.empty((rows_pad * hc,), dtype=torch.float32, device=device)
    g = torch.empty((rows, hc), dtype=torch.float32, device=device)
    out = torch.empty((rows, hs), dtype=hyper_input.dtype, device=device)
    if rows == 0:
        return out, g
    _hc_mix_persistent_fold_kernel[(num_ctas,)](
        hyper_input,
        w_down_fold,
        w_up,
        wn,
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
        ROWS=rows_pad,
        HC=hc,
        BLOCK_N=32,
        BLOCK_K=256,
        BLOCK_J=32,
        BLOCK_R=64,
        num_warps=8,
    )
    return out, g
