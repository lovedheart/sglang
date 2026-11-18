"""Triton kernels for the Hunyuan-4-Flash Engram block.

The block per token: look up one table row per local hash head, RMSNorm each
row (``lookup_gamma``), project the concatenation to 4 keys + 1 value (one GEMM
and all-reduce, outside these kernels), then gate each iHC stream by the
RMSNorm'd key/query dot product and add the normalized value to it.

Both kernels follow the reference's rounding points (bf16 results of
``F.rms_norm`` and of the bf16 elementwise ops), so they differ from the eager
module only in reduction order.
"""

import torch
import triton
import triton.language as tl

_FP32_EPS = torch.finfo(torch.float32).eps
_BF16_EPS = torch.finfo(torch.bfloat16).eps


@triton.jit
def _engram_lookup_kernel(
    hash_ptr,  # [T, n_hash] int
    hash_stride,
    table_ptr,  # int64 address of [rows, D] bf16 or fp8 (host VAs of
    # cudaHostRegister'd tables work: they are addressable from the device)
    scale_ptr,  # int64 address of [rows] per-row scale (HAS_SCALE)
    gamma_ptr,  # [heads, D] bf16
    out_ptr,  # [T, heads * D] bf16
    head_start,
    vocab_start,
    rows,
    eps,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    TABLE_FP8: tl.constexpr,
    SCALE_F32: tl.constexpr,
):
    table = table_ptr.to(tl.int64).to(
        tl.pointer_type(tl.float8e4nv if TABLE_FP8 else tl.bfloat16)
    )
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    idx = tl.load(hash_ptr + t * hash_stride + head_start + h).to(tl.int64)
    row = tl.minimum(tl.maximum(idx - vocab_start, 0), rows - 1)
    x = tl.load(table + row * D + d, mask=mask, other=0.0).to(tl.float32)
    if HAS_SCALE:
        scale = scale_ptr.to(tl.int64).to(
            tl.pointer_type(tl.float32 if SCALE_F32 else tl.bfloat16)
        )
        x = x * tl.load(scale + row).to(tl.float32)
    x = tl.where(idx < 0, 0.0, x)
    gamma = tl.load(gamma_ptr + h * D + d, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / D + eps) * gamma
    tl.store(out_ptr + t * HEADS * D + h * D + d, y.to(tl.bfloat16), mask=mask)


def hy4_flash_engram_lookup_at(
    hashes: torch.Tensor,
    head_start: int,
    heads: int,
    vocab_start: int,
    table_ptr: int,
    scale_ptr: int,
    rows: int,
    dim: int,
    table_fp8: bool,
    scale_f32: bool,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """[T, heads * D] bf16: per-head RMSNorm'd table rows of this rank's hash
    heads; a negative hash gives a zero row. The table and its per-row scales
    are addressed by raw int64 pointers so host-resident (registered) tables
    and device tables share this one entry point."""
    tokens = hashes.shape[0]
    out = torch.empty((tokens, heads * dim), dtype=torch.bfloat16, device=hashes.device)
    if tokens == 0 or heads == 0:
        return out
    assert hashes.stride(1) == 1 and gamma.is_contiguous()
    _engram_lookup_kernel[(tokens, heads)](
        hashes,
        hashes.stride(0),
        table_ptr,
        scale_ptr,
        gamma,
        out,
        head_start,
        vocab_start,
        rows,
        _FP32_EPS,
        HEADS=heads,
        D=dim,
        BLOCK_D=triton.next_power_of_2(dim),
        HAS_SCALE=scale_ptr != 0,
        TABLE_FP8=table_fp8,
        SCALE_F32=scale_f32,
        num_warps=1,
    )
    return out


def hy4_flash_engram_lookup(
    hashes: torch.Tensor,
    head_start: int,
    heads: int,
    vocab_start: int,
    table: torch.Tensor,
    table_scale: torch.Tensor | None,
    gamma: torch.Tensor,
) -> torch.Tensor:
    assert table.is_contiguous()
    has_scale = table_scale is not None and table_scale.numel() > 0
    return hy4_flash_engram_lookup_at(
        hashes,
        head_start,
        heads,
        vocab_start,
        table.data_ptr(),
        table_scale.data_ptr() if has_scale else 0,
        table.shape[0],
        table.shape[1],
        table_fp8=table.dtype == torch.float8_e4m3fn,
        scale_f32=has_scale and table_scale.dtype == torch.float32,
        gamma=gamma,
    )


@triton.jit
def _engram_combine_kernel(
    kv_ptr,  # [T, (HC + 1) * H] bf16: HC keys then the value
    hidden_ptr,  # [T, HC, H] bf16
    out_ptr,  # [T, HC, H] bf16
    norm_key_ptr,  # [HC, H] bf16
    norm_query_ptr,  # [HC, H] bf16
    pre_gamma_ptr,  # [H] bf16
    hc_scale_ptr,  # [HC, H] bf16
    hash_ptr,  # [T, n_hash] int: a token whose hashes are all negative is skipped
    hash_stride,
    n_hash,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_HASH: tl.constexpr,
    eps32: tl.constexpr,
    eps_bf16: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK_H)
    mask = d < H
    kv_row = kv_ptr + t * (HC + 1) * H

    hs = tl.arange(0, BLOCK_HASH)
    hv = tl.load(hash_ptr + t * hash_stride + hs, mask=hs < n_hash, other=-1)
    live = tl.sum((hv >= 0).to(tl.int32), axis=0) > 0

    value = tl.load(kv_row + HC * H + d, mask=mask, other=0.0).to(tl.float32)
    pre_gamma = tl.load(pre_gamma_ptr + d, mask=mask, other=0.0).to(tl.float32)
    signal = pre_gamma * value * tl.rsqrt(tl.sum(value * value, axis=0) / H + eps32)

    for c in tl.static_range(HC):
        key = tl.load(kv_row + c * H + d, mask=mask, other=0.0).to(tl.float32)
        wk = tl.load(norm_key_ptr + c * H + d, mask=mask, other=0.0).to(tl.float32)
        key_n = key * tl.rsqrt(tl.sum(key * key, axis=0) / H + eps_bf16) * wk
        key_n = key_n.to(tl.bfloat16).to(tl.float32)
        h = tl.load(hidden_ptr + (t * HC + c) * H + d, mask=mask, other=0.0).to(
            tl.float32
        )
        wq = tl.load(norm_query_ptr + c * H + d, mask=mask, other=0.0).to(tl.float32)
        q_n = h * tl.rsqrt(tl.sum(h * h, axis=0) / H + eps_bf16) * wq
        q_n = q_n.to(tl.bfloat16).to(tl.float32)
        prod = (key_n * q_n).to(tl.bfloat16).to(tl.float32)
        s = tl.sum(prod, axis=0).to(tl.bfloat16).to(tl.float32)
        s = (s * (H**-0.5)).to(tl.bfloat16).to(tl.float32)
        mag = tl.maximum(tl.abs(s), 1e-6).to(tl.bfloat16).to(tl.float32)
        root = tl.sqrt(mag).to(tl.bfloat16).to(tl.float32)
        g = tl.where(s > 0, root, tl.where(s < 0, -root, 0.0))
        gate = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
        scale = tl.load(hc_scale_ptr + c * H + d, mask=mask, other=0.0).to(tl.float32)
        result = ((gate + scale) * signal).to(tl.bfloat16).to(tl.float32)
        result = tl.where(live, result, 0.0)
        tl.store(
            out_ptr + (t * HC + c) * H + d,
            (h + result).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def hy4_flash_engram_combine(
    kv: torch.Tensor,
    hidden: torch.Tensor,
    hashes: torch.Tensor,
    norm_key: torch.Tensor,
    norm_query: torch.Tensor,
    pre_gamma: torch.Tensor,
    hc_scales: torch.Tensor,
) -> torch.Tensor:
    """hidden + per-stream gate x normalized value; kv: [T, (HC + 1) * H] (keys
    then value), hidden: [T, HC, H], hashes: [T, n_hash] of this layer."""
    tokens, hc, hidden_size = hidden.shape
    out = torch.empty_like(hidden)
    if tokens == 0:
        return out
    kv = kv.contiguous()
    hidden = hidden.contiguous()
    assert kv.shape == (tokens, (hc + 1) * hidden_size)
    assert hashes.stride(1) == 1

    _engram_combine_kernel[(tokens,)](
        kv,
        hidden,
        out,
        norm_key,
        norm_query,
        pre_gamma,
        hc_scales,
        hashes,
        hashes.stride(0),
        hashes.shape[1],
        H=hidden_size,
        HC=hc,
        BLOCK_H=triton.next_power_of_2(hidden_size),
        BLOCK_HASH=triton.next_power_of_2(max(hashes.shape[1], 2)),
        eps32=_FP32_EPS,
        eps_bf16=_BF16_EPS,
        num_warps=8,
    )
    return out


_E2M3_LUT: torch.Tensor | None = None


def _e2m3_lut() -> torch.Tensor:
    """float32 decode table for the 64 FP6 E2M3 codes (sign, exp2, mant3)."""
    global _E2M3_LUT
    if _E2M3_LUT is None:
        codes = torch.arange(64, dtype=torch.int32)
        mag_i = codes & 31
        e, m = mag_i >> 3, mag_i & 7
        mag = torch.where(
            e == 0,
            m.to(torch.float32) * 0.125,
            (1.0 + m.to(torch.float32) / 8.0)
            * torch.pow(2.0, (e - 1).to(torch.float32)),
        )
        _E2M3_LUT = torch.where(codes >= 32, -mag, mag)
    return _E2M3_LUT


def unpack_fp6_e2m3_pack4x6b(
    packed: torch.Tensor, out_dtype: torch.dtype = torch.float8_e4m3fn
) -> torch.Tensor:
    """[rows, n * 3 // 4] uint8 holding n six-bit codes per row four to three
    little-endian bytes -> [rows, n] ``out_dtype`` values. FP6 E2M3 values sit
    exactly on the FP8 E4M3 grid, so the default cast is lossless."""
    rows, nbytes = packed.shape
    assert nbytes % 3 == 0, "packed rows must hold whole 4-code groups"
    b = packed.reshape(rows, nbytes // 3, 3).to(torch.int16)
    codes = (
        torch.stack(
            (
                b[..., 0] & 0x3F,
                ((b[..., 0] >> 6) | (b[..., 1] << 2)) & 0x3F,
                ((b[..., 1] >> 4) | (b[..., 2] << 4)) & 0x3F,
                b[..., 2] >> 2,
            ),
            -1,
        )
        .reshape(rows, -1)
        .to(torch.int64)
    )
    return _e2m3_lut().to(packed.device)[codes].to(out_dtype)
