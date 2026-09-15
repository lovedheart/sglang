"""Tests for the SGLANG_QSA_ATTN_FP8 sparse-decode scratch dtype selection.

Pins (a) the pure host-side gates -- when the paged decode scratch may be
fp8_e4m3 and when fp8 KV rows may be handed to the chunk-prefill kernel
as-is -- and (b) the device-side parity the flags rely on: on an fp8_e4m3
pool the strided gather into an fp8 scratch is a bitwise copy of the bf16
gather, and the fused NVFP4 gather into an fp8 scratch reproduces the bf16
dequantized rows up to one e4m3 rounding (with the strided tail zero-filled,
not stale scratch bytes).
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_kv_gather_dequant_fp4_triton,
)

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
PAGE = 64


def test_attn_fp8_env_flag_defaults_off():
    # Unset must mean off everywhere: the fp8 scratch is strictly opt-in.
    if not envs.SGLANG_QSA_ATTN_FP8.is_set():
        assert envs.SGLANG_QSA_ATTN_FP8.get() is False


@pytest.mark.parametrize(
    "requested,query_dtype,kv_dtype,fused,gathered,expected",
    [
        # fp8 pool, plain gatherer: bitwise copy, kernel sees fp8.
        (True, BF16, FP8, False, False, FP8),
        # fused FP4 gather: dequantizes in registers, requantizes to fp8.
        (True, BF16, BF16, True, True, FP8),
        # env off keeps bf16 everywhere.
        (False, BF16, FP8, False, False, BF16),
        (False, BF16, BF16, True, True, BF16),
        # legacy (non-fused) FP4 path hands in an already-dequantized bf16
        # buffer -- casting it back to fp8 would add a rounding for free.
        (True, BF16, BF16, False, True, BF16),
        # bf16 pool stays bf16.
        (True, BF16, BF16, False, False, BF16),
        # fp16 queries have no bf16-q/fp8-KV kernel path.
        (True, torch.float16, FP8, False, False, torch.float16),
    ],
)
def test_attn_scratch_dtype_gate(
    requested, query_dtype, kv_dtype, fused, gathered, expected
):
    assert (
        QwenSparseAttnBackend._attn_scratch_dtype(
            requested, query_dtype, kv_dtype, fused_fp4=fused, gathered_rows=gathered
        )
        == expected
    )


@pytest.mark.parametrize("dtype", [FP8, BF16, torch.float16])
def test_widen_kv_for_kernel_passes_fp8_through(dtype):
    rows = torch.randn(4, 2, 16).to(dtype)
    widened = QwenSparseAttnBackend._widen_kv_for_kernel(rows, BF16)
    if dtype == FP8:
        # fp8 -> bf16 is exact; the kernel casts its K/V tiles to the query
        # dtype itself, so skip the host-side materialization entirely.
        assert widened is rows
    else:
        assert widened.dtype == BF16
        assert torch.equal(widened, rows.to(BF16))


def _make_fp4_world(batch, topk, heads, dim, pool_rows, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k_fp4 = (
        torch.randint(0, 256, (pool_rows, heads, dim // 2), generator=g, dtype=torch.int64)
        .to(torch.uint8)
        .to(device)
    )
    v_fp4 = (
        torch.randint(0, 256, (pool_rows, heads, dim // 2), generator=g, dtype=torch.int64)
        .to(torch.uint8)
        .to(device)
    )
    k_sf = (
        torch.randint(0, 127, (pool_rows, heads * (dim // 16)), generator=g, dtype=torch.int64)
        .to(torch.uint8)
        .to(device)
    )
    v_sf = (
        torch.randint(0, 127, (pool_rows, heads * (dim // 16)), generator=g, dtype=torch.int64)
        .to(torch.uint8)
        .to(device)
    )
    k_gs = (torch.rand(8, generator=g) * 0.9 + 0.1).float().to(device)
    v_gs = (torch.rand(8, generator=g) * 0.9 + 0.1).float().to(device)
    perm = torch.randperm(pool_rows - 1, generator=g) + 1
    req_to_token = perm[: batch * topk].reshape(batch, topk).to(torch.int32).to(device)
    req_indices = torch.arange(batch, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch,), topk, dtype=torch.int32, device=device)
    seq_lens[-1] = max(1, topk // 3)
    indices = torch.full((batch, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        n = int(seq_lens[b])
        indices[b, :n] = torch.arange(n, dtype=torch.int32, device=device)
    return dict(
        k_fp4=k_fp4, v_fp4=v_fp4, k_sf=k_sf, v_sf=v_sf, k_gs=k_gs, v_gs=v_gs,
        req_to_token=req_to_token, req_indices=req_indices,
        indices=indices, seq_lens=seq_lens,
    )


def test_strided_gather_into_fp8_scratch_matches_bf16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, topk, heads, dim, pool_rows = 3, 130, 2, 128, 4096
    stride = ((topk + PAGE - 1) // PAGE) * PAGE
    # an fp8 pool stores fp8; widening on the way in (the default path) must
    # give exactly the same rows as storing them still-fp8 (the opt-in path).
    k_pool = torch.randn(pool_rows, heads, dim, device=device).to(FP8)
    v_pool = k_pool.clone()
    seq_lens = torch.tensor([topk, topk, topk // 3], device=device, dtype=torch.int32)
    req_to_token = (
        torch.randperm(pool_rows, device=device)[: batch * topk]
        .reshape(batch, topk)
        .to(torch.int32)
    )
    req_indices = torch.arange(batch, device=device, dtype=torch.int32)
    indices = torch.full((batch, topk), -1, device=device, dtype=torch.int32)
    for b in range(batch):
        n = min(int(seq_lens[b]), topk)
        indices[b, :n] = torch.arange(n, device=device, dtype=torch.int32)
    cu = torch.arange(batch + 1, device=device, dtype=torch.int32) * stride

    outs = {}
    for out_dtype in (BF16, FP8):
        poison = torch.full(
            (batch * stride, heads, dim), float("nan"), device=device
        ).to(out_dtype)
        qwen_sparse_kv_extraction_compact_triton(
            k_pool, v_pool, req_to_token, req_indices, indices, seq_lens, cu,
            poison, poison.clone(), batch, topk, zero_fill_cols=stride,
        )
        outs[out_dtype] = poison

    widened = outs[FP8].to(BF16)
    ref = outs[BF16]
    # valid rows: widening fp8 storage is exact, so bitwise equal
    valid_cols = torch.zeros(batch * stride, dtype=torch.bool, device=device)
    for b in range(batch):
        valid_cols[b * stride : b * stride + int(seq_lens[b])] = True
    rows_valid = valid_cols[:, None, None].expand_as(widened)
    assert torch.equal(widened.view(BF16)[rows_valid], ref.view(BF16)[rows_valid])
    # tail: fp8 zeros are legal masked rows, never stale NaN bytes
    tail = ~rows_valid
    assert torch.equal(outs[FP8].view(torch.uint8)[tail], torch.zeros_like(outs[FP8].view(torch.uint8)[tail]))
    assert torch.equal(outs[BF16].view(torch.int16)[tail], torch.zeros_like(outs[BF16].view(torch.int16)[tail]))


def test_fp4_fused_gather_into_fp8_scratch_within_rounding():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    batch, topk, heads, dim, pool_rows, page = 3, 256, 2, 128, 8192, PAGE
    stride = ((topk + page - 1) // page) * page
    w = _make_fp4_world(batch, topk, heads, dim, pool_rows, device, seed=2)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * stride

    outs = {}
    for out_dtype in (BF16, FP8):
        k_out = torch.full(
            (batch * stride, heads, dim), float("nan"), device=device
        ).to(out_dtype)
        v_out = torch.full(
            (batch * stride, heads, dim), float("nan"), device=device
        ).to(out_dtype)
        qwen_sparse_kv_gather_dequant_fp4_triton(
            w["k_fp4"], w["v_fp4"], w["k_sf"], w["v_sf"],
            w["k_gs"][5:6], w["v_gs"][5:6],
            w["req_to_token"], w["req_indices"], w["indices"], w["seq_lens"], cu,
            k_out, v_out, batch, topk, heads, dim, zero_fill_cols=stride,
        )
        outs[out_dtype] = (k_out, v_out)

    for i, name in enumerate(("K", "V")):
        ref, got = outs[BF16][i].float(), outs[FP8][i].to(torch.float32)
        # requantizing the fp32 dequantized values into e4m3 (4 mantissa bits
        # incl. implicit, from a 16-block scale) vs into bf16 differs only by
        # the extra rounding; magnitudes above the e4m3 range saturate, the
        # same storage limit an fp8_e4m3 KV pool has (satfinite cast, never
        # NaN/Inf bytes).
        in_range = ref.abs() <= 440.0
        assert torch.allclose(got[in_range], ref[in_range], rtol=0.13, atol=0.02), (
            f"{name} drift"
        )
        sat = ~in_range
        assert torch.equal(got[sat].sign(), ref[sat].sign().clamp(min=-1)), (
            f"{name} saturation sign"
        )
        assert (got[sat].abs() <= 448.0).all(), f"{name} saturation range"
        assert not torch.isnan(got).any(), f"{name} NaN poison"
        tail = torch.zeros_like(got, dtype=torch.bool)
        for b in range(batch):
            tail[b * stride + int(w["seq_lens"][b]) : (b + 1) * stride] = True
        assert torch.equal(got[tail], torch.zeros_like(got[tail]))
        assert torch.equal(ref[tail], torch.zeros_like(ref[tail]))


def test_chunk_prefill_fp8_passthrough_is_bitwise():
    # _widen_kv_for_kernel hands the chunk-prefill kernel fp8 rows directly;
    # the kernel casts K/V to the query dtype inside the loop, so this must
    # match casting on the host bitwise (no numeric drift, no NaN rows).
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        sparse_gqa_fwd_interface_triton_ck,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    tq, tk, heads_q, heads_kv, dim = 64, 64, 12, 2, 256
    q = torch.randn(tq, heads_q, dim, device=device, dtype=BF16)
    k8 = torch.randn(tk, heads_kv, dim, device=device).to(FP8)
    v8 = torch.randn(tk, heads_kv, dim, device=device).to(FP8)
    idx = torch.randint(0, tk, (tq, heads_kv, 16), device=device, dtype=torch.int32)
    cu_q = torch.tensor([0, tq], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, tk], device=device, dtype=torch.int32)
    kv_lens = torch.tensor([tk], device=device, dtype=torch.int32)
    direct = sparse_gqa_fwd_interface_triton_ck(
        q, k8, v8, idx, cu_q, cu_k, kv_lens, dim**-0.5, max_q=tq
    )
    cast = sparse_gqa_fwd_interface_triton_ck(
        q, k8.to(BF16), v8.to(BF16), idx, cu_q, cu_k, kv_lens, dim**-0.5, max_q=tq
    )
    assert torch.isfinite(direct).all()
    assert torch.equal(direct.view(BF16), cast.view(BF16))

