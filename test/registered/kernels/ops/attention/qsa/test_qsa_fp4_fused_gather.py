"""Bitwise parity tests for the fused FP4 gather+dequant+pack kernel.

The fused kernel must produce, row for row, exactly what the legacy decode
path produced: row-gather the pool with index_select, dequantize whole rows
(NVFP4KVQuantizeUtil.dequantize) and repack them with the gathered-rows
packer. Also pins the validity rules (-1 padding, positions past the
sequence length) and the strided zero-fill semantics.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_gathered_rows_triton,
    qwen_sparse_kv_gather_dequant_fp4_triton,
)

PAGE = 64


def _make_world(batch, topk, heads, dim, pool_rows, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k_fp4 = (
        torch.randint(
            0, 256, (pool_rows, heads, dim // 2), generator=g, dtype=torch.int64
        )
        .to(torch.uint8)
        .to(device)
    )
    v_fp4 = (
        torch.randint(
            0, 256, (pool_rows, heads, dim // 2), generator=g, dtype=torch.int64
        )
        .to(torch.uint8)
        .to(device)
    )
    # keep the fp8 e4m3 NaN bytes (0x7F/0xFF) out of the scale data
    k_sf = (
        torch.randint(
            0, 127, (pool_rows, heads * (dim // 16)), generator=g, dtype=torch.int64
        )
        .to(torch.uint8)
        .to(device)
    )
    v_sf = (
        torch.randint(
            0, 127, (pool_rows, heads * (dim // 16)), generator=g, dtype=torch.int64
        )
        .to(torch.uint8)
        .to(device)
    )
    k_gs = (torch.rand(8, generator=g) * 0.9 + 0.1).float().to(device)
    v_gs = (torch.rand(8, generator=g) * 0.9 + 0.1).float().to(device)
    # scatter every request positions across the pool (slot 0 = dummy row)
    perm = torch.randperm(pool_rows - 1, generator=g) + 1
    req_to_token = perm[: batch * topk].reshape(batch, topk).to(torch.int32).to(device)
    req_indices = torch.arange(batch, dtype=torch.int32, device=device)
    # the last row is shorter than topk -> -1 padding at its tail
    seq_lens = torch.full((batch,), topk, dtype=torch.int32, device=device)
    seq_lens[-1] = max(1, topk // 3)
    indices = torch.full((batch, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        n = int(seq_lens[b])
        indices[b, :n] = torch.arange(n, dtype=torch.int32, device=device)
    return dict(
        k_fp4=k_fp4,
        v_fp4=v_fp4,
        k_sf=k_sf,
        v_sf=v_sf,
        k_gs=k_gs,
        v_gs=v_gs,
        req_to_token=req_to_token,
        req_indices=req_indices,
        indices=indices,
        seq_lens=seq_lens,
    )


def _legacy_gather(w, layer_id, batch, topk, heads, dim):
    """Mirror of QwenSparseAttnBackend._gather_kv_fp4 + _gather_topk_rows_fp4."""
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

    positions = w["indices"].clamp(min=0).long()
    slots = w["req_to_token"][w["req_indices"].long().unsqueeze(1), positions].reshape(
        -1
    )
    safe = slots.clamp(min=0).long()
    k_rows = w["k_fp4"].index_select(0, safe)
    v_rows = w["v_fp4"].index_select(0, safe)
    k_scale_rows = w["k_sf"].index_select(0, safe).view(torch.float8_e4m3fn)
    v_scale_rows = w["v_sf"].index_select(0, safe).view(torch.float8_e4m3fn)
    k = NVFP4KVQuantizeUtil.dequantize(
        k_rows, k_scale_rows, w["k_gs"][layer_id : layer_id + 1], dtype=torch.bfloat16
    ).view(batch * topk, heads, dim)
    v = NVFP4KVQuantizeUtil.dequantize(
        v_rows, v_scale_rows, w["v_gs"][layer_id : layer_id + 1], dtype=torch.bfloat16
    ).view(batch * topk, heads, dim)
    return k, v


@pytest.mark.parametrize("seed", [0, 1])
def test_fp4_fused_matches_legacy_compact(seed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    batch, topk, heads, dim, pool_rows = 4, 512, 2, 128, 4096
    w = _make_world(batch, topk, heads, dim, pool_rows, device, seed=seed)

    valid_counts = torch.empty(batch, dtype=torch.int32, device=device)
    cu = torch.empty(batch + 1, dtype=torch.int32, device=device)
    qwen_sparse_fa2_cu_seqlens_triton(
        w["seq_lens"], w["indices"], valid_counts, cu, batch, topk
    )

    k_rows, v_rows = _legacy_gather(w, 3, batch, topk, heads, dim)
    ref_k = torch.full(
        (batch * topk, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    ref_v = torch.full(
        (batch * topk, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    qwen_sparse_kv_extraction_gathered_rows_triton(
        k_rows, v_rows, w["indices"], w["seq_lens"], cu, ref_k, ref_v, batch, topk
    )

    fuse_k = torch.full(
        (batch * topk, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    fuse_v = torch.full(
        (batch * topk, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    qwen_sparse_kv_gather_dequant_fp4_triton(
        w["k_fp4"],
        w["v_fp4"],
        w["k_sf"],
        w["v_sf"],
        w["k_gs"][3:4],
        w["v_gs"][3:4],
        w["req_to_token"],
        w["req_indices"],
        w["indices"],
        w["seq_lens"],
        cu,
        fuse_k,
        fuse_v,
        batch,
        topk,
        heads,
        dim,
    )

    for name, got, ref in (("K", fuse_k, ref_k), ("V", fuse_v, ref_v)):
        same = got.view(torch.int16) == ref.view(torch.int16)
        both_nan = torch.isnan(ref) & torch.isnan(got)
        assert torch.equal(same | both_nan, torch.ones_like(same)), (
            f"compact {name} mismatch"
        )
        assert torch.isnan(ref).sum() == torch.isnan(got).sum(), f"{name} poison drift"


def test_fp4_fused_strided_zero_fill():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    batch, topk, heads, dim, pool_rows, page = 3, 2051, 2, 128, 8192, PAGE
    stride = ((topk + page - 1) // page) * page
    w = _make_world(batch, topk, heads, dim, pool_rows, device, seed=2)
    cu_strided = torch.arange(batch + 1, dtype=torch.int32, device=device) * stride

    k_rows, v_rows = _legacy_gather(w, 5, batch, topk, heads, dim)
    ref_k = torch.full(
        (batch * stride, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    ref_v = torch.full(
        (batch * stride, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    qwen_sparse_kv_extraction_gathered_rows_triton(
        k_rows,
        v_rows,
        w["indices"],
        w["seq_lens"],
        cu_strided,
        ref_k,
        ref_v,
        batch,
        topk,
    )

    fuse_k = torch.full(
        (batch * stride, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    fuse_v = torch.full(
        (batch * stride, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    qwen_sparse_kv_gather_dequant_fp4_triton(
        w["k_fp4"],
        w["v_fp4"],
        w["k_sf"],
        w["v_sf"],
        w["k_gs"][5:6],
        w["v_gs"][5:6],
        w["req_to_token"],
        w["req_indices"],
        w["indices"],
        w["seq_lens"],
        cu_strided,
        fuse_k,
        fuse_v,
        batch,
        topk,
        heads,
        dim,
        zero_fill_cols=stride,
    )

    for name, got, ref in (("K", fuse_k, ref_k), ("V", fuse_v, ref_v)):
        same = got.view(torch.int16) == ref.view(torch.int16)
        # invalid columns: legacy left the NaN poison, the fused strided mode
        # zero-fills them (matching _compact_kv); both are masked by the
        # paged kernel, so either is accepted here.
        tail_ok = torch.isnan(ref) & (got.view(torch.int16) == 0)
        assert torch.equal(same | tail_ok, torch.ones_like(same)), (
            f"strided {name} mismatch"
        )

    # the fused path additionally zeroes the page tail
    lens = w["seq_lens"].tolist()
    for b in range(batch):
        tail_k = fuse_k[b * stride + lens[b] : (b + 1) * stride]
        tail_v = fuse_v[b * stride + lens[b] : (b + 1) * stride]
        assert torch.count_nonzero(tail_k.view(torch.int16)) == 0, "K tail not zeroed"
        assert torch.count_nonzero(tail_v.view(torch.int16)) == 0, "V tail not zeroed"


def test_fp4_fused_invalid_never_reads_poison():
    """Slot 0 poisoned with NaN scale bytes: invalid columns (-1 padding)
    must never read it, so every written value stays finite and the poison
    never reaches the scratch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    batch, topk, heads, dim, pool_rows = 2, 64, 2, 128, 2048
    w = _make_world(batch, topk, heads, dim, pool_rows, device, seed=3)
    w["k_fp4"][0].fill_(0x77)
    w["v_fp4"][0].fill_(0x77)
    w["k_sf"][0].fill_(0x7F)  # fp8 e4m3 NaN
    w["v_sf"][0].fill_(0x7F)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * topk
    out_k = torch.full(
        (batch * topk, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    out_v = torch.full(
        (batch * topk, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    qwen_sparse_kv_gather_dequant_fp4_triton(
        w["k_fp4"],
        w["v_fp4"],
        w["k_sf"],
        w["v_sf"],
        w["k_gs"][0:1],
        w["v_gs"][0:1],
        w["req_to_token"],
        w["req_indices"],
        w["indices"],
        w["seq_lens"],
        cu,
        out_k,
        out_v,
        batch,
        topk,
        heads,
        dim,
    )
    for b in range(batch):
        n = int(w["seq_lens"][b])
        assert torch.isfinite(out_k[b * topk : b * topk + n]).all()
        assert torch.isfinite(out_v[b * topk : b * topk + n]).all()
        assert torch.isnan(out_k[b * topk + n : (b + 1) * topk]).all(), (
            "invalid cols written"
        )


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_fp4_fused_dims(dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    batch, topk, heads, pool_rows = 2, 128, 2, 2048
    w = _make_world(batch, topk, heads, dim, pool_rows, device, seed=7)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * topk
    k_rows, v_rows = _legacy_gather(w, 1, batch, topk, heads, dim)
    ref_k = torch.zeros(batch * topk, heads, dim, device=device, dtype=torch.bfloat16)
    ref_v = torch.zeros(batch * topk, heads, dim, device=device, dtype=torch.bfloat16)
    qwen_sparse_kv_extraction_gathered_rows_triton(
        k_rows, v_rows, w["indices"], w["seq_lens"], cu, ref_k, ref_v, batch, topk
    )
    fuse_k = torch.zeros(batch * topk, heads, dim, device=device, dtype=torch.bfloat16)
    fuse_v = torch.zeros(batch * topk, heads, dim, device=device, dtype=torch.bfloat16)
    qwen_sparse_kv_gather_dequant_fp4_triton(
        w["k_fp4"],
        w["v_fp4"],
        w["k_sf"],
        w["v_sf"],
        w["k_gs"][1:2],
        w["v_gs"][1:2],
        w["req_to_token"],
        w["req_indices"],
        w["indices"],
        w["seq_lens"],
        cu,
        fuse_k,
        fuse_v,
        batch,
        topk,
        heads,
        dim,
    )
    assert torch.equal(fuse_k.view(torch.int16), ref_k.view(torch.int16))
    assert torch.equal(fuse_v.view(torch.int16), ref_v.view(torch.int16))
