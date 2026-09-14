"""Numerics regression for the QSA paged trtllm/XQA decode path.

Pins the packed-stride arange block-table path used by sparse decode (64-row
pages, bf16) against a grouped fp32 SDPA reference. Covers the kv_heads == 1
(TP2) and kv_heads == 2 (Qwen3.8-Flash-Next TP1 GQA) head mappings, including
a short last row exercised through the per-row valid counts.
"""

from types import SimpleNamespace

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend

try:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
except ImportError:
    trtllm_batch_decode_with_kv_cache = None

HEAD_DIM = 256

_capability = (
    torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
)
pytestmark = pytest.mark.skipif(
    trtllm_batch_decode_with_kv_cache is None or _capability[0] < 10,
    reason="trtllm paged decode requires a Blackwell-class GPU",
)


def _reference(q, pool_k, pool_v, pool_rows_per_req, seq_lens, group):
    """Grouped fp32 attention: q head h attends with kv head h // group."""
    batch, q_heads, _ = q.shape
    kv_heads = pool_k.shape[1]
    seq_cap = int(seq_lens.max())
    rows = torch.arange(batch, device=q.device)
    pos = (
        rows[:, None] * pool_rows_per_req
        + torch.arange(seq_cap, device=q.device)[None, :]
    )
    out = []
    for kv_head in range(kv_heads):
        kf = pool_k[pos, kv_head].float()
        vf = pool_v[pos, kv_head].float()
        mask = torch.arange(seq_cap, device=q.device)[None, :] < seq_lens[:, None]
        for offset in range(group):
            q_head = kv_head * group + offset
            scores = torch.bmm(
                q[:, q_head].float().unsqueeze(1), kf.transpose(1, 2)
            ) * (HEAD_DIM**-0.5)
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
            out.append(torch.bmm(scores.softmax(-1), vf).squeeze(1))
    ordered = torch.empty(
        batch, q_heads, HEAD_DIM, device=q.device, dtype=torch.float32
    )
    for kv_head in range(kv_heads):
        for offset in range(group):
            ordered[:, kv_head * group + offset] = out[kv_head * group + offset]
    return ordered


def _run_case(batch, kv_heads, topk, tail_seq_len):
    torch.manual_seed(0)
    device = torch.device("cuda")
    q_heads = kv_heads * 12
    pool_rows = batch * topk
    g = torch.Generator(device="cpu").manual_seed(7)
    pool_k = (
        torch.randn(pool_rows, kv_heads, HEAD_DIM, generator=g)
        .to(torch.bfloat16)
        .to(device)
    )
    pool_v = (
        torch.randn(pool_rows, kv_heads, HEAD_DIM, generator=g)
        .to(torch.bfloat16)
        .to(device)
    )
    q = torch.randn(batch, q_heads, HEAD_DIM, generator=g).to(torch.bfloat16).to(device)
    seq_lens = torch.full((batch,), topk, dtype=torch.int32)
    seq_lens[-1] = tail_seq_len
    seq_lens = seq_lens.to(device)
    # request b owns pool rows [b*topk, (b+1)*topk); token j maps to row b*topk+j
    req_to_token = (
        torch.arange(pool_rows, dtype=torch.int32).reshape(batch, topk).to(device)
    )
    topk_indices = torch.full((batch, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        n = int(seq_lens[b])
        topk_indices[b, :n] = torch.arange(n, dtype=torch.int32, device=device)

    backend = QwenSparseAttnBackend()
    backend._fused_sparse_decode = False
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    metadata = SimpleNamespace(
        is_cuda_graph=False,
        sequence_lengths=seq_lens,
        row_req_pool_indices=None,
        fa2_valid_counts=None,
    )
    forward_batch = SimpleNamespace(
        req_pool_indices=torch.arange(batch, dtype=torch.int64, device=device)
    )
    layer = SimpleNamespace(scaling=HEAD_DIM**-0.5, layer_id=0)

    out = backend._forward_trtllm_sparse(
        q,
        pool_k,
        pool_v,
        layer,
        forward_batch,
        metadata,
        topk_indices,
        trtllm_batch_decode_with_kv_cache,
    )
    ref = _reference(q, pool_k, pool_v, topk, seq_lens, 12)
    out = out.float().reshape(ref.shape)
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    assert rel < 0.02, f"paged decode drifted: max relative error {rel}"


def test_paged_decode_single_kv_head():
    _run_case(batch=16, kv_heads=1, topk=200, tail_seq_len=200)


def test_paged_decode_two_kv_heads():
    _run_case(batch=16, kv_heads=2, topk=200, tail_seq_len=200)


def test_paged_decode_two_kv_heads_with_short_row():
    _run_case(batch=8, kv_heads=2, topk=200, tail_seq_len=123)
