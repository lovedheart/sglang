"""Tests for SGLANG_QSA_ATTN_FA4: the arch-owned Blackwell-SM120
FlashAttention-4 kernel driving QSA sparse decode over the page-aligned
packed scratch.

Pins (a) the host-side flag semantics, (b) numerics: the FA4 paged dispatch
must match a strict fp32 reference on the same gather (kernel reordering is
allowed, anything else is not), and (c) determinism: outputs must be
bit-identical across eager calls and across CUDA-graph captures and
replays, which is what MTP verify-reuse depends on.  SM120 hardware gates
the device tests; on other architectures they no-op.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-small")

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    _resolve_fa4_sm120_paged_decode,
    _resolve_flash_attn_varlen_func,
)

BF16 = torch.bfloat16
PAGE = 64
SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
requires_sm120 = pytest.mark.skipif(
    not SM120, reason="FA4 sm120 kernel requires an SM120 GPU"
)


class _Stub:
    """Just enough backend state for _fa4_paged_decode."""

    def __init__(self):
        self._fa4_out_scratch = {}


def _make_case(rows, topk, kv_heads=2, head_dim=256, q_heads=24, seed=0):
    """Page-aligned scratch + arange block table laid out exactly like
    _forward_trtllm_sparse (stride = ceil(topk/PAGE) pages per row)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages_per_row = (topk + PAGE - 1) // PAGE
    stride = pages_per_row * PAGE
    valid = torch.randint(1, topk + 1, (rows,), generator=g, device="cuda").int()
    packed_k = torch.zeros(
        rows * stride, kv_heads, head_dim, dtype=BF16, device="cuda"
    )
    packed_v = torch.zeros_like(packed_k)
    live_k = torch.randn(
        rows, topk, kv_heads, head_dim, generator=g, dtype=BF16, device="cuda"
    )
    live_v = torch.randn_like(live_k)
    for r in range(rows):
        # Valid prefix of each row is live; the tail stays zero-filled.
        packed_k[r * stride : r * stride + valid[r]] = live_k[r, : valid[r]]
        packed_v[r * stride : r * stride + valid[r]] = live_v[r, : valid[r]]
    kc = packed_k.view(-1, PAGE, kv_heads, head_dim)
    vc = packed_v.view(-1, PAGE, kv_heads, head_dim)
    block_tables = (
        torch.arange(rows, dtype=torch.int32, device="cuda")[:, None] * pages_per_row
        + torch.arange(pages_per_row, dtype=torch.int32, device="cuda")[None, :]
    ).contiguous()
    q = torch.randn(rows, q_heads, head_dim, generator=g, dtype=BF16, device="cuda")
    return q, kc, vc, block_tables, valid, stride, packed_k, packed_v


def _reference(q, packed_k, packed_v, valid, stride, scale):
    """Strict fp32 masked softmax over the per-row valid prefix."""
    rows, kv_heads, head_dim = q.shape[0], packed_k.shape[1], packed_k.shape[2]
    g = q.shape[1] // kv_heads
    out = torch.empty(rows, q.shape[1], head_dim, dtype=torch.float32, device=q.device)
    pk = packed_k.view(rows, stride, kv_heads, head_dim).float()
    pv = packed_v.view(rows, stride, kv_heads, head_dim).float()
    for r in range(rows):
        n = int(valid[r])
        kr, vr = pk[r, :n], pv[r, :n]  # (n, kvh, d)
        qe = q[r].float().view(kv_heads, g, head_dim)
        scores = torch.einsum("hgd,nhd->hgn", qe, kr) * scale
        probs = torch.softmax(scores, dim=-1)
        out[r] = torch.einsum("hgn,nhd->hgd", probs, vr).reshape(q.shape[1], head_dim)
    return out.to(BF16)


def test_fa4_env_flag_defaults_off():
    # Unset must mean off everywhere: the FA4 dispatch is strictly opt-in.
    if not envs.SGLANG_QSA_ATTN_FA4.is_set():
        assert envs.SGLANG_QSA_ATTN_FA4.get() is False


@requires_sm120
def test_fa4_resolvers_light_up_with_flag():
    with envs.SGLANG_QSA_ATTN_FA4.override(True):
        _resolve_fa4_sm120_paged_decode.cache_clear()
        _resolve_flash_attn_varlen_func.cache_clear()
        assert _resolve_fa4_sm120_paged_decode().__module__.endswith(
            "flash_attention_v4_sm120"
        )
        assert _resolve_flash_attn_varlen_func() is not None
    _resolve_fa4_sm120_paged_decode.cache_clear()
    _resolve_flash_attn_varlen_func.cache_clear()


@requires_sm120
@pytest.mark.parametrize("rows,topk", [(8, 512), (32, 512), (8, 2048), (1, 2048)])
def test_fa4_paged_matches_reference(rows, topk):
    scale = 1.0 / (256**0.5)
    q, kc, vc, bt, valid, stride, pk, pv = _make_case(rows, topk)
    stub = _Stub()
    out = QwenSparseAttnBackend._fa4_paged_decode(stub, q, kc, vc, bt, valid, scale)
    ref = _reference(q, pk, pv, valid, stride, scale)
    assert out.shape == q.shape
    # bf16 output quantization alone is 1 ulp (~0.4% of magnitude); allow
    # two ulps plus a floor on top of the exact fp32 reference.
    tol = ref.float().abs() * 2.0**-7 + 2e-3
    bad = ((out.float() - ref.float()).abs() > tol).sum().item()
    assert bad == 0, f"FA4 vs fp32 reference: {bad} elements outside tolerance"


@requires_sm120
def test_fa4_eager_bit_identical_across_calls():
    scale = 1.0 / (256**0.5)
    q, kc, vc, bt, valid, *_ = _make_case(16, 1024, seed=7)
    stub = _Stub()
    first = QwenSparseAttnBackend._fa4_paged_decode(
        stub, q, kc, vc, bt, valid, scale
    ).clone()
    for _ in range(5):
        again = QwenSparseAttnBackend._fa4_paged_decode(
            stub, q, kc, vc, bt, valid, scale
        )
        assert torch.equal(first, again), "FA4 eager outputs are not bit-identical"


@requires_sm120
def test_fa4_cuda_graph_replay_bit_identical():
    scale = 1.0 / (256**0.5)
    q, kc, vc, bt, valid, *_ = _make_case(8, 512, seed=11)
    stub = _Stub()
    # Warm the cached launch plan outside capture (host-side setup).
    ref = QwenSparseAttnBackend._fa4_paged_decode(
        stub, q, kc, vc, bt, valid, scale
    ).clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        QwenSparseAttnBackend._fa4_paged_decode(stub, q, kc, vc, bt, valid, scale)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = QwenSparseAttnBackend._fa4_paged_decode(
            stub, q, kc, vc, bt, valid, scale
        )
    # Capture records the cute launch path (the cached fast path deliberately
    # bails while the stream is capturing); replays must be bit-identical to
    # each other and within kernel-tolerance of the eager fast path.
    graph.replay()
    torch.cuda.synchronize()
    ref_replay = out.clone()
    tol = ref.float().abs() * 2.0**-6 + 2e-3
    assert ((out.float() - ref.float()).abs() <= tol).all()
    for _ in range(8):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref_replay), "FA4 graph replay diverged"

    # Re-capture must also be deterministic (plan cache reuse path).
    graph2 = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        QwenSparseAttnBackend._fa4_paged_decode(stub, q, kc, vc, bt, valid, scale)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph2):
        out2 = QwenSparseAttnBackend._fa4_paged_decode(
            stub, q, kc, vc, bt, valid, scale
        )
    graph2.replay()
    torch.cuda.synchronize()
    assert torch.equal(out2, ref_replay), "second FA4 capture diverged"


@requires_sm120
def test_fa4_matches_trtllm_dispatch():
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    except Exception:
        return  # no flashinfer on this runner: comparison not applicable
    scale = 1.0 / (256**0.5)
    q, kc, vc, bt, valid, stride, *_ = _make_case(8, 512, seed=3)
    stub = _Stub()
    fa4 = QwenSparseAttnBackend._fa4_paged_decode(stub, q, kc, vc, bt, valid, scale)
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    ref = trtllm_batch_decode_with_kv_cache(
        query=q.contiguous(),
        kv_cache=(kc.permute(0, 2, 1, 3), vc.permute(0, 2, 1, 3)),
        workspace_buffer=ws,
        block_tables=bt,
        seq_lens=valid,
        max_seq_len=stride,
        bmm1_scale=scale,
        bmm2_scale=1.0,
    )
    tol = ref.float().abs() * 2.0**-6 + 2e-3  # both sides quantize to bf16
    bad = ((fa4.float() - ref.float()).abs() > tol).sum().item()
    assert bad == 0, f"FA4 vs trtllm-gen: {bad} elements outside tolerance"
