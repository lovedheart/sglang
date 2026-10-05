"""Parity tests for the packed-NVFP4 sparse prefill kernel.

The packed kernel reads the KV cache's nibbles directly instead of dequantizing
the selected rows into BF16 scratch first.  Both arms consume the *same* packed
rows and block scales, so quantization error cancels and the only legitimate
difference is the arithmetic route: two nibble-plane dots accumulated in a
different fp32 order versus one head_dim-wide dot on a cast tile.

The bar depends on how well conditioned the softmax is.  On data like a real
cache holds (quantized Gaussians, as the pool stores them) the arms agree to a
BF16 ULP or two.  On unstructured random nibbles scaled by an uncalibrated
global scale of 1.0 the scores blow up, softmax degenerates into an argmax, and
a last-bit score difference can flip which row wins -- the output then moves by
a whole V row rather than a ULP.  That regime is asserted as a *rate*, because
it is a property of the tie and not of the kernel: a mis-read cache spoils most
of the output, not three elements in a hundred thousand.
"""

import pytest
import torch

from sglang.srt.utils import is_sm100_supported, is_sm120
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
    sparse_gqa_fwd_interface_triton_ck_nvfp4,
)
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil as U

# The nibble split is one PTX instruction that exists nowhere else; an FP4 KV
# cache does exist on SM90, where it is dequantized in software instead.
pytestmark = pytest.mark.skipif(
    not (is_sm100_supported() or is_sm120()),
    reason="the packed e2m1 path needs SM100/SM120",
)

GS = 1.0
DEV = "cuda"


def _packed_cache(rows, kv_heads, head_dim, gen):
    """Quantize Gaussian K/V the way the pool does, so scales stay in range."""
    out = []
    for _ in range(2):
        x = (torch.randn(rows, kv_heads, head_dim, generator=gen) * 0.6).to(
            torch.bfloat16
        )
        pk, sf, _ = U.quantize(x.cuda(), GS)
        out += [pk, sf.view(torch.uint8)]
    return out


def _random_rows(rows, kv_heads, head_dim, gen):
    """Legal bytes with no relation to any activation, which makes the softmax
    argmax knife edged: nibbles all over the code book, scales over the range."""
    out = []
    for _ in range(2):
        pk = torch.randint(0, 256, (rows, kv_heads, head_dim // 2), generator=gen)
        # 0x7F / 0xFF are the e4m3 NaN bytes; keep them out of the scale data.
        sf = torch.randint(1, 127, (rows, kv_heads * (head_dim // 16)), generator=gen)
        out += [
            pk.to(torch.int64).to(torch.uint8).cuda(),
            sf.to(torch.int64).to(torch.uint8).view(torch.uint8).cuda(),
        ]
    out[1] = out[1].view(torch.uint8).reshape(rows, kv_heads, head_dim // 16)
    out[3] = out[3].view(torch.uint8).reshape(rows, kv_heads, head_dim // 16)
    return out


def _build(rows, topk, lens, kv_heads, head_dim, q_ratio, seed, cache_like):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(sum(lens), kv_heads * q_ratio, head_dim, generator=gen) * 0.5).to(
        torch.bfloat16
    )
    k_pk, k_sf, v_pk, v_sf = (
        _packed_cache(rows, kv_heads, head_dim, gen)
        if cache_like
        else _random_rows(rows, kv_heads, head_dim, gen)
    )
    # Indices are positions inside a sequence's own gathered rows, which is how
    # the chunk kernel addresses them; a fraction is -1 padding.
    idx = torch.cat(
        [torch.randint(0, ln, (ln, topk), generator=gen) for ln in lens]
    ).to(torch.int32)
    idx[torch.rand(sum(lens), topk, generator=gen) < 0.15] = -1
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lens), 0)), dtype=torch.int32
    )
    return (
        q.cuda(),
        [t.cuda() for t in (k_pk, k_sf, v_pk, v_sf)],
        idx.cuda(),
        cu_q.cuda(),
        cu_q.clone().cuda(),
    )


def _both_arms(q, packed, idx, cu_q, cu_k, lens, scale):
    k_pk, k_sf, v_pk, v_sf = packed
    scale_t = torch.tensor([GS], device=DEV)
    ref = sparse_gqa_fwd_interface_triton_ck(
        q,
        U.dequantize(k_pk, k_sf, scale_t, dtype=torch.bfloat16),
        U.dequantize(v_pk, v_sf, scale_t, dtype=torch.bfloat16),
        idx,
        cu_q,
        cu_k,
        torch.tensor(lens, dtype=torch.int32, device=DEV),
        scale,
        max_q=max(lens),
    )
    got = sparse_gqa_fwd_interface_triton_ck_nvfp4(
        q,
        k_pk,
        k_sf,
        v_pk,
        v_sf,
        GS,
        GS,
        idx,
        cu_q,
        cu_k,
        torch.tensor(lens, dtype=torch.int32, device=DEV),
        scale,
        max_q=max(lens),
    )
    return ref, got


@pytest.mark.parametrize("head_dim", [256, 128, 64])
@pytest.mark.parametrize("kv_heads,q_ratio", [(4, 4), (1, 8)])
@pytest.mark.parametrize("lens,topk", [([24, 40], 32), ([64], 8), ([8, 32, 56], 16)])
def test_packed_prefill_matches_the_dequantizing_path(
    head_dim, kv_heads, q_ratio, lens, topk
):
    q, packed, idx, cu_q, cu_k = _build(
        sum(lens),
        topk,
        lens,
        kv_heads,
        head_dim,
        q_ratio,
        seed=head_dim + topk,
        cache_like=True,
    )
    ref, got = _both_arms(q, packed, idx, cu_q, cu_k, lens, head_dim**-0.5)
    assert torch.isfinite(got).all()
    delta = (got.float() - ref.float()).abs()
    # Two BF16 ULPs at these magnitudes; a mis-read cache moves ~1e-1 instead.
    assert delta.max().item() <= 2e-3, delta.max().item()


def test_packed_prefill_only_differs_at_ties_on_unstructured_data():
    q, packed, idx, cu_q, cu_k = _build(
        64, 8, [64], 4, 256, 4, seed=11, cache_like=False
    )
    ref, got = _both_arms(q, packed, idx, cu_q, cu_k, [64], 256**-0.5)
    assert torch.isfinite(got).all()
    # Random nibbles turn the softmax into an argmax, so allow the odd tie flip
    # but not a systematic difference.
    assert ((got.float() - ref.float()).abs() > 2e-3).float().mean().item() <= 1e-3


def test_packed_prefill_honours_the_causal_cap():
    """One selectable row per query: the answer must be that row's V exactly."""
    lens, topk, head_dim, kv_heads, rows = [8], 64, 256, 2, 8
    q, packed, idx, cu_q, cu_k = _build(
        rows, topk, lens, kv_heads, head_dim, 4, seed=7, cache_like=True
    )
    idx = torch.full((rows, topk), -1, dtype=torch.int32, device=DEV)
    idx[:, :1] = 0
    ref, got = _both_arms(q, packed, idx, cu_q, cu_k, lens, head_dim**-0.5)
    # The output is per query head, so head h carries the V of kv head h // q_ratio.
    expected = (
        U.dequantize(
            packed[2], packed[3], torch.tensor([GS], device=DEV), dtype=torch.bfloat16
        )[0]
        .float()
        .repeat_interleave(4, dim=0)  # q_ratio
        .unsqueeze(0)
        .expand(rows, 4 * kv_heads, head_dim)
    )
    torch.testing.assert_close(
        got.float().reshape(rows, 4 * kv_heads, head_dim), expected, rtol=0, atol=2e-3
    )
    assert (got.float() - ref.float()).abs().max().item() <= 2e-3


class _StubMethod:
    name = "nvfp4"

    def get_bmm_scales(self, layer_id):
        return GS, GS


class _StubPool:
    def __init__(self, buffers):
        self._buffers = buffers

    def get_raw_kv_buffer(self, layer_id):
        return self._buffers


class _StubLayer:
    layer_id = 0


def test_backend_helper_uses_the_packed_path_and_falls_back_correctly():
    """The call site, not just the kernel: a silent fallback would leave the flag
    doing nothing, and a wrong argument order would show up here as a mismatch."""
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )

    lens, topk, head_dim, kv_heads, rows = [16, 24], 16, 256, 2, 40
    q, packed, idx, cu_q, cu_k = _build(
        rows, topk, lens, kv_heads, head_dim, 4, seed=5, cache_like=True
    )
    all_slots = torch.arange(rows, dtype=torch.int32, device=DEV)
    kv_lens = torch.tensor(lens, dtype=torch.int32, device=DEV)

    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.kv_cache_quant_method = _StubMethod()
    # The pool hands out (k, v, k_scales, v_scales), which is a different order
    # from the attention interface's.
    k_pk, k_sf, v_pk, v_sf = packed
    backend.token_to_kv_pool = _StubPool((k_pk, v_pk, k_sf, v_sf))
    run = lambda flag: (
        backend._packed_fp4_prefill(
            _StubLayer(),
            all_slots,
            q,
            idx,
            cu_q,
            cu_k,
            kv_lens,
            head_dim**-0.5,
            max_q=max(lens),
        ),
    )[0]
    with envs.SGLANG_QSA_PREFILL_PACKED_KV.override(True):
        got = run(True)
    assert got is not None, "the packed path silently declined to run"
    with envs.SGLANG_QSA_PREFILL_PACKED_KV.override(False):
        assert run(False) is None, "the flag must gate the path"
    # A differently laid-out cache must fall back rather than read garbage.
    k_pk, k_sf, v_pk, v_sf = packed
    backend.token_to_kv_pool = _StubPool((k_pk.transpose(1, 2), v_pk, k_sf, v_sf))
    with envs.SGLANG_QSA_PREFILL_PACKED_KV.override(True):
        assert run(True) is None, "a transposed cache must not take the packed path"
    ref, _ = _both_arms(q, packed, idx, cu_q, cu_k, lens, head_dim**-0.5)
    assert (got.float() - ref.float()).abs().max().item() <= 2e-3
