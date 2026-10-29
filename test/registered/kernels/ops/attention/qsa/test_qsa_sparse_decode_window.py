"""Guard test: the fused sparse-decode kernel must not let an out-of-context
top-k token contribute KV.

The decode scorer writes only [0, compressed_len) of its empty logits buffer,
so fast_topk can select a garbage *token* index past the scored context.  On
the paged trtllm path such a token is dropped by the valid-count mask
(qwen_sparse_valid_counts_triton keeps 0 <= idx < seq_len).  The fused
gather+attend kernel must drop it with the same mask -- otherwise a stale
token reads a random (never-written or recycled) KV row with a
position-dependent softmax weight, and the two decode paths silently diverge.
"""

import math
import pathlib as _pathlib

import pytest
import torch

import sglang.srt.layers.attention.qsa.sparse_decode as _sd
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

from sglang.srt.layers.attention.qsa.sparse_decode import sparse_decode_attention

# The valid-mask line as it must appear in the kernel source.
_GUARD = "valid = tmask & (idx >= 0) & (idx < seqlen)"


def _qsa_sd_guardless_path():
    return str(_pathlib.Path(_sd.__file__).with_name("_qsa_sd_guardless.py"))


def _load_guardless():
    """Import a copy of the sparse-decode module whose valid mask ignores
    seqlen, so the A/B runs JIT-compile genuinely distinct kernels."""
    import importlib.util

    src = _pathlib.Path(_sd.__file__).read_text()
    assert _GUARD in src, "seqlen term missing from the sparse-decode valid mask"
    path = _qsa_sd_guardless_path()  # leading underscore: not pytest-collected
    with open(path, "w") as f:
        f.write(src.replace(_GUARD, "valid = tmask & (idx >= 0)", 1))
    spec = importlib.util.spec_from_file_location("_qsa_sd_guardless", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DIM, KV_HEADS, Q_HEADS = 128, 1, 8
TOPK = 2048  # final_topk == token_topk + ratio - 1 == 2051 columns wide
CTX = 4096
SLOTS = CTX + 16


def _junk_tokens(rows, cols, device):
    """Tokens past every row's context but inside the table: an
    uninitialized logits tail selects exactly these once expanded."""
    return torch.randint(TOPK + 1, CTX, (rows, cols), device=device).to(torch.int32)


def _paged_reference(q, k, v, idx, seq):
    """The paged trtllm path drops every column whose token fails the
    valid-count mask (0 <= idx < seq_len); its KV row is never attended."""
    eff = idx.long().clamp(min=0)
    kx = k[eff].float()
    vx = v[eff].float()
    scores = torch.einsum("bhd,bnhd->bhn", q.float(), kx) * (1.0 / math.sqrt(DIM))
    valid = (idx >= 0) & (idx < seq[:, None])
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhn,bnhd->bhd", probs, vx)


def _world(rows, junk_cols, device, seed):
    torch.manual_seed(seed)
    req = torch.zeros(rows, dtype=torch.int64, device=device)
    # A context shorter than the allocation: valid tokens occupy [0, TOPK);
    # positions >= TOPK hold stale KV a garbage token would wrongly gather.
    r2t = (
        torch.arange(SLOTS, dtype=torch.int32, device=device)[None, :]
        .expand(rows, -1)
        .contiguous()
    )
    seq = torch.full((rows,), TOPK, dtype=torch.int32, device=device)
    q = torch.randn(rows, Q_HEADS, DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(SLOTS, KV_HEADS, DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(SLOTS, KV_HEADS, DIM, dtype=torch.bfloat16, device=device)
    idx = torch.cat(
        [
            torch.arange(TOPK, dtype=torch.int32, device=device).expand(rows, -1),
            _junk_tokens(rows, junk_cols, device),
        ],
        dim=1,
    ).contiguous()
    return q, k, v, req, r2t, seq, idx


def _fused(fn, q, k, v, req, r2t, seq, idx):
    return fn(
        q, k, v, None, None, None, None, idx, r2t, req, seq,
        1.0 / math.sqrt(DIM), False, {},
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_garbage_tokens_are_dropped_like_the_paged_path():
    device = torch.device("cuda")
    q, k, v, req, r2t, seq, idx = _world(3, 3, device, seed=0)
    out = _fused(sparse_decode_attention, q, k, v, req, r2t, seq, idx)
    ref = _paged_reference(q, k, v, idx, seq)
    assert torch.isfinite(out).all()
    rel = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    assert rel < 5e-3, f"garbage columns leaked into the output: rel {rel}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_guard_changes_garbage_columns_not_valid_ones():
    """Dropping garbage needs the seqlen term: tokens < seqlen are dropped by
    any mask, and without the seqlen term out-of-context garbage gathers its
    stale KV row -- so removing the term moves the output exactly when
    garbage is present, and never when it is not."""
    import os

    mod = _load_guardless()
    try:
        device = torch.device("cuda")
        q, k, v, req, r2t, seq, idx = _world(2, 3, device, seed=3)
        guarded = _fused(sparse_decode_attention, q, k, v, req, r2t, seq, idx)
        unguarded = _fused(mod.sparse_decode_attention, q, k, v, req, r2t, seq, idx)
        # Garbage < seqlen: both masks already agree (identity check).
        idx_in = idx.clone()
        idx_in[:, TOPK:] = 17
        g_in = _fused(sparse_decode_attention, q, k, v, req, r2t, seq, idx_in)
        n_in = _fused(mod.sparse_decode_attention, q, k, v, req, r2t, seq, idx_in)
        assert (g_in.float() - n_in.float()).abs().max().item() == 0.0
        # Out-of-context garbage: the seqlen term is load-bearing, and with
        # it the fused path matches the paged path.
        delta = (guarded.float() - unguarded.float()).abs().max().item()
        assert delta > 1e-3, "seqlen term is not load-bearing for garbage tokens"
        ref = _paged_reference(q, k, v, idx, seq)
        rel = (guarded.float() - ref).abs().max().item() / max(
            ref.abs().max().item(), 1e-6
        )
        assert rel < 5e-3
    finally:
        os.remove(_qsa_sd_guardless_path())
