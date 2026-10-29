"""Bit-identity A/B for the C1 valid-mask change in ``_sparse_decode_split``.

The new mask adds the paged path's ``idx < seqlen`` term.  For *clean*
top-k indices (every selected token < seqlen) the term is a provable
tautology, so the compiled kernel must be bit-identical and outputs must
match the pre-fix kernel bit-for-bit.  When an out-of-context (uninitialized
logits) token is present, outputs must diverge -- that is the bug being
fixed; earlier zero-diff cases there were fp32 underflow luck
(exp(s - m) flushing to subnormal zero), not a property of the mask.
"""

import importlib.util
import math
import pathlib as _pathlib
import tempfile

import pytest
import torch

import sglang.srt.layers.attention.qsa.sparse_decode as _sd
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

from sglang.srt.layers.attention.qsa.sparse_decode import sparse_decode_attention

# The valid mask as it must appear in the kernel source (C1 form).
_NEW_MASK = "valid = tmask & (idx >= 0) & (idx < seqlen)"
_PRE_MASK = "valid = tmask & (idx >= 0)"

DIM, KV_HEADS, Q_HEADS = 128, 1, 8
TOPK = 2051  # final_topk row width: token_topk + ratio - 1
CTX = 8192
SLOTS = CTX + 16
SCALE = 1.0 / math.sqrt(DIM)


def _load_pre_fix():
    """A module copy with the C1 mask term removed, so the A/B compiles a
    genuinely distinct Triton kernel."""
    src = _pathlib.Path(_sd.__file__).read_text()
    assert _NEW_MASK in src, "C1 valid mask missing from sparse_decode"
    tmp = _pathlib.Path(tempfile.mkdtemp()) / "sparse_decode_pre_c1.py"
    tmp.write_text(src.replace(_NEW_MASK, _PRE_MASK, 1))
    spec = importlib.util.spec_from_file_location("sparse_decode_pre_c1", str(tmp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(rows, junk, seed):
    device = torch.device("cuda")
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    q = torch.randn(rows, Q_HEADS, DIM, device=device, dtype=torch.bfloat16, generator=g)
    k = torch.randn(SLOTS, KV_HEADS, DIM, device=device, dtype=torch.bfloat16, generator=g) * 3.0
    v = torch.randn(SLOTS, KV_HEADS, DIM, device=device, dtype=torch.bfloat16, generator=g) * 3.0
    req = torch.zeros(rows, dtype=torch.int64, device=device)
    r2t = (
        torch.arange(SLOTS, device=device, dtype=torch.int32)[None, :]
        .expand(rows, -1)
        .contiguous()
    )
    # seq_len < TOPK keeps the trailing -1 pad columns live: this exercises
    # the tmask term being carried into the mask, without which the pre-fix
    # kernel would read garbage at the buffer tail.
    seq = torch.full((rows,), TOPK - 3, dtype=torch.int32, device=device)
    clean = torch.arange(TOPK - 3, device=device, dtype=torch.int32).expand(rows, -1)
    if junk:
        # The fast_topk tail shape: tokens past the scored context.
        junk_t = torch.randint(
            TOPK, SLOTS, (rows, 3), device=device, dtype=torch.int32, generator=g
        )
        idx = torch.cat([clean, junk_t], dim=1).contiguous()
    else:
        idx = clean.contiguous()
    return q, k, v, req, r2t, seq, idx


def _run(fn, w):
    q, k, v, req, r2t, seq, idx = w
    return fn(q, k, v, None, None, None, None, idx, r2t, req, seq, SCALE, False, {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_clean_indices_are_bitwise_unchanged(rows, seed):
    """The C1 mask is a tautology for clean top-k: no output bit may move."""
    pre = _load_pre_fix()
    w = _world(rows, junk=False, seed=seed)
    assert torch.equal(_run(_sd.sparse_decode_attention, w), _run(pre.sparse_decode_attention, w))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_garbage_tokens_change_the_result(pre_diff_threshold=1e-3):
    """Conversely the fix must be load-bearing: with out-of-context tokens
    present, the pre-fix kernel gathers their KV and the outputs diverge."""
    pre = _load_pre_fix()
    w = _world(2, junk=True, seed=7)
    diff = (_run(_sd.sparse_decode_attention, w).float() - _run(pre.sparse_decode_attention, w).float()).abs().max().item()
    assert diff > pre_diff_threshold, f"guard did not affect garbage output: {diff}"
