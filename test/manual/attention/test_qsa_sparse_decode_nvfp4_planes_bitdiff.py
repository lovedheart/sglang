"""Bit-identity A/B for the NVFP4 V-plane rewrite of ``_sparse_decode_split``.

The V dequant switched from one interleaved tile (five ``tl.interleave``
value/scale-factor shuffles) to even/odd planes feeding two dots whose
reduction dimension is still ``CHUNK``.  Per-element operands and the HMMA
k-order are unchanged by construction, so every output bit must match the
pre-rewrite kernel.  That is the derivation this guards: a "same-looking"
edit that splits the dot along the head dim, re-expands the scales with a
different replication, or re-joins the planes through ``tl.interleave``
before the store would silently reorder the fp32 accumulation and must turn
this test red.  The baseline module is materialized from ``git show
<BASELINE_SHA>:...sparse_decode.py``, the last commit before the V-plane
rewrite.
"""

import importlib.util
import math
import pathlib as _pathlib
import subprocess
import tempfile

import pytest
import sglang.srt.layers.attention.qsa.sparse_decode as _sd
import torch

# Production QSA shape (config.json text_config): head_dim 256, 2 kv heads,
# 24 q heads, indexer budget 2048 -> final_topk row width 2051.
DIM, KV_HEADS, Q_HEADS = 256, 2, 24
TOPK = 2051
CTX = 8192
SLOTS = CTX * 4
SM_SCALE = 1 / math.sqrt(DIM)

_REPO_ROOT = _pathlib.Path(_sd.__file__).parents[6]
# The last commit whose sparse_decode.py still had the interleaved V tile.
# Pinning it (rather than HEAD) keeps the A/B meaningful after the rewrite is
# itself committed.
_BASELINE_SHA = "c13d59dfc87aa9db1399d6a55451224debd13b40"


def _baseline_module():
    """The pre-V-plane kernel source from ``_BASELINE_SHA``, as a genuinely
    distinct Triton module."""
    rel = _pathlib.Path(_sd.__file__).relative_to(_REPO_ROOT)
    src = subprocess.run(
        ["git", "show", f"{_BASELINE_SHA}:{rel}"],
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO_ROOT,
    ).stdout
    assert "for _ in tl.static_range(4):" in src, (
        f"{_BASELINE_SHA} is not the pre-V-plane baseline"
    )
    tmp = _pathlib.Path(tempfile.mkdtemp()) / "sparse_decode_pre_vplane.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("sparse_decode_pre_vplane", str(tmp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(rows, seed, mode="clean"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = torch.device("cuda")
    if mode == "pad":
        # seq_len < TOPK keeps trailing -1 columns live; the junk tail
        # mimics fast_topk selecting tokens past the scored context.
        seq = torch.full((rows,), TOPK - 3, dtype=torch.int32, device=device)
        clean = torch.arange(TOPK - 3, device=device, dtype=torch.int32).expand(
            rows, -1
        )
        junk = torch.randint(
            TOPK,
            SLOTS,
            (rows, 3),
            device=device,
            dtype=torch.int32,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        idx = torch.cat([clean, junk], dim=1).contiguous()
    elif mode == "zero":
        # CUDA-graph padded rows: empty sequence, fully -1 index row.
        seq = torch.zeros(rows, dtype=torch.int32, device=device)
        idx = torch.full((rows, TOPK), -1, dtype=torch.int32, device=device)
    else:
        seq = torch.full((rows,), CTX, dtype=torch.int32, device=device)
        idx = (
            torch.stack([torch.randperm(CTX, generator=g)[:TOPK] for _ in range(rows)])
            .to(torch.int32)
            .to(device)
        )
    # request r -> pool r % max(rows-1, 1): exercises non-identity req mapping
    req = torch.arange(rows, dtype=torch.int64, device=device) % max(rows - 1, 1)
    r2t = (
        torch.stack(
            [torch.randperm(SLOTS, generator=g)[:CTX] for _ in range(max(rows - 1, 1))]
        )
        .to(torch.int32)
        .to(device)
    )
    q = torch.randn(rows, Q_HEADS, DIM, generator=g, dtype=torch.bfloat16).to(device)
    k4 = (
        torch.randint(0, 256, (SLOTS, KV_HEADS, DIM // 2), generator=g)
        .to(torch.uint8)
        .to(device)
    )
    v4 = (
        torch.randint(0, 256, (SLOTS, KV_HEADS, DIM // 2), generator=g)
        .to(torch.uint8)
        .to(device)
    )
    # keep the fp8 e4m3 NaN bytes (0x7F/0xFF) out of the scale data
    ksf = (
        (torch.randn(SLOTS, KV_HEADS, DIM // 16, generator=g) * 0.2)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .to(device)
    )
    vsf = (
        (torch.randn(SLOTS, KV_HEADS, DIM // 16, generator=g) * 0.2)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .to(device)
    )
    return (
        q,
        k4,
        v4,
        ksf,
        vsf,
        torch.tensor([0.7], device=device),
        torch.tensor([1.3], device=device),
        idx,
        r2t,
        req,
        seq,
    )


def _run(fn, w):
    return fn(*w, SM_SCALE, True, {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["clean", "pad", "zero"])
@pytest.mark.parametrize("rows", [1, 2, 4, 16])
@pytest.mark.parametrize("seed", [0, 1])
def test_nvfp4_planes_are_bitwise_unchanged(rows, seed, mode):
    base = _baseline_module()
    w = _world(rows, seed, mode)
    a = _run(_sd.sparse_decode_attention, w)
    b = _run(base.sparse_decode_attention, w)
    assert torch.equal(a, b), (
        f"V-plane rewrite moved bits: rows={rows} seed={seed} mode={mode} "
        f"maxdiff={(a.float() - b.float()).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("pad", [False, True])
def test_bf16_branch_is_bitwise_unchanged(pad):
    """The rewrite moved the p@V dot inside a constexpr branch; the bf16
    kernel is the same source either way and must stay bit-frozen."""
    base = _baseline_module()
    w = _world(4, 0, "pad" if pad else "clean")
    g = torch.Generator(device="cpu").manual_seed(11)
    dev = torch.device("cuda")

    def bf16_pool():
        return torch.randn(SLOTS, KV_HEADS, DIM, generator=g, dtype=torch.bfloat16).to(
            dev
        )

    k, v = bf16_pool(), bf16_pool()
    a = _sd.sparse_decode_attention(
        w[0],
        k,
        v,
        None,
        None,
        None,
        None,
        *w[7:],
        SM_SCALE,
        False,
        {},
    )
    b = base.sparse_decode_attention(
        w[0],
        k,
        v,
        None,
        None,
        None,
        None,
        *w[7:],
        SM_SCALE,
        False,
        {},
    )
    assert torch.equal(a, b), "bf16 branch moved bits"
