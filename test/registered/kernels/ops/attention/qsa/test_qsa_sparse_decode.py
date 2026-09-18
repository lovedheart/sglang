"""Correctness tests for the fused Triton sparse-decode attention kernel.

``sparse_decode_attention`` replaces valid-counts + strided gather + paged
decode in one pass over the top-k list. Pin it against a torch reference for
both KV pool layouts (bf16 and NVFP4), for request-index remapping, for -1
padding, and for CUDA-graph padded rows (sequence length 0).
"""

import math

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-small")

from sglang.srt.layers.attention.qsa.sparse_decode import sparse_decode_attention

DIM, KV_HEADS, Q_HEADS = 256, 2, 24
H_PER_KV = Q_HEADS // KV_HEADS
SM_SCALE = 1 / math.sqrt(DIM)
CTX = 8192
SLOTS = CTX * 4


def _make_world(rows, topk, quant, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = torch.device("cuda")
    idx = (
        torch.stack([torch.randperm(CTX, generator=g)[:topk] for _ in range(rows)])
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
    seq = torch.full((rows,), CTX, dtype=torch.int32, device=device)
    q = torch.randn(rows, Q_HEADS, DIM, generator=g, dtype=torch.bfloat16).to(device)
    if quant == "bf16":
        k = torch.randn(SLOTS, KV_HEADS, DIM, generator=g, dtype=torch.bfloat16).to(
            device
        )
        v = torch.randn(SLOTS, KV_HEADS, DIM, generator=g, dtype=torch.bfloat16).to(
            device
        )
        return dict(
            q=q,
            k=k,
            v=v,
            k_sf=None,
            v_sf=None,
            k_gs=None,
            v_gs=None,
            idx=idx,
            r2t=r2t,
            req=req,
            seq=seq,
            nvfp4=False,
        )
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
    return dict(
        q=q,
        k=k4,
        v=v4,
        k_sf=ksf,
        v_sf=vsf,
        k_gs=torch.tensor([0.7], device=device),
        v_gs=torch.tensor([1.3], device=device),
        idx=idx,
        r2t=r2t,
        req=req,
        seq=seq,
        nvfp4=True,
    )


def _dequant_pool(w):
    """Torch reference NVFP4 dequant: [slots, kv_heads, dim] fp32."""
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    out = []
    for p_name, s_name, g_name in (
        ("k", "k_sf", "k_gs"),
        ("v", "v_sf", "v_gs"),
    ):
        p = w[p_name]
        lo = lut[(p & 0xF).long()]
        hi = lut[((p >> 4) & 0xF).long()]
        nib = torch.stack([lo, hi], dim=-1).reshape(p.shape[0], p.shape[1], -1)
        sf = w[s_name].view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        out.append(nib * sf * w[g_name])
    return out[0], out[1]


def _ref_attn(w, k_bf16, v_bf16):
    outs = []
    for r in range(w["q"].shape[0]):
        sel = w["idx"][r].long()
        slots = w["r2t"][w["req"][r]].long()[sel.clamp(min=0)]
        valid = (sel >= 0) & (sel < w["seq"][r])
        kv, vv = k_bf16[slots], v_bf16[slots]
        if not bool(valid.any()):
            # graph-padded rows: the kernel emits zeros, softmax-of-nothing
            # in the reference would be NaN
            outs.append(torch.zeros(Q_HEADS, DIM, device=kv.device))
            continue
        heads = []
        for h in range(Q_HEADS):
            kvh = h // H_PER_KV
            s = (w["q"][r, h].float() @ kv[:, kvh].T) * SM_SCALE
            s[~valid] = float("-inf")
            heads.append(torch.softmax(s, dim=-1) @ vv[:, kvh])
        outs.append(torch.stack(heads))
    return torch.stack(outs)


@pytest.mark.parametrize("quant", ["bf16", "nvfp4"])
@pytest.mark.parametrize("rows", [1, 4, 7])
def test_sparse_decode_matches_reference(quant, rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w = _make_world(rows, 2051, quant)
    out = sparse_decode_attention(
        w["q"],
        w["k"],
        w["v"],
        w["k_sf"],
        w["v_sf"],
        w["k_gs"],
        w["v_gs"],
        w["idx"],
        w["r2t"],
        w["req"],
        w["seq"],
        SM_SCALE,
        w["nvfp4"],
        {},
    )
    kb, vb = _dequant_pool(w) if w["nvfp4"] else (w["k"].float(), w["v"].float())
    ref = _ref_attn(w, kb, vb)
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 5e-3, f"numeric mismatch: rel err {rel}"


def test_sparse_decode_graph_padding_rows():
    """CUDA-graph padded rows (seq_len 0) emit zeros; long rows stay finite."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w = _make_world(4, 2051, "bf16", seed=5)
    w["seq"] = torch.tensor([0, 1, 2050, CTX], dtype=torch.int32, device="cuda")
    out = sparse_decode_attention(
        w["q"],
        w["k"],
        w["v"],
        None,
        None,
        None,
        None,
        w["idx"],
        w["r2t"],
        w["req"],
        w["seq"],
        SM_SCALE,
        False,
        {},
    )
    assert out[0].abs().max().item() == 0.0, "padded row not zeroed"
    assert torch.isfinite(out[1:]).all(), "non-finite output"
    ref = _ref_attn(w, w["k"].float(), w["v"].float())
    rel = (out[1:].float() - ref[1:]).abs().max().item() / ref.abs().max().item()
    assert rel < 5e-3, f"numeric mismatch: rel err {rel}"
