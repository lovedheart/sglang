"""FP3 W3A16 fused-MoE numerics + perf check against the real checkpoint.

Builds an 8-expert layer from layer-00000-experts.safetensors of
Qwen3.8-Flash-Next-FP3 (hidden 2560, inter 640), runs the sm12x_fp3_bf16
fused func, and compares against a float64 CPU reference that decodes the
checkpoint format directly:

    w = CB[code] * E4M3[sf & 0xFE] * scale_2 * (sf & 1 ? -1 : 1)
    CB = [-6, -2, -1, 0, 1, 2, 3, 6]
    payload: MSB-first 48-bit blocks, code j = (v >> (45 - 3j)) & 7

Run:  .venv/bin/python test/manual/quant/test_sm12x_fp3_moe.py
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from safetensors import safe_open

CKPT = "/media/lovedheart/models/Qwen3.8-Flash-Next-FP3/layer-00000-experts.safetensors"
CB = [-6.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 6.0]
E, HID, INTER, TOPK, T = 8, 2560, 640, 2, 16
PREFIX = "model.language_model.layers.0.mlp.experts"


def e4m3_bytes_to_f64(b: np.ndarray) -> np.ndarray:
    """Decode e4m3 (fn, no-nan-domain here) uint8 bytes to float64."""
    b = b.astype(np.uint16)
    sign = np.where(b >> 7, -1.0, 1.0)
    e = ((b >> 3) & 15).astype(np.int32)
    m = (b & 7).astype(np.float64)
    normal = np.ldexp(m / 8.0 + 1.0, e - 7)
    denormal = m * 2.0**-9
    return sign * np.where(e == 0, denormal, normal)


def decode_payload(payload: np.ndarray, sf: np.ndarray, gs: float) -> np.ndarray:
    """[N, K*3//8] u8 + [N, K//16] u8 -> [N, K] float64."""
    n, kb3 = payload.shape
    k = kb3 * 8 // 3
    big = payload.reshape(n, k // 16, 6).astype(np.uint64)
    v = (
        (big[..., 0] << 40) | (big[..., 1] << 32) | (big[..., 2] << 24)
        | (big[..., 3] << 16) | (big[..., 4] << 8) | big[..., 5]
    )
    codes = np.stack([(v >> (45 - 3 * j)) & 7 for j in range(16)], axis=-1)
    cb = np.array(CB)[codes]
    # Contract: e4m3 byte is sf & 0xFE; bit0 is the mirror (negated-codebook)
    # flag, NOT part of the e4m3 mantissa.
    scale = e4m3_bytes_to_f64(sf & 0xFE).astype(np.float64)
    scale = scale * np.where((sf & 1).astype(bool), -1.0, 1.0)
    w = cb * scale[..., None]
    return (w.reshape(n, k) * gs).astype(np.float64)


def load_experts():
    w13 = np.zeros((E, 2 * INTER, HID * 3 // 8), dtype=np.uint8)
    w13_sf = np.zeros((E, 2 * INTER, HID // 16), dtype=np.uint8)
    w13_gs = np.zeros((E, 2), dtype=np.float32)
    w2 = np.zeros((E, HID, INTER * 3 // 8), dtype=np.uint8)
    w2_sf = np.zeros((E, HID, INTER // 16), dtype=np.uint8)
    w2_gs = np.zeros(E, dtype=np.float32)
    with safe_open(CKPT, framework="numpy") as f:
        for e in range(E):
            p = f"{PREFIX}.{e}."
            for half, proj in ((0, "gate_proj"), (1, "up_proj")):
                w13[e, half * INTER:(half + 1) * INTER] = f.get_tensor(p + proj + ".weight")
                w13_sf[e, half * INTER:(half + 1) * INTER] = f.get_tensor(p + proj + ".weight_scale")
                w13_gs[e, half] = f.get_tensor(p + proj + ".weight_scale_2")
            w2[e] = f.get_tensor(p + "down_proj.weight")
            w2_sf[e] = f.get_tensor(p + "down_proj.weight_scale")
            w2_gs[e] = f.get_tensor(p + "down_proj.weight_scale_2")
    return w13, w13_sf, w13_gs, w2, w2_sf, w2_gs


def reference(x, ids, wts, tensors):
    """float64 CPU reference: decode all 8 experts, run the MoE directly."""
    w13, w13_sf, w13_gs, w2, w2_sf, w2_gs = tensors
    out = np.zeros((x.shape[0], HID), dtype=np.float64)
    deq13 = []
    for e in range(E):
        d = decode_payload(w13[e], w13_sf[e], 1.0)
        d[:INTER] *= w13_gs[e, 0]
        d[INTER:] *= w13_gs[e, 1]
        deq13.append(d)
    deq2 = [decode_payload(w2[e], w2_sf[e], float(w2_gs[e])) for e in range(E)]
    xd = x.astype(np.float64)
    for t in range(x.shape[0]):
        for slot in range(TOPK):
            e = int(ids[t, slot])
            g = xd[t] @ deq13[e][:INTER].T
            u = xd[t] @ deq13[e][INTER:].T
            act = g / (1.0 + np.exp(-g)) * u
            out[t] += float(wts[t, slot]) * (act @ deq2[e].T)
    return out


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tensors = load_experts()

    x = torch.randn(T, HID, dtype=torch.bfloat16, device="cuda")
    ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(T)])
    wts = torch.rand(T, TOPK, device="cuda", dtype=torch.float32)
    wts = wts / wts.sum(1, keepdim=True)

    # --- fused func (GPU) ---
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.sm12x_fp3_bf16 import (
        Sm12xFp3W3A16MoeQuantInfo,
        fused_experts_none_to_sm12x_fp3_bf16,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )

    qi = Sm12xFp3W3A16MoeQuantInfo(
        w13=torch.from_numpy(tensors[0]).cuda(),
        w13_sf=torch.from_numpy(tensors[1]).cuda(),
        w13_gs=torch.from_numpy(tensors[2]).cuda(),
        w2=torch.from_numpy(tensors[3]).cuda(),
        w2_sf=torch.from_numpy(tensors[4]).cuda(),
        w2_gs=torch.from_numpy(tensors[5]).cuda(),
    )
    cfg = MoeRunnerConfig(
        num_experts=E, num_local_experts=E, hidden_size=HID,
        intermediate_size_per_partition=INTER, top_k=TOPK,
    )
    topk_out = StandardTopKOutput(
        topk_weights=wts, topk_ids=ids.to(torch.int64), router_logits=None
    )
    dout = StandardDispatchOutput(
        hidden_states=x, hidden_states_scale=None, topk_output=topk_out
    )
    out = fused_experts_none_to_sm12x_fp3_bf16(dout, qi, cfg).hidden_states

    # --- float64 reference ---
    ref = reference(
        x.float().cpu().numpy(), ids.cpu().numpy(), wts.cpu().numpy(), tensors
    )
    got = out.float().cpu().numpy()
    # bf16 tensor-core GEMMs make elementwise max-rel meaningless near zero
    # (cancellation over K=2560); the meaningful bound is the per-token
    # relative error in the 2-norm of the output vector.
    rel = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)
    elem = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-2)
    print(f"mean|ref| = {np.abs(ref).mean():.4f}")
    print(f"max per-token rel (norm) : {rel.max():.3e}")
    print(f"mean per-token rel (norm): {rel.mean():.3e}")
    print(f"(elementwise worst, floor 1e-2: {elem.max():.3e})")
    assert (rel < 1e-2).all(), f"per-token rel err exceeds 1e-2: {rel}"
    print("NUMERICS OK: all per-token norm-rel < 1e-2 vs f64 reference")

    # --- GPU perf sanity (tiny; GPU is shared with a running server) ---
    from sglang.kernels.ops.moe.fp3_w3a16_moe_kernels import fused_experts_fp3_w3a16

    for t in (1, 32, 256):
        xt = torch.randn(t, HID, dtype=torch.bfloat16, device="cuda")
        idt = torch.randint(0, E, (t, TOPK), device="cuda")
        wt = torch.rand(t, TOPK, device="cuda")
        for _ in range(2):  # warm (first call JIT-compiles per shape bucket)
            fused_experts_fp3_w3a16(xt, qi.w13, qi.w13_sf, qi.w13_gs,
                                    qi.w2, qi.w2_sf, qi.w2_gs, wt, idt)
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fused_experts_fp3_w3a16(xt, qi.w13, qi.w13_sf, qi.w13_gs,
                                    qi.w2, qi.w2_sf, qi.w2_gs, wt, idt)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        print(f"T={t:4d} topk={TOPK} E={E}: {min(ts):8.1f} us (min of 3)")
    print("PERF OK")


if __name__ == "__main__":
    main()
