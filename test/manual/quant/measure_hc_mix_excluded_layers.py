#!/usr/bin/env python3
"""MXFP8 error of the HC-mix modules the runtime flag KEEPS in BF16.

--enable-fp8-hc-mix deliberately excludes the last main layer (47) and every
MTP/draft mix pair. This measures, on real checkpoint weights, what that
conservatism costs: fake-quant the excluded pairs with the exact shipped
quantizer (fp8_hc_mix.mxfp8_quantize_rowmajor) and compare the mix output
against the BF16-weight reference. If the last-layer / MTP relerr is the same
as a mid layer's, the exclusions are pure insurance; if it is worse, the flag
is calibrated right.

Usage:  python measure_hc_mix_excluded_layers.py [L]     (L = rows, default 256)
Env:    HC_MIX_CKPT = checkpoint dir with BF16 hyper_connection weights.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("SGLANG_PY", "/home/lovedheart/sglang-fork-dev/python"))
from sglang.srt.layers.fp8_hc_mix import (  # noqa: E402
    mxfp8_dequantize_to_bf16,
    mxfp8_quantize_rowmajor,
)

D = os.environ.get(
    "HC_MIX_CKPT",
    "/media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnFP8b128-full",
)
L = int(sys.argv[1]) if len(sys.argv) > 1 else 256
HC, HS = 4, 2560

_fb = {}


def load(p):
    with open(p, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        return h, f.read()


IDX = {}
for p in sorted(glob.glob(os.path.join(D, "*.safetensors"))):
    try:
        h, _ = load(p)
    except Exception:
        continue
    for k in h:
        if "hyper_connection" in k and k not in IDX:
            IDX[k] = p


def get(k):
    p = IDX[k]
    if p not in _fb:
        _fb[p] = load(p)
    h, b = _fb[p]
    d = h[k]
    o = d["data_offsets"]
    raw = np.frombuffer(b, np.uint8, o[1] - o[0], o[0])
    if str(d["dtype"]) != "BF16":
        raise SystemExit(f"not bf16: {k}")
    t = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(t.copy()).reshape(d["shape"]).to(torch.bfloat16)


def mix(x, wd, wu):
    # x: [L, HC*HS] bf16-normed; same math as _mix_compute
    h = torch.nn.functional.linear(x, wd).float() / HC
    t = torch.nn.functional.silu(h)
    g = torch.sigmoid(torch.nn.functional.linear(t.to(x.dtype), wu)).unflatten(-1, (HC, HS))
    return (g * x.unflatten(-1, (HC, HS))).mean(-2)


def check(name, base, seed):
    wn = get(base + ".hc_norm.weight").float()
    Wd = get(base + ".input_mix_weight_down.weight")
    Wu = get(base + ".input_mix_weight_up.weight")
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(
        rng.standard_normal((L, HC, HS)) * (1.0 + wn.numpy().reshape(HC, HS))
    ).to(torch.bfloat16)
    rms = (X.float() ** 2).mean(-1, keepdim=True).sqrt()
    x = (X.float() / rms).to(torch.bfloat16).flatten(-2)

    ref = mix(x, Wd, Wu)
    qd, sd = mxfp8_quantize_rowmajor(Wd)
    qu, su = mxfp8_quantize_rowmajor(Wu)
    got = mix(x, mxfp8_dequantize_to_bf16(qd, sd), mxfp8_dequantize_to_bf16(qu, su))
    e_mix = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    e_wd = ((mxfp8_dequantize_to_bf16(qd, sd).float() - Wd.float()).norm()
            / Wd.float().norm()).item()
    e_wu = ((mxfp8_dequantize_to_bf16(qu, su).float() - Wu.float()).norm()
            / Wu.float().norm()).item()
    gate_ref = torch.sigmoid(
        torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(x, Wd).float() / HC)
            .to(x.dtype),
            Wu,
        )
    )
    print(
        f"{name:46s} mixed {e_mix * 100:5.2f}%  "
        f"Wd {e_wd * 100:4.2f}%  Wu {e_wu * 100:4.2f}%  "
        f"gate mean {gate_ref.mean():.3f} |x|max {x.abs().max():.1f}"
    )
    return e_mix


print(f"L={L} rows, MXFP8 (e4m3 + 1x32 UE8M0, shipped quantizer)")
print("-" * 100)
check("main layer 0 attn   (quantized in config)", "model.language_model.layers.0.attn_hyper_connection", 11)
check("main layer 46 attn  (quantized in config)", "model.language_model.layers.46.attn_hyper_connection", 12)
check("main layer 47 attn  (EXCLUDED: last layer)", "model.language_model.layers.47.attn_hyper_connection", 13)
check("main layer 47 mlp   (EXCLUDED: last layer)", "model.language_model.layers.47.mlp_hyper_connection", 14)
check("main final mixer    (EXCLUDED: final mixer)", "model.language_model.hyper_connection_mixer", 15)
check("mtp layer 0 attn    (EXCLUDED: draft)", "mtp.layers.0.attn_hyper_connection", 16)
check("mtp layer 0 mlp     (EXCLUDED: draft)", "mtp.layers.0.mlp_hyper_connection", 17)
check("mtp final mixer     (EXCLUDED: draft)", "mtp.hyper_connection_mixer", 18)
