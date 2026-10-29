#!/usr/bin/env python3
"""Gain-requant error-neutrality check for the HC-mix MXFP8 fold path.

Fusion (1) folds the RMSNorm gain (1+wn) into input_mix_weight_down columns
BEFORE MXFP8 quantization (per-branch rsqrt stays a post-GEMM fp32 scalar, as
in _mix_folded). That requantizes a different matrix than today's direct path,
so per-element rounding lands differently. This sweeps EVERY module of the
real checkpoint and compares mixed-output relerr:

    direct  : silu((norm(x) @ Q(Wd).T)/HC) @ Q(Wu).T          (shipped path)
    foldrq  : g * silu(((X*(1+wn)) @ Q(Wd*row(1+wn)).T)/HC)   (proposed fold)

PASS criterion: foldrq weight relerr ~= direct (both ~2.66%) and no module's
mixed relerr exceeds direct by more than ~1.5x.

Usage:  python measure_hc_mix_gain_requant.py [L]
Env:    HC_MIX_CKPT
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
L = int(os.environ.get("L", "256"))
LAYERS = ([int(x) for x in os.environ["LAYERS"].split(",")]
          if "LAYERS" in os.environ else list(range(48)))
HC, HS = 4, 2560
EPS = 1e-6

_fb = {}


def load(p):
    with open(p, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        return h, f.read()


def header(p):
    with open(p, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


IDX = {}
for p in sorted(glob.glob(os.path.join(D, "*.safetensors"))):
    try:
        h = header(p)
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
    t = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(t.copy()).reshape(d["shape"])


def deq(w):
    q, s = mxfp8_quantize_rowmajor(w.bfloat16().contiguous())
    return mxfp8_dequantize_to_bf16(q, s)


def mix_from_down(h_down, Wu, xg):
    t = torch.nn.functional.silu(h_down.float() / HC).to(Wu.dtype)
    g = torch.sigmoid(torch.nn.functional.linear(t, Wu)).unflatten(-1, (HC, HS))
    return (g.to(xg.dtype) * xg.unflatten(-1, (HC, HS))).mean(-2)


rows = []
gain_min, gain_max = 1e9, -1e9
targets = [
    (f"model.language_model.layers.{i}.{k}_hyper_connection", f"L{i:02d}.{k}")
    for i in LAYERS
    for k in ("attn", "mlp")
]
targets.append(("model.language_model.hyper_connection_mixer", "mixer "))
targets += [
    (f"mtp.layers.{i}.{k}_hyper_connection", f"MTP{i}.{k} ")
    for i in LAYERS if i == 0
    for k in ("attn", "mlp")
]
targets.append(("mtp.hyper_connection_mixer", "MTPmixer"))

for base, tag in targets:
    try:
        wn = get(base + ".hc_norm.weight")
        Wd = get(base + ".input_mix_weight_down.weight")
        Wu = get(base + ".input_mix_weight_up.weight")
    except KeyError:
        continue
    gain = (1.0 + wn).reshape(HC, HS)
    gain_min, gain_max = min(gain_min, gain.min()), max(gain_max, gain.max())
    rng = np.random.default_rng(hash(tag) % 2**31)
    Xr = torch.from_numpy(rng.standard_normal((L, HC, HS))).float()  # raw hyper input
    gain = (1.0 + wn).reshape(HC, HS)
    gain_min, gain_max = min(gain_min, gain.min()), max(gain_max, gain.max())
    rng = np.random.default_rng(hash(tag) % 2**31)
    Xr = (torch.from_numpy(rng.standard_normal((L, HC, HS)))
          * gain).bfloat16().float()                     # raw hyper input
    g = torch.rsqrt((Xr * Xr).mean(-1, keepdim=True) + EPS)  # [L,HC,1]
    xg = (Xr * gain.unsqueeze(0) * g).bfloat16().flatten(-2)  # norm(x)
    Wu_dq = deq(Wu)

    ref = mix_from_down(
        torch.nn.functional.linear(xg.float(), Wd.float()),
        Wu_dq, xg,
    ).float()

    # direct (shipped): deq(Wd) on the normed tensor
    h_dir = torch.nn.functional.linear(xg, deq(Wd)).float()
    e_dir = ((mix_from_down(h_dir, Wu_dq, xg) - ref).norm() / ref.norm()).item()

    # foldrq: (1+wn) folded into Wd columns + requant; rsqrt after the GEMM,
    # per-branch partials (exactly _mix_folded on packed tiles)
    Wd_fold = (Wd.reshape(HC, HS).T if False else
               (Wd * gain.reshape(1, HC * HS)))
    Wd_fold_dq = deq(Wd_fold)
    partials = torch.matmul(
        Xr.bfloat16().transpose(0, 1),                  # [HC, L, HS]
        Wd_fold_dq.view(320, HC, HS).permute(1, 2, 0),  # [HC, HS, 320]
    )                                                   # [HC, L, 320]
    h_fold = torch.einsum("bnl,nb->nl", partials.float(), g.flatten(1))  # /HC inside mix_from_down
    e_fq = ((mix_from_down(h_fold, Wu_dq, xg) - ref).norm()
            / ref.norm()).item()

    e_wd_direct = ((deq(Wd).float() - Wd.float()).norm() / Wd.float().norm()).item()
    e_wd_fold = ((Wd_fold_dq.float() - Wd_fold.float()).norm()
                 / Wd_fold.float().norm()).item()
    rows.append((tag, e_dir, e_fq, e_wd_direct, e_wd_fold))

d = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
print(f"{len(rows)} modules, L={L}, gain range [{gain_min:.3f}, {gain_max:.3f}]")
print(
    f"weight relerr   direct {d[:, 2].mean() * 100:.3f}% (max {d[:, 2].max() * 100:.3f}) | "
    f"foldrq {d[:, 3].mean() * 100:.3f}% (max {d[:, 3].max() * 100:.3f})"
)
print(
    f"mixed  relerr   direct {d[:, 0].mean() * 100:.3f}% (max {d[:, 0].max() * 100:.3f}) | "
    f"foldrq {d[:, 1].mean() * 100:.3f}% (max {d[:, 1].max() * 100:.3f})"
)
ratio = d[:, 1] / d[:, 0]
print(f"fold/direct mixed ratio: mean {ratio.mean():.3f} max {ratio.max():.3f}")
bad = [r for r, rt in zip(rows, ratio) if rt > 1.5 or r[2] > 0.03]
print("OUTLIERS (ratio>1.5 or foldrq>3%):", bad if bad else "none")
worst = sorted(rows, key=lambda r: -r[2])[:3]
print("worst foldrq modules:", [(r[0], f"{r[2] * 100:.2f}%") for r in worst])
print("VERDICT:", "error-neutral" if not bad and ratio.max() < 1.5 else "NOT neutral")
