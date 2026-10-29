#!/usr/bin/env python3
"""HC mix W8A8: quantize ACTIVATIONS too (not just weights), fp32-accum emulation.

Same math as measure.py (hyperconnection.py):
    n   = GroupedGemmaRMSNorm(hyper)              # per-branch, gain (1+w)
    t   = silu( (n @ Wd.T) / 4 )                  # Wd [320,10240]
    mix = sigmoid( t @ Wu.T )                     # Wu [10240,320] -> (4,2560) gates
    mixed = mean_i( mix_i * n_i )
    inj = 2*sigmoid( (n @ Wi.T) / 4 )
W8A8 = both GEMM operands are fp8 code + per-block power-of-2 (MXFP8) or fp32
(128-block / per-row) scales, exact fp32 accumulation (fake quant, the
dequant->fp32->dot that block-scale kernels are equivalent to).
"""
import json, glob, os, sys, numpy as np

try:
    np.set_config(assume_unique=1.0)      # float32 matmul on, no BLAS warnings
    np.blas.set_config(assume_unique=1.0)
except Exception:
    pass

D = os.environ.get(
    "HC_MIX_CKPT",
    "/media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnFP8b128-full",
)
L = int(os.environ.get("L", "256"))
LAYERS = [0, 8, 16, 24, 32, 40, 47]
KINDS = {"attn": "attn_hyper_connection", "mlp": "mlp_hyper_connection"}
HC = 4


def load(p):
    with open(p, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        h = json.loads(f.read(n)); h.pop("__metadata__", None)
        return h, f.read()


IDX = {}
for p in sorted(glob.glob(os.path.join(D, "*.safetensors"))):
    try:
        h, _ = load(p)
    except Exception:
        continue
    for k in h:
        if "hyper_connection" in k and k not in IDX:
            IDX[k] = (p, str(h[k]["dtype"]), h[k]["shape"])
_fb = {}


def get(k):
    p = IDX[k][0]
    if p not in _fb:
        _fb[p] = load(p)
    h, b = _fb[p]
    d = h[k]; o = d["data_offsets"]
    raw = np.frombuffer(b, np.uint8, o[1] - o[0], o[0])
    if str(d["dtype"]) != "BF16":
        raise SystemExit("not bf16: " + k)
    return ((raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            ).reshape(d["shape"]).astype(np.float64)


def q_e4m3(y):
    a = np.minimum(np.abs(y), 448.0)
    e = np.clip(np.floor(np.log2(np.maximum(a, 2.0 ** -11))), -6, 8)
    st = np.where(a >= 2.0 ** -6, 2.0 ** (e - 3.0), 2.0 ** -9)
    return np.sign(y) * np.round(np.abs(y) / st) * st


def _blocks(x, blk):
    R, C = x.shape
    if C % blk:
        p = np.zeros((R, -(-C // blk) * blk)); p[:, :C] = x
        out = p.reshape(R // 1 if blk else R, -1)  # keep pad in, scales see zeros
        return out.reshape(R, -1, blk), True
    return x.reshape(R, C // blk, blk), False


def mx(x, blk=32):
    t, _ = _blocks(x.astype(np.float64), blk)
    s = 2.0 ** np.floor(np.log2(np.where(np.abs(t).max(2) == 0, 1e-30,
                                         np.abs(t).max(2))) - np.log2(448.0))
    return (q_e4m3(t / s[..., None]) * s[..., None]).reshape(x.shape[0], -1)[:, :x.shape[1]]


def b128(x):
    t, _ = _blocks(x.astype(np.float64), 128)
    s = np.abs(t).max(2) / 448.0
    s = np.where(s == 0, 1e-30, s)
    return (q_e4m3(t / s[..., None]) * s[..., None]).reshape(x.shape[0], -1)[:, :x.shape[1]]


def pt(x):
    s = np.abs(x).max() / 448.0
    s = max(s, 1e-30)
    return q_e4m3(x / s) * s


W_Q = {"mxfp8": mx, "b128": b128, "pt": pt, "bf16": lambda w: w}
A_Q = {
    "mx": mx,                                   # mxfp8 acts, 32-block (320%32=0)
    "b128": b128,                               # 1x128 (t@320 pads)
    "pt": pt,                                   # per-tensor (row)
    "bf16": lambda a: a,
}
CFGS = [("w-mxfp8 a-bf16", "mxfp8", "bf16"), ("w-mxfp8 a-mx", "mxfp8", "mx"),
        ("w-b128 a-b128", "b128", "b128"), ("w-pt a-pt", "pt", "pt"),
        ("w-bf16 a-mx", "bf16", "mx"), ("w-mxfp8 a-pt", "mxfp8", "pt"),
        ("w-bf16 a-b128", "bf16", "b128")]

res = {c[0]: {"mix": [], "inj": []} for c in CFGS}
act_rng = {"n": [], "t": []}
for i in LAYERS:
    for kind, pre in KINDS.items():
        base = f"model.language_model.layers.{i}.{pre}"
        wn = get(base + ".hc_norm.weight")
        Wd = get(base + ".input_mix_weight_down.weight")
        Wu = get(base + ".input_mix_weight_up.weight")
        Wi = get(base + ".block_inject_weight.weight")
        rng = np.random.default_rng(1000 + i + (0 if kind == "attn" else 7))
        X = rng.standard_normal((L, HC, 2560)) * (1.0 + wn.reshape(HC, 2560))
        rms = np.sqrt((X ** 2).mean(-1, keepdims=True))
        n = (X / rms).reshape(L, HC * 2560)
        act_rng["n"].append(np.abs(n).max())

        def mixof(fw, fa):
            nq = fa(n)
            h = (nq.astype(np.float32) @ fw(Wd).T.astype(np.float32)).astype(np.float64) / HC
            t = h / (1.0 + np.exp(-h))  # silu = h*sigmoid(h)
            act_rng["t"].append(np.abs(t).max())
            tq = fa(t)
            m = 1.0 / (1.0 + np.exp(-(tq.astype(np.float32) @ fw(Wu).T.astype(np.float32)).astype(np.float64)))
            return m.reshape(L, HC, 2560)

        def injof(fw, fa):
            return 2.0 / (1.0 + np.exp(-((fa(n).astype(np.float32)
                                          @ fw(Wi).T.astype(np.float32)).astype(np.float64) / HC)))

        mref = mixof(W_Q["bf16"], A_Q["bf16"])
        iref = injof(W_Q["bf16"], A_Q["bf16"])
        mixed_ref = (mref * X).sum(1)
        for name, wq, aq in CFGS:
            m2 = mixof(W_Q[wq], A_Q[aq])
            i2 = injof(W_Q[wq], A_Q[aq])
            res[name]["mix"].append(np.linalg.norm((m2 * X).sum(1) - mixed_ref)
                                    / np.linalg.norm(mixed_ref))
            res[name]["inj"].append(np.linalg.norm(i2 - iref) / np.linalg.norm(iref))

print(f"{len(LAYERS)*len(KINDS)} HC groups, L={L} tokens (gaussian streams, model's own hc_norm gain)")
print(f"activation |max|: n {np.mean(act_rng['n']):.1f}  t(silu out) {np.mean(act_rng['t']):.3f}")
print(f"\n{'config':18s}{'mix relerr':>12s}{'inject relerr':>15s}")
for name, _, _ in CFGS:
    r = res[name]
    print(f"{name:18s}{np.mean(r['mix']):11.2e}{np.mean(r['inj']):14.2e}")
