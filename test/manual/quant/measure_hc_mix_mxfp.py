#!/usr/bin/env python3
"""HC / gated-residual mix weights (BF16) -> FP8 block-128: measured error.
math (sglang fork, layers/hyperconnection.py):
   n   = GroupedGemmaRMSNorm(hyper)            # x/rms * (1+w)
   mix = sigmoid( W_up @ (W_down @ n)/hc )     # -> (hc=4, hs=2560) per-channel gates
   mixed_i = sum_j mix_j * x_j
   inj = 2*sigmoid( W_inject @ n / hc )        # (4,) stream->block injections
"""
import json, glob, os, numpy as np
D = os.environ.get(
    "HC_MIX_CKPT",
    "/media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnFP8b128-full",
)
def load(p):
    with open(p, "rb") as f:
        n = int.from_bytes(f.read(8), "little"); h = json.loads(f.read(n)); h.pop("__metadata__", None); return h, f.read()
IDX = {}
for p in sorted(glob.glob(os.path.join(D, "*.safetensors"))):
    try: h, _ = load(p)
    except Exception: continue
    for k, v in h.items():
        if 'hyper_connection' in k and k not in IDX: IDX[k] = (p, str(v['dtype']), v['shape'])
_fb = {}
def get(k):
    p = IDX[k][0]
    if p not in _fb: _fb[p] = load(p)
    h, b = _fb[p]; d = h[k]; o = d['data_offsets']
    raw = np.frombuffer(b, np.uint8, o[1]-o[0], o[0])
    if str(d['dtype']) == 'BF16':
        return ((raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)).reshape(d['shape']).astype(np.float64)
    raise SystemExit("not bf16: " + k + " " + str(d['dtype']))

def q_e4m3(y):
    a = np.minimum(np.abs(y), 448.0)
    e = np.clip(np.floor(np.log2(np.maximum(a, 2.0**-11))), -6, 8)
    st = np.where(a >= 2.0**-6, 2.0**(e-3.0), 2.0**-9)
    return np.sign(y)*np.round(np.abs(y)/st)*st
def fp8_2d(w, blk=128):
    R, C = w.shape
    pr, pc = -(-R//blk)*blk, -(-C//blk)*blk
    p = np.zeros((pr, pc)); p[:R, :C] = w
    t = p.reshape(pr//blk, blk, pc//blk, blk).swapaxes(1, 2)
    s = np.abs(t).max((2, 3)); s = np.where(s == 0, 1e-30, s)
    return (q_e4m3(t/s[..., None, None])*s[..., None, None]).swapaxes(1, 2).reshape(pr, pc)[:R, :C]
def fp8_k(w, blk=128):
    R, C = w.shape; t = w.reshape(R, C//blk, blk); s = np.abs(t).max(2)/448.0
    return (q_e4m3(t/s[..., None])*s[..., None]).reshape(R, C)
def fp8_pt(w):
    s = np.abs(w).max()/448.0; return q_e4m3(w/s)*s
def mxfp8(w, blk=32):
    R, C = w.shape; t = w.reshape(R, C//blk, blk)
    s = 2.0**np.floor(np.log2(np.abs(t).max(2)) - np.log2(448.0)); s = np.where(s == 0, 1e-30, s)
    return (q_e4m3(t/s[..., None])*s[..., None]).reshape(R, C)
G4 = np.array([0,.5,1,1.5,2,3,4,6.0])
def q_e2m1(y):
    a = np.minimum(np.abs(y), 6.0); return np.sign(y)*G4[np.digitize(a,(G4[1:]+G4[:-1])/2)]
def nvfp4(w, blk=16):
    R,C=w.shape; gs=np.abs(w).max()/(448.0*6.0); t=(w/gs).reshape(R,C//blk,blk)
    sq=q_e4m3(np.abs(t).max(2)/6.0); sq=np.where(sq==0,1e-30,sq)
    return (q_e2m1(t/sq[...,None])*sq[...,None]*gs).reshape(R,C)
def mxfp4(w, blk=32):
    R,C=w.shape; gs=np.abs(w).max()/(448.0*6.0); t=(w/gs).reshape(R,C//blk,blk)
    s=2.0**np.floor(np.log2(np.abs(t).max(2))-np.log2(6.0)); s=np.where(s==0,1e-30,s)
    return (q_e2m1(t/s[...,None])*s[...,None]*gs).reshape(R,C)
def snr(w, z): return 10*np.log10((w**2).sum()/max(((z-w)**2).sum(), 1e-300))

HC, HS, HCNT = 4, 2560, 4
LAYERS = [0, 8, 16, 24, 32, 40, 47]
KINDS = {"attn": "attn_hyper_connection", "mlp": "mlp_hyper_connection"}
tot = {}
res = {c: {"mix": [], "inj": [], "wd": [], "wu": [], "wi": []} for c in ["fp8_2d","fp8_k","fp8_pt","mxfp8","fp16","nvfp4","mxfp4"]}
npairs = 0
for i in LAYERS:
    for kind, pre in KINDS.items():
        base = f"model.language_model.layers.{i}.{pre}"
        wn = get(base + ".hc_norm.weight")                       # [4*2560]
        Wd = get(base + ".input_mix_weight_down.weight")         # [320, 10240]
        Wu = get(base + ".input_mix_weight_up.weight")           # [10240, 320]
        Wi = get(base + ".block_inject_weight.weight")           # [4, 10240]
        npairs += 1
        for key, w in (("wd", Wd), ("wu", Wu), ("wi", Wi)):
            tot.setdefault(key, []).append(w.size)
        rng = np.random.default_rng(1000+i+ (0 if kind == "attn" else 7))
        L = 128
        X = rng.standard_normal((L, HC, HS)) * (1.0 + wn.reshape(HC, HS))
        rms = np.sqrt((X**2).mean(-1, keepdims=True))            # Grouped Gemma RMSNorm per stream
        n = (X/rms).reshape(L, HC*HS)
        def mixof(WD, WU):
            h = n @ WD.T
            h = h/HCNT
            t = h/(1+np.exp(-h))          # silu, missing before
            m = 1/(1+np.exp(-(t @ WU.T)))
            return m.reshape(L, HC, HS)
        def injof(WI):
            return 2/(1+np.exp(-((n @ WI.T)/HCNT)))
        mref, iref = mixof(Wd, Wu), injof(Wi)
        mixed_ref = (mref*X).sum(1)                              # (L, HS) mixed input
        for c, f in (("fp8_2d", fp8_2d), ("fp8_k", lambda w: fp8_k(w) if w.shape[1] % 128 == 0 else fp8_pt(w)),
                     ("fp8_pt", fp8_pt), ("mxfp8", lambda w: mxfp8(w) if w.shape[1] % 32 == 0 else fp8_pt(w)),
                     ("nvfp4", lambda w: nvfp4(w) if w.shape[1] % 16 == 0 else fp8_pt(w)),
                     ("mxfp4", lambda w: mxfp4(w) if w.shape[1] % 32 == 0 else fp8_pt(w)),
                     ("fp16", lambda w: w.astype(np.float16).astype(np.float64))):
            m2 = mixof(f(Wd), f(Wu)); i2 = injof(f(Wi))
            mixed2 = (m2*X).sum(1)
            res[c]["mix"].append(np.linalg.norm(mixed2-mixed_ref)/np.linalg.norm(mixed_ref))
            res[c]["inj"].append(np.linalg.norm(i2-iref)/np.linalg.norm(iref))
            res[c]["wd"].append(snr(Wd, f(Wd))); res[c]["wu"].append(snr(Wu, f(Wu))); res[c]["wi"].append(snr(Wi, f(Wi)))
        if (i, kind) == (0, "attn"):
            print(f"mix gate stats: mean {mref.mean():.3f} p1 {np.percentile(mref,1):.3f} p99 {np.percentile(mref,99):.3f}"
                  f" | inject mean {iref.mean():.3f} | mixed/|x| {np.linalg.norm(mixed_ref)/np.linalg.norm(X):.3f}")
print(f"\n{npairs} HC groups sampled; shapes: down{get('model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight').shape}"
      f" up{get('model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight').shape}"
      f" inj{get('model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight').shape}")
print(f"params per layer per kind: down {tot['wd'][0]/1e6:.2f}M up {tot['wu'][0]/1e6:.2f}M inj {tot['wi'][0]/1e6:.3f}M")
print(f"\n{'scheme':9s}{'SNR down':>10s}{'SNR up':>8s}{'SNR inj':>9s}{'mix relerr':>12s}{'inject relerr':>15s}")
for c in ["fp16", "fp8_pt", "fp8_k", "fp8_2d", "mxfp8", "nvfp4", "mxfp4"]:
    r = res[c]
    print(f"{c:9s}{np.mean(r['wd']):9.2f}{np.mean(r['wu']):8.2f}{np.mean(r['wi']):9.2f}"
          f"{np.mean(r['mix']):11.2e}{np.mean(r['inj']):14.2e}")
