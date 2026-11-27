#!/usr/bin/env python3
"""HC mix W8A8 microbench: production MXFP8 path (e4m3 W 1x32 pow2 + e4m3 A
1x32 pow2, Triton _hc_mix_a8w8_*) vs an fp8_e3m4 1x128 route (e3m4 W b128
fp32 scales + e4m3 A b128 fp32 scales through sm120_fp8_e3m4_gemm/cute_ws).

Accuracy: fake-quant emulation in fp32 vs fp64 reference on REAL checkpoint
mix weights (dequant->fp32 matmul is what block-scale kernels accumulate;
pow2 scales are exact in bf16 so the emulation is storage-faithful).
Speed: real kernels, real shapes (Qwen3.8-Flash-Next: HC=4 HS=2560 LR=320;
down N=320 K=10240, up N=10240 K=320 -> pad to 384 for the cute_ws K%128=0
rule; the 64 zero columns are masked by predication, scale=1 code=0).
"""

import glob
import json
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DEV = "cuda"
D = os.environ.get(
    "HC_MIX_CKPT",
    "/media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnE3M4b128-mtpV4",
)
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,8,47").split(",")]
HC, HS, LR, KD = 4, 2560, 320, 4 * 2560
KP = 384  # padded K for the up GEMM (320 -> 3 blocks of 128)
LR_P = 384  # padded N of the down GEMM (same reason, b16 route wants N%128)
ROWS = [int(x) for x in os.environ.get("ROWS", "4096,2048,1024,512,64,16").split(",")]

E4M3_MAX, E3M4_MAX = 448.0, 30.0


# ---------- checkpoint ----------
def load_mix(layer):
    pats = [
        f"model.language_model.layers.{layer}.attn_hyper_connection.input_mix_weight_down.weight",
        f"model.language_model.layers.{layer}.attn_hyper_connection.input_mix_weight_up.weight",
    ]
    for p in sorted(glob.glob(os.path.join(D, "*.safetensors"))):
        with open(p, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
            if all(k in hdr for k in pats):
                hdr.pop("__metadata__", None)
                blob = f.read()
            else:
                blob = None
        if blob is None:
            continue
        out = []
        for k in pats:
            d = hdr[k]
            o = d["data_offsets"]
            raw = torch.frombuffer(bytearray(blob[o[0] : o[1]]), dtype=torch.uint16)
            out.append(
                ((raw.to(torch.int32) << 16).view(torch.float32))
                .reshape(d["shape"])
                .to(torch.bfloat16)
                .to(DEV)
                .contiguous()
            )
        return out
    raise SystemExit(f"layer {layer} not found")


# ---------- quantizers (ground-truth emulation) ----------
def _e3m4_tables():
    vals = torch.empty(128, dtype=torch.float64)
    for b in range(128):
        e, m = (b >> 4) & 7, b & 15
        vals[b] = (
            float("nan")
            if b == 127
            else ((m / 16) * 0.25 if e == 0 else (2.0 ** (e - 3)) * (1 + m / 16))
        )
    pos = vals[:127]
    return pos, (pos[1:] + pos[:-1]) / 2


_T = None


def encode_e3m4(x):
    global _T
    if _T is None:
        _T = _e3m4_tables()
    _, mids = _T
    mids = mids.to(x.device)
    xd = x.double().clamp(-E3M4_MAX, E3M4_MAX)
    sign = torch.where(xd.signbit(), 128, 0).to(torch.uint8)
    flat = xd.abs().reshape(-1)
    idx = torch.searchsorted(mids, flat.contiguous()).clamp_(max=126)
    tie = flat == mids.gather(0, idx.clamp(max=mids.numel() - 1))
    idx = torch.where(tie & (idx & 1 == 1), idx + 1, idx)
    return (idx.to(torch.uint8).reshape(x.shape) | sign).contiguous()


def decode_e3m4(codes):
    pos, _ = _e3m4_tables()
    pos = pos.to(codes.device)
    sign = torch.where((codes >> 7) != 0, -1.0, 1.0)
    return sign * pos[(codes & 127).long()]


def dq_w_1x32_pow2_e4m3(x):  # production MXFP8 weight storage
    x = x.float()
    n, k = x.shape
    xb = x.view(n, k // 32, 32)
    e = torch.ceil(
        torch.log2(xb.abs().amax(-1, keepdim=True).clamp_min(1e-30) / E4M3_MAX)
    ).clamp(-126, 126)
    s = torch.exp2(e)
    return (xb / s).to(torch.float8_e4m3fn).float().mul(s).view(n, k)


def dq_w_b128_fp32_e3m4(x):  # candidate storage (K padded to a 128 multiple)
    x = x.float()
    n, k = x.shape
    if k % 128:  # the serving route zero-pads K to a 128 multiple (320 -> 384)
        xp = torch.zeros(n, -(-k // 128) * 128, device=x.device)
        xp[:, :k] = x
        pad = True
    else:
        xp, pad = x, False
    xb = xp.view(n, -1, 128)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30) / E3M4_MAX
    dq = (decode_e3m4(encode_e3m4(xb / s)) * s).view(n, -1)
    return dq[:, :k] if pad else dq


def dq_w_1x32_pow2_e3m4(x):  # isolates mantissa vs scale-granularity
    x = x.float()
    n, k = x.shape
    xb = x.view(n, k // 32, 32)
    e = torch.ceil(
        torch.log2(xb.abs().amax(-1, keepdim=True).clamp_min(1e-30) / E3M4_MAX)
    ).clamp(-126, 126)
    s = torch.exp2(e)
    return (decode_e3m4(encode_e3m4(xb / s)) * s).view(n, k)


def dq_w_b64_fp32_e3m4(x):  # candidate: e3m4 1x64 fp32 scales
    x = x.float()
    n, k = x.shape
    xb = x.view(n, -1, 64)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30) / E3M4_MAX
    return (decode_e3m4(encode_e3m4(xb / s)) * s).view(n, k)


def dq_a_b64_fp32(x):  # e4m3 acts, 1x64 fp32
    x = x.float()
    r, k = x.shape
    xb = x.view(r, -1, 64)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30) / E4M3_MAX
    return (xb / s).to(torch.float8_e4m3fn).float().mul(s).view(r, k)


def dq_a_1x32_pow2(x):  # production act path (mirrors _mxfp8_quant_act_kernel)
    return dq_w_1x32_pow2_e4m3(x)


def dq_a_b128_fp32(x):  # cute_ws a_scale semantics
    x = x.float()
    r, k = x.shape
    xb = x.view(r, k // 128, 128)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30) / E4M3_MAX
    return (xb / s).to(torch.float8_e4m3fn).float().mul(s).view(r, k)


def mix_math(nrm, wd, wu):
    x = nrm.float()
    t = F.silu((x @ wd.float().T) / HC)
    g = torch.sigmoid(t @ wu.float().T).view(-1, HC, HS)
    return ((g * x.view(-1, HC, HS)).sum(1) / HC).double()


def metrics(o, r):
    rel = ((o - r).norm() / r.norm()).item()
    cos = (1 - F.cosine_similarity(o.flatten(), r.flatten(), dim=0)).item() * 1e3
    snr = 20 * math.log10((r.norm() / (o - r).norm()).item())
    return rel, cos, snr


# ---------- e3m4-route epilogues ----------
@triton.jit
def _quant_act_b128_kernel(x_ptr, q_ptr, s_ptr, M, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    kb0 = tl.program_id(1)
    for i in tl.static_range(B // 128):
        off = kb0 * B + i * 128 + tl.arange(0, 128)
        x = tl.load(x_ptr + row * K + off).to(tl.float32)
        s = tl.maximum(tl.max(tl.abs(x)), 1e-30) / 448.0
        tl.store(q_ptr + row * K + off, (x / s).to(tl.float8e4nv))
        tl.store(s_ptr + row * (K // 128) + kb0 * (B // 128) + i, s)


@triton.jit
def _quant_act_b64_kernel(x_ptr, q_ptr, s_ptr, M, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    kb0 = tl.program_id(1)
    for i in tl.static_range(B // 64):
        off = kb0 * B + i * 64 + tl.arange(0, 64)
        x = tl.load(x_ptr + row * K + off).to(tl.float32)
        s = tl.maximum(tl.max(tl.abs(x)), 1e-30) / 448.0
        tl.store(q_ptr + row * K + off, (x / s).to(tl.float8e4nv))
        tl.store(s_ptr + row * (K // 64) + kb0 * (B // 64) + i, s)


@triton.jit
def _silu_quant_pad_kernel(t_ptr, q_ptr, s_ptr, M, LO: tl.constexpr, KS: tl.constexpr):
    # t fp32 [M,LO] -> silu(t/4)? NO: caller passes t already /HC. -> silu ->
    # e4m3 [M,KS] (pad cols zero), scales [M, KS//128] fp32.
    row = tl.program_id(0)
    for kb in tl.static_range(KS // 128):
        off = kb * 128 + tl.arange(0, 128)
        if (kb + 1) * 128 <= LO:
            t = tl.load(t_ptr + row * LO + off)
            t = t * tl.sigmoid(t)
            s = tl.maximum(tl.max(tl.abs(t)), 1e-30) / 448.0
            tl.store(q_ptr + row * KS + off, (t / s).to(tl.float8e4nv))
            tl.store(s_ptr + row * (KS // 128) + kb, s)
        elif kb * 128 < LO:  # ragged last real block (LO % 128 != 0)
            m = off < LO
            t = tl.load(t_ptr + row * LO + off, mask=m, other=0.0)
            t = t * tl.sigmoid(t)
            s = tl.maximum(tl.max(tl.abs(t)), 1e-30) / 448.0
            tl.store(q_ptr + row * KS + off, (t / s).to(tl.float8e4nv), mask=m)
            tl.store(s_ptr + row * (KS // 128) + kb, s)
        else:
            tl.store(q_ptr + row * KS + off, tl.zeros((128,), tl.float8e4nv))
            tl.store(s_ptr + row * (KS // 128) + kb, 1.0)


@triton.jit
def _combine_kernel(
    g_ptr, x_ptr, out_ptr, HSx: tl.constexpr, HCc: tl.constexpr, BJ: tl.constexpr
):
    row = tl.program_id(0)
    jb = tl.program_id(1)
    j = jb * BJ + tl.arange(0, BJ)
    acc = tl.zeros((BJ,), tl.float32)
    for c in tl.static_range(HCc):
        g = tl.load(g_ptr + row * (HSx * HCc) + c * HSx + j).to(tl.float32)
        x = tl.load(x_ptr + row * (HSx * HCc) + c * HSx + j).to(tl.float32)
        acc += tl.sigmoid(g) * x
    tl.store(out_ptr + row * HSx + j, (acc / HCc).to(out_ptr.dtype.element_ty))


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    torch.manual_seed(0)
    from sglang.kernels.ops.gemm.hc_mix_fp8 import (
        _hc_mix_a8w8_down_kernel,
        _hc_mix_a8w8_up_kernel,
        _mxfp8_quant_act_kernel,
        fused_hc_mix_fp8_w8a8,
    )
    from sglang.kernels.ops.gemm.sm120_fp8_e3m4 import (
        _gemm_route,
        sm120_fp8_e3m4_gemm,
    )
    from sglang.srt.layers.fp8_hc_mix import mxfp8_quantize_rowmajor

    print(f"device={torch.cuda.get_device_name()} ckpt={os.path.basename(D)}")
    print(f"HC={HC} HS={HS} LR={LR} rows={ROWS}\n")

    # ============ accuracy ============
    print("=== ACCURACY: mix out vs fp64 ref (64 tok, real weights) ===")
    print(f"{'variant':26s} {'relL2':>9s} {'1-cos[e-3]':>10s} {'SNR dB':>7s}")
    rowsA = {}
    for layer in LAYERS:
        wd, wu = load_mix(layer)
        x = torch.randn(64, KD, device=DEV).to(torch.bfloat16)
        ref = mix_math(x, wd, wu)
        cases = {
            "w8a8 e4m3 1x32 pow2 (prod)": (
                mix_math(
                    dq_a_1x32_pow2(x), dq_w_1x32_pow2_e4m3(wd), dq_w_1x32_pow2_e4m3(wu)
                )
            ),
            "w8a8 e3m4 b128 fp32 (new)": (
                mix_math(
                    dq_a_b128_fp32(x), dq_w_b128_fp32_e3m4(wd), dq_w_b128_fp32_e3m4(wu)
                )
            ),
            "w8a8 e3m4 1x32 pow2 (isolate)": (
                mix_math(
                    dq_a_1x32_pow2(x), dq_w_1x32_pow2_e3m4(wd), dq_w_1x32_pow2_e3m4(wu)
                )
            ),
            "w8a8 e3m4 1x64 fp32": (
                mix_math(
                    dq_a_b64_fp32(x), dq_w_b64_fp32_e3m4(wd), dq_w_b64_fp32_e3m4(wu)
                )
            ),
            "act-only e4m3 b128 (w fp32)": (mix_math(dq_a_b128_fp32(x), wd, wu)),
        }
        for nm, out in cases.items():
            rel, cos, snr = metrics(out, ref)
            rowsA.setdefault(nm, []).append((rel, cos, snr))
            print(f"L{layer:<2d} {nm:22s} {rel:9.5f} {cos:10.4f} {snr:7.2f}")
    print("\nmean over layers:")
    for nm, vs in rowsA.items():
        m = [sum(v[i] for v in vs) / len(vs) for i in range(3)]
        print(f"     {nm:26s} {m[0]:9.5f} {m[1]:10.4f} {m[2]:7.2f}")

    # ============ speed ============
    print("\n=== SPEED: real kernels (ms, median of 50) ===")
    wd, wu = load_mix(LAYERS[0])
    wd8, wd8s = mxfp8_quantize_rowmajor(wd)  # e4m3 [320,10240] + u8 [320,320]
    wu8, wu8s = mxfp8_quantize_rowmajor(wu)

    # e3m4 b128 packs (speed only needs shapes/layouts; codes from encoder)
    def pack_b128(w, n_out=None):
        n, k = w.shape
        nb, kb = n // 128 if n % 128 == 0 else -(-n // 128), k // 128
        wp = torch.zeros(nb * 128, kb * 128, device=DEV)
        wp[:n, :k] = w.float()
        blocks = wp.view(nb, 128, kb, 128)
        s = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-30) / E3M4_MAX
        codes = encode_e3m4(blocks / s).view(nb * 128, kb * 128)
        return codes.view(torch.float8_e4m3fn).contiguous(), s.view(
            nb, kb
        ).contiguous().float()

    # up weight padded K 320->384
    wu_pad = torch.zeros(HS * HC, KP, device=DEV)
    wu_pad[:, :LR] = wu.float()
    wd_codes, wd_scale = pack_b128(wd)  # [320->384,10240] rows pad -> scale [3,80]
    wu_codes, wu_scale = pack_b128(wu_pad)  # [10240,384] -> scale [80,3]

    print(
        f"{'rows':>5s} {'prod_w8a8':>10s} {'e3m4_route':>11s} {'speedup':>8s}   "
        f"{'actq':>6s} {'down':>6s} {'silu+q':>7s} {'up':>7s} {'comb':>6s}  route"
    )
    for M in ROWS:
        x = torch.randn(M, KD, device=DEV).to(torch.bfloat16)

        def prod():
            return fused_hc_mix_fp8_w8a8(x, wd8, wd8s, wu8, wu8s, HC, HS)

        t_prod = bench(prod)
        # production internals, timed per kernel
        p8 = torch.empty(M, KD, dtype=torch.float8_e4m3fn, device=DEV)
        psf = torch.empty(M, KD // 32, dtype=torch.uint8, device=DEV)
        pt8 = torch.empty(M, LR, dtype=torch.float8_e4m3fn, device=DEV)
        ptsf = torch.empty(M, LR // 32, dtype=torch.uint8, device=DEV)
        pout = torch.empty(M, HS, dtype=torch.bfloat16, device=DEV)
        t_pq = bench(
            lambda: _mxfp8_quant_act_kernel[(M, KD // 1024)](
                x, p8, psf, M, K=KD, BLOCK=1024, num_warps=4
            )
        )
        t_pd = bench(
            lambda: _hc_mix_a8w8_down_kernel[(triton.cdiv(M, 128), LR // 64)](
                p8,
                psf,
                wd8,
                wd8s,
                pt8,
                ptsf,
                M,
                KD,
                LR,
                1.0 / HC,
                BM=128,
                BN=64,
                BK=256,
                NKB=KD // 32,
                num_warps=8,
                num_stages=2,
            )
        )
        t_pu = bench(
            lambda: _hc_mix_a8w8_up_kernel[(HS // 64, triton.cdiv(M, 64))](
                pt8,
                ptsf,
                wu8,
                wu8s,
                x,
                pout,
                M,
                LOWRANK=LR,
                HS=HS,
                HC=HC,
                BM=64,
                BN=64,
                BK=64,
                num_warps=4,
                num_stages=3,
            )
        )

        # e3m4 route, all real kernels
        a8 = torch.empty(M, KD, dtype=torch.float8_e4m3fn, device=DEV)
        asf = torch.empty(M, KD // 128, dtype=torch.float32, device=DEV)
        t_dn = torch.empty(M, LR_P, dtype=torch.float32, device=DEV)
        t8 = torch.empty(M, KP, dtype=torch.float8_e4m3fn, device=DEV)
        t8.zero_()  # ragged pad columns of block 2 (cols 320..383) stay zero
        tsf = torch.empty(M, KP // 128, dtype=torch.float32, device=DEV)
        g = torch.empty(M, HS * HC, dtype=torch.bfloat16, device=DEV)
        out = torch.empty(M, HS, dtype=torch.bfloat16, device=DEV)
        K128 = KD // 128

        def actq():
            _quant_act_b128_kernel[(M, KD // 1024)](x, a8, asf, M, K=KD, B=1024)

        def down():
            t_dn.copy_(sm120_fp8_e3m4_gemm(a8, wd_codes, asf, wd_scale))

        def siluq():
            _silu_quant_pad_kernel[(M,)](t_dn, t8, tsf, M, LO=LR, KS=KP)

        def up():
            g.copy_(sm120_fp8_e3m4_gemm(t8, wu_codes, tsf, wu_scale))

        def comb():
            _combine_kernel[(M, HS // 64)](g, x, out, HSx=HS, HCc=HC, BJ=64)

        def route():
            actq()
            down()
            siluq()
            up()
            comb()

        route()
        torch.cuda.synchronize()
        t_actq = bench(actq)
        t_down = bench(down)
        t_silu = bench(siluq)
        t_up = bench(up)
        t_comb = bench(comb)
        a8_64 = torch.empty(M, KD, dtype=torch.float8_e4m3fn, device=DEV)
        asf64 = torch.empty(M, KD // 64, dtype=torch.float32, device=DEV)
        t_actq64 = bench(
            lambda: _quant_act_b64_kernel[(M, KD // 1024)](
                x, a8_64, asf64, M, K=KD, B=1024
            )
        )
        t_route = bench(route)
        rt = _gemm_route(M, LR, KD)
        print(
            f"{M:5d} {t_prod:10.3f} {t_route:11.3f} {t_prod / t_route:7.2f}x   "
            f"prod: pq={t_pq:.3f} pd={t_pd:.3f} pu={t_pu:.3f} | "
            f"e3m4: q128={t_actq:.3f} q64={t_actq64:.3f} dn={t_down:.3f} sil={t_silu:.3f} up={t_up:.3f} cb={t_comb:.3f}  {rt}"
        )


if __name__ == "__main__":
    main()
