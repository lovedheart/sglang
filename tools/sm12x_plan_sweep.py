#!/usr/bin/env python
"""Offline plan sweep for the SM12x NVFP4 MoE arms (W4A16 + W4A4).

Methodology borrowed from flashinfer PR #6614: enumerate a discrete plan
variant grid offline, measure per (arm, token bucket), keep a lookup table
instead of trusting the analytic score in ``_plan``.

Timing: CUDA-event median with a 128MB cold-L2 flush between iters.
Prints one flushed line per (config) so an interrupted sweep keeps its data.

Usage:
  python tools/sm12x_plan_sweep.py baseline            # current heuristic, both arms
  python tools/sm12x_plan_sweep.py w4a16 [T]           # sweep W4A16 plan grid at bucket T
  python tools/sm12x_plan_sweep.py w4a4  [T]           # sweep W4A4 (BM/TK/ST/GY) at bucket T
"""

import gc
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import sglang.kernels.ops.moe.sm12x_nvfp4_w4a16_moe as K

E, H, I, TOPK = 512, 2560, 640, 10
N13 = 2 * I
DEV = torch.device("cuda", torch.cuda.current_device())
L2FLUSH = None


def log(*a):
    print(*a, flush=True)


def flush_l2():
    global L2FLUSH
    if L2FLUSH is None:
        L2FLUSH = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    L2FLUSH.zero_()


def make_weights(gs_scale=1e-2):
    w13 = torch.randint(0, 256, (E, N13, H // 2), dtype=torch.uint8, device=DEV)
    w13_sf = (
        torch.rand(E, N13, H // 16, device=DEV) * 0.2
    ).float().to(torch.float8_e4m3fn)
    gs13 = (10 ** torch.empty(E, device=DEV).uniform_(-3, -2)).float()
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=DEV)
    w2_sf = (torch.rand(E, H, I // 16, device=DEV) * 0.2).float().to(
        torch.float8_e4m3fn
    )
    gs2 = (10 ** torch.empty(E, device=DEV).uniform_(-3, -2)).float()
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    return {
        "w13": w13,
        "w13_sf": swizzle_blockscale(w13_sf),
        "w13_gs": gs13,
        "w2": w2,
        "w2_sf": swizzle_blockscale(w2_sf),
        "w2_gs": gs2,
    }


def make_inputs(T):
    torch.manual_seed(0)
    x = torch.randn(T, H, dtype=torch.bfloat16, device=DEV)
    ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device=DEV)
    wts = torch.softmax(torch.rand(T, TOPK, device=DEV), dim=-1).float()
    return x, ids, wts


def timeit_graph(fn, iters=20, replays=15):
    """CUDA-graph replay timing: capture `iters` launches, time replays."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(replays):
        flush_l2()
        torch.cuda.synchronize()
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters)
    times.sort()
    return times[len(times) // 2], times[0]


def timeit(fn, reps_hi=31, reps_lo=11):
    for _ in range(3):  # warm + compile
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(reps_hi):
        flush_l2()
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    times.sort()
    return times[len(times) // 2], times[0]


# ---------------------------------------------------------------- W4A16 arm
def w4a16_launch_factory(T, x, ids, wts, W, sizes):
    out = torch.empty(T, H, dtype=torch.bfloat16, device=DEV)
    ws = [
        torch.zeros(sizes[0], dtype=torch.float32, device=DEV),
        torch.zeros(sizes[1], dtype=torch.int32, device=DEV),
        torch.empty(sizes[2], dtype=torch.bfloat16, device=DEV),
    ]

    def fn():
        K.run_moe_w4a16(
            x,
            ids,
            wts,
            W["w13"],
            W["w13_sf"],
            W["w13_gs"],
            W["w2"],
            W["w2_sf"],
            W["w2_gs"],
            out,
            workspace=ws,
        )

    return fn, out


def w4a16_variants(T):
    """Candidate (persistent, consts-override) plans for one bucket."""
    S = T * TOPK
    out = []
    # what the heuristics pick today
    BM0 = K._pick_bm(S, E)
    sf13, sf2 = K._scale_shape(E, N13, H), K._scale_shape(E, H, I)
    p0 = (
        K._persistent_plan(T, H, I, N13, TOPK, S, DEV, sf13, sf2)
        if BM0 == 1
        else None
    )
    if p0 is not None:
        out.append(("heuristic", (p0[0], p0[1], p0[2], True)))
    else:
        c, fp, ip = K._tiled_plan(T, H, I, N13, E, TOPK, S, BM0, sf13, sf2)
        out.append(("heuristic", (c, fp, ip, False)))
    # forced persistent (k_fused) at any BM: sweep its joint tpb choice
    for tpb in (128, 256, 512):
        tiles = None
        r1 = K._plan(H, N13, S, 1, max_split=1, tpb_req=tpb, ret_score=True)
        r2 = K._plan(I, H, S, 1, tpb_req=tpb, ret_score=True)
        if r1 and r2:
            tiles = (r1, r2)
        if tiles:
            (_s1, TPR1, TN1, SP1, NST1, NT1), (_s2, TPR2, TN2, SP2, NST2, NT2) = tiles
            KT1, KT2 = TPR1 * K.KPT * NST1, TPR2 * K.KPT * NST2
            num_items = max(S * NT1 * SP1, S * NT2 * SP2)
            grid_max = torch.cuda.get_device_properties(DEV).multi_processor_count * K.GMUL
            per_cta = (num_items + grid_max - 1) // grid_max
            OOF = S * N13
            consts = {
                "S": S, "TOPK": TOPK, "H": H, "I": I, "N13": N13, "E": E, "SWIGLU": True,
                "CP13": sf13[2], "NSF13": sf13[1] * sf13[2], "CP2": sf2[2], "NSF2": sf2[1] * sf2[2],
                "KT1": KT1, "KTP1": KT1 + (KT1 // 32) * 8, "TPR1": TPR1, "TN1": TN1, "NT1": NT1,
                "SP1": SP1, "NST1": NST1, "KT2": KT2, "KTP2": KT2 + (KT2 // 32) * 8, "TPR2": TPR2,
                "TN2": TN2, "NT2": NT2, "SP2": SP2, "NST2": NST2, "OOF": OOF, "CTROFF": 0, "SDOFF": 4,
                "TCOFF": 4 + S, "G": (num_items + per_cta - 1) // per_cta,
                "SMEL": max(KT1 + (KT1 // 32) * 8, KT2 + (KT2 // 32) * 8),
                "FALGN": K._algn(OOF * 4, H * 4),
                "MBPM": max(1, 65536 // (K.REGCAP * (TN1 * TPR1))),
                "STREAM": bool(E >= S * 4),
            }  # order must match moe_jitA constexprs
            if TN1 * TPR1 <= 256 and TN2 * TPR2 <= 256:  # SM max block width
                out.append((f"persist_tpb{tpb}", (consts, OOF + T * H, 4 + S + T, True)))
    # tiled: force BM ladder
    for BM in (1, 2, 4, 8, 16, 32, 64):
        if BM == BM0 and BM0 != 1:
            continue
        try:
            consts, fp, ip = K._tiled_plan(T, H, I, N13, E, TOPK, S, BM, sf13, sf2)
            out.append((f"tiled_bm{BM}", (consts, fp, ip, False)))
        except Exception as exc:  # pragma: no cover
            log(f"  tiled_bm{BM} skipped: {exc}")
    # tiled: DOTC toggle at default BM
    try:
        consts, fp, ip = K._tiled_plan(T, H, I, N13, E, TOPK, S, BM0, sf13, sf2)
        for dotc in (0, 1, 2):
            c2 = dict(consts)
            c2["DOTC"] = dotc
            out.append((f"tiled_bm{BM0}_dotc{dotc}", (c2, fp, ip, False)))
    except Exception:  # pragma: no cover
        pass
    return out


def run_w4a16(T, only=None):
    x, ids, wts = make_inputs(T)
    W = WEIGHTS
    res = {}
    for name, (consts, fp, ip, pers) in w4a16_variants(T):
        if only and only not in name:
            continue
        key = (DEV, T, H, I, N13, E, TOPK)
        sizes = (fp, ip, max(1, S_of(T) * I))
        K._COMPILED[key] = [pers, consts, sizes, None]
        try:
            fn, _out = w4a16_launch_factory(T, x, ids, wts, W, sizes)
            t, best = timeit(fn)
            res[name] = t
            log(f"w4a16 T={T} {name:26s} med={t:9.1f} us  min={best:9.1f}")
        except Exception as exc:
            log(f"w4a16 T={T} {name:26s} FAILED {str(exc)[:120]}")
        del fn
        K._COMPILED.pop(key, None)
        gc.collect()
        torch.cuda.empty_cache()
    return res


def S_of(T):
    return T * TOPK


# ---------------------------------------------------------------- W4A4 arm
_W4A4_SRC = None


def w4a4_source_patch(mod, bm=None, budget1=None, budget2=None, gy_div=None):
    """Patch M._build by source-editing the four knobs (sweep-only)."""
    import inspect

    import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4 as M

    global _W4A4_SRC
    if _W4A4_SRC is None:
        _W4A4_SRC = inspect.getsource(M._build)
    s = _W4A4_SRC
    if bm is not None:
        s = s.replace("BM = 32 if P <= 24 * E else 64", f"BM = {bm}")
    if budget1 is not None:
        s = s.replace("TK1, ST1 = _pick(H, PANEL1, 128, 98000)",
                      f"TK1, ST1 = _pick(H, PANEL1, 128, {budget1})")
    if budget2 is not None:
        s = s.replace("TK2, ST2 = _pick(I, PANEL2, 256, 49152)",
                      f"TK2, ST2 = _pick(I, PANEL2, 256, {budget2})")
    if gy_div is not None:
        s = s.replace("GY = min(WMAX, min(E, P) + max(1, (P + 8 * BM - 1) // (8 * BM)))",
                      f"GY = min(WMAX, min(E, P) + max(1, (P + {gy_div} * BM - 1) // ({gy_div} * BM)))")
    code_src = s
    fname = f"<w4a4_build_{bm}_{budget1}_{budget2}_{gy_div}>"
    code = compile(code_src, fname, "exec")
    import linecache

    linecache.cache[fname] = (len(code_src), None, code_src.splitlines(True), fname)
    glb = M._build.__globals__
    exec(code, glb)  # replaces M._build in place
    M._COMPILED.clear()


def w4a4_variants():
    """(name, patch-kwargs). The knobs _build exposes: M-tile height, the two
    smem budgets that set (TK, ST), and the grid-stride GY factor."""
    out = [("default", {})]
    for bm in (32, 64):
        out.append((f"bm{bm}", {"bm": bm}))
    for b2 in (24576, 49152, 98000):
        out.append((f"gemm2_budget{b2}", {"budget2": b2}))
    out.append(("gemm1_budget49152", {"budget1": 49152}))
    for gy in (4, 16):
        out.append((f"gy{gy}", {"gy_div": gy}))
    return out


def run_w4a4(T, only=None):
    import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4 as M

    x, ids, wts = make_inputs(T)
    W = WEIGHTS
    a13 = torch.ones((), device=DEV)
    a2 = torch.ones((), device=DEV)
    out = torch.empty(T, H, dtype=torch.bfloat16, device=DEV)
    res = {}
    for name, kw in w4a4_variants():
        if only and only not in name:
            continue
        try:
            w4a4_source_patch(M, **kw)
            ws = M.allocate_workspace(T, H, I, E, TOPK, True, DEV)

            def fn():
                M.run_moe_w4a4(
                    x, ids, wts, W["w13"], W["w13_sf"], W["w13_gs"],
                    W["w2"], W["w2_sf"], W["w2_gs"], a13, a2, out, workspace=ws,
                )

            t, best = timeit(fn)
            res[name] = t
            log(f"w4a4  T={T} {name:26s} med={t:9.1f} us  min={best:9.1f}")
        except Exception as exc:
            log(f"w4a4  T={T} {name:26s} FAILED {str(exc)[:120]}")
        del ws
        gc.collect()
        torch.cuda.empty_cache()
    w4a4_source_patch(M)  # restore pristine _build
    return res


WEIGHTS = None


def main():
    global WEIGHTS
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    WEIGHTS = make_weights()
    if mode == "baseline":
        import json

        buckets = [1, 4, 16, 64, 256, 1024, 2048, 4096]
        rep = {}
        for T in buckets:
            x, ids, wts = make_inputs(T)
            fn, _ = w4a16_launch_factory(
                T, x, ids, wts, WEIGHTS, K._entry(T, H, I, N13, E, TOPK, DEV)[2]
            )
            t, _b = timeit(fn)
            rep[f"w4a16_T{T}"] = t
            line = f"baseline w4a16 T={T:5d} med={t:9.1f} us"
            if T >= 256:
                try:
                    r = run_w4a4(T)
                    rep[f"w4a4_T{T}"] = r["default"]
                    line += f"   w4a4 med={r['default']:9.1f} us  ratio={t / r['default']:.2f}"
                except Exception as exc:
                    line += f"   w4a4 FAILED {str(exc)[:80]}"
            log(line)
            del fn, x, ids, wts
            gc.collect()
            torch.cuda.empty_cache()
        log("JSON " + json.dumps(rep))
    elif mode == "w4a16":
        for T in [int(a) for a in sys.argv[2:]] or [16]:
            run_w4a16(T)
    elif mode == "w4a4":
        for T in [int(a) for a in sys.argv[2:]] or [1024]:
            run_w4a4(T)
    elif mode == "sweep":
        for T in (1, 4, 16, 64, 256, 1024):
            run_w4a16(T)
        for T in (256, 1024, 4096):
            run_w4a4(T)


def worker():
    # usage: worker <w4a16|w4a4> <T> [name-subfilter]
    global WEIGHTS
    WEIGHTS = make_weights()
    arm, T = sys.argv[2], int(sys.argv[3])
    only = sys.argv[4] if len(sys.argv) > 4 else None
    (run_w4a16 if arm == "w4a16" else run_w4a4)(T, only)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker()
    else:
        main()
