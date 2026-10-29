#!/usr/bin/env python3
"""Kernel-level decode benchmark: b12x (quant_mode="w4a16") vs sm12x_nvfp4_bf16.

Both runners consume the SAME NVFP4 (modelopt, block-16 e4m3) packed weights
for the Qwen3.8-Flash-Next geometry (E=512, H=2560, I=640, top-10), so the
comparison is kernel-vs-kernel at equal weight bytes:

  * b12x_w4a16   : flashinfer b12x_fused_moe(quant_mode="w4a16") — the CuTe-DSL
                   W4A16 arm of the same family as the MXFP4 b12x path
                   (SGLANG_MXFP4_SM120_KERNEL=b12x).
  * sm12x_w4a16  : sglang's vendored run_moe_w4a16 via the
                   sm12x_nvfp4_bf16 runner (bucket-padded, tiled launches).
  * b12x_mxfp4   : (--b12x-mxfp4) b12x quant_mode="mxfp4" W4A4 on random
                   MXFP4 weights — what SGLANG_MXFP4_SM120_KERNEL=b12x runs.
  * w2a16_cubin  : (--w2a16) feat/w2a16-offload cubin decode GEMM on TS4
                   planes; DIFFERENT quantization, context only.

Timing: CUDA-graph replay per token count, median over replays (matches the
decode regime: full_cuda_graph_backend replays these kernels)."""

import argparse
import statistics
import sys

import torch

sys.path.insert(0, "/home/lovedheart/sglang-fork-dev/python")

E, H, I, TOPK = 512, 2560, 640, 10
_A13_GS = _A2_GS = None  # set in main once weights exist


def _swizzle_sf(sf_lin):
    """[E, N, kg] -> [E, pad128(N), pad4(kg)] 128x4 swizzled storage.

    Storage order (e, m_tile, k_tile, inner_m(4), outer_m(32), inner_k(4)):
    row-major over the last two gives the 16-byte scale atom, matching the
    fork's swizzle_blockscale reshape (rows//128, 4, 32, cols//4, 4)."""
    e, n, kg = sf_lin.shape
    mt, kt = (n + 127) // 128, (kg + 3) // 4
    if mt * 128 != n or kt * 4 != kg:
        pad = sf_lin.new_zeros(e, mt * 128, kt * 4)
        pad[:, :n, :kg] = sf_lin
        sf_lin = pad
    tiles = sf_lin.view(e, mt, 4, 32, kt, 4).permute(0, 1, 4, 2, 3, 5)
    return tiles.contiguous()


def make_nvfp4_weights():
    """Same tensors for both NVFP4 arms: packed FP4 + swizzled e4m3 SF + gs.

    Quantized in slabs (row-aligned across experts, verified bit-equal to the
    per-expert call) with a throwaway global scale; the per-expert gs tensors
    ride alongside, matching how the runner consumes them."""
    from flashinfer import fp4_quantize

    ones = torch.ones((), device="cuda")
    slab = 8
    w13 = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device="cuda")
    w2 = torch.empty(E, H, I // 2, dtype=torch.uint8, device="cuda")
    s13 = torch.empty(E, 2 * I, H // 16, dtype=torch.uint8, device="cuda")
    s2 = torch.empty(E, H, I // 16, dtype=torch.uint8, device="cuda")
    for st in range(0, E, slab):
        n_e = min(slab, E - st)
        g = torch.randn(n_e * 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05
        p, sf = fp4_quantize(g, ones, is_sf_swizzled_layout=True)
        w13[st:st + n_e] = p.view(torch.uint8).view(n_e, 2 * I, H // 2)
        s13[st:st + n_e] = sf.view(torch.uint8).view(n_e, 2 * I, H // 16)
        del g, p, sf
        g = torch.randn(n_e * H, I, device="cuda", dtype=torch.bfloat16) * 0.05
        p, sf = fp4_quantize(g, ones, is_sf_swizzled_layout=True)
        w2[st:st + n_e] = p.view(torch.uint8).view(n_e, H, I // 2)
        s2[st:st + n_e] = sf.view(torch.uint8).view(n_e, H, I // 16)
        del g, p, sf
    gs13 = (10 ** torch.empty(E, device="cuda").uniform_(-3, -2)).float()
    gs2 = (10 ** torch.empty(E, device="cuda").uniform_(-3, -2)).float()
    return dict(
        w13=w13, gs13=gs13, w2=w2, gs2=gs2,
        w13_sf=_swizzle_sf(s13).view(E, -1, (H // 16 + 3) // 4 * 4).view(torch.float8_e4m3fn),
        w2_sf=_swizzle_sf(s2).view(E, -1, (I // 16 + 3) // 4 * 4).view(torch.float8_e4m3fn),
    )


def make_nvfp4_slabs():
    """NVFP4 packed tensors in b12x W4A4 layout: [up; gate] rows, 6D MMA SF.

    Same weight values the sm12x arms see (block-16 e4m3), just re-laid for
    b12x's nvfp4 quant_mode — an apples-to-apples W4A4-vs-W4A4 comparison."""
    from flashinfer import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    ones = torch.ones((), device="cuda")
    slab = 8
    w13 = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device="cuda")
    w2 = torch.empty(E, H, I // 2, dtype=torch.uint8, device="cuda")
    s13 = torch.empty(E, 2 * I, H // 16, dtype=torch.uint8, device="cuda")
    s2 = torch.empty(E, H, I // 16, dtype=torch.uint8, device="cuda")
    for st in range(0, E, slab):
        n_e = min(slab, E - st)
        g = torch.randn(n_e * 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05
        p, sf = fp4_quantize(g, ones, is_sf_swizzled_layout=True)
        w13[st:st + n_e] = p.view(torch.uint8).view(n_e, 2 * I, H // 2)
        s13[st:st + n_e] = sf.view(torch.uint8).view(n_e, 2 * I, H // 16)
        del g, p, sf
        g = torch.randn(n_e * H, I, device="cuda", dtype=torch.bfloat16) * 0.05
        p, sf = fp4_quantize(g, ones, is_sf_swizzled_layout=True)
        w2[st:st + n_e] = p.view(torch.uint8).view(n_e, H, I // 2)
        s2[st:st + n_e] = sf.view(torch.uint8).view(n_e, H, I // 16)
        del g, p, sf

    def _mma(sf_lin, n_rows, k):
        return torch.stack(
            [convert_sf_to_mma_layout(sf_lin[e], m=n_rows, k=k, num_groups=1,
                                      sf_vec_size=16)[..., 0] for e in range(E)], -1
        )

    out = dict(
        w13=w13, gs13=torch.ones(E, dtype=torch.float32, device="cuda"),
        w2=w2, gs2=torch.ones(E, dtype=torch.float32, device="cuda"),
        w13_sf=_mma(s13, 2 * I, H), w2_sf=_mma(s2, H, I),
    )
    del s13, s2
    return out


def make_mxfp4_weights():
    """MXFP4 (e2m1 + e8m0 g32) weights in b12x layout: [up; gate] rows,
    SF as per-expert 6D MMA views (what launch_sm120_moe consumes)."""
    from flashinfer import mxfp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    slab = 8
    w13 = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device="cuda")
    s13 = torch.empty(E, 2 * I, H // 32, dtype=torch.uint8, device="cuda")
    w2 = torch.empty(E, H, I // 2, dtype=torch.uint8, device="cuda")
    s2 = torch.empty(E, H, I // 32, dtype=torch.uint8, device="cuda")
    for st in range(0, E, slab):
        n_e = min(slab, E - st)
        gu = torch.randn(2 * n_e * I, H, device="cuda", dtype=torch.bfloat16) * 0.05
        pg, sg = mxfp4_quantize(gu[: n_e * I])   # gate half
        pu, su = mxfp4_quantize(gu[n_e * I:])    # up half
        # b12x wants [up; gate] row order
        w13[st:st + n_e] = torch.cat([pu, pg], 0).view(n_e, 2 * I, H // 2)
        s13[st:st + n_e] = torch.cat([su, sg], 0).view(n_e, 2 * I, H // 32)
        del gu, pg, sg, pu, su
        g = torch.randn(n_e * H, I, device="cuda", dtype=torch.bfloat16) * 0.05
        p, sf = mxfp4_quantize(g)
        w2[st:st + n_e], s2[st:st + n_e] = p.view(n_e, H, I // 2), sf.view(n_e, H, I // 32)
        del g, p, sf

    def _mma(sf_lin, n_rows, k):
        return torch.stack(
            [convert_sf_to_mma_layout(sf_lin[e], m=n_rows, k=k, num_groups=1,
                                      sf_vec_size=32)[..., 0] for e in range(E)], -1
        )

    out = dict(
        w13=w13, gs13=torch.ones(E, dtype=torch.float32, device="cuda"),
        w2=w2, gs2=torch.ones(E, dtype=torch.float32, device="cuda"),
        w13_sf=_mma(s13, 2 * I, H), w2_sf=_mma(s2, H, I),
        w13_sf_plain=s13, w2_sf_plain=s2,   # [E,N,K/32] for the fp4_e8m0_k32 arm
    )
    return out


# ------------------------------------------------------------------ arms
def make_b12x_arm(W, quant_mode="w4a16"):
    from flashinfer.fused_moe import b12x_fused_moe

    def call(x, ids, wts, out):
        return b12x_fused_moe(
            x, W["w13"], W["w13_sf"], W["w2"], W["w2_sf"], ids, wts, E, TOPK,
            w1_alpha=W["gs13"], w2_alpha=W["gs2"],
            output=out, quant_mode=quant_mode,
        )

    return call


def make_b12x_mxfp4_wrapper_arm(M, max_tokens, quant_mode="mxfp4"):
    from flashinfer.fused_moe.cute_dsl.b12x_moe import B12xMoEWrapper

    kw = {}
    sf = "modelopt"
    if quant_mode == "nvfp4":
        kw["fc2_input_scale"] = torch.ones(1, dtype=torch.float32, device="cuda")
    if quant_mode == "w4a16_e8m0":
        # MXFP4 weights, BF16 activations: b12x W4A16 arm with E8M0 K/32 scales
        quant_mode, sf = "w4a16", "fp4_e8m0_k32"
    w = B12xMoEWrapper(
        num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=I,
        use_cuda_graph=True, max_num_tokens=max_tokens, quant_mode=quant_mode,
        source_format=sf,
    )

    def call(x, ids, wts, out):
        r = w.run(
            x=x, w1_weight=M["w13"], w1_weight_sf=M["w13_sf"], w1_alpha=M["gs13"],
            w2_weight=M["w2"], w2_weight_sf=M["w2_sf"], w2_alpha=M["gs2"],
            token_selected_experts=ids, token_final_scales=wts, **kw,
        )
        out[: x.shape[0]].copy_(r)
        return out

    return call


def make_sm12x_arm(W):
    """sm12x_nvfp4_bf16 runner (vendored PR#5735 kernel, bucket ladder)."""
    from sglang.srt.layers.moe.moe_runner.sm12x_nvfp4_bf16 import (
        Sm12xNvfp4W4A16MoeQuantInfo,
        fused_experts_none_to_sm12x_nvfp4_bf16,
    )

    qi = Sm12xNvfp4W4A16MoeQuantInfo(
        w13=W["w13"], w13_sf=W["w13_sf"], w13_gs=W["gs13"],
        w2=W["w2"], w2_sf=W["w2_sf"], w2_gs=W["gs2"],
    )

    class TK:
        def __init__(s, i, w):
            s.topk_ids, s.topk_weights, s.router_logits = i, w, None

    class DO:
        def __init__(s, x, i, w):
            s.hidden_states, s.hidden_states_scale = x, None
            s.topk_output = TK(i, w)

    class CFG:
        num_fused_shared_experts = 0
        apply_router_weight_on_input = False

    def call(x, ids, wts, out):
        r = fused_experts_none_to_sm12x_nvfp4_bf16(DO(x, ids, wts), qi, CFG())
        out[: x.shape[0]].copy_(r.hidden_states)
        return out

    return call


def make_w2a16_arm():
    """w2a16_offload cubin decode path (different quantization; context only)."""
    import importlib.util
    import subprocess

    ref = "d61e47e898"
    for mod in ("ts4", "cubin"):
        p = f"/tmp/w2a16_{mod}.py"
        subprocess.run(
            ["git", "-C", "/home/lovedheart/sglang-fork-dev", "show",
             f"{ref}:python/sglang/srt/layers/moe/w2a16/{mod}.py"],
            stdout=open(p, "w"), check=True,
        )

    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"w2a16_{name}", f"/tmp/w2a16_{name}.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    ts4, cubin = load("ts4"), load("cubin")
    k13, k2 = cubin.cubin_k_for(H, cubin._MC4_KS), cubin.cubin_k_for(I, cubin._MC4_KS)
    if not cubin.shapes_supported(H, I):
        raise RuntimeError("w2a16 cubins do not cover this geometry")
    g13 = torch.randn(E, 2 * I, H, device="cuda") * 0.05
    p13, s13, _ = ts4.pack_ts4(g13)
    del g13
    g2 = torch.randn(E, H, I, device="cuda") * 0.05
    p2, s2, _ = ts4.pack_ts4(g2)
    planes = cubin.pack_planes_for_cubin(p13, s13, p2, s2, E, H, I, k13, k2, "cuda")
    del g2, p13, s13, p2, s2

    def call(x, ids, wts, out):
        return cubin.cubin_forward(x, ids, wts, planes, E, H, I, k13, k2)

    return call


def make_sm12x_w4a4_arm(W):
    """sm12x sibling W4A4 arm: flashinfer moe_nvfp4_w4a4.run_moe_w4a4 on the
    SAME NVFP4 weights (what SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS enables)."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4 import (
        allocate_workspace,
        run_moe_w4a4,
    )

    ws_cache = {}

    def call(x, ids, wts, out):
        T = x.shape[0]
        if T not in ws_cache:
            ws_cache[T] = allocate_workspace(
                T, H, I, E, TOPK, True, torch.device("cuda")
            )
        return run_moe_w4a4(
            x, ids, wts,
            W["w13"], W["w13_sf"], W["gs13"],
            W["w2"], W["w2_sf"], W["gs2"],
            _A13_GS, _A2_GS, out, workspace=ws_cache[T],
        )

    return call


# ------------------------------------------------------------------ timing
def bench_graph(fn, T, iters=50):
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device="cuda")
    wts = torch.softmax(torch.rand(T, TOPK, device="cuda"), dim=-1).float()
    out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")

    for _ in range(3):
        fn(x, ids, wts, out)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(x, ids, wts, out)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en) * 1000.0)  # us
    del g, x, ids, wts, out
    torch.cuda.empty_cache()
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,2,4,8,16,32")
    ap.add_argument("--w2a16", action="store_true",
                    help="also run the w2a16_offload cubin arm (context only)")
    ap.add_argument("--b12x-mxfp4", action="store_true",
                    help="also run b12x MXFP4 W4A4 (random mxfp4 weights)")
    ap.add_argument("--sm12x-w4a4", action="store_true",
                    help="also run sm12x sibling NVFP4 W4A4 prefill kernel")
    ap.add_argument("--wrapper-mxfp4", action="store_true",
                    help="b12x mxfp4 via B12xMoEWrapper (preallocated graph buffers)")
    ap.add_argument("--b12x-nvfp4-w4a4", action="store_true",
                    help="also run b12x NVFP4 W4A4 (same weights as sm12x arms)")
    ap.add_argument("--b12x-mxfp4-w4a16", action="store_true",
                    help="also run b12x W4A16 on MXFP4 (e8m0 k32) weights")
    args = ap.parse_args()
    Ts = [int(t) for t in args.tokens.split(",")]

    print(f"device={torch.cuda.get_device_name()} cap={torch.cuda.get_device_capability()}")
    print(f"geometry: E={E} H={H} I={I} topk={TOPK} | graph-replay median us\n")

    arms = {}
    W = make_nvfp4_weights()
    global _A13_GS, _A2_GS
    _A13_GS = torch.ones(1, dtype=torch.float32, device="cuda")
    _A2_GS = torch.ones(1, dtype=torch.float32, device="cuda")
    arms["b12x_w4a16"] = make_b12x_arm(W, "w4a16")
    arms["sm12x_w4a16"] = make_sm12x_arm(W)
    if args.sm12x_w4a4:
        try:
            arms["sm12x_w4a4"] = make_sm12x_w4a4_arm(W)
        except Exception as ex:
            print(f"[warn] sm12x w4a4 arm: {type(ex).__name__}: {ex}")
    if args.b12x_mxfp4:
        try:
            M = make_mxfp4_weights()
            if args.wrapper_mxfp4:
                arms["b12x_mxfp4"] = make_b12x_mxfp4_wrapper_arm(M, max(Ts))
                del M
            else:
                arms["b12x_mxfp4"] = make_b12x_arm(M, "mxfp4")
        except Exception as ex:
            print(f"[warn] b12x mxfp4 arm: {type(ex).__name__}: {ex}")
    if args.b12x_nvfp4_w4a4:
        try:
            N = make_nvfp4_slabs()
            if args.wrapper_mxfp4:
                arms["b12x_nvfp4_w4a4"] = make_b12x_mxfp4_wrapper_arm(
                    N, max(Ts), "nvfp4")
                del N
            else:
                arms["b12x_nvfp4_w4a4"] = make_b12x_arm(N, "nvfp4")
        except Exception as ex:
            print(f"[warn] b12x nvfp4 w4a4 arm: {type(ex).__name__}: {ex}")
    if args.b12x_mxfp4_w4a16:
        try:
            M2 = make_mxfp4_weights()
            M2["w13_sf"], M2["w2_sf"] = M2.pop("w13_sf_plain"), M2.pop("w2_sf_plain")
            arms["b12x_mxfp4_w4a16"] = make_b12x_mxfp4_wrapper_arm(
                M2, max(Ts), "w4a16_e8m0")
            del M2
        except Exception as ex:
            print(f"[warn] b12x mxfp4-w4a16 arm: {type(ex).__name__}: {ex}")
    if args.w2a16:
        try:
            arms["w2a16_cubin"] = make_w2a16_arm()
        except Exception as ex:
            print(f"[warn] w2a16 arm: {type(ex).__name__}: {ex}")

    results = {}
    for name, fn in arms.items():
        results[name] = {}
        for T in Ts:
            try:
                results[name][T] = bench_graph(fn, T)
            except Exception as ex:
                results[name][T] = None
                print(f"[warn] {name} T={T}: {type(ex).__name__}: {str(ex)[:120]}")

    hdr = "T      " + "".join(f"{n:>14}" for n in results)
    print(hdr)
    print("-" * len(hdr))
    for T in Ts:
        row = f"{T:<7}"
        for n in results:
            v = results[n].get(T)
            row += f"{v:>13.1f}u" if v else f"{'fail':>14}"
        print(row)
    base = "sm12x_w4a16" if "sm12x_w4a16" in results else None
    if base:
        for name in results:
            if name == base:
                continue
            print(f"\n{name} / {base} latency ratio (>1 = {name} slower):")
            for T in Ts:
                a, b = results[name].get(T), results[base].get(T)
                if a and b:
                    print(f"  T={T:<4} {a / b:.2f}x")


if __name__ == "__main__":
    main()
