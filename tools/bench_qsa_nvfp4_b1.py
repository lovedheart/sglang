#!/usr/bin/env python3
"""ABBA CUDA-event benchmark for the nvfp4 sparse-GQA chunk-prefill kernel.

Compares one or more *variants* of the kernel at a chosen shape. Variants are
supplied as modules exposing `run(...)` with the same signature as
`sparse_gqa_fwd_interface_triton_ck_nvfp4`, or as (name, callable) pairs.

Usage:
  python tools/bench_qsa_nvfp4_b1.py                 # baseline vs baseline (noise floor)
  Shapes: --nseq 4 --ctx 4096 --chunk 1024 --topk 2048 --nq 24 --nkv 2 --hd 256
"""

import argparse
import statistics
import sys

import torch

sys.path.insert(0, "/home/lovedheart/sglang-fork-dev/python")


def build_inputs(nseq, ctx, chunk, topk, nq, nkv, hd, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    total_q = nseq * chunk
    q = (
        torch.randn(
            total_q, nq, hd, device="cuda", generator=g, dtype=torch.float32
        ).to(torch.bfloat16)
        * 0.1
    )
    kp = torch.randint(
        0, 255, (nseq * ctx, nkv, hd // 2), device="cuda", dtype=torch.uint8
    )
    ksf = torch.randint(
        100, 130, (nseq * ctx, nkv, hd // 16), device="cuda", dtype=torch.uint8
    ).view(torch.float8_e4m3fn)
    vp = torch.randint(
        0, 255, (nseq * ctx, nkv, hd // 2), device="cuda", dtype=torch.uint8
    )
    vsf = torch.randint(
        100, 130, (nseq * ctx, nkv, hd // 16), device="cuda", dtype=torch.uint8
    ).view(torch.float8_e4m3fn)
    indices = torch.randint(
        0, ctx, (total_q, nkv, topk), device="cuda", dtype=torch.int32
    )
    # sprinkle padding
    mask = torch.rand(total_q, nkv, topk, device="cuda", generator=g) < 0.05
    indices[mask] = -1
    cu_q = torch.arange(0, nseq + 1, device="cuda", dtype=torch.int32) * chunk
    cu_k = torch.arange(0, nseq + 1, device="cuda", dtype=torch.int32) * ctx
    kv_lens = torch.full((nseq,), ctx, device="cuda", dtype=torch.int32)
    return dict(
        q=q,
        k_pk=kp,
        k_sf=ksf,
        v_pk=vp,
        v_sf=vsf,
        k_gs=1.0,
        v_gs=1.0,
        indices=indices,
        cu_q=cu_q,
        cu_k=cu_k,
        kv_lens=kv_lens,
        scale=1.0 / 16.0,
        max_q=chunk,
    )


def time_one(fn, inp, iters):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(**inp)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]  # us


def abba(fns, inp, iters=50, reps=5):
    """fns: list of (name, fn). Returns name -> median us."""
    results = {name: [] for name, _ in fns}
    # warmup
    for name, fn in fns:
        for _ in range(3):
            fn(**inp)
    torch.cuda.synchronize()
    for r in range(reps):
        seq = fns if r % 2 == 0 else list(reversed(fns))
        for name, fn in seq:
            times = time_one(fn, inp, iters)
            results[name].append(statistics.median(times))
    return {name: statistics.median(v) for name, v in results.items()}, {
        name: (min(v), max(v)) for name, v in results.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseq", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--nq", type=int, default=24)
    ap.add_argument("--nkv", type=int, default=2)
    ap.add_argument("--hd", type=int, default=256)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--check", action="store_true", help="also compare outputs between variants"
    )
    args = ap.parse_args()

    from sglang.srt.layers.attention.qsa import sparse_attn as sa

    inp = build_inputs(
        args.nseq, args.ctx, args.chunk, args.topk, args.nq, args.nkv, args.hd
    )

    variants = [("baseline", sa.sparse_gqa_fwd_interface_triton_ck_nvfp4)]
    # extra variants can be registered by importing a module given in QSA_VARIANTS
    extra = __import__("os").environ.get("QSA_VARIANTS", "")
    if extra:
        mod = __import__(extra)
        variants += mod.VARIANTS

    med, spread = abba(variants, inp, iters=args.iters, reps=args.reps)
    base = med["baseline"]
    print(
        f"shape rows={args.nseq * args.chunk} ctx={args.ctx} topk={args.topk} "
        f"nq={args.nq} nkv={args.nkv} hd={args.hd}"
    )
    for name, _ in variants:
        win = (base - med[name]) / base * 100.0
        print(
            f"  {name:32s} {med[name]:9.1f} us   spread={spread[name][0]:.0f}-{spread[name][1]:.0f}  win={win:+.1f}%"
        )

    if args.check and len(variants) > 1:
        ref = variants[0][1](**inp).float()
        for name, fn in variants[1:]:
            got = fn(**inp).float()
            d = (got - ref).abs()
            rel = d / ref.abs().clamp_min(1e-6)
            print(
                f"  diff[{name}] max_abs={d.max().item():.3e} "
                f"mean_abs={d.mean().item():.3e} max_rel={rel[d > 1e-6].max().item() if (d > 1e-6).any() else 0:.3e}"
            )


if __name__ == "__main__":
    main()
