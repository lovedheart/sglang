"""Split HC-mix vs HC-combine kernel time at decode sizes (no model needed).

Mirrors the shipped fold path: mix = fused_hc_mix_fold_fp8 (packed MXFP8),
combine = hc_combine_split_fold (rows<=32). Synthetic weights, real shapes
(hc=4, hs=2560, lowrank=320). Safe to run next to a live server: ~5 MB and
a few seconds of kernels.

Run: python test/manual/quant/bench_hc_mix_vs_combine.py
"""

import sys
import time

import torch

sys.path.insert(0, "/home/lovedheart/sglang-fork-dev/python")

from sglang.kernels.ops.elementwise.hc_combine import (  # noqa: E402
    hc_combine_split_fold,
)
from sglang.kernels.ops.gemm.hc_mix_fp8 import (  # noqa: E402
    fused_hc_mix_fold_fp8,
)
from sglang.srt.layers.fp8_hc_mix import mxfp8_quantize_rowmajor  # noqa: E402

HC, HS, LOWRANK, EPS = 4, 2560, 320, 1e-6
DEV = "cuda"


def timeit(fn, iters=2000, warmup=200):
    # CUDA-graph replay: matches serving (no CPU launch noise).
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup // 50):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters // 50):
        for _ in range(50):
            g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # µs


def main():
    torch.manual_seed(0)
    wn = (1.0 + 0.02 * torch.randn(HC * HS, device=DEV)).clamp(0.2, 4.0)
    wd = (0.02 * torch.randn(LOWRANK, HC * HS, device=DEV)).to(torch.bfloat16)
    wu = (0.05 * torch.randn(HC * HS, LOWRANK, device=DEV)).to(torch.bfloat16)
    wi = (0.5 * torch.randn(HC, HC * HS, device=DEV)).to(torch.bfloat16)
    wd_q, wd_s = mxfp8_quantize_rowmajor(wd)
    wu_q, wu_s = mxfp8_quantize_rowmajor(wu)
    wi_fold = (wi.float() * wn.float()).to(torch.bfloat16)
    print(f"{'rows':>5} {'mix us':>8} {'combine us':>11} {'comb %':>8}")
    for rows in (1, 2, 4, 8, 16):
        x = (torch.randn(rows, HC * HS, device=DEV)).to(torch.bfloat16)
        block = torch.randn(rows, HS, device=DEV).to(torch.bfloat16)
        out_m, g = fused_hc_mix_fold_fp8(
            x, wd_q, wu_q, wn.to(torch.bfloat16), HC, HS, EPS, wd_s, wu_s
        )
        t_mix = timeit(
            lambda: fused_hc_mix_fold_fp8(
                x, wd_q, wu_q, wn.to(torch.bfloat16), HC, HS, EPS, wd_s, wu_s
            )
        )
        t_comb = timeit(
            lambda: hc_combine_split_fold(
                block, x, g, wi_fold, HC, HS
            )
        )
        print(f"{rows:>5} {t_mix:8.2f} {t_comb:11.2f} {100*t_comb/(t_mix+t_comb):7.1f}%")


if __name__ == "__main__":
    main()
