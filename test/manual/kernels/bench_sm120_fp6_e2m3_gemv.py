# RUN_MANUAL=1 python test/manual/kernels/bench_sm120_fp6_e2m3_gemv.py
#
# Performance + accuracy bench for the fp6-e2m3 b32 lm-head band against an
# fp32 reference matmul, on synthetic and real lm_head weights.
# Reference: w_fp32 = dequant(codes); y_ref = x.float() @ w_fp32.T.
import statistics
import sys

import torch

sys.path.insert(0, "python")
from sglang.kernels.ops.gemm.sm120_fp6_e2m3 import (
    dequantize_e2m3_b32,
    quantize_e2m3_b32,
    sm120_fp6_e2m3_gemm,
)


def timeit(fn, iters=50):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) * 1000  # us


def lm_head_weight():
    import json
    import os

    ck = os.environ.get(
        "CKPT",
        "/media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnE3M4b128-mtpV4",
    )
    if not os.path.exists(ck):
        return None
    from safetensors import safe_open

    with open(ck + "/model.safetensors.index.json") as fh:
        ix = json.load(fh)["weight_map"]
    with safe_open(ck + "/" + ix["lm_head.weight"], "pt") as f:
        return f.get_tensor("lm_head.weight").cuda()


def bench(m, n, k, w=None, label=""):
    dev = "cuda"
    if w is None:
        torch.manual_seed(0)
        w = torch.randn(n, k, device=dev) * 0.02
    # chunked over N so full 248320-row lm_head weights fit next to a live server
    pk, sc, cd = [], [], []
    for n0 in range(0, n, 8192):
        rows = min(8192, n - n0)
        p_, s_, c_ = quantize_e2m3_b32(w[n0 : n0 + rows])
        pk.append(p_)
        sc.append(s_)
        cd.append(c_)
    packed = torch.cat(pk)
    scale = torch.cat(sc)
    codes = torch.cat(cd)
    del pk, sc, cd
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)

    # fp32 reference (chunked over N to bound memory)
    ref = torch.empty(m, n, device=dev, dtype=torch.float32)
    for n0 in range(0, n, 16384):
        rows = min(16384, n - n0)
        wq = dequantize_e2m3_b32(
            codes[n0 : n0 + rows], scale[n0 // 32 : (n0 + rows) // 32]
        )
        ref[:, n0 : n0 + rows] = x.float() @ wq.float().t()
    out = sm120_fp6_e2m3_gemm(x, packed, scale).float()
    rel = ((out - ref).norm() / ref.norm()).item()

    t_fp6 = timeit(lambda: sm120_fp6_e2m3_gemm(x, packed, scale))
    wb = w.to(torch.bfloat16)
    t_bf16 = timeit(lambda: torch.matmul(x, wb.mT))

    print(
        f"  M={m:<2} N={n:<7} K={k:<5} {label:<10} "
        f"fp6 {t_fp6:8.1f}us  bf16 {t_bf16:8.1f}us  speedup {t_bf16 / t_fp6:4.2f}x  "
        f"fp6 bytes {n * k * 6 / 8 / 1e6:6.1f}MB  eff-BW {n * k * 6 / 8 / (t_fp6 * 1e-6) / 1e12:5.2f} TB/s  "
        f"rel-L2 vs fp32-ref {rel:.2e}"
    )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.cuda.get_device_capability() == (12, 0), "SM120 only"
    print("== accuracy/perf vs fp32 reference ==")
    for m, n, k in [(1, 2560, 2560), (4, 2560, 2560), (1, 8192, 2560), (8, 8192, 2560)]:
        bench(m, n, k, label="synthetic")
    w = lm_head_weight()
    if w is not None:
        bench(1, w.shape[0], w.shape[1], w=w, label="lm_head")
        bench(4, w.shape[0], w.shape[1], w=w, label="lm_head")
    else:
        print("  (no lm_head checkpoint found; set CKPT=...)")
