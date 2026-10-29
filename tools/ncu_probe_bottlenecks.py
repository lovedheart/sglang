#!/usr/bin/env python3
"""Reproduce the two prefill bottleneck kernels at serving shapes for ncu.

Shapes mirror serve_1028_e3m4 (Qwen3.8-Flash-Next NVFP4, sm_120, bs4 x 4096
chunked prefill): 24 q heads / 2 kv heads -> GROUP_SIZE=12, head_dim=256,
topk(indexer_budget)=2048, HC hc_count=4 / hidden=2560 / lowrank=320.

Run under ncu with --kernel-name regex to isolate one family:
  ncu -k "regex:_sparse_gqa_chunk_prefill_nvfp4|_hc_mix_a8w8" ...
"""
import sys

import torch

sys.path.insert(0, "/home/lovedheart/sglang-fork-dev/python")

DEV = "cuda"
DT = torch.bfloat16


def bench_qsa_prefill():
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        sparse_gqa_fwd_interface_triton_ck_nvfp4,
    )

    # full-chunk prefill: 4 seqs x 1024 new q rows over 4096-token context
    nseq, ctx, chunk = 4, 4096, 1024
    total_q = nseq * chunk
    nq, nkv, hd, topk = 24, 2, 256, 2048
    q = torch.randn(total_q, nq, hd, device=DEV, dtype=DT) * 0.1
    # packed nvfp4 pool: half bytes + fp8-e4m3 block scales per 16 elems
    kp = torch.randint(0, 255, (nseq * ctx, nkv, hd // 2), device=DEV, dtype=torch.uint8)
    ksf = torch.randint(100, 130, (nseq * ctx, nkv, hd // 16), device=DEV, dtype=torch.uint8).view(torch.float8_e4m3fn)
    vp = torch.randint(0, 255, (nseq * ctx, nkv, hd // 2), device=DEV, dtype=torch.uint8)
    vsf = torch.randint(100, 130, (nseq * ctx, nkv, hd // 16), device=DEV, dtype=torch.uint8).view(torch.float8_e4m3fn)
    k_gs = 1.0  # get_bmm_scales returns python floats, not tensors
    v_gs = 1.0
    indices = torch.randint(0, ctx, (total_q, nkv, topk), device=DEV, dtype=torch.int32)
    cu_q = torch.arange(0, nseq + 1, device=DEV, dtype=torch.int32) * chunk
    cu_k = torch.arange(0, nseq + 1, device=DEV, dtype=torch.int32) * ctx
    kv_lens = torch.full((nseq,), ctx, device=DEV, dtype=torch.int32)
    for _ in range(3):  # warmup
        sparse_gqa_fwd_interface_triton_ck_nvfp4(
            q, kp, ksf, vp, vsf, k_gs, v_gs, indices, cu_q, cu_k, kv_lens, 1.0 / 16.0, max_q=chunk
        )
    torch.cuda.synchronize()
    for _ in range(5):  # profiled
        sparse_gqa_fwd_interface_triton_ck_nvfp4(
            q, kp, ksf, vp, vsf, k_gs, v_gs, indices, cu_q, cu_k, kv_lens, 1.0 / 16.0, max_q=chunk
        )
    torch.cuda.synchronize()
    print("qsa prefill done")


def bench_hc_mix():
    from sglang.kernels.ops.gemm.hc_mix_fp8 import fused_hc_mix_fp8_w8a8

    rows, hc, hs, lr = 4032, 4, 2560, 320
    k = hc * hs
    x = torch.randn(rows, k, device=DEV, dtype=DT) * 0.05
    w_down = torch.randint(0, 255, (lr, k), device=DEV, dtype=torch.uint8).view(torch.float8_e4m3fn)
    w_down_scale = torch.randint(100, 126, (lr, k // 32), device=DEV, dtype=torch.uint8)
    w_up = torch.randint(0, 255, (hc * hs, lr), device=DEV, dtype=torch.uint8).view(torch.float8_e4m3fn)
    w_up_scale = torch.randint(100, 126, (hc * hs, lr // 32), device=DEV, dtype=torch.uint8)
    for _ in range(3):
        fused_hc_mix_fp8_w8a8(x, w_down, w_down_scale, w_up, w_up_scale, hc, hs)
    torch.cuda.synchronize()
    for _ in range(5):
        fused_hc_mix_fp8_w8a8(x, w_down, w_down_scale, w_up, w_up_scale, hc, hs)
    torch.cuda.synchronize()
    print("hc mix done")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("both", "qsa"):
        bench_qsa_prefill()
    if what in ("both", "hc"):
        bench_hc_mix()
