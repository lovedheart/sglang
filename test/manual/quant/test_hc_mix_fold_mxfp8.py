"""GPU check: MXFP8 fold kernel == dequantized-fold reference.

Run: python test/manual/quant/test_hc_mix_fold_mxfp8.py
"""

import sys

import torch

sys.path.insert(0, "/home/lovedheart/sglang-fork-dev/python")

from sglang.kernels.ops.gemm.hc_mix import fused_hc_mix_fold  # noqa: E402
from sglang.kernels.ops.gemm.hc_mix_fp8 import (  # noqa: E402
    fused_hc_mix_fold_fp8,
)
from sglang.srt.layers.fp8_hc_mix import (  # noqa: E402
    mxfp8_dequantize_to_bf16,
    mxfp8_quantize_rowmajor,
)

HC, HS, LOWRANK, EPS = 4, 2560, 320, 1e-6
DEV = "cuda"


def ref(x, wd, wu, wn, inv_hc):
    xg = x.float().unflatten(-1, (HC, HS))
    ss = xg.pow(2).mean(-1)
    n = xg * torch.rsqrt(ss + EPS).unsqueeze(-1) * wn.float().view(HC, HS)
    nf = n.flatten(-2)
    t = torch.nn.functional.silu(nf.float() @ (wd.float() * wn.float()).t() / 4)
    g = torch.sigmoid((t @ wu.float().t()) * inv_hc).unflatten(-1, (HC, HS))
    return (g * n).mean(1).to(x.dtype)


def main():
    torch.manual_seed(0)
    inv_hc = 1.0 / HC
    wn = (1.0 + 0.02 * torch.randn(HC * HS, device=DEV)).clamp(0.2, 4.0)
    wd = (0.02 * torch.randn(LOWRANK, HC * HS, device=DEV)).to(torch.bfloat16)
    wu = (0.05 * torch.randn(HC * HS, LOWRANK, device=DEV)).to(torch.bfloat16)
    wd_q, wd_s = mxfp8_quantize_rowmajor(wd)
    wu_q, wu_s = mxfp8_quantize_rowmajor(wu)
    wd_deq = mxfp8_dequantize_to_bf16(wd_q, wd_s)
    wu_deq = mxfp8_dequantize_to_bf16(wu_q, wu_s)

    for rows in (1, 4, 16):
        x = (torch.randn(rows, HC * HS, device=DEV) * 2).to(torch.bfloat16)
        reference = ref(x, wd_deq, wu_deq, wn, inv_hc)
        out_m, g_m = fused_hc_mix_fold_fp8(
            x, wd_q, wu_q, wn.to(torch.bfloat16), HC, HS, EPS, wd_s, wu_s
        )
        out_b, g_b = fused_hc_mix_fold(
            x, wd_deq, wu_deq, wn.to(torch.bfloat16), HC, HS, EPS
        )
        d = (out_m.float() - out_b.float()).abs()
        r = d.max().item() / out_b.float().abs().max().item()
        dm = (out_m.float() - reference.float()).abs().max().item()
        print(
            f"rows={rows:3d} kernel-vs-kernel maxdiff={d.max().item():.3e} "
            f"rel={r:.3e} | mxfp8-vs-ref {dm:.3e} | g-diff "
            f"{(g_m - g_b).abs().max().item():.3e}"
        )
        assert d.max().item() < 2e-2, "fold mxfp8 kernel disagrees with bf16 fold"


if __name__ == "__main__":
    main()
    print("OK")
