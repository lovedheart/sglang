"""A3 fuse probe: fused down kernel (inline act quant) vs old two-launch path.

The OLD path is reproduced with a copy of the pre-fusion kernel source
(a8/asf loaded from global), so the comparison is against the real previous
behaviour even after hc_mix_fp8.py has been edited.
"""

import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/home/lovedheart/sglang-fork-dev/test/registered/kernels/ops/gemm")
from test_hc_mix_triton import HC_COUNT, HIDDEN_SIZE, _make_w8a8_weights  # noqa: E402

from sglang.kernels.ops.gemm.hc_mix_fp8 import (  # noqa: E402
    _hc_mix_a8w8_up_kernel,
    _mxfp8_quant_act_kernel,
    _mxfp8_quant_groups,
    fused_hc_mix_fp8_w8a8,
)


@triton.jit
def _old_down_kernel(
    a_ptr,
    asf_ptr,
    w_ptr,
    wsf_ptr,
    t_ptr,
    tsf_ptr,
    T,
    K: tl.constexpr,
    LOWRANK,
    inv_hc,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NKB: tl.constexpr,
):
    # Verbatim copy of _hc_mix_a8w8_down_kernel before A3 fusion.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < T
    mask_n = offs_n < LOWRANK
    offs_k = tl.arange(0, BK)
    offs_g = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(
            a_ptr + offs_m[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        asf = tl.load(
            asf_ptr + offs_m[:, None] * NKB + (k0 // 32) + offs_g[None, :],
            mask=mask_m[:, None],
            other=127,
        )
        b = tl.load(
            w_ptr + offs_n[None, :].to(tl.int64) * K + (k0 + offs_k)[:, None],
            mask=mask_n[None, :],
            other=0.0,
        )
        bsf = tl.load(
            wsf_ptr + offs_n[:, None] * NKB + (k0 // 32) + offs_g[None, :],
            mask=mask_n[:, None],
            other=127,
        )
        acc = tl.dot_scaled(a, asf, "e4m3", b, bsf, "e4m3", acc)
    tn = acc * inv_hc
    t = tn * tl.sigmoid(tn)
    tg = tl.reshape(t, (BM, BN // 32, 32))
    e = _mxfp8_quant_groups(tg, tl.max(tl.abs(tg), axis=2))
    q = (tg / tl.exp2(e)[:, :, None]).to(tl.float8e4nv)
    tl.store(
        t_ptr + offs_m[:, None].to(tl.int64) * LOWRANK + offs_n[None, :],
        tl.reshape(q, (BM, BN)),
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(
        tsf_ptr
        + offs_m[:, None] * (LOWRANK // 32)
        + pid_n * (BN // 32)
        + tl.arange(0, BN // 32)[None, :],
        (e + 127).to(tl.uint8),
        mask=mask_m[:, None],
    )


def old_path(x, d8, ds, u8, us, hc, hs):
    rows, k = x.shape
    lowrank = d8.shape[0]
    dev = x.device
    out = torch.empty((rows, hs), dtype=x.dtype, device=dev)
    a8 = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=dev)
    asf = torch.empty((rows, k // 32), dtype=torch.uint8, device=dev)
    t8 = torch.empty((rows, lowrank), dtype=torch.float8_e4m3fn, device=dev)
    tsf = torch.empty((rows, lowrank // 32), dtype=torch.uint8, device=dev)
    _mxfp8_quant_act_kernel[(rows, triton.cdiv(k, 1024))](
        x, a8, asf, rows, K=k, BLOCK=1024, num_warps=4
    )
    BM, BN, BK = 128, 64, 256
    _old_down_kernel[(triton.cdiv(rows, BM), triton.cdiv(lowrank, BN))](
        a8,
        asf,
        d8,
        ds,
        t8,
        tsf,
        rows,
        k,
        lowrank,
        1.0 / hc,
        BM=BM,
        BN=BN,
        BK=BK,
        NKB=k // 32,
        num_warps=8,
        num_stages=2,
    )
    # up kernel is launched (j_tiles, m_tiles) since d537d4f289
    _hc_mix_a8w8_up_kernel[(hs // 64, triton.cdiv(rows, 64))](
        t8,
        tsf,
        u8,
        us,
        x,
        out,
        rows,
        LOWRANK=lowrank,
        HS=hs,
        HC=hc,
        BM=64,
        BN=64,
        BK=64,
        num_warps=4,
        num_stages=3,
    )
    return out


def main():
    from sglang.srt.layers.fp8_hc_mix import mxfp8_quantize_rowmajor

    # Fixed weights per seed (same quantized pair for old and new).
    weight_cache = {}
    for seed in (0, 1):
        torch.manual_seed(seed)
        wd = torch.randn(320, 10240, dtype=torch.bfloat16, device="cuda") * 0.02
        wu = torch.randn(10240, 320, dtype=torch.bfloat16, device="cuda") * 0.02
        weight_cache[seed] = (
            mxfp8_quantize_rowmajor(wd),
            mxfp8_quantize_rowmajor(wu),
        )
        del wd, wu

    failures = []
    for rows in (512, 1024, 4032, 4090):
        for seed in (0, 1):
            torch.manual_seed(seed)
            x = torch.randn(
                rows, HC_COUNT * HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
            )
            (d8, ds), (u8, us) = weight_cache[seed]
            o_old = old_path(x, d8, ds, u8, us, HC_COUNT, HIDDEN_SIZE)
            o_new = fused_hc_mix_fp8_w8a8(x, d8, ds, u8, us, HC_COUNT, HIDDEN_SIZE)
            eq = torch.equal(o_old, o_new)
            print(f"rows={rows:5d} seed={seed} torch.equal={eq}")
            if not eq:
                bad = o_old != o_new
                nm = int(bad.sum())
                pos = bad.nonzero()[:5].tolist()
                for r, c in pos:
                    print(
                        f"  mismatch [{r},{c}]: old={o_old[r, c].item()} new={o_new[r, c].item()}"
                    )
                failures.append((rows, seed, nm))
            del x, o_old, o_new
            torch.cuda.empty_cache()

    if failures:
        print("FAIL", failures)
        sys.exit(1)
    print("ALL BIT-EXACT")


if __name__ == "__main__":
    main()
