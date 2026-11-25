# Fork note: this file mirrors gemm_dev/sm12x_mxfp4/mxfp4_w4a4_moe.py byte-
# identical except this header block (source of truth stays in gemm_dev).
# MXFP4 W4A4 grouped MoE on the native b12x MmaMXF4 dense_gemm (3D grouped
# contract: expert-block-major A/B, swizzled E8M0 scales, alpha=None).
import torch
import triton
import triton.language as tl


@triton.jit
def _mxfp4_quant_kernel(x_ptr, out_ptr, sf_ptr, K, KP, C, CP,
                        stride_m, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    kb = tl.program_id(1)
    offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)

    g = tl.reshape(x, (BLOCK_K // 32, 32))
    amax = tl.max(tl.abs(g), axis=1)                       # [BLOCK_K//32]
    e = tl.extra.libdevice.rint(tl.extra.libdevice.log2(tl.maximum(amax, 1e-30) / 6.0))
    e = tl.clamp(e, -126.0, 126.0)
    s = tl.extra.libdevice.exp2(e)
    byte = (e + 127.0).to(tl.uint8)

    q = g / s[:, None]                                     # scale-normalized
    aq = tl.abs(q)
    # RN onto ladder midpoints [0.25,0.75,1.25,1.75,2.5,3.5,5.0]
    mag = (aq > 0.25).to(tl.uint8) + (aq > 0.75).to(tl.uint8) + (aq > 1.25).to(tl.uint8) \
        + (aq > 1.75).to(tl.uint8) + (aq > 2.5).to(tl.uint8) + (aq > 3.5).to(tl.uint8) \
        + (aq > 5.0).to(tl.uint8)
    code = mag | ((q < 0).to(tl.uint8) << 3)               # [G, 32]
    code = tl.reshape(code, (BLOCK_K // 2, 2))
    lo, hi = tl.split(code)
    packed = lo | (hi << 4)                                # [KP-block] even-k low
    po = kb * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(out_ptr + row * KP + po, packed.to(tl.uint8), mask=po < KP)

    # SF swizzle write: global col-of-SF = kb*BLOCK_K/32 + arange
    c = kb * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)  # [G]
    rb, rr = row // 128, row % 128
    addr = (((rb * (CP // 4) + c // 4) * 32 + rr % 32) * 4 + rr // 32) * 4 + c % 4
    tl.store(sf_ptr + addr, byte, mask=c < C)


def quant_act_mxfp4(x: torch.Tensor):
    """x (M, K) bf16/fp16 -> (packed uint8 (M, K/2), sf uint8 (M_pad, C_pad))."""
    M, K = x.shape
    KP, C = K // 2, K // 32
    CP = (C + 3) // 4 * 4
    out = torch.empty((M, KP), dtype=torch.uint8, device=x.device)
    sf = torch.zeros(((M + 127) // 128 * 128, CP), dtype=torch.uint8, device=x.device)
    BLOCK_K = min(triton.next_power_of_2(K), 4096)
    _mxfp4_quant_kernel[(M, triton.cdiv(K, BLOCK_K))](
        x, out, sf, K, KP, C, CP, x.stride(0), BLOCK_K=BLOCK_K, num_warps=4)
    return out, sf
