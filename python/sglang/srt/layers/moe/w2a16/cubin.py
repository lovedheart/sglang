"""SM12x hand-written SASS cubin MoE GEMM (vLLM-Moet `moe_w2_mm`) as the
w2a16_offload prefill arm.

Reads the SAME TS4 planes the CPU engine reads, repacked once at load
(`pack_planes_for_cubin`) into the cubin fragment-major layout, padded to
the nearest cubin K (1536 -> 2048 zero-padded columns; the padding scales
are UE8M0 byte 0 ~= 0 and activation columns are zeros).

Kernel ABI (vLLM-Moet kernels/gen/gen_moe_w2.py, verified on this GPU):
    cuLaunchKernel(fn, N/16, pairs, 1,  NWARP(K)*32, 1, 1, 0, stream, argv)
    args: desc u64*, ldb=K, n_k64=K/64, ldc=N*2, n_kb128=K/128
    desc[pair][6] = [a(fp8 e4m3 rows), as(f32 [rows,K/32]), codes, scales,
                     c(bf16 rows), m_rows]   m=0 -> CTA exits (rows unwritten)
"""

from __future__ import annotations

import ctypes
import os

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_CUBIN_DIR_DEFAULT = "/home/lovedheart/vLLM-Moet/kernels/cubins-sm120"

# K -> NWARP (gen_moe_w2.py _nwarp): largest n<=min(cap, K/128) dividing K/128.
def _nwarp(k: int) -> int:
    nb = k // 128
    cap = 8 if k >= 1024 else 4
    for n in range(min(cap, nb), 0, -1):
        if nb % n == 0:
            return n
    return 1


_CU = None


def _drv():
    global _CU
    if _CU is None:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuInit.argtypes = [ctypes.c_uint]
        cu.cuDevicePrimaryCtxRetain = getattr(
            cu, "cuDevicePrimaryCtxRetain", None)
        cu.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_void_p, ctypes.c_char_p]
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p]
        _CU = cu
    return _CU


_FNS: dict[tuple[str, int], ctypes.c_void_p] = {}


def _ck(r, what):
    if isinstance(r, tuple):
        r = r[0]
    if r:
        raise RuntimeError(f"w2a16 cubin: CUDA error {r} in {what}")


def load_fn(variant: str, k: int):
    """variant in {'mc4', 'mc4afrag'}; caches CUfunction per (variant, K)."""
    key = (variant, k)
    if key in _FNS:
        return _FNS[key]
    d = os.environ.get("SGLANG_W2A16_MOET_DIR", _CUBIN_DIR_DEFAULT)
    name = f"moe_w2_mm_{'mc4afrag' if variant == 'mc4afrag' else 'mc4'}_k{k}_a32.cubin"
    path = os.path.join(d, name)
    if not os.path.exists(path):
        raise RuntimeError(f"w2a16 cubin: missing {path}")
    cu = _drv()
    mod = ctypes.c_void_p()
    _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()), "cuModuleLoad")
    fn = ctypes.c_void_p()
    _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"moe_w2_mm"),
        "cuModuleGetFunction")
    _FNS[key] = fn
    return fn


def cubin_k_for(k: int, table) -> int:
    """Smallest cubin K >= k in the given K table, or -1."""
    cands = [kx for kx in table if kx >= k and (kx * 4) % 128 == 0]
    return min(cands) if cands else -1


_MC4_KS = [384, 512, 768, 1024, 2048, 3584, 4096, 6144, 7168]


def shapes_supported(H: int, I: int) -> bool:
    return (cubin_k_for(H, _MC4_KS) > 0 and 2 * I % 16 == 0
            and cubin_k_for(I, _MC4_KS) > 0 and H % 16 == 0)


# ---------------------------------------------------------------- packing

def _codes_to_unit8(plane_u8: torch.Tensor, k: int) -> torch.Tensor:
    """[N, k/4] 2-bit-packed -> [N, k] one-code-per-byte (code at bits 2*(k%4))."""
    b = plane_u8.to(torch.int32)
    out = torch.stack([(b >> (2 * j)) & 3 for j in range(4)], dim=-1)
    return out.reshape(*b.shape[:-1], k).to(torch.uint8)


def _pack_frag_major(codes1: torch.Tensor) -> torch.Tensor:
    n, k = codes1.shape
    c = codes1.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 4).to(torch.int32)
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)
            ).to(torch.uint8)


def _pack_scales(s: torch.Tensor) -> torch.Tensor:
    n, ks = s.shape
    return s.view(n // 16, 16, ks).transpose(1, 2).contiguous().flatten()


def pack_planes_for_cubin(w13p, s13, w2p, s2, E, H, I, k13, k2, device):
    """Row-major TS4 planes ([E,N,K/4] u8 codes + [E,N,K/32] scales) ->
    cubin fragment-major, K padded (pad codes 0 -> level -4 but pad scale
    byte 0 -> 2**-127 ~= 0, and activations are zero there)."""
    def one(plane, sc, n, k, kp):
        out_p, out_s = [], []
        for e0 in range(0, E, 8):
            e1 = min(e0 + 8, E)
            u = _codes_to_unit8(plane[e0:e1].to(device), k)      # [8,n,k]
            if kp != k:
                pad = torch.zeros(*u.shape[:-1], kp - k, dtype=torch.uint8,
                                  device=device)
                u = torch.cat([u, pad], dim=-1)
                sp = torch.cat([sc[e0:e1].to(device),
                                torch.zeros(e1 - e0, n,
                                            kp // 32 - sc.shape[-1],
                                            dtype=torch.uint8, device=device)],
                               dim=-1)
            else:
                sp = sc[e0:e1].to(device)
            for i in range(e1 - e0):
                out_p.append(_pack_frag_major(u[i]))
                out_s.append(_pack_scales(sp[i]))
            del u
        return torch.cat(out_p), torch.cat(out_s)

    p13, s13b = one(w13p, s13, 2 * I, H, k13)     # rows [gate;up], N13=2I
    p2, s2b = one(w2p, s2, H, I, k2)
    return p13, s13b, p2, s2b


# ------------------------------------------------------------- kernels

@triton.jit
def _desc_kernel(eids_ptr, npad_ptr, d_ptr,
                 a1, as1, c13, a2, as2, c2,
                 p13, sc13, p2, sc2,
                 nb13, ns13, nb2, ns2,   # per-expert byte strides
                 rb13, rs13, rb2, rs2,  # pair byte strides into act/as buffers
                 rc13, rc2,             # pair byte strides into c buffers
                 pairs, cap6, E,
                 BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = tl.where(p * 16 < tl.load(npad_ptr), 16, 0).to(tl.int64)
    ok = p < pairs
    e = tl.load(eids_ptr + p, mask=ok, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), E - 1)
    for gi in tl.static_range(2):
        if gi == 0:
            a, s_, b, bs, c = a1 + p * rb13, as1 + p * rs13, p13 + e * nb13, sc13 + e * ns13, c13 + p * rc13
        else:
            a, s_, b, bs, c = a2 + p * rb2, as2 + p * rs2, p2 + e * nb2, sc2 + e * ns2, c2 + p * rc2
        d = d_ptr + gi * cap6 + p * 6
        tl.store(d + 0, a, mask=ok)
        tl.store(d + 1, s_, mask=ok)
        tl.store(d + 2, b, mask=ok)
        tl.store(d + 3, bs, mask=ok)
        tl.store(d + 4, c, mask=ok)
        tl.store(d + 5, m, mask=ok)


def _launch(fn, k, n, desc, pairs, stream):
    args = [ctypes.c_uint64(desc.data_ptr()),
            ctypes.c_uint32(k), ctypes.c_uint32(k // 64),
            ctypes.c_uint32(n * 2), ctypes.c_uint32(k // 128)]
    argv = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args])
    _ck(_drv().cuLaunchKernel(fn, n // 16, pairs, 1, _nwarp(k) * 32, 1, 1,
                              0, stream, argv, None), "launch")


def _quant_a32(x_bf16, kp):
    """[R,K] bf16 -> (fp8 e4m3 bytes [R,kp], f32 [R,kp/32]) group-32 amax/448."""
    r, k = x_bf16.shape
    g = x_bf16.float().view(r, k // 32, 32)
    s = (g.abs().amax(-1) / 448.0).clamp_min(1e-30)
    q = (g / s.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    qb = q.view(torch.uint8).view(r, k)
    if kp != k:
        qb = torch.cat([qb, torch.zeros(r, kp - k, dtype=torch.uint8,
                                        device=x_bf16.device)], dim=1)
        s = torch.cat([s, torch.zeros(r, kp // 32 - s.shape[1],
                                      dtype=torch.float32,
                                      device=x_bf16.device)], dim=1)
    return qb.view(r, kp), s


def cubin_forward(x, topk_ids, topk_weights, planes, E, H, I, k13, k2,
                  silu_clamp=20.0):
    """x [T,H] bf16; planes = (p13, sc13, p2, sc2) packed cubin planes.
    Returns [T,H] fp32 (caller casts)."""
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size)

    dev = x.device
    t = x.shape[0]
    ids = topk_ids.to(torch.int32)
    numel = ids.numel()
    sorted_ids, expert_ids, npad = moe_align_block_size(ids, 16, E)
    slots = sorted_ids.numel()
    pairs = slots // 16

    fn13 = load_fn("mc4", k13)
    fn2 = load_fn("mc4", k2)

    xq, xs = _quant_a32(x, k13)                    # [T,k13] u8, [T,k13/32] f32
    xq = torch.cat([xq, torch.zeros(1, k13, dtype=torch.uint8, device=dev)])
    xs = torch.cat([xs, torch.zeros(1, k13 // 32, dtype=torch.float32,
                                    device=dev)])
    rows = (sorted_ids.long().clamp_(min=0, max=numel)
            // topk_ids.shape[1]).clamp_(max=t)
    a1 = xq[rows]                                  # [slots,k13]
    as1 = xs[rows].contiguous()                    # [slots,k13/32]
    c13 = torch.empty(slots, 2 * I, dtype=torch.bfloat16, device=dev)
    desc = torch.zeros(2, pairs, 6, dtype=torch.int64, device=dev)
    p13, sc13, p2, sc2 = planes
    _desc_kernel[(triton.cdiv(pairs, 256),)](
        expert_ids, npad, desc,
        a1.data_ptr(), as1.data_ptr(), c13.data_ptr(), 0, 0, 0,
        p13.data_ptr(), sc13.data_ptr(), 0, 0,
        2 * I * k13 // 4, 2 * I * k13 // 32, 0, 0,
        16 * k13, 16 * (k13 // 32) * 4, 0, 0, 16 * 4 * I, 0,
        pairs, pairs * 6, E, BLOCK=256)
    stream = torch.cuda.current_stream().cuda_stream
    _launch(fn13, k13, 2 * I, desc[0], pairs, stream)
    g = c13[:, :I].float().clamp(max=silu_clamp)
    act = (g * torch.sigmoid(g) * c13[:, I:].float().clamp(
        -silu_clamp, silu_clamp)).to(torch.bfloat16)
    a2, as2 = _quant_a32(act, k2)
    c2 = torch.empty(slots, H, dtype=torch.bfloat16, device=dev)
    _desc_kernel[(triton.cdiv(pairs, 256),)](
        expert_ids, npad, desc,
        0, 0, 0, a2.data_ptr(), as2.data_ptr(), c2.data_ptr(),
        0, 0, p2.data_ptr(), sc2.data_ptr(),
        0, 0, H * k2 // 4, H * k2 // 32,
        0, 0, 16 * k2, 16 * (k2 // 32) * 4, 0, 16 * 2 * H,
        pairs, pairs * 6, E, BLOCK=256)
    _launch(fn2, k2, H, desc[1], pairs, stream)

    # weighted unpermute (deterministic: fixed-order sum over topk)
    sids = sorted_ids[:slots]
    valid = (sids >= 0) & (sids < numel)
    w = topk_weights.reshape(-1)[sids[valid].long()].float()
    c2v = c2[:slots][valid].float() * w.unsqueeze(1)
    out = torch.zeros(numel, H, dtype=torch.float32, device=dev)
    dst = sids[valid].long()
    out.index_copy_(0, dst, c2v)
    return out.view(t, topk_ids.shape[1], H).sum(1)
