"""ig_moe TS4 — 2-bit tensor-sym ({-4,-1,+1,+4}) MoE expert weights.

Format (portable, row-major; "ts4" = tensor-sym K=4 codebook):
    plane   u8 [..., K/4]   4 codes per byte along K, code(k) in bits
            2*(k%4)..+1 (little-endian pairs)
    scales  u8 [..., K/32]  UE8M0 bytes, value = 2^(byte-127), one per
            32 consecutive k (checkpoint block-32 convention)
    value   w[n,k] = LEVELS[code] * 2^(scale-127) * gs
    LEVELS = {-4, -1, +1, +4}; code order {0:-4, 1:-1, 2:+1, 3:+4}

Semantics are bit-faithful to vLLM-Moet's ``moe_w2`` 2-bit line (quantizer
ABI tag ``tsym4-e8m0-fragmajor-a32-v2``); the fragment-major CUDA/QMMA
interop packers here reproduce its planes byte-for-byte
(``pack_fragment_major`` / ``pack_scales_cuda`` / quintal planes), while
the ig_moe serving path (Vulkan shader + torch fallback) consumes the
row-major plane directly — no swizzle needed off CUDA.

The quantizer (``quantize_ts4``) is a verbatim port of the validated
vLLM-Moet pipeline: per-block-32 UE8M0 scale -> e2m1 snap with midpoint
tie-break -> nibble->code map, incl. the zero-sign balancing policy
(auto/sign/alt).
"""

from __future__ import annotations

import torch

__all__ = [
    "LEVELS", "NIBBLE_TO_CODE", "CODE_TO_NIBBLE",
    "quantize_ts4", "pack_ts4", "unpack_ts4", "ts4_dequant",
    "codes_to_plane", "plane_to_codes",
    "ts4_reference_forward", "quantize_act_i8a", "ts4_reference_forward_i8a",
    "ts4_reference_forward_a16",
    # CUDA (vLLM-Moet moe_w2) interop
    "pack_fragment_major", "unpack_fragment_major", "pack_scales_cuda",
    "unpack_scales_cuda", "mxfp4_to_codes", "mxfp4_to_nibbles",
    "nibbles_to_quintal_digits", "pack_quintal_fragment_major",
    "quintal_dequant", "reference_dequant", "TS4_QUANTIZER_ABI",
    "MOE_TS4",
]

# ------------------------------------------------------------------ tables

LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0], dtype=torch.float64)

# e2m1 nibble -> 2-bit code (proximity: mags {0,.5,1,1.5,2} -> +-1,
# {3,4,6} -> +-4; exact zeros map sign-preservingly to +-1)
NIBBLE_TO_CODE = torch.tensor(
    [2, 2, 2, 2, 2, 3, 3, 3,   # +0,.5,1,1.5,2,3,4,6
     1, 1, 1, 1, 1, 0, 0, 0],  # -0,-.5,-1,-1.5,-2,-3,-4,-6
    dtype=torch.uint8)

# 2-bit code -> e2m1 nibble of the reconstructed level (golden tests)
CODE_TO_NIBBLE = torch.tensor([0xE, 0xA, 0x2, 0x6], dtype=torch.uint8)

# e2m1 magnitudes + midpoints: bucketizing |u| against the midpoints
# reproduces the nearest-with-lo-tie-break snap (|u|==2.5 -> mag 2).
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                         dtype=torch.float64)

_QUINTAL_POW = torch.tensor([1, 5, 25, 125], dtype=torch.int64)

TS4_QUANTIZER_ABI = "tsym4-e8m0-rowmajor-a32-v2-ig"  # row-major sibling of
# vLLM-Moet's "tsym4-e8m0-fragmajor-a32-v2" (identical numerics, layout tag)


def _e8m0_to_float(sbytes: torch.Tensor) -> torch.Tensor:
    """UE8M0 bytes -> float32 powers of two, 2^(byte-127)."""
    return torch.exp2(sbytes.to(torch.float32) - 127.0)


# ------------------------------------------------------------------ quantizer

def quantize_ts4(w: torch.Tensor, zero_mode: str = "auto"):
    """Dense weights [..., K] (fp32/fp64/bf16, K%32==0) -> (codes u8 0..3,
    scales u8 e8m0).  Verbatim port of the vLLM-Moet f64 reference:
    per-block-32 UE8M0 (exp = round(log2(amax/6)), clamped [-127,127],
    all-zero blocks -> -127), e2m1 midpoint snap, nibble->code map.

    zero_mode (exact-zero sign policy; the codebook has no zero level):
      "auto" — sign-preserving, but tensors whose zeros are >95% one-signed
               switch to k-parity alternation (Kimi-K2.7 bias guard);
      "sign" — always sign-preserving;  "alt" — always k-parity alternating.
    """
    if zero_mode not in ("auto", "sign", "alt"):
        raise ValueError(f"zero_mode must be auto|sign|alt, got {zero_mode!r}")
    orig_shape = tuple(w.shape)
    K = orig_shape[-1]
    if K % 32:
        raise ValueError(f"K must be divisible by 32, got {K}")
    w64 = w.to(torch.float64).reshape(-1, K)
    N = w64.shape[0]
    wb = w64.view(N, K // 32, 32)
    amax = wb.abs().amax(dim=2)
    exp = torch.where(amax > 0,
                      torch.round(torch.log2(amax / 6.0 + 1e-30)),
                      torch.full_like(amax, -127.0)).clamp_(-127.0, 127.0)
    scale_bytes = (exp + 127.0).to(torch.uint8)
    u = wb / torch.exp2(exp).unsqueeze(2)          # exact (power-of-2 div)
    mag = torch.bucketize(u.abs().reshape(N, K), _E2M1_MID.to(w.device)).to(torch.uint8)
    neg = torch.signbit(u).reshape(N, K)

    if zero_mode != "sign":
        zero = (u == 0.0).reshape(N, K)
        nz = int(zero.sum())
        if nz:
            nneg = int(neg[zero].sum())
            one_signed = min(nneg, nz - nneg) < 0.05 * nz
            if zero_mode == "alt" or (zero_mode == "auto" and one_signed):
                parity = (torch.arange(K, device=w.device, dtype=torch.uint8)
                          & 1).view(1, K).expand(N, K)
                neg = torch.where(zero, parity.bool(), neg)

    nibbles = mag | (neg.to(torch.uint8) << 3)
    codes = NIBBLE_TO_CODE.to(w.device)[nibbles.long()]
    lead = orig_shape[:-1]
    return codes.view(*lead, K), scale_bytes.view(*lead, K // 32)


def codes_to_plane(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] u8 codes (0..3) -> row-major plane [..., K/4] u8."""
    if codes.shape[-1] % 4:
        raise ValueError("K must be divisible by 4")
    c = codes.to(torch.int32).view(*codes.shape[:-1], codes.shape[-1] // 4, 4)
    packed = (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6))
    return packed.to(torch.uint8)


def plane_to_codes(plane: torch.Tensor) -> torch.Tensor:
    """Row-major plane [..., K/4] -> [..., K] u8 codes."""
    lo = plane & 0x03
    b1 = (plane >> 2) & 0x03
    b2 = (plane >> 4) & 0x03
    hi = (plane >> 6) & 0x03
    return torch.stack((lo, b1, b2, hi), dim=-1).flatten(-2)


def pack_ts4(w: torch.Tensor, zero_mode: str = "auto"):
    """Dense [..., K] -> (row-major plane u8 [..., K/4], scales u8, gs=1.0).
    Signature-compatible with ``pack_nvfp4`` (gs is always 1.0: UE8M0
    absorbs everything)."""
    codes, scales = quantize_ts4(w, zero_mode=zero_mode)
    return codes_to_plane(codes), scales, 1.0


def unpack_ts4(plane: torch.Tensor, scales: torch.Tensor, global_f32=1.0):
    """Deprecated alias of ts4_dequant."""
    return ts4_dequant(plane, scales, global_f32)


def ts4_dequant(plane: torch.Tensor, scales: torch.Tensor, global_f32=1.0):
    """(row-major plane, e8m0 scales, fp32 scalar or fp32 [E] tensor) -> fp32 [..., K]."""
    codes = plane_to_codes(plane)
    vals = LEVELS.to(plane.device, torch.float32)[codes.long()]
    s = _e8m0_to_float(scales)
    if torch.is_tensor(global_f32):
        s = s * global_f32.to(s.device, torch.float32).view(
            (-1,) + (1,) * (s.dim() - 1))
    else:
        s = s * float(global_f32)
    K = vals.shape[-1]
    return (vals.reshape(*vals.shape[:-1], s.shape[-1], K // s.shape[-1])
            * s.unsqueeze(-1)).reshape(*vals.shape[:-1], K)


def reference_dequant(codes: torch.Tensor, scales: torch.Tensor,
                      global_f32=1.0) -> torch.Tensor:
    """[N, K] codes + [N, K/32] e8m0 bytes -> fp32 (golden reference,
    matches vLLM-Moet ``reference_dequant``)."""
    return ts4_dequant(codes_to_plane(codes), scales, global_f32)


# ------------------------------------------------- i8a: int8 activation path
# "tsym4-e8m0-i8a-v1": same weights; the fp32 h1 intermediate is carried as
# int8 + UE8M0 scale per 16 values (4x less Stage-B activation traffic).
# Intra-block dots are exact in int32; pow2 scales fold exactly. The only
# added error is the activation quantization (~0.4% RMS/block).

def quantize_act_i8a(a: torch.Tensor):
    """[..., K] fp32 -> (q int8 [..., K], e8m0 bytes uint8 [..., K/16]).
    Scale byte: ee = exponent(amax) - 6 (amax=0 -> -127), clamped [-127, 127],
    so a/2^ee lands in (-127.94, 127.94]; ties round to even (np.rint)."""
    K = a.shape[-1]
    if K % 16:
        raise ValueError("quantize_act_i8a: K must be a multiple of 16")
    ar = a.reshape(-1, K // 16, 16)
    amax = ar.abs().amax(-1)
    _, ef = torch.frexp(amax)                       # amax = fr * 2^ef, fr in [0.5,1)
    ex = torch.where(amax > 0, ef - 1, torch.full_like(ef, -127))
    ee = (ex - 6).clamp(-127, 127)
    q = torch.round(torch.ldexp(ar, (-ee).unsqueeze(-1))).clamp(-127, 127).to(torch.int8)
    eb = (ee + 127).to(torch.uint8)
    return q.reshape(a.shape), eb


def ts4_reference_forward_i8a(x, ids, w, w13, w13_scale, w13_gs,
                              w2, w2_scale, w2_gs, activation_type: int = 0):
    """i8a reference: fp path through gate/up (identical to fp32 path), then
    the h1 intermediate is quantized per 16-block and every dot is an exact
    integer sum (mirrors shaders/ts4_ffn.comp -DI8A Stage A/B)."""
    x = x.float()
    T, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    K = ids.shape[-1]
    ids = ids.long().reshape(T, K)
    w = w.float().reshape(T, K)

    deq13 = ts4_dequant(w13, w13_scale, w13_gs)                 # [E, 2I, H]
    wv = LEVELS.to(x.device, torch.int32)[plane_to_codes(w2).long()]  # [E,H,I]
    wsc = w2_scale.to(torch.int32)                              # [E,H,I/32]
    gs2e = w2_gs if torch.is_tensor(w2_gs) else None
    nb = I // 16

    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        mask = (ids == e)
        rows_t, rows_k = torch.nonzero(mask, as_tuple=True)
        if rows_t.numel() == 0:
            continue
        n = rows_t.numel()
        xe = x[rows_t]
        gate = xe @ deq13[e, :I].transpose(0, 1)
        up = xe @ deq13[e, I:].transpose(0, 1)
        if activation_type == 0:
            act = torch.nn.functional.silu(gate) * up
        elif activation_type == 1:
            act = torch.nn.functional.gelu(gate) * up
        else:
            raise ValueError(f"unsupported activation_type={activation_type}")
        q, eb = quantize_act_i8a(act)
        qf = q.float().reshape(n, nb, 16)
        wv2 = wv[e].float().reshape(H, nb, 16)
        dots = torch.einsum('nbk,hbk->nbh', qf, wv2)            # exact int sums
        sc = torch.exp2((eb.int().unsqueeze(1)
                         + wsc[e].repeat_interleave(2, dim=-1).unsqueeze(0)
                         - 254).float())
        h2 = (dots.permute(0, 2, 1) * sc).sum(-1)               # [n, H]
        gs2 = gs2e[e] if gs2e is not None else w2_gs
        contrib = h2 * gs2 * w[rows_t, rows_k].unsqueeze(-1)
        out.index_add_(0, rows_t, contrib)
    return out


# ------------------------------------------------------------------ forward

def ts4_reference_forward(x, ids, w, w13, w13_scale, w13_gs,
                          w2, w2_scale, w2_gs, activation_type: int = 0):
    """MoE FFN reference on ts4 planes (mirrors nvfp4_reference_forward).

    x  [T, H] fp16/fp32, ids [T, K] int, w [T, K] fp32 route weights,
    w13 [E, 2I, H/4] u8 + w13_scale [E, 2I, H/32] u8 + w13_gs float|[E],
    w2  [E, H,  I/4] u8 + w2_scale  [E, H,  I/32] u8 + w2_gs  float|[E].
    """
    x = x.float()
    T, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    K = ids.shape[-1]
    ids = ids.long().reshape(T, K)
    w = w.float().reshape(T, K)

    deq13 = ts4_dequant(w13, w13_scale, w13_gs)     # [E, 2I, H]
    deq2 = ts4_dequant(w2, w2_scale, w2_gs)         # [E, H, I]

    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        mask = (ids == e)
        rows_t, rows_k = torch.nonzero(mask, as_tuple=True)
        if rows_t.numel() == 0:
            continue
        xe = x[rows_t]
        gate = xe @ deq13[e, :I].transpose(0, 1)
        up = xe @ deq13[e, I:].transpose(0, 1)
        if activation_type == 0:
            act = torch.nn.functional.silu(gate) * up
        elif activation_type == 1:
            act = torch.nn.functional.gelu(gate) * up
        else:
            raise ValueError(f"unsupported activation_type={activation_type}")
        contrib = act @ deq2[e].transpose(0, 1)
        contrib = contrib * w[rows_t, rows_k].unsqueeze(-1)
        out.index_add_(0, rows_t, contrib)
    return out


# ---------------------------------------------------------------- CUDA interop
# vLLM-Moet moe_w2 fragment-major planes (QMMA m16n8k32 fragment order).
# Row-major <-> fragment-major here are exact inverses, so vLLM-Moet packs
# serve these shaders unchanged after one offline repack.

def pack_fragment_major(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] u8 codes -> fragment-major plane [N*K/4] u8 (byte-identical
    to vLLM-Moet moe_w2 planes; N%16==0, K%64==0)."""
    N, K = codes.shape
    assert N % 16 == 0 and K % 64 == 0
    c = codes.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    # [nb, tile(g|g+8), g, kb, k32(a|b), half(lo|hi), t, k4]
    # -> [nb, kb, g, t, tile, k32, half, k4]
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    c = c.view(-1, 4).to(torch.int32)
    packed = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6))
    return packed.to(torch.uint8).flatten()


def unpack_fragment_major(plane: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Inverse of pack_fragment_major -> [N, K] u8 codes."""
    assert N % 16 == 0 and K % 64 == 0
    assert plane.numel() == N * K // 4
    c = plane.to(torch.int32)
    codes = torch.stack([(c >> (2 * j)) & 3 for j in range(4)], dim=-1)  # [X,4]
    codes = codes.view(N // 16, K // 64, 8, 4, 2, 2, 2, 4)
    # packed dim order (nb,kb,g,t,tile,k32,half,k4) -> canonical
    # (nb,tile,g,kb,k32,half,t,k4): inverse of the pack permute
    codes = codes.permute(0, 4, 2, 1, 5, 6, 3, 7).contiguous().view(N, K)
    return codes.to(torch.uint8)


def pack_scales_cuda(scales: torch.Tensor) -> torch.Tensor:
    """[N, K/32] e8m0 -> fragment-major scale plane (vLLM-Moet layout)."""
    N, KS = scales.shape
    assert N % 16 == 0
    return scales.view(N // 16, 16, KS).transpose(1, 2).contiguous().flatten()


def unpack_scales_cuda(plane: torch.Tensor, N: int, KS: int) -> torch.Tensor:
    assert N % 16 == 0 and plane.numel() == N * KS
    return plane.view(N // 16, KS, 16).transpose(1, 2).contiguous().view(N, KS)


def mxfp4_to_nibbles(w_packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] u8 packed e2m1 pairs -> [..., K] u8 nibbles (low nibble = even k)."""
    lo = w_packed & 0xF
    hi = w_packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def mxfp4_to_codes(w_packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] mxfp4 -> [..., K] u8 2-bit codes."""
    return NIBBLE_TO_CODE.to(w_packed.device)[mxfp4_to_nibbles(w_packed).long()]


# --- radix-5 quintal planes (vLLM-Moet moe_w4q split-FP4, 2.5 bits/elem) ---

def nibbles_to_quintal_digits(nib: torch.Tensor) -> torch.Tensor:
    """e2m1 nibbles -> base-5 digits: magnitude index within the base-code
    class (small {0,.5,1,1.5,2} -> 0..4; big {3,4,6} -> mag-5 in 0..2)."""
    mag = (nib & 7).long()
    code = NIBBLE_TO_CODE.to(nib.device)[nib.long()]
    big = (code == 0) | (code == 3)
    return torch.where(big, mag - 5, mag)


def pack_quintal_fragment_major(nib: torch.Tensor) -> torch.Tensor:
    """[N, K] u8 e2m1 nibbles -> quintal plane (N*K*5/16 bytes), bit-exact
    vLLM-Moet moe_w4q layout (P8 256 B + P2 64 B per (nb, kb64) block)."""
    N, K = nib.shape
    assert N % 16 == 0 and K % 64 == 0
    d = nibbles_to_quintal_digits(nib)
    d = d.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    d = d.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 8, 4)
    words = (d * _QUINTAL_POW.to(d.device)).sum(-1)          # [lanes, 8] x 10 bit
    shorts = torch.zeros(words.shape[0], 5, dtype=torch.int64, device=d.device)
    for w in range(8):
        b = 10 * w
        s, off = b // 16, b % 16
        shorts[:, s] |= words[:, w] << off
        if off > 6:
            shorts[:, s + 1] |= words[:, w] >> (16 - off)
    shorts &= 0xFFFF
    by = torch.stack([shorts & 0xFF, shorts >> 8], dim=-1).view(-1, 10)
    by = by.view(-1, 32, 10).to(torch.uint8)
    return torch.cat([by[:, :, :8].reshape(-1, 256),
                      by[:, :, 8:].reshape(-1, 64)], dim=1).flatten()


def quintal_dequant(nib: torch.Tensor) -> torch.Tensor:
    """Values the quintal decode reconstructs = TRUE e2m1 (equality with
    the e2m1 table IS the bit-exactness criterion)."""
    dev = nib.device
    digit = nibbles_to_quintal_digits(nib)
    code = NIBBLE_TO_CODE.to(dev)[nib.long()]
    big = (code == 0) | (code == 3)
    mag_idx = torch.where(big, digit + 5, digit)
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
    val = mags[mag_idx]
    return torch.where(code <= 1, -val, val)


# ---------------------------------------------------------------- a16: fp16 activation path
# "tsym4-e8m0-a16-v1" (W2A16): same weights; the h1 intermediate is rounded to
# fp16 before the W2 GEMM (exactly what the -DA16 Vulkan Stage A/B do). Half
# the Stage-B activation traffic of the fp path, ~10-bit mantissa precision.

def ts4_reference_forward_a16(x, ids, w, w13, w13_scale, w13_gs,
                              w2, w2_scale, w2_gs, activation_type: int = 0):
    x = x.float()
    T, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    K = ids.shape[-1]
    ids = ids.long().reshape(T, K)
    w = w.float().reshape(T, K)

    deq13 = ts4_dequant(w13, w13_scale, w13_gs)     # [E, 2I, H]
    deq2 = ts4_dequant(w2, w2_scale, w2_gs)         # [E, H, I]

    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        mask = (ids == e)
        rows_t, rows_k = torch.nonzero(mask, as_tuple=True)
        if rows_t.numel() == 0:
            continue
        xe = x[rows_t]
        # fp64 accumulation (same as the reference); round gate/up to fp16,
        # silu*up in fp32, then round h1 to fp16 — the A16 h1 rounding.
        gate = (xe @ deq13[e, :I].transpose(0, 1)).float()
        up = (xe @ deq13[e, I:].transpose(0, 1)).float()
        if activation_type == 0:
            act = torch.nn.functional.silu(gate) * up
        elif activation_type == 1:
            act = torch.nn.functional.gelu(gate) * up
        else:
            raise ValueError(f"unsupported activation_type={activation_type}")
        act = act.half().float()                   # the A16 h1 rounding
        contrib = act @ deq2[e].transpose(0, 1)
        contrib = contrib * w[rows_t, rows_k].unsqueeze(-1)
        out.index_add_(0, rows_t, contrib)
    return out


# ------------------------------------------------------------------ fallback

def _need_tensors(name):
    raise NotImplementedError(
        f"CPU fallback needs tensors; use ts4_reference_forward "
        f"(arg {name!r} is a raw pointer; pass tensors or build the C++ ext)")


class MOE_TS4:
    """Pure-torch fallback of the pybind MOE_TS4 (same signature slots).

    Constructor takes tensors in the pointer slots:
      MOE_TS4(cfg, w13, w2, w13_scale, w2_scale, w13_gs=1.0, w2_gs=1.0)
      w13 [E,2I,H/4] u8 plane + w13_scale [E,2I,H/32] u8 e8m0, same for w2.
    Works on any torch device (CPU, CUDA, ROCm) — on a ROCm torch build
    this IS the torch-backend compute path (dequant -> grouped GEMM).
    ``quantize=True``: w13/w2 are dense [E,2I,H]/[E,H,I] tensors and are
    quantized once at construction via quantize_ts4.
    """

    def __init__(self, cfg, w13=0, w2=0, w13_scale=0, w2_scale=0,
                 w13_gs=1.0, w2_gs=1.0, gs2e=0, quantize=False, i8a=False,
                 a16=False):
        self.i8a = bool(i8a)
        self.a16 = bool(a16)
        if quantize:
            for nm, p in (("w13", w13), ("w2", w2)):
                if not torch.is_tensor(p) or p.dtype == torch.uint8:
                    raise ValueError(f"quantize=True needs float {nm}")
            w13, w13_scale, _ = pack_ts4(w13)
            w2, w2_scale, _ = pack_ts4(w2)
        for name, p in (("w13", w13), ("w2", w2),
                        ("w13_scale", w13_scale), ("w2_scale", w2_scale)):
            if not torch.is_tensor(p):
                _need_tensors(name)
        E = cfg.expert_num
        H, I = cfg.hidden_size, cfg.intermediate_size
        if H % 32 or I % 32:
            raise ValueError("ts4 requires hidden_size%32==0 and intermediate_size%32==0")
        want = {"w13": (E, 2 * I, H // 4), "w13_scale": (E, 2 * I, H // 32),
                "w2": (E, H, I // 4), "w2_scale": (E, H, I // 32)}
        for name, t in (("w13", w13), ("w13_scale", w13_scale),
                        ("w2", w2), ("w2_scale", w2_scale)):
            if tuple(t.shape) != want[name] or t.dtype != torch.uint8:
                raise ValueError(f"{name}: want uint8 {want[name]}, got "
                                 f"{t.dtype} {tuple(t.shape)}")
        self.cfg = cfg
        self.w13, self.w2, self.w13_scale, self.w2_scale = w13, w2, w13_scale, w2_scale
        self.w13_gs = w13_gs
        self.w2_gs = w2_gs
        self.gs2e = gs2e.float() if torch.is_tensor(gs2e) and gs2e.numel() > 1 else None

    def _gs13(self):
        return self.gs2e if torch.is_tensor(self.gs2e) else self.w13_gs

    def _gs2(self):
        return self.gs2e if torch.is_tensor(self.gs2e) else self.w2_gs

    def forward(self, x, ids, routing_weights):
        if getattr(self, 'a16', False):
            fn = ts4_reference_forward_a16
        elif getattr(self, 'i8a', False):
            fn = ts4_reference_forward_i8a
        else:
            fn = ts4_reference_forward
        return fn(
            x, ids, routing_weights,
            self.w13, self.w13_scale, self._gs13(),
            self.w2, self.w2_scale, self._gs2(),
            activation_type=self.cfg.activation_type)

    def cpu_decode(self, stream=0, bs=0, k=0, in_gpu=0, ids_gpu=0, w_gpu=0, out_gpu=0):
        _need_tensors("in_gpu")

    def cpu_prefill(self, qlen=0, k=0, ids_cpu=0, w_cpu=0, in_cpu=0, out_cpu=0):
        _need_tensors("in_cpu")

    def gpu_prefill(self, *a, **kw):
        return None

    def register_numa_weights(self, *a, **kw):
        return None

    def unregister_numa_weights(self, *a, **kw):
        return None

    def __repr__(self):
        c = self.cfg
        return (f"MOE_TS4(pure-torch, expert_num={c.expert_num}, top_k={c.top_k}, "
                f"hidden_size={c.hidden_size}, intermediate_size={c.intermediate_size})")
