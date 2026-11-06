"""FP3 (W3A16, 3-bit NVFP4-codebook) fused MoE kernels.

The checkpoint stores 3-bit codebook indices packed MSB-first into 48-bit
blocks (16 codes = 6 bytes, big-endian) with one uint8 e4m3-derived block
scale per group of 16 codes, plus a per-expert FP32 global scale. Serving
repacks the payload once at weight load into an LSB-first 8-codes-per-3-bytes
layout (see :func:`repack_fp3_w3a16_lsb3`) -- a pure permutation of the
checkpoint bytes, so the repacked model costs no extra VRAM. Two executors
share that payload:

* ``Fp3W3A16GemvG1/G2`` (JIT CUDA, ``csrc/moe/fp3_w3a16_gemv.cuh``): grouped
  warp-per-row SIMT GEMV, one launch per GEMM covering all active experts;
  fp32 end to end. Used for decode-size slot counts, where the tensor-core
  Triton path is launch- and SM-starved.
* ``_fp3_w3a16_gemm_kernel`` (Triton): grouped tensor-core GEMM with in-loop
  dequant, bf16 dots and FP32 accumulation; used for prefill-size batches.
* flashinfer ``moe_nvfp4_w4a4`` (SM12x): on request, prefill-size batches
  instead run block-scaled FP4 x FP4 tensor cores, fed by a lossless repack
  of the payload into canonical NVFP4 storage (the dispatcher arms this at
  or above the SGLANG_SM12X_W3A4_PREFILL_MIN_TOKENS threshold).

    w = CB[code] * E4M3[sf_byte & 0xFE] * (sf_byte & 1 ? -1 : 1) * gs
    CB = [-6, -2, -1, 0, 1, 2, 3, 6]

Layouts (E local experts, K hidden, N output, group_size 16):
    payload : uint8 [E, N, K * 3 // 8]  (LSB-first: byte triple 3g..3g+2
          holds codes 8g..8g+7, code j at bits 3j of the 24-bit group)
    scales  : uint8 [E, N, K // 16]      (raw ModelOpt sf bytes, unswizzled)
    gscale  : fp32  [E] or [E, 2] for the w13 gate/up halves

The GEMM1 global scale is selected per output tile (gate rows vs up rows of
w13), so the gate/up halves keep their exact per-half checkpoint scales and
no gate/up ratio fold is needed.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Fused MoE slot counts (tokens x topk) at or below which the grouped SIMT
# GEMV beats the Triton tensor-core path on sm_120. v6's per-block X staging
# adds traffic that grows with slots; measured crossover sits between 320 and
# 512 with the Qwen3.8-Flash-Next-FP3 expert shape.
_GEMV_MAX_SLOTS = 384

# Slot count at or below which the CuTe DSL LSB3 fork (3-bit codes expanded
# to NVFP4 nibbles in registers, sm12x W4A16 tile kernels) beats both other
# paths on sm_120; measured 1.5-3.8x over GEMV/Triton up to ~1536 tokens at
# top-k 10, Triton GEMM wins from ~2048 tokens up.
_CUTE_MAX_SLOTS = 15360
_cute_failures = 0
_cute_disabled = False
# swizzled-scale cache keyed by the raw scale tensor's storage; the entry
# keeps a reference so the data_ptr key cannot be recycled underneath it.
_cute_sf_cache: dict = {}

# flashinfer sm12x NVFP4 W4A4 grouped tensor-core arm, used for prefill-size
# batches: the 3-bit payload repacks losslessly into canonical NVFP4 storage
# (the codebook is a subset of the e2m1 grid; the sf mirror sign moves onto
# the nibble), block scales are swizzled, and the gate/up global scales fold
# as in the sm12x_nvfp4_bf16 runner (the ratio rides on gs2). Activations are
# quantized to NVFP4 inside the kernel, so batches at or above
# SGLANG_SM12X_W3A4_PREFILL_MIN_TOKENS (0 disables the arm) run fp4 x fp4
# block-scaled MMA instead of the SIMT dequantize GEMMs. The repack costs
# ~2.1 ms per 512-expert layer per call, so on a KV-tight server (cache off,
# see below) the win starts around 1536 tokens: 1.70x at 2048, 1.65x at
# 4096 vs the CuTe fork. With SGLANG_SM12X_W3A4_CACHE_MB big enough to hold
# every MoE layer the repack amortizes and the arm wins from ~96 tokens up
# (5-5.7x over Triton w3a16, 5.3x over the CuTe fork at 4096); a canonical
# copy costs ~1.5 GB per 512-expert layer, so caching all layers of a deep
# model OOMs a KV-tight server — 0 (default) repacks into allocator-
# recycled scratch per call, peak ~1.5 GB regardless of layer count.
_w4a4_failures = 0
_w4a4_disabled = False
# canonical repack LRU cache keyed by the weight storages; entries keep a
# reference to the source tensors so data_ptr keys cannot be recycled.
_w4a4_cache: OrderedDict[tuple, tuple] = OrderedDict()
_w4a4_cached_bytes = 0
# flashinfer workspaces, keyed by (bucket, h, i, e, topk, device)
_w4a4_ws: dict = {}


def _is_sm12x() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


def _cute_sf_swizzled(sf: torch.Tensor) -> torch.Tensor:
    """Fold the codebook mirror bit into the e4m3 sign and swizzle to the
    128x4 tile layout the CuTe fork consumes. Done once per weight tensor."""
    key = (sf.data_ptr(), tuple(sf.shape))
    hit = _cute_sf_cache.get(key)
    if hit is not None:
        return hit[1]
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    folded = (sf & 0xFE) ^ ((sf & 1) << 7)
    sw = swizzle_blockscale(folded.view(torch.float8_e4m3fn)).to(sf.device)
    _cute_sf_cache[key] = (sf, sw)
    return sw


def _cute_fork_experts(
    hidden_states,
    w13,
    w13_sf,
    w13_gs,
    w2,
    w2_sf,
    w2_gs,
    topk_weights,
    topk_ids,
    routed_scaling_factor,
):
    from sglang.kernels.ops.moe.sm12x_fp3_lsb3_w3a16_moe import run_moe_fp3_lsb3

    if w13_gs.dim() == 1:
        w13_gs = w13_gs.unsqueeze(1).expand(-1, 2).contiguous()
    wts = (
        topk_weights
        if routed_scaling_factor == 1.0
        else topk_weights.float() * routed_scaling_factor
    )
    out = torch.empty_like(hidden_states)
    run_moe_fp3_lsb3(
        hidden_states,
        topk_ids,
        wts,
        w13,
        _cute_sf_swizzled(w13_sf),
        w13_gs,
        w2,
        _cute_sf_swizzled(w2_sf),
        w2_gs.float(),
        out,
    )
    return out


@triton.jit
def _lsb3_to_nvfp4_kernel(
    payload_ptr, sf_ptr, out32_ptr, n_groups, BLOCK_G: tl.constexpr
):
    """Repack the LSB3 payload (8 codes per 3-byte group) into packed e2m1
    nibbles (4 codes per byte, one u32 per group). The codebook
    [-6,-2,-1,0,1,2,3,6] lands on e2m1 magnitude codes [7,4,2,0,2,4,5,7]
    with sign on codes 0..2; the sf mirror bit (bit0) flips the nibble sign
    so the canonical (non-negative) block scale is just ``sf & 0xFE``.
    Each group's 8 codes share one sf byte (16 codes per scale block).
    payload_ptr/out32_ptr are viewed as uint32 (so kb3 % 4 == 0).
    """
    pid = tl.program_id(0)
    g = pid * BLOCK_G + tl.arange(0, BLOCK_G)
    gmask = g < n_groups
    byte = g * 3
    w0 = tl.load(payload_ptr + byte // 4, mask=gmask, other=0)
    w1 = tl.load(payload_ptr + (byte + 2) // 4, mask=gmask, other=0)
    pair = w0.to(tl.uint64) | (w1.to(tl.uint64) << 32)
    v = ((pair >> ((byte % 4) * 8).to(tl.uint64)) & 0xFFFFFF).to(tl.uint32)
    j = tl.arange(0, 8)[None, :]
    code = (v[:, None] >> (3 * j)) & 7
    mag = tl.where(code == 1, 4, 0)
    mag = tl.where(code == 2, 2, mag)
    mag = tl.where(code == 4, 2, mag)
    mag = tl.where(code == 5, 4, mag)
    mag = tl.where(code == 6, 5, mag)
    mag = tl.where((code == 0) | (code == 7), 7, mag)
    sgn = (code < 3).to(tl.uint32)
    sfl = tl.load(sf_ptr + g // 2, mask=gmask, other=0).to(tl.uint32)
    nib = mag | ((sgn ^ (sfl[:, None] & 1)) << 3)
    lo, hi = tl.split(tl.reshape(nib, (BLOCK_G, 4, 2)))
    b = (lo | (hi << 4)).to(tl.uint32)
    word = tl.sum(b << (8 * tl.arange(0, 4))[None, :].to(tl.uint32), axis=1)
    tl.store(out32_ptr + g, word, mask=gmask)


@triton.jit
def _sf_fold_swizzle_kernel(
    sf_ptr, out_ptr, mc, c_len, kt_len, total, BLOCK: tl.constexpr
):
    """Fold the codebook mirror bit into the e4m3 sign and swizzle the raw
    block scales into the 128x4-tile interleaved layout in one pass (the
    torch swizzle_blockscale path allocates+zero-pads+copies the full scale
    matrix per call, ~4x the traffic for aligned shapes). Assumes M % 128 ==
    0 and C % 4 == 0 (no padding); callers fall back otherwise."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < total
    s = tl.load(sf_ptr + idx, mask=m, other=0).to(tl.uint32)
    folded = (s & 0xFE) ^ ((s & 1) << 7)
    e = idx // mc
    rem = idx % mc
    r = rem // c_len
    c = rem % c_len
    mt = r // 128
    mr = r % 128
    kt = c // 4
    kc = c % 4
    off = e * mc + ((mt * kt_len + kt) * 32 + (mr % 32)) * 16 + (mr // 32) * 4 + kc
    tl.store(out_ptr + off, folded.to(tl.uint8), mask=m)


def _sf_fold_swizzle(sf: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """swizzle_blockscale((sf & 0xFE) ^ ((sf & 1) << 7)) for 3-D uint8
    scales with sf.shape[1] % 128 == 0 and sf.shape[2] % 4 == 0. ``out``
    reuses a preallocated buffer (same shape) when given."""
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    m, c = sf.shape[1], sf.shape[2]
    if m % 128 != 0 or c % 4 != 0:
        folded = (sf & 0xFE) ^ ((sf & 1) << 7)
        return swizzle_blockscale(folded.view(torch.float8_e4m3fn)).view(torch.uint8)
    if out is None:
        out = torch.empty_like(sf)
    else:
        assert out.shape == sf.shape
    total = sf.numel()
    _sf_fold_swizzle_kernel[(triton.cdiv(total, 1024),)](
        sf.reshape(-1),
        out.reshape(-1),
        m * c,
        c,
        c // 4,
        total,
        BLOCK=1024,
        num_warps=4,
    )
    return out


def _w4a4_scales(w13_gs, w2_gs, dev):
    """Kernel scale vectors from checkpoint global scales.

    Canonical storage has one global scale per weight matrix. With the gate
    scale on w13, the gate/up ratio rescales the up half of h, which the
    GEMM2 scale absorbs exactly (same fold as the sm12x_nvfp4_bf16 runner);
    a zero gate scale keeps the raw gs2.
    """
    if w13_gs.dim() == 2:
        gate = w13_gs[:, 0].float()
        up = w13_gs[:, 1].float()
        gs13 = gate.contiguous()
        ratio = torch.where(gate > 0, up / gate.clamp_min(1e-30), torch.ones_like(up))
        gs2 = (w2_gs.float() * ratio).contiguous()
    else:
        gs13 = w13_gs.float().contiguous()
        gs2 = w2_gs.float().contiguous()
    a_gs = torch.ones(1, dtype=torch.float32, device=dev)
    return gs13, gs2, a_gs


def _w4a4_canonical(w13, w13_sf, w13_gs, w2, w2_sf, w2_gs, bufs=None):
    """Canonical NVFP4 buffers + the kernel scale vectors. Returns
    (w13_4, w13_sf4, gs13, w2_4, w2_sf4, gs2, a_gs) with a_gs the (scalar)
    activation quantization multiplier. Caching is off by default
    (SGLANG_SM12X_W3A4_CACHE_MB=0): each call repacks into scratch that the
    allocator recycles for the next layer, keeping peak overhead to ~1.5 GB
    regardless of layer count (caching every layer is what OOMed a
    KV-tight server). A positive budget enables an LRU cache of that many
    MB, for hosts with real headroom. bufs=(b13, b2) supplies preallocated
    (payload, swizzled-sf) output buffers (the prefetch double-buffer);
    shapes are derived from the payload and checked against the buffers."""
    global _w4a4_cached_bytes
    budget = envs.SGLANG_SM12X_W3A4_CACHE_MB.get() * (1 << 20)
    key = (w13.data_ptr(), w13_sf.data_ptr(), w2.data_ptr(), w2_sf.data_ptr())
    hit = _w4a4_cache.get(key)
    if hit is not None:
        _w4a4_cache.move_to_end(key)
        return hit[0]

    def repack(payload, sf, buf=None):
        e, r, kb3 = payload.shape
        assert kb3 % 4 == 0, "payload row must be uint32-viewable"
        n_groups = payload.numel() // 3
        if buf is None:
            out32 = torch.empty(
                e, r, kb3 * 4 // 3 // 4, dtype=torch.uint32, device=payload.device
            )
            sf_out = None
        else:
            out4, sf_out = buf
            assert out4.shape == (e, r, kb3 * 4 // 3)
            out32 = out4.view(torch.uint32).view(e, r, kb3 * 4 // 3 // 4)
        _lsb3_to_nvfp4_kernel[(triton.cdiv(n_groups, 512),)](
            payload.view(torch.uint32), sf, out32, n_groups, BLOCK_G=512, num_warps=4
        )
        out = out32.view(torch.uint8).view(e, r, kb3 * 4 // 3)
        return out, _sf_fold_swizzle(sf, sf_out)

    b13, b2 = bufs if bufs is not None else (None, None)
    w13_4, w13_sf4 = repack(w13, w13_sf, b13)
    w2_4, w2_sf4 = repack(w2, w2_sf, b2)
    gs13, gs2, a_gs = _w4a4_scales(w13_gs, w2_gs, w13.device)
    if budget > 0 and bufs is None:
        entry = (
            (w13_4, w13_sf4, gs13, w2_4, w2_sf4, gs2, a_gs),
            (w13, w13_sf, w2, w2_sf),  # keepalive: data_ptr keys stay valid
        )
        _w4a4_cache[key] = entry
        _w4a4_cached_bytes += sum(t.numel() * t.element_size() for t in entry[0])
        while _w4a4_cached_bytes > budget and len(_w4a4_cache) > 1:
            _, oldest = _w4a4_cache.popitem(last=False)
            _w4a4_cached_bytes -= sum(t.numel() * t.element_size() for t in oldest[0])
    return (w13_4, w13_sf4, gs13, w2_4, w2_sf4, gs2, a_gs)


# --- W4A4 side-stream repack prefetch ------------------------------------
# Arm calls walk the same chain of MoE weight sets every forward. Learn the
# chain on the first forward, then repack the successor layer on a side
# stream while the current layer's FP4 GEMMs are in flight: repack is
# bandwidth-bound, the GEMMs and the next layer's attention are not, so most
# of the ~2 ms/layer repack disappears. Costs up to two repack sets (~2.6
# GB) in flight during prefill; SGLANG_SM12X_W3A4_PREFETCH=0 reverts to the
# synchronous per-call repack.
_prefetch_stream = None
_prefetch_pending: dict = {}  # key -> (canonical tuple, done event, slot)
_prefetch_succ: dict = {}  # key -> (next key, next weight snapshot)
_prefetch_prev = None  # key of the previous arm call
# Double-buffer pool: two resident canonical sets per weight shape,
# alternated by consecutive calls, so the set the current GEMMs read is
# never the set the next prefetch overwrites. Allocating per call instead
# made the allocator cudaMalloc fresh blocks each layer (blocks freed on a
# side stream cannot recycle into the main pool): prefill slowed 2.8 ->
# 5.6 ms in the loop bench.
_pf_pool: dict = {}  # shape key -> [slot0 (b13, b2), slot1 (b13, b2)]
_pf_parity = 0


def _arm_key(w13, w13_sf, w2, w2_sf):
    return (w13.data_ptr(), w13_sf.data_ptr(), w2.data_ptr(), w2_sf.data_ptr())


def _arm_record_and_prefetch(key, weights):
    """Learn the forward's weight chain and start repacking the successor
    layer on the side stream. Payload/sf tensors are model-owned and
    stable across calls; global scales are per-call temporaries in some
    paths, so a small fp32 snapshot is taken when recording the link."""
    global _prefetch_stream, _prefetch_prev, _pf_parity
    if _prefetch_prev is not None:
        _prefetch_succ[_prefetch_prev] = (key, weights)
    _prefetch_prev = key
    if torch.cuda.is_current_stream_capturing():
        return
    nxt = _prefetch_succ.get(key)
    if nxt is None or nxt[0] in _prefetch_pending:
        return
    nkey, (n13, n13_sf, n13_gs, n2, n2_sf, n2_gs) = nxt
    skey = (n13.shape, n13_sf.shape, n2.shape, n2_sf.shape)
    bufs = _pf_pool.get(skey)
    if bufs is None:
        # Two resident (payload, swizzled-sf) sets per weight shape,
        # allocated on the main stream and never freed: the allocator is
        # never in the loop. ~2.6 GB per distinct MoE weight shape.
        def _mk(payload, sf_shape):
            e, r, kb3 = payload.shape
            return (
                torch.empty(
                    e, r, kb3 * 4 // 3, dtype=torch.uint8, device=payload.device
                ),
                torch.empty(
                    e,
                    sf_shape[1],
                    sf_shape[2],
                    dtype=torch.uint8,
                    device=payload.device,
                ),
            )

        bufs = [(_mk(n13, n13_sf.shape), _mk(n2, n2_sf.shape)) for _ in range(2)]
        _pf_pool[skey] = bufs
    slot = _pf_parity
    if any(ent[2] == slot for ent in _prefetch_pending.values()):
        return  # a live pending still pins this slot; skip this round
    _pf_parity ^= 1
    if _prefetch_stream is None:
        _prefetch_stream = torch.cuda.Stream()
    cur = torch.cuda.current_stream()
    _prefetch_stream.wait_stream(cur)
    b13, b2 = bufs[slot]
    with torch.cuda.stream(_prefetch_stream):
        try:
            canon = _w4a4_canonical(
                n13, n13_sf, n13_gs, n2, n2_sf, n2_gs, bufs=(b13, b2)
            )
        except RuntimeError:  # OOM under prefill pressure: inline next call
            canon = None
        ev = torch.cuda.Event()
        ev.record(_prefetch_stream)
    if canon is not None:
        _prefetch_pending[nkey] = (canon, ev, slot)
    while len(_prefetch_pending) > 2:
        _prefetch_pending.pop(next(iter(_prefetch_pending)))


def _w4a4_bucket(t: int) -> int:
    # flashinfer compiles and allocates per token count; prefill batch sizes
    # never repeat, so bucket to bound the compile set. Below 1024 keep a
    # power-of-two ladder (decode/verify sizes are rare); at or above 1024
    # round to 1024 multiples: a power-of-two ladder pads a 4005-token tail
    # chunk to 8192 (2x wasted GEMM work), this wastes <=25%.
    if t < 1024:
        return t if t & (t - 1) == 0 else 1 << (t - 1).bit_length()
    return (t + 1023) // 1024 * 1024


def _w4a4_experts(
    hidden_states,
    w13,
    w13_sf,
    w13_gs,
    w2,
    w2_sf,
    w2_gs,
    topk_weights,
    topk_ids,
    routed_scaling_factor,
    fused=False,
):
    if fused:
        # Fused arm: the kernel unpacks LSB3 -> FP4 in registers, so the
        # checkpoint payload and raw scales go to the GEMMs untouched.
        from sglang.kernels.ops.moe.sm12x_fp3_lsb3_w4a4_moe import (
            allocate_workspace,
            run_moe_fp3_lsb3_w4a4 as run_moe_w4a4,
        )
    else:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4 import (
            allocate_workspace,
            run_moe_w4a4,
        )

    m, h = hidden_states.shape
    e = w13.shape[0]
    topk = topk_ids.shape[1]
    inter = w2.shape[2] * 8 // 3
    prefetch_on = envs.SGLANG_SM12X_W3A4_PREFETCH.get() and not fused
    key = _arm_key(w13, w13_sf, w2, w2_sf)
    # The prefetch fast path: the previous layer scheduled a repack of
    # these weights on the side stream. It is consumed (event wait) only
    # right before the GEMMs below, so the padding and routing prep here
    # keeps the main stream busy while the repack finishes.
    pend = None
    if prefetch_on and not torch.cuda.is_current_stream_capturing():
        pend = _prefetch_pending.pop(key, None)
    if pend is not None:
        w13_4, w13_sf4, gs13, w2_4, w2_sf4, gs2, a_gs = pend[0]
    elif fused:
        w13_4, w13_sf4 = w13, w13_sf
        w2_4, w2_sf4 = w2, w2_sf
        gs13, gs2, a_gs = _w4a4_scales(w13_gs, w2_gs, hidden_states.device)
    else:
        w13_4, w13_sf4, gs13, w2_4, w2_sf4, gs2, a_gs = _w4a4_canonical(
            w13, w13_sf, w13_gs, w2, w2_sf, w2_gs
        )
    b = _w4a4_bucket(m)
    dev = hidden_states.device
    if b != m:
        ids = torch.full((b, topk), e, dtype=torch.int32, device=dev)
        ids[:m] = topk_ids.to(torch.int32)
        wts = torch.zeros(b, topk, dtype=torch.float32, device=dev)
        wts[:m] = topk_weights
        x = torch.zeros(b, h, dtype=torch.bfloat16, device=dev)
        x[:m] = hidden_states
    else:
        ids = topk_ids.to(torch.int32)
        wts = topk_weights.to(torch.float32)
        x = hidden_states
    if routed_scaling_factor != 1.0:
        wts = wts * routed_scaling_factor
    ws_key = (b, h, inter, e, topk, dev, fused)
    ws = _w4a4_ws.get(ws_key)
    if ws is None:
        ws = allocate_workspace(b, h, inter, e, topk, True, dev)
        _w4a4_ws[ws_key] = ws
    if prefetch_on:
        # Launch the successor layer's repack first (side stream waits on
        # everything queued so far, including the previous layer's GEMMs
        # that last read these recycled blocks), THEN let this call's GEMMs
        # queue behind its own repack's event. The repack is bandwidth-
        # bound, the GEMMs that run alongside it are not. The global
        # scales are per-call temporaries in the runner, so store small
        # fp32 snapshots next to the stable payload/sf tensors.
        _arm_record_and_prefetch(
            key,
            (
                w13,
                w13_sf,
                w13_gs.float().clone(),
                w2,
                w2_sf,
                w2_gs.float().clone(),
            ),
        )
        if pend is not None:
            cur = torch.cuda.current_stream()
            cur.wait_event(pend[1])
            # freed at call end: keep the allocator from recycling these
            # blocks out from under the GEMMs' reads.
            for t in pend[0]:
                t.record_stream(cur)
    out = torch.empty(b, h, dtype=torch.bfloat16, device=dev)
    run_moe_w4a4(
        x,
        ids,
        wts,
        w13_4,
        w13_sf4,
        gs13,
        w2_4,
        w2_sf4,
        gs2,
        a_gs,
        a_gs,
        out,
        workspace=ws,
    )
    return out[:m]


def repack_fp3_w3a16_lsb3(payload: torch.Tensor) -> torch.Tensor:
    """Convert the checkpoint 48-bit-block payload to the LSB3 kernel layout.

    [E, N, K * 3 // 8] uint8 (16 codes per 6-byte big-endian block, code j =
    (v >> (45 - 3j)) & 7) -> same-shape uint8 where each 3-byte group holds 8
    codes LSB-first (code j of group g at bits 3j of bytes 3g..3g+2). Run once
    per weight at load time; a pure byte permutation, so the dequantized
    values are bit-identical and VRAM is unchanged.
    """
    e, n, kb3 = payload.shape
    k = kb3 * 8 // 3
    assert k % 16 == 0, "K must be a multiple of the fp3 group size (16)"
    out = torch.empty_like(payload)
    rows = e * n
    src = payload.reshape(rows, kb3)
    dst = out.reshape(rows, kb3)
    # Chunk rows so the int64 bit-unpacking intermediates stay ~64 MiB.
    chunk = max(1, (1 << 26) // (k * 8))
    for s in range(0, rows, chunk):
        b = src[s : s + chunk].to(torch.int64).reshape(-1, k // 16, 6)
        v = (
            (b[..., 0] << 40)
            | (b[..., 1] << 32)
            | (b[..., 2] << 24)
            | (b[..., 3] << 16)
            | (b[..., 4] << 8)
            | b[..., 5]
        )
        codes = torch.stack([(v >> (45 - 3 * j)) & 7 for j in range(16)], -1)
        c = codes.reshape(-1, k // 8, 8)
        w = sum(c[..., j] << (3 * j) for j in range(8))
        dst[s : s + chunk] = (
            torch.stack([(w >> (8 * t)) & 0xFF for t in range(3)], -1)
            .reshape(-1, kb3)
            .to(torch.uint8)
        )
    return out


@triton.jit
def _fp3_e4m3_scale(sf):
    """Decode raw ModelOpt sf bytes to the (signed) FP8-E4M3 block scale.

    ``sf & 0xFE`` is the e4m3 byte (sign, 4-bit exponent, 3-bit mantissa);
    bit0 is the codebook mirror flag (the quantizer used the negated
    codebook for those blocks), folded in as a sign. Returns FP32.
    """
    b = sf & 0xFE
    e = (b >> 3) & 15
    m = b & 7
    bits = ((e + 120) << 23) | (m << 20)
    normal = bits.to(tl.uint32).to(tl.float32, bitcast=True)
    denormal = m.to(tl.float32) * 0.001953125  # m * 2**-9
    scale = tl.where(e == 0, denormal, normal)
    sign_neg = ((b >> 7) ^ (sf & 1)) & 1
    return tl.where(sign_neg == 0, scale, -scale)


@triton.jit
def _fp3_codes_to_value(codes):
    """Map 3-bit codes onto the signed codebook [-6,-2,-1,0,1,2,3,6]."""
    mag = codes - codes
    mag = tl.where(codes == 1, 2, mag)
    mag = tl.where(codes == 2, 1, mag)
    mag = tl.where(codes == 4, 1, mag)
    mag = tl.where(codes == 5, 2, mag)
    mag = tl.where(codes == 6, 3, mag)
    mag = tl.where((codes == 0) | (codes == 7), 6, mag)
    return tl.where(codes < 4, -mag.to(tl.float32), mag.to(tl.float32))


@triton.jit
def _fp3_w3a16_gemm_kernel(
    a_ptr,
    b_ptr,
    sf_ptr,
    gs_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    GS_GATE_UP,
    # Strides
    stride_am,
    stride_ak,
    stride_be,  # payload row stride, in bytes (= K * 3 // 8)
    stride_bn,
    stride_sfe,
    stride_sfn,  # scale row stride (= K // 16)
    stride_cm,
    stride_cn,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Grouped FP3-W3A16 GEMM: C[token-slot, n] = A[token, :] @ dequant(B[e])^T.

    A is [M, K] bf16 (rows addressed via sorted_token_ids // top_k), B is the
    LSB3 payload [E, N, K*3//8] with unswizzled sf [E, N, K//16]. Blocks
    beyond ``num_tokens_post_padded`` exit early; rows whose sorted slot is a
    padding sentinel are masked out of A loads and C stores.
    """
    BLOCK_K_GROUPS: tl.constexpr = BLOCK_SIZE_K // 16
    BLOCK_K_CODE_GROUPS: tl.constexpr = BLOCK_SIZE_K // 8

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    # 4-byte window per 8-code group (the 4th byte lane is masked off; the
    # payload stores 3 bytes per group).
    bgrp = (tl.arange(0, BLOCK_K_CODE_GROUPS)[:, None] * 3 + tl.arange(0, 4)[None, :])[
        None, :, :
    ]
    bsel = tl.arange(0, 4)[None, None, :] < 3
    group_off = tl.arange(0, BLOCK_K_GROUPS)[None, :]
    b_base = b_ptr + off_experts * stride_be + offs_bn[:, None] * stride_bn
    sf_base = sf_ptr + off_experts * stride_sfe + offs_bn[:, None] * stride_sfn
    shifts = (3 * tl.arange(0, 8))[None, None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kb in range(tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = kb * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)

        # Bytes 3g..3g+2 of the row window hold codes 8g..8g+7 LSB-first;
        # assemble each 24-bit group, then extract its 8 codes.
        p = tl.load(
            b_base[:, None, :] + (k_start * 3 // 8) + bgrp, mask=bsel, other=0
        ).to(tl.uint32)
        p3 = tl.sum(p << (8 * tl.arange(0, 4))[None, None, :], axis=2)  # [BN, BK/8]
        codes = tl.reshape(
            (p3.to(tl.int32)[:, :, None] >> shifts) & 7,
            (BLOCK_SIZE_N, BLOCK_SIZE_K),
        )

        sf = tl.load(sf_base + (k_start // 16) + group_off).to(tl.int32)
        scale = _fp3_e4m3_scale(sf)
        scale = tl.reshape(
            tl.broadcast_to(scale[:, :, None], (BLOCK_SIZE_N, BLOCK_K_GROUPS, 16)),
            (BLOCK_SIZE_N, BLOCK_SIZE_K),
        )

        b_val = (_fp3_codes_to_value(codes) * scale).to(compute_type)
        b = tl.trans(b_val)

        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak

    # Per-expert FP32 global scale. For the w13 GEMM the tile lies entirely in
    # the gate half or the up half (half-width divisible by BLOCK_SIZE_N), so
    # pick the matching scale_2 column.
    if GS_GATE_UP > 0:
        gs_col = tl.where(pid_n * BLOCK_SIZE_N >= N // 2, 1, 0)
        gs = tl.load(gs_ptr + off_experts * GS_GATE_UP + gs_col).to(tl.float32)
    else:
        gs = tl.load(gs_ptr + off_experts).to(tl.float32)
    accumulator *= gs

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _default_config(num_slots: int) -> dict[str, int]:
    # Tuned on sm_120 (RTX PRO 6000) with the Qwen3.8-Flash-Next-FP3 expert
    # shape (K=2560, I=640, E=512, topk=10); BLOCK_SIZE_N must divide the
    # gate/up half width of GEMM1. Decode-size slot counts go to the grouped
    # SIMT GEMV (see fused_experts_fp3_w3a16), so these tiers start above
    # that regime.
    if num_slots <= 8192:
        # Slots per expert stay small; BM16 minimizes padding waste.
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        }
    # Prefill-size batches: each expert owns enough slots to fill BM128.
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    }


def _invoke_fp3_gemm(
    a: torch.Tensor,  # [rows_a, K] bf16
    b: torch.Tensor,  # [E, N, K*3//8] uint8 LSB3 payload
    sf: torch.Tensor,  # [E, N, K//16] uint8
    gs: torch.Tensor,  # [E] or [E, 2] fp32
    c: torch.Tensor,  # [M*topk, N] bf16, addressed by sorted slot ids
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    top_k_a: int,  # slots per A row (topk for GEMM1, 1 for GEMM2)
    mul_routed_weight: bool,
    config: dict[str, int],
):
    _e, n, k3 = b.shape
    k = k3 * 8 // 3
    assert k % 16 == 0, "K must be a multiple of the fp3 group size (16)"
    assert k % config["BLOCK_SIZE_K"] == 0, "K must divide BLOCK_SIZE_K"
    grid = (
        triton.cdiv(sorted_token_ids.shape[0], config["BLOCK_SIZE_M"])
        * triton.cdiv(n, config["BLOCK_SIZE_N"]),
    )
    gs_gate_up = gs.shape[1] if gs.dim() == 2 else 0
    _fp3_w3a16_gemm_kernel[grid](
        a,
        b,
        sf,
        gs,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        sorted_token_ids.shape[0],
        num_valid_tokens,
        gs_gate_up,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        sf.stride(0),
        sf.stride(1),
        c.stride(0),
        c.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k_a,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        compute_type=tl.bfloat16,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


# ---------------------------------------------------------------------------
# Grouped SIMT GEMV decode path (JIT CUDA, see csrc/moe/fp3_w3a16_gemv.cuh)
# ---------------------------------------------------------------------------

_gemv_unavailable = False


def _gemv_blocks(num_slots: int):
    # (rows_per_block, lanes_per_row) per GEMM, tuned on sm_120 with the
    # Qwen3.8-Flash-Next-FP3 expert shape: LN = 8 lanes/row doubles the
    # prefetch horizon vs LN = 16 (measured ~1.6x at 160 slots on GEMM1).
    if num_slots <= 48:
        return (16, 16), (16, 2)
    if num_slots <= 192:
        return (16, 8), (32, 4)
    return (32, 16), (32, 4)


@cache_once
def _jit_gemv_module(gemm: int, kt: int, r: int, ln: int) -> Module:
    cls = "Fp3W3A16GemvG1" if gemm == 1 else "Fp3W3A16GemvG2"
    return load_jit(
        "fp3_w3a16_gemv",
        gemm,
        kt,
        r,
        ln,
        cuda_files=["moe/fp3_w3a16_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::{cls}<{kt}, {r}, {ln}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def _gemv_fused_experts(
    hidden_states: torch.Tensor,  # [M, K] bf16
    w13: torch.Tensor,  # [E, 2I, K*3//8] uint8 LSB3 (gate rows then up rows)
    w13_sf: torch.Tensor,  # [E, 2I, K//16] uint8
    w13_gs: torch.Tensor,  # [E, 2] fp32
    w2: torch.Tensor,  # [E, K, I*3//8] uint8 LSB3
    w2_sf: torch.Tensor,  # [E, K, I//16] uint8
    w2_gs: torch.Tensor,  # [E] fp32
    topk_weights: torch.Tensor,  # [M, topk] fp32
    topk_ids: torch.Tensor,  # [M, topk] int
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Decode-size fused MoE via two grouped SIMT GEMV launches; fp32 math."""
    m, k = hidden_states.shape
    topk = topk_ids.shape[1]
    slots = m * topk
    inter = w2.shape[2] * 8 // 3
    dev = hidden_states.device
    # etab[slot] = local expert id (low 32 bits); slot is also the Y row and
    # (for GEMM1) resolves to token slot // topk inside the kernel.
    etab = topk_ids.reshape(-1).to(torch.int64, copy=False) | (
        torch.arange(slots, dtype=torch.int64, device=dev) << 32
    )
    (r1, ln1), (r2, ln2) = _gemv_blocks(slots)
    y1 = torch.empty((slots, 2 * inter), dtype=torch.float32, device=dev)
    _jit_gemv_module(1, k // 8, r1, ln1).run(
        hidden_states, w13, w13_sf, w13_gs, etab, y1, topk
    )
    act = torch.nn.functional.silu(y1[:, :inter]) * y1[:, inter:]
    y2 = torch.empty((slots, k), dtype=torch.float32, device=dev)
    _jit_gemv_module(2, inter // 8, r2, ln2).run(act, w2, w2_sf, w2_gs, etab, y2)
    out = (y2.view(m, topk, k) * topk_weights.reshape(m, topk, 1)).sum(1)
    if routed_scaling_factor != 1.0:
        out = out * routed_scaling_factor
    return out.to(hidden_states.dtype)


def fused_experts_fp3_w3a16(
    hidden_states: torch.Tensor,  # [M, K] bf16
    w13: torch.Tensor,  # [E, 2I, K*3//8] uint8 LSB3 (gate rows then up rows)
    w13_sf: torch.Tensor,  # [E, 2I, K//16] uint8
    w13_gs: torch.Tensor,  # [E, 2] fp32 (gate, up) or [E] shared
    w2: torch.Tensor,  # [E, K, I*3//8] uint8 LSB3
    w2_sf: torch.Tensor,  # [E, K, I//16] uint8
    w2_gs: torch.Tensor,  # [E] fp32
    topk_weights: torch.Tensor,  # [M, topk] fp32
    topk_ids: torch.Tensor,  # [M, topk] int
    activation: str = "silu",
    routed_scaling_factor: float | None = None,
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    """Full W3A16 fused-MoE forward: GEMM1 -> silu*up -> GEMM2 -> topk sum.

    Dispatches decode-size slot counts to the grouped SIMT GEMV (zero VRAM
    overhead, fp32 math) and larger ones to the Triton tensor-core GEMM.
    """
    global \
        _gemv_unavailable, \
        _cute_failures, \
        _cute_disabled, \
        _w4a4_failures, \
        _w4a4_disabled
    assert hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
    assert activation == "silu", f"fp3_w3a16 supports silu, got {activation}"
    m, k = hidden_states.shape
    e, n13, k3 = w13.shape
    assert k * 3 // 8 == k3, "w13 payload width mismatch"
    topk = topk_ids.shape[1]
    inter = w2.shape[2] * 8 // 3
    slots = m * topk

    # flashinfer SM12x NVFP4 W4A4 grouped tensor-core arm (prefill sizes): the
    # payload repacks losslessly to canonical NVFP4 (cached) and the kernel
    # quantizes activations to fp4. The first call at a new token bucket
    # compiles, so it must not run under CUDA graph capture; any failure
    # falls back to the W3A16 paths below and disables the arm after 3 tries.
    w4a4_min = envs.SGLANG_SM12X_W3A4_PREFILL_MIN_TOKENS.get()
    if (
        not _w4a4_disabled
        and config is None
        and w4a4_min > 0
        and m >= w4a4_min
        and _is_sm12x()
    ):
        try:
            return _w4a4_experts(
                hidden_states,
                w13,
                w13_sf,
                w13_gs,
                w2,
                w2_sf,
                w2_gs,
                topk_weights,
                topk_ids,
                routed_scaling_factor or 1.0,
                fused=envs.SGLANG_SM12X_W3A4_FUSED.get(),
            )
        except Exception as exc:  # noqa: BLE001  (flashinfer optional: never fatal)
            _w4a4_failures += 1
            if _w4a4_failures >= 3:
                _w4a4_disabled = True
                print(
                    f"fp3_w3a16 W4A4 prefill arm disabled after {_w4a4_failures} "
                    f"failures ({exc}); using W3A16 paths"
                )

    # CuTe DSL LSB3 fork: expands the 3-bit codebook payload to NVFP4 nibbles
    # in registers and runs the sm12x W4A16 tile kernels. Fastest path on
    # sm_120/121 up to the measured crossover with the Triton GEMM; any
    # failure (shape unsupported, not compiled before graph capture) falls
    # back to GEMV/Triton for that call.
    if (
        not _cute_disabled
        and config is None
        and 0 < slots <= _CUTE_MAX_SLOTS
        and _is_sm12x()
    ):
        try:
            return _cute_fork_experts(
                hidden_states,
                w13,
                w13_sf,
                w13_gs,
                w2,
                w2_sf,
                w2_gs,
                topk_weights.float(),
                topk_ids.int(),
                routed_scaling_factor or 1.0,
            )
        except Exception as exc:  # noqa: BLE001  (fork optional: never fatal)
            _cute_failures += 1
            if _cute_failures >= 3:
                _cute_disabled = True
                print(
                    f"fp3_w3a16 CuTe DSL fork disabled after {_cute_failures} "
                    f"failures ({exc}); using GEMV/Triton"
                )

    # The quad-load GEMV needs 12-byte-aligned row slices (K/8 % 4 == 0).
    if (
        config is None
        and slots <= _GEMV_MAX_SLOTS
        and w13_gs.dim() == 2
        and (k // 8) % 4 == 0
        and (inter // 8) % 4 == 0
        and not _gemv_unavailable
    ):
        try:
            return _gemv_fused_experts(
                hidden_states,
                w13,
                w13_sf,
                w13_gs,
                w2,
                w2_sf,
                w2_gs,
                topk_weights,
                topk_ids,
                routed_scaling_factor or 1.0,
            )
        except Exception as exc:  # noqa: BLE001  (JIT build/host: stay on Triton)
            _gemv_unavailable = True
            print(f"fp3_w3a16 grouped GEMV unavailable, using Triton: {exc}")

    config = dict(config or _default_config(slots))
    assert (n13 // 2) % config["BLOCK_SIZE_N"] == 0, (
        "gate/up halves must not straddle a GEMM1 output tile"
    )

    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        act_and_mul_triton,
        moe_sum_reduce_triton,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    topk_ids_i32 = topk_ids.to(torch.int32)
    topk_weights_f32 = topk_weights.to(torch.float32).contiguous()
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids_i32, config["BLOCK_SIZE_M"], e
    )
    num_valid = slots
    dev = hidden_states.device

    cache1 = torch.empty((num_valid, n13), dtype=torch.bfloat16, device=dev)
    _invoke_fp3_gemm(
        hidden_states,
        w13,
        w13_sf,
        w13_gs,
        cache1,
        topk_weights_f32,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid,
        top_k_a=topk,
        mul_routed_weight=False,
        config=config,
    )

    cache2 = torch.empty((num_valid, inter), dtype=torch.bfloat16, device=dev)
    act_and_mul_triton(
        cache1,
        cache2,
        config,
        topk_ids=topk_ids_i32,
        activation=activation,
    )

    cache3 = torch.empty((num_valid, k), dtype=torch.bfloat16, device=dev)
    _invoke_fp3_gemm(
        cache2,
        w2,
        w2_sf,
        w2_gs,
        cache3,
        topk_weights_f32,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid,
        top_k_a=1,
        mul_routed_weight=True,
        config=config,
    )

    out = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
    moe_sum_reduce_triton(
        cache3.view(m, topk, k),
        out,
        routed_scaling_factor=routed_scaling_factor or 1.0,
    )
    return out
