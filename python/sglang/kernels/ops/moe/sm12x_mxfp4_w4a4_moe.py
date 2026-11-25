# Fork note: this file mirrors gemm_dev/sm12x_mxfp4/mxfp4_w4a4_moe.py byte-
# identical except this header block (source of truth stays in gemm_dev).
# MXFP4 W4A4 grouped MoE on the native b12x MmaMXF4 dense_gemm (3D grouped
# contract: expert-block-major A/B, swizzled E8M0 scales, alpha=None).
from __future__ import annotations

import torch
import triton
import triton.language as tl
from b12x._lib.dense_gemm import dense_gemm
from b12x._lib.intrinsics import swizzle_block_scale

try:
    from mxfp4_act_quant import quant_act_mxfp4
except ImportError:  # sglang mirror (sm12x_mxfp4_w4a4_moe.py)
    from sglang.kernels.ops.moe.sm12x_mxfp4_act_quant import quant_act_mxfp4

LADDER = (128, 256, 512, 1024)


@triton.jit
def _gather_rows_kernel(x_ptr, rowmap, out, K, stride_x, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # local row l*M + r
    src = tl.load(rowmap + row)
    for off in range(0, K, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        m = cols < K
        if src < 0:
            tl.store(out + row * K + cols, tl.zeros((BLOCK,), tl.float32), mask=m)
        else:
            v = tl.load(x_ptr + src.to(tl.int64) * stride_x + cols, mask=m, other=0.0)
            tl.store(out + row * K + cols, v, mask=m)


@triton.jit
def _silu_mul_quant_kernel(c_ptr, out, sf, I, IP, IPD, C, CP, BLOCK_I: tl.constexpr):
    # c rows are LOCAL (L*m, 2I) fully written by GEMM1; rows padded to the
    # next 128 stay numerically finite (zeros). Global row r -> block rb=r//128.
    row = tl.program_id(0).to(tl.int64)
    rb, rr = row // 128, row % 128
    for kb in range(0, tl.cdiv(I, BLOCK_I)):
        c0 = kb * BLOCK_I // 32
        cols = c0 + tl.arange(0, BLOCK_I // 32)
        offs = kb * BLOCK_I + tl.arange(0, BLOCK_I)
        mask = offs < I
        g = tl.load(c_ptr + row * 2 * I + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(c_ptr + row * 2 * I + I + offs, mask=mask, other=0.0).to(tl.float32)
        act = (g / (1.0 + tl.extra.libdevice.exp(-g))) * u
        grp = tl.reshape(act, (BLOCK_I // 32, 32))
        amax = tl.max(tl.abs(grp), axis=1)
        e = tl.extra.libdevice.rint(
            tl.extra.libdevice.log2(tl.maximum(amax, 1e-30) / 6.0)
        )
        e = tl.clamp(e, -126.0, 126.0)
        s = tl.extra.libdevice.exp2(e)
        q = grp / s[:, None]
        aq = tl.abs(q)
        mag = (
            (aq > 0.25).to(tl.uint8)
            + (aq > 0.75).to(tl.uint8)
            + (aq > 1.25).to(tl.uint8)
            + (aq > 1.75).to(tl.uint8)
            + (aq > 2.5).to(tl.uint8)
            + (aq > 3.5).to(tl.uint8)
            + (aq > 5.0).to(tl.uint8)
        )
        code = mag | ((q < 0).to(tl.uint8) << 3)
        lo, hi = tl.split(tl.reshape(code, (BLOCK_I // 2, 2)))
        po = kb * (BLOCK_I // 2) + tl.arange(0, BLOCK_I // 2)
        tl.store(out + row * IPD + po, (lo | (hi << 4)).to(tl.uint8), mask=po < IP)
        addr = (
            ((rb * (CP // 4) + cols // 4) * 32 + rr % 32) * 16
            + (rr // 32) * 4
            + cols % 4
        )
        tl.store(sf + addr, (e + 127.0).to(tl.uint8), mask=cols < C)


@triton.jit
def _finalize_kernel(c_ptr, rowmap, w_ptr, tok_ptr, ws, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(rowmap + row)
    if slot < 0:
        return
    tok = tl.load(tok_ptr + slot)
    w = tl.load(w_ptr + slot)
    for off in range(0, H, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(c_ptr + row.to(tl.int64) * H + cols, mask=cols < H, other=0.0).to(
            tl.float32
        )
        tl.atomic_add(ws + tok.to(tl.int64) * H + cols, x * w, mask=cols < H)


class Mxfp4W4A4Runner:
    def __init__(self, num_experts, w13, w13_sf, w2, w2_sf):
        # packed uint8 (E, N, K/2); sf row-major uint8 (E, N, K/32)
        self.E = num_experts
        self.H = w2.shape[1]
        self.I = w2.shape[2] * 2
        # MXFP4 can_implement requires logical K % (sf_vec*8=256); zero-pad the
        # GEMM2 K axis to the next multiple. Extra columns: code 0 x weight 0.
        Ipad = (self.I + 255) // 256 * 256
        self.Ipad = Ipad
        pad = (Ipad - self.I) // 2
        if pad:
            w2 = torch.cat([w2, w2.new_zeros(w2.shape[:-1] + (pad,))], -1)
            w2_sf = torch.cat(
                [w2_sf, w2_sf.new_zeros(w2_sf.shape[:-1] + ((Ipad - self.I) // 32,))],
                -1,
            )
        self.w13, self.w2 = w13, w2
        self.w13_sf_sw = swizzle_block_scale(w13_sf)
        self.w2_sf_sw = swizzle_block_scale(w2_sf)
        self.dev = w13.device

    def _gemm(self, A, S, W, Wsf, Cbuf, m, klog):
        dense_gemm(
            (A.permute(1, 2, 0), S),
            (W.permute(1, 2, 0), Wsf),
            Cbuf.permute(1, 2, 0),
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            expected_m=m,
            mma_tiler_mn=(128, 128),
        )

    def plan(self, topk_ids, topk_weights):
        """Data-dependent routing plan (host syncs). Call OUTSIDE graph capture;
        ``execute`` is capture-safe for a fixed plan."""
        tk = topk_ids.shape[1]
        eids = topk_ids.reshape(-1).to(torch.int64)
        order = torch.argsort(eids, stable=True)
        sids = eids[order]
        cnt = torch.bincount(sids, minlength=self.E)
        off = torch.zeros(self.E + 1, dtype=torch.int64, device=self.dev)
        off[1:] = torch.cumsum(cnt, 0)
        m = next(v for v in LADDER if v >= int(cnt.max()))
        tok = (order // tk).to(torch.int32)  # token per slot
        w = topk_weights.reshape(-1)[order].to(torch.float32)

        buckets = []
        e0 = 0
        while e0 < self.E:
            if cnt[e0].item() == 0:
                e0 += 1
                continue
            e1 = e0 + 1
            while (
                e1 < self.E
                and cnt[e1].item()
                and cnt[e1].item() <= m
                and (e1 - e0) < 64
            ):
                e1 += 1
            L = e1 - e0
            cs = cnt[e0:e1]
            lrow = torch.arange(L * m, device=self.dev, dtype=torch.int64)
            l, r = lrow // m, lrow % m
            valid = r < cs[l]
            rowmap = torch.where(valid, off[e0:e1][l] + r, torch.full_like(l, -1)).to(
                torch.int32
            )
            buckets.append((e0, L, m, rowmap))
            e0 = e1
        return tok, w, buckets

    def execute(self, hidden, topk_ids, topk_weights, tok, w, buckets, out=None):
        """Capture-safe given a fixed plan: constant shapes, no host syncs."""
        T, H = hidden.shape
        I, Ipad = self.I, self.Ipad
        xsorted = hidden[tok.long()]
        results = []
        for e0, L, m, rowmap in buckets:
            Xg = torch.empty((L * m, H), dtype=hidden.dtype, device=self.dev)
            _gather_rows_kernel[(L * m,)](
                xsorted, rowmap, Xg, H, xsorted.stride(0), BLOCK=1024, num_warps=4
            )
            A1, S1 = quant_act_mxfp4(Xg)
            C1buf = torch.empty((L, m, 2 * I), dtype=torch.bfloat16, device=self.dev)
            self._gemm(
                A1.view(L, m, H // 2),
                S1.view(L, m, -1),
                self.w13[e0 : e0 + L],
                self.w13_sf_sw[e0 : e0 + L],
                C1buf,
                m,
                H,
            )

            IPD, C2, CP2 = Ipad // 2, I // 32, Ipad // 32
            A2 = torch.zeros((L * m, IPD), dtype=torch.uint8, device=self.dev)
            S2 = torch.full((L * m, CP2), 127, dtype=torch.uint8, device=self.dev)
            _silu_mul_quant_kernel[(L * m,)](
                C1buf,
                A2,
                S2,
                I,
                I // 2,
                IPD,
                C2,
                CP2,
                BLOCK_I=min(triton.next_power_of_2(I), 4096),
                num_warps=4,
            )
            C2buf = torch.empty((L, m, H), dtype=torch.bfloat16, device=self.dev)
            self._gemm(
                A2.view(L, m, IPD),
                S2.view(L, m, CP2),
                self.w2[e0 : e0 + L],
                self.w2_sf_sw[e0 : e0 + L],
                C2buf,
                m,
                Ipad,
            )
            results.append((C2buf, rowmap))

        ws = torch.zeros((T, H), dtype=torch.float32, device=self.dev)
        for C2buf, rowmap in results:
            _finalize_kernel[(C2buf.shape[0] * C2buf.shape[1],)](
                C2buf, rowmap, w, tok, ws, H, BLOCK=1024, num_warps=4
            )
        return ws.to(torch.bfloat16) if out is None else out.copy_(ws)

    def run(self, hidden, topk_ids, topk_weights, out=None):
        tok, w, buckets = self.plan(topk_ids, topk_weights)
        return self.execute(hidden, topk_ids, topk_weights, tok, w, buckets, out)
