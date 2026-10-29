import torch, triton, triton.language as tl

@triton.jit
def _lk_decode_finish_kernel(out_ptr, gout_ptr, out_write_ptr, bs, gout_rows, rows_per_token,
    HIDDEN: tl.constexpr, COVERED: tl.constexpr, NAN: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < bs * HIDDEN
    row = off // HIDDEN
    x = tl.load(out_ptr + off, mask=m, other=0.0)
    if COVERED:
        tok = row // rows_per_token
        head = (row - tok * rows_per_token) == 0  # only the group-head row
        g = tl.load(gout_ptr + tok * HIDDEN + (off - row * HIDDEN),
                    mask=m & (tok < gout_rows) & head, other=0.0)
        x += g
    if NAN:
        x = tl.where(x == x, x, 0.0)  # nan->0; inf kept -> rounds to dtype inf,
        # matching old path: fp32 nan_to_num(nan=0.0) maps inf->3.4e38 -> inf on bf16 cast.
    tl.store(out_write_ptr + off, x.to(out_write_ptr.dtype.element_ty), mask=m)

H, dev = 512, "cuda:0"
ok = 0
for (bs, rpt, covered, nan, dt) in [(4,1,True,False,torch.bfloat16),(8,4,True,True,torch.bfloat16),
                                     (2,1,False,True,torch.bfloat16),(3,1,False,False,torch.float32),
                                     (6,2,True,False,torch.float16),(1,1,True,True,torch.bfloat16),
                                     (4,1,True,True,torch.bfloat16)]:
    rt = bs // rpt
    torch.manual_seed(0)
    out_gpu = torch.randn(bs, H, device=dev, dtype=torch.float32)
    gout = torch.randn(max(rt,1), H, device=dev, dtype=torch.float32)
    hs = torch.randn(bs, H, device=dev, dtype=dt)
    # EXACT old path, in fp32 then cast (as _cpu_decode does)
    ref = out_gpu.clone()
    if covered:
        if rpt == 1:
            ref[:rt] += gout[:rt]
        else:
            for tok in range(rt):
                ref[tok*rpt:tok*rpt+1] += gout[tok:tok+1]
    if nan:
        torch.nan_to_num(ref, nan=0.0, out=ref)
    ref = ref[:bs].to(dt)
    o = torch.empty_like(hs)
    _lk_decode_finish_kernel[(triton.cdiv(bs*H,1024),)](out_gpu, gout if covered else hs, o, bs, rt, rpt,
        HIDDEN=H, COVERED=covered, NAN=nan, BLOCK=1024)
    assert torch.equal(o, ref), f"MISMATCH bs={bs} rpt={rpt} cov={covered} nan={nan} {dt} diffs={int((o!=ref).sum())}"
    ok += 1
print(f"ALL {ok} CASES BITWISE-OK")

import time
for (bs, rpt, covered) in [(1,1,True),(2,1,True),(8,4,True)]:
    rt = bs//rpt
    out_gpu = torch.randn(bs, H, device=dev, dtype=torch.float32)
    gout = torch.randn(max(rt,1), H, device=dev, dtype=torch.float32)
    hs = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    def old():
        if covered:
            if rpt==1: out_gpu[:rt].add_(gout[:rt])
            else:
                for tok in range(rt): out_gpu[tok*rpt:tok*rpt+1].add_(gout[tok:tok+1])
        out_gpu[:bs].to(torch.bfloat16)
    o = torch.empty_like(hs)
    def new():
        _lk_decode_finish_kernel[(triton.cdiv(bs*H,1024),)](out_gpu, gout, o, bs, rt, rpt,
            HIDDEN=H, COVERED=covered, NAN=False, BLOCK=1024)
    for fn,name in [(old,"old"),(new,"fused")]:
        for _ in range(20): fn()
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(200): fn()
        torch.cuda.synchronize()
        print(f"bs={bs} rpt={rpt} {name}: {(time.perf_counter()-t)/200*1e6:.1f} us")
