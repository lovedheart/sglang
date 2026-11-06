#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {

// Grouped decode GEMV for the FP3 (W3A16) fused-MoE runner: one launch
// covers every active expert; a full warp cooperatively handles LN output
// rows (LN lanes per row, so small-K GEMMs stay conflict-free). The weight
// payload is the checkpoint's 3-bit codes repacked LSB-first, 8 codes per
// 3 bytes (E x N x K*3/8 bytes, zero VRAM growth); block scales stay raw
// uint8 e4m3 (E x N x K/16) decoded inline. fp32 math end to end.
//
// Each lane owns whole 32-code quads (12 payload bytes, 4-byte aligned):
// three __ldg words + PRMT splice rebuild the 8 3-bit groups per quad with
// zero predication. The activation row is staged into shared memory
// TRANSPOSED (element-in-quad fastest in the lane dim); a quad-major layout
// makes every lane read the same banks (16-way LDS replay, measured 2x of
// the whole kernel).
//
// etab[slot] (int64): low 32 bits = local expert id, high 32 = output row.
// Blocks are expert-major: slot = blockIdx.x / nbpe, row =
// (blockIdx.x % nbpe) * RB + (warp * RPW + sub).

/// Decode a raw ModelOpt fp3 scale byte: (sf & 0xFE) is the e4m3 byte (sign,
/// exponent bits 6..3, mantissa bits 2..0); bit 0 is the codebook mirror flag
/// folded in as a sign.
__device__ __forceinline__ float fp3_dec_sf(uint8_t sf) {
  uint32_t b = sf & 0xFEu;
  uint32_t e = (b >> 3) & 15u;
  float v = __uint_as_float(((e + 120u) << 23) | ((b & 7u) << 20));
  float den = static_cast<float>(b & 7u) * 0.001953125f;  // m * 2^-9
  v = (e == 0) ? den : v;
  return (((b >> 7) ^ (sf & 1u)) & 1u) ? -v : v;
}

// GATE_UP: X is bf16 [T, K] rows gathered by row / topk (GEMM1); GS is
// [E, 2] and the gate/up half of N picks the column. Else X is fp32 [S, K]
// indexed by row directly (GEMM2) and GS is [E].
// KT = K / 8 (code groups), RB = rows per block, LN = lanes per row,
// XR = activation element type (__nv_bfloat16 for GEMM1, float for GEMM2).
template <int KT, int RB, int LN, bool GATE_UP, class XR>
__device__ __forceinline__ void fp3_gemv_body(
    const XR* __restrict__ X,  // bf16 (GEMM1) or fp32 (GEMM2) rows
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ SF,
    const float* __restrict__ GS,
    const int64_t* __restrict__ ETAB,
    float* __restrict__ Y,
    int nbpe,
    int N,
    int NHALF,
    int topk) {
  constexpr int NQ = KT / 4;  // 32-code quads per row
  static_assert(KT % 4 == 0 && KT * 3 % 4 == 0, "quad-aligned KT");
  static_assert(LN == 2 || LN == 4 || LN == 8 || LN == 16 || LN == 32, "bad LN");
  __shared__ float lut[32];  // codebook CB[c] at index 4c: distinct banks
  __shared__ float xst[KT * 8];  // transposed activation row (single slot)
  if (threadIdx.x < 8) {
    const float c[8] = {-6.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 6.f};
    lut[threadIdx.x * 4] = c[threadIdx.x];
  }
  __syncthreads();
  const int sub = (threadIdx.x & 31) / LN, lown = (threadIdx.x & 31) % LN;
  const int RPW = 32 / LN;  // rows covered by one warp
  const int slot = blockIdx.x / nbpe;
  const int64_t et = ETAB[slot];
  const int e = static_cast<int>(et & 0xffffffffll);
  const int yrow = static_cast<int>(et >> 32);
  const int tok = GATE_UP ? yrow / topk : yrow;
  const int row = (blockIdx.x % nbpe) * RB + (threadIdx.x >> 5) * RPW + sub;
  const uint8_t* wr = W + (static_cast<size_t>(e) * N + row) * KT * 3;
  const uint8_t* sfr = SF + (static_cast<size_t>(e) * N + row) * (KT / 2);
  {  // cooperative X stage, transposed: code k = q*32+e -> xst[e*NQ+q]
    // Each iteration stages 4 codes: one float4 (fp32) or two bf162 pairs.
#pragma unroll
    for (int i = threadIdx.x; i < KT * 2; i += blockDim.x) {
      float4 v4;
      if constexpr (GATE_UP) {
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(X) +
                                   static_cast<size_t>(tok) * KT * 4 + i * 2;
        const __nv_bfloat162 lo = __ldg(xb), hi = __ldg(xb + 1);
        const float2 f01 = __bfloat1622float2(lo), f23 = __bfloat1622float2(hi);
        v4 = make_float4(f01.x, f01.y, f23.x, f23.y);
      } else {
        v4 = __ldg(reinterpret_cast<const float4*>(X) + static_cast<size_t>(tok) * KT * 2 + i);
      }
      const int k0 = i * 4;
      const int q0 = k0 >> 5, e0 = k0 & 31;
      xst[(e0 + 0) * NQ + q0] = v4.x;
      xst[(e0 + 1) * NQ + q0] = v4.y;
      xst[(e0 + 2) * NQ + q0] = v4.z;
      xst[(e0 + 3) * NQ + q0] = v4.w;
    }
  }
  __syncthreads();
  float acc = 0.f;
  uint32_t w0 = 0, w1 = 0, w2 = 0;
  {  // prime the prefetch buffer with quad `lown`
    const uint32_t* wp0 = reinterpret_cast<const uint32_t*>(wr) + static_cast<size_t>(lown) * 3;
    w0 = __ldg(wp0);
    w1 = __ldg(wp0 + 1);
    w2 = __ldg(wp0 + 2);
  }
  for (int q = lown; q < NQ; q += LN) {
    const int qn = q + LN;
    uint32_t n0 = 0, n1 = 0, n2 = 0;
    if (qn < NQ) {  // issue quad q+LN before consuming quad q
      const uint32_t* wpn = reinterpret_cast<const uint32_t*>(wr) + static_cast<size_t>(qn) * 3;
      n0 = __ldg(wpn);
      n1 = __ldg(wpn + 1);
      n2 = __ldg(wpn + 2);
    }
    const uint32_t g0 = w0 & 0xFFFFFFu;                          // codes 0-7
    const uint32_t g1 = __byte_perm(w0, w1, 0x7543) & 0xFFFFFFu;  // codes 8-15
    const uint32_t g2 = __byte_perm(w1, w2, 0x0432) & 0xFFFFFFu;  // codes 16-23
    const uint32_t g3 = w2 >> 8;                                 // codes 24-31
    w0 = n0;
    w1 = n1;
    w2 = n2;
    const uint32_t sf2 =
        static_cast<uint32_t>(__ldg(reinterpret_cast<const uint16_t*>(sfr) + q));
    float pa = 0.f, pb = 0.f;
#define SGL_FP3_CBH(g, j, dst, off)                                  \
  do {                                                               \
    const uint32_t s_ = (g) >> (6 * (j));                            \
    dst += lut[(s_ & 7u) * 4] * xst[((off) + (j) * 2) * NQ + q];     \
    dst += lut[((s_ >> 3) & 7u) * 4] * xst[((off) + (j) * 2 + 1) * NQ + q]; \
  } while (0)
    SGL_FP3_CBH(g0, 0, pa, 0);
    SGL_FP3_CBH(g0, 1, pa, 0);
    SGL_FP3_CBH(g0, 2, pa, 0);
    SGL_FP3_CBH(g0, 3, pa, 0);
    SGL_FP3_CBH(g1, 0, pa, 8);
    SGL_FP3_CBH(g1, 1, pa, 8);
    SGL_FP3_CBH(g1, 2, pa, 8);
    SGL_FP3_CBH(g1, 3, pa, 8);
    SGL_FP3_CBH(g2, 0, pb, 16);
    SGL_FP3_CBH(g2, 1, pb, 16);
    SGL_FP3_CBH(g2, 2, pb, 16);
    SGL_FP3_CBH(g2, 3, pb, 16);
    SGL_FP3_CBH(g3, 0, pb, 24);
    SGL_FP3_CBH(g3, 1, pb, 24);
    SGL_FP3_CBH(g3, 2, pb, 24);
    SGL_FP3_CBH(g3, 3, pb, 24);
#undef SGL_FP3_CBH
    acc = __fmaf_rn(pa, fp3_dec_sf(static_cast<uint8_t>(sf2 & 0xFFu)), acc);
    acc = __fmaf_rn(pb, fp3_dec_sf(static_cast<uint8_t>(sf2 >> 8)), acc);
  }
#pragma unroll
  for (int o = LN / 2; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o, LN);
  if (lown == 0) {
    float gs = GATE_UP ? GS[e * 2 + (row < NHALF ? 0 : 1)] : GS[e];
    Y[static_cast<size_t>(yrow) * N + row] = acc * gs;
  }
}

template <int KT, int RB, int LN, bool GATE_UP, class XR>
__global__ void __launch_bounds__(RB * LN) fp3_w3a16_gemv_kernel(
    const XR* __restrict__ X,
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ SF,
    const float* __restrict__ GS,
    const int64_t* __restrict__ ETAB,
    float* __restrict__ Y,
    int nbpe,
    int N,
    int NHALF,
    int topk) {
  fp3_gemv_body<KT, RB, LN, GATE_UP, XR>(X, W, SF, GS, ETAB, Y, nbpe, N, NHALF, topk);
}

// GEMM1 (gate/up): y1[s, 0:2I] = hidden[row / topk] @ dequant(w13[e])^T.
// x bf16 [T, K], w [E, 2I, K*3/8] u8, wsf [E, 2I, K/16] u8, gs [E, 2] fp32,
// etab [S] int64 (low 32 = expert, high 32 = output row), out fp32 [S, 2I].
template <int KT, int RB, int LN>
struct Fp3W3A16GemvG1 {
  static void run(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView wsf,
      const tvm::ffi::TensorView gs,
      const tvm::ffi::TensorView etab,
      const tvm::ffi::TensorView out,
      int64_t topk) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    const int64_t n = w.shape()[1];
    CHECK_HOST(w.shape()[2] == KT * 3 && wsf.shape()[1] == n && wsf.shape()[2] == KT / 2)
        << "fp3 gemv G1: payload/scale layout mismatch";
    CHECK_HOST(n % RB == 0 && n % 2 == 0) << "fp3 gemv G1: bad N";
    TensorMatcher({-1, static_cast<int64_t>(KT) * 8})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(x);
    TensorMatcher({-1, n, KT * 3}).with_dtype<uint8_t>().with_device(device).verify(w);
    TensorMatcher({-1, n, KT / 2}).with_dtype<uint8_t>().with_device(device).verify(wsf);
    TensorMatcher({-1, 2}).with_dtype<fp32_t>().with_device(device).verify(gs);
    TensorMatcher({-1, n}).with_dtype<fp32_t>().with_device(device).verify(out);
    dim3 grid(static_cast<unsigned>(etab.shape()[0] * (n / RB)), 1, 1);
    LaunchKernel(grid, RB * LN, device.unwrap())(
        fp3_w3a16_gemv_kernel<KT, RB, LN, true, __nv_bfloat16>,
        static_cast<const __nv_bfloat16*>(x.data_ptr()),
        static_cast<const uint8_t*>(w.data_ptr()),
        static_cast<const uint8_t*>(wsf.data_ptr()),
        static_cast<const float*>(gs.data_ptr()),
        static_cast<const int64_t*>(etab.data_ptr()),
        static_cast<float*>(out.data_ptr()),
        static_cast<int>(n / RB),
        static_cast<int>(n),
        static_cast<int>(n / 2),
        static_cast<int>(topk));
  }
};

// GEMM2 (down): y2[s, :] = act[s] @ dequant(w2[e])^T. act fp32 [S, I],
// w [E, K, I*3/8], wsf [E, K, I/16], gs [E] fp32, out fp32 [S, K].
template <int KT, int RB, int LN>
struct Fp3W3A16GemvG2 {
  static void run(
      const tvm::ffi::TensorView act,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView wsf,
      const tvm::ffi::TensorView gs,
      const tvm::ffi::TensorView etab,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    const int64_t n = w.shape()[1];
    CHECK_HOST(w.shape()[2] == KT * 3 && wsf.shape()[1] == n && wsf.shape()[2] == KT / 2)
        << "fp3 gemv G2: payload/scale layout mismatch";
    CHECK_HOST(n % RB == 0) << "fp3 gemv G2: bad N";
    TensorMatcher({-1, static_cast<int64_t>(KT) * 8})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(act);
    TensorMatcher({-1, n, KT * 3}).with_dtype<uint8_t>().with_device(device).verify(w);
    TensorMatcher({-1, n, KT / 2}).with_dtype<uint8_t>().with_device(device).verify(wsf);
    TensorMatcher({-1}).with_dtype<fp32_t>().with_device(device).verify(gs);
    TensorMatcher({-1, n}).with_dtype<fp32_t>().with_device(device).verify(out);
    dim3 grid(static_cast<unsigned>(etab.shape()[0] * (n / RB)), 1, 1);
    LaunchKernel(grid, RB * LN, device.unwrap())(
        fp3_w3a16_gemv_kernel<KT, RB, LN, false, float>,
        static_cast<const float*>(act.data_ptr()),
        static_cast<const uint8_t*>(w.data_ptr()),
        static_cast<const uint8_t*>(wsf.data_ptr()),
        static_cast<const float*>(gs.data_ptr()),
        static_cast<const int64_t*>(etab.data_ptr()),
        static_cast<float*>(out.data_ptr()),
        static_cast<int>(n / RB),
        static_cast<int>(n),
        0,
        0);
  }
};

}  // namespace sglang
