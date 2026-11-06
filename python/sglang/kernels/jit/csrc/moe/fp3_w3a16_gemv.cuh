#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {

// Grouped decode GEMV for the FP3 (W3A16) fused-MoE runner: one warp per
// output row, one launch covers every active expert. The weight payload is
// the checkpoint's 3-bit codes repacked LSB-first, 8 codes per 3 bytes
// (shape E x N x K*3/8, zero VRAM growth); block scales stay raw uint8 e4m3
// (E x N x K/16) decoded inline. fp32 math end to end.
//
// etab[slot] (int64): low 32 bits = local expert id. Blocks are expert-major:
// slot = blockIdx.x / nbpe, row = (blockIdx.x % nbpe) * R + warp.

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

// GATE_UP: X is bf16 [T, K] rows gathered by slot / topk (GEMM1); GS is
// [E, 2] and the gate/up half of N picks the column. Else X is fp32 [S, K]
// indexed by slot directly (GEMM2, rows = slots) and GS is [E].
template <int KT, int R, bool GATE_UP>
__device__ __forceinline__ void fp3_gemv_body(
    const void* __restrict__ X,  // bf16 or fp32 rows
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ SF,
    const float* __restrict__ GS,
    const int64_t* __restrict__ ETAB,
    float* __restrict__ Y,
    int nbpe,
    int N,
    int NHALF,
    int topk) {
  __shared__ float lut[8];
  if (threadIdx.x < 8) {
    const float c[8] = {-6.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 6.f};
    lut[threadIdx.x] = c[threadIdx.x];
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int K3 = KT * 3;  // payload bytes per weight row
  const int KB = KT / 2;  // scale bytes per weight row
  const int nblk = blockIdx.x;
  const int slot = nblk / nbpe;
  const int e = static_cast<int>(ETAB[slot] & 0xffffffffll);
  const int tok = GATE_UP ? slot / topk : slot;
  const int row = (nblk % nbpe) * R + warp;
  const uint8_t* wr = W + (static_cast<size_t>(e) * N + row) * K3;
  const uint8_t* sfr = SF + (static_cast<size_t>(e) * N + row) * KB;
  float acc = 0.f;
  if (GATE_UP) {
    const __nv_bfloat16* xk =
        static_cast<const __nv_bfloat16*>(X) + static_cast<size_t>(tok) * KT * 8;
    for (int g = lane; g < KT; g += 32) {
      const uint8_t* p3 = wr + g * 3;
      uint32_t a = static_cast<uint32_t>(p3[0]) | (static_cast<uint32_t>(p3[1]) << 8)
                   | (static_cast<uint32_t>(p3[2]) << 16);
      float p = lut[a & 7u] * __bfloat162float(xk[g * 8]);
#pragma unroll
      for (int j = 1; j < 8; ++j)
        p += lut[(a >> (3 * j)) & 7u] * __bfloat162float(xk[g * 8 + j]);
      acc = __fmaf_rn(p, fp3_dec_sf(sfr[g >> 1]), acc);
    }
  } else {
    const float* xk = static_cast<const float*>(X) + static_cast<size_t>(tok) * KT * 8;
    for (int g = lane; g < KT; g += 32) {
      const uint8_t* p3 = wr + g * 3;
      uint32_t a = static_cast<uint32_t>(p3[0]) | (static_cast<uint32_t>(p3[1]) << 8)
                   | (static_cast<uint32_t>(p3[2]) << 16);
      float p = lut[a & 7u] * xk[g * 8];
#pragma unroll
      for (int j = 1; j < 8; ++j) p += lut[(a >> (3 * j)) & 7u] * xk[g * 8 + j];
      acc = __fmaf_rn(p, fp3_dec_sf(sfr[g >> 1]), acc);
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    float gs = GATE_UP ? GS[e * 2 + (row < NHALF ? 0 : 1)] : GS[e];
    Y[static_cast<size_t>(slot) * N + row] = acc * gs;
  }
}

template <int KT, int R, bool GATE_UP>
__global__ void __launch_bounds__(32 * R) fp3_w3a16_gemv_kernel(
    const void* __restrict__ X,
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ SF,
    const float* __restrict__ GS,
    const int64_t* __restrict__ ETAB,
    float* __restrict__ Y,
    int nbpe,
    int N,
    int NHALF,
    int topk) {
  fp3_gemv_body<KT, R, GATE_UP>(X, W, SF, GS, ETAB, Y, nbpe, N, NHALF, topk);
}

// GEMM1 (gate/up): y1[s, 0:2I] = hidden[slot / topk] @ dequant(w13[e])^T.
// x bf16 [T, K], w [E, 2I, K*3/8] u8, wsf [E, 2I, K/16] u8, gs [E, 2] fp32,
// etab [S] int64 (low 32 = expert), out fp32 [S, 2I].
template <int KT, int R>
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
    CHECK_HOST(n % R == 0 && n % 2 == 0) << "fp3 gemv G1: bad N";
    TensorMatcher({-1, static_cast<int64_t>(KT) * 8})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(x);
    TensorMatcher({-1, n, KT * 3}).with_dtype<uint8_t>().with_device(device).verify(w);
    TensorMatcher({-1, n, KT / 2}).with_dtype<uint8_t>().with_device(device).verify(wsf);
    TensorMatcher({-1, 2}).with_dtype<fp32_t>().with_device(device).verify(gs);
    TensorMatcher({-1, n}).with_dtype<fp32_t>().with_device(device).verify(out);
    dim3 grid(static_cast<unsigned>(etab.shape()[0] * (n / R)), 1, 1);
    LaunchKernel(grid, 32 * R, device.unwrap())(
        fp3_w3a16_gemv_kernel<KT, R, true>,
        x.data_ptr(),
        static_cast<const uint8_t*>(w.data_ptr()),
        static_cast<const uint8_t*>(wsf.data_ptr()),
        static_cast<const float*>(gs.data_ptr()),
        static_cast<const int64_t*>(etab.data_ptr()),
        static_cast<float*>(out.data_ptr()),
        static_cast<int>(n / R),
        static_cast<int>(n),
        static_cast<int>(n / 2),
        static_cast<int>(topk));
  }
};

// GEMM2 (down): y2[s, :] = act[s] @ dequant(w2[e])^T. act fp32 [S, I],
// w [E, K, I*3/8], wsf [E, K, I/16], gs [E] fp32, out fp32 [S, K].
template <int KT, int R>
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
    CHECK_HOST(n % R == 0) << "fp3 gemv G2: bad N";
    TensorMatcher({-1, static_cast<int64_t>(KT) * 8})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(act);
    TensorMatcher({-1, n, KT * 3}).with_dtype<uint8_t>().with_device(device).verify(w);
    TensorMatcher({-1, n, KT / 2}).with_dtype<uint8_t>().with_device(device).verify(wsf);
    TensorMatcher({-1}).with_dtype<fp32_t>().with_device(device).verify(gs);
    TensorMatcher({-1, n}).with_dtype<fp32_t>().with_device(device).verify(out);
    dim3 grid(static_cast<unsigned>(etab.shape()[0] * (n / R)), 1, 1);
    LaunchKernel(grid, 32 * R, device.unwrap())(
        fp3_w3a16_gemv_kernel<KT, R, false>,
        act.data_ptr(),
        static_cast<const uint8_t*>(w.data_ptr()),
        static_cast<const uint8_t*>(wsf.data_ptr()),
        static_cast<const float*>(gs.data_ptr()),
        static_cast<const int64_t*>(etab.data_ptr()),
        static_cast<float*>(out.data_ptr()),
        static_cast<int>(n / R),
        static_cast<int>(n),
        0,
        0);
  }
};

}  // namespace sglang
