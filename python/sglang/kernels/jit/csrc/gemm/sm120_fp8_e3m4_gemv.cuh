#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

// M=1 decode GEMV for e3m4 weight codes with 128x128 block scales, e4m3
// per-128-group activations:  y[n] = sum_k dec34(w[n,k]) * dec43(x[k]) * a_s[k/128] * w_s[n/128,k/128]
// (see gemm_dev/cutlass/gemv15.cu; one warp per output row, uint4 streaming.)

__device__ __forceinline__ float dec34(uint32_t b) {  // fp8 e3m4: s eee mmmm, bias 3
  uint32_t e = (b >> 4) & 7u, m = b & 15u;
  float v = __uint_as_float(((e + 124u) << 23) | (m << 19));
  float den = static_cast<float>(m) * 0.015625f;
  v = (e == 0) ? den : v;
  return (b & 0x80u) ? -v : v;
}

__device__ __forceinline__ float dec43(uint32_t b) {  // fp8 e4m3: s eeee mmm, bias 7
  uint32_t e = (b >> 3) & 15u, m = b & 7u;
  float v = __uint_as_float(((e + 120u) << 23) | (m << 20));
  float den = static_cast<float>(m) * 0.001953125f;
  v = (e == 0) ? den : v;
  return (b & 0x80u) ? -v : v;
}

template <int K>
__global__ void __launch_bounds__(256) sm120_fp8_e3m4_gemv_kernel(
    bf16_t* __restrict__ out,
    const uint8_t* __restrict__ x,   // [K] e4m3 codes
    const uint8_t* __restrict__ w,   // [N,K] e3m4 codes
    const float* __restrict__ a_s,   // [K/128]
    const float* __restrict__ w_s,   // [N/128, K/128]
    int N) {
  constexpr int kBlk = K / 128;
  static_assert(K % 512 == 0, "K must be a multiple of 512 (4 scale blocks per pass)");
  const int lane = threadIdx.x & 31;
  const int sub = lane >> 3;  // which 128-block (of the 4) this lane covers
  const int n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (n >= N) return;
  const uint8_t* pw = w + static_cast<size_t>(n) * K;
  float acc = 0.f;
  uint4 cur = *reinterpret_cast<const uint4*>(pw + (lane << 4));
  uint4 xv = *reinterpret_cast<const uint4*>(x + (lane << 4));
#pragma unroll 2
  for (int kb = 0; kb < kBlk; kb += 4) {
    uint4 nxt;
    if (kb + 4 < kBlk)
      nxt = *reinterpret_cast<const uint4*>(pw + ((kb + 4) << 7) + (lane << 4));
    else
      nxt = cur;
    const uint32_t wb[4] = {cur.x, cur.y, cur.z, cur.w};
    const uint32_t xb[4] = {xv.x, xv.y, xv.z, xv.w};
    float d0 = 0.f, d1 = 0.f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      d0 = fmaf(dec34(wb[q] & 255u), dec43(xb[q] & 255u), d0);
      d1 = fmaf(dec34((wb[q] >> 16) & 255u), dec43((xb[q] >> 16) & 255u), d1);
      d0 = fmaf(dec34((wb[q] >> 8) & 255u), dec43((xb[q] >> 8) & 255u), d0);
      d1 = fmaf(dec34(wb[q] >> 24), dec43(xb[q] >> 24), d1);
    }
    const int kbs = kb + sub;
    const float s = __ldg(a_s + kbs) * __ldg(w_s + static_cast<size_t>(n >> 7) * kBlk + kbs);
    acc = fmaf(d0 + d1, s, acc);
    cur = nxt;
    if (kb + 4 < kBlk)
      xv = *reinterpret_cast<const uint4*>(x + ((kb + 4) << 7) + (lane << 4));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) out[n] = static_cast<bf16_t>(acc);
}

template <int K>
struct Sm120Fp8E3m4GemvKernel {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView a_s,
      const tvm::ffi::TensorView w_s,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    const int N = static_cast<int>(w.shape()[0]);
    constexpr int kBlk = K / 128;
    TensorMatcher({K}).verify(x);
    TensorMatcher({N, K}).verify(w);
    TensorMatcher({kBlk}).with_dtype<fp32_t>().with_device(device).verify(a_s);
    TensorMatcher({N >> 7, kBlk}).with_dtype<fp32_t>().with_device(device).verify(w_s);
    dim3 grid(1, (N + 7) / 8, 1);
    LaunchKernel(grid, 256, device.unwrap())(
        sm120_fp8_e3m4_gemv_kernel<K>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const uint8_t*>(x.data_ptr()),
        static_cast<const uint8_t*>(w.data_ptr()),
        static_cast<const float*>(a_s.data_ptr()),
        static_cast<const float*>(w_s.data_ptr()),
        N);
  }
};

}  // namespace sglang
