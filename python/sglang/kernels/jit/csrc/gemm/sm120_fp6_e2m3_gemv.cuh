#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {

// FP6-e2m3 (s ee mmm, bias 1, denorm m/8, max finite 7.0) blockwise-32 GEMV/GEMM
// for M <= 8 (single-token decode and MTP verify batches). Weights are stored as
// tight 6-bit packed codes (4 codes per 3 bytes, row-major [N, K]); activations
// are plain bf16 (this is the lm-head shape: no activation quantization).
//
//   y[m, n] = sum_k x[m, k] * dec6(w[n, k]) * w_s[n / 32, k / 32]
//
// One warp per output row: lane reads 4 codes (one 24-bit stream window via
// funnel shift) x K/128 passes, accumulates M dot products per code window,
// then a warp shuffle reduction. Bandwidth-bound by design: sm_120 has no fp6
// tensor-core mode, so SIMT unpack at ~96% of the 6-bit stream load floor is
// the practical optimum for this shape (measured 254us at 248320x2560 vs a
// 253us pure-load floor; bf16 torch gemv for the same weight costs 708us).
// Per row beyond M=1 costs one packed 64-bit x-load + fma per 4-code window
// (M=4: 364us, M=8: 591us on the lm-head shape).

__device__ __forceinline__ float dec6(uint32_t b) {  // fp6 e2m3: s ee mmm, bias 1
  uint32_t e = (b >> 3) & 3u, m = b & 7u;
  float v = __uint_as_float(((e + 126u) << 23) | (m << 20));
  float den = static_cast<float>(m) * 0.125f;
  v = (e == 0) ? den : v;
  return (b & 32u) ? -v : v;
}

template <int K, int M>
__global__ void __launch_bounds__(256) sm120_fp6_e2m3_gemv_kernel(
    bf16_t* __restrict__ out,       // [M, N]
    const __nv_bfloat16* __restrict__ x,  // [M, K]
    const uint8_t* __restrict__ w,        // [N*K*6/8] packed codes
    const float* __restrict__ w_s,        // [N/32, K/32]
    int N) {
  static_assert(K % 128 == 0, "K must be a multiple of 128");
  static_assert(M >= 1 && M <= 8, "this band covers M <= 8; larger M must not route here");
  constexpr int kRowBytes = K * 6 / 8;
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (n >= N) return;
  const uint8_t* pw = w + static_cast<size_t>(n) * kRowBytes;
  const float* ws_row = w_s + static_cast<size_t>(n >> 5) * (K / 32);
  const __nv_bfloat16* x_row = x;  // [M, K]
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  // Each pass consumes 128 codes: lane covers 4 codes at code index
  // p * 128 + lane * 4, i.e. byte p * 96 + lane * 3.
#pragma unroll 4
  for (int p = 0; p < K / 128; ++p) {
    const int b = p * 96 + lane * 3;
    const unsigned* ab = reinterpret_cast<const unsigned*>(pw + (b & ~3));
    const unsigned lo = __ldg(ab);
    // Codes only straddle into the next word when the window starts mid-word
    // (b % 4 >= 2); rows are 4-byte aligned and a multiple of 4 bytes long,
    // so this never reads past the packed buffer.
    const unsigned hi = (b & 3) >= 2 ? __ldg(ab + 1) : 0u;
    const unsigned u = __funnelshift_r(lo, hi, (b & 3) * 8);
    const float s = __ldg(ws_row + (p * 4 + (lane >> 3)));
    const float d0 = dec6(u & 63u);
    const float d1 = dec6((u >> 6) & 63u);
    const float d2 = dec6((u >> 12) & 63u);
    const float d3 = dec6((u >> 18) & 63u);
    const int k0 = p * 128 + lane * 4;
    // 4 bf16 activations are exactly 8 bytes (k0 % 4 == 0 keeps this aligned);
    // one 64-bit load per row + shifts beats four 16-bit LDGs by a wide margin.
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const unsigned long long q =
          __ldg(reinterpret_cast<const unsigned long long*>(x_row + static_cast<size_t>(m) * K + k0));
      const unsigned lo32 = static_cast<unsigned>(q);
      const unsigned hi32 = static_cast<unsigned>(q >> 32);
      // bf16 -> fp32 is a left shift of the raw bits (no round-trip conversion).
      const float xa = __uint_as_float(lo32 << 16);
      const float xb = __uint_as_float(lo32 & 0xFFFF0000u);
      const float xc = __uint_as_float(hi32 << 16);
      const float xd = __uint_as_float(hi32 & 0xFFFF0000u);
      const float t = d0 * xa + d1 * xb + d2 * xc + d3 * xd;
      acc[m] = fmaf(t, s, acc[m]);
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
  }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) out[static_cast<size_t>(m) * N + n] = static_cast<bf16_t>(acc[m]);
  }
}

template <int K>
struct Sm120Fp6E2m3GemvKernel {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView w_s,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    const int M = static_cast<int>(x.shape()[0]);
    const int N = static_cast<int>(out.shape()[1]);
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({(static_cast<long long>(N) * K * 6 / 8)}).verify(w);
    TensorMatcher({N >> 5, K / 32}).with_dtype<fp32_t>().with_device(device).verify(w_s);
    host::RuntimeCheck(N % 32 == 0, "fp6 e2m3 gemv needs N % 32 == 0, got ", N);
    dim3 grid(1, (N + 7) / 8, 1);
    auto launch = [&](auto kern) {
      LaunchKernel(grid, 256, device.unwrap())(
          kern,
          static_cast<bf16_t*>(out.data_ptr()),
          static_cast<const __nv_bfloat16*>(x.data_ptr()),
          static_cast<const uint8_t*>(w.data_ptr()),
          static_cast<const float*>(w_s.data_ptr()),
          N);
    };
    switch (M) {
      case 1: launch(sm120_fp6_e2m3_gemv_kernel<K, 1>); break;
      case 2: launch(sm120_fp6_e2m3_gemv_kernel<K, 2>); break;
      case 3: launch(sm120_fp6_e2m3_gemv_kernel<K, 3>); break;
      case 4: launch(sm120_fp6_e2m3_gemv_kernel<K, 4>); break;
      case 5: launch(sm120_fp6_e2m3_gemv_kernel<K, 5>); break;
      case 6: launch(sm120_fp6_e2m3_gemv_kernel<K, 6>); break;
      case 7: launch(sm120_fp6_e2m3_gemv_kernel<K, 7>); break;
      case 8: launch(sm120_fp6_e2m3_gemv_kernel<K, 8>); break;
      default:
        host::RuntimeCheck(false, "fp6 e2m3 gemv supports M <= 8, got ", M);
    }
  }
};

}  // namespace sglang
