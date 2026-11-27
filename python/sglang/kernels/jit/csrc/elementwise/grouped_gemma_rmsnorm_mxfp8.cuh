#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp8.h>

namespace sglang {

struct GroupedGemmaRMSNormMXFP8Params {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  void* quant;
  void* sf;
  uint32_t num_groups;
  float eps;
};

/**
 * \brief Grouped Gemma RMSNorm with an MXFP8 activation-quant epilogue.
 *
 * Norm math is the exact instruction sequence of grouped_gemma_rmsnorm_kernel
 * (keep the two in sync bit for bit): out = x * rsqrt(mean(x^2) + eps) * (1 + w)
 * per group_size chunk, stored as the original 2-byte dtype.
 *
 * The epilogue then quantizes the just-rounded output in 32-element groups to
 * e4m3 + ue8m0 scale bytes, row-major, matching ops/gemm/hc_mix_fp8.py's
 * _mxfp8_quant_act_kernel bit for bit:
 *   e = ceil(log2(max(amax, 1e-30) / 448)), clamped to [-126, 126], 0 if amax==0
 *   q = cvt.rn.satfinite.e4m3(x / exp2(e)),  sf byte = (uint8)(e + 127)
 * It runs on the registers the norm already produced, saving one full read of
 * the normalized tensor. Quantizing the rounded register value equals reading
 * the rounded value back from memory, so the mix GEMMs see identical inputs.
 *
 * Blackwell only: the epilogue assumes each thread owns 16 consecutive
 * elements (one 32B vector), so one 32-element group spans exactly two
 * adjacent threads (one __shfl_xor). On older arches the norm path would
 * still compile but the group mapping differs; run() refuses to launch there.
 *
 * \tparam kGroupSize Elements per variance group; must be a multiple of 512.
 * \tparam kUsePDL    Whether to emit the PDL wait/trigger pair.
 * \tparam Float      Element type: bf16_t | fp16_t.
 */
template <int64_t kGroupSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(kGroupSize / 16) void grouped_gemma_rmsnorm_mxfp8_kernel(
    const GroupedGemmaRMSNormMXFP8Params __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
#if SGL_ARCH_BLACKWELL_OR_GREATER
  using Storage = AlignedVector<Float2, 8>;
  constexpr uint32_t kNumLoads = 1;
#else
  using Storage = AlignedVector<Float2, 4>;
  constexpr uint32_t kNumLoads = 2;
#endif
  constexpr uint32_t kVecLen = kNumLoads == 1 ? 8 : 4;
  constexpr auto kNumThreads = kGroupSize / 16;
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const uint32_t bid = blockIdx.x;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[kWarpThreads];

  PDLWaitPrimary<kUsePDL>();

  const auto input_ptr = pointer::offset<Float>(params.input, static_cast<int64_t>(bid) * kGroupSize);
  const auto output_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(bid) * kGroupSize);
  const auto weight_ptr =
      pointer::offset<Float>(params.weight, static_cast<int64_t>(bid % params.num_groups) * kGroupSize);

  Storage input_vec[kNumLoads];
  Storage weight_vec[kNumLoads];
#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
    input_vec[j] = gmem.load(input_ptr, j);
    weight_vec[j] = gmem.load(weight_ptr, j);
  }

  float sum_of_squares = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
#pragma unroll
    for (uint32_t i = 0; i < kVecLen; ++i) {
      const auto [x, y] = cast<fp32x2_t>(input_vec[j][i]);
      sum_of_squares += x * x + y * y;
    }
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  float norm_factor;
  if constexpr (kNumWarps == 1) {
    norm_factor = math::rsqrt(sum_of_squares / kGroupSize + params.eps);
  } else {
    const auto warp_id = threadIdx.x / kWarpThreads;
    smem[warp_id] = sum_of_squares;
    __syncthreads();
    if (warp_id == 0) {
      const auto tx = threadIdx.x;
      const auto local_sum = tx < kNumWarps ? smem[tx] : 0.0f;
      sum_of_squares = warp::reduce_sum(local_sum);
      smem[tx] = math::rsqrt(sum_of_squares / kGroupSize + params.eps);
    }
    __syncthreads();
    norm_factor = smem[warp_id];
  }

#if SGL_ARCH_BLACKWELL_OR_GREATER
  auto quant_ptr = static_cast<__nv_fp8_storage_t*>(params.quant) + static_cast<int64_t>(bid) * kGroupSize;
  auto sf_ptr = static_cast<uint8_t*>(params.sf) + static_cast<int64_t>(bid) * (kGroupSize / 32);
#endif

#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
    Storage output_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen; ++i) {
      const auto [ix, iy] = cast<fp32x2_t>(input_vec[j][i]);
      const auto [wx, wy] = cast<fp32x2_t>(weight_vec[j][i]);
      output_vec[i] = cast<Float2>(fp32x2_t{ix * norm_factor * (1.0f + wx), iy * norm_factor * (1.0f + wy)});
    }
    gmem.store(output_ptr, output_vec, j);

#if SGL_ARCH_BLACKWELL_OR_GREATER
    // One j == one 32B vector == 16 consecutive elements (kNumLoads == 1).
    // The epilogue's group mapping needs one 32B vector per thread (16
    // consecutive elements); pre-Blackwell Storage would be 16B with 2 loads.
    static_assert(sizeof(Storage) == 32, "MXFP8 epilogue assumes a single 32B vector per thread");
    float v[16];
    float amax = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen; ++i) {
      const auto [ox, oy] = cast<fp32x2_t>(output_vec[i]);
      v[2 * i] = ox;
      v[2 * i + 1] = oy;
      amax = fmaxf(amax, fmaxf(fabsf(ox), fabsf(oy)));
    }
    // A 32-element group spans this thread and its xor-1 partner; both lanes
    // derive the same exponent from the same amax, so only even lanes store it.
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1u));
    // Same libdevice log2 (log2f lowers to __nv_log2f, and load_jit strips
    // --use_fast_math) and same div.full (__fdividef) as the Triton quantizer.
    const float e =
        amax > 0.0f ? fminf(fmaxf(ceilf(log2f(__fdividef(fmaxf(amax, 1e-30f), 448.0f))), -126.0f), 126.0f) : 0.0f;
    const float d = exp2f(e);
    uint16_t bytes[8];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const float2 pair = {__fdividef(v[2 * i], d), __fdividef(v[2 * i + 1], d)};
      bytes[i] = __nv_cvt_float2_to_fp8x2(pair, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint4*>(quant_ptr + static_cast<ptrdiff_t>(threadIdx.x) * 16) =
        *reinterpret_cast<const uint4*>(bytes);
    if ((threadIdx.x & 1u) == 0u) {
      sf_ptr[threadIdx.x >> 1] = static_cast<uint8_t>(e + 127.0f);
    }
#endif
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kGroupSize, bool kUsePDL, typename DType>
struct GroupedGemmaRMSNormMXFP8Kernel {
  static_assert(sizeof(DType) == 2, "GroupedGemmaRMSNormMXFP8 only supports 2-byte dtypes");
  static_assert(kGroupSize % 512 == 0, "kGroupSize must be a multiple of 512");
  static constexpr auto kernel = grouped_gemma_rmsnorm_mxfp8_kernel<kGroupSize, kUsePDL, DType>;
  static constexpr auto kBlockSize = static_cast<uint32_t>(kGroupSize / 16);

  /**
   * \brief Validate tensors and launch one CTA per (token, group) chunk.
   * \param input  [M, H] contiguous, H % kGroupSize == 0
   * \param weight [H]
   * \param output [M, H] contiguous, same shape/dtype/device as input
   * \param quant  [M, H] contiguous uint8 view of the e4m3 output
   * \param sf     [M, H/32] contiguous uint8 ue8m0 scale bytes
   * \param eps    RMSNorm epsilon
   */
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView quant,
      const tvm::ffi::TensorView sf,
      float eps) {
    using namespace host;
    CHECK_HOST(kGroupSize % 32 == 0) << "grouped_gemma_rmsnorm_mxfp8: kGroupSize must be a multiple of 32";
#if !SGL_ARCH_BLACKWELL_OR_GREATER
    CHECK_HOST(false) << "grouped_gemma_rmsnorm_mxfp8 requires SM100+ (Blackwell)";
#endif
    auto M = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto Sf = SymbolicSize{"sf_cols"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({M, H})  // input
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({H})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({M, H})  // output
        .with_dtype<DType>()
        .with_device(device)
        .verify(output);
    TensorMatcher({M, H})  // quant (passed as a uint8 view)
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(quant);
    TensorMatcher({M, Sf})  // sf
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(sf);

    const int64_t hidden_size = H.unwrap();
    CHECK_HOST(Sf.unwrap() * 32 == hidden_size) << "grouped_gemma_rmsnorm_mxfp8: sf must have hidden_size/32 columns";
    CHECK_HOST(hidden_size % kGroupSize == 0) << "grouped_gemma_rmsnorm_mxfp8: hidden_size (" << hidden_size
                                              << ") must be divisible by group_size (" << kGroupSize << ")";

    const auto params = GroupedGemmaRMSNormMXFP8Params{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .quant = quant.data_ptr(),
        .sf = sf.data_ptr(),
        .num_groups = static_cast<uint32_t>(hidden_size / kGroupSize),
        .eps = eps,
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const uint32_t num_blocks = num_tokens * params.num_groups;
    if (num_blocks == 0) return;
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
