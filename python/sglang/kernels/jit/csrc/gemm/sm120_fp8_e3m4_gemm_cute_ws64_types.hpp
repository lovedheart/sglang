#pragma once
// Types + host Params builder for the CuTe warp-specialized e3m4 64x64 GEMM
// (cute_ws_64x64, shipped config A_S_GMEM=0). Compiled host-side against the
// flashinfer-bundled CUTLASS 4.5.0 tree; the embedded cubin in
// sm120_fp8_e3m4_gemm_cute_ws64.cuh was built from the SAME headers
// (nvcc -ptx --expt-relaxed-constexpr -DA_S_GMEM=0 | cu13 ptxas -arch=sm_120a
// -O3 | mix3 0 2, gemm_dev/sg_integ/build_64x64_sg.sh), so the Params layout
// matches the kernel exactly. Source spells e4m3; mix3 flipped the QMMA
// B-format bits to e3m4 (A stays e4m3).
//
// Scale contract (fp32 everywhere):
//   A e4m3 [M,K] + a_s fp32 [M, K/64] row-major            (1x64 act groups)
//   B e3m4 [N,K] + w_s fp32 [ceil(N/64), K/64] row-major   (64x64 blocks)

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>

namespace e3m4_cute_ws64 {
using namespace cute;

static constexpr int STAGES = 2;
static constexpr int NUM_THREADS = 384;      // 1 producer WG + 2 consumer WGs
static constexpr int TILE_M = 128, TILE_N = 128, TILE_K = 128;
static constexpr int KG = TILE_K / 64;       // 2 scale groups per K tile
static constexpr int BANDS = TILE_N / 64;    // 2 weight-scale bands per N tile
// 1024 hdr + 2*32768 stages + 2*KG*(128+BANDS)*4 staged a_s/w_s + 64 pad = 68704
static constexpr int SMEM_BYTES =
    1024 + STAGES * 32768 + STAGES * KG * (TILE_M + BANDS) * 4 + 64;

using ElementA = cutlass::float_e4m3_t;      // activations (mix3 A=0: stays e4m3)
using ElementB = cutlass::float_e4m3_t;      // weights     (mix3 B=2: -> e3m4)
using ElementSF = float;
using TmaInternalElementA = uint8_t;
using TmaInternalElementB = uint8_t;

using TiledMma = decltype(cute::make_tiled_mma(
    SM120_16x8x32_TN<ElementA, ElementB, float>{},
    Layout<Shape<_4, _2, _1>>{},
    Tile<_128, _32, _32>{}));

using SmemLayoutAtomA = UMMA::Layout_K_SW128_Atom<uint8_t>;
using SmemLayoutAtomB = UMMA::Layout_K_SW128_Atom<uint8_t>;
using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(_128{}, _128{}, Int<STAGES>{})));
using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(_128{}, _128{}, Int<STAGES>{})));

using StrideA = Stride<int64_t, _1, int64_t>;
using StrideB = Stride<int64_t, _1, int64_t>;

struct Params {
  using TMA_A = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
      SmemLayoutA{}(_, _, 0),
      make_shape(_128{}, _128{}),
      _1{}));
  using TMA_B = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
      SmemLayoutB{}(_, _, 0),
      make_shape(_128{}, _128{}),
      _1{}));

  TMA_A tma_load_a;
  TMA_B tma_load_b;
  uint32_t tma_transaction_bytes = 2 * 128 * 128;
  ElementSF const* ptr_SFA;   // [M, gK] row-major
  ElementSF const* ptr_SFB;   // [bands, gK] row-major
  int gK;                     // ceil(K/64)
  int bands;                  // ceil(N/64)
};

inline Params make_params(void* ptrA, int64_t ldA, void* ptrB, int64_t ldB,
                          float const* ptrSFA, float const* ptrSFB,
                          int M, int N, int K) {
  auto ptr_a = recast_ptr<TmaInternalElementA>((ElementA*)ptrA);
  auto ptr_b = recast_ptr<TmaInternalElementB>((ElementB*)ptrB);
  Tensor tensor_a = make_tensor(ptr_a, make_shape(M, K, 1), make_stride(ldA, _1{}, int64_t(0)));
  Tensor tensor_b = make_tensor(ptr_b, make_shape(N, K, 1), make_stride(ldB, _1{}, int64_t(0)));
  Params p;
  p.tma_load_a = make_tma_copy(SM90_TMA_LOAD{}, tensor_a, SmemLayoutA{}(_, _, 0),
                               make_shape(_128{}, _128{}), _1{});
  p.tma_load_b = make_tma_copy(SM90_TMA_LOAD{}, tensor_b, SmemLayoutB{}(_, _, 0),
                               make_shape(_128{}, _128{}), _1{});
  p.tma_transaction_bytes = 2 * 128 * 128;
  p.ptr_SFA = ptrSFA;
  p.ptr_SFB = ptrSFB;
  p.gK = (K + 63) / 64;
  p.bands = (N + 63) / 64;
  return p;
}

}  // namespace e3m4_cute_ws64
