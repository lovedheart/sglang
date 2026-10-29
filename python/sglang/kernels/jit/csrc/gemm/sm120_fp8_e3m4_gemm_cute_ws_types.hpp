#pragma once
// Types + host Params builder for the self-owned CuTe warp-specialized e3m4
// GEMM (cute_ws). Compiled host-side against the flashinfer-bundled CUTLASS
// 4.5.0 tree; the embedded cubin in sm120_fp8_e3m4_gemm_cute_ws.cuh was built
// from the SAME headers (nvcc -ptx | cu13.3 ptxas -arch=sm_120a -O3 | mix3 0 2)
// and its SASS is byte-identical to the CUTLASS 4.1 build, so Params layout
// matches the kernel exactly. Source spells e4m3; mix3 flipped the QMMA
// B-format bits to e3m4 (A stays e4m3).


#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/detail/blockwise_scale_layout.hpp>

namespace e3m4_cute_ws {
using namespace cute;

static constexpr int STAGES = 3;
static constexpr int NUM_THREADS = 384;      // 1 producer WG (warp0 works) + 2 consumer WGs
static constexpr int TILE_M = 128, TILE_N = 128, TILE_K = 128;
// 1024 (hdr/barriers) + 3*32768 (stages) + 3*128*4 (sSFA) + 16 (sSFB) + 64 pad
static constexpr int SMEM_BYTES = 1024 + STAGES * 32768 + 3 * 128 * 4 + 80;

using ElementA = cutlass::float_e4m3_t;      // activations (mix3 A=0: stays e4m3)
using ElementB = cutlass::float_e4m3_t;      // weights     (mix3 B=2: -> e3m4)
using ElementSF = float;
using TmaInternalElementA = uint8_t;         // avoid TMA rounding (CUTLASS uses uint_bit_t<8>)
using TmaInternalElementB = uint8_t;

using TileShape = Shape<_128, _128, _128>;

// Cooperative TiledMma: 8 mma warps (2 consumer WGs) over the 128x128 tile,
// atom layout (4,2,1), per-atom tile (128,32,32) — EXACTLY the builder's choice
// (sm120_blockwise_mma_builder.inl: TiledMma = rr_op_selector_sm120 x (4,2,1) x Tile<_128,_32,_32>).
using TiledMma = decltype(cute::make_tiled_mma(
    SM120_16x8x32_TN<ElementA, ElementB, float>{},
    Layout<Shape<_4, _2, _1>>{},
    Tile<_128, _32, _32>{}));

using SmemLayoutAtomA = UMMA::Layout_K_SW128_Atom<uint8_t>;
using SmemLayoutAtomB = UMMA::Layout_K_SW128_Atom<uint8_t>;
using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(_128{}, _128{}, Int<STAGES>{})));
using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(_128{}, _128{}, Int<STAGES>{})));

// Scale layout (1,128,128): SFA [M,K/128] row-major memory (sfa[kb*M+m]), SFB [N/128,K/128] k-major.
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using StrideA = Stride<int64_t, _1, int64_t>;   // (M,K,L) row-major A
using StrideB = Stride<int64_t, _1, int64_t>;   // (N,K,L) row-major B (col-major w/ TN)

// Host-side params: TMA atoms (hold CUtensorMap descriptors) + scale pointers.
// Device kernel takes this by value __grid_constant__ (same as CUTLASS device_kernel).
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
  uint32_t tma_transaction_bytes = 2 * 128 * 128;   // A+B bytes per stage (scales ride cp.async)
  ElementSF const* ptr_SFA;
  LayoutSFA layout_SFA;
  ElementSF const* ptr_SFB;
  LayoutSFB layout_SFB;
};

// Host helper: build Params exactly like CollectiveMma::to_underlying_arguments.
// ptrA: M x K e4m3 codes; ptrB: N x K; SFA/SFB layouts from ScaleConfig::tile_atom_to_shape_*.
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
  p.ptr_SFA = ptrSFA; p.layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  p.ptr_SFB = ptrSFB; p.layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  return p;
}

}  // namespace e3m4_cute_ws
