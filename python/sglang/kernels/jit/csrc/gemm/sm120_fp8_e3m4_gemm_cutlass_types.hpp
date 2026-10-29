// sm120_fp8_e3m4_gemm_cutlass_types.hpp — exact type recipe of the CUTLASS
// SM120 blockwise instantiation that the embedded cubin was built from.
//
// The cubin (mix3-patched e3m4 QMMA bits, see sm120_fp8_e3m4_gemm_cutlass.cuh)
// takes its Params by value, so this header MUST be compiled against the same
// CUTLASS tree that produced the cubin: the flashinfer-bundled 4.5.0 tree that
// load_jit resolves for extra_dependencies=["cutlass"] (verified by rebuilding
// the cubin against it; SNR 55.6 dB at 256x128x256 / 1024x4096x2560).
//
// Source uses e4m3 spellings (PTX cannot name e3m4); the B-operand QMMA format
// bits are patched e4m3 -> e3m4 post-hoc by mix3.py 0 2.
#pragma once

#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/version.h>

// Trip wire: Params is passed by value, so its layout is ABI-pinned to the
// CUTLASS tree the embedded cubin was built from. If flashinfer bumps its
// bundled cutlass, fail at load time (the python loader then silently falls
// back to the handwritten kernels: slower, never wrong) instead of silently
// reinterpreting Params and emitting garbage. Fix: rebuild the cubin against
// the new tree (~/gemm_dev/sg_integ/MAINTENANCE.md), expect test_sg SNR
// 55.6 dB, and update the pinned version here.
#if CUTLASS_VERSION != 450
#error "sm120_fp8_e3m4_gemm_cutlass: embedded cubin pinned to CUTLASS 4.5.0; rebuild it (comment above)"
#endif

namespace sglang::e3m4_cutlass {

using namespace cute;

using ElementA = cutlass::float_e4m3_t;   // activations (patched: stays e4m3)
using ElementB = cutlass::float_e4m3_t;   // weights (QMMA B-format patched -> e3m4)
using ElementD = cutlass::bfloat16_t;
using ElementC = void;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ClusterShape = Shape<_1, _1, _1>;

using MmaTileShape = Shape<_128, _128, _128>;
using PerSmTileShape = Shape<_128, _128, _128>;
using ScalesPerTile = Shape<_128, _1, _1>;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;

// majorSFA = Major::K: the SFA buffer is [M, K/128] ROW-major (m*KB + kb),
// exactly the [M, K/128] row-major layout sglang's per-token-group quantizer
// produces, so the wrapper never transposes it (the device code is compiled
// with this same choice). majorSFB = Major::K -> [N/128, K/128], w_scale as-is.
// ScaleGranularity = MmaTile / ScalesPerTile, same as the blockwise path.
constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;  // 1
constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});  // 128
constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});  // 128

using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
    ScaleGranularityM,
    ScaleGranularityN,
    ScaleGranularityK,
    cute::UMMA::Major::K,
    cute::UMMA::Major::K>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    PerSmTileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutCTag,
    AlignmentC,
    ElementD,
    LayoutDTag,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename CollectiveEpilogue::SharedStorage))>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    cute::tuple<LayoutATag, LayoutSFA>,
    AlignmentA,
    ElementB,
    cute::tuple<LayoutBTag, LayoutSFB>,
    AlignmentB,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    StageCount,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

using GemmKernelVoid =
    cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
// Persistent cooperative StreamK: splits K across the 188-SM grid for tall
// shapes; get_workspace_size > 0, cleared in run() before each launch.
using GemmKernelStreamK =
    cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
                                          cutlass::gemm::StreamKScheduler>;

}  // namespace sglang::e3m4_cutlass
