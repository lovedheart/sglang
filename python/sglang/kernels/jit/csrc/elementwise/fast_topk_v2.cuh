// Split-row radix-select top-k (v2 of the JIT fast_topk kernel): selects like
// fast_topk.cuh but spreads each long row's stage-1 histogram over several
// blocks at small batch, where the one-block-per-row kernel runs at single-SM
// bandwidth. For each row b, select the kTopK largest scores in
// [row_starts[b], row_starts[b] + lengths[b]) and write their indices
// relative to row_starts[b]; order within a row is unspecified.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/once.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace fast_topk_v2_detail {

constexpr uint32_t kThreadsPerBlock = 1024;
// Each radix pass needs at most ~kTopK candidates in the threshold bin, so
// 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient. The
// sorted path appends a kTopK-entry bitonic scratch (indices + keys) after
// the candidate lists.
constexpr size_t kSmemBytes = 8 * 1024 * sizeof(uint32_t);  // 32KB
template <int kTopK>
constexpr size_t kSortScratchBytes = 2 * kTopK * sizeof(int32_t);
template <int kTopK>
constexpr size_t kSmemBytesSorted = kSmemBytes + kSortScratchBytes<kTopK>;

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, kTopK]
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
  bool sorted;  // emit rows in torch.topk order (costs the in-kernel bitonic)
};

// Split-row path. The one-block kernel pins a row to a single SM, so at small
// batch the memory-bound stage-1 histogram runs at one-SM bandwidth. The split
// path spreads each row's coarse histogram over `split` blocks (pass 1), then
// re-runs the exact radix refinement per row (pass 2) reading the combined
// histogram instead of rescanning the row for stage 1. The refinement is the
// same code path as the one-block kernel, so the selected index SET matches
// bit-for-bit; slot order within a row stays unspecified as before.
constexpr uint32_t kCoarseBins = 256;

// Rows at or below this keep the one-block path in pass 2 (and retire early
// from pass 1): the second launch and cross-block combine cost more than the
// scan they save below this length.
constexpr uint32_t kSplitFloor = 8192;

// Upper bound on rows the persistent workspace serves; beyond it the host
// falls back to the one-block path.
constexpr uint32_t kSplitMaxRows = 4096;

struct FastTopKSplitParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, kTopK]
  const int32_t* __restrict__ lengths;     // [B]
  uint32_t* __restrict__ hist;             // [B, kCoarseBins], zeroed, self-cleaning
  int64_t input_stride;
  uint32_t split;                          // ranks per row (pass 1); pass 2 reads it from gridDim-free logic
  bool sorted;                             // refine emits torch.topk order (see FastTopKParams)
};

SGL_DEVICE auto convert_to_uint8(float x) -> uint8_t {
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

SGL_DEVICE auto convert_to_uint32(float x) -> uint32_t {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// When length <= kTopK, write the indices directly.
template <int kTopK>
SGL_DEVICE void naive_topk(int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// Radix-select top-k. Assumes length > kTopK (checked by the caller).
// With `kPreCounted`, `coarse_hist` holds this row's stage-1 bin counts
// (combined across blocks by the split pass); they are loaded into smem and
// zeroed back through (the workspace stays clean for the next call, which is
// what makes CUDA-graph replay safe). Everything after stage 1 is identical
// to the one-block path, so the selected index SET matches bit-for-bit.
// With `sorted`, the row is emitted already sorted: winners are collected in
// atomic order as usual, then one in-smem bitonic network reorders them by
// (ordered score desc, index asc) -- matching torch.topk's tie convention.
// The network is data-order independent, so both the SET and the SLOT ORDER
// are run-invariant, and callers can drop their own sort fold.  Runtime flag:
// QSA decode defaults it OFF (the expand kernel's own fold is cheaper), so the
// scratch smem is simply unused on that path.
template <int kTopK, bool kPreCounted = false>
SGL_DEVICE void radix_select_topk(
    const float* __restrict__ input, int* __restrict__ index, int row_start, int length,
    uint32_t* __restrict__ coarse_hist, bool sorted) {
  int topk = kTopK;
  constexpr auto BLOCK_SIZE = kThreadsPerBlock;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmemBytes / (2 * sizeof(int));
  // Bitonic sort of the <=kTopK winners. The pair-merge form below (compare
  // q against q+step, both inside one block of sz) sorts N slots when the
  // scratch has N = kTopK slots and kTopK is a power of two; winners fill the
  // first n slots, the rest get a sentinel key that sorts last.
  constexpr auto SORT_SCRATCH = kTopK;

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // The dynamic region holds the two candidate lists (SMEM_INPUT_SIZE each).
  // The sort scratch is a separate dynamic allocation placed after both lists,
  // so it never aliases live candidate data.
  extern __shared__ int s_dyn[];
  auto s_input_idx = reinterpret_cast<int(*)[SMEM_INPUT_SIZE]>(s_dyn);
  auto* s_sort_idx = s_dyn + 2 * SMEM_INPUT_SIZE;
  auto* s_sort_key = reinterpret_cast<uint32_t*>(s_sort_idx + kTopK);

  const int tx = threadIdx.x;

  // In-smem bitonic sort of the first `n` entries of `index` (n <= kTopK),
  // descending by ordered score, ascending index inside a tie. Every lane
  // reads the same keys, so the network result never depends on the atomic
  // arrival order that filled the row.
  auto sort_topk_row = [&](int n) {
    if (!sorted) return;
    for (int i = tx; i < SORT_SCRATCH; i += BLOCK_SIZE) {
      if (i < n) {
        const auto idx = index[i];
        s_sort_idx[i] = idx;
        s_sort_key[i] = convert_to_uint32(input[idx + row_start]);
      } else {
        s_sort_idx[i] = -1;
        s_sort_key[i] = 0u;  // below every real key: pads sort last
      }
    }
    __syncthreads();
#pragma unroll
    for (int sz = 2; sz <= SORT_SCRATCH; sz <<= 1) {
#pragma unroll
      for (int step = sz >> 1; step > 0; step >>= 1) {
        for (int q = tx; q < SORT_SCRATCH; q += BLOCK_SIZE) {
          if ((q & step) == 0) {  // pairs (q, q+step); skip the upper halves
            // Standard bitonic direction bit: blocks of `sz` alternate
            // between descending and ascending runs.
            const bool desc_block = ((q & sz) == 0);
            const auto ka = s_sort_key[q];
            const auto kb = s_sort_key[q + step];
            // "a first": q precedes q+step in the final order -- key
            // descending, index ascending inside a tie, -1 last.
            const bool a_first =
                (ka != kb) ? (ka > kb) : (s_sort_idx[q] <= s_sort_idx[q + step]);
            const bool swap = desc_block ? !a_first : a_first;
            if (swap) {
              s_sort_key[q] = kb;
              s_sort_key[q + step] = ka;
              const auto t = s_sort_idx[q];
              s_sort_idx[q] = s_sort_idx[q + step];
              s_sort_idx[q + step] = t;
            }
          }
        }
        __syncthreads();
      }
    }
    for (int i = tx; i < n; i += BLOCK_SIZE) {
      index[i] = s_sort_idx[i];
    }
    __syncthreads();
  };

  // stage 1: 8bit coarse histogram
  if constexpr (!kPreCounted) {
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = convert_to_uint8(input[idx + row_start]);
      ::atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();
  } else {
    // Load the combined counts and hand the workspace back zeroed. The
    // trailing slot of the smem buffer is not part of the histogram.
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    if (tx < RADIX) {
      s_histogram[tx] = static_cast<int>(coarse_hist[tx]);
      coarse_hist[tx] = 0u;
    }
    __syncthreads();
  }

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  if constexpr (kPreCounted) {
    // A pre-counted row whose histogram sums below kTopK means the producer
    // scored fewer tokens than the length claims (an out-of-contract caller);
    // emit the naive prefix instead of reading an unset threshold bin.
    // After run_cumsum, s_histogram[0] holds the total of all bins.
    if (s_histogram[0] < topk) {
      for (int i = tx; i < kTopK; i += BLOCK_SIZE) {
        index[i] = (i < length) ? i : -1;
      }
      __syncthreads();
      return;
    }
  }
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    sort_topk_row(min(s_counter, kTopK));
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        // fuse the histogram computation here
        if (pos < int(SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      sort_topk_row(min(s_counter, kTopK));
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            // Bit-identical values: collect now, deposit below by ascending
            // index rank. The old arrival-order `atomicAdd(&s_last_remain,-1)`
            // deposit made the selected SET depend on warp scheduling.
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < int(SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < int(SMEM_INPUT_SIZE)) {
              // fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();

      if (round == 3) {
        // Final tie-bin deposit: every candidate left here is bit-identical.
        // Select by ascending index rank (torch.topk convention): a candidate
        // wins iff fewer than `remain` smaller-indexed candidates tie with it.
        // Counting is a pure function of the candidate set, so the selected
        // SET is run-invariant. The tie list caps at SMEM_INPUT_SIZE; a pop
        // that large (only reachable with >4096 identical scores) falls back
        // to the legacy arrival-order deposit.
        int tie_count = min(s_num_input[r_idx ^ 1], int(SMEM_INPUT_SIZE));
        int* tie_list = s_input_idx[r_idx ^ 1];
        if (s_num_input[r_idx ^ 1] > int(SMEM_INPUT_SIZE)) {
          for (int i = tx; i < tie_count; i += BLOCK_SIZE) {
            const auto idx = tie_list[i];
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[kTopK - pos] = idx;
            }
          }
        } else {
          // Every candidate in the list is bit-identical after 4 exact 8-bit
          // radix passes, so all of them tie. `topk` is already the number of
          // slots left after strictly-greater deposits this round.
          const int remain = topk;
          for (int i = tx; i < tie_count; i += BLOCK_SIZE) {
            const auto idx = tie_list[i];
            const auto key = convert_to_uint32(input[idx + row_start]);
            int rank = 0;
            for (int j = 0; j < tie_count; ++j) {
              const auto idx2 = tie_list[j];
              if (idx2 < idx &&
                  convert_to_uint32(input[idx2 + row_start]) == key) {
                ++rank;
              }
            }
            if (rank < remain) {
              const auto pos = ::atomicAdd(&s_counter, 1);
              index[pos] = idx;
            }
          }
        }
        __syncthreads();
        // The overflow fallback above deposits tie winners from the tail with
        // s_last_remain, which is run-varying in slot order; the sort below
        // normalizes the whole row either way.
        sort_topk_row(min(s_counter + topk, kTopK));
        break;
      }
    }
  }
}

template <int kTopK, bool kUsePDL>
__global__ __launch_bounds__(fast_topk_v2_detail::kThreadsPerBlock) void fast_topk_v2_kernel(
    const fast_topk_v2_detail::FastTopKParams __grid_constant__ params) {
  using namespace fast_topk_v2_detail;
  device::PDLWaitPrimary<kUsePDL>();

  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = params.row_starts[bid];
  const auto length = params.lengths[bid];
  const auto indice = params.indices + bid * kTopK;
  const auto score = params.input + bid * params.input_stride;
  if (length <= kTopK) {
    naive_topk<kTopK>(indice, length);
  } else {
    // Tail pre-fill: see the identical note in fast_topk.cuh -- the radix
    // select leaves slots past the final counter untouched, which a graph
    // replay would otherwise inherit from the previous capture.
    for (int i = threadIdx.x; i < kTopK; i += kThreadsPerBlock) {
      indice[i] = -1;
    }
    __syncthreads();
    radix_select_topk<kTopK>(score, indice, row_start, length, nullptr,
                     params.sorted);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

// Split pass 1: this rank's slice of each row's stage-1 coarse histogram,
// atomically merged into the persistent workspace. The rank-0 block also
// pre-fills the row's output with -1: the refine pass writes only the
// strictly-greater winners and the tie deposit, so on a CUDA-graph replay
// (same buffer) every slot it does not win must start clean. Short rows keep
// the identity prefix (naive semantics); long rows get all -1 and are fully
// overwritten by pass 2 up to its final counter.
template <int kTopK>
__global__ __launch_bounds__(fast_topk_v2_detail::kThreadsPerBlock) void fast_topk_v2_split_hist_kernel(
    const fast_topk_v2_detail::FastTopKSplitParams __grid_constant__ params) {
  using namespace fast_topk_v2_detail;
  const auto row = static_cast<uint64_t>(blockIdx.y);
  const auto rank = blockIdx.x;
  const auto length = params.lengths[row];
  if (rank == 0) {
    for (auto i = threadIdx.x; i < kTopK; i += kThreadsPerBlock) {
      params.indices[row * kTopK + i] =
          (static_cast<uint32_t>(length) <= kSplitFloor && i < length) ? static_cast<int32_t>(i) : -1;
    }
  }
  if (static_cast<uint32_t>(length) <= kSplitFloor) return;
  // Row base INCLUDING row_starts: this manual scan indexes score[idx] for
  // idx in [0,length), matching radix_select_topk's input[idx + row_start]
  // (whose own base omits row_starts). Both bin the same window.
  const auto score = params.input + row * params.input_stride + params.row_starts[row];
  __shared__ uint32_t s_hist[kCoarseBins];
  if (threadIdx.x < kCoarseBins) s_hist[threadIdx.x] = 0u;
  __syncthreads();
  for (auto idx = threadIdx.x + rank * kThreadsPerBlock; idx < length;
       idx += kThreadsPerBlock * gridDim.x) {
    ::atomicAdd(&s_hist[convert_to_uint8(score[idx])], 1u);
  }
  __syncthreads();
  if (threadIdx.x < kCoarseBins && s_hist[threadIdx.x] != 0u) {
    ::atomicAdd(&params.hist[row * kCoarseBins + threadIdx.x], s_hist[threadIdx.x]);
  }
}

// Split pass 2: one block per row. Short rows retire (pass 1 wrote them);
// long rows run the exact refinement over the combined histogram.
template <int kTopK, bool kUsePDL>
__global__ __launch_bounds__(fast_topk_v2_detail::kThreadsPerBlock) void fast_topk_v2_split_refine_kernel(
    const fast_topk_v2_detail::FastTopKSplitParams __grid_constant__ params) {
  using namespace fast_topk_v2_detail;
  device::PDLWaitPrimary<kUsePDL>();

  const auto row = static_cast<uint64_t>(blockIdx.x);
  const auto length = params.lengths[row];
  if (static_cast<uint32_t>(length) > kSplitFloor) {
    // Sorted deposit: the row leaves this kernel already in torch.topk order
    // (score desc, index asc), so the expand kernel's sort fold is redundant.
    radix_select_topk<kTopK, /*kPreCounted=*/true>(
        params.input + row * params.input_stride,
        params.indices + row * kTopK,
        params.row_starts[row],
        length,
        params.hist + row * kCoarseBins,
        params.sorted);
  } else {
    // Short row (length <= kSplitFloor): pass 1 either wrote an identity
    // prefix (normal split path) or nothing at all (external-hist refine,
    // split==0). In BOTH cases this block owns the whole row and must select
    // from the scores -- pass 1 never fills the histogram for a short row, so
    // recomputing stage 1 here is required, not optional. Trusting the earlier
    // "pass 1 wrote them" assumption produced an identity prefix whenever the
    // tensor was wider than the row (pick_split keys on width, not length).
    // Hand the histogram back zeroed in case a fused producer bumped it.
    if (threadIdx.x < kCoarseBins) params.hist[row * kCoarseBins + threadIdx.x] = 0u;
    for (int i = threadIdx.x; i < kTopK; i += kThreadsPerBlock) {
      params.indices[row * kTopK + i] = -1;
    }
    __syncthreads();
    if (static_cast<uint32_t>(length) <= kTopK) {
      naive_topk<kTopK>(params.indices + row * kTopK, length);
    } else {
      radix_select_topk<kTopK, /*kPreCounted=*/false>(
          params.input + row * params.input_stride,
          params.indices + row * kTopK,
          params.row_starts[row],
          length,
          nullptr,
          params.sorted);
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace fast_topk_v2_detail

/**
 * \brief Per-row top-k selection over ragged rows of a fp32 score matrix.
 *
 * Row b selects the kTopK largest values in
 * score[b, row_starts[b] : row_starts[b] + lengths[b]) and writes their
 * indices (relative to row_starts[b]) into indices[b]. Unfilled slots are
 * -1 when lengths[b] < kTopK.
 *
 * Rows longer than `kSplitFloor` at small batch go through the split path: a
 * histogram pass that spreads each row over `split` blocks plus a per-row
 * refinement pass. The split workspace is persistent and self-cleaning (the
 * refine pass zeroes it), so CUDA-graph capture and replay need no reset.
 */
template <int kTopK, bool kUsePDL>
struct FastTopKSplitKernel {
  static constexpr auto kernel = fast_topk_v2_detail::fast_topk_v2_kernel<kTopK, kUsePDL>;

  /// The sorted path's dynamic smem (candidate lists + bitonic scratch) can
  /// exceed the default 48KB cap; opt the launched kernels in once.
  static void ensure_smem_opt_in() {
    static bool done = [] {
      auto set = [](const void* f) {
        ::cudaFuncSetAttribute(
            f, cudaFuncAttributeMaxDynamicSharedMemorySize,
            fast_topk_v2_detail::kSmemBytesSorted<kTopK>);
      };
      set(reinterpret_cast<const void*>(kernel));
      set(reinterpret_cast<const void*>(fast_topk_v2_detail::fast_topk_v2_split_refine_kernel<kTopK, false>));
      return true;
    }();
    (void)done;
  }

  /// Persistent zeroed coarse-histogram workspace, one allocation per device.
  static auto split_hist_buffer(DLDevice device) -> host::ffi::Tensor& {
    // The factory's type is the cache discriminator (see once.h), so the key
    // is just the device.
    return host::init_once<int32_t>(device.device_id, [&] {
      constexpr int64_t kNumBins =
          static_cast<int64_t>(fast_topk_v2_detail::kSplitMaxRows) * fast_topk_v2_detail::kCoarseBins;
      auto buffer = host::ffi::empty({kNumBins}, DLDataType{kDLUInt, 32, 1}, device);
      CHECK_CUDA(cudaMemsetAsync(
          buffer.data_ptr(), 0, kNumBins * sizeof(uint32_t), host::LaunchKernel::resolve_device(device)));
      return buffer;
    });
  }

  /// Blocks to spread one row's stage-1 histogram over: fill the machine with
  /// rows * split blocks, bounded so the cross-block combine stays cheap.
  static auto pick_split(uint32_t num_rows, uint32_t max_length, DLDevice device) -> uint32_t {
    if (max_length <= fast_topk_v2_detail::kSplitFloor) return 0;
    const auto occupancy = host::runtime::get_blocks_per_sm(
        fast_topk_v2_detail::fast_topk_v2_split_hist_kernel<kTopK>, fast_topk_v2_detail::kThreadsPerBlock);
    const auto sm_count = host::runtime::get_sm_count(device.device_id);
    const auto target = std::max<uint32_t>(1u, occupancy * sm_count / std::max<uint32_t>(1u, num_rows));
    // A rank shorter than a few passes per thread is pure launch overhead.
    constexpr auto kMinRankItems = 8u * fast_topk_v2_detail::kThreadsPerBlock;
    const auto by_length = std::max<uint32_t>(1u, fast_topk_v2_detail::kSplitFloor / kMinRankItems);
    return std::min(target, std::max<uint32_t>(1u, max_length / by_length));
  }

  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView row_starts,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths,
      const bool sorted) {
    using namespace fast_topk_v2_detail;
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"length"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(score);
    TensorMatcher({B})  // row_starts
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(row_starts);
    TensorMatcher({B, kTopK})  // indices
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(indices);
    TensorMatcher({B})  // lengths
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(lengths);

    ensure_smem_opt_in();
    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    const auto base_params = FastTopKParams{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = static_cast<const int32_t*>(row_starts.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = S.unwrap(),
        .sorted = sorted,
    };

    const auto split = num_rows <= kSplitMaxRows ? pick_split(num_rows, L.unwrap(), device.unwrap()) : 0u;
    if (split <= 1) {
      LaunchKernel(num_rows, kThreadsPerBlock, device.unwrap(), kSmemBytesSorted<kTopK>)
          .enable_pdl(kUsePDL)(kernel, base_params);
      return;
    }

    auto& hist = split_hist_buffer(device.unwrap());
    const FastTopKSplitParams params{
        .input = base_params.input,
        .row_starts = base_params.row_starts,
        .indices = base_params.indices,
        .lengths = base_params.lengths,
        .hist = static_cast<uint32_t*>(hist.data_ptr()),
        .input_stride = base_params.input_stride,
        .split = split,
        .sorted = sorted,
    };
    // Pass 1 and pass 2 must not overlap: the refine reads the histogram the
    // histogram pass accumulates. PDL would let pass 2 start before pass 1
    // drained, so the split launches deliberately do not enable it.
    LaunchKernel({split, num_rows}, kThreadsPerBlock, device.unwrap())(
        fast_topk_v2_split_hist_kernel<kTopK>, params);
    LaunchKernel(num_rows, kThreadsPerBlock, device.unwrap(), kSmemBytesSorted<kTopK>)(
        fast_topk_v2_split_refine_kernel<kTopK, false>, params);
  }

  /// Refine-only entry for a coarse histogram produced elsewhere (the QSA
  /// scorer's fused epilogue atomically accumulates [B, kCoarseBins] counts
  /// in the convert_to_uint8 binning). Runs the split-pass-2 refinement per
  /// row -- which also zeroes the workspace back, keeping it self-cleaning --
  /// and emits short rows with the naive identity prefix here, since no pass-1
  /// pre-fill ran for them.
  static void
  run_refine(const tvm::ffi::TensorView score,
             const tvm::ffi::TensorView row_starts,
             const tvm::ffi::TensorView indices,
             const tvm::ffi::TensorView lengths,
             const tvm::ffi::TensorView coarse_hist) {
    using namespace fast_topk_v2_detail;
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"length"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({B, L}).with_strides({S, 1}).with_dtype<fp32_t>().with_device(device).verify(score);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(row_starts);
    TensorMatcher({B, kTopK}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B.unwrap(), kCoarseBins})
        .with_dtype<uint32_t>()
        .with_device(device)
        .verify(coarse_hist);

    ensure_smem_opt_in();
    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    const FastTopKSplitParams params{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = static_cast<const int32_t*>(row_starts.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .hist = static_cast<uint32_t*>(coarse_hist.data_ptr()),
        .input_stride = S.unwrap(),
        .split = 0,  // marks "no pass 1 ran": refine also fixes up short rows
        .sorted = true,
    };
    // No PDL here: the fused-scorer producer is a tilelang kernel that never
    // triggers the programmatic-launch dependency, so the secondary wait
    // would hang; and only the <kTopK,false> instantiation got the smem
    // opt-in above.
    LaunchKernel(num_rows, kThreadsPerBlock, device.unwrap(), kSmemBytesSorted<kTopK>)(
        fast_topk_v2_detail::fast_topk_v2_split_refine_kernel<kTopK, false>, params);
  }
};

}  // namespace sglang
