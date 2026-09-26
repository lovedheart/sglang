import math
from dataclasses import dataclass
from typing import Optional, Type

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
from quack.cute_dsl_utils import ParamsBase

from sglang.kernels.ops.attention.flash_attn.cute import utils



@cute.jit
def _nvfp4_to_bf16(code: cutlass.Uint32) -> cutlass.BFloat16:
    # NVFP4 (e2m1) -> BFloat16: magnitudes {0,.5,1,1.5,2,3,4,6} and the sign
    # bit are exactly representable, so a select chain is exact and cheap.
    mag = code & 0x7
    v = cutlass.BFloat16(0.0)
    if mag == 1:
        v = cutlass.BFloat16(0.5)
    elif mag == 2:
        v = cutlass.BFloat16(1.0)
    elif mag == 3:
        v = cutlass.BFloat16(1.5)
    elif mag == 4:
        v = cutlass.BFloat16(2.0)
    elif mag == 5:
        v = cutlass.BFloat16(3.0)
    elif mag == 6:
        v = cutlass.BFloat16(4.0)
    else:
        if mag == 7:
            v = cutlass.BFloat16(6.0)
    if ((code >> 3) & 0x1) == 1:
        v = cutlass.BFloat16(0.0) - v
    return v

@dataclass
class Sm120PagedKVManager(ParamsBase):
    """SM120 paged-KV loader for the stage-sliced FA4 pipeline."""

    mPageTable: cute.Tensor
    mK_paged: cute.Tensor
    mV_paged: cute.Tensor
    thread_idx: Int32

    page_size_divmod: FastDivmodDivisor
    seqlen_k: Int32
    leftpad_k: Int32
    n_block_size: cutlass.Constexpr[Int32]
    num_threads: cutlass.Constexpr[Int32]
    head_dim_padded: cutlass.Constexpr[Int32]
    head_dim_v_padded: cutlass.Constexpr[Int32]

    gmem_threads_per_row: cutlass.Constexpr[Int32]
    page_entry_per_thread: cutlass.Constexpr[Int32]
    async_copy_elems: cutlass.Constexpr[Int32]

    gmem_tiled_copy_KV: cute.TiledCopy
    gmem_thr_copy_KV: cute.TiledCopy
    tPrPage: cute.Tensor
    tPrPageOffset: cute.Tensor
    # Native packed-NVFP4 KV support (SM120 SIMT loader path only).
    packed_fp4: cutlass.Constexpr[bool] = False
    mKsf_paged: Optional[cute.Tensor] = None
    mVsf_paged: Optional[cute.Tensor] = None
    compute_dtype: Type[cutlass.Numeric] = cutlass.BFloat16
    dump_gmem: Optional[cute.Tensor] = None

    @staticmethod
    def create(
        mPageTable: cute.Tensor,
        mK_paged: cute.Tensor,
        mV_paged: cute.Tensor,
        page_size_divmod: FastDivmodDivisor,
        bidb: Int32,
        bidh: Int32,
        thread_idx: Int32,
        seqlen_k: Int32,
        leftpad_k: Int32,
        n_block_size: cutlass.Constexpr[Int32],
        head_dim_padded: cutlass.Constexpr[Int32],
        head_dim_v_padded: cutlass.Constexpr[Int32],
        num_threads: cutlass.Constexpr[Int32],
        dtype: Type[cutlass.Numeric],
        packed_fp4: cutlass.Constexpr[bool] = False,
        mKsf_paged: Optional[cute.Tensor] = None,
        mVsf_paged: Optional[cute.Tensor] = None,
        compute_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        dump_gmem: Optional[cute.Tensor] = None,
    ):
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // dtype.width
        dtype_bytes = dtype.width // 8
        gmem_k_block_size = math.gcd(
            head_dim_padded,
            head_dim_v_padded,
            128 // dtype_bytes,
        )
        assert gmem_k_block_size % async_copy_elems == 0
        gmem_threads_per_row = gmem_k_block_size // async_copy_elems
        assert cute.arch.WARP_SIZE % gmem_threads_per_row == 0

        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        thr_layout = cute.make_ordered_layout(
            (num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(
            atom_async_copy, thr_layout, val_layout
        )
        gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(thread_idx)

        # SM120 decode tiles can have fewer rows than DMA threads. Keep one
        # register entry per thread so those shapes do not create zero-sized
        # register tensors.
        page_entry_per_thread = max(1, (n_block_size + num_threads - 1) // num_threads)
        tPrPage = cute.make_rmem_tensor((page_entry_per_thread,), Int32)
        tPrPageOffset = cute.make_rmem_tensor((page_entry_per_thread,), Int32)

        return Sm120PagedKVManager(
            mPageTable[bidb, None],
            mK_paged[None, None, bidh, None],
            mV_paged[None, None, bidh, None],
            thread_idx,
            page_size_divmod,
            seqlen_k,
            leftpad_k,
            n_block_size,
            num_threads,
            head_dim_padded,
            head_dim_v_padded,
            gmem_threads_per_row,
            page_entry_per_thread,
            async_copy_elems,
            gmem_tiled_copy_KV,
            gmem_thr_copy_KV,
            tPrPage,
            tPrPageOffset,
            packed_fp4,
            mKsf_paged[None, None, bidh, None] if packed_fp4 else None,
            mVsf_paged[None, None, bidh, None] if packed_fp4 else None,
            compute_dtype,
            dump_gmem,
        )

    @cute.jit
    def _load_page_table_entry(self, i: Int32, n_block: Int32):
        row = (
            i * self.num_threads
            + (self.thread_idx % self.gmem_threads_per_row)
            * (self.num_threads // self.gmem_threads_per_row)
            + (self.thread_idx // self.gmem_threads_per_row)
        )
        row_idx = n_block * self.n_block_size + row
        page_idx, page_offset = divmod(row_idx + self.leftpad_k, self.page_size_divmod)
        is_valid = (
            (i + 1) * self.num_threads <= self.n_block_size or row < self.n_block_size
        ) and row_idx < self.seqlen_k
        page = self.mPageTable[page_idx] if is_valid else 0
        self.tPrPage[i] = page
        self.tPrPageOffset[i] = page_offset

    @cute.jit
    def load_page_table(self, n_block: Int32):
        # The entry count is a specialization constant for SM120. Expanding
        # this small loop removes a measurable dynamic-loop cost in decode.
        for i in cutlass.range_constexpr(self.page_entry_per_thread):
            self._load_page_table_entry(i, n_block)

    @cute.jit
    def compute_X_ptr(self, K_or_V: str):
        tPrXPtr = cute.make_rmem_tensor((self.page_entry_per_thread,), cutlass.Int64)
        mX = self.mK_paged if const_expr(K_or_V == "K") else self.mV_paged
        for i in cutlass.range_constexpr(self.page_entry_per_thread):
            page = self.tPrPage[i]
            page_offset = self.tPrPageOffset[i]
            # SGLang stores both paged K and paged V as
            # (page_size, head_dim, num_pages).
            tPrXPtr[i] = utils.elem_pointer(mX, (page_offset, 0, page)).toint()
        return tPrXPtr

    @cute.jit
    def compute_SF_ptr(self, K_or_V: str):
        tPrSfPtr = cute.make_rmem_tensor((self.page_entry_per_thread,), cutlass.Int64)
        mX = self.mKsf_paged if const_expr(K_or_V == "K") else self.mVsf_paged
        for i in cutlass.range_constexpr(self.page_entry_per_thread):
            page = self.tPrPage[i]
            page_offset = self.tPrPageOffset[i]
            tPrSfPtr[i] = utils.elem_pointer(mX, (page_offset, 0, page)).toint()
        return tPrSfPtr

    @cute.jit
    def _copy_row_async(
        self,
        tXsX: cute.Tensor,
        tXcX: cute.Tensor,
        mX_paged_cur_copy: cute.Tensor,
        m: Int32,
        should_load: cute.Tensor,
    ):
        for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
            ki = tXcX[0, 0, k][1] // self.async_copy_elems
            mX_paged_cur_copy_ki = mX_paged_cur_copy[None, ki]
            tXsX_k = tXsX[None, m, k]
            mX_paged_cur_copy_ki = cute.make_tensor(
                mX_paged_cur_copy_ki.iterator, tXsX_k.layout
            )
            cute.copy(
                self.gmem_tiled_copy_KV,
                mX_paged_cur_copy_ki,
                tXsX_k,
                pred=should_load,
            )

    @cute.jit
    def load_KV(self, n_block: Int32, sX: cute.Tensor, K_or_V: str):
        assert K_or_V in ("K", "V")
        if const_expr(self.packed_fp4):
            self._load_KV_fp4(n_block, sX, K_or_V)
            return

        tPrXPtr = self.compute_X_ptr(K_or_V)

        # The SM120 pipeline passes one stage at a time. V has already been
        # transposed by the caller's shared-memory view.
        sX_pi = cute.group_modes(sX, 0, 1)
        head_dim = (
            self.head_dim_v_padded
            if const_expr(K_or_V == "V")
            else self.head_dim_padded
        )
        cX = cute.make_identity_tensor((self.n_block_size, head_dim))
        tXsX = self.gmem_thr_copy_KV.partition_D(sX_pi)
        tXcX = self.gmem_thr_copy_KV.partition_S(cX)
        # D-side identity partition: exact (row, col) of every destination
        # slot (the V view is transposed, so S-side coords do not match).
        tXdX = self.gmem_thr_copy_KV.partition_D(cX)

        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            row_valid = True
            should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], cute.Boolean)
            should_load.fill(row_valid)

            x_ptr_i64 = utils.shuffle_sync(
                tPrXPtr[m // self.gmem_threads_per_row],
                m % self.gmem_threads_per_row,
                width=self.gmem_threads_per_row,
            )
            x_gmem_ptr = cute.make_ptr(
                self.mK_paged.element_type,
                x_ptr_i64,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            mX_paged_cur = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_paged_cur_copy = cute.tiled_divide(
                mX_paged_cur, (self.async_copy_elems,)
            )
            self._copy_row_async(tXsX, tXcX, mX_paged_cur_copy, m, should_load)

    @cute.jit
    def _load_KV_fp4(self, n_block: Int32, sX: cute.Tensor, K_or_V: str):
        # Native packed-NVFP4 loader: same (m, k) -> (row, col) ownership as
        # the proven bf16 cp.async ``load_KV`` (identical tiled-copy geometry,
        # entry-pointer warp shuffles), but each 16-byte source word covers
        # 32 elements; every element is dequantized (e2m1 x e4m3 SF, both
        # exact in fp32) before the SIMT store into the swizzled tile.
        tPrXPtr = self.compute_X_ptr(K_or_V)
        tPrSfPtr = self.compute_SF_ptr(K_or_V)

        sX_pi = cute.group_modes(sX, 0, 1)
        head_dim = (
            self.head_dim_v_padded
            if const_expr(K_or_V == "V")
            else self.head_dim_padded
        )
        cX = cute.make_identity_tensor((self.n_block_size, head_dim))
        tXsX = self.gmem_thr_copy_KV.partition_D(sX_pi)
        tXcX = self.gmem_thr_copy_KV.partition_S(cX)
        # D-side identity partition: exact (row, col) of every destination
        # slot (the V view is transposed, so S-side coords do not match).
        tXdX = self.gmem_thr_copy_KV.partition_D(cX)

        gptr = self.gmem_threads_per_row
        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            row_valid = True
            x_ptr_i64 = utils.shuffle_sync(
                tPrXPtr[m // gptr], m % gptr, width=gptr
            )
            sf_ptr_i64 = utils.shuffle_sync(
                tPrSfPtr[m // gptr], m % gptr, width=gptr
            )
            x_gmem_ptr = cute.make_ptr(
                cutlass.Uint8, x_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            sf_gmem_ptr = cute.make_ptr(
                cutlass.Uint8, sf_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            mX_row = cute.make_tensor(
                cute.recast_ptr(x_gmem_ptr, dtype=cutlass.Uint64),
                cute.make_layout((head_dim // 16,)),
            )
            mX_sf = cute.make_tensor(
                cute.recast_ptr(sf_gmem_ptr, dtype=cutlass.Uint32),
                cute.make_layout((head_dim // 64,)),
            )
            for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                # Values land in a register fragment indexed identically to
                # the S-side identity partition (whose coordinates are the
                # logical (row, col) of the tile); autovec_copy then writes
                # the fragment into the swizzled SMEM tile through the
                # tiled-copy layout algebra, exactly like the bf16 path.
                frag = cute.make_fragment_like(tXsX[None, m, k])
                sc0 = tXcX[((0, 0), m, k)]
                row = sc0[0]
                row_valid = n_block * self.n_block_size + row < self.seqlen_k
                # Each fragment owns 16 *contiguous* columns starting at
                # ``col0`` (a multiple of 16) -- verified exhaustively by the
                # gmem-dump check -- so the whole fragment shares one packed
                # u64 word and one e4m3 scale byte.  This hoists the div/mod
                # (and their gmem loads) out of the 16-element loop.
                col0 = sc0[1]
                word_idx = col0 // 16
                packed64 = mX_row[word_idx]
                sf_word = mX_sf[word_idx // 4].to(cutlass.Int32)
                sf_byte = (sf_word >> ((word_idx % 4) * 8)) & 0xFF
                sf_frag = cute.make_rmem_tensor((1,), cutlass.Uint8)
                sf_frag[0] = sf_byte.to(cutlass.Uint8)
                sf_f16 = cute.make_tensor(
                    cute.recast_ptr(
                        sf_frag.iterator, dtype=cutlass.Float8E4M3FN
                    ),
                    cute.make_layout((1,)),
                )[0].to(cutlass.Float16)
                # Hardware fp4->fp16 converts (four cvt.rn.f16x2.e2m1x2 per
                # word); nibble i of the packed word is element i.  The fp16
                # multiply by the (broadcast, exact) e4m3 scale keeps at most
                # 6 significand bits, so the fp16 result is exact and the
                # fp16 -> bf16 convert is lossless: bit-identical to the
                # fp32 scalar reference.
                ssv_frag = cute.make_rmem_tensor((8,), cutlass.Float16)
                for j in cutlass.range_constexpr(8):
                    ssv_frag[j] = sf_f16
                ssv = ssv_frag.load()
                r_lo = cute.TensorSSA(
                    cute.arch.cvt_f4e2m1x8_to_f16x8(
                        packed64.to(cutlass.Uint32).ir_value()
                    ),
                    (8,),
                    cutlass.Float16,
                )
                r_hi = cute.TensorSSA(
                    cute.arch.cvt_f4e2m1x8_to_f16x8(
                        (packed64 >> 32).to(cutlass.Uint32).ir_value()
                    ),
                    (8,),
                    cutlass.Float16,
                )
                p_lo = r_lo * ssv
                p_hi = r_hi * ssv
                for e in cutlass.range_constexpr(
                    cute.size(frag, mode=[0, 0])
                ):
                    value = (
                        p_lo[e] if e < 8 else p_hi[e - 8]
                    ).to(self.compute_dtype)
                    frag[e] = value
                    if const_expr(self.dump_gmem is not None):
                        if const_expr(K_or_V == "K"):
                            seq_pos = n_block * self.n_block_size + row
                            self.dump_gmem[seq_pos, col0 + e] = value
                if row_valid:
                    cute.autovec_copy(frag, tXsX[None, m, k])
