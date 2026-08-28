"""KV pools carrying the QSA sparse-attention indexer caches.

``QSATokenToKVPool`` (compressed, Qwen4-Exp) adds the per-token BF16 index-key
state, its RoPE coordinates, and the paged compressed-K cache on top of the
hybrid full/linear KV pool. ``QwenDSATokenToKVPool`` (tokenwise,
Qwen3Next-DSA) adds only the flat per-token index-K cache.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.mem_cache.memory_pool import GB, HybridLinearKVPool, MambaPool


def _index_k_bytes(*, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    return kv_heads * head_dim * dtype.itemsize


class QSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool with the minimal BF16 state required by simple QSA."""

    # DSV4-style compressed addressing: the full-KV allocator is paged with
    # the page a multiple of the compress ratio, so every compression
    # group's raw tokens are contiguous in one page and
    # ``compressed_slot = full_slot // compress_ratio`` is a stable,
    # collision-free mapping (any of the group's slots floor-divides to the
    # same value). The compressed cache therefore has exactly
    # ``full_slots // ratio`` slots, its lifecycle rides the full-KV
    # allocator and radix tree (page-granular sharing shares compressed
    # slots by arithmetic), and no ownership bookkeeping exists. Full slot 0
    # is the pools' reserved padding slot, so compressed slot 0 stays the
    # inert dump target for non-boundary rows.
    index_state_dtype = torch.bfloat16
    # FP8-indexer mode stores compressed keys as fp8_e4m3 (one byte per
    # element) plus one shared constant-1.0 fp32 scale vector; compressed keys
    # are RMS-normed, so a unit scale never overflows and is written once.
    qsa_fp8_state_dtype = torch.float8_e4m3fn

    @classmethod
    def qsa_bytes_per_token(
        cls, *, kv_heads: int, head_dim: int, compress_ratio: int, num_layers: int
    ) -> int:
        """Per-token cost of the QSA index caches: the compressed keys only.

        Pre-compression state is a per-request ring of ``compress_ratio``
        slots (the pending group's members), not a per-token cache, so it
        does not price per token; its total is bounded by the request-slot
        count and stays outside this budget like the other per-request
        buffers.
        """
        dtype = (
            cls.qsa_fp8_state_dtype
            if cls.qsa_use_fp8_indexer_enabled()
            else cls.index_state_dtype
        )
        index_k_bytes = _index_k_bytes(
            kv_heads=kv_heads, head_dim=head_dim, dtype=dtype
        )
        return index_k_bytes // compress_ratio * num_layers

    @classmethod
    def qsa_use_fp8_indexer_enabled(cls) -> bool:
        from sglang.srt.environ import envs

        return envs.SGLANG_QWEN_DSA_USE_FP8_INDEXER.get()


    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        qsa_index_kv_heads: int,
        qsa_index_head_dim: int,
        qsa_compress_ratio: int,
        qsa_token_topk: int,
        num_request_slots: int,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
    ):
        if page_size <= 1 or page_size % qsa_compress_ratio != 0:
            raise ValueError(
                "compressed QSA requires a paged full-KV cache with the page "
                "a multiple of the compress ratio (compressed slots are "
                f"full_slot // ratio): page_size={page_size}, "
                f"ratio={qsa_compress_ratio}. With MambaRadixCache this "
                "needs the mamba extra-buffer strategy or "
                "--disable-radix-cache (see the Qwen4-Exp arg overrides)."
            )
        # The base __init__ computes mem_usage through the overridden
        # get_kv_size_bytes before the QSA buffers exist; give them empty
        # placeholders first and recompute mem_usage at the end.
        self.qsa_key_state_buffer_pool = []
        self.qsa_compressed_k_buffer_pool = []
        self.qsa_rope_position_buffer = torch.empty(0)
        super().__init__(
            size=size,
            dtype=dtype,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            mamba_pool=mamba_pool,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            use_mla=False,
            start_layer=start_layer,
            full_kv_pool_class=full_kv_pool_class,
            quant_method=quant_method,
            post_capture_active=post_capture_active,
        )
        if (
            min(
                qsa_index_kv_heads,
                qsa_index_head_dim,
                qsa_compress_ratio,
                qsa_token_topk,
            )
            <= 0
        ):
            raise ValueError("QSA cache configuration values must be positive")
        if qsa_token_topk % qsa_compress_ratio != 0:
            raise ValueError("qsa_token_topk must be divisible by qsa_compress_ratio")
        self.qsa_compress_ratio = int(qsa_compress_ratio)
        self.qsa_index_head_dim = int(qsa_index_head_dim)
        self.qsa_index_kv_heads = int(qsa_index_kv_heads)
        self.qsa_token_topk = int(qsa_token_topk)
        self.qsa_block_topk = self.qsa_token_topk // self.qsa_compress_ratio
        state_size = size + page_size
        # Compressed slots mirror the full-KV slot space 1:ratio; the "page"
        # seen by the scoring kernels is one full-KV page's worth of groups.
        self.qsa_compressed_page_size = page_size // self.qsa_compress_ratio
        self.qsa_compressed_capacity = -(state_size // -self.qsa_compress_ratio)
        from sglang.srt.environ import envs

        self.qsa_use_fp8_indexer = envs.SGLANG_QWEN_DSA_USE_FP8_INDEXER.get()
        if self.qsa_use_fp8_indexer and self.qsa_index_kv_heads != 1:
            raise ValueError(
                "FP8-indexer compressed QSA requires index_kv_heads = 1 "
                f"(one fp8 scale per compressed key), got {self.qsa_index_kv_heads}"
            )
        # Pre-compression index-K state is a per-request RING, not a
        # per-token cache: once a group's compressed key is written, its raw
        # members are never read again, and page-granular prefix sharing
        # keeps every extend chunk group-aligned, so the only state that
        # must survive a forward is the pending group's members -- at most
        # ``ratio`` tokens per request, addressed as
        # ``req_pool_idx * ratio + position % ratio``. Request slot 0 is
        # never allocated, so ring rows [0, ratio) double as the inert dump
        # for tokens whose group already compressed in the same forward.
        if num_request_slots <= 0:
            raise ValueError(
                f"QSA pending ring needs request slots, got {num_request_slots}"
            )
        self.qsa_num_request_slots = int(num_request_slots)
        ring_slots = self.qsa_num_request_slots * self.qsa_compress_ratio
        self.qsa_key_state_buffer_pool = [
            torch.zeros(
                (ring_slots, self.qsa_index_kv_heads, self.qsa_index_head_dim),
                dtype=self.index_state_dtype,
                device=device,
            )
            for _ in full_attention_layer_ids
        ]
        # RoPE coordinates are layer-independent.  Keep the exact Qwen4-Exp MRoPE
        # position of every pending key so compression can rotate the pooled
        # key with the group's real starting coordinate.
        self.qsa_rope_position_buffer = torch.zeros(
            (ring_slots, 3), dtype=torch.int64, device=device
        )
        # One contiguous allocation behind per-layer views: every layer's
        # compressed pages are addressable from a single base pointer.  In
        # FP8-indexer mode the same view shape carries fp8_e4m3 payloads and
        # the BF16 flat twin is not allocated (compressed keys are
        # RMS-normed, so the DeepGEMM per-key scales are a constant 1.0 that
        # each call site materializes over its gathered rows).  The fused
        # payload+scale page layout of the tokenwise pool is not needed here:
        # the SM120 paged kernel requires block_kv 64 while compressed pages
        # are ratio-shrunken, so the compressed indexer scores through the
        # packed kernel on gathers.
        if self.qsa_use_fp8_indexer:
            self.qsa_compressed_k_buffer_pool = [
                torch.zeros(
                    (
                        self.qsa_compressed_capacity,
                        self.qsa_index_kv_heads,
                        self.qsa_index_head_dim,
                    ),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                for _ in full_attention_layer_ids
            ]
            self.qsa_compressed_flat = torch.empty(
                0, dtype=self.index_state_dtype, device=device
            )
        else:
            self.qsa_compressed_flat = torch.zeros(
                (
                    len(full_attention_layer_ids),
                    self.qsa_compressed_capacity
                    * self.qsa_index_kv_heads
                    * self.qsa_index_head_dim,
                ),
                dtype=self.index_state_dtype,
                device=device,
            )
            self.qsa_compressed_k_buffer_pool = [
                self.qsa_compressed_flat[layer_offset].view(
                    self.qsa_compressed_capacity,
                    self.qsa_index_kv_heads,
                    self.qsa_index_head_dim,
                )
                for layer_offset in range(len(full_attention_layer_ids))
            ]
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_qsa_key_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_key_state_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_key_state_buffer(
        self, layer_id: int, loc: torch.Tensor, token_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_key_state_buffer(layer_id)
        buffer[loc.long()] = token_k.to(buffer.dtype)

    def set_qsa_rope_position_buffer(
        self, loc: torch.Tensor, positions: torch.Tensor
    ) -> None:
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                f"QSA RoPE positions must be [tokens] or [3, tokens], got {positions.shape}"
            )
        self.qsa_rope_position_buffer[loc.long()] = positions.transpose(0, 1)

    def get_qsa_rope_position_buffer(self, loc: torch.Tensor) -> torch.Tensor:
        return self.qsa_rope_position_buffer[loc.long()]

    def get_qsa_compressed_k_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_compressed_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_compressed_k_buffer(
        self, layer_id: int, loc: torch.Tensor, compressed_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_compressed_k_buffer(layer_id)
        buffer[loc.long()] = compressed_k.to(buffer.dtype)

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        qsa_k_size = (
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_key_state_buffer_pool
            )
            + sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_compressed_k_buffer_pool
            )
            + self.qsa_rope_position_buffer.numel() * 8
        )
        return k_size + qsa_k_size, v_size


class QwenDSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool carrying the per-token index-K cache of tokenwise QSA.

    Only the BF16 reference layout is ported: one flat
    ``[size + page_size, index_kv_heads, index_head_dim]`` buffer per
    full-attention (DSA) layer, addressed by raw KV slots.  The FP8
    deep_gemm layout is intentionally not ported yet and the caller-side
    fast paths fail loudly instead of silently degrading.
    """

    index_state_dtype = torch.bfloat16

    @classmethod
    def qsa_bytes_per_token(
        cls, *, kv_heads: int, head_dim: int, num_layers: int
    ) -> int:
        return (
            _index_k_bytes(
                kv_heads=kv_heads, head_dim=head_dim, dtype=cls.index_state_dtype
            )
            * num_layers
        )

    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        qsa_index_kv_heads: int,
        qsa_index_head_dim: int,
        qsa_token_budget: int,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
    ):
        if page_size != 64:
            raise ValueError(
                "tokenwise QSA requires KV-cache page_size 64 for its paged "
                f"indexer buffer, got {page_size}"
            )
        self.dsa_index_k_buffer_pool = []
        super().__init__(
            size=size,
            dtype=dtype,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            mamba_pool=mamba_pool,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            use_mla=False,
            start_layer=start_layer,
            full_kv_pool_class=full_kv_pool_class,
            quant_method=quant_method,
            post_capture_active=post_capture_active,
        )
        if qsa_index_kv_heads != 1:
            raise ValueError(
                f"tokenwise QSA requires index_kv_heads = 1 (MQA), got "
                f"{qsa_index_kv_heads}"
            )
        if min(qsa_index_kv_heads, qsa_index_head_dim, qsa_token_budget) <= 0:
            raise ValueError("QSA cache configuration values must be positive")
        self.qsa_compress_ratio = 1
        self.qsa_index_kv_heads = int(qsa_index_kv_heads)
        self.qsa_index_head_dim = int(qsa_index_head_dim)
        self.qsa_token_topk = int(qsa_token_budget)
        self.qsa_block_topk = int(qsa_token_budget)
        state_size = size + page_size
        # FP8 paged layout for the DeepGEMM decode indexer
        # (SGLANG_QWEN_DSA_USE_FP8_INDEXER): fused
        # [pages, 64, 1, head_dim + 4] uint8 per layer -- head_dim fp8_e4m3
        # bytes followed by 4 fp32 scale bytes whose value is a constant 1.0
        # (written once; indexer K is RMS-normed so scale=1 never overflows).
        # Slots map to (slot // 64) page, offset slot % 64, so the block
        # table is slot_table[:, ::64] // 64.  In this mode the BF16 twin is
        # not allocated: every read gathers from the fp8 pages.
        from sglang.srt.environ import envs

        self.qsa_use_fp8_indexer = envs.SGLANG_QWEN_DSA_USE_FP8_INDEXER.get()
        self.qsa_index_page_size = page_size
        num_pages = state_size // page_size
        assert state_size % page_size == 0, "index-K pages must align"
        hd = self.qsa_index_head_dim
        if self.qsa_use_fp8_indexer:
            self.dsa_index_k_buffer_pool = None
            paged = []
            for _ in full_attention_layer_ids:
                fused = torch.zeros(
                    (num_pages, page_size, 1, hd + 4),
                    dtype=torch.uint8,
                    device=device,
                )
                fused.view(num_pages, -1)[:, page_size * hd :].view(
                    torch.float32
                ).fill_(1.0)
                paged.append(fused)
            self.dsa_index_paged_pool = paged
        else:
            self.dsa_index_k_buffer_pool = [
                torch.zeros(
                    (state_size, self.qsa_index_kv_heads, self.qsa_index_head_dim),
                    dtype=self.index_state_dtype,
                    device=device,
                )
                for _ in full_attention_layer_ids
            ]
            self.dsa_index_paged_pool = None
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def _qsa_index_slot_offsets(self, loc: torch.Tensor) -> torch.Tensor:
        """Flat byte offsets of each slot's fp8 payload inside a fused page.

        Page layout (DeepGEMM fused-KV contract): first ``page_size *
        head_dim`` bytes of data (token i at i * head_dim), then
        ``page_size * 4`` fp32 scale bytes.
        """
        head_dim = self.qsa_index_head_dim
        page_stride = self.qsa_index_page_size * (head_dim + 4)
        loc = loc.long()
        return (loc // self.qsa_index_page_size) * page_stride + (
            loc % self.qsa_index_page_size
        ) * head_dim

    def set_dsa_index_k_buffer(
        self, layer_id: int, loc: torch.Tensor, index_k: torch.Tensor
    ) -> None:
        layer = self._transfer_full_attention_id(layer_id)
        if self.dsa_index_paged_pool is not None:
            fused = self.dsa_index_paged_pool[layer]
            head_dim = self.qsa_index_head_dim
            offs = self._qsa_index_slot_offsets(loc).unsqueeze(1) + torch.arange(
                head_dim, dtype=torch.int64, device=loc.device
            )
            k_fp8 = index_k.to(torch.float8_e4m3fn).reshape(-1, head_dim)
            fused.view(-1)[offs] = k_fp8.view(torch.uint8)
            return
        buffer = self.dsa_index_k_buffer_pool[layer]
        buffer[loc.long()] = index_k.to(buffer.dtype)

    def get_dsa_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        if self.dsa_index_k_buffer_pool is None:
            raise RuntimeError(
                "tokenwise QSA BF16 index-K buffer is absent in FP8-indexer "
                "mode; gather via get_dsa_index_k_fp8 instead"
            )
        return self.dsa_index_k_buffer_pool[self._transfer_full_attention_id(layer_id)]

    def get_dsa_index_paged(self, layer_id: int) -> torch.Tensor:
        if self.dsa_index_paged_pool is None:
            raise RuntimeError("QSA FP8 paged index-K is disabled")
        return self.dsa_index_paged_pool[self._transfer_full_attention_id(layer_id)]

    def get_dsa_index_k_fp8(self, layer_id: int, slots: torch.Tensor) -> torch.Tensor:
        """Gather head_dim fp8 payloads for ``slots`` ([N] -> [N, head_dim])."""
        fused = self.get_dsa_index_paged(layer_id)
        head_dim = self.qsa_index_head_dim
        idx = self._qsa_index_slot_offsets(slots).unsqueeze(1) + torch.arange(
            head_dim, dtype=torch.int64, device=slots.device
        )
        return fused.view(-1)[idx].view(torch.float8_e4m3fn)

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        if getattr(self, "dsa_index_paged_pool", None) is not None:
            dsa_k_size = sum(
                tensor.numel() for tensor in self.dsa_index_paged_pool
            )
        else:
            dsa_k_size = sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.dsa_index_k_buffer_pool
            )
        return k_size + dsa_k_size, v_size
