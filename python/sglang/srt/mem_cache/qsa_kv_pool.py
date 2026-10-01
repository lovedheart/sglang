"""KV pools carrying the QSA sparse-attention indexer caches.

``QSATokenToKVPool`` (compressed, Qwen4-Exp) adds the per-token BF16 index-key
state, its RoPE coordinates, and the paged compressed-K cache on top of the
hybrid full/linear KV pool. ``QwenDSATokenToKVPool`` (tokenwise,
Qwen3Next-DSA) adds only the flat per-token index-K cache.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.qsa.metadata import qsa_ring_stride
from sglang.srt.mem_cache.memory_pool import GB, HybridLinearKVPool, MambaPool

logger = logging.getLogger(__name__)

# State layer IDs are serialized as uint32 by the disaggregation protocols.
# Reserve the value below PLE's request-wide sentinel for QSA's request-wide
# RoPE ring, which is shared by all full-attention layers.
QSA_ROPE_STATE_LAYER_ID = (1 << 32) - 2


def _index_k_bytes(*, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    return kv_heads * head_dim * dtype.itemsize


# ``--qsa-indexer-dtype`` choices. The pending key ring stays bf16 regardless.
QSA_INDEXER_DTYPE_CHOICES = ("auto", "bfloat16", "fp8_e4m3")


def resolve_qsa_indexer_dtype(name: str) -> torch.dtype:
    """Storage dtype of the compressed QSA indexer cache for a CLI value."""
    if name in ("auto", "bfloat16"):
        return torch.bfloat16
    if name == "fp8_e4m3":
        return torch.float8_e4m3fn
    raise ValueError(
        f"Unsupported --qsa-indexer-dtype {name!r}; expected one of "
        f"{QSA_INDEXER_DTYPE_CHOICES}"
    )


class QSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool with the minimal BF16 state required by simple QSA."""

    # Full-KV pages are a multiple of the compress ratio, so no group straddles pages;
    # ``compressed_slot = full_slot // ratio`` needs no ownership bookkeeping;
    # lifecycle rides the full-KV allocator and radix tree.
    # Full slot 0 is the reserved padding slot; compressed slot 0 is the inert dump.
    # Pending-ring dtype: the raw keys are averaged from here, so it stays bf16.
    index_state_dtype = torch.bfloat16
    # Compressed keys stay BF16 in the pool even under
    # SGLANG_QSA_USE_FP8_INDEXER: the SM120 paged FP8 kernel requires
    # block_kv 64 while compressed pages are ratio-shrunken, so a gathered
    # fp8 decode scores far below the TileLang paged BF16 path.  Only the
    # packed prefill scorer consumes fp8, cast per call from these BF16 pages
    # (see QSAIndexer.select_prefill_tokens).

    @classmethod
    def qsa_bytes_per_token(
        cls,
        *,
        kv_heads: int,
        head_dim: int,
        compress_ratio: int,
        num_layers: int,
        compressed_dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Per-token cost of the QSA index caches: the compressed keys only.

        Pre-compression state is a per-request ring of ``compress_ratio``
        slots (the pending group's members), not a per-token cache, so it
        does not price per token; its total is bounded by the request-slot
        count and stays outside this budget like the other per-request
        buffers.
        """
        index_k_bytes = _index_k_bytes(
            kv_heads=kv_heads, head_dim=head_dim, dtype=compressed_dtype
        )
        return index_k_bytes // compress_ratio * num_layers

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
        qsa_indexer_dtype: torch.dtype = torch.bfloat16,
    ):
        if page_size <= 1 or page_size % qsa_compress_ratio != 0:
            raise ValueError(
                "compressed QSA requires a paged full-KV cache with the page "
                "a multiple of the compress ratio (compressed slots are "
                f"full_slot // ratio): page_size={page_size}, "
                f"ratio={qsa_compress_ratio}. This needs the mamba "
                "extra-buffer strategy or "
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
        if qsa_indexer_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError(
                "QSA compressed indexer cache dtype must be bfloat16 or "
                f"float8_e4m3fn, got {qsa_indexer_dtype}"
            )
        # Storage dtype of the compressed keys and the index Q (the GEMM operands).
        self.qsa_compressed_dtype = qsa_indexer_dtype
        logger.info(
            "QSA compressed indexer cache dtype: %s (pending ring %s)",
            self.qsa_compressed_dtype,
            self.index_state_dtype,
        )
        state_size = size + page_size
        # Compressed slots mirror the full-KV slot space 1:ratio; the "page"
        # seen by the scoring kernels is one full-KV page's worth of groups.
        self.qsa_compressed_page_size = page_size // self.qsa_compress_ratio
        self.qsa_compressed_capacity = -(state_size // -self.qsa_compress_ratio)
        # Pre-compression index-K state is a per-request RING, not a
        # per-token cache: once a group's compressed key is written, its raw
        # members are never read again, and page-granular prefix sharing
        # keeps every extend chunk group-aligned, so the only state that
        # must survive a forward is the pending group's members -- at most
        # ``ratio`` tokens per request, addressed as
        # ``req_pool_idx * stride + position % stride`` (stride = 2*ratio via
        # qsa_ring_stride, so a speculative verify window cannot alias the
        # pending group it compresses). Request slot 0 is never allocated, so
        # ring rows [0, stride) double as the inert dump for tokens whose
        # group already compressed in the same forward.
        if num_request_slots <= 0:
            raise ValueError(
                f"QSA pending ring needs request slots, got {num_request_slots}"
            )
        self.qsa_num_request_slots = int(num_request_slots)
        ring_slots = self.qsa_num_request_slots * qsa_ring_stride(
            self.qsa_compress_ratio
        )
        # These buffers participate in Mooncake PD transfer just like the base
        # KV and Mamba pools.  Keep their allocation in the same memory-saver
        # and Mooncake custom-pool regions; otherwise MNNVL cannot resolve the
        # ordinary CUDA allocation when the first QSA state page is sent.
        allocation_pool = self.full_kv_pool
        with (
            allocation_pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(allocation_pool.custom_mem_pool)
                if allocation_pool.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            self.qsa_key_state_buffer_pool = [
                torch.zeros(
                    (
                        ring_slots,
                        self.qsa_index_kv_heads,
                        self.qsa_index_head_dim,
                    ),
                    dtype=self.index_state_dtype,
                    device=device,
                )
                for _ in full_attention_layer_ids
            ]
            # RoPE coordinates are layer-independent. Keep the exact Qwen4-Exp
            # MRoPE position of every incomplete key so compression can rotate
            # the pooled key with the group's real starting coordinate.
            self.qsa_rope_position_buffer = torch.zeros(
                (ring_slots, 3), dtype=torch.int64, device=device
            )
            # One contiguous allocation behind per-layer views: every layer's
            # compressed pages are addressable from a single base pointer.
            self.qsa_compressed_flat = torch.zeros(
                (
                    len(full_attention_layer_ids),
                    self.qsa_compressed_capacity
                    * self.qsa_index_kv_heads
                    * self.qsa_index_head_dim,
                ),
                dtype=self.qsa_compressed_dtype,
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
        # The indexer reads compressed keys before attention reads the full KV.
        self._wait_for_layer(layer_id)
        return self.qsa_compressed_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_compressed_k_buffer(
        self, layer_id: int, loc: torch.Tensor, compressed_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_compressed_k_buffer(layer_id)
        buffer[loc.long()] = compressed_k.to(buffer.dtype)

    @staticmethod
    def _get_paged_state_buf_infos(tensors, page_size: int):
        return (
            [tensor.data_ptr() for tensor in tensors],
            [tensor.nbytes for tensor in tensors],
            [tensor[0].nbytes * page_size for tensor in tensors],
        )

    def get_qsa_pending_state_buf_infos(self):
        """Per-request pending key-state and RoPE ring transfer buffers."""
        # A PP stage without a local QSA layer never writes the shared RoPE
        # ring.  Do not register it as a transfer source: otherwise that stage
        # can race with a QSA-owning stage and overwrite valid positions with
        # its zero-initialized or stale contents.
        if not self.full_attention_layer_id_mapping:
            return [], [], []
        tensors = [*self.qsa_key_state_buffer_pool, self.qsa_rope_position_buffer]
        return self._get_paged_state_buf_infos(
            tensors,
            qsa_ring_stride(self.qsa_compress_ratio),
        )

    def get_qsa_pending_state_layer_ids(self):
        """Global layer metadata for the compact QSA pending-state list."""
        if not self.full_attention_layer_id_mapping:
            return []
        return [
            *self.full_attention_layer_id_mapping.keys(),
            QSA_ROPE_STATE_LAYER_ID,
        ]

    def get_qsa_compressed_state_layer_ids(self):
        """Global layer metadata for the compact compressed-K list."""
        return list(self.full_attention_layer_id_mapping.keys())

    def get_qsa_compressed_state_buf_infos(self):
        """Per-full-page compressed-K transfer buffers.

        One full KV page maps to one compressed page because the full page size
        is an integer multiple of the compression ratio.
        """
        return self._get_paged_state_buf_infos(
            self.qsa_compressed_k_buffer_pool,
            self.qsa_compressed_page_size,
        )

    def qsa_hicache_regions(self, page_size: int):
        """KV-page-indexed uint8 views of the compressed-K cache for HiCache.

        Row ``i`` of each returned buffer is the byte block of compressed
        slots ``[i * page_size // ratio, (i + 1) * page_size // ratio)``, i.e.
        exactly the compressed keys belonging to full-KV page ``i``
        (``compressed_slot = full_slot // ratio`` keeps every page's groups
        contiguous). ``DeepSeekV4PagedHostPool`` maps a KV token index to a
        buffer row via ``index // page_size``, so these views ride the generic
        KV-derived sidecar transfer path with no extra index derivation.
        """
        ratio = self.qsa_compress_ratio
        if page_size % ratio != 0:
            raise ValueError(
                "QSA HiCache regions require page_size to be a multiple of "
                f"the compress ratio: page_size={page_size}, ratio={ratio}"
            )
        groups_per_page = page_size // ratio
        row_bytes = (
            groups_per_page
            * self.qsa_index_kv_heads
            * self.qsa_index_head_dim
            * self.qsa_compressed_dtype.itemsize
        )
        rows = self.qsa_compressed_capacity // groups_per_page
        buffers = [
            layer.view(torch.uint8)[: rows * row_bytes].view(rows, row_bytes)
            for layer in self.qsa_compressed_flat
        ]
        return buffers, row_bytes

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
