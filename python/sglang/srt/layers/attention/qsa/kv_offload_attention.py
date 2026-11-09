"""QSA sparse attention over the K/V host offload (``--enable-qsa-kv-offload``).

Under the offload, the device K/V pool of the QSA layers holds only the hot-cache
and scratch rows of ``mem_cache/qsa_kv_offload.py``. These functions are the
backend's offload paths: store the new K/V into device rows and through to the
host, gather a prefill's cached context from the host, and run decode / target
verify attention through the offload's per-forward device-row table. In the
debug verify mode the full pool stays on the GPU and every offloaded read is
compared with the full pool's.
"""

from __future__ import annotations

from typing import Tuple

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.sparse_attn import (
    pack_qsa_prefill_kv,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)


class QSAKVGatherSource(msgspec.Struct, frozen=True):
    """Where sparse attention reads K/V: row ``i``, column ``j`` reads
    ``k_buffer[slot_table[row_requests[i], indices[i, j]]]`` for
    ``0 <= indices[i, j] < sequence_lengths[i]``."""

    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    slot_table: torch.Tensor
    row_requests: torch.Tensor
    sequence_lengths: torch.Tensor
    # A quantized (FP4) offload additionally addresses its device scale planes.
    k_sf_buffer: torch.Tensor | None = None
    v_sf_buffer: torch.Tensor | None = None


def _pack_gathered_kv(source: QSAKVGatherSource, indices: torch.Tensor):
    """(valid counts, packed K, packed V) of the decode gather, for verification."""
    rows, width = indices.shape
    device = indices.device
    counts = torch.empty(rows, dtype=torch.int32, device=device)
    cu = torch.arange(rows + 1, dtype=torch.int32, device=device) * width
    packed_k = torch.empty(
        (rows * width, *source.k_buffer.shape[1:]),
        dtype=source.k_buffer.dtype,
        device=device,
    )
    packed_v = torch.empty_like(packed_k)
    qwen_sparse_valid_counts_triton(
        source.sequence_lengths, indices, counts, rows, width
    )
    qwen_sparse_kv_extraction_compact_triton(
        source.k_buffer,
        source.v_buffer,
        source.slot_table,
        source.row_requests,
        indices,
        source.sequence_lengths,
        cu,
        packed_k,
        packed_v,
        rows,
        width,
        zero_fill_cols=width,
    )
    return counts, packed_k, packed_v


def store_offloaded_kv(*, backend, layer, forward_batch, k, v) -> None:
    """Store a forward's new K/V into the offload's device rows (decode: the
    tail ring, extend: the scratch) and write it through to the host."""
    offload = backend._kv_offload
    mode = forward_batch.forward_mode
    if mode.is_decode() or backend._is_speculative_paged_mode(mode):
        metadata = backend._resolve_metadata(forward_batch)
        rows = offload.new_token_rows(
            row_seq_lens=metadata.sequence_lengths,
            row_req=backend._row_requests(
                forward_batch=forward_batch, metadata=metadata
            ),
        )
    else:
        rows = offload.extend_rows(forward_batch.out_cache_loc.numel())
    pool = backend.token_to_kv_pool
    layer_idx = pool.full_attention_layer_index(layer.layer_id)
    if offload.verify:
        if layer_idx == 0:
            offload.snapshot_mapping()
        backend.kv_layers.layer(layer.layer_id).store(
            layer, forward_batch.out_cache_loc, k, v
        )
        offload.mirror(
            layer_idx=layer_idx,
            rows=rows,
            loc=forward_batch.out_cache_loc,
            pool_k=pool.full_kv_pool.k_buffer[layer_idx],
            pool_v=pool.full_kv_pool.v_buffer[layer_idx],
            pool_ksf=pool.full_kv_pool.k_scale_buffer[layer_idx]
            if offload.quantized
            else None,
            pool_vsf=pool.full_kv_pool.v_scale_buffer[layer_idx]
            if offload.quantized
            else None,
        )
        return
    backend.kv_layers.layer(layer.layer_id).store(layer, rows, k, v)
    offload.write_through(
        layer_idx=layer_idx, rows=rows, loc=forward_batch.out_cache_loc
    )


def gather_offloaded_context(
    *,
    backend,
    layer,
    forward_batch,
    sequence_lens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_k: int,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Every context token of each request, in ``cu_seqlens_k`` order, read
    from the host (the layout ``pack_qsa_prefill_kv`` builds from the pool)."""
    device = sequence_lens.device
    owner = torch.repeat_interleave(
        torch.arange(sequence_lens.numel(), device=device),
        sequence_lens.long(),
        output_size=total_k,
    )
    positions = torch.arange(total_k, device=device) - cu_seqlens_k[:-1].long()[owner]
    offload = backend._kv_offload
    slots = offload.req_to_token[
        forward_batch.req_pool_indices.long()[owner], positions
    ]
    pool = backend.token_to_kv_pool
    layer_idx = pool.full_attention_layer_index(layer.layer_id)
    extend_lens = forward_batch.extend_seq_lens
    hot_tail = (
        envs.SGLANG_QSA_OFFLOAD_HOT_TAIL_GATHER.get()
        and not offload.verify
        # The masked row copies have data-dependent shapes; a captured graph
        # needs the fixed-shape all-host gather.
        and not is_in_breakable_cuda_graph()
        and not torch.cuda.is_current_stream_capturing()
        and extend_lens is not None
        and extend_lens.numel() == sequence_lens.numel()
    )
    if hot_tail:
        # The current chunk's rows are in the device mirror scratch right now
        # (store_offloaded_kv wrote them this layer); read them there instead
        # of waiting on their host round-trip. Token (b, p) of the chunk owns
        # scratch row scratch_base + (rows taken by earlier requests) + its
        # offset in the chunk. Same cu order, bitwise same content.
        seq = sequence_lens.long()
        e = extend_lens.long()
        starts = seq[owner] - e[owner]
        tail_mask = positions >= starts
        rows_before = torch.cumsum(e, 0) - e
        tail_rows = (offload.layout.scratch_base + rows_before[owner] + (positions - starts))[
            tail_mask
        ]
        k, v, k_sf, v_sf = offload.gather_context_hot_tail(
            layer_idx=layer_idx,
            slots=slots,
            tail_mask=tail_mask,
            tail_rows=tail_rows,
        )
    else:
        k, v, k_sf, v_sf = offload.gather_context(layer_idx=layer_idx, slots=slots)
    if k_sf is not None:
        if offload.verify:
            # Packed rows compare bit for bit against the full pool's rows of
            # the same logical slots; both sides quantized identically.
            pool_kp, pool_vp, pool_ksf, pool_vsf = pool.get_raw_kv_buffer(
                layer.layer_id
            )
            offload.check(
                what="prefill context",
                layer_idx=layer_idx,
                actual=(k, v, k_sf, v_sf),
                expected=(
                    pool_kp[slots],
                    pool_vp[slots],
                    pool_ksf[slots],
                    pool_vsf[slots],
                ),
            )
        # Quantized host tier: dequantize the gathered packed rows here; the
        # prefill attention runs on plain K/V either way.
        return backend.kv_cache_quant_method.dequantize_gathered_kv(
            k, k_sf, v, v_sf, layer.layer_id, output_dtype
        )
    pool_k = pool.layer_router.read(layer.layer_id).key()
    pool_v = pool.layer_router.read(layer.layer_id).value()
    packed = (
        k.view(pool_k.dtype).to(output_dtype),
        v.view(pool_k.dtype).to(output_dtype),
    )
    if offload.verify:
        expected = pack_qsa_prefill_kv(
            k=pool_k,
            v=pool_v,
            req_to_token=offload.req_to_token,
            req_indices=forward_batch.req_pool_indices,
            cu_k=cu_seqlens_k,
            total_k=total_k,
            max_k=int(sequence_lens.max()),
            output_dtype=output_dtype,
        )
        offload.check(
            what="prefill context",
            layer_idx=layer_idx,
            actual=packed,
            expected=expected,
        )
    return packed


def forward_offloaded_attention(
    *,
    backend,
    q: torch.Tensor,
    layer,
    forward_batch,
    topk_indices: torch.Tensor,
    trtllm_decode,
) -> torch.Tensor:
    """Decode / target verify over the offload's per-forward device-row table."""
    pool = backend.token_to_kv_pool
    offload = backend._kv_offload
    layer_idx = pool.full_attention_layer_index(layer.layer_id)
    metadata = backend._resolve_metadata(forward_batch)
    positions = backend._expand_block_indices(topk_indices, metadata)
    row_requests = backend._row_requests(forward_batch=forward_batch, metadata=metadata)
    num_rows, width = positions.shape
    rows_per_request = num_rows // forward_batch.batch_size
    rows = offload.resolve_rows(
        layer_idx=layer_idx,
        positions=positions,
        row_seq_lens=metadata.sequence_lengths,
        row_req=row_requests,
        rows_per_request=rows_per_request,
    )
    # Valid positions lead each row, so column ids keep the packing order.
    columns = torch.arange(width, dtype=torch.int32, device=rows.device)
    indices = torch.where(rows >= 0, columns, -1)
    quantized = backend.kv_cache_quant_method is not None
    if quantized:
        # The device rows of a quantized offload are packed rows already; a
        # quantized layer_router read would dequantize the whole pool.
        k_source = offload.dev_k[layer_idx]
        v_source = offload.dev_v[layer_idx]
        k_sf_source = offload.dev_ksf[layer_idx]
        v_sf_source = offload.dev_vsf[layer_idx]
    else:
        pool_k = pool.layer_router.read(layer.layer_id).key()
        pool_v = pool.layer_router.read(layer.layer_id).value()
        k_source = offload.dev_k[layer_idx].view(pool_k.dtype)
        v_source = offload.dev_v[layer_idx].view(pool_v.dtype)
        k_sf_source = v_sf_source = None
    source = QSAKVGatherSource(
        k_buffer=k_source,
        v_buffer=v_source,
        slot_table=rows,
        row_requests=torch.arange(num_rows, device=rows.device),
        sequence_lengths=torch.full(
            (num_rows,), width, dtype=torch.int32, device=rows.device
        ),
        k_sf_buffer=k_sf_source,
        v_sf_buffer=v_sf_source,
    )
    if offload.verify:
        if quantized:
            valid = indices >= 0
            dev_rows = rows.long()[valid]
            ref_loc = offload.req_to_token[
                row_requests.long()[:, None], positions.long().clamp(min=0)
            ][valid]
            pool_kp, pool_vp, pool_ksf, pool_vsf = pool.get_raw_kv_buffer(
                layer.layer_id
            )
            offload.check(
                what="decode rows",
                layer_idx=layer_idx,
                actual=(
                    k_source[dev_rows],
                    v_source[dev_rows],
                    offload.dev_ksf[layer_idx][dev_rows],
                    offload.dev_vsf[layer_idx][dev_rows],
                ),
                expected=(
                    pool_kp[ref_loc],
                    pool_vp[ref_loc],
                    pool_ksf[ref_loc],
                    pool_vsf[ref_loc],
                ),
            )
        else:
            reference = QSAKVGatherSource(
                k_buffer=pool_k,
                v_buffer=pool_v,
                slot_table=offload.req_to_token,
                row_requests=row_requests,
                sequence_lengths=metadata.sequence_lengths,
            )
            # CUDA-graph padding rows resolve to no device row by design.
            real_rows = offload.real_rows(
                num_rows=num_rows, rows_per_request=rows_per_request
            )
            reference_positions = torch.where(real_rows[:, None], positions, -1)
            offload.check(
                what="decode gather",
                layer_idx=layer_idx,
                actual=_pack_gathered_kv(source=source, indices=indices),
                expected=_pack_gathered_kv(
                    source=reference,
                    indices=reference_positions.to(torch.int32).contiguous(),
                ),
            )
    indices = indices.to(torch.int32).contiguous()
    if quantized:
        if backend._fused_fp4_gather:
            # The fused FP4 reads address the packed device rows directly
            # through the source table; the buffers only carry metadata.
            k_buffer = torch.empty(
                (0, k_source.shape[1], k_source.shape[2] * 2),
                dtype=backend._fp4_attn_dtype,
                device=q.device,
            )
            v_buffer = k_buffer
        else:
            k_buffer, v_buffer = backend._gather_topk_rows_fp4(
                layer, forward_batch, metadata, indices, source=source
            )
    else:
        k_buffer = source.k_buffer
        v_buffer = source.v_buffer
    if trtllm_decode is not None:
        return backend._forward_trtllm_sparse(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            layer=layer,
            forward_batch=forward_batch,
            metadata=metadata,
            topk_indices=indices,
            trtllm_decode=trtllm_decode,
            source=source,
            gathered_rows=quantized,
        )
    return backend._forward_varlen_sparse(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        layer=layer,
        forward_batch=forward_batch,
        metadata=metadata,
        topk_indices=indices,
        source=source,
    )
