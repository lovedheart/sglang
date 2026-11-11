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

import logging
from typing import List, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.sparse_attn import (
    pack_qsa_prefill_kv,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
)
from sglang.srt.mem_cache.qsa_kv_offload import _copy_rows
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)

logger = logging.getLogger(__name__)


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


def _slabs_for(offload, req_ids, sequence_lens, extend_lens, layer_idx, slot0s):
    """Per-request slabs, or None per request where the slab cannot serve.

    A slab is dropped and re-reserved when it cannot be this request's:
    its cursor for THIS layer (layers advance independently) passed the
    prior, or its stamped first row differs from the slot table's — a
    reused request slot with a radix hit keeps prior > 0, so cursor
    depth alone cannot tell tenants apart (decode releases do not run
    inside replayed graphs). The drop precedes the reserve, so a dead
    slab never runs the grow/evict path. A slab that a later reserve
    evicted from under its request aliases another's rows, so every
    entry is re-checked against the allocator's table.
    """
    slabs = []
    for req, seq, ext, slot0 in zip(req_ids, sequence_lens, extend_lens, slot0s):
        prior = seq - ext
        slab = offload._slabs.get(int(req))
        if slab is not None and (
            slab.cursors[layer_idx] > prior or slab.slot0 != slot0
        ):
            offload.slab_release(req)
        slab = offload.slab_reserve(req, seq)
        if slab is not None and max(slab.cursors) == 0:
            slab.slot0 = slot0
        slabs.append(slab)
    return [
        slab if slab is not None and offload._slabs.get(int(req)) is slab else None
        for req, slab in zip(req_ids, slabs)
    ]


def _gather_union_from_slab(
    *,
    offload,
    layer_idx: int,
    slabs: list,
    owners: torch.Tensor,
    positions: torch.Tensor,
    sequence_lens: List[int],
    extend_lens: List[int],
    req_ids: List[int],
):
    """Stage every selected row in the device slab, then gather the union
    rows entirely on the device.

    Rows the slab does not hold yet cross PCIe exactly once (host to slab),
    the chunk's rows move from the scratch; afterwards the union reads slab
    rows, so a chunk with a 95% union rate still costs no more PCIe than
    its new rows. Bytes equal the host store's row for row, so the packed
    result matches ``gather_context`` exactly."""
    device = positions.device
    k, v, k_sf, v_sf = offload._gather_output(layer_idx, positions.numel())
    quant = offload.quantized
    ext_before = 0
    for b, (seq, ext, slab) in enumerate(zip(sequence_lens, extend_lens, slabs)):
        cached = slab.cursors[layer_idx]
        prior = seq - ext
        if prior > cached:
            _copy_rows(
                k_dst=offload.slab_k[layer_idx],
                v_dst=offload.slab_v[layer_idx],
                dst=torch.arange(slab.base + cached, slab.base + prior, device=device),
                k_src=offload.host_k[layer_idx],
                v_src=offload.host_v[layer_idx],
                src=offload.req_to_token[req_ids[b], cached:prior].long(),
                ks_dst=offload.slab_ksf[layer_idx] if quant else None,
                vs_dst=offload.slab_vsf[layer_idx] if quant else None,
                ks_src=offload.host_ksf[layer_idx] if quant else None,
                vs_src=offload.host_vsf[layer_idx] if quant else None,
            )
        if ext:
            _copy_rows(
                k_dst=offload.slab_k[layer_idx],
                v_dst=offload.slab_v[layer_idx],
                dst=torch.arange(slab.base + prior, slab.base + seq, device=device),
                k_src=offload.dev_k[layer_idx],
                v_src=offload.dev_v[layer_idx],
                src=torch.arange(
                    offload.layout.scratch_base + ext_before,
                    offload.layout.scratch_base + ext_before + ext,
                    device=device,
                ),
                ks_dst=offload.slab_ksf[layer_idx] if quant else None,
                vs_dst=offload.slab_vsf[layer_idx] if quant else None,
                ks_src=offload.dev_ksf[layer_idx] if quant else None,
                vs_src=offload.dev_vsf[layer_idx] if quant else None,
            )
        slab.cursors[layer_idx] = seq
        ext_before += ext
    _copy_rows(
        k_dst=k,
        v_dst=v,
        dst=torch.arange(positions.numel(), dtype=torch.int64, device=device),
        k_src=offload.slab_k[layer_idx],
        v_src=offload.slab_v[layer_idx],
        src=torch.tensor([slab.base for slab in slabs], device=device)[owners]
        + positions,
        ks_dst=k_sf,
        vs_dst=v_sf,
        ks_src=offload.slab_ksf[layer_idx] if quant else None,
        vs_src=offload.slab_vsf[layer_idx] if quant else None,
    )
    return k, v, k_sf, v_sf


def gather_offloaded_context_union(
    *,
    backend,
    layer,
    forward_batch,
    topk_indices: torch.Tensor,
    extend_lens: List[int],
    sequence_lens: List[int],
    max_seq: int,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``gather_offloaded_context`` bytes restricted to the rows the
    chunk's top-k references.

    The sparse-prefill kernel walks ``indices`` per query, so rows no query
    selected never load; the per-request selection bitmap packs the union
    (row-major, the order ``indices`` remaps into) and every query keeps its
    column order and count, making the attention output bit-identical to the
    dense gather. Returns the packed K/V (dequantized like the dense path),
    the compact ``cu_seqlens_k`` and the remapped indices (``-1`` padding
    preserved; ``kv_lens`` stays the true context lengths so the kernel's
    visible-prefix truncation is unchanged)."""
    offload = backend._kv_offload
    pool = backend.token_to_kv_pool
    layer_idx = pool.full_attention_layer_index(layer.layer_id)
    device = topk_indices.device
    num_sequences = len(sequence_lens)
    # Query-row owner from the (possibly DP-padded) top-k row count, not the
    # extend lengths: padded rows past the indexer's count still map to the
    # request that owns the last extend row.
    extend_end = torch.cumsum(
        torch.tensor(extend_lens, dtype=torch.int32, device=device), 0
    )
    q_owner = torch.searchsorted(
        extend_end,
        torch.arange(topk_indices.shape[0], dtype=torch.int32, device=device),
        right=True,
    ).clamp(max=max(num_sequences - 1, 0))
    entry_owner = q_owner.view(-1, *([1] * (topk_indices.ndim - 1))).expand(
        topk_indices.shape
    )
    selected = topk_indices >= 0
    chosen = torch.zeros((num_sequences, max_seq), dtype=torch.bool, device=device)
    chosen[entry_owner[selected], topk_indices[selected].long()] = True
    counts = chosen.sum(1, dtype=torch.int32)
    cu_seqlens_k = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            counts.cumsum(0, dtype=torch.int32),
        ]
    )
    # Local row of a selection = selections of its request before it.
    rank = chosen.cumsum(1, dtype=torch.int32) - 1
    owners, positions = chosen.nonzero(as_tuple=True)
    slots = offload.req_to_token[
        forward_batch.req_pool_indices.long()[owners], positions
    ].long()
    # The chunk's own rows are in the mirror scratch (stored this layer);
    # read them there, exactly as the dense path's hot tail does.
    starts = torch.tensor(
        [s - e for s, e in zip(sequence_lens, extend_lens)],
        dtype=torch.int32,
        device=device,
    )[owners]
    before = torch.tensor(
        [sum(extend_lens[:i]) for i in range(num_sequences)],
        dtype=torch.int32,
        device=device,
    )[owners]
    tail_mask = positions >= starts
    tail_rows = (offload.layout.scratch_base + before + (positions - starts))[tail_mask]
    # NOTE: capture-safety (extend-only, no graph) is enforced by the one
    # caller; any new caller must gate this block the same way.
    slabs = None
    req_ids = None
    if offload.slab_rows > 0 and not offload.verify:
        req_ids = forward_batch.req_pool_indices.tolist()
        slot0s = offload.req_to_token[forward_batch.req_pool_indices, 0].tolist()
        slabs = _slabs_for(
            offload, req_ids, sequence_lens, extend_lens, layer_idx, slot0s
        )
        if any(slab is None for slab in slabs):
            slabs = None
    if slabs is not None:
        k, v, k_sf, v_sf = _gather_union_from_slab(
            offload=offload,
            layer_idx=layer_idx,
            slabs=slabs,
            owners=owners,
            positions=positions,
            sequence_lens=sequence_lens,
            extend_lens=extend_lens,
            req_ids=req_ids,
        )
    elif tail_mask.any():
        k, v, k_sf, v_sf = offload.gather_context_hot_tail(
            layer_idx=layer_idx,
            slots=slots,
            tail_mask=tail_mask,
            tail_rows=tail_rows,
        )
    else:
        k, v, k_sf, v_sf = offload.gather_context(layer_idx=layer_idx, slots=slots)
    # One line per forward (the first QSA layer): the union rate that decides
    # whether this gather beats the dense one. The device->host read stays
    # behind this test; other layers must not sync.
    if layer_idx == 0 and envs.SGLANG_QSA_OFFLOAD_UNION_GATHER_LOG.get():
        total_selected = int(cu_seqlens_k[-1])
        logger.info(
            "QSA union gather: %d/%d context rows (%.1f%%)",
            total_selected,
            sum(sequence_lens),
            100.0 * total_selected / max(sum(sequence_lens), 1),
        )
    remapped = torch.where(
        selected, rank[entry_owner, topk_indices.clamp(min=0)], topk_indices
    ).contiguous()
    if k_sf is not None:
        return (
            *backend.kv_cache_quant_method.dequantize_gathered_kv(
                k, k_sf, v, v_sf, layer.layer_id, output_dtype
            ),
            cu_seqlens_k,
            remapped,
        )
    pool_k = pool.layer_router.read(layer.layer_id).key()
    return (
        k.view(pool_k.dtype).to(output_dtype),
        v.view(pool_k.dtype).to(output_dtype),
        cu_seqlens_k,
        remapped,
    )


def _gather_prefill_context_slab(
    *, offload, layer_idx: int, slots: torch.Tensor, forward_batch
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Packed context whose prefix rows below the slab cursor are served from
    the device slab instead of the host store.

    Rows between the cursor and the chunk are re-staged from the host (a
    slab evicted by pool pressure), and the current chunk's rows come from
    the mirror scratch this layer just stored; the new rows then go into
    the slab, so the next chunk finds them without a PCIe round-trip. The
    layout and bytes match ``gather_context`` exactly: the slab holds byte
    copies of the same host rows in cu order."""
    dev = slots.device
    seq_lens = [int(x) for x in forward_batch.seq_lens_cpu]
    ext_lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
    reqs = forward_batch.req_pool_indices.tolist()
    total_k = sum(seq_lens)
    k = torch.empty(
        (total_k, *offload.dev_k[layer_idx].shape[1:]),
        dtype=offload.dev_k[layer_idx].dtype,
        device=dev,
    )
    v = torch.empty_like(k)
    quant = offload.quantized
    k_sf = (
        torch.empty(
            (total_k, *offload.dev_ksf[layer_idx].shape[1:]),
            dtype=offload.dev_ksf[layer_idx].dtype,
            device=dev,
        )
        if quant
        else None
    )
    v_sf = torch.empty_like(k_sf) if quant else None
    planes = (
        {
            "ks_dst": k_sf,
            "vs_dst": v_sf,
            "ks_src": offload.host_ksf[layer_idx],
            "vs_src": offload.host_vsf[layer_idx],
        }
        if quant
        else {}
    )

    def span(a: int, b: int) -> torch.Tensor:
        return torch.arange(a, b, dtype=torch.int64, device=dev)

    slabs = _slabs_for(
        offload,
        reqs,
        seq_lens,
        ext_lens,
        layer_idx,
        # First row per request; a reused slot's new tenant differs here.
        offload.req_to_token[reqs, 0].tolist(),
    )
    ext_seen = 0
    start = 0
    for seq, ext, slab in zip(seq_lens, ext_lens, slabs):
        prior = seq - ext
        if slab is None:
            _copy_rows(
                k_dst=k,
                v_dst=v,
                dst=span(start, start + seq),
                k_src=offload.host_k[layer_idx],
                v_src=offload.host_v[layer_idx],
                src=slots[start : start + seq].long(),
                **planes,
            )
        else:
            cached = slab.cursors[layer_idx]
            if cached:
                _copy_rows(
                    k_dst=k,
                    v_dst=v,
                    dst=span(start, start + cached),
                    k_src=offload.slab_k[layer_idx],
                    v_src=offload.slab_v[layer_idx],
                    src=span(slab.base, slab.base + cached),
                    **(
                        {
                            "ks_src": offload.slab_ksf[layer_idx],
                            "vs_src": offload.slab_vsf[layer_idx],
                            "ks_dst": k_sf,
                            "vs_dst": v_sf,
                        }
                        if quant
                        else {}
                    ),
                )
            if prior > cached:
                _copy_rows(
                    k_dst=k,
                    v_dst=v,
                    dst=span(start + cached, start + prior),
                    k_src=offload.host_k[layer_idx],
                    v_src=offload.host_v[layer_idx],
                    src=slots[start + cached : start + prior].long(),
                    **planes,
                )
            if ext:
                _copy_rows(
                    k_dst=k,
                    v_dst=v,
                    dst=span(start + prior, start + seq),
                    k_src=offload.dev_k[layer_idx],
                    v_src=offload.dev_v[layer_idx],
                    src=(
                        offload.layout.scratch_base + ext_seen + span(0, ext)
                    ),
                    **(
                        {
                            "ks_src": offload.dev_ksf[layer_idx],
                            "vs_src": offload.dev_vsf[layer_idx],
                            "ks_dst": k_sf,
                            "vs_dst": v_sf,
                        }
                        if quant
                        else {}
                    ),
                )
            offload.slab_append(
                layer_idx=layer_idx,
                slab=slab,
                packed=(k, v, k_sf, v_sf),
                # The packed rows NOT yet in the slab start at cached, not
                # at the request's slice start.
                packed_start=start + cached,
                count=seq - cached,
            )
            slab.cursors[layer_idx] = seq
        ext_seen += ext
        start += seq
    return k, v, k_sf, v_sf


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
    slab_on = (
        offload.slab_rows > 0
        and not offload.verify
        and not is_in_breakable_cuda_graph()
        and not torch.cuda.is_current_stream_capturing()
        and forward_batch.extend_seq_lens_cpu is not None
    )
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
    if slab_on:
        k, v, k_sf, v_sf = _gather_prefill_context_slab(
            offload=offload,
            layer_idx=layer_idx,
            slots=slots,
            forward_batch=forward_batch,
        )
    elif hot_tail:
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
    if (
        offload.slab_rows > 0
        and layer_idx == 0
        and forward_batch.forward_mode.is_decode()
        # Replayed graphs run no Python; decode-only release would miss the
        # step. Fresh-extend resets (below) are the authoritative reclaim.
        and not torch.cuda.is_current_stream_capturing()
    ):
        # These requests left extend: their prefill slabs are dead weight,
        # decode reads through the ring/LRU. (Idempotent; target verify
        # keeps the slab, its request is done extending already.)
        for req in forward_batch.req_pool_indices.tolist():
            offload.slab_release(int(req))
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
