"""Host offload of the QSA full-attention K/V with a per-request GPU hot cache.

The full K/V of every QSA layer lives in pinned host memory, indexed by the
logical KV slot (the allocator index stored in ``req_to_token``). The allocator,
radix tree, chunked prefill and the compressed indexer keep addressing logical
slots, so a radix prefix shared by several requests shares its host rows too.

The GPU K/V pool of the QSA layers holds only device rows:

* request slot ``r`` owns rows ``[r * region, (r + 1) * region)`` with
  ``region = (hot_blocks + 2) * compress_ratio``: a two-block tail ring (block
  ``b`` in ring slot ``b % 2``) for the newest tokens, then ``hot_blocks``
  least-recently-used slots of one compressed block (``compress_ratio`` tokens)
  each;
* an extend scratch above the request regions.

Every forward writes its new K/V to device rows and through to the host. A
decode or target-verify forward maps each selected logical position to a device
row, copying missing blocks from the host, and attention gathers through that
row table instead of ``req_to_token``. A block copy is valid only while
``req_to_token`` still maps its positions to the logical slots it was filled
from; the radix cache rewrites a running request's mapping when it deduplicates
an inserted prefix.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import msgspec
import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    _cuda_host_unregister,
    alloc_with_host_register,
)
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

# The block of a request's first new token and the next one: up to
# ``compress_ratio`` new tokens span at most two blocks.
TAIL_BLOCKS = 2
GB = 1e9


class _PrefillSlab(msgspec.Struct):
    """One request's device-side slab of its prefill context. ``cursors``
    counts, per layer, the prefix rows already in the slab: layer ``l``'s
    gather of the next chunk trusts rows below ``cursors[l]`` and appends
    the rest, so layers advance independently within a chunk."""

    base: int
    length: int
    cursors: list[int]
    # First context row's host slot when the slab was stamped: a reused
    # request slot with the same cursor depth but a different slot table
    # is a different tenant, and its rows must not be trusted.
    slot0: int = -1


class _QsaGatherPlan(msgspec.Struct):
    """One extend forward's context rows, resolved once at forward entry.

    ``slots`` mirrors the per-layer index ``gather_offloaded_context`` reads
    out of ``req_to_token``: the scheduler (that table's only writer) runs on
    this thread between forwards, so the rows stay what every layer's gather
    will read. Plane 0 is prefetched right after the plan is built (its
    runway: the embedding and the first GDN run) and each layer's gather
    triggers the plane after it. ``seqs``/``exts`` mirror the request
    lengths, so the chunk tail refills per request without a host sync.
    """

    seqs: list[int]  # context length per request, as the gather sees them
    exts: list[int]  # rows this forward stores itself, per request
    num_rows: int  # sum(seqs): context rows per layer gather
    slots: torch.Tensor  # (num_rows,) int64 device, the slot of each row
    next_layer: int = 0  # the next plane whose gather may be prefetched


class _QsaPrefetch(msgspec.Struct):
    """A scheduled host -> device gather of one plane's context K/V rows.

    The copy runs on the prefetch stream, chained behind the compute stream
    at issue time; ``done`` is recorded when it ends. The rows holding this
    forward's own chunk are copied too but always overwritten from the
    device scratch at consume time, so their (possibly stale) host bytes
    never reach the attention. The scale planes of a quantized K/V are not
    prefetched: they are too small for the fused transfer kernel and are
    copied inline at consume, exactly as the un-prefetched gather does.
    """

    layer_idx: int
    k: torch.Tensor
    v: torch.Tensor
    done: torch.cuda.Event


class QSAKVOffloadLayout(msgspec.Struct, frozen=True):
    """Device-row layout of an offloaded QSA K/V pool."""

    compress_ratio: int
    hot_blocks: int
    num_request_slots: int
    scratch_rows: int
    token_topk: int
    rows_per_request: int

    @property
    def region_rows(self) -> int:
        return (self.hot_blocks + TAIL_BLOCKS) * self.compress_ratio

    @property
    def tail_rows(self) -> int:
        return TAIL_BLOCKS * self.compress_ratio

    @property
    def scratch_base(self) -> int:
        return self.num_request_slots * self.region_rows

    @property
    def device_rows(self) -> int:
        return self.scratch_base + self.scratch_rows


def compute_offload_layout(
    *,
    compress_ratio: int,
    token_topk: int,
    hot_blocks: int,
    num_request_slots: int,
    rows_per_request: int,
    max_extend_tokens: int,
    page_size: int,
) -> QSAKVOffloadLayout:
    """Layout for ``num_request_slots`` request slots (``req_to_token`` rows).

    ``rows_per_request`` is the query rows of one request in a forward (the
    verify rows under speculative decoding, else 1); their selected blocks are
    resolved together and must fit in the hot cache at once.
    """
    if page_size % compress_ratio:
        raise ValueError(
            f"QSA K/V offload needs a page size ({page_size}) that is a multiple "
            f"of the compress ratio ({compress_ratio})"
        )
    if not 1 <= rows_per_request <= compress_ratio:
        raise ValueError(
            "QSA K/V offload supports at most compress_ratio "
            f"({compress_ratio}) query rows per request, got {rows_per_request}"
        )
    needed = rows_per_request * (token_topk // compress_ratio)
    if hot_blocks < needed:
        raise ValueError(
            f"--qsa-kv-offload-hot-blocks {hot_blocks} is below {needed}: "
            f"{rows_per_request} query row(s) per request x "
            f"{token_topk // compress_ratio} selected blocks must fit at once"
        )
    # Page-aligned so the device pool size is a whole number of pages.
    scratch = max_extend_tokens + 2 * page_size
    regions = num_request_slots * (hot_blocks + TAIL_BLOCKS) * compress_ratio
    scratch += -(regions + scratch) % page_size
    return QSAKVOffloadLayout(
        compress_ratio=compress_ratio,
        hot_blocks=hot_blocks,
        num_request_slots=num_request_slots,
        scratch_rows=scratch,
        token_topk=token_topk,
        rows_per_request=rows_per_request,
    )


def offload_layout_from_config(
    *, hf_config, num_request_slots: int, page_size: int
) -> QSAKVOffloadLayout:
    """Layout from the resolved server configuration and the model's QSA profile."""
    from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
    from sglang.srt.runtime_context import (
        get_memory,
        get_schedule,
        max_speculative_num_draft_tokens,
    )

    profile = parse_qsa_profile(hf_config)
    schedule = get_schedule()
    return compute_offload_layout(
        compress_ratio=profile.compress_ratio,
        token_topk=profile.budget,
        hot_blocks=get_memory().qsa_kv_offload_hot_blocks,
        num_request_slots=num_request_slots,
        rows_per_request=max_speculative_num_draft_tokens() or 1,
        max_extend_tokens=max(
            schedule.chunked_prefill_size or 0, schedule.max_prefill_tokens or 0
        ),
        page_size=page_size,
    )


class QSAKVOffloadBudget(msgspec.Struct, frozen=True):
    """GPU and host cost of the offload, for the KV pool sizing."""

    # K/V bytes per logical token of the offloaded layers (all of them).
    offloaded_bytes_per_token: int
    # Device rows per request slot, and the scratch plus padding page.
    region_rows: int
    fixed_rows: int
    # Hot-cache tables per request slot, all layers.
    table_bytes_per_slot: int
    host_size_gb: float
    page_size: int
    # Verify mode keeps the full GPU pool: the whole cell stays on the GPU.
    verify: bool = False

    def slot_bytes(self) -> int:
        return self.region_rows * self.offloaded_bytes_per_token + (
            self.table_bytes_per_slot
        )

    def gpu_cell(self, cell_bytes: int) -> int:
        cell = cell_bytes - (0 if self.verify else self.offloaded_bytes_per_token)
        if cell <= 0:
            raise ValueError(f"QSA K/V offload: per-token GPU cost {cell} B")
        return cell

    def host_tokens(self) -> int:
        if self.host_size_gb <= 0:
            raise ValueError("--qsa-kv-offload-host-size must be positive")
        host_rows = int(self.host_size_gb * GB) // self.offloaded_bytes_per_token
        return host_rows - self.page_size

    def max_total_tokens(
        self, *, available_bytes: int, cell_bytes: int, num_request_slots: int
    ) -> int:
        """Logical KV pool size for ``num_request_slots`` request slots.

        Per logical token the GPU keeps the cell minus the offloaded K/V (the
        QSA index and any draft pool); each request slot costs its device rows
        and tables. The host budget caps the pool at its K/V rows.
        """
        fixed = num_request_slots * self.slot_bytes() + (
            self.fixed_rows * self.offloaded_bytes_per_token
        )
        if available_bytes <= fixed:
            raise ValueError(
                f"QSA K/V offload: the hot cache of {num_request_slots} request "
                f"slots ({fixed / 2**30:.2f} GiB) exceeds the "
                f"{available_bytes / 2**30:.2f} GiB KV budget; lower "
                "--qsa-kv-offload-hot-blocks or --max-running-requests"
            )
        gpu_tokens = (available_bytes - fixed) // self.gpu_cell(cell_bytes)
        tokens = min(gpu_tokens, self.host_tokens())
        logger.info(
            "QSA K/V offload sizing: %d request slots, hot cache %.2f GiB, GPU "
            "allows %d tokens, host %.1f GB allows %d tokens",
            num_request_slots,
            fixed / 2**30,
            gpu_tokens,
            self.host_size_gb,
            self.host_tokens(),
        )
        return tokens // self.page_size * self.page_size

    def tokens_with_unused_slots(
        self, *, max_total_tokens: int, cell_bytes: int, unused_slots: int
    ) -> int:
        """Pool size once ``unused_slots`` budgeted request slots are not built:
        their GPU bytes hold logical tokens instead, up to the host budget."""
        extra = unused_slots * self.slot_bytes() // self.gpu_cell(cell_bytes)
        tokens = min(max_total_tokens + extra, self.host_tokens())
        return tokens // self.page_size * self.page_size


def offload_table_bytes_per_slot(
    *, num_layers: int, hot_blocks: int, compress_ratio: int, context_len: int
) -> int:
    """Upper bound of the per-request-slot hot-cache tables of either resolve:
    block->slot (int32 per block of the context), per-slot block, age, LRU
    order, marks and source records, and the ring records."""
    per_layer = (
        (context_len // compress_ratio + 2 * TAIL_BLOCKS) * 4
        + hot_blocks * (8 + 8 + 2 * 4 + 4 + 4 * compress_ratio)
        + TAIL_BLOCKS * (8 + 4 * compress_ratio)
        + 8
    )
    return num_layers * per_layer


class HotTailSplit(msgspec.Struct, frozen=True):
    """Row split of one hot-tail gather, resolved entirely from host counts.

    A request's masked rows (``position >= chunk start``) are the suffix of
    its gathered context rows, so the masked ``gather_context_hot_tail``
    indices are contiguous per-request ranges: the host part first, the chunk
    part after. ``host_idx`` / ``tail_idx`` are those output row ids and
    ``tail_rows`` the tail slots' mirror rows (masked order); the device
    tensors are the CPU arrays lifted by one pinned copy each.
    """

    host_idx: torch.Tensor  # (host rows,) device row ids read from the host store
    host_idx_cpu: torch.Tensor  # the same ids, for the CPU scale-plane leg
    tail_idx: torch.Tensor  # (tail rows,) device row ids read from the mirror
    tail_rows: torch.Tensor  # (tail rows,) mirror scratch rows, masked order
    host_count: int
    tail_count: int


_SF_STAGE_DEPTH = 4  # pinned stage slots per plane, see _sf_host_stage


def _pinned_h2d(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type != "cuda":
        return cpu.to(device)
    return cpu.pin_memory().to(device, non_blocking=True)


def make_hot_tail_split(
    *,
    counts: List[int],
    host_counts: List[int],
    tail_rows_cpu: torch.Tensor,
    device: torch.device,
) -> HotTailSplit:
    """Build one forward's hot-tail gather indices on the host, from the
    per-request row counts and tail rows; no device readback."""
    assert len(counts) == len(host_counts)
    host_np, tail_np = [], []
    total = 0
    for rows, host in zip(counts, host_counts):
        if not 0 <= host <= rows:
            raise ValueError(
                f"QSA K/V offload: {host} host rows for a {rows}-row request"
            )
        host_np.append(np.arange(total, total + host, dtype=np.int64))
        tail_np.append(np.arange(total + host, total + rows, dtype=np.int64))
        total += rows
    assert total == sum(host_counts) + tail_rows_cpu.numel()
    host_idx_cpu = torch.from_numpy(
        np.concatenate(host_np) if counts else np.empty(0, np.int64)
    )
    return HotTailSplit(
        host_idx=_pinned_h2d(host_idx_cpu, device),
        host_idx_cpu=host_idx_cpu,
        tail_idx=_pinned_h2d(
            torch.from_numpy(
                np.concatenate(tail_np) if counts else np.empty(0, np.int64)
            ),
            device,
        ),
        tail_rows=_pinned_h2d(tail_rows_cpu, device),
        host_count=int(host_idx_cpu.numel()),
        tail_count=int(tail_rows_cpu.numel()),
    )


def _copy_plane(
    k_dst: torch.Tensor,
    v_dst: torch.Tensor,
    dst: torch.Tensor,
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """One K/V plane's rows ``src[i]`` -> ``dst[i]`` across devices. The TMA
    transfer kernel wants rows that are a multiple of one 32-lane package
    group (128 B); the small scale planes of a quantized K/V fall back to
    index copies, which is fine: they only travel on eager reference paths."""
    row_bytes = k_dst[0].numel() * k_dst.element_size()
    if dst.is_cuda and row_bytes % 128 == 0:
        from sglang.kernels.ops.kvcache.hicache import transfer_hicache_one_layer

        transfer_hicache_one_layer(
            k_cache_dst=k_dst,
            v_cache_dst=v_dst,
            indices_dst=dst,
            k_cache_src=k_src,
            v_cache_src=v_src,
            indices_src=src,
            element_dim=k_dst[0].numel(),
        )
        return
    gathered_k = k_src.index_select(0, src.to(k_src.device))
    gathered_v = v_src.index_select(0, src.to(v_src.device))
    k_dst.index_copy_(0, dst.to(k_dst.device), gathered_k.to(k_dst.device))
    v_dst.index_copy_(0, dst.to(v_dst.device), gathered_v.to(v_dst.device))


def _copy_rows(
    *,
    k_dst: torch.Tensor,
    v_dst: torch.Tensor,
    dst: torch.Tensor,
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    src: torch.Tensor,
    ks_dst: Optional[torch.Tensor] = None,
    vs_dst: Optional[torch.Tensor] = None,
    ks_src: Optional[torch.Tensor] = None,
    vs_src: Optional[torch.Tensor] = None,
) -> None:
    """K/V rows ``src[i]`` -> ``dst[i]`` between device and pinned host buffers.
    A quantized K/V passes its per-block scale planes along; they travel with
    the data rows of the same logical slots."""
    if dst.numel() == 0:
        return
    _copy_plane(k_dst, v_dst, dst, k_src, v_src, src)
    if ks_dst is not None:
        _copy_plane(ks_dst, vs_dst, dst, ks_src, vs_src, src)


class QSAKVHostOffload:
    """Host K/V store and GPU hot cache of one QSA K/V pool.

    On CUDA the resolve is the fused, CUDA-graph-safe one
    (``qsa_kv_offload_kernels``). The torch resolve here is the reference: eager,
    with host synchronizations; ``SGLANG_USE_QSA_KV_OFFLOAD_TORCH_RESOLVE``
    selects it on CUDA.
    """

    def __init__(
        self,
        *,
        layout: QSAKVOffloadLayout,
        k_buffers: List[torch.Tensor],
        v_buffers: List[torch.Tensor],
        host_rows: int,
        k_sf_buffers: Optional[List[torch.Tensor]] = None,
        v_sf_buffers: Optional[List[torch.Tensor]] = None,
        pin_memory: bool = True,
        verify: bool = False,
    ):
        self.layout = layout
        # Debug: the full K/V pool stays on the GPU; see check().
        self.verify = verify
        self._verify_logged = 0.0
        self.cr = layout.compress_ratio
        self.hot_blocks = layout.hot_blocks
        self.region = layout.region_rows
        self.tail_rows = layout.tail_rows
        self.dev_k = k_buffers
        self.dev_v = v_buffers
        # A quantized (NVFP4) pool adds row-aligned per-block scale planes.
        self.quantized = k_sf_buffers is not None
        self.dev_ksf = k_sf_buffers
        self.dev_vsf = v_sf_buffers
        self.num_layers = len(k_buffers)
        self.device = k_buffers[0].device
        row_shape = tuple(k_buffers[0].shape[1:])
        if tuple(v_buffers[0].shape[1:]) != row_shape:
            raise ValueError("QSA K/V offload needs equal K and V row shapes")
        if k_buffers[0].shape[0] < layout.device_rows:
            raise ValueError(
                f"device K/V buffers have {k_buffers[0].shape[0]} rows, the "
                f"offload layout needs {layout.device_rows}"
            )
        self.host_rows = host_rows
        dtype = k_buffers[0].dtype

        # Pageable memory of the exact size, page-locked with cudaHostRegister:
        # torch's pinned caching allocator rounds each block up to a power of two.
        def alloc(row_shape):
            return alloc_with_host_register(
                dims=(self.num_layers, host_rows, *row_shape),
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
                allocator=HostTensorAllocator(),
            )

        self.host_k, self.host_v = alloc(row_shape), alloc(row_shape)
        if k_sf_buffers is not None:
            sf_row_shape = tuple(k_sf_buffers[0].shape[1:])
            if tuple(v_sf_buffers[0].shape[1:]) != sf_row_shape:
                raise ValueError("QSA K/V offload needs equal K and V scale row shapes")
            if k_sf_buffers[0].dtype != dtype or v_sf_buffers[0].dtype != dtype:
                raise ValueError("QSA K/V offload scale planes must share the dtype")
            self.host_ksf, self.host_vsf = alloc(sf_row_shape), alloc(sf_row_shape)
        else:
            self.host_ksf = self.host_vsf = None
        self.pin_memory = pin_memory
        self.host_bytes = 2 * self.host_k.numel() * self.host_k.element_size() + (
            0
            if self.host_ksf is None
            else 2 * self.host_ksf.numel() * self.host_ksf.element_size()
        )
        logger.info(
            "QSA K/V offload: %d layers, host K/V %.2f GB (%d rows), %d device "
            "rows (%d request slots x %d hot blocks + %d scratch rows)",
            self.num_layers,
            self.host_bytes / GB,
            host_rows,
            layout.device_rows,
            layout.num_request_slots,
            layout.hot_blocks,
            layout.scratch_rows,
        )
        self.req_to_token: Optional[torch.Tensor] = None
        # Layer-ahead gather prefetch (see SGLANG_QSA_OFFLOAD_PREFETCH_*):
        # the per-forward plan, the one outstanding copy, and the event the
        # next host write must wait on (see the top of write_through).
        self._gather_plan: Optional[_QsaGatherPlan] = None
        self._prefetched: Optional[_QsaPrefetch] = None
        self._prefetch_write_gate: Optional[torch.cuda.Event] = None
        self._prefetch_stream = None  # the "qsa_kv_h2d" lease, taken eagerly
        # Rotating pinned staging for the CPU scale-plane leg of
        # gather_context_hot_tail_split, keyed "k" / "v"; each entry holds
        # _SF_STAGE_DEPTH stage slots and the HtoD event per slot.
        self._sf_stages: Dict[str, dict] = {}
        # The fused resolve reads the host store through raw UVA pointers,
        # validated on CUDA only.
        self.use_fused = (
            self.device.type == "cuda"
            and not is_hip()
            and not envs.SGLANG_USE_QSA_KV_OFFLOAD_TORCH_RESOLVE.get()
        )
        self.fused = None
        self._resolve_errors_host = torch.zeros(
            1, dtype=torch.int64, pin_memory=self.use_fused
        )
        self._step = 0
        if verify:
            # [mismatching checks, checks]
            self._verify_counts = torch.zeros(2, dtype=torch.int64, device=self.device)
        self._arange = torch.arange(
            max(layout.scratch_rows, self.cr), dtype=torch.int64, device=self.device
        )
        # Chunked-prefill context slab (see SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS):
        # a device byte-copy of each extending request's context rows, so the
        # next chunk appends its rows instead of re-gathering the whole
        # prefix from the pinned host. Pool exhausted -> that request's chunk
        # uses the all-host gather; correctness never depends on the slab.
        self.slab_rows = (
            0 if verify else envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.get()
        )
        self.slab_grow = max(layout.scratch_rows, 4096)
        if self.slab_rows > 0:

            def device_planes(shape):
                return torch.empty(
                    (self.num_layers, self.slab_rows, *shape),
                    dtype=dtype,
                    device=self.device,
                )

            self.slab_k = device_planes(row_shape)
            self.slab_v = device_planes(row_shape)
            self.slab_ksf = device_planes(sf_row_shape) if self.quantized else None
            self.slab_vsf = device_planes(sf_row_shape) if self.quantized else None
            logger.info(
                "QSA K/V offload: prefill slab %d device rows (%.2f GB)",
                self.slab_rows,
                (
                    2 * self.slab_k.numel() * self.slab_k.element_size()
                    + (
                        0
                        if self.slab_ksf is None
                        else 2 * self.slab_ksf.numel() * self.slab_ksf.element_size()
                    )
                )
                / GB,
            )
        else:
            self.slab_k = self.slab_v = self.slab_ksf = self.slab_vsf = None
        # req_pool_idx -> slab, and the pool's free spans (base, length).
        self._slabs: dict[int, _PrefillSlab] = {}
        self._slab_spans: list[Tuple[int, int]] = (
            [(0, self.slab_rows)] if self.slab_rows > 0 else []
        )

    def mirror(
        self,
        layer_idx: int,
        rows: torch.Tensor,
        loc: torch.Tensor,
        pool_k: torch.Tensor,
        pool_v: torch.Tensor,
        pool_ksf: Optional[torch.Tensor] = None,
        pool_vsf: Optional[torch.Tensor] = None,
    ) -> None:
        """Verify mode: the new K/V the full pool holds at ``loc`` -> device
        rows ``rows`` and host rows ``loc`` (no host sync: graph-capturable)."""
        loc, rows = loc.long(), rows.long()
        self.dev_k[layer_idx][rows] = pool_k[loc]
        self.dev_v[layer_idx][rows] = pool_v[loc]
        if pool_ksf is not None:
            self.dev_ksf[layer_idx][rows] = pool_ksf[loc]
            self.dev_vsf[layer_idx][rows] = pool_vsf[loc]
        self.write_through(layer_idx=layer_idx, rows=rows, loc=loc)

    def check(self, what: str, layer_idx: int, actual, expected) -> None:
        """Verify mode: count offloaded tensors that differ from the full pool's,
        bit for bit, on the device (graph-capturable); ``poll_verify`` raises."""
        for got, ref in zip(actual, expected):
            self._verify_counts[0] += (
                got.view(torch.uint8) != ref.view(torch.uint8)
            ).any()
        self._verify_counts[1] += 1

    def poll_verify(self) -> None:
        """Verify mode: raise if any check so far differed (host sync)."""
        mismatches, checks = self._verify_counts.tolist()
        if mismatches:
            raise RuntimeError(
                f"QSA K/V offload verify: {mismatches} of {checks} checks differ "
                "from the full pool"
            )
        now = time.monotonic()
        if now - self._verify_logged >= 10:
            self._verify_logged = now
            logger.info("QSA K/V offload verify: %d checks, 0 mismatches", checks)

    def destroy(self) -> None:
        """Unpin the host K/V (see HostKVCache.destroy). Idempotent."""
        if self._prefetch_stream is not None:
            # A last copy may still be reading the host planes about to be
            # unpinned; join it (there is no later write to ride the gate).
            self._prefetch_stream.synchronize()
            self._prefetched = None
            self._prefetch_write_gate = None
        if self.verify:
            self._verify_logged = 0.0
            self.poll_verify()
        if self.fused is not None and int(self.fused.errors):
            logger.error(
                "QSA K/V offload: %d selected K/V columns had no device row",
                int(self.fused.errors),
            )
        if self.pin_memory:
            for buffer in (self.host_k, self.host_v, self.host_ksf, self.host_vsf):
                if buffer is not None:
                    _cuda_host_unregister(buffer)
        self.pin_memory = False

    def bind(self, req_to_token: torch.Tensor) -> None:
        """Allocate the hot-cache tables for this ``req_to_token``."""
        if self.req_to_token is not None:
            return
        num_slots, width = req_to_token.shape
        if num_slots != self.layout.num_request_slots:
            raise ValueError(
                f"req_to_token has {num_slots} request slots, the QSA K/V offload "
                f"was built for {self.layout.num_request_slots}"
            )
        # Verify mode reads one snapshot of the mapping per forward
        # (snapshot_mapping); serving reads the live table, as upstream does.
        self._live_req_to_token = req_to_token
        self.req_to_token = req_to_token.clone() if self.verify else req_to_token
        self.width = width
        self.max_blocks = width // self.cr + TAIL_BLOCKS
        if self.use_fused:
            from sglang.srt.mem_cache.qsa_kv_offload_kernels import QSAKVFusedResolve

            self.fused = QSAKVFusedResolve(
                num_request_slots=num_slots,
                hot_blocks=self.hot_blocks,
                compress_ratio=self.cr,
                token_topk=self.layout.token_topk,
                width=width,
                region_rows=self.region,
                dev_k=self.dev_k,
                dev_v=self.dev_v,
                host_k=self.host_k,
                host_v=self.host_v,
                dev_ksf=self.dev_ksf,
                dev_vsf=self.dev_vsf,
                host_ksf=self.host_ksf,
                host_vsf=self.host_vsf,
                max_rows_per_request=self.layout.rows_per_request,
            )
            return
        layers, hot, cr, dev = self.num_layers, self.hot_blocks, self.cr, self.device

        def table(*shape, dtype=torch.int64):
            return torch.full(shape, -1, dtype=dtype, device=dev)

        # Block -> LRU slot, and per LRU slot its block, last use and the
        # logical slots its rows were filled from.
        self.blk_slot = table(layers, num_slots, self.max_blocks, dtype=torch.int32)
        self.slot_blk = table(layers, num_slots, hot)
        self.slot_age = table(layers, num_slots, hot)
        self.slot_src = table(layers, num_slots, hot, cr, dtype=torch.int32)
        # Tail ring: the block in each ring slot, the logical slots its rows
        # were filled from, and the end of the positions written into it.
        self.ring_blk = table(layers, num_slots, TAIL_BLOCKS)
        self.ring_src = table(layers, num_slots, TAIL_BLOCKS, cr, dtype=torch.int32)
        self.ring_end = table(layers, num_slots)

    def snapshot_mapping(self) -> None:
        """Verify mode: copy the live ``req_to_token`` once per forward. The
        scheduler rewrites a running request's mapping (radix dedup in
        ``checkpoint``) on its own stream while a forward may run, so the
        resolve and the full-pool reference must read the same copy
        (graph-capturable)."""
        if self.verify:
            self.req_to_token.copy_(self._live_req_to_token)

    def set_num_real(self, num_requests: int) -> None:
        """Requests of the next forward; CUDA-graph padding past them is skipped.
        A single device fill: graph-replay metadata prep may be captured."""
        if self.fused is not None:
            self.fused.num_real.fill_(num_requests)

    def real_rows(self, *, num_rows: int, rows_per_request: int) -> torch.Tensor:
        """Device mask of the forward's query rows that belong to real requests,
        not CUDA-graph padding (no host sync: graph-capturable)."""
        rows = torch.arange(num_rows, device=self.device)
        if self.fused is None:
            return rows >= 0
        return rows < self.fused.num_real * rows_per_request

    @property
    def graph_safe(self) -> bool:
        return self.use_fused

    def begin_forward(
        self,
        req_to_token: torch.Tensor,
        reset_req_pool_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Per-forward setup; an extend passes its requests to reset."""
        self._gather_plan = None
        self._prefetched = None
        if (
            self.fused is None
            and self.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "The QSA K/V offload resolve runs eagerly; launch with "
                "--disable-cuda-graph"
            )
        self.bind(req_to_token)
        if reset_req_pool_indices is not None:
            self.reset(reset_req_pool_indices)
        if self.fused is not None:
            self._check_resolve_errors()
        if self.verify:
            self.poll_verify()

    def _check_resolve_errors(self) -> None:
        """Raise if the fused resolve left a selected column without a device row
        (it would not be attended). Reads the copy started at the previous eager
        forward, so it adds no host sync."""
        if int(self._resolve_errors_host[0]):
            raise RuntimeError(
                f"QSA K/V offload: {int(self._resolve_errors_host[0])} selected K/V "
                "columns had no device row"
            )
        self._resolve_errors_host.copy_(self.fused.errors, non_blocking=True)

    # ---- extend --------------------------------------------------------------

    def reset(self, req_pool_indices: torch.Tensor) -> None:
        """Empty the hot cache and tail ring of these request slots.

        Every extend calls this for its requests (new request, radix hit or
        the next prefill chunk); exact because the host holds every token.
        """
        if self.fused is not None:
            self.fused.reset(req_pool_indices)
            return
        idx = req_pool_indices.long()
        for table in (
            self.blk_slot,
            self.slot_blk,
            self.slot_age,
            self.slot_src,
            self.ring_blk,
            self.ring_src,
            self.ring_end,
        ):
            table.index_fill_(1, idx, -1)

    def extend_rows(self, num_tokens: int) -> torch.Tensor:
        """Device scratch rows that hold this extend's new K/V."""
        if num_tokens > self.layout.scratch_rows:
            raise RuntimeError(
                f"QSA K/V offload: extend of {num_tokens} tokens exceeds the "
                f"{self.layout.scratch_rows}-row scratch"
            )
        return self.layout.scratch_base + self._arange[:num_tokens]

    def write_through(
        self, layer_idx: int, rows: torch.Tensor, loc: torch.Tensor
    ) -> None:
        """Device rows ``rows`` -> host rows ``loc`` (their logical slots)."""
        if self._prefetch_write_gate is not None:
            # The scheduler may have freed a row from the radix tree between
            # forwards while a prefetch copy still reads it, and ``loc`` can
            # be that row now. The copies are FIFO on one stream, so the
            # newest event drains all of them.
            torch.cuda.current_stream().wait_event(self._prefetch_write_gate)
            self._prefetch_write_gate = None
        if self.fused is not None:
            self.fused.write_through(layer_idx=layer_idx, rows=rows, loc=loc)
            return
        _copy_rows(
            k_dst=self.host_k[layer_idx],
            v_dst=self.host_v[layer_idx],
            dst=loc.long(),
            k_src=self.dev_k[layer_idx],
            v_src=self.dev_v[layer_idx],
            src=rows.long(),
            ks_dst=self.host_ksf[layer_idx] if self.quantized else None,
            vs_dst=self.host_vsf[layer_idx] if self.quantized else None,
            ks_src=None if self.dev_ksf is None else self.dev_ksf[layer_idx],
            vs_src=None if self.dev_vsf is None else self.dev_vsf[layer_idx],
        )

    def _gather_output(self, layer_idx: int, rows: int):
        shape = (rows, *self.dev_k[layer_idx].shape[1:])
        k = torch.empty(shape, dtype=self.dev_k[layer_idx].dtype, device=self.device)
        v = torch.empty_like(k)
        if self.quantized:
            sf_shape = (rows, *self.dev_ksf[layer_idx].shape[1:])
            ksf = torch.empty(
                sf_shape, dtype=self.dev_ksf[layer_idx].dtype, device=self.device
            )
            vsf = torch.empty_like(ksf)
        else:
            ksf = vsf = None
        return k, v, ksf, vsf

    def gather_context(
        self, layer_idx: int, slots: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Host K/V rows of these logical slots, in order, on the device (plus
        the scale planes of a quantized K/V)."""
        k, v, ksf, vsf = self._gather_output(layer_idx, slots.numel())
        dst = torch.arange(slots.numel(), dtype=torch.int64, device=self.device)
        _copy_rows(
            k_dst=k,
            v_dst=v,
            dst=dst,
            k_src=self.host_k[layer_idx],
            v_src=self.host_v[layer_idx],
            src=slots.long(),
            ks_dst=ksf,
            vs_dst=vsf,
            ks_src=self.host_ksf[layer_idx] if self.quantized else None,
            vs_src=self.host_vsf[layer_idx] if self.quantized else None,
        )
        return k, v, ksf, vsf

    def gather_context_hot_tail(
        self,
        layer_idx: int,
        slots: torch.Tensor,
        tail_mask: torch.Tensor,
        tail_rows: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """``gather_context`` reading the rows selected by ``tail_mask`` from
        the DEVICE mirror rows ``tail_rows`` (cu order, one per masked slot)
        and the rest from the host store. Bit-identical to ``gather_context``
        whenever ``tail_rows`` holds exactly those slots' rows, which
        ``store`` + ``write_through`` guarantee for the current chunk: the
        host copy is a bitwise copy of the device rows it was written from.

        The mask indexes are data-dependent shapes whose ``nonzero`` syncs;
        serving uses the split-driven ``gather_context_hot_tail_split`` (the
        same bytes, indices prebuilt on the host), and the bit-identity tests
        pin this reference against it.
        """
        k, v, ksf, vsf = self._gather_output(layer_idx, slots.numel())
        dst = torch.arange(slots.numel(), dtype=torch.int64, device=self.device)
        host = ~tail_mask
        _copy_rows(
            k_dst=k,
            v_dst=v,
            dst=dst[host],
            k_src=self.host_k[layer_idx],
            v_src=self.host_v[layer_idx],
            src=slots[host].long(),
            ks_dst=ksf,
            vs_dst=vsf,
            ks_src=self.host_ksf[layer_idx] if self.quantized else None,
            vs_src=self.host_vsf[layer_idx] if self.quantized else None,
        )
        _copy_rows(
            k_dst=k,
            v_dst=v,
            dst=dst[tail_mask],
            k_src=self.dev_k[layer_idx],
            v_src=self.dev_v[layer_idx],
            src=tail_rows.long(),
            ks_dst=ksf,
            vs_dst=vsf,
            ks_src=self.dev_ksf[layer_idx] if self.quantized else None,
            vs_src=self.dev_vsf[layer_idx] if self.quantized else None,
        )
        return k, v, ksf, vsf

    def _sf_host_stage(
        self, key: str, rows: int, like: torch.Tensor
    ) -> Tuple[torch.Tensor, dict, int]:
        """Next slot of a rotating pinned staging ring for the CPU scale-plane
        leg, plus its entry and index. The CPU may not overwrite a slot whose
        async HtoD has not run, so the slot's recorded event is waited on; the
        rotation makes that event ``_SF_STAGE_DEPTH`` stage-uses old, which the
        CPU has overtaken only if the GPU fell that far behind (then waiting
        is the correct, and rare, answer)."""
        entry = self._sf_stages.get(key)
        if entry is None or entry["cap"] < rows or entry["stage"].dtype != like.dtype:
            cap = max(rows, 8192)
            entry = {
                "stage": torch.empty(
                    (_SF_STAGE_DEPTH * cap, *like.shape[1:]),
                    dtype=like.dtype,
                    pin_memory=like.device.type == "cuda",
                ),
                "cap": cap,
                "events": [None] * _SF_STAGE_DEPTH,
                "next": 0,
            }
            self._sf_stages[key] = entry
        i = entry["next"]
        entry["next"] = (i + 1) % _SF_STAGE_DEPTH
        if entry["events"][i] is not None:
            entry["events"][i].synchronize()
            entry["events"][i] = None
        return (
            entry["stage"][i * entry["cap"] : (i + 1) * entry["cap"]][:rows],
            entry,
            i,
        )

    def gather_context_hot_tail_split(
        self,
        *,
        layer_idx: int,
        slots: torch.Tensor,
        split: HotTailSplit,
        slots_cpu: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """``gather_context_hot_tail`` driven by a host-resolved ``split``.

        Same rows in the same order as the masked gather — the split is that
        mask read as counts — but every index arrives prebuilt from the host,
        so no leg carries a device synchronization or a data-dependent shape:
        the scale planes' host leg gathers on the CPU (``slots_cpu`` is
        ``slots`` read back once, away from this gather; pass it only for a
        quantized pool, whose CPU host-plane reads also barrier the layer's
        write-through copies, and it is read back here if absent) and rides
        one pinned copy up, which the masked path's 32-byte-row
        ``_copy_plane`` fallback cannot do.
        """
        k, v, ksf, vsf = self._gather_output(layer_idx, slots.numel())
        if split.host_count:
            _copy_plane(
                k,
                v,
                split.host_idx,
                self.host_k[layer_idx],
                self.host_v[layer_idx],
                slots.index_select(0, split.host_idx).long(),
            )
            if self.quantized:
                if slots_cpu is None:
                    # Quantized host leg: this also drains this layer's
                    # write-through copies before the CPU reads host rows.
                    slots_cpu = slots.to("cpu")
                hi_slots = slots_cpu.index_select(0, split.host_idx_cpu)
                for dst, src, key in (
                    (ksf, self.host_ksf[layer_idx], "k"),
                    (vsf, self.host_vsf[layer_idx], "v"),
                ):
                    stage, entry, slot = self._sf_host_stage(key, split.host_count, dst)
                    torch.index_select(src, 0, hi_slots, out=stage)
                    dst.index_copy_(
                        0, split.host_idx, stage.to(dst.device, non_blocking=True)
                    )
                    if stage.is_pinned():
                        ev = torch.cuda.Event()
                        ev.record()
                        entry["events"][slot] = ev
        if split.tail_count:
            _copy_rows(
                k_dst=k,
                v_dst=v,
                dst=split.tail_idx,
                k_src=self.dev_k[layer_idx],
                v_src=self.dev_v[layer_idx],
                src=split.tail_rows,
                ks_dst=ksf,
                vs_dst=vsf,
                ks_src=self.dev_ksf[layer_idx] if self.quantized else None,
                vs_src=self.dev_vsf[layer_idx] if self.quantized else None,
            )
        return k, v, ksf, vsf

    # ---- layer-ahead gather prefetch ------------------------------------------

    def _prefetch_stream_lease(self) -> torch.cuda.Stream:
        # The gather copies own their stream — sharing the weights
        # offloader's "offload" lease would serialize unrelated copy work.
        if self._prefetch_stream is None:
            from sglang.srt.runtime_context import get_stream

            self._prefetch_stream = get_stream("qsa_kv_h2d")
        return self._prefetch_stream

    def _prefetch_eligible(self, forward_batch) -> bool:
        """Whether this extend forward may prefetch its context gathers.

        The plan mirrors the dense/hot-tail gather row-for-row, so the same
        fields the gather reads must exist, and no branch that would not
        consume the copy (verify, slab, union, a captured graph) may be in
        play. The row cap bounds the extra device buffers and the window in
        which the copy stream reads host rows the scheduler may recycle.
        """
        if not envs.SGLANG_QSA_OFFLOAD_PREFETCH_GATHER.get():
            return False
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )

        if (
            self.verify
            or self.slab_rows > 0
            or envs.SGLANG_QSA_OFFLOAD_UNION_GATHER.get()
            or self.device.type != "cuda"
            or not self.pin_memory
            # Like the slab and hot-tail gathers: inside a BCG prefill the
            # consume could land in a captured segment, where waiting on an
            # event recorded outside the capture is unsupported.
            or is_in_breakable_cuda_graph()
            or torch.cuda.is_current_stream_capturing()
        ):
            return False
        row = self.host_k[0][0]
        if row.numel() * row.element_size() % 128:
            # The fused transfer kernel wants 128-byte rows; its fallback
            # index copy syncs, which has no place on the copy stream.
            return False
        seqs = forward_batch.seq_lens_cpu
        exts = forward_batch.extend_seq_lens_cpu
        if seqs is None or exts is None or forward_batch.extend_seq_lens is None:
            return False
        if len(seqs) != len(exts):
            return False
        total = sum(int(x) for x in seqs)
        if not 0 < total <= envs.SGLANG_QSA_OFFLOAD_PREFETCH_MAX_ROWS.get():
            return False
        # forward_extend hands the new K/V straight to the kernel when no
        # request has a cached prefix: no gather ever runs to take a copy.
        return any(s > e for s, e in zip((int(x) for x in seqs), exts))

    def prefetch_prepare(self, forward_batch) -> None:
        """Resolve this extend forward's context rows once and prefetch the
        first QSA layer's gather on the copy stream (its runway: the
        embedding and the first GDN run)."""
        if not self._prefetch_eligible(forward_batch):
            return
        seqs = [int(x) for x in forward_batch.seq_lens_cpu]
        exts = [int(x) for x in forward_batch.extend_seq_lens_cpu]
        total = sum(seqs)
        starts_list = [0]
        for seq in seqs[:-1]:
            starts_list.append(starts_list[-1] + seq)
        starts = torch.tensor(starts_list, dtype=torch.int64, device=self.device)
        owner = torch.repeat_interleave(
            torch.arange(len(seqs), device=self.device),
            torch.tensor(seqs, dtype=torch.int64, device=self.device),
            output_size=total,
        )
        positions = torch.arange(total, device=self.device) - starts[owner]
        slots = self.req_to_token[
            forward_batch.req_pool_indices.long()[owner], positions
        ].long()
        self._gather_plan = _QsaGatherPlan(
            seqs=seqs, exts=exts, num_rows=total, slots=slots, next_layer=0
        )
        self._issue_prefetch(0)

    def maybe_prefetch_next(self, layer_idx: int) -> None:
        """Trigger from a QSA layer's gather: the plane after it gets a
        copy runway of the attention and GDN layers before its gather.
        Must run after that gather took its record: one record slot."""
        if self._gather_plan is not None:
            self._issue_prefetch(layer_idx + 1)

    def _issue_prefetch(self, layer_idx: int) -> None:
        plan = self._gather_plan
        if (
            plan is None
            or plan.next_layer != layer_idx
            or layer_idx >= self.num_layers
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        plan.next_layer += 1
        shape = (plan.num_rows, *self.dev_k[layer_idx].shape[1:])
        k = torch.empty(shape, dtype=self.dev_k[layer_idx].dtype, device=self.device)
        v = torch.empty_like(k)
        stream = self._prefetch_stream_lease()
        done = torch.cuda.Event()
        # Chain the copy behind everything enqueued so far: those are the
        # writes to the host rows it reads. A *later* store could land on a
        # host row the scheduler freed from the radix tree while this copy
        # still reads it; that writer, not this reader, is gated — see the
        # wait at the top of write_through.
        stream.wait_stream(torch.cuda.current_stream())
        plan.slots.record_stream(stream)
        # Allocated on the compute stream, written on the copy stream: lease
        # both so the allocator cannot hand them to a third party mid-copy.
        k.record_stream(stream)
        v.record_stream(stream)
        with torch.cuda.stream(stream):
            # The index vector is allocated on the copy stream too, so the
            # caching allocator returns it to *this* stream's pool: a
            # compute-stream reuse would otherwise need no event for the
            # copy's still-pending read of it.
            _copy_plane(
                k_dst=k,
                v_dst=v,
                dst=torch.arange(plan.num_rows, dtype=torch.int64, device=self.device),
                k_src=self.host_k[layer_idx],
                v_src=self.host_v[layer_idx],
                src=plan.slots,
            )
            done.record(stream)
        self._prefetched = _QsaPrefetch(layer_idx, k, v, done)
        self._prefetch_write_gate = done

    def take_prefetch(self, layer_idx: int) -> Optional[_QsaPrefetch]:
        """The scheduled copy for this layer's gather, if one was issued;
        ``finish_prefetched_gather`` assembles the planes from it. The
        stream wait behind the copy's completion rides on the current
        stream, as does everything the caller launches after it."""
        rec = self._prefetched
        if rec is None or rec.layer_idx != layer_idx:
            return None
        self._prefetched = None
        torch.cuda.current_stream().wait_event(rec.done)
        return rec

    def finish_prefetched_gather(
        self,
        rec: _QsaPrefetch,
        layer_idx: int,
        slots: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Assemble one layer's gather from its copy: the chunk tail (whose
        host rows the copy could not have seen written) is overwritten from
        the device-scratch rows the hot-tail gather reads, and the scale
        planes of a quantized K/V are read from the host exactly as the
        un-prefetched gather reads them. Bit-identical to
        ``gather_context``; all index math rides on the plan's host-side
        lengths, so no extra host sync enters the forward."""
        plan = self._gather_plan
        dst_rows = []
        dst_rows_src = []
        off = scr = 0
        base = self.layout.scratch_base
        for seq, ext in zip(plan.seqs, plan.exts):
            if ext:
                dst_rows += [
                    rec.k[off + seq - ext : off + seq],
                    rec.v[off + seq - ext : off + seq],
                ]
                dst_rows_src += [
                    self.dev_k[layer_idx][base + scr : base + scr + ext],
                    self.dev_v[layer_idx][base + scr : base + scr + ext],
                ]
                scr += ext
            off += seq
        if dst_rows:
            torch._foreach_copy_(dst_rows, dst_rows_src)
        k_sf = v_sf = None
        if self.quantized:
            k_sf = torch.empty(
                (plan.num_rows, *self.dev_ksf[layer_idx].shape[1:]),
                dtype=self.dev_ksf[layer_idx].dtype,
                device=self.device,
            )
            v_sf = torch.empty_like(k_sf)
            _copy_rows(
                k_dst=k_sf,
                v_dst=v_sf,
                dst=torch.arange(plan.num_rows, device=self.device),
                k_src=self.host_ksf[layer_idx],
                v_src=self.host_vsf[layer_idx],
                src=slots,
            )
        if envs.SGLANG_QSA_OFFLOAD_PREFETCH_DEBUG.get():
            # The all-host gather reads the same rows *now* (after all this
            # forward's host writes): a difference means the copy raced a
            # host-plane write or read a row the plan should not have held.
            ref = self.gather_context(layer_idx, slots)
            for name, got, want in (
                ("k", rec.k, ref[0]),
                ("v", rec.v, ref[1]),
                ("k_sf", k_sf, ref[2]),
                ("v_sf", v_sf, ref[3]),
            ):
                if got is None and want is None:
                    continue
                if not torch.equal(got, want):
                    raise RuntimeError(
                        f"QSA gather prefetch: layer {layer_idx} {name} differs "
                        "from the inline all-host gather"
                    )
        return rec.k, rec.v, k_sf, v_sf

    # ---- decode / target verify ----------------------------------------------

    def new_token_rows(
        self, row_seq_lens: torch.Tensor, row_req: torch.Tensor
    ) -> torch.Tensor:
        """Ring row of each query row's new token (position ``seq_len - 1``)."""
        pos = row_seq_lens.long() - 1
        return (
            row_req.long() * self.region
            + (pos // self.cr) % TAIL_BLOCKS * self.cr
            + pos % self.cr
        )

    def resolve_rows(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        row_seq_lens: torch.Tensor,
        row_req: torch.Tensor,
        rows_per_request: int,
    ) -> torch.Tensor:
        """Device row of every selected logical position.

        ``positions`` is ``[R, W]`` request-relative positions (``-1`` padding),
        the rows of one request contiguous and ``rows_per_request`` of them;
        ``row_seq_lens`` is each row's query position + 1. Returns int32
        ``[R, W]`` device rows, ``-1`` exactly where the position is not
        attended (``< 0`` or ``>= row_seq_lens``). Call after ``bind``; the
        forward's new K/V may already be stored at ``new_token_rows``.
        """
        if self.fused is not None:
            return self.fused.resolve(
                layer_idx=layer_idx,
                positions=positions,
                row_seq_lens=row_seq_lens.contiguous(),
                row_req=row_req,
                rows_per_request=rows_per_request,
                req_to_token=self.req_to_token,
            )
        cr, q = self.cr, rows_per_request
        num_rows, width = positions.shape
        if num_rows % q:
            raise ValueError(f"{num_rows} rows are not groups of {q}")
        groups = num_rows // q
        if layer_idx == 0:
            self._step += 1
        seq = row_seq_lens.long()
        req = row_req.long()
        group_req = req[::q]
        first_new = seq[::q] - 1  # committed length of each request
        self._load_ring(
            layer_idx=layer_idx,
            group_req=group_req,
            first_new=first_new,
            rows_per_request=q,
        )
        self._drop_remapped(layer_idx=layer_idx, group_req=group_req)

        pos = positions.long()
        valid = (pos >= 0) & (pos < seq[:, None])
        ring_start = torch.repeat_interleave(first_new // cr * cr, q)
        in_ring = valid & (pos >= ring_start[:, None])
        need = (valid & ~in_ring).reshape(groups, q * width)
        blk = torch.where(need, pos.reshape(groups, q * width) // cr, -1)
        blk_slot = self.blk_slot[layer_idx]
        req_x = group_req[:, None].expand_as(blk)
        slot = blk_slot[req_x, blk.clamp(min=0)].long()
        miss = need & (slot < 0)
        if bool(miss.any()):
            self._swap_in(
                layer_idx=layer_idx,
                group_req=group_req,
                blk=blk,
                slot=slot,
                need=need,
                miss=miss,
            )
            slot = blk_slot[req_x, blk.clamp(min=0)].long()
        hit_g, hit_c = need.nonzero(as_tuple=True)
        self.slot_age[layer_idx][group_req[hit_g], slot[hit_g, hit_c]] = self._step

        slot = slot.reshape(num_rows, width)
        offset = pos.remainder(cr)
        base = (req * self.region)[:, None]
        ring_rows = base + (pos // cr) % TAIL_BLOCKS * cr + offset
        lru_rows = base + self.tail_rows + slot.clamp(min=0) * cr + offset
        rows = torch.where(in_ring, ring_rows, lru_rows)
        return torch.where(valid, rows, -1).to(torch.int32)

    def _load_ring(
        self,
        layer_idx: int,
        group_req: torch.Tensor,
        first_new: torch.Tensor,
        rows_per_request: int,
    ) -> None:
        """Make each request's tail ring hold the committed rows of block
        ``first_new // cr`` and take the forward's new rows of that block and
        the next one."""
        cr = self.cr
        offsets = self._arange[:cr]
        block = first_new // cr
        ring_slot = block % TAIL_BLOCKS
        committed = offsets[None, :] < (first_new % cr)[:, None]
        cols = block[:, None] * cr + offsets
        src0 = self.req_to_token[group_req[:, None], cols.clamp(max=self.width - 1)]
        cols1 = cols + cr
        src1 = self.req_to_token[group_req[:, None], cols1.clamp(max=self.width - 1)]
        ring_blk = self.ring_blk[layer_idx]
        ring_src = self.ring_src[layer_idx]
        held = (
            (ring_blk[group_req, ring_slot] == block)
            & (first_new <= self.ring_end[layer_idx][group_req])
            & ((ring_src[group_req, ring_slot] == src0) | ~committed).all(1)
        )
        load = committed & ~held[:, None]
        if bool(load.any()):
            load_g, load_c = load.nonzero(as_tuple=True)
            dst = (
                group_req[load_g] * self.region
                + ring_slot[load_g] * cr
                + offsets[load_c]
            )
            _copy_rows(
                k_dst=self.dev_k[layer_idx],
                v_dst=self.dev_v[layer_idx],
                dst=dst,
                k_src=self.host_k[layer_idx],
                v_src=self.host_v[layer_idx],
                src=src0[load_g, load_c].long(),
                ks_dst=self.dev_ksf[layer_idx] if self.quantized else None,
                vs_dst=self.dev_vsf[layer_idx] if self.quantized else None,
                ks_src=self.host_ksf[layer_idx] if self.quantized else None,
                vs_src=self.host_vsf[layer_idx] if self.quantized else None,
            )
        # A block that re-enters the ring is rewritten there: drop its LRU copy.
        for b in (block, block + 1):
            slot = self.blk_slot[layer_idx][group_req, b.clamp(max=self.max_blocks - 1)]
            hit = (slot >= 0).nonzero(as_tuple=True)[0]
            if hit.numel():
                self.slot_src[layer_idx][group_req[hit], slot[hit].long()] = -1
        other = 1 - ring_slot
        ring_blk[group_req, ring_slot] = block
        ring_blk[group_req, other] = block + 1
        ring_src[group_req, ring_slot] = src0
        ring_src[group_req, other] = src1
        self.ring_end[layer_idx][group_req] = first_new + rows_per_request

    def _drop_remapped(self, layer_idx: int, group_req: torch.Tensor) -> None:
        """Invalidate LRU slots whose positions no longer map to the logical
        slots they were filled from."""
        slot_blk = self.slot_blk[layer_idx][group_req]
        live = slot_blk >= 0
        cols = slot_blk.clamp(min=0)[..., None] * self.cr + self._arange[: self.cr]
        current = self.req_to_token[
            group_req[:, None, None], cols.clamp(max=self.width - 1)
        ]
        stale = live & (current != self.slot_src[layer_idx][group_req]).any(-1)
        stale_g, stale_s = stale.nonzero(as_tuple=True)
        if stale_g.numel():
            req = group_req[stale_g]
            self.blk_slot[layer_idx][req, slot_blk[stale_g, stale_s]] = -1
            self.slot_blk[layer_idx][req, stale_s] = -1
            self.slot_age[layer_idx][req, stale_s] = -1
            self.slot_src[layer_idx][req, stale_s] = -1

    def _swap_in(
        self,
        layer_idx: int,
        group_req: torch.Tensor,
        blk: torch.Tensor,
        slot: torch.Tensor,
        need: torch.Tensor,
        miss: torch.Tensor,
    ) -> None:
        """Copy every missing block of each request into its least recently
        used slots that this forward does not read."""
        cr, groups = self.cr, blk.shape[0]
        group_ids = torch.arange(groups, device=blk.device)[:, None].expand_as(blk)
        key = torch.unique((group_ids * self.max_blocks + blk)[miss])
        miss_g = key // self.max_blocks
        miss_blk = key % self.max_blocks
        counts = torch.bincount(miss_g, minlength=groups)
        rank = (
            torch.arange(key.numel(), device=blk.device)
            - (torch.cumsum(counts, 0) - counts)[miss_g]
        )
        age = self.slot_age[layer_idx][group_req].clone()
        pinned = torch.iinfo(torch.int64).max
        hit_g, hit_c = (need & ~miss).nonzero(as_tuple=True)
        age[hit_g, slot[hit_g, hit_c]] = pinned
        if bool((counts > (age != pinned).sum(1)).any()):
            raise RuntimeError(
                f"QSA K/V offload: {self.hot_blocks} hot blocks cannot hold this "
                "forward's selection"
            )
        victim = torch.argsort(age, dim=1, stable=True)[miss_g, rank]
        req = group_req[miss_g]
        old = self.slot_blk[layer_idx][req, victim]
        evicted = old >= 0
        self.blk_slot[layer_idx][req[evicted], old[evicted]] = -1
        self.slot_blk[layer_idx][req, victim] = miss_blk
        self.blk_slot[layer_idx][req, miss_blk] = victim.to(torch.int32)
        offsets = self._arange[:cr]
        src = self.req_to_token[req[:, None], miss_blk[:, None] * cr + offsets]
        self.slot_src[layer_idx][req, victim] = src
        dst = req[:, None] * self.region + self.tail_rows + victim[:, None] * cr
        _copy_rows(
            k_dst=self.dev_k[layer_idx],
            v_dst=self.dev_v[layer_idx],
            dst=(dst + offsets).reshape(-1),
            k_src=self.host_k[layer_idx],
            v_src=self.host_v[layer_idx],
            src=src.long().reshape(-1),
            ks_dst=self.dev_ksf[layer_idx] if self.quantized else None,
            vs_dst=self.dev_vsf[layer_idx] if self.quantized else None,
            ks_src=self.host_ksf[layer_idx] if self.quantized else None,
            vs_src=self.host_vsf[layer_idx] if self.quantized else None,
        )

    # -- chunked-prefill context slab (see SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS)

    def slab_reserve(self, req: int, need: int) -> Optional[_PrefillSlab]:
        """A contiguous slab of ``need`` rows for extending request ``req``,
        keeping its cached rows (moving them if the slab must grow). None
        when the pool cannot fit it even after freeing every slab it may
        touch (other requests' and, if that is what it takes, its own);
        the request then gathers from the host and any partial slab it
        still holds stays valid."""
        if self.slab_rows <= 0 or need > self.slab_rows:
            # Never servable, however much it evicts (need only grows during
            # the extend): give up without touching neighbors. Drop this
            # request's own now-dead slab too, or the pool would keep it
            # pinned as the largest slab while its owner runs the host
            # gather every chunk -- the opposite of the eviction heuristic.
            self.slab_release(req)
            return None
        slab = self._slabs.get(req)
        if slab is not None and slab.length >= need:
            return slab
        want = min(need + self.slab_grow, self.slab_rows)
        # The grow slack is a preference, not a requirement: an exact-need
        # span still saves the PCIe re-gather, so try it before discarding
        # any cached rows. ``taken`` is the span actually reserved; the
        # slab length must match it or release over-returns.
        base = self._slab_take(want)
        taken = want
        if base is None:
            base = self._slab_take(need)
            taken = need
        if base is None:
            # Destructive plans, cheapest first: free the biggest other
            # slab (its rows re-gather once), hand back this request's own
            # rows (cached rows are re-gatherable, the rebuild starts
            # empty), or both. Preview each BEFORE giving up anything --
            # _slabs_for retries per layer and chunk, so a plan that
            # cannot serve the need must never destroy the most-cached
            # rows; a previewed plan cannot then fail its need-sized
            # take, so no release is ever made for nothing.
            victim = self._slab_biggest_other(req)
            plans = []
            if victim is not None:
                v = self._slabs[victim]
                plans.append((True, False, [(v.base, v.length)]))
            if slab is not None:
                plans.append((False, True, [(slab.base, slab.length)]))
                if victim is not None:
                    v = self._slabs[victim]
                    plans.append(
                        (True, True, [(v.base, v.length), (slab.base, slab.length)])
                    )
            for evict, own_release, freed in plans:
                if self._slab_max_run(freed) < need:
                    continue
                # The preview bounds the post-release layout, so the
                # need-sized take below cannot fail; the want-sized one
                # may, which is why it is only a preference.
                if evict:
                    self._slab_evict_other(req)
                if own_release:
                    self._slab_put_span(slab.base, slab.length)
                    self._slabs.pop(req, None)
                    base = self._slab_take(want)
                    taken = want
                    if base is None:
                        base = self._slab_take(need)
                        taken = need
                    self._slabs[req] = _PrefillSlab(
                        base=base, length=taken, cursors=[0] * self.num_layers
                    )
                    return self._slabs[req]
                base = self._slab_take(want)
                taken = want
                if base is None:
                    base = self._slab_take(need)
                    taken = need
                break
        if base is None:
            return None
        if slab is not None:
            cached = max(slab.cursors)
            for layer_idx in range(self.num_layers):
                self._slab_move_rows(layer_idx, dst=base, src=slab.base, count=cached)
            self._slab_put_span(slab.base, slab.length)
            slab.base = base
            slab.length = taken
        else:
            slab = _PrefillSlab(base=base, length=taken, cursors=[0] * self.num_layers)
            self._slabs[req] = slab
        return slab

    def slab_release(self, req: int) -> None:
        """Hand back the slab rows of a request that left extend (its decode
        steps read the pool through the ring/LRU, not the slab)."""
        slab = self._slabs.pop(req, None)
        if slab is not None:
            self._slab_put_span(slab.base, slab.length)

    def slab_append(
        self,
        *,
        layer_idx: int,
        slab: _PrefillSlab,
        packed: Tuple[torch.Tensor, ...],
        packed_start: int,
        count: int,
    ) -> None:
        """Stage ``count`` packed context rows starting at ``packed_start``
        of this chunk's gather output into the slab behind its cursor for
        ``layer_idx``; the caller advances the cursor."""
        if count <= 0:
            return
        cursor = slab.cursors[layer_idx]
        k, v, k_sf, v_sf = packed
        _copy_rows(
            k_dst=self.slab_k[layer_idx],
            v_dst=self.slab_v[layer_idx],
            dst=torch.arange(
                slab.base + cursor,
                slab.base + cursor + count,
                dtype=torch.int64,
                device=self.device,
            ),
            k_src=k,
            v_src=v,
            src=torch.arange(
                packed_start,
                packed_start + count,
                dtype=torch.int64,
                device=self.device,
            ),
            ks_dst=self.slab_ksf[layer_idx] if self.quantized else None,
            vs_dst=self.slab_vsf[layer_idx] if self.quantized else None,
            ks_src=k_sf,
            vs_src=v_sf,
        )

    def _slab_take(self, need: int) -> Optional[int]:
        for i, (base, length) in enumerate(self._slab_spans):
            if length >= need:
                rest = length - need
                self._slab_spans[i : i + 1] = [(base + need, rest)] if rest else []
                return base
        return None

    def _slab_put_span(self, base: int, length: int) -> None:
        self._slab_spans.append((base, length))
        self._slab_spans.sort()
        merged = []
        for b, n in self._slab_spans:
            if merged and merged[-1][0] + merged[-1][1] == b:
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((b, n))
        self._slab_spans = merged

    def _slab_biggest_other(self, req: int) -> Optional[int]:
        """The request id whose slab _slab_evict_other would pick (the most
        cached other request), or None when there is no eviction victim."""
        cands = [
            (max(slab.cursors), other)
            for other, slab in self._slabs.items()
            if other != req
        ]
        return max(cands)[1] if cands else None

    def _slab_max_run(self, extra: list[Tuple[int, int]]) -> int:
        """The longest contiguous row run the pool could hold if the spans
        in ``extra`` were handed back to the free list. Free spans and slab
        spans tile the pool, so this is exactly what releasing them can
        yield beside the existing free space (want-sized takes can still
        fail on the final layout; need-sized ones cannot if this holds)."""
        spans = sorted(list(self._slab_spans) + list(extra))
        best = 0
        run_start = run_end = 0
        for base, length in spans:
            if base > run_end:
                best = max(best, run_end - run_start)
                run_start = base
            run_end = base + length
        return max(best, run_end - run_start)

    def _slab_evict_other(self, req: int) -> None:
        # The largest other request's slab is the cheapest way back to free
        # rows: its next chunk re-gathers from the host once and re-caches.
        cands = [
            (max(slab.cursors), other, slab)
            for other, slab in self._slabs.items()
            if other != req
        ]
        if cands:
            _, other, _ = max(cands)
            self.slab_release(other)

    def _slab_move_rows(
        self, layer_idx: int, *, dst: int, src: int, count: int
    ) -> None:
        if count <= 0:
            return
        rows_src = torch.arange(src, src + count, dtype=torch.int64, device=self.device)
        rows_dst = torch.arange(dst, dst + count, dtype=torch.int64, device=self.device)
        _copy_rows(
            k_dst=self.slab_k[layer_idx],
            v_dst=self.slab_v[layer_idx],
            dst=rows_dst,
            k_src=self.slab_k[layer_idx],
            v_src=self.slab_v[layer_idx],
            src=rows_src,
            ks_dst=self.slab_ksf[layer_idx] if self.quantized else None,
            vs_dst=self.slab_vsf[layer_idx] if self.quantized else None,
            ks_src=self.slab_ksf[layer_idx] if self.quantized else None,
            vs_src=self.slab_vsf[layer_idx] if self.quantized else None,
        )
