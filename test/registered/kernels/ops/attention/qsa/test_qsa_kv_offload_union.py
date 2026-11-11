"""Union gather of the QSA K/V offload: bit-identical sparse prefill.

``gather_offloaded_context_union`` packs only the context rows the chunk's
top-k references and remaps ``indices`` into the compact layout. The sparse
prefill kernel walks ``indices``, so the attention output must be bitwise
equal to the dense gather for every query. Both paths run the real Triton
kernel on the same offload state and the outputs are compared. With the
prefill slab on, the union rows are served from the device slab and the
parity must hold across chunks.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.kv_offload_attention import (
    gather_offloaded_context,
    gather_offloaded_context_union,
)
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.srt.mem_cache.qsa_kv_offload import (
    QSAKVHostOffload,
    compute_offload_layout,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

CR, HEADS, DIM = 4, 2, 64
HOST_ROWS = 1 << 14


class _PlainRows:
    @staticmethod
    def key():
        return torch.empty(0, dtype=torch.float32)

    @staticmethod
    def value():
        return torch.empty(0, dtype=torch.float32)


class _PlainRouter:
    @staticmethod
    def read(layer_id):
        return _PlainRows()


class _PlainPool:
    layer_router = _PlainRouter()

    @staticmethod
    def full_attention_layer_index(layer_id):
        return layer_id


class _FakeBackend:
    def __init__(self, offload):
        self._kv_offload = offload
        self.token_to_kv_pool = _PlainPool()
        self.kv_cache_quant_method = None


def _setup(seed, lens, chunks):
    """Offload holding ``lens[b]`` context rows per request, the last
    ``chunks[b]`` of them stored in the extend scratch (the current chunk);
    everything else lives in the host store."""
    torch.manual_seed(seed)
    num_seqs = len(lens)
    layout = compute_offload_layout(
        compress_ratio=CR,
        token_topk=64 * CR,
        hot_blocks=512,
        num_request_slots=num_seqs + 1,
        rows_per_request=1,
        max_extend_tokens=sum(chunks),
        page_size=64,
    )
    shape = (layout.device_rows + 64, HEADS, DIM)
    offload = QSAKVHostOffload(
        layout=layout,
        k_buffers=[torch.zeros(shape, device="cuda")],
        v_buffers=[torch.zeros(shape, device="cuda")],
        host_rows=HOST_ROWS,
        pin_memory=False,
    )
    req_to_token = torch.full(
        (num_seqs + 1, max(lens)), -1, dtype=torch.int32, device="cuda"
    )
    free = torch.randperm(HOST_ROWS - 64, device="cuda") + 64
    cursor = 0
    ext_before = 0
    prefix_row = 0
    for b, (n, e) in enumerate(zip(lens, chunks), start=1):
        loc = free[cursor : cursor + n].to(torch.int32)
        cursor += n
        req_to_token[b, :n] = loc
        data_k = torch.randn(n, HEADS, DIM, device="cuda")
        data_v = torch.randn(n, HEADS, DIM, device="cuda")
        # The chunk's rows sit in the scratch; the cached prefix in plain
        # device rows. Both are written through to the host store, as the
        # serving path does.
        rows = torch.cat(
            [
                torch.arange(prefix_row, prefix_row + n - e, device="cuda"),
                torch.arange(
                    layout.scratch_base + ext_before,
                    layout.scratch_base + ext_before + e,
                    device="cuda",
                ),
            ]
        )
        prefix_row += n - e
        ext_before += e
        offload.dev_k[0][rows] = data_k
        offload.dev_v[0][rows] = data_v
        offload.write_through(0, rows, loc.long())
    offload.req_to_token = req_to_token
    forward_batch = SimpleNamespace(
        req_pool_indices=torch.arange(1, num_seqs + 1, device="cuda"),
        extend_seq_lens=torch.tensor(chunks, dtype=torch.int32, device="cuda"),
        extend_seq_lens_cpu=torch.tensor(chunks, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(lens, dtype=torch.int32),
    )
    return offload, _FakeBackend(offload), forward_batch


def _run_pair(seed, lens, chunks, topk, state=None, pos_cap=None):
    """Dense and union sparse-prefill attention over the same offload state.

    ``state`` reuses a live offload (multi-chunk slab tests); the rows
    beyond its original context must already be stored, which ``_setup``
    guarantees for the harness sizes."""
    offload, backend, base_fb = state or _setup(seed, lens, chunks)
    forward_batch = SimpleNamespace(
        req_pool_indices=base_fb.req_pool_indices,
        extend_seq_lens=torch.tensor(chunks, dtype=torch.int32, device="cuda"),
        extend_seq_lens_cpu=torch.tensor(chunks, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(lens, dtype=torch.int32),
    )
    layer = SimpleNamespace(layer_id=0)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    cu_k = F.pad(seq.cumsum(0), (1, 0)).contiguous()
    num_q = sum(chunks)
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(num_q, 4 * HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    # Each query selects ``topk`` causal-window positions, ~15% padding.
    starts = [sum(chunks[:b]) for b in range(len(chunks))]
    owner = torch.repeat_interleave(
        torch.arange(len(lens), device="cuda"),
        torch.tensor(chunks, device="cuda"),
    )
    q_rel = torch.arange(num_q, device="cuda") - torch.tensor(starts, device="cuda")[
        owner
    ]
    visible = (torch.tensor(lens, device="cuda") - torch.tensor(chunks, device="cuda"))[
        owner
    ] + q_rel + 1
    if pos_cap is not None:  # harness: un-stored chunk rows cannot be referenced
        visible = visible.clamp(max=pos_cap)
    pick = (torch.rand(num_q, topk, device="cuda", generator=g) * visible[:, None]).to(
        torch.int32
    )
    topk_indices = torch.where(
        torch.rand(num_q, topk, device="cuda", generator=g) < 0.85,
        pick,
        torch.full((num_q, topk), -1, dtype=torch.int32, device="cuda"),
    ).contiguous()
    cu_q = F.pad(
        torch.tensor(chunks, dtype=torch.int32, device="cuda").cumsum(0), (1, 0)
    ).contiguous()

    dk, dv = gather_offloaded_context(
        backend=backend,
        layer=layer,
        forward_batch=forward_batch,
        sequence_lens=seq,
        cu_seqlens_k=cu_k,
        total_k=sum(lens),
        output_dtype=q.dtype,
    )
    dense_out = sparse_gqa_fwd_interface_triton_ck(
        q, dk, dv, topk_indices, cu_q, cu_k, seq, 0.1, max_q=max(chunks)
    )
    uk, uv, cu_u, idx_u = gather_offloaded_context_union(
        backend=backend,
        layer=layer,
        forward_batch=forward_batch,
        topk_indices=topk_indices,
        extend_lens=chunks,
        sequence_lens=lens,
        max_seq=max(lens),
        output_dtype=q.dtype,
    )
    union_out = sparse_gqa_fwd_interface_triton_ck(
        q, uk, uv, idx_u, cu_q, cu_u, seq, 0.1, max_q=max(chunks)
    )
    assert torch.equal(dense_out, union_out)
    return offload, backend, forward_batch


class TestUnionGather:
    def test_single_request(self):
        _run_pair(1, [1500], [500], 64)

    def test_two_requests_tail_chunk(self):
        _run_pair(2, [1200, 900], [300, 128], 64)

    def test_tiny_union(self):
        _run_pair(3, [2048], [64], 16)


class TestUnionSlab:
    def test_slab_serves_union(self):
        with envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.override(16384):
            _run_pair(4, [1500, 900], [400, 128], 64)

    def test_second_chunk_rides_the_slab(self):
        with envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.override(16384):
            # Table/scratch sized for the 3700-row request; two forwards of
            # the same chunked state over one live offload. The second
            # forward's dense path must ride the slab the first one filled
            # (the harness never write-throughs between forwards, so only
            # the _setup chunking is ground-truth consistent here).
            state = _setup(5, [3700], [1000])
            _run_pair(6, [3700], [1000], 64, state=state)
            _run_pair(7, [3700], [1000], 64, state=state)
            assert state[0]._slabs[1].cursors[0] == 3700

    def test_stale_cursor_is_discarded(self):
        with envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.override(16384):
            state = _setup(5, [3700], [1000])
            _run_pair(6, [3000], [1000], 64, state=state)
            # A previous tenant's cursor must never be trusted, even when
            # it is larger than anything the new layout could stage.
            state[0]._slabs[1].cursors[0] = 5000
            _run_pair(7, [3700], [700], 64, state=state, pos_cap=3000)
            assert state[0]._slabs[1].cursors[0] == 3700

    def test_new_tenant_is_not_served_the_old_slab(self):
        with envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.override(16384):
            # Tenant B lands on the slot with a radix hit at exactly
            # tenant A's cursor depth: cursor depth alone cannot tell the
            # tenants apart; the stamped first row must.
            state = _setup(8, [4600], [900])
            _run_pair(9, [3700], [900], 64, state=state)
            old = state[0]._slabs[1]
            state[0].req_to_token[1, 0] = 1
            _run_pair(10, [4600], [900], 64, state=state)
            assert state[0]._slabs[1] is not old

    def test_reused_slot_resets_the_slab(self):
        with envs.SGLANG_QSA_OFFLOAD_PREFILL_SLAB_ROWS.override(16384):
            state = _setup(8, [800], [500])
            _run_pair(9, [800], [500], 64, state=state)
            # A new request takes the slot with a fresh extend: the old
            # occupant's rows (now zeroed in the host store) must not be
            # trusted through the stale slab.
            state[0].host_k[0].zero_()
            state[0].host_v[0].zero_()
            _run_pair(10, [500], [500], 64, state=state)
            assert state[0]._slabs[1].cursors[0] == 500
