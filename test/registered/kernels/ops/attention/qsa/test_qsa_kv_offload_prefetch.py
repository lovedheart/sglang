"""Layer-ahead gather prefetch of the QSA K/V host offload.

The prefetch copies each QSA plane's context rows host -> device on a side
stream so the copy hides behind the GDN layers between two QSA layers;
``finish_prefetched_gather`` then overwrites this forward's chunk tail from
the device mirror scratch (whose host rows the copy could not have seen
written) and adds the scale planes inline. The assembled K/V must be
bit-identical to both ``gather_context`` and ``gather_context_hot_tail``.
The tail mapping mirrors ``kv_offload_attention.gather_offloaded_context``.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.qsa_kv_offload import (
    QSAKVHostOffload,
    compute_offload_layout,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

CR, BLOCK_TOPK, HEADS, DIM = 4, 512, 2, 256
TOPK = BLOCK_TOPK * CR + CR - 1
HOST_ROWS = 1 << 18
LAYERS = 2


def _make_offload(lens, tails, quant):
    layout = compute_offload_layout(
        compress_ratio=CR,
        token_topk=TOPK,
        hot_blocks=2048,
        num_request_slots=len(lens) + 1,
        rows_per_request=4,
        max_extend_tokens=sum(tails),
        page_size=64,
    )
    dev_k = [
        torch.zeros(
            (layout.device_rows + 64, HEADS, DIM // 2), device="cuda", dtype=torch.uint8
        )
        for _ in range(LAYERS)
    ]  # fmt: skip
    dev_v = [torch.zeros_like(t) for t in dev_k]
    ksf = [
        torch.zeros(
            (layout.device_rows + 64, HEADS, DIM // 16), device="cuda", dtype=torch.uint8
        )
        for _ in range(LAYERS)
    ]  # fmt: skip
    vsf = [torch.zeros_like(t) for t in ksf]
    return QSAKVHostOffload(
        layout=layout,
        k_buffers=dev_k,
        v_buffers=dev_v,
        host_rows=HOST_ROWS,
        pin_memory=True,
        k_sf_buffers=ksf if quant else None,
        v_sf_buffers=vsf if quant else None,
    )


def _scene(offload, lens, tails, seed, mirror="both"):
    """Fill host prefix rows, scratch chunk rows, and the (req_to_token,
    forward_batch, slots) the production gather is entered with. Globally
    disjoint random logical slots, random bytes everywhere."""
    g = torch.Generator().manual_seed(seed)
    perm = (
        torch.randperm(
            HOST_ROWS - 64, generator=torch.Generator().manual_seed(seed)
        ).cuda()
        + 64
    )
    slots_by_req, p0 = [], 0
    for n in lens:
        slots_by_req.append(perm[p0 : p0 + n].long())
        p0 += n
    all_slots = torch.cat(slots_by_req)
    for planes in (offload.host_k, offload.host_v):
        for plane in planes:
            plane[all_slots.cpu()] = torch.randint(
                0, 256, (p0, HEADS, DIM // 2), generator=g, dtype=torch.uint8
            )
    if offload.quantized:
        for planes in (offload.host_ksf, offload.host_vsf):
            for plane in planes:
                plane[all_slots.cpu()] = torch.randint(
                    0, 256, (p0, HEADS, DIM // 16), generator=g, dtype=torch.uint8
                )  # fmt: skip
    # The current chunk: device scratch rows (what store leaves behind).
    rows = offload.extend_rows(sum(tails))
    for planes in (offload.dev_k, offload.dev_v):
        for d in planes:
            d[rows] = (
                torch.randint(
                    0, 256, (sum(tails), HEADS, DIM // 2), generator=g, dtype=torch.uint8
                )
                .cuda()
            )  # fmt: skip
    if offload.quantized:
        for planes in (offload.dev_ksf, offload.dev_vsf):
            for d in planes:
                d[rows] = (
                    torch.randint(
                        0,
                        256,
                        (sum(tails), HEADS, DIM // 16),
                        generator=g,
                        dtype=torch.uint8,
                    )
                    .cuda()
                )  # fmt: skip
    tail_slots = torch.cat(
        [
            slots_by_req[b][lens[b] - tails[b] :]
            for b in range(len(lens))
            if tails[b] > 0
        ]
    ).cpu()
    if mirror in ("both", "scale"):
        if mirror == "both":
            for hs, ds in (
                (offload.host_k, offload.dev_k),
                (offload.host_v, offload.dev_v),
            ):
                for h, d in zip(hs, ds):
                    h[tail_slots] = d[rows].cpu()
        if offload.quantized:
            for hs, ds in (
                (offload.host_ksf, offload.dev_ksf),
                (offload.host_vsf, offload.dev_vsf),
            ):
                for h, d in zip(hs, ds):
                    h[tail_slots] = d[rows].cpu()
    req_to_token = torch.zeros(
        (len(lens) + 1, max(lens)), dtype=torch.int32, device="cuda"
    )
    for r, s in enumerate(slots_by_req):
        req_to_token[r, : lens[r]] = s.int()
    fb = SimpleNamespace(
        seq_lens_cpu=torch.tensor(lens, dtype=torch.int64),
        extend_seq_lens_cpu=list(tails),
        extend_seq_lens=torch.tensor(tails, dtype=torch.int32, device="cuda"),
        req_pool_indices=torch.arange(len(lens), dtype=torch.int64, device="cuda"),
    )
    return req_to_token, fb, all_slots.int()


def _hot_tail_args(offload, lens, tails):
    seq = torch.tensor(lens, device="cuda", dtype=torch.int64)
    e = torch.tensor(tails, device="cuda", dtype=torch.int64)
    total = int(seq.sum())
    owner = torch.repeat_interleave(
        torch.arange(len(lens), device="cuda"), seq, output_size=total
    )
    starts = seq[owner] - e[owner]
    positions = (
        torch.arange(total, device="cuda")
        - (torch.cat([seq.new_zeros(1), seq.cumsum(0)[:-1]])[owner])
    )
    tail_mask = positions >= starts
    rows_before = e.cumsum(0) - e
    tail_rows = (
        offload.layout.scratch_base + rows_before[owner] + (positions - starts)
    )[tail_mask]
    return tail_mask, tail_rows


@pytest.mark.parametrize("quant", [True, False])
def test_prefetch_bit_exact(quant, monkeypatch):
    monkeypatch.setenv("SGLANG_QSA_OFFLOAD_PREFETCH_GATHER", "true")
    lens, tails = [130, 51, 97], [16, 0, 33]
    offload = _make_offload(lens, tails, quant)
    req_to_token, fb, slots = _scene(offload, lens, tails, seed=7)
    offload.begin_forward(req_to_token, fb.req_pool_indices)
    offload.prefetch_prepare(fb)
    assert offload._gather_plan is not None
    tail_mask, tail_rows = _hot_tail_args(offload, lens, tails)
    for layer in range(LAYERS):
        if layer:
            # The previous QSA layer's store triggers this plane's copy.
            offload.maybe_prefetch_next(layer - 1)
        rec = offload.take_prefetch(layer)
        assert rec is not None, f"layer {layer} was never scheduled"
        got = offload.finish_prefetched_gather(rec, layer_idx=layer, slots=slots.long())
        ref = offload.gather_context(layer_idx=layer, slots=slots)
        hot = offload.gather_context_hot_tail(
            layer_idx=layer, slots=slots, tail_mask=tail_mask, tail_rows=tail_rows
        )
        for a, b, c in zip(got, ref, hot):
            if a is None:
                assert b is None
                continue
            assert torch.equal(a, b) and torch.equal(a, c), f"layer {layer}"
    offload.destroy()


def test_prefetch_refills_stale_host_tail(monkeypatch):
    # Host rows of the chunk's K/V data hold OLD bytes (their write-through
    # raced the copy); the scratch refill must save them. The scale planes
    # are mirrored: they are never prefetched, the consume-time host read
    # is the production one.
    monkeypatch.setenv("SGLANG_QSA_OFFLOAD_PREFETCH_GATHER", "true")
    lens, tails = [64, 40], [8, 12]
    offload = _make_offload(lens, tails, quant=True)
    req_to_token, fb, slots = _scene(offload, lens, tails, seed=11, mirror="scale")
    offload.begin_forward(req_to_token, fb.req_pool_indices)
    offload.prefetch_prepare(fb)
    tail_mask, tail_rows = _hot_tail_args(offload, lens, tails)
    for layer in range(LAYERS):
        if layer:
            offload.maybe_prefetch_next(layer - 1)
        rec = offload.take_prefetch(layer)
        got = offload.finish_prefetched_gather(rec, layer_idx=layer, slots=slots.long())
        hot = offload.gather_context_hot_tail(
            layer_idx=layer, slots=slots, tail_mask=tail_mask, tail_rows=tail_rows
        )
        ref = offload.gather_context(layer_idx=layer, slots=slots)
        for a, b in zip(got, hot):
            if a is None:
                continue
            assert torch.equal(a, b), f"layer {layer}: tail refill diverged"
        assert not torch.equal(got[0], ref[0]), f"layer {layer}: stale bytes survived"
    offload.destroy()


def test_gates_leave_no_plan(monkeypatch):
    # One offload for the whole gating matrix: every pinned host plane is a
    # cudaHostRegister, and the machine may have little left to spare.
    lens, tails = [40, 20], [6, 4]
    off = _make_offload(lens, tails, quant=True)
    rt, fb, _ = _scene(off, lens, tails, seed=3)
    off.begin_forward(rt, fb.req_pool_indices)

    # Env off (default): no plan, and nothing to take.
    off.prefetch_prepare(fb)
    assert off._gather_plan is None
    assert off.take_prefetch(0) is None

    monkeypatch.setenv("SGLANG_QSA_OFFLOAD_PREFETCH_GATHER", "true")

    # Row cap: the plan would copy more rows than allowed.
    monkeypatch.setenv("SGLANG_QSA_OFFLOAD_PREFETCH_MAX_ROWS", "10")
    off.prefetch_prepare(fb)
    assert off._gather_plan is None
    monkeypatch.delenv("SGLANG_QSA_OFFLOAD_PREFETCH_MAX_ROWS")

    # Verify offload: gathers go through check(), never a copy.
    off.verify = True
    off.prefetch_prepare(fb)
    assert off._gather_plan is None
    off.verify = False

    # Missing host mirror: the plan cannot be resolved without a sync.
    keep = fb.extend_seq_lens
    fb.extend_seq_lens = None
    off.prefetch_prepare(fb)
    assert off._gather_plan is None
    fb.extend_seq_lens = keep

    # No request has a cached prefix: forward_extend skips the gather
    # entirely, so no copy would ever be taken.
    fb.seq_lens_cpu = torch.tensor(lens, dtype=torch.int64).clone()
    fb.extend_seq_lens_cpu = list(lens)
    fb.extend_seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    off.prefetch_prepare(fb)
    assert off._gather_plan is None

    off.destroy()


def test_debug_mode_detects_divergence(monkeypatch):
    monkeypatch.setenv("SGLANG_QSA_OFFLOAD_PREFETCH_GATHER", "true")
    monkeypatch.setenv("SGLANG_QSA_OFFLOAD_PREFETCH_DEBUG", "true")
    lens, tails = [64, 32], [8, 0]
    offload = _make_offload(lens, tails, quant=True)
    req_to_token, fb, slots = _scene(offload, lens, tails, seed=5)
    offload.begin_forward(req_to_token, fb.req_pool_indices)
    offload.prefetch_prepare(fb)
    torch.cuda.synchronize()
    # Simulate the race the debug mode exists to catch: a prefix row the
    # copy already read changes on the host before the gather consumes it.
    prefix_slot = int(slots[0].cpu())
    offload.host_k[0][prefix_slot] = 0xA5
    rec = offload.take_prefetch(0)
    with pytest.raises(RuntimeError, match="differs"):
        offload.finish_prefetched_gather(rec, layer_idx=0, slots=slots.long())
    offload.destroy()


def test_production_order_engages(monkeypatch):
    """The wiring in gather_offloaded_context must take the prefetch on EVERY
    QSA layer when gathers run in production order (take plane L, then
    trigger plane L+1). Guards the record slot against the store-time
    trigger regression: a trigger that issues L+1 before take(L) would
    clobber the record and silently fall back."""
    from sglang.srt.layers.attention.qsa.kv_offload_attention import (
        gather_offloaded_context,
    )

    monkeypatch.setenv("SGLANG_QSA_OFFLOAD_PREFETCH_GATHER", "true")
    lens, tails = [48, 30], [8, 6]
    offload = _make_offload(lens, tails, quant=False)
    req_to_token, fb, slots = _scene(offload, lens, tails, seed=23)

    def _no_fallback(*a, **k):
        raise AssertionError("fell back off the prefetch gather path")

    monkeypatch.setattr(offload, "gather_context_hot_tail", _no_fallback)
    monkeypatch.setattr(offload, "gather_context", _no_fallback)

    pool = SimpleNamespace(
        full_attention_layer_index=lambda layer_id: layer_id,
        layer_router=SimpleNamespace(
            read=lambda layer_id: SimpleNamespace(
                key=lambda: offload.dev_k[layer_id],
                value=lambda: offload.dev_v[layer_id],
            )
        ),
    )
    backend = SimpleNamespace(_kv_offload=offload, token_to_kv_pool=pool)
    offload.begin_forward(req_to_token, fb.req_pool_indices)
    offload.prefetch_prepare(fb)
    seq = torch.tensor(lens, device="cuda", dtype=torch.int32)
    cu = torch.nn.functional.pad(seq.cumsum(0), (1, 0)).long()
    for layer in range(LAYERS):
        gather_offloaded_context(
            backend=backend,
            layer=SimpleNamespace(layer_id=layer),
            forward_batch=fb,
            sequence_lens=seq,
            cu_seqlens_k=cu,
            total_k=int(seq.sum()),
            output_dtype=torch.uint8,
        )
        # The record was taken and (except after the last plane) the copy
        # of the NEXT plane — and only of the next plane — is queued.
        if layer + 1 < LAYERS:
            assert offload._prefetched is not None
            assert offload._prefetched.layer_idx == layer + 1
        else:
            assert offload._prefetched is None
    offload.destroy()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
