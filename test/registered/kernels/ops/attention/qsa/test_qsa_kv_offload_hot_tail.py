"""Hot-tail split gather of the QSA K/V host offload (quantized pools).

``gather_context_hot_tail`` reads the current prefill chunk's rows from the
offload's device mirror scratch (where ``store_offloaded_kv`` just wrote them)
and the cached prefix from the host store. The assembled packed K/V and scale
planes must be bit-identical to the all-host ``gather_context``. The
tail-mapping expression mirrors the one in
``kv_offload_attention.gather_offloaded_context``.
"""

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


def _fill_and_check(lens, tails, seed):
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
        torch.zeros((layout.device_rows + 64, HEADS, DIM // 2), device="cuda", dtype=torch.uint8)
        for _ in range(LAYERS)
    ]  # fmt: skip
    dev_ksf = [
        torch.zeros((layout.device_rows + 64, HEADS, DIM // 16), device="cuda", dtype=torch.uint8)
        for _ in range(LAYERS)
    ]  # fmt: skip
    offload = QSAKVHostOffload(
        layout=layout,
        k_buffers=dev_k,
        v_buffers=[torch.zeros_like(t) for t in dev_k],
        host_rows=HOST_ROWS,
        pin_memory=True,
        k_sf_buffers=dev_ksf,
        v_sf_buffers=[torch.zeros_like(t) for t in dev_ksf],
    )
    g = torch.Generator().manual_seed(seed)
    # Globally disjoint logical slots; the host plane holds the past-chunk rows.
    perm = (
        torch.randperm(HOST_ROWS - 64, generator=torch.Generator().manual_seed(seed))
        + 64
    )
    slots_by_req, p0 = [], 0
    for n in lens:
        slots_by_req.append(perm[p0 : p0 + n].cuda().int())
        p0 += n
    for r, n in enumerate(lens):
        s = slots_by_req[r].long().cpu()
        for plane in (offload.host_k, offload.host_v):
            plane[0][s] = torch.randint(
                0, 256, (n, HEADS, DIM // 2), generator=g, dtype=torch.uint8
            )
        for plane in (offload.host_ksf, offload.host_vsf):
            plane[0][s] = torch.randint(0, 256, (n, HEADS, DIM // 16), generator=g, dtype=torch.uint8)  # fmt: skip
    # Mirror the current chunk: device scratch rows plus host copies at the
    # tail slots (what store + write_through leave behind).
    rows = offload.extend_rows(sum(tails))
    tail_slots = (
        torch.cat(
            [
                slots_by_req[b][lens[b] - tails[b] :]
                for b in range(len(lens))
                if tails[b] > 0
            ]
        )
        .long()
        .cpu()
    )
    for dst, src in (
        (offload.host_k, offload.dev_k),
        (offload.host_v, offload.dev_v),
        (offload.host_ksf, offload.dev_ksf),
        (offload.host_vsf, offload.dev_vsf),
    ):
        dst[0][tail_slots] = src[0][rows].cpu()
    # Production shape: ``req_to_token`` is int32 and the gather is entered
    # with its raw int32 lookup (gather_context converts internally).
    slots = torch.cat([s.long() for s in slots_by_req]).to(torch.int32)
    seq = torch.tensor(lens, device="cuda", dtype=torch.int32)
    cu = torch.nn.functional.pad(seq.cumsum(0)[:-1], (1, 0)).long()
    total_k = int(seq.sum())
    owner = torch.repeat_interleave(
        torch.arange(len(lens), device="cuda"), seq.long(), output_size=total_k
    )
    positions = torch.arange(total_k, device="cuda") - cu[owner]
    # The exact tail mapping from gather_offloaded_context.
    e = torch.tensor(tails, device="cuda", dtype=torch.int32).long()
    starts = seq.long()[owner] - e[owner]
    tail_mask = positions >= starts
    rows_before = torch.cumsum(e, 0) - e
    tail_rows = (
        offload.layout.scratch_base + rows_before[owner] + (positions - starts)
    )[tail_mask]
    for layer in range(LAYERS):
        ref = offload.gather_context(layer_idx=layer, slots=slots)
        got = offload.gather_context_hot_tail(
            layer_idx=layer, slots=slots, tail_mask=tail_mask, tail_rows=tail_rows
        )
        assert all(torch.equal(a, b) for a, b in zip(got, ref)), f"layer {layer}"
    offload.destroy()


@pytest.mark.parametrize(
    "lens,tails",
    [
        ([512, 400, 460, 430], [256, 200, 240, 210]),  # mixed prefix + tail
        ([512, 400, 460, 430], [512, 400, 460, 430]),  # everything is current chunk
        ([512, 400, 460, 430], [0, 400, 460, 430]),  # a zero-extend row
        ([600], [600]),  # single request
        ([600], [64]),  # single request, small tail
    ],
)
def test_hot_tail_matches_host_gather(lens, tails):
    _fill_and_check(lens, tails, seed=len(lens) * 1000 + sum(tails))
