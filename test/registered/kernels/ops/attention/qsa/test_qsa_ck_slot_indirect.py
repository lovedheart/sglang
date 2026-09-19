"""Bitwise parity test for the SGLANG_QSA_CK_SLOT_INDIRECT chunk-prefill path.

The slot-indirect kernel must be *bit-identical* to the gather-then-attend
path, not merely close: it reads the same K/V values in the same tile order
and accumulates identically, so every output bit must match.  This test
builds realistic chunk-prefill scenarios (multiple sequences, non-zero
prefixes, scattered pool slots, -1-padded top-k rows) and asserts
``torch.equal`` between:

* the reference -- what ``forward_extend`` does today: vectorized slot
  gather, ``index_select`` into packed full-context K/V, kernel over the copy;
* the candidate -- the same kernel family addressing the pool through the
  slot table (``slots=`` kwarg), no packed copy.

Coverage: bitwise-equal on bf16 and fp8_e4m3 pools; on NVFP4-packed pools the
in-register dequant changes only the score dot's MMA accumulation order, so
the contract pinned there is >=99% bitwise-equal with every other element
within 2 bf16 ulps of the row's output magnitude.  Also covered: MQA/GQA head
groupings, head dims 64/128,
int32 and int64 slot tables, duplicate top-k entries, fully-visible rows, and
single-token queries.
"""

import random

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-small")

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
)

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
DEVICE = "cuda"


def test_ck_slot_indirect_env_flag_defaults_off():
    # Unset must mean off everywhere: the indirect path is strictly opt-in.
    if not envs.SGLANG_QSA_CK_SLOT_INDIRECT.is_set():
        assert envs.SGLANG_QSA_CK_SLOT_INDIRECT.get() is False


def _build_case(
    *,
    seq_specs,
    q_heads,
    kv_heads,
    head_dim,
    kv_dtype,
    slot_dtype,
    topk,
    dense,
    allow_duplicates,
    pool_slots,
    seed,
    fp4=False,
):
    """Materialize one chunk-prefill scenario.

    seq_specs: list of (prefix_len, extend_len) per sequence.
    Returns q, pool_k, pool_v, indices, cu_q, cu_k, kv_lens, all_slots,
    max_q, fp4_ctx (None unless fp4, else the packed pool's SF/GS tensors).
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    bs = len(seq_specs)
    seq_lens = [p + e for p, e in seq_specs]
    extend_lens = [e for _, e in seq_specs]
    total_q = sum(extend_lens)
    total_ctx = sum(seq_lens)
    assert total_ctx + 32 <= pool_slots

    # Disjoint random pool slots per sequence, like a real allocator.
    perm = torch.randperm(pool_slots, generator=torch.Generator().manual_seed(seed))[
        :total_ctx
    ]
    req_to_token = torch.zeros((bs, max(seq_lens)), dtype=slot_dtype)
    for b, length in enumerate(seq_lens):
        req_to_token[b, :length] = perm[sum(seq_lens[:b]) : sum(seq_lens[: b + 1])]
    all_slots = torch.cat(
        [req_to_token[b, :length] for b, length in enumerate(seq_lens)]
    ).to(DEVICE)

    q = torch.randn(total_q, q_heads, head_dim, device=DEVICE, dtype=BF16)
    fp4_ctx = None
    if fp4:
        gp = torch.Generator(device="cpu").manual_seed(seed + 99991)
        pool_k = (
            torch.randint(
                0,
                256,
                (pool_slots, kv_heads, head_dim // 2),
                generator=gp,
                dtype=torch.int64,
            )
            .to(torch.uint8)
            .to(DEVICE)
        )
        pool_v = (
            torch.randint(
                0,
                256,
                (pool_slots, kv_heads, head_dim // 2),
                generator=gp,
                dtype=torch.int64,
            )
            .to(torch.uint8)
            .to(DEVICE)
        )
        # keep the fp8 e4m3 NaN bytes (0x7F/0xFF) out of the scale data
        fp4_ctx = dict(
            k_sf=torch.randint(
                0,
                127,
                (pool_slots, kv_heads, head_dim // 16),
                generator=gp,
                dtype=torch.int64,
            )
            .to(torch.uint8)
            .to(DEVICE),
            v_sf=torch.randint(
                0,
                127,
                (pool_slots, kv_heads, head_dim // 16),
                generator=gp,
                dtype=torch.int64,
            )
            .to(torch.uint8)
            .to(DEVICE),
            k_gs=(torch.rand(1, generator=gp) * 0.9 + 0.1).float().to(DEVICE),
            v_gs=(torch.rand(1, generator=gp) * 0.9 + 0.1).float().to(DEVICE),
        )
    else:
        pool_k = torch.randn(pool_slots, kv_heads, head_dim).to(kv_dtype).to(DEVICE)
        pool_v = torch.randn(pool_slots, kv_heads, head_dim).to(kv_dtype).to(DEVICE)

    # One row per query token; entry j is the j-th selected context position
    # of that row's visible prefix window, -1-padded past the selection.
    indices = torch.full((total_q, topk), -1, dtype=torch.int32)
    row = 0
    for b, (prefix, extend) in enumerate(seq_specs):
        for local in range(extend):
            visible = prefix + local + 1
            width = min(topk, visible)
            if dense:
                picked = list(range(width))
            elif allow_duplicates:
                picked = [rng.randrange(visible) for _ in range(width)]
            else:
                picked = rng.sample(range(visible), width)
            rng.shuffle(picked)
            indices[row, :width] = torch.tensor(picked, dtype=torch.int32)
            row += 1
    indices = indices.to(DEVICE)

    cu_q = (
        torch.tensor([0] + extend_lens, dtype=torch.int32, device=DEVICE)
        .cumsum(0, dtype=torch.int32)
        .contiguous()
    )
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    cu_k = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            kv_lens.cumsum(0, dtype=torch.int32),
        ]
    ).contiguous()
    return (
        q,
        pool_k,
        pool_v,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        all_slots,
        max(extend_lens),
        fp4_ctx,
    )


def _run_parity(**case_kwargs):
    assert torch.cuda.is_available(), "parity check needs a CUDA device"
    (
        q,
        pool_k,
        pool_v,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        all_slots,
        max_q,
        fp4_ctx,
    ) = _build_case(**case_kwargs)
    scale = 1.0 / (case_kwargs["head_dim"] ** 0.5)

    if fp4_ctx is None:
        # Reference: today's forward_extend -- gather packed full-context K/V.
        k_all = pool_k.index_select(0, all_slots.long())
        v_all = pool_v.index_select(0, all_slots.long())
        ref = sparse_gqa_fwd_interface_triton_ck(
            q, k_all, v_all, indices, cu_q, cu_k, kv_lens, scale, max_q=max_q
        )
        # Candidate: the same kernel family, pool addressed through the slots.
        got = sparse_gqa_fwd_interface_triton_ck(
            q,
            pool_k,
            pool_v,
            indices,
            cu_q,
            cu_k,
            kv_lens,
            scale,
            max_q=max_q,
            slots=all_slots,
        )
    else:
        # Reference: today's FP4 forward_extend -- row-gather the packed pool
        # and dequantize the whole context with NVFP4KVQuantizeUtil (on
        # SM100+/120 this dispatches to flashinfer's nvfp4_kv_dequantize).
        #
        # The FP4 candidate cannot be bitwise-equal: its dequantized K tile
        # reaches tl.dot as a computed (not loaded) operand, which Triton
        # lowers to a different MMA accumulation order on this backend.  The
        # per-element values are exact (verified); only the fp32 summation
        # order of the score dot differs, so outputs are bitwise-equal except
        # for a few tenths of a percent of elements, each within 2 bf16 ulps
        # of the row's output magnitude (<= ~0.8% relative; more in relative
        # terms only under cancellation).  Pin exactly that contract here.
        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

        safe = all_slots.clamp(min=0).long()
        k_all = NVFP4KVQuantizeUtil.dequantize(
            pool_k.index_select(0, safe),
            fp4_ctx["k_sf"].index_select(0, safe).view(torch.float8_e4m3fn),
            fp4_ctx["k_gs"],
            dtype=BF16,
        )
        v_all = NVFP4KVQuantizeUtil.dequantize(
            pool_v.index_select(0, safe),
            fp4_ctx["v_sf"].index_select(0, safe).view(torch.float8_e4m3fn),
            fp4_ctx["v_gs"],
            dtype=BF16,
        )
        ref = sparse_gqa_fwd_interface_triton_ck(
            q, k_all, v_all, indices, cu_q, cu_k, kv_lens, scale, max_q=max_q
        )
        got = sparse_gqa_fwd_interface_triton_ck(
            q,
            pool_k,
            pool_v,
            indices,
            cu_q,
            cu_k,
            kv_lens,
            scale,
            max_q=max_q,
            slots=all_slots,
            fp4=True,
            k_sf=fp4_ctx["k_sf"],
            v_sf=fp4_ctx["v_sf"],
            k_gs=fp4_ctx["k_gs"],
            v_gs=fp4_ctx["v_gs"],
        )
        # One bf16 ulp of the ROW's output magnitude bounds every deviation:
        # the score dot's fp32 accumulation order can flip one output bit, and
        # near-perfect cancellation can amplify a fixed absolute 1-ulp shift
        # in relative terms, so scale by row magnitude, not element binade.
        r, g = ref.float(), got.float()
        d = (r - g).abs()
        changed = d != 0
        row_scale = r.abs().amax(dim=tuple(range(1, d.dim())), keepdim=True).clamp(
            min=2**-126
        )
        row_ulp = torch.ldexp(
            torch.ones_like(row_scale), torch.floor(torch.log2(row_scale)) - 7.0
        )
        within = d <= 2 * row_ulp + 2**-133
        frac_changed = float(changed.float().mean())
        assert int((~within & changed).sum()) == 0, (
            f"FP4 slot-indirect output exceeds the 2-row-ulp accumulation-"
            f"order bound ({int((~within & changed).sum())} elements, "
            f"max abs {float(d.max()):g})"
        )
        assert frac_changed < 0.01, (
            f"FP4 slot-indirect changed {100 * frac_changed:.3f}% of outputs; "
            "expected <1% (single-ulp softmax-order flips only)"
        )
    if fp4_ctx is None:
        assert torch.equal(ref, got), (
            f"slot-indirect output is not bitwise-equal "
            f"(max diff {(ref.float() - got.float()).abs().max().item():g})"
        )
    assert not torch.isnan(got).any()


# (name, case kwargs)
CASES = [
    (
        "gqa128-mixed-prefixes",
        dict(
            seq_specs=[(256, 64), (512, 16), (64, 33)],
            q_heads=8,
            kv_heads=2,
            head_dim=128,
            kv_dtype=BF16,
            slot_dtype=torch.int64,
            topk=128,
            dense=False,
            allow_duplicates=False,
            pool_slots=4096,
            seed=1,
        ),
    ),
    (
        "mqa128-single-long-ctx",
        dict(
            seq_specs=[(4096, 17)],
            q_heads=16,
            kv_heads=1,
            head_dim=128,
            kv_dtype=BF16,
            slot_dtype=torch.int64,
            topk=512,
            dense=False,
            allow_duplicates=False,
            pool_slots=8192,
            seed=2,
        ),
    ),
    (
        "fp8-pool-two-seq",
        dict(
            seq_specs=[(1024, 128), (2048, 64)],
            q_heads=8,
            kv_heads=2,
            head_dim=128,
            kv_dtype=FP8,
            slot_dtype=torch.int64,
            topk=256,
            dense=False,
            allow_duplicates=False,
            pool_slots=8192,
            seed=3,
        ),
    ),
    (
        "int32-slot-table",
        dict(
            seq_specs=[(300, 24), (75, 9)],
            q_heads=8,
            kv_heads=2,
            head_dim=128,
            kv_dtype=BF16,
            slot_dtype=torch.int32,
            topk=128,
            dense=False,
            allow_duplicates=False,
            pool_slots=4096,
            seed=4,
        ),
    ),
    (
        "dense-fully-visible-rows",
        dict(
            seq_specs=[(8, 12), (0, 33)],
            q_heads=8,
            kv_heads=2,
            head_dim=128,
            kv_dtype=BF16,
            slot_dtype=torch.int64,
            topk=256,
            dense=True,
            allow_duplicates=False,
            pool_slots=2048,
            seed=5,
        ),
    ),
    (
        "duplicate-topk-entries",
        dict(
            seq_specs=[(512, 32)],
            q_heads=8,
            kv_heads=2,
            head_dim=128,
            kv_dtype=BF16,
            slot_dtype=torch.int64,
            topk=128,
            dense=False,
            allow_duplicates=True,
            pool_slots=4096,
            seed=6,
        ),
    ),
    (
        "grouped-gqa8-head64",
        dict(
            seq_specs=[(130, 7), (4096, 3)],
            q_heads=8,
            kv_heads=1,
            head_dim=64,
            kv_dtype=BF16,
            slot_dtype=torch.int64,
            topk=64,
            dense=False,
            allow_duplicates=False,
            pool_slots=8192,
            seed=7,
        ),
    ),
    (
        "single-query-token",
        dict(
            seq_specs=[(1000, 1)],
            q_heads=8,
            kv_heads=2,
            head_dim=128,
            kv_dtype=BF16,
            slot_dtype=torch.int64,
            topk=256,
            dense=False,
            allow_duplicates=False,
            pool_slots=4096,
            seed=8,
        ),
    ),
    (
        "nvfp4-gqa128-mixed-prefixes",
        dict(
            seq_specs=[(256, 64), (512, 16), (64, 33)],
            q_heads=8,
            kv_heads=2,
            head_dim=128,
            kv_dtype=BF16,
            slot_dtype=torch.int64,
            topk=128,
            dense=False,
            allow_duplicates=False,
            pool_slots=4096,
            seed=9,
            fp4=True,
        ),
    ),
    (
        "nvfp4-mqa-long-ctx",
        dict(
            seq_specs=[(8192, 17)],
            q_heads=16,
            kv_heads=1,
            head_dim=128,
            kv_dtype=BF16,
            slot_dtype=torch.int64,
            topk=512,
            dense=False,
            allow_duplicates=False,
            pool_slots=16384,
            seed=10,
            fp4=True,
        ),
    ),
    (
        "nvfp4-head64-duplicates",
        dict(
            seq_specs=[(130, 7), (2048, 5)],
            q_heads=8,
            kv_heads=1,
            head_dim=64,
            kv_dtype=BF16,
            slot_dtype=torch.int64,
            topk=64,
            dense=False,
            allow_duplicates=True,
            pool_slots=4096,
            seed=11,
            fp4=True,
        ),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[name for name, _ in CASES])
def test_ck_slot_indirect_bitwise(case):
    _run_parity(**case[1])
