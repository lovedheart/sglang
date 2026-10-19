# SPDX-License-Identifier: Apache-2.0
"""Owner tests for the sm12x_nvfp4_bf16_disk backend: the streamed slot
cache must be numerically identical to the resident backend on a synthetic
checkpoint -- cold fill, all-hit, LRU thrash with the slot pool pinned at
the expert count, CUDA-graph replay through a breakable capture. SM120/
SM121 only; a few MiB of VRAM."""

import os
import types

import pytest
import torch
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.sm12x_nvfp4_bf16 import (
    Sm12xNvfp4W4A16MoeQuantInfo,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from sglang.srt.runtime_context import get_context, get_flags, reset_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-small")

cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
if cc not in ((12, 0), (12, 1)):
    pytest.skip("sm12x kernels require SM120/SM121", allow_module_level=True)

DEV = "cuda"
E, H, I, TOPK = 8, 256, 128, 2
LAYERS = (0, 1)
RES = MoeRunnerBackend.SM12X_NVFP4_BF16
DSK = MoeRunnerBackend.SM12X_NVFP4_BF16_DISK
# run_moe_w4a16 is not bit-deterministic (fp32 atomic combine order); the
# parity gate is noise-derived in _gate(), floored at 2 bf16 ulp.
_GATE = 2e-3

_CFG = MoeRunnerConfig(
    num_experts=E,
    hidden_size=H,
    intermediate_size_per_partition=I,
    top_k=TOPK,
    is_gated=True,
    activation="silu",
    num_fused_shared_experts=0,
)


def _expert_raw(L, e):
    """The per-expert tensors exactly as a modelopt checkpoint stores them
    (FP4-packed payload bytes, raw-layout scale factors, per-proj F32
    global scales)."""
    g = torch.Generator().manual_seed(1000 * L + e)
    w13 = torch.randint(0, 256, (2 * I, H // 2), dtype=torch.uint8, generator=g)
    w13_sf = (torch.rand(2 * I, H // 16, generator=g) * 0.2).to(torch.float8_e4m3fn)
    gs = (10 ** torch.empty(2).uniform_(-3, -2, generator=g)).float()
    w2 = torch.randint(0, 256, (H, I // 2), dtype=torch.uint8, generator=g)
    w2_sf = (torch.rand(H, I // 16, generator=g) * 0.2).to(torch.float8_e4m3fn)
    w2_gs = (10 ** torch.empty(1).uniform_(-3, -2, generator=g)).float()
    return {
        "w13": w13,
        "w13_sf": w13_sf,
        "gs": gs,
        "w2": w2,
        "w2_sf": w2_sf,
        "w2_gs": w2_gs,
    }


def _write_ckpt(d, raws):
    from safetensors.torch import save_file

    tensors = {}
    for L in LAYERS:
        for e in range(E):
            r = raws[L][e]
            pre = f"model.layers.{L}.mlp.experts.{e}"
            for nm, w, s, gs in (
                ("gate_proj", r["w13"][:I], r["w13_sf"][:I], r["gs"][0]),
                ("up_proj", r["w13"][I:], r["w13_sf"][I:], r["gs"][1]),
                ("down_proj", r["w2"], r["w2_sf"], r["w2_gs"]),
            ):
                tensors[f"{pre}.{nm}.weight"] = w.contiguous()
                tensors[f"{pre}.{nm}.weight_scale"] = s.contiguous()
                tensors[f"{pre}.{nm}.weight_scale_2"] = gs.reshape(())
    save_file(tensors, os.path.join(str(d), "model.safetensors"))


def _qi(L, raw, meta_payload):
    """Layer params -> the production sm12x weight prep -> quant_info.
    Byte payloads are real VRAM tensors (resident) or meta (streamed);
    scales are always resident and prepped identically."""
    dev = "meta" if meta_payload else DEV
    w13 = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device=dev)
    w2 = torch.empty(E, H, I // 2, dtype=torch.uint8, device=dev)
    if not meta_payload:
        for e in range(E):
            w13[e].copy_(raw[e]["w13"])
            w2[e].copy_(raw[e]["w2"])
    layer = torch.nn.Module()
    layer.moe_runner_config = types.SimpleNamespace(
        is_gated=True, activation="silu", num_fused_shared_experts=0
    )
    for nm, t in (
        ("w13_weight", w13),
        ("w13_weight_scale", torch.stack([r["w13_sf"] for r in raw]).to(DEV)),
        ("w13_weight_scale_2", torch.stack([r["gs"] for r in raw]).to(DEV)),
        ("w2_weight", w2),
        ("w2_weight_scale", torch.stack([r["w2_sf"] for r in raw]).to(DEV)),
        (
            "w2_weight_scale_2",
            torch.stack([r["w2_gs"][0] for r in raw]).to(DEV),
        ),
    ):
        setattr(layer, nm, torch.nn.Parameter(t, requires_grad=False))
    ModelOptNvFp4FusedMoEMethod._prepare_sm12x_w4a16_weights(None, layer)
    return Sm12xNvfp4W4A16MoeQuantInfo(
        w13=layer.w13_weight.data,
        w13_sf=layer.w13_weight_scale.data,
        w13_gs=layer.w13_weight_scale_2.data,
        w2=layer.w2_weight.data,
        w2_sf=layer.w2_weight_scale.data,
        w2_gs=layer.w2_weight_scale_2.data,
        layer_id=L,
    )


@pytest.fixture()
def ck(request, tmp_path):
    """Synthetic ckpt published as model_path. ``slots_mb`` (default 1 ->
    K=16 slots) via indirect parametrisation; 0 pins K to the expert count
    so every call must thrash."""
    expert_slots = getattr(request, "param", 1)
    raws = {L: [_expert_raw(L, e) for e in range(E)] for L in LAYERS}
    _write_ckpt(tmp_path, raws)
    override = get_context().override_server_args(model_path=str(tmp_path))
    override.install()
    knobs = (
        (envs.SGLANG_SM12X_DISK_VRAM_EXPERT_SLOTS, expert_slots),
        (envs.SGLANG_SM12X_DISK_PREFETCH, 0),
        (envs.SGLANG_SM12X_DISK_PREWARM, False),
        (envs.SGLANG_SM12X_DISK_RESIDENT_LAYERS, ""),
    )
    saved = [(k, k.get()) for k, _ in knobs]
    for k, v in knobs:
        k.set(v)
    try:
        yield raws
    finally:
        for k, v in saved:
            k.set(v)
        override.restore()
        reset_context()


def _inp(tag, t):
    g = torch.Generator(device=DEV).manual_seed(555 + abs(hash(tag)) % 100003)
    x = (torch.randn(t, H, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    ids = torch.randint(0, E, (t, TOPK), generator=g, device=DEV, dtype=torch.int32)
    wts = torch.rand(t, TOPK, generator=g, device=DEV)
    return StandardDispatchOutput(
        x,
        None,
        StandardTopKOutput(
            topk_weights=wts,
            topk_ids=ids,
            router_logits=torch.zeros(t, E, device=DEV),
        ),
    )


def _run(backend, qi, disp):
    with get_flags().moe.override(
        runner_backend=backend, a2a_backend=MoeA2ABackend.NONE
    ):
        return MoeRunner(backend, _CFG).run(disp, qi).hidden_states.clone()


def _gate(qi, disp):
    """Kernel run-noise gate from two resident runs of the same batch."""
    global _GATE
    a = _run(RES, qi, disp)
    b = _run(RES, qi, disp)
    _GATE = max((a.float() - b.float()).abs().max().item(), 1e-3) * 2
    return a


def _check(got, want, tag):
    dd = (got.float() - want.float()).abs().max().item()
    assert torch.isfinite(got.float()).all(), f"{tag}: nonfinite output"
    assert dd <= _GATE, f"{tag}: maxdiff {dd:.2e} > gate {_GATE:.2e}"


def test_cold_then_hit_matches_resident(ck):
    raws = ck
    res0 = _qi(LAYERS[0], raws[LAYERS[0]], meta_payload=False)
    dsk0 = _qi(LAYERS[0], raws[LAYERS[0]], meta_payload=True)
    ref = _gate(res0, _inp("warm", 8))
    _check(_run(DSK, dsk0, _inp("cold", 8)), ref, "cold fill")
    got = _run(DSK, dsk0, _inp("warm", 8))  # same batch: all-hit path
    _check(got, ref, "all hit")
    assert got.dtype == torch.bfloat16


@pytest.mark.parametrize("ck", [0], indirect=True)
def test_thrash_at_k_equals_e(ck):
    # K == E slots shared across two layers: every call evicts and refills,
    # and correctness has to survive the churn.
    raws = ck
    res = {L: _qi(L, raws[L], meta_payload=False) for L in LAYERS}
    dsk = {L: _qi(L, raws[L], meta_payload=True) for L in LAYERS}
    _gate(res[LAYERS[0]], _inp("w", 8))
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_gather import get_slot_cache
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_registry import (
        get_expert_registry,
    )

    for r in range(4):
        for L in LAYERS:
            d = _inp((r, L), 8)
            _check(_run(DSK, dsk[L], d), _run(RES, res[L], d), f"thrash r{r} L{L}")
    cache = get_slot_cache(get_expert_registry(), dsk[LAYERS[0]], E)
    if not envs.SGLANG_SM12X_DISK_INGRAPH.get():
        assert cache.misses > 3 * cache.hits, "budget 0 should thrash, not hit"


def test_padding_bucket(ck):
    raws = ck
    res0 = _qi(LAYERS[0], raws[LAYERS[0]], meta_payload=False)
    dsk0 = _qi(LAYERS[0], raws[LAYERS[0]], meta_payload=True)
    ref = _gate(res0, _inp("warm3", 3))
    _check(_run(DSK, dsk0, _inp("warm3", 3)), ref, "padded bucket")


# ---- layout B: per-expert ffn naming, scale factors streamed from disk ----


def _write_ckpt_b(d, raws):
    from safetensors.torch import save_file

    tensors = {}
    for L in LAYERS:
        for e in range(E):
            r = raws[L][e]
            pre = f"layers.{L}.ffn.experts.{e}"
            for nm, w, s, gs in (
                ("w1", r["w13"][:I], r["w13_sf"][:I], r["gs"][0]),
                ("w3", r["w13"][I:], r["w13_sf"][I:], r["gs"][1]),
                ("w2", r["w2"], r["w2_sf"], r["w2_gs"][0]),
            ):
                tensors[f"{pre}.{nm}.weight"] = w.contiguous()
                tensors[f"{pre}.{nm}.weight_scale"] = s.contiguous()
                tensors[f"{pre}.{nm}.weight_scale_2"] = gs.reshape(())
    save_file(tensors, os.path.join(str(d), "model.safetensors"))


def _qi_b(L, raw):
    """Streamed quant info for layout B: payload and scales are meta (they
    come from disk); the fp32 global scales stay resident, with the same
    gated gate/up -> w13_gs / w2_gs fold _prepare_sm12x_w4a16_weights does."""
    gs = torch.stack([r["gs"] for r in raw]).to(DEV).to(torch.float32)
    w2_gs = torch.stack([r["w2_gs"][0] for r in raw]).to(DEV).to(torch.float32)
    return Sm12xNvfp4W4A16MoeQuantInfo(
        w13=torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device="meta"),
        w13_sf=torch.empty(
            E, 2 * I, H // 16, dtype=torch.float8_e4m3fn, device="meta"
        ),
        w13_gs=gs[:, 0].contiguous(),
        w2=torch.empty(E, H, I // 2, dtype=torch.uint8, device="meta"),
        w2_sf=torch.empty(E, H, I // 16, dtype=torch.float8_e4m3fn, device="meta"),
        w2_gs=(w2_gs * gs[:, 1] / gs[:, 0]).contiguous(),
        layer_id=L,
    )


@pytest.fixture()
def ckb(request, tmp_path):
    """Same synthetic bytes as ``ck``, published in layout B naming."""
    expert_slots = getattr(request, "param", 1)
    raws = {L: [_expert_raw(L, e) for e in range(E)] for L in LAYERS}
    _write_ckpt_b(tmp_path, raws)
    override = get_context().override_server_args(model_path=str(tmp_path))
    override.install()
    knobs = (
        (envs.SGLANG_SM12X_DISK_VRAM_EXPERT_SLOTS, expert_slots),
        (envs.SGLANG_SM12X_DISK_PREFETCH, 0),
        (envs.SGLANG_SM12X_DISK_PREWARM, False),
        (envs.SGLANG_SM12X_DISK_RESIDENT_LAYERS, ""),
    )
    saved = [(k, k.get()) for k, _ in knobs]
    for k, v in knobs:
        k.set(v)
    try:
        yield raws
    finally:
        for k, v in saved:
            k.set(v)
        override.restore()
        reset_context()


def test_b_cold_then_hit_matches_resident(ckb):
    raws = ckb
    res0 = _qi(LAYERS[0], raws[LAYERS[0]], meta_payload=False)
    dsk0 = _qi_b(LAYERS[0], raws[LAYERS[0]])
    ref = _gate(res0, _inp("bwarm", 8))
    _check(_run(DSK, dsk0, _inp("bwarm", 8)), ref, "B cold fill (scales streamed)")
    _check(_run(DSK, dsk0, _inp("bwarm", 8)), ref, "B all hit")
    # The synthetic MoE output is insensitive to x, so check the streamed
    # slot BYTES directly: payload and swizzled scales must equal what the
    # resident prep would have held.
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_gather import get_slot_cache
    from sglang.srt.layers.moe.moe_runner.sm12x_disk_registry import (
        get_expert_registry,
    )
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    cache = get_slot_cache(get_expert_registry(), dsk0, E)
    for e in range(E):
        slot = cache.slot_of[(LAYERS[0], e)]
        r = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in raws[LAYERS[0]][e].items()}
        assert torch.equal(cache.slot_w13[slot], r["w13"])
        assert torch.equal(cache.slot_w2[slot], r["w2"])
        want = swizzle_blockscale(r["w13_sf"]).view(torch.uint8)
        got = cache.slot_w13_sf[slot].view(torch.uint8)
        assert torch.equal(got, want), f"streamed w13 sf bytes wrong, expert {e}"
        want2 = swizzle_blockscale(r["w2_sf"]).view(torch.uint8)
        assert torch.equal(cache.slot_w2_sf[slot].view(torch.uint8), want2)


@pytest.mark.parametrize("ckb", [0], indirect=True)
def test_b_thrash(ckb):
    raws = ckb
    res = {L: _qi(L, raws[L], meta_payload=False) for L in LAYERS}
    dsk = {L: _qi_b(L, raws[L]) for L in LAYERS}
    for r in range(4):
        for L in LAYERS:
            d = _inp((r, L), 8)
            _check(_run(DSK, dsk[L], d), _run(RES, res[L], d), f"B thrash r{r} L{L}")


@pytest.mark.parametrize("ck", [0], indirect=True)
def test_breakable_graph_replay(ck):
    raws = ck
    dsk = {L: _qi(L, raws[L], meta_payload=True) for L in LAYERS}

    t = 1
    x = (torch.randn(t, H, device=DEV) * 0.5).to(torch.bfloat16)
    ids = torch.randint(0, E, (t, TOPK), device=DEV, dtype=torch.int32)
    wts = torch.rand(t, TOPK, device=DEV)

    def chain():
        out = x
        for L in LAYERS:
            disp = StandardDispatchOutput(
                out,
                None,
                StandardTopKOutput(
                    topk_weights=wts,
                    topk_ids=ids,
                    router_logits=torch.zeros(t, E, device=DEV),
                ),
            )
            with get_flags().moe.override(
                runner_backend=DSK, a2a_backend=MoeA2ABackend.NONE
            ):
                out = MoeRunner(DSK, _CFG).run(disp, dsk[L]).hidden_states
        return out.clone()

    ref = chain()
    ref2 = chain()
    gate = max((ref.float() - ref2.float()).abs().max().item(), 1e-3) * 2

    graph = BreakableCUDAGraph()
    with (
        enable_breakable_cuda_graph(),
        torch.cuda.stream(torch.cuda.Stream()),
        BreakableCUDAGraphCapture(graph),
    ):
        out_cap = chain()
    expected_breaks = 0 if envs.SGLANG_SM12X_DISK_INGRAPH.get() else len(LAYERS)
    assert len(graph._break_fns) == expected_breaks, "eager breaks per layer"
    graph.replay()
    torch.cuda.synchronize()
    assert (out_cap.float() - ref.float()).abs().max().item() <= gate, "first replay"

    g = torch.Generator(device=DEV).manual_seed(77)
    ids.copy_(torch.randint(0, E, (t, TOPK), generator=g, device=DEV))
    graph.replay()
    torch.cuda.synchronize()
    want = chain()  # eager ground truth against the mutated buffers
    assert (out_cap.float() - want.float()).abs().max().item() <= gate, "mutated replay"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
