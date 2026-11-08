"""Packed fp6-e2m3 storage for the offloaded Qwen4-Exp PLE table.

The checkpoint holds each shard as uint8 codes (4 six-bit codes per 3-byte
little-endian group) plus one float32 scale per row, produced by an offline
per-row 21-divisor sweep (ReMe digest/tasks/ple-fp6-e2m3). This test covers
the serving side: the Triton gather unpacks codes, applies the per-row scale,
and emits the same bf16 a dequant reference computes across the vocabulary
shard boundaries.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpPinnedHostEmbedding,
    _ple_table_is_fp6,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_accelerator = torch.accelerator.current_accelerator()
DEVICE = _accelerator.type if _accelerator is not None else None

pytestmark = pytest.mark.skipif(DEVICE is None, reason="a device is required.")

E2M3 = torch.tensor(
    [
        0.0,
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
        1.125,
        1.25,
        1.375,
        1.5,
        1.625,
        1.75,
        1.875,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
    ],
    dtype=torch.float32,
)


def quantize_e2m3_rows(values: torch.Tensor):
    """values [N, D] fp32 -> (uint8 packed codes [N, D//4*3], fp32 scales [N,1])
    in the offline builder's layout, plus the bf16 dequant reference."""
    scales = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 6.0
    u = (values / scales).clamp(-7.5, 7.5)
    e2m3 = E2M3.to(values.device)
    grid = torch.bucketize(u.abs(), (e2m3[1:] + e2m3[:-1]) / 2)
    codes = (grid | torch.where(u < 0, 32, 0)).to(torch.int32)
    ref = torch.sign(u) * e2m3[grid] * scales
    packed4 = (
        codes[:, 0::4]
        | (codes[:, 1::4] << 6)
        | (codes[:, 2::4] << 12)
        | (codes[:, 3::4] << 18)
    )
    packed = (
        packed4.view(torch.uint8)
        .reshape(values.shape[0], -1, 4)[:, :, :3]
        .reshape(values.shape[0], -1)
    )
    return packed.to(torch.uint8), scales, ref.to(torch.bfloat16)


def _make_fp6_source(*, embedding_dim=160, vocab_start=0, vocab_end=8):
    local_rows = vocab_end - vocab_start
    weight = nn.Parameter(
        torch.zeros(
            (local_rows, embedding_dim // 4 * 3), dtype=torch.uint8, device=DEVICE
        ),
        requires_grad=False,
    )
    set_weight_attrs(
        weight,
        {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": lambda *_a, **_k: None,
        },
    )
    shard_indices = VocabParallelEmbeddingShardIndices(
        padded_org_vocab_start_index=vocab_start,
        padded_org_vocab_end_index=vocab_end,
        padded_added_vocab_start_index=vocab_end,
        padded_added_vocab_end_index=vocab_end,
        org_vocab_start_index=vocab_start,
        org_vocab_end_index=vocab_end,
        added_vocab_start_index=vocab_end,
        added_vocab_end_index=vocab_end,
    )
    return SimpleNamespace(
        weight=weight,
        quant_config=None,
        enable_tp=True,
        use_attn_tp_group=False,
        tp_size=1,
        num_embeddings=vocab_end,
        org_vocab_size=vocab_end,
        padding_size=1,
        num_added_embeddings=0,
        use_presharded_weights=False,
        org_vocab_size_padded=vocab_end,
        num_embeddings_padded=vocab_end,
        shard_indices=shard_indices,
        embedding_dim=embedding_dim,
        weight_scale=torch.ones(1, dtype=torch.bfloat16, device=DEVICE),
        quant_method=UnquantizedEmbeddingMethod(),
        num_embeddings_per_partition=local_rows,
        num_org_embeddings_per_partition=local_rows,
        num_added_embeddings_per_partition=0,
    )


def test_packed_fp6_gather_matches_dequant_reference():
    torch.manual_seed(0)
    embedding_dim = 160
    rows = torch.randn(8, embedding_dim, device=DEVICE) * torch.exp(
        torch.randn(8, 1, device=DEVICE) * 3
    )
    packed, scales, ref = quantize_e2m3_rows(rows)
    offloaded = Qwen4ExpPinnedHostEmbedding(_make_fp6_source())
    offloaded.weight.copy_(packed.to(offloaded.weight.device))
    offloaded.row_scales.copy_(scales.to(offloaded.row_scales.device))

    ids = torch.tensor([[0, 7, 3], [4, 1, 6]], device=DEVICE)
    torch.testing.assert_close(
        offloaded.gather(ids),
        ref.index_select(0, ids.flatten()).reshape(2, 3, embedding_dim),
        rtol=0,
        atol=0,
    )


def test_packed_fp6_gather_tp_shard_boundaries():
    torch.manual_seed(1)
    rows = torch.randn(4, 160, device=DEVICE)
    packed, scales, ref = quantize_e2m3_rows(rows)
    # This rank owns the vocabulary window [4, 8).
    offloaded = Qwen4ExpPinnedHostEmbedding(
        _make_fp6_source(vocab_start=4, vocab_end=8)
    )
    offloaded.weight.copy_(packed.to(offloaded.weight.device))
    offloaded.row_scales.copy_(scales.to(offloaded.row_scales.device))

    out = offloaded.gather(torch.tensor([[-1, 3, 4], [7, 8, 100]], device=DEVICE))
    torch.testing.assert_close(out[0, 2], ref[0], rtol=0, atol=0)  # global 4 -> row 0
    torch.testing.assert_close(out[1, 0], ref[3], rtol=0, atol=0)  # global 7 -> row 3
    assert torch.count_nonzero(out[0, :2]) == 0  # 3 and below: other shard
    assert torch.count_nonzero(out[1, 1:]) == 0  # 8 and 100: out of vocabulary


def test_packed_fp6_config_gate_and_guards():
    from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig

    cfg = Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=16,
        hc_count=2,
        ple_embed_dim=64,
        ple_embedding_dtype="fp6_e2m3",
    )
    assert _ple_table_is_fp6(cfg)
    assert not _ple_table_is_fp6(
        Qwen4ExpTextConfig(vocab_size=64, hidden_size=16, hc_count=2, ple_embed_dim=64)
    )
    with pytest.raises(ValueError, match="multiple of 32"):
        Qwen4ExpPinnedHostEmbedding(_make_fp6_source(embedding_dim=48))
    fp16 = _make_fp6_source()
    fp16.weight = nn.Parameter(
        torch.zeros((8, 160), dtype=torch.float16, device=DEVICE), requires_grad=False
    )
    with pytest.raises(TypeError, match="packed fp6"):
        Qwen4ExpPinnedHostEmbedding(fp16)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
