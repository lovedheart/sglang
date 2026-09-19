"""Device-name-independent dispatch tests for the sparse prefill tiling."""

import pytest

from sglang.srt.layers.attention.qsa import sparse_attn
from sglang.srt.layers.attention.qsa.sparse_attn import _get_best_config


@pytest.mark.parametrize(
    "name,expected_table",
    [
        ("NVIDIA H20", "h20"),
        ("NVIDIA H200", "h20"),  # substring match is intentional
        ("NVIDIA L20", "l20"),
        ("NVIDIA GeForce RTX 5090", "l20"),
        ("AMD Instinct MI300X", "l20"),
        ("", "l20"),
    ],
)
def test_config_table_selection_is_substring_based(monkeypatch, name, expected_table):
    monkeypatch.setattr(
        sparse_attn.torch.cuda, "get_device_name", lambda idx=0: name
    )
    table = sparse_attn._H20_CONFIGS if expected_table == "h20" else sparse_attn._L20_CONFIGS
    for total_q in (1, 32, 64, 128, 512, 4096, 100000):
        assert _get_best_config(total_q) == next(
            cfg for limit, cfg in table if total_q <= limit
        ), f"{name!r} at total_q={total_q}"


def test_tables_cover_every_total_q():
    import math

    assert _get_best_config(1) == (32, 8, 2)
    assert _get_best_config(10**9) == (16, 1, 2)
    assert math.isinf(sparse_attn._L20_CONFIGS[-1][0])
    assert math.isinf(sparse_attn._H20_CONFIGS[-1][0])
