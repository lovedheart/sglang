"""Device-independent dispatch tests for the sparse prefill tiling."""

import math
import sys

import pytest

from sglang.srt.layers.attention.qsa import sparse_attn
from sglang.srt.layers.attention.qsa.sparse_attn import _get_best_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_TABLE_ATTR = {
    "h20": "_H20_CONFIGS",
    "l20": "_L20_CONFIGS",
    "sm120": "_SM120_CONFIGS",
}


def _patch_device(monkeypatch, name, capability):
    monkeypatch.setattr(sparse_attn.torch.cuda, "get_device_name", lambda idx=0: name)
    monkeypatch.setattr(
        sparse_attn.torch.cuda, "get_device_capability", lambda idx=0: capability
    )


@pytest.mark.parametrize(
    "name,capability,expected",
    [
        ("NVIDIA H20", (9, 0), "h20"),
        ("NVIDIA H200", (9, 0), "h20"),  # substring match is intentional
        ("NVIDIA L20", (8, 9), "l20"),
        ("NVIDIA GeForce RTX 5090", (12, 0), "sm120"),
        ("NVIDIA RTX PRO 6000 Blackwell", (12, 0), "sm120"),
        ("NVIDIA GB10", (12, 1), "sm120"),  # every SM12x SKU, by capability
        ("AMD Instinct MI300X", (9, 0), "l20"),
        ("", (8, 0), "l20"),
    ],
)
def test_config_table_selection(monkeypatch, name, capability, expected):
    _patch_device(monkeypatch, name, capability)
    table = getattr(sparse_attn, _TABLE_ATTR[expected])
    for total_q in (1, 16, 32, 64, 128, 512, 4096, 100000):
        assert _get_best_config(total_q) == next(
            cfg for limit, cfg in table if total_q <= limit
        ), f"{name!r} at total_q={total_q}"


def test_tables_cover_every_total_q(monkeypatch):
    forcing = [
        (sparse_attn._H20_CONFIGS, ("NVIDIA H20", (9, 0))),
        (sparse_attn._L20_CONFIGS, ("NVIDIA L20", (8, 9))),
        (sparse_attn._SM120_CONFIGS, ("NVIDIA RTX PRO 6000", (12, 0))),
    ]
    for table, device in forcing:
        _patch_device(monkeypatch, *device)
        assert math.isinf(table[-1][0])
        previous_limit = 0
        for limit, cfg in table:
            assert _get_best_config(previous_limit + 1) == cfg
            assert _get_best_config(limit) == cfg
            if math.isinf(limit):
                break
            previous_limit = limit


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
