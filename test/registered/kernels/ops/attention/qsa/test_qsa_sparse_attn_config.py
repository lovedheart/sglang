"""Device-name-independent dispatch tests for the sparse prefill tiling."""

import pytest

from sglang.srt.layers.attention.qsa import sparse_attn
from sglang.srt.layers.attention.qsa.sparse_attn import _get_best_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


@pytest.fixture(autouse=True)
def _not_sm120(monkeypatch):
    """The name-keyed tables are a fallback; pin SM120 off so these tests say
    the same thing on an SM120 box as on the CPU runners they run on."""
    import sglang.srt.utils

    monkeypatch.setattr(sglang.srt.utils, "is_sm120", lambda: False)


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
    monkeypatch.setattr(sparse_attn.torch.cuda, "get_device_name", lambda idx=0: name)
    table = (
        sparse_attn._H20_CONFIGS
        if expected_table == "h20"
        else sparse_attn._L20_CONFIGS
    )
    for total_q in (1, 32, 64, 128, 512, 4096, 100000):
        assert _get_best_config(total_q) == next(
            cfg for limit, cfg in table if total_q <= limit
        ), f"{name!r} at total_q={total_q}"


def test_sm120_table_wins_by_capability_not_name(monkeypatch):
    """SM120 gets its measured table regardless of the name (a "GeForce RTX
    5090" string must not fall through to the L20 hand-me-downs)."""
    import sglang.srt.utils

    monkeypatch.setattr(sglang.srt.utils, "is_sm120", lambda: True)
    monkeypatch.setattr(
        sparse_attn.torch.cuda,
        "get_device_name",
        lambda idx=0: "NVIDIA GeForce RTX 5090",
    )
    # The bucket that moved: NEXTN verify rows run at <=32 queries.
    assert _get_best_config(16) == (64, 4, 2) != sparse_attn._L20_CONFIGS[0][1]
    for limit, cfg in sparse_attn._SM120_CONFIGS:
        assert _get_best_config(limit) == cfg


def test_tables_cover_every_total_q():
    import math

    assert _get_best_config(1) == (32, 8, 2)
    assert _get_best_config(10**9) == (16, 1, 2)
    for table in (
        sparse_attn._L20_CONFIGS,
        sparse_attn._H20_CONFIGS,
        sparse_attn._SM120_CONFIGS,
    ):
        assert math.isinf(table[-1][0])


def test_multi_warp_flag_only_replaces_the_one_warp_bucket(monkeypatch):
    from sglang.srt.environ import envs

    def set_name(name):
        monkeypatch.setattr(
            sparse_attn.torch.cuda, "get_device_name", lambda idx=0: name
        )

    set_name("NVIDIA L20")
    with envs.SGLANG_QSA_PREFILL_MULTI_WARP.override(False):
        assert _get_best_config(10**9) == (16, 1, 2)
    with envs.SGLANG_QSA_PREFILL_MULTI_WARP.override(True):
        assert _get_best_config(10**9) == sparse_attn._MULTI_WARP_CONFIG == (16, 2, 3)
        # Every bucket someone else tuned keeps its own launch, on either table.
        for name, table in (
            ("NVIDIA L20", sparse_attn._L20_CONFIGS),
            ("NVIDIA H20", sparse_attn._H20_CONFIGS),
        ):
            set_name(name)
            for limit, cfg in table:
                if cfg[1] == 1:
                    continue
                # A bucket's own limit is the smallest total_q that reaches it.
                assert _get_best_config(limit) == cfg, f"{name} at {limit}"
        # The SM120 table spells the tail multi-warp already, so the flag is a
        # no-op on it by construction.
        import sglang.srt.utils

        monkeypatch.setattr(sglang.srt.utils, "is_sm120", lambda: True)
        assert sparse_attn._SM120_CONFIGS[-1][1] == sparse_attn._MULTI_WARP_CONFIG
        assert _get_best_config(10**9) == sparse_attn._MULTI_WARP_CONFIG
