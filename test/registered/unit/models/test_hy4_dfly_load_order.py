"""Real-weights DFly load order: draft fixups run after draft weights exist.

DefaultModelLoader calls DeepseekV2WeightLoaderMixin.do_load_weights, which
invokes post_load_weights from inside itself; Hy4 then loads the draft weights
afterwards. Deferring only the draft fixups prevents the real checkpoint from
hitting the "draft FFNs absent" fallback before their shapes have materialized
them. Dummy/sharded loads bypass model.load_weights and must still be fixed
up directly from post_load_weights.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from sglang.srt.models import hunyuan_v4_flash as hy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


@pytest.fixture
def model(monkeypatch):
    m = object.__new__(hy.HYV4FlashForConditionalGeneration)
    m.config = SimpleNamespace(tie_word_embeddings=False)
    m.dfly = MagicMock()
    m.lm_head = MagicMock()
    m.visual = None
    m.model = SimpleNamespace(layers=[], engram_hasher=SimpleNamespace(_loaded=True))
    m.engram_hasher = SimpleNamespace(_loaded=True)
    m._dfly_load_in_progress = False
    features = SimpleNamespace(enable_fp8_weights=False, lm_head_dtype="bf16")
    monkeypatch.setattr(
        hy,
        "get_exec",
        lambda: SimpleNamespace(
            features=features,
            offload=SimpleNamespace(hy4_engram_offload=False),
        ),
    )
    monkeypatch.setattr(
        hy.DeepseekV2WeightLoaderMixin, "post_load_weights", lambda *a, **kw: None
    )
    return m


def test_dummy_post_load_still_fixes_draft(model):
    model.post_load_weights()
    model.dfly.post_load_weights.assert_called_once()


def test_post_load_defers_when_real_draft_weights_are_pending(model):
    model._dfly_load_in_progress = True
    model.post_load_weights()
    model.dfly.post_load_weights.assert_not_called()


def test_real_loader_fixes_draft_after_loading_its_weights(model, monkeypatch):
    events = []

    def fake_do_load(weights):
        assert model._dfly_load_in_progress
        list(weights)  # fills dfly_weights before post_load runs
        events.append("main")
        model.post_load_weights()

    def fake_dfly_load(weights):
        assert len(list(weights)) == 1
        events.append("draft-load")

    def fake_dfly_post():
        events.append("draft-post")

    monkeypatch.setattr(model, "do_load_weights", fake_do_load)
    monkeypatch.setattr(model.dfly, "load_weights", fake_dfly_load)
    monkeypatch.setattr(model.dfly, "post_load_weights", fake_dfly_post)

    model.load_weights([("model.dfly_layers.0.mlp.gate_proj.weight", torch.zeros(1))])

    assert events == ["main", "draft-load", "draft-post"]
    assert not model._dfly_load_in_progress
