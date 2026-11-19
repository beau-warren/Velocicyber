import json

import pytest

from velocicyber.engine.context import ContextBudget
from velocicyber.engine.runner import Engine
from velocicyber.models.catalog import model_catalog
from velocicyber.state import State


def test_context_follows_loaded_model_and_never_reuses_another_models_window(settings):
    settings.options["LOCAL_MODEL_CONTEXT_WINDOWS"] = json.dumps({settings.local_model: 16384})
    state = State(
        active_provider="local",
        model_context_window=65536,
        model_context_model=settings.local_model,
    )
    engine = Engine(settings, state)
    assert engine.context_budget().window == 65536  # Actual server takes precedence.
    settings.local_model = "different/model"
    assert engine.context_budget().window == 32768  # Not the previous model's 64K limit.
    settings.local_model = state.model_context_model
    state.model_context_window = 0
    assert engine.context_budget().window == 16384  # Explicit serving profile fallback.
    state.active_provider = "api"
    assert engine.context_budget().window == 32768


@pytest.mark.parametrize("value", ["[]", "invalid", '{"model":true}', '{"model":1024}'])
def test_bad_context_profile_fails_before_launch(settings, value):
    settings.options["LOCAL_MODEL_CONTEXT_WINDOWS"] = value
    with pytest.raises(ValueError):
        settings.local_context_window()


@pytest.mark.parametrize("alias,name,model,window", [
    ("cyberglm", "CyberGLM", "lordx64/cyberglm", 131072),
    ("cyberseek", "CyberSeek", "adverserial/cyberseek", 1048576),
])
def test_api_models_use_their_own_context_budget_and_name(settings, alias, name, model, window):
    settings.api_model = alias
    engine = Engine(settings, State())
    assert settings.api_model_name == name
    assert engine.context_budget().window == window
    assert engine.selected_model == model
    settings.api_model = model
    assert settings.api_model_name == name
    assert engine.context_budget().window == window
    assert settings.is_api_model(alias)
    assert settings.is_api_model(model)
    assert settings.is_api_model("lordx64/cyberkimi")
    assert settings.is_api_model("cyberglm")
    settings.api_model = "lordx64/cyberkimi"
    assert engine.context_budget().window == 32768
    assert settings.api_model_name == "CyberKimi"


def test_small_model_skill_budget_preserves_room_for_coding():
    budget = ContextBudget(16384)
    assert budget.skills_catalog_limit == 1024
    assert budget.compact_at == 12288
    assert budget.window == 16384
    assert ContextBudget(131072).skills_catalog_limit >= 131072 * 2 // 100


def test_unknown_models_keep_fallback_and_profiles_use_serving_limits():
    assert model_catalog("unknown/custom-model", ContextBudget(), "instructions") is None
    for window in (16384, 32768, 65536):
        profile = model_catalog(
            "Blackfrost-Research/M.O.G.-SEC-27B-1M-CTX-BF16", ContextBudget(window), "Frozen session instructions"
        )["models"][0]
        assert profile["context_window"] == profile["max_context_window"] == window
        assert profile["auto_compact_token_limit"] < window
        assert profile["model_messages"]["instructions_template"] == "Frozen session instructions"
        assert profile["input_modalities"] == ["text"]
