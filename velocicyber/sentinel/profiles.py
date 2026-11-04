"""Sentinel choices are independent from the main coding model."""

from dataclasses import dataclass

from velocicyber.models.pi_profiles import PI_MODEL_REPO


@dataclass(frozen=True)
class SentinelProfile:
    model: str
    label: str
    local: bool
    lock_file: str
    receipt: str
    protocol: str
    review_timeout: int = 120


SENTINEL_MODELS = {
    "qwen": SentinelProfile(PI_MODEL_REPO, "Qwen3.5-35B-A3B IQK", True,
                            "config/locks/pi-model.lock.json", "pi-model/installed.json",
                            "qwen-scope-actions-v1", 195),
    "granite": SentinelProfile("ibm-granite/granite-guardian-4.1-8b",
                               "Granite Guardian 4.1 8B Q4", True,
                               "config/locks/sentinel-model.lock.json", "sentinel-observer/installed.json",
                               "granite-guardian-4.1-actions-v3"),
}


def sentinel_profile(state):
    try:
        return SENTINEL_MODELS[state.sentinel_model]
    except KeyError:
        raise ValueError("Unknown Sentinel model selection.") from None


def local_sentinel_active(state):
    return (state.sentinel_enabled or state.sentinel_ready) and sentinel_profile(state).local
