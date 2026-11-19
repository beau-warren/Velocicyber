"""Offline checks for the narrow SDK compatibility adapter and its fail-closed policy."""

import json
from types import SimpleNamespace

import pytest

from scripts.runtime.api_gateway import POLICY_BINDINGS, policy_dispatcher
from velocicyber.providers.api import ConfidentialGateway


@pytest.fixture
def policy():
    return {
        "status": "active", "model_id": "adverserial/cyberseek",
        "receipt_keys": {"fixture": {"kty": "EC"}},
        **{key: "sha256:fixture-" + key for _, key in POLICY_BINDINGS},
    }


@pytest.fixture
def sdk(policy):
    core = SimpleNamespace(CANONICAL_MODELS=frozenset({"lordx64/cyberglm", "lordx64/cyberkimi"}))

    class VerificationFailed(Exception):
        pass

    class Dispatcher:
        def __init__(self):
            self.calls = []
            self.error = None
            self.proof = SimpleNamespace(
                model_id=policy["model_id"], dev_mode=False,
                **{field: policy[key] for field, key in POLICY_BINDINGS},
            )

        def proof_for(self, model):
            assert model in core.CANONICAL_MODELS
            self.calls.append(model)
            if self.error:
                raise self.error
            return self.proof

    core.Dispatcher = Dispatcher
    core.VerificationFailed = VerificationFailed
    return core


def test_cyberseek_adapter_preserves_sdk_verification_and_exact_model(policy, sdk):
    dispatcher = policy_dispatcher(sdk, policy)()
    assert dispatcher.proof_for("adverserial/cyberseek") is dispatcher.proof
    assert dispatcher.calls == ["adverserial/cyberseek"]
    assert sdk.CANONICAL_MODELS == {"lordx64/cyberglm", "lordx64/cyberkimi", "adverserial/cyberseek"}
    for model in ("cyberseek", "lordx64/cyberseek", "lordx64/cyberglm", "unknown/model"):
        with pytest.raises(sdk.VerificationFailed, match="No active"):
            dispatcher.proof_for(model)
    assert dispatcher.calls == ["adverserial/cyberseek"]
    dispatcher.error = sdk.VerificationFailed("hardware or receipt verification failed")
    with pytest.raises(sdk.VerificationFailed, match="hardware or receipt"):
        dispatcher.proof_for("adverserial/cyberseek")


@pytest.mark.parametrize("field", [field for field, _ in POLICY_BINDINGS] + ["model_id", "dev_mode"])
def test_wrong_workload_or_dev_proof_is_rejected(policy, sdk, field):
    dispatcher = policy_dispatcher(sdk, policy)()
    setattr(dispatcher.proof, field, True if field == "dev_mode" else "wrong")
    with pytest.raises(sdk.VerificationFailed):
        dispatcher.proof_for("adverserial/cyberseek")


@pytest.mark.parametrize("key,value", [
    ("status", "inactive"), ("receipt_keys", {}), ("model_id", "unknown/model"),
    *[(key, "") for _, key in POLICY_BINDINGS],
])
def test_invalid_policy_does_not_extend_sdk_models(policy, sdk, key, value):
    policy[key] = value
    original = sdk.CANONICAL_MODELS
    with pytest.raises(ValueError, match="complete active"):
        policy_dispatcher(sdk, policy)
    assert sdk.CANONICAL_MODELS == original


def test_old_client_requires_explicit_refresh_before_launch(settings, monkeypatch):
    folder = settings.runtime / "adverserial-confidential"
    (folder / "venv/bin").mkdir(parents=True)
    (folder / "venv/bin/python").touch()
    (folder / "source.json").write_text(json.dumps({"commit": "old"}))
    (settings.root / "config/locks/api-client.lock.json").write_text(json.dumps({"commit": "new"}))
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: pytest.fail("Old client was launched"))
    with pytest.raises(RuntimeError, match="/setup"):
        ConfidentialGateway(settings).start()
