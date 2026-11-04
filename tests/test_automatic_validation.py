import importlib.util
from pathlib import Path

import httpx
import pytest

from velocicyber.models.gpu_profiles import MOG_SEC_REPO
from velocicyber.state import State

spec = importlib.util.spec_from_file_location(
    "automatic_validation", Path(__file__).resolve().parents[1] / "scripts/validation/validate_mog_when_ready.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


async def test_waiter_requires_exact_model_and_idle_user_task(settings, monkeypatch):
    State(instance_id="existing").persist(settings.runtime / "gpu.json")
    requests, statuses = [], []

    def serve(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": MOG_SEC_REPO if len(requests) > 1 else "other"}]})

    monkeypatch.setattr(validation, "user_task_running", lambda _: len(requests) < 3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        await validation.wait_for_model(settings, "existing", client, statuses.append, interval=0, timeout=2)
    assert statuses == ["waiting_for_model", "waiting_for_user_task"]
    assert len(requests) == 3 and all(r.method == "GET" for r in requests)


async def test_waiter_stops_when_original_instance_is_gone(settings):
    State(instance_id="replacement").persist(settings.runtime / "gpu.json")
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: pytest.fail("Must not contact a replacement instance")
    )) as client:
        with pytest.raises(RuntimeError, match="instance changed"):
            await validation.wait_for_model(settings, "original", client, lambda _: None, interval=0)
