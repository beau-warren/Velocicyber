import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import OptionList

from velocicyber.models.pi_profiles import PI_MODEL_REPO
from velocicyber.state import GPUStatus
from velocicyber.ui.app import Menu, VelocicyberApp


@pytest.fixture
def swap_case(settings, monkeypatch):
    settings.options["PI_LOCAL_MODEL"] = PI_MODEL_REPO
    folder = settings.root / "SCOPE"
    folder.mkdir()
    (folder / "SCOPE.MD").write_text("Only inspect fixture files inside the workspace.")
    app = VelocicyberApp(settings, demo=True)
    app.state.active_provider = app.state.private_target = "pi"
    app.state.pi_ready = True
    app.pi.process = SimpleNamespace(returncode=None)
    events = []

    async def stop_pi():
        events.append("pi stopped")
        app.state.pi_ready = False
        app.pi.process = None

    async def start_sentinel(notify):
        assert not app.state.pi_ready and app.pi.process is None
        assert app.state.active_provider != "pi"
        events.append("sentinel started: " + app.state.sentinel_model)
        app.state.sentinel_enabled = app.state.sentinel_ready = True
        app.state.sentinel_stage = "READY"

    async def stop_sentinel():
        app.state.sentinel_enabled = app.state.sentinel_ready = False
        app.state.sentinel_stage = "OFF"

    monkeypatch.setattr(app.pi, "stop", stop_pi)
    monkeypatch.setattr(app.sentinel, "start", start_sentinel)
    monkeypatch.setattr(app.sentinel, "stop", stop_sentinel)
    return app, events


async def choose(pilot, index):
    menu = pilot.app.screen.query_one(OptionList)
    menu.highlighted = menu.get_option_index(f"choice-{index}")
    menu.action_select()
    await pilot.pause()


async def test_on_menu_offers_swap_and_cancel_keeps_running_pi(swap_case):
    app, events = swap_case
    async with app.run_test() as pilot:
        app.demo = False
        app.command("sentinel")
        await pilot.pause()
        await choose(pilot, 0)
        assert app.screen.id == "sentinel-swap"
        assert PI_MODEL_REPO in app.screen.details and "(API)" in app.screen.details
        assert not events and app.state.pi_ready
        await choose(pilot, 0)
        assert not events and app.state.pi_ready and app.state.active_provider == "pi"
        assert not app.state.sentinel_enabled and not app.sentinel_busy


@pytest.mark.parametrize("model", ["qwen", "granite"])
@pytest.mark.parametrize("active,ready_gpu,expected", [
    ("pi", False, "api"), ("pi", True, "local"),
    ("api", True, "api"), ("local", True, "local"),
])
async def test_confirmed_swap_stops_before_start_and_pauses_queue(swap_case, model, active, ready_gpu, expected):
    app, events = swap_case
    async with app.run_test() as pilot:
        app.demo = False
        app.state.active_provider = active
        app.state.gpu_status = GPUStatus.RUNNING if ready_gpu else GPUStatus.STOPPED
        app.state.model_ready = ready_gpu
        app.queued_prompts.append("Previously queued Pi task")
        app.select_sentinel_model(model)
        await pilot.pause()
        assert app.screen.id == "sentinel-swap" and not events
        assert app.chat_model_name(expected) in app.screen.details
        await choose(pilot, 1)
        await app.workers.wait_for_complete()
        assert events == ["pi stopped", "sentinel started: " + model]
        assert app.state.sentinel_ready and app.state.sentinel_model == model
        assert app.state.active_provider == expected and not app.state.pi_ready
        assert app.state.model_ready == ready_gpu
        if expected == "local":
            assert app.state.private_target == "gpu"
        assert app.queue_paused and list(app.queued_prompts) == ["Previously queued Pi task"]
        assert not app.pi_busy and not app.sentinel_busy and app.gpu._client is None


async def test_swap_reserves_operation_and_waits_for_pi_exit(swap_case, monkeypatch):
    app, events = swap_case
    entered, released = asyncio.Event(), asyncio.Event()
    stop = app.pi.stop

    async def slow_stop():
        entered.set()
        await released.wait()
        await stop()

    monkeypatch.setattr(app.pi, "stop", slow_stop)
    monkeypatch.setattr(app.engine, "run", lambda *a: pytest.fail("Unexpected task during handoff"))
    async with app.run_test() as pilot:
        app.demo = False
        app.set_sentinel(True)
        await pilot.pause()
        await choose(pilot, 1)
        await asyncio.wait_for(entered.wait(), 2)
        try:
            assert app.pi_busy and app.sentinel_busy and not events
            assert app.state.sentinel_stage == "UNLOADING PI MODEL"
            old_api = app.settings.api_model
            app.select_api_model("cyberglm")
            app.select_local_model(PI_MODEL_REPO)
            app.select_sentinel_model("granite")
            await app.run_prompt("Do not run while swapping").wait()
            assert app.settings.api_model == old_api and app.state.sentinel_model == "qwen"
            assert not events
        finally:
            released.set()
        await app.workers.wait_for_complete()
        assert events == ["pi stopped", "sentinel started: qwen"]


async def test_missing_scope_does_not_unload_or_change_models(swap_case):
    app, events = swap_case
    async with app.run_test() as pilot:
        app.demo = False
        app.select_sentinel_model("granite")
        await pilot.pause()
        (app.settings.root / "SCOPE/SCOPE.MD").unlink()
        await choose(pilot, 1)
        await app.workers.wait_for_complete()
        assert not events and app.state.pi_ready and app.state.active_provider == "pi"
        assert app.state.sentinel_model == "qwen" and not app.state.sentinel_enabled
        assert not app.pi_busy and not app.sentinel_busy


async def test_changed_provider_needs_a_new_confirmation(swap_case):
    app, events = swap_case
    async with app.run_test() as pilot:
        app.demo = False
        app.set_sentinel(True)
        await pilot.pause()
        assert "CyberKimi" in app.screen.details
        app.settings.api_model = "lordx64/cyberglm"
        await choose(pilot, 1)
        # The replacement confirmation is deferred until the next refresh.
        # Waiting only for idle messages can return before that refresh occurs.
        async with asyncio.timeout(5):
            while app.screen.id != "sentinel-swap" or not app.screen.is_mounted:
                await pilot.pause()
        await pilot.pause()
        assert app.screen.id == "sentinel-swap" and "CyberGLM" in app.screen.details
        assert not events and app.state.pi_ready
        await choose(pilot, 0)
        assert not events and app.state.active_provider == "pi"


@pytest.mark.parametrize("failed_step", ["stop", "start"])
async def test_failed_handoff_does_not_run_both_models_or_unmonitored_tasks(swap_case, monkeypatch, failed_step):
    app, events = swap_case
    original_stop = app.pi.stop

    async def fail(*args):
        raise RuntimeError("fixture " + failed_step + " failure")

    monkeypatch.setattr(app.pi if failed_step == "stop" else app.sentinel,
                        "stop" if failed_step == "stop" else "start", fail)
    async with app.run_test() as pilot:
        app.demo = False
        app.set_sentinel(True)
        await pilot.pause()
        await choose(pilot, 1)
        await app.workers.wait_for_complete()
        assert not app.pi_busy and not app.sentinel_busy
        if failed_step == "stop":
            assert not events and app.state.pi_ready and app.state.active_provider == "pi"
            assert app.state.sentinel_stage == "OFF"
            monkeypatch.setattr(app.pi, "stop", original_stop)
        else:
            assert events == ["pi stopped"]
            assert not app.state.pi_ready and not app.state.sentinel_ready
            assert app.state.sentinel_enabled and app.state.sentinel_stage == "ERROR"
            with pytest.raises(RuntimeError, match="Sentinel is unavailable"):
                await app.engine.run("Must not run without Sentinel", lambda event: None)
            assert app.engine.process is None
            app.set_sentinel(False)
            await app.workers.wait_for_complete()
            assert not app.state.sentinel_enabled


@pytest.mark.parametrize("busy", ["busy", "pi_busy", "gpu_busy", "sentinel_busy", "data_busy", "restoring_gpu", "session_ending"])
async def test_swap_refuses_active_work(swap_case, busy):
    app, events = swap_case
    async with app.run_test() as pilot:
        app.demo = False
        setattr(app, busy, True)
        app.set_sentinel(True)
        app.select_sentinel_model("granite")
        await pilot.pause()
        assert not isinstance(app.screen, Menu) and not events and app.state.pi_ready
        setattr(app, busy, False)


async def test_demo_swap_never_stops_or_loads_a_model(swap_case):
    app, events = swap_case
    async with app.run_test() as pilot:
        app.select_sentinel_model("granite")
        await pilot.pause()
        await choose(pilot, 1)
        await app.workers.wait_for_complete()
        assert not events and app.state.pi_ready and app.state.active_provider == "pi"
        assert not app.state.sentinel_ready and not app.pi_busy and not app.sentinel_busy
