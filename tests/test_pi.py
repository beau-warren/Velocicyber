import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest
from textual.widgets import Input, OptionList, Static

from velocicyber.config import PI_MODEL_NOTICE, PI_MODEL_REPO, Settings
from velocicyber.engine.runner import Engine
from velocicyber.models.pi_profiles import BUGTRACE_PI_REPO, PI_PROFILES
from velocicyber.providers.lambda_cloud import GPUManager
from velocicyber.providers.pi import PiModelManager
from velocicyber.state import GPUStatus, State
from velocicyber.ui.app import Menu, VelocicyberApp


def test_configured_cpu_model_is_separate_from_gpu_choices():
    settings = Settings.load(Path(__file__).resolve().parents[1])
    assert PI_MODEL_REPO in settings.model_choices
    assert PI_MODEL_REPO not in settings.gpu_model_choices
    assert BUGTRACE_PI_REPO in settings.pi_model_choices
    assert BUGTRACE_PI_REPO not in settings.gpu_model_choices
    assert not any(repo == "example-org/removed-model" for repo in settings.model_choices)


@pytest.mark.parametrize("repo", [PI_MODEL_REPO, BUGTRACE_PI_REPO])
def test_cpu_profile_never_enters_gpu_deployment(settings, monkeypatch, repo):
    settings.options["PI_LOCAL_MODEL"] = repo
    window = PI_PROFILES[repo].context_window
    state = State(active_provider="pi", pi_ready=True, pi_context_window=window)
    gpu = GPUManager(settings, state)
    with pytest.raises(ValueError, match="local CPU"):
        gpu.inspect_model(repo)
    assert gpu._client is None
    engine = Engine(settings, state)
    assert engine.context_budget().window == window
    assert engine.response_timeout() >= 1800
    assert engine.selected_model == repo
    command = PiModelManager(settings, state).command(
        {
            "model_path": "/example/model.gguf",
            "binary_path": "/example/llama-server",
            "context_window": window,
        }
    )
    for flag, value in [
        ("-ngl", "0"),
        ("-np", "1"),
        ("--host", "127.0.0.1"),
        ("-c", str(window)),
        ("--alias", repo),
    ]:
        assert command[command.index(flag) + 1] == value
    assert "--jinja" in command and "--chat-template-kwargs" in command
    assert state.gpu_status == GPUStatus.STOPPED and not state.model_ready


def test_install_receipt_requires_matching_profile_size_and_timestamp(settings):
    manager = PiModelManager(settings, State())
    cfg = {"repo": PI_MODEL_REPO, "revision": "pinned", "weight_bytes": 4}
    (settings.root / "config/locks/pi-model.lock.json").write_text(json.dumps(cfg))
    model, binary = manager.folder / "model.gguf", manager.folder / "llama-server"
    model.write_bytes(b"GGUF")
    binary.write_text("fixture")
    receipt = {
        **cfg,
        "model_path": str(model),
        "binary_path": str(binary),
        "model_mtime_ns": model.stat().st_mtime_ns,
    }
    (manager.folder / "installed.json").write_text(json.dumps(receipt))
    assert manager.installed() == receipt
    cfg["server_args"] = ["--repeat-penalty", "1.0"]
    cfg["chat_template"] = {"path": "chat-templates/gemma4.jinja", "sha256": "fixture"}
    (settings.root / "config/locks/pi-model.lock.json").write_text(json.dumps(cfg))
    assert manager.installed() == {**receipt, **cfg}
    assert json.loads((manager.folder / "installed.json").read_text()) == receipt
    model.write_bytes(b"changed")
    assert manager.installed() is None


def test_pi_template_is_pinned_and_confined_to_project(settings):
    import hashlib

    settings.options["PI_LOCAL_MODEL"] = BUGTRACE_PI_REPO
    template = settings.root / "template.jinja"
    template.write_text("fixture template")
    cfg = {
        "model_path": "/example/model.gguf",
        "binary_path": "/example/llama-server",
        "context_window": 8192,
        "chat_template": {
            "path": "template.jinja",
            "sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        },
    }
    manager = PiModelManager(settings, State())
    command = manager.command(cfg)
    assert command[command.index("--chat-template-file") + 1] == str(template)
    template.write_text("modified")
    with pytest.raises(ValueError, match="checksum"):
        manager.command(cfg)
    cfg["chat_template"]["path"] = "../outside.jinja"
    with pytest.raises(ValueError, match="inside this project"):
        manager.command(cfg)


def test_bugtrace_pinned_serving_settings():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config/locks/pi-bugtrace.lock.json").read_text())
    template = root / cfg["chat_template"]["path"]
    assert hashlib.sha256(template.read_bytes()).hexdigest() == cfg["chat_template"]["sha256"]
    assert "OpenAI Chat Completions:" in template.read_text()
    assert cfg["server_args"][cfg["server_args"].index("--repeat-penalty") + 1] == "1.0"


async def test_select_pi_and_switch_api_without_gpu_side_effects(settings):
    settings.options["PI_LOCAL_MODEL"] = PI_MODEL_REPO
    settings.options["PI_LOCAL_MODEL_OPTIONS"] = f"{PI_MODEL_REPO},{BUGTRACE_PI_REPO}"
    gpu_model = settings.local_model
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(100, 30)) as pilot:
        app.menu_selected("local")
        await pilot.pause()
        assert isinstance(app.screen, Menu)
        app.screen.query_one(Input).value = "RPi5"
        await pilot.pause()
        option = app.screen.query_one(OptionList).highlighted_option
        assert str(option.prompt) == PI_MODEL_REPO
        await pilot.press("enter")
        assert app.screen.id == "local-model-choice" and not app.state.pi_ready
        assert not app.conversation
        assert PI_MODEL_NOTICE in app.screen.heading
        await pilot.press(*"more", "enter")
        assert PI_MODEL_NOTICE in app.screen.details
        assert "13 GiB available RAM" in app.screen.details
        assert not app.state.pi_ready and app.gpu._client is None
        assert not app.conversation
        await pilot.press("enter")  # Back.
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert app.conversation and not app.query_one("#main-menu").display
        assert app.focused is app.query_one("#prompt")
        assert app.engine.session is None and not app.engine.running
        assert app.state.active_provider == "pi" and app.state.pi_ready
        assert app.state.gpu_status == GPUStatus.STOPPED and not app.state.model_ready
        assert not app.state.instance_id and app.gpu._client is None
        assert settings.local_model == gpu_model
        dashboard = app.query_one("#dashboard", Static).render()
        assert "GPUs: OFF" in str(dashboard) and "GPU inference: unavailable" in str(dashboard)
        assert "Pi CPU: Ready" in str(dashboard)
        assert any(
            PI_MODEL_REPO in text and style.foreground.hex == "#39FF14"
            for text, style in dashboard.render()
        )
        assert "Pi CPU" in app.query_one("#log").border_title
        app.confirm_termination()  # GPU OFF must not deselect the Pi.
        assert app.state.active_provider == "pi" and app.state.pi_ready
        app.select_api_model("cyberkimi")
        assert app.state.active_provider == "api" and app.state.pi_ready
        app.select_local_model(PI_MODEL_REPO)
        await app.workers.wait_for_complete()
        assert app.state.active_provider == "pi"
        app.menu_selected("local")
        await pilot.pause()
        app.screen.query_one(Input).value = "terrorswift"
        await pilot.pause()
        await pilot.press("enter")
        assert app.screen.id == "local-model-choice"
        assert PI_MODEL_NOTICE in app.screen.heading
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert settings.pi_model == BUGTRACE_PI_REPO
        assert PI_MODEL_REPO in settings.pi_model_choices
        assert app.engine.selected_model == BUGTRACE_PI_REPO
        assert app.engine.context_budget().window == 8192
        assert app.state.gpu_status == GPUStatus.STOPPED and app.gpu._client is None
        dashboard = app.query_one("#dashboard", Static).render()
        assert any(
            BUGTRACE_PI_REPO in text and style.foreground.hex == "#39FF14"
            for text, style in dashboard.render()
        )
        app.select_local_model("pi-stop")
        await app.workers.wait_for_complete()
        assert not app.state.pi_ready and app.state.gpu_status == GPUStatus.STOPPED


async def test_failed_pi_start_does_not_fall_back_to_api(settings, monkeypatch):
    settings.options["PI_LOCAL_MODEL"] = PI_MODEL_REPO
    app = VelocicyberApp(settings, demo=False)
    notifications = []
    monkeypatch.setattr(app, "notify", lambda message, **kw: notifications.append(message))

    monkeypatch.setattr(app.pi, "installed", lambda repo=None: {"installed": True})

    async def fail(notify, *, allow_download=False):
        raise RuntimeError("Fixture: insufficient RAM")

    monkeypatch.setattr(app.pi, "start", fail)
    async with app.run_test():
        app.select_local_model(PI_MODEL_REPO)
        await app.workers.wait_for_complete()
        assert app.conversation and not notifications
        assert "Fixture: insufficient RAM" in app.query_one("#log")._copy_text
        assert app.state.active_provider == "pi" and not app.state.pi_ready
        with pytest.raises(RuntimeError, match="not ready"):
            await app.engine.run("Do not send this to an API")
        assert app.gpu._client is None


async def test_pi_loading_progress_is_inline_in_automatic_new_session(settings, monkeypatch):
    settings.options["PI_LOCAL_MODEL"] = PI_MODEL_REPO
    app = VelocicyberApp(settings)
    started, finish = asyncio.Event(), asyncio.Event()
    notifications = []
    monkeypatch.setattr(app, "notify", lambda message, **kw: notifications.append(message))
    monkeypatch.setattr(app.pi, "install_status", lambda: "Downloaded; ready to load")

    monkeypatch.setattr(app.pi, "installed", lambda repo=None: {"installed": True})

    async def start(notify, *, allow_download=False):
        notify("Loading the Pi model into memory…")
        started.set()
        await finish.wait()
        app.state.pi_ready = True
        app.state.pi_stage = "Ready"
        notify("Pi model ready.")

    monkeypatch.setattr(app.pi, "start", start)
    async with app.run_test() as pilot:
        assert not notifications and not app.conversation
        assert "installation complete" in app.query_one("#home-notice").render().plain
        app.select_local_model(PI_MODEL_REPO)
        await asyncio.wait_for(started.wait(), timeout=5)
        await pilot.pause()
        assert app.conversation and app.pi_busy and not app.state.pi_ready
        assert "Loading the Pi model" in app.query_one("#log")._copy_text
        assert app.engine.session is None and not app.engine.running
        finish.set()
        await app.workers.wait_for_complete()
        assert app.state.pi_ready and not notifications
        assert "Pi model ready." in app.query_one("#log")._copy_text
        assert app.state.gpu_status == GPUStatus.STOPPED and app.gpu._client is None


async def test_pi_switch_unloads_before_spawn_and_lock_blocks_other_harness(settings, monkeypatch):
    settings.options["PI_LOCAL_MODEL"] = PI_MODEL_REPO
    state = State(active_provider="pi")
    manager = PiModelManager(settings, state)
    processes, commands = [], []

    class Process:
        returncode = None

        def __init__(self):
            self.done = asyncio.Event()

        async def wait(self):
            await self.done.wait()
            return self.returncode

        def terminate(self):
            self.returncode = 0
            self.done.set()

    async def spawn(*args, **kwargs):
        assert all(p.returncode == 0 for p in processes), "Two Pi models would overlap"
        commands.append(args)
        process = Process()
        processes.append(process)
        return process

    def installed():
        return {
            "repo": manager.repo,
            "revision": "fixture",
            "binary_path": "/fixture/server",
            "model_path": "/fixture/weights.gguf",
            "context_window": PI_PROFILES[manager.repo].context_window,
        }

    def response(request):
        return httpx.Response(200, json={"data": [{"id": manager.repo}]})

    client = httpx.AsyncClient
    monkeypatch.setattr("velocicyber.providers.pi.available_memory", lambda: 14 * 1024**3)
    monkeypatch.setattr("velocicyber.providers.pi.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr(manager, "installed", installed)
    monkeypatch.setattr(
        "velocicyber.providers.pi.httpx.AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(response), **kwargs),
    )
    try:
        await manager.start()
        other_settings = copy.deepcopy(settings)
        other_settings.options["PI_LOCAL_MODEL"] = BUGTRACE_PI_REPO
        other = PiModelManager(other_settings, State())
        with pytest.raises(RuntimeError, match="another harness"):
            await other.start()
        assert processes[0].returncode is None
        for repo in (BUGTRACE_PI_REPO, PI_MODEL_REPO):
            settings.options["PI_LOCAL_MODEL"] = repo
            await manager.start()
            assert state.pi_ready and manager.repo == repo
            assert state.pi_context_window == PI_PROFILES[repo].context_window
            assert commands[-1][commands[-1].index("--alias") + 1] == repo
        assert len(processes) == 3
        assert state.gpu_status == GPUStatus.STOPPED
    finally:
        await manager.stop()
    assert all(p.returncode == 0 for p in processes)
