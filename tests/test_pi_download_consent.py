"""A missing Pi model needs download consent without disturbing current work."""

import json
from unittest.mock import AsyncMock

import pytest
from textual.widgets import RichLog

from velocicyber.models.pi_profiles import BUGTRACE_PI_REPO, PI_MODEL_REPO, PI_PROFILES
from velocicyber.providers.pi import PiModelManager
from velocicyber.state import GPUStatus, State
from velocicyber.ui.app import VelocicyberApp

PROMPT = (
    "This model is not downloaded yet onto this local device. "
    "Would you like to download it now?"
)


def current_model(settings, selected):
    previous = BUGTRACE_PI_REPO if selected == PI_MODEL_REPO else PI_MODEL_REPO
    settings.options["PI_LOCAL_MODEL"] = previous
    settings.options["PI_LOCAL_MODEL_OPTIONS"] = ",".join(PI_PROFILES)
    return previous


@pytest.mark.parametrize("repo", list(PI_PROFILES))
@pytest.mark.parametrize("keys", [("enter",), ("escape",), (*"cancel", "enter")])
async def test_decline_or_escape_preserves_current_model_and_session(settings, monkeypatch, repo, keys):
    previous = current_model(settings, repo)
    app = VelocicyberApp(settings)
    start = AsyncMock()
    monkeypatch.setattr(app.pi, "start", start)
    async with app.run_test() as pilot:
        app.state.pi_ready = True
        app.state.active_provider = app.state.private_target = "pi"
        app.show_conversation()
        app.say("Keep this conversation")
        app.review_local_model(repo)
        await pilot.press("enter")
        assert app.screen.id == "pi-download-confirm"
        assert PROMPT in app.screen.heading and repo in app.screen.heading
        assert settings.pi_model == previous and app.pi.repo == previous
        start.assert_not_awaited()
        await pilot.press(*keys)
        await app.workers.wait_for_complete()
        start.assert_not_awaited()
        assert settings.pi_model == previous and app.pi.repo == previous
        assert app.state.pi_ready and app.state.active_provider == "pi"
        assert app.state.private_target == "pi" and app.conversation
        assert "Keep this conversation" in app.query_one("#log", RichLog)._copy_text
        assert app.state.gpu_status == GPUStatus.STOPPED and app.gpu._client is None


@pytest.mark.parametrize("repo", list(PI_PROFILES))
async def test_yes_starts_only_the_selected_download(settings, monkeypatch, repo):
    previous = current_model(settings, repo)
    app = VelocicyberApp(settings)
    start = AsyncMock()
    monkeypatch.setattr(app.pi, "start", start)
    async with app.run_test() as pilot:
        app.select_local_model(repo)
        await pilot.pause()
        assert app.screen.id == "pi-download-confirm"
        assert not app.conversation and settings.pi_model == previous
        start.assert_not_awaited()
        await pilot.press(*"yes", "enter")
        await app.workers.wait_for_complete()
        start.assert_awaited_once_with(app.say, allow_download=True)
        assert settings.pi_model == repo and app.conversation
        assert app.state.active_provider == "pi" and app.state.private_target == "pi"
        assert app.state.gpu_status == GPUStatus.STOPPED and app.gpu._client is None


@pytest.mark.parametrize("repo", list(PI_PROFILES))
async def test_installed_selection_starts_without_download_permission(settings, monkeypatch, repo):
    current_model(settings, repo)
    app = VelocicyberApp(settings)
    checked = []
    def installed(selected):
        checked.append(selected)
        return {"installed": True}
    monkeypatch.setattr(app.pi, "installed", installed)
    start = AsyncMock()
    monkeypatch.setattr(app.pi, "start", start)
    async with app.run_test():
        app.select_local_model(repo)
        await app.workers.wait_for_complete()
        assert checked == [repo] and app.screen.id != "pi-download-confirm"
        start.assert_awaited_once_with(app.say, allow_download=False)
        assert settings.pi_model == repo and app.conversation


@pytest.mark.parametrize("approved", [False, True])
async def test_provider_never_downloads_without_explicit_permission(settings, monkeypatch, approved):
    settings.options["PI_LOCAL_MODEL"] = PI_MODEL_REPO
    manager = PiModelManager(settings, State())
    monkeypatch.setattr(manager, "installed", lambda: None)
    install = AsyncMock(side_effect=RuntimeError("approved installer reached"))
    monkeypatch.setattr(manager, "install", install)
    error = "approved installer reached" if approved else "download needs confirmation"
    with pytest.raises(RuntimeError, match=error):
        if approved:
            await manager.start(allow_download=True)
        else:
            await manager.start()
    assert install.await_count == int(approved)
    assert manager.lock is None and manager.process is None


def test_inspection_checks_requested_model_without_switching(settings):
    settings.options["PI_LOCAL_MODEL"] = PI_MODEL_REPO
    manager = PiModelManager(settings, State())
    cfg = {"repo": PI_MODEL_REPO, "revision": "fixture", "weight_bytes": 4}
    (settings.root / PI_PROFILES[PI_MODEL_REPO].lock_file).write_text(json.dumps(cfg))
    model, binary = manager.folder / "model.gguf", manager.folder / "server"
    model.write_bytes(b"GGUF")
    binary.write_bytes(b"fixture")
    receipt = {**cfg, "model_path": str(model), "binary_path": str(binary),
               "model_mtime_ns": model.stat().st_mtime_ns}
    (manager.folder / "installed.json").write_text(json.dumps(receipt))
    assert manager.installed(PI_MODEL_REPO) == receipt
    assert manager.installed(BUGTRACE_PI_REPO) is None
    assert manager.repo == settings.pi_model == PI_MODEL_REPO
    assert not (settings.runtime / PI_PROFILES[BUGTRACE_PI_REPO].directory).exists()


async def test_approval_rechecks_busy_before_switching(settings, monkeypatch):
    previous = current_model(settings, BUGTRACE_PI_REPO)
    app = VelocicyberApp(settings)
    start = AsyncMock()
    monkeypatch.setattr(app.pi, "start", start)
    async with app.run_test() as pilot:
        app.select_local_model(BUGTRACE_PI_REPO)
        await pilot.pause()
        app.pi_busy = True
        await pilot.press(*"yes", "enter")
        start.assert_not_awaited()
        assert settings.pi_model == previous and not app.conversation
        app.pi_busy = False
