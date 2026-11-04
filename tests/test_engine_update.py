import json

import httpx
import pytest

from velocicyber.engine.installation import (
    ACTIVE_FILE,
    RECOVERY_FILE,
    UPDATE_NOTICE_FILE,
    engine_lease,
    take_update_notice,
)
from velocicyber.engine.releases import REPOSITORY
from velocicyber.engine.updater import ASSET, EngineUpdater, interactive_update, save_json, sha256


def make_engine(folder, version):
    (folder / "bin").mkdir(parents=True)
    binary = folder / "bin/interpreter"
    binary.write_text(f"#!/bin/sh\nprintf 'interpreter {version}\\n'\n")
    binary.chmod(0o755)
    (folder / "codex-package.json").write_text(json.dumps({
        "variant": "open-interpreter", "version": version,
    }))
    return binary


@pytest.fixture
def maintenance(settings):
    current = make_engine(settings.runtime / "engine", "0.0.45")
    (settings.root / "tests").mkdir()
    (settings.root / "tests/test_engine_integration.py").touch()
    save_json(settings.root / "config/locks/upstream.lock.json", {"repository": REPOSITORY, "tag": "rust-v0.0.45"})
    folder = settings.runtime / "engine-updates/candidate"
    binary = make_engine(folder, "0.0.55")
    candidate = {
        "schema": 1, "entrypoint": str(binary.relative_to(settings.runtime)),
        "release": {"tag": "rust-v0.0.55"},
    }
    receipt = {
        "passed": True, "entrypoint": candidate["entrypoint"],
        "binary_sha256": sha256(binary), "previous_entrypoint": str(current),
    }
    history = settings.runtime / "conversations/example"
    history.mkdir(parents=True)
    (history / "session.lock").touch()
    (history / "transcript.jsonl").write_text("old history\n")
    updater = EngineUpdater(settings, lambda _: None)
    return updater, candidate, receipt


def test_no_update_work_without_explicit_approval(maintenance, monkeypatch):
    updater, _, _ = maintenance
    monkeypatch.setattr(updater, "preflight", lambda: pytest.fail("Work started before approval"))
    with pytest.raises(PermissionError):
        updater.run()


@pytest.mark.parametrize("answer", ["", "no", "yes", "update"])
def test_prompt_cancellation_starts_nothing(settings, monkeypatch, answer):
    monkeypatch.setattr("builtins.input", lambda _: answer)
    monkeypatch.setattr(EngineUpdater, "run", lambda *a, **k: pytest.fail("Unapproved update"))
    interactive_update(settings)
    assert not (settings.runtime / "engine-updates").exists()


def test_confirmation_is_explicit_and_passed_to_work(settings, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "UPDATE")
    calls = []
    monkeypatch.setattr(EngineUpdater, "run", lambda self, **kw: calls.append(kw))
    interactive_update(settings)
    assert calls == [{"approved": True, "action": "update", "tag": None}]


def test_failed_compatibility_gate_preserves_engine_and_history(maintenance, monkeypatch):
    updater, candidate, _ = maintenance
    current = updater.settings.engine()
    monkeypatch.setattr(updater, "stage", lambda tag: candidate)

    def failure(candidate):
        raise RuntimeError("Compatibility tests failed")

    monkeypatch.setattr(updater, "validate", failure)
    with pytest.raises(RuntimeError, match="Compatibility tests failed"):
        updater.run(approved=True)
    assert updater.settings.engine() == current
    assert not (updater.runtime / ACTIVE_FILE).exists()
    assert (updater.runtime / "conversations/example/transcript.jsonl").read_text() == "old history\n"


def test_stage_only_never_activates(maintenance, monkeypatch):
    updater, candidate, receipt = maintenance
    monkeypatch.setattr(updater, "stage", lambda tag: candidate)
    monkeypatch.setattr(updater, "validate", lambda candidate: receipt)
    updater.run(approved=True, action="stage")
    assert not (updater.runtime / ACTIVE_FILE).exists()
    assert take_update_notice(updater.runtime) is None
    assert not list(updater.root.glob("backup-*"))


def test_activation_and_rollback_preserve_both_histories(maintenance, monkeypatch):
    updater, candidate, receipt = maintenance
    old = updater.settings.engine()
    monkeypatch.setattr(updater, "stage", lambda tag: candidate)
    monkeypatch.setattr(updater, "validate", lambda candidate: receipt)
    updater.run(approved=True)
    assert updater.settings.engine() == updater.runtime / candidate["entrypoint"]
    active = json.loads((updater.runtime / ACTIVE_FILE).read_text())
    backup = updater.runtime / active["backup"]
    assert (backup / "conversations/example/transcript.jsonl").read_text() == "old history\n"
    assert (backup.stat().st_mode & 0o777) == 0o700
    assert ((updater.runtime / ACTIVE_FILE).stat().st_mode & 0o777) == 0o600
    (updater.runtime / "conversations/new-session").mkdir()
    (updater.runtime / "conversations/new-session/transcript.jsonl").write_text("new history\n")
    updater.run(approved=True, action="rollback")
    assert take_update_notice(updater.runtime) is None
    assert updater.settings.engine() == old
    assert not (updater.runtime / "conversations/new-session").exists()
    archive = next(updater.root.glob("rollback-*/conversations-after-update"))
    assert (archive / "new-session/transcript.jsonl").read_text() == "new history\n"
    assert (updater.runtime / "conversations/example/transcript.jsonl").read_text() == "old history\n"


def test_activation_notification_is_consumed_once(maintenance):
    updater, candidate, receipt = maintenance
    updater.activate(candidate, receipt)
    assert take_update_notice(updater.runtime) == candidate["release"]["tag"]
    assert take_update_notice(updater.runtime) is None


def test_failed_activation_does_not_announce_update(maintenance, monkeypatch):
    updater, candidate, receipt = maintenance

    def fail_active_write(path, value):
        if path.name == ACTIVE_FILE:
            raise OSError("Fixture: cannot activate")
        save_json(path, value)

    monkeypatch.setattr("velocicyber.engine.updater.save_json", fail_active_write)
    with pytest.raises(OSError, match="cannot activate"):
        updater.activate(candidate, receipt)
    assert (updater.runtime / UPDATE_NOTICE_FILE).exists()
    assert take_update_notice(updater.runtime) is None


def test_modified_candidate_cannot_use_old_test_receipt(maintenance):
    updater, candidate, receipt = maintenance
    (updater.runtime / candidate["entrypoint"]).write_text("changed")
    with pytest.raises(RuntimeError, match="has not passed validation"):
        updater.activate(candidate, receipt)
    assert not (updater.runtime / ACTIVE_FILE).exists()


@pytest.mark.parametrize("problem", ["skipped", "missing", "malformed", "failed-exit"])
def test_incomplete_test_reports_never_pass_validation(maintenance, monkeypatch, problem):
    from pathlib import Path

    updater, candidate, _ = maintenance
    current = updater.settings.engine()

    class TestProcess:
        def __init__(self, command, **kwargs):
            report = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
            if problem != "missing":
                body = "<testcase/>" * 12
                if problem == "skipped":
                    body += "<testcase><skipped/></testcase>"
                report.write_text("<broken" if problem == "malformed" else f"<testsuites>{body}</testsuites>")

        def wait(self, timeout=None):
            return 1 if problem == "failed-exit" else 0

    monkeypatch.setattr("velocicyber.engine.updater.subprocess.Popen", TestProcess)
    with pytest.raises(RuntimeError, match="[Cc]ompatibility"):
        updater.validate(candidate)
    assert updater.settings.engine() == current
    assert not (updater.runtime / ACTIVE_FILE).exists()


def test_harness_and_maintenance_exclude_each_other(settings):
    with engine_lease(settings.runtime):
        with pytest.raises(RuntimeError, match="Close all"):
            with engine_lease(settings.runtime, maintenance=True):
                pytest.fail("Concurrent engine replacement")
    with engine_lease(settings.runtime, maintenance=True):
        with pytest.raises(RuntimeError, match="maintenance is running"):
            with engine_lease(settings.runtime):
                pytest.fail("Started harness mid-update")


def test_open_legacy_conversation_blocks_activation(maintenance):
    import fcntl

    updater, candidate, receipt = maintenance
    with (updater.runtime / "conversations/example/session.lock").open() as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="still open"):
            updater.activate(candidate, receipt)
    assert not (updater.runtime / ACTIVE_FILE).exists()


def test_interrupted_rollback_blocks_launch_and_can_resume(maintenance, monkeypatch):
    updater, candidate, receipt = maintenance
    old = updater.settings.engine()
    updater.activate(candidate, receipt)

    def interrupted(path, value):
        if path.name == ACTIVE_FILE:
            raise OSError("Simulated interrupted pointer switch")
        save_json(path, value)

    monkeypatch.setattr("velocicyber.engine.updater.save_json", interrupted)
    with pytest.raises(OSError):
        updater.rollback()
    assert (updater.runtime / RECOVERY_FILE).exists()
    with pytest.raises(RuntimeError, match="interrupted"):
        updater.settings.engine()
    monkeypatch.setattr("velocicyber.engine.updater.save_json", save_json)
    updater.rollback()
    assert not (updater.runtime / RECOVERY_FILE).exists()
    assert updater.settings.engine() == old


def test_release_requires_official_matching_asset_digest(settings):
    updater = EngineUpdater(settings)
    release = {
        "tag_name": "rust-v0.0.55", "draft": False, "prerelease": False,
        "assets": [{"name": ASSET, "size": 10}],
    }
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=release))) as client:
        with pytest.raises(RuntimeError, match="SHA256"):
            updater.release(client)
        release["assets"][0].update({
            "digest": "sha256:" + "a" * 64,
            "browser_download_url": "https://untrusted.invalid/engine",
        })
        with pytest.raises(RuntimeError, match="Unexpected upstream"):
            updater.release(client)


def test_bad_active_pointer_never_falls_back_to_old_engine(maintenance):
    updater, _, _ = maintenance
    save_json(updater.runtime / ACTIVE_FILE, {"schema": 1, "entrypoint": "../../outside"})
    with pytest.raises(ValueError, match="escapes"):
        updater.settings.engine()


async def test_update_menu_requires_confirmation_and_demo_cannot_mutate(settings, monkeypatch):
    from textual.widgets import OptionList

    from velocicyber.ui.app import Menu, VelocicyberApp

    app = VelocicyberApp(settings)
    exits = []
    monkeypatch.setattr(app, "action_quit", lambda: exits.append(True))
    async with app.run_test() as pilot:
        app.command("update-engine")
        await pilot.pause()
        assert isinstance(app.screen, Menu)
        assert app.screen.query_one(OptionList).highlighted_option.id == "choice-0"
        await pilot.press("enter")
        assert not exits and not app.maintenance_action
        app.confirm_engine_maintenance("update")
        await pilot.pause()
        assert app.screen.query_one(OptionList).highlighted_option.id == "choice-0"
        await pilot.press("enter")
        assert not exits and not app.maintenance_action
        app.confirm_engine_maintenance("update")
        await pilot.pause()
        await pilot.press("down", "enter")
        assert exits == [True] and app.maintenance_action == "update"
        app.maintenance_action = ""
        app.demo = True
        app.command("update-engine")
        assert not app.maintenance_action and exits == [True]
