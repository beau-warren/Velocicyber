import json
import sys
import time
from types import SimpleNamespace

import httpx
import pytest

from velocicyber import cli
from velocicyber.config import Settings
from velocicyber.updates import startup
from velocicyber.updates.source import REPOSITORY, SourceChecker
from velocicyber.updates.updater import PENDING

OLD, NEW = "a" * 40, "b" * 40


@pytest.fixture
def launch(settings, monkeypatch):
    settings.options["AUTO_UPDATE_HARNESS"] = "ON"
    receipt = settings.runtime / "source-install.json"
    receipt.write_text(json.dumps({"repository": REPOSITORY, "commit": OLD}))
    calls = []

    async def check(self):
        if self.checked_this_launch:
            return
        calls.append("check")
        self.checked_this_launch = True
        self.result = {"schema": 1, "installed": OLD, "latest": NEW,
                       "status": "behind", "error": "", "checked_at": time.time()}

    def update(self, plan, *, approved):
        assert approved and plan == {"installed": OLD, "target": NEW}
        calls.append("update")
        receipt.write_text(json.dumps({"repository": REPOSITORY, "commit": NEW}))

    monkeypatch.delenv(startup.HANDOFF, raising=False)
    monkeypatch.setattr(SourceChecker, "check", check)
    monkeypatch.setattr(startup.SourceUpdater, "run", update)
    return calls


@pytest.mark.parametrize("value", [None, "OFF", "invalid"])
def test_automatic_updates_require_explicit_opt_in(settings, launch, value):
    settings.options.pop("AUTO_UPDATE_HARNESS")
    if value is not None:
        settings.options["AUTO_UPDATE_HARNESS"] = value
    assert startup.startup_updates(settings) is None
    assert launch == []


def test_preference_preserves_private_options_and_is_owner_only(settings):
    path = settings.root / "Modules.txt"
    path.write_text("LAMBDA_SSH_KEY_NAME=keep-me\n")
    startup.set_automatic_updates(settings, True)
    assert Settings.load(settings.root).options["AUTO_UPDATE_HARNESS"] == "ON"
    assert "LAMBDA_SSH_KEY_NAME=keep-me" in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    startup.set_automatic_updates(settings, False)
    assert not startup.automatic_updates(settings)


def test_success_restarts_with_original_flags_and_handoff_prevents_repeat(settings, launch, monkeypatch):
    captured = {}

    def execute(executable, arguments, environment):
        captured.update(executable=executable, arguments=arguments, environment=environment)
        raise SystemExit(0)

    monkeypatch.setattr(startup.os, "execve", execute)
    monkeypatch.setattr(sys, "argv", ["velocicyber", "--root", str(settings.root), "--sandbox", "open"])
    with pytest.raises(SystemExit):
        startup.startup_updates(settings)
    assert launch == ["check", "update"]
    assert captured["arguments"][1:] == ["-m", "velocicyber", *sys.argv[1:]]
    monkeypatch.setenv(startup.HANDOFF, captured["environment"][startup.HANDOFF])
    checker = startup.startup_updates(settings)
    assert checker.checked_this_launch and checker.installed == NEW
    assert checker.indicator()[0] == "UP TO DATE"
    assert startup.HANDOFF not in startup.os.environ
    assert launch == ["check", "update"]
    assert "successfully" in checker.startup_notice


@pytest.mark.parametrize("change", ["root", "commit", "time", "error", "malformed"])
def test_restart_marker_must_match_root_revision_and_fresh_cache(settings, launch, monkeypatch, change):
    checked_at = time.time()
    cached = {"schema": 1, "installed": OLD, "latest": OLD, "status": "identical",
              "error": "", "checked_at": checked_at}
    marker = {"root": str(settings.root), "commit": OLD, "checked_at": checked_at}
    if change == "root":
        marker["root"] += "/different"
    elif change == "commit":
        marker["commit"] = NEW
    elif change == "time":
        marker["checked_at"] = cached["checked_at"] = time.time() - 301
    elif change == "error":
        cached["error"] = "offline"
    (settings.runtime / "harness-updates.json").write_text(json.dumps(cached))
    monkeypatch.setenv(startup.HANDOFF, "[]" if change == "malformed" else json.dumps(marker))
    checker = SourceChecker(settings)
    assert not startup.take_handoff(checker)
    assert not checker.checked_this_launch and startup.HANDOFF not in startup.os.environ


@pytest.mark.parametrize("status,error", [("identical", ""), ("ahead", ""), ("diverged", ""),
                                         ("behind", "network check failed")])
def test_never_installs_stale_unverified_or_non_forward_update(settings, launch, monkeypatch, status, error):
    async def check(self):
        self.checked_this_launch = True
        self.result = {"latest": NEW, "status": status, "error": error}
    monkeypatch.setattr(SourceChecker, "check", check)
    checker = startup.startup_updates(settings)
    assert checker.checked_this_launch and launch == []


@pytest.mark.parametrize("failure", ["candidate tests failed", "active session", "local modifications"])
def test_failed_update_keeps_working_install_and_reports_deferral(settings, launch, monkeypatch, failure):
    def reject(*args, **kwargs):
        raise RuntimeError(failure)
    monkeypatch.setattr(startup.SourceUpdater, "run", reject)
    checker = startup.startup_updates(settings)
    assert checker.installed == OLD and failure in checker.startup_notice
    assert launch == ["check"]


def test_interrupted_activation_blocks_ui_and_asks_for_recovery(settings, launch, monkeypatch):
    def reject(*args, **kwargs):
        (settings.runtime / PENDING).write_text("{}")
        raise RuntimeError("rollback failed")
    monkeypatch.setattr(startup.SourceUpdater, "run", reject)
    with pytest.raises(RuntimeError, match="needs recovery"):
        startup.startup_updates(settings)
    with pytest.raises(RuntimeError, match="Interrupted"):
        startup.startup_updates(settings)
    assert launch == ["check"]


def test_cli_passes_completed_check_to_ui_before_starting(settings, monkeypatch):
    import velocicyber.onboarding as onboarding
    import velocicyber.ui.app as ui
    checker = SimpleNamespace(startup_notice="update deferred")
    calls = []

    class App:
        def __init__(self, *args, **kwargs):
            self.updates = SimpleNamespace()
            self.maintenance_action = ""

        def run(self):
            assert self.updates.source is checker
            assert self.startup_update_message == "update deferred"
            calls.append("UI")

    def update(*args):
        calls.append("update")
        return checker

    monkeypatch.setattr(cli.Settings, "load", lambda root: settings)
    monkeypatch.setattr(startup, "startup_updates", update)
    monkeypatch.setattr(onboarding, "run_setup", lambda *args, **kw: calls.append("setup"))
    monkeypatch.setattr(ui, "VelocicyberApp", App)
    monkeypatch.setattr(sys, "argv", ["velocicyber"])
    cli.main()
    assert calls == ["update", "setup", "UI"]


@pytest.mark.parametrize("arguments", [["doctor"], ["setup"], ["--demo"]])
def test_cli_maintenance_and_demo_never_auto_update(settings, monkeypatch, arguments):
    import velocicyber.onboarding as onboarding
    import velocicyber.ui.app as ui
    monkeypatch.setattr(cli.Settings, "load", lambda root: settings)
    monkeypatch.setattr(startup, "startup_updates", lambda *a: pytest.fail("Unexpected source update"))
    monkeypatch.setattr(onboarding, "run_setup", lambda *a, **kw: None)
    monkeypatch.setattr(ui.VelocicyberApp, "run", lambda self: None)
    monkeypatch.setattr(sys, "argv", ["velocicyber", *arguments])
    cli.main()


def test_download_network_failure_preserves_old_version_and_hides_url(settings, launch, monkeypatch):
    def reject(*args, **kwargs):
        raise httpx.ReadTimeout("private download URL")
    monkeypatch.setattr(startup.SourceUpdater, "run", reject)
    checker = startup.startup_updates(settings)
    assert checker.installed == OLD
    assert "ReadTimeout" in checker.startup_notice and "private download URL" not in checker.startup_notice
