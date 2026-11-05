import sys
from pathlib import Path

import pytest

from velocicyber import cli
from velocicyber.datasets.pipeline import RawStore, Scrubber


@pytest.mark.parametrize("sandbox_open", [False, True])
@pytest.mark.parametrize(
    "flags,collect,upload,compaction",
    [
        ([], False, False, True),
        (["--collect", "--upload"], True, True, True),
        (["--collect"], True, False, True),
        (["--upload"], False, False, True),
        (["--no-collect"], False, False, True),
        (["--no-upload"], False, False, True),
        (["--no-allow-compaction"], False, False, False),
    ],
)
def test_run_collection_upload_opt_in(
    settings, episode, monkeypatch, flags, collect, upload, compaction, sandbox_open
):
    from velocicyber.updates import startup
    monkeypatch.setattr(startup, "startup_updates", lambda *a: pytest.fail("Unexpected source update"))
    calls = []

    class Engine:
        def __init__(self, settings, state, workspace=None):
            self.state = state
            self.scrubber = Scrubber()

        async def run(self, prompt, callback):
            assert self.state.sandbox_open == sandbox_open
            assert self.state.collection == collect
            assert self.state.allow_compaction == compaction
            if collect:
                RawStore(settings.data).append(episode)

        def close(self):
            calls.append("closed")

    class Uploader:
        def __init__(self, settings):
            pass

        def upload(self, artifact, *, approved):
            assert approved and (artifact / "manifest.json").exists()
            calls.append("uploaded")
            return {"training_hash": "fixture"}

    monkeypatch.setattr(cli.Settings, "load", lambda root: settings)
    monkeypatch.setattr(cli, "Engine", Engine)
    monkeypatch.setattr(cli, "Uploader", Uploader)
    access_flags = ["--sandbox", "open"] if sandbox_open else []
    monkeypatch.setattr(sys, "argv", ["velocicyber", *access_flags, "run", "Fixture task", *flags])
    cli.main()
    assert ("uploaded" in calls) == upload
    assert calls[-1] == "closed"



def test_menu_setup_runs_after_ui_closes_and_reloads_secure_configuration(settings, monkeypatch, capsys):
    from velocicyber import onboarding
    from velocicyber.config import Settings
    from velocicyber.engine.installation import engine_lease
    from velocicyber.ui import app as ui
    from velocicyber.updates import UpdateChecker

    for key, _ in onboarding.CREDENTIALS:
        monkeypatch.delenv(key, raising=False)
    calls, apps = [], []
    desktop = settings.root / "Desktop"
    desktop.mkdir()
    (settings.root / "assets").mkdir()
    (settings.root / "assets/velocicyber.svg").write_text("<svg/>")
    (settings.root / "velocicyber.sh").write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(onboarding, "desktop_directory", lambda: desktop)
    monkeypatch.setattr(onboarding, "install_component", lambda *a, **k: pytest.fail("Unexpected install"))
    original_setup = onboarding.run_setup
    answers = iter(["owner/main", "owner/sentinel", "n", "n", "", "y", ""])
    secrets = iter(["lambda-private-fixture", "api-private-fixture", ""])

    def wizard(root, *, force=False):
        if not force:
            return False
        assert calls == ["UI closed"]
        # The UI's shared lease was released; setup now owns exclusive maintenance.
        with pytest.raises(RuntimeError):
            with engine_lease(settings.runtime):
                pytest.fail("Setup must exclude concurrent model sessions")
        calls.append("setup")
        return original_setup(root, force=True, interactive=True,
                              input_fn=lambda _: next(answers), secret_fn=lambda _: next(secrets))

    class App:
        def __init__(self, configuration, **kwargs):
            self.settings = configuration
            assert kwargs["sandbox_open"] and kwargs["workspace"] == settings.root / "workspace"
            self.updates = UpdateChecker(configuration)
            self.maintenance_action = "setup" if not apps else ""
            apps.append(self)

        def run(self):
            if len(apps) == 1:
                self.updates.source.checked_this_launch = self.updates.engine.checked_this_launch = True
                calls.append("UI closed")
            else:
                assert self.settings.secrets["ADVERSARIAL_API"] == "api-private-fixture"
                assert self.settings.dataset_repo == "owner/main"
                assert self.updates.source is apps[0].updates.source
                assert self.updates.engine is apps[0].updates.engine
                assert self.updates.source.settings is self.settings
                assert (desktop / "Velocicyber.desktop").is_file()
                calls.append("UI reopened")

    monkeypatch.setattr(onboarding, "run_setup", wizard)
    monkeypatch.setattr(ui, "VelocicyberApp", App)
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(sys, "argv", ["velocicyber", "--root", str(settings.root),
                                    "--workspace", str(settings.root / "workspace"), "--sandbox", "open"])
    cli.main()
    assert calls == ["UI closed", "setup", "UI reopened"]
    for name in [".env", "Modules.txt"]:
        assert (settings.root / name).stat().st_mode & 0o777 == 0o600
    output = capsys.readouterr()
    assert "private-fixture" not in output.out + output.err
    assert not settings.data.exists()
    assert Settings.load(settings.root).options["SENTINEL_FINETUNING_REPO"] == "owner/sentinel"


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt(), ValueError("Hidden input is unavailable")])
def test_menu_setup_interruptions_and_validation_errors_return_control(settings, monkeypatch, capsys, error):
    from velocicyber import onboarding
    from velocicyber.engine.installation import engine_lease

    def interrupted(*args, **kwargs):
        raise error
    monkeypatch.setattr(onboarding, "run_setup", interrupted)
    monkeypatch.setattr("builtins.input", lambda _: "")
    cli.run_menu_setup(settings)
    output = capsys.readouterr()
    assert "Setup" in output.out + output.err
    with engine_lease(settings.runtime):
        pass


def test_menu_setup_refuses_prompts_while_another_harness_is_open(settings, monkeypatch):
    from velocicyber import onboarding
    from velocicyber.engine.installation import engine_lease
    monkeypatch.setattr(onboarding, "run_setup", lambda *a, **k: pytest.fail("Unexpected prompts"))
    monkeypatch.setattr("builtins.input", lambda _: "")
    with engine_lease(settings.runtime):
        cli.run_menu_setup(settings)


def test_real_ui_tears_down_before_menu_setup_and_reopens(settings, monkeypatch):
    from velocicyber import onboarding
    from velocicyber.ui import app as ui

    opened, lifecycle = [], []
    original = ui.VelocicyberApp

    class HeadlessApp(original):
        CSS_PATH = Path(ui.__file__).with_name("styles.tcss")

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

        def on_mount(self):
            super().on_mount()
            self.call_after_refresh(self.begin_setup if len(opened) == 1 else self.action_quit)

        async def on_unmount(self):
            await super().on_unmount()
            lifecycle.append("closed")

        def run(self):
            return super().run(headless=True)

    def wizard(root, *, force=False):
        if force:
            assert lifecycle == ["closed"]
            assert not opened[0].is_running
            assert opened[0].engine.session is None
            lifecycle.append("hidden prompts")

    monkeypatch.setattr(onboarding, "run_setup", wizard)
    monkeypatch.setattr(ui, "VelocicyberApp", HeadlessApp)
    monkeypatch.setattr(HeadlessApp, "refresh_updates", lambda self: None)
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(sys, "argv", ["velocicyber", "--root", str(settings.root)])
    cli.main()
    assert len(opened) == 2
    assert lifecycle == ["closed", "hidden prompts", "closed"]
