import getpass
import json
import stat
import warnings
from pathlib import Path

import pytest

from velocicyber import onboarding
from velocicyber.config import Settings, read_assignments
from velocicyber.state import State


@pytest.fixture(autouse=True)
def isolated_setup(monkeypatch):
    for key, _ in onboarding.CREDENTIALS:
        monkeypatch.delenv(key, raising=False)

    def no_install(*args, **kwargs):
        pytest.fail("Unexpected runtime installation")

    def no_create(*args, **kwargs):
        pytest.fail("Unexpected dataset creation")

    monkeypatch.setattr(onboarding, "install_component", no_install)
    monkeypatch.setattr(onboarding, "desktop_directory", lambda: None)
    monkeypatch.setattr(onboarding, "install_shortcut", no_install)
    monkeypatch.setattr("huggingface_hub.HfApi.create_repo", no_create)


def responses(*values):
    answers = iter(values)
    return lambda _: next(answers)


def setup(root, answers=(), secrets=("", "", ""), ssh_answers=("",), **kwargs):
    return onboarding.run_setup(
        root, interactive=True, input_fn=responses(*answers, *ssh_answers),
        secret_fn=responses(*secrets), **kwargs,
    )


def test_skip_everything_then_marker_prevents_repeat(tmp_path, capsys):
    assert setup(tmp_path, ("", "skip", "n", ""))
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / "Modules.txt").exists()
    marker = tmp_path / ".runtime" / onboarding.MARKER
    assert json.loads(marker.read_text())["schema"] == 1
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert not setup(tmp_path)
    assert "skip" in capsys.readouterr().out.lower()
    state = State()
    assert not state.collection and not state.upload_approved


def test_separate_dataset_prompts_hidden_secrets_and_private_files(tmp_path, capsys):
    secrets = ("lambda-fixture-key", "api-fixture-key", "hf-fixture-key")
    assert setup(tmp_path, ("owner/main-data", "owner/sentinel-data", "n", "n", "n", ""), secrets)
    settings = Settings.load(tmp_path)
    assert settings.dataset_repo == "owner/main-data"
    assert settings.options["SENTINEL_FINETUNING_REPO"] == "owner/sentinel-data"
    assert tuple(settings.secrets[key] for key, _ in onboarding.CREDENTIALS) == secrets
    for name in (".env", "Modules.txt"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600
    output = capsys.readouterr().out
    assert all(key not in output for key in secrets)
    assert all(key not in (tmp_path / ".runtime" / onboarding.MARKER).read_text() for key in secrets)


def test_existing_credentials_are_preserved_and_prompts_can_be_reopened(tmp_path):
    original = "# owner configuration\nLAMBDA_API=existing-lambda\nADVERSARIAL_API=existing-api\nHUGGINGFACE_API=existing-hf\nCUSTOM=value\n"
    path = tmp_path / ".env"
    path.write_text(original)
    path.chmod(0o644)
    assert setup(tmp_path, ("", "", "n", "n", ""), secrets=())
    assert path.read_text() == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert setup(tmp_path, ("", "", "n", "n", ""), ("new-lambda", "", ""), force=True)
    values = read_assignments(path)
    assert values["LAMBDA_API"] == "new-lambda"
    assert values["CUSTOM"] == "value" and values["HUGGINGFACE_API"] == "existing-hf"


def test_invalid_or_identical_dataset_ids_are_reprompted(tmp_path):
    setup(tmp_path, ("just-name", "owner/name.", "owner/main", "owner/main", "../escape", "owner/sentinel", "n", ""))
    values = read_assignments(tmp_path / "Modules.txt")
    assert values == {"FINETUNING_REPO": "owner/main", "SENTINEL_FINETUNING_REPO": "owner/sentinel"}


def test_dataset_creation_requires_explicit_consent_and_uses_private_repos(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("huggingface_hub.HfApi.create_repo", lambda self, **kwargs: calls.append(kwargs))
    setup(tmp_path, ("owner/main", "owner/sentinel", "yes", "n", ""), ("", "", "hf-fixture"))
    assert calls == [
        {"repo_id": "owner/main", "repo_type": "dataset", "private": True, "exist_ok": True},
        {"repo_id": "owner/sentinel", "repo_type": "dataset", "private": True, "exist_ok": True},
    ]


def test_failed_creation_never_prints_key(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("provider rejected hf-private-fixture-key")

    monkeypatch.setattr("huggingface_hub.HfApi.create_repo", fail)
    setup(tmp_path, ("owner/main", "", "yes", "n", ""), ("", "", "hf-private-fixture-key"))
    assert "hf-private-fixture-key" not in capsys.readouterr().out


@pytest.mark.parametrize("choice,models", [
    ("1", [onboarding.PI_MODEL_REPO]),
    ("2", [onboarding.BUGTRACE_PI_REPO]),
    ("3", [onboarding.PI_MODEL_REPO, onboarding.BUGTRACE_PI_REPO]),
    ("4", [onboarding.LFM3_PI_REPO]),
    ("5", [onboarding.PI_MODEL_REPO, onboarding.BUGTRACE_PI_REPO, onboarding.LFM3_PI_REPO]),
])
def test_optional_pi_download_dispatch(tmp_path, monkeypatch, choice, models):
    calls = []
    monkeypatch.setattr(onboarding, "install_component", lambda *args: calls.append(args))
    setup(tmp_path, ("", "", "n", choice))
    assert calls == [(tmp_path, "install_pi_model.py", "--model", model) for model in models]


def test_headless_mode_does_not_prompt_or_claim_setup_done(tmp_path):
    assert not onboarding.run_setup(tmp_path, interactive=False)
    assert not (tmp_path / ".runtime" / onboarding.MARKER).exists()
    with pytest.raises(ValueError, match="terminal"):
        onboarding.run_setup(tmp_path, force=True, interactive=False)


@pytest.mark.parametrize("name", [".env", "Modules.txt", ".runtime"])
def test_setup_rejects_symlinked_configuration(tmp_path, name):
    target = tmp_path / "target"
    target.write_text("private")
    (tmp_path / name).symlink_to(target)
    with pytest.raises(ValueError, match="regular local"):
        setup(tmp_path)
    assert target.read_text() == "private"


def test_assignment_rejects_injected_lines_and_symlinks(tmp_path):
    target = tmp_path / ".env"
    with pytest.raises(ValueError, match="control"):
        onboarding.private_assignments(target, {"LAMBDA_API": "x\nINJECTED=1"})
    assert not target.exists()
    target.symlink_to(tmp_path / "other")
    with pytest.raises(ValueError, match="symlink"):
        onboarding.private_assignments(target, {"LAMBDA_API": "fixture"})
    assert not (tmp_path / "other").exists()


def test_unavailable_hidden_input_fails_without_echoing(tmp_path):
    def unsafe_input(_):
        warnings.warn("unavailable", getpass.GetPassWarning)
        pytest.fail("Should not fall back to echoing secrets")

    with pytest.raises(ValueError, match="Hidden input"):
        onboarding.run_setup(tmp_path, interactive=True, secret_fn=unsafe_input)


def test_example_configuration_is_portable_and_local_overrides_win(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "Modules.example.txt").write_bytes((root / "Modules.example.txt").read_bytes())
    settings = Settings.load(tmp_path)
    assert len(settings.gpu_model_choices) == 6
    assert len(settings.pi_model_choices) == 3
    assert not settings.local_model and not settings.dataset_repo
    assert not settings.options["SENTINEL_FINETUNING_REPO"]
    (tmp_path / "Modules.txt").write_text("FINETUNING_REPO=owner/main\nSENTINEL_FINETUNING_REPO=owner/sentinel\n")
    settings = Settings.load(tmp_path)
    assert settings.dataset_repo == "owner/main"
    assert len(settings.gpu_model_choices) == 6


@pytest.mark.parametrize("choice,created", [("yes", True), ("", False), ("n", False)])
def test_desktop_launcher_requires_explicit_choice(tmp_path, monkeypatch, choice, created):
    directory = tmp_path / "Desktop"
    calls = []
    monkeypatch.setattr(onboarding, "desktop_directory", lambda: directory)

    def install(root, destination):
        calls.append((root, destination))
        return destination / "Velocicyber.desktop"

    monkeypatch.setattr(onboarding, "install_shortcut", install)
    assert setup(tmp_path, ("", "", "n", "", choice))
    assert calls == ([(tmp_path, directory)] if created else [])
    assert not setup(tmp_path)  # Completed setup does not ask on every launch.


def test_optional_desktop_failure_does_not_block_setup(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(onboarding, "desktop_directory", lambda: tmp_path / "Desktop")

    def fail(*args):
        raise PermissionError("Desktop folder is read-only")

    monkeypatch.setattr(onboarding, "install_shortcut", fail)
    assert setup(tmp_path, ("", "", "n", "", "y"))
    assert (tmp_path / ".runtime" / onboarding.MARKER).is_file()
    output = capsys.readouterr().out
    assert "could not be created" in output and "setup to retry later" in output
