import json

import pytest

from velocicyber.config import read_assignments
from velocicyber.onboarding import MARKER, run_setup
from velocicyber.ssh_setup import lambda_ssh_options, private_key_path

KEY = b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----\nsynthetic-fixture-only\n-----END OPENSSH PRIVATE KEY-----\n"


def answers(*values):
    values = iter(values)
    return lambda _: next(values)


def test_existing_local_key_is_selected_by_number_and_secured(tmp_path, capsys):
    folder = tmp_path / "ssh-keys"
    folder.mkdir()
    (folder / "README.md").write_text("docs")
    key = folder / "test.pem"
    key.write_bytes(KEY)
    key.chmod(0o644)
    options = lambda_ssh_options(tmp_path, {}, answers("lambda-key-label", "1"))
    assert options == {"LAMBDA_SSH_KEY_NAME": "lambda-key-label", "SSH_PRIVATE_KEY": "ssh-keys/test.pem"}
    assert key.stat().st_mode & 0o777 == 0o600
    assert "synthetic-fixture" not in capsys.readouterr().out


def test_external_key_is_imported_without_overwriting_original_or_existing_keys(tmp_path):
    root = tmp_path / "installation"
    root.mkdir()
    key = tmp_path / "external.pem"
    key.write_bytes(KEY)
    first = private_key_path(root, str(key))
    second = private_key_path(root, str(key))
    assert first != second and first.startswith("ssh-keys/")
    assert (root / first).read_bytes() == key.read_bytes() == KEY
    assert (root / first).stat().st_mode & 0o777 == 0o600
    assert (root / second).read_bytes() == KEY


@pytest.mark.parametrize("value", [b"ssh-ed25519 public-key", b"API_SECRET=value", b"x" * 65537])
def test_non_private_keys_are_rejected_without_echoing_contents(tmp_path, value):
    key = tmp_path / "not-a-private-key"
    key.write_bytes(value)
    with pytest.raises(ValueError, match="private SSH key"):
        private_key_path(tmp_path, key.name)


def test_import_refuses_symlinked_key_directory(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "ssh-keys").symlink_to(outside)
    key = tmp_path / "external.pem"
    key.write_bytes(KEY)
    with pytest.raises(ValueError, match="symlink"):
        private_key_path(root, str(key))
    assert list(outside.iterdir()) == []


def test_skip_and_invalid_input_do_not_create_configuration(tmp_path, capsys):
    assert lambda_ssh_options(tmp_path, {}, answers("")) == {}
    assert lambda_ssh_options(tmp_path, {}, answers("-----BEGIN " + "PRIVATE KEY-----", "skip")) == {}
    assert not list(tmp_path.iterdir())
    assert "no key contents" in capsys.readouterr().out


def test_missing_file_reprompts_and_name_only_can_be_saved(tmp_path):
    options = lambda_ssh_options(tmp_path, {}, answers("registered-label", "missing.pem", "skip"))
    assert options == {"LAMBDA_SSH_KEY_NAME": "registered-label"}


def test_existing_install_gets_only_new_ssh_prompt_once_and_keeps_secrets(tmp_path, capsys):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    marker = runtime / MARKER
    marker.write_text('{"schema":1,"completed_at":"previous"}\n')
    (tmp_path / ".env").write_text("LAMBDA_API=keep-private\nADVERSARIAL_API=keep-api\n")
    (tmp_path / "Modules.txt").write_text("FINETUNING_REPO=owner/keep\nCUSTOM=keep\n")
    key = tmp_path / "lambda.pem"
    key.write_bytes(KEY)

    def no_secret(_):
        pytest.fail("Migration must not re-prompt provider credentials")

    assert run_setup(tmp_path, interactive=True, input_fn=answers("registered-key", "lambda.pem"), secret_fn=no_secret)
    values = read_assignments(tmp_path / "Modules.txt")
    assert values["LAMBDA_SSH_KEY_NAME"] == "registered-key"
    assert values["SSH_PRIVATE_KEY"] == "lambda.pem"
    assert values["FINETUNING_REPO"] == "owner/keep" and values["CUSTOM"] == "keep"
    assert (tmp_path / "Modules.txt").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / ".env").read_text() == "LAMBDA_API=keep-private\nADVERSARIAL_API=keep-api\n"
    assert json.loads(marker.read_text())["lambda_ssh_setup"] == 1
    assert not run_setup(tmp_path, interactive=True, input_fn=no_secret, secret_fn=no_secret)
    assert "keep-private" not in capsys.readouterr().out


def test_headless_launch_defers_new_prompt_and_skip_is_remembered(tmp_path):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    marker = runtime / MARKER
    marker.write_text('{"schema":1}\n')
    assert not run_setup(tmp_path, interactive=False)
    assert "lambda_ssh_setup" not in json.loads(marker.read_text())
    assert run_setup(tmp_path, interactive=True, input_fn=answers(""))
    assert not (tmp_path / "Modules.txt").exists()
    assert not run_setup(tmp_path, interactive=True, input_fn=answers())


def test_first_run_saves_ssh_settings_after_optional_setup(tmp_path, monkeypatch):
    from velocicyber import onboarding
    monkeypatch.setattr(onboarding, "desktop_directory", lambda: None)
    key = tmp_path / "key.pem"
    key.write_bytes(KEY)
    for name, _ in onboarding.CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    assert run_setup(tmp_path, interactive=True, secret_fn=answers("", "", ""),
                     input_fn=answers("", "", "n", "", "registered-key", "key.pem"))
    values = read_assignments(tmp_path / "Modules.txt")
    assert values["LAMBDA_SSH_KEY_NAME"] == "registered-key"
    assert values["SSH_PRIVATE_KEY"] == "key.pem"
    assert (tmp_path / "Modules.txt").stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / ".env").exists()
