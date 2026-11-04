import subprocess

import pytest

from velocicyber import desktop


@pytest.fixture
def installation(tmp_path):
    root = tmp_path / 'install with spaces, $cash and "quotes" %f = yes'
    root.mkdir()
    (root / "velocicyber.sh").write_text("#!/bin/sh\nprintf 'launched\\n' > launched.txt\n")
    (root / "assets").mkdir()
    (root / "assets/velocicyber.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    return root


@pytest.fixture(autouse=True)
def isolated_desktop(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    monkeypatch.setattr(desktop.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(desktop.shutil, "which", lambda _: None)


def test_localized_xdg_desktop_and_disabled_home(tmp_path, monkeypatch):
    folder = tmp_path / "Bureau"
    folder.mkdir()
    (tmp_path / "Desktop").mkdir()
    calls = []

    def resolve(arguments, **kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, str(folder) + "\n", "")

    monkeypatch.setattr(desktop.shutil, "which", lambda _: "/usr/bin/xdg-user-dir")
    monkeypatch.setattr(desktop.subprocess, "run", resolve)
    assert desktop.desktop_directory() == folder
    assert calls == [["/usr/bin/xdg-user-dir", "DESKTOP"]]
    folder = tmp_path  # XDG uses HOME to disable a special directory.
    assert desktop.desktop_directory() is None


@pytest.mark.parametrize("output", ["relative/Desktop\n", "", "/missing-fixture/Desktop\n"])
def test_invalid_configured_desktop_does_not_fall_back(tmp_path, monkeypatch, output):
    (tmp_path / "Desktop").mkdir()
    monkeypatch.setattr(desktop.shutil, "which", lambda _: "/usr/bin/xdg-user-dir")
    monkeypatch.setattr(desktop.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, output, ""))
    assert desktop.desktop_directory() is None


def test_fallback_does_not_create_desktop_and_non_linux_is_skipped(tmp_path, monkeypatch):
    assert desktop.desktop_directory() is None
    assert not (tmp_path / "Desktop").exists()
    folder = tmp_path / "Desktop"
    folder.mkdir()
    assert desktop.desktop_directory() == folder
    monkeypatch.setattr(desktop.sys, "platform", "darwin")
    assert desktop.desktop_directory() is None


def test_xdg_timeout_uses_existing_desktop(tmp_path, monkeypatch):
    (tmp_path / "Desktop").mkdir()

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 3)

    monkeypatch.setattr(desktop.shutil, "which", lambda _: "/usr/bin/xdg-user-dir")
    monkeypatch.setattr(desktop.subprocess, "run", timeout)
    assert desktop.desktop_directory() == tmp_path / "Desktop"


def test_existing_shortcut_preserved_and_repeat_is_idempotent(tmp_path, installation):
    folder = tmp_path / "Desktop"
    folder.mkdir()
    previous = folder / "Velocicyber.desktop"
    previous.write_text("existing prototype launcher")
    launcher = desktop.install_shortcut(installation, folder)
    assert launcher.name == "Velocicyber (2).desktop"
    assert previous.read_text() == "existing prototype launcher"
    assert launcher.stat().st_mode & 0o777 == 0o755
    assert "Terminal=true\n" in launcher.read_text()
    assert desktop.install_shortcut(installation, folder) == launcher
    assert {path.name for path in folder.iterdir()} == {
        "Velocicyber.desktop", "Velocicyber (2).desktop",
    }
    assert not (installation / "launched.txt").exists()


def test_existing_symlink_is_not_followed_or_replaced(tmp_path, installation):
    folder = tmp_path / "Desktop"
    folder.mkdir()
    target = tmp_path / "unrelated-file"
    target.write_text("keep me")
    (folder / "Velocicyber.desktop").symlink_to(target)
    assert desktop.install_shortcut(installation, folder).name == "Velocicyber (2).desktop"
    assert target.read_text() == "keep me"
    assert (folder / "Velocicyber.desktop").is_symlink()


def test_missing_assets_or_injected_paths_leave_no_shortcut(tmp_path, installation):
    folder = tmp_path / "Desktop"
    folder.mkdir()
    (installation / "assets/velocicyber.svg").unlink()
    with pytest.raises(ValueError, match="missing"):
        desktop.install_shortcut(installation, folder)
    assert list(folder.iterdir()) == []
    (installation / "assets/velocicyber.svg").write_text("icon")
    renamed = installation.with_name("injected\nExec=unexpected")
    installation.rename(renamed)
    with pytest.raises(ValueError, match="control"):
        desktop.install_shortcut(renamed, folder)
    assert list(folder.iterdir()) == []
