import importlib.util
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("bootstrap", Path(__file__).resolve().parents[1] / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
REVISION = "a" * 40
FILES = {
    "install.sh": b"#!/bin/sh\nprintf ready > installed.txt\n",
    "pyproject.toml": b'[project]\nname = "velocicyber"\n',
    "velocicyber/cli.py": b"# fixture\n",
    "velocicyber.sh": b"#!/bin/sh\nexit 0\n",
}


def archive_bytes(extra=()):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for name, data in FILES.items():
            member = tarfile.TarInfo("source/" + name)
            member.size = len(data)
            member.mode = 0o755 if name.endswith(".sh") else 0o644
            bundle.addfile(member, io.BytesIO(data))
        for member in extra:
            bundle.addfile(member, io.BytesIO(b"x" * member.size) if member.isfile() else None)
    return output.getvalue()


@pytest.fixture
def transport(monkeypatch):
    calls = []
    run = subprocess.run
    monkeypatch.setenv("GH_TOKEN", "private_fixture_token")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: "/usr/bin/curl")

    def fake_run(argv, **kwargs):
        calls.append(argv)
        assert "GH_TOKEN" not in kwargs["env"] and "GITHUB_TOKEN" not in kwargs["env"]
        assert not any("private_fixture_token" in arg for arg in argv)
        if argv[0] != "curl":
            return run(argv, **kwargs)
        assert kwargs["input"] == b"Authorization: Bearer private_fixture_token\n"
        assert argv[1] == "-q" and "--location-trusted" not in argv and "--insecure" not in argv
        assert argv[argv.index("--proto-redir") + 1] == "=https"
        output = Path(argv[argv.index("--output") + 1])
        if "/commits/" in argv[-1]:
            output.write_text(json.dumps({"sha": REVISION}))
        else:
            assert argv[-1].endswith("/tarball/" + REVISION)
            output.write_bytes(archive_bytes())
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("download_only", [True, False])
def test_download_pins_commit_and_keeps_auth_out_of_installed_files(tmp_path, transport, capsys, download_only):
    destination = tmp_path / "installation with spaces"
    bootstrap.install(destination, "feature/curl", download_only=download_only)
    assert transport[0][-1].endswith("/commits/feature%2Fcurl")
    assert len(transport) == (2 if download_only else 3)
    assert (destination / "installed.txt").exists() == (not download_only)
    receipt = destination / ".runtime/source-install.json"
    assert json.loads(receipt.read_text())["commit"] == REVISION
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert destination.stat().st_mode & 0o777 == 0o700
    assert not list(tmp_path.glob(".velocicyber-download-*"))
    assert "private_fixture_token" not in capsys.readouterr().out
    assert all(b"private_fixture_token" not in p.read_bytes() for p in destination.rglob("*") if p.is_file())


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_destination_is_never_touched_or_downloaded(tmp_path, transport, kind):
    destination = tmp_path / "existing"
    if kind == "directory":
        destination.mkdir()
        (destination / ".env").write_text("preserve")
    elif kind == "file":
        destination.write_text("preserve")
    else:
        destination.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="already exists"):
        bootstrap.install(destination)
    assert not transport
    if kind == "directory":
        assert (destination / ".env").read_text() == "preserve"
    elif kind == "file":
        assert destination.read_text() == "preserve"
    else:
        assert destination.is_symlink() and not (tmp_path / "absent").exists()


@pytest.mark.parametrize("name,kind", [
    ("source/../../escaped", tarfile.REGTYPE),
    ("/absolute", tarfile.REGTYPE),
    ("source/link", tarfile.SYMTYPE),
    ("source/link", tarfile.LNKTYPE),
    ("source/fifo", tarfile.FIFOTYPE),
    ("source/install.sh", tarfile.REGTYPE),
    ("other-root/file", tarfile.REGTYPE),
])
def test_rejects_unsafe_or_duplicate_archive_entries(tmp_path, name, kind):
    member = tarfile.TarInfo(name)
    member.type = kind
    member.linkname = "../../outside"
    archive = tmp_path / "download.tar.gz"
    archive.write_bytes(archive_bytes([member]))
    with pytest.raises(ValueError):
        bootstrap.unpack(archive, tmp_path / "unpacked")
    assert not (tmp_path / "escaped").exists()


def test_archive_size_limit_is_checked_before_writing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "MAX_SOURCE_BYTES", 8)
    archive = tmp_path / "download.tar.gz"
    archive.write_bytes(archive_bytes())
    folder = tmp_path / "unpacked"
    with pytest.raises(ValueError, match="too large"):
        bootstrap.unpack(archive, folder)
    assert list(folder.iterdir()) == []


def test_failed_download_cleans_staging_and_never_starts_installer(tmp_path, transport, monkeypatch):
    def fail(*args):
        raise RuntimeError("download failed")

    monkeypatch.setattr(bootstrap, "fetch", fail)
    with pytest.raises(RuntimeError, match="download failed"):
        bootstrap.install(tmp_path / "new")
    assert not transport and list(tmp_path.iterdir()) == []


def test_invalid_auth_is_rejected_before_network_or_directory_creation(tmp_path, transport, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "fixture\nInjected: header")
    with pytest.raises(ValueError, match="invalid format"):
        bootstrap.install(tmp_path / "new")
    assert not transport and list(tmp_path.iterdir()) == []


def test_destination_created_during_download_is_preserved(tmp_path, transport, monkeypatch):
    original = bootstrap.unpack
    destination = tmp_path / "new"

    def unpack_then_create(*args):
        original(*args)
        destination.mkdir()
        (destination / ".env").write_text("preserve")

    monkeypatch.setattr(bootstrap, "unpack", unpack_then_create)
    with pytest.raises(FileExistsError):
        bootstrap.install(destination)
    assert (destination / ".env").read_text() == "preserve"
    assert not (destination / "install.sh").exists()


def test_curl_failure_does_not_echo_token_or_provider_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(
        a[0], 22, stdout=b"", stderr=b"private_fixture_token"))
    with pytest.raises(RuntimeError, match="curl exit 22") as error:
        bootstrap.fetch(bootstrap.API, tmp_path / "download", "private_fixture_token")
    assert "private_fixture_token" not in str(error.value) + capsys.readouterr().out


def test_dependency_failure_retains_downloaded_source_for_retry(tmp_path, transport, monkeypatch):
    fake_run = bootstrap.subprocess.run

    def fail_install(argv, **kwargs):
        if argv[0] == "sh":
            return subprocess.CompletedProcess(argv, 1)
        return fake_run(argv, **kwargs)

    monkeypatch.setattr(bootstrap.subprocess, "run", fail_install)
    destination = tmp_path / "new"
    with pytest.raises(RuntimeError, match="Source was retained"):
        bootstrap.install(destination)
    assert (destination / "install.sh").is_file()
    assert (destination / ".runtime/source-install.json").is_file()
    assert not list(tmp_path.glob(".velocicyber-download-*"))
