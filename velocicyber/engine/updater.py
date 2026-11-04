"""Manually approved, staged engine updates with compatibility gates and rollback."""

import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import uuid
import xml.etree.ElementTree as ET
from contextlib import ExitStack, contextmanager

import httpx

from velocicyber.engine.downloads import download
from velocicyber.engine.installation import (
    ACTIVE_FILE,
    RECOVERY_FILE,
    UPDATE_NOTICE_FILE,
    active_record,
    engine_lease,
    runtime_path,
)
from velocicyber.engine.releases import API, REPOSITORY, rust_version, stable_rust_release
from velocicyber.engine.sessions import private_write

ASSET = "open-interpreter-package-aarch64-unknown-linux-musl.tar.gz"


def save_json(path, value):
    private_write(path, json.dumps(value, indent=2) + "\n")
    # Persist the rename itself as well as the file contents.
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binary_tag(binary):
    result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, timeout=10, check=True,
    )
    match = re.fullmatch(r"interpreter (\d+\.\d+\.\d+)", result.stdout.strip())
    if not match:
        raise RuntimeError("The installed engine did not report a supported version.")
    return "rust-v" + match[1]


@contextmanager
def idle_conversations(runtime):
    """Also detect conversations held by harness versions predating the global lock."""
    with ExitStack() as stack:
        for path in (runtime / "conversations").glob("*/session.lock"):
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            stream = stack.enter_context(os.fdopen(fd, "r+"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("A saved conversation is still open. Close its harness first.") from None
        yield


class EngineUpdater:
    def __init__(self, settings, notify=print):
        self.settings, self.notify = settings, notify
        self.runtime = settings.runtime
        self.root = self.runtime / "engine-updates"

    def preflight(self, *, testing=True):
        if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
            raise RuntimeError("This updater currently supports the Raspberry Pi Linux ARM64 bundle.")
        if self.settings.options.get("INTERPRETER_BIN"):
            raise RuntimeError("Remove the INTERPRETER_BIN override before using managed updates.")
        for name in ("pytest", "pytest_asyncio") if testing else ():
            if importlib.util.find_spec(name) is None:
                raise RuntimeError("Install the project's development dependencies before updating.")
        if testing and not (self.settings.root / "tests/test_engine_integration.py").is_file():
            raise RuntimeError("The compatibility test suite is missing; refusing an untested update.")
        self.root.mkdir(mode=0o700, exist_ok=True)

    def release(self, client, tag=None):
        if tag and not rust_version(tag):
            raise ValueError("Select a stable rust-vMAJOR.MINOR.PATCH tag.")
        response = client.get(API + ("/releases/tags/" + tag if tag else "/releases/latest"))
        response.raise_for_status()
        release = response.json()
        if not tag and not stable_rust_release(release):
            response = client.get(API + "/releases?per_page=20")
            response.raise_for_status()
            releases = [item for item in response.json() if stable_rust_release(item)]
            if not releases:
                raise RuntimeError("No stable Rust release found.")
            release = max(releases, key=lambda item: rust_version(item["tag_name"]))
        if not stable_rust_release(release):
            raise RuntimeError("The selected release is not a stable Rust release.")
        tag = release["tag_name"]
        asset = next((a for a in release.get("assets", []) if a["name"] == ASSET), None)
        if not asset or not re.fullmatch(r"sha256:[a-f0-9]{64}", asset.get("digest") or ""):
            raise RuntimeError("The ARM64 bundle or its official SHA256 digest is missing.")
        url = f"{REPOSITORY}/releases/download/{tag}/{ASSET}"
        if (
            asset.get("browser_download_url") != url
            or not isinstance(asset.get("size"), int) or not 0 < asset["size"] <= 1024**3
        ):
            raise RuntimeError("Unexpected upstream asset metadata.")
        return {
            "repository": REPOSITORY, "tag": tag, "asset": ASSET,
            "sha256": asset["digest"].split(":", 1)[1], "size": asset["size"], "url": url,
        }

    def stage(self, tag=None):
        self.preflight()
        with httpx.Client(
            timeout=httpx.Timeout(60, connect=15), follow_redirects=True,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "Velocicyber-updater"},
        ) as client:
            release = self.release(client, tag)
            current = binary_tag(self.settings.engine())
            if rust_version(release["tag"]) <= rust_version(current):
                raise RuntimeError(f"{current} is already current for the selected release.")
            folder = self.root / (release["tag"] + "-" + release["sha256"][:16])
            folder.mkdir(mode=0o700, exist_ok=True)
            self.notify(f"Staging {release['tag']} ({release['size'] / 1024**2:.1f} MiB).")
            archive = folder / ASSET
            if not archive.exists() or sha256(archive) != release["sha256"]:
                if shutil.disk_usage(self.runtime).free < release["size"] * 10 + 1024**3:
                    raise RuntimeError("Not enough free disk space to stage and test this engine.")
                partial = folder / "download.partial"
                download(client, release["url"], partial, release["size"], self.notify)
                if partial.stat().st_size != release["size"] or sha256(partial) != release["sha256"]:
                    raise RuntimeError("Engine checksum/size mismatch; the active engine is unchanged.")
                partial.replace(archive)
                shutil.rmtree(partial.with_suffix(".parts"))
            # Re-extract verified bytes for every approved attempt; old test receipts are not reused.
            bundle = folder / ("bundle-" + uuid.uuid4().hex)
            bundle.mkdir(mode=0o700)
            with tarfile.open(archive) as tar:
                members = tar.getmembers()
                if sum(item.size for item in members) > 4 * 1024**3:
                    raise RuntimeError("Engine archive expands beyond the supported size limit.")
                tar.extractall(bundle, members=members, filter="data")
            package = json.loads((bundle / "codex-package.json").read_text())
            if (
                package.get("variant") != "open-interpreter"
                or package.get("target") != "aarch64-unknown-linux-musl"
                or package.get("entrypoint") != "bin/interpreter"
                or "rust-v" + package.get("version", "") != release["tag"]
            ):
                raise RuntimeError("Downloaded package identity does not match its release.")
            binary = bundle / "bin/interpreter"
            if binary_tag(binary) != release["tag"]:
                raise RuntimeError("Downloaded engine's actual version does not match its release.")
            record = {
                "schema": 1, "entrypoint": str(binary.relative_to(self.runtime)),
                "release": release, "archive": str(archive.relative_to(self.runtime)),
            }
            save_json(bundle.parent / "candidate.json", record)
            self.notify("Verified and staged; the active engine has not changed.")
            return record

    def validate(self, candidate):
        binary = runtime_path(self.runtime, candidate["entrypoint"])
        run = self.root / ("validation-" + uuid.uuid4().hex)
        run.mkdir(mode=0o700)
        report, log = run / "results.xml", run / "tests.log"
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "LANG", "TMPDIR"}}
        env.update({
            "PYTHONPATH": str(self.settings.root), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "VELOCICYBER_TEST_ENGINE": str(binary),
            "VELOCICYBER_PREVIOUS_ENGINE": str(self.settings.engine()),
        })
        command = [
            sys.executable, "-m", "pytest", "-p", "pytest_asyncio.plugin", "-q",
            "tests/test_engine_integration.py", "tests/test_sessions.py", "tests/test_data.py",
            "tests/test_sandbox.py", "tests/test_instruction_scope.py", "tests/test_sentinel_observer.py",
            "--basetemp=" + str(run / "fixtures"), "--junitxml=" + str(report),
        ]
        self.notify("Testing tools, filesystem access, responses, compaction, data capture, and old/new session resume…")
        with log.open("w") as stream:
            process = subprocess.Popen(
                command, cwd=self.settings.root, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                code = process.wait(timeout=300)
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        try:
            cases = ET.parse(report).findall(".//testcase") if report.exists() else []
        except ET.ParseError:
            raise RuntimeError(f"Invalid compatibility report; engine unchanged. See {log}") from None
        if code or len(cases) < 12 or any(
            case.find(kind) is not None for case in cases for kind in ("failure", "error", "skipped")
        ):
            raise RuntimeError(f"Compatibility tests failed or were skipped. Engine unchanged. See {log}")
        receipt = {
            "passed": True, "tests": len(cases), "entrypoint": candidate["entrypoint"],
            "binary_sha256": sha256(binary), "report": str(report.relative_to(self.runtime)),
            "previous_entrypoint": str(self.settings.engine()),
        }
        save_json(run / "validation.json", receipt)
        self.notify(f"Passed {len(cases)} compatibility tests. Report: {report}")
        return receipt

    def activate(self, candidate, receipt):
        binary = runtime_path(self.runtime, candidate["entrypoint"])
        if (
            not receipt.get("passed") or receipt.get("entrypoint") != candidate["entrypoint"]
            or receipt.get("binary_sha256") != sha256(binary)
            or receipt.get("previous_entrypoint") != str(self.settings.engine())
        ):
            raise RuntimeError("The candidate has not passed validation against the active engine.")
        previous = active_record(self.runtime) or {
            "schema": 1,
            "entrypoint": str(self.settings.engine().relative_to(self.runtime)),
            "release": json.loads((self.settings.root / "config/locks/upstream.lock.json").read_text()),
        }
        backup = self.root / ("backup-" + uuid.uuid4().hex)
        backup.mkdir(mode=0o700)
        with idle_conversations(self.runtime):
            self.notify("Backing up private saved conversations before activation…")
            conversations = self.runtime / "conversations"
            if conversations.is_symlink():
                raise RuntimeError("Conversation storage must not be a symlink.")
            size = sum(p.stat().st_size for p in conversations.rglob("*") if p.is_file() and not p.is_symlink())
            if shutil.disk_usage(self.runtime).free < size + 512 * 1024**2:
                raise RuntimeError("Not enough space for the required conversation backup.")
            if conversations.exists():
                shutil.copytree(conversations, backup / "conversations", symlinks=True)
            else:
                (backup / "conversations").mkdir(mode=0o700)
            save_json(backup / "previous-engine.json", previous)
            save_json(backup / "validation.json", receipt)
            new = {**candidate, "previous": previous, "backup": str(backup.relative_to(self.runtime))}
            save_json(self.runtime / UPDATE_NOTICE_FILE, {
                "activation": new["backup"], "entrypoint": new["entrypoint"],
                "tag": candidate["release"]["tag"],
            })
            save_json(self.runtime / ACTIVE_FILE, new)
        self.notify(f"Activated {candidate['release']['tag']}. Previous engine and history backup: {backup}")
        self.notify("Reopen Velocicyber to use the updated engine.")

    def rollback(self):
        journal = self.runtime / RECOVERY_FILE
        with idle_conversations(self.runtime):
            if not journal.exists():
                active = active_record(self.runtime)
                if not active or not active.get("previous") or not active.get("backup"):
                    raise RuntimeError("No managed update is available to roll back.")
                backup = runtime_path(self.runtime, active["backup"]) / "conversations"
                previous = active["previous"]
                binary = runtime_path(self.runtime, previous["entrypoint"])
                binary_tag(binary)
                transaction = self.root / ("rollback-" + uuid.uuid4().hex)
                transaction.mkdir(mode=0o700)
                restore = transaction / "restore"
                shutil.copytree(backup, restore, symlinks=True)
                save_json(journal, {
                    "previous": previous, "restore": str(restore.relative_to(self.runtime)),
                    "archive": str((transaction / "conversations-after-update").relative_to(self.runtime)),
                })
            recovery = json.loads(journal.read_text())
            restore = runtime_path(self.runtime, recovery["restore"])
            archive = runtime_path(self.runtime, recovery["archive"])
            current = self.runtime / "conversations"
            runtime_path(self.runtime, recovery["previous"]["entrypoint"])
            if restore.exists():
                if current.exists():
                    if archive.exists():
                        raise RuntimeError("Rollback needs review: both current and archived history exist.")
                    current.rename(archive)
                restore.rename(current)
            elif not current.is_dir():
                raise RuntimeError("Rollback recovery history is missing.")
            save_json(self.runtime / ACTIVE_FILE, recovery["previous"])
            (self.runtime / UPDATE_NOTICE_FILE).unlink(missing_ok=True)
            journal.unlink()
        self.notify("Restored the previous engine and its saved conversations.")
        self.notify(f"Conversations created since the update are preserved separately at {archive}")

    def run(self, *, approved=False, action="update", tag=None):
        if not approved:
            raise PermissionError("Explicit approval is required before starting engine maintenance.")
        if action not in {"update", "stage", "rollback"}:
            raise ValueError("Unknown engine maintenance action.")
        if action == "rollback" and tag:
            raise ValueError("Rollback restores the previous engine; --target applies only to updates/tests.")
        self.preflight(testing=action != "rollback")
        if action == "stage":
            candidate = self.stage(tag)
            self.validate(candidate)
            self.notify("Validation only: the working engine remains unchanged.")
            return
        with engine_lease(self.runtime, maintenance=True):
            if action == "rollback":
                self.rollback()
            else:
                # Refuse before downloading if a legacy harness still owns a conversation.
                with idle_conversations(self.runtime):
                    pass
                candidate = self.stage(tag)
                receipt = self.validate(candidate)
                self.activate(candidate, receipt)


def interactive_update(settings, *, action="update", tag=None, approved=False):
    if not approved:
        explanation = {
            "update": "Download, verify and test the latest engine, back up conversations, then activate only if tests pass.",
            "stage": "Download and test the engine in isolation. Keep the working engine unchanged.",
            "rollback": "Restore the previous engine and pre-update conversations; archive newer conversations separately.",
        }[action]
        print(explanation)
        word = {"update": "UPDATE", "stage": "TEST", "rollback": "ROLLBACK"}[action]
        try:
            approved = input(f"Type {word} to proceed, or Enter to cancel: ").strip() == word
        except EOFError:
            approved = False
        if not approved:
            print("Cancelled. No engine maintenance started.")
            return
    try:
        EngineUpdater(settings, lambda text: print(text, flush=True)).run(
            approved=True, action=action, tag=tag,
        )
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Engine maintenance stopped: {exc}") from exc
