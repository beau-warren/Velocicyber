"""Selection and inter-process locking for managed engine versions."""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

ACTIVE_FILE = "engine-active.json"
RECOVERY_FILE = "engine-rollback.json"
UPDATE_NOTICE_FILE = "engine-update-notice.json"


def runtime_path(runtime, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Invalid managed engine path.")
    path = (runtime / relative).resolve()
    if path == runtime.resolve() or not path.is_relative_to(runtime.resolve()):
        raise ValueError("Managed engine path escapes this project's runtime.")
    return path


def active_record(runtime):
    path = runtime / ACTIVE_FILE
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (
        not isinstance(record, dict) or record.get("schema") != 1
        or not isinstance(record.get("entrypoint"), str)
    ):
        raise ValueError("Invalid active engine record; use the updater's rollback command.")
    runtime_path(runtime, record["entrypoint"])
    return record


def managed_engine(runtime):
    if (runtime / RECOVERY_FILE).exists():
        raise RuntimeError("Engine rollback was interrupted. Run ./velocicyber.sh update-engine --rollback.")
    record = active_record(runtime)
    path = runtime_path(runtime, record["entrypoint"]) if record else runtime / "engine/bin/interpreter"
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError("The selected Velocicyber execution engine is missing or not executable.")
    return path


def take_update_notice(runtime):
    """Consume a successful activation notice once; never notify about a staged update."""
    path = runtime / UPDATE_NOTICE_FILE
    try:
        notice = json.loads(path.read_text())
        active = active_record(runtime)
        path.unlink()
        if (
            isinstance(notice, dict) and active
            and notice.get("activation") == active.get("backup")
            and notice.get("entrypoint") == active["entrypoint"]
            and notice.get("tag") == active.get("release", {}).get("tag")
            and isinstance(notice.get("tag"), str)
        ):
            return notice["tag"]
    except (OSError, ValueError, TypeError):
        pass  # An optional notification must never prevent startup.
    return None


def provenance(binary):
    try:
        package = json.loads((binary.parent.parent / "codex-package.json").read_text())
        if package["variant"] == "open-interpreter":
            return "open-interpreter-rust-v" + package["version"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return "custom-engine"


@contextmanager
def engine_lease(runtime, *, maintenance=False):
    """Hold a shared lock for the harness lifetime, exclusive during activation."""
    fd = os.open(runtime / "engine-use.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if maintenance else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Close all Velocicyber harness windows before updating or rolling back."
                if maintenance else "Engine maintenance is running; reopen the harness when it finishes."
            ) from None
        if not maintenance and (runtime / RECOVERY_FILE).exists():
            raise RuntimeError("Finish the interrupted rollback with ./velocicyber.sh update-engine --rollback.")
        yield
