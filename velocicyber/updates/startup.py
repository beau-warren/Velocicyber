"""Opt-in source updates before the UI starts; engine upgrades stay manual."""

import asyncio
import json
import os
import subprocess
import sys
import time

import httpx

from velocicyber.engine.sessions import private_write
from velocicyber.onboarding import private_assignments
from velocicyber.updates.source import SourceChecker, installed_source
from velocicyber.updates.updater import PENDING, SourceUpdater

HANDOFF = "VELOCICYBER_STARTUP_UPDATE_HANDOFF"


def automatic_updates(settings):
    return settings.options.get("AUTO_UPDATE_HARNESS", "OFF").strip().upper() == "ON"


def set_automatic_updates(settings, enabled):
    value = "ON" if enabled else "OFF"
    private_assignments(settings.root / "Modules.txt", {"AUTO_UPDATE_HARNESS": value})
    settings.options["AUTO_UPDATE_HARNESS"] = value


def take_handoff(checker):
    """Consume the restart marker once, bound to this root and freshly checked commit."""
    raw = os.environ.pop(HANDOFF, "")
    if not raw:
        return False
    try:
        marker = json.loads(raw)
        result = checker.result
        valid = (marker["root"] == str(checker.settings.root.resolve())
                 and marker["commit"] == checker.installed == result.get("latest")
                 and marker["checked_at"] == result.get("checked_at")
                 and 0 <= time.time() - marker["checked_at"] < 300
                 and result.get("status") == "identical" and not result.get("error")
                 and not checker.config_error)
    except (ValueError, TypeError, KeyError):
        return False
    if valid:
        checker.checked_this_launch = True
    return valid


def restart(settings, checker):
    """Load the activated source and Python environment, retaining the original CLI flags."""
    environment = dict(os.environ)
    environment[HANDOFF] = json.dumps({"root": str(settings.root.resolve()),
                                      "commit": checker.installed,
                                      "checked_at": checker.result["checked_at"]})
    python = settings.root / ".venv/bin/python"
    executable = str(python) if python.is_file() else sys.executable
    os.execve(executable, [executable, "-m", "velocicyber", *sys.argv[1:]], environment)


def startup_updates(settings, notify=print):
    """Return a reusable checker, or None when the ordinary UI check should handle it."""
    if (settings.runtime / PENDING).exists():
        raise RuntimeError("Interrupted harness update. Run ./velocicyber.sh update --rollback-harness first.")
    if not automatic_updates(settings) and HANDOFF not in os.environ:
        return None
    checker = SourceChecker(settings)
    if take_handoff(checker):
        checker.startup_notice = "Velocicyber updated and restarted successfully. Engine upgrades remain manual."
        return checker
    if not automatic_updates(settings):
        return None
    notify("Checking for Velocicyber repo updates before starting…")
    asyncio.run(checker.check())
    # A cached UPDATE AVAILABLE is useful in the UI, but cannot authorize an
    # automatic install when today's network check failed.
    if (checker.config_error or checker.result.get("error")
            or checker.result.get("status") != "behind"):
        return checker
    previous, target = checker.installed, checker.result["latest"]
    try:
        SourceUpdater(settings, notify).run({"installed": previous, "target": target}, approved=True)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, httpx.HTTPError) as exc:
        if (settings.runtime / PENDING).exists():
            raise RuntimeError("Harness update needs recovery. Run ./velocicyber.sh update --rollback-harness.") from exc
        if installed_source(settings.root)[1] != previous:
            raise RuntimeError("The source changed during maintenance. Close and reopen the harness.") from exc
        detail = ("GitHub download failed (" + type(exc).__name__ + "). Retry on next launch."
                  if isinstance(exc, httpx.HTTPError) else str(exc))
        checker.startup_notice = "Automatic repo update deferred: " + detail
        notify(checker.startup_notice)
        return checker
    checker = SourceChecker(settings)
    if checker.installed != target:
        raise RuntimeError("Updated source could not be verified. Close and reopen the harness.")
    checker.result = {"schema": 1, "installed": target, "latest": target,
                      "status": "identical", "error": "", "checked_at": time.time()}
    private_write(checker.path, json.dumps(checker.result) + "\n")
    notify("Restarting Velocicyber with the tested update…")
    try:
        restart(settings, checker)
    except OSError as exc:
        raise RuntimeError("Update installed. Reopen the harness to use the new version.") from exc
    # execve never returns; a failed or substituted restart must not launch a UI
    # with a mixture of old imports and new source.
    raise RuntimeError("Update installed. Reopen the harness to use the new version.")
