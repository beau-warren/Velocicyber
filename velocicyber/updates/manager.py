"""Explicit approval for the exact harness/engine components shown to the user."""

import asyncio
import subprocess

import httpx

from velocicyber.engine.updater import EngineUpdater
from velocicyber.updates import UpdateChecker
from velocicyber.updates.updater import SourceUpdater


def describe(plan):
    return " + ".join(name for key, name in (("harness", "Velocicyber"), ("engine", "engine"))
                      if key in plan)


def apply_updates(settings, plan, *, approved=False, stage_only=False, notify=print):
    if not approved:
        raise PermissionError("Explicit approval is required before installing updates.")
    if not plan or set(plan) - {"engine", "harness"}:
        raise ValueError("No supported update components were selected.")
    # Each component has its own test gate and rollback. Test source against the
    # installed engine after its approved update; never hot-reload an updater.
    if "engine" in plan:
        notify("Updating engine…" if not stage_only else "Validating engine…")
        EngineUpdater(settings, notify).run(approved=True, action="stage" if stage_only else "update",
                                            tag=plan["engine"]["target"])
    if "harness" in plan:
        notify("Updating Velocicyber…" if not stage_only else "Validating Velocicyber…")
        SourceUpdater(settings, notify).run(plan["harness"], approved=True, stage_only=stage_only)


def interactive_updates(settings, *, plan=None, approved=False, harness_only=False,
                        stage_only=False, rollback_harness=False, input_fn=input):
    def notify(text):
        print(text, flush=True)
    if rollback_harness:
        print("Restore the previous harness source and Python environment. Keep local user data.")
        if not approved and input_fn("Type ROLLBACK to proceed, or Enter to cancel: ").strip() != "ROLLBACK":
            print("Cancelled. No updates started.")
            return
        SourceUpdater(settings, notify).run(approved=True, rollback=True)
        return
    if plan is None:
        checker = UpdateChecker(settings)
        asyncio.run(checker.check())
        print(checker.details())
        plan = checker.plan()
    if harness_only:
        plan = {key: value for key, value in plan.items() if key == "harness"}
    if not plan:
        print("No verified updates are available to install. Resolve any check errors before retrying.")
        return
    print(("Test only: " if stage_only else "Update: ") + describe(plan))
    for component, item in plan.items():
        print(component + " target: " + item["target"])
    if not approved:
        word = "TEST" if stage_only else "UPDATE"
        try:
            approved = input_fn(f"Type {word} to proceed, or Enter to cancel: ").strip() == word
        except EOFError:
            approved = False
    if not approved:
        print("Cancelled. No updates started.")
        return
    try:
        apply_updates(settings, plan, approved=True, stage_only=stage_only, notify=notify)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        message = type(exc).__name__ if isinstance(exc, (httpx.HTTPError, subprocess.SubprocessError)) else str(exc)
        raise RuntimeError("Update stopped: " + message + " Any component already reported as updated remains installed; rollback is available.") from None
