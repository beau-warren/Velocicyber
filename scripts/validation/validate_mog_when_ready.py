"""Wait without an LLM, then run one isolated M.O.G. coding test on the existing GPU."""

import asyncio
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import httpx

from velocicyber.config import Settings
from velocicyber.datasets.pipeline import Scrubber
from velocicyber.engine.capture import utcnow
from velocicyber.engine.installation import engine_lease
from velocicyber.engine.sessions import private_write
from velocicyber.models.gpu_profiles import MOG_SEC_REPO
from velocicyber.state import State


def user_task_running(runtime):
    for path in (runtime / "conversations").glob("*/session.json"):
        try:
            data = json.loads(path.read_text())
            if data.get("status") != "running" or data.get("provider") != "local":
                continue
            with (path.parent / "session.lock").open("r") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
        except (OSError, ValueError):
            continue
    return False


async def wait_for_model(settings, instance_id, client, record, *, interval=60, timeout=4 * 3600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = State.restore(settings.runtime / "gpu.json")
        if state.instance_id != instance_id:
            raise RuntimeError("The tracked instance changed or was terminated; test canceled.")
        ready = False
        try:
            response = await client.get(state.local_url + "/models")
            response.raise_for_status()
            ready = any(row.get("id") == MOG_SEC_REPO for row in response.json()["data"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            pass
        if ready:
            if not user_task_running(settings.runtime):
                return
            record("waiting_for_user_task")
        else:
            record("waiting_for_model")
        await asyncio.sleep(interval)
    raise TimeoutError("Model was not ready within four hours; no test or GPU termination performed.")


async def run(settings, record):
    instance_id = State.restore(settings.runtime / "gpu.json").instance_id
    if not instance_id:
        raise RuntimeError("No existing GPU is tracked. This watcher never launches instances.")
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        await wait_for_model(settings, instance_id, client, record)
    result_path = settings.runtime / "diagnostics/mog-coding-smoke-result.json"
    result_path.unlink(missing_ok=True)
    record("testing")
    # Engine maintenance remains possible throughout the download wait.
    with engine_lease(settings.runtime):
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(settings.runtime / "diagnostics/mog-coding-smoke.py"),
            cwd=settings.root,
            env={**os.environ, "PYTHONPATH": str(settings.root)},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 360)
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    if process.returncode or not result.get("passed"):
        detail = result.get("error") or (stdout + stderr).decode(errors="replace")[-2000:]
        raise RuntimeError("Coding test failed: " + (detail or "no result recorded"))
    record("passed", elapsed_s=result.get("elapsed_s"), result=str(result_path))


def main():
    settings = Settings.load(Path(__file__).resolve().parents[2])
    folder = settings.runtime / "diagnostics"
    folder.mkdir(mode=0o700, exist_ok=True)
    scrub = Scrubber(settings.secrets.values())
    report = {"pid": os.getpid(), "model": MOG_SEC_REPO, "started_at": utcnow(),
              "collection": False, "uploads": False, "automatic_retries": False}

    def record(status, **details):
        if report.get("status") == status and not details:
            return
        report.update(status=status, updated_at=utcnow(), **details)
        private_write(folder / "mog-autotest.json", json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)

    with (folder / "mog-autotest.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("M.O.G. automatic validation is already running.", flush=True)
            return
        record("starting")
        try:
            asyncio.run(run(settings, record))
        except Exception as exc:
            record("failed", error=scrub.text(str(exc)))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
