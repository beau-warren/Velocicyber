"""Wait for installation, then run one local coding check without uploads or GPUs."""

import argparse
import asyncio
import fcntl
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velocicyber.config import Settings  # noqa: E402
from velocicyber.engine.runner import Engine  # noqa: E402
from velocicyber.models.pi_profiles import PI_PROFILES  # noqa: E402
from velocicyber.providers.pi import PiModelManager  # noqa: E402
from velocicyber.state import State  # noqa: E402


async def validate(model=None):
    os.umask(0o077)
    settings = Settings.load(ROOT)
    if model:
        settings.options["PI_LOCAL_MODEL"] = model
    state = State(active_provider="pi", collection=False, upload_approved=False)
    manager = PiModelManager(settings, state)
    folder = manager.folder

    def report(stage, **details):
        result = {"stage": stage, "updated_at": time.time(), **details}
        tmp = folder / "validation.tmp"
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(folder / "validation.json")
        print(json.dumps(result), flush=True)

    with (folder / "validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = None
        try:
            report("waiting_for_installation")
            deadline = time.monotonic() + 24 * 3600
            while manager.installed() is None:
                with (folder / "install.lock").open("a") as installer:
                    try:
                        fcntl.flock(installer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        if manager.installed() is None:
                            raise RuntimeError(
                                "Installation stopped before completion; see installation log."
                            )
                if time.monotonic() > deadline:
                    raise TimeoutError("Installation still incomplete after 24 hours.")
                await asyncio.sleep(60)
            report("loading")
            started = time.monotonic()
            await manager.start(print)
            load_seconds = round(time.monotonic() - started, 1)
            workspace = folder / "validation-workspaces" / str(time.time_ns())
            engine = Engine(settings, state, workspace=workspace)
            engine.verifier = [
                sys.executable,
                "-c",
                "from pathlib import Path; assert Path('result.txt').read_text().strip() == '42'",
            ]
            report("coding_check", load_seconds=load_seconds, workspace=str(workspace))
            events = []
            started = time.monotonic()
            await asyncio.wait_for(
                engine.run(
                    "Use your tools to create result.txt in the current workspace containing "
                    "the result of 6 * 7. Read it back to verify it, then report completion briefly.",
                    events.append,
                ),
                timeout=1800,
            )
            if not any(e.get("type") == "turn.completed" for e in events) or any(
                e.get("type") in {"turn.failed", "engine.error"} for e in events
            ):
                raise RuntimeError("The coding turn did not complete successfully.")
            if (workspace / "result.txt").read_text().strip() != "42":
                raise RuntimeError("Coding check produced the wrong result.")
            report(
                "passed",
                load_seconds=load_seconds,
                coding_seconds=round(time.monotonic() - started, 1),
                workspace=str(workspace),
                event_types=sorted({event["type"] for event in events}),
            )
        except asyncio.CancelledError:
            report("interrupted")
            raise
        except Exception as exc:
            report("failed", error=str(exc))
        finally:
            if engine:
                await engine.cancel()
                if engine.process and engine.process.returncode is None:
                    try:
                        await asyncio.wait_for(engine.process.wait(), 10)
                    except TimeoutError:
                        engine.process.kill()
                        await engine.process.wait()
                engine.close()
            await manager.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=PI_PROFILES)
    asyncio.run(validate(parser.parse_args().model))
