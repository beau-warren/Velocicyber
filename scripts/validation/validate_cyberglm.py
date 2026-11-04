"""Explicit live CyberGLM tool test through Engine. No GPU calls or dataset uploads."""

import asyncio
import json
import sys
import time
from pathlib import Path

from velocicyber.config import Settings
from velocicyber.engine.runner import Engine
from velocicyber.state import State


async def main():
    settings = Settings.load(Path(__file__).resolve().parents[2])
    settings.api_model = "lordx64/cyberglm"
    folder = settings.runtime / "validation" / ("cyberglm-" + str(time.time_ns()))
    folder.mkdir(parents=True)
    (folder / "AGENTS.md").write_text("Only perform the user's small validation task here.\n")
    engine = Engine(settings, State(collection=False, upload_approved=False), folder)
    engine.verifier = [sys.executable, "-c",
        'from pathlib import Path; assert Path("check.txt").read_text() == "CYBERGLM_OK\\n"']
    report = {"model": settings.api_model, "workspace": str(folder), "events": []}
    start = time.monotonic()

    def event(e):
        if e.get("type") in {"engine.error", "turn.failed", "turn.completed", "item.completed"}:
            value = engine.scrubber.clean(e)
            report["events"].append(value)
            print(json.dumps(value), flush=True)

    try:
        await asyncio.wait_for(engine.run(
            "In this current validation directory only, create check.txt containing exactly "
            "CYBERGLM_OK followed by a newline. Read it back using a tool and finish with "
            "a brief confirmation. This is the entire task; ignore project development milestones.",
            event), 180)
        report["file_verified"] = (folder / "check.txt").read_text() == "CYBERGLM_OK\n"
        report["session_status"] = engine.session.metadata["status"]
        report["passed"] = report["file_verified"] and report["session_status"] == "complete"
    except Exception as exc:
        report.update(passed=False, error=engine.scrubber.text(str(exc)))
    finally:
        await engine.cancel()
        report["api_health"] = engine.state.api_health
        report["seconds"] = round(time.monotonic() - start, 2)
        engine.close()
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "events"}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
