"""One explicit live API tool/capture test; no uploads or GPUs."""

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
    workspace = settings.runtime / "live-api-test-workspace-native-v1"
    workspace.mkdir(mode=0o700, exist_ok=True)
    if (workspace / "api_check.txt").exists():
        raise RuntimeError("This test has already executed; use its saved report.")
    state = State(collection=True)
    engine = Engine(settings, state, workspace)
    engine.new_task("live_smoke_api_file_native_v1")
    engine.verifier = [
        sys.executable,
        "-c",
        'from pathlib import Path; assert Path("api_check.txt").read_text() == "API_TOOLS_OK\\n"',
    ]
    report = {"started_at_unix": time.time(), "events": []}

    def event(value):
        if value.get("type") in (
            "turn.completed",
            "turn.failed",
            "dataset.prepared",
            "error",
            "engine.error",
        ):
            report["events"].append(engine.scrubber.clean(value))
            print(json.dumps(report["events"][-1]), flush=True)

    try:
        raw_path = await asyncio.wait_for(
            engine.run(
                "Use the shell tool to create api_check.txt containing exactly API_TOOLS_OK followed by a newline, and read it back in the same tool call. Only work in the current directory. Finish with a brief confirmation.",
                event,
            ),
            180,
        )
        episode = json.loads(raw_path.read_text())
        report.update(
            raw_path=str(raw_path),
            outcome=episode["outcome"],
            tool_events=len(episode["tool_events"]),
            context_complete=episode["context_complete"],
            api_health=state.api_health,
        )
        if not episode["outcome"]["success"] or not episode["tool_events"]:
            raise RuntimeError("Live API tool task failed verification.")
        report["passed"] = True
    except Exception as exc:
        report.update(
            passed=False, error=type(exc).__name__, message=engine.scrubber.text(str(exc))
        )
        raise
    finally:
        await engine.cancel()
        engine.close()
        target = settings.runtime / "live-api-test-native-v1.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        target.chmod(0o600)
        print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
