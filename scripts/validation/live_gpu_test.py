"""Live validation. GPU lifetime follows explicit user session termination.

Uses the cached, pinned model inspection and a freshly checked capacity catalog.
Publishes no data: derived snapshots are retained for review and an approved upload.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

from velocicyber.config import Settings
from velocicyber.engine.runner import Engine
from velocicyber.providers.lambda_cloud import GPUManager, LambdaClient, LaunchPlan, ModelSpec
from velocicyber.state import State


async def main(args):
    settings = Settings.load(Path(__file__).resolve().parents[2])
    if args.model not in settings.gpu_model_choices:
        raise ValueError("Choose a configured GPU model repository from /local & cloud models.")
    settings.local_model = args.model
    if args.region:
        settings.options["LAMBDA_REGION"] = args.region
    state = State.restore(settings.runtime / "gpu.json")
    if (state.instance_id or state.launch_uncertain) and not args.resume:
        raise RuntimeError("Existing GPU state requires reconciliation before a live test.")
    if args.resume and not state.instance_id:
        raise RuntimeError("No tracked instance to resume.")
    client = LambdaClient(settings.require_secret("LAMBDA_API"))
    manager = GPUManager(settings, state, client)
    row = next(
        row
        for row in json.loads((settings.runtime / "model-preflight.json").read_text())["models"]
        if row["repo"] == settings.local_model
    )
    manager.spec = ModelSpec(**{field.name: row[field.name] for field in fields(ModelSpec)})
    report_path = settings.runtime / "live-gpu-test.json"
    previous = json.loads(report_path.read_text()) if args.resume else None
    if previous:
        details = previous["plan"].copy()
        details["model"] = ModelSpec(**details["model"])
        plan = LaunchPlan(**details)
    else:
        plan = manager.plan(40)
    started = time.monotonic()
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "plan": asdict(plan),
        "stages": [],
        "artifacts": [],
        "episodes": [],
    }
    if previous:
        report = previous
        started -= time.time() - datetime.fromisoformat(report["started_at"]).timestamp()
    elif report_path.exists():
        archive = settings.runtime / "test-history"
        archive.mkdir(mode=0o700, exist_ok=True)
        report_path.replace(archive / (str(time.time_ns()) + "-live-gpu-test.json"))
    engine = None
    report["runner_pid"] = os.getpid()
    attempt_start = len(report["stages"])
    report["attempt_started_at"] = datetime.now(timezone.utc).isoformat()

    def save_json(path, value):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(path)

    def record(stage, **data):
        event = {"stage": stage, "elapsed_s": round(time.monotonic() - started, 1), **data}
        report["stages"].append(event)
        save_json(report_path, report)
        stages = {event["stage"] for event in report["stages"][attempt_start:]}
        save_json(
            settings.runtime / "gpu-notification.json",
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "run_started_at": report["started_at"],
                "attempt_started_at": report["attempt_started_at"],
                "stage": stage,
                "instance_id": state.instance_id,
                "deployment_completed": bool(stages & {"server_started", "server_reused"}),
                "inference_ready": "inference_ready" in stages,
                "tests_passed": "gpu_test_passed" in stages,
                "error_recorded": "test_error" in stages,
                "gpu_lifetime": "until_user_ends_session",
            },
        )
        print(json.dumps(event), flush=True)

    async def remote(program):
        process = await asyncio.create_subprocess_exec(
            *manager.ssh_args(),
            "ubuntu@" + state.instance_ip,
            "python3 -",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            output, error = await asyncio.wait_for(process.communicate(program.encode()), 60)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        if process.returncode:
            raise RuntimeError("SSH diagnostic failed: " + error.decode(errors="replace")[-1200:])
        return output.decode()

    async def workflow():
        nonlocal engine
        if not args.resume:
            record("launching", hourly_usd=plan.hourly_cents / 100, region=plan.region)
            await asyncio.to_thread(manager.launch, plan, approved=True)
            record("launched", instance_id=state.instance_id)
        else:
            record("resumed", instance_id=state.instance_id, lifetime="until_user_ends_session")
        for _ in range(30):
            await asyncio.sleep(20)
            status = await asyncio.to_thread(manager.refresh)
            record("instance_status", status=status)
            if status == "active":
                break
            if status in ("terminated", "unhealthy"):
                raise RuntimeError("Instance became " + status)
        else:
            raise RuntimeError("Instance did not become active within ten minutes.")
        diagnostics = await remote("""import json, shutil, subprocess
result = {"docker": shutil.which("docker"), "nvidia_ctk": shutil.which("nvidia-ctk")}
for name, command in [("gpu", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]), ("docker_runtime", ["sudo", "-n", "docker", "info", "--format", "{{json .Runtimes}}"] )]:
    try:
        value = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if name == "docker_runtime" and value.returncode == 0:
            value.stdout = json.dumps(sorted(json.loads(value.stdout)))
        result[name] = {"returncode": value.returncode, "stdout": value.stdout, "stderr": value.stderr[-500:]}
    except Exception as exc:
        result[name] = {"error": type(exc).__name__}
print(json.dumps(result))
""")
        record("remote_diagnostics", result=json.loads(diagnostics))
        if not json.loads(diagnostics)["docker"]:
            raise RuntimeError("Lambda image needs Docker installation before deployment.")
        if args.reuse_server:
            await manager.connect()
            record("server_reused")
        else:
            record("deploying")
            await manager.deploy()
            record("server_started")
        for _ in range(30):
            if await manager.check_ready():
                record("inference_ready")
                break
            await asyncio.sleep(15)
        else:
            logs = await remote("""import subprocess
p = subprocess.run(["sudo", "-n", "docker", "logs", "--tail", "60", "velocicyber-model"], capture_output=True, text=True)
print(p.stdout + p.stderr)
""")
            record("server_logs", text=logs[-12000:])
            raise RuntimeError("Model readiness timed out.")
        workspace = settings.runtime / "live-test-workspace"
        workspace.mkdir(mode=0o700, exist_ok=True)
        if (workspace / "solution.py").exists():
            raise RuntimeError(
                "Live-test workspace already contains a solution; choose a fresh task."
            )
        (workspace / "verify.py").write_text("""import unittest
from solution import parse_ports
class Checks(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(parse_ports("80,443,80,22"), [22, 80, 443])
        self.assertEqual(parse_ports(" 8080, 53 "), [53, 8080])
        self.assertEqual(parse_ports(""), [])
    def test_invalid(self):
        for item in ["0", "65536", "abc", "80,,443", "-1", "1.5"]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse_ports(item)
unittest.main()
""")
        state.collection = True
        engine = Engine(settings, state, workspace)
        engine.new_task("live_smoke_parse_ports_v1")
        engine.verifier = [sys.executable, "verify.py"]

        def event(value):
            kind = value.get("type")
            if kind == "dataset.prepared":
                report["artifacts"].append(value["path"])
                report["episodes"].append(value["episode_id"])
                record("dataset_prepared", path=value["path"])
            elif kind in (
                "thread.started",
                "turn.completed",
                "turn.failed",
                "error",
                "engine.error",
            ):
                record("engine_event", event=engine.scrubber.clean(value))
            elif kind == "item.completed":
                item = value.get("item", {})
                record("engine_item", item_type=item.get("type"), exit_code=item.get("exit_code"))

        prompt = "Implement solution.py with parse_ports(text): parse comma-separated decimal TCP port numbers, return sorted unique integers. Allow whitespace. Empty or whitespace-only input gives []. Reject empty components, non-integers, and ports outside 1..65535 with ValueError. Inspect verify.py, implement the function, and run python3 verify.py. Only work in the current directory; do not change verify.py or read credentials. Keep the final response brief."
        record("coding_tests_started")
        path = await asyncio.wait_for(engine.run(prompt, event), 300)
        episode = json.loads(path.read_text())
        record(
            "coding_result",
            outcome=episode["outcome"],
            tool_events=len(episode["tool_events"]),
            context_complete=episode["context_complete"],
        )
        if not episode["outcome"]["success"] or not episode["tool_events"]:
            raise RuntimeError("Live coding task failed verification or did not execute tools.")
        followup = "Add test_extra.py with standard-library unittest tests for whitespace-only input, boundary ports 1 and 65535, and duplicate ports. Run python3 test_extra.py and python3 verify.py. Do not modify verify.py. Keep the final response brief."
        path = await asyncio.wait_for(engine.run(followup, event), 300)
        episode = json.loads(path.read_text())
        record(
            "continuation_result",
            outcome=episode["outcome"],
            tool_events=len(episode["tool_events"]),
            context_complete=episode["context_complete"],
        )
        if not episode["outcome"]["success"] or not (workspace / "test_extra.py").exists():
            raise RuntimeError("Continuation task failed verification.")
        record("gpu_test_passed")

    try:
        await workflow()
    except BaseException as exc:
        record("test_error", error=type(exc).__name__, message=str(exc))
        raise
    finally:
        if engine:
            await engine.cancel()
            engine.close()
        await manager.close_tunnel()
        record(
            "validation_stopped_instance_retained",
            instance_id=state.instance_id,
            hourly_usd=plan.hourly_cents / 100,
            estimated_cost_so_far_usd=round(
                (time.monotonic() - started) / 3600 * plan.hourly_cents / 100, 3
            ),
        )
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="GPU model repository to test")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue on the tracked GPU")
    parser.add_argument(
        "--reuse-server",
        action="store_true",
        help="Reconnect to an existing server without redeploying",
    )
    parser.add_argument("--region", help="Optional test-only region; Modules.txt remains unchanged")
    args = parser.parse_args()
    if not args.approved:
        parser.error("Explicit user approval is required. Use --approved only after approval.")
    if args.reuse_server and not args.resume:
        parser.error("--reuse-server requires --resume on an existing tracked GPU.")
    asyncio.run(main(args))
