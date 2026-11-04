"""Write completion flags from local report events, without network polling."""

import ctypes
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path


def main():
    runtime = Path(__file__).resolve().parents[2] / ".runtime"
    report = runtime / "live-gpu-test.json"
    flag = runtime / "gpu-notification.json"
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    try:
        # Register before reading so completion cannot race with watch setup.
        if libc.inotify_add_watch(fd, os.fsencode(runtime), 0x8 | 0x80) < 0:
            raise OSError(ctypes.get_errno(), "inotify_add_watch")
        seen = 0
        while True:
            data = json.loads(report.read_text())
            for event in data["stages"][seen:]:
                stage = event["stage"]
                if stage in {
                    "server_started", "inference_ready", "gpu_test_passed",
                    "test_error", "validation_stopped_instance_retained",
                }:
                    payload = {
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        "run_started_at": data["started_at"],
                        "stage": stage,
                        "deployment_completed": any(
                            e["stage"] == "server_started" for e in data["stages"]
                        ),
                        "tests_passed": any(
                            e["stage"] == "gpu_test_passed" for e in data["stages"]
                        ),
                        "error_recorded": any(
                            e["stage"] == "test_error" for e in data["stages"]
                        ),
                        "gpu_lifetime": "until_user_ends_session",
                    }
                    temporary = flag.with_suffix(".tmp")
                    temporary.write_text(json.dumps(payload, indent=2) + "\n")
                    temporary.chmod(0o600)
                    temporary.replace(flag)
                    print(json.dumps(payload), flush=True)
                if stage == "validation_stopped_instance_retained":
                    return
            seen = len(data["stages"])
            # Block on filesystem events; never poll providers or the report.
            while True:
                events = os.read(fd, 65536)
                offset = 0
                changed = False
                while offset < len(events):
                    _, mask, _, length = struct.unpack_from("iIII", events, offset)
                    name = events[offset + 16:offset + 16 + length].rstrip(b"\0")
                    changed |= name == report.name.encode() or bool(mask & 0x4000)
                    offset += 16 + length
                if changed:
                    break
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
