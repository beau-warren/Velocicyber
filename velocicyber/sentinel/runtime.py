"""Local Sentinel servers, independent of the coding engine and GPU lifecycle."""

import asyncio
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

from velocicyber.engine.sessions import private_write
from velocicyber.providers.pi import PiModelManager, available_memory
from velocicyber.sentinel.profiles import sentinel_profile

MODEL = "ibm-granite/granite-guardian-4.1-8b"
LABEL = "Granite Guardian 4.1 8B Q4"
URL = "http://127.0.0.1:18002"


class SentinelRuntime:
    def __init__(self, settings, state):
        self.settings, self.state = settings, state
        self.process = self.install_process = self.lock = self.watcher = None
        self.running_model = None

    @property
    def profile(self):
        return sentinel_profile(self.state)

    @property
    def folder(self):
        folder = self.settings.runtime / "sentinel-observer"
        if folder.is_symlink():
            raise ValueError("Sentinel runtime must not be a symlink.")
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        return folder

    def installed(self):
        if self.state.sentinel_model == "qwen":
            manager = PiModelManager(self.settings, self.state)
            manager.repo = self.profile.model
            cfg = manager.installed()
            if cfg is None:
                return None
            # Share the existing verified download; never copy or redownload it.
            binary_hash = hashlib.sha256(Path(cfg["binary_path"]).read_bytes()).hexdigest()
            return {**cfg, "minimum_available_gib": 13, "context_window": 4096,
                    "binary_sha256": binary_hash,
                    "runtime": {"repo": cfg["runtime_repo"], "revision": cfg["runtime_revision"]}}
        marker = self.folder / "installed.json"
        if not marker.exists():
            return None
        cfg = json.loads((self.settings.root / "config/locks/sentinel-model.lock.json").read_text())
        saved = json.loads(marker.read_text())
        if any(saved.get(k) != v for k, v in cfg.items()):
            return None
        if not saved.get("runtime_files"):
            return None
        model = Path(saved["model_path"])
        binary = Path(saved["binary_path"])
        if (not model.resolve().is_relative_to(self.folder.resolve())
                or not binary.resolve().is_relative_to(self.settings.runtime.resolve())):
            raise ValueError("Sentinel installation escapes the project.")
        if (not model.is_file() or not binary.is_file()
                or model.stat().st_size != cfg["weight_bytes"]
                or model.stat().st_mtime_ns != saved["model_mtime_ns"]):
            return None
        with binary.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != saved["binary_sha256"]:
                raise ValueError("The sentinel runtime changed; reinstall its receipt.")
        for name, expected in saved["runtime_files"].items():
            path = self.folder / name
            if not path.resolve().is_relative_to(self.folder.resolve()) or not path.is_file():
                raise ValueError("Sentinel runtime dependency is missing or outside its installation.")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("Sentinel runtime dependency changed; reinstall it before use.")
        return saved

    def command(self, cfg):
        command = [
            sys.executable, str(self.settings.root / "scripts/runtime/serve_pi_model.py"),
            str(os.getpid()), cfg["binary_path"], "-m", cfg["model_path"],
            "--host", "127.0.0.1", "--port", "18002", "--alias", self.profile.model,
            "-ngl", "0", "-t", "3", "--threads-batch", "4", "-c", "4096",
            "-np", "1", "-b", "256", "-ub", "128", "--cache-ram", "0",
        ]
        if self.state.sentinel_model == "qwen":
            command += ["--jinja", "--chat-template-kwargs", '{"enable_thinking":false}',
                        "--ctx-checkpoints", "4", "--ctx-checkpoints-interval", "128"]
        return command

    async def start(self, notify=lambda message: None):
        if self.state.active_provider == "pi" or self.state.pi_ready:
            raise RuntimeError("Stop the Pi coding model and select an API or Lambda model first.")
        if self.process and self.process.returncode is None:
            if self.running_model == self.state.sentinel_model:
                return
            await self.stop()
        try:
            folder = self.settings.runtime / "pi-model"
            folder.mkdir(parents=True, exist_ok=True)
            self.lock = (folder / "server.lock").open("a")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("The Pi is already serving a coding or sentinel model.") from None
            cfg = self.installed()
            if cfg is None:
                if self.state.sentinel_model == "qwen":
                    raise RuntimeError("The existing Qwen Pi installation did not validate. "
                                       "Install or repair it via /local before using it as Sentinel.")
                self.state.sentinel_stage = "DOWNLOADING"
                self.install_process = await asyncio.create_subprocess_exec(
                    sys.executable, str(self.settings.root / "scripts/install/install_sentinel.py"),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                    cwd=self.settings.root, start_new_session=True,
                )
                with (self.folder / "install.log").open("ab") as log:
                    async for line in self.install_process.stdout:
                        log.write(line)
                        log.flush()
                        try:
                            event = json.loads(line)
                            if event.get("total"):
                                self.state.sentinel_stage = (
                                    f"DOWNLOADING {event.get('bytes', 0) * 100 // event['total']}%"
                                )
                        except (ValueError, TypeError):
                            pass
                code = await self.install_process.wait()
                self.install_process = None
                if code:
                    raise RuntimeError(f"Sentinel download stopped; see {self.folder / 'install.log'}.")
                cfg = self.installed()
            if cfg is None:
                raise RuntimeError("Sentinel installation did not validate.")
            if available_memory() < cfg["minimum_available_gib"] * 1024**3:
                raise RuntimeError(f"{self.profile.label} needs at least "
                                   f"{cfg['minimum_available_gib']} GiB available RAM. "
                                   "Stop the Pi coding model.")
            self.state.sentinel_stage = "LOADING"
            with (self.folder / "server.log").open("ab") as log:
                self.process = await asyncio.create_subprocess_exec(
                    *self.command(cfg),
                    stdout=log, stderr=asyncio.subprocess.STDOUT,
                    pass_fds=(self.lock.fileno(),), cwd=self.settings.root,
                )
            self.running_model = self.state.sentinel_model
            deadline = time.monotonic() + 300
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if self.process.returncode is not None:
                        raise RuntimeError("Sentinel server exited; inspect sentinel-observer/server.log.")
                    try:
                        response = await client.get(URL + "/v1/models")
                        if response.status_code == 200:
                            ids = [m["id"] for m in response.json()["data"]]
                            if self.profile.model not in ids:
                                raise RuntimeError("Another model is using the sentinel port.")
                            break
                    except (httpx.ConnectError, httpx.TimeoutException):
                        pass
                    await asyncio.sleep(2)
                else:
                    raise RuntimeError("Sentinel loading timed out.")
            from velocicyber.sentinel.observer import validate_guardian

            self.state.sentinel_stage = "CHECKING"
            checks = await validate_guardian(model=self.state.sentinel_model)
            private_write(self.folder / "self-check.json", json.dumps({
                "checked_at": time.time(), "model": self.profile.model,
                "runtime": cfg["runtime"], "checks": checks,
            }, indent=2) + "\n")
            private_write(self.folder / "active-model.json", json.dumps({
                "model": self.profile.model, "artifact": {k: cfg[k] for k in (
                    "repo", "revision", "filename", "sha256", "binary_sha256", "runtime"
                ) if k in cfg},
            }, indent=2) + "\n")
            self.state.sentinel_enabled = self.state.sentinel_ready = True
            self.state.sentinel_stage = "READY"
            self.watcher = asyncio.create_task(self.watch_exit(notify))
        except BaseException:
            await self.stop()
            raise

    async def watch_exit(self, notify):
        await self.process.wait()
        self.state.sentinel_ready = False
        self.state.sentinel_stage = "ERROR"
        notify("Sentinel server stopped. Restart it with /sentinel; monitored tasks will stop.")

    async def stop(self):
        if self.watcher:
            self.watcher.cancel()
            await asyncio.gather(self.watcher, return_exceptions=True)
            self.watcher = None
        for process in (self.install_process, self.process):
            if process and process.returncode is None:
                if process is self.install_process:
                    os.killpg(process.pid, 15)
                else:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        self.install_process = self.process = None
        self.running_model = None
        if self.lock:
            self.lock.close()
            self.lock = None
        self.state.sentinel_enabled = self.state.sentinel_ready = False
        self.state.sentinel_stage = "OFF"
