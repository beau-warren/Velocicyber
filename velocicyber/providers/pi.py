"""CPU-only serving on this Raspberry Pi; independent from the Lambda lifecycle."""

import asyncio
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from velocicyber.models.pi_profiles import PI_MODEL_REPO, PI_PROFILES
from velocicyber.sentinel.profiles import local_sentinel_active


def available_memory():
    return next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )


class PiModelManager:
    def __init__(self, settings, state):
        self.settings, self.state = settings, state
        self.process = None
        self.install_process = None
        self.lock = None
        self.watcher = None
        self.repo = settings.pi_model or PI_MODEL_REPO

    @property
    def folder(self):
        path = self.settings.runtime / PI_PROFILES[self.repo].directory
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        return path

    def installed(self):
        marker = self.folder / "installed.json"
        if not marker.exists():
            return None
        cfg = json.loads((self.settings.root / PI_PROFILES[self.repo].lock_file).read_text())
        saved = json.loads(marker.read_text())
        # Serving-only changes reuse verified weights and the pinned executable.
        # Return the current serving config while preserving the install receipt.
        if any(
            saved.get(key) != value
            for key, value in cfg.items()
            if key not in {"server_args", "chat_template"}
        ):
            return None
        model, binary = Path(saved["model_path"]), Path(saved["binary_path"])
        if not all(p.resolve().is_relative_to(self.folder.resolve()) for p in (model, binary)):
            raise ValueError("Pi model installation paths must stay inside this project.")
        if (
            not model.is_file()
            or not binary.is_file()
            or model.stat().st_size != (saved.get("converted_weight_bytes") if cfg.get("conversion") else cfg["weight_bytes"])
            or model.stat().st_mtime_ns != saved["model_mtime_ns"]
        ):
            return None
        if cfg.get("conversion"):
            from velocicyber.models.pi_artifacts import valid_receipt

            if not valid_receipt(cfg, self.folder, saved, verify_model=False):
                return None
        return {**saved, **cfg}

    async def install(self, notify):
        # A CLI/background installation may already own the transfer. Join it
        # instead of starting another download or probing Hugging Face again.
        joined = False
        with (self.folder / "install.lock").open("a") as lock:
            while self.installed() is None:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    joined = True
                    self.state.pi_stage = self.install_status()
                    await asyncio.sleep(5)
                    continue
                fcntl.flock(lock, fcntl.LOCK_UN)
                if joined:
                    raise RuntimeError(
                        f"Background Pi installation stopped; inspect {self.folder / 'install.log'}."
                    )
                break
            else:
                return
        self.state.pi_stage = "Downloading/building"
        self.install_process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self.settings.root / "scripts/install/install_pi_model.py"),
            "--model",
            self.repo,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.settings.root,
            start_new_session=True,
        )
        with (self.folder / "install.log").open("ab") as log:
            async for line in self.install_process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("stage") == "download_progress":
                    percent = event["bytes"] * 100 // event["total"]
                    self.state.pi_stage = f"Downloading {percent}%"
                elif event.get("stage") in {"installing_runtime", "preparing_converter", "converting"}:
                    self.state.pi_stage = event["stage"].replace("_", " ").capitalize()
                elif event.get("part") == "setup" and event.get("stage") == "ready":
                    notify("Raspberry Pi model download and runtime build complete.")
        result = await self.install_process.wait()
        self.install_process = None
        if result:
            raise RuntimeError(
                f"Pi setup stopped. See {self.folder / 'install.log'}; select it again to resume."
            )

    def install_status(self):
        if (self.folder / "installed.json").exists():
            return "Downloaded — select to load"
        build = self.folder / "build.json"
        if build.exists():
            try:
                stage = json.loads(build.read_text()).get("stage")
                if stage == "failed":
                    return "Setup stopped — see installation log"
                if stage in {"preparing_converter", "converting"}:
                    return stage.replace("_", " ").capitalize()
            except (OSError, ValueError):
                pass
        path = self.folder / "download.json"
        if path.exists():
            try:
                progress = json.loads(path.read_text())
                if progress.get("stage") == "failed":
                    return "Download stopped — see installation log"
                if progress.get("stage") == "verifying":
                    return "Verifying downloaded model"
                if progress.get("total"):
                    return f"Downloading {progress.get('bytes', 0) * 100 // progress['total']:.0f}% / building"
            except (OSError, ValueError):
                pass
        return "Not installed"

    def command(self, cfg):
        endpoint = urlsplit(self.state.pi_url)
        if endpoint.hostname != "127.0.0.1" or endpoint.scheme != "http" or not endpoint.port:
            raise ValueError("The Pi CPU server must bind to a loopback HTTP port.")
        template_args = []
        if template := cfg.get("chat_template"):
            path = (self.settings.root / template["path"]).resolve()
            if not path.is_relative_to(self.settings.root.resolve()):
                raise ValueError("Pi chat template must stay inside this project.")
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != template["sha256"]
            ):
                raise ValueError(
                    "Pi chat template is missing or does not match the pinned checksum."
                )
            template_args = ["--chat-template-file", str(path)]
        return (
            [
                sys.executable,
                str(self.settings.root / "scripts/runtime/serve_pi_model.py"),
                str(os.getpid()),
                cfg["binary_path"],
                "-m",
                cfg["model_path"],
                "--host",
                "127.0.0.1",
                "--port",
                str(endpoint.port),
                "--alias",
                self.repo,
                "-ngl",
                "0",
                "-t",
                "3",
                "--threads-batch",
                "4",
                "-c",
                str(cfg["context_window"]),
                "-np",
                "1",
                "-b",
                "256",
                "-ub",
                "128",
                "--jinja",
                "--chat-template-kwargs",
                json.dumps({"enable_thinking": cfg.get("enable_thinking", False)}),
            ]
            + cfg.get("server_args", [])
            + template_args
        )

    async def start(self, notify=lambda message: None):
        if local_sentinel_active(self.state):
            raise RuntimeError("Turn /sentinel OFF before loading a Pi coding model.")
        if self.settings.pi_model not in PI_PROFILES:
            raise ValueError("Select a configured Raspberry Pi model first.")
        if self.repo != self.settings.pi_model:
            notify("Unloading the previous Pi model before loading " + self.settings.pi_model)
            await self.stop()
            self.repo = self.settings.pi_model
        if self.process and self.process.returncode is None and self.state.pi_ready:
            self.state.active_provider = "pi"
            self.state.private_target = "pi"
            return
        if self.process:
            await self.stop()
        try:
            if self.lock is None:
                # Retain the original lock path for compatibility with older harnesses.
                lock_folder = self.settings.runtime / "pi-model"
                lock_folder.mkdir(mode=0o700, exist_ok=True)
                self.lock = (lock_folder / "server.lock").open("a")
                try:
                    fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self.lock.close()
                    self.lock = None
                    raise RuntimeError(
                        "The Pi model is already in use by another harness."
                    ) from None
            cfg = self.installed()
            if cfg is None:
                await self.install(notify)
                cfg = self.installed()
            if cfg is None:
                raise RuntimeError("Pi model installation did not validate.")
            available = available_memory()
            minimum = cfg.get("minimum_available_gib", 13)
            if available < minimum * 1024**3:
                raise RuntimeError(
                    f"The Pi model needs about {minimum} GiB available RAM. "
                    "Close other memory-heavy applications, then select it again."
                )
            self.state.pi_stage = "Loading"
            with (self.folder / "server.log").open("ab") as log:
                self.process = await asyncio.create_subprocess_exec(
                    *self.command(cfg),
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    pass_fds=(self.lock.fileno(),),
                    cwd=self.settings.root,
                )
            deadline = time.monotonic() + 600
            async with httpx.AsyncClient(timeout=10) as client:
                while time.monotonic() < deadline:
                    if self.process.returncode is not None:
                        raise RuntimeError(
                            f"Pi CPU server exited. See {self.folder / 'server.log'}."
                        )
                    try:
                        health = await client.get(self.state.pi_url.removesuffix("/v1") + "/health")
                        if health.status_code == 200:
                            models = await client.get(self.state.pi_url + "/models")
                            models.raise_for_status()
                            if any(m["id"] == self.repo for m in models.json()["data"]):
                                break
                            raise RuntimeError(
                                "A different model is already using the Pi inference port."
                            )
                    except (httpx.ConnectError, httpx.TimeoutException):
                        pass
                    await asyncio.sleep(3)
                else:
                    raise RuntimeError("Pi model loading timed out; inspect the local server log.")
            self.state.pi_ready = True
            self.state.pi_stage = "Ready"
            self.state.pi_revision = cfg["revision"]
            self.state.pi_context_window = cfg["context_window"]
            self.state.private_target = "pi"
            self.watcher = asyncio.create_task(self.watch_exit(notify))
            notify(
                "Raspberry Pi CPU model ready."
                + (
                    " Inference now runs on this Pi."
                    if self.state.active_provider == "pi"
                    else " Select it under Local Model Selection to switch from the API."
                )
            )
        except BaseException:
            await self.stop()
            self.state.pi_stage = "Stopped — see output"
            raise

    async def watch_exit(self, notify):
        await self.process.wait()
        self.state.pi_ready = False
        self.state.pi_stage = "Stopped"
        # Keep the selected provider: never silently send a Pi prompt to an API.
        notify("Raspberry Pi model stopped. Select it again to restart, or choose another model.")

    async def stop(self):
        if self.watcher:
            self.watcher.cancel()
            await asyncio.gather(self.watcher, return_exceptions=True)
            self.watcher = None
        for process in (self.process, self.install_process):
            if process and process.returncode is None:
                if process is self.install_process:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 15)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        self.process = self.install_process = None
        self.state.pi_ready = False
        self.state.pi_stage = ""
        if self.lock:
            self.lock.close()
            self.lock = None
