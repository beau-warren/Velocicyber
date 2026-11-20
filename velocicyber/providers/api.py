"""Managed, loopback-only access to Adverserial's confidential SDK gateway."""

import json
import os
import selectors
import signal
import subprocess
from contextlib import contextmanager
from urllib.parse import urlsplit

from velocicyber.providers.transport import validate_inference_url


class ConfidentialGateway:
    def __init__(self, settings):
        self.settings = settings
        self.process = None
        self.url = ""

    def start(self):
        if self.process and self.process.poll() is None:
            return self.url
        folder = self.settings.runtime / "adverserial-confidential"
        python = folder / "venv/bin/python"
        if not python.is_file() or not (folder / "source.json").is_file():
            raise RuntimeError(
                "Adverserial now requires confidential inference. Run "
                ".venv/bin/python scripts/install/install_api_gateway.py first."
            )
        installed = json.loads((folder / "source.json").read_text())
        required = json.loads((self.settings.root / "config/locks/api-client.lock.json").read_text())
        if installed.get("commit") != required["commit"]:
            raise RuntimeError("Refresh the Adverserial confidential API client through /setup before using API models.")
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
        # No billing key in argv, the environment, or the gateway's logs.
        with (folder / "gateway.log").open("ab") as log:
            self.process = subprocess.Popen(
                [str(python), str(self.settings.root / "scripts/runtime/api_gateway.py"), str(folder)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log,
                env=env, cwd=self.settings.root, start_new_session=True,
            )
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                if not selector.select(30):
                    raise RuntimeError("Confidential gateway startup timed out.")
                result = json.loads(self.process.stdout.readline())
            if result.get("error"):
                raise RuntimeError(result["error"])
            port = result["port"]
            if type(port) is not int or not 0 < port < 65536:
                raise ValueError("Invalid confidential gateway port.")
            self.url = f"http://127.0.0.1:{port}/v1"
            return self.url
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
            self.process.stdout.close()
            self.process = None


def needs_confidential_gateway(url):
    return urlsplit(url).hostname == "api.adverserial.ai"


@contextmanager
def api_connection(settings):
    url = validate_inference_url(settings.api_url)
    gateway = None
    try:
        if needs_confidential_gateway(url):
            gateway = ConfidentialGateway(settings)
            yield validate_inference_url(gateway.start())
        else:
            yield url
    finally:
        if gateway:
            gateway.stop()
