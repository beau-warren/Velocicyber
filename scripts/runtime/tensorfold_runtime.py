"""TensorFold helpers appended to serve_model.py and run only on the GPU host.

One managed container owns two supervised ranks. The public API stays behind
the same localhost SSH tunnel as the other backends. No automatic model downloads
are permitted by the inference process.
"""

import hashlib
import json
import re
import subprocess
import tempfile
import urllib.request
import uuid
from pathlib import Path

TENSORFOLD_RUNNER = r'''
import os
import signal
import subprocess
import sys
import time

def run():
    repo, snapshot, context = sys.argv[1:]
    common = [
        "tensorfold", "serve", snapshot, "--backend", "cuda", "--name", repo,
        "--tp", "2", "--master", "127.0.0.1", "--master-port", "29551",
        "--host", "0.0.0.0", "--port", "8000", "--context", context,
        "--parallel", "1", "--max-tokens", "4096", "--reasoning-effort", "high",
        "--no-drafts", "--drafter", "none", "--no-update-check",
    ]
    stopping = False
    children = []

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for rank in (1, 0):
            if stopping:
                break
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), TF_GLM_MTP="0")
            child = subprocess.Popen(
                common + ["--rank", str(rank)], env=env, start_new_session=True,
            )
            children.append((rank, child))
        while not stopping:
            for rank, child in children:
                code = child.poll()
                if code is not None:
                    print(f"TensorFold rank {rank} exited ({code}); stopping both ranks.", flush=True)
                    return 1
            time.sleep(1)
        return 0
    finally:
        for _, child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        for _, child in children:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()

if __name__ == "__main__":
    sys.exit(run())
'''

TENSORFOLD_CHECK = r'''
import subprocess
import torch
import tensorfold
import tensorfold.families.glm5_next.cuda.weights

assert torch.cuda.device_count() == 2, "TensorFold profile requires exactly two GPUs"
assert torch.distributed.is_nccl_available(), "NCCL is required"
for device in range(2):
    p = torch.cuda.get_device_properties(device)
    assert p.major == 10 and "B200" in p.name, "This profile requires B200 GPUs"
    assert p.total_memory >= 128 * 2**30, "At least 128 GiB per GPU is required"
    with torch.cuda.device(device):
        assert (torch.ones(1, device=f"cuda:{device}") + 1).item() == 2
subprocess.run(["nvcc", "--version"], check=True, capture_output=True)
subprocess.run(["tensorfold", "serve", "--help"], check=True, capture_output=True)
print("TENSORFOLD_RUNTIME_OK", flush=True)
'''


def tensorfold_dockerfile(pin):
    """Use the base's CUDA/PyTorch toolchain; TensorFold's core deps exclude torch."""
    revision, checksum, base = pin["source_revision"], pin["source_sha256"], pin["image"]
    if (
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not re.fullmatch(r"[0-9a-f]{64}", checksum)
        or not re.fullmatch(r"nvcr.io/nvidia/pytorch@sha256:[0-9a-f]{64}", base)
    ):
        raise ValueError("TensorFold runtime must have a pinned base image and source checksum.")
    return (
        f"FROM {base}\n"
        "COPY tensorfold.tar.gz /tmp/tensorfold.tar.gz\n"
        "RUN mkdir -p /opt/tensorfold /opt/velocicyber && "
        "tar -xzf /tmp/tensorfold.tar.gz --strip-components=1 -C /opt/tensorfold && "
        "python3 -m pip install --no-cache-dir /opt/tensorfold "
        "'huggingface-hub>=1.32,<2' 'hf-xet>=1.2,<2' ninja && "
        "rm /tmp/tensorfold.tar.gz\n"
        "COPY tensorfold_server.py /opt/velocicyber/tensorfold_server.py\n"
        "ENV TENSORFOLD_NO_UPDATE_CHECK=1 TENSORFOLD_NO_LIVE=1 "
        "PYTHONUNBUFFERED=1 TORCH_EXTENSIONS_DIR=/root/.cache/tensorfold/torch_extensions\n"
        f'LABEL io.velocicyber.tensorfold-source="{revision}"\n'
    )


def prepare_tensorfold(cfg, folder, docker):
    """Build/reuse the source-pinned image, then check it before any weights download."""
    pin = cfg["tensorfold_runtime"]
    dockerfile = tensorfold_dockerfile(pin)
    if cfg["gpus"] != 2:
        raise ValueError("TensorFold GLM requires exactly two B200 GPUs.")
    fingerprint = hashlib.sha256(
        (dockerfile + TENSORFOLD_RUNNER + pin["source_sha256"]).encode()
    ).hexdigest()[:24]
    image = "velocicyber-tensorfold:" + fingerprint
    found = subprocess.run([*docker, "image", "inspect", image], capture_output=True)
    if found.returncode:
        print(json.dumps({"stage": "runtime_build"}), flush=True)
        with tempfile.TemporaryDirectory(prefix="tensorfold-build-", dir=folder) as temp:
            work = Path(temp)
            source = work / "tensorfold.tar.gz"
            url = (
                "https://codeload.github.com/ashhart/TensorFold/tar.gz/" + pin["source_revision"]
            )
            digest = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(url, timeout=60) as response, source.open("wb") as out:
                while block := response.read(1024**2):
                    total += len(block)
                    if total > 64 * 1024**2:
                        raise ValueError("TensorFold source archive unexpectedly exceeds 64 MiB.")
                    out.write(block)
                    digest.update(block)
            if digest.hexdigest() != pin["source_sha256"]:
                raise ValueError("TensorFold source archive checksum mismatch.")
            (work / "Dockerfile").write_text(dockerfile)
            (work / "tensorfold_server.py").write_text(TENSORFOLD_RUNNER)
            subprocess.run([*docker, "build", "--tag", image, str(work)], check=True)
    print(json.dumps({"stage": "runtime_check"}), flush=True)
    name = "velocicyber-runtime-check-" + uuid.uuid4().hex
    try:
        checked = subprocess.run(
            [*docker, "run", "--rm", "--name", name, "--gpus", "all", "--network", "none",
             image, "python3", "-c", TENSORFOLD_CHECK],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        subprocess.run([*docker, "rm", "--force", name], capture_output=True, timeout=20)
        raise RuntimeError("TensorFold runtime check timed out before model download.") from None
    if checked.returncode:
        raise RuntimeError("TensorFold runtime check failed before download: " + checked.stderr[-1600:])
    return image


def tensorfold_server_options(cfg, folder):
    snapshot = (
        "/root/.cache/huggingface/hub/models--" + cfg["repo"].replace("/", "--")
        + "/snapshots/" + cfg["revision"]
    )
    # Tokens are needed by the downloader only. The server uses an immutable snapshot.
    mounts = [
        "--volume", str(folder / "hf-cache") + ":/root/.cache/huggingface:ro",
        "--volume", str(folder / "tensorfold-cache") + ":/root/.cache/tensorfold",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--init",
    ]
    arguments = [
        "python3", "/opt/velocicyber/tensorfold_server.py", cfg["repo"], snapshot,
        str(cfg["context_window"]),
    ]
    return mounts, arguments
