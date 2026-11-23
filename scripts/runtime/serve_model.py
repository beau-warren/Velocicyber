"""Run on an approved Lambda instance, with configuration sent through SSH stdin.

Requires Docker and NVIDIA Container Toolkit. Weights download on the GPU host.
"""

import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import quote

# Runs inside the pinned serving image, which supplies huggingface_hub/Xet.
# Credentials come from the private Docker env file, never command arguments.
SNAPSHOT_DOWNLOADER = r"""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import hf_hub_download

cfg = json.load(sys.stdin)
total = cfg["weight_bytes"]
done = 0
print(json.dumps({"stage": "download_progress", "bytes": 0, "total": total}), flush=True)

def download(file):
    path = Path(hf_hub_download(
        cfg["repo"], file["filename"], revision=cfg["revision"],
        token=os.environ.get("HF_TOKEN") or False,
    ))
    if path.stat().st_size != file["size"]:
        raise RuntimeError("Downloaded file size mismatch: " + file["filename"])
    if file["sha256"]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != file["sha256"]:
            # A corrupt cache file must not be silently reused on the next try.
            path.resolve().unlink()
            raise RuntimeError("Downloaded file checksum mismatch: " + file["filename"])
    return file["size"] if file["filename"].endswith(".safetensors") else 0

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(download, file) for file in cfg["download_files"]]
    try:
        for future in as_completed(futures):
            size = future.result()
            if size:
                done += size
                print(json.dumps({"stage": "download_progress", "bytes": done, "total": total}), flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
if done != total:
    raise RuntimeError("Downloaded weight manifest total does not match model size")
"""


def download_safetensors(cfg, folder, docker, mounts):
    files = cfg["download_files"]
    if (
        not files
        or sum(file["size"] for file in files if file["filename"].endswith(".safetensors"))
        != cfg["weight_bytes"]
    ):
        raise ValueError("A complete pinned weight manifest is required before downloading.")
    snapshot = (
        folder
        / "hf-cache"
        / "hub"
        / ("models--" + cfg["repo"].replace("/", "--"))
        / "snapshots"
        / cfg["revision"]
    )
    missing = 0
    for file in files:
        if ((Path(file["filename"]).name != file["filename"] and not (
                cfg.get("repo") == "Blackfrost-AI/Inkling-2-Mini-Derisked"
                and file["filename"] == "tiktoken/tokenizer.model"
        )) or file["size"] < 0):
            raise ValueError("Invalid model file manifest.")
        cached = snapshot / file["filename"]
        if not cached.is_file() or cached.stat().st_size != file["size"]:
            missing += file["size"]
    if shutil.disk_usage(folder).free < missing + 64 * 2**30:
        raise RuntimeError("Insufficient disk space for the model plus 64 GiB working space.")
    subprocess.run(
        [
            *docker,
            "run",
            "--rm",
            "--interactive",
            *mounts,
            "--env",
            "HF_HUB_DISABLE_PROGRESS_BARS=1",
            "--env",
            "HF_HUB_DISABLE_TELEMETRY=1",
            "--entrypoint",
            "python3",
            cfg["image"],
            "-c",
            SNAPSHOT_DOWNLOADER,
        ],
        input=json.dumps(
            {key: cfg[key] for key in ("repo", "revision", "weight_bytes", "download_files")}
        ),
        text=True,
        check=True,
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def private_file(path, contents):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(contents)


def progress(stage, **values):
    print(json.dumps({"stage": stage, **values}), flush=True)


def download_gguf(cfg, folder):
    stall_seconds = int(cfg.get("download_stall_seconds", 180))
    minimum_speed = int(cfg.get("download_min_bytes_per_second", 1024))
    chunk_bytes = int(cfg.get("download_chunk_mib", 256)) * 1024**2
    total_bytes = int(cfg["weight_bytes"])
    if min(stall_seconds, minimum_speed, chunk_bytes, total_bytes) <= 0:
        raise ValueError("Download sizes, stall duration and minimum speed must be positive.")
    destination = folder / "weights" / cfg["revision"] / Path(cfg["filename"]).name
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists():
        progress("download_verifying")
        if file_sha256(destination) == cfg["sha256"]:
            return destination
    partial = destination.with_suffix(".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total_bytes:
        raise RuntimeError(
            "Partial download exceeds the pinned model size; inspect it before retrying."
        )
    progress("download_progress", bytes=offset, total=total_bytes)
    chunk = destination.with_suffix(".chunk")
    response_headers = destination.with_suffix(".headers")
    headers = folder / "download.headers"
    token = cfg["token"].replace("\n", "").replace("\r", "")
    private_file(headers, "Authorization: Bearer " + token + "\n" if token else "")
    url = f"https://huggingface.co/{cfg['repo']}/resolve/{cfg['revision']}/" + quote(
        cfg["filename"], safe="/"
    )
    try:
        while offset < total_bytes:
            end = min(offset + chunk_bytes, total_bytes) - 1
            size = end - offset + 1
            for attempt in range(3):
                # Keep signed redirect headers private and never include them in logs.
                private_file(response_headers, "")
                result = subprocess.run(
                    [
                        "curl",
                        "--disable",  # Ignore curlrc, including any hidden max-time/retry options.
                        "--http1.1",
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--connect-timeout",
                        "15",
                        "--speed-limit",
                        str(minimum_speed),
                        "--speed-time",
                        str(stall_seconds),
                        "--range",
                        f"{offset}-{end}",
                        "--max-filesize",
                        str(size),
                        "--header",
                        "@" + str(headers),
                        "--dump-header",
                        str(response_headers),
                        "--output",
                        str(chunk),
                        "--write-out",
                        "%{http_code}",
                        url,
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    # Redirect responses may include their own headers; only the final
                    # response is allowed to describe the bytes we are about to append.
                    final_headers = {}
                    for line in response_headers.read_text().splitlines():
                        if line.startswith("HTTP/"):
                            final_headers = {}
                        elif ":" in line:
                            key, value = line.split(":", 1)
                            final_headers[key.lower()] = value.strip()
                    if (
                        result.stdout.strip() != "206"
                        or final_headers.get("content-range")
                        != f"bytes {offset}-{end}/{total_bytes}"
                        or not chunk.exists()
                        or chunk.stat().st_size != size
                    ):
                        raise RuntimeError(
                            "Server returned an unexpected byte range or length; "
                            f"{offset} partial bytes retained without appending the response."
                        )
                    break
                # Retry only this bounded chunk. Authentication, rate limits and
                # servers ignoring Range stop immediately, without another request.
                if result.returncode not in (18, 28, 56, 92) or attempt == 2:
                    raise RuntimeError(
                        f"Model chunk {offset}-{end} failed after {attempt + 1} attempt(s) "
                        f"(curl {result.returncode}, HTTP {result.stdout.strip()}); "
                        f"{offset} partial bytes retained for resume."
                    )
                time.sleep(2**attempt)
            with partial.open("ab") as output, chunk.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024**2)
                output.flush()
                os.fsync(output.fileno())
            offset += size
            chunk.unlink()
            progress("download_progress", bytes=offset, total=total_bytes)
        progress("download_verifying")
        if file_sha256(partial) != cfg["sha256"]:
            partial.unlink()
            raise RuntimeError("GGUF checksum verification failed; model was not started.")
        partial.replace(destination)
        return destination
    finally:
        headers.unlink(missing_ok=True)
        response_headers.unlink(missing_ok=True)
        chunk.unlink(missing_ok=True)


def check_runtime(cfg, docker):
    """Exercise dependency imports before downloading weights or replacing inference."""
    name = "velocicyber-runtime-check-" + uuid.uuid4().hex
    program = (
        "import torch, torchvision; "
        "from torchvision.ops import nms; "
        "from vllm.engine.arg_utils import EngineArgs; "
        "assert nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "assert (torch.ones(1, device='cuda') + 1).item() == 2"
    )
    if cfg.get("vllm_cli"):
        program += (
            "; import vllm.tool_parsers.inkling_tool_parser"
            "; import vllm.reasoning.inkling_reasoning_parser"
            "; assert torch.cuda.get_device_capability()[0] >= 9, 'Inkling requires Hopper/Blackwell'"
        )
    progress("runtime_check")
    try:
        result = subprocess.run(
            [*docker, "run", "--rm", "--name", name, "--network", "none", "--gpus", "all",
             "--entrypoint", "python3", cfg["image"], "-c", program],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        subprocess.run([*docker, "rm", "--force", name], capture_output=True, timeout=20)
        raise RuntimeError("Serving runtime check timed out before model download.") from None
    if result.returncode:
        raise RuntimeError("Serving runtime check failed before model download: " + result.stderr[-1600:])


def serve(cfg, *, tensorfold_prepare=None, tensorfold_options=None):
    cfg = dict(cfg)
    context_window = cfg.get("context_window", 32768)
    if type(context_window) is not int or context_window < 4096:
        raise ValueError("Serving context window must be at least 4096 tokens.")
    folder = Path.home() / ".velocicyber"
    folder.mkdir(mode=0o700, exist_ok=True)
    environment = folder / "serve.env"
    docker = ["sudo", "-n", "docker"]
    backend = cfg.get("backend", "vllm")
    if backend not in ("llama.cpp", "vllm", "tensorfold"):
        raise ValueError("Unsupported serving backend.")
    if backend == "tensorfold" and (tensorfold_prepare is None or tensorfold_options is None):
        raise ValueError("TensorFold runtime helpers are required.")
    health_options = (
        [
            "--health-cmd",
            "curl --fail --silent --show-error http://127.0.0.1:8000/health",
            "--health-start-period",
            "120s",
        ]
        if backend == "llama.cpp"
        else []
    )
    existing = subprocess.run(
        [
            *docker,
            "inspect",
            "--format",
            '{{index .Config.Labels "io.velocicyber.managed"}}',
            "velocicyber-model",
        ],
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0 and existing.stdout.strip() != "true":
        raise RuntimeError("The velocicyber-model container is not managed by this harness.")
    try:
        image_pulled = False
        if backend == "llama.cpp":
            model = download_gguf(cfg, folder)
            mounts = ["--volume", str(folder / "weights") + ":/models:ro"]
            arguments = [
                "--model",
                "/models/" + str(model.relative_to(folder / "weights")),
                "--alias",
                cfg["repo"],
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--ctx-size",
                str(context_window),
                "--parallel",
                "1",
                "--n-gpu-layers",
                "all",
                "--split-mode",
                "layer",
                "--jinja",
            ]
            arguments += list(cfg.get("extra_arguments", ()))
        else:
            private_file(
                environment,
                "HF_TOKEN=" + cfg["token"].replace("\n", "").replace("\r", "") + "\n"
                "HUGGING_FACE_HUB_TOKEN=\nHF_TOKEN_PATH=/dev/null\n",
            )
            mounts = [
                "--env-file",
                str(environment),
                "--volume",
                str(folder / "hf-cache") + ":/root/.cache/huggingface",
            ]
            arguments = [
                "--model",
                cfg["repo"],
                "--revision",
                cfg["revision"],
                "--served-model-name",
                cfg["repo"],
                "--tensor-parallel-size",
                str(cfg["gpus"]),
                "--max-model-len",
                str(context_window),
                "--max-num-seqs",
                str(cfg.get("max_num_seqs", 2)),
                "--gpu-memory-utilization",
                "0.90",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                cfg["tool_parser"],
            ]
            if cfg.get("vllm_cli"):
                arguments[:2] = ["serve", cfg["repo"]]
                arguments += ["--host", "0.0.0.0", "--port", "8000"]
            if cfg["reasoning_parser"]:
                arguments += ["--reasoning-parser", cfg["reasoning_parser"]]
            arguments += list(cfg.get("extra_arguments", ()))
            progress("server_starting")
            if backend == "tensorfold":
                cfg["image"] = tensorfold_prepare(cfg, folder, docker)
            else:
                subprocess.run([*docker, "pull", cfg["image"]], check=True)
                check_runtime(cfg, docker)
            image_pulled = True
            if cfg.get("download_files"):
                # Await the entire snapshot before starting the readiness timer.
                # Preserve the existing inference container on download failure.
                download_safetensors(cfg, folder, docker, mounts)
                mounts += [
                    "--env",
                    "HF_HUB_OFFLINE=1",
                    "--env",
                    "TRANSFORMERS_OFFLINE=1",
                    "--env",
                    "VLLM_WORKER_MULTIPROC_METHOD=spawn",
                ]
            if backend == "tensorfold":
                mounts, arguments = tensorfold_options(cfg, folder)
            elif cfg.get("vllm_cli"):
                mounts += ["--entrypoint", "vllm"]
        # Resolve the image before replacing a working server.
        progress("server_starting")
        if not image_pulled:
            subprocess.run([*docker, "pull", cfg["image"]], check=True)
        if existing.returncode == 0:
            subprocess.run([*docker, "rm", "--force", "velocicyber-model"], check=True)
        subprocess.run(
            [
                *docker,
                "run",
                "--detach",
                "--name",
                "velocicyber-model",
                "--label",
                "io.velocicyber.managed=true",
                "--gpus",
                "all",
                "--ipc",
                "host",
                "--restart",
                "no" if backend == "tensorfold" else "unless-stopped",
                "--publish",
                "127.0.0.1:8000:8000",
                *health_options,
                *mounts,
                cfg["image"],
                *arguments,
            ],
            check=True,
        )
        progress("model_loading")
    finally:
        environment.unlink(missing_ok=True)
