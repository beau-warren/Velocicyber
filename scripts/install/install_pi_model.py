"""Install the pinned CPU runtime and Pi GGUF locally; never uses Lambda or uploads."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.runtime import pi_download  # noqa: E402
from scripts.runtime.serve_model import file_sha256  # noqa: E402
from velocicyber.models.pi_profiles import PI_MODEL_REPO, PI_PROFILES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=PI_PROFILES, default=PI_MODEL_REPO)
    args = parser.parse_args()
    os.umask(0o077)
    if platform.machine() not in {"aarch64", "arm64"}:
        raise RuntimeError("This installation profile requires a 64-bit ARM system.")
    profile = PI_PROFILES[args.model]
    cfg = json.loads((ROOT / profile.lock_file).read_text())
    folder = ROOT / ".runtime" / profile.directory
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "install.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"Pi model installation is already running; progress is in {folder}."
            ) from None

        def progress(part, stage, **values):
            data = {"stage": stage, "updated_at": time.time(), **values}
            tmp = folder / (part + ".tmp")
            tmp.write_text(json.dumps(data) + "\n")
            tmp.replace(folder / (part + ".json"))
            print(json.dumps({"part": part, **data}), flush=True)

        def download():
            uploaded = folder / cfg["filename"]
            if uploaded.exists():
                progress("download", "verifying")
                if (
                    uploaded.stat().st_size != cfg["weight_bytes"]
                    or file_sha256(uploaded) != cfg["sha256"]
                ):
                    raise RuntimeError(
                        "Uploaded Pi GGUF checksum mismatch; inspect it before retrying."
                    )
                progress("download", "ready", bytes=cfg["weight_bytes"], total=cfg["weight_bytes"])
                return uploaded
            remaining = cfg["weight_bytes"]
            if shutil.disk_usage(folder).free < 2 * remaining + 2 * 1024**3:
                raise RuntimeError(
                    "At least 25 GiB free disk is needed for download parts and assembly."
                )
            model = pi_download.download(
                cfg, folder, lambda stage, **values: progress("download", stage, **values)
            )
            progress("download", "ready", bytes=cfg["weight_bytes"], total=cfg["weight_bytes"])
            return model

        def build():
            source = folder / "ik_llama.cpp"
            binary = source / "build" / "bin" / "llama-server"
            marker = folder / "runtime-revision.txt"
            if (
                binary.exists()
                and marker.exists()
                and marker.read_text().strip() == cfg["runtime_revision"]
            ):
                return binary
            progress("build", "building")
            # Reuse a matching, locally compiled static runtime without rebuilding Qwen.
            # Copy it into this model's folder so both installations remain independent.
            previous = ROOT / ".runtime" / "pi-model" / "installed.json"
            if args.model != PI_MODEL_REPO and previous.exists():
                saved = json.loads(previous.read_text())
                candidate = Path(saved["binary_path"])
                revision_marker = previous.parent / "runtime-revision.txt"
                if (
                    all(saved.get(k) == cfg[k] for k in ("runtime_repo", "runtime_revision"))
                    and candidate.resolve().is_relative_to(previous.parent.resolve())
                    and candidate.is_file()
                    and revision_marker.exists()
                    and revision_marker.read_text().strip() == cfg["runtime_revision"]
                ):
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, binary)
                    marker.write_text(cfg["runtime_revision"] + "\n")
                    progress("build", "ready", reused=True)
                    return binary
            with (folder / "build.log").open("a") as log:

                def run(argv, **kwargs):
                    subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT, **kwargs)

                cmake = Path(sys.executable).parent / "cmake"
                if not cmake.exists():
                    run([sys.executable, "-m", "pip", "install", "cmake>=3.28,<4"])
                if not (source / ".git").exists():
                    run(["git", "init", str(source)])
                    run(["git", "-C", str(source), "remote", "add", "origin", cfg["runtime_repo"]])
                present = (
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(source),
                            "cat-file",
                            "-e",
                            cfg["runtime_revision"] + "^{commit}",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    ).returncode
                    == 0
                )
                if not present:
                    run(
                        [
                            "git",
                            "-c",
                            "http.version=HTTP/1.1",
                            "-C",
                            str(source),
                            "fetch",
                            "--depth=1",
                            "origin",
                            cfg["runtime_revision"],
                        ]
                    )
                run(["git", "-C", str(source), "checkout", "--detach", cfg["runtime_revision"]])
                run(
                    [
                        str(cmake),
                        "-S",
                        str(source),
                        "-B",
                        str(source / "build"),
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_NATIVE=ON",
                        "-DGGML_BLAS=OFF",
                        "-DBUILD_SHARED_LIBS=OFF",
                        "-DLLAMA_CURL=OFF",
                        "-DLLAMA_BUILD_TESTS=OFF",
                    ]
                )
                run(
                    [
                        str(cmake),
                        "--build",
                        str(source / "build"),
                        "--target",
                        "llama-server",
                        "-j2",
                    ]
                )
            marker.write_text(cfg["runtime_revision"] + "\n")
            progress("build", "ready")
            return binary

        def tracked(part, operation):
            try:
                return operation()
            except Exception as exc:
                progress(part, "failed", error=str(exc))
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            weights, runtime = (
                pool.submit(tracked, "download", download),
                pool.submit(tracked, "build", build),
            )
            model, binary = weights.result(), runtime.result()
        result = {
            **cfg,
            "model_path": str(model),
            "binary_path": str(binary),
            "model_mtime_ns": model.stat().st_mtime_ns,
        }
        temporary = folder / "installed.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(folder / "installed.json")
        progress("setup", "ready", model_path=str(model), binary_path=str(binary))


if __name__ == "__main__":
    main()
