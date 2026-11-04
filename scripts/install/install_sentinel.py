"""Install the pinned Guardian GGUF and its separate, verified ARM64 CPU runtime."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.runtime.pi_download import DownloadTransportError, download  # noqa: E402
from velocicyber.engine.sessions import private_write  # noqa: E402


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_manifest(folder):
    result = {}
    for path in folder.rglob("*"):
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Sentinel runtime file escapes its installation.")
        if path.is_file():
            result[str(path.relative_to(folder))] = file_hash(path)
    return result


def install_runtime(pin, folder):
    base = folder / "runtime"
    if base.is_symlink():
        raise ValueError("Sentinel runtime directory must not be a symlink.")
    base.mkdir(exist_ok=True, mode=0o700)
    archive, target = base / pin["filename"], base / pin["tag"]
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        received = 0
        with httpx.Client(timeout=60, follow_redirects=True, trust_env=False) as client:
            with client.stream("GET", pin["url"]) as response, partial.open("wb") as stream:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    received += len(chunk)
                    if received > pin["bytes"]:
                        raise ValueError("Sentinel runtime download exceeded its pinned size.")
                    stream.write(chunk)
        if received != pin["bytes"] or file_hash(partial) != pin["sha256"]:
            raise ValueError("Sentinel runtime download checksum mismatch.")
        partial.replace(archive)
    if archive.is_symlink() or archive.stat().st_size != pin["bytes"] or file_hash(archive) != pin["sha256"]:
        raise ValueError("Sentinel runtime archive checksum mismatch.")
    with TemporaryDirectory(prefix="extract-", dir=base) as temporary:
        extracted = Path(temporary)
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            if sum(member.size for member in members) > 128 * 1024**2:
                raise ValueError("Sentinel runtime archive exceeds its extraction limit.")
            bundle.extractall(extracted, members=members, filter="data")
        expected = runtime_manifest(extracted)
        if target.exists():
            if target.is_symlink() or runtime_manifest(target) != expected:
                raise ValueError("Sentinel runtime files changed; installation was not activated.")
        else:
            extracted.rename(target)
    binary = target / pin["binary"]
    if not binary.resolve().is_relative_to(target.resolve()) or not os.access(binary, os.X_OK):
        raise ValueError("Verified sentinel server binary is missing or not executable.")
    files = {str((target / name).relative_to(folder)): checksum for name, checksum in expected.items()}
    return binary, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=4,
                        help="Parallel resumable download chunks (default: 4).")
    args = parser.parse_args()
    os.umask(0o077)
    cfg = json.loads((ROOT / "config/locks/sentinel-model.lock.json").read_text())
    folder = ROOT / ".runtime/sentinel-observer"
    if folder.is_symlink():
        raise ValueError("Sentinel runtime must not be a symlink.")
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (folder / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing_model = folder / "weights" / cfg["revision"] / Path(cfg["filename"]).name
        required = 256 * 1024**2 if existing_model.is_file() else 2 * cfg["weight_bytes"] + 1024**3
        if shutil.disk_usage(folder).free < required:
            raise RuntimeError("Allow 11 GiB free for download and assembly (5 GB retained).")

        def progress(stage, **values):
            event = {"stage": stage, **values}
            private_write(folder / "download.json", json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)

        model = download(cfg, folder, progress, workers=args.workers)
        progress("installing_runtime")
        binary, runtime_files = install_runtime(cfg["runtime"], folder)
        receipt = {
            **cfg, "model_path": str(model), "model_mtime_ns": model.stat().st_mtime_ns,
            "binary_path": str(binary), "binary_sha256": file_hash(binary),
            "runtime_files": runtime_files,
        }
        private_write(folder / "installed.json", json.dumps(receipt, indent=2) + "\n")
        progress("ready", bytes=cfg["weight_bytes"], total=cfg["weight_bytes"])


if __name__ == "__main__":
    try:
        main()
    except DownloadTransportError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(75)  # Temporary failure; a supervisor can retry with backoff.
