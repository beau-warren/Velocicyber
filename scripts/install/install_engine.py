"""Install the pinned official ARM64 bundle into this workspace only."""

import hashlib
import json
import platform
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
lock = json.loads((ROOT / "config/locks/upstream.lock.json").read_text())
if platform.machine() not in ("aarch64", "arm64") or platform.system() != "Linux":
    raise SystemExit("This lock selects Linux ARM64. Select a verified asset for other platforms.")
runtime = ROOT / ".runtime"
runtime.mkdir(exist_ok=True, mode=0o700)
archive = runtime / lock["asset"]
if not archive.exists():
    url = f"{lock['repository']}/releases/download/{lock['tag']}/{lock['asset']}"
    partial = archive.with_suffix(".partial")
    print(f"Downloading {lock['asset']}", flush=True)
    urllib.request.urlretrieve(url, partial)
    partial.rename(archive)
with archive.open("rb") as f:
    digest = hashlib.file_digest(f, "sha256").hexdigest()
if digest != lock["sha256"]:
    raise SystemExit("SHA256 mismatch; refusing to install. Remove the archive and try again.")
destination = runtime / "engine"
destination.mkdir(exist_ok=True)
with tarfile.open(archive) as bundle:
    bundle.extractall(destination, filter="data")
print(f"Verified and installed the Velocicyber execution engine ({lock['tag']}) into {destination}")
for p in destination.rglob("interpreter"):
    if p.is_file():
        print(f"Entrypoint: {p}")
