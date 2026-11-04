#!/usr/bin/env python3
"""Download Velocicyber with curl, then install its isolated Python environment."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

REPOSITORY = "beau-warren/Velocicyber"
API = "https://api.github.com/repos/" + REPOSITORY
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024


def child_environment():
    # Download credentials are not needed by curl's environment or package installers.
    return {key: value for key, value in os.environ.items()
            if key not in {"GH_TOKEN", "GITHUB_TOKEN"}}


def fetch(url, destination, token):
    headers = "Authorization: Bearer " + token + "\n" if token else ""
    result = subprocess.run([
        "curl", "-q", "--fail", "--silent", "--show-error", "--location",
        "--proto", "=https", "--proto-redir", "=https",
        "--connect-timeout", "15", "--max-time", "180", "--retry", "2",
        "--max-filesize", str(MAX_ARCHIVE_BYTES),
        "--header", "Accept: application/vnd.github+json",
        "--header", "X-GitHub-Api-Version: 2022-11-28",
        "--header", "@-", "--output", str(destination), url,
    ], input=headers.encode(), capture_output=True, env=child_environment())
    if result.returncode:
        # Avoid echoing provider messages or temporary signed redirect URLs.
        raise RuntimeError(
            f"GitHub download failed (curl exit {result.returncode}). "
            "For this private repository, set GH_TOKEN to a GitHub token with Contents read "
            "access. Check repository access, connectivity and API rate limits before retrying."
        )


def unpack(archive, folder):
    """Extract only regular files/directories from one bounded source archive."""
    folder.mkdir()
    with tarfile.open(archive, "r:gz") as bundle:
        members, names, roots, size = [], set(), set(), 0
        for count, member in enumerate(bundle):
            parts = PurePosixPath(member.name).parts
            if (not parts or member.name.startswith("/") or ".." in parts
                    or "\\" in member.name or any(ord(c) < 32 for c in member.name)
                    or not (member.isdir() or member.isfile())):
                raise ValueError("Source archive contains an unsafe path or non-regular entry.")
            roots.add(parts[0])
            if len(roots) != 1 or count >= 5000:
                raise ValueError("Unexpected source archive layout or file count.")
            if len(parts) == 1:
                if not member.isdir():
                    raise ValueError("Source archive must contain a top-level directory.")
                continue
            relative = PurePosixPath(*parts[1:])
            if relative in names:
                raise ValueError("Source archive contains duplicate paths.")
            names.add(relative)
            size += member.size
            if member.size < 0 or size > MAX_SOURCE_BYTES:
                raise ValueError("Source archive is too large.")
            members.append((member, relative))
        for member, relative in members:
            target = folder / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
    for required in ("install.sh", "pyproject.toml", "velocicyber/cli.py"):
        if not (folder / required).is_file():
            raise ValueError("Downloaded archive is not a complete Velocicyber source tree.")


def install(destination, ref="main", *, download_only=False):
    destination = Path(os.path.abspath(Path(destination).expanduser()))
    if os.path.lexists(destination):
        raise ValueError("Destination already exists; choose a new directory: " + str(destination))
    if sys.version_info < (3, 11):
        raise ValueError("Python 3.11 or newer is required.")
    if not shutil.which("curl"):
        raise ValueError("Install curl before running this installer.")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    if token and not re.fullmatch(r"[A-Za-z0-9_]+", token):
        raise ValueError("GitHub token has an invalid format; its value was not displayed.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".velocicyber-download-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        metadata = staging / "commit.json"
        fetch(API + "/commits/" + quote(ref, safe=""), metadata, token)
        details = json.loads(metadata.read_text())
        revision = details.get("sha") if isinstance(details, dict) else None
        if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("GitHub did not return a valid commit ID.")
        print("Downloading Velocicyber at " + revision[:12] + "…", flush=True)
        archive = staging / "source.tar.gz"
        fetch(API + "/tarball/" + revision, archive, token)
        source = staging / "source"
        unpack(archive, source)
        # Exclusive creation also catches a destination created while downloading.
        destination.mkdir(mode=0o700)
        for item in source.iterdir():
            item.rename(destination / item.name)
    runtime = destination / ".runtime"
    runtime.mkdir(mode=0o700)
    receipt = runtime / "source-install.json"
    with receipt.open("x") as output:
        os.fchmod(output.fileno(), 0o600)
        json.dump({"repository": REPOSITORY, "commit": revision}, output, indent=2)
        output.write("\n")
    print("Source saved to " + str(destination), flush=True)
    if not download_only:
        result = subprocess.run(["sh", str(destination / "install.sh")],
                                cwd=destination, env=child_environment())
        if result.returncode:
            raise RuntimeError("Python installation failed. Source was retained; retry: "
                               + shlex.join(["sh", str(destination / "install.sh")]))
    print("Launch: " + shlex.quote(str(destination / "velocicyber.sh")))
    if download_only:
        print("Source-only download. Run ./install.sh in that directory before launching.")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path.home() / "Velocicyber",
                        help="New installation directory (default: ~/Velocicyber)")
    parser.add_argument("--ref", default="main", help="Git branch, tag or commit to download")
    parser.add_argument("--download-only", action="store_true", help="Skip Python dependency installation")
    args = parser.parse_args()
    try:
        install(args.dir, args.ref, download_only=args.download_only)
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        parser.exit(1, "Installation stopped: " + str(exc) + "\n")
    except KeyboardInterrupt:
        parser.exit(130, "Installation interrupted. Any downloaded installation was retained.\n")


if __name__ == "__main__":
    main()
