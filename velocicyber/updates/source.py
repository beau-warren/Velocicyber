"""Identify this installation and check its canonical source revision once on launch."""

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path

import httpx

from velocicyber.engine.sessions import private_write

REPOSITORY = "beau-warren/Velocicyber"
API = "https://api.github.com/repos/" + REPOSITORY
REMOTE = "https://github.com/" + REPOSITORY + ".git"


def revision(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{40}", value) is not None


def clean_environment():
    return {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT")
            if key in os.environ}


def git(root, *arguments, env=None):
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *arguments],
        env=env or clean_environment(), capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        # Git diagnostics can include credentials or signed URLs. Never echo them.
        raise RuntimeError("Git operation failed: " + arguments[0] + ". Check repository access and local changes.")
    return result.stdout.strip()


def installed_source(root):
    root = Path(root).resolve()
    if (root / ".git").exists():
        origin = git(root, "remote", "get-url", "origin")
        if origin not in {REMOTE, REMOTE.removesuffix(".git"), "git@github.com:" + REPOSITORY + ".git"}:
            raise ValueError("Automatic updates require the official Velocicyber origin; update forks manually.")
        commit = git(root, "rev-parse", "HEAD")
        kind = "git"
    else:
        receipt = json.loads((root / ".runtime/source-install.json").read_text())
        if receipt.get("repository") != REPOSITORY:
            raise ValueError("This source download has no official installation receipt.")
        commit, kind = receipt.get("commit"), "archive"
    if not revision(commit):
        raise ValueError("The installed source revision is invalid.")
    return kind, commit


def github_token(settings):
    return next((settings.secrets.get(key) for key in ("GITHUB_API", "GH_TOKEN", "GITHUB_TOKEN")
                 if settings.secrets.get(key)), "")


def github_headers(settings):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "Velocicyber-updates",
               "X-GitHub-Api-Version": "2022-11-28"}
    if token := github_token(settings):
        if not re.fullmatch(r"[A-Za-z0-9_]+", token):
            raise ValueError("Invalid GitHub credential format; its value was not displayed.")
        headers["Authorization"] = "Bearer " + token
    return headers


class SourceChecker:
    def __init__(self, settings):
        self.settings = settings
        self.path = settings.runtime / "harness-updates.json"
        self.result = {}
        self.config_error = ""
        self.checked_this_launch = False
        self.kind, self.installed = "", ""
        try:
            self.kind, self.installed = installed_source(settings.root)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            self.config_error = "Source revision unavailable. Use an official Git clone or curl installation."
        try:
            cached = json.loads(self.path.read_text())
            if (isinstance(cached, dict) and cached.get("schema") == 1
                    and cached.get("installed") == self.installed
                    and revision(cached.get("latest"))
                    and cached.get("status") in {"identical", "behind", "ahead", "diverged"}
                    and isinstance(cached.get("error"), str)):
                self.result = cached
        except (OSError, ValueError):
            pass

    def indicator(self):
        if self.config_error:
            return "VERSION UNKNOWN", "yellow"
        if self.result.get("status") == "behind":
            return "UPDATE AVAILABLE", "red"
        if not self.result:
            return "CHECKING UPDATES", "yellow"
        if self.result.get("error"):
            return "UPDATE CHECK FAILED", "yellow"
        if self.result.get("status") == "identical":
            return "UP TO DATE", "#39ff14"
        return "LOCAL SOURCE CHANGES", "yellow"

    async def check(self, *, transport=None):
        if self.config_error or self.checked_this_launch:
            return
        self.checked_this_launch = True
        result = {**self.result, "schema": 1, "installed": self.installed,
                  "checked_at": time.time(), "error": ""}
        try:
            async with httpx.AsyncClient(
                headers=github_headers(self.settings), transport=transport,
                follow_redirects=False, timeout=httpx.Timeout(30, connect=15),
            ) as client:
                response = await asyncio.wait_for(client.get(API + "/commits/main"), 45)
                response.raise_for_status()
                latest = response.json().get("sha")
                if not revision(latest):
                    raise ValueError("Invalid source revision")
                status = "identical"
                if latest != self.installed:
                    response = await asyncio.wait_for(
                        client.get(API + "/compare/" + self.installed + "..." + latest), 45,
                    )
                    response.raise_for_status()
                    # GitHub describes the remote head relative to our installed base.
                    status = {"ahead": "behind", "behind": "ahead", "identical": "identical",
                              "diverged": "diverged"}.get(response.json().get("status"))
                    if status is None:
                        raise ValueError("Invalid source comparison")
                result.update(latest=latest, status=status)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, TimeoutError) as exc:
            code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            detail = f"HTTP {code}" if code else type(exc).__name__
            result["error"] = f"Velocicyber source check failed ({detail}); retry on next launch."
            if code in {401, 403, 404}:
                result["error"] += " Private repositories require GITHUB_API in .env with Contents read access."
        self.result = result
        try:
            private_write(self.path, json.dumps(result) + "\n")
        except OSError:
            pass

    def details(self):
        return "\n".join([
            "Velocicyber harness: " + self.indicator()[0],
            "Installed source: " + (self.installed[:12] or "unknown"),
            "Latest main: " + (self.result.get("latest", "")[:12] or "not checked"),
            self.config_error or self.result.get("error", ""),
            "Source: https://github.com/" + REPOSITORY,
        ]).strip()
