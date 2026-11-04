"""Cached, read-only checks of the official Rust releases and public advisories."""

import asyncio
import json
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPOSITORY = "https://github.com/openinterpreter/openinterpreter"
API = "https://api.github.com/repos/openinterpreter/openinterpreter"


def rust_version(tag):
    match = re.fullmatch(r"rust-v(\d+)\.(\d+)\.(\d+)", str(tag))
    return tuple(map(int, match.groups())) if match else None


def stable_rust_release(item):
    return (
        isinstance(item, dict) and rust_version(item.get("tag_name"))
        and item.get("draft") is False and item.get("prerelease") is False
    )


class UpdateChecker:
    def __init__(self, settings):
        self.settings = settings
        self.path = settings.runtime / "upstream-updates.json"
        self.result = {}
        self.checked_this_launch = False
        self.config_error = ""
        try:
            lock = json.loads((settings.root / "config/locks/upstream.lock.json").read_text())
            if lock.get("repository") != REPOSITORY or not rust_version(lock.get("tag")):
                raise ValueError("The engine is not pinned to the official Rust release series.")
            self.engine = settings.engine()
            stat = self.engine.stat()
            self.identity = [str(self.engine), stat.st_mtime_ns, stat.st_size, lock["tag"]]
        except (OSError, ValueError, RuntimeError, AttributeError) as exc:
            self.config_error = f"Engine version unavailable: {exc}"
            self.identity = []
        try:
            cached = json.loads(self.path.read_text())
            if (
                isinstance(cached, dict)
                and cached.get("schema") == 1
                and cached.get("identity") == self.identity
                and all(
                    isinstance(cached.get(key, 0), (int, float))
                    and 0 <= cached.get(key, 0) <= time.time()
                    for key in ("checked_at", "release_checked_at", "advisory_checked_at")
                )
                and isinstance(cached.get("error", ""), str)
                and isinstance(cached.get("advisories", []), list)
                and all(
                    isinstance(item, dict) and {"id", "summary", "severity"} <= item.keys()
                    for item in cached.get("advisories", [])
                )
            ):
                self.result = cached
        except (OSError, ValueError):
            pass

    def indicator(self):
        installed = rust_version(self.result.get("installed"))
        latest = rust_version(self.result.get("latest"))
        if installed and latest and latest > installed and not self.config_error:
            return "UPDATE AVAILABLE", "red"
        if self.config_error:
            return "VERSION UNKNOWN", "yellow"
        if not self.result:
            return "CHECKING UPDATES", "yellow"
        if self.result.get("error"):
            return "UPDATE CHECK FAILED", "yellow"
        if installed and latest:
            return "UP TO DATE", "#39ff14"
        return "VERSION UNKNOWN", "yellow"

    async def installed_tag(self):
        process = await asyncio.create_subprocess_exec(
            str(self.engine), "--version",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        match = re.fullmatch(r"interpreter (\d+\.\d+\.\d+)", stdout.decode().strip())
        if process.returncode or not match:
            raise ValueError("Could not identify the installed engine's release version.")
        return "rust-v" + match[1]

    async def check(self, *, transport=None):
        if self.config_error or self.checked_this_launch:
            return
        self.checked_this_launch = True
        result = {
            "schema": 1, "identity": self.identity, "checked_at": time.time(),
            "installed": self.result.get("installed", ""),
            "latest": self.result.get("latest", ""),
            "release_checked_at": self.result.get("release_checked_at", 0),
            "advisories": self.result.get("advisories", []),
            "advisory_checked_at": self.result.get("advisory_checked_at", 0),
            "advisories_more": self.result.get("advisories_more", False),
            "error": "",
        }
        errors = []
        try:
            result["installed"] = await self.installed_tag()
        except (OSError, ValueError, TimeoutError):
            # Do not report green from a pin when the actual installed binary is unknown.
            result["installed"] = ""
            errors.append("Could not read the installed engine version.")
        async with httpx.AsyncClient(
            # DNS/TLS can be slow just after startup. Allow connection setup time
            # within the existing total deadline without adding requests or retries.
            timeout=httpx.Timeout(30, connect=15), transport=transport, follow_redirects=False,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "Velocicyber-updates"},
        ) as client:
            for kind, url in (
                ("release", API + "/releases/latest"),
                ("advisory", API + "/security-advisories?state=published&per_page=30"),
            ):
                try:
                    # Cap the entire response too, including slow trickling transfers.
                    response = await asyncio.wait_for(client.get(url), timeout=45)
                    response.raise_for_status()
                    items = response.json()
                    if kind == "release":
                        if (
                            not isinstance(items, dict)
                            or not isinstance(items.get("tag_name"), str)
                            or not isinstance(items.get("draft"), bool)
                            or not isinstance(items.get("prerelease"), bool)
                        ):
                            raise ValueError("Unexpected release response format")
                        if stable_rust_release(items):
                            items = [items]
                        else:
                            # The repo also has Python releases. Only query a larger
                            # listing when its designated latest release is not Rust.
                            response = await asyncio.wait_for(
                                client.get(API + "/releases?per_page=20"), timeout=45,
                            )
                            response.raise_for_status()
                            items = response.json()
                    if not isinstance(items, list) or any(not isinstance(i, dict) for i in items):
                        raise ValueError("Unexpected response format")
                    if kind == "release":
                        tags = [
                            item["tag_name"] for item in items
                            if stable_rust_release(item)
                        ]
                        if not tags:
                            raise ValueError("No stable Rust release was returned")
                        result["latest"] = max(tags, key=rust_version)
                        result["release_checked_at"] = result["checked_at"]
                    else:
                        result["advisories"] = [
                            {
                                "id": item.get("ghsa_id", ""),
                                "summary": item.get("summary", ""),
                                "severity": item.get("severity", "unknown"),
                            }
                            for item in items
                            if item.get("state") == "published" and not item.get("withdrawn_at")
                        ]
                        result["advisories_more"] = 'rel="next"' in response.headers.get("link", "")
                        result["advisory_checked_at"] = result["checked_at"]
                except (httpx.HTTPError, ValueError, KeyError, TypeError, TimeoutError) as exc:
                    status = (
                        f"HTTP {exc.response.status_code}"
                        if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
                    )
                    errors.append(
                        f"GitHub {kind} check failed ({status}); retry on next launch."
                    )
        result["error"] = " ".join(errors)
        self.result = result
        # Show the previous result during startup; each launch checks afresh once.
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as file:
                temp = file.name
                json.dump(result, file)
                file.write("\n")
            Path(temp).replace(self.path)
        except OSError:
            if temp:
                Path(temp).unlink(missing_ok=True)

    def details(self):
        def stamp(value):
            return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "Open Interpreter engine: " + self.indicator()[0],
            "Installed: " + str(self.result.get("installed") or "unknown"),
            "Latest stable Rust release: " + str(self.result.get("latest") or "not checked"),
        ]
        if self.result.get("release_checked_at"):
            lines.append("Release check: " + stamp(self.result["release_checked_at"]))
        if self.result.get("advisory_checked_at"):
            advisories = self.result.get("advisories", [])
            suffix = "+" if self.result.get("advisories_more") else ""
            lines.append(f"Published repository security advisories: {len(advisories)}{suffix}")
            lines.append("Advisory check: " + stamp(self.result["advisory_checked_at"]))
            lines.extend(
                f"  {item['id']} ({item['severity']}): {item['summary']}"
                for item in advisories[:10]
            )
        if error := self.config_error or self.result.get("error"):
            lines.append(error)
        lines.extend([
            "Checks run once on launch only. "
            "Updates are installed manually.",
            "Up to date refers to stable releases; it is not a security audit. "
            "Published advisories require review for affected packages and versions.",
            "Release notes: " + REPOSITORY + "/releases",
            "Security advisories: " + REPOSITORY + "/security/advisories",
        ])
        return "\n".join(lines)
