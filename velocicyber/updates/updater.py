"""Stage and test source updates; preserve local data and retain a rollback copy."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from velocicyber.engine.installation import engine_lease
from velocicyber.engine.sessions import private_write
from velocicyber.updates.source import (
    API,
    REMOTE,
    REPOSITORY,
    clean_environment,
    git,
    github_headers,
    github_token,
    installed_source,
    revision,
)

PENDING = "harness-update-pending.json"
BACKUP = "harness-update-backup.json"
PROTECTED = {".git", ".venv", ".runtime", ".alpha-audit", "workspace", "Modules.txt",
             "velocicyber_finetuning-data", "velocicyber_sentinel-data", "upstream"}


def source_path(name):
    path = Path(name)
    if (path.is_absolute() or not path.parts or ".." in path.parts
            or any(ord(c) < 32 or ord(c) == 127 for c in name) or "\\" in name
            or path.parts[0] in PROTECTED
            or (path.parts[0].startswith(".env") and name != ".env.example")
            or (path.parts[0] == "ssh-keys" and name != "ssh-keys/README.md")
            or (path.parts[0] == "SCOPE" and name not in {"SCOPE/README.md", "SCOPE/SCOPE.example.md"})
            or path.suffix in {".pem", ".key", ".gguf", ".safetensors"}):
        raise ValueError("Update contains a protected or unsafe source path.")
    return path


def manifest(folder):
    result = {}
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError("Source updates cannot contain symlinks.")
        if path.is_file():
            name = str(path.relative_to(folder))
            source_path(name)
            result[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "mode": path.stat().st_mode & 0o777}
    return result


def matches(path, item):
    return (path.is_file() and not path.is_symlink()
            and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            and (bool(path.stat().st_mode & 0o111) == bool(item["mode"] & 0o111)))


class SourceUpdater:
    def __init__(self, settings, notify=print):
        self.settings, self.notify = settings, notify
        self.root, self.runtime = settings.root, settings.runtime

    def download(self, commit, destination, *, transport=None):
        if not revision(commit):
            raise ValueError("Invalid update revision.")
        archive = destination.with_suffix(".tar.gz")
        url = API + "/tarball/" + commit
        headers = github_headers(self.settings)
        with httpx.Client(timeout=httpx.Timeout(60, connect=15), follow_redirects=False, transport=transport) as client:
            for _ in range(4):
                with client.stream("GET", url, headers=headers) as response:
                    if response.is_redirect:
                        next_url = str(response.url.join(response.headers["Location"]))
                        parsed = urlsplit(next_url)
                        if (parsed.scheme != "https" or parsed.hostname not in {"api.github.com", "codeload.github.com"}
                                or parsed.username or parsed.password or parsed.port not in {None, 443}):
                            raise ValueError("Unexpected source-download redirect.")
                        url = next_url
                        if parsed.hostname != "api.github.com":
                            headers = {k: v for k, v in headers.items() if k != "Authorization"}
                        continue
                    if response.status_code != 200:
                        raise RuntimeError(f"Source download failed (HTTP {response.status_code}); check GitHub access.")
                    size = 0
                    with archive.open("xb") as output:
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > 16 * 1024**2:
                                raise ValueError("Source archive exceeds the download limit.")
                            output.write(chunk)
                    break
            else:
                raise ValueError("Too many source-download redirects.")
        spec = importlib.util.spec_from_file_location("source_bootstrap", self.root / "bootstrap.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.unpack(archive, destination)
        return manifest(destination)

    def command(self, arguments, folder, log, *, env=None, timeout=600):
        with log.open("ab") as output:
            os.fchmod(output.fileno(), 0o600)
            result = subprocess.run(arguments, cwd=folder, env=env or clean_environment(),
                                    stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError("Update validation/installation failed; see " + str(log))

    def validate(self, source, run):
        changed = any((source / name).read_bytes() != (self.root / name).read_bytes()
                      for name in ("requirements.lock", "pyproject.toml"))
        python = Path(sys.executable)
        if changed:
            self.notify("Preparing an isolated Python environment for changed dependencies…")
            venv = run / "venv"
            self.command(["python3", "-m", "venv", str(venv)], source, run / "install.log")
            python = venv / "bin/python"
            self.command([str(python), "-m", "pip", "install", "-r", "requirements.lock"], source, run / "install.log")
            self.command([str(python), "-m", "pip", "install", "--no-deps", "-e", "."], source, run / "install.log")
        env = clean_environment()
        env.update(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        try:
            env["VELOCICYBER_TEST_ENGINE"] = str(self.settings.engine())
        except RuntimeError:
            pass
        report = run / "tests.xml"
        self.notify("Testing the staged harness with offline fixtures; the active source is unchanged…")
        self.command([str(python), "-m", "pytest", "-p", "pytest_asyncio.plugin", "-q",
                      "--basetemp=" + str(run / "fixtures"), "--junitxml=" + str(report)],
                     source, run / "tests.log", env=env)
        cases = ET.parse(report).findall(".//testcase")
        if not cases or all(case.find("skipped") is not None for case in cases) or any(
                case.find(kind) is not None for case in cases for kind in ("failure", "error")):
            raise RuntimeError("The staged harness did not pass its test suite.")
        self.notify(f"Harness tests passed: {len(cases)} cases ({sum(c.find('skipped') is not None for c in cases)} skipped).")
        return changed

    def preflight(self, installed):
        if self.runtime.is_symlink() or not self.runtime.resolve().is_relative_to(self.root):
            raise ValueError("Update storage must stay inside this installation.")
        if (self.runtime / PENDING).exists():
            raise RuntimeError("Finish recovery first: ./velocicyber.sh update --rollback-harness")
        kind, current = installed_source(self.root)
        if current != installed:
            raise RuntimeError("The installed source changed. Restart and review the update again.")
        if kind == "git":
            if git(self.root, "status", "--porcelain", "--untracked-files=all"):
                raise RuntimeError("Local source changes are present. Commit or save them before updating; nothing was overwritten.")
            if git(self.root, "branch", "--show-current") != "main":
                raise RuntimeError("Automatic source updates require the main branch.")
        return kind

    def fetch_git(self, target, run):
        helper = run / "askpass.py"
        private_write(helper, "#!/usr/bin/env python3\nimport os, sys\nprint('x-access-token' if 'Username' in sys.argv[-1] else os.environ.get('VELOCICYBER_UPDATE_TOKEN', ''))\n")
        helper.chmod(0o700)
        env = clean_environment()
        env.update(GIT_ASKPASS=str(helper), GIT_TERMINAL_PROMPT="0", VELOCICYBER_UPDATE_TOKEN=github_token(self.settings))
        try:
            git(self.root, "-c", "credential.helper=", "fetch", "--no-tags", REMOTE, target, env=env)
        finally:
            helper.unlink(missing_ok=True)
        if git(self.root, "rev-parse", "FETCH_HEAD") != target:
            raise RuntimeError("Git fetched an unexpected source revision.")
        git(self.root, "merge-base", "--is-ancestor", "HEAD", target)

    def check_files(self, before, after):
        for name, item in before.items():
            path = self.root / source_path(name)
            if not matches(path, item):
                raise RuntimeError("Local source differs from the installed release: " + name)
        for name in after:
            path = self.root / source_path(name)
            if not path.resolve().is_relative_to(self.root) or any(p.is_symlink() for p in (path, *path.parents) if p != self.root.parent):
                raise ValueError("A source path crosses a symlink.")
            if name not in before and path.exists():
                raise RuntimeError("Update would overwrite a local file: " + name)

    def write_source(self, source, before, after):
        for name in after:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".update-tmp")
            with temporary.open("xb") as output:
                output.write((source / name).read_bytes())
            temporary.chmod(after[name]["mode"])
            temporary.replace(path)
        for name in before.keys() - after.keys():
            (self.root / name).unlink()

    def restore(self, record):
        run = self.runtime / record["run"]
        if not run.resolve().is_relative_to(self.runtime):
            raise ValueError("Invalid update backup path.")
        if record["kind"] == "git":
            # --keep refuses to discard new local edits made after the update.
            git(self.root, "reset", "--keep", record["installed"])
        else:
            self.write_source(run / "backup", record["after"], record["before"])
            private_write(self.runtime / "source-install.json", record["old_receipt"])
        old_env = run / "previous-venv"
        if old_env.exists() or old_env.is_symlink():
            current = self.root / ".venv"
            if current.is_symlink() and current.resolve() == (run / "venv").resolve():
                current.unlink()
            elif current.exists():
                raise RuntimeError("The Python environment changed; restore the saved environment manually.")
            old_env.rename(current)
        (self.runtime / PENDING).unlink(missing_ok=True)

    def rollback(self):
        record_path = self.runtime / PENDING if (self.runtime / PENDING).exists() else self.runtime / BACKUP
        if not record_path.exists():
            raise RuntimeError("No harness update backup is available.")
        record = json.loads(record_path.read_text())
        if not (self.runtime / PENDING).exists():
            self.preflight(record["target"])
            self.check_files(record["after"], record["before"])
        self.restore(record)
        (self.runtime / BACKUP).unlink(missing_ok=True)
        self.notify("Previous harness restored. Credentials, models and conversations were preserved.")

    def run(self, plan=None, *, approved=False, rollback=False, stage_only=False):
        if not approved:
            raise PermissionError("Explicit approval is required before updating the harness.")
        with engine_lease(self.runtime, maintenance=True):
            if rollback:
                return self.rollback()
            if not plan or not all(revision(plan.get(key)) for key in ("installed", "target")):
                raise ValueError("Invalid source update plan.")
            kind = self.preflight(plan["installed"])
            run = self.runtime / "harness-updates" / uuid.uuid4().hex
            run.mkdir(mode=0o700, parents=True)
            source = run / "source"
            self.notify("Downloading Velocicyber " + plan["target"][:12] + "…")
            after = self.download(plan["target"], source)
            if kind == "git":
                before = {}
                for name in filter(None, git(self.root, "ls-files", "-z").split("\0")):
                    path = self.root / source_path(name)
                    before[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "mode": path.stat().st_mode & 0o777}
                self.fetch_git(plan["target"], run)
            else:
                before = self.download(plan["installed"], run / "baseline")
            self.check_files(before, after)
            changed_dependencies = self.validate(source, run)
            if stage_only:
                self.notify("Harness validation complete; source and dependencies remain unchanged.")
                return
            self.preflight(plan["installed"])
            self.check_files(before, after)
            for name in before:
                target = run / "backup" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / name, target)
            record = {**plan, "kind": kind, "run": str(run.relative_to(self.runtime)),
                      "before": before, "after": after,
                      "old_receipt": (self.runtime / "source-install.json").read_text() if kind == "archive" else ""}
            private_write(self.runtime / PENDING, json.dumps(record))
            try:
                if kind == "git":
                    git(self.root, "merge", "--ff-only", plan["target"])
                else:
                    self.write_source(source, before, after)
                    private_write(self.runtime / "source-install.json", json.dumps({"repository": REPOSITORY, "commit": plan["target"]}))
                if changed_dependencies:
                    current = self.root / ".venv"
                    current.rename(run / "previous-venv")
                    current.symlink_to(run / "venv", target_is_directory=True)
                    self.command([str(run / "venv/bin/python"), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(self.root)],
                                 self.root, run / "install.log")
                private_write(self.runtime / BACKUP, json.dumps(record))
                (self.runtime / PENDING).unlink()
            except BaseException:
                self.restore(record)
                raise
            self.notify("Velocicyber updated to " + plan["target"][:12] + ". Restart the harness to use it.")
