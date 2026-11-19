"""Optional first-run setup. Never launches GPUs or enables data uploads."""

import getpass
import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

from velocicyber.config import Settings
from velocicyber.datasets.pipeline import Scrubber
from velocicyber.desktop import desktop_directory, install_shortcut
from velocicyber.models.pi_profiles import BUGTRACE_PI_REPO, LFM3_PI_REPO, PI_MODEL_REPO
from velocicyber.ssh_setup import lambda_ssh_options

CREDENTIALS = (
    ("LAMBDA_API", "Lambda API Key"),
    ("ADVERSARIAL_API", "Adverserial.ai API key"),
    ("HUGGINGFACE_API", "Hugging Face API Key"),
)
DATASETS = (
    ("FINETUNING_REPO", "Main LLM fine-tuning dataset"),
    ("SENTINEL_FINETUNING_REPO", "Sentinel LLM fine-tuning dataset"),
)
MARKER = "setup-complete.json"


def private_assignments(path, changes):
    """Update local assignments atomically; reject symlinks and control characters."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symlink: {path.name}")
    for name, value in changes.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError("Invalid configuration field.")
        if any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("Configuration values cannot contain control characters.")
    lines = path.read_text().splitlines() if path.exists() else []
    kept = []
    for line in lines:
        key = line.strip().removeprefix("export ").partition("=")[0].strip()
        if key not in changes:
            kept.append(line)
    kept.extend(name + "=" + value for name, value in changes.items())
    descriptor, temporary = tempfile.mkstemp(prefix=".setup-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write("\n".join(kept) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def confirm(prompt, input_fn):
    return input_fn(prompt + " [y/N]: ").strip().lower() in {"y", "yes"}


def dataset_choice(label, current, input_fn):
    from huggingface_hub.utils import validate_repo_id

    while True:
        suffix = f" (current: {current}; Enter keeps it)" if current else " (Enter to skip for now)"
        value = input_fn(label + " [account/dataset-name]" + suffix + ": ").strip()
        if not value or value.lower() in {"skip", "skip for now"}:
            return current
        try:
            validate_repo_id(value)
        except ValueError:
            pass
        else:
            if value.count("/") == 1:
                return value
        print("Use a full Hugging Face dataset ID, such as your-account/coding-data.")


def install_component(root, script, *arguments):
    try:
        subprocess.run([sys.executable, str(root / "scripts/install" / script), *arguments],
                       cwd=root, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("Optional installer failed: " + script + ". Run setup again to retry.") from None


def run_setup(root, *, force=False, input_fn=input, secret_fn=getpass.getpass, interactive=None):
    root = Path(root).resolve()
    for name in (".env", "Modules.txt", ".runtime"):
        if (root / name).is_symlink():
            raise ValueError("Setup requires regular local configuration paths: " + name)
    settings = Settings.load(root)
    marker = settings.runtime / MARKER
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if marker.is_file() and not force:
        try:
            saved = json.loads(marker.read_text())
        except (ValueError, OSError):
            saved = {}
        if not isinstance(saved, dict):
            saved = {}
        if not interactive or saved.get("lambda_ssh_setup") == 1:
            return False
        print("Additional setup: Lambda SSH access. Existing provider and dataset settings are kept.")
        if options := lambda_ssh_options(root, settings.options, input_fn):
            private_assignments(root / "Modules.txt", options)
        saved["lambda_ssh_setup"] = 1
        from velocicyber.engine.sessions import private_write
        private_write(marker, json.dumps(saved) + "\n")
        return True
    if not interactive:
        if force:
            raise ValueError("Run setup in a terminal, or copy .env.example and Modules.example.txt manually.")
        return False
    print("\nVelocicyber setup — skip any item for now. No GPUs will be launched.")
    print("Credentials stay in this installation's .env with owner-only permissions.")
    changes = {}
    for name, label in CREDENTIALS:
        configured = bool(settings.secrets.get(name))
        if configured and not force:
            print(label + ": already configured (value hidden).")
            continue
        suffix = "Enter keeps current" if configured else "Enter to skip for now"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                value = secret_fn(label + " (hidden; " + suffix + "): ").strip()
        except getpass.GetPassWarning:
            raise ValueError("Hidden input is unavailable. Run setup in a terminal.") from None
        if value and value.lower() not in {"skip", "skip for now"}:
            if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
                raise ValueError(label + " must be a single token; its value was not displayed.")
            changes[name] = value
    if changes:
        private_assignments(root / ".env", changes)
    elif (root / ".env").is_file() and not (root / ".env").is_symlink():
        (root / ".env").chmod(0o600)
    settings = Settings.load(root)
    print("\nOptional dataset destinations. Collection and uploads start OFF.")
    datasets = {}
    for name, label in DATASETS:
        while True:
            value = dataset_choice(label, settings.options.get(name, ""), input_fn)
            if name == "SENTINEL_FINETUNING_REPO" and value and value == datasets.get("FINETUNING_REPO"):
                print("Use a separate dataset for Sentinel observations.")
                continue
            datasets[name] = value
            break
    if any(datasets.values()):
        private_assignments(root / "Modules.txt", datasets)
    if settings.secrets.get("HUGGINGFACE_API") and any(datasets.values()):
        if confirm("Create missing datasets as PRIVATE on Hugging Face now? Existing datasets keep their visibility", input_fn):
            from huggingface_hub import HfApi

            api = HfApi(token=settings.require_secret("HUGGINGFACE_API"))
            scrubber = Scrubber(settings.secrets.values())
            for repo in dict.fromkeys(value for value in datasets.values() if value):
                try:
                    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
                    print("Dataset available: " + repo)
                except Exception as exc:
                    print("Dataset creation failed: " + scrubber.text(str(exc)))
    elif any(datasets.values()):
        print("Dataset names saved. Add a Hugging Face key and create the datasets before uploading.")
    print("\nOptional runtime installation. You can return with ./velocicyber.sh setup.")
    try:
        settings.engine()
        engine_missing = False
    except RuntimeError:
        engine_missing = True
    if engine_missing and confirm("Download and install the pinned execution engine (about 234 MB)?", input_fn):
        install_component(root, "install_engine.py")
    if settings.secrets.get("ADVERSARIAL_API"):
        if confirm("Install or refresh the Adverserial confidential API client? Requires Node.js/npm and network access", input_fn):
            install_component(root, "install_api_gateway.py")
    print("\nOptional Pi 5 models (16 GB RAM, Linux ARM64; one coding model at a time):")
    print("1: Qwen3.5 IQK — 12.23 GB; 2: BugTrace APEX Mini — 12.26 GB; 3: Qwen + BugTrace.")
    print("4: Blackfrost LFM3 1.7B — 3.48 GB source download, converted locally to ~1.8–2.0 GB Q8; 5: all three.")
    print("LFM3: allow ~3 GB installed and 12 GiB free during setup; conversion needs 6 GiB available RAM.")
    print("Qwen/BugTrace: allow 25 GiB free for one, or about 40 GiB for both plus build space.")
    print("Weights are pinned and checksum-verified. Review the publishers' model licenses first.")
    print("https://huggingface.co/" + PI_MODEL_REPO)
    print("https://huggingface.co/" + BUGTRACE_PI_REPO)
    print("https://huggingface.co/" + LFM3_PI_REPO)
    while True:
        choice = input_fn("Download/build Pi models now? [1/2/3/4/5; Enter to skip for now]: ").strip().lower()
        if choice in {"", "0", "skip", "skip for now"}:
            models = ()
            break
        if choice in {"1", "2", "3", "4", "5"}:
            models = {"1": (PI_MODEL_REPO,), "2": (BUGTRACE_PI_REPO,),
                      "3": (PI_MODEL_REPO, BUGTRACE_PI_REPO), "4": (LFM3_PI_REPO,),
                      "5": (PI_MODEL_REPO, BUGTRACE_PI_REPO, LFM3_PI_REPO)}[choice]
            break
        print("Choose 1–5, or Enter to skip for now.")
    for model in models:
        install_component(root, "install_pi_model.py", "--model", model)
    directory = desktop_directory()
    if directory is not None:
        if confirm("Create a desktop launcher and Velocicyber icon? Enter skips for now", input_fn):
            try:
                shortcut = install_shortcut(root, directory)
                print("Desktop launcher ready: " + str(shortcut))
                print("If your desktop asks, choose Allow Launching or Trust and Launch.")
            except (OSError, ValueError) as exc:
                print("Desktop shortcut could not be created: " + str(exc))
                print("Continue normally; run ./velocicyber.sh setup to retry later.")
    else:
        print("Desktop shortcut skipped: no Linux desktop folder was found.")
    settings = Settings.load(root)
    if options := lambda_ssh_options(root, settings.options, input_fn):
        private_assignments(root / "Modules.txt", options)
    payload = {"schema": 1, "completed_at": datetime.now(timezone.utc).isoformat(),
               "lambda_ssh_setup": 1}
    descriptor, temporary = tempfile.mkstemp(prefix="setup-", dir=settings.runtime)
    try:
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(payload, stream)
        os.replace(temporary, marker)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print("Setup saved. Use /data-collection and /upload-perms to opt in when ready.\n")
    return True
