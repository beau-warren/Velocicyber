"""Separate, immutable observer records and append-only, confirmed HF publication."""

import hashlib
import json
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from velocicyber.datasets.pipeline import Scrubber, canonical, immutable_write

DEFAULT_REPO = ""


class SentinelStore:
    def __init__(self, settings, api=None):
        self.settings, self.api = settings, api
        self.root = settings.root / "velocicyber_sentinel-data"
        self.scrubber = Scrubber(
            settings.secrets.values(), settings.options.get("PRIVATE_HOSTS", "").split(",")
        )

    def check_paths(self):
        for path in (self.root, self.root / "raw", self.root / "uploaded"):
            if path.is_symlink():
                raise ValueError("Sentinel data paths must not be symlinks.")

    def append(self, record):
        self.check_paths()
        body = canonical(record) + b"\n"
        key = hashlib.sha256(body).hexdigest()
        path = self.root / "raw" / (key + ".jsonl")
        if not path.exists():
            immutable_write(path, body)
        return path

    def upload(self, *, approved=False, allow_compaction=True):
        if not approved:
            raise PermissionError("Approve dataset uploads in /upload-perms first.")
        self.check_paths()
        local = []
        for path in sorted((self.root / "raw").glob("*.jsonl")):
            if path.is_symlink():
                raise ValueError("Sentinel records must not be symlinks.")
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != path.stem:
                raise ValueError("Sentinel record checksum mismatch.")
            record = json.loads(body)
            session_id = record.get("session_id", "")
            compacted = record.get("context_compacted", False)
            if re.fullmatch(r"[a-f0-9]{32}", session_id):
                metadata = self.settings.runtime / "conversations" / session_id / "session.json"
                if metadata.is_file():
                    compacted |= json.loads(metadata.read_text()).get("context_compacted", False)
            if compacted and not allow_compaction:
                continue
            clean = canonical(self.scrubber.clean(record)) + b"\n"
            remote = "observations/" + hashlib.sha256(clean).hexdigest() + ".jsonl"
            local.append((path, remote, clean))
        if not local:
            return None
        repo = self.settings.options.get("SENTINEL_FINETUNING_REPO", DEFAULT_REPO)
        if not repo or repo == self.settings.dataset_repo:
            raise ValueError("Sentinel data requires a separate Hugging Face dataset.")
        from huggingface_hub import CommitOperationAdd, HfApi

        api = self.api or HfApi(token=self.settings.require_secret("HUGGINGFACE_API"))
        info = api.dataset_info(repo, timeout=30)
        names = {p.rfilename for p in info.siblings}
        with TemporaryDirectory(prefix="sentinel-upload-", dir=self.settings.runtime) as cache:
            for _, name, body in local:
                if name in names:
                    remote = api.hf_hub_download(
                        repo_id=repo, repo_type="dataset", filename=name,
                        revision=info.sha, cache_dir=cache,
                    )
                    if Path(remote).read_bytes() != body:
                        raise ValueError("Remote sentinel shard changed; local records retained.")
        files = {name: body for _, name, body in local if name not in names}
        # Content-addressed shards preserve every earlier upload and survive retries.
        # No readme or metadata owned by the user is replaced.
        if "README.md" not in names:
            files["README.md"] = (
                "# Velocicyber sentinel observations\n\n"
                "These sanitized records contain scope, observed agent events, model identity, "
                "latency and Sentinel predictions. Predictions are NOT verified labels. "
                "Review and correct them before supervised training. Monitor failures are "
                "not scope violations. No hidden reasoning unavailable to the harness is captured.\n"
            ).encode()
        commit = info.sha
        if files:
            result = api.create_commit(
                repo_id=repo, repo_type="dataset", parent_commit=info.sha,
                operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=b)
                            for n, b in files.items()],
                commit_message="Append Velocicyber sentinel observations",
            )
            commit = result.oid
        if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40,64}", commit):
            raise RuntimeError("No confirmed HF commit; sentinel records kept locally.")
        receipt = {"repo": repo, "commit": commit, "records": len(local),
                   "files": [name for _, name, _ in local]}
        marker = self.root / "uploaded" / (hashlib.sha256(canonical(receipt)).hexdigest() + ".json")
        if not marker.exists():
            immutable_write(marker, canonical(receipt) + b"\n")
        for path, _, _ in local:
            path.unlink()  # Only after the remote commit and durable local receipt.
        return receipt

    def delete(self, *, approved=False):
        if not approved:
            raise PermissionError("Confirm local fine-tuning data deletion first.")
        self.check_paths()
        if self.root.exists():
            shutil.rmtree(self.root)
            return True
        return False
