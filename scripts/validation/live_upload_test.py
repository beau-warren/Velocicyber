"""Approved current-view upload validation; preserves remote archive files."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

from velocicyber.config import Settings
from velocicyber.datasets.pipeline import validate_messages
from velocicyber.datasets.upload import Uploader


def inventory(info):
    return {
        file.rfilename: {
            "blob_id": file.blob_id,
            "lfs_sha256": file.lfs.sha256 if file.lfs else None,
            "size": file.size,
        }
        for file in info.siblings
    }


def main(args):
    settings = Settings.load(Path(__file__).resolve().parents[2])
    artifact = settings.data / "derived" / args.dataset_hash
    manifest = json.loads((artifact / "manifest.json").read_text())
    for name in manifest["files"]:
        text = (artifact / name).read_text()
        if any(secret and secret in text for secret in settings.secrets.values()):
            raise RuntimeError("A configured secret survived curation.")
        if name.endswith(("train.jsonl", "validation.jsonl")):
            for line in text.splitlines():
                if not validate_messages(json.loads(line)["messages"]):
                    raise RuntimeError("Invalid message contract in upload candidate.")
    api = HfApi(token=settings.require_secret("HUGGINGFACE_API"))
    before = api.dataset_info(settings.dataset_repo, files_metadata=True)
    existing = inventory(before)
    report_path = settings.runtime / "live-upload-test.json"
    report = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else {"repo": settings.dataset_repo, "tests": []}
    )
    attempt = {
        "dataset_hash": artifact.name,
        "before_revision": before.sha,
        "before_files": existing,
        "private": before.private,
    }
    report["tests"].append(attempt)

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        report_path.chmod(0o600)

    save()
    try:
        uploader = Uploader(settings, api)
        plan = uploader.plan(artifact)
        receipt = uploader.upload(artifact, approved=True)
        if not receipt:
            raise RuntimeError("The candidate has no eligible examples.")
        attempt["receipt"] = receipt
        save()
        after = api.dataset_info(settings.dataset_repo, files_metadata=True)
        current = inventory(after)
        preserved = {
            name: value
            for name, value in existing.items()
            if name != "README.md" and not name.startswith("current/")
        }
        if any(current.get(name) != value for name, value in preserved.items()):
            raise RuntimeError("An archived or unrelated remote file changed during upload.")
        for name, body in plan.files.items():
            remote = current[name]
            if remote["lfs_sha256"]:
                matches = hashlib.sha256(body).hexdigest() == remote["lfs_sha256"]
            else:
                matches = (
                    hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
                    == remote["blob_id"]
                )
            if not matches:
                raise RuntimeError("Remote content hash does not match the prepared artifact.")

        # A recorded receipt must suppress a second commit without contacting the provider.
        class NoRequests:
            def create_commit(self, **kwargs):
                raise AssertionError("A duplicate upload attempted a second commit.")

        assert Uploader(settings, NoRequests()).upload(artifact, approved=True) == receipt
        attempt.update(
            passed=True,
            after_revision=after.sha,
            preserved_files=len(preserved),
            new_files=len(current) - len(existing),
            duplicate_commit_suppressed=True,
        )
        print(json.dumps({k: v for k, v in attempt.items() if k != "before_files"}, indent=2))
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_hash")
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args()
    if not args.approved:
        parser.error("Use --approved only after explicit user approval.")
    if len(args.dataset_hash) != 64 or any(c not in "0123456789abcdef" for c in args.dataset_hash):
        parser.error("Expected a SHA256 dataset directory name.")
    main(args)
