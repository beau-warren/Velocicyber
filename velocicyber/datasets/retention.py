"""Remove acknowledged collection files without touching pending work or sessions."""

import hashlib
import json
import re
import shutil
from collections import defaultdict

from velocicyber.datasets.pipeline import digest, normalized_messages, validate_messages


def guarded_path(root, relative):
    path = root / relative
    if path.is_absolute() and not path.is_relative_to(root):
        raise ValueError("Cleanup path escapes the collection folder.")
    if ".." in path.parts:
        raise ValueError("Invalid cleanup path.")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("Local cleanup refused a symlink.")
        if parent == root:
            break
    return path


def cleanup_plan(root, artifact, manifest, published_files, scrubber):
    """Capture exact files whose sanitized conversations are represented remotely.

    Near duplicates and task-capped examples are retained unless the complete
    message sequence is actually present (possibly as a prefix of a continuation).
    This is a plan only; it must not run until the commit succeeds.
    """
    published = defaultdict(list)
    for name, body in published_files.items():
        for line in body.splitlines():
            row = json.loads(line)
            key = (row["meta"]["campaign_id"], digest(row["tools"]))
            messages = (
                [row["chosen"], row["rejected"]]
                if name.endswith("preferences.jsonl")
                else [row["messages"]]
            )
            published[key].extend(normalized_messages(m) for m in messages)

    excluded = {
        row["episode_id"]: row["reason"]
        for line in (artifact / "excluded.jsonl").read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    files, episodes = {}, set()
    for source in manifest["source_episodes"]:
        eid = source["episode_id"]
        if not re.fullmatch(r"ep_[a-zA-Z0-9_-]+", eid):
            raise ValueError("Invalid cleanup episode ID.")
        if excluded.get(eid) not in (None, "duplicate", "conversation_prefix", "verified_failure"):
            continue
        raw = guarded_path(root, f"raw/{eid}.jsonl")
        if not raw.exists():
            continue
        body = raw.read_bytes()
        episode = json.loads(body)
        if digest(episode) != source["sha256"]:
            raise ValueError("Raw episode changed since dataset preparation.")
        messages = scrubber.clean(episode.get("messages", []))
        if not validate_messages(messages):
            continue
        key = (episode["campaign_id"], digest(scrubber.clean(episode.get("tools", []))))
        normalized = normalized_messages(messages)
        if not any(normalized == other[: len(normalized)] for other in published[key]):
            continue
        sidecar = guarded_path(root, f"raw/{eid}.sha256")
        raw_hash = hashlib.sha256(body).hexdigest()
        if sidecar.read_text() != raw_hash:
            raise ValueError("Raw episode integrity failure before cleanup.")
        files[f"raw/{eid}.jsonl"] = raw_hash
        files[f"raw/{eid}.sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        episodes.add(eid)

    snapshots = {}
    # Only snapshots already present before the commit may be pruned. Their
    # remaining pending examples can be re-derived from the untouched raw files.
    derived = guarded_path(root, "derived")
    for folder in sorted(derived.glob("*")):
        if not re.fullmatch(r"[a-f0-9]{64}", folder.name):
            continue
        manifest_path = guarded_path(root, f"derived/{folder.name}/manifest.json")
        if not manifest_path.exists():
            continue
        body = manifest_path.read_bytes()
        saved = json.loads(body)
        if digest({k: v for k, v in saved.items() if k != "dataset_hash"}) != folder.name:
            raise ValueError("Snapshot integrity failure before cleanup.")
        if episodes.intersection(s["episode_id"] for s in saved["source_episodes"]):
            snapshots[f"derived/{folder.name}"] = hashlib.sha256(body).hexdigest()
    return {"episodes": sorted(episodes), "files": files, "snapshots": snapshots}


def cleanup_confirmed(root, plan):
    """Idempotent, hash-checked deletion using a persisted upload acknowledgment."""
    files, snapshots = [], []
    # Preflight all targets before deleting anything. A changed file is pending
    # data, never something to remove on the strength of an older receipt.
    for relative, expected in plan["files"].items():
        if not re.fullmatch(r"raw/ep_[a-zA-Z0-9_-]+\.(jsonl|sha256)", relative):
            raise ValueError("Invalid raw cleanup target.")
        path = guarded_path(root, relative)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Local collection file changed; cleanup deferred.")
        files.append(path)
    for relative, expected in plan["snapshots"].items():
        if not re.fullmatch(r"derived/[a-f0-9]{64}", relative):
            raise ValueError("Invalid snapshot cleanup target.")
        path = guarded_path(root, relative)
        if path.exists():
            manifest = guarded_path(root, relative + "/manifest.json")
            if manifest.exists() and hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
                raise ValueError("Local snapshot changed; cleanup deferred.")
            snapshots.append(path)
    # Remove derived copies first. If cleanup is interrupted, the receipt still
    # contains enough information to resume without another upload.
    for path in snapshots:
        shutil.rmtree(path)
    for path in files:
        path.unlink(missing_ok=True)
    return {"complete": True, "episodes": len(plan["episodes"])}
