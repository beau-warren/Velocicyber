"""Publish a single deduplicated training view, after explicit opt-in."""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from velocicyber.datasets.pipeline import (
    PIPELINE_VERSION,
    Filters,
    Scrubber,
    canonical,
    digest,
    immutable_write,
    select_preference_rows,
    select_sft_rows,
    validate_messages,
)
from velocicyber.datasets.retention import cleanup_confirmed, cleanup_plan

TRAINING_FILE = re.compile(
    r"campaigns/([a-zA-Z0-9_-]{1,128})/(train|validation|preferences)\.jsonl"
)


def checked_manifest(body, expected_hash=None):
    manifest = json.loads(body)
    actual = digest({k: v for k, v in manifest.items() if k != "dataset_hash"})
    if actual != manifest["dataset_hash"] or (expected_hash and actual != expected_hash):
        raise ValueError("Dataset manifest hash mismatch.")
    return manifest


def training_rows(manifest, read):
    """Validate artifacts before they can participate in a publication."""
    result = []
    for name, expected in manifest["files"].items():
        match = TRAINING_FILE.fullmatch(name)
        if not match and name != "excluded.jsonl":
            raise ValueError("Unexpected path in dataset manifest.")
        body = read(name)
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("Derived artifact hash mismatch.")
        rows = [json.loads(line) for line in body.splitlines() if line.strip()]
        if len(rows) != manifest["counts"][name]:
            raise ValueError("Dataset row count mismatch.")
        if not match:
            continue
        campaign, kind = match.groups()
        for row in rows:
            meta = row["meta"]
            if meta["campaign_id"] != campaign or meta["split"] not in {"train", "validation"}:
                raise ValueError("Dataset campaign/split mismatch.")
            unverified = (
                meta.get("success", False) is None
                and meta.get("verification") == "unverified"
                and "quality_score" in meta
                and meta["quality_score"] is None
            )
            if kind != "preferences":
                if meta["split"] != kind or not (meta["success"] is True or unverified):
                    raise ValueError("Invalid supervised training record.")
            elif meta["success"] is not True or row.get("margin") != "verifier_graded":
                raise ValueError("Preferences require verified outcomes.")
            trajectories = (
                [row["chosen"], row["rejected"]] if kind == "preferences" else [row["messages"]]
            )
            if not all(validate_messages(messages) for messages in trajectories):
                raise ValueError("Invalid dataset tool-call sequence.")
            result.append((kind, row))
    return result


def current_files(records, filters=Filters(), preferred_ids=()):
    """Merge approved examples without retaining repeated snapshots or prefixes."""
    identities = {}
    for kind, row in records:
        key = (kind, row["id"])
        if key in identities and identities[key] != row:
            # Older resumed checkpoints could have been published without these
            # annotations. Upgrade only additive preparation metadata on identical
            # content; never resolve conflicting messages/outcomes by overwriting.
            annotations = {
                "context_compacted",
                "checkpoint_id",
                "segment_index",
                "continuation_start_index",
                "context_validation",
                "context_utf8_bytes_with_reserve",
            }
            previous = identities[key]
            old_meta, new_meta = previous["meta"], row["meta"]
            old_base = {
                **previous,
                "meta": {k: v for k, v in old_meta.items() if k not in annotations},
            }
            new_base = {**row, "meta": {k: v for k, v in new_meta.items() if k not in annotations}}
            old_notes = {k: v for k, v in old_meta.items() if k in annotations}
            new_notes = {k: v for k, v in new_meta.items() if k in annotations}
            if old_base != new_base or any(
                old_notes[k] != new_notes[k] for k in old_notes.keys() & new_notes.keys()
            ):
                raise ValueError("Conflicting dataset records share an ID.")
            row = {**row, "meta": {**new_meta, **old_notes}}
        identities[key] = row
    sft, _ = select_sft_rows(
        [r for (kind, _), r in identities.items() if kind != "preferences"], filters, preferred_ids
    )
    preferences = select_preference_rows(
        [r for (kind, _), r in identities.items() if kind == "preferences"], preferred_ids
    )
    grouped = defaultdict(list)
    for kind, rows in [(None, sft), ("preferences", preferences)]:
        for row in rows:
            meta = row["meta"]
            name = f"campaigns/{meta['campaign_id']}/{kind or meta['split']}.jsonl"
            grouped[name].append(row)
    bodies = {
        name: b"".join(canonical(row) + b"\n" for row in rows)
        for name, rows in sorted(grouped.items())
    }
    counts = {name: len(rows) for name, rows in sorted(grouped.items())}
    return bodies, counts


def dataset_card(existing, files):
    from huggingface_hub import DatasetCard

    card = DatasetCard(existing or "---\nlicense: apache-2.0\n---\n")
    configs = defaultdict(list)
    for name in sorted(files):
        campaign, kind = TRAINING_FILE.fullmatch(name).groups()
        config = ("preferences" if kind == "preferences" else "sft") + "-" + campaign
        configs[config].append(
            {"split": "train" if kind == "preferences" else kind, "path": "current/" + name}
        )
    ordered = sorted(configs, key=lambda name: (not name.startswith("sft-"), name))
    card.data.configs = [
        {"config_name": name, "data_files": configs[name], **({"default": True} if i == 0 else {})}
        for i, name in enumerate(ordered)
    ]
    # Old auto-generated statistics/features can describe the archived schema.
    card.data.__dict__.pop("dataset_info", None)
    heading = "## Velocicyber current training data"
    if heading not in card.text:
        card.text = (
            card.text.rstrip()
            + "\n\n"
            + heading
            + "\n\n"
            + (
                "The dataset configurations select only current/ files. Each campaign has\n"
                "its own supervised/preference configuration. Historical versions/ files\n"
                "are archives; do not combine them with the current training view.\n"
                "Pin the Hugging Face commit revision for reproducible training.\n"
            )
        )
    policy_heading = "## Outcome labels"
    if policy_heading not in card.text:
        card.text = (
            card.text.rstrip()
            + "\n\n"
            + policy_heading
            + "\n\nCompleted conversations without a machine verifier have meta.success=null,\n"
            "meta.quality_score=null and meta.verification=unverified. Completion does\n"
            "not establish correctness. Filter meta.success=true for verified-only SFT.\n"
            "Preference pairs require verified outcomes. Blank/error replies, known\n"
            "failures and incomplete context are excluded from SFT.\n"
        )
    # Update only our old generated policy sentence, preserving the owner's text.
    card.text = card.text.replace(
        "failures, incomplete context and oversized examples are excluded from SFT.",
        "failures and incomplete context are excluded from SFT.",
    )
    context_heading = "## Checkpoints and training preparation"
    if context_heading not in card.text:
        card.text = (
            card.text.rstrip() + "\n\n" + context_heading + "\n\n"
            "These are canonical conversations, without a student chat template. Before\n"
            "training, tokenize messages AND tools with the target tokenizer/template,\n"
            "then split or reject examples exceeding its context window; never silently\n"
            "truncate. UTF-8 bytes and teacher usage are not student token counts. Rows\n"
            "marked meta.context_validation=requires_target_tokenizer exceed the old\n"
            "byte heuristic, which does not establish token overflow.\n\n"
            "Compacted continuations retain the actual checkpoint and role/tool schema.\n"
            "meta.context_compacted=true identifies them; checkpoint_id identifies the\n"
            "sanitized checkpoint, and segment_index counts known compactions (null when\n"
            "unknown). Missing context_compacted in older rows does not prove no compaction.\n"
            "For these rows, only assistant messages at or after the zero-based\n"
            "meta.continuation_start_index are targets. Mask earlier messages, checkpoints,\n"
            "user messages and tool results from the loss. Summary generation is not a\n"
            "training target. Use prompt/completion masking in your training adapter;\n"
            "metadata alone does not apply a loss mask. Keep the existing task split.\n"
        )
    return str(card).encode()


@dataclass
class PublicationPlan:
    revision: str
    files: dict[str, bytes]
    training_hash: str
    counts: dict[str, int]
    cleanup: dict


class Uploader:
    def __init__(self, settings, api=None):
        self.settings = settings
        self.api = api

    def local_artifact(self, artifact):
        root = (self.settings.data / "derived").resolve()
        artifact = artifact.resolve()
        if artifact.parent != root:
            raise ValueError("Only derived dataset snapshots can be uploaded.")
        manifest = checked_manifest((artifact / "manifest.json").read_bytes(), artifact.name)

        def read(name):
            file = (artifact / name).resolve()
            if not file.is_relative_to(artifact):
                raise ValueError("Derived artifact path escapes snapshot.")
            return file.read_bytes()

        return manifest, training_rows(manifest, read)

    def client(self):
        from huggingface_hub import HfApi

        if not self.settings.dataset_repo:
            raise ValueError("Set FINETUNING_REPO in Modules.txt first.")
        return self.api or HfApi(token=self.settings.require_secret("HUGGINGFACE_API"))

    def plan(self, artifact):
        """Read-only preview at a pinned revision; never creates a remote commit."""
        # Downloaded merge inputs must not leave another local training-data copy.
        with TemporaryDirectory(prefix="hf-merge-", dir=self.settings.runtime) as cache:
            return self._plan(artifact, cache)

    def _plan(self, artifact, cache_dir):
        manifest, records = self.local_artifact(artifact)
        api = self.client()
        info = api.dataset_info(self.settings.dataset_repo, timeout=30)
        names = {file.rfilename for file in info.siblings}
        cache = {}

        def read(name):
            if name not in names:
                raise ValueError("Remote dataset references a missing file.")
            if name not in cache:
                path = api.hf_hub_download(
                    repo_id=self.settings.dataset_repo,
                    repo_type="dataset",
                    filename=name,
                    revision=info.sha,
                    cache_dir=cache_dir,
                )
                cache[name] = Path(path).read_bytes()
            return cache[name]

        old, migrated, preferred_ids = {}, set(), set()
        if "current/manifest.json" in names:
            old = checked_manifest(read("current/manifest.json"))
            if old.get("publication_format") != "current-v1":
                raise ValueError("Unsupported current dataset format.")
            remote_rows = training_rows(old, lambda name: read("current/" + name))
            records += remote_rows
            preferred_ids.update(row["id"] for _, row in remote_rows)
            migrated.update(old.get("legacy_versions", []))
        legacy = sorted(
            name
            for name in names
            if re.fullmatch(r"versions/[a-f0-9]{64}/manifest\.json", name)
            and name.split("/")[1] not in migrated
        )
        for name in legacy:
            version = name.split("/")[1]
            legacy_manifest = checked_manifest(read(name), version)
            records += training_rows(
                legacy_manifest, lambda part, version=version: read(f"versions/{version}/{part}")
            )
            migrated.add(version)
        filters = Filters(**manifest["filters"])
        bodies, counts = current_files(records, filters, preferred_ids)
        cleanup = cleanup_plan(
            self.settings.data,
            artifact,
            manifest,
            bodies,
            Scrubber(
                self.settings.secrets.values(),
                self.settings.options.get("PRIVATE_HOSTS", "").split(","),
            ),
        )
        hashes = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
        training_hash = digest(hashes)
        existing_card = read("README.md").decode() if "README.md" in names else ""
        card = dataset_card(existing_card, bodies)
        if old.get("files") == hashes and card == existing_card.encode():
            return PublicationPlan(info.sha, {}, training_hash, counts, cleanup)
        current = {
            "publication_format": "current-v1",
            "pipeline_version": PIPELINE_VERSION,
            "filters": asdict(filters),
            "files": hashes,
            "counts": counts,
            "legacy_versions": sorted(migrated),
            "source_snapshots": sorted(
                set(old.get("source_snapshots", [])) | migrated | {artifact.name}
            ),
        }
        current["dataset_hash"] = digest(current)
        files = {
            "current/" + name: body
            for name, body in bodies.items()
            if old.get("files", {}).get(name) != hashes[name]
        }
        files["current/manifest.json"] = canonical(current) + b"\n"
        if card != existing_card.encode():
            files["README.md"] = card
        return PublicationPlan(info.sha, files, training_hash, counts, cleanup)

    def finish_cleanup(self, receipt):
        try:
            result = cleanup_confirmed(self.settings.data, receipt["cleanup_plan"])
        except (OSError, ValueError) as exc:
            # The upload succeeded. Retain the receipt so cleanup can be retried
            # without resending the training data or reporting a failed upload.
            result = {"complete": False, "error": str(exc)}
        return {**receipt, "local_cleanup": result}

    def upload(self, artifact: Path, *, approved=False):
        if not approved:
            raise PermissionError("Approve Hugging Face uploads before publishing a dataset.")
        artifact = artifact.resolve()
        if artifact.parent != (self.settings.data / "derived").resolve():
            raise ValueError("Only derived dataset snapshots can be uploaded.")
        marker = (
            self.settings.data
            / "uploaded"
            / "current-v1"
            / (digest([self.settings.dataset_repo, artifact.name]) + ".retention-v1.json")
        )
        if marker.exists():
            return self.finish_cleanup(json.loads(marker.read_text()))
        _, records = self.local_artifact(artifact)
        if not records:
            return None
        plan = self.plan(artifact)
        commit = plan.revision
        if plan.files:
            from huggingface_hub import CommitOperationAdd

            # A concurrent writer aborts this commit; never silently overwrite it
            # or retry a provider error. The next user-approved attempt re-merges.
            result = self.client().create_commit(
                repo_id=self.settings.dataset_repo,
                repo_type="dataset",
                parent_commit=plan.revision,
                operations=[
                    CommitOperationAdd(path_in_repo=name, path_or_fileobj=body)
                    for name, body in plan.files.items()
                ],
                commit_message=f"Update unique Velocicyber training data {plan.training_hash[:12]}",
            )
            commit = result.oid
        if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40,64}", commit):
            raise RuntimeError("Hugging Face did not confirm a commit; local data was retained.")
        receipt = {
            "dataset_hash": artifact.name,
            "training_hash": plan.training_hash,
            "commit": commit,
            "repo": self.settings.dataset_repo,
            "publication_format": "current-v1",
            "unchanged": not bool(plan.files),
            "cleanup_plan": plan.cleanup,
        }
        immutable_write(marker, canonical(receipt) + b"\n")
        return self.finish_cleanup(receipt)
