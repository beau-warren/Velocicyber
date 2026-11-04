"""Immutable raw episodes and deterministic, versioned derived artifacts."""

import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

from velocicyber.engine.compaction import (
    checkpoint_index,
    compaction_exchanges,
    valid_continuation,
    was_compacted,
)

PIPELINE_VERSION = "1.4.0"


def dataset_summary(artifact: Path):
    """Describe the prepared rows and exclusions without reading raw conversations."""
    manifest = json.loads((artifact / "manifest.json").read_text())
    reasons = Counter()
    excluded = artifact / "excluded.jsonl"
    if excluded.exists():
        for line in excluded.read_text().splitlines():
            if line.strip():
                reasons[json.loads(line)["reason"]] += 1
    return {
        "eligible": sum(
            count for name, count in manifest["counts"].items() if name != "excluded.jsonl"
        ),
        "excluded": sum(reasons.values()),
        "exclusion_reasons": dict(reasons),
        "unverified": manifest.get("verification_counts", {}).get("unverified", 0),
    }


def delete_local_finetuning_data(project_root: Path, *, approved=False):
    """Explicit local purge; never follow a dataset-root symlink or contact HF."""
    if not approved:
        raise PermissionError("Confirm deletion of all local fine-tuning data first.")
    target = project_root.resolve() / "velocicyber_finetuning-data"
    if target.is_symlink():
        raise ValueError("The fine-tuning data folder is a symlink; deletion was refused.")
    if not target.exists():
        return False
    if not target.is_dir():
        raise ValueError("The fine-tuning data path is not a directory.")
    # On the Pi, rmtree uses directory descriptors and does not traverse symlinks.
    shutil.rmtree(target)
    return True


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def immutable_write(path: Path, body: bytes):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # O_EXCL prevents accidental overwrite. fsync persists data before publication.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    path.chmod(0o400)


class RawStore:
    def __init__(self, root: Path):
        self.root = root

    def append(self, episode: dict) -> Path:
        if not re.fullmatch(r"ep_[a-zA-Z0-9_-]+", episode["episode_id"]):
            raise ValueError("Invalid episode ID")
        path = self.root / "raw" / (episode["episode_id"] + ".jsonl")
        immutable_write(path, canonical(episode) + b"\n")
        immutable_write(
            path.with_suffix(".sha256"), hashlib.sha256(path.read_bytes()).hexdigest().encode()
        )
        return path

    def episodes(self):
        for path in sorted((self.root / "raw").glob("*.jsonl")):
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != path.with_suffix(".sha256").read_text():
                raise ValueError(f"Raw log integrity failure: {path.name}")
            yield json.loads(raw)


class Scrubber:
    def __init__(self, secrets=(), private_hosts=()):
        self.secrets = sorted({v for v in secrets if len(v) >= 4}, key=len, reverse=True)
        self.private_hosts = tuple(h for h in private_hosts if h)

    def text(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED_SECRET]")
        for host in self.private_hosts:
            text = text.replace(host, "[REDACTED_HOST]")
        text = re.sub(
            r"(?i)\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|ai|dev|internal|local|corp)\b",
            "[REDACTED_HOST]",
            text,
        )
        text = re.sub(
            r"(?i)(https?://)(?:[^\s/@]+@)?[a-z0-9.-]+(?::\d+)?", r"\1[REDACTED_HOST]", text
        )
        text = re.sub(
            r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
            "[REDACTED_PRIVATE_KEY]",
            text,
            flags=re.S,
        )
        text = re.sub(r"(?i)\b(?:sk-|hf_|ghp_|github_pat_)[a-z0-9_\-]{8,}", "[REDACTED_KEY]", text)
        text = re.sub(r"(?i)(bearer\s+)[a-z0-9._~+/-]+=*", r"\1[REDACTED_TOKEN]", text)
        text = re.sub(
            r'(?i)((?:api[_-]?key|password|secret|access[_-]?token)["\s]*[:=]["\s]*)[^\s"\x27,}]+',
            r"\1[REDACTED_SECRET]",
            text,
        )

        def address(match):
            try:
                ipaddress.ip_address(match[0])
                return "[REDACTED_IP]"
            except ValueError:
                return match[0]

        text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", address, text)
        text = re.sub(r"(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F:.]+", address, text)

        def entropy(match):
            token = match[0]
            counts = Counter(token)
            bits = -sum((n / len(token)) * math.log2(n / len(token)) for n in counts.values())
            return "[REDACTED_HIGH_ENTROPY]" if bits > 4.2 else token

        return re.sub(r"[A-Za-z0-9_+/=\-]{24,}", entropy, text)

    def clean(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, dict):
            # Preserve schema keys, tool names and call IDs; sanitize values.
            return {
                k: self.clean(v) if k not in ("id", "tool_call_id", "name", "type", "role") else v
                for k, v in value.items()
            }
        return value


@dataclass(frozen=True)
class Filters:
    max_context_tokens: int = 32768
    template_reserve: int = 2048
    max_per_task: int = 8
    validation_percent: int = 10
    near_duplicate_ratio: float = 0.97
    allow_compaction: bool = True


def validate_messages(messages):
    pending = set()
    seen = set()
    for message in messages:
        if message.get("role") == "assistant":
            if pending:
                return False
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")
                if not call_id or call_id in seen or call.get("type") != "function":
                    return False
                try:
                    json.loads(call["function"]["arguments"])
                except (KeyError, TypeError, ValueError):
                    return False
                seen.add(call_id)
                pending.add(call_id)
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                return False
            pending.remove(call_id)
        elif pending:
            return False
    return not pending and bool(messages) and messages[-1].get("role") == "assistant"


def normalized_messages(messages):
    """Ignore generated call IDs and JSON formatting when comparing trajectories.

    Keep content/code whitespace intact; normalization must not change meaning.
    The original messages are always retained in the exported training row.
    """
    result = json.loads(json.dumps(messages))
    ids = {}
    for message in result:
        for call in message.get("tool_calls") or []:
            ids.setdefault(call["id"], f"call_{len(ids)}")
            call["id"] = ids[call["id"]]
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        if "tool_call_id" in message:
            message["tool_call_id"] = ids[message["tool_call_id"]]
    return [canonical(message) for message in result]


def select_sft_rows(rows, filters=Filters(), preferred_ids=()):
    """Keep unique completed examples, preferring continuations over their prefixes."""
    kept, excluded, seen = [], [], set()
    groups, counts = defaultdict(list), Counter()
    # Prefer longer eligible records before caps. A failed/oversized continuation
    # must never replace an eligible success: callers supply validated SFT rows.
    for row in sorted(
        rows,
        key=lambda r: (
            -len(r["messages"]),
            r["meta"]["success"] is not True,
            r["meta"]["split"] != "validation",
            r["id"] not in preferred_ids,
            r["id"],
        ),
    ):
        messages = normalized_messages(row["messages"])
        schema = (row["meta"]["campaign_id"], digest(row["tools"]))
        fingerprint = digest([schema, [m.decode() for m in messages]])
        key = (*schema, row["meta"]["task_id"])
        users = [m for m, raw in zip(messages, row["messages"]) if raw["role"] == "user"]
        text = b"\n".join(messages).decode()
        reason = ""
        if fingerprint in seen:
            reason = "duplicate"
        elif any(messages == old[: len(messages)] for old, _, _ in groups[key]):
            reason = "conversation_prefix"
        elif any(
            users == old_users
            and SequenceMatcher(None, text, old_text, autojunk=False).ratio()
            >= filters.near_duplicate_ratio
            for _, old_users, old_text in groups[key]
        ):
            reason = "duplicate"
        elif counts[key] >= filters.max_per_task:
            reason = "task_cap"
        if reason:
            excluded.append({"episode_id": row["source_episode"], "reason": reason})
            continue
        seen.add(fingerprint)
        groups[key].append((messages, users, text))
        counts[key] += 1
        kept.append(row)
    return sorted(kept, key=lambda r: r["id"]), excluded


def select_preference_rows(rows, preferred_ids=()):
    kept, seen = [], set()
    for row in sorted(rows, key=lambda r: (r["id"] not in preferred_ids, r["id"])):
        fingerprint = digest(
            [
                row["meta"]["campaign_id"],
                row["tools"],
                [m.decode() for m in normalized_messages(row["chosen"])],
                [m.decode() for m in normalized_messages(row["rejected"])],
            ]
        )
        if fingerprint not in seen:
            kept.append(row)
            seen.add(fingerprint)
    return kept


class Curator:
    def __init__(self, root: Path, scrubber: Scrubber, filters=Filters()):
        self.root, self.scrubber, self.filters = root, scrubber, filters

    def run(self) -> Path:
        splits = {"train": [], "validation": []}
        candidates = []
        pairs, rejected, sources = [], [], []
        grouped = defaultdict(list)
        for episode in RawStore(self.root).episodes():
            eid = episode["episode_id"]
            sources.append({"episode_id": eid, "sha256": digest(episode)})
            outcome = episode.get("outcome", {})
            verified = bool(outcome.get("verifier") not in (None, "", "none")) and (
                type(outcome.get("success")) is bool
            )
            success = outcome.get("success") if verified else None
            compacted = was_compacted(episode)
            reason = ""
            last = (episode.get("messages") or [{}])[-1]
            content = last.get("content")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            if (
                outcome.get("response_error")
                or last.get("role") != "assistant"
                or not isinstance(content, str)
                or not content.strip()
                or last.get("tool_calls")
            ):
                reason = "no_complete_answer"
            elif outcome.get("turn_completed") is False or outcome.get("exit_code") not in (
                None,
                0,
            ):
                reason = "incomplete_turn"
            elif compacted and not self.filters.allow_compaction:
                reason = "compaction_disabled"
            elif compacted and not valid_continuation(episode):
                reason = "incomplete_checkpoint"
            elif not compacted and not episode.get("context_complete"):
                reason = "incomplete_context"
            messages = self.scrubber.clean(episode.get("messages", []))
            tools = self.scrubber.clean(episode.get("tools", []))
            if not reason and not validate_messages(messages):
                reason = "invalid_role_sequence"
            # Bytes are not tokens. Preserve canonical examples for target-tokenizer
            # preparation; an oversized byte bound does not prove token overflow.
            byte_bound = len(canonical([messages, tools])) + self.filters.template_reserve
            if reason:
                rejected.append({"episode_id": eid, "reason": reason})
                continue
            task = episode["task"]["id"]
            campaign = episode["campaign_id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", campaign):
                rejected.append({"episode_id": eid, "reason": "invalid_campaign_id"})
                continue
            split = (
                "validation"
                if int(digest(task)[:8], 16) % 100 < self.filters.validation_percent
                else "train"
            )
            row = {
                "id": "sft_" + eid,
                "source_episode": eid,
                "messages": messages,
                "tools": tools,
                "meta": {
                    "success": success,
                    "source": "teacher",
                    "quality_score": (1.0 if success else 0.0) if verified else None,
                    "task_id": "task_" + digest(task)[:20],
                    "campaign_id": campaign,
                    "split": split,
                },
            }
            if not verified:
                # A complete answer can be collected without claiming the task passed.
                row["meta"]["verification"] = "unverified"
            if byte_bound > self.filters.max_context_tokens:
                row["meta"]["context_validation"] = "requires_target_tokenizer"
                row["meta"]["context_utf8_bytes_with_reserve"] = byte_bound
            if compacted:
                checkpoint = checkpoint_index(messages)
                row["meta"].update(
                    context_compacted=True,
                    segment_index=episode.get("compaction_count")
                    or (len(compaction_exchanges(episode.get("provider_exchanges", []))) or None),
                    checkpoint_id="ckpt_" + digest(messages[checkpoint])[:20],
                    # Context, including an assistant-authored summary, is input only.
                    # Train only assistant messages at/after this boundary.
                    continuation_start_index=checkpoint + 1,
                )
            # Match the entire prompt before the first assistant response. This prevents
            # pairing attempts with different user context or instruction contracts.
            prefix = []
            for message in messages:
                if message.get("role") == "assistant":
                    break
                prefix.append(message)
            if verified:
                grouped[(campaign, task, digest([prefix, tools]))].append(row)
            if outcome.get("success") is False:
                rejected.append(
                    {
                        "episode_id": eid,
                        "reason": "verified_failure" if verified else "failed_outcome",
                    }
                )
                continue
            candidates.append(row)
        selected, duplicates = select_sft_rows(candidates, self.filters)
        rejected.extend(duplicates)
        for row in selected:
            splits[row["meta"]["split"]].append(row)
        for rows in grouped.values():
            wins = [r for r in rows if r["meta"]["success"] is True][: self.filters.max_per_task]
            losses = [r for r in rows if r["meta"]["success"] is False][: self.filters.max_per_task]
            for chosen, loser in zip(wins, losses):
                pairs.append(
                    {
                        "id": "dpo_" + digest([chosen["id"], loser["id"]])[:20],
                        "prompt_episode": chosen["source_episode"],
                        "rejected_episode": loser["source_episode"],
                        "chosen": chosen["messages"],
                        "rejected": loser["messages"],
                        "tools": chosen["tools"],
                        "meta": chosen["meta"],
                        "margin": "verifier_graded",
                    }
                )
        artifacts = {"excluded.jsonl": rejected}
        for kind, rows in [
            ("train", splits["train"]),
            ("validation", splits["validation"]),
            ("preferences", select_preference_rows(pairs)),
        ]:
            for row in rows:
                name = f"campaigns/{row['meta']['campaign_id']}/{kind}.jsonl"
                artifacts.setdefault(name, []).append(row)
        files = {
            name: b"".join(canonical(row) + b"\n" for row in rows)
            for name, rows in artifacts.items()
        }
        manifest = {
            "pipeline_version": PIPELINE_VERSION,
            "filters": asdict(self.filters),
            "source_episodes": sources,
            "context_accounting": "target_tokenizer_required_before_training",
            "outcome_policy": "completed_or_verified",
            "verification_counts": dict(
                Counter(
                    "verified" if r["meta"]["success"] is True else "unverified" for r in selected
                )
            ),
            "counts": {k: len(v) for k, v in artifacts.items()},
            "files": {name: hashlib.sha256(body).hexdigest() for name, body in files.items()},
        }
        version = digest(manifest)
        manifest["dataset_hash"] = version
        output = self.root / "derived" / version
        if output.exists() and (output / "manifest.json").exists():
            return output
        for name, body in files.items():
            path = output / name
            if not path.exists():
                immutable_write(path, body)
            elif path.read_bytes() != body:
                raise ValueError("Derived artifact hash collision or corruption")
        immutable_write(output / "manifest.json", canonical(manifest) + b"\n")
        return output
