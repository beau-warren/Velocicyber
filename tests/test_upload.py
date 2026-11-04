import copy
import json
from types import SimpleNamespace

import pytest
from huggingface_hub import DatasetCard

from velocicyber.datasets.pipeline import (
    Curator,
    Filters,
    RawStore,
    Scrubber,
    delete_local_finetuning_data,
)
from velocicyber.datasets.upload import Uploader, current_files, training_rows


class FakeHub:
    def __init__(self, root):
        self.root = root
        self.files = {
            "README.md": b"---\nlicense: apache-2.0\n---\nKeep this description.\n",
            ".gitattributes": b"keep",
        }
        self.calls = []
        self.reads = 0
        self.sha = "a" * 40
        self.conflict = False

    def dataset_info(self, repo, **kwargs):
        self.reads += 1
        return SimpleNamespace(
            sha=self.sha, siblings=[SimpleNamespace(rfilename=name) for name in self.files]
        )

    def hf_hub_download(self, *, filename, revision, **kwargs):
        assert revision == self.sha
        target = self.root / revision / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)

    def create_commit(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["parent_commit"] == self.sha
        if self.conflict:
            raise RuntimeError("Concurrent remote change")
        for op in kwargs["operations"]:
            assert type(op).__name__ == "CommitOperationAdd"
            with op.as_file() as file:
                self.files[op.path_in_repo] = file.read()
        self.sha = f"{len(self.calls):040x}"
        return SimpleNamespace(oid=self.sha)

    def rows(self):
        manifest = json.loads(self.files["current/manifest.json"])
        return [
            json.loads(line)
            for name in manifest["files"]
            for line in self.files["current/" + name].splitlines()
        ]


@pytest.fixture
def hub(tmp_path):
    return FakeHub(tmp_path / "fake-hub")


def prepare(settings, episode):
    RawStore(settings.data).append(episode)
    return Curator(settings.data, Scrubber()).run()


def test_compaction_toggle_upload_cleanup_and_existing_remote_preserved(
    settings, episode, compacted_episode, hub
):
    raw = RawStore(settings.data).append(compacted_episode)
    uploader = Uploader(settings, hub)
    no = Curator(settings.data, Scrubber(), Filters(allow_compaction=False)).run()
    assert uploader.upload(no, approved=True) is None
    assert hub.reads == 0 and raw.exists()
    yes = Curator(settings.data, Scrubber()).run()
    result = uploader.upload(yes, approved=True)
    assert result["local_cleanup"]["complete"] and not raw.exists()
    published = hub.rows()[0]
    assert published["meta"]["context_compacted"]
    assert b"continuation_start_index" in hub.files["README.md"]
    # Turning NO back on applies to incoming examples, not previously published rows.
    RawStore(settings.data).append(episode)
    no = Curator(settings.data, Scrubber(), Filters(allow_compaction=False)).run()
    uploader.upload(no, approved=True)
    assert published in hub.rows() and len(hub.rows()) == 2


def test_old_checkpoint_annotations_upgrade_without_duplicate_or_content_overwrite(
    settings, compacted_episode
):
    artifact = prepare(settings, compacted_episode)
    manifest = json.loads((artifact / "manifest.json").read_text())
    [(kind, row)] = training_rows(manifest, lambda name: (artifact / name).read_bytes())
    old = copy.deepcopy(row)
    for name in ("context_compacted", "checkpoint_id", "segment_index", "continuation_start_index"):
        del old["meta"][name]
    for records in ([(kind, old), (kind, row)], [(kind, row), (kind, old)]):
        files, counts = current_files(records)
        assert sum(counts.values()) == 1
        assert json.loads(next(iter(files.values()))) == row
    old["messages"][-1]["content"] = "Conflicting answer"
    with pytest.raises(ValueError, match="Conflicting"):
        current_files([(kind, old), (kind, row)])


def test_upload_requires_consent_and_repeat_is_idempotent(settings, episode, hub):
    artifact = prepare(settings, episode)
    uploader = Uploader(settings, hub)
    with pytest.raises(PermissionError):
        uploader.upload(artifact)
    assert hub.reads == 0 and not hub.calls
    receipt = uploader.upload(artifact, approved=True)
    assert uploader.upload(artifact, approved=True) == receipt
    assert len(hub.calls) == 1 and hub.reads == 1
    assert hub.files[".gitattributes"] == b"keep"
    assert b"Keep this description." in hub.files["README.md"]
    card = DatasetCard(hub.files["README.md"].decode())
    assert card.data.license == "apache-2.0"
    assert card.data.configs[0]["default"]
    assert all(f["path"].startswith("current/") for c in card.data.configs for f in c["data_files"])
    assert len(hub.rows()) == 1
    assert not any(name.startswith("versions/") for name in hub.files)
    with pytest.raises(ValueError):
        uploader.upload(settings.data / "raw", approved=True)


def test_excluded_only_changes_do_not_publish_another_copy(settings, episode, hub):
    uploader = Uploader(settings, hub)
    first = prepare(settings, episode)
    uploader.upload(first, approved=True)
    blank = copy.deepcopy(episode)
    blank["episode_id"] = "ep_blank"
    blank["messages"][-1]["content"] = ""
    second = prepare(settings, blank)
    assert first != second
    receipt = uploader.upload(second, approved=True)
    assert receipt is None  # The previous successful example was cleaned locally.
    assert (settings.data / "raw" / "ep_blank.jsonl").exists()
    assert len(hub.calls) == 1


def test_completed_unverified_chat_uploads_and_deduplicates(settings, episode, hub):
    episode["outcome"] = {
        "success": None,
        "verifier": "none",
        "turn_completed": True,
        "exit_code": 0,
    }
    episode["messages"][-1]["content"] += " hf_abcdefgh1234567890"
    artifact = prepare(settings, episode)
    uploader = Uploader(settings, hub)
    receipt = uploader.upload(artifact, approved=True)
    assert receipt and not receipt["unchanged"]
    assert len(hub.rows()) == 1
    meta = hub.rows()[0]["meta"]
    assert meta["success"] is None and meta["quality_score"] is None
    assert meta["verification"] == "unverified"
    assert b"hf_abcdefgh1234567890" not in b"".join(hub.files.values())
    assert b"meta.success=null" in hub.files["README.md"]
    # Losing receipts must still avoid republishing identical conversations.
    delete_local_finetuning_data(settings.root, approved=True)
    episode["episode_id"] = "ep_another_id"
    again = uploader.upload(prepare(settings, episode), approved=True)
    assert again["unchanged"] and len(hub.calls) == 1


def test_verified_duplicate_preferred_over_unknown(settings, episode, hub):
    unknown = copy.deepcopy(episode)
    unknown["episode_id"] = "ep_aaa_unknown"
    unknown["outcome"] = {"success": None, "verifier": "none"}
    uploader = Uploader(settings, hub)
    uploader.upload(prepare(settings, unknown), approved=True)
    uploader.upload(prepare(settings, episode), approved=True)
    assert len(hub.rows()) == 1
    assert hub.rows()[0]["meta"]["success"] is True


def test_remote_dedup_survives_deleted_local_data_and_changed_ids(settings, episode, hub):
    uploader = Uploader(settings, hub)
    uploader.upload(prepare(settings, episode), approved=True)
    delete_local_finetuning_data(settings.root, approved=True)
    duplicate = copy.deepcopy(episode)
    duplicate["episode_id"] = "ep_aaa"
    duplicate["messages"][2]["tool_calls"][0]["id"] = "other"
    duplicate["messages"][3]["tool_call_id"] = "other"
    duplicate["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{ "command": "pytest" }'
    receipt = uploader.upload(prepare(settings, duplicate), approved=True)
    assert receipt["unchanged"] and len(hub.calls) == 1
    assert hub.rows()[0]["source_episode"] == episode["episode_id"]
    delete_local_finetuning_data(settings.root, approved=True)
    different = copy.deepcopy(episode)
    different["episode_id"] = "ep_other"
    different["task"]["id"] = "unrelated"
    different["messages"][1]["content"] = "Implement an unrelated feature"
    uploader.upload(prepare(settings, different), approved=True)
    assert len(hub.rows()) == 2 and len(hub.calls) == 2


def test_legacy_snapshot_migration_preserves_archives_and_removes_prefix(settings, episode, hub):
    first = prepare(settings, episode)
    for file in first.rglob("*"):
        if file.is_file():
            hub.files[f"versions/{first.name}/{file.relative_to(first)}"] = file.read_bytes()
    archived = {name: body for name, body in hub.files.items() if name.startswith("versions/")}
    followup = copy.deepcopy(episode)
    followup["episode_id"] = "ep_followup"
    followup["messages"] += [
        {"role": "user", "content": "Continue"},
        {"role": "assistant", "content": "Done."},
    ]
    second = prepare(settings, followup)
    uploader = Uploader(settings, hub)
    plan = uploader.plan(second)
    assert not hub.calls
    assert sum(plan.counts.values()) == 1
    uploader.upload(second, approved=True)
    assert all(hub.files[name] == body for name, body in archived.items())
    assert hub.rows()[0]["messages"] == followup["messages"]
    assert not any(op.path_in_repo.startswith("versions/") for op in hub.calls[0]["operations"])


def test_remote_conflict_is_not_retried_or_marked_uploaded(settings, episode, hub):
    artifact = prepare(settings, episode)
    hub.conflict = True
    with pytest.raises(RuntimeError, match="Concurrent"):
        Uploader(settings, hub).upload(artifact, approved=True)
    assert len(hub.calls) == 1
    assert not (settings.data / "uploaded").exists()
    assert "current/manifest.json" not in hub.files
    assert artifact.exists()
    assert (settings.data / "raw" / (episode["episode_id"] + ".jsonl")).exists()


def test_success_cleans_published_data_and_old_snapshots_preserving_pending_and_resume(
    settings, episode, hub
):
    first = prepare(settings, episode)
    followup = copy.deepcopy(episode)
    followup["episode_id"] = "ep_followup"
    followup["messages"] += [
        {"role": "user", "content": "Add another check"},
        {"role": "assistant", "content": "Added another check."},
    ]
    second = prepare(settings, followup)
    pending = copy.deepcopy(episode)
    pending["episode_id"] = "ep_pending"
    pending["messages"][-1]["content"] = ""
    artifact = prepare(settings, pending)
    pending_raw = settings.data / "raw" / "ep_pending.jsonl"
    before = pending_raw.read_bytes()
    session = settings.runtime / "conversations" / "keep" / "events.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text("Saved conversation")
    uploader = Uploader(settings, hub)
    preview = uploader.plan(artifact)
    assert not hub.calls and first.exists() and second.exists() and artifact.exists()
    assert set(preview.cleanup["episodes"]) == {episode["episode_id"], "ep_followup"}
    receipt = uploader.upload(artifact, approved=True)
    assert receipt["local_cleanup"] == {"complete": True, "episodes": 2}
    assert not first.exists() and not second.exists() and not artifact.exists()
    assert sorted(p.name for p in (settings.data / "raw").iterdir()) == [
        "ep_pending.jsonl",
        "ep_pending.sha256",
    ]
    assert pending_raw.read_bytes() == before and session.read_text() == "Saved conversation"
    assert hub.rows()[0]["messages"] == followup["messages"]
    assert uploader.upload(artifact, approved=True) == receipt


def test_records_created_during_commit_are_never_deleted(settings, episode, hub, monkeypatch):
    artifact = prepare(settings, episode)
    original_commit = hub.create_commit
    created = []

    def commit(**kwargs):
        new = copy.deepcopy(episode)
        new["episode_id"] = "ep_new_during_upload"
        new["messages"][-1]["content"] = "New work that has not been uploaded."
        created.append(prepare(settings, new))
        return original_commit(**kwargs)

    monkeypatch.setattr(hub, "create_commit", commit)
    receipt = Uploader(settings, hub).upload(artifact, approved=True)
    assert receipt["local_cleanup"]["complete"]
    assert (settings.data / "raw" / "ep_new_during_upload.jsonl").exists()
    assert created[0].exists()
    assert not artifact.exists()


def test_cleanup_failure_preserves_receipt_and_can_retry_without_upload(
    settings, episode, hub, monkeypatch
):
    from velocicyber.datasets import upload

    artifact = prepare(settings, episode)
    real_cleanup = upload.cleanup_confirmed

    def unavailable(*args):
        raise PermissionError("Fixture disk permission failure")

    monkeypatch.setattr(upload, "cleanup_confirmed", unavailable)
    uploader = Uploader(settings, hub)
    receipt = uploader.upload(artifact, approved=True)
    assert not receipt["local_cleanup"]["complete"]
    assert receipt["commit"] == hub.sha
    assert artifact.exists() and len(hub.calls) == 1
    monkeypatch.setattr(upload, "cleanup_confirmed", real_cleanup)
    retried = uploader.upload(artifact, approved=True)
    assert retried["local_cleanup"]["complete"] and not artifact.exists()
    assert len(hub.calls) == 1 and hub.reads == 1


def test_changed_or_symlinked_raw_is_preserved_after_remote_success(
    settings, episode, hub, monkeypatch
):
    artifact = prepare(settings, episode)
    original_commit = hub.create_commit
    raw = settings.data / "raw" / (episode["episode_id"] + ".jsonl")
    sentinel = settings.root / "preserve.jsonl"
    sentinel.write_text("New data")

    def commit(**kwargs):
        raw.unlink()
        raw.symlink_to(sentinel)
        return original_commit(**kwargs)

    monkeypatch.setattr(hub, "create_commit", commit)
    receipt = Uploader(settings, hub).upload(artifact, approved=True)
    assert not receipt["local_cleanup"]["complete"]
    assert "symlink" in receipt["local_cleanup"]["error"]
    assert sentinel.read_text() == "New data" and raw.is_symlink() and artifact.exists()


def test_lost_commit_confirmation_retains_data_until_remote_is_checked(
    settings, episode, hub, monkeypatch
):
    artifact = prepare(settings, episode)
    original_commit = hub.create_commit

    def lost_response(**kwargs):
        original_commit(**kwargs)
        raise RuntimeError("Connection lost before confirmation")

    monkeypatch.setattr(hub, "create_commit", lost_response)
    uploader = Uploader(settings, hub)
    with pytest.raises(RuntimeError, match="before confirmation"):
        uploader.upload(artifact, approved=True)
    assert artifact.exists() and not (settings.data / "uploaded").exists()
    assert len(list((settings.data / "raw").glob("*.jsonl"))) == 1
    # An explicit retry sees the committed data and cleans up without another commit.
    receipt = uploader.upload(artifact, approved=True)
    assert receipt["unchanged"] and receipt["local_cleanup"]["complete"]
    assert len(hub.calls) == 1 and not artifact.exists()


def test_near_duplicate_kept_locally_when_its_exact_content_was_not_uploaded(
    settings, episode, hub
):
    episode["messages"][-1]["content"] = "A long explanation. " * 200
    prepare(settings, episode)
    similar = copy.deepcopy(episode)
    similar["episode_id"] = "ep_z_similar"
    similar["messages"][-1]["content"] += "Extra detail."
    artifact = prepare(settings, similar)
    receipt = Uploader(settings, hub).upload(artifact, approved=True)
    assert receipt["local_cleanup"] == {"complete": True, "episodes": 1}
    assert (settings.data / "raw" / "ep_z_similar.jsonl").exists()
    assert hub.rows()[0]["source_episode"] == episode["episode_id"]


def test_interrupted_snapshot_cleanup_resumes_using_receipt(settings, episode, hub, monkeypatch):
    from velocicyber.datasets import retention

    artifact = prepare(settings, episode)
    original_remove = retention.shutil.rmtree

    def interrupted(path, *args, **kwargs):
        if path == artifact:
            (path / "manifest.json").unlink()
            raise PermissionError("Fixture interrupted removal")
        return original_remove(path, *args, **kwargs)

    monkeypatch.setattr(retention.shutil, "rmtree", interrupted)
    uploader = Uploader(settings, hub)
    receipt = uploader.upload(artifact, approved=True)
    assert not receipt["local_cleanup"]["complete"] and artifact.exists()
    monkeypatch.setattr(retention.shutil, "rmtree", original_remove)
    assert uploader.upload(artifact, approved=True)["local_cleanup"]["complete"]
    assert not artifact.exists() and len(hub.calls) == 1


def test_corrupt_local_or_remote_file_is_rejected_before_commit(settings, episode, hub):
    artifact = prepare(settings, episode)
    uploader = Uploader(settings, hub)
    uploader.upload(artifact, approved=True)
    delete_local_finetuning_data(settings.root, approved=True)
    artifact = prepare(settings, episode)
    training_file = next(name for name in hub.files if name.endswith("train.jsonl"))
    hub.files[training_file] += b"tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        uploader.upload(artifact, approved=True)
    assert len(hub.calls) == 1
    local = next(artifact.rglob("train.jsonl"))
    local.chmod(0o600)
    local.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        uploader.upload(artifact, approved=True)
    assert len(hub.calls) == 1
