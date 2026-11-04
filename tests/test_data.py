import copy
import json
from dataclasses import replace

import pytest

from velocicyber.datasets.pipeline import (
    Curator,
    Filters,
    RawStore,
    Scrubber,
    delete_local_finetuning_data,
)


def rows(path, filename):
    return [
        json.loads(line) for file in path.rglob(filename) for line in file.read_text().splitlines()
    ]


def test_raw_is_immutable_and_integrity_checked(tmp_path, episode):
    store = RawStore(tmp_path)
    path = store.append(episode)
    assert list(store.episodes()) == [episode]
    with pytest.raises(FileExistsError):
        store.append(episode)
    path.chmod(0o600)
    path.write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        list(store.episodes())


def test_delete_local_data_requires_consent_and_preserves_other_files(settings, episode):
    raw = RawStore(settings.data).append(episode)
    snapshot = Curator(settings.data, Scrubber()).run()
    receipt = settings.data / "uploaded" / (snapshot.name + ".json")
    receipt.parent.mkdir()
    receipt.write_text('{"commit": "already-uploaded"}')
    protected = [
        settings.root / ".env",
        settings.root / "Modules.txt",
        settings.runtime / "gpu.json",
        settings.root / "ssh-keys" / "test.pem",
        settings.root / "workspace" / "main.py",
    ]
    for path in protected:
        path.parent.mkdir(exist_ok=True)
        path.write_text("preserve")
    (settings.data / "linked-workspace").symlink_to(protected[-1].parent, target_is_directory=True)
    with pytest.raises(PermissionError):
        delete_local_finetuning_data(settings.root)
    assert raw.exists() and receipt.exists()
    assert delete_local_finetuning_data(settings.root, approved=True)
    assert not settings.data.exists()
    assert all(path.read_text() == "preserve" for path in protected)
    assert not delete_local_finetuning_data(settings.root, approved=True)
    # Explicitly re-enabling collection can create a new store after deletion.
    assert RawStore(settings.data).append(episode).exists()


def test_delete_local_data_refuses_dataset_root_symlink(settings):
    elsewhere = settings.root / "keep"
    elsewhere.mkdir()
    sentinel = elsewhere / "data.jsonl"
    sentinel.write_text("preserve")
    settings.data.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        delete_local_finetuning_data(settings.root, approved=True)
    assert sentinel.read_text() == "preserve"
    assert settings.data.is_symlink()


def test_recovery_kept_failures_paired_and_reproducible(tmp_path, episode, failed_episode):
    store = RawStore(tmp_path)
    store.append(episode)
    store.append(failed_episode)
    curator = Curator(tmp_path, Scrubber())
    path = curator.run()
    assert curator.run() == path
    sft = rows(path, "train.jsonl") + rows(path, "validation.jsonl")
    assert len(sft) == 1
    assert sft[0]["messages"][3]["content"].startswith("FAILED")
    pair = rows(path, "preferences.jsonl")[0]
    assert pair["chosen"] == episode["messages"]
    assert pair["rejected"] == failed_episode["messages"]


def test_scrubbing_preserves_raw_and_tool_schema(tmp_path, episode):
    secret = "credential_from_actual_environment"
    episode["messages"][-1]["content"] += f" {secret} http://private.example/a 192.168.10.2"
    raw = RawStore(tmp_path).append(episode)
    path = Curator(tmp_path, Scrubber([secret])).run()
    data = b"".join(f.read_bytes() for f in path.rglob("*.jsonl"))
    assert secret.encode() not in data
    assert b"private.example" not in data
    assert b"192.168.10.2" not in data
    assert secret in raw.read_text()
    result = (rows(path, "train.jsonl") + rows(path, "validation.jsonl"))[0]
    assert result["tools"] == episode["tools"]
    assert result["messages"][2]["tool_calls"] == episode["messages"][2]["tool_calls"]


def test_unknown_outcome_collected_invalid_calls_excluded_long_context_flagged(tmp_path, episode):
    store = RawStore(tmp_path)
    unknown = copy.deepcopy(episode)
    unknown["episode_id"] = "ep_unknown"
    unknown["outcome"]["success"] = None
    unknown["outcome"]["verifier"] = "none"
    store.append(unknown)
    broken = copy.deepcopy(episode)
    broken["episode_id"] = "ep_broken"
    broken["messages"][3]["tool_call_id"] = "does_not_exist"
    store.append(broken)
    long = copy.deepcopy(episode)
    long["episode_id"] = "ep_long"
    long["messages"][-1]["content"] = "a" * 40000
    store.append(long)
    path = Curator(tmp_path, Scrubber()).run()
    assert {r["reason"] for r in rows(path, "excluded.jsonl")} == {
        "invalid_role_sequence",
    }
    kept = rows(path, "train.jsonl") + rows(path, "validation.jsonl")
    assert len(kept) == 2
    unknown_row = next(r for r in kept if r["source_episode"] == "ep_unknown")
    assert unknown_row["meta"]["success"] is None
    assert unknown_row["meta"]["quality_score"] is None
    assert unknown_row["meta"]["verification"] == "unverified"
    long_row = next(r for r in kept if r["source_episode"] == "ep_long")
    assert long_row["messages"] == long["messages"]
    assert long_row["meta"]["context_validation"] == "requires_target_tokenizer"
    assert not rows(path, "preferences.jsonl")


def test_compaction_policy_checkpoint_metadata_and_immutable_raw(tmp_path, compacted_episode):
    raw = RawStore(tmp_path).append(compacted_episode)
    original = raw.read_bytes()
    no = Curator(tmp_path, Scrubber(), Filters(allow_compaction=False)).run()
    assert rows(no, "excluded.jsonl") == [
        {"episode_id": "ep_compacted", "reason": "compaction_disabled"},
    ]
    yes = Curator(tmp_path, Scrubber()).run()
    assert not rows(yes, "excluded.jsonl")
    row = (rows(yes, "train.jsonl") + rows(yes, "validation.jsonl"))[0]
    assert row["messages"] == compacted_episode["messages"]
    assert row["meta"]["context_compacted"] is True
    assert row["meta"]["segment_index"] == 1
    assert row["meta"]["checkpoint_id"].startswith("ckpt_")
    assert row["meta"]["continuation_start_index"] == 3
    assert raw.read_bytes() == original


@pytest.mark.parametrize("problem", ["missing_summary", "missing_capture", "rewrite", "blank"])
def test_allow_compaction_does_not_waive_context_or_response_validation(
    tmp_path, compacted_episode, problem
):
    if problem == "missing_summary":
        compacted_episode["messages"][2]["content"] = "Invented checkpoint"
    elif problem == "missing_capture":
        compacted_episode["provider_exchanges"] = []
        compacted_episode["context_compacted"] = True
    elif problem == "rewrite":
        tail = copy.deepcopy(compacted_episode["provider_exchanges"][-1])
        tail["request"]["messages"][1]["content"] = "Unrelated replacement context"
        compacted_episode["provider_exchanges"].insert(1, tail)
    else:
        compacted_episode["messages"][-1]["content"] = ""
    RawStore(tmp_path).append(compacted_episode)
    artifact = Curator(tmp_path, Scrubber()).run()
    assert not rows(artifact, "train.jsonl") + rows(artifact, "validation.jsonl")
    assert rows(artifact, "excluded.jsonl")[0]["reason"] == (
        "no_complete_answer" if problem == "blank" else "incomplete_checkpoint"
    )


def test_old_resumed_checkpoint_detected_without_compaction_event(tmp_path, compacted_episode):
    compacted_episode["provider_exchanges"] = compacted_episode["provider_exchanges"][-1:]
    compacted_episode["context_complete"] = True
    RawStore(tmp_path).append(compacted_episode)
    artifact = Curator(tmp_path, Scrubber(), Filters(allow_compaction=False)).run()
    assert rows(artifact, "excluded.jsonl")[0]["reason"] == "compaction_disabled"
    artifact = Curator(tmp_path, Scrubber()).run()
    row = (rows(artifact, "train.jsonl") + rows(artifact, "validation.jsonl"))[0]
    assert row["meta"]["context_compacted"] and row["meta"]["segment_index"] is None


def test_unverified_conversation_does_not_become_preference_signal(
    tmp_path, episode, failed_episode
):
    unknown = copy.deepcopy(episode)
    unknown["episode_id"] = "ep_unknown"
    unknown["outcome"] = {
        "success": None,
        "verifier": "none",
        "exit_code": 0,
        "turn_completed": True,
    }
    unknown["messages"][-1]["content"] = "Here is an explanation, without independent verification."
    store = RawStore(tmp_path)
    for e in [episode, failed_episode, unknown]:
        store.append(e)
    path = Curator(tmp_path, Scrubber()).run()
    kept = rows(path, "train.jsonl") + rows(path, "validation.jsonl")
    assert len(kept) == 2
    pairs = rows(path, "preferences.jsonl")
    assert len(pairs) == 1
    assert pairs[0]["prompt_episode"] == episode["episode_id"]
    assert pairs[0]["rejected_episode"] == failed_episode["episode_id"]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("response_error", "ReadTimeout", "no_complete_answer"),
        ("exit_code", -15, "incomplete_turn"),
        ("turn_completed", False, "incomplete_turn"),
        ("success", False, "failed_outcome"),
    ],
)
def test_unverified_errors_never_upload_even_with_nonblank_answer(
    tmp_path, episode, field, value, reason
):
    episode["outcome"] = {"success": None, "verifier": "none", field: value}
    RawStore(tmp_path).append(episode)
    path = Curator(tmp_path, Scrubber()).run()
    assert not rows(path, "train.jsonl") + rows(path, "validation.jsonl")
    assert rows(path, "excluded.jsonl")[0]["reason"] == reason


@pytest.mark.parametrize("content", ["", "  ", None, []])
def test_blank_answers_never_enter_training_even_if_marked_verified(tmp_path, episode, content):
    from velocicyber.datasets.pipeline import dataset_summary

    episode["messages"][-1]["content"] = content
    raw = RawStore(tmp_path).append(episode)
    before = raw.read_bytes()
    artifact = Curator(tmp_path, Scrubber()).run()
    assert dataset_summary(artifact) == {
        "eligible": 0,
        "excluded": 1,
        "exclusion_reasons": {"no_complete_answer": 1},
        "unverified": 0,
    }
    assert raw.read_bytes() == before


def test_task_splits_campaign_isolation_dedup_and_cap(tmp_path, episode):
    store = RawStore(tmp_path)
    for index in range(5):
        e = copy.deepcopy(episode)
        e["episode_id"] = f"ep_{index}"
        e["campaign_id"] = "one" if index < 3 else "two"
        if index in (2, 4):
            e["messages"][-1]["content"] = "Different path " * 50
        store.append(e)
    path = Curator(tmp_path, Scrubber(), replace(Filters(), max_per_task=1)).run()
    all_rows = rows(path, "train.jsonl") + rows(path, "validation.jsonl")
    assert len(all_rows) == 2
    assert len({r["meta"]["split"] for r in all_rows}) == 1
    assert {p.name for p in (path / "campaigns").iterdir()} == {"one", "two"}
    assert {r["reason"] for r in rows(path, "excluded.jsonl")} == {"duplicate", "task_cap"}


def test_duplicate_with_new_task_and_call_ids_is_removed(tmp_path, episode):
    duplicate = copy.deepcopy(episode)
    duplicate["episode_id"] = "ep_copy"
    duplicate["task"]["id"] = "another-task-id"
    duplicate["messages"][2]["tool_calls"][0]["id"] = "random-new-id"
    duplicate["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{ "command" : "pytest" }'
    duplicate["messages"][3]["tool_call_id"] = "random-new-id"
    store = RawStore(tmp_path)
    store.append(episode)
    store.append(duplicate)
    path = Curator(tmp_path, Scrubber()).run()
    assert len(rows(path, "train.jsonl") + rows(path, "validation.jsonl")) == 1
    assert rows(path, "excluded.jsonl")[0]["reason"] == "duplicate"
    assert len(list(store.episodes())) == 2


@pytest.mark.parametrize("eligible", [True, False])
def test_longest_eligible_conversation_keeps_full_context(tmp_path, episode, eligible):
    followup = copy.deepcopy(episode)
    followup["episode_id"] = "ep_followup"
    followup["messages"] += [
        {"role": "user", "content": "Add an extra test"},
        {"role": "assistant", "content": "Added the extra test and it passed."},
    ]
    followup["outcome"]["success"] = eligible
    store = RawStore(tmp_path)
    original = store.append(episode).read_bytes()
    store.append(followup)
    path = Curator(tmp_path, Scrubber(), replace(Filters(), max_per_task=1)).run()
    kept = rows(path, "train.jsonl") + rows(path, "validation.jsonl")
    assert len(kept) == 1
    assert kept[0]["messages"] == (followup if eligible else episode)["messages"]
    assert store.root.joinpath("raw", "ep_success.jsonl").read_bytes() == original
    if eligible:
        assert rows(path, "excluded.jsonl")[0]["reason"] == "conversation_prefix"
