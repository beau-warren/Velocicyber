import json
import uuid

import pytest

from velocicyber.datasets.pipeline import Scrubber, delete_local_finetuning_data
from velocicyber.engine.runner import Engine
from velocicyber.engine.sessions import SessionStore
from velocicyber.state import State


def saved_conversation(settings):
    engine = Engine(settings, State(collection=False, upload_approved=False))
    saved = engine.ensure_session("Continue the parser fix")
    thread_id = str(uuid.uuid4())
    (saved.home / "sessions").mkdir()
    (saved.home / "sessions" / f"rollout-fixture-{thread_id}.jsonl").write_text("{}\n")
    saved.save(
        thread_id=thread_id,
        task_id="parser-task",
        verifier=["python3", "check.py"],
        model="CyberKimi",
    )
    saved.append({"type": "user.prompt", "text": "Continue the parser fix"})
    return engine, saved


def test_resume_restores_context_metadata_and_protects_concurrent_writers(settings):
    engine, saved = saved_conversation(settings)
    session_id = saved.metadata["id"]
    other = Engine(settings, State(collection=False, upload_approved=False))
    with pytest.raises(RuntimeError, match="another harness"):
        other.resume_session(session_id)
    assert other.session is None
    engine.close()
    other.resume_session(session_id)
    assert other.thread_id == saved.metadata["thread_id"]
    assert other.task_id == "parser-task" and other.verifier == ["python3", "check.py"]
    assert list(other.session.events()) == [
        {"type": "user.prompt", "text": "Continue the parser fix"}
    ]
    assert (saved.folder / "session.json").stat().st_mode & 0o777 == 0o600
    other.new_task()
    assert other.thread_id is None and other.session is None
    assert len(other.sessions.list()) == 1
    other.close()


def test_interrupted_session_and_partial_display_log_remain_resumable(settings):
    engine, saved = saved_conversation(settings)
    saved.save(status="running")
    with (saved.folder / "transcript.jsonl").open("a") as stream:
        stream.write('{"type":')
    saved.lock.close()  # Simulate the OS releasing the lock on a killed process.
    saved.lock = None
    restored = engine.sessions.open(saved.metadata["id"])
    assert restored.metadata["status"] == "interrupted"
    assert len(list(restored.events())) == 1
    restored.append({"type": "user.prompt", "text": "Continue after interruption"})
    assert len(list(restored.events())) == 2
    restored.close()
    engine.close()


def test_resume_rejects_paths_and_missing_workspace(settings):
    engine, saved = saved_conversation(settings)
    session_id = saved.metadata["id"]
    engine.close()
    for invalid in ("../other", "/tmp", "not-an-id"):
        with pytest.raises(ValueError, match="ID"):
            engine.resume_session(invalid)
    meta_file = saved.folder / "session.json"
    meta = json.loads(meta_file.read_text())
    meta["workspace"] = "/tmp"
    meta_file.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="workspace"):
        engine.resume_session(session_id)
    assert engine.session is None


def test_saved_conversations_are_independent_of_finetuning_deletion(settings):
    engine, saved = saved_conversation(settings)
    settings.data.mkdir()
    (settings.data / "fixture").write_text("training data")
    delete_local_finetuning_data(settings.root, approved=True)
    assert (saved.folder / "session.json").exists()
    assert not settings.data.exists()
    engine.close()
    assert SessionStore(settings, Scrubber()).list()[0]["id"] == saved.metadata["id"]


def test_collection_upload_and_gpu_readiness_start_off(settings):
    state = State.restore(settings.runtime / "no-gpu.json")
    assert not state.collection and not state.upload_approved
    assert not state.model_ready and not state.instance_id
