import asyncio
import uuid

import pytest
from textual.widgets import Input, Static

from velocicyber.ui.app import VelocicyberApp


@pytest.fixture
def app(settings, monkeypatch):
    app = VelocicyberApp(settings)
    app.state.collection = app.state.upload_approved = False
    monkeypatch.setattr(app, "refresh_updates", lambda: None)
    return app


async def submit(app, pilot, text):
    app.query_one("#prompt", Input).value = text
    await pilot.pause()
    await pilot.press("enter")


async def test_followups_run_fifo_in_same_conversation_without_overlap(app, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    calls, sessions = [], []
    active = 0

    async def run(prompt, callback):
        nonlocal active
        active += 1
        assert active == 1
        calls.append(prompt)
        sessions.append(app.engine.ensure_session(prompt).metadata["id"])
        if len(calls) == 1:
            started.set()
            await release.wait()
        callback({"type": "turn.completed"})
        active -= 1

    monkeypatch.setattr(app.engine, "run", run)
    async with app.run_test(size=(80, 24)) as pilot:
        worker = app.run_prompt("First")
        await asyncio.wait_for(started.wait(), 3)
        await submit(app, pilot, "Second test-key")
        await submit(app, pilot, "Third")
        assert calls == ["First"]
        assert list(app.queued_prompts) == ["Second test-key", "Third"]
        assert app.query_one("#prompt", Input).value == ""
        indicator = app.query_one("#prompt-queue", Static)
        assert indicator.display
        assert "Queued: 2" in indicator.render().plain
        assert "test-key" not in indicator.render().plain
        assert "REDACTED" in indicator.render().plain
        assert app.query_one("#prompt").region.bottom < app.size.height
        release.set()
        await asyncio.wait_for(worker.wait(), 3)
        await pilot.pause()
        assert calls == ["First", "Second test-key", "Third"]
        assert len(set(sessions)) == 1
        assert not app.busy and not app.queued_prompts and not indicator.display


async def test_two_workers_submitted_together_only_start_one_engine(app, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def run(prompt, callback):
        calls.append(prompt)
        if len(calls) == 1:
            started.set()
            await release.wait()

    monkeypatch.setattr(app.engine, "run", run)
    async with app.run_test():
        first = app.run_prompt("First")
        second = app.run_prompt("Second")
        await asyncio.wait_for(started.wait(), 3)
        await asyncio.wait_for(second.wait(), 3)
        assert calls == ["First"] and list(app.queued_prompts) == ["Second"]
        release.set()
        await asyncio.wait_for(first.wait(), 3)
        assert calls == ["First", "Second"]


@pytest.mark.parametrize("reason", ["ctrl+x", "error", "sentinel"])
async def test_stop_or_failure_holds_queue_until_explicit_resume(app, monkeypatch, reason):
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def run(prompt, callback):
        calls.append(prompt)
        if len(calls) == 1:
            started.set()
            await release.wait()
            if reason == "sentinel":
                callback({
                    "type": "sentinel.stopped",
                    "message": "Fixture scope violation.",
                    "active_process_stopped": True,
                })
            raise RuntimeError("Fixture interruption")

    async def cancel():
        release.set()

    monkeypatch.setattr(app.engine, "run", run)
    monkeypatch.setattr(app.engine, "cancel", cancel)
    async with app.run_test() as pilot:
        worker = app.run_prompt("First")
        await asyncio.wait_for(started.wait(), 3)
        await submit(app, pilot, "Second")
        if reason == "ctrl+x":
            await pilot.press("ctrl+x")
        else:
            release.set()
        await asyncio.wait_for(worker.wait(), 3)
        assert calls == ["First"] and app.queue_paused
        assert list(app.queued_prompts) == ["Second"]
        assert "paused" in app.query_one("#prompt-queue", Static).render().plain
        await submit(app, pilot, "/queue")
        assert app.screen.id == "prompt-queue-menu"
        await pilot.press(*"resume", "enter")
        await app.workers.wait_for_complete()
        assert calls == ["First", "Second"]
        assert not app.queued_prompts and not app.busy


async def test_queue_menu_can_pause_and_clear_while_task_is_running(app, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def run(prompt, callback):
        calls.append(prompt)
        started.set()
        await release.wait()

    monkeypatch.setattr(app.engine, "run", run)
    async with app.run_test() as pilot:
        worker = app.run_prompt("First")
        await asyncio.wait_for(started.wait(), 3)
        await submit(app, pilot, "Second")
        await submit(app, pilot, "/queue")
        assert app.screen.id == "prompt-queue-menu"
        assert "Second" in app.screen.details
        await pilot.press(*"pause", "enter")
        release.set()
        await asyncio.wait_for(worker.wait(), 3)
        assert calls == ["First"] and app.queue_paused
        await submit(app, pilot, "/queue")
        await pilot.press(*"clear", "enter")
        assert not app.queued_prompts and calls == ["First"]


@pytest.mark.parametrize("operation", ["new", "resume"])
async def test_conversation_changes_discard_unsent_queue(app, operation):
    async with app.run_test():
        saved = app.engine.ensure_session("Saved conversation")
        session_id = saved.metadata["id"]
        thread_id = str(uuid.uuid4())
        (saved.home / "sessions").mkdir()
        (saved.home / "sessions" / f"rollout-fixture-{thread_id}.jsonl").write_text("{}\n")
        saved.save(thread_id=thread_id)
        app.engine.new_task()
        app.show_conversation()
        app.queue_prompt("Must not run in another conversation")
        app.queue_paused = True
        if operation == "new":
            app.command("new")
        else:
            app.resume_selected(session_id)
            assert app.engine.session.metadata["id"] == session_id
        assert not app.queued_prompts


async def test_quit_never_starts_queued_work(app, monkeypatch):
    started, release, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def run(prompt, callback):
        calls.append(prompt)
        started.set()
        await release.wait()

    async def cancel():
        release.set()

    monkeypatch.setattr(app.engine, "run", run)
    monkeypatch.setattr(app.engine, "cancel", cancel)
    monkeypatch.setattr(app, "exit", exited.set)
    async with app.run_test() as pilot:
        app.run_prompt("First")
        await asyncio.wait_for(started.wait(), 3)
        await submit(app, pilot, "Second")
        app.action_quit()
        await asyncio.wait_for(exited.wait(), 3)
        assert calls == ["First"]
        assert app.session_ending and not app.queued_prompts
        await app.run_prompt("Must not run while closing").wait()
        assert calls == ["First"]


@pytest.mark.parametrize("blocker", ["gpu_busy", "pi_busy", "sentinel_busy", "data_busy"])
async def test_startup_block_preserves_typed_prompt(app, monkeypatch, blocker):
    calls = []

    async def run(prompt, callback):
        calls.append(prompt)

    monkeypatch.setattr(app.engine, "run", run)
    if blocker == "pi_busy":
        app.state.active_provider = "pi"
    async with app.run_test() as pilot:
        setattr(app, blocker, True)
        await submit(app, pilot, "Write hello.txt")
        assert not calls
        assert app.query_one("#prompt", Input).value == "Write hello.txt"
        setattr(app, blocker, False)
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert calls == ["Write hello.txt"]


async def test_queue_waits_for_current_turn_upload_handling(app, monkeypatch):
    uploading, release = asyncio.Event(), asyncio.Event()
    calls = []
    app.state.collection = app.state.upload_approved = True

    async def run(prompt, callback):
        calls.append(prompt)

    async def upload():
        if len(calls) == 1:
            uploading.set()
            await release.wait()

    monkeypatch.setattr(app.engine, "run", run)
    monkeypatch.setattr(app, "flush_uploads", upload)
    async with app.run_test() as pilot:
        worker = app.run_prompt("First")
        await asyncio.wait_for(uploading.wait(), 3)
        await submit(app, pilot, "Second")
        assert calls == ["First"] and app.busy
        release.set()
        await asyncio.wait_for(worker.wait(), 3)
        assert calls == ["First", "Second"]


async def test_fresh_task_after_failure_can_accept_automatic_followups(app, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def run(prompt, callback):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("Fixture failure with no queued work")
        if len(calls) == 2:
            started.set()
            await release.wait()

    monkeypatch.setattr(app.engine, "run", run)
    async with app.run_test() as pilot:
        await app.run_prompt("First").wait()
        assert app.queue_paused and not app.queued_prompts
        worker = app.run_prompt("Fresh task")
        await asyncio.wait_for(started.wait(), 3)
        await submit(app, pilot, "Follow-up")
        release.set()
        await asyncio.wait_for(worker.wait(), 3)
        assert calls == ["First", "Fresh task", "Follow-up"]
