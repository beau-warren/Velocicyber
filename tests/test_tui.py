import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from textual.widgets import Input, Label, OptionList, RichLog, Static

from velocicyber.config import Settings
from velocicyber.datasets.pipeline import Curator, RawStore, Scrubber
from velocicyber.engine.installation import ACTIVE_FILE, UPDATE_NOTICE_FILE
from velocicyber.state import GPUStatus, State
from velocicyber.ui.app import Menu, VelocicyberApp
from velocicyber.ui.presentation import title_art


async def test_startup_has_no_gpu_model_until_confirmed(tmp_path):
    repo = "example-org/coding-model"
    (tmp_path / "Modules.txt").write_text("HUGGINGFACE_MODEL=" + repo + "\n")
    settings = Settings.load(tmp_path)
    assert not settings.local_model and settings.gpu_model_choices == [repo]
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test() as pilot:
        app.gpu_selected("on")
        await pilot.pause()
        assert app.screen.id == "local-models"
        await pilot.press("enter")
        assert app.screen.id == "local-model-choice"
        await pilot.press("escape")
        assert not settings.local_model and not app.state.model_ready
        app.select_local_model(repo)
        await pilot.pause()
        assert settings.local_model == repo
        assert app.state.gpu_status == GPUStatus.STOPPED
        assert app.gpu._client is None
    # A configured menu choice is never a startup selection, even on the next launch.
    assert not Settings.load(tmp_path).local_model


async def test_model_confirmation_from_home_starts_new_session_and_cancel_does_not(
    settings, monkeypatch,
):
    app = VelocicyberApp(settings, demo=True)
    notifications = []
    monkeypatch.setattr(app, "notify", lambda message, **kw: notifications.append(message))
    async with app.run_test() as pilot:
        app.command("new")
        old = app.engine.ensure_session("Previous task")
        old.append({"type": "user.prompt", "text": "Previous task"})
        app.say("Previous task transcript.")
        app.command("home")
        app.menu_selected("local")
        await pilot.pause()
        await pilot.press("enter")  # Review a GPU model; no selection yet.
        assert app.screen.id == "local-model-choice"
        await pilot.press("escape")
        assert not app.conversation and app.engine.session is old
        assert "Previous task transcript" in app.query_one("#log")._copy_text
        app.select_api_model("cyberkimi")
        await pilot.pause()
        assert app.conversation and app.engine.session is None
        assert not app.engine.running and app.state.active_provider == "api"
        assert app.focused is app.query_one("#prompt")
        assert "Previous task transcript" not in app.query_one("#log")._copy_text
        assert (old.folder / "session.json").exists()  # Saved history remains available.
        app.say("Keep this visible during an in-session model switch.")
        app.select_api_model("cyberglm")
        assert "Keep this visible" in app.query_one("#log")._copy_text
        assert not notifications


async def test_only_completed_update_shows_popup_once(settings, monkeypatch):
    active = {
        "schema": 1, "entrypoint": "engine/bin/interpreter",
        "backup": "engine-updates/backup-test", "release": {"tag": "rust-v0.0.99"},
    }
    (settings.runtime / ACTIVE_FILE).write_text(json.dumps(active))
    (settings.runtime / UPDATE_NOTICE_FILE).write_text(json.dumps({
        "activation": active["backup"], "entrypoint": active["entrypoint"],
        "tag": active["release"]["tag"],
    }))
    notifications = []
    for _ in range(2):
        app = VelocicyberApp(settings)
        monkeypatch.setattr(app, "notify", lambda message, **kw: notifications.append(message))
        async with app.run_test() as pilot:
            app.say("Routine status message.")
            await pilot.pause()
            assert app.query_one("#home-notice").display
            assert app.query_one("#home-notice").render().plain == "Routine status message."
            assert notifications == ["Velocicyber engine updated to rust-v0.0.99."]
            app.command("new")
            app.say("Routine session message.")
            await pilot.pause()
            assert not app.query_one("#home-notice").display
            assert len(notifications) == 1


def test_tool_argument_error_is_visible_and_allows_recovery(settings):
    app = VelocicyberApp(settings)
    messages = []
    app.say = messages.append
    app.busy = True
    app.task_failed = False
    app.engine_event(
        {
            "type": "tool.error",
            "tool": "write",
            "message": "failed to parse function arguments: missing field `path`",
        }
    )
    assert "Tool error (write)" in messages[0]
    assert "missing field `path`" in messages[0]
    assert not app.task_failed


@pytest.mark.parametrize("completion", ["actions_completed", "response_incomplete"])
async def test_completed_actions_without_summary_show_harness_notice(settings, monkeypatch, completion):
    app = VelocicyberApp(settings)
    app.state.collection = app.state.upload_approved = False
    calls = []

    async def run(prompt, callback):
        calls.append(prompt)
        callback({"type": "turn.completed"})
        if len(calls) == 1:
            callback(
                {
                    "type": "engine.notice",
                    "code": "missing_final_summary",
                    "completion_status": completion,
                    "message": "1 tool action completed successfully. No final summary.",
                }
            )
        else:
            callback(
                {"type": "item.completed", "item": {"type": "agent_message", "text": "Ready."}}
            )

    monkeypatch.setattr(app.engine, "run", run)
    async with app.run_test(size=(100, 30)):
        await app.run_prompt("Delete the test files.").wait()
        transcript = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
        assert "Harness: 1 tool action completed successfully" in transcript
        assert "Velocicyber:" not in transcript
        expected = (
            "Actions completed — no model summary" if completion == "actions_completed"
            else "Task paused — no model summary; enter a follow-up"
        )
        assert not app.task_failed and app.activity == expected
        await app.run_prompt("Continue.").wait()
        assert "Task complete" in app.activity and not app.completion_notice
        assert len(calls) == 2


async def choose(pilot, index):
    app = pilot.app
    if isinstance(app.screen, Menu):
        options = app.screen.query_one(OptionList)
        options.highlighted = options.get_option_index(f"choice-{index}")
    else:
        options = app.query_one("#commands", OptionList)
        options.highlighted = index
    options.action_select()
    await pilot.pause()


async def test_menu_collection_and_demo_gpu(settings):
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("f2")
        await choose(pilot, 4)
        await choose(pilot, 0)
        assert app.state.collection
        await pilot.press("f2")
        await choose(pilot, 0)
        await choose(pilot, 0)
        await choose(pilot, 3)
        await pilot.pause(0.8)
        assert app.state.gpu_status == GPUStatus.RUNNING
        assert app.state.model_ready
        assert settings.local_model in str(app.query_one("#dashboard", Static).render())
        app.confirm_termination()
        assert not app.state.model_ready
        assert not settings.data.exists()


async def test_delete_data_menu_requires_explicit_confirmation(settings, episode, monkeypatch):
    raw = RawStore(settings.data).append(episode)
    app = VelocicyberApp(settings)
    monkeypatch.setattr(app.uploader, "upload", lambda *a, **kw: pytest.fail("Unexpected upload"))
    async with app.run_test(size=(80, 24)) as pilot:
        app.state.collection = app.state.upload_approved = True
        app.query_one("#prompt", Input).value = "\\data-collection"
        await pilot.pause()
        await pilot.press("enter")
        assert isinstance(app.screen, Menu)
        app.screen.query_one("#menu-filter", Input).value = "delete"
        await pilot.pause()
        assert app.screen.query_one(OptionList).option_count == 1
        await pilot.press("enter")
        assert "Hugging Face data stays intact" in app.screen.heading
        assert "cannot be undone" in app.screen.heading
        assert app.screen.query_one(OptionList).highlighted_option.id == "choice-0"
        label = app.screen.query_one(Label)
        assert label.size.height >= 5
        assert app.screen.query_one("#menu-filter", Input).region.bottom < app.size.height
        await pilot.press("enter")  # Cancel is the default.
        assert raw.exists()
        assert app.state.collection and app.state.upload_approved
        app.menu_selected("data-collection")
        await pilot.pause()
        await choose(pilot, 2)
        await pilot.press("down", "enter")
        await app.workers.wait_for_complete()
        assert not settings.data.exists()
        assert not app.state.collection and not app.state.upload_approved
        assert not app.data_busy


async def test_delete_data_demo_preserves_files(settings, episode):
    raw = RawStore(settings.data).append(episode)
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test():
        app.confirm_data_deletion("delete-all")
        await app.workers.wait_for_complete()
        assert raw.exists()
        assert not app.data_busy


async def test_data_collection_submenu_manual_upload(settings, episode, monkeypatch):
    RawStore(settings.data).append(episode)
    artifact = Curator(settings.data, Scrubber()).run()
    app = VelocicyberApp(settings)
    app.state.upload_approved = False  # Exercise the explicit opt-out and re-enable controls.
    calls = []

    def upload(path, *, approved):
        calls.append(path)
        assert approved and path == artifact
        return {"training_hash": "abc123", "unchanged": False}

    monkeypatch.setattr(app.uploader, "upload", upload)
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", Input).value = "/"
        await pilot.pause()
        commands = app.query_one("#commands", OptionList)
        ids = {commands.get_option_at_index(i).id for i in range(commands.option_count)}
        assert "data-collection" in ids
        assert "upload-perms" in ids
        assert not {"data", "upload", "upload-pending", "upload-manual"} & ids
        app.query_one("#prompt", Input).value = "/data-collection"
        await pilot.pause()
        await pilot.press("enter")
        app.screen.query_one("#menu-filter", Input).value = "manual upload"
        await pilot.pause()
        assert app.screen.query_one(OptionList).option_count == 1
        assert str(app.screen.query_one(OptionList).highlighted_option.prompt) == "Manual Upload"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert not calls  # Upload approval remains required.
        app.query_one("#prompt", Input).value = "/upload-perms"
        await pilot.pause()
        await pilot.press("enter")
        assert isinstance(app.screen, Menu)
        assert "Automatic uploads" in app.screen.heading
        await choose(pilot, 0)
        assert app.state.upload_approved
        app.menu_selected("data-collection")
        await pilot.pause()
        await choose(pilot, 3)
        await app.workers.wait_for_complete()
        assert calls == [artifact]
        app.command("upload-pending")
        await pilot.pause()
        assert calls == [artifact]


async def test_compaction_menu_filters_yes_no_and_respects_busy_state(settings):
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(80, 24)) as pilot:
        assert app.state.allow_compaction
        app.menu_selected("data-collection")
        await pilot.pause()
        app.screen.query_one("#menu-filter", Input).value = "compaction"
        await pilot.pause()
        assert app.screen.query_one(OptionList).option_count == 1
        assert (
            str(app.screen.query_one(OptionList).highlighted_option.prompt) == "Allow compaction?"
        )
        await pilot.press("enter")
        assert "Currently YES" in app.screen.heading
        app.screen.query_one("#menu-filter", Input).value = "no"
        await pilot.pause()
        await pilot.press("enter")
        assert not app.state.allow_compaction
        assert not app.state.collection and not app.state.upload_approved
        app.data_selected("allow-compaction")
        await pilot.pause()
        assert "Currently NO" in app.screen.heading
        await pilot.press("escape")
        assert not app.state.allow_compaction
        app.busy = True
        app.set_allow_compaction(True)
        assert not app.state.allow_compaction
        app.busy = False
        app.data_selected("allow-compaction")
        await pilot.pause()
        await choose(pilot, 0)
        assert app.state.allow_compaction


async def test_manual_upload_uses_compaction_policy(settings, compacted_episode, monkeypatch):
    from velocicyber.datasets.pipeline import dataset_summary

    raw = RawStore(settings.data).append(compacted_episode)
    app = VelocicyberApp(settings)
    app.state.collection = app.state.upload_approved = True
    counts = []

    def upload(path, *, approved):
        assert approved
        counts.append(dataset_summary(path)["eligible"])
        return None

    monkeypatch.setattr(app.uploader, "upload", upload)
    async with app.run_test():
        app.state.allow_compaction = False
        await app.flush_uploads()
        app.state.allow_compaction = True
        await app.flush_uploads()
        app.state.upload_approved = False
        with pytest.raises(PermissionError):
            await app.flush_uploads()
        assert counts == [0, 1] and raw.exists()


async def test_delete_data_blocks_concurrent_tasks_and_uploads(settings, episode, monkeypatch):
    raw = RawStore(settings.data).append(episode)
    app = VelocicyberApp(settings)
    started, release = asyncio.Event(), asyncio.Event()

    async def uploading():
        started.set()
        await release.wait()

    async with app.run_test() as pilot:
        app.busy = True
        app.data_selected("delete")
        app.confirm_data_deletion("delete-all")
        assert not isinstance(app.screen, Menu)
        assert not app.data_busy and raw.exists()
        app.busy = False
        upload = app.run_worker(uploading(), group="upload")
        await started.wait()
        app.confirm_data_deletion("delete-all")
        assert not app.data_busy and raw.exists()
        release.set()
        await upload.wait()
        await pilot.pause()

        # Hold an approved deletion pending and try every competing entry point.
        monkeypatch.setattr(app, "delete_finetuning_data", lambda: None)
        app.state.collection = app.state.upload_approved = True
        app.confirm_data_deletion("delete-all")
        assert app.data_busy
        app.set_collection(True)
        app.set_upload(True)
        await app.run_prompt("Do not start a task during deletion").wait()
        with pytest.raises(RuntimeError, match="deletion"):
            await app.flush_uploads()
        assert not app.busy
        assert not app.state.collection and not app.state.upload_approved
        assert raw.exists()
        app.data_busy = False


async def test_backslash_menu_and_small_terminal(settings):
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(80, 28)) as pilot:
        assert not app.conversation and app.query_one("#main-menu").display
        assert not app.query_one("#status-panel").display
        guide = str(app.query_one("#status-guide", Static).render())
        for label in (
            "Updates",
            "Private model",
            "GPUs",
            "API",
            "GPU inference",
            "Pi CPU",
            "Using",
            "Data",
            "Access",
        ):
            assert label + ":" in guide
        actions = app.query_one("#home-actions", Static)
        assert actions.render().plain.splitlines() == [
            "/new to start a new session",
            "/resume to pick up where you left off.",
        ]
        assert app.query_one("#status-guide").region.y >= app.query_one("#title").region.bottom
        assert actions.region.bottom <= app.query_one("#prompt").region.y
        app.query_one("#prompt", Input).value = "\\"
        await pilot.pause()
        commands = app.query_one("#commands", OptionList)
        assert commands.display and commands.option_count >= 6
        prompt = app.query_one("#prompt", Input)
        assert commands.region.bottom <= prompt.region.y
        assert prompt.region.bottom < app.size.height
        assert not isinstance(app.screen, Menu)
        await pilot.press("l", "o", "c")
        assert commands.option_count == 1
        assert commands.highlighted_option.id == "local"
        await pilot.press("enter")
        assert isinstance(app.screen, Menu)
        assert app.screen.id == "local-models"
        assert "Turn GPUs ON" in str(app.screen.query_one(Label).render())
        assert not app.screen.query_one(OptionList).get_option("choice-0").disabled
        await pilot.press("escape")
        assert not isinstance(app.screen, Menu)
        assert title_art().plain in str(app.query_one("#title", Static).render())
        await pilot.resize_terminal(120, 28)
        await pilot.pause()
        assert title_art().plain in str(app.query_one("#title", Static).render())
        await pilot.press("slash", *"new", "enter")
        assert app.conversation and not app.query_one("#main-menu").display
        assert app.query_one("#status-panel").display
        assert not app.busy and app.engine.session is None
        app.say("Retain this session text.")
        app.command("home")
        await pilot.pause()
        assert app.query_one("#main-menu").display and not app.conversation
        app.command("chat")
        await pilot.pause()
        assert "Retain this session text." in "\n".join(
            line.text for line in app.query_one("#log", RichLog).lines
        )


async def test_alternative_model_selection_preserves_configured_default(settings):
    default = settings.local_model
    alternative = "example-org/bf16-model"
    settings.options.update(HUGGINGFACE_MODEL=default, LOCAL_MODEL_OPTIONS=alternative)
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(140, 45)) as pilot:
        app.menu_selected("local")
        await pilot.pause()
        assert isinstance(app.screen, Menu)
        assert not app.screen.query_one(OptionList).get_option("choice-0").disabled
        app.state.gpu_status = GPUStatus.RUNNING
        app.animate()
        await pilot.pause()
        assert [repo for repo, _ in app.screen.options][:2] == [default, alternative]
        assert not app.screen.query_one(OptionList).get_option("choice-0").disabled
        await choose(pilot, 1)
        assert app.screen.id == "local-model-choice"
        assert settings.local_model == default and not app.state.model_ready
        assert not app.conversation
        await choose(pilot, 0)
        await pilot.pause()
        assert settings.local_model == alternative
        assert app.state.model_ready and app.state.active_provider == "local"
        assert app.conversation and app.engine.session is None
        assert settings.model_choices == [default, alternative]
        assert settings.options["HUGGINGFACE_MODEL"] == default


async def test_rejected_model_switch_keeps_current_model(settings, monkeypatch):
    from velocicyber.providers.lambda_cloud import ModelSpec

    alternative = "example-org/bf16-model"
    app = VelocicyberApp(settings)
    default = settings.local_model
    app.state.total_vram_gib = 40
    app.state.gpu_status = GPUStatus.RUNNING
    app.state.model_ready = True
    app.state.active_provider = "local"
    monkeypatch.setattr(
        app.gpu,
        "inspect_model",
        lambda repo: ModelSpec(repo, "revision", 55563008568, "qwen3_coder", "qwen3", 24),
    )
    monkeypatch.setattr(app.gpu, "refresh", lambda: "active")
    async with app.run_test(size=(140, 45)):
        await app.gpu_operation("load", alternative).wait()
        assert settings.local_model == default
        assert app.state.model_ready and app.state.active_provider == "local"


async def test_user_quit_waits_for_confirmed_gpu_termination(settings, monkeypatch):
    app = VelocicyberApp(settings)
    app.termination_poll_seconds = 0.001
    calls = []
    exited = asyncio.Event()

    def terminate(*, approved):
        assert approved
        calls.append("terminate")

    def refresh():
        calls.append("refresh")
        if calls.count("refresh") == 1:
            return "terminating"
        app.state.instance_id = ""
        app.state.gpu_status = GPUStatus.STOPPED
        return "terminated"

    monkeypatch.setattr(app.gpu, "terminate", terminate)
    monkeypatch.setattr(app.gpu, "refresh", refresh)
    monkeypatch.setattr(app, "exit", lambda: (calls.append("exit"), exited.set()))
    async with app.run_test(size=(140, 45)):
        app.state.instance_id = "test-instance"
        app.action_quit()
        await asyncio.wait_for(exited.wait(), 5)
        assert calls == ["terminate", "refresh", "refresh", "exit"]
        assert not app.state.instance_id


async def test_download_error_keeps_gpu_for_the_session(settings, monkeypatch):
    app = VelocicyberApp(settings)
    app.state.total_vram_gib = 40
    app.state.gpu_status = GPUStatus.RUNNING
    monkeypatch.setattr(app.gpu, "inspect_model", lambda repo: SimpleNamespace(required_gib=36))
    monkeypatch.setattr(app.gpu, "refresh", lambda: "active")

    async def failed_download():
        app.state.model_stage = "download"
        raise RuntimeError("Interrupted transfer")

    monkeypatch.setattr(app.gpu, "deploy", failed_download)
    calls = []
    monkeypatch.setattr(app.gpu, "terminate", lambda **kwargs: calls.append("terminate"))
    async with app.run_test(size=(140, 45)):
        app.state.instance_id = "retained-instance"
        await app.gpu_operation("load").wait()
        assert not calls and app.state.instance_id == "retained-instance"
        assert not app.exit_after_termination
        assert not app.state.model_stage


async def test_compact_model_download_status_and_readiness(settings):
    app = VelocicyberApp(settings)
    async with app.run_test(size=(80, 24)) as pilot:
        app.show_conversation()
        app.state.gpu_status = GPUStatus.RUNNING
        app.state.download_total = 1000
        app.gpu.deployment_progress('{"stage":"download_progress","bytes":600,"total":1000}')
        app.animate()
        await pilot.pause()
        dashboard = app.query_one("#dashboard", Static)
        assert "GPU inference: UPLOADING MODEL [||||||    ] 60%" in str(dashboard.render())
        assert dashboard.region.height == 7
        assert app.query_one("#log").region.height >= 9
        assert not app.state.model_ready
        for event, label in [
            ("download_verifying", "VERIFYING MODEL"),
            ("server_starting", "STARTING SERVER"),
            ("model_loading", "LOADING MODEL"),
        ]:
            app.gpu.deployment_progress('{"stage":"' + event + '"}')
            app.animate()
            assert "GPU inference: " + label in str(dashboard.render())
            assert "UPLOADING MODEL" not in str(dashboard.render())
        app.state.model_ready = True
        app.animate()
        assert "GPU inference: Ready for coding" in str(dashboard.render())
        app.state.model_ready = False
        app.state.model_stage = "failed"
        app.animate()
        assert "GPU inference: STARTUP FAILED" in str(dashboard.render())


async def test_timeout_and_empty_uploads_are_readable(settings, episode, monkeypatch):
    episode["messages"][-1]["content"] = ""
    RawStore(settings.data).append(episode)
    app = VelocicyberApp(settings)
    app.state.collection = app.state.upload_approved = True

    async def run(prompt, callback):
        callback(
            {
                "type": "item.completed",
                "item": {"type": "error", "message": "Model metadata unavailable."},
            }
        )
        callback({"type": "item.completed", "item": {"type": "agent_message", "text": ""}})
        callback({"type": "turn.completed"})
        callback(
            {"type": "engine.error", "message": "Model response timed out; no complete answer."}
        )
        callback(
            {
                "type": "dataset.prepared",
                "path": "/fixture/artifact",
                "eligible": 0,
                "excluded": 1,
                "exclusion_reasons": {"no_complete_answer": 1},
            }
        )
        raise RuntimeError("Model response timed out; no complete answer.")

    monkeypatch.setattr(app.engine, "run", run)
    monkeypatch.setattr(
        app.uploader, "client", lambda: pytest.fail("Empty dataset must not contact Hugging Face")
    )
    async with app.run_test(size=(80, 24)):
        await app.run_prompt("Fixture task").wait()
        assert app.task_failed and "Task failed" in app.activity
        await app.flush_uploads()  # Manual attempt also explains why it sends nothing.
        transcript = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
        assert "Engine notice: Model metadata unavailable." in transcript
        assert "Model response timed out" in transcript
        assert "0 eligible rows" in transcript and "no complete answer: 1" in transcript
        assert "Hugging Face: nothing eligible to upload." in transcript
        assert "'type':" not in transcript and '"type":' not in transcript
        assert "Velocicyber:" not in transcript
        assert "Task complete" not in app.activity


async def test_upload_failure_does_not_hide_successful_reply(settings, monkeypatch):
    app = VelocicyberApp(settings)
    app.state.collection = app.state.upload_approved = True

    async def run(prompt, callback):
        callback(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Your task is complete."},
            }
        )
        callback(
            {"type": "dataset.prepared", "path": "/fixture/artifact", "eligible": 1, "excluded": 0}
        )

    async def failed_upload():
        raise RuntimeError("fixture upload failure")

    monkeypatch.setattr(app.engine, "run", run)
    monkeypatch.setattr(app, "flush_uploads", failed_upload)
    async with app.run_test(size=(80, 24)):
        await app.run_prompt("Fixture task").wait()
        transcript = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
        assert "Your task is complete." in transcript
        assert "Hugging Face upload failed" in transcript
        assert not app.task_failed and "Task complete" in app.activity


async def test_resume_menu_filters_restores_transcript_and_waits_for_prompt(settings, monkeypatch):
    app = VelocicyberApp(settings)
    notifications = []
    monkeypatch.setattr(app, "notify", lambda message, **kwargs: notifications.append(message))
    saved = app.engine.ensure_session("Repair the parser")
    thread = str(uuid.uuid4())
    (saved.home / "sessions").mkdir()
    (saved.home / "sessions" / f"rollout-fixture-{thread}.jsonl").write_text("{}\n")
    saved.save(thread_id=thread, model=settings.api_model, task_id="parser-task")
    saved.append({"type": "user.prompt", "text": "Repair the parser"})
    saved.append(
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "id": "legacy-command",
                "command": ["python3", "check.py"],
                "aggregated_output": "Parser checks passed.",
                "status": "completed",
                "exit_code": 0,
            },
        }
    )
    saved.append(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "The parser fix is ready."},
        }
    )
    app.engine.new_task()
    monkeypatch.setattr(
        app.engine, "run", lambda *a: pytest.fail("Resume must wait for user input")
    )
    monkeypatch.setattr(
        app.uploader, "upload", lambda *a, **k: pytest.fail("Resume must not upload")
    )
    async with app.run_test(size=(80, 24)) as pilot:
        assert not app.state.collection and not app.state.upload_approved
        await pilot.press("slash", *"resume", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.screen.id == "saved-sessions"
        await pilot.press(*"parser")
        assert app.screen.query_one(OptionList).option_count == 1
        await pilot.press("enter")
        assert app.engine.thread_id == thread and app.engine.task_id == "parser-task"
        assert app.state.active_provider == "api"
        assert app.conversation and not app.busy
        transcript = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
        assert "Repair the parser" in transcript and "The parser fix is ready." in transcript
        assert "$ python3 check.py" in transcript and "Parser checks passed." in transcript
        assert "Next reply uses CyberKimi" in transcript
        assert not notifications
        assert not settings.data.exists()


async def test_opted_in_collection_uploads_eligible_rows_and_can_be_disabled(
    settings, episode, monkeypatch
):
    app = VelocicyberApp(settings)
    app.state.collection = app.state.upload_approved = True
    uploads = []
    episode["outcome"] = {
        "success": None,
        "verifier": "none",
        "turn_completed": True,
        "exit_code": 0,
    }

    async def run(prompt, callback):
        assert app.state.collection
        RawStore(settings.data).append(episode)
        callback(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Here is the coding task response."},
            }
        )

    def upload(artifact, *, approved):
        from velocicyber.datasets.upload import Uploader

        assert approved
        _, records = Uploader(settings).local_artifact(artifact)
        assert len(records) == 1
        assert records[0][1]["meta"]["verification"] == "unverified"
        uploads.append(artifact)
        return {"training_hash": "test-hash"}

    monkeypatch.setattr(app.engine, "run", run)
    monkeypatch.setattr(app.uploader, "upload", upload)
    async with app.run_test(size=(80, 24)):
        await app.run_prompt("Fix the parser").wait()
        assert len(uploads) == 1 and not app.task_failed
        app.set_upload(False)
        app.set_collection(False)
        assert not app.state.upload_approved and not app.state.collection


@pytest.mark.parametrize("result", ["completed", "failed"])
async def test_compaction_notice_stays_visible_while_summarizing(settings, result):
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(80, 24)):
        app.busy = True
        app.engine_event({"type": "context.compaction.started"})
        assert app.compacting and app.activity == "Compacting conversation…"
        app.engine_event(
            {"type": "provider.progress", "stage": "Writing response", "text": "Internal summary"}
        )
        assert app.activity == "Compacting conversation…" and not app.live_text
        app.engine_event({"type": "context.compaction." + result})
        assert not app.compacting
        transcript = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
        assert "Compacting conversation" in transcript
        assert (
            "Compaction complete" if result == "completed" else "Compaction failed"
        ) in transcript


async def test_quit_during_launch_waits_for_the_instance_id(settings, monkeypatch):
    app = VelocicyberApp(settings)
    app.gpu_busy = True
    app.termination_poll_seconds = 0.001
    release_launch, exited = asyncio.Event(), asyncio.Event()
    calls = []

    async def finish_launch():
        await release_launch.wait()
        app.state.instance_id = "new-instance"

    def terminate(*, approved):
        assert approved and app.state.instance_id == "new-instance"
        calls.append("terminate")

    def refresh():
        app.state.instance_id = ""
        return "terminated"

    monkeypatch.setattr(app.gpu, "terminate", terminate)
    monkeypatch.setattr(app.gpu, "refresh", refresh)
    monkeypatch.setattr(app, "exit", exited.set)
    async with app.run_test(size=(140, 45)) as pilot:
        app.launch_task = asyncio.create_task(finish_launch())
        app.action_quit()
        await pilot.pause()
        assert not calls and not exited.is_set()
        release_launch.set()
        await asyncio.wait_for(exited.wait(), 5)
        assert calls == ["terminate"]


async def test_api_picker_requires_explicit_model_selection(settings):
    app = VelocicyberApp(settings, demo=True)
    app.state.active_provider = "local"
    app.state.gpu_status = GPUStatus.RUNNING
    app.state.model_ready = True

    def model_colors():
        display = app.query_one("#dashboard", Static).render()
        return [
            next(style.foreground.hex for text, style in display.render() if name in text)
            for name in (settings.local_model, "CyberKimi")
        ]

    async with app.run_test(size=(80, 28)) as pilot:
        assert model_colors() == ["#39FF14", "#9399A3"]
        await pilot.press("backslash", "a", "p", "i", "enter")
        assert isinstance(app.screen, Menu)
        assert app.screen.options == [
            ("cyberkimi", "CyberKimi — Adverserial.ai"),
            ("cyberglm", "CyberGLM — Adverserial.ai"),
            ("cyberseek", "CyberSeek — Adverserial.ai"),
        ]
        assert app.state.active_provider == "local"
        await pilot.press("escape")
        assert app.state.active_provider == "local"
        await pilot.press("f2")
        await choose(pilot, 2)
        await choose(pilot, 0)
        assert app.state.active_provider == "api"
        app.animate()
        assert model_colors() == ["#9399A3", "#39FF14"]
        app.state.api_health = "payment_required"
        app.animate()
        display = app.query_one("#dashboard", Static).render()
        error_color = next(
            style.foreground for text, style in display.render() if "payment required" in text
        )
        assert error_color.r > error_color.g and error_color.r > error_color.b


async def test_command_filter_navigation_escape_and_arguments(settings, monkeypatch):
    app = VelocicyberApp(settings, demo=True)
    calls = []
    monkeypatch.setattr(app, "run_prompt", calls.append)
    async with app.run_test(size=(80, 28)) as pilot:
        prompt = app.query_one("#prompt", Input)
        commands = app.query_one("#commands", OptionList)
        await pilot.press("backslash", "down", "enter")
        assert app.screen.id == "local-models"
        await pilot.press("escape")
        for query in ("\\cloud", "/local & cloud models", "/local"):
            prompt.value = query
            await pilot.pause()
            assert commands.option_count == 1
            assert str(commands.highlighted_option.prompt).startswith(
                query[0] + "local & cloud models"
            )
            await pilot.press("tab")
            assert prompt.value == query[0] + "local & cloud models"
            await pilot.press("escape", "enter")  # Direct submission without the palette.
            assert app.screen.id == "local-models"
            assert app.screen.heading.startswith("Local & Cloud Models")
            await pilot.press("escape")
        assert not calls and app.gpu._client is None
        await pilot.press("slash", "a", "p", "tab")
        assert prompt.value == "/api"
        await pilot.press("escape")
        assert not commands.display
        prompt.value = "\\nothingmatches"
        await pilot.pause()
        assert commands.highlighted_option is None
        await pilot.press("enter")
        assert not calls
        prompt.value = "\\verif"
        await pilot.pause()
        assert commands.highlighted_option is None
        prompt.value = "/"
        await pilot.pause()
        ids = {commands.get_option_at_index(i).id for i in range(commands.option_count)}
        assert not {"task", "verifier"} & ids
        prompt.value = "/data-collection"
        await pilot.pause()
        await pilot.press("enter", *"advanced", "enter")
        assert app.screen.id == "data-advanced"
        assert "Verification command: OFF" in app.screen.heading
        await pilot.press(*"verification", "enter")
        assert prompt.value == "\\verifier "
        assert not commands.display
        await pilot.press("p", "y", "t", "e", "s", "t", "enter")
        assert app.engine.verifier == ["pytest"]
        prompt.value = "/data-collection"
        await pilot.pause()
        await pilot.press("enter", *"advanced", "enter")
        assert "Verification command: pytest" in app.screen.heading
        await pilot.press(*"task", "enter")
        assert prompt.value == "\\task "
        assert app.engine.task_id is None
        await pilot.press(*"parser-fix", "enter")
        assert app.engine.task_id == "parser-fix" and app.engine.verifier == ["pytest"]
        app.command('verifier "unterminated')
        assert app.engine.verifier == ["pytest"]
        prompt.value = "/verifier off"  # Direct aliases remain available.
        await pilot.pause()
        await pilot.press("enter")
        assert not app.engine.verifier and not calls
        prompt.value = "Just a task"
        await pilot.pause()
        assert not commands.display
        await pilot.press("enter")
        assert calls == ["Just a task"]


async def test_saved_gpu_refreshes_once_and_static_title(settings, monkeypatch):
    State(instance_id="saved-instance").persist(settings.runtime / "gpu.json")
    app = VelocicyberApp(settings)
    calls = []

    def refresh():
        calls.append("refresh")
        app.state.gpu_status = GPUStatus.RUNNING
        return "active"

    monkeypatch.setattr(app.gpu, "refresh", refresh)
    async with app.run_test(size=(80, 28)) as pilot:
        await app.workers.wait_for_complete()
        assert calls == ["refresh"]
        assert not app.restoring_gpu
        assert app.state.gpu_status == GPUStatus.RUNNING
        assert not app.state.model_ready
        title = str(app.query_one("#title", Static).render())
        assert not app.query("#raptor")
        first = app.query_one("#dashboard", Static).render()
        assert "GPUs: RUNNING" in str(first)
        assert "reconnect / load" in str(first)
        app.frame += 4
        app.animate()
        second = app.query_one("#dashboard", Static).render()
        assert first.spans == second.spans
        assert (
            next(
                style.foreground.hex
                for text, style in first.render()
                if settings.local_model in text
            )
            == "#9399A3"
        )
        assert (
            next(style.foreground.hex for text, style in first.render() if "CyberKimi" in text)
            == "#39FF14"
        )
        assert str(app.query_one("#title", Static).render()) == title
        assert app.query_one("#title").styles.color.hex == "#39FF14"
        colors = {
            style.foreground.hex
            for _, style in app.query_one("#title", Static).render().render()
            if style.foreground
        }
        assert {"#39FF14", "#FF39C8"} <= colors
        assert app.screen.styles.background.hex == "#000000"
        await pilot.pause(0.3)
        assert calls == ["refresh"]


async def test_conversation_progress_and_tool_output_in_small_terminal(settings, monkeypatch):
    app = VelocicyberApp(settings)
    started, finish = asyncio.Event(), asyncio.Event()

    async def run(prompt, on_event):
        on_event({"type": "turn.started"})
        on_event(
            {"type": "provider.progress", "stage": "Writing response", "text": "Checking files"}
        )
        started.set()
        await finish.wait()
        command = {
            "id": "tool1",
            "type": "command_execution",
            "command": "ls",
            "aggregated_output": "",
        }
        on_event({"type": "item.started", "item": command})
        on_event({"type": "item.updated", "item": {**command, "aggregated_output": "file.py\n"}})
        on_event(
            {
                "type": "item.completed",
                "item": {
                    **command,
                    "aggregated_output": "file.py\n",
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        )
        on_event(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Found file.py."}}
        )
        on_event({"type": "turn.completed"})

    monkeypatch.setattr(app.engine, "run", run)
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", Input).value = "Inspect the files"
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), 5)
        await pilot.pause(0.3)
        assert app.conversation and app.busy
        log = app.query_one("#log", RichLog)
        assert "Chat model: CyberKimi" in log.border_title
        assert "bold" in log.border_title and "#39ff14" in log.border_title.lower()
        assert log.styles.border_title_align == "left"
        # GPU startup may finish during an API reply. Keep the reply's actual
        # model in the chat border until that reply finishes.
        app.state.active_provider = "local"
        app.state.model_ready = True
        app.state.gpu_status = GPUStatus.RUNNING
        app.animate()
        assert "CyberKimi" in log.border_title
        assert app.query_one("#masthead").region.height == 9
        assert "Writing response" in str(app.query_one("#activity", Static).render())
        preview = app.query_one("#live-response", RichLog)
        assert preview.display
        assert "Checking files" in "".join(line.text for line in preview.lines)
        assert app.query_one("#prompt").region.bottom < app.size.height
        finish.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        transcript = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
        assert transcript.count("$ ls") == 1
        assert transcript.count("file.py") == 2  # One tool output, one response.
        assert "Command completed · exit 0" in transcript
        segments = [segment for line in app.query_one("#log", RichLog).lines for segment in line]
        assert any(
            "$ ls" in segment.text and segment.style.color.get_truecolor().hex == "#ff39c8"
            for segment in segments
            if segment.style and segment.style.color
        )
        assert any(
            segment.text.strip() == "file.py"
            and segment.style.color.get_truecolor().hex == "#9399a3"
            for segment in segments
            if segment.style and segment.style.color
        )
        assert not preview.display and not app.busy
        assert settings.local_model in log.border_title
        assert "Task complete" in str(app.query_one("#activity", Static).render())
        assert app.query_one("#log").region.height >= 9
        title = app.query_one("#title", Static)
        assert title.region.height == 9
        assert max(map(len, title.render().plain.splitlines())) <= title.size.width


async def test_stop_task_shortcut_and_gpu_off_label(settings, monkeypatch):
    app = VelocicyberApp(settings, demo=True)
    stopped = asyncio.Event()

    async def cancel():
        stopped.set()

    monkeypatch.setattr(app.engine, "cancel", cancel)
    async with app.run_test(size=(80, 24)) as pilot:
        app.busy = True
        await pilot.press("ctrl+x")
        assert stopped.is_set() and app.stop_requested
        app.busy = False
        app.state.gpu_status = GPUStatus.STOPPING
        app.animate()
        assert "GPUs: Turning OFF" in str(app.query_one("#dashboard", Static).render())
        app.state.gpu_status = GPUStatus.STOPPED
        app.animate()
        text = str(app.query_one("#dashboard", Static).render())
        assert "GPUs: OFF" in text
        assert "INITIALIZED" not in text


async def test_all_submenus_filter_and_select_from_keyboard(settings):
    alternative = "example-org/bf16-model"
    settings.options["LOCAL_MODEL_OPTIONS"] = alternative
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("backslash", *"local", "enter", *"bf16", "enter", "enter")
        assert settings.local_model == alternative
        assert app.state.gpu_status == GPUStatus.STOPPED
        assert not app.state.model_ready and app.state.active_provider == "api"
        await pilot.press("backslash", "g", "p", "u", "enter")
        assert isinstance(app.screen, Menu)
        assert app.screen.styles.background.a == 0
        assert (
            app.screen.query_one(OptionList).styles.border.top
            == app.query_one("#commands", OptionList).styles.border.top
        )
        await pilot.press("o", "n", "enter")
        assert "Total GPU memory" in app.screen.heading
        assert alternative in app.screen.heading
        assert [size for size, _ in app.screen.options] == [
            24,
            40,
            48,
            80,
            180,
            360,
            640,
            720,
            1440,
            1800,
        ]
        await pilot.press("8", "0", "enter")
        await app.workers.wait_for_complete()
        assert settings.local_model == alternative
        assert app.state.gpu_status == GPUStatus.RUNNING
        await pilot.press("backslash", "a", "p", "i", "enter")
        await pilot.press(*"cyber")
        assert app.screen.query_one(OptionList).option_count == 3
        await pilot.press(*"kimi")
        assert app.screen.query_one(OptionList).option_count == 1
        await pilot.press("enter")
        assert app.state.active_provider == "api"
        await pilot.press("backslash", "d", "a", "t", "a", "enter")
        await pilot.press("o", "n", "enter")
        assert app.state.collection


async def test_blackfrost_selection_shows_requirements_without_launching(settings):
    from velocicyber.models.gpu_profiles import CYBER_FROST_REPO

    settings.options["LOCAL_MODEL_OPTIONS"] = CYBER_FROST_REPO
    app = VelocicyberApp(settings)
    default = settings.local_model
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("backslash", *"local", "enter", *"blackfrost")
        assert all(
            repo == label for repo, label in app.screen.options if repo in settings.model_choices
        )
        await pilot.press("enter")
        assert app.screen.id == "local-model-choice"
        assert [label for _, label in app.screen.options] == [
            "Confirm selected model choice",
            "More Details",
        ]
        assert settings.local_model == default and app.gpu._client is None
        await pilot.press("escape")
        assert settings.local_model == default
        await pilot.press("backslash", *"local", "enter", *"blackfrost", "enter", *"more", "enter")
        assert app.screen.id == "local-model-details"
        assert "360 GB" in app.screen.details and "6–14 hours" in app.screen.details
        assert "720 GB (4× B200)" in app.screen.details
        assert "512 GiB host RAM" in app.screen.details
        assert "Projected = untested" in app.screen.details
        assert settings.local_model == default and app.gpu._client is None
        await pilot.press("pagedown")
        assert app.screen.query_one("#model-details-body").scroll_y > 0
        assert app.screen.query_one("#menu-filter").region.bottom < app.size.height
        await pilot.press("enter")  # Back to the two choices.
        assert app.screen.id == "local-model-choice"
        await pilot.press("enter")  # Confirm.
        assert settings.local_model == CYBER_FROST_REPO
        assert app.state.gpu_status == GPUStatus.STOPPED
        assert not app.state.model_ready and app.state.active_provider == "api"
        assert app.gpu._client is None
        await pilot.press("backslash", *"gpu", "enter", *"on", "enter")
        assert "720 GB (4× B200)" in app.screen.heading
        app.screen.query_one("#menu-filter", Input).value = "cluster 720"
        await pilot.pause()
        options = app.screen.query_one(OptionList)
        assert options.option_count == 2
        assert "720 GB" in str(options.highlighted_option.prompt)
        await pilot.press("escape")
        assert not app.state.instance_id and app.gpu._client is None


async def test_mog_selection_filters_separately_and_shows_single_gpu_guidance(settings):
    from velocicyber.models.gpu_profiles import CYBER_FROST_REPO, MOG_SEC_REPO

    settings.options["LOCAL_MODEL_OPTIONS"] = ",".join((CYBER_FROST_REPO, MOG_SEC_REPO))
    app = VelocicyberApp(settings)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("backslash", *"local", "enter", *"m.o.g.", "enter")
        assert settings.local_model != MOG_SEC_REPO and app.gpu._client is None
        assert app.screen.id == "mog-context"
        await pilot.press(*"64k", "enter", *"more", "enter")
        assert "10–20 min" in app.screen.details
        assert "Download to a running GPU: ~2–5 min" in app.screen.details
        assert "Single 80 GB (H100)" in app.screen.details
        assert "128 GiB host RAM" in app.screen.details
        await pilot.press(*"confirm", "enter")
        assert settings.local_model == MOG_SEC_REPO
        assert app.state.gpu_status == GPUStatus.STOPPED and not app.state.model_ready
        assert app.state.active_provider == "api" and app.gpu._client is None
        await pilot.press("backslash", *"gpu", "enter", *"on", "enter")
        assert "Single 80 GB (H100)" in app.screen.heading
        assert "Harness context: 64K" in app.screen.heading
        app.screen.query_one("#menu-filter", Input).value = "single 80"
        await pilot.pause()
        assert str(app.screen.query_one(OptionList).highlighted_option.prompt) == (
            "80 GB — ($3.29 – $4.29) / hr"
        )
        await pilot.press("escape")
        assert not app.state.instance_id and app.gpu._client is None


async def test_confirmation_palette_filters_and_escape_does_not_approve(settings, monkeypatch):
    app = VelocicyberApp(settings)
    calls = []
    monkeypatch.setattr(app, "terminate_during_startup", lambda: calls.append("terminate"))
    async with app.run_test(size=(80, 24)) as pilot:
        app.state.instance_id = "test-resource"
        await pilot.press("backslash", *"terminate", "enter")
        await pilot.press(*"terminate")
        assert app.screen.query_one(OptionList).option_count == 1
        assert not calls
        await pilot.press("escape")
        assert not calls and not isinstance(app.screen, Menu)


@pytest.mark.parametrize("memory", [80, 720, 1800])
async def test_model_preselection_is_offline_and_gpu_flow_still_requires_approval(
    settings, monkeypatch, memory
):
    default = settings.local_model
    alternative = "example-org/bf16-model"
    settings.options.update(HUGGINGFACE_MODEL=default, LOCAL_MODEL_OPTIONS=alternative)
    app = VelocicyberApp(settings)
    calls = []

    def plan(memory):
        calls.append(("plan", memory, settings.local_model))
        return SimpleNamespace(
            instance_type="test-gpu",
            region="test-region",
            total_vram_gib=memory,
            gpu_count={80: 1, 720: 4, 1800: 10}[memory],
            hourly_cents=199,
            model=SimpleNamespace(repo=settings.local_model),
        )

    monkeypatch.setattr(app.gpu, "plan", plan)
    monkeypatch.setattr(app.gpu, "launch", lambda *args, **kwargs: calls.append("launch"))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("backslash", *"local", "enter", *"bf16", "enter")
        assert not calls and app.gpu._client is None
        assert settings.local_model == default
        await pilot.press("enter")
        assert settings.local_model == alternative
        assert settings.options["HUGGINGFACE_MODEL"] == default
        assert not app.state.instance_id and not app.state.model_ready
        await pilot.press("backslash", *"gpu", "enter", *"on", "enter")
        assert "Total GPU memory" in app.screen.heading
        assert alternative in app.screen.heading
        options = app.screen.query_one(OptionList)
        assert options.get_option_at_index(0).disabled
        assert "Single" in str(options.get_option_at_index(0).prompt)
        assert "Cluster" in str(options.get_option_at_index(6).prompt)
        group = "single" if memory == 80 else "cluster"
        app.screen.query_one("#menu-filter", Input).value = f"{group} {memory}"
        await pilot.pause()
        if memory != 80:
            assert options.option_count == 2  # Heading plus one selectable size.
            assert options.highlighted_option.id == {720: "choice-7", 1800: "choice-9"}[memory]
        await pilot.press("enter")
        await pilot.pause()
        assert calls == [("plan", memory, alternative)]
        assert "Launch test-gpu" in app.screen.heading
        assert alternative in app.screen.heading
        await pilot.press("escape")
        await app.workers.wait_for_complete()
        assert calls == [("plan", memory, alternative)]
        assert not app.state.instance_id


async def test_failed_startup_refresh_leaves_models_visible(settings, monkeypatch):
    State(instance_id="saved-instance").persist(settings.runtime / "gpu.json")
    app = VelocicyberApp(settings)
    calls = []

    def refresh():
        calls.append("refresh")
        raise RuntimeError("Lambda temporarily unavailable")

    monkeypatch.setattr(app.gpu, "refresh", refresh)
    async with app.run_test(size=(80, 28)) as pilot:
        await app.workers.wait_for_complete()
        app.menu_selected("local")
        await pilot.pause()
        assert isinstance(app.screen, Menu)
        assert app.screen.query_one(OptionList).get_option("choice-0").disabled
        assert calls == ["refresh"]


@pytest.mark.parametrize("query,model,name", [
    ("glm", "lordx64/cyberglm", "CyberGLM"),
    ("seek", "adverserial/cyberseek", "CyberSeek"),
])
async def test_api_picker_filters_routes_and_updates_active_model(settings, query, model, name):
    app = VelocicyberApp(settings, demo=True)
    app.state.gpu_status = GPUStatus.RUNNING
    app.state.model_ready = True
    app.state.active_provider = "local"
    async with app.run_test(size=(100, 30)) as pilot:
        app.menu_selected("api")
        await pilot.pause()
        app.screen.query_one(Input).value = query
        await pilot.pause()
        await pilot.press("enter")
        assert settings.api_model == model
        assert app.engine.selected_model == model
        assert app.state.active_provider == "api"
        assert app.state.gpu_status == GPUStatus.RUNNING and app.state.model_ready
        app.show_conversation()
        assert name in app.query_one("#log").border_title
        display = app.query_one("#dashboard", Static).render()
        assert any(
            name in text and style.foreground.hex == "#39FF14"
            for text, style in display.render()
        )
        app.state.api_health = "ok"
        app.engine.thread_id = "prior-conversation"
        app.select_api_model("cyberkimi")
        assert settings.api_model == "lordx64/cyberkimi"
        assert app.engine.thread_id is None
        assert app.state.api_health == "unverified"
        assert "CyberKimi" in app.query_one("#log").border_title


@pytest.mark.parametrize("after_turn", [False, True])
@pytest.mark.parametrize("preview", ["Hello from CyberGLM.", "Hello from"])
async def test_delayed_response_preview_is_not_printed_twice(
    settings, monkeypatch, after_turn, preview,
):
    app = VelocicyberApp(settings)
    app.state.collection = app.state.upload_approved = False
    answer = "Hello from CyberGLM."
    previews = []

    async def run(prompt, callback):
        callback({"type": "item.completed", "item": {
            "type": "agent_message", "text": answer,
        }})
        if after_turn:
            callback({"type": "turn.completed"})
        # The proxy posts previews from another thread, independently of engine stdout.
        asyncio.get_running_loop().call_soon(callback, {
            "type": "provider.progress", "stage": "Writing response", "text": preview,
        })
        await asyncio.sleep(0)
        previews.append(app.live_text)
        if not after_turn:
            callback({"type": "turn.completed"})

    monkeypatch.setattr(app.engine, "run", run)
    async with app.run_test(size=(100, 30)):
        await app.run_prompt("Hello.").wait()
        transcript = app.query_one("#log")._copy_text
        assert previews == [""]
        assert transcript.count(answer) == 1
        assert "Partial response:" not in transcript
        assert not app.query_one("#live-response").display
        assert not app.task_failed and "Task complete" in app.activity


@pytest.mark.parametrize("earlier_answer", [False, True])
async def test_interrupted_response_keeps_unfinished_text(settings, monkeypatch, earlier_answer):
    app = VelocicyberApp(settings)
    app.state.collection = app.state.upload_approved = False

    async def run(prompt, callback):
        if earlier_answer:
            callback({"type": "item.completed", "item": {
                "type": "agent_message", "text": "Checking the project.",
            }})
        callback({"type": "provider.progress", "stage": "Writing response",
                  "text": "Checking the project. Found an unfinished"})
        raise RuntimeError("Fixture connection interrupted")

    monkeypatch.setattr(app.engine, "run", run)
    async with app.run_test(size=(100, 30)):
        await app.run_prompt("Check the project.").wait()
        transcript = app.query_one("#log")._copy_text
        assert "Partial response:\nChecking the project. Found an unfinished" in transcript
        assert transcript.count("Found an unfinished") == 1
        assert app.task_failed and not app.query_one("#live-response").display


async def test_response_preview_completion_resets_for_followup(settings, monkeypatch):
    app = VelocicyberApp(settings)
    app.state.collection = app.state.upload_approved = False
    calls = []
    answer = "Hello again."

    async def run(prompt, callback):
        calls.append(prompt)
        callback({"type": "provider.progress", "stage": "Writing response", "text": answer})
        assert app.live_text == answer
        if len(calls) == 1:
            callback({"type": "item.completed", "item": {
                "type": "agent_message", "text": answer,
            }})
            callback({"type": "turn.completed"})
        else:
            raise RuntimeError("Fixture connection interrupted")

    monkeypatch.setattr(app.engine, "run", run)
    async with app.run_test(size=(100, 30)):
        await app.run_prompt("Hello.").wait()
        await app.run_prompt("Again.").wait()
        transcript = app.query_one("#log")._copy_text
        assert transcript.count(answer) == 2
        assert "Partial response:\nHello again." in transcript
        assert app.task_failed


@pytest.mark.parametrize("components", [{"engine"}, {"harness"}, {"engine", "harness"}])
async def test_update_menu_confirms_exact_available_components(settings, monkeypatch, components):
    from velocicyber.updates import UpdateChecker

    async def no_check(self):
        pass

    monkeypatch.setattr(UpdateChecker, "check", no_check)
    app = VelocicyberApp(settings)
    plan = {key: {"target": "b" * 40 if key == "harness" else "rust-v0.0.59"}
            for key in components}
    monkeypatch.setattr(app.updates, "plan", lambda: plan)
    exits = []
    monkeypatch.setattr(app, "action_quit", lambda: exits.append(True))
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        app.command("update")
        await pilot.pause()
        await choose(pilot, 1)
        assert not exits and not app.maintenance_action
        await choose(pilot, 0)
        assert not exits
        app.command("update")
        await pilot.pause()
        await choose(pilot, 1)
        await choose(pilot, 1)
        assert exits == [True]
        assert app.maintenance_action == "components" and app.maintenance_plan == plan


async def test_automatic_source_update_menu_requires_one_opt_in_and_can_disable(settings, monkeypatch):
    from velocicyber.updates import UpdateChecker
    from velocicyber.updates.startup import automatic_updates

    async def no_check(self):
        pass

    monkeypatch.setattr(UpdateChecker, "check", no_check)
    app = VelocicyberApp(settings)
    monkeypatch.setattr(app, "action_quit", lambda: pytest.fail("Preference must not end the session"))
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        app.command("update")
        await pilot.pause()
        await choose(pilot, 3)
        assert not automatic_updates(settings)
        await choose(pilot, 0)
        assert not (settings.root / "Modules.txt").exists()
        app.command("update")
        await pilot.pause()
        await choose(pilot, 3)
        await choose(pilot, 1)
        assert automatic_updates(settings)
        app.command("update")
        await pilot.pause()
        await choose(pilot, 3)
        assert not automatic_updates(settings)
        assert not app.maintenance_action


async def test_setup_is_filterable_and_requires_confirmation_before_shutdown(settings, monkeypatch):
    from velocicyber.updates import UpdateChecker

    async def no_check(self):
        pass

    monkeypatch.setattr(UpdateChecker, "check", no_check)
    app = VelocicyberApp(settings)
    shutdown = []
    monkeypatch.setattr(app, "terminate_during_startup", lambda: shutdown.append("terminate"))
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.press("backslash", *"setup", "enter")
        assert app.screen.id == "setup-confirm"
        assert "terminates its GPU" in app.screen.heading
        assert "hidden terminal prompts" in app.screen.heading
        assert not app.maintenance_action and not shutdown
        await choose(pilot, 0)
        assert not app.maintenance_action and not shutdown
        app.state.instance_id = "fixture-running-instance"
        app.command("setup")
        await pilot.pause()
        await choose(pilot, 1)
        assert app.maintenance_action == "setup"
        assert app.exit_after_termination and shutdown == ["terminate"]
        assert not (settings.root / ".env").exists()


@pytest.mark.parametrize("blocked", ["demo", "busy", "gpu_busy", "pi_busy", "data_busy", "sentinel_busy", "session_ending"])
async def test_setup_cannot_start_during_operations_or_demo(settings, monkeypatch, blocked):
    app = VelocicyberApp(settings, demo=True)
    monkeypatch.setattr(app, "action_quit", lambda: pytest.fail("Unexpected setup shutdown"))
    async with app.run_test() as pilot:
        app.demo = blocked == "demo"
        setattr(app, blocked, True)
        app.setup_menu()
        await pilot.pause()
        assert not isinstance(app.screen, Menu)
        # Check again at confirmation time to cover a task starting after opening the menu.
        app.begin_setup()
        assert not app.maintenance_action
        setattr(app, blocked, False)


async def test_setup_never_echoes_credentials_passed_as_command_arguments(settings):
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test():
        app.command("setup accidental-private-token")
        transcript = app.query_one("#log")._copy_text
        assert "hidden prompts" in transcript and "accidental-private-token" not in transcript
        assert not isinstance(app.screen, Menu)
