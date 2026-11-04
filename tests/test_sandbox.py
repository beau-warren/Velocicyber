"""Access settings and real engine boundaries, using only local test fixtures."""

import asyncio
import json
import shlex
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Input, OptionList, Static

from velocicyber.datasets.pipeline import RawStore
from velocicyber.engine.runner import Engine
from velocicyber.state import State
from velocicyber.ui.app import Menu, VelocicyberApp


def test_open_is_not_restored_from_state(settings):
    path = settings.runtime / "gpu.json"
    State(sandbox_open=True).persist(path)
    assert not State.restore(path).sandbox_open
    # Even a historical/manual preference must not grant permission on launch.
    path.write_text('{"sandbox_open": true}')
    assert not State.restore(path).sandbox_open


def test_external_workspace_and_resume_require_current_open_selection(settings):
    external = settings.root / "external"
    settings.root = settings.root / "harness"
    settings.root.mkdir()
    with pytest.raises(ValueError, match="--sandbox open"):
        Engine(settings, State(), external)
    assert not external.exists()
    engine = Engine(settings, State(sandbox_open=True), external)
    saved = engine.ensure_session("External project")
    thread_id = str(uuid.uuid4())
    (saved.home / "sessions").mkdir()
    (saved.home / "sessions" / f"rollout-test-{thread_id}.jsonl").write_text("{}\n")
    saved.save(thread_id=thread_id, sandbox_mode="danger-full-access", verifier=["check"])
    session_id = saved.metadata["id"]
    engine.close()

    engine = Engine(settings, State())
    try:
        with pytest.raises(ValueError, match="Select /sandbox > Open"):
            engine.resume_session(session_id)
        assert engine.session is None and not engine.state.sandbox_open
        engine.set_sandbox_open(True)
        engine.resume_session(session_id)
        assert engine.workspace == external
        assert engine.verifier == ["check"]
        assert engine.set_sandbox_open(False)  # Narrowing changes the workspace.
        assert engine.workspace == settings.root / "workspace"
        assert engine.session is None and not engine.thread_id and not engine.verifier
        assert saved.folder.exists()
        with pytest.raises(ValueError, match="Select /sandbox > Open"):
            engine.resume_session(session_id)
    finally:
        engine.close()


def test_access_change_preserves_context_and_frozen_prompt(settings, monkeypatch):
    monkeypatch.setattr(settings, "engine", lambda: Path("/test-engine"))
    engine = Engine(settings, State())
    try:
        session = engine.ensure_session()
        frozen = "Legacy prompt. Stay within the configured workspace\nand respect tool approval and sandbox rules.\n"
        (session.home / "system-prompt.md").write_text(frozen)
        engine.thread_id = str(uuid.uuid4())
        original_thread = engine.thread_id
        for enabled in (True, False):
            assert not engine.set_sandbox_open(enabled)
            command = engine.command(SimpleNamespace(url="http://fixture/v1"), settings.api_model)
            assert command[command.index("--sandbox") + 1] == (
                "danger-full-access" if enabled else "workspace-write"
            )
            assert engine.thread_id == original_thread and engine.session is session
            assert (session.home / "system-prompt.md").read_text() == frozen
            effective = (session.home / "execution-instructions.md").read_text()
            assert "Filesystem access: " + ("OPEN" if enabled else "WORKSPACE") in effective
            assert "Stay within the configured workspace" not in effective
        engine.running = True
        with pytest.raises(RuntimeError, match="Stop the current task"):
            engine.set_sandbox_open(True)
        assert not engine.state.sandbox_open
    finally:
        engine.close()


def test_closed_workspace_rejects_external_symlinks(settings):
    external = settings.root / "external"
    external.mkdir()
    settings.root = settings.root / "harness"
    settings.root.mkdir()
    (settings.root / "workspace").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        Engine(settings, State())
    engine = Engine(settings, State(sandbox_open=True))
    try:
        with pytest.raises(ValueError, match="must not link outside"):
            engine.set_sandbox_open(False)
        assert engine.state.sandbox_open  # Failed switch doesn't misreport actual access.
    finally:
        engine.close()


async def test_filterable_sandbox_menu_status_cancel_and_busy_guard(settings):
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(100, 30)) as pilot:
        assert "Access: WORKSPACE" in str(app.query_one("#dashboard", Static).render())
        app.query_one("#prompt", Input).value = "/sandbox"
        await pilot.pause()
        await pilot.press("enter")
        assert isinstance(app.screen, Menu)
        await pilot.press("escape")
        assert not app.state.sandbox_open
        app.command("sandbox")
        await pilot.pause()
        app.screen.query_one("#menu-filter", Input).value = "open"
        await pilot.pause()
        assert app.screen.query_one(OptionList).option_count == 1
        await pilot.press("enter")
        assert app.state.sandbox_open
        assert "Access: OPEN" in str(app.query_one("#dashboard", Static).render())
        app.busy = True
        app.set_sandbox(False)
        assert app.state.sandbox_open
        app.busy = False
        app.command("sandbox")
        await pilot.pause()
        await pilot.press("enter")  # Workspace is the first choice.
        assert not app.state.sandbox_open
        assert "Access: WORKSPACE" in str(app.query_one("#dashboard", Static).render())


async def test_real_engine_switches_filesystem_access_on_same_conversation(settings, native_engine):
    """A sibling marker stays within pytest's directory but outside the agent workspace."""
    requests = []
    targets = [settings.root / f"outside-{index}.txt" for index in range(3)]

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            index = len(requests) - 1
            if index % 2 == 0 and index < 6:
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"write_{index}",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": json.dumps(
                                    {
                                        "cmd": "printf marker > "
                                        + shlex.quote(str(targets[index // 2])),
                                        "login": False,
                                    }
                                ),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                delta = {"role": "assistant", "content": "The file access check has finished."}
                finish = "stop"
            packet = {
                "id": f"reply_{index}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": settings.api_model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: " + json.dumps(packet).encode() + b"\n\ndata: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings.api_url = f"http://127.0.0.1:{server.server_port}/v1"
    settings.options["INTERPRETER_BIN"] = str(native_engine)
    engine = Engine(settings, State(collection=True, upload_approved=False))
    events = []
    original_thread = None
    try:
        for index, enabled in enumerate((False, True, False)):
            engine.set_sandbox_open(enabled)
            await asyncio.wait_for(
                engine.run(f"Check write access to {targets[index]}", events.append), 45
            )
            assert len(requests) == 2 * (index + 1), str(events)
            assert targets[index].exists() == enabled, str(events)
            if enabled:
                assert targets[index].read_text() == "marker"
            original_thread = original_thread or engine.thread_id
            assert engine.thread_id == original_thread
            systems = [m for m in requests[index * 2]["messages"] if m["role"] == "system"]
            assert "Filesystem access: " + ("OPEN" if enabled else "WORKSPACE") in str(systems)
            assert engine.session.metadata["sandbox_mode"] == engine.sandbox_mode
        modes = [e["execution"]["sandbox_mode"] for e in RawStore(settings.data).episodes()]
        assert modes.count("workspace-write") == 2 and modes.count("danger-full-access") == 1
    finally:
        engine.close()
        server.shutdown()
        server.server_close()
        thread.join()
