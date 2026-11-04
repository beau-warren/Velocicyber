"""Check instruction discovery with the installed engine and a fake Pi model."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from velocicyber.config import PI_MODEL_REPO
from velocicyber.engine.runner import Engine
from velocicyber.state import State


@pytest.mark.parametrize(
    "workspace_kind,sandbox_open",
    [("managed", False), ("managed", True), ("project", False)],
)
async def test_greeting_uses_workspace_instructions_without_harness_build_brief(
    settings, native_engine, workspace_kind, sandbox_open
):
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            packet = {
                "id": "greeting-fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": PI_MODEL_REPO,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello! How can I help?"},
                    "finish_reason": "stop",
                }],
            }
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: " + json.dumps(packet).encode() + b"\n\ndata: [DONE]\n\n")

    # Reproduce the harness repository enclosing its normal user workspace.
    (settings.root / ".git").mkdir()
    (settings.root / ".git/HEAD").write_text("ref: refs/heads/main\n")
    (settings.root / "AGENTS.md").write_text(
        "HARNESS_BUILD_BRIEF: Build the Velocicyber coding harness.\n"
    )
    project = settings.root / "workspace"
    project.mkdir()
    if workspace_kind == "project":
        project = project / "user-project"
        project.mkdir()
        (project / ".git").mkdir()
        (project / ".git/HEAD").write_text("ref: refs/heads/main\n")
        workspace = project / "src"
        workspace.mkdir()
    else:
        workspace = project
    (project / "AGENTS.md").write_text("USER_PROJECT_RULE: Preserve existing files.\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings.options.update({"INTERPRETER_BIN": str(native_engine), "PI_LOCAL_MODEL": PI_MODEL_REPO})
    state = State(
        collection=False, upload_approved=False, active_provider="pi", pi_ready=True,
        pi_url=f"http://127.0.0.1:{server.server_port}/v1", sandbox_open=sandbox_open,
    )
    engine = Engine(settings, state, workspace)
    try:
        for prompt in ("hello!", "hello again!"):
            events = []
            await asyncio.wait_for(engine.run(prompt, events.append), 45)
            assert any(e["type"] == "turn.completed" for e in events)
            assert not any(
                e.get("item", {}).get("type") in {"command_execution", "file_change"}
                for e in events
            )
            messages = requests[-1]["messages"]
            text = json.dumps(messages)
            assert "HARNESS_BUILD_BRIEF" not in text
            assert "USER_PROJECT_RULE" in text
            assert "Configured coding model: " + PI_MODEL_REPO in text
            content = messages[-1]["content"]
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content)
            assert content == prompt
            # Reopen the saved native conversation for the second turn as /resume does.
            session_id = engine.session.metadata["id"]
            engine.close()
            engine = Engine(settings, state, workspace)
            engine.resume_session(session_id)
        assert len(requests) == 2
        assert not settings.data.exists()
    finally:
        engine.close()
        server.shutdown()
        server.server_close()
        thread.join()
