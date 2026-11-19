"""Runs the real pinned ARM64 Open Interpreter against a local fake model only."""

import asyncio
import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from velocicyber.datasets.pipeline import Curator, RawStore, canonical
from velocicyber.engine.runner import Engine
from velocicyber.state import State


@pytest.mark.parametrize(
    "mode", ["success", "verified", "verifier_failed", "command_failed", "length"]
)
async def test_empty_final_reply_after_actions_is_classified_without_retry(
    settings, native_engine, mode
):
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if len(requests) == 1:
                cmd = "exit 1" if mode == "command_failed" else "rm test1.txt test2.txt"
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "delete_files",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": json.dumps({"cmd": cmd}),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                delta = {"role": "assistant", "content": "" if len(requests) == 2 else "Ready."}
                finish = "length" if mode == "length" else "stop"
            packet = {
                "id": "fixture",
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
    state = State(collection=True, upload_approved=False)
    engine = Engine(settings, state)
    for name in ("test1.txt", "test2.txt", "keep.txt"):
        (engine.workspace / name).write_text("fixture")
    if mode in {"verified", "verifier_failed"}:
        engine.verifier = [
            sys.executable,
            "-c",
            "from pathlib import Path; assert "
            + (
                "False"
                if mode == "verifier_failed"
                else "not Path('test1.txt').exists() and not Path('test2.txt').exists()"
            ),
        ]
    events = []
    try:
        if mode in {"success", "verified"}:
            await asyncio.wait_for(engine.run("Delete test1.txt and test2.txt.", events.append), 45)
            expected = "verified_no_summary" if mode == "verified" else "actions_completed"
            assert engine.session.metadata["status"] == expected
            assert state.api_health == "ok"
            notices = [e for e in events if e["type"] == "engine.notice"]
            assert len(notices) == 1 and notices[0]["completion_status"] == expected
            assert not any(e["type"] == "engine.error" for e in events)
            assert not any(e.get("item", {}).get("text", "").strip() for e in events)
        else:
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(
                    engine.run("Delete test1.txt and test2.txt.", events.append), 45
                )
            assert engine.session.metadata["status"] == "failed"
            assert not any(e["type"] == "engine.notice" for e in events)
        assert len(requests) == 2  # No extra summary request and no repeated deletion.
        assert (engine.workspace / "test1.txt").exists() == (mode == "command_failed")
        assert (engine.workspace / "test2.txt").exists() == (mode == "command_failed")
        assert (engine.workspace / "keep.txt").read_text() == "fixture"
        episode = next(RawStore(settings.data).episodes())
        # Truncated responses end at the real tool output; ordinary empty replies
        # retain the empty assistant message. Neither path invents a summary.
        assert not any(m.get("content") for m in episode["messages"] if m["role"] == "assistant")
        if mode == "length":
            assert episode["messages"][-1]["role"] == "tool"
            assert not episode["context_complete"]
        assert episode["outcome"]["success"] == (
            True if mode == "verified" else False if mode == "verifier_failed" else None
        )
        prepared = next(e for e in events if e["type"] == "dataset.prepared")
        assert prepared["eligible"] == 0 and prepared["exclusion_reasons"] == {
            "no_complete_answer": 1
        }
        if mode == "success":
            saved_id = engine.session.metadata["id"]
            engine.close()
            engine = Engine(settings, state)
            engine.resume_session(saved_id)
            assert any(e["type"] == "engine.notice" for e in engine.session.events())
            await asyncio.wait_for(engine.run("Continue.", events.append), 45)
            assert engine.session.metadata["status"] == "complete"
            assert engine.session.metadata["completion_status"] is None
    finally:
        engine.close()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("capitalized", [False, True])
async def test_native_alias_contracts_execute_and_capture_recovery(
    settings, native_engine, capitalized
):
    """Exercise real handlers, including the exact malformed call from BugTrace's stalled turn."""
    requests = []
    calls = [
        ("write", {"input": "test_test3.txt"}),  # Original error must remain a real error.
        ("write", {"path": "test_test3.txt", "content": "before"}),
        ("read", {"path": "test_test3.txt"}),
        ("edit", {"path": "test_test3.txt", "old_string": "before", "new_string": "after"}),
        ("glob", {"pattern": "*.txt"}),
        ("grep", {"pattern": "after"}),
        ("bash", {"command": "cat test_test3.txt"}),
    ]

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            index = len(requests) - 1
            if index < len(calls):
                name, arguments = calls[index]
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": name.capitalize() if capitalized else name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                delta, finish = {"role": "assistant", "content": "Created and verified."}, "stop"
            packet = {
                "id": f"reply_{index}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
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
    try:
        await asyncio.wait_for(engine.run("Create test_test3.txt.", events.append), 45)
        assert (engine.workspace / "test_test3.txt").read_text() == "after"
        assert len(requests) == len(calls) + 1
        tools = {t["function"]["name"]: t["function"] for t in requests[0]["tools"]}
        for index, (name, arguments) in enumerate(calls[1:], 1):
            name = name.capitalize() if capitalized else name
            assert set(arguments) <= tools[name]["parameters"]["properties"].keys()
            reply = requests[index + 1]["messages"][-1]
            assert reply["role"] == "tool"
            assert "failed" not in str(reply).lower(), reply
        assert "before" in requests[3]["messages"][-1]["content"]
        assert "test_test3.txt" in requests[5]["messages"][-1]["content"]
        assert "test_test3.txt" in requests[6]["messages"][-1]["content"]
        assert "after" in requests[7]["messages"][-1]["content"]
        errors = [e for e in events if e["type"] == "tool.error"]
        assert len(errors) == 1 and "missing field `path`" in errors[0]["message"]
        assert any(e["type"] == "turn.completed" for e in events)
        episode = next(RawStore(settings.data).episodes())
        assert episode["tools"] == requests[-1]["tools"]
        exchange = episode["provider_exchanges"][0]
        original = exchange.get("engine_request", exchange["request"])
        for tool in original["tools"]:
            function = tool["function"]
            if function["name"].lower() == "write":
                if set(function["parameters"]["properties"]) == {"input"}:
                    assert "engine_request" in exchange
                    assert tools[function["name"]]["parameters"]["required"] == ["path", "content"]
                else:
                    # A future engine with a proper schema needs no compatibility rewrite.
                    assert tools[function["name"]] == function
        assert "missing field `path`" in json.dumps(episode["messages"])
    finally:
        engine.close()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "window,allow_compaction,collect,provider,reply_mode",
    [
        (32768, True, True, "local", "summary"),
        (65536, True, True, "local", "summary"),
        (32768, False, True, "local", "summary"),
        (32768, True, False, "local", "summary"),
        (16384, True, True, "pi", "summary"),
        (8192, True, True, "pi", "empty"),
        (8192, True, True, "pi", "auxiliary"),
        (8192, True, True, "pi", "failed_command"),
        (8192, True, True, "pi", "truncated"),
    ],
)
async def test_local_real_engine_compacts_automatically_and_continues(
    settings, window, allow_compaction, collect, provider, reply_mode, native_engine
):
    from velocicyber.config import PI_MODEL_REPO
    from velocicyber.engine.context import COMPACTION_PROMPT, is_compaction_request

    binary = native_engine
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            compacting = is_compaction_request(payload)
            if len(requests) == 1:
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "write_result",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": json.dumps(
                                    {
                                        "cmd": (
                                            "exit 1" if reply_mode == "failed_command"
                                            else "printf done > result.txt"
                                        ),
                                        "login": False,
                                    }
                                ),
                            },
                        }
                    ],
                }
                finish, tokens = "tool_calls", window * 3 // 4 + 100
            elif reply_mode == "auxiliary" and len(requests) == 3 and not compacting:
                delta = {"role": "assistant", "tool_calls": [{
                    "index": 0, "id": "aux", "type": "function",
                    "function": {"name": "get_goal", "arguments": "{}"},
                }]}
                finish, tokens = "tool_calls", 800
            else:
                text = (
                    "CHECKPOINT: User requested result.txt containing done. The file was written. "
                    "Remaining work: report completion. Preserve the user's constraints."
                    if compacting
                    else "Created result.txt containing done."
                )
                delta, finish, tokens = {"role": "assistant", "content": text}, "stop", 800
                if not compacting and reply_mode != "summary" and len(requests) <= (
                    4 if reply_mode == "auxiliary" else 3
                ):
                    delta["content"] = ""
                    finish = "length" if reply_mode == "truncated" else "stop"
            packet = {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": settings.local_model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            chunks = [packet]
            # Reproduce the real failure: usage is absent unless explicitly requested.
            if payload.get("stream_options", {}).get("include_usage"):
                chunks.append(
                    {
                        **packet,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": tokens,
                            "completion_tokens": 100,
                            "total_tokens": tokens + 100,
                        },
                    }
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings.options["INTERPRETER_BIN"] = str(binary)
    settings.options["PI_LOCAL_MODEL"] = PI_MODEL_REPO
    state = State(
        collection=collect,
        allow_compaction=allow_compaction,
        upload_approved=False,
        active_provider=provider,
        pi_ready=True,
        pi_url=f"http://127.0.0.1:{server.server_port}/v1",
        pi_context_window=window,
        model_ready=True,
        model_context_window=window,
        model_context_model=settings.local_model,
        local_url=f"http://127.0.0.1:{server.server_port}/v1",
    )
    engine = Engine(settings, state)
    events = []
    try:
        run = engine.run("Create result.txt containing done.", events.append)
        if reply_mode in {"failed_command", "truncated"}:
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(run, 45)
            assert engine.session.metadata["status"] == "failed"
        else:
            await asyncio.wait_for(run, 45)
            expected_status = {
                "summary": "complete", "empty": "actions_completed",
                "auxiliary": "response_incomplete",
            }[reply_mode]
            assert engine.session.metadata["status"] == expected_status
            assert not any(e["type"] == "engine.error" for e in events)
        if reply_mode != "failed_command":
            assert (engine.workspace / "result.txt").read_text() == "done"
        messages = [e.get("item", {}).get("message", e.get("message", "")) for e in events]
        assert not any("Defaulting to fallback metadata" in text for text in messages)
        assert not any("skills context budget" in text for text in messages)
        # Descriptions fit without disabling the skills themselves.
        first_prompt = json.dumps(requests[0]["messages"])
        assert "skill-creator" in first_prompt and "skill-installer" in first_prompt
        assert len(requests) == (4 if reply_mode == "auxiliary" else 3), [
            (is_compaction_request(r), len(str(r))) for r in requests
        ]
        assert requests[1]["messages"][-1]["content"] == COMPACTION_PROMPT
        assert "CHECKPOINT" in json.dumps(requests[2]["messages"])
        notices = [e["type"] for e in events if e["type"].startswith("context.compaction.")]
        assert notices == ["context.compaction.started", "context.compaction.completed"]
        assert state.model_ready
        assert engine.session.metadata["context_compacted"]
        assert engine.session.metadata["compaction_count"] == 1
        if not allow_compaction or not collect:
            assert not list(RawStore(settings.data).episodes())
            assert any(e["type"] == "dataset.skipped" for e in events) == collect
        else:
            episode = next(RawStore(settings.data).episodes())
            assert not episode["context_complete"]  # Raw rewrite flag remains truthful.
            assert episode["context_compacted"] and episode["compaction_count"] == 1
            assert episode["models"]["provider"] == provider
            assert episode["models"]["teacher"] == (
                PI_MODEL_REPO if provider == "pi" else settings.local_model
            )
            assert episode["provider_exchanges"][1]["compaction"]
            assert episode["provider_exchanges"][0]["request"]["stream_options"]["include_usage"]
            assert "stream_options" not in episode["provider_exchanges"][0]["engine_request"]
            prepared = next(e for e in events if e["type"] == "dataset.prepared")
            assert prepared["eligible"] == (1 if reply_mode == "summary" else 0)
            assert prepared["excluded"] == (0 if reply_mode == "summary" else 1)
            rows = [
                json.loads(line)
                for p in Path(prepared["path"]).rglob("*.jsonl")
                if p.name != "excluded.jsonl"
                for line in p.read_text().splitlines()
            ]
            if reply_mode == "summary":
                assert rows[0]["meta"]["context_compacted"]
                assert rows[0]["messages"] == engine.scrubber.clean(episode["messages"])
            else:
                assert not rows and episode["outcome"]["success"] is None
                assert episode["outcome"].get("response_error")
        if reply_mode in {"empty", "auxiliary"}:
            before = len(requests)
            saved_id = engine.session.metadata["id"]
            engine.close()
            engine = Engine(settings, state)
            engine.resume_session(saved_id)
            assert any(e.get("code") == "missing_final_summary" for e in engine.session.events())
            await asyncio.wait_for(engine.run("Summarize the result.", events.append), 30)
            assert len(requests) == before + 1  # Only a user follow-up requests another reply.
            assert engine.session.metadata["status"] == "complete"
        if window == 32768 and collect:
            # Reopen the durable session: a later turn may not itself compact.
            session_id = engine.session.metadata["id"]
            before = list(RawStore(settings.data).episodes())
            engine.close()
            state.allow_compaction = False
            engine = Engine(settings, state)
            engine.resume_session(session_id)
            events.clear()
            await asyncio.wait_for(engine.run("Summarize the result.", events.append), 30)
            assert len(requests) == 4
            assert not is_compaction_request(requests[-1])
            assert any(e["type"] == "dataset.skipped" for e in events)
            assert list(RawStore(settings.data).episodes()) == before
            assert engine.session.metadata["status"] == "complete"
    finally:
        engine.close()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("reply_kind", ["complete", "truncated", "empty"])
async def test_real_engine_provider_capture_and_opt_out(settings, reply_kind, native_engine):
    binary = native_engine
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            packet = {
                "id": "test-response",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "Local integration passed."
                            if reply_kind == "complete"
                            else "",
                        },
                        "finish_reason": None if reply_kind == "truncated" else "stop",
                    }
                ],
            }
            self.wfile.write(b"data: " + json.dumps(packet).encode() + b"\n\ndata: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings.api_url = f"http://127.0.0.1:{server.server_port}/v1"
    settings.options["INTERPRETER_BIN"] = str(binary)
    settings.api_model = "test-model"
    state = State(collection=False, upload_approved=False)
    engine = Engine(settings, state)
    from types import SimpleNamespace

    assert "model_providers.velocicyber.stream_idle_timeout_ms=615000" in engine.command(
        SimpleNamespace(url="http://127.0.0.1:1/v1"), settings.api_model
    )
    events = []
    try:
        if reply_kind != "complete":
            with pytest.raises(RuntimeError, match="Model"):
                await asyncio.wait_for(engine.run("Say hello", events.append), 45)
            assert not settings.data.exists()
            state.collection = True
            engine.verifier = [sys.executable, "-c", "raise SystemExit(0)"]
            with pytest.raises(RuntimeError, match="Model"):
                await asyncio.wait_for(engine.run("Say hello again", events.append), 45)
            episode = next(RawStore(settings.data).episodes())
            assert (
                episode["outcome"]["success"] is None
            )  # Even a passing verifier must not label a blank answer.
            assert episode["outcome"]["response_error"]
            assert not episode["context_complete"]
            assert any(e["type"] == "engine.error" for e in events)
            prepared = next(e for e in events if e["type"] == "dataset.prepared")
            assert prepared["eligible"] == 0
            assert prepared["exclusion_reasons"] == {"no_complete_answer": 1}
            assert len(requests) == 2  # No automatic retries.
            return
        assert await asyncio.wait_for(engine.run("Say hello", events.append), 45) is None
        assert not settings.data.exists()
        assert state.api_health == "ok"
        assert any(e["type"] == "turn.completed" for e in events)
        saved_id, thread_id = engine.session.metadata["id"], engine.thread_id
        saved_folder = engine.session.folder
        engine.close()
        assert (saved_folder / "session.json").exists()
        engine = Engine(settings, state)
        engine.resume_session(saved_id)
        assert engine.thread_id == thread_id
        state.collection = True
        path = await asyncio.wait_for(engine.run("Say hello again", events.append), 45)
        episode = json.loads(path.read_text())
        assert episode["messages"][-1] == {
            "role": "assistant",
            "content": "Local integration passed.",
        }
        assert episode["provider_exchanges"][-1]["request"]["messages"] == requests[-1]["messages"]
        assert episode["outcome"]["success"] is None  # Protocol completion is not verification.
        assert episode["system_prompt"]
        assert episode["context_complete"]
        assert len(requests) == 2
        assert "Say hello" in json.dumps(requests[-1]["messages"])
        assert "Local integration passed." in json.dumps(requests[-1]["messages"])
        assert len([e for e in engine.session.events() if e["type"] == "user.prompt"]) == 2

        # Surviving pre-feature rollouts can also be imported without old SQLite files.
        old_home = settings.runtime / "session-legacy-fixture"
        old_home.mkdir()
        shutil.copytree(engine.session.home / "sessions", old_home / "sessions")
        shutil.copy2(engine.session.home / "system-prompt.md", old_home / "system-prompt.md")
        engine.close()
        engine = Engine(settings, state)
        assert engine.sessions.import_legacy() == 1
        assert engine.sessions.import_legacy() == 0
        imported = next(row for row in engine.sessions.list() if row.get("legacy_source"))
        engine.resume_session(imported["id"])
        assert engine.thread_id == thread_id
        await asyncio.wait_for(engine.run("Continue the imported conversation", events.append), 45)
        assert "Say hello again" in json.dumps(requests[-1]["messages"])
        assert len(requests) == 3
    finally:
        engine.close()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "model",
    [
        "lordx64/cyberkimi",
        "cyberglm",
        "cyberseek",
        "adverserial/cyberseek",
        "example-org/coding-model",
        "terrorswift/BugTraceAI-Apex-G4-26B-APEX-GGUF",
    ],
)
async def test_real_engine_executes_and_recovers_before_verification(
    settings, model, native_engine
):
    binary = native_engine
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            count = len(requests)
            if count <= 2:
                command = (
                    "cat missing-file-for-recovery"
                    if count == 1
                    else "printf verified > result.txt"
                )
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call_{count}",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": json.dumps({"cmd": command, "login": False}),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                delta, finish = {"role": "assistant", "content": "Created result.txt."}, "stop"
            packet = {
                "id": f"reply_{count}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
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
    settings.options["INTERPRETER_BIN"] = str(binary)
    settings.api_model = model
    engine = Engine(settings, State(collection=True))
    (engine.workspace / "AGENTS.md").write_text("Preserve the CONTRACT_INSTRUCTIONS_MARKER rule.\n")
    engine.verifier = [
        sys.executable,
        "-c",
        'from pathlib import Path; assert Path("result.txt").read_text() == "verified"',
    ]
    try:
        events = []
        await asyncio.wait_for(
            engine.run("Create result.txt containing verified.", events.append), 45
        )
        assert (engine.workspace / "result.txt").read_text() == "verified", str(events)
        assert not any(
            "Defaulting to fallback metadata" in e.get("item", {}).get("message", "")
            for e in events
        )
        episode = next(RawStore(settings.data).episodes())
        assert episode["outcome"]["success"] is True
        assert episode["outcome"]["verifier"] == "command_exit"
        assert episode["context_complete"]
        assert len(episode["tool_events"]) == 4
        assert [e["type"] for e in episode["tool_events"]] == [
            "tool_call",
            "tool_output",
            "tool_call",
            "tool_output",
        ]
        assert "No such file" in str(episode["tool_events"][1])
        assert len(requests) == 3
        assert "CONTRACT_INSTRUCTIONS_MARKER" in json.dumps(requests[0]["messages"])
        # API aliases resolve to the canonical provider ID in inference and provenance.
        expected_model = settings.api_profile[1]
        assert all(request["model"] == expected_model for request in requests)
        assert episode["models"]["teacher"] == expected_model
        assert engine.session.metadata["model"] == expected_model
        assert "You are Velocicyber" in episode["system_prompt"]
        assert len(canonical([episode["messages"], episode["tools"]])) + 2048 < 32768
        artifact = Curator(settings.data, engine.scrubber).run()
        manifest = json.loads((artifact / "manifest.json").read_text())
        assert (
            sum(
                n
                for name, n in manifest["counts"].items()
                if name.endswith(("train.jsonl", "validation.jsonl"))
            )
            == 1
        )
    finally:
        engine.close()
        server.shutdown()
        server.server_close()
        thread.join()


async def test_resume_across_engine_upgrade_and_rollback(settings, native_engine):
    previous = Path(os.environ.get("VELOCICYBER_PREVIOUS_ENGINE", str(native_engine)))
    assert previous.is_file()
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            packet = {
                "id": "cross-version",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": settings.api_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": f"Saved reply number {len(requests)}.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: " + json.dumps(packet).encode() + b"\n\ndata: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings.api_url = f"http://127.0.0.1:{server.server_port}/v1"
    saved = None
    try:
        for binary, prompt in (
            (previous, "Remember PROJECT_MARKER_42."),
            (native_engine, "Continue after upgrading."),
            (previous, "Continue after rolling back."),
        ):
            settings.options["INTERPRETER_BIN"] = str(binary)
            engine = Engine(settings, State(collection=False, upload_approved=False))
            try:
                if saved:
                    engine.resume_session(saved)
                await asyncio.wait_for(engine.run(prompt), 45)
                saved = engine.session.metadata["id"]
                assert engine.session.metadata["status"] == "complete"
            finally:
                engine.close()
        assert len(requests) == 3
        assert "PROJECT_MARKER_42" in json.dumps(requests[-1]["messages"])
        assert "Saved reply number 1" in json.dumps(requests[-1]["messages"])
        assert "Saved reply number 2" in json.dumps(requests[-1]["messages"])
        assert not settings.data.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
