import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from velocicyber.engine.capture import CaptureProxy, ReplyProgress, parse_reply
from velocicyber.engine.tool_schemas import repair_tool_schemas
from velocicyber.models.pi_profiles import BUGTRACE_PI_REPO


def test_local_provider_uses_fresh_connections_without_replaying_requests():
    connections = []

    class Provider(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            connections.append(self.client_address)
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "OK"},
                                            "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with CaptureProxy(f"http://127.0.0.1:{server.server_port}/v1", "fixture") as proxy:
            for _ in range(2):
                response = httpx.post(
                    proxy.url + "/chat/completions", json={"messages": []}, trust_env=False,
                    headers={"Authorization": "Bearer " + proxy.local_key},
                )
                assert response.json()["choices"][0]["message"]["content"] == "OK"
            assert proxy.response_error() is None
        assert len(connections) == 2 and connections[0] != connections[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_provider_tool_call_observation_captures_reads_missing_from_native_events():
    events = []
    call = {"id": "read-fixture", "type": "function", "function": {
        "name": "read", "arguments": '{"path":"/outside/fixture.txt"}',
    }}
    body = json.dumps({"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "", "tool_calls": [call],
    }}]}).encode()
    with CaptureProxy("https://provider.example/v1", "fixture", on_progress=events.append,
                      transport=httpx.MockTransport(lambda _: httpx.Response(
                          200, content=body, headers={"content-type": "application/json"},
                      ))) as proxy:
        response = httpx.post(proxy.url + "/chat/completions", json={"messages": []},
                              headers={"Authorization": "Bearer " + proxy.local_key})
        assert response.content == body
        assert [e for e in events if e["type"] == "provider.tool_call"] == [
            {"type": "provider.tool_call", "tool_call": call},
        ]


def placeholder_tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Open Interpreter harness compatibility alias for {name}.",
            "parameters": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "input": {"type": "string", "description": "Harness-native tool input."}
                },
            },
        },
    }


def test_alias_repair_keeps_actual_forwarded_contract_and_original_request():
    request = {"messages": [], "stream": True, "tools": [placeholder_tool("write")]}
    original = json.dumps(request, sort_keys=True)
    seen = []

    def handler(r):
        seen.append(json.loads(r.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
                ]
            },
        )

    with CaptureProxy(
        "https://provider.example/v1",
        "secret",
        request_usage=True,
        transport=httpx.MockTransport(handler),
    ) as proxy:
        httpx.post(
            proxy.url + "/chat/completions",
            json=request,
            headers={"Authorization": "Bearer " + proxy.local_key},
        )
        exchange = proxy.events[0]
        assert exchange["engine_request"] == request
        assert exchange["request"] == seen[0]
        assert seen[0]["stream_options"]["include_usage"] is True
        parameters = seen[0]["tools"][0]["function"]["parameters"]
        assert parameters["required"] == ["path", "content"]
        assert "input" not in parameters["properties"]
        assert proxy.transcript()[1] == seen[0]["tools"]
    assert json.dumps(request, sort_keys=True) == original


def test_real_custom_and_future_engine_schemas_are_not_overwritten():
    custom = placeholder_tool("write")
    custom["function"]["description"] = "User-defined tool taking input."
    future = placeholder_tool("Write")
    future["function"]["parameters"]["properties"] = {"new_contract": {"type": "string"}}
    request = {"tools": [custom, future, placeholder_tool("UnknownAlias")]}
    assert repair_tool_schemas(request) == request


def test_question_alias_reuses_the_real_question_contract():
    question = {
        "type": "function",
        "function": {
            "name": "request_user_input",
            "description": "Ask only when clarification is needed.",
            "parameters": {"type": "object", "properties": {"questions": {"type": "array"}}},
        },
    }
    result = repair_tool_schemas({"tools": [placeholder_tool("AskUserQuestion"), question]})
    alias = result["tools"][0]["function"]
    assert alias["name"] == "AskUserQuestion"
    assert alias["parameters"] == question["function"]["parameters"]
    assert alias["description"] == question["function"]["description"]


def test_bugtrace_uses_single_canonical_tools_without_changing_history_or_other_models():
    tools = [placeholder_tool(n) for n in ("Read", "read", "Write", "write", "Bash", "bash")]
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Run a shell command.",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            },
        }
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "old", "function": {"name": "Bash", "arguments": '{"command":"pwd"}'}}
            ],
        }
    ]
    request = {
        "model": BUGTRACE_PI_REPO,
        "tools": tools,
        "messages": messages,
        "parallel_tool_calls": True,
    }
    repaired = repair_tool_schemas(request)
    assert [t["function"]["name"] for t in repaired["tools"]] == ["read", "write", "exec_command"]
    assert repaired["messages"] == messages
    assert repaired["parallel_tool_calls"] is False
    assert len(request["tools"]) == 7 and request["parallel_tool_calls"] is True
    request["model"] = "cyberkimi"
    other = repair_tool_schemas(request)
    assert len(other["tools"]) == 7 and other["parallel_tool_calls"] is True
    # Third-party tools sharing a familiar name must not be silently removed.
    request["model"] = BUGTRACE_PI_REPO
    tools[4]["function"]["description"] = "Custom Bash tool"
    repaired = repair_tool_schemas(request)
    assert "Bash" in [t["function"]["name"] for t in repaired["tools"]]


def test_tool_argument_errors_reported_once_without_leaking_other_outputs():
    events = []
    proxy = CaptureProxy("https://provider.example/v1", "secret", on_progress=events.append)
    try:
        request = {
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "write"}}]},
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "failed to parse function arguments: missing field `path` at line 1 column 26",
                },
                {"role": "tool", "tool_call_id": "c2", "content": "private file contents"},
            ]
        }
        proxy.report_tool_errors(request)
        proxy.report_tool_errors(request)
        assert len(events) == 1
        assert events[0]["type"] == "tool.error" and events[0]["tool"] == "write"
        assert "missing field `path`" in events[0]["message"]
        assert "private file contents" not in json.dumps(events)
    finally:
        proxy.client.close()


def test_resuming_does_not_redisplay_historical_tool_errors():
    events = []
    proxy = CaptureProxy("https://provider.example/v1", "secret", on_progress=events.append)
    error = "failed to parse function arguments: missing field `path`"
    request = {"messages": [{"role": "tool", "tool_call_id": "old", "content": error}]}
    try:
        proxy.report_tool_errors(request, historical=True)
        proxy.report_tool_errors(request)
        assert events == []
        request["messages"].append({"role": "tool", "tool_call_id": "new", "content": error})
        proxy.report_tool_errors(request)
        assert len(events) == 1
    finally:
        proxy.client.close()


def test_tool_progress_names_counts_and_elapsed_without_argument_contents(monkeypatch):
    events = []
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    progress = ReplyProgress(events.append)
    for index, name, arguments in [(0, "wr", '{"path":'), (1, "read", "{}")]:
        packet = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": index, "function": {"name": name, "arguments": arguments}}
                        ]
                    }
                }
            ]
        }
        progress.feed(b"data: " + json.dumps(packet).encode() + b"\n\n")
    monkeypatch.setattr(time, "monotonic", lambda: 15.0)
    packet = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"name": "ite", "arguments": '"secret.txt"}'}}
                    ]
                }
            }
        ]
    }
    body = b"data: " + json.dumps(packet).encode() + b"\n\n"
    for byte in body:
        progress.feed(bytes([byte]))
    progress.emit()
    assert events[-1]["stage"] == "Preparing write, read · 23 argument chars · 5s"
    assert "secret.txt" not in json.dumps(events)


def test_fragmented_tool_arguments_roundtrip():
    packets = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc1",
                                "function": {"name": "shell", "arguments": '{"cmd":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"pwd"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    body = b"".join(b"data: " + json.dumps(p).encode() + b"\n\n" for p in packets)
    reply = parse_reply(body, "text/event-stream")
    assert reply["tool_calls"] == [
        {
            "id": "tc1",
            "type": "function",
            "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
        }
    ]
    with pytest.raises(ValueError, match="Incomplete"):
        parse_reply(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n', "text/event-stream")


def test_proxy_preserves_bytes_masks_auth_and_reports_payment():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(402, json={"error": "wallet empty"})

    with CaptureProxy(
        "https://provider.example/v1", "never-log-this", transport=httpx.MockTransport(handler)
    ) as proxy:
        response = httpx.post(
            proxy.url + "/chat/completions",
            json={"model": "test", "messages": []},
            headers={"Authorization": "Bearer " + proxy.local_key},
        )
        assert response.status_code == 402
        assert proxy.health == "payment_required"
        assert "never-log-this" not in json.dumps(proxy.events)
        assert seen[0].headers["authorization"] == "Bearer never-log-this"


@pytest.mark.parametrize(
    "status,body,expected,health",
    [
        (
            503,
            b'{"error":{"type":"model_maintenance","message":"private-request-data"}}',
            "CyberKimi is unavailable during provider maintenance",
            "maintenance",
        ),
        (
            502,
            b"<html>502 Bad Gateway private-request-data</html>",
            "HTTP 502 (Bad Gateway)",
            "temporarily_unavailable",
        ),
        (503, b"[]", "HTTP 503 (Service Unavailable)", "temporarily_unavailable"),
        (504, b"null", "HTTP 504 (Gateway Timeout)", "temporarily_unavailable"),
        (401, b'{"error":{"type":"model_maintenance"}}', "HTTP 401", "authentication_error"),
    ],
)
def test_provider_maintenance_and_gateway_errors_are_clear_without_retries(
    status, body, expected, health
):
    requests = []

    def handler(request):
        requests.append(request)
        # CyberKimi's maintenance JSON is returned as text/plain, not application/json.
        return httpx.Response(status, content=body, headers={"content-type": "text/plain"})

    with CaptureProxy(
        "https://provider.example/v1", "secret", transport=httpx.MockTransport(handler)
    ) as proxy:
        response = httpx.post(
            proxy.url + "/chat/completions",
            json={"model": "lordx64/cyberkimi", "messages": []},
            headers={"Authorization": "Bearer " + proxy.local_key},
        )
        assert response.status_code == status and response.content == body
        message = proxy.response_error()
        assert expected in message and "no automatic retry" in message
        assert "private-request-data" not in message
        assert proxy.health == health and len(requests) == 1
        assert proxy.transcript() == ([], [], False)
        assert bytes.fromhex("".join(proxy.events[0]["response_chunks"])) == body


def test_null_tool_calls_and_final_delta_from_live_provider():
    packets = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Done.", "tool_calls": None},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": None, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"total_tokens": 10}},
    ]
    body = b"".join(b"data: " + json.dumps(p).encode() + b"\n\n" for p in packets)
    assert parse_reply(body, "text/event-stream") == {"role": "assistant", "content": "Done."}


def test_live_preview_handles_fragmented_utf8_without_showing_reasoning():
    events = []
    progress = ReplyProgress(events.append)
    packets = [
        {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]},
        {"choices": [{"delta": {"content": "Hello π"}}]},
        {"choices": [{"delta": None, "finish_reason": "stop"}]},
    ]
    body = b"".join(
        b"data: " + json.dumps(p, ensure_ascii=False).encode() + b"\n\n" for p in packets
    )
    for byte in body:
        progress.feed(bytes([byte]))
    progress.emit()
    assert events[-1]["text"] == "Hello π"
    assert "private reasoning" not in json.dumps(events)


def test_proxy_previews_before_completion_and_preserves_exact_bytes():
    events = []
    first = b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
    last = b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'

    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            time.sleep(0.12)
            yield first
            assert any(e["text"] == "Hello " for e in events)
            yield last

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Chunks()
        )
    )
    with CaptureProxy(
        "https://provider.example/v1", "secret", transport=transport, on_progress=events.append
    ) as proxy:
        response = httpx.post(
            proxy.url + "/chat/completions",
            json={"messages": []},
            headers={"Authorization": "Bearer " + proxy.local_key},
        )
        assert response.content == first + last
        assert bytes.fromhex("".join(proxy.events[0]["response_chunks"])) == first + last
        assert events[-1]["text"] == "Hello world"


def test_provider_timeout_is_reported_even_after_http_200():
    first = b'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n\n'
    calls = []

    class TimeoutStream(httpx.SyncByteStream):
        def __iter__(self):
            yield first
            raise httpx.ReadTimeout("fixture")

    def handler(request):
        calls.append(request)
        assert request.extensions["timeout"]["read"] == 600
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=TimeoutStream()
        )

    with CaptureProxy(
        "https://provider.example/v1", "secret", transport=httpx.MockTransport(handler)
    ) as proxy:
        response = httpx.post(
            proxy.url + "/chat/completions",
            json={"messages": []},
            headers={"Authorization": "Bearer " + proxy.local_key},
        )
        assert response.status_code == 200 and response.content == first
        assert "600 seconds" in proxy.response_error()
        assert proxy.health == "connection_error"
        assert not proxy.transcript()[2]
        assert len(calls) == 1
        assert bytes.fromhex("".join(proxy.events[-1]["response_chunks"])) == first


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_invalid_response_timeout(settings, value):
    settings.options["MODEL_RESPONSE_TIMEOUT_SECONDS"] = value
    with pytest.raises(ValueError, match="positive integer"):
        settings.provider_read_timeout()


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_compaction_notices_and_usage_capture_preserve_provider_bytes(finish):
    from velocicyber.engine.context import COMPACTION_PROMPT

    events, requests = [], []
    packet = {
        "choices": [{"index": 0, "delta": {"content": "Checkpoint."}, "finish_reason": finish}]
    }
    body = b"data: " + json.dumps(packet).encode() + b"\n\ndata: [DONE]\n\n"

    def handler(request):
        assert events == [{"type": "context.compaction.started"}]
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    with CaptureProxy(
        "https://provider.example/v1",
        "secret",
        transport=httpx.MockTransport(handler),
        on_progress=events.append,
        request_usage=True,
    ) as proxy:
        original = {
            "stream": True,
            "stream_options": {"include_usage": False},
            "messages": [
                {"role": "user", "content": COMPACTION_PROMPT},
            ],
        }
        response = httpx.post(
            proxy.url + "/chat/completions",
            json=original,
            headers={"Authorization": "Bearer " + proxy.local_key},
        )
        assert response.content == body
        assert proxy.events[0]["engine_request"] == original
        assert proxy.events[0]["request"] == requests[0]
        assert requests[0]["stream_options"]["include_usage"] is True
        assert bytes.fromhex("".join(proxy.events[0]["response_chunks"])) == body
        assert events == [
            {"type": "context.compaction.started"},
            {
                "type": "context.compaction.completed"
                if finish == "stop"
                else "context.compaction.failed"
            },
        ]
        if finish == "length":
            assert "token limit" in proxy.response_error()
            assert not proxy.transcript()[2]


def test_context_overflow_is_reported_as_a_context_limit():
    error = {
        "error": {"type": "exceed_context_size_error", "n_prompt_tokens": 32778, "n_ctx": 32768}
    }
    with CaptureProxy(
        "https://fixture/v1",
        "secret",
        transport=httpx.MockTransport(lambda request: httpx.Response(400, json=error)),
    ) as proxy:
        httpx.post(
            proxy.url + "/chat/completions",
            json={"messages": []},
            headers={"Authorization": "Bearer " + proxy.local_key},
        )
        assert "32,778 requested; 32,768 available" in proxy.response_error()
        assert proxy.health == "context_limit"


def test_stream_receipt_error_after_finish_is_not_a_success():
    packet = {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}
    stream = ("data: " + json.dumps(packet) + '\n\ndata: {"error":{"message":"receipt failed"}}\n\n').encode()
    with pytest.raises(ValueError, match="unverified"):
        parse_reply(stream, "text/event-stream")


@pytest.mark.parametrize("explicit", [None, 512])
def test_output_allowance_is_forwarded_and_original_request_preserved(explicit):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}]})

    original = {"model": "lordx64/cyberglm", "messages": []}
    if explicit is not None:
        original["max_tokens"] = explicit
    with CaptureProxy("https://fixture/v1", "secret", default_max_tokens=16384,
                      transport=httpx.MockTransport(handler)) as proxy:
        httpx.post(proxy.url + "/chat/completions", json=original,
                   headers={"Authorization": "Bearer " + proxy.local_key})
        assert requests[0]["max_tokens"] == (explicit or 16384)
        assert proxy.events[0]["request"] == requests[0]
        if explicit is None:
            assert proxy.events[0]["engine_request"] == original
        assert proxy.response_error() is None


@pytest.mark.parametrize("resumed", [False, True])
def test_truncated_tool_history_stops_before_provider_dispatch(resumed):
    calls = []
    call = {"id": "broken", "type": "function", "function": {
        "name": "exec_command", "arguments": '{"cmd":"unfinished'}}
    packet = {"choices": [{"delta": {"tool_calls": [{"index": 0, **call}]},
                           "finish_reason": "length"}]}
    body = b"data: " + json.dumps(packet).encode() + b"\n\ndata: [DONE]\n\n"

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    with CaptureProxy("https://fixture/v1", "secret", reject_malformed_tool_history=True,
                      transport=httpx.MockTransport(handler)) as proxy:
        headers = {"Authorization": "Bearer " + proxy.local_key}
        if not resumed:
            first = httpx.post(proxy.url + "/chat/completions", json={"messages": []}, headers=headers)
            assert first.content == body
        history = {"messages": [{"role": "assistant", "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "broken", "content": "failed to parse function arguments"}]}
        response = httpx.post(proxy.url + "/chat/completions", json=history, headers=headers)
        assert response.status_code == 400 and len(calls) == (0 if resumed else 1)
        assert "No request was sent" in proxy.response_error()
        if not resumed:
            assert "output token limit" in proxy.response_error()
        assert proxy.events[-1]["rejected_before_dispatch"]
        assert proxy.events[-1]["request"] == history
        assert not proxy.transcript()[2]


def test_encrypted_gateway_error_is_readable_without_changing_raw_capture():
    body = json.dumps({"error": {"type": "confidential_gateway_error",
        "message": "POST /chat/completions failed with HTTP 400: \u0000\u001b\ufffdciphertext"}}).encode()
    with CaptureProxy("https://fixture/v1", "secret", transport=httpx.MockTransport(
            lambda _: httpx.Response(400, content=body))) as proxy:
        httpx.post(proxy.url + "/chat/completions", json={"messages": []},
                   headers={"Authorization": "Bearer " + proxy.local_key})
        assert "HTTP 400" in proxy.response_error()
        assert "unreadable encrypted error" in proxy.response_error()
        assert "ciphertext" not in proxy.response_error()
        assert bytes.fromhex("".join(proxy.events[-1]["response_chunks"])) == body


@pytest.mark.parametrize("alias,key", [
    ("cyberglm", "CYBERGLM_MAX_OUTPUT_TOKENS"),
    ("cyberseek", "CYBERSEEK_MAX_OUTPUT_TOKENS"),
    ("adverserial/cyberseek", "CYBERSEEK_MAX_OUTPUT_TOKENS"),
])
def test_api_output_budget(settings, alias, key):
    assert settings.api_output_limit() is None
    settings.api_model = alias
    assert settings.api_output_limit() == 16384
    settings.options[key] = "8192"
    assert settings.api_output_limit() == 8192
    for value in ("0", "65537", "not-a-number"):
        settings.options[key] = value
        with pytest.raises(ValueError, match="1 to 65536"):
            settings.api_output_limit()
