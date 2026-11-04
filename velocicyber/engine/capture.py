"""Loopback Chat Completions bridge: preserve the actual model/tool contract.

Open Interpreter's exec events are a UI summary. Capturing provider requests and
SSE responses also preserves exact prompts, tool schemas, retries and outputs.
Authorization headers are never part of the recorded events.
"""

import copy
import json
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import httpx

from velocicyber.config import API_MODELS
from velocicyber.engine.context import is_compaction_request
from velocicyber.engine.tool_schemas import repair_tool_schemas


class ResponseLengthError(ValueError):
    pass


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def malformed_tool_history(request):
    for message in request.get("messages", []):
        for call in message.get("tool_calls") or []:
            if call.get("type", "function") != "function":
                continue
            arguments = (call.get("function") or {}).get("arguments")
            try:
                json.loads(arguments)
            except (ValueError, TypeError):
                return True
    return False


def parse_reply(body: bytes, content_type: str) -> dict:
    if "text/event-stream" not in content_type:
        choice = json.loads(body)["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ResponseLengthError("Model response reached its token limit")
        if choice.get("finish_reason") not in ("stop", "tool_calls"):
            raise ValueError("Incomplete or length-limited provider response")
        return choice["message"]
    message = {"role": "assistant", "content": ""}
    calls = {}
    finished = False
    for line in body.decode("utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if value == "[DONE]":
            continue
        packet = json.loads(value)
        if packet.get("error"):
            raise ValueError("Provider stream ended with an error; response is unverified.")
        for choice in packet.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            if choice.get("finish_reason") == "length":
                raise ResponseLengthError("Model response reached its token limit")
            finished |= choice.get("finish_reason") in ("stop", "tool_calls")
            delta = choice.get("delta") or {}
            for field in ("content", "reasoning_content", "reasoning"):
                if delta.get(field):
                    message[field] = message.get(field, "") + delta[field]
            for call in delta.get("tool_calls") or []:
                target = calls.setdefault(
                    call["index"],
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call.get("id"):
                    target["id"] = call["id"]
                for field in ("name", "arguments"):
                    target["function"][field] += call.get("function", {}).get(field, "")
    if not finished:
        raise ValueError("Incomplete or length-limited provider response")
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return message


class ReplyProgress:
    """Observe SSE for display without changing the forwarded or recorded bytes."""

    def __init__(self, callback):
        self.callback = callback
        self.pending = b""
        self.text = ""
        self.stage = "Waiting for model"
        self.last_sent = 0.0
        self.tool_calls = {}
        self.tool_started = None

    def feed(self, chunk):
        self.pending += chunk
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            if not line.startswith(b"data:"):
                continue
            try:
                packet = json.loads(line[5:].strip())
                for choice in packet.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta") or {}
                    if isinstance(delta.get("content"), str) and delta["content"]:
                        self.text += delta["content"]
                        self.stage = "Writing response"
                    elif delta.get("tool_calls"):
                        if self.tool_started is None:
                            self.tool_started = time.monotonic()
                        for call in delta["tool_calls"]:
                            entry = self.tool_calls.setdefault(call.get("index", 0), ["", 0])
                            function = call.get("function") or {}
                            entry[0] = (entry[0] + (function.get("name") or ""))[:100]
                            entry[1] += len(function.get("arguments") or "")
                        names = ", ".join(entry[0] or "tool" for entry in self.tool_calls.values())
                        chars = sum(entry[1] for entry in self.tool_calls.values())
                        elapsed = int(time.monotonic() - self.tool_started)
                        self.stage = f"Preparing {names} · {chars:,} argument chars · {elapsed}s"
                    elif delta.get("reasoning_content") or delta.get("reasoning"):
                        self.stage = "Model working"
            except (ValueError, TypeError, AttributeError):
                continue  # Preview failures must not change inference/capture behavior.
        if time.monotonic() - self.last_sent >= 0.1:
            self.emit()

    def emit(self):
        self.last_sent = time.monotonic()
        try:
            self.callback({"type": "provider.progress", "stage": self.stage, "text": self.text})
        except Exception:
            pass  # A display callback cannot interrupt the model's response.


class CaptureProxy:
    def __init__(
        self,
        base_url,
        api_key,
        *,
        transport=None,
        on_progress=None,
        read_timeout=600,
        request_usage=False,
        default_max_tokens=None,
        reject_malformed_tool_history=False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.local_key = secrets.token_urlsafe(32)
        self.events = []
        self.health = "unverified"
        local = urlsplit(self.base_url).hostname in {"localhost", "127.0.0.1", "::1"}
        connection_options = (
            {"limits": httpx.Limits(max_keepalive_connections=0), "trust_env": False}
            if local else {}
        )
        # A tunnel/server can close an idle HTTP connection between tool turns.
        # Open a fresh local connection without replaying any model request.
        self.client = httpx.Client(
            timeout=httpx.Timeout(read_timeout, connect=15, write=30, pool=15),
            transport=transport, **connection_options,
        )
        self.server = None
        self.on_progress = on_progress
        self.request_usage = request_usage
        self.default_max_tokens = default_max_tokens
        self.reject_malformed_tool_history = reject_malformed_tool_history
        self.reported_tool_errors = set()
        self.empty_answer = False

    def report_tool_errors(self, request, *, historical=False):
        """Surface argument rejection promptly, without treating recovery as turn failure."""
        if not self.on_progress:
            return
        names = {}
        for message in request.get("messages", []):
            for call in message.get("tool_calls") or []:
                names[call.get("id")] = call.get("function", {}).get("name", "tool")
            content = message.get("content")
            call_id = message.get("tool_call_id")
            if (
                message.get("role") == "tool"
                and isinstance(content, str)
                and content.startswith("failed to parse function arguments:")
                and call_id not in self.reported_tool_errors
            ):
                self.reported_tool_errors.add(call_id)
                if historical:
                    continue  # Resuming must not replay errors from earlier turns.
                try:
                    self.on_progress(
                        {
                            "type": "tool.error",
                            "tool": names.get(call_id, "tool"),
                            "message": content[:240],
                        }
                    )
                except Exception:
                    pass

    def notify(self, kind):
        if self.on_progress:
            try:
                self.on_progress({"type": kind})
            except Exception:
                pass

    def report_tool_calls(self, exchange):
        """Observe completed provider calls, including tools absent from exec's UI events."""
        if not self.on_progress or exchange.get("status") != 200:
            return
        try:
            reply = parse_reply(
                bytes.fromhex("".join(exchange["response_chunks"])), exchange["content_type"]
            )
        except (ValueError, KeyError, TypeError, IndexError, UnicodeDecodeError):
            return
        for call in reply.get("tool_calls") or []:
            try:
                self.on_progress({"type": "provider.tool_call", "tool_call": call})
            except Exception:
                pass  # Observation does not become a synchronous execution gate.

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                if (
                    self.path != "/v1/chat/completions"
                    or self.headers.get("Authorization") != f"Bearer {owner.local_key}"
                ):
                    self.send_error(403)
                    return
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 32 * 1024 * 1024:
                    self.send_error(413)
                    return
                body = self.rfile.read(length)
                event = {
                    "type": "provider_exchange",
                    "at": utcnow(),
                    "request": json.loads(body),
                    "response_chunks": [],
                }
                owner.events.append(event)
                forwarded = repair_tool_schemas(event["request"])
                if owner.default_max_tokens is not None:
                    forwarded.setdefault("max_tokens", owner.default_max_tokens)
                if owner.request_usage and forwarded.get("stream"):
                    # Request real usage so the native engine can trigger compaction.
                    # Keep the original request as well as the exact forwarded payload.
                    forwarded["stream_options"] = {
                        **(forwarded.get("stream_options") or {}),
                        "include_usage": True,
                    }
                if forwarded != event["request"]:
                    event["engine_request"] = event["request"]
                    event["request"] = forwarded
                    body = json.dumps(event["request"]).encode()
                compacting = is_compaction_request(event["request"])
                event["compaction"] = compacting
                if compacting:
                    owner.notify("context.compaction.started")
                progress = (
                    ReplyProgress(owner.on_progress)
                    if owner.on_progress and not compacting
                    else None
                )
                if progress:
                    progress.emit()
                owner.report_tool_errors(event["request"], historical=len(owner.events) == 1)
                if owner.reject_malformed_tool_history and malformed_tool_history(forwarded):
                    # A truncated call cannot be replayed to this provider, even on /resume.
                    # Preserve the submitted history locally without inventing arguments.
                    detail = (
                        "The conversation contains an incomplete or malformed tool call. "
                        "No request was sent to the provider. Start /new with a short task "
                        "summary and ask for smaller tool calls."
                    )
                    for previous in owner.events[:-1]:
                        if previous.get("status") != 200:
                            continue
                        try:
                            parse_reply(bytes.fromhex("".join(previous["response_chunks"])),
                                        previous["content_type"])
                        except ResponseLengthError:
                            detail = "The previous model response reached its output token limit. " + detail
                            break
                        except (ValueError, KeyError, TypeError, IndexError, UnicodeDecodeError):
                            pass
                    payload = json.dumps({"error": {"type": "invalid_tool_call_history",
                                                    "message": detail}}).encode()
                    event.update(status=400, content_type="application/json",
                                 response_chunks=[payload.hex()], rejected_before_dispatch=True)
                    owner.health = "invalid_response"
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    if compacting:
                        owner.notify("context.compaction.failed")
                    return
                try:
                    with owner.client.stream(
                        "POST",
                        owner.base_url + "/chat/completions",
                        content=body,
                        headers={
                            "Authorization": f"Bearer {owner.api_key}",
                            "Content-Type": "application/json",
                        },
                    ) as response:
                        event["status"] = response.status_code
                        event["content_type"] = response.headers.get(
                            "content-type", "application/json"
                        )
                        if response.status_code == 402:
                            owner.health = "payment_required"
                        elif response.status_code in (401, 403):
                            owner.health = "authentication_error"
                        elif response.status_code == 429:
                            owner.health = "rate_limited"
                        elif response.is_success:
                            owner.health = "ok"
                        else:
                            owner.health = "provider_error"
                        self.send_response(response.status_code)
                        self.send_header("Content-Type", event["content_type"])
                        self.send_header("Connection", "close")
                        self.end_headers()
                        for chunk in response.iter_bytes():
                            # Hex preserves byte boundaries and malformed UTF-8 exactly.
                            event["response_chunks"].append(chunk.hex())
                            self.wfile.write(chunk)
                            self.wfile.flush()
                            if (
                                progress
                                and response.is_success
                                and "text/event-stream" in event["content_type"]
                            ):
                                progress.feed(chunk)
                        if progress:
                            progress.emit()
                        owner.report_tool_calls(event)
                except Exception as exc:
                    event["transport_error"] = type(exc).__name__
                    owner.health = "connection_error"
                    self.close_connection = True
                finally:
                    if compacting:
                        owner.notify(
                            "context.compaction.completed"
                            if owner.response_error() is None
                            else "context.compaction.failed"
                        )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def response_error(self):
        """Validate the final exchange independently of the engine's exit code."""
        self.empty_answer = False
        if not self.events:
            return "No model response was received."
        last = self.events[-1]
        if error := last.get("transport_error"):
            if error == "ReadTimeout":
                return (
                    f"Model response timed out after {self.client.timeout.read:g} seconds "
                    "without data. No complete answer was received; no automatic retry."
                )
            return f"Model connection interrupted ({error}); no automatic retry."
        if last.get("status") != 200:
            try:
                payload = json.loads(bytes.fromhex("".join(last["response_chunks"])))
                error = payload.get("error", {}) if isinstance(payload, dict) else {}
            except (ValueError, KeyError, TypeError):
                error = {}
            if last.get("status") == 426:
                return (
                    "Adverserial requires encrypted confidential inference (HTTP 426). "
                    "Install or refresh the client with scripts/install/install_api_gateway.py."
                )
            if isinstance(error, dict) and error.get("type") == "confidential_gateway_error":
                # The gateway is local; redact the credential even if billing echoed it.
                detail = str(error.get("message", "Request rejected"))
                if self.api_key:
                    detail = detail.replace(self.api_key, "[REDACTED_SECRET]")
                if "\ufffd" in detail or any(ord(c) < 32 and c not in "\n\r\t" for c in detail):
                    status = re.search(r"HTTP (\d{3})", detail)
                    detail = (
                        "Request failed" + (" with HTTP " + status[1] if status else "")
                        + "; the gateway returned unreadable encrypted error details."
                    )
                return "Confidential API: " + detail[:700] + " No automatic retry."
            if isinstance(error, dict) and error.get("type") == "invalid_tool_call_history":
                self.health = "invalid_response"
                return str(error["message"])
            if (
                last.get("status") == 503
                and isinstance(error, dict)
                and error.get("type") == "model_maintenance"
            ):
                self.health = "maintenance"
                model = (last.get("request") or {}).get("model")
                name = next(
                    (profile[0] for profile in API_MODELS.values() if profile[1] == model),
                    "The selected model",
                )
                # Do not echo arbitrary provider text: it can contain request data.
                return (
                    f"{name} is unavailable during provider maintenance (HTTP 503). "
                    "Choose another model or try again later; no automatic retry."
                )
            if isinstance(error, dict) and error.get("type") == "exceed_context_size_error":
                self.health = "context_limit"
                prompt, window = error.get("n_prompt_tokens"), error.get("n_ctx")
                detail = (
                    f" ({prompt:,} requested; {window:,} available)"
                    if (type(prompt) is int and type(window) is int)
                    else ""
                )
                return (
                    "Conversation exceeds the model's context window"
                    + detail
                    + ". Start /new with a short task summary. Reconnecting the tunnel will not help."
                )
            unavailable = {502: "Bad Gateway", 503: "Service Unavailable", 504: "Gateway Timeout"}
            if last.get("status") in unavailable:
                self.health = "temporarily_unavailable"
                return (
                    f"Model provider returned HTTP {last['status']} "
                    f"({unavailable[last['status']]}). The upstream service could not complete "
                    "the request. Try again later or choose another model; no automatic retry."
                )
            return (
                f"Model provider returned HTTP {last.get('status', 'unknown')}; no automatic retry."
            )
        try:
            reply = parse_reply(
                bytes.fromhex("".join(last["response_chunks"])), last["content_type"]
            )
        except ResponseLengthError:
            self.health = "context_limit"
            return (
                "Model reached its output token limit before completing the response. "
                "The partial answer was excluded from training. "
                "Reasoning and tool arguments share this limit. Use smaller tool calls "
                "or increase the model's output allowance; no automatic retry."
            )
        except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError):
            self.health = "invalid_response"
            return "Model response ended before completion or was malformed; no automatic retry."
        if reply.get("tool_calls"):
            return "Task ended with an unfinished model tool call; no final answer was received."
        content = reply.get("content")
        if not content or (isinstance(content, str) and not content.strip()):
            self.empty_answer = True
            self.health = "invalid_response"
            return "Model returned an empty answer. No automatic retry."
        return None

    def transcript(self):
        exchanges = [e for e in self.events if e.get("status") == 200]
        if not exchanges:
            return [], [], False
        last = exchanges[-1]
        messages = copy.deepcopy(last["request"].get("messages", []))
        complete = False
        try:
            reply = parse_reply(
                bytes.fromhex("".join(last["response_chunks"])), last["content_type"]
            )
            messages.append(reply)
            complete = not reply.get("tool_calls")
        except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError):
            pass
        previous = []
        for exchange in exchanges:
            current = exchange["request"].get("messages", [])
            if current[: len(previous)] != previous:
                complete = False  # Compaction/rewrite requires separate curation.
            previous = current
        return messages, last["request"].get("tools", []), complete

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.client.close()
