"""Conservative execution evidence for a turn that ends without a final reply."""

from collections import Counter

from velocicyber.engine.capture import parse_reply
from velocicyber.engine.context import is_compaction_request


def _native_actions(events):
    items = {}
    for event in events:
        if event.get("type") in {"turn.failed", "engine.error", "error"}:
            return None
        if event.get("type") not in {"item.started", "item.updated", "item.completed"}:
            continue
        item = event.get("item", {})
        if item.get("type") not in {"command_execution", "file_change"}:
            continue
        if not item.get("id"):
            return None
        items[item["id"]] = (event["type"], item)
    actual = Counter()
    for kind, item in items.values():
        if kind != "item.completed" or item.get("status") != "completed":
            return None
        if item["type"] == "command_execution" and item.get("exit_code") != 0:
            return None
        actual[item["type"]] += 1
    return actual


def completed_native_actions(events):
    """Count visible successful actions; other tools and task completion remain unverified."""
    return sum((_native_actions(events) or {}).values())


def successful_actions(events, exchanges):
    """Count actions only when every current-turn call has a successful native result.

    Exit zero/file-change completion proves execution, not overall task correctness.
    Unsupported tools, missing results, background jobs and failures need review.
    Earlier turns' successful calls cannot justify this turn's empty answer.
    """
    groups = {
        "exec_command": "command_execution",
        "Bash": "command_execution",
        "bash": "command_execution",
        "write": "file_change",
        "Write": "file_change",
        "edit": "file_change",
        "Edit": "file_change",
    }
    expected = Counter()
    call_ids, returned = set(), set()
    for exchange in exchanges:
        if exchange.get("status") != 200 or exchange.get("transport_error"):
            return 0
        try:
            reply = parse_reply(
                bytes.fromhex("".join(exchange["response_chunks"])), exchange["content_type"]
            )
        except (ValueError, KeyError, TypeError, IndexError, UnicodeDecodeError):
            return 0
        # The result immediately preceding compaction may appear only in the
        # checkpoint request: the continuation has already replaced that history.
        returned.update(
            message.get("tool_call_id")
            for message in exchange["request"].get("messages", [])
            if message.get("role") == "tool"
        )
        if exchange.get("compaction") or is_compaction_request(exchange["request"]):
            if reply.get("tool_calls") or not str(reply.get("content") or "").strip():
                return 0
            continue
        for call in reply.get("tool_calls") or []:
            name = call.get("function", {}).get("name")
            if name not in groups or not call.get("id") or call["id"] in call_ids:
                return 0
            call_ids.add(call["id"])
            expected[groups[name]] += 1
    if not expected or not call_ids <= returned:
        return 0
    actual = _native_actions(events)
    return sum(actual.values()) if actual == expected else 0
