"""Recognize native checkpoints and validate the continuation actually captured."""

from velocicyber.engine.capture import parse_reply
from velocicyber.engine.context import is_compaction_request

# The native engine inserts this wrapper around the model's checkpoint.
CHECKPOINT_HEADER = (
    "Another language model started to solve this problem and produced a summary "
    "of its thinking process."
)


def checkpoint_index(messages):
    indices = [
        i
        for i, message in enumerate(messages)
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].startswith(CHECKPOINT_HEADER)
    ]
    return indices[-1] if indices else None


def compaction_exchanges(exchanges):
    return [
        e for e in exchanges if e.get("compaction") or is_compaction_request(e.get("request", {}))
    ]


def was_compacted(episode):
    return bool(
        episode.get("context_compacted")
        or compaction_exchanges(episode.get("provider_exchanges", []))
        or checkpoint_index(episode.get("messages", [])) is not None
    )


def valid_continuation(episode):
    """A compaction flag alone cannot excuse a missing or unrelated rewritten context."""
    messages = episode.get("messages", [])
    checkpoint = checkpoint_index(messages)
    if checkpoint is None or checkpoint >= len(messages) - 1:
        return False
    if episode.get("context_complete"):
        return True  # A later/resumed turn whose captured requests only append.
    exchanges = episode.get("provider_exchanges", [])
    compactions = compaction_exchanges(exchanges)
    if not compactions:
        return False
    latest = compactions[-1]
    tail = exchanges[exchanges.index(latest) + 1 :]
    if latest.get("status") != 200 or not tail:
        return False
    try:
        summary = parse_reply(
            bytes.fromhex("".join(latest["response_chunks"])), latest["content_type"]
        )
        content = summary.get("content")
        if (
            summary.get("tool_calls")
            or not isinstance(content, str)
            or not content.strip()
            or content not in messages[checkpoint]["content"]
        ):
            return False
        previous = []
        for exchange in tail:
            if exchange.get("status") != 200:
                continue  # A recovered failed request is not a training target.
            current = exchange["request"].get("messages", [])
            if current[: len(previous)] != previous or checkpoint_index(current) is None:
                return False
            previous = current
        last = tail[-1]
        reply = parse_reply(bytes.fromhex("".join(last["response_chunks"])), last["content_type"])
        return last.get("status") == 200 and previous == messages[:-1] and reply == messages[-1]
    except (ValueError, KeyError, TypeError, IndexError, UnicodeDecodeError):
        return False
