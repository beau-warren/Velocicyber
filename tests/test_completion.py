import copy
import json

import pytest

from velocicyber.engine.completion import completed_native_actions, successful_actions
from velocicyber.engine.context import COMPACTION_PROMPT


def exchange(reply, messages=()):
    return {
        "status": 200,
        "content_type": "application/json",
        "request": {"messages": list(messages)},
        "response_chunks": [
            json.dumps(
                {
                    "choices": [
                        {
                            "message": reply,
                            "finish_reason": "tool_calls" if reply.get("tool_calls") else "stop",
                        }
                    ]
                }
            )
            .encode()
            .hex()
        ],
    }


@pytest.fixture
def completed_action():
    call = {
        "id": "call1",
        "type": "function",
        "function": {
            "name": "exec_command",
            "arguments": '{"cmd":"rm test1.txt test2.txt"}',
        },
    }
    exchanges = [
        exchange({"role": "assistant", "tool_calls": [call]}),
        exchange(
            {"role": "assistant", "content": ""},
            [
                {"role": "tool", "tool_call_id": "call1", "content": "exit 0"},
            ],
        ),
    ]
    events = [
        {
            "type": "item.completed",
            "item": {
                "id": "item1",
                "type": "command_execution",
                "status": "completed",
                "exit_code": 0,
            },
        }
    ]
    return events, exchanges


def test_current_native_success_is_execution_evidence(completed_action):
    events, exchanges = completed_action
    assert successful_actions(events, exchanges) == 1
    assert successful_actions(events * 2, exchanges) == 1  # Repeated UI updates aren't more calls.


@pytest.mark.parametrize(
    "failure",
    [
        "failed",
        "background",
        "missing_result",
        "unknown_tool",
        "old_history",
        "transport",
        "truncated",
        "turn_failed",
    ],
)
def test_uncertain_or_failed_work_cannot_be_labelled_successful(completed_action, failure):
    events, exchanges = copy.deepcopy(completed_action)
    if failure == "failed":
        events[0]["item"]["exit_code"] = 1
    elif failure == "background":
        events[0]["item"]["status"] = "in_progress"
    elif failure == "missing_result":
        exchanges[-1]["request"]["messages"] = []
    elif failure == "unknown_tool":
        reply = {
            "role": "assistant",
            "tool_calls": [{"id": "call1", "function": {"name": "unknown"}}],
        }
        exchanges[0] = exchange(reply)
    elif failure == "old_history":
        exchanges = exchanges[-1:]  # Only a historical tool output exists; no call this turn.
    elif failure == "transport":
        exchanges[-1]["transport_error"] = "ReadTimeout"
    elif failure == "truncated":
        exchanges[-1]["response_chunks"] = [b'{"choices":[]}'.hex()]
    elif failure == "turn_failed":
        events.append({"type": "turn.failed"})
    assert successful_actions(events, exchanges) == 0


def test_file_write_requires_a_matching_native_completion(completed_action):
    events, exchanges = completed_action
    exchanges[0] = exchange(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call1",
                    "function": {"name": "write", "arguments": '{"path":"a","content":"b"}'},
                }
            ],
        }
    )
    assert successful_actions(events, exchanges) == 0
    events[0]["item"] = {"id": "file1", "type": "file_change", "status": "completed"}
    assert successful_actions(events, exchanges) == 1


@pytest.mark.parametrize("bad_summary", [False, True])
def test_tool_result_in_compaction_request_is_preserved_as_evidence(completed_action, bad_summary):
    events, exchanges = completed_action
    checkpoint = exchange(
        {"role": "assistant", "content": "" if bad_summary else "Checkpoint: deletion completed."},
        [
            *exchanges[-1]["request"]["messages"],
            {"role": "user", "content": COMPACTION_PROMPT},
        ],
    )
    exchanges.insert(1, checkpoint)
    exchanges[-1]["request"]["messages"] = [
        {"role": "user", "content": "Checkpoint: deletion completed."},
    ]
    assert successful_actions(events, exchanges) == (0 if bad_summary else 1)


def test_native_activity_does_not_prove_auxiliary_tool_success(completed_action):
    events, exchanges = completed_action
    exchanges.insert(1, exchange({
        "role": "assistant", "tool_calls": [
            {"id": "aux", "function": {"name": "get_goal", "arguments": "{}"}},
        ],
    }))
    assert completed_native_actions(events) == 1
    assert successful_actions(events, exchanges) == 0
    events[0]["item"]["exit_code"] = 1
    assert completed_native_actions(events) == 0
