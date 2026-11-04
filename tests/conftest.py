import copy
import json
import os
from pathlib import Path

import pytest

from velocicyber.config import Settings


@pytest.fixture
def native_engine():
    if configured := os.environ.get("VELOCICYBER_TEST_ENGINE"):
        binary = Path(configured)
        assert binary.is_file(), "The requested candidate engine is missing"
        return binary
    root = Path(__file__).resolve().parents[1]
    try:
        return Settings(root=root, local_model="", dataset_repo="").engine()
    except RuntimeError:
        pytest.skip("Install the pinned ARM64 engine for integration tests")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    import velocicyber.models.catalog as profiles

    # Register the fake server model only in tests; never ship it as a menu option.
    monkeypatch.setattr(profiles, "SUPPORTED_MODELS",
                        profiles.SUPPORTED_MODELS | {"example-org/coding-model"})
    (tmp_path / "config/locks").mkdir(parents=True)
    return Settings(
        root=tmp_path,
        local_model="example-org/coding-model",
        dataset_repo="owner/dataset",
        options={"SENTINEL_FINETUNING_REPO": "owner/sentinel-dataset"},
        secrets={"ADVERSARIAL_API": "test-key"},
    )


@pytest.fixture
def episode():
    return {
        "episode_id": "ep_success",
        "schema_version": "1.0",
        "campaign_id": "frozen-prompt",
        "task": {"id": "test-task", "prompt": "Fix the test"},
        "system_prompt": "Be helpful",
        "messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Fix the test"},
            {
                "role": "assistant",
                "content": "Run a check",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command":"pytest"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "FAILED: expected 2, got 3"},
            {"role": "assistant", "content": "Corrected it and reran the suite."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ],
        "context_complete": True,
        "outcome": {"success": True, "verifier": "test_suite"},
        "territory": {"secrets_scrubbed": False},
    }


@pytest.fixture
def failed_episode(episode):
    failed = copy.deepcopy(episode)
    failed["episode_id"] = "ep_failure"
    failed["outcome"]["success"] = False
    failed["messages"][-1]["content"] = "The suite still fails."
    return failed


@pytest.fixture
def compacted_episode(episode):
    from velocicyber.engine.compaction import CHECKPOINT_HEADER
    from velocicyber.engine.context import COMPACTION_PROMPT

    episode = copy.deepcopy(episode)
    episode["episode_id"] = "ep_compacted"
    summary = "The test failed with 3 instead of 2. Next: correct it and rerun the suite."
    checkpoint = {"role": "user", "content": CHECKPOINT_HEADER + "\n" + summary}
    before = copy.deepcopy(episode["messages"])
    episode["messages"].insert(2, checkpoint)
    episode["context_complete"] = False

    def exchange(messages, reply, compacting=False):
        return {
            "request": {"messages": messages, "tools": episode["tools"]},
            "status": 200,
            "content_type": "application/json",
            "compaction": compacting,
            "response_chunks": [
                json.dumps(
                    {
                        "choices": [{"message": reply, "finish_reason": "stop"}],
                    }
                )
                .encode()
                .hex()
            ],
        }

    episode["provider_exchanges"] = [
        exchange(
            before + [{"role": "user", "content": COMPACTION_PROMPT}],
            {"role": "assistant", "content": summary},
            True,
        ),
        exchange(copy.deepcopy(episode["messages"][:-1]), episode["messages"][-1]),
    ]
    return episode
