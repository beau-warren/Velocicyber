"""Offline Inkling integration: no weights, paid instances, or model execution."""

import ast
import asyncio
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from textual.widgets import Input, Static

from velocicyber.config import Settings
from velocicyber.engine.capture import parse_reply
from velocicyber.engine.context import ContextBudget
from velocicyber.models.catalog import model_catalog
from velocicyber.models.gpu_profiles import INKLING_REPO
from velocicyber.providers.lambda_cloud import GPUManager
from velocicyber.state import GPUStatus, State
from velocicyber.ui.app import VelocicyberApp

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_BYTES = 20_161_820_584


@pytest.fixture
def inkling(settings, monkeypatch):
    import huggingface_hub

    settings.local_model = INKLING_REPO
    settings.options.update(LAMBDA_SSH_KEY_NAME="fixture", SSH_PRIVATE_KEY="fake-key")
    (settings.root / "fake-key").write_text("fixture")
    for name in ("config/locks/serving.lock.json", "scripts/runtime/serve_model.py"):
        p = settings.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    config = settings.root / "config.json"
    config.write_text(json.dumps({
        "architectures": ["Inkling2LoopForConditionalGeneration"], "model_type": "inkling2_loop",
        "text_config": {"model_max_length": 1048576, "num_attention_heads": 24},
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }))
    sizes = {
        "model-00001-of-00001.safetensors": 19526019168, "mtp.safetensors": 635801416,
        "config.json": config.stat().st_size, "hf_quant_config.json": 67924,
        "model.safetensors.index.json": 109300, "chat_template.jinja": 6868,
        "tokenizer.json": 26178678, "tokenizer_config.json": 13235,
        "tiktoken/tokenizer.model": 3386494,
        "DWM_RECIPE/private.json": 100, "deployment-kit/scripts/serve.sh": 100,
    }
    info = SimpleNamespace(sha="a" * 40, siblings=[
        SimpleNamespace(rfilename=name, size=size, lfs=SimpleNamespace(sha256="b" * 64))
        for name, size in sizes.items()
    ])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(model_info=lambda *a, **kw: info))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(config))
    manager = GPUManager(settings, State())
    return manager, info


def test_complete_checkpoint_and_engine_context(inkling):
    manager, _ = inkling
    spec = manager.inspect_model()
    assert spec.weight_bytes == WEIGHT_BYTES and spec.required_gib == 32
    assert spec.tool_parser == spec.reasoning_parser == "inkling"
    assert spec.context_window == 65536
    names = {f["filename"] for f in spec.download_files}
    assert {"mtp.safetensors", "hf_quant_config.json", "tiktoken/tokenizer.model"} <= names
    assert not any(n.startswith(("DWM_RECIPE/", "deployment-kit/")) for n in names)
    assert "@sha256:" in manager.serving_image(spec)
    catalog = model_catalog(spec.repo, ContextBudget(spec.context_window), "instructions")
    assert catalog["models"][0]["context_window"] == 65536


@pytest.mark.parametrize("missing", ["mtp.safetensors", "hf_quant_config.json", "tiktoken/tokenizer.model"])
def test_incomplete_checkpoint_stops_before_lambda(inkling, missing):
    manager, info = inkling
    info.siblings = [f for f in info.siblings if f.rfilename != missing]
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        manager.plan(80)
    assert manager._client is None


def test_changed_quantization_and_short_context_are_rejected(inkling):
    manager, _ = inkling
    p = manager.settings.root / "config.json"
    config = json.loads(p.read_text())
    config["quantization_config"]["quant_algo"] = "different"
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="native ModelOpt NVFP4"):
        manager.inspect_model()
    config["quantization_config"]["quant_algo"] = "NVFP4"
    config["text_config"]["model_max_length"] = 8192
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="64K context"):
        manager.inspect_model()


def test_24gb_stops_before_capacity_request(inkling):
    manager, _ = inkling
    with pytest.raises(ValueError, match="too small"):
        manager.plan(24)
    assert manager._client is None


@pytest.mark.parametrize("family,memory,ram,disk,allowed", [
    ("H100", 80, 64, 128, True), ("B200", 180, 64, 128, True),
    ("A100", 80, 64, 128, False), ("A100", 40, 64, 128, False),
    ("A6000", 48, 64, 128, False), ("H100", 80, 32, 128, False),
    ("H100", 80, 64, 64, False),
])
def test_family_and_resources_checked_before_launch(inkling, family, memory, ram, disk, allowed):
    manager, _ = inkling
    offer = {"instance_type": {
        "name": "gpu_1x_" + family.lower(), "gpu_description": f"{family} {memory}GB",
        "architecture": "x86_64", "price_cents_per_hour": 100,
        "specs": {"gpus": 1, "memory_gib": ram, "storage_gib": disk},
    }, "regions_with_capacity_available": [{"name": "test-region"}]}
    manager._client = SimpleNamespace(catalog=lambda: {"offer": offer})
    if allowed:
        plan = manager.plan(memory)
        with pytest.raises(PermissionError):
            manager.launch(plan)
    else:
        with pytest.raises(ValueError, match="No available"):
            manager.plan(memory)
    assert not manager.state.instance_id


async def test_remote_manifest_entrypoint_and_speculation(inkling, monkeypatch):
    manager, _ = inkling
    manager.state.gpu_status = GPUStatus.RUNNING
    manager.state.instance_type = "gpu_1x_h100"
    manager.state.instance_ip = "192.0.2.1"
    manager.state.gpu_count, manager.state.total_vram_gib = 1, 80
    captured = {}

    class Process:
        returncode = 0

        def __init__(self):
            self.stdin = self
            self.stdout, self.stderr = asyncio.StreamReader(), asyncio.StreamReader()
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        def write(self, program):
            captured.update(json.loads(ast.literal_eval(ast.parse(program).body[-1].value.args[0].args[0])))

        async def drain(self): pass
        def close(self): pass
        async def wait(self): return 0

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=Process()))
    monkeypatch.setattr(manager, "connect", AsyncMock())
    await manager.deploy()
    module_spec = importlib.util.spec_from_file_location("inkling_serve", ROOT / "scripts/runtime/serve_model.py")
    serving = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(serving)
    commands = []
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: manager.settings.root))
    monkeypatch.setattr(serving.subprocess, "run", lambda args, **kw: commands.append(args) or SimpleNamespace(returncode=0, stdout="true\n"))
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=200 * 2**30))
    serving.serve(captured)
    launch = commands[-1]
    assert launch[launch.index("--entrypoint") + 1] == "vllm"
    assert launch[launch.index(captured["image"]) + 1:][:2] == ["serve", INKLING_REPO]
    assert launch[launch.index("--max-model-len") + 1] == "65536"
    assert launch[launch.index("--max-num-seqs") + 1] == "1"
    assert launch[launch.index("--tool-call-parser") + 1] == "inkling"
    assert launch[launch.index("--reasoning-parser") + 1] == "inkling"
    assert json.loads(launch[launch.index("--speculative-config") + 1])["num_speculative_tokens"] == 7
    assert "HF_HUB_OFFLINE=1" in launch and "127.0.0.1:8000:8000" in launch
    assert "--trust-remote-code" not in launch and "--network" not in launch
    assert any("inkling_tool_parser" in arg for command in commands for arg in command)
    captured["download_files"] = [{"filename": "../secret", "size": 1, "sha256": ""},
                                  {"filename": "model.safetensors", "size": WEIGHT_BYTES, "sha256": "b" * 64}]
    with pytest.raises(ValueError, match="Invalid model file"):
        serving.download_safetensors(captured, manager.settings.root, ["docker"], [])


@pytest.mark.parametrize("failure", ["", "tool", "followup"])
async def test_readiness_uses_correct_effort_and_requires_tool_round_trip(inkling, monkeypatch, failure):
    manager, _ = inkling
    requests = []

    def respond(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": INKLING_REPO, "max_model_len": 65536}]})
        body = json.loads(request.content)
        requests.append(body)
        assert body["chat_template_kwargs"] == {"reasoning_effort": "none"}
        message = {"content": "OK", "reasoning": ""}
        if "tools" in body and failure != "tool":
            message = {"tool_calls": [{"id": "probe", "type": "function", "function": {
                "name": "readiness_probe", "arguments": '{"value":"OK"}',
            }}]}
        if len(body["messages"]) == 3:
            assert json.loads(body["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {"value": "OK"}
            assert body["messages"][2]["tool_call_id"] == "probe"
            if failure == "followup":
                message = {"content": ""}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(**kw, transport=httpx.MockTransport(respond)))
    manager.tunnel = SimpleNamespace(returncode=None)
    manager.state.gpu_status = GPUStatus.RUNNING
    if failure:
        with pytest.raises(RuntimeError, match="parser check failed|follow-up returned no text"):
            await manager.check_ready()
        assert not manager.state.model_ready
    else:
        assert await manager.check_ready()
        assert len(requests) == 3


async def test_menu_details_are_offline(settings):
    assert INKLING_REPO in Settings.load(ROOT).gpu_model_choices
    settings.options["LOCAL_MODEL_OPTIONS"] = INKLING_REPO
    settings.local_model = ""
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test() as pilot:
        app.menu_selected("local")
        await pilot.pause()
        app.screen.query_one(Input).value = "inkling"
        await pilot.press("enter")
        app.screen.query_one(Input).value = "more details"
        await pilot.press("enter")
        text = str(app.screen.query_one("#model-details-body Static", Static).render())
        assert "80 GB (H100)" in text and "64K" in text and "Projected" in text
        assert "20.16 GB" in text and "128 GiB disk" in text
        await pilot.press("escape")
        assert not settings.local_model and app.gpu._client is None


def test_native_reasoning_field_is_captured():
    body = b'data: {"choices":[{"delta":{"reasoning":"Checking scope."}}]}\n\n'
    body += b'data: {"choices":[{"delta":{"content":"Ready."},"finish_reason":"stop"}]}\n\n'
    result = parse_reply(body, "text/event-stream")
    assert result["reasoning"] == "Checking scope." and result["content"] == "Ready."
