"""Offline SAGE planning, deployment and UI coverage; no paid provider calls."""

import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from textual.widgets import Input, Static

from velocicyber.config import Settings
from velocicyber.engine.context import ContextBudget
from velocicyber.models.catalog import model_catalog
from velocicyber.models.gpu_profiles import SAGE_REPO
from velocicyber.models.startup import model_details
from velocicyber.providers.lambda_cloud import GPUManager
from velocicyber.state import GPUStatus, State
from velocicyber.ui.app import VelocicyberApp

ROOT = Path(__file__).resolve().parents[1]
# Public Hub file sizes reviewed at revision 625ca8d6528fcb65407a7e5685778b8d4b16022c.
SHARD_BYTES = (
    4324097448, 4506714344, 4989093232, 3962454952,
    4506715976, 4989095576, 3962456776, 4506716432,
    4989095576, 3962456776, 4506716432, 4989095576,
    3962456776, 4506716432, 4989095576, 2565780424,
)
MTP_BYTES = 1689376064


@pytest.fixture
def sage(settings, monkeypatch):
    import huggingface_hub

    settings.local_model = SAGE_REPO
    settings.options.update(LAMBDA_SSH_KEY_NAME="fixture", SSH_PRIVATE_KEY="fake-key")
    (settings.root / "fake-key").write_text("fixture")
    (settings.root / "config/locks/serving.lock.json").write_bytes((ROOT / "config/locks/serving.lock.json").read_bytes())
    script = settings.root / "scripts/runtime/serve_model.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes((ROOT / "scripts/runtime/serve_model.py").read_bytes())
    config = settings.root / "config.json"
    config.write_text(json.dumps({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_attention_heads": 16, "num_key_value_heads": 2,
            "num_hidden_layers": 40, "max_position_embeddings": 262144,
            "mamba_ssm_dtype": "float32", "dtype": "bfloat16",
        },
    }))
    weights = [SimpleNamespace(
        rfilename=f"model-{i:05d}-of-00016.safetensors", size=size,
        lfs=SimpleNamespace(sha256=f"{i:064x}"),
    ) for i, size in enumerate(SHARD_BYTES, 1)]
    # The actual index includes MTP despite the publisher's card saying otherwise.
    weights.append(SimpleNamespace(rfilename="model-mtp.safetensors", size=MTP_BYTES,
                                   lfs=SimpleNamespace(sha256="b" * 64)))
    info = SimpleNamespace(sha="a" * 40, siblings=weights + [
        SimpleNamespace(rfilename="config.json", size=config.stat().st_size, lfs=None),
        SimpleNamespace(rfilename="model.safetensors.index.json", size=1000, lfs=None),
        SimpleNamespace(rfilename="chat_template.jinja", size=9025, lfs=None),
        SimpleNamespace(rfilename="tokenizer.json", size=19989325, lfs=None),
        SimpleNamespace(rfilename="ASSETS/banner.png", size=100, lfs=None),
    ])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(
        model_info=lambda *a, **kw: info,
    ))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(config))
    return GPUManager(settings, State())


def test_inspection_keeps_all_indexed_weights_and_caps_engine_context(sage):
    before = (sage.settings.root / "config.json").read_bytes()
    spec = sage.inspect_model()
    assert spec.architecture == "qwen3_5_moe" and spec.backend == "vllm"
    assert spec.revision == "a" * 40 and spec.weight_bytes == 71908134368
    assert spec.required_gib == 96 and spec.context_window == 32768
    assert spec.tool_parser == "qwen3_coder" and spec.reasoning_parser == "qwen3"
    weights = [f for f in spec.download_files if f["filename"].endswith(".safetensors")]
    assert len(weights) == 17 and sum(f["size"] for f in weights) == spec.weight_bytes
    assert any(f["filename"] == "model-mtp.safetensors" for f in weights)
    assert all("/" not in f["filename"] for f in spec.download_files)
    assert (sage.settings.root / "config.json").read_bytes() == before
    sage.settings.options["VLLM_IMAGE"] = "unvalidated-image"
    assert "@sha256:" in sage.serving_image(spec)
    catalog = model_catalog(spec.repo, ContextBudget(spec.context_window), "instructions")
    assert catalog["models"][0]["context_window"] == 32768


@pytest.mark.parametrize("memory", [24, 40, 48, 80])
def test_small_gpu_rejected_before_requesting_lambda(sage, memory):
    with pytest.raises(ValueError, match="too small"):
        sage.plan(memory)
    assert sage._client is None


def offer(*, family="B200", ram=224, disk=1024, count=1):
    return {
        "instance_type": {
            "name": f"gpu_{count}x_{family.lower()}", "gpu_description": f"{family} 180GB",
            "architecture": "x86_64", "price_cents_per_hour": 100,
            "specs": {"gpus": count, "memory_gib": ram, "storage_gib": disk},
        },
        "regions_with_capacity_available": [{"name": "test-region"}],
    }


@pytest.mark.parametrize("count,memory", [(1, 180), (2, 360)])
def test_single_and_cluster_plans_require_explicit_launch_approval(sage, count, memory):
    sage._client = SimpleNamespace(catalog=lambda: {"offer": offer(count=count)})
    plan = sage.plan(memory)
    assert plan.gpu_count == count and plan.total_vram_gib == memory
    with pytest.raises(PermissionError):
        sage.launch(plan)
    assert not sage.state.instance_id


@pytest.mark.parametrize("row", [offer(family="A100"), offer(ram=64), offer(disk=128)])
def test_incompatible_instance_is_not_offered(sage, row):
    sage._client = SimpleNamespace(catalog=lambda: {"offer": row})
    with pytest.raises(ValueError, match="128 GiB host RAM and 256 GiB disk"):
        sage.plan(180)


def test_architecture_change_stops_planning_before_lambda(sage):
    config = sage.settings.root / "config.json"
    config.write_text(json.dumps({"model_type": "different"}))
    with pytest.raises(ValueError, match="architecture changed"):
        sage.plan(180)
    assert sage._client is None


async def test_deployment_carries_manifest_and_correct_vllm_arguments(sage, monkeypatch):
    sage.settings.options["LOCAL_MODEL_CONTEXT_WINDOWS"] = json.dumps({SAGE_REPO: 262144})
    sage.state.gpu_status = GPUStatus.RUNNING
    sage.state.instance_type = "gpu_1x_b200"
    sage.state.instance_ip = "192.0.2.1"
    sage.state.gpu_count, sage.state.total_vram_gib = 1, 180
    captured = {}

    class Process:
        returncode = 0

        def __init__(self):
            self.stdin = self
            self.stdout, self.stderr = asyncio.StreamReader(), asyncio.StreamReader()
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        def write(self, program):
            tree = ast.parse(program)
            captured.update(json.loads(ast.literal_eval(tree.body[-1].value.args[0].args[0])))

        async def drain(self): pass
        def close(self): pass
        async def wait(self): return 0

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=Process()))
    monkeypatch.setattr(sage, "connect", AsyncMock())
    await asyncio.wait_for(sage.deploy(), timeout=15)
    assert captured["context_window"] == 32768 and captured["max_num_seqs"] == 1
    assert captured["weight_bytes"] == 71908134368
    assert len([f for f in captured["download_files"] if f["filename"].endswith(".safetensors")]) == 17
    module_spec = importlib.util.spec_from_file_location("sage_serve", ROOT / "scripts/runtime/serve_model.py")
    serving = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(serving)
    commands = []
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: sage.settings.root))
    monkeypatch.setattr(serving.subprocess, "run", lambda args, **kw:
                        commands.append(args) or SimpleNamespace(returncode=0, stdout="true\n"))
    monkeypatch.setattr(serving, "download_safetensors", lambda *a: None)
    serving.serve(captured)
    launch = commands[-1]
    for flag, value in {
        "--tensor-parallel-size": "1", "--max-model-len": "32768",
        "--max-num-seqs": "1", "--dtype": "bfloat16",
        "--mamba-ssm-cache-dtype": "float32", "--tool-call-parser": "qwen3_coder",
        "--reasoning-parser": "qwen3", "--served-model-name": SAGE_REPO,
    }.items():
        assert launch[launch.index(flag) + 1] == value
    assert "--language-model-only" in launch and "HF_HUB_OFFLINE=1" in launch
    assert "--speculative-config" not in launch and "--trust-remote-code" not in launch
    assert sage._client is None and not sage.state.model_ready


@pytest.mark.parametrize("valid_tool", [True, False])
async def test_ready_requires_a_valid_coding_tool_response(sage, monkeypatch, valid_tool):
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": SAGE_REPO, "max_model_len": 32768}]})
        message = {"content": "OK"}
        if "tools" in json.loads(request.content) and valid_tool:
            message = {"tool_calls": [{"type": "function", "function": {
                "name": "readiness_probe", "arguments": '{"value":"OK"}',
            }}]}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw:
                        client(**kw, transport=httpx.MockTransport(handler)))
    sage.tunnel = SimpleNamespace(returncode=None)
    sage.state.gpu_status = GPUStatus.RUNNING
    if valid_tool:
        assert await sage.check_ready()
        assert sage.state.active_provider == "local"
    else:
        with pytest.raises(RuntimeError, match="coding-tool parser check failed"):
            await sage.check_ready()
        assert not sage.state.model_ready


async def test_menu_details_and_confirmation_are_offline(settings):
    configured = Settings.load(ROOT)
    assert SAGE_REPO in configured.gpu_model_choices and not configured.local_model
    assert SAGE_REPO not in configured.pi_model_choices
    settings.local_model = ""
    settings.options["LOCAL_MODEL_OPTIONS"] = SAGE_REPO
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test() as pilot:
        app.menu_selected("local")
        await pilot.pause()
        app.screen.query_one(Input).value = "sage"
        await pilot.press("enter")
        assert app.screen.id == "local-model-choice" and not settings.local_model
        app.screen.query_one(Input).value = "more details"
        await pilot.press("enter")
        assert app.screen.id == "local-model-details"
        details = app.screen.query_one("#model-details-body Static", Static).render()
        assert "180 GB (B200)" in str(details) and "Projected" in str(details)
        assert "32K" in str(details) and "71.9 GB" in str(details)
        app.screen.query_one(Input).value = "confirm"
        await pilot.press("enter")
        assert settings.local_model == SAGE_REPO and app.state.gpu_status == GPUStatus.STOPPED
        assert app.gpu._client is None and not app.state.model_ready
    assert "262K context are not enabled" in model_details(SAGE_REPO)
