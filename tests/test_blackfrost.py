import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from velocicyber.engine.context import ContextBudget
from velocicyber.models.catalog import model_catalog
from velocicyber.models.gpu_profiles import CYBER_FROST_REPO, GPU_PROFILES, MOG_SEC_REPO
from velocicyber.models.startup import model_option_label
from velocicyber.providers.lambda_cloud import GPUManager
from velocicyber.state import GPUStatus, State


@pytest.fixture
def blackfrost(settings, monkeypatch):
    import huggingface_hub

    settings.local_model = CYBER_FROST_REPO
    (settings.root / "fake-key").write_text("fixture")
    settings.options.update(LAMBDA_SSH_KEY_NAME="name", SSH_PRIVATE_KEY="fake-key")
    root = Path(__file__).resolve().parents[1]
    (settings.root / "config/locks/serving.lock.json").write_bytes((root / "config/locks/serving.lock.json").read_bytes())
    config = settings.root / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"num_attention_heads": 24, "max_position_embeddings": 262144},
            }
        )
    )
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=[
            SimpleNamespace(
                rfilename="model.safetensors",
                size=360_000_192_888,
                lfs=SimpleNamespace(sha256="b" * 64),
            ),
            SimpleNamespace(rfilename="config.json", size=config.stat().st_size, lfs=None),
            SimpleNamespace(rfilename="chat_template.jinja", size=200, lfs=None),
            SimpleNamespace(rfilename="DEPLOYMENT/LAUNCH.sh", size=100, lfs=None),
        ],
    )
    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(model_info=lambda *a, **kw: info)
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(config))
    return GPUManager(settings, State())


def test_inspection_pins_weights_runtime_and_serving_context(blackfrost):
    spec = blackfrost.inspect_model()
    assert spec.required_gib == 480
    assert spec.weight_bytes == 360_000_192_888
    assert spec.revision == "a" * 40
    assert spec.context_window == 32768
    assert spec.tool_parser == "qwen3_coder" and spec.reasoning_parser == "qwen3"
    assert len(spec.download_files) == 3
    blackfrost.settings.options["VLLM_IMAGE"] = "old-incompatible-image"
    assert "@sha256:" in blackfrost.serving_image(spec)
    profile = model_catalog(spec.repo, ContextBudget(spec.context_window), "instructions")
    assert profile["models"][0]["context_window"] == 32768
    assert profile["models"][0]["input_modalities"] == ["text"]


@pytest.mark.parametrize("memory", [24, 40, 48, 80, 180])
def test_small_instance_rejected_without_lambda_request(blackfrost, memory):
    with pytest.raises(ValueError, match="too small"):
        blackfrost.plan(memory)
    assert blackfrost._client is None


def offer(family, per_gpu, memory=1800, disk=22000, gpu_count=8):
    return {
        "instance_type": {
            "name": f"gpu_{gpu_count}x_" + family.lower() + "_sxm",
            "gpu_description": f"NVIDIA {family} {per_gpu}GB",
            "architecture": "x86_64",
            "specs": {"gpus": gpu_count, "memory_gib": memory, "storage_gib": disk},
            "price_cents_per_hour": 100,
        },
        "regions_with_capacity_available": [{"name": "test-region"}],
    }


@pytest.mark.parametrize(
    "family,per_gpu,total,gpu_count",
    [("H100", 80, 640, 8), ("B200", 180, 720, 4), ("B200", 180, 1440, 8)],
)
def test_supported_cluster_planning_still_requires_approval(
    blackfrost, family, per_gpu, total, gpu_count
):
    blackfrost._client = SimpleNamespace(
        catalog=lambda: {"offer": offer(family, per_gpu, gpu_count=gpu_count)}
    )
    plan = blackfrost.plan(total)
    assert plan.gpu_count == gpu_count and plan.total_vram_gib == total
    with pytest.raises(PermissionError):
        blackfrost.launch(plan)
    assert not blackfrost.state.instance_id
    blackfrost.state.instance_type = plan.instance_type
    blackfrost.state.gpu_count = plan.gpu_count
    blackfrost.validate_loaded_hardware(plan.model)


@pytest.mark.parametrize(
    "row", [offer("A100", 80), offer("H100", 80, memory=256), offer("H100", 80, disk=256)]
)
def test_memory_alone_does_not_qualify_incompatible_hardware(blackfrost, row):
    blackfrost._client = SimpleNamespace(catalog=lambda: {"offer": row})
    with pytest.raises(ValueError, match="No available multi-GPU"):
        blackfrost.plan(640)


async def test_unsupported_existing_instance_rejected_before_deployment(blackfrost):
    blackfrost.state.gpu_status = GPUStatus.RUNNING
    blackfrost.state.total_vram_gib = 640
    blackfrost.state.gpu_count = 8
    blackfrost.state.instance_type = "gpu_8x_a100_sxm4"
    with pytest.raises(ValueError, match="incompatible"):
        await blackfrost.deploy()
    assert blackfrost._client is None


@pytest.mark.parametrize("tool_ok", [False, True])
@pytest.mark.parametrize("repo", [CYBER_FROST_REPO, MOG_SEC_REPO])
async def test_ready_requires_parsed_tool_call_without_executing_it(
    blackfrost, monkeypatch, tool_ok, repo
):
    blackfrost.settings.local_model = repo
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{
                "id": repo, "max_model_len": blackfrost.settings.local_context_window(),
            }]})
        payload = json.loads(request.content)
        message = {"content": "OK"}
        if "tools" in payload and tool_ok:
            message = {
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "readiness_probe",
                            "arguments": '{"value":"OK"}',
                        },
                    }
                ]
            }
        return httpx.Response(200, json={"choices": [{"message": message}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: client(**kw, transport=httpx.MockTransport(handler))
    )
    blackfrost.tunnel = SimpleNamespace(returncode=None)
    blackfrost.state.gpu_status = GPUStatus.RUNNING
    if tool_ok:
        assert await blackfrost.check_ready()
        assert blackfrost.state.active_provider == "local"
    else:
        with pytest.raises(RuntimeError, match="coding-tool parser check failed"):
            await blackfrost.check_ready()
        assert not blackfrost.state.model_ready and blackfrost.state.active_provider == "api"
    assert len(requests) == 4  # Health, identity, text, then one tool request; no retries.


def test_profile_notice_and_default_selection_are_separate():
    from velocicyber.config import Settings

    settings = Settings.load(Path(__file__).resolve().parents[1])
    assert not settings.local_model
    assert "example-org/coding-model" not in settings.gpu_model_choices
    assert CYBER_FROST_REPO in settings.gpu_model_choices
    label = model_option_label(CYBER_FROST_REPO)
    assert "360 GB" in label and "720 GB (4× B200)" in label and "Projected" in label
    assert GPU_PROFILES[CYBER_FROST_REPO].context_window == 32768
