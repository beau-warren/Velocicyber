import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from velocicyber.config import Settings
from velocicyber.engine.context import ContextBudget
from velocicyber.models.catalog import model_catalog
from velocicyber.models.gpu_profiles import MOG_SEC_REPO
from velocicyber.models.startup import model_option_label
from velocicyber.providers.lambda_cloud import GPUManager
from velocicyber.state import GPUStatus, State

module_spec = importlib.util.spec_from_file_location(
    "serve_model", Path(__file__).resolve().parents[1] / "scripts/runtime/serve_model.py"
)
serve_model = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(serve_model)


@pytest.fixture
def mog(settings, monkeypatch):
    import huggingface_hub

    settings.local_model = MOG_SEC_REPO
    settings.options.update(LAMBDA_SSH_KEY_NAME="name", SSH_PRIVATE_KEY="fake-key")
    (settings.root / "fake-key").write_text("fixture")
    root = Path(__file__).resolve().parents[1]
    (settings.root / "config/locks/serving.lock.json").write_bytes((root / "config/locks/serving.lock.json").read_bytes())
    (settings.root / "scripts/runtime").mkdir(parents=True)
    (settings.root / "scripts/runtime/serve_model.py").write_bytes(
        (root / "scripts/runtime/serve_model.py").read_bytes()
    )
    config = settings.root / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "num_attention_heads": 24,
                    "max_position_embeddings": 1000000,
                    "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
                },
            }
        )
    )
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=[
            SimpleNamespace(
                rfilename="model.safetensors",
                size=55_563_006_776,
                lfs=SimpleNamespace(sha256="b" * 64),
            ),
            SimpleNamespace(rfilename="config.json", size=config.stat().st_size, lfs=None),
            SimpleNamespace(rfilename="chat_template.jinja", size=200, lfs=None),
            SimpleNamespace(rfilename="deployment-kit/overlay/model.py", size=100, lfs=None),
        ],
    )

    def model_info(repo, files_metadata):
        assert repo == MOG_SEC_REPO and files_metadata
        return info

    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(model_info=model_info)
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(config))
    return GPUManager(settings, State())


def test_mog_inspection_preserves_checkpoint_but_caps_serving_context(mog):
    before = (mog.settings.root / "config.json").read_bytes()
    spec = mog.inspect_model()
    assert spec.repo == MOG_SEC_REPO and spec.revision == "a" * 40
    assert spec.backend == "vllm" and spec.weight_bytes == 55_563_006_776
    assert spec.required_gib == 72 and spec.context_window == 32768
    assert spec.tool_parser == "qwen3_coder" and spec.reasoning_parser == "qwen3"
    assert len(spec.download_files) == 3
    assert (mog.settings.root / "config.json").read_bytes() == before
    catalog = model_catalog(spec.repo, ContextBudget(spec.context_window), "system")
    assert catalog["models"][0]["context_window"] == 32768


@pytest.mark.parametrize("memory", [24, 40, 48])
def test_mog_rejects_small_gpu_before_lambda_request(mog, memory):
    with pytest.raises(ValueError, match="too small"):
        mog.plan(memory)
    assert mog._client is None


def offer(family, vram, ram=225, disk=1024):
    return {
        "instance_type": {
            "name": f"gpu_1x_{family.lower()}",
            "gpu_description": f"NVIDIA {family} {vram}GB",
            "architecture": "x86_64",
            "specs": {"gpus": 1, "memory_gib": ram, "storage_gib": disk},
            "price_cents_per_hour": 100,
        },
        "regions_with_capacity_available": [{"name": "test-region"}],
    }


@pytest.mark.parametrize("family,memory", [("H100", 80), ("B200", 180)])
def test_mog_single_gpu_plan_requires_consent(mog, family, memory):
    mog._client = SimpleNamespace(catalog=lambda: {"offer": offer(family, memory)})
    plan = mog.plan(memory)
    assert plan.gpu_count == 1 and plan.total_vram_gib == memory
    with pytest.raises(PermissionError):
        mog.launch(plan)
    assert not mog.state.instance_id


@pytest.mark.parametrize(
    "row", [offer("A100", 80), offer("H100", 80, ram=64), offer("H100", 80, disk=128)]
)
def test_mog_rejects_incompatible_specs_with_its_own_requirements(mog, row):
    mog._client = SimpleNamespace(catalog=lambda: {"offer": row})
    with pytest.raises(ValueError, match="128 GiB host RAM and 256 GiB disk"):
        mog.plan(80)


async def test_mog_deployment_transmits_manifest_and_launches_32k_bf16(mog, monkeypatch):
    mog.settings.options["LOCAL_MODEL_CONTEXT_WINDOWS"] = json.dumps({MOG_SEC_REPO: 1000000})
    mog.state.gpu_status = GPUStatus.RUNNING
    mog.state.instance_type = "gpu_1x_h100_pcie"
    mog.state.instance_ip = "192.0.2.1"
    mog.state.gpu_count = 1
    mog.state.total_vram_gib = 80
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

        async def drain(self):
            pass

        def close(self):
            pass

        async def wait(self):
            return 0

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=Process()))
    monkeypatch.setattr(mog, "connect", AsyncMock())
    await mog.deploy()
    assert captured["repo"] == MOG_SEC_REPO and captured["gpus"] == 1
    assert captured["context_window"] == 32768 and captured["max_num_seqs"] == 1
    assert len(captured["download_files"]) == 3
    assert "@sha256:" in captured["image"]

    commands = []
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: mog.settings.root))
    monkeypatch.setattr(
        serve_model.subprocess,
        "run",
        lambda args, **kw: commands.append(args) or SimpleNamespace(returncode=0, stdout="true\n"),
    )
    monkeypatch.setattr(serve_model, "download_safetensors", lambda *args: None)
    serve_model.serve(captured)
    launch = commands[-1]
    assert launch[launch.index("--tensor-parallel-size") + 1] == "1"
    assert launch[launch.index("--max-model-len") + 1] == "32768"
    assert launch[launch.index("--mamba-ssm-cache-dtype") + 1] == "float32"
    assert launch[launch.index("--dtype") + 1] == "bfloat16"
    assert launch[launch.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert launch[launch.index("--served-model-name") + 1] == MOG_SEC_REPO
    assert "HF_HUB_OFFLINE=1" in launch and "--language-model-only" in launch
    assert mog._client is None and not mog.state.model_ready


def test_mog_is_gpu_only_with_clear_context_and_startup_notice():
    settings = Settings.load(Path(__file__).resolve().parents[1])
    assert MOG_SEC_REPO in settings.gpu_model_choices
    assert not settings.local_model
    assert "example-org/coding-model" not in settings.gpu_model_choices
    assert settings.pi_model != MOG_SEC_REPO
    label = model_option_label(MOG_SEC_REPO)
    assert "Single 80 GB (H100)" in label and "55.6 GB" in label
    assert "Harness context: 32K" in label and "1M context needs a separate" in label
    assert "Estimated: 10–20 min fresh · 2–4 min cached" in label
