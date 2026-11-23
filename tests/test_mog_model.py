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
                    "rope_parameters": {"rope_type": "yarn", "factor": 4.0,
                                        "original_max_position_embeddings": 262144},
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


@pytest.mark.parametrize("variant,window,floor", [("64k", 65536, 72), ("1m", 1000000, 160)])
def test_mog_inspection_preserves_checkpoint_but_caps_serving_context(mog, variant, window, floor):
    mog.settings.options["MOG_CONTEXT_PROFILE"] = variant
    before = (mog.settings.root / "config.json").read_bytes()
    spec = mog.inspect_model()
    assert spec.repo == MOG_SEC_REPO and spec.revision == "a" * 40
    assert spec.backend == "vllm" and spec.weight_bytes == 55_563_006_776
    assert spec.required_gib == floor and spec.context_window == window
    assert spec.profile_variant == variant
    assert spec.tool_parser == "qwen3_coder" and spec.reasoning_parser == "qwen3"
    assert len(spec.download_files) == 3
    assert (mog.settings.root / "config.json").read_bytes() == before
    catalog = model_catalog(spec.repo, ContextBudget(spec.context_window), "system")
    assert catalog["models"][0]["context_window"] == window


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


@pytest.mark.parametrize("variant,window,family,memory", [
    ("64k", 65536, "h100", 80), ("1m", 1000000, "b200", 180),
])
async def test_mog_deployment_transmits_manifest_and_launches_selected_context(
    mog, monkeypatch, variant, window, family, memory
):
    mog.settings.options["MOG_CONTEXT_PROFILE"] = variant
    mog.settings.options["LOCAL_MODEL_CONTEXT_WINDOWS"] = json.dumps({MOG_SEC_REPO: 1000000})
    mog.state.gpu_status = GPUStatus.RUNNING
    mog.state.instance_type = f"gpu_1x_{family}"
    mog.state.instance_ip = "192.0.2.1"
    mog.state.gpu_count = 1
    mog.state.total_vram_gib = memory
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
    assert captured["context_window"] == window and captured["max_num_seqs"] == 1
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
    assert launch[launch.index("--max-model-len") + 1] == str(window)
    if variant == "1m":
        assert launch[launch.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
        assert launch[launch.index("--max-num-batched-tokens") + 1] == "2048"
        assert "--enable-chunked-prefill" in launch and "--enable-prefix-caching" in launch
        assert "--speculative-config" not in launch
    else:
        assert "--kv-cache-dtype" not in launch
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
    assert "Harness context: 64K" in label and "new 64K limit" in label
    assert "Projected: 10–20 min fresh · 2–4 min cached" in label


def test_mog_profile_switch_invalidates_inspection_and_stale_launch_plan(mog):
    mog._client = SimpleNamespace(catalog=lambda: {"offer": offer("H100", 80)})
    plan = mog.plan(80)
    assert plan.model.profile_variant == "64k"
    mog.settings.options["MOG_CONTEXT_PROFILE"] = "1m"
    with pytest.raises(ValueError, match="context profile changed"):
        mog.launch(plan, approved=True)
    assert not mog.state.launch_uncertain and not mog.state.instance_id
    changed = mog.inspect_model()
    assert changed.context_window == 1000000 and changed.required_gib == 160
    assert changed is not plan.model
    assert changed.revision == plan.model.revision and changed.download_files == plan.model.download_files


def test_mog_1m_rejects_80gb_before_request_and_accepts_b200_plan(mog):
    mog.settings.options["MOG_CONTEXT_PROFILE"] = "1m"
    with pytest.raises(ValueError, match="160 GiB"):
        mog.plan(80)
    assert mog._client is None
    mog._client = SimpleNamespace(catalog=lambda: {"offer": offer("B200", 180, ram=360)})
    plan = mog.plan(180)
    assert plan.model.context_window == 1000000 and plan.gpu_count == 1
    with pytest.raises(PermissionError):
        mog.launch(plan)
    assert not mog.state.instance_id


@pytest.mark.parametrize("field,value", [("max_position_embeddings", 262144),
                                         ("rope_parameters", {"rope_type": "default"})])
def test_mog_1m_checks_checkpoint_context_configuration_before_cloud(mog, field, value):
    mog.settings.options["MOG_CONTEXT_PROFILE"] = "1m"
    path = mog.settings.root / "config.json"
    config = json.loads(path.read_text())
    config["text_config"][field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="checkpoint|YaRN"):
        mog.plan(180)
    assert mog._client is None


def test_invalid_mog_variant_fails_without_provider_access(mog):
    mog.settings.options["MOG_CONTEXT_PROFILE"] = "invalid"
    with pytest.raises(ValueError, match="64k or 1m"):
        mog.plan(180)
    assert mog._client is None


@pytest.mark.parametrize("variant,window,compact,timeout", [
    ("64k", 65536, 49152, 600), ("1m", 1000000, 750000, 3600),
])
def test_mog_engine_uses_selected_budget_and_timeout(mog, variant, window, compact, timeout):
    from velocicyber.engine.runner import Engine

    mog.settings.options.update(MOG_CONTEXT_PROFILE=variant,
                               LOCAL_MODEL_CONTEXT_WINDOWS=json.dumps({MOG_SEC_REPO: 32768}))
    mog.state.active_provider = "local"
    engine = Engine(mog.settings, mog.state)
    assert engine.context_budget().window == window
    assert engine.context_budget().compact_at == compact
    assert engine.response_timeout() == timeout
    mog.state.active_provider = "api"
    assert engine.response_timeout() == 600
    mog.state.active_provider = "pi"
    assert engine.response_timeout() == 1800


@pytest.mark.parametrize("variant,reported,ready", [
    ("1m", 65536, False), ("64k", 1000000, False), ("64k", 32768, False),
    ("1m", None, False), ("1m", 1000000, True), ("64k", 65536, True),
])
async def test_mog_readiness_requires_actual_selected_context(mog, monkeypatch, variant, reported, ready):
    import httpx

    # Include a stale inspection cache from the other variant.
    mog.settings.options["MOG_CONTEXT_PROFILE"] = "64k" if variant == "1m" else "1m"
    mog.inspect_model()
    mog.settings.options["MOG_CONTEXT_PROFILE"] = variant
    mog.tunnel = SimpleNamespace(returncode=None)
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MOG_SEC_REPO, "max_model_len": reported}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(**kw, transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(mog, "check_tools", AsyncMock())
    if ready:
        assert await mog.check_ready()
        assert mog.state.model_context_window == reported
        mog.check_tools.assert_awaited_once()
    else:
        with pytest.raises(RuntimeError, match="Select MOG in /local & cloud models to reload"):
            await mog.check_ready()
        assert not mog.state.model_ready
        assert mog.state.model_context_window == 0
        assert calls == ["/health", "/v1/models"]
        mog.check_tools.assert_not_awaited()


async def test_mog_1m_menu_details_cancel_then_confirm_without_launch(mog):
    from velocicyber.ui.app import VelocicyberApp

    mog.settings.options["LOCAL_MODEL_OPTIONS"] = MOG_SEC_REPO
    mog.settings.local_model = ""
    app = VelocicyberApp(mog.settings)
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.press("slash", *"local", "enter", *"m.o.g.", "enter")
        assert app.screen.id == "mog-context"
        await pilot.press(*"1 million", "enter", *"more", "enter")
        assert "160 GiB" in app.screen.details and "FP8" in app.screen.details
        assert "15–35 min" in app.screen.details and "750K" in app.screen.details
        assert "unvalidated" in app.screen.details and "64K" not in app.screen.heading
        assert not mog.settings.local_model
        assert mog.settings.options.get("MOG_CONTEXT_PROFILE", "64k") == "64k"
        await pilot.press("escape")
        assert not mog.settings.local_model and app.gpu._client is None
        app.review_local_model(MOG_SEC_REPO)
        await pilot.pause()
        await pilot.press(*"1m", "enter", *"confirm", "enter")
        assert mog.settings.local_model == MOG_SEC_REPO
        assert mog.settings.options["MOG_CONTEXT_PROFILE"] == "1m"
        assert not app.state.model_ready and app.state.active_provider == "api"
        assert app.gpu._client is None and app.conversation
        await pilot.press("slash", *"gpu", "enter", *"on", "enter")
        assert "MOG 1M" in app.screen.heading
        assert "160 GiB" in app.screen.heading
        await pilot.press("escape")
        assert app.gpu._client is None and not app.state.instance_id


@pytest.mark.parametrize("previous,chosen,memory,loads", [
    ("64k", "1m", 80, False), ("64k", "1m", 180, True), ("1m", "64k", 180, True),
])
async def test_mog_switch_reloads_or_preserves_working_profile_on_small_gpu(
    mog, monkeypatch, previous, chosen, memory, loads
):
    from velocicyber.models.gpu_profiles import MOG_CONTEXT_PROFILES
    from velocicyber.ui.app import VelocicyberApp

    mog.settings.options["MOG_CONTEXT_PROFILE"] = previous
    app = VelocicyberApp(mog.settings)
    async with app.run_test() as pilot:
        app.state.gpu_status = GPUStatus.RUNNING
        app.state.instance_type = "gpu_1x_b200" if memory == 180 else "gpu_1x_h100"
        app.state.gpu_count = 1
        app.state.total_vram_gib = memory
        app.state.model_ready = True
        app.state.model_context_window = MOG_CONTEXT_PROFILES[previous].context_window
        app.state.model_context_model = MOG_SEC_REPO
        app.state.active_provider = "local"
        monkeypatch.setattr(app.gpu, "refresh", lambda: "active")
        monkeypatch.setattr(app, "load_model", AsyncMock())
        app.select_local_model(MOG_SEC_REPO, mog_context=chosen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        if loads:
            app.load_model.assert_awaited_once()
            assert mog.settings.options["MOG_CONTEXT_PROFILE"] == chosen
            assert not app.state.model_ready
        else:
            app.load_model.assert_not_awaited()
            assert mog.settings.options["MOG_CONTEXT_PROFILE"] == previous
            assert app.state.model_ready and app.state.active_provider == "local"
        assert app.gpu._client is None


@pytest.mark.parametrize("window,label", [(65536, "64K context"), (1000000, "1M context")])
def test_mog_dashboard_shows_served_context(mog, window, label):
    from velocicyber.ui.dashboard import dashboard_text

    mog.state.gpu_status = GPUStatus.RUNNING
    mog.state.model_ready = True
    mog.state.model_context_window = window
    mog.state.model_context_model = MOG_SEC_REPO
    assert "Ready for coding · " + label in dashboard_text(mog.settings, mog.state).plain
