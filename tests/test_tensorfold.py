import ast
import asyncio
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Input, OptionList

from velocicyber.config import Settings
from velocicyber.models.gpu_profiles import GLM53_EXL3_REPO, GPU_PROFILES
from velocicyber.models.startup import model_details
from velocicyber.providers.lambda_cloud import GPUManager
from velocicyber.state import GPUStatus, State
from velocicyber.ui.app import VelocicyberApp

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/runtime" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serving = load_script("serve_model")
runtime = load_script("tensorfold_runtime")


@pytest.fixture
def glm(settings, monkeypatch):
    import huggingface_hub

    settings.local_model = GLM53_EXL3_REPO
    settings.options.update(LAMBDA_SSH_KEY_NAME="fixture", SSH_PRIVATE_KEY="key")
    (settings.root / "key").write_text("fixture")
    (settings.root / "config/locks/serving.lock.json").write_bytes((ROOT / "config/locks/serving.lock.json").read_bytes())
    (settings.root / "scripts/runtime").mkdir(parents=True)
    for script in ("serve_model.py", "tensorfold_runtime.py"):
        (settings.root / "scripts/runtime" / script).write_bytes((ROOT / "scripts/runtime" / script).read_bytes())
    config = settings.root / "config.json"
    config.write_text(json.dumps({
        "model_type": "glm5_next",
        "text_config": {"max_position_embeddings": 1048576, "num_attention_heads": 64},
        "quantization_config": {
            "quant_method": "exl3", "bits": 4, "codebook": "mcg",
            "scope": "glm53_routed_experts_only",
        },
    }))
    info = SimpleNamespace(sha="a" * 40, siblings=[
        SimpleNamespace(rfilename="model.safetensors", size=175_642_267_944,
                        lfs=SimpleNamespace(sha256="b" * 64)),
        SimpleNamespace(rfilename="config.json", size=config.stat().st_size, lfs=None),
        SimpleNamespace(rfilename="tokenizer.json", size=1234, lfs=None),
        SimpleNamespace(rfilename="chat_template.jinja", size=123, lfs=None),
    ])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(
        model_info=lambda *a, **kw: info,
    ))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(config))
    return GPUManager(settings, State())


def offer(family="B200", gpus=2, per_gpu=180, ram=720, disk=5500):
    return {
        "instance_type": {
            "name": f"gpu_{gpus}x_{family.lower()}", "gpu_description": f"{family} {per_gpu}GB",
            "architecture": "x86_64", "price_cents_per_hour": 1378,
            "specs": {"gpus": gpus, "memory_gib": ram, "storage_gib": disk},
        },
        "regions_with_capacity_available": [{"name": "test-region"}],
    }


def test_glm_uses_exl3_backend_with_pinned_weights_and_32k_limit(glm):
    spec = glm.inspect_model()
    assert spec.backend == "tensorfold" and spec.required_gib == 256
    assert spec.context_window == 32768 and spec.weight_bytes == 175_642_267_944
    assert len(spec.download_files) == 4 and spec.revision == "a" * 40
    assert spec.tool_parser == "tensorfold"
    assert glm.serving_image(spec).startswith("nvcr.io/nvidia/pytorch@sha256:")
    glm._client = SimpleNamespace(catalog=lambda: {"offer": offer()})
    plan = glm.plan(360)
    assert plan.total_vram_gib == 360 and plan.gpu_count == 2 and plan.hourly_cents == 1378
    with pytest.raises(PermissionError):
        glm.launch(plan)


@pytest.mark.parametrize("size", [24, 40, 48, 80, 180])
def test_single_gpu_rejected_before_capacity_or_launch(glm, size):
    with pytest.raises(ValueError, match="too small"):
        glm.plan(size)
    assert glm._client is None


@pytest.mark.parametrize("kind,memory", [
    (offer("H100", 8, 80), 640), (offer(gpus=4), 720),
    (offer(gpus=1, per_gpu=360), 360),
    (offer(ram=128), 360), (offer(disk=128), 360),
])
def test_only_supported_two_gpu_hardware_is_admitted(glm, kind, memory):
    glm._client = SimpleNamespace(catalog=lambda: {"offer": kind})
    with pytest.raises(ValueError, match="No available multi-GPU"):
        glm.plan(memory)
    assert not glm.state.instance_id


def test_wrong_exl3_layout_rejected_before_launch(glm):
    p = glm.settings.root / "config.json"
    cfg = json.loads(p.read_text())
    cfg["quantization_config"]["bits"] = 3
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="4-bit MCG EXL3"):
        glm.inspect_model()
    assert glm._client is None


async def test_deploy_passes_source_pin_and_never_sends_exl3_to_vllm(glm, monkeypatch):
    glm.state.gpu_status = GPUStatus.RUNNING
    glm.state.instance_type = "gpu_2x_b200_sxm6"
    glm.state.gpu_count = 2
    glm.state.total_vram_gib = 360
    glm.state.instance_ip = "192.0.2.1"
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
            call = tree.body[-1].value
            assert {x.arg for x in call.keywords} == {"tensorfold_prepare", "tensorfold_options"}
            captured.update(json.loads(ast.literal_eval(call.args[0].args[0])))

        async def drain(self):
            pass

        def close(self):
            pass

        async def wait(self):
            return 0

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=Process()))
    monkeypatch.setattr(glm, "connect", AsyncMock())
    await glm.deploy()
    assert captured["backend"] == "tensorfold" and captured["context_window"] == 32768
    assert len(captured["tensorfold_runtime"]["source_sha256"]) == 64
    commands, order = [], []
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: glm.settings.root))

    def run(args, **kw):
        commands.append(args)
        if "rm" in args:
            order.append("replace")
        return SimpleNamespace(returncode=0, stdout="true\n")

    monkeypatch.setattr(serving.subprocess, "run", run)
    monkeypatch.setattr(serving, "download_safetensors", lambda *a: order.append("download"))
    monkeypatch.setattr(serving, "check_runtime", lambda *a: pytest.fail("vLLM preflight called"))

    def prepare(*args):
        order.append("preflight")
        return "velocicyber-tensorfold:fixture"

    serving.serve(captured, tensorfold_prepare=prepare, tensorfold_options=runtime.tensorfold_server_options)
    launch = commands[-1]
    assert order == ["preflight", "download", "replace"]
    assert "127.0.0.1:8000:8000" in launch and "--tensor-parallel-size" not in launch
    assert launch[launch.index("--restart") + 1] == "no"
    assert "--env-file" not in launch and "HF_HUB_OFFLINE=1" in launch
    assert "/opt/velocicyber/tensorfold_server.py" in launch
    assert not (glm.settings.root / ".velocicyber/serve.env").exists()


def test_broken_runtime_stops_before_downloading_or_replacing(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    commands = []
    monkeypatch.setattr(serving.subprocess, "run", lambda args, **kw: (
        commands.append(args) or SimpleNamespace(returncode=0, stdout="true\n")
    ))
    monkeypatch.setattr(serving, "download_safetensors", lambda *a: pytest.fail("Downloaded weights"))

    def broken(*args):
        raise RuntimeError("preflight failed")

    cfg = dict(backend="tensorfold", context_window=32768, token="fixture", repo=GLM53_EXL3_REPO,
               revision="a" * 40, gpus=2, tool_parser="tensorfold", reasoning_parser="", image="base")
    with pytest.raises(RuntimeError, match="preflight failed"):
        serving.serve(cfg, tensorfold_prepare=broken, tensorfold_options=runtime.tensorfold_server_options)
    assert all("rm" not in c and "run" not in c for c in commands)
    assert not (tmp_path / ".velocicyber/serve.env").exists()


def test_rank_failure_stops_peer_without_retry(monkeypatch):
    ns = {"__name__": "test_runner"}
    exec(compile(runtime.TENSORFOLD_RUNNER, "tensorfold_server.py", "exec"), ns)
    calls, killed, waited = [], [], []

    class Child:
        def __init__(self, rank):
            self.rank, self.pid = rank, 12300 + rank

        def poll(self):
            return 7 if self.rank == 1 else None

        def wait(self, **kwargs):
            waited.append(self.rank)
            return 0

    def spawn(args, **kwargs):
        calls.append((args, kwargs))
        return Child(int(args[-1]))

    monkeypatch.setattr(ns["sys"], "argv", ["runner", GLM53_EXL3_REPO, "/weights/rev", "32768"])
    monkeypatch.setattr(ns["signal"], "signal", lambda *a: None)
    monkeypatch.setattr(ns["subprocess"], "Popen", spawn)
    monkeypatch.setattr(ns["os"], "killpg", lambda pid, sig: killed.append(pid))
    assert ns["run"]() == 1
    assert len(calls) == 2 and killed == [12300] and waited == [1, 0]
    assert [kw["env"]["CUDA_VISIBLE_DEVICES"] for _, kw in calls] == ["1", "0"]
    assert all("--no-drafts" in cmd and "--no-update-check" in cmd for cmd, _ in calls)


def test_missing_source_pin_blocks_before_lambda_catalog(glm):
    p = glm.settings.root / "config/locks/serving.lock.json"
    lock = json.loads(p.read_text())
    del lock["model_images"][GLM53_EXL3_REPO]["source_sha256"]
    p.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="pinned NVIDIA image"):
        glm.plan(360)
    assert glm._client is None


def test_source_checksum_failure_never_builds_or_starts_container(tmp_path, monkeypatch):
    pin = json.loads((ROOT / "config/locks/serving.lock.json").read_text())["model_images"][GLM53_EXL3_REPO]
    commands = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda cmd, **kw: (
        commands.append(cmd) or SimpleNamespace(returncode=1)
    ))
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad source"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        runtime.prepare_tensorfold({"tensorfold_runtime": pin, "gpus": 2}, tmp_path, ["docker"])
    assert len(commands) == 1 and commands[0][1:3] == ["image", "inspect"]
    assert not list(tmp_path.iterdir())


def test_cached_image_still_checks_both_gpus_without_downloading_source(tmp_path, monkeypatch):
    pin = json.loads((ROOT / "config/locks/serving.lock.json").read_text())["model_images"][GLM53_EXL3_REPO]
    commands = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda cmd, **kw: (
        commands.append(cmd) or SimpleNamespace(returncode=0, stderr="")
    ))
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Network"))
    image = runtime.prepare_tensorfold({"tensorfold_runtime": pin, "gpus": 2}, tmp_path, ["docker"])
    assert image.startswith("velocicyber-tensorfold:") and len(commands) == 2
    check = commands[-1]
    assert "--network" in check and "none" in check and "--env-file" not in check
    assert "torch.cuda.device_count() == 2" in check[-1]
    assert "nvcc" in check[-1]


async def test_glm_menu_confirmation_details_and_360gb_cluster(settings):
    settings.options["LOCAL_MODEL_OPTIONS"] = GLM53_EXL3_REPO
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.press("backslash", *"local", "enter", *"mia", "enter")
        assert app.screen.id == "local-model-choice"
        await pilot.press(*"more", "enter")
        assert "360 GB (2× B200)" in app.screen.details
        assert "Experimental" in app.screen.details and "$13.78/hour" in app.screen.details
        await pilot.press(*"confirm", "enter")
        assert settings.local_model == GLM53_EXL3_REPO and app.gpu._client is None
        app.gpu_selected("on")
        await pilot.pause()
        app.screen.query_one("#menu-filter", Input).value = "cluster 360"
        await pilot.pause()
        assert "360 GB" in str(app.screen.query_one(OptionList).highlighted_option.prompt)
        assert not app.state.instance_id


def test_config_keeps_model_explicit_and_details_do_not_claim_live_validation():
    settings = Settings.load(ROOT)
    assert not settings.local_model and GLM53_EXL3_REPO in settings.gpu_model_choices
    details = model_details(GLM53_EXL3_REPO)
    assert "25–60 min" in details and "Projected" in details
    assert "Lambda validation pending" in details
    assert GPU_PROFILES[GLM53_EXL3_REPO].gpu_counts == (2,)
