import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from velocicyber.config import Settings
from velocicyber.models.gpu_profiles import GPU_PROFILES, ORCACYBER_REPO
from velocicyber.providers.lambda_cloud import GPUManager
from velocicyber.state import GPUStatus, State
from velocicyber.ui.app import VelocicyberApp

ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = "OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf"


@pytest.fixture
def orca(settings, monkeypatch):
    import huggingface_hub

    import velocicyber.providers.gguf

    settings.local_model = ORCACYBER_REPO
    settings.options.update(
        LAMBDA_SSH_KEY_NAME="fixture",
        SSH_PRIVATE_KEY="fake-key",
        # Another model's generic override must not select the small/draft GGUF.
        GGUF_FILE="OrcaCyber-Nano-1.0-AI-Defense-Mix-13G.gguf",
    )
    (settings.root / "fake-key").write_text("fixture")
    (settings.root / "config/locks/serving.lock.json").write_bytes((ROOT / "config/locks/serving.lock.json").read_bytes())
    (settings.root / "scripts/runtime").mkdir(parents=True)
    (settings.root / "scripts/runtime/serve_model.py").write_bytes(
        (ROOT / "scripts/runtime/serve_model.py").read_bytes()
    )
    files = [
        ("OrcaCyber-Nano-1.0-AI-Defense-DFlash2-Q8_0.gguf", 2_056_414_112),
        ("OrcaCyber-Nano-1.0-AI-Defense-Mix-13G.gguf", 12_993_304_352),
        (MAIN_FILE, 15_695_148_832),
        ("mmproj-OrcaCyber-Nano-1.0-AI-Defense-F16.gguf", 927_606_848),
    ]
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=[
            SimpleNamespace(rfilename=name, size=size, lfs=SimpleNamespace(sha256="b" * 64))
            for name, size in files
        ],
    )

    def model_info(repo, files_metadata):
        assert repo == ORCACYBER_REPO and files_metadata
        return info

    def header(repo, revision, filename, token):
        assert (repo, revision, filename) == (ORCACYBER_REPO, info.sha, MAIN_FILE)
        return {"general.architecture": "qwen35", "qwen35.context_length": 262144}

    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(model_info=model_info)
    )
    monkeypatch.setattr(velocicyber.providers.gguf, "inspect_header", header)
    return GPUManager(settings, State()), info


def test_selects_large_gguf_with_single_agent_budget_and_own_runtime(orca):
    manager, _ = orca
    spec = manager.inspect_model()
    assert spec.filename == MAIN_FILE and spec.weight_bytes == 15_695_148_832
    assert spec.backend == "llama.cpp" and spec.architecture == "qwen35"
    assert spec.context_window == 32768 and spec.required_gib == 24
    lock = json.loads((manager.settings.root / "config/locks/serving.lock.json").read_text())
    assert manager.serving_image(spec) == lock["model_images"][ORCACYBER_REPO]["image"]
    assert manager.serving_image(spec) != lock["llama_cpp_image"]
    assert not manager._client and not manager.state.instance_id


@pytest.mark.parametrize("problem", ["missing_variant", "wrong_architecture", "missing_checksum"])
def test_invalid_artifact_stops_before_lambda_access(orca, monkeypatch, problem):
    import velocicyber.providers.gguf

    manager, info = orca
    if problem == "missing_variant":
        info.siblings = [file for file in info.siblings if file.rfilename != MAIN_FILE]
    elif problem == "missing_checksum":
        next(file for file in info.siblings if file.rfilename == MAIN_FILE).lfs = None
    else:
        monkeypatch.setattr(
            velocicyber.providers.gguf, "inspect_header", lambda *a: {"general.architecture": "dflash"}
        )
    with pytest.raises(ValueError):
        manager.plan(24)
    assert manager._client is None


@pytest.mark.parametrize("family,memory", [("A10", 24), ("A6000", 48)])
def test_single_gpu_plan_still_requires_launch_approval(orca, family, memory):
    manager, _ = orca
    manager._client = SimpleNamespace(catalog=lambda: {
        "fixture": {
            "instance_type": {
                "name": "gpu_1x_" + family.lower(),
                "gpu_description": f"NVIDIA {family} {memory}GB",
                "specs": {"gpus": 1, "memory_gib": 100, "storage_gib": 512},
                "price_cents_per_hour": 109,
            },
            "regions_with_capacity_available": [{"name": "test-region"}],
        },
    })
    plan = manager.plan(memory)
    assert plan.model.filename == MAIN_FILE and plan.gpu_count == 1
    with pytest.raises(PermissionError):
        manager.launch(plan)
    assert not manager.state.instance_id


async def test_deployment_caps_context_and_transmits_only_large_gguf(orca, monkeypatch):
    manager, _ = orca
    manager.settings.options["LOCAL_MODEL_CONTEXT_WINDOWS"] = json.dumps({ORCACYBER_REPO: 262144})
    manager.state.gpu_status = GPUStatus.RUNNING
    manager.state.instance_type = "gpu_1x_a10"
    manager.state.instance_ip = "192.0.2.1"
    manager.state.gpu_count = 1
    manager.state.total_vram_gib = 24
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
    monkeypatch.setattr(manager, "connect", AsyncMock())
    await manager.deploy()
    assert captured["filename"] == MAIN_FILE and captured["sha256"] == "b" * 64
    assert captured["revision"] == "a" * 40 and captured["context_window"] == 32768
    assert captured["download_files"] == [] and captured["backend"] == "llama.cpp"
    assert captured["extra_arguments"] == list(GPU_PROFILES[ORCACYBER_REPO].arguments)

    spec = importlib.util.spec_from_file_location("orca_serve_model", ROOT / "scripts/runtime/serve_model.py")
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    commands, downloads = [], []
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: manager.settings.root))
    monkeypatch.setattr(
        remote.subprocess, "run",
        lambda args, **kw: commands.append(args) or SimpleNamespace(returncode=0, stdout="true\n"),
    )

    def download(config, folder):
        downloads.append(config["filename"])
        return folder / "weights" / config["revision"] / config["filename"]

    monkeypatch.setattr(remote, "download_gguf", download)
    remote.serve(captured)
    assert downloads == [MAIN_FILE]
    launch = commands[-1]
    assert launch[launch.index("--alias") + 1] == ORCACYBER_REPO
    assert launch[launch.index("--ctx-size") + 1] == "32768"
    assert launch[launch.index("--parallel") + 1] == "1"
    assert launch[launch.index("--temp") + 1] == "1.0" and "--jinja" in launch
    assert "--mmproj" not in launch and "--spec-type" not in launch
    assert not manager.state.model_ready and manager._client is None


async def test_local_picker_details_and_confirmation_remain_offline(settings):
    configured = Settings.load(ROOT)
    assert ORCACYBER_REPO in configured.gpu_model_choices
    assert ORCACYBER_REPO not in configured.pi_model_choices and not configured.local_model
    settings.options["LOCAL_MODEL_OPTIONS"] = ORCACYBER_REPO
    previous = settings.local_model
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("backslash", *"local", "enter", *"orcacyber", "enter")
        assert app.screen.id == "local-model-choice" and settings.local_model == previous
        await pilot.press(*"more", "enter")
        assert "Mix-15.7G" in app.screen.details and "24 GB" in app.screen.details
        assert "32 GiB host RAM" in app.screen.details and "15–35 min" in app.screen.details
        assert "Projected: 25–50 min fresh" in app.screen.details
        assert app.gpu._client is None and settings.local_model == previous
        await pilot.press(*"confirm", "enter")
        assert settings.local_model == ORCACYBER_REPO
        assert app.state.gpu_status == GPUStatus.STOPPED and not app.state.model_ready
        assert app.gpu._client is None and not app.state.instance_id
