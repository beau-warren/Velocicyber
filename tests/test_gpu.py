import ast
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from velocicyber.providers.lambda_cloud import (
    GPUManager,
    LambdaClient,
    LambdaError,
    LaunchPlan,
    ModelSpec,
    total_vram,
)
from velocicyber.state import GPUStatus, State


async def test_empty_selection_blocks_gpu_work_even_with_cached_spec(settings, monkeypatch):
    manager = GPUManager(settings, State(gpu_status=GPUStatus.RUNNING))
    manager.spec = ModelSpec(settings.local_model, "rev", 1024, "qwen3_coder", "", 32)
    settings.local_model = ""

    def unexpected(*args, **kwargs):
        pytest.fail("No selection must stop before contacting a provider")

    monkeypatch.setattr("huggingface_hub.HfApi", unexpected)
    with pytest.raises(ValueError, match="Choose a GPU model"):
        manager.inspect_model()
    with pytest.raises(ValueError, match="Choose a GPU model"):
        manager.plan(80)
    with pytest.raises(ValueError, match="Choose a GPU model"):
        await manager.deploy()
    assert manager._client is None


def test_total_memory_not_host_ram_or_per_gpu():
    assert (
        total_vram({"gpu_description": "NVIDIA A100 80GB", "specs": {"gpus": 2, "memory_gib": 512}})
        == 160
    )
    assert total_vram({"gpu_description": "unknown", "specs": {"gpus": 1, "memory_gib": 80}}) == 0


def test_launch_requires_consent_and_tracks_termination(settings):
    calls = []
    responses = [
        {"instance_ids": ["owned-123"]},
        {"status": "active", "ip": "192.0.2.1", "instance_type": {"specs": {"gpus": 1}}},
        {"terminated_instances": []},
        {"status": "terminating", "instance_type": {"specs": {"gpus": 1}}},
        {"status": "active", "instance_type": {"specs": {"gpus": 1}}},
        {"status": "terminated", "instance_type": {"specs": {"gpus": 1}}},
    ]

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": responses.pop(0)})

    settings.options["LAMBDA_SSH_KEY_NAME"] = "test-key"
    state = State()
    manager = GPUManager(
        settings, state, LambdaClient("fake", transport=httpx.MockTransport(handler))
    )
    plan = LaunchPlan(
        "gpu_1x_a100",
        "us-east-1",
        80,
        1,
        100,
        ModelSpec(settings.local_model, "revision", 54 * 2**30, "hermes", "qwen3", 32),
        time.time(),
    )
    with pytest.raises(PermissionError):
        manager.launch(plan)
    assert not calls
    manager.launch(plan, approved=True)
    assert state.gpu_status == GPUStatus.INITIALIZED and state.instance_id == "owned-123"
    assert manager.refresh() == "active"
    assert state.gpu_status == GPUStatus.RUNNING and not state.model_ready
    with pytest.raises(PermissionError):
        manager.terminate()
    manager.terminate(approved=True)
    assert state.gpu_status == GPUStatus.STOPPING
    assert state.instance_id == "owned-123"  # Do not forget a still-billable resource.
    assert manager.refresh() == "terminating"
    assert state.gpu_status == GPUStatus.STOPPING
    assert manager.refresh() == "active"  # Provider status can briefly lag the request.
    assert state.gpu_status == GPUStatus.STOPPING
    assert manager.refresh() == "terminated"
    assert state.gpu_status == GPUStatus.STOPPED and not state.instance_id
    assert len(calls) == 6


def test_ambiguous_launch_never_retried(settings):
    def handler(request):
        raise httpx.ReadTimeout("timeout", request=request)

    settings.options["LAMBDA_SSH_KEY_NAME"] = "test-key"
    state = State()
    manager = GPUManager(
        settings, state, LambdaClient("fake", transport=httpx.MockTransport(handler))
    )
    plan = LaunchPlan(
        "gpu_1x_a100",
        "us-east-1",
        80,
        1,
        100,
        ModelSpec(settings.local_model, "revision", 1, "hermes", "", 32),
        time.time(),
    )
    with pytest.raises(httpx.ReadTimeout):
        manager.launch(plan, approved=True)
    assert State.restore(settings.runtime / "gpu.json").launch_uncertain
    with pytest.raises(RuntimeError, match="unresolved"):
        manager.launch(plan, approved=True)


@pytest.mark.parametrize(
    "status,code,rejected",
    [
        (400, "instance-operations/launch/insufficient-capacity", True),
        (403, "global/account-inactive", True),
        (429, "global/rate-limit", True),
        (408, "global/timeout", False),
        (503, "provider/internal-unavailable", False),
        (400, "provider/unknown", False),
    ],
)
def test_launch_rejections_clear_state_but_uncertain_errors_keep_retry_block(
    settings, status, code, rejected,
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"code": code, "message": "Fixture failure"}})

    settings.options["LAMBDA_SSH_KEY_NAME"] = "test-key"
    manager = GPUManager(
        settings, State(), LambdaClient("fake", transport=httpx.MockTransport(handler))
    )
    plan = LaunchPlan(
        "gpu_1x_a6000", "test-region", 48, 1, 109,
        ModelSpec(settings.local_model, "revision", 1, "jinja", "auto", 0), time.time(),
    )
    with pytest.raises(LambdaError, match="Fixture failure"):
        manager.launch(plan, approved=True)
    state = State.restore(settings.runtime / "gpu.json")
    assert state.launch_uncertain is not rejected and not state.instance_id
    assert state.instance_type == "gpu_1x_a6000" and state.total_vram_gib == 48
    assert state.region == "test-region" and len(calls) == 1
    if not rejected:
        with pytest.raises(RuntimeError, match="unresolved"):
            manager.launch(plan, approved=True)
        assert len(calls) == 1


def test_lambda_error_retains_reason_and_suggestion_without_secret_or_unrelated_fields(settings):
    secret = "sample-provider-key-123456789"
    settings.secrets["LAMBDA_API"] = secret
    client = LambdaClient(secret, transport=httpx.MockTransport(lambda request: httpx.Response(
        400, json={
            "error": {
                "code": "global/quota-exceeded",
                "message": "Quota exceeded for " + secret,
                "suggestion": "Contact support",
                "debug": "THIS-MUST-NOT-BE-EXPOSED",
            },
            "request": {"authorization": secret},
        },
    )))
    with pytest.raises(LambdaError) as caught:
        client.request("POST", "instance-operations/launch", {})
    error = caught.value
    assert error.code == "global/quota-exceeded"
    assert "Quota exceeded" in str(error) and "Contact support" in str(error)
    assert secret not in str(error) and "THIS-MUST-NOT-BE-EXPOSED" not in str(error)
    manager = GPUManager(settings, State(), client)
    manager.record_error("launch/load", error, memory=48)
    path = settings.runtime / "diagnostics/gpu-errors.jsonl"
    row = json.loads(path.read_text())
    assert row["provider_code"] == "global/quota-exceeded"
    assert row["http_status"] == 400 and row["requested_vram_gib"] == 48
    assert secret not in path.read_text() and path.stat().st_mode & 0o777 == 0o600
    assert not settings.data.exists()


@pytest.mark.parametrize("body", [b"<html>private upstream body</html>", b'[]', b'{"error":"private"}'])
def test_malformed_lambda_error_uses_status_only(body):
    client = LambdaClient("fake", transport=httpx.MockTransport(
        lambda request: httpx.Response(503, content=body)
    ))
    with pytest.raises(LambdaError, match="HTTP 503") as caught:
        client.request("GET", "instances")
    assert not caught.value.code and "private" not in str(caught.value)


def test_model_too_large_blocks_before_lambda_request(settings):
    key = settings.root / "test-key"
    key.write_text("fake")
    settings.options.update(
        LAMBDA_SSH_KEY_NAME="name", LAMBDA_REGION="us-east-1", SSH_PRIVATE_KEY="test-key"
    )
    manager = GPUManager(settings, State())
    manager.spec = ModelSpec(settings.local_model, "revision", 54 * 2**30, "hermes", "qwen3", 32)
    with pytest.raises(ValueError, match="too small"):
        manager.plan(24)
    assert manager._client is None


def test_terminated_instance_removed_by_provider(settings):
    state = State(instance_id="previously-owned", gpu_status=GPUStatus.INITIALIZED)
    client = LambdaClient(
        "fake",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(404, json={"error": "instance-not-found"})
        ),
    )
    manager = GPUManager(settings, state, client)
    assert manager.refresh() == "terminated"
    assert state.gpu_status == GPUStatus.STOPPED
    assert not state.instance_id


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"meta": {"n_ctx": 16384, "n_ctx_train": 262144}}, 16384),
        ({"max_model_len": 65536}, 65536),
    ],
)
async def test_model_not_ready_until_health_and_inference_succeed(
    settings, monkeypatch, metadata, expected
):
    calls = []
    healthy = False

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200 if healthy else 503)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": settings.local_model, **metadata}]})
        assert request.extensions["timeout"]["read"] == 120
        assert request.extensions["timeout"]["connect"] == 10
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(handler)),
    )
    state = State(gpu_status=GPUStatus.RUNNING)
    manager = GPUManager(settings, state)
    manager.tunnel = SimpleNamespace(returncode=None)
    assert not await manager.check_ready()
    assert calls == ["/health"] and not state.model_ready
    healthy = True
    assert await manager.check_ready()
    assert state.model_ready and state.active_provider == "local"
    assert state.model_context_window == expected
    assert state.model_context_model == settings.local_model
    assert calls == ["/health", "/health", "/v1/models", "/v1/chat/completions"]


def test_region_auto_chooses_available_capacity_by_price(settings):
    key = settings.root / "fake-key"
    key.write_text("fixture")
    settings.options.update(
        LAMBDA_SSH_KEY_NAME="name", SSH_PRIVATE_KEY="fake-key", LAMBDA_REGION="auto"
    )

    def offer(name, cents, regions):
        return {
            "instance_type": {
                "name": name,
                "gpu_description": "NVIDIA H100 80GB",
                "specs": {"gpus": 1},
                "price_cents_per_hour": cents,
            },
            "regions_with_capacity_available": [{"name": r} for r in regions],
        }

    catalog = {
        "no-capacity": offer("unavailable", 100, []),
        "east": offer("expensive", 350, ["us-east-1"]),
        "west": offer("available", 250, ["us-west-1"]),
    }
    # A cheaper two-GPU offer with the same total must not satisfy "Single".
    catalog["multi"] = offer("wrong-category", 1, ["us-west-1"])
    catalog["multi"]["instance_type"].update(gpu_description="NVIDIA A100 40GB", specs={"gpus": 2})
    manager = GPUManager(settings, State(), SimpleNamespace(catalog=lambda: catalog))
    manager.spec = ModelSpec(settings.local_model, "sha", 54 * 2**30, "hermes", "qwen3", 32)
    plan = manager.plan(80)
    assert (plan.instance_type, plan.region, plan.hourly_cents) == ("available", "us-west-1", 250)
    settings.options["LAMBDA_REGION"] = "us-east-1"
    assert manager.plan(80).instance_type == "expensive"


def test_40gb_reselection_checks_live_capacity_once_without_launching(settings):
    (settings.root / "fake-key").write_text("fixture")
    settings.options.update(
        LAMBDA_SSH_KEY_NAME="name", SSH_PRIVATE_KEY="fake-key", LAMBDA_REGION="auto"
    )
    calls = []
    regions = []
    status = 200

    def handler(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(
            status,
            json={
                "data": {
                    "gpu_1x_a100_sxm4": {
                        "instance_type": {
                            "name": "gpu_1x_a100_sxm4",
                            "gpu_description": "A100 (40 GB SXM4)",
                            "architecture": "x86_64",
                            "specs": {"gpus": 1, "memory_gib": 200},
                            "price_cents_per_hour": 199,
                        },
                        "regions_with_capacity_available": [{"name": r} for r in regions],
                    }
                }
            },
        )

    client = LambdaClient("fake", transport=httpx.MockTransport(handler))
    manager = GPUManager(settings, State(), client)
    manager.spec = ModelSpec(
        settings.local_model, "sha", 28_595_764_032, "jinja", "auto", 0, backend="llama.cpp"
    )
    try:
        with pytest.raises(ValueError, match="No available single-GPU instance with exactly 40"):
            manager.plan(40)
        assert len(calls) == 1

        # Capacity can open or disappear immediately, without restarting the harness.
        regions.append("us-east-1")
        plan = manager.plan(40)
        assert (plan.instance_type, plan.region, plan.total_vram_gib, plan.gpu_count) == (
            "gpu_1x_a100_sxm4",
            "us-east-1",
            40,
            1,
        )
        assert len(calls) == 2

        regions.clear()
        with pytest.raises(ValueError, match="No instance launched"):
            manager.plan(40)
        assert len(calls) == 3

        # Provider errors must not silently use old capacity or trigger retries.
        status = 503
        with pytest.raises(LambdaError, match="HTTP 503"):
            manager.plan(40)
        assert calls == [("GET", "/api/v1/instance-types")] * 4
        assert not manager.state.instance_id and not manager.state.launch_uncertain
        assert not (settings.runtime / "gpu.json").exists()
    finally:
        client.close()


def test_cluster_requires_exact_multi_gpu_capacity(settings):
    (settings.root / "fake-key").write_text("fixture")
    settings.options.update(LAMBDA_SSH_KEY_NAME="name", SSH_PRIVATE_KEY="fake-key")

    def offer(name, per_gpu, count, price):
        return {
            "instance_type": {
                "name": name,
                "gpu_description": f"Test GPU {per_gpu}GB",
                "specs": {"gpus": count},
                "price_cents_per_hour": price,
            },
            "regions_with_capacity_available": [{"name": "test-region"}],
        }

    # Synthetic catalog; this does not claim real provider availability.
    catalog = {
        "cluster": offer("cluster-match", 180, 10, 2000),
        "single": offer("wrong-category", 1800, 1, 1),
        "smaller": offer("wrong-total", 180, 8, 1),
    }
    manager = GPUManager(settings, State(), SimpleNamespace(catalog=lambda: catalog))
    manager.spec = ModelSpec(
        settings.local_model, "sha", 28 * 2**30, "jinja", "auto", 0, backend="llama.cpp"
    )
    plan = manager.plan(1800)
    assert (plan.instance_type, plan.total_vram_gib, plan.gpu_count) == ("cluster-match", 1800, 10)
    del catalog["cluster"]
    with pytest.raises(ValueError, match="No available multi-GPU instance with exactly 1800"):
        manager.plan(1800)
    assert not manager.state.instance_id


@pytest.mark.parametrize(
    "key", ["DOWNLOAD_STALL_SECONDS", "DOWNLOAD_MIN_BYTES_PER_SECOND", "DOWNLOAD_CHUNK_MIB"]
)
@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_download_policy_blocks_before_paid_requests(settings, key, value):
    settings.options[key] = value
    manager = GPUManager(settings, State())
    with pytest.raises(ValueError, match=key):
        manager.plan(40)
    with pytest.raises(ValueError, match=key):
        manager.launch(None, approved=True)
    assert manager._client is None


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_long_deployment_and_cancellation_keep_resource(settings, monkeypatch, outcome):
    """Advance the clock beyond the old deadline without waiting 15 minutes."""
    root = Path(__file__).resolve().parents[1]
    (settings.root / "scripts/runtime").mkdir(parents=True)
    (settings.root / "scripts/runtime/serve_model.py").write_bytes(
        (root / "scripts/runtime/serve_model.py").read_bytes()
    )
    (settings.root / "config/locks/serving.lock.json").write_text(
        json.dumps({"llama_cpp_image": "fixture@sha256:pinned"})
    )
    (settings.root / "test-key").write_text("fixture")
    settings.options.update(
        SSH_PRIVATE_KEY="test-key",
        DOWNLOAD_STALL_SECONDS="240",
        DOWNLOAD_MIN_BYTES_PER_SECOND="512",
    )
    state = State(
        instance_id="owned",
        instance_ip="192.0.2.1",
        gpu_status=GPUStatus.RUNNING,
        total_vram_gib=40,
        gpu_count=1,
    )
    manager = GPUManager(settings, state)
    manager.spec = ModelSpec(
        settings.local_model,
        "revision",
        1024,
        "",
        "",
        24,
        backend="llama.cpp",
        filename="model.gguf",
        sha256="a" * 64,
    )
    loop = asyncio.get_running_loop()
    clock = loop.time
    elapsed = 0
    monkeypatch.setattr(loop, "time", lambda: clock() + elapsed)
    started = asyncio.Event()
    connected = False

    class Process:
        returncode = None
        terminated = False

        def __init__(self):
            self.stdin = self
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()

        def write(self, program):
            tree = ast.parse(program)
            cfg = json.loads(ast.literal_eval(tree.body[-1].value.args[0].args[0]))
            assert cfg["download_stall_seconds"] == 240
            assert cfg["download_min_bytes_per_second"] == 512
            assert cfg["download_chunk_mib"] == 256
            assert cfg["weight_bytes"] == 1024

        async def drain(self):
            nonlocal elapsed
            started.set()
            if outcome == "cancel":
                await asyncio.Event().wait()
            elapsed = 1800  # A normal transfer outlasting the former 900s limit.
            self.stdout.feed_data(b'{"stage":"download_progress","bytes":512,"total":1024}\n')
            for _ in range(5):
                await asyncio.sleep(0)
            # Progress is visible while SSH is still running, not buffered until exit.
            assert self.returncode is None
            assert state.model_stage == "download" and state.download_bytes == 512
            assert not state.model_ready
            self.returncode = 1 if outcome == "error" else 0
            if self.returncode:
                self.stderr.feed_data(b"transport attempts exhausted")
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        def close(self):
            pass

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        async def wait(self):
            return self.returncode

    process = Process()

    async def spawn(*args, **kwargs):
        assert "ServerAliveInterval=30" in args
        assert "ServerAliveCountMax=3" in args
        return process

    async def connect():
        nonlocal connected
        connected = True

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(manager, "connect", connect)
    if outcome == "cancel":
        task = asyncio.create_task(manager.deploy())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.terminated
    elif outcome == "error":
        with pytest.raises(RuntimeError, match="transport attempts exhausted"):
            await manager.deploy()
    else:
        await manager.deploy()
    assert connected == (outcome == "success")
    assert state.model_stage == ("load" if outcome == "success" else "")
    assert state.instance_id == "owned"
    assert manager._client is None  # No provider request, including termination.


def test_transfer_progress_throttled_validated_and_not_restored(settings, monkeypatch):
    import velocicyber.providers.lambda_cloud as gpu

    clock = 1000
    monkeypatch.setattr(gpu, "time", SimpleNamespace(monotonic=lambda: clock))
    state = State(download_total=1000)
    manager = GPUManager(settings, state)

    def progress(done):
        manager.deployment_progress(
            json.dumps({"stage": "download_progress", "bytes": done, "total": 1000})
        )

    progress(200)  # A resumed transfer reports its current position immediately.
    assert state.model_stage == "download" and state.download_bytes == 200
    clock += 299
    progress(500)
    assert state.download_bytes == 200
    clock += 1
    progress(600)
    assert state.download_bytes == 600
    for line in (
        b"docker output",
        b"\xff",
        b"[]",
        b"{}",
        b'{"stage":"download_progress","bytes":900,"total":900}',
    ):
        manager.deployment_progress(line)
    progress(-1)
    progress(1001)
    assert state.download_bytes == 600
    progress(1000)  # Completion is immediate even inside the five-minute window.
    assert state.download_bytes == 1000 and not state.model_ready
    manager.deployment_progress('{"stage":"download_verifying"}')
    assert state.model_stage == "verify" and not state.model_ready
    state.persist(settings.runtime / "gpu.json")
    restored = State.restore(settings.runtime / "gpu.json")
    assert not restored.model_stage and not restored.download_total
    assert not restored.model_ready


@pytest.mark.parametrize("status", ["running", "restarting", "exited", "dead"])
async def test_loading_detects_crashed_container_without_restarting_it(settings, monkeypatch, status):
    state = State(instance_ip="192.0.2.1", model_stage="load")
    manager = GPUManager(settings, state)
    monkeypatch.setattr(manager, "ssh_args", lambda: ["ssh"])
    commands = []

    async def communicate():
        return json.dumps({"Status": status, "ExitCode": 1}).encode(), None

    async def spawn(*args, **kwargs):
        commands.append(args)
        return SimpleNamespace(returncode=0, communicate=communicate)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    if status == "running":
        await manager.check_model_process()
        assert state.model_stage == "load"
    else:
        with pytest.raises(RuntimeError, match="startup failed"):
            await manager.check_model_process()
        assert state.model_stage == "failed" and not state.model_ready
    assert len(commands) == 1
    assert commands[0][-1] == "sudo -n docker inspect --format '{{json .State}}' velocicyber-model"


@pytest.mark.parametrize("failure", ["timeout", "http", "malformed", "empty"])
async def test_inference_failure_reports_cause_without_retry(settings, monkeypatch, failure):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": settings.local_model}]})
        if failure == "timeout":
            raise httpx.ReadTimeout("fixture", request=request)
        if failure == "http":
            return httpx.Response(500, json={"error": "fixture"})
        if failure == "malformed":
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(handler)),
    )
    state = State(gpu_status=GPUStatus.RUNNING)
    manager = GPUManager(settings, state)
    manager.tunnel = SimpleNamespace(returncode=None)
    expected = {
        "timeout": "ReadTimeout",
        "http": "HTTP 500",
        "malformed": "IndexError",
        "empty": "no text",
    }
    with pytest.raises(RuntimeError, match=expected[failure]):
        await manager.check_ready()
    assert not state.model_ready and state.active_provider == "api"
    assert calls == ["/health", "/v1/models", "/v1/chat/completions"]
