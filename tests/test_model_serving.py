import hashlib
import importlib.util
import json
import shutil
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from velocicyber.providers.gguf import inspect_header, metadata_from_prefix
from velocicyber.providers.lambda_cloud import GPUManager
from velocicyber.state import State

module_spec = importlib.util.spec_from_file_location(
    "serve_model", Path(__file__).resolve().parents[1] / "scripts/runtime/serve_model.py"
)
serve_model = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(serve_model)


def gguf_prefix():
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    return (
        b"GGUF"
        + struct.pack("<IQQ", 3, 851, 2)
        + string("general.architecture")
        + struct.pack("<I", 8)
        + string("qwen35")
        # A tokenizer array intentionally exceeds the prefix available to the reader.
        + string("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", 9, 8, 248320)
        + string("first token")
    )


def test_gguf_prefix_and_bounded_download_on_ignored_range():
    requested = []
    consumed = []

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            first = gguf_prefix().ljust(65536, b"\0")
            for offset in range(0, len(first), 8192):
                consumed.append(offset)
                yield first[offset : offset + 8192]
            raise AssertionError("Attempted to download model tensors")

    def handler(request):
        requested.append(request)
        if request.url.host == "huggingface.co":
            return httpx.Response(302, headers={"Location": "https://cdn.example/weights"})
        return httpx.Response(200, stream=Stream())

    meta = inspect_header(
        "owner/model",
        "immutable-revision",
        "model.gguf",
        "private-token",
        transport=httpx.MockTransport(handler),
    )
    assert meta["general.architecture"] == "qwen35"
    assert requested[0].headers["range"] == "bytes=0-65535"
    assert "immutable-revision" in str(requested[0].url)
    assert "authorization" not in requested[1].headers
    assert len(consumed) == 8
    with pytest.raises(ValueError, match="header"):
        metadata_from_prefix(b"not a gguf")


def test_both_model_formats_and_vram_requirements(settings, monkeypatch):
    import huggingface_hub

    import velocicyber.providers.gguf

    default = settings.local_model
    alternative = "example-org/bf16-model"
    metadata = {
        default: SimpleNamespace(
            sha="gguf-revision",
            siblings=[
                SimpleNamespace(
                    rfilename="Qwen3.8-27B-Uncensored-OrcaRouter-Q8_0.gguf",
                    size=28595764032,
                    lfs=SimpleNamespace(sha256="a" * 64),
                )
            ],
        ),
        alternative: SimpleNamespace(
            sha="safetensors-revision",
            siblings=[
                SimpleNamespace(rfilename=f"model-{i}.safetensors", size=size)
                for i, size in enumerate((49825162976, 4888445168, 849400424))
            ],
        ),
    }
    calls = []

    def model_info(repo, **kwargs):
        calls.append(repo)
        return metadata[repo]

    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(model_info=model_info)
    )
    config = settings.runtime / "config.json"
    config.write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {"num_attention_heads": 24}})
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kw: str(config))
    monkeypatch.setattr(
        velocicyber.providers.gguf, "inspect_header", lambda *args: metadata_from_prefix(gguf_prefix())
    )
    manager = GPUManager(settings, State())
    spec = manager.inspect_model()
    assert spec.backend == "llama.cpp" and spec.required_gib == 36
    assert spec.sha256 == "a" * 64 and spec.architecture == "qwen35"
    assert manager.inspect_model() is spec
    other = manager.inspect_model(alternative)
    assert other.backend == "vllm" and other.required_gib == 67
    assert other.tool_parser == "qwen3_coder" and other.attention_heads == 24
    assert settings.local_model == default
    assert calls == [default, alternative]


@pytest.fixture
def remote_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    weights = b"tiny stand-in for model weights"
    cfg = dict(
        backend="llama.cpp",
        repo="owner/model",
        revision="a" * 40,
        filename="model.gguf",
        sha256=hashlib.sha256(weights).hexdigest(),
        weight_bytes=len(weights),
        token="secret-token",
        image="server@sha256:fixture",
        gpus=1,
        tool_parser="qwen3_coder",
        reasoning_parser="qwen3",
    )
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        assert cfg["token"] not in " ".join(args)
        if "inspect" in args:
            return SimpleNamespace(returncode=0, stdout="true\n")
        if args[0] == "curl":
            headers = Path(args[args.index("--header") + 1][1:])
            assert headers.stat().st_mode & 0o777 == 0o600
            assert headers.read_text() == "Authorization: Bearer secret-token\n"
            start, end = map(int, args[args.index("--range") + 1].split("-"))
            Path(args[args.index("--output") + 1]).write_bytes(weights[start : end + 1])
            Path(args[args.index("--dump-header") + 1]).write_text(
                f"HTTP/1.1 206 Partial Content\nContent-Range: bytes {start}-{end}/{len(weights)}\n\n"
            )
            return SimpleNamespace(returncode=0, stdout="206", stderr="")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(serve_model.subprocess, "run", run)
    return cfg, commands, tmp_path / ".velocicyber"


def test_gguf_verified_before_replacing_server(remote_setup, capsys):
    cfg, commands, folder = remote_setup
    serve_model.serve(cfg)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events == [
        {"stage": "download_progress", "bytes": 0, "total": cfg["weight_bytes"]},
        {"stage": "download_progress", "bytes": cfg["weight_bytes"], "total": cfg["weight_bytes"]},
        {"stage": "download_verifying"},
        {"stage": "server_starting"},
        {"stage": "model_loading"},
    ]
    download = next(cmd for cmd in commands if cmd[0] == "curl")
    assert "--http1.1" in download
    assert download[1] == "--disable"
    assert "--max-time" not in download
    assert download[download.index("--speed-time") + 1] == "180"
    assert download[download.index("--speed-limit") + 1] == "1024"
    assert (
        download[-1] == f"https://huggingface.co/owner/model/resolve/{cfg['revision']}/model.gguf"
    )
    launch = commands[-1]
    assert "--jinja" in launch and "--env-file" not in launch
    assert launch[launch.index("--alias") + 1] == cfg["repo"]
    assert "127.0.0.1:8000:8000" in launch
    assert launch[launch.index("--health-cmd") + 1].endswith("http://127.0.0.1:8000/health")
    assert [cmd[3] for cmd in commands[-3:]] == ["pull", "rm", "run"]
    assert not (folder / "download.headers").exists()
    before = len(commands)
    serve_model.serve(cfg)
    assert not any(cmd[0] == "curl" for cmd in commands[before:])
    assert [json.loads(line)["stage"] for line in capsys.readouterr().out.splitlines()] == [
        "download_verifying",
        "server_starting",
        "model_loading",
    ]


def test_bad_checksum_keeps_existing_server(remote_setup):
    cfg, commands, folder = remote_setup
    cfg["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checksum"):
        serve_model.serve(cfg)
    assert not any("rm" in cmd or "run" in cmd for cmd in commands)
    assert not (folder / "download.headers").exists()
    assert not list(folder.rglob("*.partial"))


def test_safetensors_uses_pinned_revision_and_vllm(remote_setup):
    cfg, commands, folder = remote_setup
    cfg["backend"] = "vllm"
    serve_model.serve(cfg)
    command = commands[-1]
    assert command[command.index("--revision") + 1] == cfg["revision"]
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert "--env-file" in command and "--jinja" not in command
    assert not (folder / "serve.env").exists()
    assert not any(cmd[0] == "curl" for cmd in commands)


@pytest.mark.parametrize("download_ok", [False, True])
def test_large_snapshot_finishes_before_server_replacement(remote_setup, monkeypatch, download_ok):
    from velocicyber.models.gpu_profiles import CYBER_FROST_REPO, GPU_PROFILES

    cfg, commands, folder = remote_setup
    cfg.update(
        backend="vllm",
        repo=CYBER_FROST_REPO,
        download_files=[{"fixture": True}],
        extra_arguments=GPU_PROFILES[CYBER_FROST_REPO].arguments,
        max_num_seqs=1,
    )

    def download(config, path, docker, mounts):
        assert config == cfg and path == folder
        assert [*docker, "pull", cfg["image"]] in commands
        assert "--entrypoint" in commands[-1]  # Dependency check precedes weights.
        assert not any("rm" in cmd or "--detach" in cmd for cmd in commands)
        if not download_ok:
            raise RuntimeError("download failed")

    monkeypatch.setattr(serve_model, "download_safetensors", download)
    if download_ok:
        serve_model.serve(cfg)
        launch = commands[-1]
        assert "HF_HUB_OFFLINE=1" in launch
        assert "--language-model-only" in launch
        assert launch[launch.index("--dtype") + 1] == "bfloat16"
        assert launch[launch.index("--max-num-seqs") + 1] == "1"
        assert launch[launch.index("--mamba-ssm-cache-dtype") + 1] == "bfloat16"
        assert len([cmd for cmd in commands if "pull" in cmd]) == 1
    else:
        with pytest.raises(RuntimeError, match="download failed"):
            serve_model.serve(cfg)
        assert not any("rm" in cmd or "--detach" in cmd for cmd in commands)
    assert not (folder / "serve.env").exists()


def test_broken_runtime_stops_before_weight_download_or_server_replacement(remote_setup, monkeypatch):
    cfg, commands, folder = remote_setup
    cfg.update(backend="vllm", download_files=[{"fixture": True}])
    original = serve_model.subprocess.run

    def run(args, **kwargs):
        if "--entrypoint" in args:
            commands.append(args)
            assert "--env-file" not in args and "--volume" not in args
            assert args[args.index("--network") + 1] == "none"
            return SimpleNamespace(returncode=1, stderr="operator torchvision::nms does not exist")
        return original(args, **kwargs)

    monkeypatch.setattr(serve_model.subprocess, "run", run)
    monkeypatch.setattr(serve_model, "download_safetensors", lambda *a: pytest.fail("Unexpected download"))
    with pytest.raises(RuntimeError, match="runtime check failed.*torchvision::nms"):
        serve_model.serve(cfg)
    assert not any("rm" in cmd or "--detach" in cmd for cmd in commands)
    assert not (folder / "serve.env").exists()


def test_snapshot_requires_space_before_starting_download(remote_setup, monkeypatch):
    cfg, commands, folder = remote_setup
    cfg["download_files"] = [
        {"filename": "model.safetensors", "size": cfg["weight_bytes"], "sha256": cfg["sha256"]}
    ]
    folder.mkdir()
    monkeypatch.setattr(serve_model.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="Insufficient disk"):
        serve_model.download_safetensors(cfg, folder, ["docker"], [])
    assert not commands


@pytest.mark.parametrize("corrupt", [False, True])
def test_snapshot_downloader_verifies_shards_and_reports_weight_bytes(
    tmp_path, monkeypatch, capsys, corrupt
):
    import io
    import sys

    import huggingface_hub

    files = []
    for name, data in (("config.json", b"{}"), ("model.safetensors", b"weights")):
        (tmp_path / name).write_bytes(data)
        files.append(
            {"filename": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    if corrupt:
        (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    cfg = {"repo": "owner/model", "revision": "a" * 40, "weight_bytes": 7, "download_files": files}
    calls = []

    def download(repo, filename, revision):
        calls.append((repo, filename, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(cfg)))
    if corrupt:
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            exec(serve_model.SNAPSHOT_DOWNLOADER, {})
        assert not (tmp_path / "model.safetensors").exists()
    else:
        exec(serve_model.SNAPSHOT_DOWNLOADER, {})
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert events[-1] == {"stage": "download_progress", "bytes": 7, "total": 7}
    assert len(calls) == 2 and all(call[2] == cfg["revision"] for call in calls)


@pytest.mark.parametrize("backend,flag", [("llama.cpp", "--ctx-size"), ("vllm", "--max-model-len")])
def test_serving_uses_selected_context_window(remote_setup, backend, flag):
    cfg, commands, _ = remote_setup
    cfg.update(backend=backend, context_window=16384)
    serve_model.serve(cfg)
    command = commands[-1]
    assert command[command.index(flag) + 1] == "16384"


def seed_partial(cfg, folder, content):
    partial = folder / "weights" / cfg["revision"] / "model.partial"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(content)
    return partial


@pytest.mark.parametrize(
    "error_code,expected_attempts", [(18, 2), (28, 2), (56, 2), (92, 2), (22, 1)]
)
def test_transfer_resume_does_not_retry_http_errors(
    remote_setup, monkeypatch, capsys, error_code, expected_attempts
):
    cfg, commands, folder = remote_setup
    original_run = serve_model.subprocess.run
    attempts = []
    partial = seed_partial(cfg, folder, b"tiny ")

    def run(command, **kwargs):
        if command[0] == "curl":
            attempts.append(command)
            assert partial.read_bytes() == b"tiny "
            assert command[command.index("--range") + 1] == f"5-{cfg['weight_bytes'] - 1}"
            if len(attempts) == 1:
                Path(command[command.index("--output") + 1]).write_bytes(b"interrupted chunk")
                return SimpleNamespace(
                    returncode=error_code, stdout="403" if error_code == 22 else "206"
                )
        return original_run(command, **kwargs)

    monkeypatch.setattr(serve_model.subprocess, "run", run)
    monkeypatch.setattr(serve_model.time, "sleep", lambda _: None)
    if error_code == 22:
        with pytest.raises(RuntimeError, match="HTTP 403"):
            serve_model.serve(cfg)
        assert partial.read_bytes() == b"tiny "
    else:
        serve_model.serve(cfg)
    assert len(attempts) == expected_attempts
    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first == {"stage": "download_progress", "bytes": 5, "total": cfg["weight_bytes"]}
    assert not (folder / "download.headers").exists()
    assert not list(folder.rglob("*.chunk"))


def test_stalls_stop_after_three_attempts_and_keep_partial(remote_setup, monkeypatch):
    cfg, commands, folder = remote_setup
    original_run = serve_model.subprocess.run
    attempts = []
    cfg.update(download_stall_seconds=45, download_min_bytes_per_second=128)
    partial = seed_partial(cfg, folder, b"partial")

    def run(command, **kwargs):
        if command[0] == "curl":
            attempts.append(command)
            assert command[command.index("--speed-time") + 1] == "45"
            assert command[command.index("--speed-limit") + 1] == "128"
            Path(command[command.index("--output") + 1]).write_bytes(b"bad chunk")
            return SimpleNamespace(returncode=28, stdout="206")
        return original_run(command, **kwargs)

    monkeypatch.setattr(serve_model.subprocess, "run", run)
    monkeypatch.setattr(serve_model.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match=r"3 attempt\(s\).*7 partial bytes retained"):
        serve_model.serve(cfg)
    assert len(attempts) == 3
    assert partial.read_bytes() == b"partial"
    assert not (folder / "download.headers").exists()
    assert not list(folder.rglob("*.chunk"))
    assert not any("rm" in cmd or "run" in cmd for cmd in commands)


def test_completed_partial_verified_without_network(remote_setup):
    cfg, commands, folder = remote_setup
    seed_partial(cfg, folder, b"tiny stand-in for model weights")
    serve_model.serve(cfg)
    assert not any(cmd[0] == "curl" for cmd in commands)
    assert not list(folder.rglob("*.partial"))


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl is required")
@pytest.mark.parametrize(
    "scenario",
    [
        "progress",
        "stall_resume",
        "chunks",
        "ignore_range",
        "wrong_range",
        "short_body",
        401,
        403,
        429,
    ],
)
def test_real_curl_transfer_policy(tmp_path, monkeypatch, scenario):
    """Exercise curl, bounded ranges, resume and bad responses on loopback only."""
    payload = b"test weights\n" * (200000 if scenario == "chunks" else 16384)
    requests = []
    release_stall = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            range_header = self.headers.get("Range")
            requests.append(range_header)
            if isinstance(scenario, int):
                self.send_response(scenario)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            offset, end = map(int, range_header.removeprefix("bytes=").split("-"))
            body = payload[offset : end + 1]
            if scenario == "short_body":
                body = body[:-1]
            self.send_response(200 if scenario == "ignore_range" else 206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            reported_start = offset + 1 if scenario == "wrong_range" else offset
            self.send_header("Content-Range", f"bytes {reported_start}-{end}/{len(payload)}")
            self.end_headers()
            try:
                if scenario == "stall_resume" and len(requests) == 1:
                    self.wfile.write(body[:4096])
                    self.wfile.flush()
                    release_stall.wait(20)
                    self.close_connection = True
                elif scenario == "progress":
                    for start in range(0, len(body), 8192):
                        self.wfile.write(body[start : start + 8192])
                        self.wfile.flush()
                        time.sleep(0.15)
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    actual_run = serve_model.subprocess.run

    def local_curl(command, **kwargs):
        assert command[0] == "curl"
        local = [*command[:-1], f"http://127.0.0.1:{server.server_port}/weights"]
        return actual_run(local, **kwargs, timeout=20)

    monkeypatch.setattr(serve_model.subprocess, "run", local_curl)
    cfg = {
        "repo": "fixture/model",
        "revision": "immutable",
        "filename": "model.gguf",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "weight_bytes": len(payload),
        "token": "fixture-token",
        "download_stall_seconds": 2,
        "download_min_bytes_per_second": 1024,
        "download_chunk_mib": 1,
    }
    prefix_size = 4096 if scenario in ("stall_resume", "chunks") else 0
    partial = seed_partial(cfg, tmp_path, payload[:prefix_size])
    first_range = f"bytes={prefix_size}-{min(prefix_size + 1024**2, len(payload)) - 1}"
    started = time.monotonic()
    try:
        if isinstance(scenario, int):
            with pytest.raises(RuntimeError, match=f"HTTP {scenario}"):
                serve_model.download_gguf(cfg, tmp_path)
            assert requests == [first_range]
        elif scenario in ("ignore_range", "wrong_range", "short_body"):
            with pytest.raises(RuntimeError, match="unexpected byte range or length"):
                serve_model.download_gguf(cfg, tmp_path)
            assert partial.read_bytes() == payload[:prefix_size]
            assert requests == [first_range]
        else:
            destination = serve_model.download_gguf(cfg, tmp_path)
            assert destination.read_bytes() == payload
            if scenario == "progress":
                assert time.monotonic() - started > cfg["download_stall_seconds"]
                assert requests == [first_range]
            elif scenario == "stall_resume":
                assert requests == [first_range, first_range]
            else:
                assert requests == [
                    f"bytes={start}-{min(start + 1024**2, len(payload)) - 1}"
                    for start in range(prefix_size, len(payload), 1024**2)
                ]
            assert not list(tmp_path.rglob("*.partial"))
        assert not (tmp_path / "download.headers").exists()
        assert not list(tmp_path.rglob("*.chunk"))
        assert not list(tmp_path.rglob("*.headers"))
    finally:
        release_stall.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
