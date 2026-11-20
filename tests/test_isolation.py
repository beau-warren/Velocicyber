"""Exercise the filesystem boundary with real processes and synthetic secrets."""

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from velocicyber.engine.isolation import bubblewrap, isolate, validate_workspace
from velocicyber.engine.runner import Engine
from velocicyber.engine.sessions import private_read, private_write
from velocicyber.state import State


@pytest.fixture
def isolation_backend():
    if sys.platform != "linux" or not shutil.which("bwrap", path="/usr/bin:/bin"):
        pytest.skip("The Linux Bubblewrap integration backend is not installed")
    return bubblewrap()


def execute(settings, code, *, home=None):
    workspace = settings.root / "workspace"
    workspace.mkdir(exist_ok=True)
    command, env = isolate(["/usr/bin/python3", "-c", code], {
        "PATH": str(settings.root), "HOME": str(settings.root),
        "ADVERSARIAL_API": "synthetic-provider-secret", "SSH_AUTH_SOCK": "/run/private-agent",
        "PYTHONPATH": str(settings.root), "VELOCICYBER_BRIDGE_TOKEN": "ephemeral-fixture",
    }, workspace=workspace, home=home)
    return subprocess.run(command, env=env, cwd=workspace, capture_output=True,
                          text=True, timeout=15)


def test_closed_filesystem_hides_credentials_symlinks_processes_and_host_tmp(settings, isolation_backend):
    engine = Engine(settings, State())
    session = engine.ensure_session()
    host_secret = settings.root / ".env"
    private_write(host_secret, "PRIVATE_CANARY_MUST_STAY_OUT")
    (engine.workspace / "escape").symlink_to(host_secret)
    (engine.workspace / "visible.txt").write_text("project data")
    # Host process memory/environment and /proc/PID/root must not bypass mounts.
    parent = os.getpid()
    paths = [str(host_secret), str(engine.workspace / "escape"),
             f"/proc/{parent}/root{host_secret}", f"/proc/{parent}/environ",
             "/proc/1/root" + str(host_secret)]
    try:
        result = execute(settings, f'''
import json,os
from pathlib import Path
hidden={paths!r}
assert Path('visible.txt').read_text() == 'project data'
Path('created.txt').write_text('done')
for path in hidden:
    try:
        Path(path).read_bytes()
    except (FileNotFoundError,PermissionError):
        pass
    else:
        raise AssertionError('private file became readable: ' + path)
assert 'ADVERSARIAL_API' not in os.environ
assert 'SSH_AUTH_SOCK' not in os.environ
assert 'PYTHONPATH' not in os.environ
assert os.environ['HOME']=='/home/agent'
assert os.environ['VELOCICYBER_BRIDGE_TOKEN']=='ephemeral-fixture'
assert not Path('/run').exists()
print('isolation passed')
''', home=session.home)
        assert result.returncode == 0, result.stderr + result.stdout
        assert result.stdout.strip() == "isolation passed"
        assert (engine.workspace / "created.txt").read_text() == "done"
        assert host_secret.read_text() == "PRIVATE_CANARY_MUST_STAY_OUT"
    finally:
        engine.close()


def test_conversation_control_files_are_read_only_but_history_is_writable(settings, isolation_backend):
    engine = Engine(settings, State())
    session = engine.ensure_session()
    for name in ("execution-instructions.md", "model-catalog.json"):
        private_write(session.home / name, "host-controlled")
    controls = [str(session.home / name) for name in
                ("system-prompt.md", "execution-instructions.md", "model-catalog.json")]
    try:
        result = execute(settings, f'''
from pathlib import Path
for name in {controls!r}:
    p=Path(name)
    assert p.read_text()
    for operation in (lambda: p.write_text('changed'),lambda: p.unlink(),
                      lambda: p.rename(p.with_suffix('.moved'))):
        try: operation()
        except OSError: pass
        else: raise AssertionError('control file modified')
Path({str(session.home / 'history-fixture.json')!r}).write_text('history')
''', home=session.home)
        assert result.returncode == 0, result.stderr
        assert (session.home / "history-fixture.json").read_text() == "history"
        assert (session.home / "execution-instructions.md").read_text() == "host-controlled"
    finally:
        engine.close()


def test_ephemeral_tmp_does_not_expose_host_files(settings, isolation_backend, tmp_path):
    # The real host temp path is deliberately outside the mounted workspace.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="velocicyber-isolation-") as temporary:
        secret = Path(temporary) / "private.txt"
        secret.write_text("host-temp-canary")
        code = f"from pathlib import Path; assert not Path({str(secret)!r}).exists(); Path('/tmp/scratch').write_text('ok')"
        result = execute(settings, code)
        assert result.returncode == 0, result.stderr
        result = execute(settings, "from pathlib import Path; assert not Path('/tmp/scratch').exists()")
        assert result.returncode == 0, result.stderr
        assert secret.read_text() == "host-temp-canary"


@pytest.mark.parametrize("workspace", ["", ".runtime/project", ".venv", "ssh-keys", "../outside"])
def test_private_workspace_roots_require_open(settings, workspace):
    with pytest.raises(ValueError):
        validate_workspace(settings.root, settings.root / workspace)


def test_backend_missing_fails_closed_before_provider_access(settings, monkeypatch):
    monkeypatch.setattr("velocicyber.engine.isolation.shutil.which", lambda *a, **kw: None)
    engine = Engine(settings, State())
    try:
        with pytest.raises(RuntimeError, match="requires Bubblewrap"):
            asyncio.run(engine.run("hello", lambda event: None))
        assert engine.session is None and engine.process is None
    finally:
        engine.close()


def test_control_file_io_never_follows_temporary_or_destination_symlinks(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("keep")
    target = tmp_path / "control.md"
    target.with_suffix(".tmp").symlink_to(secret)
    target.symlink_to(secret)
    with pytest.raises(OSError):
        private_read(target)
    private_write(target, "replacement")
    assert target.read_text() == "replacement" and not target.is_symlink()
    assert secret.read_text() == "keep"
    assert target.stat().st_mode & 0o777 == 0o600


def test_control_file_symlink_is_rejected(settings, isolation_backend):
    engine = Engine(settings, State())
    session = engine.ensure_session()
    (session.home / "system-prompt.md").unlink()
    (session.home / "system-prompt.md").symlink_to(settings.root / ".env")
    try:
        with pytest.raises(ValueError, match="symlinks"):
            isolate(["true"], {}, workspace=engine.workspace, home=session.home)
    finally:
        engine.close()


@pytest.mark.parametrize("provider", ["api", "pi", "local"])
@pytest.mark.parametrize("sandbox_open", [False, True])
async def test_real_native_tools_follow_explicit_access_mode(settings, native_engine, isolation_backend, provider, sandbox_open, monkeypatch):
    """Use adversarial native read/bash calls through the real bridge and engine."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []
    secret = settings.root / ".env"
    private_write(secret, "DO_NOT_DISCLOSE_HOST_CREDENTIAL_CANARY")
    state = State(collection=False, upload_approved=False, active_provider=provider,
                  pi_ready=True, model_ready=True, sandbox_open=sandbox_open)
    if sandbox_open:
        monkeypatch.setattr("velocicyber.engine.runner.check_isolation",
                            lambda: pytest.fail("Open mode must not require the isolation backend"))
    engine = Engine(settings, state)
    (engine.workspace / "escape").symlink_to(secret)
    (engine.workspace / "project.txt").write_text("ordinary project input")
    session = engine.ensure_session()
    commands = [
        ("read", {"path": str(secret)}),
        ("read", {"path": str(engine.workspace / "escape")}),
        ("exec_command", {"cmd": "cat " + str(secret), "login": False}),
        ("write", {"path": str(settings.root / "outside-write.txt"), "content": "native write"}),
        ("read", {"path": str(engine.workspace / "project.txt")}),
        ("exec_command", {"cmd": "printf done > result.txt", "login": False}),
    ]

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            index = len(requests) - 1
            if index < len(commands):
                name, arguments = commands[index]
                delta = {"role": "assistant", "tool_calls": [{
                    "index": 0, "id": f"boundary_{index}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }]}
                finish = "tool_calls"
            else:
                delta = {"role": "assistant", "content": "Workspace task completed."}
                finish = "stop"
            packet = {"id": "isolation-fixture", "object": "chat.completion.chunk", "created": 1,
                      "model": payload["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: " + json.dumps(packet).encode() + b"\n\ndata: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    settings.api_url = state.pi_url = state.local_url = url
    settings.options["INTERPRETER_BIN"] = str(native_engine)
    # The verifier executes project-controlled code, so it needs containment too.
    (engine.workspace / "verify.py").write_text(
        "from pathlib import Path\n"
        + f"assert Path({str(secret)!r}).exists() == {sandbox_open!r}\n"
        + "assert Path('result.txt').read_text() == 'done'\n"
    )
    engine.verifier = [sys.executable, "verify.py"]
    events = []
    try:
        await asyncio.wait_for(engine.run("Read the requested files and create result.txt.", events.append), 60)
        assert len(requests) == len(commands) + 1, events
        transcript = json.dumps(requests)
        assert ("DO_NOT_DISCLOSE_HOST_CREDENTIAL_CANARY" in transcript) == sandbox_open
        assert "ordinary project input" in transcript
        assert (engine.workspace / "result.txt").read_text() == "done"
        assert secret.read_text() == "DO_NOT_DISCLOSE_HOST_CREDENTIAL_CANARY"
        assert (settings.root / "outside-write.txt").exists() == sandbox_open
        assert session.metadata["filesystem_isolation"] == ("none" if sandbox_open else "bubblewrap")
        assert session.metadata["status"] == "complete"
    finally:
        engine.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_namespace_failure_is_explicit_and_does_not_widen_access(monkeypatch):
    from types import SimpleNamespace

    from velocicyber.engine.isolation import check_isolation
    monkeypatch.setattr("velocicyber.engine.isolation.bubblewrap", lambda: "/usr/bin/bwrap")
    monkeypatch.setattr("velocicyber.engine.isolation.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="access was not widened"):
        check_isolation()


async def test_cancelled_isolation_stops_detached_children(settings, isolation_backend):
    from velocicyber.sentinel.observer import stop_process_tree

    workspace = settings.root / "workspace"
    workspace.mkdir()
    heartbeat = workspace / "heartbeat"
    child = "import time; from pathlib import Path; p=Path('heartbeat'); " + "\nwhile True: p.write_text(str(time.time_ns())); time.sleep(.03)"
    code = f"import subprocess,time; subprocess.Popen(['/usr/bin/python3','-c',{child!r}],start_new_session=True); print('ready',flush=True); time.sleep(60)"
    command, env = isolate(["/usr/bin/python3", "-c", code], {}, workspace=workspace)
    process = await asyncio.create_subprocess_exec(*command, env=env, start_new_session=True,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        assert await asyncio.wait_for(process.stdout.readline(), 10) == b"ready\n"
        for _ in range(100):
            if heartbeat.exists():
                break
            await asyncio.sleep(.02)
        assert heartbeat.exists()
        await stop_process_tree(process)
        await asyncio.wait_for(process.wait(), 5)
        before = heartbeat.read_text()
        await asyncio.sleep(.15)
        assert heartbeat.read_text() == before
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def test_symlinked_engine_home_is_rejected_before_host_control_read(settings):
    from types import SimpleNamespace

    engine = Engine(settings, State())
    session = engine.ensure_session()
    original = session.folder / "original-engine"
    session.home.rename(original)
    session.home.symlink_to(original, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="stay inside this conversation"):
            engine.command(SimpleNamespace(url="http://fixture/v1"), settings.api_model)
    finally:
        engine.close()
