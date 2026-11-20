"""Rootless filesystem/process isolation around the unmodified execution engine.

The host owns provider credentials and the inference bridge. Closed execution
gets a fresh filesystem view with explicit mounts, including its own /proc and
/tmp. Networking is shared for the bridge; this is not a target/network firewall.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
SYSTEM_DIRS = ("/usr/bin", "/usr/sbin", "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/share")
SYSTEM_CONFIG = ("/etc/ld.so.cache", "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/ssl/certs")
CONTROL_FILES = ("system-prompt.md", "execution-instructions.md", "model-catalog.json")
PRIVATE_DIRS = {".git", ".venv", ".runtime", ".alpha-audit", "ssh-keys", "upstream",
                "velocicyber_finetuning-data", "velocicyber_sentinel-data"}


def bubblewrap():
    binary = shutil.which("bwrap", path="/usr/bin:/bin") if sys.platform == "linux" else None
    if not binary:
        raise RuntimeError(
            "Closed workspace access requires Bubblewrap on Linux. On Raspberry Pi OS/Debian, "
            "install it with: sudo apt install bubblewrap. No agent command was started. "
            "Run ./velocicyber.sh doctor to check setup."
        )
    return binary


def validate_workspace(root, workspace):
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    if not workspace.is_relative_to(root):
        raise ValueError("The agent workspace is outside the Velocicyber project. "
                         "Select /sandbox > Open first, or launch with --sandbox open.")
    relative = workspace.relative_to(root)
    if not relative.parts or relative.parts[0] in PRIVATE_DIRS:
        raise ValueError("Closed access requires a project subdirectory, not the harness root "
                         "or private storage. Use the workspace directory or select /sandbox > Open.")


def _mount(arguments, path, *, writable=False):
    path = Path(path)
    arguments.extend(["--bind" if writable else "--ro-bind", str(path), str(path)])


def _engine_mounts(arguments, executable):
    """Expose only the selected executable and its official bundled helpers."""
    executable = Path(executable).absolute()
    if not executable.is_file():
        raise RuntimeError("The selected execution engine is missing.")
    bundle = executable.parent.parent
    metadata = bundle / "codex-package.json"
    try:
        packaged = executable.parent.name == "bin" and json.loads(metadata.read_text()).get("variant") == "open-interpreter"
    except (OSError, ValueError, AttributeError):
        packaged = False
    if packaged:
        for name in ("bin", "codex-path", "codex-resources", "codex-package.json"):
            path = bundle / name
            if path.exists():
                _mount(arguments, path)
    elif not any(executable.is_relative_to(path) for path in SYSTEM_DIRS):
        _mount(arguments, executable)


def isolate(command, environment, *, workspace, home=None, engine=None):
    """Return a Bubblewrap command and a small, deterministic child environment.

    No provider secret or user shell/agent config is inherited. The short-lived
    bridge credential is passed through the process environment, never argv.
    """
    arguments = [bubblewrap(), "--unshare-user", "--unshare-pid", "--unshare-ipc",
                 "--unshare-uts", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    for path in SYSTEM_DIRS:
        if Path(path).is_dir():
            _mount(arguments, path)
    for name in ("bin", "sbin", "lib", "lib64"):
        path = Path("/") / name
        if path.is_symlink():
            arguments.extend(["--symlink", os.readlink(path), str(path)])
        elif path.is_dir():
            _mount(arguments, path)
    for path in SYSTEM_CONFIG:
        if Path(path).exists():
            _mount(arguments, path)
    arguments.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                      "--tmpfs", "/home/agent", "--dir", "/tmp/velocicyber-cache"])
    if engine is not None:
        if Path(engine).resolve().is_relative_to(Path(workspace).resolve()):
            raise ValueError("The execution engine must be installed outside the agent workspace.")
        _engine_mounts(arguments, engine)
    # A user-configured verifier can be a standalone executable outside /usr.
    executable = Path(command[0])
    if engine is None and executable.is_absolute() and executable.is_file():
        if not executable.is_relative_to(workspace) and not any(executable.is_relative_to(p) for p in SYSTEM_DIRS):
            _mount(arguments, executable)
    if home is not None:
        home = Path(home)
        if home.is_symlink() or not home.is_dir():
            raise ValueError("Saved engine history must be a real conversation directory.")
        _mount(arguments, home, writable=True)
        # Engine rollouts are writable; host-controlled prompts/catalogs are not.
        for name in CONTROL_FILES:
            path = home / name
            if path.is_symlink():
                raise ValueError("Conversation control files must not be symlinks.")
            if path.is_file():
                _mount(arguments, path)
    _mount(arguments, workspace, writable=True)
    arguments.extend(["--remount-ro", "/", "--chdir", str(workspace), "--", *command])
    child = {k: v for k, v in environment.items() if k in {
        "LANG", "TERM", "INTERPRETER_HOME", "VELOCICYBER_BRIDGE_TOKEN", "HF_HUB_DISABLE_TELEMETRY",
    }}
    child.update(PATH=SYSTEM_PATH, HOME="/home/agent", TMPDIR="/tmp",
                 XDG_CACHE_HOME="/tmp/velocicyber-cache")
    return arguments, child


def check_isolation():
    """Exercise kernel support without contacting providers or running user code."""
    bubblewrap()
    with tempfile.TemporaryDirectory(prefix="velocicyber-sandbox-check-") as temporary:
        command, environment = isolate(["/usr/bin/true"], {}, workspace=Path(temporary))
        try:
            result = subprocess.run(command, env=environment, capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Closed filesystem isolation could not start. Run "
                               "./velocicyber.sh doctor and check Bubblewrap/user namespace support.") from exc
        if result.returncode:
            raise RuntimeError("Closed filesystem isolation is unavailable: Bubblewrap could not "
                               "create its filesystem/process namespaces. Check Linux user namespace "
                               "permissions. No agent command was started; access was not widened.")
