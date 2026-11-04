"""Opt-in Linux desktop shortcuts using the installation's bundled icon."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def desktop_directory():
    """Respect localized XDG desktop locations and explicitly disabled desktops."""
    if not sys.platform.startswith("linux"):
        return None
    home = Path.home().resolve()
    if command := shutil.which("xdg-user-dir"):
        try:
            result = subprocess.run(
                [command, "DESKTOP"], capture_output=True, text=True, timeout=3, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        else:
            if result.returncode == 0:
                value = result.stdout.rstrip("\r\n")
                directory = Path(value)
                if not value or not directory.is_absolute():
                    return None
                if directory.resolve() == home or not directory.is_dir():
                    return None
                return directory
    fallback = home / "Desktop"
    return fallback if fallback.is_dir() else None


def _checked(value):
    value = str(value)
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Desktop launcher paths cannot contain control characters.")
    return value


def _value(value):
    return _checked(value).replace("\\", "\\\\").replace(" ", "\\s")


def _argument(value):
    # Desktop entries have a string-escape layer followed by Exec argument parsing.
    value = _checked(value).replace("%", "%%")
    value = "".join("\\" + char if char in '\\"`$' else char for char in value)
    return '"' + value.replace("\\", "\\\\") + '"'


def install_shortcut(root, directory):
    """Create a launcher without replacing another installation's shortcut."""
    if not sys.platform.startswith("linux"):
        raise ValueError("Desktop shortcuts currently support Linux desktops.")
    root, directory = Path(root).resolve(), Path(directory)
    launcher, icon = root / "velocicyber.sh", root / "assets/velocicyber.svg"
    if not launcher.is_file() or not icon.is_file():
        raise ValueError("This installation is missing its launcher or desktop icon.")
    if not directory.is_dir():
        raise ValueError("The configured desktop folder is unavailable.")
    content = (
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Application\n"
        "Name=Velocicyber\n"
        "Comment=Code with local and API models\n"
        f"Exec=/bin/sh {_argument(launcher)}\n"
        "TryExec=/bin/sh\n"
        f"Path={_value(root)}\n"
        f"Icon={_value(icon)}\n"
        "Terminal=true\n"
        "Categories=Development;\n"
        "Keywords=Velocicyber;Coding;AI;\n"
        "StartupNotify=false\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=".velocicyber-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o755)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        for number in range(1, 101):
            name = "Velocicyber" if number == 1 else f"Velocicyber ({number})"
            destination = directory / (name + ".desktop")
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if not destination.is_symlink() and destination.is_file():
                    if destination.read_bytes() == content:
                        destination.chmod(0o755)
                        return destination
                continue
            return destination
        raise ValueError("Too many existing Velocicyber shortcuts; remove an unused one first.")
    finally:
        Path(temporary).unlink(missing_ok=True)
