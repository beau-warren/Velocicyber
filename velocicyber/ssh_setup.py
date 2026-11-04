"""Optional Lambda SSH setup: save a name/path, never key contents, in configuration."""

import os
import re
import tempfile
from pathlib import Path

SKIP = {"skip", "skip for now"}


def private_key_path(root, value):
    """Use a project-local key, or import an explicitly selected external key."""
    root = Path(root).resolve()
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("Choose a key path without control characters.")
    path = Path(value).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise ValueError("Choose an existing private key file, or type skip for now.")
    with path.open("rb") as stream:
        data = stream.read(64 * 1024 + 1)
    if len(data) > 64 * 1024 or not re.match(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----", data.lstrip()):
        raise ValueError("Choose the private SSH key (.pem or OpenSSH), not a public key or another file.")
    if path.is_relative_to(root):
        path.chmod(0o600)
        return str(path.relative_to(root))
    folder = root / "ssh-keys"
    if folder.is_symlink():
        raise ValueError("ssh-keys must be a directory inside this installation, not a symlink.")
    folder.mkdir(mode=0o700, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="lambda-", suffix=".pem", dir=folder)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return str(Path(temporary).relative_to(root))


def lambda_ssh_options(root, options, input_fn=input):
    print("\nOptional Lambda SSH setup. Enter skips missing values; type skip to leave this section.")
    print("Use the SSH key name registered in your Lambda account. Do not paste private-key contents.")
    current = options.get("LAMBDA_SSH_KEY_NAME", "")
    while True:
        suffix = f" (Enter keeps {current})" if current else " (Enter skips for now)"
        name = input_fn("Lambda SSH key name" + suffix + ": ").strip()
        if name.lower() in SKIP:
            return {}
        name = name or current
        if not name:
            return {}
        if any(ord(c) < 32 or ord(c) == 127 for c in name) or "PRIVATE KEY" in name or "-----BEGIN" in name:
            print("Enter only the registered key name; no key contents or control characters.")
            continue
        break
    choices = []
    folder = Path(root) / "ssh-keys"
    if folder.is_dir() and not folder.is_symlink():
        choices = sorted(p for p in folder.iterdir() if p.is_file() and p.name != "README.md")
    if choices:
        print("Keys already in this installation:")
        for number, path in enumerate(choices, 1):
            print(f"  {number}: {path.relative_to(root)}")
    print("Enter a key path or a number above. Files outside this installation are copied into ssh-keys with 0600 permissions.")
    configured = options.get("SSH_PRIVATE_KEY", "")
    default = configured if configured and (Path(root) / configured).is_file() else ""
    while True:
        suffix = f" (Enter keeps {default})" if default else " (Enter skips for now)"
        value = input_fn("Private SSH key file" + suffix + ": ").strip()
        if value.lower() in SKIP or not (value or default):
            print("SSH name saved. Select a private key through setup before launching a GPU.")
            return {"LAMBDA_SSH_KEY_NAME": name}
        value = value or default
        if value.isdigit() and 1 <= int(value) <= len(choices):
            value = str(choices[int(value) - 1])
        try:
            relative = private_key_path(root, value)
        except (OSError, ValueError) as exc:
            print(str(exc) if isinstance(exc, ValueError) else "The key file could not be read or saved; choose another file.")
            continue
        print("SSH key selected: " + relative + ". No Lambda request was made.")
        return {"LAMBDA_SSH_KEY_NAME": name, "SSH_PRIVATE_KEY": relative}
