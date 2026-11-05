"""Integrity checks shared by the Pi converter and normal model startup."""

import hashlib
import os
from pathlib import Path


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


IDENTITY_KEYS = ("repo", "revision", "filename", "source_files", "runtime", "conversion")


def local_file(folder, name, *, allow_internal_symlink=False):
    path = Path(name)
    if not path.is_absolute():
        path = folder / path
    if ((path.is_symlink() and not allow_internal_symlink)
            or not path.resolve().is_relative_to(folder.resolve())):
        raise ValueError("Pi conversion artifact escapes its installation.")
    return path


def valid_receipt(cfg, folder, saved, *, verify_model=True):
    if any(saved.get(key) != cfg[key] for key in IDENTITY_KEYS):
        return False
    try:
        model = local_file(folder, saved["model_path"])
        binary = local_file(folder, saved["binary_path"])
        if (not model.is_file() or not binary.is_file()
                or not os.access(binary, os.X_OK)
                or model.stat().st_size != saved["converted_weight_bytes"]
                or model.stat().st_mtime_ns != saved["model_mtime_ns"]
                or len(saved["model_sha256"]) != 64):
            return False
        if verify_model and file_hash(model) != saved["model_sha256"]:
            return False
        files, stats = saved["runtime_files"], saved["runtime_stats"]
        if str(binary.relative_to(folder)) not in files or files.keys() != stats.keys():
            return False
        for name, checksum in files.items():
            path = local_file(folder, name, allow_internal_symlink=True)
            if not path.is_file() or [path.stat().st_size, path.stat().st_mtime_ns] != stats[name]:
                return False
            if verify_model and file_hash(path) != checksum:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True

