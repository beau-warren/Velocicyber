"""Validated, atomic edits to the personal scope file (never a Git artifact)."""

import hashlib
import os
import stat
import uuid
from contextlib import contextmanager

MAX_SCOPE_BYTES = 6000
NAME = "SCOPE.MD"


def validate_scope(body):
    if not body.strip() or len(body) > MAX_SCOPE_BYTES:
        raise ValueError("SCOPE.MD must contain 1–6,000 UTF-8 bytes; no scope is silently truncated.")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("SCOPE.MD must be UTF-8 text.") from None
    if "EXAMPLE ONLY" in text:
        raise ValueError("Replace the example with your actual scope before enabling Sentinel.")
    return text


@contextmanager
def scope_directory(root, *, create=False):
    folder = root / "SCOPE"
    if create:
        folder.mkdir(mode=0o700, exist_ok=True)
    try:
        fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        yield None
        return
    except OSError:
        raise ValueError("SCOPE/SCOPE.MD must be a regular file inside the project.") from None
    try:
        yield fd
    finally:
        os.close(fd)


def read_at(fd):
    if fd is None:
        return None
    try:
        file_fd = os.open(NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    except FileNotFoundError:
        return None
    except OSError:
        raise ValueError("SCOPE/SCOPE.MD must be a regular file inside the project.") from None
    if not stat.S_ISREG(os.fstat(file_fd).st_mode):
        os.close(file_fd)
        raise ValueError("SCOPE/SCOPE.MD must be a regular file inside the project.")
    with os.fdopen(file_fd, "rb") as stream:
        body = stream.read(MAX_SCOPE_BYTES + 1)
    if len(body) > MAX_SCOPE_BYTES:
        raise ValueError("SCOPE.MD exceeds 6,000 bytes; shorten it before opening the editor.")
    return body


def read_scope(root):
    """Return an exact editing snapshot, or None when the scope does not exist."""
    with scope_directory(root) as fd:
        return read_at(fd)


def load_scope(root):
    body = read_scope(root)
    if body is None:
        raise ValueError("Place your approved scope in SCOPE/SCOPE.MD before turning Sentinel ON.")
    return validate_scope(body), hashlib.sha256(body).hexdigest()


def save_scope(root, text, *, expected):
    body = text.encode("utf-8")
    validate_scope(body)
    with scope_directory(root, create=True) as fd:
        if read_at(fd) != expected:
            raise ValueError("SCOPE.MD changed outside this editor. Cancel and reopen before saving.")
        temporary = ".scope-" + uuid.uuid4().hex + ".tmp"
        try:
            file_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=fd)
            with os.fdopen(file_fd, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            # Check again before replacement; never follow a substituted scope link.
            if read_at(fd) != expected:
                raise ValueError("SCOPE.MD changed outside this editor. Cancel and reopen before saving.")
            directory = os.stat(root / "SCOPE", follow_symlinks=False)
            opened = os.fstat(fd)
            if (directory.st_dev, directory.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError("SCOPE directory changed. Cancel and reopen before saving.")
            os.replace(temporary, NAME, src_dir_fd=fd, dst_dir_fd=fd)
            os.fsync(fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError:
                pass
