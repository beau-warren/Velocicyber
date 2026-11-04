import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runtime import pi_download


@pytest.mark.parametrize("mode", ["success", "resume", "interrupted", "bad_range", "bad_hash"])
def test_parallel_download_preserves_bytes_and_checks_integrity(tmp_path, monkeypatch, mode):
    payload = b"GGUF0123456789abcdefghijklmnopqrstuv"
    cfg = {
        "repo": "fixture/model",
        "revision": "pinned",
        "filename": "model.gguf",
        "weight_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    folder = tmp_path / "weights" / "pinned"
    folder.mkdir(parents=True)
    if mode == "resume":
        (folder / "model.partial").write_bytes(payload[:13])
    calls = []
    failed_once = False

    def curl(args, **kwargs):
        nonlocal failed_once
        start, end = map(int, args[args.index("--range") + 1].split("-"))
        calls.append((start, end))
        data = payload[start : end + 1]
        code = 0
        if mode == "interrupted" and not failed_once:
            failed_once = True
            data = data[:3]
            code = 28
        if mode == "bad_hash":
            data = b"x" * len(data)
        Path(args[args.index("--output") + 1]).write_bytes(data)
        reported_start = start + 1 if mode == "bad_range" else start
        Path(args[args.index("--dump-header") + 1]).write_text(
            f"HTTP/1.1 206 Partial Content\nContent-Range: bytes {reported_start}-{end}/{len(payload)}\n"
        )
        return SimpleNamespace(returncode=code, stdout="206", stderr="")

    monkeypatch.setattr(pi_download.subprocess, "run", curl)
    monkeypatch.setattr(pi_download.time, "sleep", lambda _: None)
    events = []

    def report(stage, **values):
        events.append({"stage": stage, **values})

    if mode.startswith("bad_"):
        with pytest.raises(RuntimeError, match="range response|checksum mismatch"):
            pi_download.download(cfg, tmp_path, report, workers=2, chunk_bytes=8)
        assert not (folder / "model.gguf").exists()
        return
    result = pi_download.download(cfg, tmp_path, report, workers=2, chunk_bytes=8)
    assert result.read_bytes() == payload
    if mode == "resume":
        assert min(start for start, _ in calls) == 13
    if mode == "interrupted":
        assert any(start % 8 == 3 for start, _ in calls)
    assert events[-1]["stage"] == "verifying"
    assert events[-1]["bytes"] == len(payload)
    # A completed, verified model is reused without any network request.
    monkeypatch.setattr(pi_download.subprocess, "run", lambda *a, **k: pytest.fail("network"))
    assert pi_download.download(cfg, tmp_path, report, chunk_bytes=8) == result


def test_incompatible_part_manifest_is_rejected(tmp_path):
    parts = tmp_path / "weights" / "pinned" / "model.parts"
    parts.mkdir(parents=True)
    (parts / "manifest.json").write_text(json.dumps({"revision": "wrong"}))
    cfg = {
        "repo": "fixture/model",
        "revision": "pinned",
        "filename": "model.gguf",
        "sha256": "hash",
        "weight_bytes": 10,
    }
    with pytest.raises(RuntimeError, match="different model"):
        pi_download.download(cfg, tmp_path, lambda *a, **k: None)


@pytest.mark.parametrize("failures,curl_code,expected_calls", [
    (1, 28, 2), (9, 28, 4), (1, 6, 2), (9, 6, 4), (1, 52, 2), (1, 0, 1),
])
def test_redirect_connection_failure_retries_without_accepting_redirect_body(
    tmp_path, monkeypatch, failures, curl_code, expected_calls,
):
    payload = b"GGUFfixture"
    cfg = {"repo": "fixture/model", "revision": "pinned", "filename": "model.gguf",
           "weight_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    calls = []

    def curl(args, **kwargs):
        calls.append(args[args.index("--range") + 1])
        output = Path(args[args.index("--output") + 1])
        headers = Path(args[args.index("--dump-header") + 1])
        if len(calls) <= failures:
            output.write_bytes(b"redirect")
            headers.write_text("HTTP/1.1 302 Found\nContent-Length: 8\n")
            return SimpleNamespace(returncode=curl_code, stdout="302", stderr="timeout")
        output.write_bytes(payload)
        headers.write_text(f"HTTP/1.1 206 Partial Content\nContent-Range: bytes 0-{len(payload)-1}/{len(payload)}\n")
        return SimpleNamespace(returncode=0, stdout="206", stderr="")

    monkeypatch.setattr(pi_download.subprocess, "run", curl)
    monkeypatch.setattr(pi_download.time, "sleep", lambda _: None)
    events = []
    def report(stage, **values):
        events.append({"stage": stage, **values})
    if failures == 1 and curl_code in (6, 28, 52):
        assert pi_download.download(cfg, tmp_path, report, workers=1).read_bytes() == payload
    else:
        error = pi_download.DownloadTransportError if curl_code else RuntimeError
        with pytest.raises(error, match="302, curl"):
            pi_download.download(cfg, tmp_path, report, workers=1)
        assert not (tmp_path / "weights/pinned/model.gguf").exists()
        assert events[-1]["bytes"] == 0
    assert calls == [f"0-{len(payload)-1}"] * expected_calls
