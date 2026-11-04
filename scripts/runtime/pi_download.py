"""Bounded parallel HTTP downloads with durable parts and a final pinned checksum."""

import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

from scripts.runtime.serve_model import file_sha256


class DownloadTransportError(RuntimeError):
    """A transient network failure; verified partial bytes may be resumed."""


def response_range(headers):
    final = {}
    for line in headers.read_text().splitlines():
        if line.startswith("HTTP/"):
            final = {"status": line.split()[1]}
        elif ":" in line:
            key, value = line.split(":", 1)
            final[key.lower()] = value.strip()
    return final


def download(cfg, folder, report, *, workers=4, chunk_bytes=16 * 1024**2):
    destination = folder / "weights" / cfg["revision"] / Path(cfg["filename"]).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        report("verifying")
        if (
            destination.stat().st_size == cfg["weight_bytes"]
            and file_sha256(destination) == cfg["sha256"]
        ):
            return destination
        raise RuntimeError("Existing Pi model checksum mismatch; inspect it before retrying.")
    parts = destination.with_suffix(".parts")
    parts.mkdir(exist_ok=True)
    manifest = parts / "manifest.json"
    identity = {k: cfg[k] for k in ("repo", "revision", "filename", "sha256", "weight_bytes")}
    identity["chunk_bytes"] = chunk_bytes
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError("Download parts belong to a different model or chunk profile.")
    manifest.write_text(json.dumps(identity))
    total = cfg["weight_bytes"]
    ranges = [(i, min(i + chunk_bytes, total)) for i in range(0, total, chunk_bytes)]

    # Import the durable contiguous prefix from the former sequential downloader.
    prefix = destination.with_suffix(".partial")
    if prefix.exists():
        if prefix.stat().st_size > total:
            raise RuntimeError("Saved prefix exceeds the pinned model size.")
        with prefix.open("rb") as source:
            for start, end in ranges:
                block = source.read(end - start)
                if not block:
                    break
                part = parts / f"{start:012d}.part"
                if not part.exists() or part.stat().st_size < len(block):
                    part.write_bytes(block)

    def retained():
        return sum(p.stat().st_size for p in parts.glob("*.part"))

    initial, started = retained(), time.monotonic()

    def progress():
        current = retained()
        rate = max(0, current - initial) / max(1, time.monotonic() - started)
        report(
            "download_progress",
            bytes=current,
            total=total,
            bytes_per_second=round(rate),
            eta_seconds=round((total - current) / rate) if rate else None,
            method="http_range",
            workers=workers,
        )

    base = f"https://huggingface.co/{cfg['repo']}/resolve/{cfg['revision']}/" + quote(
        cfg["filename"], safe="/"
    )

    def transfer(bounds):
        start, end = bounds
        part = parts / f"{start:012d}.part"
        temporary, headers = part.with_suffix(".tmp"), part.with_suffix(".headers")

        def adopt(offset):
            if not temporary.exists() or not headers.exists():
                return False
            final = response_range(headers)
            valid = (
                final.get("status") == "206"
                and final.get("content-range") == f"bytes {offset}-{end - 1}/{total}"
                and temporary.stat().st_size <= end - offset
            )
            if valid:
                with part.open("ab") as out, temporary.open("rb") as src:
                    shutil.copyfileobj(src, out)
                    out.flush()
                    os.fsync(out.fileno())
            temporary.unlink(missing_ok=True)
            headers.unlink(missing_ok=True)
            return valid

        # A previous process may have exited during a valid partial HTTP response.
        adopt(start + (part.stat().st_size if part.exists() else 0))
        for attempt in range(4):
            offset = start + (part.stat().st_size if part.exists() else 0)
            if offset == end:
                return
            if offset > end:
                raise RuntimeError("Saved download part exceeds its expected range.")
            requested = f"{offset}-{end - 1}"
            result = subprocess.run(
                [
                    "curl",
                    "--disable",
                    "--http1.1",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "180",
                    "--speed-time",
                    "45",
                    "--speed-limit",
                    "16384",
                    "--range",
                    requested,
                    "--max-filesize",
                    str(end - offset),
                    "--dump-header",
                    str(headers),
                    "--output",
                    str(temporary),
                    "--write-out",
                    "%{http_code}",
                    base + "?download=true&range=" + requested,
                ],
                capture_output=True,
                text=True,
            )
            valid = adopt(offset)
            if (
                # A connection can fail after HF redirects to its object store.
                # curl then reports the last redirect, rather than HTTP 000.
                result.stdout.strip() in {"000", "301", "302", "303", "307", "308"}
                and result.returncode in (5, 6, 7, 28, 35, 52, 55, 56, 92)
            ):
                if attempt == 3:
                    raise DownloadTransportError(
                        f"Download connection failed ({result.stdout.strip()}, "
                        f"curl {result.returncode}); saved parts retained."
                    )
                time.sleep(2**attempt)
                continue
            if result.stdout.strip() != "206" or not valid:
                raise RuntimeError(
                    f"Rejected unexpected HTTP range response ({result.stdout.strip()}, "
                    f"curl {result.returncode}); saved parts retained."
                )
            if result.returncode == 0 and part.stat().st_size == end - start:
                return
            if result.returncode in (18, 28, 56, 92) and attempt == 3:
                raise DownloadTransportError(
                    f"Download stalled (curl {result.returncode}); saved parts retained for resume."
                )
            if result.returncode not in (18, 28, 56, 92):
                raise RuntimeError(
                    f"Download stalled (curl {result.returncode}); saved parts retained for resume."
                )
            time.sleep(2**attempt)

    progress()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(transfer, bounds) for bounds in ranges}
        try:
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=30, return_when=concurrent.futures.FIRST_EXCEPTION
                )
                progress()
                for future in done:
                    future.result()
        except BaseException:
            for future in pending:
                future.cancel()
            # In-flight workers may retain more bytes before exiting. Publish the
            # final retained amount instead of leaving a stale progress reading.
            pool.shutdown(wait=True, cancel_futures=True)
            progress()
            raise
    report("verifying", bytes=total, total=total)
    assembled = destination.with_suffix(".assembling")
    with assembled.open("wb") as out:
        for start, end in ranges:
            part = parts / f"{start:012d}.part"
            if part.stat().st_size != end - start:
                raise RuntimeError("Incomplete download part; model was not activated.")
            with part.open("rb") as source:
                shutil.copyfileobj(source, out, length=8 * 1024**2)
        out.flush()
        os.fsync(out.fileno())
    if file_sha256(assembled) != cfg["sha256"]:
        raise RuntimeError(
            "Pi model checksum mismatch; model was not activated. Download parts retained for inspection."
        )
    assembled.replace(destination)
    shutil.rmtree(parts)
    prefix.unlink(missing_ok=True)
    return destination
