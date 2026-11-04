"""Bounded, resumable range transfer for the verified upstream engine archive."""

import concurrent.futures
import shutil
import threading
import time

import httpx


def download(client, url, target, total, notify, *, chunk_size=8 * 1024**2):
    parts = target.with_suffix(".parts")
    parts.mkdir(mode=0o700, exist_ok=True)
    ranges = [(start, min(total, start + chunk_size)) for start in range(0, total, chunk_size)]
    # Preserve bytes left by an interrupted sequential download.
    if target.exists():
        if target.stat().st_size > total:
            raise RuntimeError("Saved engine download exceeds the expected size.")
        with target.open("rb") as prefix:
            for start, end in ranges:
                content = prefix.read(end - start)
                if not content:
                    break
                part = parts / str(start)
                if not part.exists() or part.stat().st_size < len(content):
                    part.write_bytes(content)
    stop = threading.Event()
    deadline = time.monotonic() + 1800

    def transfer(bounds):
        start, end = bounds
        part = parts / str(start)
        for attempt in range(3):
            have = part.stat().st_size if part.exists() else 0
            if have == end - start:
                return
            if have > end - start:
                raise RuntimeError("Saved engine range exceeds its expected size.")
            if stop.is_set() or time.monotonic() > deadline:
                raise RuntimeError("Engine download interrupted or timed out; saved parts retained.")
            offset = start + have
            try:
                with client.stream("GET", url, headers={
                    "Range": f"bytes={offset}-{end - 1}", "Accept-Encoding": "identity",
                }) as response:
                    response.raise_for_status()
                    expected = f"bytes {offset}-{end - 1}/{total}"
                    if response.status_code != 206 or response.headers.get("content-range") != expected:
                        raise RuntimeError("Unexpected HTTP range response; refusing unverified bytes.")
                    with part.open("ab") as stream:
                        for chunk in response.iter_bytes(256 * 1024):
                            if stop.is_set() or time.monotonic() > deadline:
                                raise RuntimeError("Engine transfer stopped; saved parts retained.")
                            have += len(chunk)
                            if have > end - start:
                                raise RuntimeError("Engine range response exceeded its expected size.")
                            stream.write(chunk)
                if have == end - start:
                    return
            except httpx.HTTPError:
                if attempt == 2:
                    raise
        raise RuntimeError("Engine transfer was incomplete; saved parts retained.")

    last_percent = -1
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(transfer, bounds) for bounds in ranges}
        try:
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                percent = sum(path.stat().st_size for path in parts.iterdir()) * 100 // total
                if percent != last_percent:
                    notify(f"Download: {percent}%")
                    last_percent = percent
        finally:
            stop.set()
            for future in pending:
                future.cancel()
    assembled = target.with_suffix(".assembling")
    with assembled.open("wb") as stream:
        for start, end in ranges:
            part = parts / str(start)
            if part.stat().st_size != end - start:
                raise RuntimeError("Incomplete engine part; refusing assembly.")
            with part.open("rb") as source:
                shutil.copyfileobj(source, stream)
    assembled.replace(target)
    # The caller must verify the full upstream SHA256 before extracting or executing.
