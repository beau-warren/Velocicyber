"""Read a bounded GGUF header without downloading model tensors."""

import struct
from urllib.parse import quote

import httpx


def metadata_from_prefix(data):
    if data[:4] != b"GGUF":
        raise ValueError("Selected model does not have a GGUF header.")
    offset = 4
    formats = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }

    def number(fmt):
        nonlocal offset
        value = struct.unpack_from("<" + fmt, data, offset)[0]
        offset += struct.calcsize("<" + fmt)
        return value

    def string():
        nonlocal offset
        size = number("Q")
        if offset + size > len(data):
            raise EOFError
        value = data[offset : offset + size].decode("utf-8")
        offset += size
        return value

    def read_value(kind, depth=0):
        if depth > 1:
            raise ValueError("Unsupported nested GGUF metadata.")
        if kind == 8:
            return string()
        if kind in formats:
            return number(formats[kind])
        if kind == 9:
            item_type, count = number("I"), number("Q")
            for _ in range(count):
                read_value(item_type, depth + 1)
            return None
        raise ValueError("Unknown GGUF metadata type.")

    version = number("I")
    if version not in (2, 3):
        raise ValueError(f"Unsupported GGUF version {version}.")
    number("Q")  # tensor count
    count = number("Q")
    metadata = {}
    try:
        for _ in range(count):
            key = string()
            value = read_value(number("I"))
            if value is not None:
                metadata[key] = value
    except (EOFError, struct.error):
        pass  # Tokenizer arrays typically extend beyond the bounded prefix.
    if "general.architecture" not in metadata:
        raise ValueError("GGUF architecture was not found in the first 64 KiB.")
    return metadata


def inspect_header(repo, revision, filename, token, *, transport=None):
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(filename, safe='/')}"
    headers = {"Range": "bytes=0-65535"}
    if token:
        headers["Authorization"] = "Bearer " + token
    with httpx.Client(timeout=30, follow_redirects=True, transport=transport) as client:
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes(8192):
                data.extend(chunk[: 65536 - len(data)])
                if len(data) >= 65536:
                    break
    return metadata_from_prefix(data)
