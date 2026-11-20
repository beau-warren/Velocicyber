"""Transport rules for credential-bearing model inference requests."""

from ipaddress import ip_address

import httpx


def is_loopback_url(url):
    host = httpx.URL(url).host
    if host in {"localhost", "localhost."}:
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or bool(mapped and mapped.is_loopback)


def validate_inference_url(value):
    """Reject unsafe URLs before opening a connection; never echo their contents."""
    try:
        if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError
        url = httpx.URL(value)
        if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
                or (url.port is not None and not 1 <= url.port <= 65535)):
            raise ValueError
    except (ValueError, httpx.InvalidURL):
        raise ValueError(
            "Invalid inference endpoint. Set API_BASE_URL in Modules.txt to an absolute "
            "HTTP(S) URL with a valid port and no embedded username or password. "
            "Keep API keys in .env."
        ) from None
    if url.scheme == "http":
        if not is_loopback_url(url):
            raise ValueError(
                "Remote API endpoints require HTTPS. Change API_BASE_URL in Modules.txt "
                "to the provider's HTTPS endpoint. HTTP is allowed only on loopback "
                "for local inference or an SSH tunnel."
            )
        if url.host in {"localhost", "localhost."}:
            # Keep cleartext local even if host lookup is overridden on this machine.
            url = url.copy_with(host="127.0.0.1")
    return str(url)
