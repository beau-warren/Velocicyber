"""Offline regressions for API transport, redirects, and local inference."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from scripts.validation.smoke_api import main as smoke_api
from velocicyber.engine.capture import CaptureProxy
from velocicyber.providers.api import api_connection
from velocicyber.providers.transport import validate_inference_url

ANSWER = {"choices": [{"finish_reason": "stop", "message": {
    "role": "assistant", "content": "VELOCICYBER_OK",
}}]}


@pytest.mark.parametrize("url", [
    "http://provider.example/v1", "http://api.adverserial.ai/v1",
    "http://192.168.10.2:8000/v1", "http://10.0.0.1:8000/v1",
    "http://0.0.0.0:8000/v1", "http://[::]:8000/v1", "http://[fd00::1]/v1",
    "http://localhost.provider.example/v1", "http://127.0.0.1.provider.example/v1",
    "http://[::ffff:192.0.2.1]/v1", "http://127.1/v1", "http://2130706433/v1",
    "http://0x7f000001/v1", "http://local-provider.example/v1",
])
def test_remote_http_rejected_before_client_or_gateway(settings, monkeypatch, url):
    settings.api_url = url
    monkeypatch.setattr("httpx.Client", lambda *a, **kw: pytest.fail("HTTP client created"))
    monkeypatch.setattr("velocicyber.providers.api.ConfidentialGateway.start",
                        lambda _: pytest.fail("Gateway launched for rejected URL"))
    with pytest.raises(ValueError, match="require HTTPS"):
        with api_connection(settings):
            pytest.fail("Unsafe API endpoint accepted")
    with pytest.raises(ValueError, match="require HTTPS"):
        CaptureProxy(url, "private-fixture-key")
    with pytest.raises(ValueError, match="require HTTPS"):
        smoke_api(settings)


@pytest.mark.parametrize("url", [
    "", "/v1", "provider.example/v1", "https:///v1", "ftp://provider.example/v1",
    "https://private-user:private-password@provider.example/v1",
    "http://private-user:private-password@localhost:8080/v1",
    "https://provider.example:invalid/v1", "https://provider.example:65536/v1",
    "http://127.0.0.1:0/v1", "http://[::1/v1",
    "http://local\nhost/v1", " https://provider.example/v1",
])
def test_invalid_urls_fail_without_disclosing_their_contents(url):
    with pytest.raises(ValueError, match="Invalid inference endpoint") as caught:
        validate_inference_url(url)
    assert "private-user" not in str(caught.value)
    assert "private-password" not in str(caught.value)
    assert "provider.example" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("url,expected", [
    ("https://provider.example/v1", "https://provider.example/v1"),
    ("https://192.0.2.1:8443/v1", "https://192.0.2.1:8443/v1"),
    ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
    ("http://127.0.0.2:8000/v1", "http://127.0.0.2:8000/v1"),
    ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
    ("http://[::ffff:127.0.0.1]:8000/v1", "http://[::ffff:127.0.0.1]:8000/v1"),
    ("http://LOCALHOST:8000/v1", "http://127.0.0.1:8000/v1"),
    ("http://localhost.:8000/v1", "http://127.0.0.1:8000/v1"),
    ("https://localhost:8000/v1", "https://localhost:8000/v1"),
])
def test_supported_endpoints_keep_secure_or_loopback_transport(settings, monkeypatch, url, expected):
    settings.api_url = url
    monkeypatch.setattr("velocicyber.providers.api.ConfidentialGateway.start",
                        lambda _: pytest.fail("Custom endpoint must not start the gateway"))
    with api_connection(settings) as connection:
        assert connection == expected


def test_adverserial_https_keeps_confidential_gateway_and_cleanup(settings, monkeypatch):
    events = []

    def start(_):
        events.append("start")
        return "http://127.0.0.1:8765/v1"

    monkeypatch.setattr("velocicyber.providers.api.ConfidentialGateway.start", start)
    monkeypatch.setattr("velocicyber.providers.api.ConfidentialGateway.stop", lambda _: events.append("stop"))
    with api_connection(settings) as connection:
        assert connection == "http://127.0.0.1:8765/v1"
        assert events == ["start"]
    assert events == ["start", "stop"]


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_inference_does_not_follow_remote_redirects(status):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, headers={"location": "http://provider.example/v1"})

    with CaptureProxy("https://provider.example/v1", "private-fixture-key",
                      transport=httpx.MockTransport(handler)) as proxy:
        response = httpx.post(proxy.url + "/chat/completions", json={"messages": []},
                              headers={"Authorization": "Bearer " + proxy.local_key}, trust_env=False)
        assert response.status_code == status
    assert len(seen) == 1
    assert seen[0].url.scheme == "https"
    assert seen[0].headers["Authorization"] == "Bearer private-fixture-key"


@pytest.mark.parametrize("entrypoint", ["harness", "smoke"])
def test_loopback_requests_ignore_environment_proxy(settings, monkeypatch, entrypoint):
    received = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            received.append((self.headers["Authorization"],
                             json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            body = json.dumps(ANSWER).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    for key in ["NO_PROXY", "no_proxy"]:
        monkeypatch.delenv(key, raising=False)
    settings.api_url = f"http://localhost:{server.server_port}/v1"
    try:
        if entrypoint == "smoke":
            smoke_api(settings)
        else:
            with CaptureProxy(settings.api_url, settings.secrets["ADVERSARIAL_API"]) as proxy:
                response = httpx.post(proxy.url + "/chat/completions", json={"messages": []},
                                      headers={"Authorization": "Bearer " + proxy.local_key}, trust_env=False)
                assert response.json() == ANSWER
        assert len(received) == 1
        assert received[0][0] == "Bearer " + settings.secrets["ADVERSARIAL_API"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("status", [200, 307])
def test_smoke_api_uses_https_without_following_redirects(settings, monkeypatch, status):
    settings.api_url = "https://provider.example/v1"
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json=ANSWER, headers={"location": "http://provider.example/v1"})

    real_client = httpx.Client
    monkeypatch.setattr("httpx.Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    if status == 200:
        smoke_api(settings)
    else:
        with pytest.raises(SystemExit):
            smoke_api(settings)
    assert len(seen) == 1
    assert seen[0].url.scheme == "https"
    assert seen[0].headers["Authorization"] == "Bearer " + settings.secrets["ADVERSARIAL_API"]
