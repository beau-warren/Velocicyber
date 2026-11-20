"""One explicit, bounded request to the configured API model; no tools or collection."""

import json
from pathlib import Path

import httpx

from velocicyber.config import Settings
from velocicyber.providers.api import api_connection
from velocicyber.providers.transport import is_loopback_url


def main(settings=None):
    settings = settings or Settings.load(Path(__file__).resolve().parents[2])
    with api_connection(settings) as url, httpx.Client(
        timeout=60, follow_redirects=False, trust_env=not is_loopback_url(url),
    ) as client:
        response = client.post(
            url.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + settings.require_secret("ADVERSARIAL_API")},
            json={
                "model": settings.api_profile[1],
                "messages": [{"role": "user", "content": "Reply with exactly VELOCICYBER_OK."}],
                "max_tokens": 128,
                "reasoning_effort": "low",
                "stream": False,
            },
        )
    print("HTTP", response.status_code)
    if response.is_success:
        result = response.json()
        choice = result["choices"][0]
        print("finish_reason:", choice.get("finish_reason"))
        print("answer:", choice["message"].get("content", ""))
        print("usage:", json.dumps(result.get("usage", {})))
    else:
        messages = {
            401: "API key not accepted.",
            402: "Provider reports payment required.",
            403: "API access denied.",
            429: "Provider reports a rate or quota limit.",
        }
        print(messages.get(response.status_code, "Provider request failed."))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
