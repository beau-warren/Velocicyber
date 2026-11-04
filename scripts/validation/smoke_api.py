"""One explicit, bounded request to the configured API model; no tool execution or data collection."""

import json
from pathlib import Path

import httpx

from velocicyber.config import Settings

settings = Settings.load(Path(__file__).resolve().parents[2])
with httpx.Client(timeout=60) as client:
    response = client.post(
        settings.api_url + "/chat/completions",
        headers={"Authorization": "Bearer " + settings.require_secret("ADVERSARIAL_API")},
        json={
            "model": settings.api_model,
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
