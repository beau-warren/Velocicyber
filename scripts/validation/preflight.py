"""Explicit setup check (may save the matching SSH key name). No launch, training, or dataset upload calls."""

import argparse
import json
import subprocess
from pathlib import Path

from velocicyber.config import Settings
from velocicyber.providers.lambda_cloud import MEMORY_CHOICES, GPUManager, LambdaClient, total_vram
from velocicyber.state import State

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True, help="GPU model repository to inspect")
args = parser.parse_args()
settings = Settings.load(root)
if args.model not in settings.gpu_model_choices:
    parser.error("Choose a configured GPU model repository from /local & cloud models.")
settings.local_model = args.model
key = (root / settings.options["SSH_PRIVATE_KEY"]).resolve()
if not key.is_relative_to(root):
    raise SystemExit("Private key must be within the workspace.")
public = subprocess.run(
    ["ssh-keygen", "-y", "-P", "", "-f", str(key)], capture_output=True, text=True, check=True
).stdout.split()[:2]
client = LambdaClient(settings.require_secret("LAMBDA_API"))
report = {"model": settings.local_model}
try:
    registered = client.request("GET", "ssh-keys")
    matches = [k["name"] for k in registered if k.get("public_key", "").split()[:2] == public]
    if len(matches) == 1:
        name = matches[0]
        report["ssh_key_name"] = name
        config = root / "Modules.txt"
        lines = config.read_text().splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.split("=", 1)[0].strip() == "LAMBDA_SSH_KEY_NAME":
                lines[i] = "LAMBDA_SSH_KEY_NAME=" + name + "\n"
        config.write_text("".join(lines))
    else:
        report["ssh_key_match"] = "No unique registered key matched the local private key."
    manager = GPUManager(settings, State(), client)
    try:
        spec = manager.inspect_model()
        report.update(
            model_revision=spec.revision,
            weight_bytes=spec.weight_bytes,
            minimum_vram_gib=spec.required_gib,
            tool_parser=spec.tool_parser,
        )
    except Exception as exc:
        report["model_check_error"] = type(exc).__name__ + ": " + str(exc).split("\n")[0]
    catalog = client.catalog()
    available = []
    for row in catalog.values():
        kind = row["instance_type"]
        memory = total_vram(kind)
        if memory in MEMORY_CHOICES:
            for region in row["regions_with_capacity_available"]:
                available.append(
                    {
                        "type": kind["name"],
                        "total_vram_gib": memory,
                        "gpu_count": kind["specs"]["gpus"],
                        "region": region["name"],
                        "region_description": region.get("description", ""),
                        "hourly_cents": kind["price_cents_per_hour"],
                    }
                )
    report["available_instances"] = available
    output = settings.runtime / "preflight.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.chmod(0o600)
    print(json.dumps(report, indent=2))
finally:
    client.close()
