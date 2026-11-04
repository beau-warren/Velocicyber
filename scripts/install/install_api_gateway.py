"""Install the pinned provider client locally; refresh its public production policy."""

import io
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path

COMMIT = "351aa8b7263a935b9c0dd36a9d857b1966d4e195"


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / ".runtime/adverserial-confidential"
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / ("confidential-sdk-" + COMMIT)
    if not source.is_dir():
        url = f"https://codeload.github.com/AdverserialAI/confidential-sdk/tar.gz/{COMMIT}"
        with urllib.request.urlopen(url, timeout=60) as response:
            archive = response.read(10 * 1024 * 1024 + 1)
        if len(archive) > 10 * 1024 * 1024:
            raise ValueError("Unexpected SDK archive size.")
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(folder, filter="data")
    python = folder / "venv/bin/python"
    if not python.is_file():
        subprocess.run(["python3", "-m", "venv", str(folder / "venv")], check=True)
    subprocess.run([str(python), "-m", "pip", "install", str(source / "python"),
                    str(source / "gateway")], check=True)
    subprocess.run(["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
                   cwd=source / "typescript", check=True)
    subprocess.run(["npm", "run", "build"], cwd=source / "typescript", check=True)
    with urllib.request.urlopen("https://verify.adverserial.ai/policies/production.json",
                                timeout=30) as response:
        policy = json.load(response)
    if policy.get("status") != "active" or not policy.get("receipt_keys"):
        raise ValueError("No active production policy; client not enabled.")
    (folder / "production.json").write_text(json.dumps(policy, indent=2) + "\n")
    (folder / "source.json").write_text(json.dumps({"commit": COMMIT,
        "directory": source.name}, indent=2) + "\n")
    print("Confidential API client installed. No inference request was sent.")


if __name__ == "__main__":
    main()
