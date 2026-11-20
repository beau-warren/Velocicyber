#!/bin/sh
# Install into this checkout only; no system packages, GPUs, or datasets are changed.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
printf '%s\n' 'Python environment ready. No model weights have been downloaded.'
printf '%s\n' 'Run ./velocicyber.sh for first-run setup, or ./velocicyber.sh --demo for the offline UI.'
if ! command -v bwrap >/dev/null 2>&1; then
    printf '%s\n' 'Closed workspace access also needs Bubblewrap and Linux user namespaces.'
    printf '%s\n' 'On Raspberry Pi OS/Debian: sudo apt install bubblewrap'
    printf '%s\n' 'Then run ./velocicyber.sh doctor to verify isolation.'
fi
