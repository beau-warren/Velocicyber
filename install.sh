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
