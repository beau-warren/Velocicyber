#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$PROJECT_DIR/.venv/bin/python" ]; then
    printf '%s\n' 'Run ./install.sh first to create the local Python environment.' >&2
    exit 1
fi
cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/python" -m velocicyber --root "$PROJECT_DIR" "$@"
