#!/bin/sh
# Open the harness in a terminal large enough for the conversation and dashboard.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
exec lxterminal --title=Velocicyber --geometry=120x36 -e "$PROJECT_DIR/velocicyber.sh"
