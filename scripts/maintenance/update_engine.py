#!/usr/bin/env python3
"""Run with .venv/bin/python scripts/maintenance/update_engine.py; approval is always interactive."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velocicyber.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--root", str(ROOT), "update-engine", *sys.argv[1:]]
    main()
