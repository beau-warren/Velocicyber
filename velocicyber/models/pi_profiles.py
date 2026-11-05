"""Pinned CPU profiles; each model has separate files and shares one server lock."""

from dataclasses import dataclass

PI_MODEL_REPO = "mtrpires/Qwen3.5-35B-A3B-IQK-RPi5-16GB"
BUGTRACE_PI_REPO = "terrorswift/BugTraceAI-Apex-G4-26B-APEX-GGUF"
LFM3_PI_REPO = "Blackfrost-AI/LFM3-1.7B-DERISKED"
PI_MODEL_NOTICE = "No GPUs Needed! Can run locally without GPUs."


@dataclass(frozen=True)
class PiProfile:
    lock_file: str
    directory: str
    context_window: int


PI_PROFILES = {
    LFM3_PI_REPO: PiProfile("config/locks/pi-lfm3.lock.json", "pi-models/lfm3-derisked", 16384),
    PI_MODEL_REPO: PiProfile("config/locks/pi-model.lock.json", "pi-model", 16384),
    BUGTRACE_PI_REPO: PiProfile("config/locks/pi-bugtrace.lock.json", "pi-models/bugtrace-apex", 8192),
}
