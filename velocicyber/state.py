import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path


class GPUStatus(StrEnum):
    STOPPED = "STOPPED"
    STOPPING = "STOPPING"
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"


@dataclass
class State:
    gpu_status: GPUStatus = GPUStatus.STOPPED
    instance_id: str = ""
    instance_ip: str = ""
    instance_type: str = ""
    region: str = ""
    gpu_count: int = 0
    total_vram_gib: int = 0
    model_ready: bool = False
    model_revision: str = ""
    model_context_window: int = 0
    model_context_model: str = ""
    model_stage: str = ""
    download_bytes: int = 0
    download_total: int = 0
    active_provider: str = "api"
    api_health: str = "unverified"
    sandbox_open: bool = False
    collection: bool = False
    allow_compaction: bool = True
    upload_approved: bool = False
    local_url: str = "http://127.0.0.1:18000/v1"
    private_target: str = "gpu"
    pi_ready: bool = False
    pi_url: str = "http://127.0.0.1:18001/v1"
    pi_stage: str = ""
    pi_revision: str = ""
    pi_context_window: int = 16384
    launch_uncertain: bool = False
    sentinel_enabled: bool = False
    sentinel_ready: bool = False
    sentinel_stage: str = "OFF"
    sentinel_model: str = "qwen"

    def persist(self, path: Path):
        # Persist only resource identity. Preferences use the user-approved defaults.
        fields = (
            "instance_id",
            "instance_ip",
            "instance_type",
            "region",
            "gpu_count",
            "total_vram_gib",
            "launch_uncertain",
        )
        data = {k: v for k, v in asdict(self).items() if k in fields}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.chmod(0o600)
        tmp.replace(path)

    @classmethod
    def restore(cls, path: Path):
        state = cls()
        if path.exists():
            data = json.loads(path.read_text())
            for key in (
                "instance_id",
                "instance_ip",
                "instance_type",
                "region",
                "gpu_count",
                "total_vram_gib",
                "launch_uncertain",
            ):
                if key in data:
                    setattr(state, key, data[key])
            if state.instance_id or state.launch_uncertain:
                state.gpu_status = GPUStatus.INITIALIZED
        return state
