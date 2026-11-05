"""Read project configuration without executing shell code or exposing credentials."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from velocicyber.models.pi_profiles import PI_MODEL_NOTICE as PI_MODEL_NOTICE
from velocicyber.models.pi_profiles import PI_MODEL_REPO as PI_MODEL_REPO
from velocicyber.models.pi_profiles import PI_PROFILES

API_MODELS = {
    "cyberkimi": ("CyberKimi", "lordx64/cyberkimi", 32768),
    "cyberglm": ("CyberGLM", "lordx64/cyberglm", 131072),
}


def read_assignments(path: Path) -> dict[str, str]:
    result = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key.strip()] = value
    return result


@dataclass
class Settings:
    root: Path
    local_model: str
    dataset_repo: str
    api_model: str = "lordx64/cyberkimi"
    api_url: str = "https://api.adverserial.ai/v1"
    secrets: dict[str, str] = field(default_factory=dict, repr=False)
    options: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path):
        root = root.resolve()
        modules = read_assignments(root / "Modules.example.txt")
        modules.update(read_assignments(root / "Modules.txt"))
        values = read_assignments(root / ".env")
        names = {"LAMBDA_API", "ADVERSARIAL_API", "HUGGINGFACE_API", "GITHUB_API", "GH_TOKEN", "GITHUB_TOKEN"}
        values.update({k: os.environ[k] for k in values.keys() | names if k in os.environ})
        return cls(
            root=root,
            # Configured repositories populate the picker; loading requires selection.
            local_model="",
            dataset_repo=modules.get("FINETUNING_REPO", ""),
            api_model=modules.get("API_MODEL", "lordx64/cyberkimi"),
            api_url=modules.get("API_BASE_URL", "https://api.adverserial.ai/v1"),
            secrets=values,
            options=modules,
        )

    @property
    def api_profile(self):
        if self.api_model in API_MODELS:
            return API_MODELS[self.api_model]
        return next(
            (profile for profile in API_MODELS.values() if profile[1] == self.api_model),
            (self.api_model, self.api_model, 32768),
        )

    @property
    def api_model_name(self):
        return self.api_profile[0]

    def is_api_model(self, model):
        return model in API_MODELS or model == self.api_model or any(
            profile[1] == model for profile in API_MODELS.values()
        )

    @property
    def model_choices(self):
        default = self.options.get("HUGGINGFACE_MODEL", self.local_model)
        return list(
            dict.fromkeys(
                model.strip()
                for model in [
                    default,
                    *self.options.get("LOCAL_MODEL_OPTIONS", "").split(","),
                    *self.pi_model_choices,
                ]
                if model.strip()
            )
        )

    @property
    def pi_model(self):
        repo = self.options.get("PI_LOCAL_MODEL", "")
        if repo and repo not in PI_PROFILES:
            raise ValueError("PI_LOCAL_MODEL requires a validated Raspberry Pi serving profile.")
        return repo

    @property
    def pi_model_choices(self):
        repos = [self.pi_model, *self.options.get("PI_LOCAL_MODEL_OPTIONS", "").split(",")]
        choices = list(dict.fromkeys(repo.strip() for repo in repos if repo.strip()))
        if any(repo not in PI_PROFILES for repo in choices):
            raise ValueError("PI_LOCAL_MODEL_OPTIONS requires a registered Raspberry Pi profile.")
        return choices

    @property
    def gpu_model_choices(self):
        return [repo for repo in self.model_choices if repo not in PI_PROFILES]

    def download_policy(self):
        """Validate transfer settings before requesting paid resources."""
        policy = {}
        for key, default in (
            ("DOWNLOAD_STALL_SECONDS", 180),
            ("DOWNLOAD_MIN_BYTES_PER_SECOND", 1024),
            ("DOWNLOAD_CHUNK_MIB", 256),
        ):
            try:
                value = int(self.options.get(key, str(default)))
            except ValueError:
                raise ValueError(f"{key} must be a positive integer in Modules.txt.") from None
            if value <= 0:
                raise ValueError(f"{key} must be a positive integer in Modules.txt.")
            policy[key.lower()] = value
        return policy

    def provider_read_timeout(self):
        key = "MODEL_RESPONSE_TIMEOUT_SECONDS"
        try:
            seconds = int(self.options.get(key, "600"))
        except ValueError:
            raise ValueError(f"{key} must be a positive integer.") from None
        if seconds <= 0:
            raise ValueError(f"{key} must be a positive integer.")
        return seconds

    def api_output_limit(self):
        if self.api_profile[1] != "lordx64/cyberglm":
            return None
        key = "CYBERGLM_MAX_OUTPUT_TOKENS"
        try:
            value = int(self.options.get(key, "16384"))
        except ValueError:
            raise ValueError(f"{key} must be an integer from 1 to 65536.") from None
        if not 1 <= value <= 65536:
            raise ValueError(f"{key} must be an integer from 1 to 65536.")
        return value

    def local_context_window(self, model=None):
        from velocicyber.engine.context import DEFAULT_CONTEXT_WINDOW, ContextBudget
        from velocicyber.models.gpu_profiles import GPU_PROFILES

        try:
            profiles = json.loads(self.options.get("LOCAL_MODEL_CONTEXT_WINDOWS", "{}"))
        except ValueError:
            raise ValueError("LOCAL_MODEL_CONTEXT_WINDOWS must be a JSON object.") from None
        if not isinstance(profiles, dict):
            raise ValueError("LOCAL_MODEL_CONTEXT_WINDOWS must be a JSON object.")
        for value in profiles.values():
            ContextBudget(value)
        repo = model or self.local_model
        default = GPU_PROFILES[repo].context_window if repo in GPU_PROFILES else DEFAULT_CONTEXT_WINDOW
        return profiles.get(repo, default)

    @property
    def runtime(self):
        path = self.root / ".runtime"
        path.mkdir(mode=0o700, exist_ok=True)
        return path

    @property
    def data(self):
        return self.root / "velocicyber_finetuning-data"

    def engine(self) -> Path:
        if configured := self.options.get("INTERPRETER_BIN"):
            path = (self.root / configured).resolve()
            if path.is_file():
                return path
        from velocicyber.engine.installation import ACTIVE_FILE, RECOVERY_FILE, managed_engine

        if (self.runtime / ACTIVE_FILE).exists() or (self.runtime / RECOVERY_FILE).exists():
            return managed_engine(self.runtime)
        # Use the managed package entrypoint, which locates its bundled sandbox helpers.
        for path in sorted((self.runtime / "engine").rglob("interpreter")):
            if path.is_file() and os.access(path, os.X_OK):
                return path
        raise RuntimeError(
            "The Velocicyber execution engine is not installed. Run python3 scripts/install/install_engine.py"
        )

    def require_secret(self, name: str) -> str:
        value = self.secrets.get(name, "")
        if not value:
            raise ValueError(f"Configure {name} in .env first.")
        return value
