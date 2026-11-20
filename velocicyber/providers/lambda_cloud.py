"""Explicitly approved Lambda lifecycle, capacity checks, and SSH model serving."""

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from velocicyber.engine.context import ContextBudget, served_context_window
from velocicyber.models.gpu_profiles import GPU_PROFILES, MOG_SEC_REPO, gpu_profile
from velocicyber.models.pi_profiles import PI_PROFILES
from velocicyber.providers.huggingface import model_download_token
from velocicyber.state import GPUStatus

SINGLE_MEMORY_CHOICES = (24, 40, 48, 80, 180)
CLUSTER_MEMORY_CHOICES = (360, 640, 720, 1440, 1800)
MEMORY_CHOICES = SINGLE_MEMORY_CHOICES + CLUSTER_MEMORY_CHOICES


class LambdaError(RuntimeError):
    def __init__(self, status_code, *, code="", message="", suggestion=""):
        self.status_code = status_code
        self.code = code
        detail = " ".join(value for value in (code, message, suggestion) if value)
        super().__init__(
            f"Lambda returned HTTP {status_code}"
            + (": " + detail if detail else "")
            + "; no automatic retry."
        )

    @property
    def launch_rejected(self):
        # Timeouts and server/provider failures can follow an accepted request.
        return self.status_code in {400, 401, 403, 404, 409, 422, 429} and not self.code.startswith(
            "provider/"
        )


def total_vram(instance_type):
    description = instance_type.get("gpu_description", "").upper()
    match = re.search(r"\b(\d+)\s*G(?:I)?B\b", description)
    if match:
        per_gpu = int(match[1])
    else:
        known = {"A10": 24, "A6000": 48, "H100": 80, "H200": 141, "B200": 180}
        per_gpu = next(
            (size for name, size in known.items() if re.search(r"\b" + name + r"\b", description)),
            0,
        )
    return per_gpu * instance_type["specs"]["gpus"]


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    weight_bytes: int
    tool_parser: str
    reasoning_parser: str
    attention_heads: int
    backend: str = "vllm"
    filename: str = ""
    sha256: str = ""
    architecture: str = ""
    context_window: int = 0
    download_files: tuple[dict, ...] = ()
    profile_variant: str = ""

    @property
    def profile(self):
        return gpu_profile(self.repo, self.profile_variant)

    @property
    def required_gib(self):
        # Lower bound with working space. Serving validation is still required.
        estimate = math.ceil(self.weight_bytes / 2**30 * 1.20 + 4)
        profile = self.profile
        return max(estimate, profile.minimum_vram_gib) if profile else estimate


@dataclass(frozen=True)
class LaunchPlan:
    instance_type: str
    region: str
    total_vram_gib: int
    gpu_count: int
    hourly_cents: int
    model: ModelSpec
    created_at: float


class LambdaClient:
    def __init__(self, api_key, *, transport=None):
        from velocicyber.datasets.pipeline import Scrubber

        self.scrubber = Scrubber((api_key,))
        self.client = httpx.Client(
            base_url="https://cloud.lambda.ai/api/v1/",
            timeout=30,
            headers={"Authorization": "Bearer " + api_key},
            transport=transport,
        )

    def request(self, method, endpoint, payload=None):
        response = self.client.request(method, endpoint, json=payload)
        if response.is_error:
            # Retain only documented error fields, with secrets/control characters removed.
            # Never print the request, arbitrary response bodies or headers.
            detail = {}
            try:
                body = response.json()
                error = body.get("error") if isinstance(body, dict) else None
                if isinstance(error, dict):
                    for field in ("code", "message", "suggestion"):
                        value = error.get(field)
                        if isinstance(value, str):
                            value = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value)
                            detail[field] = self.scrubber.text(value)[:500]
            except ValueError:
                pass
            raise LambdaError(response.status_code, **detail)
        return response.json()["data"]

    def catalog(self):
        # Capacity changes independently of instance specs. Each explicit plan
        # needs one fresh snapshot; never reuse availability from an earlier try.
        return self.request("GET", "instance-types")

    def close(self):
        self.client.close()


class GPUManager:
    def __init__(self, settings, state, client=None):
        self.settings, self.state, self._client = settings, state, client
        self.spec = None
        self.tunnel = None
        self._download_updated_at = None

    @property
    def client(self):
        if self._client is None:
            self._client = LambdaClient(self.settings.require_secret("LAMBDA_API"))
        return self._client

    def save(self):
        self.state.persist(self.settings.runtime / "gpu.json")

    def record_error(self, operation, error, *, memory=None, model=None):
        """Keep local diagnostics for UI failures; never add them to training data."""
        from velocicyber.datasets.pipeline import Scrubber

        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "model": model or self.settings.local_model,
            "requested_vram_gib": memory,
            "instance_type": self.state.instance_type,
            "region": self.state.region,
            "launch_uncertain": self.state.launch_uncertain,
            "stage": self.state.model_stage,
            "error_type": type(error).__name__,
            "error": Scrubber(self.settings.secrets.values()).text(str(error))[:4000],
        }
        if isinstance(error, LambdaError):
            row["http_status"] = error.status_code
            row["provider_code"] = error.code
        try:
            folder = self.settings.runtime / "diagnostics"
            folder.mkdir(mode=0o700, exist_ok=True)
            fd = os.open(folder / "gpu-errors.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "w") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(Scrubber(self.settings.secrets.values()).text(json.dumps(row)) + "\n")
        except OSError:
            pass  # Reporting an error must not hide it when diagnostic storage is unavailable.

    def inspect_model(self, repo=None, *, profile_variant=None):
        repo = repo or self.settings.local_model
        if not repo:
            raise ValueError("Choose a GPU model in /local before loading or uploading it.")
        if repo in PI_PROFILES:
            raise ValueError("This Raspberry Pi model uses the local CPU runtime, not Lambda GPUs.")
        profile = (
            self.settings.gpu_profile(repo) if profile_variant is None
            else gpu_profile(repo, profile_variant)
        )
        token = model_download_token(self.settings)
        if (self.spec and self.spec.repo == repo
                and self.spec.profile_variant == (profile.variant if profile else "")):
            return self.spec
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi(token=token).model_info(repo, files_metadata=True)
        ggufs = [f for f in info.siblings if f.rfilename.endswith(".gguf")]
        if profile and profile.gguf_filename:
            ggufs = [f for f in ggufs if f.rfilename == profile.gguf_filename]
            if not ggufs:
                raise ValueError("The selected model's configured GGUF variant is missing.")
        if ggufs:
            from velocicyber.providers.gguf import inspect_header

            if profile and profile.backend != "llama.cpp":
                raise ValueError("GGUF weights do not match this model's serving profile.")
            requested = (
                profile.gguf_filename if profile and profile.gguf_filename
                else self.settings.options.get("GGUF_FILE")
            )
            if requested:
                ggufs = [f for f in ggufs if f.rfilename == requested]
            if len(ggufs) != 1 or not ggufs[0].size:
                raise ValueError("Select a single complete GGUF using GGUF_FILE in Modules.txt.")
            file = ggufs[0]
            if re.search(r"-\d{5}-of-\d{5}\.gguf$", file.rfilename):
                raise ValueError("Split GGUF checkpoints require a multi-file serving profile.")
            if not file.lfs or not file.lfs.sha256:
                raise ValueError("GGUF checksum metadata is required before deployment.")
            meta = inspect_header(repo, info.sha, file.rfilename, token)
            architecture = meta["general.architecture"]
            if profile and architecture != profile.family:
                raise ValueError("Model architecture changed; review its GPU serving profile first.")
            if (architecture not in {"qwen2", "qwen3", "qwen3moe", "qwen35", "qwen35moe"}
                    and not (profile and architecture == "flint")):
                raise ValueError(
                    f"GGUF serving needs validation for architecture {architecture!r}."
                )
            context_window = int(meta.get(architecture + ".context_length", 0))
            if profile:
                context_window = min(context_window or profile.context_window, profile.context_window)
            self.spec = ModelSpec(
                repo,
                info.sha,
                file.size,
                "jinja",
                "auto",
                0,
                "llama.cpp",
                file.rfilename,
                file.lfs.sha256,
                architecture,
                context_window,
            )
            return self.spec
        weights = [
            f
            for f in info.siblings
            if "/" not in f.rfilename and f.rfilename.endswith(".safetensors")
        ]
        if not weights or any(not f.size for f in weights):
            raise ValueError(
                "Model repository needs complete safetensors weights with size metadata."
            )
        if any(f.rfilename == "adapter_config.json" for f in info.siblings):
            raise ValueError(
                "The selected repository is a LoRA adapter. Select merged weights for serving."
            )
        path = hf_hub_download(
            repo,
            "config.json",
            revision=info.sha,
            token=token,
            cache_dir=self.settings.runtime / "hf-cache",
        )
        config = json.loads(Path(path).read_text())
        family = config.get("model_type", "")
        if profile and family != profile.family:
            raise ValueError("Model architecture changed; review its GPU serving profile first.")
        parsers = {
            "qwen2": ("hermes", ""),
            "qwen3": ("hermes", "qwen3"),
            "qwen3_moe": ("hermes", "qwen3"),
            "qwen3_5": ("qwen3_coder", "qwen3"),
            "qwen3_5_moe": ("qwen3_coder", "qwen3"),
        }
        if profile and profile.backend == "tensorfold":
            quant = config.get("quantization_config", {})
            if (
                quant.get("quant_method") != "exl3" or quant.get("bits") != 4
                or quant.get("codebook") != "mcg"
                or quant.get("scope") != "glm53_routed_experts_only"
            ):
                raise ValueError("TensorFold profile requires 4-bit MCG EXL3 routed experts.")
            parsers[profile.family] = ("tensorfold", "tensorfold")
        elif profile:
            # The packaged template emits <tool_call><function=...><parameter=...>.
            parsers[profile.family] = ("qwen3_coder", "qwen3")
        if family not in parsers:
            raise ValueError(
                f"Serving profile needs validation for model_type={family!r} before launch."
            )
        tool, reasoning = parsers[family]
        files = []
        if profile:
            for file in info.siblings:
                if "/" in file.rfilename or not file.rfilename.endswith(
                    (".safetensors", ".json", ".jinja", ".model", ".txt", ".tiktoken")
                ):
                    continue
                checksum = getattr(getattr(file, "lfs", None), "sha256", "") or ""
                if file.size is None or (
                    file.rfilename.endswith(".safetensors")
                    and not re.fullmatch(r"[0-9a-f]{64}", checksum)
                ):
                    raise ValueError("Model file sizes and weight SHA256 metadata are required.")
                files.append({"filename": file.rfilename, "size": file.size, "sha256": checksum})
        text_config = config.get("text_config", config)
        context_window = int(text_config.get("max_position_embeddings", 0))
        if repo == MOG_SEC_REPO:
            if context_window < profile.context_window:
                raise ValueError("MOG checkpoint does not support the selected context window.")
            if profile.variant == "1m":
                rope = text_config.get("rope_parameters", {})
                if (rope.get("rope_type") != "yarn" or rope.get("factor") != 4.0
                        or rope.get("original_max_position_embeddings") != 262144):
                    raise ValueError("MOG 1M requires the checkpoint's reviewed YaRN configuration.")
        if profile:
            context_window = min(context_window or profile.context_window, profile.context_window)
        self.spec = ModelSpec(
            repo,
            info.sha,
            sum(f.size for f in weights),
            tool,
            reasoning,
            config.get("text_config", config).get("num_attention_heads", 0),
            backend=profile.backend if profile else "vllm",
            architecture=family,
            context_window=context_window,
            download_files=tuple(files),
            profile_variant=profile.variant if profile else "",
        )
        return self.spec

    def plan(self, memory):
        if not self.settings.local_model:
            raise ValueError("Choose a GPU model in /local before launching GPUs.")
        self.settings.download_policy()
        self.settings.local_context_window()
        if memory not in MEMORY_CHOICES:
            raise ValueError("Choose one of the displayed total VRAM sizes.")
        if self.state.instance_id or self.state.launch_uncertain:
            raise RuntimeError(
                "An instance exists or launch is unresolved. Refresh or terminate it first."
            )
        for name in ("LAMBDA_SSH_KEY_NAME", "SSH_PRIVATE_KEY"):
            if not self.settings.options.get(name):
                raise ValueError(f"Configure {name} with ./velocicyber.sh setup before requesting a launch plan.")
        key = (self.settings.root / self.settings.options["SSH_PRIVATE_KEY"]).resolve()
        if not key.is_file() or not key.is_relative_to(self.settings.root):
            raise ValueError("SSH_PRIVATE_KEY must point to a key file inside this project. Run ./velocicyber.sh setup to select or import it.")
        spec = self.inspect_model()
        if spec.required_gib > memory:
            raise ValueError(
                f"Model requires at least ~{spec.required_gib} GiB including working space; "
                f"{memory} GB is too small. No instance launched."
            )
        preferred_region = self.settings.options.get("LAMBDA_REGION", "auto") or "auto"
        profile = spec.profile
        if profile:
            self.serving_image(spec)  # Missing runtime pin must block before paid launch.
        from velocicyber.providers.pricing import cache_catalog_prices

        catalog = self.client.catalog()
        cache_catalog_prices(self.settings.runtime, catalog)
        candidates = []
        for row in catalog.values():
            kind = row["instance_type"]
            regions = [r["name"] for r in row["regions_with_capacity_available"]]
            if total_vram(kind) == memory and kind.get("architecture", "x86_64") == "x86_64":
                count = kind["specs"]["gpus"]
                if (memory in SINGLE_MEMORY_CHOICES and count != 1) or count < 1:
                    continue
                if memory in CLUSTER_MEMORY_CHOICES and count < 2:
                    continue
                if spec.attention_heads and spec.attention_heads % count:
                    continue
                if profile and (
                    not profile.supports_hardware(
                        kind.get("gpu_description", "") + " " + kind["name"], count
                    )
                    or kind["specs"].get("memory_gib", 0) < profile.minimum_host_memory_gib
                    or kind["specs"].get("storage_gib", 0) < profile.minimum_disk_gib
                ):
                    continue
                for region in regions:
                    if preferred_region == "auto" or preferred_region == region:
                        candidates.append((kind, region))
        if not candidates:
            category = "single-GPU" if memory in SINGLE_MEMORY_CHOICES else "multi-GPU"
            raise ValueError(
                f"No available {category} instance with exactly {memory} GB total VRAM "
                f"in {preferred_region if preferred_region != 'auto' else 'any region'} "
                "in Lambda's latest capacity response. No instance launched. "
                "Select the size again later to check fresh availability."
                + (" " + profile.hardware_requirements() if profile else "")
            )
        kind, region = min(
            candidates, key=lambda row: (row[0]["price_cents_per_hour"], row[1], row[0]["name"])
        )
        return LaunchPlan(
            kind["name"],
            region,
            memory,
            kind["specs"]["gpus"],
            kind["price_cents_per_hour"],
            spec,
            time.time(),
        )

    def launch(self, plan, *, approved=False):
        if not approved:
            raise PermissionError("Explicit approval is required to launch billable GPUs.")
        self.settings.download_policy()
        if self.state.instance_id or self.state.launch_uncertain:
            raise RuntimeError("An existing or unresolved launch must be reconciled first.")
        selected_profile = self.settings.gpu_profile()
        if (time.time() - plan.created_at > 300 or plan.model.repo != self.settings.local_model
                or plan.model.profile_variant != (selected_profile.variant if selected_profile else "")):
            raise ValueError("Launch plan expired or model/context profile changed. Prepare a new plan.")
        model_download_token(self.settings)  # Recheck before requesting paid resources.
        self.state.instance_type, self.state.region = plan.instance_type, plan.region
        self.state.gpu_count = plan.gpu_count
        self.state.total_vram_gib = plan.total_vram_gib
        self.state.instance_ip = ""
        self.state.launch_uncertain = True
        self.save()  # A network timeout must never cause an automatic duplicate launch.
        try:
            data = self.client.request(
                "POST",
                "instance-operations/launch",
                {
                    "region_name": plan.region,
                    "instance_type_name": plan.instance_type,
                    "ssh_key_names": [self.settings.options["LAMBDA_SSH_KEY_NAME"]],
                    "quantity": 1,
                    "name": "velocicyber",
                },
            )
        except LambdaError as exc:
            if exc.launch_rejected:
                self.state.launch_uncertain = False
                self.state.gpu_status = GPUStatus.STOPPED
                self.state.model_ready = False
                self.save()
            raise
        self.state.instance_id = data["instance_ids"][0]
        self.state.instance_type, self.state.region = plan.instance_type, plan.region
        self.state.gpu_count = plan.gpu_count
        self.state.total_vram_gib = plan.total_vram_gib
        self.state.gpu_status = GPUStatus.INITIALIZED
        self.state.model_ready = False
        self.state.launch_uncertain = False
        self.spec = plan.model
        self.save()

    def refresh(self):
        if self.state.launch_uncertain and not self.state.instance_id:
            candidates = [
                i for i in self.client.request("GET", "instances") if i.get("name") == "velocicyber"
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "Ambiguous launch: inspect Lambda console before retrying. "
                    "No further launch will be attempted."
                )
            self.state.instance_id = candidates[0]["id"]
            self.state.launch_uncertain = False
        if not self.state.instance_id:
            return "terminated"
        try:
            item = self.client.request("GET", "instances/" + self.state.instance_id)
        except LambdaError as exc:
            if exc.status_code != 404:
                raise
            item = {"status": "terminated", "instance_type": {"specs": {"gpus": 0}}}
        status = item["status"]
        self.state.instance_ip = item.get("ip", "")
        self.state.gpu_count = item["instance_type"]["specs"]["gpus"]
        self.state.total_vram_gib = total_vram(item["instance_type"])
        if status in ("terminated", "unhealthy"):
            self.state.model_ready = False
            if self.state.active_provider == "local":
                self.state.active_provider = "api"
            self.state.gpu_status = (
                GPUStatus.STOPPED if status == "terminated" else GPUStatus.INITIALIZED
            )
            if status == "terminated":
                self.state.instance_id = ""
        else:
            if status == "terminating" or self.state.gpu_status == GPUStatus.STOPPING:
                self.state.gpu_status = GPUStatus.STOPPING
            else:
                self.state.gpu_status = (
                    GPUStatus.RUNNING if status == "active" else GPUStatus.INITIALIZED
                )
            if self.state.gpu_status != GPUStatus.RUNNING:
                self.state.model_ready = False
        self.save()
        return status

    def terminate(self, *, approved=False):
        if not approved:
            raise PermissionError("Confirm termination of the displayed instance first.")
        if not self.state.instance_id:
            raise RuntimeError("No tracked instance. Refresh an unresolved launch first.")
        self.client.request(
            "POST", "instance-operations/terminate", {"instance_ids": [self.state.instance_id]}
        )
        self.state.model_ready = False
        if self.state.active_provider == "local":
            self.state.active_provider = "api"
        # Keep the ID until the provider confirms termination; billing may continue meanwhile.
        self.state.gpu_status = GPUStatus.STOPPING
        self.save()

    def ssh_args(self):
        import ipaddress

        ipaddress.ip_address(self.state.instance_ip)
        key = (self.settings.root / self.settings.options["SSH_PRIVATE_KEY"]).resolve()
        if not key.is_relative_to(self.settings.root) or not key.is_file():
            raise ValueError("Configure a project-local SSH_PRIVATE_KEY.")
        return [
            "ssh",
            "-i",
            str(key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=" + str(self.settings.runtime / "known_hosts"),
        ]

    def deployment_progress(self, line):
        """Consume only structured progress; never display arbitrary remote stdout."""
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(event, dict):
            return
        stage = event.get("stage")
        if stage == "download_progress":
            done, total = event.get("bytes"), event.get("total")
            if (
                type(done) is not int
                or type(total) is not int
                or total != self.state.download_total
                or total <= 0
                or not self.state.download_bytes <= done <= total
            ):
                return
            now = time.monotonic()
            if (
                self._download_updated_at is None
                or now - self._download_updated_at >= 60
                or done == total
            ):
                self.state.model_stage = "download"
                self.state.download_bytes = done
                self._download_updated_at = now
        elif stage in (
            "download_verifying", "server_starting", "model_loading", "runtime_check", "runtime_build",
        ):
            self.state.model_stage = {
                "download_verifying": "verify",
                "server_starting": "start",
                "model_loading": "load",
                "runtime_check": "check",
                "runtime_build": "build",
            }[stage]

    async def deploy(self):
        if not self.settings.local_model:
            raise ValueError("Choose a GPU model in /local before loading or uploading it.")
        if self.state.gpu_status != GPUStatus.RUNNING:
            raise RuntimeError("Wait for the GPU instance to be active before loading the model.")
        download_policy = self.settings.download_policy()
        spec = self.spec or await asyncio.to_thread(self.inspect_model)
        if spec.repo != self.settings.local_model:
            spec = await asyncio.to_thread(self.inspect_model)
        if spec.required_gib > self.state.total_vram_gib:
            raise ValueError(
                f"Selected model needs at least ~{spec.required_gib} GiB; "
                "the current instance is too small or its memory is unknown."
            )
        self.validate_loaded_hardware(spec)
        token = await asyncio.to_thread(model_download_token, self.settings)
        self.state.model_revision = spec.revision
        profile = spec.profile
        script = (self.settings.root / "scripts/runtime/serve_model.py").read_text()
        if spec.backend == "tensorfold":
            script += "\n" + (self.settings.root / "scripts/runtime/tensorfold_runtime.py").read_text()
        config = {
            **download_policy,
            "repo": spec.repo,
            "revision": spec.revision,
            "gpus": self.state.gpu_count,
            "tool_parser": spec.tool_parser,
            "reasoning_parser": spec.reasoning_parser,
            "token": token or "",
            "backend": spec.backend,
            "filename": spec.filename,
            "sha256": spec.sha256,
            "weight_bytes": spec.weight_bytes,
            "context_window": ContextBudget(
                min(self.settings.local_context_window(), spec.context_window)
                if spec.context_window
                else self.settings.local_context_window()
            ).window,
            "image": self.serving_image(spec),
            "download_files": spec.download_files,
            "extra_arguments": profile.arguments if profile else (),
            "max_num_seqs": 1 if profile else 2,
        }
        if spec.backend == "tensorfold":
            lock = json.loads((self.settings.root / "config/locks/serving.lock.json").read_text())
            config["tensorfold_runtime"] = lock["model_images"][spec.repo]
        # Only the verified read-only token travels through encrypted SSH stdin.
        # The dataset-upload token is never included in remote configuration.
        helpers = (
            ", tensorfold_prepare=prepare_tensorfold, tensorfold_options=tensorfold_server_options"
            if spec.backend == "tensorfold" else ""
        )
        program = script + "\nimport json\nserve(json.loads(" + repr(json.dumps(config)) + ")" + helpers + ")\n"
        process = await asyncio.create_subprocess_exec(
            *self.ssh_args(),
            "ubuntu@" + self.state.instance_ip,
            "python3 -",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.state.model_ready = False
        self.state.model_stage = "prepare"
        self.state.download_bytes = 0
        self.state.download_total = spec.weight_bytes
        self._download_updated_at = None

        async def read_progress():
            async for line in process.stdout:
                self.deployment_progress(line)

        async def read_errors():
            tail = b""
            while block := await process.stderr.read(8192):
                tail = (tail + block)[-8192:]
            return tail

        readers = [asyncio.create_task(read_progress()), asyncio.create_task(read_errors())]
        try:
            # Large progressing transfers may take much longer than 15 minutes.
            # Curl detects stalled transfers; SSH keepalives detect a lost host.
            # User cancellation still closes this process in the finally block.
            process.stdin.write(program.encode())
            await process.stdin.drain()
            process.stdin.close()
            _, error = await asyncio.gather(*readers)
            await process.wait()
            if process.returncode:
                from velocicyber.datasets.pipeline import Scrubber

                raise RuntimeError(
                    "Model deployment failed: "
                    + Scrubber(self.settings.secrets.values()).text(
                        error.decode(errors="replace")[-1500:]
                    )
                )
            self.state.model_stage = "load"
            await self.connect()
        except BaseException:
            self.state.model_stage = ""
            raise
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)

    def serving_image(self, spec):
        lock = json.loads((self.settings.root / "config/locks/serving.lock.json").read_text())
        if spec.repo in GPU_PROFILES:
            image = lock.get("model_images", {}).get(spec.repo, {}).get("image")
            if not image:
                raise ValueError("This model needs its dedicated runtime pin in config/locks/serving.lock.json.")
            if spec.backend == "tensorfold":
                pin = lock["model_images"][spec.repo]
                if not re.fullmatch(r"[0-9a-f]{40}", pin.get("source_revision", "")) or not re.fullmatch(
                    r"[0-9a-f]{64}", pin.get("source_sha256", "")
                ) or not re.fullmatch(r"nvcr.io/nvidia/pytorch@sha256:[0-9a-f]{64}", image):
                    raise ValueError("TensorFold requires pinned NVIDIA image and source checksums.")
            return image
        if spec.backend == "llama.cpp":
            return self.settings.options.get("LLAMA_CPP_IMAGE", lock["llama_cpp_image"])
        return self.settings.options.get("VLLM_IMAGE", lock["vllm_image"])

    def validate_loaded_hardware(self, spec):
        profile = spec.profile
        if profile and not profile.supports_hardware(
            self.state.instance_type, self.state.gpu_count
        ):
            raise ValueError(
                "The current instance is incompatible with this model. "
                + profile.hardware_requirements()
                + "\n"
                + profile.notice
            )

    async def connect(self):
        await self.close_tunnel()
        self.tunnel = await asyncio.create_subprocess_exec(
            *self.ssh_args(),
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            "-L",
            "127.0.0.1:18000:127.0.0.1:8000",
            "ubuntu@" + self.state.instance_ip,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.sleep(1)
        if self.tunnel.returncode is not None:
            raise RuntimeError("SSH inference tunnel failed to start.")

    async def check_model_process(self):
        """Detect a crashed container during loading instead of waiting out the readiness timer."""
        if self.state.model_stage != "load" or not self.state.instance_ip:
            return
        process = await asyncio.create_subprocess_exec(
            *self.ssh_args(), "ubuntu@" + self.state.instance_ip,
            "sudo -n docker inspect --format '{{json .State}}' velocicyber-model",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), 25)
            if process.returncode:
                return  # A failed diagnostic does not prove the server has crashed.
            state = json.loads(output)
        except (TimeoutError, ValueError):
            return
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        if isinstance(state, dict) and state.get("Status") in {"restarting", "exited", "dead"}:
            self.state.model_ready = False
            self.state.model_stage = "failed"
            code = state.get("ExitCode")
            detail = f" (exit {code})" if type(code) is int else ""
            raise RuntimeError(
                f"Model server is {state['Status']}{detail}; startup failed. "
                "Inspect the velocicyber-model container logs. The GPU remains running."
            )

    async def check_ready(self):
        self.state.model_ready = False
        if not self.tunnel or self.tunnel.returncode is not None:
            return False
        loaded = False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                health = await client.get(self.state.local_url.removesuffix("/v1") + "/health")
                if health.status_code == 503:
                    return False
                health.raise_for_status()
                response = await client.get(self.state.local_url + "/models")
                response.raise_for_status()
                served_model = next(
                    (
                        item
                        for item in response.json()["data"]
                        if item["id"] == self.settings.local_model
                    ),
                    None,
                )
                loaded = served_model is not None
                if loaded:
                    configured_window = self.settings.local_context_window()
                    if (
                        self.spec
                        and self.spec.repo == self.settings.local_model
                        and self.spec.context_window
                        and self.spec.profile == self.settings.gpu_profile()
                    ):
                        configured_window = min(configured_window, self.spec.context_window)
                    served_window = served_context_window(served_model)
                    if self.settings.local_model == MOG_SEC_REPO and served_window != configured_window:
                        self.state.model_context_window = 0
                        self.state.model_context_model = ""
                        raise RuntimeError(
                            f"MOG server context is {served_window or 'unknown'}; selected profile "
                            f"requires {configured_window} tokens. Select MOG in /local to reload "
                            "the chosen context profile on compatible GPUs."
                        )
                    self.state.model_context_window = served_window or configured_window
                    self.state.model_context_model = self.settings.local_model
                    # One small inference validates readiness after model loading. Failure
                    # stops the workflow instead of repeatedly spending generation tokens.
                    probe = await client.post(
                        self.state.local_url + "/chat/completions",
                        timeout=httpx.Timeout(120, connect=10, write=10, pool=10),
                        json={
                            "model": self.settings.local_model,
                            "messages": [{"role": "user", "content": "Reply with OK."}],
                            "max_tokens": 32,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    probe.raise_for_status()
                    if not probe.json()["choices"][0]["message"].get("content"):
                        raise RuntimeError(
                            "Model loaded but inference returned no text. Check serving configuration."
                        )
                    if self.settings.local_model in GPU_PROFILES:
                        await self.check_tools(client)
                    self.state.model_ready = True
        except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError) as exc:
            if loaded:
                detail = (
                    f"HTTP {exc.response.status_code}"
                    if isinstance(exc, httpx.HTTPStatusError)
                    else type(exc).__name__
                )
                raise RuntimeError(
                    f"Model loaded but its inference check failed ({detail}). "
                    "The generation read timeout is 120 seconds; inspect the server logs."
                ) from None
        if self.state.model_ready:
            self.state.model_stage = ""
            if self.state.active_provider != "pi":
                self.state.active_provider = "local"
                self.state.private_target = "gpu"
        return self.state.model_ready

    async def check_tools(self, client):
        """One non-executing parser check before enabling a new agentic profile."""
        response = await client.post(
            self.state.local_url + "/chat/completions",
            timeout=httpx.Timeout(120, connect=10, write=10, pool=10),
            json={
                "model": self.settings.local_model,
                "messages": [{"role": "user", "content": "Call readiness_probe with value OK."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "readiness_probe",
                            "description": "Return a readiness marker; no actions are performed.",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "max_tokens": 128,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        calls = response.json()["choices"][0]["message"].get("tool_calls") or []
        try:
            valid = (
                len(calls) == 1
                and calls[0]["type"] == "function"
                and calls[0]["function"]["name"] == "readiness_probe"
                and json.loads(calls[0]["function"]["arguments"]) == {"value": "OK"}
            )
        except (ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise RuntimeError(
                "Text inference works, but the coding-tool parser check failed. "
                "The model has not been enabled; inspect the serving configuration."
            )

    async def close_tunnel(self):
        self.state.model_ready = False
        self.state.model_context_window = 0
        self.state.model_context_model = ""
        if self.tunnel and self.tunnel.returncode is None:
            self.tunnel.terminate()
            await self.tunnel.wait()
        self.tunnel = None
