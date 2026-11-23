"""Model-specific GPU requirements; estimates do not imply live validation."""

import re
from dataclasses import dataclass

INKLING_REPO = "Blackfrost-AI/Inkling-2-Mini-Derisked"

CYBER_FROST_REPO = "Blackfrost-AI/CYBER-FROST-3.8-BF16"
MOG_SEC_REPO = "Blackfrost-Research/M.O.G.-SEC-27B-1M-CTX-BF16"
GLM53_EXL3_REPO = "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold"
FLINT_REPO = "orcarouter/Flint-abliterated-GGUF"
ORCACYBER_REPO = "orcarouter/OrcaCyber-Nano-1.0"
SAGE_REPO = "Blackfrost-AI/SAGE-A1P5-DERISKED-BF16-V2"


@dataclass(frozen=True)
class ServingProfile:
    family: str
    context_window: int
    minimum_vram_gib: int
    minimum_host_memory_gib: int
    minimum_disk_gib: int
    gpu_families: tuple[str, ...]
    gpu_counts: tuple[int, ...]
    arguments: tuple[str, ...]
    notice: str
    backend: str = "vllm"
    gguf_filename: str = ""
    variant: str = ""
    response_timeout_seconds: int = 0

    def supports_hardware(self, description, gpu_count):
        return gpu_count in self.gpu_counts and any(
            re.search(r"(?:^|[^a-z0-9])" + family + r"(?:$|[^a-z0-9])", description, re.I)
            for family in self.gpu_families
        )

    def hardware_requirements(self):
        return (
            f"Supported GPUs: {', '.join(self.gpu_families)}; "
            f"GPU counts: {', '.join(map(str, self.gpu_counts))}; "
            f"at least {self.minimum_host_memory_gib} GiB host RAM and "
            f"{self.minimum_disk_gib} GiB disk."
        )


GPU_PROFILES = {
    INKLING_REPO: ServingProfile(
        family="inkling2_loop",
        context_window=65536,
        minimum_vram_gib=32,
        minimum_host_memory_gib=64,
        minimum_disk_gib=128,
        gpu_families=("H100", "H200", "B200", "B300"),
        gpu_counts=(1,),
        arguments=(
            "--dtype", "bfloat16",
            "--language-model-only",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "2048",
            "--speculative-config",
            '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}',
        ),
        notice=(
            "Lambda GPU required · native NVFP4/MXFP8 weights + MTP: 20.16 GB / 18.78 GiB.\n"
            "Recommended / smallest supported menu choice: Single 80 GB (H100).\n"
            "Single 180 GB (B200) is also supported; no cluster required.\n"
            "Planning floor: 32 GiB VRAM, but A10/A6000/A100 are not enabled for this runtime.\n"
            "Harness context: 64K, text/coding, one request; DSpark draft module preserved.\n"
            "Dedicated CUDA 13 vLLM image; compatible NVIDIA driver required (R580+ recommended).\n"
            "Projected requirements; Lambda inference/tool use awaiting live validation."
        ),
    ),
    FLINT_REPO: ServingProfile(
        family="flint",
        context_window=65536,
        minimum_vram_gib=32,
        minimum_host_memory_gib=64,
        minimum_disk_gib=96,
        gpu_families=("A6000", "A100", "H100", "H200", "B200", "B300"),
        gpu_counts=(1,),
        arguments=("--temp", "1.0", "--top-p", "0.95", "--top-k", "20"),
        backend="llama.cpp",
        gguf_filename="Flint-abliterated-Mix-24G.gguf",
        notice=(
            "Lambda GPU required · Mix-24G GGUF (23.99 GB / 22.35 GiB).\n"
            "Minimum menu choice: Single 40 GB (A100). Recommended: Single 48 GB (A6000).\n"
            "Profile minimum: 32 GiB VRAM; 24 GB GPUs are too small for this variant.\n"
            "Harness context: 64K, text/coding, one request at a time.\n"
            "Projected requirements; Lambda inference/tool use awaiting live validation."
        ),
    ),
    SAGE_REPO: ServingProfile(
        family="qwen3_5_moe",
        context_window=32768,
        minimum_vram_gib=96,
        minimum_host_memory_gib=128,
        minimum_disk_gib=256,
        gpu_families=("H100", "H200", "B200", "B300"),
        gpu_counts=(1, 2, 4, 8),
        arguments=(
            "--dtype", "bfloat16",
            "--mamba-ssm-cache-dtype", "float32",
            "--language-model-only",
        ),
        notice=(
            "Lambda GPU required · ~36B MoE · 71.9 GB BF16 weights (~67.0 GiB).\n"
            "Recommended: Single 180 GB (B200). No cluster required.\n"
            "Conservative profile minimum: 96 GiB total VRAM; 80 GB is not enabled.\n"
            "Harness context: 32K, text/coding, one request at a time.\n"
            "Projected requirements; Lambda inference/tool use awaiting live validation."
        ),
    ),
    ORCACYBER_REPO: ServingProfile(
        family="qwen35",
        context_window=32768,
        minimum_vram_gib=24,
        minimum_host_memory_gib=32,
        minimum_disk_gib=64,
        gpu_families=("A10", "A6000", "A100", "H100", "H200", "B200", "B300"),
        gpu_counts=(1,),
        arguments=("--temp", "1.0", "--top-p", "0.95", "--top-k", "20"),
        backend="llama.cpp",
        gguf_filename="OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf",
        notice=(
            "Lambda GPU required · Mix-15.7G GGUF (15.70 GB / 14.62 GiB).\n"
            "Minimum: Single 24 GB (A10). Recommended headroom: Single 48 GB (A6000).\n"
            "Harness context: 32K, text/coding, one request at a time.\n"
            "Projected requirements; Lambda inference/tool use awaiting live validation."
        ),
    ),
    GLM53_EXL3_REPO: ServingProfile(
        family="glm5_next",
        context_window=32768,
        minimum_vram_gib=256,
        minimum_host_memory_gib=256,
        minimum_disk_gib=512,
        gpu_families=("B200",),
        gpu_counts=(2,),
        arguments=(),
        backend="tensorfold",
        notice=(
            "Lambda GPUs required · 175.6 GB EXL3/BF16 weights (~163.6 GiB).\n"
            "Experimental: Cluster 360 GB (2× B200); Lambda validation pending.\n"
            "Planning allowance: 256 GiB total; two GPUs required. Single 180 GB is unsupported.\n"
            "Harness context: 32K, text/coding, one request at a time.\n"
            "Publisher tested 2× DGX Spark (128 GB each); its 1M context is not enabled here."
        ),
    ),
    CYBER_FROST_REPO: ServingProfile(
        family="qwen4_exp",
        context_window=32768,
        minimum_vram_gib=480,
        minimum_host_memory_gib=512,
        minimum_disk_gib=512,
        gpu_families=("H100", "H200", "B200", "B300"),
        gpu_counts=(4, 8),
        arguments=(
            "--dtype",
            "bfloat16",
            "--mamba-ssm-cache-dtype",
            "bfloat16",
            "--language-model-only",
        ),
        notice=(
            "Lambda GPUs required · 360 GB BF16 weights.\n"
            "Recommended first test: Cluster 720 GB (4× B200).\n"
            "Alternatives: 640 GB (8× H100) or 1440 GB (8× B200).\n"
            "32K context; Lambda inference/tool use awaiting live validation."
        ),
    ),
    MOG_SEC_REPO: ServingProfile(
        family="qwen3_5",
        variant="64k",
        context_window=65536,
        minimum_vram_gib=72,
        minimum_host_memory_gib=128,
        minimum_disk_gib=256,
        gpu_families=("H100", "H200", "B200", "B300"),
        gpu_counts=(1, 2, 4, 8),
        arguments=(
            "--dtype",
            "bfloat16",
            "--mamba-ssm-cache-dtype",
            "float32",
            "--language-model-only",
        ),
        notice=(
            "Lambda GPUs required · 55.6 GB BF16 weights.\n"
            "Recommended first test: Single 80 GB (H100); Single 180 GB (B200) also supported.\n"
            "Harness context: 64K (65,536 tokens), BF16 KV cache.\n"
            "Earlier 32K coding/tool check passed on H100; the new 64K limit "
            "and long-session stability await live validation."
        ),
    ),
}


MOG_CONTEXT_PROFILES = {
    "64k": GPU_PROFILES[MOG_SEC_REPO],
    "1m": ServingProfile(
        family="qwen3_5",
        variant="1m",
        context_window=1_000_000,
        minimum_vram_gib=160,
        minimum_host_memory_gib=256,
        minimum_disk_gib=256,
        gpu_families=("H100", "H200", "B200", "B300"),
        gpu_counts=(1, 2, 4, 8),
        arguments=(
            "--dtype", "bfloat16",
            "--mamba-ssm-cache-dtype", "float32",
            "--language-model-only",
            "--kv-cache-dtype", "fp8_e4m3",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "2048",
            "--enable-prefix-caching",
            "--mamba-cache-mode", "align",
            "--prefix-cache-retention-interval", "0",
        ),
        response_timeout_seconds=3600,
        notice=(
            "Experimental MOG 1M context (1,000,000 tokens).\n"
            "Same 55.6 GB BF16 weights; FP8 KV cache and float32 recurrent state.\n"
            "Recommended first test: Single 180 GB (B200). More headroom: 2× B200 (360 GB).\n"
            "Planning floor: 160 GiB VRAM; Single 80 GB is unsupported.\n"
            "One request at a time. Full-context speed, accuracy and tool use are unvalidated.\n"
            "Long prompts may take many minutes before output; compaction starts near 750K tokens."
        ),
    ),
}


def gpu_profile(repo, variant=""):
    """Resolve a serving variant without changing the Hugging Face repository ID."""
    if repo == MOG_SEC_REPO:
        try:
            return MOG_CONTEXT_PROFILES[variant or "64k"]
        except KeyError:
            raise ValueError("MOG_CONTEXT_PROFILE must be 64k or 1m.") from None
    return GPU_PROFILES.get(repo)
