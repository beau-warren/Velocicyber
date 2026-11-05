"""Offline planning estimates, based on September–November 2026 GPU validation.

Fresh startup includes boot, transfer, checksum, model load and inference check.
Cached startup assumes weights remain on an already-running compatible GPU.
Transfer rates vary by model and download path; M.O.G. uses parallel Xet shards.
The Raspberry Pi profile has its own installation/runtime notice.
"""

from dataclasses import dataclass

from velocicyber.config import PI_MODEL_NOTICE, PI_MODEL_REPO
from velocicyber.models.gpu_profiles import (
    CYBER_FROST_REPO,
    FLINT_REPO,
    GLM53_EXL3_REPO,
    GPU_PROFILES,
    MOG_SEC_REPO,
    ORCACYBER_REPO,
    SAGE_REPO,
)
from velocicyber.models.pi_profiles import BUGTRACE_PI_REPO, LFM3_PI_REPO, PI_PROFILES


@dataclass(frozen=True)
class StartupEstimate:
    fresh: str
    cached: str
    projected: bool = False

    def label(self):
        basis = "Projected" if self.projected else "Estimated"
        return f"{basis}: {self.fresh} fresh · {self.cached} cached"


ESTIMATES = {
    FLINT_REPO: StartupEstimate("35–70 min", "3–6 min", projected=True),
    SAGE_REPO: StartupEstimate("15–30 min", "3–8 min", projected=True),
    CYBER_FROST_REPO: StartupEstimate("6–14 hours", "10–30 min", projected=True),
    MOG_SEC_REPO: StartupEstimate("10–20 min", "2–4 min"),
    GLM53_EXL3_REPO: StartupEstimate("25–60 min", "5–15 min", projected=True),
    ORCACYBER_REPO: StartupEstimate("25–50 min", "2–5 min", projected=True),
}

MENU_HELP = (
    "Fresh: GPU boot + download + ready for coding.\n"
    "Cached: weights already on a running GPU.\n"
    "Times vary by region/network. Projected = untested."
)


def model_option_label(repo, *, ready=False):
    if repo in PI_PROFILES:
        return (
            repo
            + "\n"
            + PI_MODEL_NOTICE
            + (
                "\nReady now — Raspberry Pi CPU"
                if ready
                else "\nFirst use: 3.48 GB source download + local Q8 conversion; then local loading."
                if repo == LFM3_PI_REPO
                else "\nFirst use: 11.4 GiB weights + CPU runtime setup; then local loading."
            )
        )
    if estimate := ESTIMATES.get(repo):
        detail = estimate.label()
    else:
        detail = "Startup estimate unavailable for this model"
    if ready:
        detail += "\nReady now — already loaded"
    if profile := GPU_PROFILES.get(repo):
        detail = profile.notice + "\n" + detail
    return repo + "\n" + detail


def model_details(repo, *, ready=False):
    """Offline model information for the details view, separate from picker labels."""
    summary = model_option_label(repo, ready=ready).partition("\n")[2]
    if repo == LFM3_PI_REPO:
        return (
            summary + "\n\nHardware: Raspberry Pi 5 (Linux ARM64); about 4 GiB available RAM "
            "for serving, 6 GiB during conversion. No Lambda GPU required.\n"
            "Source weights: 3.46 GB BF16; ~3.48 GB with tokenizer/configuration.\n"
            "Setup converts the exact Blackfrost checkpoint to Q8_0 GGUF locally.\n"
            "Estimated Q8 weights: 1.8–2.0 GB; allow about 3 GB installed.\n"
            "Keep at least 12 GiB free during setup for downloads, conversion and dependencies.\n"
            "Temporary BF16 files and the converter environment are removed after successful conversion.\n"
            "Context: 16K, text/coding tools; thinking enabled. One Pi model at a time.\n"
            "First setup and generation speeds are unmeasured; conversion runs once.\n"
            "CPU runtime verified on Pi; full model conversion and coding validation pending.\n"
            "Model license: LFM Open License v1.0; review the publisher's terms."
        )
    if repo == BUGTRACE_PI_REPO:
        return (
            summary + "\n\nVariant: APEX Mini GGUF (12.3 GB / 11.4 GiB).\n"
            "Hardware: Raspberry Pi 5 with 16 GB RAM; about 13 GiB available RAM "
            "needed to start.\nGPU boot/upload: not needed.\n"
            "Context: 8,192 tokens, Q8 KV cache; text and coding tools.\n"
            "Switching Pi models unloads the previous model first.\n"
            "Existing uploads are verified and reused.\n"
            "Observed cached loading: about 3 seconds; cold starts vary.\n"
            "Local coding check: about 5 minutes including the initial prompt; "
            "generation about 3.4–3.6 tokens/second."
        )
    if repo == PI_MODEL_REPO:
        return (
            summary + "\n\nHardware: Raspberry Pi 5 with 16 GB RAM; about 13 GiB available RAM "
            "needed to start.\nGPU boot/upload: not needed.\n"
            "Startup time: no measured estimate yet; first download and CPU runtime "
            "build depend on the network and local storage."
        )
    if profile := GPU_PROFILES.get(repo):
        hardware = (
            f"Estimated minimum VRAM: {profile.minimum_vram_gib} GiB total.\n"
            + profile.hardware_requirements()
        )
    else:
        hardware = "Lambda GPU required. Exact hardware requirements are checked before launch."
    timing = (
        "\n\nDownload to a running GPU: ~2–5 min on the tested fast connection.\n"
        "Download + loading until ready: ~5–10 min with the serving image available.\n"
        "Observed on H100 SXM5 (2026-10-08): 55.6 GB transferred in ~2 min 13 sec; "
        "server initialization ~2 min 18 sec. One run, not a guarantee.\n"
        "Fresh startup estimate allows normal boot time; capacity waits, slow boots "
        "and a first container-image download can add time."
        if repo == MOG_SEC_REPO else ""
    )
    if repo == GLM53_EXL3_REPO:
        timing = (
            "\n\nTransfer only: ~7–20 min on a fast GPU-host connection; unmeasured.\n"
            "First startup includes an NVIDIA image, TensorFold install and CUDA compilation.\n"
            "Estimate extrapolates the M.O.G. transfer, with extra headroom; slow links "
            "or image/build delays can take hours. Cached figures are also projected.\n"
            "Listed 2× B200 price (2026-10-12): ~$13.78/hour total before tax.\n"
            "The launch confirmation uses current Lambda pricing and availability.\n"
            "TensorFold Python runtime with two CUDA ranks; EXL3 is not served by vLLM.\n"
            "Initial profile uses BF16 dense weights/KV and disables speculative drafts.\n"
            "No extra drafter is downloaded. Published Spark speed figures do not predict Lambda speed."
        )
    if repo == FLINT_REPO:
        timing = (
            "\n\nVariant: Flint-abliterated-Mix-24G.gguf (39.8B MoE, ~2.5B active).\n"
            "Publisher recommends this higher-fidelity variant over Mix-12G.\n"
            "Hugging Face download access is required; request access on the model page "
            "using the account associated with your HF token. Access is checked before GPU launch.\n"
            "Transfer only to a running GPU: ~25–50 min, projected from prior GGUF "
            "transfers at 8–19 MB/s; faster links can finish sooner.\n"
            "Fresh startup adds GPU boot, checksum, container download and loading; "
            "slow boots or networks can exceed this range. Cached times are also unmeasured.\n"
            "Downloads only Mix-24G; no Mix-12G, imatrix or full BF16 checkpoint.\n"
            "Pinned llama.cpp CUDA runtime with native Flint/tool-template support.\n"
            "Reasoning is always enabled and shares the context budget with the answer.\n"
            "The advertised 262K context requires a separate serving profile.\n"
            "The Pi coordinates inference; weights reside on Lambda.\n"
            "The launch confirmation uses current Lambda pricing and availability."
        )
    if repo == ORCACYBER_REPO:
        timing = (
            "\n\nVariant: OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf.\n"
            "Transfer only to a running GPU: ~15–35 min, projected from prior GGUF "
            "transfers at 8–19 MB/s; faster links can finish sooner.\n"
            "Fresh startup includes GPU boot, model transfer/checksum, container setup "
            "and loading. Slow boots or image downloads can exceed this range.\n"
            "Cached startup assumes weights and the serving image remain on a running GPU.\n"
            "Text and coding tools via llama.cpp; downloads only the selected main GGUF.\n"
            "Vision, speculative decoding and the advertised 262K context require "
            "separate configuration and validation.\n"
            "The Raspberry Pi coordinates inference; weights reside on Lambda.\n"
            "Listed prices (2026-10-19): A10 24 GB ~$1.29/hour; "
            "A6000 48 GB ~$1.09/hour, before tax.\n"
            "The launch confirmation uses current Lambda pricing and availability."
        )
    if repo == SAGE_REPO:
        timing = (
            "\n\nTransfer only to a running GPU: ~3–10 min on a fast connection, projected.\n"
            "Fresh startup includes boot, transfer/checksums, serving-image setup and loading.\n"
            "Estimate extrapolates the earlier 55.6 GB M.O.G. transfer; it is not measured for SAGE.\n"
            "Slow boots, image downloads or network congestion can exceed this range.\n"
            "Cached startup assumes the weights and serving image remain on a running GPU.\n"
            "BF16 weights alone use ~67 GiB; the 96 GiB planning floor allows runtime/KV space.\n"
            "Single 180 GB (B200) is the smallest currently listed single-GPU option above that floor.\n"
            "The complete checkpoint includes its indexed MTP file; speculative decoding is disabled.\n"
            "Vision, verification-protocol mode and the advertised 262K context are not enabled.\n"
            "The launch confirmation shows current Lambda pricing and availability."
        )
    return summary + "\n\n" + hardware + timing + "\n\n" + MENU_HELP
