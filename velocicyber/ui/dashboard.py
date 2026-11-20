"""Render dashboard state without polling providers or changing application state."""

from rich.text import Text

from velocicyber.models.gpu_profiles import MOG_SEC_REPO
from velocicyber.providers import temperature
from velocicyber.sentinel.profiles import sentinel_profile
from velocicyber.state import GPUStatus
from velocicyber.ui.presentation import CODE_GRAY, NEON_BLUE, NEON_GREEN


def dashboard_text(settings, state, *, restoring_gpu=False, sentinel_busy=False,
                   pi_temperature_c=None):
    display = Text("Private model: ", style="#e3ffe7", no_wrap=True, overflow="ellipsis")
    model = (
        settings.pi_model
        if state.private_target == "pi"
        else settings.local_model
    ) or "Not selected"
    local_active = (
        state.active_provider == "local"
        and state.model_ready
        and state.gpu_status == GPUStatus.RUNNING
    )
    pi_active = state.active_provider == "pi" and state.pi_ready
    display.append(
        model, style=f"bold {NEON_GREEN}" if local_active or pi_active else CODE_GRAY
    )
    gpu_color = {
        GPUStatus.RUNNING: "#39ff14",
        GPUStatus.INITIALIZED: "yellow",
        GPUStatus.STOPPING: "yellow",
        GPUStatus.STOPPED: "red",
    }[state.gpu_status]
    gpu_label = {
        GPUStatus.STOPPED: "OFF",
        GPUStatus.STOPPING: "Turning OFF",
    }.get(state.gpu_status, state.gpu_status.value)
    display.append(f"\nGPUs: {gpu_label}", style=gpu_color)
    if restoring_gpu:
        readiness = "Checking GPU status…"
    elif state.gpu_status == GPUStatus.STOPPING:
        readiness = "Waiting for shutdown confirmation"
    elif state.model_ready:
        readiness = "Ready for coding"
        if settings.local_model == MOG_SEC_REPO and state.model_context_window:
            context = "1M" if state.model_context_window == 1000000 else f"{state.model_context_window // 1024}K"
            readiness += f" · {context} context"
    elif state.gpu_status == GPUStatus.RUNNING and state.model_stage:
        if state.model_stage == "download" and state.download_total:
            percent = min(100, state.download_bytes * 100 // state.download_total)
            filled = percent // 10
            bar = "|" * filled + " " * (10 - filled)
            readiness = f"UPLOADING MODEL [{bar}] {percent}%"
        else:
            label = {
                "prepare": "PREPARING MODEL",
                "verify": "VERIFYING MODEL",
                "start": "STARTING SERVER",
                "load": "LOADING MODEL",
                "check": "CHECKING SERVER RUNTIME",
                "build": "BUILDING SERVER RUNTIME",
                "failed": "STARTUP FAILED — see output",
            }.get(state.model_stage, "LOADING MODEL")
            readiness = label
    elif state.gpu_status == GPUStatus.RUNNING:
        readiness = "reconnect / load"
    else:
        readiness = "unavailable"
    health = state.api_health
    api_color = (
        "green"
        if health == "ok"
        else "red"
        if health in ("payment_required", "authentication_error")
        else "#a2baa9"
    )
    display.append("\nAPI: ")
    display.append(
        settings.api_model_name,
        style=f"bold {NEON_GREEN}" if state.active_provider == "api" else CODE_GRAY,
    )
    display.append(f" ({health.replace('_', ' ')})", style=api_color)
    display.append("\nGPU inference: " + readiness, style=NEON_BLUE)
    if state.private_target == "pi":
        display.append(
            "\nPi CPU: " + (state.pi_stage or "OFF"),
            style=NEON_GREEN if state.pi_ready else NEON_BLUE,
        )
        if pi_temperature_c is None:
            display.append(" (N/A)", style=CODE_GRAY)
        else:
            fahrenheit = pi_temperature_c * 9 / 5 + 32
            display.append(
                f" ({fahrenheit:.1f}°F)",
                style="bold #ff3333"
                if pi_temperature_c >= temperature.WARNING_C
                else NEON_GREEN,
            )
    display.append("\nUsing: " + state.active_provider.upper(), style=NEON_BLUE)
    sentinel_label = (
        "ON · " + sentinel_profile(state).label + " · " + state.sentinel_stage
        if state.sentinel_enabled else state.sentinel_stage
    )
    if state.sentinel_enabled or sentinel_busy:
        display.append(
            "\nSentinel: " + sentinel_label,
            style="bold #ff3b30" if state.sentinel_stage == "STOPPED TASK" else
            NEON_GREEN if state.sentinel_ready else
            "red" if state.sentinel_enabled else CODE_GRAY,
        )
    display.append(
        "\nData: " + ("ON" if state.collection else "OFF"),
        style=NEON_GREEN if state.collection else "red",
    )
    display.append(
        "\nAccess: " + ("OPEN" if state.sandbox_open else "WORKSPACE"),
        style="bold #ff39c8" if state.sandbox_open else NEON_GREEN,
    )
    return display
