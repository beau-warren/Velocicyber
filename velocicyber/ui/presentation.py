"""Literal transcript styling; formatting never changes captured model data."""

import re

from rich.text import Text

NEON_GREEN, NEON_PINK, NEON_BLUE = "#39ff14", "#ff39c8", "#00cfff"
CODE_GRAY = "#9399a3"
TEXT_GRAY = "#d3d7dd"
SHELL_LANGUAGES = {"bash", "sh", "zsh", "fish", "shell", "powershell", "ps1", "cmd", "bat"}
CONSOLE_LANGUAGES = {"console", "terminal", "shell-session"}
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)")

# Hand-lettered "Velocicyber": only V is capital, with a descending y.
TITLE = r"""_   __         __              _               __
| | / / ___   / / ___  ____   (_) ____ __ __  / /_ ___  ____
| |/ / / -_) / / / _ \/ __/  / / / __// // / / _ \/ -_)/ __/
|___/  \__/ /_/  \___/\__/  /_/  \__/ \_, / /_.__/\__/ /_/
                                      /___/"""


def title_art():
    """Static italic lettering with a one-column pink extrusion."""
    rows = TITLE.splitlines()
    result = Text()
    for y, row in enumerate(rows):
        foreground = row.ljust(len(row) + 1)
        shadow = " " + row
        for front, back in zip(foreground, shadow):
            if front != " ":
                result.append(front, style=f"bold italic {NEON_GREEN}")
            elif back != " ":
                result.append(back, style=f"italic {NEON_PINK}")
            else:
                result.append(" ")
        if y != len(rows) - 1:
            result.append("\n")
    return result


def compact_title_art(*, padded=True):
    """Desktop icon sampled at 16×16, rendered with colored half-block pixels.

    This preserves the green V and pink slash from assets/velocicyber.png in
    terminals without inline image support, including the desktop's LXTerminal.
    """
    pixels = (
        "................",
        "................",
        "................",
        "...gg.....gggp..",
        "...ggg....ggpp..",
        "....gg....ggp...",
        "....ggg..gggp...",
        ".....gg..ggpp...",
        ".....gg.gggp....",
        ".....gggggpp....",
        "......ggggp.....",
        "......ggggp.....",
        "......gggpp.....",
        "................",
        "................",
        "................",
    )
    if not padded:
        pixels = pixels[3:-3]  # Preserve the V; omit the icon's black top/bottom padding.
    palette = {".": "#000000", "g": NEON_GREEN, "p": NEON_PINK}
    result = Text()
    result.append("Velocicyber", style=f"bold italic {NEON_GREEN}")
    for y in range(0, len(pixels), 2):
        result.append("\n")
        for top, bottom in zip(pixels[y], pixels[y + 1]):
            result.append(
                " " if top == bottom == "." else "▀",
                style=f"{palette[top]} on {palette[bottom]}",
            )
    return result


def status_guide():
    rows = (
        ("Updates", "Whether the harness and engine are up to date."),
        ("Private model", "Selected GPU or Raspberry Pi model."),
        ("GPUs", "Lambda GPU instance status."),
        ("API", "Selected API model and request health."),
        ("GPU inference", "Model upload progress and readiness."),
        ("Pi CPU", "Pi model status and CPU temperature (°F)."),
        ("Using", "Provider selected for your next message."),
        ("Sentinel", "Scope observer and its model; reacts by stopping tasks."),
        ("Data", "Fine-tuning data collection on or off."),
        ("Access", "Agent filesystem access mode."),
    )
    result = Text(style=TEXT_GRAY)
    for index, (label, description) in enumerate(rows):
        if index:
            result.append("\n")
        result.append(label + ": ", style=f"bold {NEON_BLUE}")
        result.append(description)
    return result


def transcript_text(message, kind="text"):
    if kind == "sentinel_stop":
        return Text(message, style="bold #ff3b30")
    if kind == "command":
        return Text(message, style=NEON_PINK)
    if kind == "output":
        return Text(message, style=CODE_GRAY)

    result = Text(style=TEXT_GRAY)
    marker, language = "", ""
    for line in message.splitlines(keepends=True):
        match = FENCE.match(line)
        if marker:
            closing = (
                match
                and match[1][0] == marker[0]
                and len(match[1]) >= len(marker)
                and not match[2].strip()
            )
            command = (
                not closing
                and line.strip()
                and not line.lstrip().startswith("#")
                and (
                    language in SHELL_LANGUAGES
                    or (language in CONSOLE_LANGUAGES and line.lstrip().startswith(("$ ", "> ")))
                )
            )
            result.append(line, style=NEON_GREEN if command else CODE_GRAY)
            if closing:
                marker, language = "", ""
        elif match:
            marker = match[1]
            language = match[2].strip().split(" ", 1)[0].lower()
            result.append(line, style=CODE_GRAY)
        else:
            result.append(line, style=NEON_BLUE if line.lstrip().startswith("$ ") else TEXT_GRAY)
    return result
