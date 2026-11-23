"""Commands offered by the slash menu."""

CONTROLS = [
    ("gpu", "GPU / GPU Cluster"),
    ("local", "Pi CPU or Lambda GPU"),
    ("api", "API Model"),
    ("sentinel", "Scope observer — ON / OFF"),
    ("data-collection", "Data Collection — Fine-tuning Data"),
    ("upload-perms", "Dataset uploads — ON / OFF"),
    ("terminate", "Terminate Instance"),
    ("resume", "Resume saved conversation"),
    ("sandbox", "Sandbox / filesystem access"),
]

COMMANDS = CONTROLS + [
    ("new", "Start a new conversation"),
    ("queue", "View, pause, resume or clear queued prompts"),
    ("help", "Show command help"),
    ("setup", "Secure setup — credentials, datasets, SSH and desktop launcher"),
    ("updates-info", "Harness/engine versions and security advisories"),
    ("update", "Manually update the harness, engine, or both"),
    ("home", "Return to the main menu"),
    ("quit", "End session and terminate its GPU"),
]


def command_name(name):
    """User-facing command name; keep internal routing IDs and shortcuts stable."""
    return "local & cloud models" if name == "local" else name
