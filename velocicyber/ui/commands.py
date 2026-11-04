"""Commands offered by the slash menu."""

CONTROLS = [
    ("gpu", "GPU / GPU Cluster"),
    ("local", "Local Model Selection"),
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
    ("updates-info", "Engine updates and security advisories"),
    ("update-engine", "Manually update or roll back the execution engine"),
    ("home", "Return to the main menu"),
    ("quit", "End session and terminate its GPU"),
]
