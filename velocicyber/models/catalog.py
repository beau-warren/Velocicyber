"""Explicit text/tool profiles for models configured in this harness.

Context limits come from the active serving profile, not model training capacity.
Unregistered models retain the upstream fallback warning.
"""

from velocicyber.config import API_MODELS
from velocicyber.models.gpu_profiles import GPU_PROFILES
from velocicyber.models.pi_profiles import PI_PROFILES

SUPPORTED_MODELS = {
    *GPU_PROFILES,
    *PI_PROFILES,
    *(profile[1] for profile in API_MODELS.values()),
}


def model_catalog(model, budget, system_prompt):
    if model not in SUPPORTED_MODELS:
        return None
    return {
        "models": [
            {
                "slug": model,
                "display_name": model,
                "description": "Velocicyber text and coding-tool integration",
                "supported_reasoning_levels": [],
                "reasoning_control": "none",
                "shell_type": "default",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 0,
                "model_messages": {"instructions_template": system_prompt},
                "include_skills_usage_instructions": False,
                "include_plugin_usage_instructions": False,
                "include_apps_usage_instructions": False,
                "supports_reasoning_summaries": False,
                "supports_reasoning_summary_parameter": False,
                "support_verbosity": False,
                "truncation_policy": {"mode": "bytes", "limit": budget.tool_output_limit * 4},
                "context_window": budget.window,
                "max_context_window": budget.window,
                "auto_compact_token_limit": budget.compact_at,
                "effective_context_window_percent": 95,
                "experimental_supported_tools": [],
                "input_modalities": ["text"],
                "supports_search_tool": False,
                "tool_mode": "direct",
            }
        ]
    }
