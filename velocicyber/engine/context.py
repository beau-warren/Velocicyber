"""Context budgets for the window actually configured on a model server."""

from dataclasses import dataclass

DEFAULT_CONTEXT_WINDOW = 32768
COMPACTION_PROMPT = """Create a concise checkpoint so Velocicyber can continue this coding task.
Preserve the user's goal and constraints, project instructions, decisions, files changed,
tests and their actual results, unresolved errors, and the exact next steps. Include
essential tool results and file paths. Distinguish completed work from plans and
unverified claims. Preserve information needed to continue safely. Do not execute
tools or continue the task in this response; return only the checkpoint summary.
""".strip()


@dataclass(frozen=True)
class ContextBudget:
    window: int = DEFAULT_CONTEXT_WINDOW

    def __post_init__(self):
        if type(self.window) is not int or self.window < 4096:
            raise ValueError("Model context window must be an integer of at least 4096 tokens.")

    @property
    def compact_at(self):
        return self.window * 3 // 4

    @property
    def tool_output_limit(self):
        return min(2048, self.window // 16)

    @property
    def skills_catalog_limit(self):
        # Fit the bundled skill descriptions on small local contexts. Larger
        # windows retain upstream's 2% allowance, capped at its supported maximum.
        return min(10000, max(self.window // 50, min(1024, self.window // 16)))


def served_context_window(model):
    """Read serving limits, never the model's advertised training context length."""
    value = model.get("max_model_len") or model.get("meta", {}).get("n_ctx")
    return ContextBudget(value).window if value is not None else None


def is_compaction_request(request):
    last = (request.get("messages") or [{}])[-1]
    content = last.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return last.get("role") == "user" and content == COMPACTION_PROMPT
