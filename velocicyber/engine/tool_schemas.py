"""Repair native-engine alias placeholders before presenting tools to a model.

Open Interpreter 0.0.55 advertises several native aliases as ``input: string``
although its handlers deserialize named JSON fields. These contracts match
codex-rs/core/src/tools/handlers/harness_aliases.rs. Only the exact placeholder
is replaced; real schemas supplied by a future engine are left intact.
"""

import copy

from velocicyber.models.pi_profiles import BUGTRACE_PI_REPO


def field(kind, description, **extra):
    return {"type": kind, "description": description, **extra}


def schema(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


PATH = field("string", "File path, relative to the workspace or absolute within permitted access.")
ALIASES = {
    "bash": (
        "Run a shell command within the current sandbox.",
        schema(
            {
                "command": field("string", "Shell command to execute."),
                "description": field("string", "Brief description of the command."),
                "timeout": field("integer", "Time to wait in milliseconds, up to 30000."),
                "run_in_background": field(
                    "boolean", "Return a session for a long-running command."
                ),
            },
            ["command"],
        ),
    ),
    "read": (
        "Read a UTF-8 text file. Read relevant code before editing it.",
        schema(
            {
                "path": PATH,
                "line_offset": field("integer", "First line to read (1-based).", minimum=1),
                "n_lines": field("integer", "Maximum number of lines to read.", minimum=1),
            },
            ["path"],
        ),
    ),
    "write": (
        "Create or overwrite a UTF-8 text file with content. Use empty content for an empty file.",
        schema(
            {
                "path": PATH,
                "content": field("string", "Exact text to write; may be empty."),
                "mode": field(
                    "string",
                    "Overwrite by default; append adds text to the end.",
                    enum=["overwrite", "append"],
                ),
            },
            ["path", "content"],
        ),
    ),
    "edit": (
        "Replace exact text in an existing file. Read the file first.",
        schema(
            {
                "path": PATH,
                "old_string": field("string", "Exact existing text to replace."),
                "new_string": field("string", "Replacement text."),
                "replace_all": field("boolean", "Replace all matches; default false."),
            },
            ["path", "old_string", "new_string"],
        ),
    ),
    "glob": (
        "Find paths matching a glob pattern, relative to the supplied directory or workspace.",
        schema(
            {
                "pattern": field("string", "Glob pattern, e.g. **/*.py."),
                "path": field("string", "Directory to search; defaults to the workspace."),
                "include_dirs": field("boolean", "Include directories; default true."),
            },
            ["pattern"],
        ),
    ),
    "grep": (
        "Search file contents using a regular expression; returns matching file paths.",
        schema(
            {
                "pattern": field("string", "Regular expression to search for."),
                "path": field("string", "File or directory to search; defaults to the workspace."),
                "glob": field("string", "Optional glob filter for searched files."),
            },
            ["pattern"],
        ),
    ),
}


def repair_tool_schemas(request):
    """Return the forwarded payload, preserving the original for capture provenance."""
    result = copy.deepcopy(request)
    tools = result.get("tools") or []
    question = next(
        (t["function"] for t in tools if t.get("function", {}).get("name") == "request_user_input"),
        None,
    )
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function = tool.get("function", {})
        name = function.get("name", "")
        if (
            function.get("description")
            != f"Open Interpreter harness compatibility alias for {name}."
        ):
            continue
        parameters = function.get("parameters", {})
        if parameters != {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Harness-native tool input."}
            },
            "additionalProperties": True,
        }:
            continue
        if name in {n for base in ALIASES for n in (base, base.capitalize())}:
            description, corrected = ALIASES[name.lower()]
            function["description"] = description
            function["parameters"] = copy.deepcopy(corrected)
        elif name == "AskUserQuestion" and question:
            # This alias delegates directly to RequestUserInputHandler.
            function["description"] = question.get("description", "Ask the user a question.")
            function["parameters"] = copy.deepcopy(question["parameters"])
    if request.get("model") == BUGTRACE_PI_REPO:
        # A small CPU model need not choose between duplicate spellings of the
        # same tool. Retain the native handlers and their actual argument schema.
        equivalents = {
            "Bash": "exec_command",
            "bash": "exec_command",
            "Read": "read",
            "Write": "write",
            "Edit": "edit",
            "Glob": "glob",
            "Grep": "grep",
            "AskUserQuestion": "request_user_input",
        }
        names = {t.get("function", {}).get("name") for t in tools}
        redundant = {
            t["function"]["name"]
            for t in request.get("tools") or []
            if t.get("function", {}).get("name") in equivalents
            and equivalents[t["function"]["name"]] in names
            and t["function"].get("description")
            == f"Open Interpreter harness compatibility alias for {t['function']['name']}."
        }
        result["tools"] = [t for t in tools if t.get("function", {}).get("name") not in redundant]
        result["parallel_tool_calls"] = False
    return result
