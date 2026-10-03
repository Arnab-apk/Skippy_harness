import json
import subprocess
from pathlib import Path
import os
import stat
import tempfile

from . import history
from . import sandbox
from .permissions import check
from .subagent import TASK_SCHEMA, task
from .skills import read_skill
from .todos import TODO_SCHEMA, write_todos


def bash(command: str) -> str:
    """Run a shell command and return its combined stdout and stderr."""
    try:
        result = sandbox.run(command)
    except subprocess.TimeoutExpired as expired:
        # Hand the failure back as a result. A slow command is the model's
        # problem to work around, not a reason to take the session down.
        return (
            f"Timed out after {expired.timeout}s and was killed. "
            "Narrow it down - search inside the working directory rather than /."
        )
    output = (result.stdout + result.stderr) or "(no output)"
    return history.cap(f"Exit code: {result.returncode}\n{output}")


def read_file(path: str) -> str:
    """Read a file and return its contents."""
    with open(path, encoding="utf-8") as f:
        return history.cap(f.read())


def write_file(path: str, content: str) -> str:
    """Create a file, or overwrite it if it already exists."""
    write_atomic(path, content)
    return f"Wrote {path}"


def write_atomic(path, content):
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        if destination.exists():
            temporary.chmod(stat.S_IMODE(destination.stat().st_mode))
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def str_replace(path, old_str, new_str, allow_multi_edit=False):
    """Swap exact text in a file. old_str must match exactly once."""
    if not old_str:
        return "Error: old_str must not be empty."
    with open(path, encoding="utf-8") as f:
        content = f.read()

    count = content.count(old_str)
    if count == 0:
        return f"Error: old_str was not found in {path}"
    if count > 1 and not allow_multi_edit:
        return (
            f"Error: old_str matches {count} times in {path}. "
            "Add surrounding lines to make it unique, "
            "or set allow_multi_edit to replace them all."
        )

    write_atomic(path, content.replace(old_str, new_str))
    return f"Replaced {count} match(es) in {path}"


def execute(tool_call, allowed_names=None, read_only=False):
    """Run one tool call through the permission layer.

    Shared by the main loop and by subagents, so a subagent is fenced in by
    exactly the same rules - it is not a way around them.

    A tool call is text the model wrote, so all of it is untrusted: the name
    may not exist, the arguments may not be JSON, and they may not match the
    signature. Every one of those comes back as a result the model can read
    and retry. None of them is allowed to end the session.
    """
    from .ui import ui

    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError) as broken:
        return {}, f"Error: arguments were not valid JSON ({broken})."

    if not isinstance(args, dict):
        return {}, "Error: tool arguments must be a JSON object."
    if allowed_names is not None and name not in allowed_names:
        return args, f"Blocked by policy: {name} is not available to this subagent."

    if name not in TOOLS:
        return args, f"Error: no tool named '{name}'. Available: {', '.join(TOOLS)}."

    parameters = next(s["function"]["parameters"] for s in TOOL_SCHEMAS if s["function"]["name"] == name)
    for required in parameters.get("required", []):
        if required not in args:
            return args, f"Error: {name} requires {required}."
    types = {"string": str, "boolean": bool, "array": list, "object": dict}
    for key, value in args.items():
        prop = parameters.get("properties", {}).get(key)
        if prop is None:
            return args, f"Error: unknown argument {key} for {name}."
        expected = types.get(prop.get("type"))
        if expected is not None and not isinstance(value, expected):
            return args, f"Error: {key} for {name} must be a {prop['type']}."

    try:
        action, reason = check(name, args)
        if read_only and (name in ("write_file", "str_replace", "write_todos", "task") or action != "allow"):
            return args, "Blocked by policy: exploration subagents may only use approved read-only tools."
        if action == "deny":
            return args, f"Blocked by policy: {reason}"
        if action == "ask" and not ui.approve(reason):
            return args, "The user denied this tool call."
        return args, TOOLS[name](**args)
    except TypeError as mismatch:
        return args, f"Error: wrong arguments for {name} ({mismatch})."
    except KeyError as missing:
        return args, f"Error: {name} needs an argument you did not send: {missing}."
    except Exception as failure:  # noqa: BLE001 - the model gets to see and retry
        return args, f"Error: {name} failed - {type(failure).__name__}: {failure}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return its combined stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "Open a skill by name and return its full instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to open",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a file, or overwrite it if it already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to write"},
                    "content": {"type": "string", "description": "The full contents"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": (
                "Replace exact text in a file. old_str must appear exactly once, "
                "so include surrounding lines if needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to edit"},
                    "old_str": {"type": "string", "description": "Exact text to find"},
                    "new_str": {"type": "string", "description": "Text to put in its place"},
                    "allow_multi_edit": {
                        "type": "boolean",
                        "description": "Replace every match instead of failing",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    TODO_SCHEMA,
    TASK_SCHEMA,
]

TOOLS = {
    "bash": bash,
    "read_file": read_file,
    "write_file": write_file,
    "str_replace": str_replace,
    "read_skill": read_skill,
    "write_todos": write_todos,
    "task": task,
}
