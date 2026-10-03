"""The plan. Lives here, not in the transcript, and is re-injected every turn."""

import json

MARKS = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}

TODOS = []  # [{"content": ..., "activeForm": ..., "status": ...}]


def write_todos(todos):
    """Replace the whole list. Exactly one task may be in_progress."""
    if not isinstance(todos, list) or any(
        not isinstance(t, dict) or not isinstance(t.get("status"), str) or t.get("status") not in MARKS
        or not isinstance(t.get("content"), str) or not isinstance(t.get("activeForm"), str)
        for t in todos
    ):
        return "Error: todos must be a list with content, activeForm, and a valid status."
    active = [t for t in todos if t["status"] == "in_progress"]
    if len(active) > 1:
        return f"Error: {len(active)} tasks are in_progress. Only one may be."

    TODOS[:] = todos
    return todos_prompt() or "Todo list cleared."


def restore(messages, initial=None):
    TODOS.clear()
    if initial is not None:
        write_todos(initial)
    results = {m.get("tool_call_id"): m.get("content", "") for m in messages if m.get("role") == "tool"}
    for message in messages:
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] != "write_todos":
                continue
            result = results.get(call["id"], "Error:")
            if result.startswith(("Error:", "Blocked", "The user denied", "Interrupted", "Skipped:")):
                continue
            try:
                write_todos(json.loads(call["function"]["arguments"])["todos"])
            except (ValueError, TypeError, KeyError):
                continue


def todos_prompt():
    return "\n".join(f"{MARKS[t['status']]} {t['content']}" for t in TODOS)


def active_form():
    """What the agent is doing right now, for the spinner."""
    for todo in TODOS:
        if todo["status"] == "in_progress":
            return todo["activeForm"]
    return "thinking"


TODO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_todos",
        "description": (
            "Record the plan for a multi-step task. Send the whole list every "
            "time. Keep exactly one task in_progress and update it as you go."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The task, imperative: 'Fix the parser'",
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Present continuous: 'Fixing the parser'",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "done"],
                            },
                        },
                        "required": ["content", "activeForm", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
    },
}
