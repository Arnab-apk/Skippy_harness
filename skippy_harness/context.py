"""Machine context, kept separate from the user's actual request."""

import hashlib
import subprocess
from datetime import datetime

from .todos import todos_prompt
from pathlib import Path

LABELS = {"M": "modified", "D": "deleted", "A": "added", "??": "new"}


def git(command):
    result = subprocess.run(
        f"git {command}", shell=True, capture_output=True, text=True
    )
    return result.stdout


def file_hash(path):
    file = Path(path)
    return hashlib.md5(file.read_bytes()).hexdigest() if file.is_file() else None


def git_state():
    """path -> (status, content hash) for every file git sees as changed."""
    state = {}
    for line in git("status --porcelain").splitlines():
        path = line[3:]
        state[path] = (line[:2].strip(), file_hash(path))
    return state


LAST_STATE = git_state()


def file_changes():
    """Files whose status or contents moved since the previous turn."""
    global LAST_STATE
    now = git_state()
    changed = {p: v[0] for p, v in now.items() if LAST_STATE.get(p) != v}
    LAST_STATE = now
    return changed


def changes_note():
    changed = file_changes()
    if not changed:
        return ""
    lines = [f"{LABELS.get(code, code)}: {path}" for path, code in changed.items()]
    return (
        "\n<system-reminder>\n"
        "These files changed since your last turn. Read them again before "
        "editing:\n" + "\n".join(lines) + "\n</system-reminder>"
    )


def todos_note():
    plan = todos_prompt()
    return f"\n<todos>\n{plan}\n</todos>" if plan else ""


def reminder():
    """Current metadata, not an additional user message."""
    return {
        "role": "system",
        "content": (
            "<env>\n"
            f"time: {datetime.now():%Y-%m-%d %H:%M}\n"
            f"git branch: {git('branch --show-current').strip() or '(detached)'}\n"
            "</env>" + todos_note() + changes_note()
        ),
    }


def prepare_messages(messages, injection):
    """Merge metadata into a copy of the system message for this request.

    A trailing synthetic user turn confused local models into responding to
    the environment instead of the user. Keep tool/result ordering and the
    original transcript intact, with a single system message for templates
    that only support a system message at the beginning.
    """
    prepared = list(messages)
    metadata = "\n\nMachine context (metadata, not a user request):\n" + injection["content"]
    if prepared and prepared[0].get("role") == "system":
        prepared[0] = {**prepared[0], "content": (prepared[0].get("content") or "") + metadata}
    else:
        prepared.insert(0, {"role": "system", "content": metadata})
    return prepared
