"""Transcripts on disk. One JSONL file per chat."""

import json
from datetime import datetime
from pathlib import Path

PROJECT = Path.cwd().resolve()
SESSION_DIR = PROJECT / ".skippy" / "sessions"
LEGACY_DIR = Path.home() / ".agents" / "sessions" / str(PROJECT).replace("/", "-")
CURRENT = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
WRITTEN = 0  # how many messages are already on disk
BASE_TODOS = []  # plan at the most recent compaction boundary


def path_for(session_id):
    return SESSION_DIR / f"{session_id}.jsonl"


def save(messages):
    """Append what is new. Never rewrite what is already on disk."""
    global WRITTEN
    append_entries(messages[WRITTEN:])
    WRITTEN = len(messages)


def append_entries(entries):
    """Keep a torn final JSON line from swallowing the next journal entry."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    with path_for(CURRENT).open("ab+") as stream:
        if stream.tell():
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        for entry in entries:
            stream.write((json.dumps(entry) + "\n").encode("utf-8"))


def rewind_to(count):
    """Record a rewind as an entry, so the old messages stay in the file."""
    global WRITTEN
    append_entries([{"rewind_to": count}])
    WRITTEN = count


def compacted(messages):
    """Compaction rewrites history, so record the result and start from it."""
    global WRITTEN, BASE_TODOS
    from .todos import TODOS
    append_entries([{"compacted": messages, "todos": TODOS}])
    BASE_TODOS = json.loads(json.dumps(TODOS))
    WRITTEN = len(messages)


def load(session_id, restore_plan=False, repair_history=True):
    """Replay the log: messages accumulate, rewinds cut them back."""
    global BASE_TODOS
    messages = []
    from . import todos
    path = path_for(session_id)
    if not path.exists():
        path = LEGACY_DIR / f"{session_id}.jsonl"
    plan = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # A half-written last line, usually from a kill mid-save. Skipping
            # it costs one message; raising would break /sessions for every
            # chat in the project, because listing them all calls load().
            continue
        if not isinstance(entry, dict):
            continue
        if "rewind_to" in entry:
            del messages[entry["rewind_to"]:]
            if not any("<summary>" in (m.get("content") or "") for m in messages):
                plan = None
        elif "compacted" in entry:
            messages = list(entry["compacted"])
            plan = entry.get("todos")
        else:
            if entry.get("role") in ("system", "user", "assistant", "tool"):
                messages.append(entry)
    if repair_history:
        messages = repair(messages)
    if restore_plan:
        BASE_TODOS = json.loads(json.dumps(plan or []))
        todos.restore(messages, plan)
    return messages


def repair(messages):
    """Complete interrupted tool batches so resuming produces a valid request."""
    repaired, pending = [], {}
    for message in messages:
        if message.get("role") != "tool":
            for call_id in pending:
                repaired.append({"role": "tool", "tool_call_id": call_id,
                                 "content": "Interrupted before a result was saved. Inspect current files before retrying."})
            pending = {}
        else:
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                continue
            del pending[call_id]
        repaired.append(message)
        if message.get("tool_calls"):
            pending = {c["id"]: c for c in message["tool_calls"]}
    for call_id in pending:
        repaired.append({"role": "tool", "tool_call_id": call_id,
                         "content": "Interrupted before a result was saved. Inspect current files before retrying."})
    return repaired


def open_session(session_id):
    """Switch to a past chat and become it."""
    global CURRENT, WRITTEN
    messages = load(session_id, restore_plan=True)
    CURRENT = session_id
    # Rebuild when importing a legacy log or repairing an interrupted batch.
    raw = load(session_id, repair_history=False)
    WRITTEN = len(messages)
    if not path_for(session_id).exists():
        WRITTEN = 0
        save(messages)
        compacted(messages)
    elif raw != messages:
        compacted(messages)
    return messages


def title(messages):
    for message in messages:
        if message["role"] == "user" and "<summary>" not in (message.get("content") or ""):
            return " ".join(str(message.get("content") or "").split())[:60]
    return "(empty)"


def all_sessions():
    """Newest first."""
    files_by_id = {p.stem: p for p in LEGACY_DIR.glob("*.jsonl")}
    files_by_id.update({p.stem: p for p in SESSION_DIR.glob("*.jsonl")})
    files = sorted(files_by_id.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"id": p.stem, "title": title(load(p.stem))} for p in files]
