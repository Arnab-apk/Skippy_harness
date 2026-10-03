"""Which tool calls need a human.

The sandbox decides what is *possible*. These rules only decide what is worth
interrupting you for - so read-only commands run silently, and the risky ones
still stop and ask.
"""

from fnmatch import fnmatch
from pathlib import Path
import re

PROJECT = Path.cwd().resolve()

# Last matching rule wins, so put the catch-all first.
BASH_RULES = {
    "*": "ask",
    # read-only: let them through
    "ls": "allow",
    "ls *": "allow",
    "dir": "allow",
    "dir *": "allow",
    "pwd": "allow",
    "cd *": "allow",
    "echo *": "allow",
    "sort *": "allow",
    "uniq *": "ask",
    "cut *": "allow",
    "basename *": "allow",
    "dirname *": "allow",
    "date": "allow",
    "env": "allow",
    "cat *": "allow",
    "head *": "allow",
    "tail *": "allow",
    "wc *": "allow",
    "file *": "allow",
    "which *": "allow",
    "grep *": "allow",
    "rg *": "allow",
    "find *": "ask",
    "tree": "allow",
    "tree *": "allow",
    "git status": "allow",
    "git status *": "allow",
    "git diff": "allow",
    "git diff *": "allow",
    "git log": "allow",
    "git log *": "allow",
    "git show": "allow",
    "git show *": "allow",
    "git ls-files": "allow",
    "git ls-files *": "allow",
    "pytest*": "ask",
    "python -m pytest*": "ask",
    "Get-ChildItem": "allow",
    "Get-ChildItem *": "allow",
    "Get-Content *": "allow",
    "Get-Location": "allow",
    "Get-Command": "allow",
    "Get-Command *": "allow",
    "Select-String *": "allow",
    "Test-Path *": "allow",
    # risky: never, even if the user says yes
    "rm *": "deny",
    "sudo *": "deny",
    "chmod *": "deny",
    "chown *": "deny",
    "curl *": "deny",
    "wget *": "deny",
    "git push*": "deny",
    "git reset*": "deny",
    "git clean*": "deny",
}

def split_command(command):
    """Split a compound command on the separators that actually separate.

    A naive split on | and ; also cuts inside quotes, so `rg "cap|max"` breaks
    into fragments that match no rule and fall through to "ask". Anything
    quoted or backslash-escaped is an argument, not a separator.
    """
    parts, current, quote = [], [], None
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            current.append(char)
            quote = None if char == quote else quote
        elif char == "\\":
            current.append(char)
            index += 1
            if index < len(command):
                current.append(command[index])
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char in "&|;":
            parts.append("".join(current))
            current = []
            while index + 1 < len(command) and command[index + 1] in "&|":
                index += 1
        else:
            current.append(char)
        index += 1

    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def decide(command):
    """Rate every part of a compound command; the strictest verdict wins."""
    # Redirection, expansion, script blocks, newlines and command substitution
    # require approval even when the leading executable looks read-only.
    needs_approval = (re.search(r"[\r\n<>`$(){}]", command) or
                      re.search(r"--(?:pre|ext-diff|textconv|output|exec)(?:[=\s]|$)|(?:^|\s)-(?:o|exec|delete)(?:[=\s]|$)", command))
    verdicts = []
    for part in split_command(command):
        action = "ask"
        for pattern, rule in BASH_RULES.items():
            if fnmatch(part.lower(), pattern.lower()):
                action = rule
        verdicts.append(action)

    for strictest in ("deny", "ask"):
        if strictest in verdicts:
            return strictest
    return "ask" if needs_approval else "allow"


def inside_project(path):
    resolved = Path(path).resolve()
    return resolved == PROJECT or PROJECT in resolved.parents


def check(name, args):
    """Return (action, reason). Action is allow, ask or deny."""
    if name == "bash":
        return decide(args["command"]), f"run: {args['command']}"

    if name in ("write_file", "str_replace") and not inside_project(args["path"]):
        return "ask", f"{name} outside {PROJECT}: {args['path']}"
    if name in ("write_file", "str_replace"):
        resolved = Path(args["path"]).resolve()
        if ".git" in resolved.relative_to(PROJECT).parts:
            return "deny", "Direct changes to Git internals are blocked."

    return "allow", None
