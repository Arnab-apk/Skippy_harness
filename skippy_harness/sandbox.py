"""Kernel-enforced limits on what bash can touch.

One policy - read anything, write only inside the project, no network - and a
different enforcement mechanism per OS. The idea ports; the mechanism never does.

The macOS profile is adapted from openai/codex (Apache-2.0), simplified.
https://github.com/openai/codex
"""

import shutil
import subprocess
import sys
import tempfile
import re
from pathlib import Path

PROJECT = Path.cwd().resolve()

PROFILE = f"""(version 1)
(deny default)
(allow process-exec process-fork signal)
(allow file-read*)
(allow sysctl-read)
(deny network*)
(allow file-write* (subpath "{PROJECT}") (literal "/dev/null"))
(deny file-write* (subpath "{PROJECT}/.git"))
"""


def wrap(command):
    """Wrap a shell command in an OS sandbox. None means we have no sandbox."""
    if sys.platform == "darwin":
        profile = Path(tempfile.gettempdir()) / "skippy_harness.sb"
        profile.write_text(PROFILE)
        return ["sandbox-exec", "-f", str(profile), "/bin/sh", "-c", command]

    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return [
            "bwrap",
            "--ro-bind", "/", "/",
            "--bind", str(PROJECT), str(PROJECT),
            "--dev", "/dev", "--proc", "/proc",
            "--unshare-net", "--die-with-parent",
            "/bin/sh", "-c", command,
        ]

    return None  # Windows, or Linux without bubblewrap


def name():
    if sys.platform == "darwin":
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bubblewrap"
    return "none"


def command_error(command):
    """Explain common incompatible commands before asking to execute them."""
    if sys.platform != "win32":
        return None
    # Ignore quoted strings and PowerShell backtick escapes. Never rewrite a
    # compound command: changing its separators could change its meaning.
    visible, quote, escaped = [], None, False
    for char in command:
        if escaped:
            visible.append(" ")
            escaped = False
        elif char == "`":
            escaped = True
            visible.append(" ")
        elif quote:
            if char == quote:
                quote = None
            visible.append(" ")
        elif char in "\"'":
            quote = char
            visible.append(" ")
        else:
            visible.append(char)
    unquoted = "".join(visible)
    if ("&&" in unquoted or "||" in unquoted) and not shutil.which("pwsh"):
        return "Windows PowerShell 5.1 does not support && or ||. Run separate bash tool calls, or use native PowerShell control flow."
    if re.search(r"(?:^|[;|&])\s*(?:ls|dir)\s+(?:-(?:la|al|l|a|lh|lah|alh)|/[bBsS])(?:\s|$)", unquoted):
        return "This shell is PowerShell, not bash or cmd. Use Get-ChildItem -Name to list filenames or Get-ChildItem -Force to include hidden files."
    if re.search(r"(?:^|[;|&])\s*which(?:\s|$)", unquoted):
        return "Use Get-Command <program> to find a program in Windows PowerShell."
    return None


def run(command, timeout=60):
    """Run a command, sandboxed when the OS lets us."""
    error = command_error(command)
    if error:
        return subprocess.CompletedProcess(command, 1, "", error)
    sandboxed = wrap(command)
    invocation = sandboxed or command
    if sys.platform == "win32":
        executable = shutil.which("pwsh") or shutil.which("powershell")
        if not executable:
            raise RuntimeError("PowerShell was not found. Install PowerShell to run shell tools on Windows.")
        script = ("$ErrorActionPreference = 'Stop'\n"
                  "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding\n"
                  + command + "\nif ($null -ne $LASTEXITCODE) { exit $LASTEXITCODE }")
        invocation = [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script]
    return subprocess.run(
        invocation,
        shell=sandboxed is None and sys.platform != "win32",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT,
        timeout=timeout,
    )
