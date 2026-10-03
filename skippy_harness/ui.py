"""Terminal presentation layer.

Knows nothing about LLMs, providers or tools - it only receives plain strings
and dicts and decides how they look.
"""

import json
import os
import sys
import re
import difflib
from collections import OrderedDict
from pathlib import Path
from contextlib import contextmanager

from rich.console import Console, Group
from rich.json import JSON
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import prompt
from .todos import MARKS

ACCENT = "#7aa2f7"
USER = "#9ece6a"
TOOL = "#e0af68"
MUTED = "#565f89"
OUTPUT = "#a9b1d6"
ERROR = "#f7768e"

MAX_TOOL_OUTPUT_LINES = 4

LOGO = r"""   _____ __ __ ____ ____ ____ __  __
  / ___// //_/  _// __ \/ __ \\ \/ /
  \__ \/ ,<  / / / /_/ / /_/ /\  /
 ___/ / /| |/ / / ____/ ____/ / /
/____/_/ |_/___/_/   /_/     /_/"""

TOOL_LABELS = {
    "bash": "Shell", "read_file": "Read", "write_file": "Write",
    "str_replace": "Edit", "read_skill": "Skill", "task": "Explore",
    "write_todos": "Plan",
}

TODO_STYLES = {
    "done": f"{MUTED} strike",
    "in_progress": f"bold {ACCENT}",
    "pending": MUTED,
}


class UI:
    def __init__(self):
        if os.name == "nt":
            for stream in (sys.stdout, sys.stderr):
                if hasattr(stream, "reconfigure"):
                    try:
                        stream.reconfigure(encoding="utf-8", errors="replace")
                    except (OSError, ValueError):
                        pass
        self.console = Console()
        self._totals = {}
        self.verbose = False
        self.tools_expanded = False
        self._tool_results = OrderedDict()

    # ---------------------------------------------------------------- input

    def banner(self, sandbox_name="none", provider="", model="", art=True):
        self.console.print()
        if art:
            logo = LOGO if self.console.width >= 44 else "S K I P P Y"
            colors = ["#bb9af7", "#9d7cd8", ACCENT, "#7dcfff", "#73daca"]
            for index, line in enumerate(logo.splitlines()):
                self.console.print(Padding(Text(line, style=f"bold {colors[index % len(colors)]}"), (0, 0, 0, 2)))
            self.console.print(Padding(Text("terminal coding companion", style=OUTPUT), (0, 0, 0, 3)))
            self.console.print()
        else:
            self.console.print(Padding(Text("SKIPPY", style=f"bold {ACCENT}"), (0, 0, 0, 2)))
        parts = []
        if provider:
            parts.append(provider)
        if model:
            parts.append(model)
        parts.append(f"sandbox: {sandbox_name}")
        info_line = "  /  ".join(parts)
        if self.console.width < 60:
            info_line = " / ".join(parts[:-1]) + "\n" + parts[-1]
        self.console.print(
            Padding(Text(info_line, style=OUTPUT), (0, 0, 0, 2))
        )
        shortcuts = "/model  /provider  |  Ctrl+O: details  |  Alt+Enter: newline  |  Ctrl+D: exit"
        if self.console.width < 70:
            shortcuts = "/model  /provider\nCtrl+O: details  Ctrl+D: exit"
        self.console.print(Padding(Text(shortcuts, style=MUTED), (0, 0, 0, 2)))
        self.console.print(Padding(Rule(style=MUTED), (1, 2, 0, 2)))

    def clear(self):
        self.console.clear()

    def resumed(self, messages, label="resumed"):
        turns = sum(1 for m in messages if m["role"] == "user")
        self.console.print(
            Padding(
                Text(f"{label} · {len(messages)} messages · {turns} turns", style=MUTED),
                (0, 0, 0, 2),
            )
        )

    def replay(self, messages):
        """Redraw a loaded transcript so the screen matches the history."""
        results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
        for message in messages:
            if message["role"] == "user":
                self.user(message["content"])
            elif message["role"] == "assistant":
                if message.get("content"):
                    self.agent(message["content"])
                for call in message.get("tool_calls") or []:
                    try:
                        args = json.loads(call["function"]["arguments"])
                        if not isinstance(args, dict):
                            args = {"arguments": args}
                    except (TypeError, json.JSONDecodeError):
                        args = {"arguments": call["function"]["arguments"]}
                    self.tool(
                        call["function"]["name"],
                        args,
                        results.get(call["id"], ""),
                        call_id=call["id"],
                    )

    def approve(self, reason):
        self.console.print(Padding(Text(reason, style=f"bold {TOOL}"), (1, 0, 0, 2)))
        try:
            answer = prompt.read("  allow? (y/n)> ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.lower().startswith("y")

    def note(self, text):
        self.console.print(Padding(Text(text, style=MUTED), (1, 0, 0, 2)))

    def pick(self, title, rows):
        """Searchable, paginated list using the same 1-based numbering as setup."""
        indices, page = list(range(len(rows))), 0
        while True:
            self.console.print(Padding(Text(title, style=f"bold {ACCENT}"), (1, 0, 0, 2)))
            visible = indices[page * 20:(page + 1) * 20]
            for i in visible:
                self.console.print(Padding(Text(f"{i + 1:>3}  {rows[i]}", style=MUTED), (0, 0, 0, 2)))
            try:
                answer = prompt.read("\n  number, search text, n/p page, Enter to cancel> ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not answer:
                return None
            if answer.isdigit():
                selected = int(answer) - 1
                if selected in visible:
                    return selected
                self.note("Choose a number displayed on this page.")
            elif answer.lower() == "n":
                if (page + 1) * 20 < len(indices):
                    page += 1
            elif answer.lower() == "p":
                page = max(0, page - 1)
            else:
                indices = [i for i, row in enumerate(rows) if answer.lower() in row.lower()]
                page = 0
                if not indices:
                    self.note("No matches. Enter another search term.")

    def ask(self):
        self.console.print()
        try:
            return prompt.read("> ").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return None

    # --------------------------------------------------------------- output

    def user(self, text):
        self.console.print(
            Padding(Text(text.strip(), style=f"bold {USER}"), (1, 0, 0, 2))
        )

    def agent(self, text):
        self.console.print(
            Padding(
                Group(
                    Text("agent", style=f"bold {ACCENT}"),
                    Padding(Markdown(text.strip()), (1, 0, 0, 0)),
                ),
                (1, 2, 0, 2),
            )
        )

    def tool(self, name, args, result, nested=False, call_id=None):
        if call_id:
            result = self._tool_results.get(call_id, result)
            self._tool_results[call_id] = result
            self._tool_results.move_to_end(call_id)
            while len(self._tool_results) > 200:
                self._tool_results.popitem(last=False)
        if name == "write_todos" and args.get("todos") and not result.startswith("Error:"):
            return self.todos(args["todos"])

        failed, status = self._tool_status(result)
        color = ERROR if failed else USER
        label = TOOL_LABELS.get(name, name)
        detail = self._tool_detail(name, args)
        if not self.tools_expanded:
            detail_text = Text(detail)
            detail_text.truncate(max(8, self.console.width - len(label) - len(status) - (6 if nested else 2) - 12), overflow="ellipsis")
            detail = detail_text.plain
        header = Text.assemble(("! " if failed else "+ ", color),
                               (label, f"bold {TOOL}"), (f"  {detail}", OUTPUT),
                               (f"  [{status}]", color))
        indent = 6 if nested else 2
        self.console.print(Padding(header, (1, 2, 0, indent)))
        expanded = self.tools_expanded
        if name == "str_replace" and not failed and isinstance(args.get("old_str"), str) and isinstance(args.get("new_str"), str):
            difference = list(difflib.unified_diff(args["old_str"].splitlines(), args["new_str"].splitlines(), lineterm="", n=1))[2:]
            body = "\n".join(difference)
        elif name == "write_file" and not failed and not expanded:
            return  # The path and line count are the useful result of a write.
        elif name == "write_file" and not failed and expanded:
            body = str(args.get("content", ""))
        else:
            body = re.sub(r"^Exit code: -?\d+\n?", "", result)
        if body.strip():
            rendered = self._format_result(body, expanded=expanded, failed=failed, diff=name == "str_replace" and not failed)
            self.console.print(Padding(rendered, (0, 2, 0, indent + 2)))

    def subagent(self, description):
        """Shown to you, never to the main agent - it only gets the report."""
        self.console.print(Padding(Text.assemble(("> Explore  ", f"bold {ACCENT}"), (self._one_line(description), OUTPUT)), (1, 2, 0, 2)))

    def injection(self, text):
        if not self.verbose:
            return
        self.console.print(
            Padding(
                Panel(
                    Text(text.strip(), style=MUTED),
                    title=Text("late injection", style=f"italic {MUTED}"),
                    title_align="left",
                    border_style=MUTED,
                    padding=(0, 1),
                ),
                (1, 2, 0, 2),
            )
        )

    def debug(self, data):
        self.console.print(
            Padding(
                Panel(
                    JSON.from_data(data),
                    title=Text("raw response", style=f"italic {MUTED}"),
                    title_align="left",
                    border_style=TOOL,
                    padding=(0, 1),
                ),
                (1, 2, 0, 2),
            )
        )

    @contextmanager
    def working(self, label="thinking"):
        with self.console.status(
            Text(label, style=MUTED), spinner="dots", spinner_style=ACCENT
        ):
            yield

    @contextmanager
    def executing(self, name, args):
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        if name == "task":  # Its own loop renders a spinner.
            yield
        else:
            label = TOOL_LABELS.get(name, name)
            with self.working(f"{label.lower()} {self._tool_detail(name, args)}"):
                yield

    def toggle_tools(self):
        self.tools_expanded = not self.tools_expanded
        return "tool details expanded" if self.tools_expanded else "tool details collapsed"

    # ---------------------------------------------------------------- usage

    def usage(self, stats):
        for key, value in stats.items():
            self._totals[key] = self._totals.get(key, 0) + (value or 0)

        if not self.verbose:
            return

        parts = " · ".join(
            f"{value:,} {key.replace('_tokens', '')}"
            for key, value in stats.items()
            if value
        )
        self.console.print(Padding(Text(parts, style=MUTED), (1, 0, 0, 2)))

    def summary(self):
        if not self._totals:
            return

        table = Table.grid(padding=(0, 2))
        table.add_column(style=MUTED)
        table.add_column(style=f"bold {ACCENT}", justify="right")
        for key, value in self._totals.items():
            table.add_row(key.replace("_", " "), f"{value:,}")

        self.console.print(Padding(table, (1, 2)))
        self.console.print(Rule(style=MUTED))
        self.console.print()

    def compacted(self, before, messages):
        summary = next(
            (m["content"] for m in messages if "<summary>" in (m.get("content") or "")),
            "",
        )
        self.console.print(
            Padding(
                Panel(
                    Markdown(summary.replace("<summary>", "").replace("</summary>", "")),
                    title=Text(
                        f"compacted · {before} → {len(messages)} messages",
                        style=f"bold {TOOL}",
                    ),
                    title_align="left",
                    border_style=TOOL,
                    padding=(0, 1),
                ),
                (1, 2, 0, 2),
            )
        )

    def todos(self, todos):
        """The plan, as a checklist. The raw tool output is never worth showing."""
        done = sum(1 for t in todos if t["status"] == "done")

        if not self.tools_expanded:
            active = next((t["content"] for t in todos if t["status"] == "in_progress"), "")
            suffix = f"  {active}" if active else ""
            line = Text.assemble(("+ Plan  ", f"bold {TOOL}"), (f"{done}/{len(todos)} done", USER), (suffix, OUTPUT))
            self.console.print(Padding(line, (1, 2, 0, 2)))
            return

        rows = Table.grid(padding=(0, 1))
        rows.add_column(no_wrap=True)
        rows.add_column(overflow="fold")
        for todo in todos:
            status = todo["status"]
            style = TODO_STYLES[status]
            rows.add_row(
                Text(MARKS[status], style=style),
                Text(todo["content"], style=style),
            )

        self.console.print(Padding(Text(f"+ Plan  {done}/{len(todos)} done", style=f"bold {TOOL}"), (1, 2, 0, 2)))
        self.console.print(Padding(rows, (0, 2, 0, 4)))

    # -------------------------------------------------------------- helpers

    def _format_args(self, args):
        if len(args) == 1:
            return str(next(iter(args.values())))
        return json.dumps(args)

    def _one_line(self, value):
        return " ".join(str(value).split())

    def _tool_detail(self, name, args):
        if name == "bash":
            return self._one_line(args.get("command", ""))
        if name in ("read_file", "write_file", "str_replace"):
            path = str(args.get("path", ""))
            try:
                path = Path(path).resolve().relative_to(Path.cwd()).as_posix()
            except (OSError, ValueError):
                pass
            if name == "write_file" and isinstance(args.get("content"), str):
                path += f"  ({len(args['content'].splitlines())} lines)"
            return path
        return self._one_line(args.get("description", args.get("name", self._format_args(args))))

    def _tool_status(self, result):
        exit_match = re.match(r"Exit code: (-?\d+)", result)
        if exit_match:
            code = int(exit_match[1])
            return code != 0, "ok" if code == 0 else f"exit {code}"
        for prefix, status in (("Error:", "error"), ("Blocked by policy", "blocked"),
                               ("The user denied", "denied"), ("Timed out", "timeout"),
                               ("Interrupted before", "interrupted"), ("No skill named", "missing"),
                               ("Skipped:", "skipped"),
                               ("(stopped after", "incomplete")):
            if result.startswith(prefix):
                return True, status
        return False, "ok" if result else "no result"

    def _format_result(self, result, expanded=False, failed=False, diff=False):
        lines = result.strip().splitlines() or ["(no output)"]
        shown = lines if expanded else lines[:MAX_TOOL_OUTPUT_LINES]
        body = Text()
        width = max(12, self.console.width - 10)
        for index, line in enumerate(shown):
            if index:
                body.append("\n")
            style = ERROR if failed else OUTPUT
            if diff:
                style = ERROR if line.startswith("-") else USER if line.startswith("+") else MUTED
            rendered_line = Text(line, style=style)
            if not expanded:
                rendered_line.truncate(width, overflow="ellipsis")
            body.append_text(rendered_line)
        hidden = len(lines) - len(shown)
        if hidden > 0:
            body.append(f"\n... {hidden} more lines | Ctrl+O or /tools to expand", style=MUTED)
        return body


ui = UI()
