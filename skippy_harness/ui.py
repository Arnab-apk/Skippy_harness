"""Terminal presentation layer.

Knows nothing about LLMs, providers or tools - it only receives plain strings
and dicts and decides how they look.
"""

import json
import os
import sys
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

MAX_TOOL_OUTPUT_LINES = 12

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

    # ---------------------------------------------------------------- input

    def banner(self, sandbox_name="none", provider="", model=""):
        self.console.print()
        self.console.print(
            Rule(Text(" Skippy Harness ", style=f"bold {ACCENT}"), style=MUTED)
        )
        parts = []
        if provider:
            parts.append(f"provider: {provider}")
        if model:
            parts.append(f"model: {model}")
        parts.append(f"sandbox: {sandbox_name}")
        info_line = "  ·  ".join(parts) + "  ·  opt-enter for a newline  ·  ctrl-d to exit"
        self.console.print(
            Padding(Text(info_line, style=MUTED), (0, 0, 0, 2))
        )

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

    def tool(self, name, args, result, nested=False):
        if name == "write_todos" and args.get("todos") and not result.startswith("Error:"):
            return self.todos(args["todos"])

        header = Text.assemble(
            (f"{name} ", f"bold {TOOL}"),
            (self._format_args(args), MUTED),
        )
        self.console.print(
            Padding(
                Panel(
                    Group(header, Rule(style=MUTED), self._format_result(result)),
                    border_style=MUTED,
                    padding=(0, 1),
                ),
                (1, 2, 0, 6 if nested else 2),
            )
        )

    def subagent(self, description):
        """Shown to you, never to the main agent - it only gets the report."""
        self.console.print(
            Padding(
                Panel(
                    Text(description.strip(), style=MUTED),
                    title=Text("subagent · own context", style=f"bold {ACCENT}"),
                    title_align="left",
                    border_style=ACCENT,
                    padding=(0, 1),
                ),
                (1, 2, 0, 4),
            )
        )

    def injection(self, text):
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

    # ---------------------------------------------------------------- usage

    def usage(self, stats):
        for key, value in stats.items():
            self._totals[key] = self._totals.get(key, 0) + (value or 0)

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

        self.console.print(
            Padding(
                Panel(
                    rows,
                    title=Text(f"todos {done}/{len(todos)}", style=f"bold {TOOL}"),
                    title_align="left",
                    border_style=MUTED,
                    padding=(0, 1),
                ),
                (1, 2, 0, 2),
            )
        )

    # -------------------------------------------------------------- helpers

    def _format_args(self, args):
        if len(args) == 1:
            return str(next(iter(args.values())))
        return json.dumps(args)

    def _format_result(self, result):
        lines = result.strip().splitlines() or ["(no output)"]
        shown = lines[:MAX_TOOL_OUTPUT_LINES]
        body = Text("\n".join(shown), style=MUTED)
        hidden = len(lines) - len(shown)
        if hidden > 0:
            body.append(f"\n… {hidden} more lines", style=f"italic {TOOL}")
        return body


ui = UI()
