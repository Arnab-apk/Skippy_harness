# Skippy Harness

A lightweight, modular, and extensible coding agent harness in Python built for autonomous terminal-driven pair programming.

## Features

- **Interactive Terminal Chat**: Powered by `prompt_toolkit` and `rich`, featuring multi-line input, history recall, and clean terminal rendering.
- **Core Tooling**: File inspection, creation, and surgical text replacements (`str_replace`), plus sandboxed shell command execution (`bash`).
- **Configurable LLM Backend**: Connects to any OpenAI-compatible API endpoint (OpenAI, DeepSeek, Ollama, OpenRouter, etc.).
- **Permissions & OS Sandboxing**: Enforces write boundaries and prevents network access where supported (Seatbelt on macOS, Bubblewrap on Linux).
- **Subagents**: Spawns isolated exploration subagents with separate context windows to search codebases without polluting the primary context.
- **Skills System**: Dynamically loads custom capabilities and operational guides from `.agents/skills`.
- **Todo & Task Planning**: Automatic multi-step task planning with `<todos>` injection into prompts for reliable execution.
- **Session Management**: Full conversation persistence with `/sessions` to browse/resume previous chats and `/rewind` to roll back turns.
- **Context Compaction**: Automated token budgeting and compaction to retain key context across long-running sessions, plus manual `/compact`.
- **Git State Tracking**: Tracks git branch and automatically injects reminders about modified, added, or deleted files between turns.

---

## Getting Started

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd Skippy_harness
uv sync
```

Alternatively, install using `pip`:

```bash
pip install -e .
```

### Configuration

Skippy Harness reads configuration from environment variables or `~/.agents/env`:

Create `~/.agents/env` or export environment variables:

```bash
# Required
BASE_URL="https://api.openai.com/v1"
API_KEY="your-api-key-here"

# Optional
MODEL="deepseek/deepseek-v4-flash"
CONTEXT_WINDOW=128000
```

---

## Usage

Start Skippy Harness:

```bash
uv run skippy
```

Or using the standard entrypoint:

```bash
python -m skippy_harness.agent
```

### CLI Options

- `--resume`: Automatically restore and continue your most recent session.
- `--debug`: Display raw model responses and token usage diagnostics.

```bash
uv run skippy --resume
```

### Interactive Commands

During a session, you can use built-in slash commands:

- `/compact` - Manually trigger context compaction.
- `/sessions` - List and resume previous chat sessions.
- `/rewind [n]` - Rewind conversation by `n` turns (default: 1).
- `Option+Enter` / `Alt+Enter` - Insert a newline in prompt.
- `Ctrl+D` - Exit the agent session.

---

## License

MIT
