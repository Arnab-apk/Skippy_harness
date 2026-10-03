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

### Configuration & Multi-Provider Support

Skippy Harness supports multiple LLM providers out of the box with auto-detection:

- **OpenRouter** (Claude 3.7/3.5, GPT-4o, DeepSeek, etc.)
- **OpenAI** (GPT-4o, o3-mini, o1)
- **Groq** (Ultra-fast Llama 3.3, Qwen 2.5 Coder)
- **DeepSeek** (DeepSeek-V3, DeepSeek-R1)
- **Google Gemini** (Gemini 2.5 Flash, Gemini 2.5 Pro)
- **Mistral AI** (Codestral, Mistral Large)
- **Together AI** (Open-source model hosting)
- **Ollama** (Local models running on `localhost:11434`)
- **Custom** (Any OpenAI-compatible endpoint)

#### Quick Setup Wizard

If no configuration exists, simply run:

```bash
uv run skippy
```

Skippy will launch an interactive setup wizard to select your provider and save your key to `.env`. You can re-run this setup at any time with:

```bash
uv run skippy --setup
```

#### Manual Configuration (`.env`)

Copy `.env.example` to `.env` and fill in whichever keys you have:

```bash
cp .env.example .env
```

```ini
# Active provider to use (optional, auto-detected from set keys)
PROVIDER=openrouter

# Store as many keys as you like:
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=...
```

---

## Usage

Start Skippy Harness:

```bash
uv run skippy
```

### CLI Options

- `--provider <name>`: Switch provider directly on launch (e.g. `uv run skippy --provider groq`)
- `--model <name>`: Override model name (e.g. `uv run skippy --model gpt-4o-mini`)
- `--setup`: Launch the interactive provider configuration wizard
- `--resume`: Automatically restore and continue your most recent session
- `--debug`: Display raw model responses and token usage diagnostics

Examples:
```bash
uv run skippy --provider openrouter --model anthropic/claude-3.7-sonnet
uv run skippy --provider groq
uv run skippy --resume
```

### Interactive In-Chat Commands

During a session, you can use built-in slash commands:

- `/provider` - View active provider or switch between providers on the fly
- `/model [name]` - View current model, choose from recommendations, or set a custom model
- `/compact` - Manually trigger context compaction
- `/sessions` - List and resume previous chat sessions
- `/rewind [n]` - Rewind conversation by `n` turns (default: 1)
- `/help` - Show all available commands
- `Alt+Enter` / `Opt+Enter` - Insert a newline in prompt
- `Ctrl+D` - Exit the agent session

---

## License

MIT
