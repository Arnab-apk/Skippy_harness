# Skippy Harness

A lightweight, modular, and extensible coding agent harness in Python built for autonomous terminal-driven pair programming.

Ollama and OpenRouter are the primary supported backends. Model menus use live provider catalogues: Ollama lists installed models, and OpenRouter lists models with text output and tool support. Provider switching remembers a separate model for each backend and saves menu selections to `.env`.

### Start with Ollama or OpenRouter

```powershell
uv run skippy --provider ollama
```

Start Ollama first and install a model with tool support. If the saved model is missing, Skippy opens the model picker. The picker uses **1-based numbering**, supports search text, and uses `n` / `p` to change pages. Enter cancels a menu; an empty chat prompt keeps the session open.

For OpenRouter, set `OPENROUTER_API_KEY` in `.env`, then run:

```powershell
uv run skippy --provider openrouter
```

Inside chat:

```text
/provider ollama
/provider openrouter
/model
/model qwen3.5:2b
```

Model IDs are provider-specific. An Ollama model must be installed and support tools; an OpenRouter model must appear in its current tool-capable catalogue. A valid OpenRouter key can still fail inference if its monthly spending limit or credits are exhausted; Skippy explains that error and lets you switch to Ollama.

Check connectivity, installed/available model, and tool support without starting chat:

```powershell
uv run skippy --check
uv run skippy --provider openrouter --model google/gemini-2.5-flash --check
```

`--check` checks metadata and OpenRouter authentication. It does not make a paid inference request or guarantee remaining inference allowance.

## Features

- **Interactive Terminal Chat**: Powered by `prompt_toolkit` and `rich`, featuring multi-line input, history recall, and clean terminal rendering.
- **Core Tooling**: UTF-8 file inspection, atomic file creation/replacement, and surgical text replacements (`str_replace`). Shell output includes the command's exit code. On Windows, the `bash` tool runs PowerShell; on macOS/Linux it runs the POSIX shell.
- **Configurable LLM Backend**: Connects to any OpenAI-compatible API endpoint (OpenAI, DeepSeek, Ollama, OpenRouter, etc.).
- **Permissions & OS Sandboxing**: File writes outside the project and shell commands with effects require approval. Seatbelt on macOS and Bubblewrap on Linux restrict shell writes/network where available. Windows and Linux without Bubblewrap have no OS sandbox; the permission checker is an approval mechanism, not a security boundary.
- **Subagents**: Exploration subagents have separate message histories. The executor blocks write tools, recursive delegation, plan changes, and shell commands requiring approval.
- **Skills System**: Dynamically loads custom capabilities and operational guides from `.agents/skills`.
- **Todo & Task Planning**: Automatic multi-step task planning with `<todos>` injection into prompts for reliable execution.
- **Session Management**: Conversation logs are stored in `.skippy/sessions/`, ignored by Git. `/sessions` and `--resume` restore chats, plans, and interrupted tool batches. Older logs remain readable and are imported when opened. `/rewind` rolls back conversation turns; it does not undo file edits.
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

- **NVIDIA NIM** (Llama 3.3, Nemotron, DeepSeek R1, Qwen 2.5 Coder via `integrate.api.nvidia.com`)
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
- `--check`: Check provider connectivity, authentication, and selected model metadata without inference

Examples:
```bash
uv run skippy --provider openrouter --model anthropic/claude-3.7-sonnet
uv run skippy --provider groq
uv run skippy --resume
```

### Interactive In-Chat Commands

During a session, you can use built-in slash commands:

- `/provider` - View active provider or switch between providers on the fly
- `/provider ollama` / `/provider openrouter` - Switch directly to that provider
- `/model [name]` - View current model, choose from recommendations, or set a custom model
- `/compact` - Manually trigger context compaction
- `/tools` or `Ctrl+O` - Expand/collapse tool details; Ctrl+O preserves your unfinished prompt
- `/sessions` - List and resume previous chat sessions
- `/rewind` - Select a user turn and rewind the conversation to before it
- `/rewind n` - Remove the last `n` user turns from the conversation
- `/help` - Show all available commands
- `Alt+Enter` / `Opt+Enter` - Insert a newline in prompt
- `Ctrl+D` - Exit the agent session

Tool calls use compact rows with commands or relative file paths, success/error status, and short output previews. Edits show a small colored diff; writes show the path and line count. Expand with `/tools` to inspect recent output. The preview does not change what is sent to the model or saved in session logs; existing output and context limits still apply. The UI keeps the last 200 tool results for expansion even after in-memory history is shortened. Older or resumed results may already be trimmed by context management.

Startup shows a colored ASCII Skippy wordmark with a small-terminal fallback. Repeated environment reminders, per-request token diagnostics, and raw responses are displayed with `--debug`.

---

## License

MIT

## Validation and runtime settings

Run the regression suite without extra test dependencies:

```powershell
uv run python -m unittest discover -s tests -v
```

Optional live coding smoke test (OpenRouter inference may incur charges):

```powershell
uv run python -m scripts.smoke_backend --provider ollama --model qwen3.5:2b
```

This asks the model to create, edit, and read a temporary file through the harness, then checks the actual contents.

Optional `.env` settings:

| Setting | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Saved Ollama endpoint; `OLLAMA_HOST` is also accepted |
| `OLLAMA_MODEL` / `OPENROUTER_MODEL` | Provider default or picker selection | Separate saved model IDs |
| `CONTEXT_WINDOW` | 8192 for Ollama; 128000 otherwise | Request budget; OpenRouter metadata can reduce it |
| `LLM_TIMEOUT` | 300 seconds for Ollama; 180 otherwise | Inference timeout |
| `MAX_OUTPUT_TOKENS` | 4096 | Completion limit for Ollama and OpenRouter |
| `MAX_AGENT_STEPS` | 50 | Maximum model/tool iterations per user turn |

Ollama uses its [native chat API](https://docs.ollama.com/api/chat), passing the configured context size as `options.num_ctx`. OpenRouter discovery uses its [model catalogue](https://openrouter.ai/docs/api/api-reference/models/get-models) and filters for tool support. Backend discovery needs the model server/provider connection; file and shell tools remain subject to the harness permission layer.
