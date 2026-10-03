"""Settings and multi-provider configuration for Skippy Harness."""

import os
import sys
from pathlib import Path

PROJECT_ENV = Path.cwd() / ".env"
USER_ENV = Path.home() / ".agents" / "env"

PROVIDERS = {
    "nvidia": {
        "name": "NVIDIA NIM",
        "description": "NVIDIA Inference Microservices (Llama 3.3, Nemotron, DeepSeek R1)",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "default_model": "meta/llama-3.2-90b-vision-instruct",
        "models": [
            "meta/llama-3.2-90b-vision-instruct",
            "meta/llama-3.2-11b-vision-instruct",
            "meta/codellama-70b",
            "ibm/granite-34b-code-instruct",
        ],
    },
    "openrouter": {
        "name": "OpenRouter",
        "description": "Unified API (Claude 3.7, GPT-4o, DeepSeek, Llama)",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-3.7-sonnet",
        "models": [
            "anthropic/claude-3.7-sonnet",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "deepseek/deepseek-chat",
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemini-2.5-flash",
        ],
    },
    "openai": {
        "name": "OpenAI",
        "description": "Official OpenAI API (GPT-4o, o3-mini, o1)",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "o3-mini", "o1"],
    },
    "groq": {
        "name": "Groq",
        "description": "Ultra-fast inference for open-source models",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "models": [
            "llama-3.3-70b-versatile",
            "qwen-2.5-coder-32b",
            "deepseek-r1-distill-llama-70b",
        ],
    },
    "deepseek": {
        "name": "DeepSeek",
        "description": "DeepSeek official API (V3, R1)",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "gemini": {
        "name": "Google Gemini",
        "description": "Google Gemini API via OpenAI-compatible endpoint",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
    },
    "mistral": {
        "name": "Mistral AI",
        "description": "Mistral AI & Codestral for coding",
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "default_model": "codestral-latest",
        "models": ["codestral-latest", "mistral-large-latest"],
    },
    "together": {
        "name": "Together AI",
        "description": "Fast cloud hosting for open-source models",
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
        ],
    },
    "ollama": {
        "name": "Ollama (Local)",
        "description": "Locally running models at localhost:11434",
        "base_url": "http://localhost:11434/v1",
        "env_key": "OLLAMA_API_KEY",
        "default_model": "qwen2.5-coder",
        "models": ["qwen2.5-coder", "llama3.2", "deepseek-coder-v2"],
    },
    "custom": {
        "name": "Custom Endpoint",
        "description": "Custom OpenAI-compatible base URL and API key",
        "base_url": "",
        "env_key": "API_KEY",
        "default_model": "gpt-4o",
        "models": [],
    },
}


def load_file(path: Path):
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                os.environ.setdefault(key, val)
    except Exception:
        pass


def load_env():
    load_file(PROJECT_ENV)
    load_file(USER_ENV)


load_env()

ACTIVE_PROVIDER = ""
BASE_URL = ""
API_KEY = ""
MODEL = ""
CONTEXT_WINDOW = int(os.environ.get("CONTEXT_WINDOW", 128_000))
COMPACT_AT = 0.85
COMPACT_TO = 0.35


def detect_provider():
    # 1. Explicit PROVIDER env var
    req = os.environ.get("PROVIDER", "").lower().strip()
    if req in ("nvidia", "nim"):
        return "nvidia"
    if req and req in PROVIDERS:
        return req

    # 2. Check for NVIDIA keys
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("NV_API_KEY") or os.environ.get("NGC_API_KEY"):
        return "nvidia"

    # 3. Check for configured provider keys
    for p_id, p_info in PROVIDERS.items():
        if p_id in ("custom", "ollama", "nvidia"):
            continue
        if os.environ.get(p_info["env_key"]):
            return p_id

    # 4. Check for custom endpoint
    if os.environ.get("BASE_URL") and os.environ.get("API_KEY"):
        return "custom"

    # 5. Check for Ollama
    if os.environ.get("OLLAMA_API_KEY") or os.environ.get("OLLAMA_HOST"):
        return "ollama"

    return None


def is_configured():
    provider = detect_provider()
    if not provider:
        return False
    if provider == "ollama":
        return True
    if provider == "nvidia":
        return bool(
            os.environ.get("NVIDIA_API_KEY")
            or os.environ.get("NV_API_KEY")
            or os.environ.get("NGC_API_KEY")
            or os.environ.get("API_KEY")
        )
    p_info = PROVIDERS[provider]
    key = os.environ.get(p_info["env_key"]) or os.environ.get("API_KEY")
    return bool(key)


def configure(provider=None, model=None, api_key=None, base_url=None):
    global ACTIVE_PROVIDER, BASE_URL, API_KEY, MODEL, CONTEXT_WINDOW

    if not provider:
        provider = detect_provider() or "nvidia"

    provider = provider.lower().strip()
    if provider in ("nvidia", "nim"):
        provider = "nvidia"
    elif provider not in PROVIDERS:
        provider = "custom"

    p_info = PROVIDERS[provider]
    ACTIVE_PROVIDER = provider

    if base_url:
        BASE_URL = base_url
    elif provider == "custom":
        BASE_URL = os.environ.get("BASE_URL", "")
    else:
        BASE_URL = p_info["base_url"]

    if api_key:
        API_KEY = api_key
    elif provider == "ollama":
        API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
    elif provider == "nvidia":
        API_KEY = (
            os.environ.get("NVIDIA_API_KEY")
            or os.environ.get("NV_API_KEY")
            or os.environ.get("NGC_API_KEY")
            or os.environ.get("API_KEY", "")
        )
    else:
        API_KEY = (
            os.environ.get(p_info["env_key"])
            or os.environ.get("API_KEY", "")
        )

    if model:
        MODEL = model
    elif provider and os.environ.get("PROVIDER") and os.environ.get("PROVIDER") != provider:
        MODEL = p_info["default_model"]
    elif os.environ.get("MODEL"):
        env_m = os.environ["MODEL"]
        if provider == "nvidia" and env_m.startswith(("anthropic/", "google/", "openai/")):
            MODEL = p_info["default_model"]
        elif provider == "openai" and env_m.startswith(("anthropic/", "google/", "meta/", "nvidia/")):
            MODEL = p_info["default_model"]
        else:
            MODEL = env_m
    else:
        MODEL = p_info["default_model"]

    CONTEXT_WINDOW = int(os.environ.get("CONTEXT_WINDOW", 128_000))


def save_env_var(key: str, value: str, env_path: Path = PROJECT_ENV):
    lines = []
    found = False
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{key}=") or stripped.startswith(f"#{key}="):
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interactive_setup(provider=None, model=None):
    """Guide the user interactively to select and configure a provider."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print()
    console.print(
        Panel(
            "[bold #7aa2f7]Skippy Harness — Provider Configuration[/]\n"
            "Select an LLM provider and enter your API key to get started.",
            border_style="#7aa2f7",
        )
    )

    p_keys = list(PROVIDERS.keys())
    table = Table(show_header=True, header_style="bold #7aa2f7")
    table.add_column("#", style="bold #e0af68", width=3)
    table.add_column("Provider", style="bold", width=16)
    table.add_column("Description", style="#9ece6a")
    table.add_column("Status")

    for i, p_id in enumerate(p_keys, 1):
        p_data = PROVIDERS[p_id]
        if p_id == "nvidia":
            has_key = bool(
                os.environ.get("NVIDIA_API_KEY")
                or os.environ.get("NV_API_KEY")
                or os.environ.get("NGC_API_KEY")
            )
        elif p_id == "ollama":
            has_key = True
        else:
            has_key = bool(os.environ.get(p_data["env_key"]))
        status = "[#9ece6a]Ready[/]" if has_key else "[#565f89]Key not set[/]"
        table.add_row(str(i), p_data["name"], p_data["description"], status)

    console.print(table)
    console.print()

    selected_provider = provider
    if selected_provider and selected_provider.lower() in ("nvidia", "nim"):
        selected_provider = "nvidia"

    if not selected_provider:
        while True:
            try:
                choice = input(f"Select provider [1-{len(p_keys)}] (default 1: NVIDIA NIM): ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[red]Setup cancelled.[/]")
                sys.exit(0)
            if not choice:
                selected_provider = "nvidia"
                break
            if choice.lower() in ("nvidia", "nim"):
                selected_provider = "nvidia"
                break
            if choice.lower() in p_keys:
                selected_provider = choice.lower()
                break
            if choice.isdigit() and 1 <= int(choice) <= len(p_keys):
                selected_provider = p_keys[int(choice) - 1]
                break
            console.print("[red]Invalid selection. Please choose a valid number or provider name.[/]")

    p_info = PROVIDERS[selected_provider]
    console.print(f"\nConfiguring [bold #7aa2f7]{p_info['name']}[/]...")

    # Prompt API key if needed
    api_key = ""
    base_url = p_info["base_url"]

    if selected_provider == "custom":
        try:
            base_url = input("Enter OpenAI-compatible BASE_URL: ").strip()
            api_key = input("Enter API key: ").strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
    elif selected_provider == "ollama":
        try:
            custom_url = input(f"Enter Ollama URL [default: {p_info['base_url']}]: ").strip()
            if custom_url:
                base_url = custom_url
            api_key = "ollama"
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
    else:
        existing_key = (
            os.environ.get("NVIDIA_API_KEY")
            or os.environ.get("NV_API_KEY")
            if selected_provider == "nvidia"
            else os.environ.get(p_info["env_key"], "")
        )
        if existing_key:
            console.print(f"Found existing key in environment: [dim]{existing_key[:6]}...{existing_key[-4:]}[/]")
            try:
                use_existing = input("Keep existing key? [Y/n]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                sys.exit(0)
            if use_existing in ("", "y", "yes"):
                api_key = existing_key

        if not api_key:
            try:
                api_key = input(f"Enter your {p_info['name']} API key ({p_info['env_key']}): ").strip()
            except (KeyboardInterrupt, EOFError):
                sys.exit(0)
            while not api_key:
                console.print("[red]API key cannot be empty.[/]")
                try:
                    api_key = input(f"Enter your {p_info['name']} API key: ").strip()
                except (KeyboardInterrupt, EOFError):
                    sys.exit(0)

    # Prompt Model
    chosen_model = model
    if not chosen_model:
        default_m = p_info["default_model"]
        if p_info["models"]:
            console.print("\nRecommended models:")
            for m in p_info["models"]:
                console.print(f"  · {m}")
        try:
            m_input = input(f"\nEnter model [default: {default_m}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
        chosen_model = m_input if m_input else default_m

    # Ask to save to .env
    try:
        save = input("\nSave this configuration to .env? [Y/n]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)

    if save in ("", "y", "yes"):
        save_env_var("PROVIDER", selected_provider)
        if selected_provider == "custom":
            save_env_var("BASE_URL", base_url)
            save_env_var("API_KEY", api_key)
        else:
            save_env_var(p_info["env_key"], api_key)
        save_env_var("MODEL", chosen_model)
        console.print(f"[bold #9ece6a]Saved configuration to {PROJECT_ENV}[/]")

    # Set in os.environ for immediate use
    os.environ["PROVIDER"] = selected_provider
    if selected_provider == "custom":
        os.environ["BASE_URL"] = base_url
        os.environ["API_KEY"] = api_key
    else:
        os.environ[p_info["env_key"]] = api_key
    os.environ["MODEL"] = chosen_model

    configure(
        provider=selected_provider,
        model=chosen_model,
        api_key=api_key,
        base_url=base_url,
    )
    console.print(f"[bold #9ece6a][OK] Provider configured:[/] {p_info['name']} ({chosen_model})\n")


# Initial configuration if already available in environment
if is_configured():
    configure()
