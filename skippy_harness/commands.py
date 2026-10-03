"""Slash commands. Anything typed starting with / lands here."""

import os
from . import compact as compaction
from . import config
from . import llm
from . import sandbox
from . import session
from .ui import ui

COMMANDS = {
    "/provider": "view or switch active LLM provider (OpenRouter, OpenAI, Groq, etc.)",
    "/model": "view or switch the active model",
    "/rewind": "jump back to an earlier point in this chat",
    "/sessions": "open a past chat",
    "/compact": "summarise the history so far and free up the context window",
    "/help": "list all available commands",
}


def preview(message):
    if message.get("tool_calls"):
        return "-> " + message["tool_calls"][0]["function"]["name"]
    return " ".join(str(message.get("content") or "").split())[:70]


def redraw(messages, label):
    """The screen no longer matches the history, so wipe it and draw again."""
    ui.clear()
    ui.banner(sandbox.name(), config.ACTIVE_PROVIDER, config.MODEL)
    ui.resumed(messages, label)
    ui.replay(messages)
    return messages


def rewind(messages):
    rows = [f"{m['role']:<9} {preview(m)}" for m in messages]
    choice = ui.pick("rewind to", rows)
    if choice is None:
        return messages
    session.rewind_to(choice + 1)
    return redraw(messages[: choice + 1], "rewound")


def sessions(messages):
    saved = session.all_sessions()
    if not saved:
        ui.note("no saved chats yet")
        return messages
    rows = [f"{s['id']}  {s['title']}" for s in saved]
    choice = ui.pick("open chat", rows)
    if choice is None:
        return messages

    return redraw(session.open_session(saved[choice]["id"]), "opened")


def compact(messages):
    before = len(messages)
    try:
        with ui.working("compacting"):
            compacted = compaction.compact(messages)
    except Exception as failure:  # noqa: BLE001
        # Compaction is one more API call, and it fires when the window is
        # nearly full - the worst moment to lose the session over a rate limit.
        ui.note(f"compaction failed ({type(failure).__name__}); transcript kept as is")
        return messages
    if len(compacted) == before:
        ui.note("nothing old enough to compact yet")
        return messages
    session.compacted(compacted)
    ui.compacted(before, compacted)
    return compacted


def switch_provider(messages):
    p_keys = list(config.PROVIDERS.keys())
    rows = []
    for p_id in p_keys:
        p_info = config.PROVIDERS[p_id]
        has_key = bool(os.environ.get(p_info["env_key"]) or (p_id == "ollama"))
        active = "[ACTIVE] " if p_id == config.ACTIVE_PROVIDER else ""
        key_status = "(configured)" if has_key else "(key missing)"
        rows.append(f"{active}{p_info['name']} {key_status} - {p_info['description']}")

    rows.append(">> Configure new API key or custom endpoint")

    choice = ui.pick(f"active: {config.ACTIVE_PROVIDER} ({config.MODEL}) | Select provider:", rows)
    if choice is None:
        return messages

    if choice == len(p_keys):
        # Configure new provider
        config.interactive_setup()
        llm.reset_client()
        ui.note(f"Switched to {config.ACTIVE_PROVIDER} ({config.MODEL})")
        return redraw(messages, f"switched to {config.ACTIVE_PROVIDER}")

    selected_p = p_keys[choice]
    p_info = config.PROVIDERS[selected_p]
    has_key = bool(os.environ.get(p_info["env_key"]) or (selected_p == "ollama"))

    if not has_key:
        ui.note(f"{p_info['name']} has no API key set. Starting configuration...")
        config.interactive_setup(provider=selected_p)
    else:
        config.configure(provider=selected_p)

    llm.reset_client()
    ui.note(f"Active provider switched to {p_info['name']} with model: {config.MODEL}")
    return redraw(messages, f"switched to {config.ACTIVE_PROVIDER}")


def switch_model(command_text, messages):
    parts = command_text.split(maxsplit=1)
    if len(parts) > 1 and parts[1].strip():
        new_model = parts[1].strip()
        config.configure(provider=config.ACTIVE_PROVIDER, model=new_model)
        llm.reset_client()
        ui.note(f"Model changed to: {new_model}")
        return redraw(messages, f"model: {new_model}")

    p_info = config.PROVIDERS.get(config.ACTIVE_PROVIDER, config.PROVIDERS["openrouter"])
    options = list(p_info["models"]) if p_info.get("models") else []
    if config.MODEL not in options:
        options.insert(0, config.MODEL)

    rows = [f"{'[ACTIVE] ' if m == config.MODEL else ''}{m}" for m in options]
    rows.append(">> Enter custom model name")

    choice = ui.pick(f"Current model: {config.MODEL} | Select new model:", rows)
    if choice is None:
        return messages

    if choice == len(options):
        custom_m = input("Enter model name: ").strip()
        if custom_m:
            config.configure(provider=config.ACTIVE_PROVIDER, model=custom_m)
            llm.reset_client()
            ui.note(f"Model changed to: {custom_m}")
            return redraw(messages, f"model: {custom_m}")
        return messages

    chosen_m = options[choice]
    config.configure(provider=config.ACTIVE_PROVIDER, model=chosen_m)
    llm.reset_client()
    ui.note(f"Model changed to: {chosen_m}")
    return redraw(messages, f"model: {chosen_m}")


def handle(command, messages):
    cmd_lower = command.strip().lower()
    if cmd_lower == "/compact":
        return compact(messages)
    if cmd_lower == "/rewind":
        return rewind(messages)
    if cmd_lower == "/sessions":
        return sessions(messages)
    if cmd_lower.startswith("/provider"):
        return switch_provider(messages)
    if cmd_lower.startswith("/model"):
        return switch_model(command, messages)
    if cmd_lower in ("/help", "/?"):
        ui.note("\n".join(f"{name:<12} -  {help_txt}" for name, help_txt in COMMANDS.items()))
        return messages

    ui.note("\n".join(f"{name:<12} -  {help_txt}" for name, help_txt in COMMANDS.items()))
    return messages
