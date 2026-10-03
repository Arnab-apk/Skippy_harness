"""Slash commands. Anything typed starting with / lands here."""

from . import compact as compaction
from . import config
from . import llm
from . import sandbox
from . import session
from . import todos
from . import models
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


def rewind(messages, count=None):
    starts = [i for i, m in enumerate(messages) if m["role"] == "user" and "<summary>" not in (m.get("content") or "")]
    if count is not None:
        if count < 1 or count > len(starts):
            ui.note("Choose a positive number no greater than the number of user turns.")
            return messages
        cut = starts[-count]
    else:
        choice = ui.pick("rewind before this user turn (file edits remain on disk)", [preview(messages[i]) for i in starts])
        if choice is None:
            return messages
        cut = starts[choice]
    session.rewind_to(cut)
    todos.restore(messages[:cut], session.BASE_TODOS)
    return redraw(messages[:cut], "rewound")


def sessions(messages):
    saved = session.all_sessions()
    if not saved:
        ui.note("no saved chats yet")
        return messages
    rows = [f"{s['id']}  {s['title']}" for s in saved]
    choice = ui.pick("open chat", rows)
    if choice is None:
        return messages

    restored = session.open_session(saved[choice]["id"])
    return redraw(restored, "opened")


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
    if compacted == messages:
        ui.note("nothing old enough to compact yet")
        return messages
    session.compacted(compacted)
    ui.compacted(before, compacted)
    return compacted


def switch_provider(messages, provider=None):
    p_keys = ["ollama", "openrouter"] + [p for p in config.PROVIDERS if p not in ("ollama", "openrouter")]
    rows = []
    for p_id in p_keys:
        p_info = config.PROVIDERS[p_id]
        has_key = config.is_configured(p_id)
        active = "[ACTIVE] " if p_id == config.ACTIVE_PROVIDER else ""
        key_status = "(configured)" if has_key else "(key missing)"
        rows.append(f"{active}{p_info['name']} {key_status} - {p_info['description']}")

    rows.append(">> Configure new API key or custom endpoint")

    if provider:
        provider = provider.lower().strip()
        if provider not in p_keys:
            ui.note(f"Unknown provider: {provider}. Use /provider ollama or /provider openrouter.")
            return messages
        choice = p_keys.index(provider)
    else:
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
    has_key = config.is_configured(selected_p)

    if not has_key:
        ui.note(f"{p_info['name']} has no API key set. Starting configuration...")
        config.interactive_setup(provider=selected_p)
    else:
        config.configure(provider=selected_p)

    try:
        config.MODEL = models.validate_model(config.MODEL, refresh=True)
        models.apply_context_limit()
    except Exception as err:
        ui.note(str(err))
        switch_model("/model", messages)
    save_selection()
    llm.reset_client()
    ui.note(f"Active provider switched to {p_info['name']} with model: {config.MODEL}")
    return redraw(messages, f"switched to {config.ACTIVE_PROVIDER}")


def save_selection():
    try:
        config.persist_selection()
    except OSError as err:
        ui.note(f"Selection works for this session, but could not be saved: {err}")


def select_model(name, messages):
    try:
        name = models.validate_model(name)
        config.configure(provider=config.ACTIVE_PROVIDER, model=name)
        models.apply_context_limit()
    except Exception as err:
        ui.note(str(err))
        return messages
    save_selection()
    ui.note(f"Model changed to: {config.MODEL}")
    return redraw(messages, f"model: {config.MODEL}")


def switch_model(command_text, messages):
    parts = command_text.split(maxsplit=1)
    if len(parts) > 1 and parts[1].strip():
        return select_model(parts[1].strip(), messages)

    try:
        options = list(models.available_models())
    except Exception as err:
        ui.note(str(err))
        return messages

    rows = [f"{'[ACTIVE] ' if m == config.MODEL else ''}{m}" for m in options]
    rows.append(">> Enter custom model name")

    choice = ui.pick(f"Current model: {config.MODEL} | Select new model:", rows)
    if choice is None:
        return messages

    if choice == len(options):
        from . import prompt
        try:
            custom_m = prompt.read("Enter model name: ").strip()
        except (EOFError, KeyboardInterrupt):
            return messages
        if custom_m:
            return select_model(custom_m, messages)
        return messages

    chosen_m = options[choice]
    return select_model(chosen_m, messages)


def handle(command, messages):
    parts = command.strip().split(maxsplit=1)
    if not parts:
        return messages
    cmd_lower = parts[0].lower()
    if cmd_lower == "/compact":
        return compact(messages)
    if cmd_lower == "/rewind":
        try:
            count = int(parts[1]) if len(parts) > 1 else None
        except ValueError:
            ui.note("Usage: /rewind [positive number of turns]")
            return messages
        return rewind(messages, count)
    if cmd_lower == "/sessions":
        return sessions(messages)
    if cmd_lower == "/provider":
        return switch_provider(messages, parts[1] if len(parts) > 1 else None)
    if cmd_lower == "/model":
        return switch_model(command, messages)
    if cmd_lower in ("/help", "/?"):
        ui.note("\n".join(f"{name:<12} -  {help_txt}" for name, help_txt in COMMANDS.items()))
        return messages

    ui.note("\n".join(f"{name:<12} -  {help_txt}" for name, help_txt in COMMANDS.items()))
    return messages
