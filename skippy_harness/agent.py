import argparse

from . import commands
from . import compact
from . import config
from . import history
from . import sandbox
from . import session
from . import models
from .history import estimate
from .context import reminder
from .llm import SYSTEM_PROMPT, call_llm
from .todos import active_form
from .tools import execute, TOOL_SCHEMAS
from .ui import ui


def main():
    parser = argparse.ArgumentParser(description="Skippy Harness: A minimal coding agent harness")
    parser.add_argument("--resume", action="store_true", help="continue the last session")
    parser.add_argument("--debug", action="store_true", help="show the raw model response")
    parser.add_argument(
        "--provider",
        type=str,
        help="LLM provider (openrouter, openai, groq, deepseek, gemini, mistral, together, ollama, custom)",
    )
    parser.add_argument("--model", type=str, help="model identifier to use")
    parser.add_argument("--setup", action="store_true", help="run provider setup wizard")
    parser.add_argument("--check", action="store_true", help="check provider connection and model without starting chat")
    cli = parser.parse_args()

    # Apply CLI selection before checking credentials for that provider.
    try:
        config.configure(provider=cli.provider, model=cli.model)
    except ValueError as err:
        parser.error(str(err))
    if cli.check:
        try:
            if not config.is_configured():
                raise ValueError(f"No credentials configured for {config.ACTIVE_PROVIDER}.")
            available = models.available_models()
            validated = models.validate_model(config.MODEL)
            ui.note(f"Connected to {config.ACTIVE_PROVIDER}: {len(available)} available models. Selected: {validated} (tools supported).")
            if config.ACTIVE_PROVIDER == "openrouter":
                key = models.request_json("GET", config.BASE_URL.rstrip("/") + "/key",
                                          headers={"Authorization": f"Bearer {config.API_KEY}"}).get("data", {})
                ui.note("OpenRouter API key accepted.")
                if key.get("limit_remaining") is not None and key["limit_remaining"] <= 0:
                    raise ValueError("OpenRouter reports no remaining allowance for this key. Adjust its limit in OpenRouter or use Ollama.")
        except Exception as err:
            ui.note(str(err))
            raise SystemExit(1)
        return
    if cli.setup or not config.is_configured():
        config.interactive_setup(provider=cli.provider, model=cli.model)

    try:
        config.MODEL = models.validate_model(config.MODEL, refresh=True)
        models.apply_context_limit()
    except Exception as err:
        ui.note(str(err))
        commands.switch_model("/model", [])

    ui.banner(sandbox.name(), config.ACTIVE_PROVIDER, config.MODEL)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if cli.resume:
        saved = session.all_sessions()
        if saved:
            messages = session.open_session(saved[0]["id"])
            history.strip(messages)
            ui.resumed(messages)
            ui.replay(messages)

    while True:
        user_input = ui.ask()
        if user_input is None:
            break
        if not user_input:
            continue

        if user_input.startswith("/"):
            messages = commands.handle(user_input, messages)
            session.save(messages)
            continue

        messages.append({"role": "user", "content": user_input})
        usage = None

        for _ in range(config.positive_int("MAX_AGENT_STEPS", 50)):
            injection = reminder()
            ui.injection(injection["content"])

            if history.fit(messages):
                ui.note("dropped old tool output to make this request fit")

            budget = config.CONTEXT_WINDOW * config.COMPACT_AT
            if estimate(messages + [injection]) + estimate(TOOL_SCHEMAS) > budget:
                messages = commands.compact(messages)
                if estimate(messages + [injection]) + estimate(TOOL_SCHEMAS) > budget:
                    ui.note("This request is too large for the selected context window. Shorten the prompt, rewind, or use a model with a larger window.")
                    break

            try:
                with ui.working(active_form()):
                    message, usage = call_llm(messages + [injection])
            except Exception as err:
                detail = str(err).replace(config.API_KEY, "[redacted]") if config.API_KEY else str(err)
                ui.note(f"API Error ({config.ACTIVE_PROVIDER} / {config.MODEL}):\n  {detail}\nTip: Type /model to change model, or /provider ollama or /provider openrouter to switch.")
                break

            if not message.content and not message.tool_calls:
                ui.note("The model returned an empty answer. Retry or select another model with /model.")
                break
            messages.append(message.model_dump(exclude_none=True))
            session.save(messages)
            ui.usage(usage)

            if cli.debug:
                ui.debug(message.model_dump(exclude_none=True))

            if message.content:
                ui.agent(message.content)

            if not message.tool_calls:
                break

            for tool_call in message.tool_calls:
                args, result = execute(tool_call)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                session.save(messages)
                ui.tool(tool_call.function.name, args, result)
        else:
            ui.note("Stopped at the agent step limit. Send another message to continue, or adjust MAX_AGENT_STEPS.")

        history.sweep()   # the turn is over: bin its temp files
        history.strip(messages)  # ...and shrink the tool output it produced

        if usage and compact.needed(usage):
            messages = commands.compact(messages)

    ui.summary()


if __name__ == "__main__":
    main()
