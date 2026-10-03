import argparse

from . import commands
from . import compact
from . import config
from . import history
from . import sandbox
from . import session
from .context import reminder
from .llm import SYSTEM_PROMPT, call_llm
from .todos import active_form
from .tools import execute
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
    cli = parser.parse_args()

    # Ensure provider is configured or launch interactive setup
    if cli.setup or not config.is_configured():
        config.interactive_setup(provider=cli.provider, model=cli.model)
    elif cli.provider or cli.model:
        config.configure(provider=cli.provider, model=cli.model)

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
        if not user_input:
            break

        if user_input.startswith("/"):
            messages = commands.handle(user_input, messages)
            session.save(messages)
            continue

        messages.append({"role": "user", "content": user_input})

        while True:
            injection = reminder()
            ui.injection(injection["content"])

            if history.fit(messages):
                ui.note("dropped old tool output to make this request fit")

            with ui.working(active_form()):
                message, usage = call_llm(messages + [injection])

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
                ui.tool(tool_call.function.name, args, result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                session.save(messages)

        history.sweep()   # the turn is over: bin its temp files
        history.strip(messages)  # ...and shrink the tool output it produced

        if compact.needed(usage):
            messages = commands.compact(messages)

    ui.summary()


if __name__ == "__main__":
    main()
