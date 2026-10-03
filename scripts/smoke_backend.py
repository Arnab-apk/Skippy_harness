"""Exercise an actual model -> tools -> model loop in a temporary workspace.

Run from the repository root with python -m scripts.smoke_backend.
This makes live inference requests to the selected provider.
"""

import argparse
import tempfile
from pathlib import Path

from skippy_harness import config, models, llm, tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["ollama", "openrouter"])
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    config.configure(provider=args.provider, model=args.model)
    config.MODEL = models.validate_model(config.MODEL, refresh=True)
    models.apply_context_limit()
    schemas = [s for s in tools.TOOL_SCHEMAS if s["function"]["name"] in {"read_file", "write_file", "str_replace"}]
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        target = Path(directory) / "smoke.txt"
        messages = [
            {"role": "system", "content": "You are a coding agent. Use the provided tools to perform the requested file operations. Do not claim success without doing them. Return a brief answer after verifying the file."},
            {"role": "user", "content": f"Create {target.as_posix()} containing exactly 'hello world', then use str_replace to replace 'world' with 'harness', then read_file to verify it contains 'hello harness'. Execute all three steps using tools, then report completion."},
        ]
        used = []
        for step in range(8):
            message, usage = llm.call_llm(messages, tools=schemas)
            messages.append(message.model_dump(exclude_none=True))
            print(f"Step {step + 1}: {usage['prompt_tokens']} input, {usage['completion_tokens']} output; tools={[c.function.name for c in message.tool_calls or []]}", flush=True)
            if not message.tool_calls:
                break
            for tool_call in message.tool_calls:
                if tool_call.function.name not in {"read_file", "write_file", "str_replace"}:
                    raise RuntimeError("Unexpected tool requested")
                import json
                arguments = json.loads(tool_call.function.arguments)
                if Path(arguments.get("path", "")).resolve() != target.resolve():
                    raise RuntimeError("The smoke model requested an unexpected path")
                _, result = tools.execute(tool_call)
                print(f"  {tool_call.function.name}: {result}", flush=True)
                used.append(tool_call.function.name)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        if not target.exists() or target.read_text(encoding="utf-8") != "hello harness":
            raise RuntimeError("Live coding smoke test failed: expected file contents not produced")
        if not {"write_file", "str_replace", "read_file"}.issubset(used):
            raise RuntimeError("Live coding smoke test failed: required tool workflow was not completed")
        print(f"PASS: {args.provider} / {args.model} created, edited, and verified the file through the harness.", flush=True)


if __name__ == "__main__":
    main()
