import json
import os
import uuid


from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from . import config
from .skills import skills_prompt
from .tools import TOOLS, TOOL_SCHEMAS

_client = None


def get_client():
    global _client
    if _client is None:
        if not config.API_KEY:
            raise ValueError(f"No API key configured for {config.ACTIVE_PROVIDER}. Use /provider to configure it.")
        _client = OpenAI(
            base_url=config.BASE_URL,
            api_key=config.API_KEY or "ollama",
            timeout=config.positive_int("LLM_TIMEOUT", 180),
            max_retries=1,
        )
    return _client


def reset_client():
    global _client
    _client = None


class ClientProxy:
    """Proxy object so existing imports of `client` work seamlessly across provider switches."""
    def __getattr__(self, name):
        return getattr(get_client(), name)


client = ClientProxy()

SYSTEM_PROMPT = f"""
You are a coding agent. Complete the user's request and verify your changes.
Use the bash tool to inspect files.
Use write_file to create files and str_replace to edit them.
Answer back to the user once the requested work is done. Report checks and any remaining limitations.
Shell commands run in {"PowerShell on Windows. Use Get-ChildItem, Get-Content, and native PowerShell syntax; do not assume bash, grep, or head are installed." if os.name == "nt" else "the POSIX shell."}

For any task that takes more than one step, call write_todos first and plan it
out. Send the whole list every time you call it - it replaces the old one.
Keep exactly one task in_progress, mark it done the moment it is finished, and
move the next one to in_progress in the same call. Do not batch up completions
at the end. Skip the tool entirely for single-step tasks; it is noise there.

The current list is injected back to you every turn inside <todos> tags, so
that block - not the transcript - is the truth about where you are.

When you need to understand how something works - where a feature lives, how
data flows, what calls what - send a task subagent instead of grepping your
way there yourself. It explores in its own context window and hands you back
just the findings, so the search does not fill yours. It cannot see this
conversation, so write the question so it stands alone. Do all editing
yourself; the subagent only reads.

Long tool output is cut short, and the whole thing is written to a temp file
whose path is given at the cut. Page through it with head, tail, sed -n or
grep rather than asking for it again. That file only exists for the current
turn, so read it now or re-run the command later.

Your current working directory is: {os.getcwd()}

You have skills available. Each one is a set of instructions for a task.
If a skill matches what the user wants, call read_skill first and follow it.

{skills_prompt()}
"""


def ollama_messages(messages):
    converted, names = [], {}
    for message in messages:
        item = {"role": message["role"], "content": message.get("content") or ""}
        if message.get("tool_calls"):
            item["tool_calls"] = []
            for call in message["tool_calls"]:
                function = call["function"]
                names[call["id"]] = function["name"]
                try:
                    arguments = json.loads(function["arguments"])
                    if not isinstance(arguments, dict):
                        arguments = {}
                except (ValueError, TypeError):
                    arguments = {}
                item["tool_calls"].append({"function": {"name": function["name"], "arguments": arguments}})
        if message["role"] == "tool":
            item["tool_name"] = names.get(message.get("tool_call_id"), "")
        converted.append(item)
    return converted


def call_ollama(messages, tools):
    from .models import ollama_root, ollama_headers, request_json
    payload = {"model": config.MODEL, "messages": ollama_messages(messages),
               "stream": False, "think": False,
               "options": {"num_ctx": config.CONTEXT_WINDOW, "num_predict": config.positive_int("MAX_OUTPUT_TOKENS", 4096)}}
    if tools:
        payload["tools"] = tools
    data = request_json("POST", ollama_root() + "/api/chat", json_body=payload,
                        headers=ollama_headers(),
                        timeout=config.positive_int("LLM_TIMEOUT", 300))
    if data.get("error"):
        raise ValueError(f"Ollama: {data['error']}")
    if not data.get("message"):
        raise ValueError("Ollama returned no message.")
    message = data["message"]
    calls = []
    for call in message.get("tool_calls") or []:
        function = call["function"]
        arguments = function.get("arguments", {})
        calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function",
                      "function": {"name": function["name"], "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}})
    return ChatCompletionMessage(role="assistant", content=message.get("content") or "", tool_calls=calls or None), {
        "prompt_tokens": data.get("prompt_eval_count", 0), "completion_tokens": data.get("eval_count", 0),
        "reasoning_tokens": None, "cached_tokens": data.get("prompt_eval_cached_count", 0)}


def call_llm(messages, tools=None):
    selected_tools = TOOL_SCHEMAS if tools is None else tools
    if config.ACTIVE_PROVIDER == "ollama":
        return call_ollama(messages, selected_tools)
    active_client = get_client()
    kwargs = {"model": config.MODEL, "messages": messages}
    if config.ACTIVE_PROVIDER == "openrouter":
        kwargs["max_tokens"] = config.positive_int("MAX_OUTPUT_TOKENS", 4096)
    if selected_tools:
        kwargs["tools"] = selected_tools
        if config.ACTIVE_PROVIDER == "openrouter":
            kwargs["extra_body"] = {"provider": {"require_parameters": True}}
    try:
        response = active_client.chat.completions.create(**kwargs)
    except Exception as err:
        text = str(err).lower()
        if config.ACTIVE_PROVIDER == "openrouter":
            if "key limit exceeded" in text or "monthly limit" in text:
                raise ValueError("OpenRouter's monthly limit for this API key has been reached. Increase/reset the key limit in OpenRouter, or use /provider ollama to continue locally.") from err
            if getattr(err, "status_code", None) == 402:
                raise ValueError("OpenRouter has insufficient credits. Add credits in OpenRouter or use /provider ollama.") from err
            if getattr(err, "status_code", None) == 401:
                raise ValueError("OpenRouter rejected the API key. Reconfigure it with /provider.") from err
        raise

    if not response.choices:
        raise ValueError("Provider returned no completion. Check the model and account limits.")

    message = response.choices[0].message

    completion_details = getattr(response.usage, "completion_tokens_details", None)
    prompt_details = getattr(response.usage, "prompt_tokens_details", None)

    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
        "cached_tokens": getattr(prompt_details, "cached_tokens", None),
    }

    return message, usage


if __name__ == "__main__":
    user_input = input("Enter your prompt> ")

    message, usage = call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ])

    print("\nAgent: ", message.content, "\n")

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        result = TOOLS[tool_call.function.name](**args)
        print("Tool: ", tool_call.function.name, args)
        print(result, "\n")

    print(usage)
