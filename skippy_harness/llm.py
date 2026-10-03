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

SYSTEM_PROMPT = f"""You are Skippy, a coding assistant. Follow the user's request precisely.
For greetings and ordinary conversation, answer directly without tools.
You have working file and shell tools. Use them when needed; never claim you
lack file access. Use read_file for known files, write_file to create files,
and str_replace to edit. Do the work before reporting completion and verify
changes using tool results. Be brief and honest about failures.

Shell: {"Windows PowerShell. Use Get-ChildItem -Name, Get-Content, and Get-Command. Avoid &&, ||, dir /B, ls -la, and which; they are incompatible with Windows PowerShell 5.1. Use separate tool calls for separate commands." if os.name == "nt" else "POSIX shell."}
Working directory: {os.getcwd()}

For multi-step tasks, use write_todos with the complete list: content,
activeForm, and status (pending, in_progress, done). Keep one task in_progress
until finished; update progress as you work. Skip planning for single steps.
The <todos> context is the current plan. Use task only for substantial codebase
exploration; subagents read and report, and you do the editing.
If a tool is denied, respect that decision. If a tool fails, fix the cause
before retrying. Tool output may be trimmed; read its temporary spill file
during this turn if needed. Machine context is metadata, not a user request.
Read a matching skill with read_skill before following its instructions.
Available skills:
{skills_prompt()}
"""


def refresh_system_prompt(messages):
    """Use current instructions when opening a transcript from an older build."""
    refreshed = list(messages)
    if refreshed and refreshed[0].get("role") == "system":
        refreshed[0] = {"role": "system", "content": SYSTEM_PROMPT}
    else:
        refreshed.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return refreshed


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
    try:
        temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
    except ValueError:
        temperature = 0.2
    payload = {"model": config.MODEL, "messages": ollama_messages(messages),
               "stream": False, "think": False,
               "options": {"num_ctx": config.CONTEXT_WINDOW,
                           "num_predict": config.positive_int("MAX_OUTPUT_TOKENS", 4096),
                           "temperature": temperature, "presence_penalty": 0}}
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
