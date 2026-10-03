"""Provider model discovery and actionable connection errors."""

import json
import socket
import urllib.error
import urllib.request

from . import config

_CACHE = {}


def request_json(method, url, *, json_body=None, headers=None, timeout=15):
    try:
        body = json.dumps(json_body).encode("utf-8") if json_body is not None else None
        request_headers = dict(headers or {})
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
        if data.get("error"):
            raise ValueError(str(data["error"]))
        return data
    except urllib.error.HTTPError as err:
        status = err.code
        hint = "Check your API key." if status in (401, 403) else "Check the endpoint and model ID with /model."
        try:
            detail = json.loads(err.read()).get("error", "")
            if isinstance(detail, dict):
                detail = detail.get("message", "")
            detail = str(detail)[:500]
            if config.API_KEY:
                detail = detail.replace(config.API_KEY, "[redacted]")
        except (ValueError, OSError):
            detail = ""
        raise ValueError(f"Provider returned HTTP {status}: {detail}. {hint}") from err
    except (TimeoutError, socket.timeout) as err:
        raise ValueError("Provider timed out. Try a smaller Ollama model or increase LLM_TIMEOUT in .env.") from err
    except urllib.error.URLError as err:
        if config.ACTIVE_PROVIDER == "ollama":
            raise ValueError(f"Cannot connect to Ollama at {config.BASE_URL}. Start Ollama (or run ollama serve), then retry /model.") from err
        raise ValueError("Cannot connect to the provider. Check your connection and endpoint.") from err


def ollama_root():
    return config.BASE_URL.rstrip("/").removesuffix("/v1")


def ollama_headers():
    return {"Authorization": f"Bearer {config.API_KEY}"} if config.API_KEY not in ("", "ollama") else {}


def available_models(refresh=True):
    key = (config.ACTIVE_PROVIDER, config.BASE_URL)
    if not refresh and key in _CACHE:
        return _CACHE[key]
    if config.ACTIVE_PROVIDER == "ollama":
        data = request_json("GET", ollama_root() + "/api/tags", headers=ollama_headers())
        models = {m["name"]: m for m in data.get("models", [])}
        if not models:
            raise ValueError("No Ollama models are installed. Run ollama pull <model>, then retry /model.")
    else:
        headers = {"Authorization": f"Bearer {config.API_KEY}"} if config.API_KEY else {}
        data = request_json("GET", config.BASE_URL.rstrip("/") + "/models", headers=headers)
        models = {}
        for m in data.get("data", []):
            if config.ACTIVE_PROVIDER == "openrouter":
                if "tools" not in m.get("supported_parameters", []):
                    continue
                if "text" not in m.get("architecture", {}).get("output_modalities", ["text"]):
                    continue
            models[m["id"]] = m
        if not models:
            raise ValueError("The provider returned no models supporting coding tools.")
    _CACHE[key] = dict(sorted(models.items()))
    return _CACHE[key]


def validate_model(name, refresh=False):
    models = available_models(refresh=refresh)
    if config.ACTIVE_PROVIDER == "ollama" and name not in models and name + ":latest" in models:
        name += ":latest"
    if name not in models:
        if config.ACTIVE_PROVIDER == "ollama":
            raise ValueError(f"Ollama model '{name}' is not installed. Choose /model or run ollama pull {name}.")
        raise ValueError(f"Model '{name}' is unavailable or does not support tools. Choose one using /model.")
    if config.ACTIVE_PROVIDER == "ollama":
        details = request_json("POST", ollama_root() + "/api/show", json_body={"model": name}, headers=ollama_headers())
        if "capabilities" in details and "tools" not in details["capabilities"]:
            raise ValueError(f"Ollama model '{name}' does not support tools. Choose another model using /model.")
    return name


def apply_context_limit():
    if config.ACTIVE_PROVIDER == "openrouter":
        details = available_models(refresh=False).get(config.MODEL, {})
        if details.get("context_length"):
            config.CONTEXT_WINDOW = min(config.CONTEXT_WINDOW, int(details["context_length"]))
