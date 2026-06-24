"""Provider-agnostic LLM client for Team Doctor.

One function, many backends. Most providers speak the OpenAI chat-completions
dialect, so they share a single code path; Anthropic has its own shape. No
provider SDKs — just HTTP via requests — so adding a provider is a one-line
config entry and the app stays dependency-light.

The model is used ONLY to (a) turn a messy team description into structure and
(b) narrate findings. Every actual diagnosis comes from the deterministic engine
(raci.check / health.coach), so "explainable, can't hallucinate" still holds.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Dict, List, Optional

import requests

# Cloud models answer in seconds; a local model must load into VRAM and generate
# a large JSON, so it gets a much longer leash (especially on the first call).
TIMEOUT = 90
LOCAL_TIMEOUT = 300


def _timeout_for(cfg) -> int:
    return LOCAL_TIMEOUT if "localhost" in cfg.get("base_url", "") else TIMEOUT

# Free options first on purpose — they're the headline for a no-cost demo.
PROVIDERS: Dict[str, Dict] = {
    "Google Gemini (free)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
        "get_key": "https://aistudio.google.com/apikey",
        "kind": "openai",
        "supports_json_mode": False,
        "supports_vision": True,
        "needs_key": True,
    },
    "Groq (free)": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "get_key": "https://console.groq.com/keys",
        "kind": "openai",
        "supports_json_mode": True,
        "needs_key": True,
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "get_key": "https://platform.deepseek.com/api_keys",
        "kind": "openai",
        "supports_json_mode": True,
        "needs_key": True,
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "get_key": "https://platform.openai.com/api-keys",
        "kind": "openai",
        "supports_json_mode": True,
        "supports_vision": True,
        "needs_key": True,
    },
    "Anthropic (Claude)": {
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-3-5-haiku-latest",
        "get_key": "https://console.anthropic.com/settings/keys",
        "kind": "anthropic",
        "supports_json_mode": False,
        # Default model (haiku) is text-only; leave vision off so a booth visitor
        # gets the clear "switch to Gemini/OpenAI" message instead of an API error.
        "supports_vision": False,
        "needs_key": True,
    },
    "Ollama (local, free)": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1",
        "get_key": "https://ollama.com/download",
        "kind": "openai",
        "supports_json_mode": True,
        "needs_key": False,
    },
}


class LLMError(RuntimeError):
    """Raised with a human-friendly message the page can show directly."""


def chat(provider: str, model: str, api_key: str, messages: List[Dict],
         temperature: float = 0.3, json_mode: bool = False) -> str:
    cfg = PROVIDERS.get(provider)
    if not cfg:
        raise LLMError(f"Unknown provider: {provider}")
    model = (model or cfg["default_model"]).strip()
    if cfg["kind"] == "anthropic":
        return _anthropic(cfg, model, api_key, messages, temperature)
    return _openai(cfg, model, api_key, messages, temperature,
                   json_mode and cfg.get("supports_json_mode", False))


VISION_PROMPT = (
    "These image(s) describe a team — e.g. a photo of a whiteboard, an org chart, "
    "a roster, a screenshot, or handwritten notes. Read everything in them and "
    "write a clear plain-English description of the team: who is on it, each "
    "person's role, the areas of work, and who owns or does what. Transcribe any "
    "text faithfully. Output ONLY the description, as prose — no preamble.")


def transcribe_images(provider: str, model: str, api_key: str,
                      images: List[tuple]) -> str:
    """Turn uploaded image(s) into a plain-text team description via the visitor's
    OWN vision-capable model. ``images`` is a list of (mime_type, raw_bytes).

    Only providers whose default model can see are enabled; others get a friendly
    nudge to switch. Keeps the BYOK promise — it's the visitor's key and quota.
    """
    cfg = PROVIDERS.get(provider)
    if not cfg:
        raise LLMError(f"Unknown provider: {provider}")
    if not cfg.get("supports_vision"):
        raise LLMError("This provider can't read images. Switch to "
                       "**Google Gemini (free)** or **OpenAI** in the sidebar, or "
                       "upload a PDF / Word / text file instead.")
    if not images:
        return ""
    model = (model or cfg["default_model"]).strip()
    content = [{"type": "text", "text": VISION_PROMPT}]
    for mime, raw in images:
        b64 = base64.b64encode(raw).decode("ascii")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return _openai(cfg, model, api_key,
                   [{"role": "user", "content": content}], 0.2, False)


def _openai(cfg, model, api_key, messages, temperature, json_mode) -> str:
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=_timeout_for(cfg))
    except requests.exceptions.ConnectionError:
        raise LLMError(_conn_hint(cfg))
    except requests.exceptions.Timeout:
        raise LLMError("The model took too long. Try again or pick a smaller model.")
    if r.status_code != 200:
        raise LLMError(_http_hint(r))
    try:
        return r.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError):
        raise LLMError("Got an unexpected response shape from the provider.")


def _anthropic(cfg, model, api_key, messages, temperature) -> str:
    url = cfg["base_url"].rstrip("/") + "/messages"
    system, convo = "", []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            convo.append({"role": m["role"], "content": m["content"]})
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {"model": model, "max_tokens": 1500, "temperature": temperature,
            "messages": convo}
    if system:
        body["system"] = system
    try:
        r = requests.post(url, headers=headers, json=body, timeout=_timeout_for(cfg))
    except requests.exceptions.ConnectionError:
        raise LLMError("Couldn't reach Anthropic. Check your internet connection.")
    except requests.exceptions.Timeout:
        raise LLMError("Anthropic took too long. Try again.")
    if r.status_code != 200:
        raise LLMError(_http_hint(r))
    try:
        return "".join(p.get("text", "") for p in r.json().get("content", []))
    except ValueError:
        raise LLMError("Got an unexpected response shape from Anthropic.")


def _conn_hint(cfg) -> str:
    if "localhost" in cfg["base_url"]:
        return ("Couldn't reach Ollama at localhost:11434. Start it with "
                "`ollama serve` and pull the model (`ollama pull llama3.1`).")
    return "Couldn't reach the provider. Check your internet connection."


def _http_hint(r) -> str:
    msg = ""
    try:
        body = r.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        err = body.get("error", body)
        msg = err.get("message", "") if isinstance(err, dict) else str(err)
    elif body is not None:
        msg = str(body)[:200]
    else:
        msg = r.text[:200]
    msg = (msg or "").strip()
    if r.status_code in (401, 403):
        return f"Authentication failed ({r.status_code}). Check your API key. {msg}".strip()
    if r.status_code == 404:
        return f"Model not found ({r.status_code}). Check the model name. {msg}".strip()
    if r.status_code == 429:
        return "Rate limit hit. Wait a moment, then try again (or switch provider)."
    return f"Provider error {r.status_code}: {msg or r.text[:200]}"


def extract_json(text: str) -> Optional[dict]:
    """Pull a JSON object from a model reply, tolerating fences and stray prose."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start:end + 1]
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
