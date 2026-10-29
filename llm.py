"""
One place that talks to an AI model, so the rest of the app doesn't care which one you use.

Set in config.env:
  LLM_PROVIDER = anthropic | gemini | groq | openrouter | ollama | custom
  LLM_API_KEY  = your key for that provider (not needed for ollama)
  LLM_MODEL    = optional; each provider has a sensible default below
  LLM_BASE_URL = only for "custom" (any OpenAI-compatible endpoint)

If nothing is set but ANTHROPIC_API_KEY is, it uses Claude (the original behavior).
If a provider rate-limits us, AI calls pause for 10 minutes and the app falls back to rules.
"""
import json, os, re, time

import requests

from common import log

PROVIDERS = {  # base URL, default model for quick jobs, default model for the brief
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.8-flash", "gemini-3.8-flash"),
    "groq":       ("https://api.groq.com/openai/v1/", "openai/gpt-oss-120b", "openai/gpt-oss-120b"),
    "openrouter": ("https://openrouter.ai/api/v1/", "", ""),
    "ollama":     ("http://localhost:11434/v1/", "llama3.1:8b", "llama3.1:8b"),
}
_paused_until = 0.0


def provider():
    p = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if p:
        return p
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else ""


def enabled():
    p = provider()
    if not p:
        return False
    if p == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY"))
    if p == "ollama":
        return True
    return bool(os.environ.get("LLM_API_KEY"))


def label():
    p = provider()
    return {"anthropic": "Claude", "gemini": "Gemini", "groq": "Groq", "openrouter": "OpenRouter",
            "ollama": "local model"}.get(p, p or "none")


def _model(kind):
    p = provider()
    if kind == "brief" and os.environ.get("LLM_BRIEF_MODEL"):
        return os.environ["LLM_BRIEF_MODEL"]
    if os.environ.get("LLM_MODEL"):
        return os.environ["LLM_MODEL"]
    if p == "anthropic":
        return os.environ.get("BRIEF_MODEL", "claude-sonnet-5") if kind == "brief" \
            else os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    base = PROVIDERS.get(p, ("", "", ""))
    return base[2] if kind == "brief" else base[1]


def chat(prompt, max_tokens=800, kind="quick", timeout=90, json_mode=False):
    """Send one prompt, return the reply text, or None on any failure (callers fall back to rules)."""
    global _paused_until
    if not enabled() or time.time() < _paused_until:
        return None
    p, model = provider(), _model(kind)
    try:
        if p == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")
            r = requests.post("https://api.anthropic.com/v1/messages",
                              headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json={"model": model, "max_tokens": max_tokens,
                                    "messages": [{"role": "user", "content": prompt}]}, timeout=timeout)
        else:
            base = os.environ.get("LLM_BASE_URL") or PROVIDERS.get(p, ("",))[0]
            if not base or not model:
                log(f"[ai] {p}: set LLM_BASE_URL and/or LLM_MODEL in config.env")
                return None
            body = {"model": model, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
            # Reasoning models (Gemini Flash, Groq's gpt-oss) spend part of max_tokens "thinking" before
            # answering; keeping that short leaves room for the actual reply.
            effort = os.environ.get("LLM_REASONING", "low" if p in ("gemini", "groq") else "")
            if effort:
                body["reasoning_effort"] = effort
            if json_mode and p in ("groq", "gemini", "openrouter"):
                body["response_format"] = {"type": "json_object"}   # forces a clean JSON reply
            headers = {"content-type": "application/json"}
            if os.environ.get("LLM_API_KEY"):
                headers["Authorization"] = f"Bearer {os.environ['LLM_API_KEY']}"
            r = requests.post(base.rstrip("/") + "/chat/completions", headers=headers, json=body, timeout=timeout)

        if r.status_code == 429:
            _paused_until = time.time() + 600
            log(f"[ai] {label()} rate limit reached; pausing AI for 10 minutes (rules keep working)")
            return None
        if r.status_code >= 400:
            log(f"[ai] {label()} error {r.status_code}: {r.text[:200]}")
            if r.status_code in (401, 403) or "api key" in r.text.lower():
                _paused_until = time.time() + 3600      # bad key: don't retry every minute
                log("[ai] check LLM_API_KEY in config.env, then restart; AI paused for an hour")
            return None
        data = r.json()
        if p == "anthropic":
            return "".join(b.get("text", "") for b in data.get("content", []))
        text = (data["choices"][0]["message"].get("content") or "").strip()
        if not text:
            log(f"[ai] {label()} returned an empty reply (it likely used its token budget thinking); using rules")
            return None
        return text
    except Exception as e:
        log(f"[ai] {label()} request failed: {e}")
        return None


def chat_json(prompt, max_tokens=800, kind="quick", timeout=90):
    """Like chat(), but pulls the first {...} JSON object out of the reply."""
    text = chat(prompt, max_tokens=max_tokens, kind=kind, timeout=timeout, json_mode=True)
    if not text:
        return None
    try:
        return json.loads(re.search(r"\{.*\}", re.sub(r"```(json)?", "", text), re.S).group(0))
    except Exception:
        snippet = text[:120].replace("\n", " ")
        log(f"[ai] {label()} reply wasn't valid JSON ({len(text)} chars: {snippet!r}); using rules")
        return None
