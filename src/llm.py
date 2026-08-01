"""Pluggable LLM backend with timeouts, retries and usage capture.

    LLM_PROVIDER = ollama | openai | anthropic     (unset -> deterministic mode)
    LLM_MODEL    = <model name>                    (optional override)

Deterministic mode (no provider) is the default: the categoriser uses transparent
keyword rules and the rationale is a template, so the whole system runs end-to-end
offline with no keys — which is what CI and the evaluation harness rely on.
The LLM paths are real and activate the moment a provider is configured.

Design decisions:
  * temperature 0 and a fixed seed wherever the provider supports it;
  * JSON mode requested where supported (Ollama `format=json`,
    OpenAI `response_format=json_object`); Anthropic uses prompt-enforced JSON;
  * every call returns latency + token usage for logging and evaluation;
  * retries with exponential backoff; a provider that still fails raises
    `LLMUnavailable`, and the caller falls back to rules + a guardrail warning —
    a provider outage must never break an assessment, only make it conservative.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from config import llm_config


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """Provider unreachable / kept failing after retries."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class CallStats:
    calls: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms_total: float = 0.0
    errors: list[str] = field(default_factory=list)


def extract_json(text: str) -> dict:
    """Parse the first JSON object out of a model reply (handles code fences)."""
    if not text:
        raise ValueError("empty model response")
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        raise ValueError("no JSON object found in model response")
    return json.loads(m.group(0))


class BaseLLM:
    name = "base"
    model = ""

    def _chat(self, system: str, user: str, json_mode: bool) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def chat(
        self, system: str, user: str, json_mode: bool = False, stats: CallStats | None = None
    ) -> LLMResponse:
        """Chat with retries + exponential backoff. Raises LLMUnavailable at the end."""
        cfg = llm_config()
        last_err: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._chat(system, user, json_mode)
                r.latency_ms = (time.perf_counter() - t0) * 1000
                if stats:
                    stats.calls += 1
                    stats.retries += attempt
                    stats.prompt_tokens += r.prompt_tokens
                    stats.completion_tokens += r.completion_tokens
                    stats.latency_ms_total += r.latency_ms
                return r
            except Exception as e:  # provider/network error -> backoff and retry
                last_err = e
                if stats:
                    stats.errors.append(f"{type(e).__name__}: {e}")
                if attempt < cfg.max_retries:
                    time.sleep(min(2**attempt * 0.5, 4.0))
        if stats:
            stats.failures += 1
        raise LLMUnavailable(
            f"{self.name}/{self.model} failed after {cfg.max_retries + 1} attempts: {last_err}"
        ) from last_err


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, model: str | None = None):
        cfg = llm_config()
        self.model = model or cfg.model or "llama3.2"
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def _chat(self, system, user, json_mode):
        import requests

        cfg = llm_config()
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {
                "temperature": cfg.temperature,
                "seed": cfg.seed,
                "num_predict": cfg.max_output_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{self.host}/api/chat", timeout=cfg.timeout_s, json=payload)
        r.raise_for_status()
        data = r.json()
        return LLMResponse(
            text=data["message"]["content"],
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )


class OpenAILLM(BaseLLM):
    name = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        cfg = llm_config()
        self.model = model or cfg.model or "gpt-4o-mini"
        self.client = OpenAI(timeout=cfg.timeout_s, max_retries=0)  # retries handled here

    def _chat(self, system, user, json_mode):
        cfg = llm_config()
        kwargs = dict(
            model=self.model,
            temperature=cfg.temperature,
            seed=cfg.seed,
            max_tokens=cfg.max_output_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = self.client.chat.completions.create(**kwargs)
        u = r.usage
        return LLMResponse(
            text=r.choices[0].message.content or "",
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        )


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic

        cfg = llm_config()
        self.model = model or cfg.model or "claude-3-5-haiku-latest"
        self.client = anthropic.Anthropic(timeout=cfg.timeout_s, max_retries=0)

    def _chat(self, system, user, json_mode):
        cfg = llm_config()
        if json_mode:  # no native JSON mode -> enforce via prompt + validation layer
            system = system + "\nRespond with a single valid JSON object and nothing else."
        r = self.client.messages.create(
            model=self.model,
            max_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return LLMResponse(
            text="".join(b.text for b in r.content if b.type == "text"),
            prompt_tokens=r.usage.input_tokens,
            completion_tokens=r.usage.output_tokens,
        )


def get_llm() -> BaseLLM | None:
    """Return the configured provider, or None for deterministic mode."""
    p = llm_config().provider
    if p == "openai":
        return OpenAILLM()
    if p == "anthropic":
        return AnthropicLLM()
    if p == "ollama":
        return OllamaLLM()
    if p in ("", "none", "deterministic"):
        return None
    raise ValueError(f"unknown LLM_PROVIDER {p!r} (use ollama|openai|anthropic or unset)")
