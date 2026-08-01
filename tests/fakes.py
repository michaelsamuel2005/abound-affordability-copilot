"""Test doubles for the LLM layer — scripted, offline, deterministic.

FakeLLM returns queued responses in order (repeating the last one), so tests can
script exact model behaviour: valid output, malformed JSON, invented IDs, missing
items, ungrounded rationales. BrokenLLM fails every call, exercising the
LLMUnavailable fallback path.
"""

from __future__ import annotations

import json

from llm import BaseLLM, LLMResponse


class FakeLLM(BaseLLM):
    name = "fake"
    model = "scripted"

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def _chat(self, system: str, user: str, json_mode: bool) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):  # scripted mid-sequence provider failure
            raise item
        return LLMResponse(text=item, prompt_tokens=100, completion_tokens=50)


class BrokenLLM(BaseLLM):
    name = "broken"
    model = "down"

    def _chat(self, system, user, json_mode):
        raise ConnectionError("provider unreachable")


def valid_items_json(
    txns: list[dict],
    category: str = "groceries",
    confidence: float = 0.9,
    overrides: dict | None = None,
) -> str:
    """A schema-valid categorisation reply covering every input transaction."""
    overrides = overrides or {}
    items = [
        {
            "transaction_id": t["transaction_id"],
            "category": overrides.get(t["transaction_id"], category),
            "confidence": confidence,
        }
        for t in txns
    ]
    return json.dumps({"items": items})
