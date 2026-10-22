"""Rebuild a non-streaming chat completion from the engine's SSE deltas.

The gateway asks the engine for a stream even when the client sent
``stream: false`` (see gateway.py). Two reasons, both learned the hard way:

* a non-streamed request is **invisible** while it runs — no token deltas, so
  the UI shows "queued, 0 tokens" for a request that has been burning the GPU
  for half an hour (that is exactly how a Qwen3 thinking loop hid in plain
  sight: 26 min, ~100k tokens, no way to tell it apart from a stuck queue);
* a non-streamed request is **unabortable** — the engine consumes the whole
  generation inside its own process, so closing the client connection does
  nothing. vLLM only aborts a request when its output generator is closed,
  which can only happen through the streaming path.

So this module has exactly one job: turn those deltas back into the JSON body a
non-streaming client expects, byte-for-byte close enough that nothing can tell.
"""

from __future__ import annotations

import time
from typing import Any

# Fields the engine echoes on every chunk; keep the first non-null one.
_ECHO = ("id", "object", "created", "model", "system_fingerprint", "service_tier")


def _text_of(content: Any) -> str:
    """Flatten `content` (str, or a list of parts) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


class ChatAssembler:
    """Accumulate OpenAI chat SSE events into one `chat.completion` object."""

    def __init__(self) -> None:
        self.meta: dict[str, Any] = {}
        self.choices: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None
        self.events = 0
        self.error: dict[str, Any] | None = None

    # -- accumulation -------------------------------------------------------
    def feed_event(self, obj: dict) -> None:
        if not isinstance(obj, dict):
            return
        self.events += 1
        for key in _ECHO:
            if self.meta.get(key) is None and obj.get(key) is not None:
                self.meta[key] = obj[key]
        if isinstance(obj.get("error"), (dict, str)):
            self.error = obj["error"] if isinstance(obj["error"], dict) else {"message": obj["error"]}
        # An `include_usage` stream ends with a usage-only chunk (`choices: []`).
        if isinstance(obj.get("usage"), dict) and obj["usage"]:
            self.usage = obj["usage"]
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            try:
                index = int(choice.get("index") or 0)
            except (TypeError, ValueError):
                index = 0
            slot = self.choices.setdefault(index, {
                "role": "assistant", "content": [], "reasoning": [],
                "tool_calls": {}, "refusal": [], "finish_reason": None, "logprobs": None,
            })
            delta = choice.get("delta")
            if isinstance(delta, dict):
                self._delta(slot, delta)
            # Some backends answer a "stream" request with a plain message.
            message = choice.get("message")
            if isinstance(message, dict):
                self._delta(slot, message)
            if choice.get("finish_reason"):
                slot["finish_reason"] = str(choice["finish_reason"])
            if choice.get("logprobs") is not None:
                slot["logprobs"] = choice["logprobs"]

    def _delta(self, slot: dict[str, Any], delta: dict) -> None:
        if isinstance(delta.get("role"), str) and delta["role"]:
            slot["role"] = delta["role"]
        text = _text_of(delta.get("content"))
        if text:
            slot["content"].append(text)
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            slot["reasoning"].append(reasoning)
        refusal = delta.get("refusal")
        if isinstance(refusal, str) and refusal:
            slot["refusal"].append(refusal)
        for call in delta.get("tool_calls") or []:
            self._tool_call(slot, call)

    def _tool_call(self, slot: dict[str, Any], call: Any) -> None:
        """Merge one tool-call fragment. Arguments arrive split across chunks."""
        if not isinstance(call, dict):
            return
        try:
            index = int(call.get("index") or 0)
        except (TypeError, ValueError):
            index = 0
        acc = slot["tool_calls"].setdefault(index, {
            "id": None, "type": "function", "function": {"name": "", "arguments": ""},
        })
        if call.get("id"):
            acc["id"] = call["id"]
        if call.get("type"):
            acc["type"] = call["type"]
        function = call.get("function")
        if isinstance(function, dict):
            if isinstance(function.get("name"), str) and function["name"]:
                # The name is sent once by the engines we serve; a client that
                # resends it would otherwise get "get_weatherget_weather".
                acc["function"]["name"] = function["name"]
            if isinstance(function.get("arguments"), str):
                acc["function"]["arguments"] += function["arguments"]

    # -- result -------------------------------------------------------------
    def result(self) -> dict[str, Any]:
        choices = []
        for index in sorted(self.choices):
            slot = self.choices[index]
            message: dict[str, Any] = {
                "role": slot["role"],
                "content": "".join(slot["content"]),
            }
            reasoning = "".join(slot["reasoning"])
            if reasoning:
                message["reasoning_content"] = reasoning
            if slot["tool_calls"]:
                message["tool_calls"] = [slot["tool_calls"][i] for i in sorted(slot["tool_calls"])]
            if slot["refusal"]:
                message["refusal"] = "".join(slot["refusal"])
            choices.append({
                "index": index,
                "message": message,
                "logprobs": slot["logprobs"],
                "finish_reason": slot["finish_reason"] or "stop",
            })
        body: dict[str, Any] = {
            "id": self.meta.get("id"),
            "object": "chat.completion",
            "created": self.meta.get("created") or int(time.time()),
            "model": self.meta.get("model"),
            "choices": choices,
        }
        for key in ("system_fingerprint", "service_tier"):
            if self.meta.get(key) is not None:
                body[key] = self.meta[key]
        if self.usage is not None:
            body["usage"] = self.usage
        return body
