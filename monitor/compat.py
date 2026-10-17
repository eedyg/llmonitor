"""Legacy /v1/completions <-> /v1/chat/completions translation.

Only used for NInfer, which has no legacy completions endpoint. The same
contract as ../compat_openai.py, but expressed as stream transforms so the
gateway can tap the raw chat stream and translate on the way out.
"""

from __future__ import annotations

import json
import time
from typing import Iterable, Iterator

PASSTHROUGH_FIELDS = (
    "max_tokens",
    "temperature",
    "top_p",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "user",
)


def to_chat_request(payload: dict) -> tuple[dict, str]:
    prompt = payload.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(str(item) for item in prompt)
    prompt = str(prompt)
    chat: dict = {
        "model": payload.get("model") or "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": bool(payload.get("stream")),
    }
    # Legacy clients want plain continuation text, not a thinking trace.
    chat["reasoning_effort"] = payload.get("reasoning_effort", "none")
    for field in PASSTHROUGH_FIELDS:
        if field in payload:
            chat[field] = payload[field]
    return chat, prompt


def chat_to_completion(data: dict, prompt: str, echo: bool) -> dict:
    choices = []
    for index, choice in enumerate(data.get("choices") or []):
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if echo:
            text = prompt + text
        choices.append(
            {
                "text": text,
                "index": choice.get("index", index),
                "logprobs": None,
                "finish_reason": choice.get("finish_reason") or "stop",
            }
        )
    return {
        "id": data.get("id") or "cmpl-monitor",
        "object": "text_completion",
        "created": data.get("created") or int(time.time()),
        "model": data.get("model"),
        "choices": choices,
        "usage": data.get("usage") or {},
    }


class CompletionStream:
    """Stateful translator for one SSE response."""

    def __init__(self, prompt: str, echo: bool, model: str | None) -> None:
        self.prompt = prompt
        self.echo = echo
        self.model = model
        self.id = "cmpl-monitor"
        self.created = int(time.time())
        self.echo_sent = False
        self.buf = b""

    def feed(self, chunk: bytes) -> bytes:
        self.buf += chunk
        out: list[bytes] = []
        while True:
            idx = self.buf.find(b"\n\n")
            if idx < 0:
                break
            block, self.buf = self.buf[:idx], self.buf[idx + 2 :]
            for line in block.split(b"\n"):
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                try:
                    out.extend(self._event(event))
                except Exception:
                    continue
        return b"".join(out)

    def _emit(self, text: str, finish: str | None) -> bytes:
        chunk = {
            "id": self.id,
            "object": "text_completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {"text": text, "index": 0, "logprobs": None, "finish_reason": finish}
            ],
        }
        return ("data: %s\n\n" % json.dumps(chunk, ensure_ascii=False)).encode()

    def start(self) -> bytes:
        if self.echo and self.echo_sent is False:
            self.echo_sent = True
            return self._emit(self.prompt, None)
        return b""

    def _event(self, event: dict) -> list[bytes]:
        out: list[bytes] = []
        if event.get("id"):
            self.id = event["id"]
        if event.get("model"):
            self.model = event["model"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
            if text:
                out.append(self._emit(text, None))
            if choice.get("finish_reason"):
                out.append(self._emit("", choice["finish_reason"]))
        return out

    def end(self) -> bytes:
        return b"data: [DONE]\n\n"


def translate_stream(chunks: Iterable[bytes], prompt: str, echo: bool,
                     model: str | None) -> Iterator[bytes]:
    state = CompletionStream(prompt, echo, model)
    head = state.start()
    if head:
        yield head
    for chunk in chunks:
        out = state.feed(chunk)
        if out:
            yield out
    rest = state.feed(b"\n\n")
    if rest:
        yield rest
    yield state.end()
