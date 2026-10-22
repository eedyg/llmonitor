"""Per-request tap: parse streamed protocol deltas while bytes pass through.

Works for all three wire protocols both engines speak:
  * OpenAI chat completions (incl. reasoning_content thinking split)
  * OpenAI legacy text completions
  * OpenAI Responses
  * Anthropic messages

Nothing is buffered for the client: the gateway yields the raw bytes first and
feeds a *copy* here.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

CHAT = "openai_chat"
TEXT = "openai_text"
RESPONSES = "openai_responses"
ANTHROPIC = "anthropic"

PROTOCOL_BY_PATH = {
    "/v1/chat/completions": CHAT,
    "/v1/completions": TEXT,
    "/v1/responses": RESPONSES,
    "/v1/messages": ANTHROPIC,
}

GENERATION_PATHS = set(PROTOCOL_BY_PATH)

TAIL_CHARS = 900


def est_tokens(text: str) -> int:
    """Cheap token estimate: CJK ~0.65 tok/char, latin ~1 tok / 3.8 chars."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u2e80" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af")
    other = len(text) - cjk
    return int(cjk * 0.65 + other / 3.8) + 1


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    out.append(part["text"])
                elif part.get("type") in ("image_url", "input_image", "image"):
                    out.append("[image]")
        return "".join(out)
    return ""


@dataclass
class Record:
    id: str = field(default_factory=lambda: "mreq_" + uuid.uuid4().hex[:16])
    engine: str | None = None
    endpoint: str = ""
    protocol: str = CHAT
    stream: bool = False
    model: str | None = None
    method: str = "POST"

    started_at: float = field(default_factory=time.time)
    started_iso: str = ""
    finished_at: float | None = None
    first_token_at: float | None = None
    first_content_at: float | None = None
    phase: str = "queued"

    client_ip: str | None = None
    client_ua: str | None = None

    request_bytes: int = 0
    prompt_messages: int = 0
    prompt_chars: int = 0
    prompt_tokens_est: int = 0
    has_media: bool = False
    tools: int = 0
    max_tokens: int | None = None
    sampling: dict = field(default_factory=dict)
    thinking_requested: bool | None = None

    prompt_tokens: int | None = None
    prompt_cached: int | None = None
    thinking_tokens: int = 0
    content_tokens: int = 0
    thinking_chunks: int = 0
    content_chunks: int = 0
    thinking_chars: int = 0
    content_chars: int = 0
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)

    status: int | None = None
    error: str | None = None
    engine_request_id: str | None = None

    thinking_text: str = ""
    content_text: str = ""
    thinking_tail: str = ""
    content_tail: str = ""

    engine_record: dict | None = None

    # -- derived ------------------------------------------------------------
    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started_at) * 1000.0, 2)

    @property
    def ttfc_ms(self) -> float | None:
        if self.first_content_at is None:
            return None
        return round((self.first_content_at - self.started_at) * 1000.0, 2)

    @property
    def total_ms(self) -> float | None:
        end = self.finished_at or time.time()
        return round((end - self.started_at) * 1000.0, 2)

    @property
    def decode_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return round(((self.finished_at or time.time()) - self.first_token_at) * 1000.0, 2)

    @property
    def decode_tps(self) -> float | None:
        if self.first_token_at is None or not self.finished_at:
            return None
        span = self.finished_at - self.first_token_at
        tokens = max(0, self.thinking_tokens + self.content_tokens - 1)
        if span <= 0.05 or tokens <= 0:
            return None
        return round(tokens / span, 2)

    @property
    def prompt_kind(self) -> str:
        if self.prompt_cached and self.prompt_tokens:
            return f"{self.prompt_cached}/{self.prompt_tokens}"
        return str(self.prompt_tokens or self.prompt_tokens_est or "?")

    def public(self, include_text: bool = False) -> dict:
        data = {
            "id": self.id,
            "engine": self.engine,
            "endpoint": self.endpoint,
            "protocol": self.protocol,
            "stream": self.stream,
            "model": self.model,
            "phase": self.phase,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": self.total_ms,
            "ttft_ms": self.ttft_ms,
            "ttfc_ms": self.ttfc_ms,
            "decode_ms": self.decode_ms,
            "decode_tps": self.decode_tps,
            "prompt_tokens": self.prompt_tokens,
            "prompt_tokens_est": self.prompt_tokens_est,
            "prompt_cached": self.prompt_cached,
            "prompt_chars": self.prompt_chars,
            "request_bytes": self.request_bytes,
            "prompt_messages": self.prompt_messages,
            "has_media": self.has_media,
            "tools": self.tools,
            "thinking_tokens": self.thinking_tokens,
            "content_tokens": self.content_tokens,
            "thinking_chars": self.thinking_chars,
            "content_chars": self.content_chars,
            "capacity": self.thinking_tokens + self.content_tokens,
            "finish_reason": self.finish_reason,
            "status": self.status,
            "error": self.error,
            "max_tokens": self.max_tokens,
            "sampling": self.sampling,
            "thinking_requested": self.thinking_requested,
            "client_ip": self.client_ip,
            "client_ua": self.client_ua,
            "started_iso": self.started_iso,
            "engine_request_id": self.engine_request_id,
        }
        if include_text:
            data["thinking_text"] = self.thinking_text
            data["content_text"] = self.content_text
            data["engine_record"] = self.engine_record
        return data

    def live(self) -> dict:
        data = self.public()
        data["thinking_tail"] = self.thinking_tail
        data["content_tail"] = self.content_tail
        # live counters: usage only arrives at the end, chunk counts are exact
        # enough while streaming (one delta == one token on both engines).
        if not data["thinking_tokens"]:
            data["thinking_tokens"] = self.thinking_chunks
        if not data["content_tokens"]:
            data["content_tokens"] = self.content_chunks
        if self.first_token_at is not None:
            span = time.time() - self.first_token_at
            tokens = max(0, self.thinking_chunks + self.content_chunks - 1)
            data["live_decode_tps"] = round(tokens / span, 2) if span > 0.15 and tokens > 0 else None
        else:
            data["live_decode_tps"] = None
        return data

    def to_row(self) -> dict:
        row = asdict(self)
        row["ttft_ms"] = self.ttft_ms
        row["ttfc_ms"] = self.ttfc_ms
        row["elapsed_ms"] = self.total_ms
        row["decode_ms"] = self.decode_ms
        row["decode_tps"] = self.decode_tps
        row["sampling_json"] = json.dumps(self.sampling, ensure_ascii=False)
        row["usage_json"] = json.dumps(self.usage, ensure_ascii=False)
        row["engine_record_json"] = json.dumps(self.engine_record, ensure_ascii=False) if self.engine_record else None
        return row


class StreamTap:
    """Consumes SSE bytes and mutates one Record."""

    def __init__(self, record: Record, wire: str | None = None,
                 on_first_token: Callable[[Record], None] | None = None,
                 on_event: Callable[[dict], None] | None = None) -> None:
        self.record = record
        self.wire = wire or record.protocol
        self.buf = b""
        self.done = False
        self.on_first_token = on_first_token
        # Raw parsed event hook: lets the gateway rebuild a non-streaming body
        # from the same deltas it taps (see assemble.ChatAssembler).
        self.on_event = on_event

    # -- SSE framing --------------------------------------------------------
    def feed(self, chunk: bytes) -> None:
        self.buf += chunk
        while True:
            idx_nn = self.buf.find(b"\n\n")
            idx_rn = self.buf.find(b"\r\n\r\n")
            if idx_nn < 0 and idx_rn < 0:
                break
            if idx_rn >= 0 and (idx_nn < 0 or idx_rn < idx_nn):
                block, self.buf = self.buf[:idx_rn], self.buf[idx_rn + 4 :]
            else:
                block, self.buf = self.buf[:idx_nn], self.buf[idx_nn + 2 :]
            self._block(block)

    def _block(self, block: bytes) -> None:
        event = ""
        data_lines = []
        for raw in block.split(b"\n"):
            line = raw.rstrip(b"\r").decode("utf-8", "replace")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            return
        payload = "\n".join(data_lines)
        if payload.strip() in ("[DONE]", "done"):
            self.done = True
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return
        if not isinstance(obj, dict):
            return
        if self.on_event is not None:
            try:
                self.on_event(obj)
            except Exception:  # an assembler bug must never break the stream
                pass
        record = self.record
        try:
            if self.wire == ANTHROPIC or event.startswith("message_") or event.startswith("content_block_"):
                self._anthropic(event, obj, record)
            elif self.wire == RESPONSES or event.startswith("response."):
                self._responses(event, obj, record)
            else:
                self._openai(obj, record)
        except Exception:  # never break the stream for a tap bug
            pass

    # -- protocol handlers --------------------------------------------------
    def _mark_token(self, record: Record, kind: str, text: str) -> None:
        now = time.time()
        if record.first_token_at is None:
            record.first_token_at = now
            record.phase = "decode"
            if self.on_first_token:
                self.on_first_token(record)
        if kind == "thinking":
            record.thinking_chunks += 1
            record.thinking_chars += len(text)
            record.thinking_text = (record.thinking_text + text)[-8000:]
            record.thinking_tail = (record.thinking_tail + text)[-TAIL_CHARS:]
        elif kind == "content":
            if record.first_content_at is None:
                record.first_content_at = now
            record.content_chunks += 1
            record.content_chars += len(text)
            record.content_text = (record.content_text + text)[-8000:]
            record.content_tail = (record.content_tail + text)[-TAIL_CHARS:]
        else:  # text completions
            if record.first_content_at is None:
                record.first_content_at = now
            record.content_chunks += 1
            record.content_chars += len(text)
            record.content_text = (record.content_text + text)[-8000:]
            record.content_tail = (record.content_tail + text)[-TAIL_CHARS:]

    def _usage(self, record: Record, usage: dict) -> None:
        if not isinstance(usage, dict) or not usage:
            return
        record.usage = usage
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        if prompt is not None:
            record.prompt_tokens = int(prompt)
        cached = None
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
        if cached is None:
            cached = usage.get("cache_read_input_tokens")
        if cached is not None:
            record.prompt_cached = int(cached)
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        reasoning = None
        odetails = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        if isinstance(odetails, dict):
            reasoning = odetails.get("reasoning_tokens")
        if reasoning is not None:
            record.thinking_tokens = int(reasoning)
            if completion is not None:
                record.content_tokens = max(0, int(completion) - int(reasoning))
        elif completion is not None:
            record.content_tokens = max(0, int(completion) - record.thinking_tokens)

    def _openai(self, obj: dict, record: Record) -> None:
        choices = obj.get("choices") or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    self._mark_token(record, "thinking", reasoning)
                content = delta.get("content")
                if isinstance(content, str) and content:
                    self._mark_token(record, "content", content)
            text = choice.get("text")
            if isinstance(text, str) and text:
                self._mark_token(record, "content", text)
            message = choice.get("message")
            if isinstance(message, dict):
                self._absorb_message(message, record)
            if choice.get("finish_reason"):
                record.finish_reason = str(choice["finish_reason"])
        if isinstance(obj.get("usage"), dict):
            self._usage(record, obj["usage"])

    def _absorb_message(self, message: dict, record: Record) -> None:
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            self._mark_token(record, "thinking", reasoning)
        content = _text_of(message.get("content"))
        if content:
            self._mark_token(record, "content", content)

    def _anthropic(self, event: str, obj: dict, record: Record) -> None:
        kind = obj.get("type") or event
        if kind == "message_start":
            message = obj.get("message") or {}
            if isinstance(message.get("usage"), dict):
                self._usage(record, message["usage"])
            return
        if kind == "content_block_start":
            block = obj.get("content_block") or {}
            if block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
                self._mark_token(record, "thinking", block["thinking"])
            elif block.get("type") == "text" and isinstance(block.get("text"), str):
                self._mark_token(record, "content", block["text"])
            return
        if kind == "content_block_delta":
            delta = obj.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "thinking_delta" and isinstance(delta.get("thinking"), str):
                self._mark_token(record, "thinking", delta["thinking"])
            elif dtype == "text_delta" and isinstance(delta.get("text"), str):
                self._mark_token(record, "content", delta["text"])
            return
        if kind == "message_delta":
            delta = obj.get("delta") or {}
            if delta.get("stop_reason"):
                record.finish_reason = str(delta["stop_reason"])
            if isinstance(obj.get("usage"), dict):
                self._usage(record, obj["usage"])
            return

    def _responses(self, event: str, obj: dict, record: Record) -> None:
        kind = obj.get("type") or event
        if kind in ("response.output_text.delta", "response.text.delta"):
            if isinstance(obj.get("delta"), str):
                self._mark_token(record, "content", obj["delta"])
            return
        if kind in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning.delta",
        ):
            if isinstance(obj.get("delta"), str):
                self._mark_token(record, "thinking", obj["delta"])
            return
        if kind in ("response.completed", "response.done", "response.incomplete"):
            resp = obj.get("response") or {}
            if isinstance(resp.get("usage"), dict):
                self._usage(record, resp["usage"])
                record.prompt_tokens = record.prompt_tokens or int(
                    resp["usage"].get("input_tokens") or 0
                ) or None
            if resp.get("status"):
                record.finish_reason = str(resp["status"])
            return
        if kind == "response.failed":
            resp = obj.get("response") or {}
            err = resp.get("error") or {}
            record.error = str(err.get("message") or "response.failed")
            record.phase = "error"

    # -- non-stream body ----------------------------------------------------
    def feed_body(self, body: bytes) -> None:
        try:
            obj = json.loads(body)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        record = self.record
        if record.protocol == ANTHROPIC:
            if isinstance(obj.get("usage"), dict):
                self._usage(record, obj["usage"])
            for block in obj.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
                    self._mark_token(record, "thinking", block["thinking"])
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    self._mark_token(record, "content", block["text"])
            if obj.get("stop_reason"):
                record.finish_reason = str(obj["stop_reason"])
            return
        if record.protocol == RESPONSES:
            if isinstance(obj.get("usage"), dict):
                self._usage(record, obj["usage"])
            for item in obj.get("output") or []:
                if not isinstance(item, dict):
                    continue
                for part in item.get("content") or []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in ("output_text", "text") and isinstance(part.get("text"), str):
                        self._mark_token(record, "content", part["text"])
                    elif part.get("type") in ("reasoning_text", "summary_text") and isinstance(part.get("text"), str):
                        self._mark_token(record, "thinking", part["text"])
            if obj.get("status"):
                record.finish_reason = str(obj["status"])
            if obj.get("error"):
                record.error = str((obj.get("error") or {}).get("message") or "error")
            return
        if isinstance(obj.get("usage"), dict):
            self._usage(record, obj["usage"])
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                self._absorb_message(message, record)
            if isinstance(choice.get("text"), str):
                self._mark_token(record, "content", choice["text"])
            if choice.get("finish_reason"):
                record.finish_reason = str(choice["finish_reason"])


def request_meta(record: Record, payload: Any) -> None:
    """Extract request shape from the JSON body."""
    if not isinstance(payload, dict):
        return
    record.model = payload.get("model") or record.model
    record.stream = bool(payload.get("stream"))
    record.max_tokens = payload.get("max_completion_tokens") or payload.get("max_tokens")
    if record.max_tokens is not None:
        try:
            record.max_tokens = int(record.max_tokens)
        except (TypeError, ValueError):
            record.max_tokens = None
    sampling = {}
    for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                "frequency_penalty", "seed", "reasoning_effort", "thinking_budget"):
        if key in payload and payload[key] is not None:
            sampling[key] = payload[key]
    record.sampling = sampling
    if "reasoning_effort" in payload:
        record.thinking_requested = payload.get("reasoning_effort") not in (None, "none")
    elif "thinking_budget" in payload or "thinking" in payload:
        record.thinking_requested = bool(payload.get("thinking_budget") or payload.get("thinking"))
    record.tools = len(payload.get("tools") or [])
    if payload.get("tool_choice") and payload.get("tool_choice") != "auto":
        record.sampling["tool_choice"] = payload.get("tool_choice")

    def scan(content: Any) -> None:
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "image_url", "input_image", "image", "video_url", "input_video"
                ):
                    record.has_media = True

    if record.protocol == ANTHROPIC:
        msgs = payload.get("messages") or []
        record.prompt_messages = len(msgs) + (1 if payload.get("system") else 0)
        scan(payload.get("system"))
        for msg in msgs:
            if isinstance(msg, dict):
                scan(msg.get("content"))
        total_chars = len(json.dumps(payload.get("messages") or [], ensure_ascii=False))
        record.prompt_chars = total_chars
        record.prompt_tokens_est = est_tokens(
            json.dumps(payload.get("system") or "", ensure_ascii=False)
            + json.dumps(payload.get("messages") or [], ensure_ascii=False)
        )
        return

    if record.protocol == RESPONSES:
        text = payload.get("instructions") or ""
        items = payload.get("input")
        if isinstance(items, str):
            text += items
            record.prompt_messages = 1
        elif isinstance(items, list):
            record.prompt_messages = len(items)
            for item in items:
                if isinstance(item, dict):
                    scan(item.get("content"))
                    for part in item.get("content") or []:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            text += part["text"]
                elif isinstance(item, str):
                    text += item
        record.prompt_chars = len(text)
        record.prompt_tokens_est = est_tokens(text)
        if "max_output_tokens" in payload:
            try:
                record.max_tokens = int(payload["max_output_tokens"])
            except (TypeError, ValueError):
                pass
        return

    if record.protocol == TEXT:
        prompt = payload.get("prompt")
        if isinstance(prompt, list):
            text = "\n".join(str(p) for p in prompt)
            record.prompt_messages = len(prompt)
        else:
            text = str(prompt or "")
            record.prompt_messages = 1
        record.prompt_chars = len(text)
        record.prompt_tokens_est = est_tokens(text)
        return

    msgs = payload.get("messages") or []
    record.prompt_messages = len(msgs)
    text = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        scan(content)
        text.append(_text_of(content))
        if msg.get("tool_calls"):
            text.append(json.dumps(msg["tool_calls"], ensure_ascii=False))
    joined = "\n".join(text)
    record.prompt_chars = len(joined)
    record.prompt_tokens_est = est_tokens(joined)


class TapRegistry:
    """Live in-flight requests."""

    def __init__(self) -> None:
        self.live: dict[str, Record] = {}

    def add(self, record: Record) -> None:
        self.live[record.id] = record

    def drop(self, record: Record) -> None:
        self.live.pop(record.id, None)

    def snapshot(self) -> list[dict]:
        rows = [r.live() for r in self.live.values()]
        rows.sort(key=lambda r: r["started_at"])
        return rows
