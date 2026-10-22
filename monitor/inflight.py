"""In-flight request handles: the only place a request can be aborted.

Why this exists: the gateway used to keep nothing but a counter, so a request
that had run away (a Qwen3 thinking loop can generate six figures of tokens with
nobody watching) could only be stopped by restarting the whole engine.

The abort lever is the upstream connection. vLLM's ``AsyncLLM.generate`` aborts
a request when its output generator is closed — see the comment in
``vllm/v1/engine/async_llm.py``: "If the request is disconnected by the client,
generate() is cancelled or the generator is garbage collected. So, we abort the
request if we end up here." Closing a *streaming* upstream response does exactly
that. A non-streaming upstream response cannot be aborted: the engine consumes
the whole generation internally, so we report that honestly instead of pretending.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Inflight:
    """One request the gateway is currently relaying."""

    id: str
    abortable: bool = False
    record: Any = None
    upstream: Any = None
    started_at: float = field(default_factory=time.time)
    cancelled: bool = False
    cancelled_at: float | None = None
    reason: str | None = None

    async def cancel(self, reason: str = "cancelled from the monitor") -> bool:
        """Abort the engine-side generation. Returns True if that really happened."""
        if self.cancelled:
            return self.abortable and self.upstream is not None
        self.cancelled = True
        self.cancelled_at = time.time()
        self.reason = reason
        if self.record is not None:
            self.record.finish_reason = "cancelled"
        if not self.abortable or self.upstream is None:
            return False
        try:
            # Closing the response ends the SSE iteration in the relay, which
            # drops the connection; the engine then aborts the request.
            await self.upstream.aclose()
        except Exception:  # pragma: no cover - already closed/gone
            pass
        return True

    def status(self) -> dict:
        record = self.record
        return {
            "id": self.id,
            "abortable": self.abortable,
            "cancelled": self.cancelled,
            "reason": self.reason,
            "elapsed_ms": round((time.time() - self.started_at) * 1000.0, 2),
            "phase": getattr(record, "phase", None),
            "thinking_tokens": getattr(record, "thinking_chunks", 0) or getattr(record, "thinking_tokens", 0),
            "content_tokens": getattr(record, "content_chunks", 0) or getattr(record, "content_tokens", 0),
            "model": getattr(record, "model", None),
            "client_ip": getattr(record, "client_ip", None),
            "client_ua": getattr(record, "client_ua", None),
        }


class InflightRegistry:
    """Live handles, keyed by monitor request id (`mreq_...`)."""

    def __init__(self) -> None:
        self._items: dict[str, Inflight] = {}

    def __len__(self) -> int:
        return len(self._items)

    def add(self, handle: Inflight) -> Inflight:
        self._items[handle.id] = handle
        return handle

    def get(self, request_id: str) -> Inflight | None:
        return self._items.get(request_id)

    def drop(self, request_id: str) -> None:
        self._items.pop(request_id, None)

    def snapshot(self) -> list[dict]:
        return [item.status() for item in list(self._items.values())]

    async def cancel(self, request_id: str, reason: str = "cancelled from the monitor") -> dict:
        handle = self._items.get(request_id)
        if handle is None:
            return {"ok": False, "found": False, "id": request_id,
                    "error": "unknown or already finished request id"}
        aborted = await handle.cancel(reason)
        return {
            "ok": True,
            "found": True,
            "id": request_id,
            "aborted": aborted,
            "engine_aborted": aborted,
            "abortable": handle.abortable,
            "reason": reason,
            "note": "" if aborted else
                    "upstream is not streaming for this request; the engine keeps generating "
                    "(only an engine restart can stop it)",
        }
