"""The gateway asks the engine for a stream even when the client did not.

Non-streamed requests used to be invisible (no deltas for the tap, so the UI
showed "queued / 0 tokens" for 26 minutes) and unabortable (the engine owns the
whole generation). These tests pin the rewrite, the reassembly, and the handles
that make cancel possible.

Note on the fake engine: httpx's MockTransport reads the whole body even with
`stream=True` (it then raises StreamConsumed), which would hide the very path we
test, so the fake transport below streams the body in small chunks instead.
"""
import asyncio
import json

import httpx
import pytest

from monitor.config import Config
from monitor.gateway import build_gateway
from monitor.inflight import Inflight, InflightRegistry
from monitor.runtime import Runtime


@pytest.fixture()
def rt(tmp_path):
    cfg = Config(str(tmp_path / "config.yaml"))
    cfg.data["paths"]["run_dir"] = "run"
    cfg.data["paths"]["state_dir"] = "state"
    (tmp_path / "run").mkdir(exist_ok=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    yield Runtime(cfg)


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def __aiter__(self):
        for i in range(0, len(self.data), 23):
            yield self.data[i:i + 23]


class _FakeEngine(httpx.AsyncBaseTransport):
    def __init__(self, handler) -> None:
        self.handler = handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        status, headers, data = self.handler(request)
        return httpx.Response(status, headers=headers,
                              stream=_ChunkedStream(data), request=request)


def _sse(events) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return (body + "data: [DONE]\n\n").encode()


def _chunk(**over):
    base = {"id": "chatcmpl-9", "object": "chat.completion.chunk", "created": 1792500000,
            "model": "qwen3.8-27b", "choices": []}
    base.update(over)
    return base


def _engine(seen: dict, sse_events=None, json_body=None):
    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        if sse_events is not None:
            return 200, {"content-type": "text/event-stream"}, _sse(sse_events)
        return 200, {"content-type": "application/json"}, json.dumps(json_body).encode()
    return _FakeEngine(handler)


def _call(app, payload, path="/v1/chat/completions"):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
            return await client.post(path, json=payload)
    return asyncio.run(go())


def test_non_stream_request_is_rewritten_and_reassembled(rt):
    seen = {}
    events = [
        _chunk(choices=[{"index": 0, "delta": {"role": "assistant"}}]),
        _chunk(choices=[{"index": 0, "delta": {"reasoning_content": "想一想"}}]),
        _chunk(choices=[{"index": 0, "delta": {"content": "42"}, "finish_reason": "stop"}]),
        _chunk(choices=[], usage={"prompt_tokens": 9, "completion_tokens": 3,
                                  "total_tokens": 12}),
    ]
    app = build_gateway(rt)
    app.state.client = httpx.AsyncClient(transport=_engine(seen, sse_events=events))
    resp = _call(app, {"model": "qwen3.8-27b", "stream": False,
                       "messages": [{"role": "user", "content": "hi"}]})

    # upstream was asked for a stream: that is what makes the request visible
    # to the tap and cancellable.
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"]["include_usage"] is True

    # the client still gets exactly one non-streaming body
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "42"
    assert body["choices"][0]["message"]["reasoning_content"] == "想一想"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 3

    # ... and the tapped record carries real progress numbers instead of zeros
    record = rt.store.get(resp.headers["x-monitor-request-id"])
    assert record["status"] == 200 and record["finish_reason"] == "stop"
    assert record["thinking_tokens"] == 1   # delta fallback (usage had no reasoning_tokens)
    assert record["content_tokens"] == 3   # usage.completion_tokens, thinking not split out
    assert record["prompt_tokens"] == 9
    assert record["ttft_ms"] is not None


def test_multi_choice_is_left_on_the_buffered_path(rt):
    seen = {}
    app = build_gateway(rt)
    app.state.client = httpx.AsyncClient(transport=_engine(seen, json_body={
        "id": "chatcmpl-2", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "a"},
                     "finish_reason": "stop"},
                    {"index": 1, "message": {"role": "assistant", "content": "b"},
                     "finish_reason": "stop"}],
    }))
    resp = _call(app, {"model": "m", "n": 2, "messages": [{"role": "user", "content": "hi"}]})
    assert "stream" not in seen["body"]          # untouched
    assert [c["message"]["content"] for c in resp.json()["choices"]] == ["a", "b"]


def test_default_max_tokens_is_injected_when_the_client_omits_it(rt):
    rt.cfg.data["gateway"]["default_max_tokens"] = 1234
    seen = {}
    app = build_gateway(rt)
    app.state.client = httpx.AsyncClient(transport=_engine(seen, sse_events=[
        _chunk(choices=[{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}])]))
    resp = _call(app, {"model": "m", "stream": False,
                       "messages": [{"role": "user", "content": "hi"}]})
    assert seen["body"]["max_tokens"] == 1234
    assert resp.json()["choices"][0]["message"]["content"] == "x"


def test_client_supplied_max_tokens_wins(rt):
    rt.cfg.data["gateway"]["default_max_tokens"] = 1234
    seen = {}
    app = build_gateway(rt)
    app.state.client = httpx.AsyncClient(transport=_engine(seen, sse_events=[
        _chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])]))
    _call(app, {"model": "m", "stream": False, "max_tokens": 77,
                "messages": [{"role": "user", "content": "hi"}]})
    assert seen["body"]["max_tokens"] == 77


class _FakeUpstream:
    """Stands in for httpx's streaming response: cancel just closes it."""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def test_cancel_closes_the_upstream_and_marks_the_record():
    registry = InflightRegistry()
    upstream = _FakeUpstream()
    handle = registry.add(Inflight(id="mreq_x", abortable=True, upstream=upstream))
    result = asyncio.run(registry.cancel("mreq_x", "runaway"))
    assert result == {"ok": True, "found": True, "id": "mreq_x", "aborted": True,
                      "engine_aborted": True, "abortable": True, "reason": "runaway",
                      "note": ""}
    assert upstream.closed and handle.cancelled


def test_cancel_reports_honestly_when_the_upstream_is_not_streaming():
    registry = InflightRegistry()
    registry.add(Inflight(id="mreq_y", abortable=False, upstream=_FakeUpstream()))
    result = asyncio.run(registry.cancel("mreq_y"))
    assert result["found"] is True and result["aborted"] is False
    assert "not streaming" in result["note"]


def test_cancel_unknown_id():
    result = asyncio.run(InflightRegistry().cancel("mreq_nope"))
    assert result["found"] is False and result["ok"] is False


def _get_or_post(app, path, method="post"):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ui") as client:
            return await getattr(client, method)(path)
    return asyncio.run(go())


def test_cancel_endpoint_contract(rt):
    from monitor.api import build_api

    app = build_api(rt)
    assert _get_or_post(app, "/api/requests/mreq_nope/cancel").status_code == 404

    # an engine-side generation we cannot reach: say so, do not pretend
    rt.inflight.add(Inflight(id="mreq_blocked", abortable=False, upstream=_FakeUpstream()))
    blocked = _get_or_post(app, "/api/requests/mreq_blocked/cancel")
    assert blocked.status_code == 409
    assert "not streaming" in blocked.json()["detail"]

    upstream = _FakeUpstream()
    rt.inflight.add(Inflight(id="mreq_ok", abortable=True, upstream=upstream))
    ok = _get_or_post(app, "/api/requests/mreq_ok/cancel")
    assert ok.status_code == 200
    assert ok.json()["engine_aborted"] is True
    assert upstream.closed


def test_live_view_marks_what_can_be_cancelled(rt):
    from monitor.api import build_api
    from monitor.tap import Record

    record = Record(endpoint="/v1/chat/completions", engine="vllm")
    rt.taps.add(record)
    app = build_api(rt)

    live = _get_or_post(app, "/api/requests/live", method="get").json()["items"]
    assert live[0]["cancellable"] is False        # no streaming upstream handle

    rt.inflight.add(Inflight(id=record.id, abortable=True, record=record,
                             upstream=_FakeUpstream()))
    live = _get_or_post(app, "/api/requests/live", method="get").json()["items"]
    assert live[0]["cancellable"] is True
