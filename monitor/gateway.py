"""The gateway: owns the public port, proxies to the engine, taps the stream."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .compat import chat_to_completion, to_chat_request
from .runtime import Runtime
from .tap import CHAT, PROTOCOL_BY_PATH, Record, StreamTap, request_meta

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}

METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")


def build_gateway(rt: Runtime) -> FastAPI:
    cfg = rt.cfg
    app = FastAPI(title="llm-monitor gateway", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.rt = rt
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=float(cfg.root.gateway.connect_timeout_s), read=None,
                              write=None, pool=None),
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        follow_redirects=False,
    )
    base = f"http://127.0.0.1:{int(cfg.root.server.engine_port)}"
    max_capture = int(cfg.root.gateway.max_capture_bytes)
    force_usage = bool(cfg.root.gateway.force_usage_chunk)

    def _finish(record: Record, status: int | None, error: str | None = None) -> None:
        if record.finished_at is not None:
            return
        record.finished_at = time.time()
        record.status = status
        if error:
            record.error = error
        if record.error:
            record.phase = "error"
        else:
            record.phase = "done"
        # token fallbacks
        if record.thinking_tokens == 0 and record.thinking_chunks:
            record.thinking_tokens = record.thinking_chunks
        if record.content_tokens == 0 and record.content_chunks:
            record.content_tokens = record.content_chunks
        if record.prompt_tokens is None:
            record.prompt_tokens = record.prompt_tokens_est or None
        rt.taps.drop(record)
        try:
            rt.store.add(record)
        except Exception as exc:  # pragma: no cover - never fail a request for this
            rt.publish({"type": "tap_error", "error": str(exc)})
        rt.publish({"type": "request_done", "request": record.public()})
        rt.gateway["in_flight"] = max(0, rt.gateway["in_flight"] - 1)

    @app.api_route("/{path:path}", methods=METHODS)
    async def proxy(path: str, request: Request) -> Response:
        url_path = "/" + path
        declared = int(request.headers.get("content-length") or 0)
        # Very large bodies (base64 media) are streamed through, not buffered:
        # we keep the byte count but skip deep request parsing for them.
        stream_body = bool(declared and declared > max_capture)
        body = b"" if stream_body else await request.body()
        rt.gateway["requests"] += 1
        rt.gateway["bytes_in"] += len(body) if not stream_body else declared

        proto = PROTOCOL_BY_PATH.get(url_path)
        if request.headers.get("x-monitor-probe"):
            proto = None
        record: Record | None = None
        payload: Any = None
        if proto:
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = None
            record = Record(
                endpoint=url_path,
                protocol=proto,
                engine=rt.engine.kind or (rt.controller.kind if rt.controller else None),
                method=request.method,
                started_iso=_iso(time.time()),
                client_ip=request.client.host if request.client else None,
                client_ua=request.headers.get("user-agent"),
                request_bytes=len(body) if not stream_body else declared,
            )
            if isinstance(payload, dict):
                request_meta(record, payload)
            rt.taps.add(record)
            rt.gateway["in_flight"] += 1

        # ---- legacy completions -> chat (NInfer only) ---------------------
        upstream_path = url_path
        upstream_body = body
        wire = proto
        prompt = ""
        echo = False
        translate = False
        live_kind = rt.engine.kind or (rt.controller.kind if rt.controller else None)
        if proto == "openai_text" and live_kind == "ninfer" and isinstance(payload, dict):
            chat, prompt = to_chat_request(payload)
            echo = bool(payload.get("echo"))
            if record is not None:
                record.protocol = "openai_text"
                record.stream = bool(chat.get("stream"))
                record.max_tokens = chat.get("max_tokens")
                record.prompt_tokens_est = record.prompt_tokens_est or 0
                record.sampling = {
                    k: v for k, v in chat.items()
                    if k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "seed")
                }
            upstream_path = "/v1/chat/completions"
            upstream_body = json.dumps(chat).encode()
            wire = CHAT
            translate = True

        # ---- outbound headers ---------------------------------------------
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["accept-encoding"] = "identity"
        api_key = rt.engine_api_key or (cfg.root.engine.extra_env or {}).get("API_KEY")
        if api_key and "authorization" not in {k.lower() for k in headers}:
            headers["authorization"] = f"Bearer {api_key}"

        if force_usage and proto in ("openai_chat", "openai_text"):
            if translate and isinstance(payload, dict) and payload.get("stream"):
                chat_body = json.loads(upstream_body)
                options = chat_body.get("stream_options")
                if not isinstance(options, dict):
                    options = {}
                options.setdefault("include_usage", True)
                chat_body["stream_options"] = options
                upstream_body = json.dumps(chat_body).encode()
            elif isinstance(payload, dict) and payload.get("stream"):
                options = payload.get("stream_options")
                if not isinstance(options, dict):
                    options = {}
                options.setdefault("include_usage", True)
                payload["stream_options"] = options
                upstream_body = json.dumps(payload).encode()

        query = request.url.query
        url = f"{base}{upstream_path}" + (f"?{query}" if query else "")

        tap: StreamTap | None = None
        if record is not None:
            tap = StreamTap(record, wire=wire)

        client: httpx.AsyncClient = app.state.client
        try:
                upstream = await client.send(
                client.build_request(
                    request.method,
                    url,
                    headers=headers,
                    content=request.stream() if stream_body else upstream_body,
                ),
                stream=True,
            )
        except Exception as exc:
            if record is not None:
                _finish(record, 502, f"upstream unreachable: {type(exc).__name__}: {exc}")
            rt.gateway["errors"] += 1
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"engine unreachable at {base} ({type(exc).__name__}: {exc})."
                        " Is the engine running?",
                        "type": "monitor_gateway_error",
                    }
                },
            )

        out_headers = {
            k: v for k, v in upstream.headers.items()
            if k.lower() not in ("content-length", "content-encoding", "transfer-encoding", "connection")
        }
        content_type = (upstream.headers.get("content-type") or "").lower()
        is_sse = "text/event-stream" in content_type

        if record is not None:
            record.engine_request_id = upstream.headers.get("x-request-id") or record.engine_request_id
        if record is not None and record.status is None:
            record.status = upstream.status_code
            if record.phase == "queued":
                record.phase = "prefill" if upstream.status_code < 400 else "error"
        if record is not None and upstream.status_code >= 400:
            out_headers["x-monitor-request-id"] = record.id

        # ---- non-streaming: buffer, tap, return ---------------------------
        if not is_sse:
            try:
                if proto is None:
                    raw = b""
                    chunks: list[bytes] = []
                    async for chunk in upstream.aiter_raw():
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                else:
                    chunks = []
                    total = 0
                    async for chunk in upstream.aiter_raw():
                        chunks.append(chunk)
                        total += len(chunk)
                        if total > max_capture:
                            break
                    raw = b"".join(chunks)
                if tap is not None and upstream.status_code < 400:
                    tap.feed_body(raw)
                if translate and isinstance(payload, dict) and upstream.status_code < 400:
                    try:
                        data = json.loads(raw or b"{}")
                        raw = json.dumps(
                            chat_to_completion(data, prompt, echo), ensure_ascii=False
                        ).encode()
                        out_headers["content-type"] = "application/json"
                    except json.JSONDecodeError:
                        pass
                if record is not None:
                    if upstream.status_code >= 400:
                        _finish(record, upstream.status_code, _error_text(raw))
                    else:
                        _finish(record, upstream.status_code)
                rt.gateway["bytes_out"] += len(raw)
                status = upstream.status_code
                if record is not None:
                    out_headers["x-monitor-request-id"] = record.id
                return Response(content=raw, status_code=status, headers=out_headers)
            finally:
                await upstream.aclose()

        # ---- streaming ----------------------------------------------------
        async def relay():
            error: str | None = None
            try:
                if translate and isinstance(payload, dict):
                    from .compat import CompletionStream

                    state = CompletionStream(prompt, echo, payload.get("model"))
                    head = state.start()
                    if head:
                        yield head
                    async for chunk in upstream.aiter_raw():
                        if tap is not None:
                            tap.feed(chunk)
                        out = state.feed(chunk)
                        if out:
                            rt.gateway["bytes_out"] += len(out)
                            yield out
                    rest = state.feed(b"\n\n")
                    if rest:
                        yield rest
                    yield state.end()
                else:
                    async for chunk in upstream.aiter_raw():
                        if tap is not None:
                            tap.feed(chunk)
                        rt.gateway["bytes_out"] += len(chunk)
                        yield chunk
            except Exception as exc:  # client gone or upstream died mid-stream
                error = f"{type(exc).__name__}: {exc}"
            finally:
                await upstream.aclose()
                if record is not None:
                    if error and ("Cancelled" in error or "GeneratorExit" in error):
                        _finish(record, upstream.status_code, "client disconnected")
                    else:
                        _finish(record, upstream.status_code, error)

        if record is not None:
            out_headers["x-monitor-request-id"] = record.id
        return StreamingResponse(relay(), status_code=upstream.status_code, headers=out_headers)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.client.aclose()

    return app


def _error_text(raw: bytes) -> str:
    try:
        data = json.loads(raw or b"{}")
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:500]
        if isinstance(err, str):
            return err[:500]
        return json.dumps(data, ensure_ascii=False)[:500]
    except Exception:
        return (raw[:300].decode("utf-8", "replace") or "upstream error")
