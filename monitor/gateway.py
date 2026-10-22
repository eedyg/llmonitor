"""The gateway: owns the public port, proxies to the engine, taps the stream."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .assemble import ChatAssembler
from .compat import chat_to_completion, to_chat_request
from .inflight import Inflight
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
    # Ask the engine for a stream even when the client asked for one JSON body:
    # that is the only way to see per-token progress and the only way to abort
    # (see assemble.py / inflight.py). See config.yaml → gateway.upstream_stream.
    upstream_stream = bool(cfg.root.gateway.upstream_stream)
    default_max_tokens = int(cfg.root.gateway.default_max_tokens or 0)

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

        # ---- request policy: cap, then streaming upstream -------------------
        # 1) A default max_tokens keeps a looping model from generating until it
        #    hits the context limit (a Qwen3 thinking loop did 100k+ tokens and
        #    held the GPU for 26 minutes); 0 = never inject, leave clients alone.
        # 2) A non-streamed request is invisible and unabortable: the engine
        #    consumes the whole generation inside its own process. Rewriting it
        #    to a stream gives the tap every delta and gives the cancel API a
        #    handle to close (vLLM aborts when its generator is closed).
        assembler: ChatAssembler | None = None
        if proto == "openai_chat" and isinstance(payload, dict) and not translate:
            if (default_max_tokens and payload.get("max_tokens") is None
                    and payload.get("max_completion_tokens") is None):
                payload = {**payload, "max_tokens": default_max_tokens}
                upstream_body = json.dumps(payload).encode()
                if record is not None:
                    record.max_tokens = default_max_tokens
            if (upstream_stream and not payload.get("stream") and not stream_body
                    and _can_assemble(payload)):
                chat_body = dict(payload)
                options = chat_body.get("stream_options")
                options = dict(options) if isinstance(options, dict) else {}
                options.setdefault("include_usage", True)
                chat_body["stream"] = True
                chat_body["stream_options"] = options
                payload = chat_body
                upstream_body = json.dumps(chat_body).encode()
                assembler = ChatAssembler()
                if record is not None:
                    # The client still gets a single JSON body: `stream` keeps
                    # the value the client asked for, that is what the UI shows.
                    record.stream = False

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
            tap = StreamTap(record, wire=wire,
                            on_event=assembler.feed_event if assembler is not None else None)

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

        # ---- assembled: streamed upstream, one JSON body for the client ---
        if assembler is not None:
            handle = None
            if record is not None:
                handle = rt.inflight.add(Inflight(id=record.id, abortable=True,
                                                  record=record, upstream=upstream))
            try:
                if upstream.status_code >= 400:
                    raw = await upstream.aread()
                    if record is not None:
                        _finish(record, upstream.status_code, _error_text(raw))
                    rt.gateway["errors"] += 1
                    return Response(content=raw, status_code=upstream.status_code,
                                    headers=out_headers)
                async for chunk in upstream.aiter_raw():
                    if tap is not None:
                        tap.feed(chunk)
                raw = json.dumps(assembler.result(), ensure_ascii=False).encode()
                rt.gateway["bytes_out"] += len(raw)
                out_headers["content-type"] = "application/json"
                if record is not None:
                    out_headers["x-monitor-request-id"] = record.id
                    _finish(record, 200)
                return Response(content=raw, status_code=upstream.status_code, headers=out_headers)
            except Exception as exc:
                cancelled = handle is not None and handle.cancelled
                reason = (handle.reason if handle is not None and handle.reason
                          else f"upstream died mid-stream: {type(exc).__name__}: {exc}")
                if record is not None:
                    _finish(record, 499 if cancelled else 502, reason)
                rt.gateway["errors"] += 1
                return JSONResponse(
                    status_code=499 if cancelled else 502,
                    content={"error": {
                        "message": reason,
                        "type": "monitor_cancelled" if cancelled else "monitor_upstream_error",
                    }},
                )
            finally:
                if record is not None:
                    rt.inflight.drop(record.id)
                    if record.finished_at is None:
                        # The client vanished mid-generation (or we were
                        # cancelled): finish the bookkeeping anyway, otherwise
                        # the record leaks in the live list and in_flight never
                        # drops. aclose() below still aborts the engine.
                        _finish(record, 499, "client disconnected")
                await upstream.aclose()

        # ---- non-streaming: buffer, tap, return ---------------------------
        if not is_sse:
            if record is not None:
                # Registered so /cancel can answer honestly: the engine has
                # already swallowed this generation, nothing here can stop it.
                rt.inflight.add(Inflight(id=record.id, abortable=False,
                                         record=record, upstream=upstream))
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
                if record is not None:
                    rt.inflight.drop(record.id)
                    if record.finished_at is None:
                        _finish(record, 499, "client disconnected")
                await upstream.aclose()

        # ---- streaming ----------------------------------------------------
        live = None
        if record is not None:
            live = rt.inflight.add(Inflight(id=record.id, abortable=True,
                                           record=record, upstream=upstream))

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
                if live is not None and live.cancelled:
                    # A dropped connection looks like a clean finish to an SSE
                    # client, so say it out loud before the stream ends.
                    out = json.dumps({"error": {
                        "message": live.reason or "cancelled from the monitor",
                        "type": "monitor_cancelled",
                    }}, ensure_ascii=False)
                    chunk = f"data: {out}\n\n".encode()
                    rt.gateway["bytes_out"] += len(chunk)
                    yield chunk
            finally:
                if record is not None:
                    rt.inflight.drop(record.id)
                await upstream.aclose()
                if record is not None:
                    if live is not None and live.cancelled:
                        _finish(record, 499, live.reason or "cancelled from the monitor")
                    elif error and ("Cancelled" in error or "GeneratorExit" in error):
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


def _can_assemble(payload: dict) -> bool:
    """Only the plain single-choice text path may be rewritten to a stream.

    `n > 1` and `logprobs` shape the wire response in ways that are easy to get
    subtly wrong, so those clients keep the old buffer-and-forward path. They
    also stay unabortable, which /api/requests/{id}/cancel reports honestly.
    """
    try:
        if int(payload.get("n") or 1) > 1:
            return False
    except (TypeError, ValueError):
        return False
    return not (payload.get("logprobs") or payload.get("top_logprobs"))


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
