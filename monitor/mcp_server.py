"""MCP server: every control/observation the web UI has, exposed as tools.

Transports:
    <UI process>                                      # /mcp on the control port (mount_http)
    python -m monitor.mcp_server                      # stdio (pi, claude code, …)
    python -m monitor.mcp_server --transport http     # standalone streamable HTTP on :9001/mcp

The UI process serves the MCP endpoint itself (http://<ui_host>:9000/mcp), the
same way most agent products do: one port, no extra daemon to babysit. The
standalone HTTP/SSE transports stay for clients that cannot reach the UI or
cannot spawn a subprocess.

It is a thin client of the monitor REST API, so the UI and the MCP tools can
never drift apart.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from typing import Any, AsyncIterator

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

DEFAULT_URL = os.environ.get("MONITOR_URL", "http://127.0.0.1:9000")
TOKEN = os.environ.get("MONITOR_TOKEN", "")

mcp = FastMCP(
    "llm-monitor",
    instructions=(
        "Control and observe a local LLM serving stack (vLLM or NInfer) behind the "
        "llm-monitor gateway. Use llm_status first; write operations (start/stop/"
        "restart/switch) require confirm=true and return a preview otherwise."
    ),
)

_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        headers = {"x-monitor-token": TOKEN} if TOKEN else {}
        _client = httpx.Client(base_url=DEFAULT_URL, timeout=120.0, headers=headers)
    return _client


def mount_http(app: Any, *, base_url: str | None = None, token: str | None = None,
               path: str = "/mcp") -> None:
    """Serve the MCP endpoint from an existing FastAPI/Starlette app.

    Keeps the one-port layout: the control plane answers the UI on `/`, the REST
    API on `/api/*` and MCP on `/mcp`, so no extra process, no extra port, and
    the UI's auth middleware covers MCP too.

    Stateless is deliberate: the tools are pure REST calls, so there is nothing
    to keep per session, and a long-lived session id would break as soon as the
    UI process restarts.

    The caller must run `http_lifespan` in its own lifespan; Starlette does not
    propagate lifespans to mounted apps.
    """
    global DEFAULT_URL, TOKEN, _client
    if base_url:
        DEFAULT_URL = base_url.rstrip("/")
        _client = None
    if token is not None:
        TOKEN = token
        _client = None
    mcp.settings.streamable_http_path = path
    mcp.settings.stateless_http = True
    # FastMCP pins the Host header to loopback when it binds 127.0.0.1, but this
    # endpoint lives on the UI port, which is served on 0.0.0.0 and reached by
    # LAN IP / hostname — that pin would answer every real client with 421.
    # The UI's token middleware is the guard here, as it is for the REST API.
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False)
    sub = mcp.streamable_http_app()
    # Copy the route instead of app.mount(): a Mount answers /mcp with a 307 to
    # /mcp/, and a POST that gets redirected once per request is just noise.
    app.router.routes.extend(sub.routes)
    app.state.mcp_session_manager = mcp.session_manager


@contextlib.asynccontextmanager
async def http_lifespan(app: Any = None) -> AsyncIterator[None]:
    """Lifespan for an app that mounted the MCP route via `mount_http`."""
    manager = getattr(getattr(app, "state", None), "mcp_session_manager", None)
    if manager is None:
        yield
        return
    async with manager.run():
        yield


def _get(path: str, **params: Any) -> Any:
    resp = client().get(path, params={k: v for k, v in params.items() if v is not None})
    resp.raise_for_status()
    return resp.json()


def _post(path: str, payload: dict | None = None) -> Any:
    resp = client().post(path, json=payload or {})
    resp.raise_for_status()
    return resp.json()


def _compact(snap: dict) -> dict:
    engine = snap.get("engine") or {}
    gpu = ((snap.get("gpu") or {}).get("gpus") or [{}])[0]
    ctrl = snap.get("controller") or {}
    history = (snap.get("tokens") or {}).get("history") or {}
    session = (snap.get("tokens") or {}).get("session") or {}
    money = ((snap.get("billing") or {}).get("history") or {})
    ds = money.get("deepseek") or {}
    power = money.get("energy") or {}
    return {
        "state": ctrl.get("state"),
        "engine": ctrl.get("engine"),
        "detail": ctrl.get("detail"),
        "model": (engine.get("model") or {}).get("served"),
        "requests": engine.get("requests"),
        "tokens": engine.get("tokens"),
        "cache": {k: v for k, v in (engine.get("cache") or {}).items() if k != "pools"},
        "speed": engine.get("speed"),
        "spec": engine.get("spec"),
        "gpu": {
            "name": gpu.get("name"),
            "util_pct": gpu.get("util_pct"),
            "mem_used_mib": gpu.get("mem_used_mib"),
            "mem_total_mib": gpu.get("mem_total_mib"),
            "power_w": gpu.get("power_w"),
            "temp_c": gpu.get("temp_c"),
            "fan_pct": gpu.get("fan_pct"),
            "throttle": gpu.get("throttle"),
        },
        "live": len(snap.get("live") or []),
        # Cumulative request ledger (SQLite) + equivalent money. The numbers in
        # "tokens" above are the *engine's* counters, which reset with the
        # engine process; this block covers the requests table and survives it.
        "ledger": {
            "requests": history.get("requests"),
            "total_tokens": history.get("total"),
            "prompt": history.get("prompt"),
            "prompt_cached": history.get("prompt_cached"),
            "prompt_miss": history.get("prompt_miss"),
            "output": history.get("output"),
            "thinking": history.get("thinking"),
            "content": history.get("content"),
            "cache_hit_pct": history.get("cache_hit_pct"),
            "cache_estimated": history.get("cache_estimated"),
            "first_at": history.get("first_at"),
            "last_at": history.get("last_at"),
            "session_total_tokens": session.get("total"),
            "deepseek_cny": ds.get("cost"),
            "energy_kwh": power.get("kwh"),
            "electricity_cny": power.get("cost"),
        },
    }


# ── read tools ───────────────────────────────────────────────────────────────
@mcp.tool()
def llm_status() -> dict:
    """Current engine state, model, request counts, cache, speed, spec-decode and GPU summary."""
    return _compact(_get("/api/state"))


@mcp.tool()
def llm_metrics(section: str = "all") -> dict:
    """Full normalized metrics snapshot.

    section: all|requests|tokens|cache|speed|spec|config (engine counters, reset
    with the engine process) or ledger|billing (request table + money, durable).
    """
    snap = _get("/api/state")
    engine = snap.get("engine") or {}
    if section == "all":
        return {
            "engine": engine,
            "gpu": (snap.get("gpu") or {}).get("gpus"),
            "gateway": snap.get("gateway"),
            "summary": snap.get("summary"),
            "ledger": snap.get("tokens"),
            "billing": snap.get("billing"),
        }
    if section == "ledger":
        return {"ledger": snap.get("tokens")}
    if section == "billing":
        return {"billing": snap.get("billing")}
    if section in engine:
        return {section: engine.get(section)}
    return {"error": f"unknown section '{section}'",
            "available": list(engine.keys()) + ["ledger", "billing"]}


@mcp.tool()
def llm_ledger(start: str | None = None, end: str | None = None) -> dict:
    """Cumulative token ledger from the request table (survives engine restarts).

    No arguments: this process (`session`) + everything retained in SQLite
    (`history`), each split into prompt / cached / miss / output / thinking /
    content. With `start`/`end` (YYYY-MM-DD, local days, both inclusive) it
    returns that window's aggregate plus one row per day.
    """
    if start or end:
        data = _get("/api/tokens", start=start, end=end)
        daily = [
            {
                "date": row.get("date"),
                "requests": row.get("requests"),
                "total": row.get("total"),
                "prompt": row.get("prompt"),
                "prompt_cached": row.get("prompt_cached"),
                "prompt_miss": row.get("prompt_miss"),
                "output": row.get("output"),
                "thinking": row.get("thinking"),
                "content": row.get("content"),
                "deepseek_cny": ((row.get("billing") or {}).get("deepseek") or {}).get("cost"),
                "energy_kwh": ((row.get("billing") or {}).get("energy") or {}).get("kwh"),
                "electricity_cny": ((row.get("billing") or {}).get("energy") or {}).get("cost"),
            }
            for row in data.get("daily") or []
        ]
        return {
            "start": data.get("start"),
            "end": data.get("end"),
            "days": data.get("days"),
            "scope": data.get("scope"),
            "billing": (data.get("billing") or {}).get("scope"),
            "daily": daily,
        }
    snap = _get("/api/state")
    tokens = snap.get("tokens") or {}
    return {
        "session": tokens.get("session"),
        "history": tokens.get("history"),
        "billing": (snap.get("billing") or {}).get("session"),
        "billing_history": (snap.get("billing") or {}).get("history"),
    }


@mcp.tool()
def llm_cost(start: str | None = None, end: str | None = None) -> dict:
    """What the recorded tokens would cost at the deepseek-flash rate card, plus GPU electricity.

    Two numbers with different confidence, both estimates: the API figure uses
    the official CNY rate card with peak/off-peak split by request timestamp;
    electricity integrates the GPU's own power samples (no CPU/PSU losses).
    Prices come from `billing.*` in config.yaml.
    """
    if start or end:
        data = _get("/api/tokens", start=start, end=end)
        return {
            "start": data.get("start"),
            "end": data.get("end"),
            "days": data.get("days"),
            "price": (data.get("billing") or {}).get("price"),
            "cost": (data.get("billing") or {}).get("scope"),
            "per_day": [
                {
                    "date": row.get("date"),
                    "deepseek_cny": ((row.get("billing") or {}).get("deepseek") or {}).get("cost"),
                    "energy_cny": ((row.get("billing") or {}).get("energy") or {}).get("cost"),
                    "total_cny": ((row.get("billing") or {}).get("scope") or {}).get("total"),
                }
                for row in data.get("daily") or []
            ],
        }
    billing = _get("/api/state").get("billing") or {}
    return {
        "price": billing.get("price"),
        "session": billing.get("session"),
        "history": billing.get("history"),
    }


@mcp.tool()
def llm_gpu() -> dict:
    """GPU telemetry: utilization, memory, power, temperature, fan, clocks, throttle reasons, per-process memory."""
    snap = _get("/api/state")
    return snap.get("gpu") or {}


@mcp.tool()
def llm_live_requests() -> dict:
    """Requests currently in flight, with phase (queued/prefill/decode), tokens so far and output tails."""
    return _get("/api/requests/live")


@mcp.tool()
def llm_cancel_request(request_id: str, confirm: bool = False) -> dict:
    """Abort one in-flight request (closes the upstream stream; the engine aborts it).

    Use llm_live_requests for the ids. Requests whose upstream is not streaming
    cannot be aborted - the API answers 409 and the engine keeps generating.
    """
    if not confirm:
        return {"hint": "call again with confirm=true to abort this request",
                "request_id": request_id}
    return _post(f"/api/requests/{request_id}/cancel")


@mcp.tool()
def llm_requests(limit: int = 20, engine: str | None = None, status: str | None = None,
                 search: str | None = None) -> dict:
    """Recent request history (metadata only). status: ok|error."""
    return _get("/api/requests", limit=limit, engine=engine, status=status, search=search)


@mcp.tool()
def llm_request(request_id: str) -> dict:
    """Full detail for one request: timings, token split, thinking/content text, sampling, engine raw record."""
    return _get(f"/api/requests/{request_id}")


@mcp.tool()
def llm_series(window_s: int = 3600) -> dict:
    """Time series (1 Hz) for throughput, latency, cache, GPU power/temperature over the last window_s seconds."""
    return _get("/api/series", window=window_s)


@mcp.tool()
def llm_logs(tail: int = 200) -> dict:
    """Tail of the engine log (docker logs for vLLM, run/*.log for NInfer)."""
    return _get("/api/logs", tail=tail)


@mcp.tool()
def llm_probe(extended: bool = False) -> dict:
    """Transparency-contract probe: /health, /v1/models, /v1/chat/completions.

    With extended=true it also sends /v1/responses and /v1/messages. Those are
    real requests: on some engines (NInfer in this environment) they can trip an
    engine-side bug and leave it "unavailable", so they are opt-in.
    """
    return _post("/api/engine/probe", {"extended": extended})


@mcp.tool()
def llm_dry_run(engine: str = "vllm", profile: str = "stage1", ctx: str | None = None,
                kv_capacity: str | None = None, spec_tokens: str | None = None) -> dict:
    """Show the exact launch plan the monitor would execute, without running it.

    Returns the resolved argv, mounts, env, labels, warnings and preflight errors
    (plus `plan_source`).
    """
    return _get("/api/engine/plan", engine=engine, profile=profile, ctx=ctx,
                kv_capacity=kv_capacity, spec_tokens=spec_tokens)


# ── docker (vLLM scope only) ─────────────────────────────────────────────────
@mcp.tool()
def llm_docker_status() -> dict:
    """Docker resources in the vLLM scope: containers, images (with removable/blocked reason), disk use.

    Reads enumerate every container/image so we can refuse unsafe deletions, but
    every write tool can only touch resources under the configured scope
    (default: repo vllm/vllm-openai, container vllm-qwen38).
    """
    return _get("/api/docker/summary")


@mcp.tool()
def llm_docker_pull(ref: str = "vllm/vllm-openai:cu130-sm120", confirm: bool = False) -> dict:
    """Pull/refresh a vLLM image. Returns a job id; poll it with llm_docker_job.

    Also acts as an update check: if the tag moved, `changed` becomes true and the
    previous image id is recorded as superseded (and therefore removable).
    """
    if not confirm:
        return {"hint": "call again with confirm=true to pull", "ref": ref}
    return _post("/api/docker/pull", {"ref": ref})


@mcp.tool()
def llm_docker_job(job_id: str, lines: int = 80) -> dict:
    """Progress + log tail of a docker pull job."""
    return _get(f"/api/docker/jobs/{job_id}", lines=lines)


@mcp.tool()
def llm_docker_remove_image(image_id: str, confirm: bool = False) -> dict:
    """Remove one image (id or sha256:…). Refuses if any container uses it, if it has
    out-of-scope tags only, or if it is not in the vLLM ledger."""
    if not confirm:
        return _post("/api/docker/images/remove", {"id": image_id})
    return _post("/api/docker/images/remove", {"id": image_id, "confirm": True})


@mcp.tool()
def llm_docker_remove_container(name: str = "vllm-qwen38", confirm: bool = False) -> dict:
    """Remove the stopped vLLM container. Running containers are refused."""
    if not confirm:
        return _post("/api/docker/containers/remove", {"name": name})
    return _post("/api/docker/containers/remove", {"name": name, "confirm": True})


@mcp.tool()
def llm_docker_prune(confirm: bool = False) -> dict:
    """Remove exactly the images reported as removable in the vLLM scope.

    Never calls `docker image prune`/`system prune`; each image is re-validated
    individually before `docker rmi`.
    """
    if not confirm:
        return _post("/api/docker/prune", {})
    return _post("/api/docker/prune", {"confirm": True})


@mcp.tool()
def llm_docker_restart_with(ref: str, profile: str = "stage1", ctx: str | None = None,
                            confirm: bool = False) -> dict:
    """Restart vLLM using a specific local image (e.g. after pulling an update)."""
    if not confirm:
        return {"hint": "call again with confirm=true to recreate the container",
                "ref": ref, "profile": profile, "ctx": ctx}
    return _post("/api/docker/restart-with",
                 {"ref": ref, "profile": profile, "ctx": ctx})


# ── write tools ──────────────────────────────────────────────────────────────
def _opts(engine: str, profile: str, ctx: str | None, kv_capacity: str | None,
          spec_tokens: str | None, extra_args: list[str] | None) -> dict:
    return {
        "engine": engine,
        "profile": profile,
        "ctx": ctx,
        "kv_capacity": kv_capacity,
        "spec_tokens": spec_tokens,
        "extra_args": extra_args or [],
    }


@mcp.tool()
def llm_start(engine: str = "vllm", profile: str = "stage1", ctx: str | None = None,
              kv_capacity: str | None = None, spec_tokens: str | None = None,
              extra_args: list[str] | None = None, confirm: bool = False) -> dict:
    """Start an engine (engine: vllm|ninfer; profile: stage1|mtp|dflash|dflash2|kv2|kv3).

    Without confirm=true this only returns the command preview.
    """
    opts = _opts(engine, profile, ctx, kv_capacity, spec_tokens, extra_args)
    if not confirm:
        return {"preview": _get("/api/engine/dry-run", engine=engine, profile=profile, ctx=ctx,
                                kv_capacity=kv_capacity, spec_tokens=spec_tokens),
                "hint": "call again with confirm=true to actually start"}
    return _post("/api/engine/start", opts)


@mcp.tool()
def llm_stop(confirm: bool = False) -> dict:
    """Stop the running engine. Without confirm=true nothing happens."""
    if not confirm:
        return {"hint": "call again with confirm=true to stop the engine"}
    return _post("/api/engine/stop", {})


@mcp.tool()
def llm_restart(engine: str = "vllm", profile: str = "stage1", ctx: str | None = None,
                kv_capacity: str | None = None, spec_tokens: str | None = None,
                confirm: bool = False) -> dict:
    """Recreate the engine (picks up changed profile/context). Requires confirm=true."""
    if not confirm:
        return {"hint": "call again with confirm=true to restart"}
    return _post("/api/engine/restart",
                 _opts(engine, profile, ctx, kv_capacity, spec_tokens, None))


@mcp.tool()
def llm_switch(engine: str, profile: str = "stage1", ctx: str | None = None,
               kv_capacity: str | None = None, spec_tokens: str | None = None,
               confirm: bool = False) -> dict:
    """Atomically switch engines: stop → wait for VRAM release → start → health. Requires confirm=true."""
    if not confirm:
        return {"preview": _get("/api/engine/dry-run", engine=engine, profile=profile, ctx=ctx),
                "hint": "call again with confirm=true to switch"}
    return _post("/api/engine/switch",
                 _opts(engine, profile, ctx, kv_capacity, spec_tokens, None))


@mcp.tool()
def llm_bench(prompt: str = "Explain in one paragraph what speculative decoding is.",
              max_tokens: int = 256, model: str | None = None) -> dict:
    """Send a real streaming chat request through the gateway and report TTFT / decode speed."""
    cfg = _get("/api/config")
    gateway_port = cfg["server"]["gateway_port"]
    served = model or (_get("/api/state").get("engine", {}).get("model", {}) or {}).get("served")
    payload = {
        "model": served or "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    started = time.time()
    first = None
    chunks = 0
    usage = {}
    openai = cfg.get("server", {}).get("auth_token")
    headers = {"content-type": "application/json"}
    if openai and openai != "***":
        headers["authorization"] = f"Bearer {openai}"
    with httpx.Client(timeout=None) as cli:
        with cli.stream("POST", f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                        json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                        chunks += 1
                        if first is None:
                            first = time.time()
    total = time.time() - started
    ttft = (first - started) if first else None
    completion = usage.get("completion_tokens") or chunks
    decode_s = (time.time() - first) if first else None
    return {
        "ok": True,
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "total_s": round(total, 3),
        "completion_tokens": completion,
        "decode_tps": round(max(0, completion - 1) / decode_s, 2) if decode_s and decode_s > 0.05 else None,
        "usage": usage,
    }


def main(argv: list[str] | None = None) -> int:
    global DEFAULT_URL, TOKEN, _client
    parser = argparse.ArgumentParser(prog="llm-monitor-mcp")
    parser.add_argument("--url", default=DEFAULT_URL, help="monitor control API base URL")
    parser.add_argument("--token", default=TOKEN)
    parser.add_argument("--transport", default="stdio", choices=["stdio", "http", "sse"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args(argv)

    DEFAULT_URL = args.url.rstrip("/")
    TOKEN = args.token
    _client = None

    if args.transport in ("http", "sse"):
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.settings.streamable_http_path = "/mcp"
        mcp.run(transport="streamable-http" if args.transport == "http" else "sse")
    else:
        mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
