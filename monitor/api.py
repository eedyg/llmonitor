"""Control plane: REST + WebSocket + static UI + a unified /metrics export."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import billing, launcher
from .docker_scope import DockerError, ResourceInUse, ScopeViolation
from .runtime import Runtime
from .store import day_window

WEB_DIR = Path(__file__).parent / "web"


def build_api(rt: Runtime) -> FastAPI:
    cfg = rt.cfg
    app = FastAPI(title="llm-monitor", docs_url="/api/docs", redoc_url=None)
    app.state.rt = rt

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        """Never let the browser cache the UI or the API.

        The frontend is four unversioned files. StaticFiles answers with an ETag
        and Last-Modified but no Cache-Control, so a browser may keep an old
        app.js and silently half-render a page whose index.html is already new
        (that is exactly how the money row ended up showing "—")."""
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith(("/static/", "/api/")):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.middleware("http")
    async def auth(request: Request, call_next):
        token = str(cfg.root.server.auth_token or "")
        if token and not request.url.path.startswith("/static"):
            supplied = request.headers.get("x-monitor-token") or request.query_params.get("token")
            if supplied != token:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    # -- state --------------------------------------------------------------
    @app.get("/api/state")
    async def state() -> dict:
        return rt.snapshot()

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "ts": time.time(), "engine": rt.controller.info()["state"]}

    @app.get("/api/config")
    async def config() -> dict:
        data = cfg.as_dict()
        data["server"]["auth_token"] = "***" if data["server"]["auth_token"] else ""
        data["paths"]["run_llm"] = str(cfg.p("paths.run_llm"))
        data["paths"]["run_dir"] = str(cfg.p("paths.run_dir"))
        data["paths"]["state_dir"] = str(cfg.p("paths.state_dir"))
        data["storage"]["db"] = str(cfg.p("storage.db"))
        return data

    @app.get("/api/series")
    async def series(since: float | None = None, window: int | None = None,
                     keys: str | None = None) -> dict:
        if since is None:
            since = time.time() - (window or 3600)
        wanted = [k.strip() for k in keys.split(",")] if keys else None
        return rt.store.series(since=since, keys=wanted)

    @app.get("/api/tokens")
    async def tokens(start: str | None = None, end: str | None = None) -> dict:
        """Token ledger for a local-date window (inclusive, day precision)."""
        try:
            since, until = day_window(start, end)
        except ValueError:
            raise HTTPException(400, "start/end must be YYYY-MM-DD (or an epoch)")
        data = rt.store.token_range(since, until, cache_ratio=rt.engine.cache_hit_ratio())
        data["start"] = datetime.fromtimestamp(since).date().isoformat()
        data["end"] = datetime.fromtimestamp(until - 1).date().isoformat()
        data["days"] = len(data["daily"])
        kwh = rt.store.energy_by_day(data["start"], data["end"])
        data["billing"] = billing.range_report(cfg.root.billing, data, kwh)
        return data

    # -- requests -----------------------------------------------------------
    @app.get("/api/requests")
    async def requests(limit: int = 100, offset: int = 0, engine: str | None = None,
                       status: str | None = None, since: float | None = None,
                       search: str | None = None) -> dict:
        return rt.store.history(limit=limit, offset=offset, engine=engine, status=status,
                                since=since, search=search)

    @app.get("/api/requests/live")
    async def live_requests() -> dict:
        return {"items": rt.taps.snapshot()}

    @app.get("/api/requests/{request_id}")
    async def request_detail(request_id: str) -> dict:
        row = rt.store.get(request_id)
        if not row:
            raise HTTPException(404, "unknown request id")
        return row

    @app.delete("/api/requests")
    async def clear_requests(keep: int = 0) -> dict:
        deleted = rt.store.clear(keep_last=keep)
        return {"deleted": deleted}

    @app.get("/api/events")
    async def events(limit: int = 50) -> dict:
        return {"items": list(rt.events)[-max(1, min(limit, 300)):]}

    @app.get("/api/logs")
    async def logs(tail: int = 200) -> dict:
        return await rt.controller.logs(tail=tail)

    # -- lifecycle ----------------------------------------------------------
    @app.post("/api/engine/start")
    async def start(payload: dict) -> dict:
        return await rt.controller.start(payload or {})

    @app.post("/api/engine/stop")
    async def stop(payload: dict | None = None) -> dict:
        force = bool((payload or {}).get("force"))
        return await rt.controller.stop(force=force)

    @app.post("/api/engine/restart")
    async def restart(payload: dict | None = None) -> dict:
        return await rt.controller.restart(payload or {})

    @app.post("/api/engine/switch")
    async def switch(payload: dict) -> dict:
        return await rt.controller.switch(payload or {})

    @app.post("/api/engine/probe")
    async def probe(payload: dict | None = None) -> dict:
        extended = bool((payload or {}).get("extended"))
        return await rt.controller.probe(extended=extended)

    async def _plan_payload(opts: dict) -> dict:
        if rt.controller.use_launcher:
            plan = await asyncio.to_thread(rt.controller.build_plan, opts)
            return plan.public()
        cmd, _env = rt.controller.build_command(opts)
        return {"engine": opts.get("engine"), "kind": "run_llm.sh", "ok": True,
                "command": cmd, "argv": cmd, "warnings": [], "errors": [],
                "plan_source": "run_llm.sh"}

    @app.get("/api/engines")
    async def engines() -> dict:
        data = {}
        for name in launcher.available_engines():
            try:
                data[name] = launcher.engine_schema(name)
                data[name]["default_profile"] = rt.controller.default_profile(name)
                rt.controller.apply_kvcache_defaults(data[name])
            except Exception as exc:  # pragma: no cover
                data[name] = {"engine": name, "error": str(exc), "options": []}
        return {"engines": data, "default": rt.cfg.root.engine.default,
                "gateway_port": rt.controller.gateway_port,
                "engine_port": rt.controller.engine_port}

    @app.get("/api/kvcache")
    async def kvcache() -> dict:
        """KV cache tiers (GPU/CPU/disk) + disk health, low-frequency sampled."""
        return rt.kvcache.view()

    @app.post("/api/engine/plan")
    async def engine_plan_post(payload: dict | None = None) -> dict:
        return await _plan_payload(dict(payload or {}))

    @app.get("/api/engine/plan")
    async def engine_plan(engine: str = "vllm", profile: str = "stage1", ctx: str | None = None,
                          kv_capacity: str | None = None, spec_tokens: str | None = None,
                          extra: str | None = None) -> dict:
        opts = {
            "engine": engine, "profile": profile, "ctx": ctx,
            "kv_capacity": kv_capacity, "spec_tokens": spec_tokens,
            "extra_args": extra.split() if extra else [],
        }
        return await _plan_payload(opts)

    @app.get("/api/engine/dry-run")
    async def dry_run(engine: str, profile: str = "stage1", ctx: str | None = None,
                      kv_capacity: str | None = None, spec_tokens: str | None = None) -> dict:
        opts = {"engine": engine, "profile": profile, "ctx": ctx,
                "kv_capacity": kv_capacity, "spec_tokens": spec_tokens}
        cmd, _env = rt.controller.build_command(opts)
        return {"command": cmd}

    # -- docker (vLLM scope only) -------------------------------------------
    DOCKER_JOB_MAX = 40

    def _job_public(job: dict, lines: int = 0) -> dict:
        data = {
            "id": job["id"], "kind": job["kind"], "ref": job["ref"],
            "state": job["state"], "started_at": job["started_at"],
            "finished_at": job["finished_at"], "error": job["error"],
            "old_id": job["old_id"], "new_id": job["new_id"],
            "changed": job["changed"], "progress": job["progress"],
            "timeout_hit": job.get("timeout_hit", False),
        }
        if lines:
            data["lines"] = list(job["lines"])[-lines:]
        return data

    def _job_proc(job: dict, proc) -> None:
        job["proc"] = proc

    async def _pull_timeout(job: dict, seconds: float) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        proc = job.get("proc")
        if job["state"] == "running" and proc is not None and proc.returncode is None:
            job["cancel_requested"] = True
            job["timeout_hit"] = True
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    async def _pull_task(job: dict, ref: str) -> None:
        job["state"] = "running"
        rt.publish({"type": "docker_job", "job": _job_public(job)})
        last = 0.0
        timeout = float(cfg.root.docker.pull_timeout_s)
        watchdog = asyncio.create_task(_pull_timeout(job, timeout))
        try:
            job["old_id"] = await asyncio.to_thread(rt.docker.image_id, ref)
            async for line in rt.docker.pull_stream(ref, on_proc=lambda p: _job_proc(job, p)):
                job["lines"].append(line)
                if line.strip():
                    job["progress"] = line.strip()[:160]
                now = time.time()
                if now - last > 0.3:
                    last = now
                    rt.publish({"type": "docker_job", "job": _job_public(job)})
            new_id = await asyncio.to_thread(rt.docker.image_id, ref)
            job["new_id"] = new_id
            if new_id:
                rt.store.track_image(new_id, [ref], "pull")
            job["changed"] = bool(job["old_id"] and new_id and job["old_id"] != new_id)
            if job["changed"] and job["old_id"]:
                rt.store.mark_superseded(job["old_id"])
            job["state"] = "done"
        except Exception as exc:
            if job.get("cancel_requested"):
                job["state"] = "cancelled"
                job["error"] = "cancelled" + (" (timeout)" if job.get("timeout_hit") else "")
            else:
                job["state"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            watchdog.cancel()
            job["finished_at"] = time.time()
            rt.publish({"type": "docker_job", "job": _job_public(job)})
            while len(rt.docker_jobs) > DOCKER_JOB_MAX:
                rt.docker_jobs.pop(next(iter(rt.docker_jobs)), None)

    @app.get("/api/docker/summary")
    async def docker_summary() -> dict:
        if not bool(cfg.root.docker.enabled):
            return {"available": False, "enabled": False, "reason": "docker.enabled=false"}
        try:
            data = await asyncio.to_thread(rt.docker.summary, rt.store.tracked_image_ids())
        except Exception as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}",
                    "scope": rt.docker_scope.as_dict()}
        data["enabled"] = True
        data["image_override"] = rt.image_override
        data["running_jobs"] = [
            _job_public(j) for j in rt.docker_jobs.values() if j["state"] == "running"
        ]
        data["ledger"] = rt.store.docker_ledger(limit=20)
        return data

    @app.get("/api/docker/jobs")
    async def docker_jobs() -> dict:
        return {"items": [_job_public(j) for j in reversed(list(rt.docker_jobs.values()))]}

    @app.get("/api/docker/jobs/{job_id}")
    async def docker_job(job_id: str, lines: int = 200) -> dict:
        job = rt.docker_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        return _job_public(job, lines=lines)

    @app.post("/api/docker/pull")
    async def docker_pull(payload: dict) -> dict:
        if not bool(cfg.root.docker.enabled):
            raise HTTPException(400, "docker management is disabled")
        ref = str((payload or {}).get("ref") or "").strip()
        if not ref:
            raise HTTPException(400, "ref is required")
        if not rt.docker_scope.image_ref_allowed(ref):
            raise HTTPException(403, f"image '{ref}' is outside the vLLM scope")
        job = {
            "id": "dpull_" + uuid.uuid4().hex[:10], "kind": "pull", "ref": ref,
            "state": "queued", "started_at": time.time(), "finished_at": None,
            "error": None, "old_id": None, "new_id": None, "changed": False,
            "progress": "", "lines": __import__("collections").deque(maxlen=500),
            "proc": None, "cancel_requested": False, "timeout_hit": False,
        }
        rt.docker_jobs[job["id"]] = job
        asyncio.create_task(_pull_task(job, ref))
        return {"ok": True, "job": _job_public(job)}

    @app.post("/api/docker/images/remove")
    async def docker_remove_image(payload: dict) -> dict:
        payload = payload or {}
        image_id = str(payload.get("id") or "")
        if not image_id:
            raise HTTPException(400, "id is required")
        tracked = rt.store.tracked_image_ids()
        try:
            if not payload.get("confirm"):
                return {"preview": await asyncio.to_thread(
                    rt.docker.remove_image, image_id, tracked, True)}
            result = await asyncio.to_thread(rt.docker.remove_image, image_id, tracked)
            rt.store.mark_image_removed(image_id)
            rt.publish({"type": "docker_changed", "action": "rmi", "id": image_id})
            return {"ok": True, **result}
        except ResourceInUse as exc:
            raise HTTPException(409, str(exc)) from exc
        except ScopeViolation as exc:
            raise HTTPException(403, str(exc)) from exc
        except DockerError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/docker/jobs/{job_id}/cancel")
    async def docker_job_cancel(job_id: str) -> dict:
        job = rt.docker_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if job["state"] != "running":
            return {"ok": False, "state": job["state"]}
        job["cancel_requested"] = True
        proc = job.get("proc")
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        return {"ok": True, "state": "cancelling"}

    @app.post("/api/docker/containers/remove")
    async def docker_remove_container(payload: dict) -> dict:
        payload = payload or {}
        name = str(payload.get("name") or "")
        if not name:
            raise HTTPException(400, "name is required")
        try:
            if not payload.get("confirm"):
                return {"preview": await asyncio.to_thread(
                    rt.docker.remove_container, name, True)}
            result = await asyncio.to_thread(rt.docker.remove_container, name)
            rt.publish({"type": "docker_changed", "action": "rm", "name": name})
            return {"ok": True, **result}
        except ScopeViolation as exc:
            raise HTTPException(403, str(exc)) from exc
        except DockerError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/docker/prune")
    async def docker_prune(payload: dict | None = None) -> dict:
        payload = payload or {}
        tracked = rt.store.tracked_image_ids()
        images = await asyncio.to_thread(rt.docker.images, tracked)
        targets = [img for img in images if img["removable"]]
        reclaim = round(sum(img["size_mb"] or 0 for img in targets), 1)
        if not payload.get("confirm"):
            return {
                "preview": [
                    {"id": img["id"], "tags": img["in_scope_tags"] or ["<none>"],
                     "size_mb": img["size_mb"]}
                    for img in targets
                ],
                "count": len(targets),
                "reclaimable_mb": reclaim,
                "hint": "call again with confirm=true to remove exactly these images",
            }
        results = []
        for img in targets:
            try:
                result = await asyncio.to_thread(
                    rt.docker.remove_image, img["id"], rt.store.tracked_image_ids())
                rt.store.mark_image_removed(img["id"])
                results.append({"id": img["id"], "ok": True, "result": result})
            except Exception as exc:
                results.append({"id": img["id"], "ok": False, "error": str(exc)})
        rt.publish({"type": "docker_changed", "action": "prune", "count": len(results)})
        return {"ok": True, "removed": results, "reclaimable_mb": reclaim}

    @app.post("/api/docker/restart-with")
    async def docker_restart_with(payload: dict) -> dict:
        payload = payload or {}
        ref = str(payload.get("ref") or "").strip()
        if not ref:
            raise HTTPException(400, "ref is required")
        if not rt.docker_scope.image_ref_allowed(ref):
            raise HTTPException(403, f"image '{ref}' is outside the vLLM scope")
        if not await asyncio.to_thread(lambda: rt.docker.image_id(ref) is not None):
            raise HTTPException(400, f"localhost has no image '{ref}' (pull it first)")
        rt.image_override = ref
        engine = payload.get("engine") or "vllm"
        opts = {
            "engine": engine,
            "profile": payload.get("profile") or "stage1",
            "ctx": payload.get("ctx"),
            "kv_capacity": payload.get("kv_capacity"),
            "spec_tokens": payload.get("spec_tokens"),
            "image": ref,
        }
        result = await rt.controller.restart(opts)
        result["image"] = ref
        return result

    # -- prometheus export --------------------------------------------------
    @app.get("/metrics")
    async def metrics() -> Response:
        return PlainTextResponse(render_prometheus(rt.snapshot()), media_type="text/plain; version=0.0.4")

    # -- websocket ----------------------------------------------------------
    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        token = str(cfg.root.server.auth_token or "")
        if token and socket.query_params.get("token") != token:
            await socket.close(code=4401)
            return
        await socket.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        rt.subscribers.add(queue)
        try:
            await socket.send_json(rt.snapshot())
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    await socket.send_json(event)
                except asyncio.TimeoutError:
                    await socket.send_json(rt.snapshot())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            rt.subscribers.discard(queue)

    # -- static -------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    async def index() -> Response:
        """Serve the shell with a content stamp on every asset URL.

        Belt and braces on top of the no-store header: if a browser, an
        extension or a proxy ever ignores caching headers, a changed app.js
        still lands on a *new* URL, so a plain reload is always enough.
        (A stale app.js next to a fresh index.html is how the money row showed
        "—" while the backend was serving the numbers just fine.)
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        stamps = {}
        for name in ("app.js", "style.css"):
            try:
                stamps[name] = int((WEB_DIR / name).stat().st_mtime)
            except OSError:
                stamps[name] = 0
        for name, stamp in stamps.items():
            html = html.replace(f'/static/{name}"', f'/static/{name}?v={stamp}"')
        html = html.replace(
            "<head>",
            f'<head>\n<meta name="monitor-build" content="{max(stamps.values())}" />',
            1,
        )
        return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})

    return app


def render_prometheus(snap: dict) -> str:
    """Unified metric namespace so Grafana can scrape one endpoint."""
    engine = snap.get("engine") or {}
    model = engine.get("model") or {}
    req = engine.get("requests") or {}
    tokens = engine.get("tokens") or {}
    cache = engine.get("cache") or {}
    speed = engine.get("speed") or {}
    spec = engine.get("spec") or {}
    ctrl = snap.get("controller") or {}
    labels = f'engine="{ctrl.get("engine") or "none"}",model="{model.get("served") or "none"}"'
    lines: list[str] = []
    seen: set[str] = set()

    def add(name: str, value, kind: str = "gauge", labels_str: str = "", help_text: str = "") -> None:
        if value is None:
            return
        if name not in seen:
            seen.add(name)
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
        suffix = f"{{{labels_str}}}" if labels_str else ""
        lines.append(f"{name}{suffix} {value}")

    add("llm_engine_up", 1 if engine.get("ok") else 0, "gauge", labels, "Engine is serving")
    add("llm_engine_state", 1, "gauge", f'{labels},state="{ctrl.get("state")}"',
        "Current lifecycle state (one-hot)")
    add("llm_requests_running", req.get("running"), "gauge", labels, "Resident requests")
    add("llm_requests_waiting", req.get("waiting"), "gauge", labels, "Queued requests")
    add("llm_requests_prefilling", req.get("prefilling"), "gauge", labels, "Requests in prefill")
    add("llm_requests_paused", req.get("paused"), "gauge", labels, "Paused requests")
    add("llm_preemptions_total", req.get("preemptions"), "counter", labels, "Pressure preemptions")
    add("llm_prompt_tokens_total", tokens.get("prompt"), "counter", labels, "Prompt tokens")
    add("llm_prompt_tokens_cached_total", tokens.get("prompt_cached"), "counter", labels,
        "Prompt tokens served from cache")
    add("llm_prefill_tokens_total", tokens.get("prefill"), "counter", labels, "Prefill tokens computed")
    add("llm_generation_tokens_total", tokens.get("generation"), "counter", labels, "Generated tokens")
    add("llm_kv_usage_percent", cache.get("kv_pct"), "gauge", labels, "KV cache occupancy")
    add("llm_prefix_cache_queries_total", cache.get("prefix_queries"), "counter", labels,
        "Prefix cache queries")
    add("llm_prefix_cache_hits_total", cache.get("prefix_hits"), "counter", labels,
        "Prefix cache hits")
    for key, metric, kind in (
        ("decode_tps", "llm_decode_tokens_per_second", "gauge"),
        ("prefill_tps", "llm_prefill_tokens_per_second", "gauge"),
        ("ttft_p50_ms", "llm_ttft_p50_milliseconds", "gauge"),
        ("ttft_p90_ms", "llm_ttft_p90_milliseconds", "gauge"),
        ("ttft_p99_ms", "llm_ttft_p99_milliseconds", "gauge"),
        ("e2e_p50_ms", "llm_e2e_p50_milliseconds", "gauge"),
        ("itl_p50_ms", "llm_itl_p50_milliseconds", "gauge"),
    ):
        add(metric, speed.get(key), kind, labels, metric.replace("_", " "))
    add("llm_spec_accept_rate", spec.get("accept_rate"), "gauge", labels,
        "Speculative decoding acceptance rate")
    add("llm_spec_drafted_tokens_total", spec.get("drafted"), "counter", labels, "Draft tokens")
    add("llm_spec_accepted_tokens_total", spec.get("accepted"), "counter", labels, "Accepted tokens")

    for gpu in (snap.get("gpu") or {}).get("gpus") or []:
        g = f'gpu="{gpu.get("index")}",name="{gpu.get("name")}"'
        add("llm_gpu_memory_used_mib", gpu.get("mem_used_mib"), "gauge", g, "GPU memory used")
        add("llm_gpu_memory_total_mib", gpu.get("mem_total_mib"), "gauge", g, "GPU memory total")
        add("llm_gpu_memory_used_percent", gpu.get("mem_used_pct"), "gauge", g, "GPU memory used %")
        add("llm_gpu_utilization_percent", gpu.get("util_pct"), "gauge", g, "GPU utilization")
        add("llm_gpu_temperature_celsius", gpu.get("temp_c"), "gauge", g, "GPU temperature")
        add("llm_gpu_power_watts", gpu.get("power_w"), "gauge", g, "GPU power draw")
        add("llm_gpu_power_limit_watts", gpu.get("power_limit_w"), "gauge", g, "GPU power limit")
        add("llm_gpu_fan_percent", gpu.get("fan_pct"), "gauge", g, "GPU fan speed")
        add("llm_gpu_sm_clock_mhz", gpu.get("sm_clock_mhz"), "gauge", g, "SM clock")

    gw = snap.get("gateway") or {}
    add("llm_gateway_requests_total", gw.get("requests"), "counter", "", "Requests through the tap")
    add("llm_gateway_in_flight", gw.get("in_flight"), "gauge", "", "In-flight requests")
    add("llm_gateway_errors_total", gw.get("errors"), "counter", "", "Gateway errors")

    # Cumulative request ledger (this process, read from the history DB).
    ledger = (snap.get("tokens") or {}).get("session") or {}
    add("llm_monitor_tokens_total", ledger.get("total"), "counter", "",
        "Tokens in + out recorded by this process")
    add("llm_monitor_prompt_tokens_total", ledger.get("prompt"), "counter", "",
        "Prompt tokens recorded by this process")
    add("llm_monitor_prompt_tokens_cached_total", ledger.get("prompt_cached"), "counter", "",
        "Prompt tokens served from the prefix cache")
    add("llm_monitor_prompt_tokens_miss_total", ledger.get("prompt_miss"), "counter", "",
        "Prompt tokens not served from the prefix cache")
    add("llm_monitor_output_tokens_total", ledger.get("output"), "counter", "",
        "Output tokens (thinking + content)")
    add("llm_monitor_thinking_tokens_total", ledger.get("thinking"), "counter", "",
        "Output tokens spent thinking")
    add("llm_monitor_content_tokens_total", ledger.get("content"), "counter", "",
        "Output tokens actually returned as content")
    add("llm_monitor_requests_recorded_total", ledger.get("requests"), "counter", "",
        "Requests recorded in the history DB by this process")

    # Equivalent money (retained history, see billing.py)
    money = (snap.get("billing") or {}).get("history") or {}
    add("llm_monitor_api_cost_cny", (money.get("deepseek") or {}).get("cost"), "gauge", "",
        "What the recorded tokens would cost at the deepseek-flash rate card")
    add("llm_monitor_energy_kwh", (money.get("energy") or {}).get("kwh"), "gauge", "",
        "GPU energy integrated over the recorded days")
    add("llm_monitor_electricity_cost_cny", (money.get("energy") or {}).get("cost"), "gauge", "",
        "Electricity cost of the GPU at billing.electricity.cny_per_kwh")
    add("llm_monitor_cost_cny", money.get("total"), "gauge", "",
        "API equivalent + electricity")
    return "\n".join(lines) + "\n"
