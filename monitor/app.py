"""Process entry point: two uvicorn servers (gateway + control/UI) in one loop."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time

import httpx
import uvicorn

from .api import build_api
from .config import Config
from .controller import Controller
from .engine import Metrics, adapter_for, blank, detect_kind
from .gateway import build_gateway
from .runtime import Runtime

log = logging.getLogger("monitor")


async def metrics_loop(rt: Runtime) -> None:
    interval = float(rt.cfg.root.collect.metrics_interval_s)
    static_key = (rt.cfg.root.engine.extra_env or {}).get("API_KEY")
    base = f"http://127.0.0.1:{int(rt.cfg.root.server.engine_port)}"
    last_models = 0.0
    async with httpx.AsyncClient() as client:
        while True:
            api_key = rt.engine_api_key or static_key
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            try:
                if time.time() - last_models > 30:
                    last_models = time.time()
                    try:
                        resp = await client.get(f"{base}/v1/models", headers=headers, timeout=5.0)
                        if resp.status_code == 200:
                            data = (resp.json() or {}).get("data") or []
                            if data:
                                rt.engine.models = {
                                    "served": data[0].get("id"),
                                    "max_model_len": data[0].get("max_model_len"),
                                    "owned_by": data[0].get("owned_by"),
                                }
                    except Exception:
                        pass
                kind = rt.engine.kind or (rt.controller.kind if rt.controller else None)
                adapter = adapter_for(kind, rt.engine)
                metrics = await adapter.fetch(client, base, headers)
                if metrics is not None:
                    detected = detect_kind(rt.engine.raw_text)
                    if detected and detected != kind:
                        kind = detected
                        rt.engine.kind = detected
                        adapter = adapter_for(detected, rt.engine)
                        metrics = Metrics(rt.engine.raw_text)
                    else:
                        rt.engine.kind = kind
                    snap = adapter.normalize(metrics, time.time())
                    if rt.engine.models:
                        (snap.get("model") or {}).update(rt.engine.models)
                    rt.engine.snapshot = snap
                    rt.engine.failures = 0
                else:
                    rt.engine.failures += 1
                    rt.engine.snapshot["ok"] = False
                    if rt.engine.failures >= 3:
                        fresh = blank()
                        if rt.engine.models:
                            fresh["model"].update(rt.engine.models)
                        fresh["error"] = rt.engine.last_error
                        rt.engine.snapshot = fresh
            except Exception as exc:  # pragma: no cover - defensive
                rt.engine.last_error = f"{type(exc).__name__}: {exc}"
                log.debug("metrics loop error: %s", exc)
            await asyncio.sleep(interval)


async def gpu_loop(rt: Runtime) -> None:
    interval = float(rt.cfg.root.collect.gpu_interval_s)
    while True:
        try:
            await asyncio.to_thread(rt.gpu.sample)
        except Exception as exc:  # pragma: no cover
            log.debug("gpu loop error: %s", exc)
        await asyncio.sleep(interval)


async def series_loop(rt: Runtime) -> None:
    while True:
        rt.push_point()
        await asyncio.sleep(1.0)


async def controller_loop(rt: Runtime) -> None:
    interval = float(rt.cfg.root.collect.runtime_interval_s)
    while True:
        try:
            await rt.controller.refresh()
        except Exception as exc:  # pragma: no cover
            log.debug("controller loop error: %s", exc)
        await asyncio.sleep(interval)


async def jsonl_loop(rt: Runtime) -> None:
    """Merge NInfer's authoritative JSONL request records into tapped rows."""
    if not bool(rt.cfg.root.engine.ninfer_jsonl):
        return
    path = rt.cfg.p("engine.ninfer_jsonl_path")
    offset = 0
    while True:
        try:
            if rt.engine.kind == "ninfer" and path.exists():
                size = path.stat().st_size
                if size < offset:
                    offset = 0
                if size > offset:
                    with path.open("rb") as handle:
                        handle.seek(offset)
                        blob = handle.read()
                        offset = handle.tell()
                    lines = blob.split(b"\n")
                    if blob and not blob.endswith(b"\n"):
                        offset -= len(lines[-1])
                        lines = lines[:-1]
                    for raw in lines:
                        if not raw.strip():
                            continue
                        try:
                            record = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if record.get("event") != "request_done":
                            continue
                        req = record.get("request") or {}
                        engine_id = req.get("http_request_id")
                        if engine_id and rt.store.attach_engine_record(engine_id, record):
                            rt.publish({"type": "request_enriched", "id": engine_id})
        except Exception as exc:  # pragma: no cover
            log.debug("jsonl loop error: %s", exc)
        await asyncio.sleep(1.0)


def _disable_uvicorn_signals() -> None:
    """Two uvicorn servers must not fight over SIGINT/SIGTERM (the last one wins,
    leaving the other serving forever). We own the signals ourselves."""
    if getattr(uvicorn.Server, "_monitor_patched", False):
        return

    def capture_signals(self):  # noqa: ANN001
        @contextlib.contextmanager
        def _null():
            yield

        return _null()

    uvicorn.Server.capture_signals = capture_signals  # type: ignore[assignment]
    uvicorn.Server._monitor_patched = True  # type: ignore[attr-defined]


def _port_free(host: str, port: int) -> bool:
    import socket

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((probe_host, port))
            return True
        except OSError:
            return False


def _preflight(cfg: Config) -> None:
    gw = int(cfg.root.server.gateway_port)
    ui = int(cfg.root.server.ui_port)
    engine = int(cfg.root.server.engine_port)
    if gw == engine:
        raise SystemExit(f"gateway_port and engine_port are both {gw}; the gateway needs its own port")
    if not _port_free(str(cfg.root.server.gateway_host), gw):
        raise SystemExit(
            f"gateway port {gw} is already in use.\n"
            f"  Most likely the engine is still listening on {gw}: the monitor owns the public\n"
            f"  port and the engine must move to {engine} (run_llm.sh HOST_PORT={engine}).\n"
            f"  Either stop what holds :{gw} or set MONITOR_GATEWAY_PORT to a free port."
        )
    if not _port_free(str(cfg.root.server.ui_host), ui):
        raise SystemExit(f"control/UI port {ui} is already in use (set MONITOR_UI_PORT)")


async def run(cfg: Config) -> None:
    _preflight(cfg)
    rt = Runtime(cfg)
    rt.controller = Controller(rt)
    gateway = build_gateway(rt)
    api = build_api(rt)

    tasks = [
        asyncio.create_task(metrics_loop(rt), name="metrics"),
        asyncio.create_task(gpu_loop(rt), name="gpu"),
        asyncio.create_task(series_loop(rt), name="series"),
        asyncio.create_task(controller_loop(rt), name="controller"),
        asyncio.create_task(jsonl_loop(rt), name="jsonl"),
    ]

    ui_host = str(cfg.root.server.ui_host)
    ui_port = int(cfg.root.server.ui_port)
    gw_host = str(cfg.root.server.gateway_host)
    gw_port = int(cfg.root.server.gateway_port)
    engine_port = int(cfg.root.server.engine_port)

    log.info("control  : http://%s:%s/  (UI + REST + WS)", ui_host, ui_port)
    log.info("gateway  : http://%s:%s/  -> engine 127.0.0.1:%s", gw_host, gw_port, engine_port)

    _disable_uvicorn_signals()
    servers = [
        uvicorn.Server(uvicorn.Config(api, host=ui_host, port=ui_port, log_level="warning",
                                      access_log=False)),
        uvicorn.Server(uvicorn.Config(gateway, host=gw_host, port=gw_port, log_level="warning",
                                      access_log=False)),
    ]

    stop_event = asyncio.Event()

    def _stop(*_: object) -> None:
        stop_event.set()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _stop)
            except NotImplementedError:  # pragma: no cover
                pass
        serve = [asyncio.create_task(server.serve(), name=f"uvicorn-{i}") for i, server in enumerate(servers)]
        done, pending = await asyncio.wait(
            [*serve, asyncio.create_task(stop_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*serve, return_exceptions=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("monitor stopped")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-monitor", description="Unified vLLM/NInfer monitor")
    parser.add_argument("--config", default=os.environ.get("MONITOR_CONFIG", "config.yaml"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = Config(args.config)
    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:  # pragma: no cover
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
