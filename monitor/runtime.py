"""Shared runtime: config, store, collectors, tap registry, event bus."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

from .config import Config
from .docker_scope import Docker, Scope
from .engine import EngineStore
from .nvml import GpuCollector
from .store import Store
from .tap import TapRegistry


class Runtime:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        storage = cfg.root.storage
        self.store = Store(
            cfg.p("storage.db"),
            ring_seconds=storage.ring_seconds,
            text_keep=storage.text_keep,
            text_cap=storage.text_cap_bytes,
            retention_days=storage.retention_days,
        )
        self.gpu = GpuCollector(
            interval_s=float(cfg.root.collect.gpu_interval_s),
            fallback_csv=bool(cfg.root.gpu.fallback_csv),
        )
        self.engine = EngineStore(window_s=float(cfg.root.collect.rate_window_s))
        self.taps = TapRegistry()
        self.controller: Any = None
        self.subscribers: set[asyncio.Queue] = set()
        self.started_at = time.time()
        self.gateway = {"requests": 0, "in_flight": 0, "errors": 0, "bytes_in": 0, "bytes_out": 0}
        self.events: deque[dict] = deque(maxlen=300)
        self.jsonl_offset = 0
        containers = list(cfg.root.docker.containers or [])
        if not containers:
            containers = [cfg.root.engine.container_names.vllm]
        self.docker_scope = Scope(
            image_repos=tuple(cfg.root.docker.image_repos),
            containers=tuple(containers),
        )
        self.docker = Docker(self.docker_scope)
        self.docker_jobs: dict[str, dict] = {}
        self.image_override: str | None = None
        self.engine_api_key: str | None = None

    # -- events -------------------------------------------------------------
    def publish(self, event: dict) -> None:
        self.events.append(event)
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # -- series -------------------------------------------------------------
    def make_point(self) -> dict:
        snap = self.engine.snapshot
        gpu = self.gpu.first
        speed = snap.get("speed") or {}
        cache = snap.get("cache") or {}
        req = snap.get("requests") or {}
        return {
            "ts": round(time.time(), 3),
            "decode_tps": speed.get("decode_tps"),
            "prefill_tps": speed.get("prefill_tps"),
            "ttft_p50_ms": speed.get("ttft_p50_ms"),
            "ttft_p90_ms": speed.get("ttft_p90_ms"),
            "ttft_p99_ms": speed.get("ttft_p99_ms"),
            "itl_ms": speed.get("itl_p50_ms"),
            "kv_pct": cache.get("kv_pct"),
            "running": req.get("running"),
            "waiting": req.get("waiting"),
            "gpu_util": gpu.get("util_pct"),
            "vram_pct": gpu.get("mem_used_pct"),
            "vram_used_mib": gpu.get("mem_used_mib"),
            "power_w": gpu.get("power_w"),
            "temp_c": gpu.get("temp_c"),
            "fan_pct": gpu.get("fan_pct"),
        }

    def push_point(self) -> None:
        self.store.push_sample(self.make_point())

    # -- snapshot -----------------------------------------------------------
    def snapshot(self) -> dict:
        controller = self.controller.info() if self.controller else {"state": "unknown"}
        return {
            "type": "snapshot",
            "ts": time.time(),
            "uptime_s": round(time.time() - self.started_at, 1),
            "controller": controller,
            "engine": self.engine.snapshot,
            "engine_error": self.engine.last_error,
            "gpu": self.gpu.info(),
            "live": self.taps.snapshot(),
            "summary": self.store.summary(),
            "gateway": dict(self.gateway),
            "point": self.make_point(),
            "events": list(self.events)[-12:],
            "image_override": self.image_override,
        }
