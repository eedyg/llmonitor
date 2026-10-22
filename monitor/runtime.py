"""Shared runtime: config, store, collectors, tap registry, event bus."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

from .billing import snapshot_report
from .config import Config
from .docker_scope import Docker, Scope
from .engine import EngineStore
from .inflight import InflightRegistry
from .kvcache import KvCacheMonitor, parse_size as parse_kv_size
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
            peak_hours=tuple(cfg.root.billing.deepseek.peak_hours or ()),
        )
        self.gpu = GpuCollector(
            interval_s=float(cfg.root.collect.gpu_interval_s),
            fallback_csv=bool(cfg.root.gpu.fallback_csv),
        )
        self.engine = EngineStore(window_s=float(cfg.root.collect.rate_window_s))
        kv = cfg.root.kvcache
        self.kvcache = KvCacheMonitor(
            enabled=bool(kv.enabled),
            directory=cfg.p("kvcache.disk_dir"),
            cpu_budget_bytes=parse_kv_size(kv.cpu_bytes_to_use) if bool(kv.enabled) else 0,
            disk_limit_bytes=int(float(kv.disk_limit_gb or 0) * 1024 ** 3),
            disk_enabled=bool(getattr(kv, "disk_enabled", True)),
            device=str(kv.device or ""),
            interval_s=float(kv.collect_interval_s),
            health_interval_s=float(kv.health_interval_s),
            reclaim_target_ratio=float(kv.reclaim_target_ratio),
        )
        self.taps = TapRegistry()
        self.controller: Any = None
        self.subscribers: set[asyncio.Queue] = set()
        self.started_at = time.time()
        self.gateway = {"requests": 0, "in_flight": 0, "errors": 0, "bytes_in": 0, "bytes_out": 0}
        # Live handles for /api/requests/{id}/cancel (see inflight.py).
        self.inflight = InflightRegistry()
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
        kv = self.kvcache.view()
        kvc = kv.get("cpu") or {}
        kvd = kv.get("disk") or {}
        used = kvd.get("used_bytes") or 0
        cap = kvd.get("capacity_bytes") or 0
        read_bps = kvd.get("read_bps")
        write_bps = kvd.get("write_bps")
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
            # KV offload tiers (low-frequency samples, carried between ticks).
            "kv_cpu_used_pct": (
                round(100.0 * (kvc.get("region_used_bytes") or 0) / kvc["budget_bytes"], 2)
                if kvc.get("budget_bytes") else None
            ),
            "kv_cpu_active_pct": kvc.get("active_pct"),
            "kv_disk_pct": round(100.0 * used / cap, 2) if cap else None,
            "kv_disk_used_gb": round(used / 1024 ** 3, 2),
            "kv_disk_read_mbps": round(read_bps / 1024 ** 2, 2) if read_bps is not None else None,
            "kv_disk_write_mbps": round(write_bps / 1024 ** 2, 2) if write_bps is not None else None,
        }

    def push_point(self) -> None:
        self.store.push_sample(self.make_point())

    # -- snapshot -----------------------------------------------------------
    def live_requests(self) -> list[dict]:
        """In-flight rows, annotated with whether they can be aborted.

        The UI is WebSocket-driven (snapshot → live), so this flag has to ride
        along on the snapshot; /api/requests/live uses the same rows.
        """
        rows = self.taps.snapshot()
        for row in rows:
            handle = self.inflight.get(str(row.get("id")))
            row["cancellable"] = bool(handle is not None and handle.abortable
                                       and not handle.cancelled)
            row["cancelled"] = bool(handle is not None and handle.cancelled)
        return rows

    def snapshot(self) -> dict:
        controller = self.controller.info() if self.controller else {"state": "unknown"}
        tokens = self.store.token_totals(cache_ratio=self.engine.cache_hit_ratio())
        return {
            "type": "snapshot",
            "ts": time.time(),
            "uptime_s": round(time.time() - self.started_at, 1),
            "controller": controller,
            "engine": self.engine.snapshot,
            "engine_error": self.engine.last_error,
            "gpu": self.gpu.info(),
            "kvcache": self.kvcache.view(),
            "live": self.live_requests(),
            "summary": self.store.summary(),
            "tokens": tokens,
            "billing": snapshot_report(self.cfg.root.billing, tokens, self.store.energy_totals()),
            "gateway": dict(self.gateway),
            "point": self.make_point(),
            "events": list(self.events)[-12:],
            "image_override": self.image_override,
        }
