"""KV cache tier monitoring: GPU HBM → CPU RAM → disk.

The engine side is vLLM's 3-tier offload (``OffloadingConnector`` +
``TieringOffloadingSpec``). vLLM exposes per-tier transfer counters, but it does
*not* expose everything an operator wants to see, so this module is explicit
about what is measured where:

* GPU tier  — real occupancy: ``vllm:kv_cache_usage_perc`` (+ block counts from
  ``vllm:cache_config_info``).
* CPU tier  — vLLM only reports the fraction of CPU chunks **pinned by in-flight
  transfers** (``vllm:kv_offload_cpu_cache_usage_perc``), not the fraction
  holding cached KV. That gauge is therefore reported as "active" (pressure),
  and the *capacity* is taken from configuration and cross-checked against the
  shared mmap region vLLM creates at ``/dev/shm/vllm_offload_<engine_id>.mmap``.
* disk tier — vLLM reports nothing about its footprint, so the directory is
  measured on the host, and the tier's read/write throughput comes from
  ``vllm:kv_offload_tiering_read_bytes`` / ``write_bytes`` per tier label.

vLLM's ``fs`` secondary tier has **no capacity limit and no eviction** (its
constructor takes no size argument), so ``disk_limit_bytes`` is enforced here by
reclaiming the oldest block files. Reclaiming a file that a request is about to
read makes that promotion fail (vLLM then recomputes the prefix), which is why
the reaper only runs above the limit and cuts down to ``target_ratio`` of it.

Everything runs at a low frequency; see config.yaml → ``kvcache``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .engine import Rates
from .prom import Metrics

# -- vLLM metric names (v1/kv_offload/{cpu/common,tiering/base}.py) -------------
GPU_USAGE = "vllm:kv_cache_usage_perc"
CPU_ACTIVE = "vllm:kv_offload_cpu_cache_usage_perc"
CPU_ACTIVE_WRITE = "vllm:kv_offload_cpu_cache_write_usage_perc"
STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
TIER_READ_BYTES = "vllm:kv_offload_tiering_read_bytes"
TIER_WRITE_BYTES = "vllm:kv_offload_tiering_write_bytes"
TIER_READ_TIME = "vllm:kv_offload_tiering_read_time"
TIER_WRITE_TIME = "vllm:kv_offload_tiering_write_time"
TIER_QUERIES = "vllm:kv_offload_tiering_chunk_queries"
TIER_HITS = "vllm:kv_offload_tiering_chunk_hits"
TIER_PROMO_FAIL = "vllm:kv_offload_tiering_promotion_job_failures"
TIER_CASCADE_FAIL = "vllm:kv_offload_tiering_cascade_job_failures"
BP_DROPPED_STORES = "vllm:kv_offload_tiering_backpressure_stores_dropped"
BP_DROPPED_BLOCKS = "vllm:kv_offload_tiering_backpressure_blocks_dropped"

SHM_GLOB = "vllm_offload_*.mmap"

EMPTY: dict[str, Any] = {
    "enabled": False,
    "dir": "",
    "device": "",
    "gpu": {"pct": None, "blocks": None, "block_size": None},
    "cpu": {"budget_bytes": 0, "active_pct": None, "write_pct": None,
            "region_bytes": 0, "region_used_bytes": 0, "stores_skipped": 0},
    "disk": {"used_bytes": 0, "capacity_bytes": 0, "limit_bytes": 0, "files": 0,
             "read_bps": None, "write_bps": None, "read_time_s": 0.0,
             "write_time_s": 0.0, "queries": 0, "hits": 0, "hit_rate": None,
             "promo_fail": 0, "cascade_fail": 0,
             "dropped_stores": 0, "dropped_blocks": 0, "tier": None},
    "health": {"device": "", "ok": False, "error": None, "pct_used": None,
               "temp_c": None, "spare_pct": None, "media_errors": None,
               "critical_warning": None, "written_gb": None, "read_gb": None,
               "power_on_hours": None, "unsafe_shutdowns": None},
    "reclaim": {"last_bytes": 0, "last_files": 0, "last_at": 0.0, "error": None},
    "ts": 0.0,
    "collected_at": 0.0,
}


def blank() -> dict:
    import copy

    return copy.deepcopy(EMPTY)


# -- helpers -------------------------------------------------------------------
def parse_size(raw: Any) -> int:
    """'12G' | '8192m' | 12884901888 -> bytes (0 when unset/invalid)."""
    if raw in (None, "", 0, "0"):
        return 0
    text = str(raw).strip().lower().replace(" ", "").replace("_", "")
    mult = 1
    if text[-1] in ("k", "m", "g", "t"):
        mult = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}[text[-1]]
        text = text[:-1]
    try:
        return int(float(text) * mult)
    except (TypeError, ValueError):
        return 0


def dir_usage(path: Path) -> tuple[int, int]:
    """(bytes, files) of every regular file below ``path``."""
    total = 0
    files = 0
    stack = [str(path)]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                    files += 1
            except OSError:
                continue
    return total, files


def oldest_files(path: Path, limit: int = 20_000) -> list[tuple[float, int, str]]:
    """[(mtime, size, path)] sorted oldest-first, capped at ``limit`` files."""
    out: list[tuple[float, int, str]] = []
    stack = [str(path)]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    stat = entry.stat(follow_symlinks=False)
                    out.append((stat.st_mtime, stat.st_size, entry.path))
            except OSError:
                continue
    out.sort(key=lambda item: item[0])
    return out[:limit]


def filesystem_usage(path: Path) -> tuple[int, int]:
    """(capacity, free) of the filesystem holding ``path``."""
    try:
        stat = os.statvfs(str(path))
    except OSError:
        return 0, 0
    return stat.f_blocks * stat.f_frsize, stat.f_bavail * stat.f_frsize


def device_for_path(path: Path) -> str | None:
    """Best-effort: the NVMe *controller* device backing ``path``.

    Walks /proc/mounts for the longest matching mount point, resolves any
    device-mapper (LVM) layer through /sys/block/<dev>/slaves, and strips a
    partition suffix (nvme1n1p1 -> nvme1n1) because SMART lives on the
    controller, not on the partition.
    """
    mount_source = None
    best_len = -1
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        source, target = parts[0], parts[1].replace("\\040", " ")
        try:
            target_path = Path(target)
        except (TypeError, ValueError):
            continue
        if target_path == path or target_path in path.parents:
            if len(target) > best_len:
                best_len, mount_source = len(target), source
    if not mount_source or not mount_source.startswith("/dev/"):
        return None
    name = os.path.basename(mount_source)
    if mount_source.startswith("/dev/mapper/"):
        name = _dm_name(name) or name
    seen: set[str] = set()
    while name and name not in seen:
        seen.add(name)
        try:
            kids = [p.name for p in (Path("/sys/block") / name / "slaves").iterdir()]
        except OSError:
            kids = []
        if not kids:
            return _controller_device(name)
        name = kids[0]
    return None


def _controller_device(name: str) -> str:
    """``nvme1n1p1`` -> ``/dev/nvme1n1`` (SMART is per controller)."""
    match = re.match(r"(nvme\d+n\d+)", name)
    return f"/dev/{match.group(1)}" if match else f"/dev/{name}"


def _dm_name(mapper: str) -> str | None:
    """``vg_ssd1-lv_kvcache`` -> ``dm-2`` (the /dev/mapper link target)."""
    try:
        for entry in Path("/sys/block").iterdir():
            if not entry.name.startswith("dm-"):
                continue
            try:
                if (entry / "dm/name").read_text().strip() == mapper:
                    return entry.name
            except OSError:
                continue
    except OSError:
        return None
    try:
        return os.path.basename(os.path.realpath(f"/dev/mapper/{mapper}"))
    except OSError:
        return None


def parse_smart(payload: dict) -> dict:
    """nvme-cli ``smart-log -o json`` -> normalized health dict."""
    from_k = lambda value: round((float(value) - 273.15), 1) if value is not None else None
    units = payload.get("data_units_written")
    read_units = payload.get("data_units_read")
    # NVMe counts in 1000 x 512-byte units.
    gib = lambda value: round(float(value) * 1000 * 512 / 1024 ** 3, 1) if value is not None else None
    return {
        "ok": int(payload.get("critical_warning") or 0) == 0,
        "pct_used": payload.get("percent_used"),
        "temp_c": from_k(payload.get("temperature")),
        "spare_pct": payload.get("avail_spare"),
        "media_errors": payload.get("media_errors"),
        "critical_warning": payload.get("critical_warning"),
        "written_gb": gib(units),
        "read_gb": gib(read_units),
        "power_on_hours": payload.get("power_on_hours"),
        "unsafe_shutdowns": payload.get("unsafe_shutdowns"),
        "error": None,
    }


# vLLM registers counters without the suffix but the Prometheus exporter renders
# them as ``..._total`` (0.31), so every lookup tries both spellings.
def _names(name: str) -> tuple[str, ...]:
    return (name,) if name.endswith("_total") else (name, f"{name}_total")


def _value(m: Metrics, name: str) -> float | None:
    for candidate in _names(name):
        value = m.value(candidate)
        if value is not None:
            return value
    return None


def _label_value(m: Metrics, name: str, label: str, value: str) -> float | None:
    for candidate in _names(name):
        for sample in m.samples(candidate):
            if sample.labels.get(label) == value:
                return sample.value
    return None


def disk_tier_label(m: Metrics) -> str | None:
    """The tier label vLLM assigns to the fs tier, e.g. ``"1:fs"``."""
    for name in (TIER_READ_BYTES, TIER_WRITE_BYTES, TIER_QUERIES):
        for candidate in _names(name):
            for sample in m.samples(candidate):
                tier = sample.labels.get("tier") or ""
                if ":fs" in tier:
                    return tier
    return None


@dataclass
class KvCacheMonitor:
    """Low-frequency collector for the 3 KV tiers (+ disk health)."""

    enabled: bool = False
    directory: Path = field(default_factory=lambda: Path("/mnt/kvcache/kvoffload"))
    cpu_budget_bytes: int = 0
    disk_limit_bytes: int = 0
    device: str = ""
    interval_s: float = 30.0
    health_interval_s: float = 120.0
    window_s: float = 90.0
    snapshot: dict = field(default_factory=blank)
    _rates: Rates = field(default=None)  # type: ignore[assignment]
    _last_health: float = 0.0
    _shm_region_bytes: int = 0

    def __post_init__(self) -> None:
        if self._rates is None:
            self._rates = Rates(window_s=self.window_s, min_span_s=1.0)
        self.snapshot["enabled"] = bool(self.enabled)
        self.snapshot["dir"] = str(self.directory)
        self.snapshot["cpu"]["budget_bytes"] = int(self.cpu_budget_bytes)
        self.snapshot["disk"]["limit_bytes"] = int(self.disk_limit_bytes)
        self.snapshot["device"] = self.device or (device_for_path(self.directory) or "")
        self.snapshot["health"]["device"] = self.snapshot["device"]

    def apply_launch(self, variables: dict) -> None:
        """Follow the launch that actually started, not just config.yaml.

        The engine page lets a single launch override the disk-cache directory,
        the disk budget and the CPU budget, so the collector has to re-target
        itself whenever an engine starts.
        """
        enabled = str(variables.get("offload_enabled") or "0").strip().lower() in (
            "1", "true", "yes", "on",
        )
        self.enabled = enabled
        directory = variables.get("kvoffload_dir")
        if directory not in (None, ""):
            self.directory = Path(str(directory))
        limit = variables.get("kv_disk_limit_gb")
        if limit not in (None, ""):
            try:
                self.disk_limit_bytes = int(float(limit) * 1024 ** 3)
            except (TypeError, ValueError):
                pass
        budget = variables.get("cpu_bytes_to_use")
        if budget not in (None, ""):
            size = parse_size(budget)
            if size:
                self.cpu_budget_bytes = size
        snap = self.snapshot
        snap["enabled"] = enabled
        snap["dir"] = str(self.directory)
        snap["cpu"]["budget_bytes"] = int(self.cpu_budget_bytes)
        snap["disk"]["limit_bytes"] = int(self.disk_limit_bytes)
        # The directory may live on another filesystem now: re-detect the device.
        self.device = device_for_path(self.directory) or ""
        snap["device"] = self.device
        snap["health"]["device"] = self.device

    # -- engine side ---------------------------------------------------------
    def collect_engine(self, m: Metrics | None, now: float) -> None:
        snap = self.snapshot
        if m is None:
            return
        fam = m.families.get("vllm:cache_config_info")
        labels = fam.samples[0].labels if fam and fam.samples else {}
        blocks = labels.get("num_gpu_blocks")
        snap["gpu"]["pct"] = round(100.0 * (m.value(GPU_USAGE) or 0.0), 2)
        snap["gpu"]["blocks"] = int(blocks) if blocks not in (None, "") else None
        snap["gpu"]["block_size"] = (
            int(labels["block_size"]) if labels.get("block_size") not in (None, "") else None
        )
        active = _value(m, CPU_ACTIVE)
        write = _value(m, CPU_ACTIVE_WRITE)
        snap["cpu"]["active_pct"] = round(100.0 * active, 2) if active is not None else None
        snap["cpu"]["write_pct"] = round(100.0 * write, 2) if write is not None else None
        snap["cpu"]["stores_skipped"] = int(_value(m, STORES_SKIPPED) or 0)

        tier = disk_tier_label(m)
        if tier:
            disk = snap["disk"]
            disk["tier"] = tier
            read_bytes = _label_value(m, TIER_READ_BYTES, "tier", tier) or 0.0
            write_bytes = _label_value(m, TIER_WRITE_BYTES, "tier", tier) or 0.0
            self._rates.push("read", read_bytes, now)
            self._rates.push("write", write_bytes, now)
            disk["read_bps"] = self._rates.rate("read", now)
            disk["write_bps"] = self._rates.rate("write", now)
            disk["read_time_s"] = round(_label_value(m, TIER_READ_TIME, "tier", tier) or 0.0, 3)
            disk["write_time_s"] = round(_label_value(m, TIER_WRITE_TIME, "tier", tier) or 0.0, 3)
            queries = _label_value(m, TIER_QUERIES, "tier", tier) or 0.0
            hits = _label_value(m, TIER_HITS, "tier", tier) or 0.0
            disk["queries"] = int(queries)
            disk["hits"] = int(hits)
            disk["hit_rate"] = round(hits / queries, 4) if queries else None
            disk["promo_fail"] = int(_label_value(m, TIER_PROMO_FAIL, "tier", tier) or 0)
            disk["cascade_fail"] = int(_label_value(m, TIER_CASCADE_FAIL, "tier", tier) or 0)
            disk["dropped_stores"] = int(_label_value(m, BP_DROPPED_STORES, "tier", tier) or 0)
            disk["dropped_blocks"] = int(_label_value(m, BP_DROPPED_BLOCKS, "tier", tier) or 0)
        snap["ts"] = now
        snap["collected_at"] = now

    # -- host side -----------------------------------------------------------
    def collect_disk(self, now: float, reclaim: bool = True) -> None:
        snap = self.snapshot
        used, files = dir_usage(self.directory) if self.directory.exists() else (0, 0)
        capacity, _free = filesystem_usage(self.directory)
        snap["disk"]["used_bytes"] = used
        snap["disk"]["files"] = files
        snap["disk"]["capacity_bytes"] = (
            int(self.disk_limit_bytes) if self.disk_limit_bytes > 0 else capacity
        )
        snap["cpu"]["region_bytes"] = self._shm_bytes()
        snap["cpu"]["region_used_bytes"] = self._shm_allocated()
        if reclaim and self.disk_limit_bytes > 0 and used > self.disk_limit_bytes:
            result = self.reclaim(now)
            snap["disk"]["used_bytes"] = result["bytes"]
            snap["disk"]["files"] = result["files"]
        snap["ts"] = now
        snap["collected_at"] = now

    def _shm_bytes(self) -> int:
        """Declared size of vLLM's CPU-tier region (proves the budget applied)."""
        total = 0
        try:
            for entry in Path("/dev/shm").glob(SHM_GLOB):
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
        except OSError:
            return self._shm_region_bytes
        self._shm_region_bytes = total
        return total

    def _shm_allocated(self) -> int:
        """Bytes actually backed in tmpfs (st_blocks), i.e. how full the pool got.

        vLLM exposes no occupancy gauge for the CPU tier: the only metric it
        publishes (kv_offload_cpu_cache_usage_perc) is the share *pinned by
        in-flight transfers*, which reads 0 % on an idle engine even when the
        pool is completely full. The region is a tmpfs-backed mmap, so its
        allocated blocks are the closest honest host-side answer — they reach
        the pool size once the tier has been filled and stay there (ARC reuses
        slots without returning pages), which makes them a high-water mark.
        """
        total = 0
        try:
            for entry in Path("/dev/shm").glob(SHM_GLOB):
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                total += getattr(stat, "st_blocks", 0) * 512
        except OSError:
            return 0
        return total

    def reclaim(self, now: float, target_ratio: float = 0.9) -> dict:
        """Delete the oldest block files until usage is back under the limit.

        The files are created by the engine (root inside the container) while the
        monitor runs unprivileged, so a plain unlink hits EACCES on the parent
        directory. Failures are therefore retried through ``sudo -n rm``; if that
        is unavailable too the error is recorded and surfaced in the UI instead
        of silently doing nothing.
        """
        target = int(self.disk_limit_bytes * target_ratio)
        used, files = dir_usage(self.directory)
        victims: list[tuple[str, int]] = []
        projected = used
        for _mtime, size, path in oldest_files(self.directory):
            if projected <= target:
                break
            victims.append((path, size))
            projected -= size
        freed = deleted = 0
        error: str | None = None
        if victims:
            freed, deleted, error = self._unlink_all(victims)
        self.snapshot["reclaim"] = {
            "last_bytes": freed, "last_files": deleted, "last_at": now,
            "error": error,
        }
        return {"bytes": used - freed, "files": files - deleted, "freed": freed}

    def _unlink_all(self, victims: list[tuple[str, int]]) -> tuple[int, int, str | None]:
        """Unlink the given (path, size) pairs; returns (freed, deleted, error)."""
        freed = deleted = 0
        pending: list[tuple[str, int]] = []
        for path, size in victims:
            try:
                os.unlink(path)
            except FileNotFoundError:
                continue
            except OSError:
                pending.append((path, size))
                continue
            freed += size
            deleted += 1
        error: str | None = None
        for start in range(0, len(pending), 200):
            chunk = pending[start:start + 200]
            try:
                proc = subprocess.run(
                    ["sudo", "-n", "rm", "-f", "--", *[p for p, _ in chunk]],
                    capture_output=True, text=True, timeout=120,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                error = detail[-1] if detail else f"sudo rm rc={proc.returncode}"
                break
            for path, size in chunk:
                if not os.path.exists(path):
                    freed += size
                    deleted += 1
        return freed, deleted, error

    # -- health --------------------------------------------------------------
    def health_due(self, now: float) -> bool:
        return not self._last_health or (now - self._last_health) >= self.health_interval_s

    def collect_health(self, now: float) -> None:
        device = self.snapshot.get("device") or self.device
        self._last_health = now
        if not device:
            self.snapshot["health"] = {**blank()["health"], "error": "no device"}
            return
        try:
            proc = subprocess.run(
                ["sudo", "-n", "nvme", "smart-log", device, "-o", "json"],
                capture_output=True, text=True, timeout=20,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.snapshot["health"] = {**blank()["health"], "device": device,
                                      "error": f"{type(exc).__name__}: {exc}"}
            return
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            self.snapshot["health"] = {**blank()["health"], "device": device,
                                       "error": detail[-1] if detail else f"rc={proc.returncode}"}
            return
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            self.snapshot["health"] = {**blank()["health"], "device": device,
                                       "error": f"bad json: {exc}"}
            return
        self.snapshot["health"] = {**parse_smart(payload), "device": device}

    # -- view ----------------------------------------------------------------
    def view(self) -> dict:
        return self.snapshot
