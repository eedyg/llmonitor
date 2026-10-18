"""Configuration loading: YAML + defaults + env overrides + path resolution."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "paths": {
        "run_llm": "../run_llm.sh",
        "run_dir": "../run",
        "state_dir": "./state",
    },
    "server": {
        "ui_host": "0.0.0.0",
        "ui_port": 9000,
        "gateway_host": "0.0.0.0",
        "gateway_port": 8000,
        "engine_port": 8001,
        "auth_token": "",
    },
    "engine": {
        "default": "vllm",
        "bind_host": "127.0.0.1",
        "container_names": {"vllm": "vllm-qwen38", "ninfer": "ninfer-qwen38"},
        "profiles": ["stage1", "mtp", "dflash", "dflash2", "kv2", "kv3"],
        "default_ctx": {"vllm": "128k", "ninfer": "256k"},
        "launcher": True,
        "stop_grace_s": 25,
        "health_timeout_s": 1800,
        "health_poll_s": 2.0,
        "ninfer_jsonl": True,
        "ninfer_jsonl_path": "../run/ninfer-requests.jsonl",
        "ninfer_stats_interval_ms": 1000,
        "extra_env": {},
    },
    "gateway": {
        "force_usage_chunk": False,
        "max_capture_bytes": 8 * 1024 * 1024,
        "connect_timeout_s": 10,
    },
    "collect": {
        "metrics_interval_s": 2.0,
        "gpu_interval_s": 1.0,
        "runtime_interval_s": 3.0,
        "rate_window_s": 10.0,
    },
    "storage": {
        "db": "./state/monitor.db",
        "ring_seconds": 7200,
        "text_keep": 200,
        "text_cap_bytes": 2 * 1024 * 1024,
        "retention_days": 30,
        # Hard safety net on top of retention_days (0 = no cap).
        "max_db_mb": 512,
    },
    "logs": {
        # Trim (keep the tail, drop the head) once a log crosses the line.
        "max_log_mb": 64,
        "keep_tail_mb": 8,
        "lifecycle_max_mb": 8,
        "jsonl_max_mb": 64,
        "check_interval_s": 30,
    },
    "billing": {
        "deepseek": {
            "model": "deepseek-flash",
            # CNY per 1M tokens, PEAK rate (2026-09-10 rate card); off-peak is
            # exactly half. Peak = Beijing time Mon-Fri 9-12, 14-18.
            "cache_hit": 0.04,
            "cache_miss": 2.0,
            "output": 8.0,
            "offpeak_factor": 0.5,
            "peak_hours": [9, 10, 11, 14, 15, 16, 17],
        },
        "electricity": {
            # Chengdu: residential tier-1 0.5224, commercial single-rate ~0.61
            # (State Grid Sichuan, 2026-10). 0.6 is a usable middle ground.
            "cny_per_kwh": 0.6,
        },
    },
    "gpu": {"fallback_csv": True},
    "docker": {
        "enabled": True,
        "image_repos": ["vllm/vllm-openai"],
        "containers": [],
        "pull_timeout_s": 3600,
    },
}


class Dot(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - config typo guard
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Dot({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(old: Any, new: str) -> Any:
    if isinstance(old, bool):
        return new.lower() in ("1", "true", "yes", "on")
    if isinstance(old, int):
        return int(new)
    if isinstance(old, float):
        return float(new)
    return new


_ENV_MAP = {
    "MONITOR_UI_HOST": ("server", "ui_host"),
    "MONITOR_UI_PORT": ("server", "ui_port"),
    "MONITOR_GATEWAY_HOST": ("server", "gateway_host"),
    "MONITOR_GATEWAY_PORT": ("server", "gateway_port"),
    "MONITOR_ENGINE_PORT": ("server", "engine_port"),
    "MONITOR_AUTH_TOKEN": ("server", "auth_token"),
    "MONITOR_RUN_LLM": ("paths", "run_llm"),
    "MONITOR_RUN_DIR": ("paths", "run_dir"),
    "MONITOR_STATE_DIR": ("paths", "state_dir"),
    "MONITOR_DB": ("storage", "db"),
    "MONITOR_TEXT_KEEP": ("storage", "text_keep"),
    "MONITOR_DB_MAX_MB": ("storage", "max_db_mb"),
    "MONITOR_LOG_MAX_MB": ("logs", "max_log_mb"),
    "MONITOR_JSONL_MAX_MB": ("logs", "jsonl_max_mb"),
    "MONITOR_KWH_PRICE": ("billing", "electricity", "cny_per_kwh"),
    "MONITOR_FORCE_USAGE_CHUNK": ("gateway", "force_usage_chunk"),
    "MONITOR_ENGINE_DEFAULT": ("engine", "default"),
    "MONITOR_DOCKER_PULL_TIMEOUT": ("docker", "pull_timeout_s"),
    "MONITOR_ENGINE_LAUNCHER": ("engine", "launcher"),
}


class Config:
    def __init__(self, path: str | Path = "config.yaml") -> None:
        self.path = Path(path).expanduser().resolve()
        self.base = self.path.parent
        raw = {}
        if self.path.exists():
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        data = _merge(DEFAULTS, raw)
        for env_key, path in _ENV_MAP.items():
            if env_key not in os.environ:
                continue
            *parents, key = path
            node = data
            for part in parents:
                node = node[part]
            node[key] = _coerce(node[key], os.environ[env_key])
        self.data = _wrap(data)

    # -- typed access -------------------------------------------------------
    @property
    def root(self) -> Dot:
        return self.data

    def p(self, dotted: str) -> Path:
        """Resolve a configured path relative to the config file."""
        node: Any = self.data
        for part in dotted.split("."):
            node = node[part]
        path = Path(str(node)).expanduser()
        if not path.is_absolute():
            path = (self.base / path).resolve()
        return path

    def ensure_dirs(self) -> None:
        for key in ("paths.state_dir", "paths.run_dir"):
            self.p(key).mkdir(parents=True, exist_ok=True)
        self.p("storage.db").parent.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict:
        return copy.deepcopy(dict(self.data))


CFG = Config()
