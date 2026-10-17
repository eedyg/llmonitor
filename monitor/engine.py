"""Engine adapters: scrape Prometheus from vLLM / NInfer and normalize it.

The whole UI/API/MCP only ever sees the normalized shape produced here.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from .prom import Metrics


@dataclass
class Rates:
    """Counter -> tok/s over a sliding window."""

    window_s: float = 10.0
    series: dict[str, deque] = field(default_factory=dict)

    def push(self, key: str, value: float | None, now: float) -> None:
        if value is None:
            return
        q = self.series.setdefault(key, deque())
        q.append((now, value))
        while len(q) > 2 and now - q[1][0] >= self.window_s:
            q.popleft()

    def rate(self, key: str, now: float) -> float | None:
        q = self.series.get(key)
        if not q or len(q) < 2:
            return None
        t0, v0 = q[0]
        t1, v1 = q[-1]
        dt = t1 - t0
        if dt <= 0.2:
            return None
        return max(0.0, (v1 - v0) / dt)


def _f(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _ms(value: float | None) -> float | None:
    return None if value is None else round(value * 1000.0, 2)


def _avg_hist(m: Metrics, name: str) -> float | None:
    hist = m.histogram(name)
    if not hist:
        return None
    _b, total, count = hist
    return total / count if count > 0 else None


EMPTY = {
    "model": {"name": None, "served": None, "engine": None, "version": None},
    "requests": {"running": 0, "waiting": 0, "paused": 0, "prefilling": 0,
                 "decode_ready": 0, "replaying": 0, "preemptions": 0},
    "tokens": {"prompt": 0, "prompt_cached": 0, "prefill": 0, "generation": 0,
               "decode": 0, "replayed": 0},
    "cache": {"kv_pct": 0.0, "kv_used_pages": 0, "kv_capacity_pages": 0,
              "prefix_queries": 0, "prefix_hits": 0, "hit_rate": 0.0,
              "host_used_bytes": 0, "host_capacity_bytes": 0, "pools": []},
    "speed": {"decode_tps": None, "prefill_tps": None, "ttft_p50_ms": None,
              "ttft_p90_ms": None, "ttft_p99_ms": None, "queue_p50_ms": None,
              "e2e_p50_ms": None, "e2e_p90_ms": None, "itl_p50_ms": None,
              "prefill_mean_ms": None, "decode_mean_ms": None,
              "prompt_tokens_avg": None},
    "spec": {"backend": None, "draft_window": None, "rounds": 0, "drafted": 0,
             "accepted": 0, "accept_rate": 0.0, "fallback": 0},
    "config": {},
    "ok": False,
    "error": None,
    "scraped_at": 0.0,
}


def blank() -> dict:
    import copy

    return copy.deepcopy(EMPTY)


class EngineStore:
    """Holds the latest snapshot plus derived series."""

    def __init__(self, window_s: float = 10.0) -> None:
        self.rates = Rates(window_s)
        self.snapshot: dict = blank()
        self.kind: str | None = None
        self.last_error: str | None = None
        self.last_ok: float = 0.0
        self.raw_text: str = ""
        self.headers: dict[str, str] = {}
        self.models: dict = {}
        self.failures: int = 0


class EngineAdapter:
    kind = "unknown"

    def __init__(self, state: EngineStore) -> None:
        self.state = state

    # -- HTTP ---------------------------------------------------------------
    async def fetch(self, client: httpx.AsyncClient, base: str, headers: dict[str, str]) -> Metrics | None:
        try:
            resp = await client.get(f"{base}/metrics", headers=headers, timeout=5.0)
            if resp.status_code != 200:
                self.state.last_error = f"/metrics HTTP {resp.status_code}"
                return None
            self.state.raw_text = resp.text
            metrics = Metrics(resp.text)
            if not metrics:
                self.state.last_error = "/metrics returned no samples"
                return None
            self.state.last_error = None
            self.state.last_ok = time.time()
            return metrics
        except Exception as exc:
            self.state.last_error = f"{type(exc).__name__}: {exc}"
            return None

    # -- normalize ----------------------------------------------------------
    def normalize(self, m: Metrics, now: float) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def detect(self, text: str) -> bool:
        return False


class VllmAdapter(EngineAdapter):
    kind = "vllm"

    def detect(self, text: str) -> bool:
        return "vllm:" in text

    def normalize(self, m: Metrics, now: float) -> dict:
        snap = blank()
        st = self.state
        info = m.value("vllm:cache_config_info") or 0.0
        cfg_labels = {}
        fam = m.families.get("vllm:cache_config_info")
        if fam and fam.samples:
            cfg_labels = fam.samples[0].labels
        model_name = cfg_labels.get("model_name") or m.label_values("vllm:num_requests_running", "model_name")
        if isinstance(model_name, list):
            model_name = model_name[0] if model_name else None

        prompt = m.value("vllm:prompt_tokens_total")
        cached = m.value("vllm:prompt_tokens_cached_total")
        local_compute = m.value("vllm:prompt_tokens_by_source_total", source="local_compute")
        generation = m.value("vllm:generation_tokens_total")
        st.rates.push("generation", generation, now)
        st.rates.push("prefill", local_compute if local_compute is not None else prompt, now)
        st.rates.push("prompt", prompt, now)

        queries = m.value("vllm:prefix_cache_queries_total") or 0.0
        hits = m.value("vllm:prefix_cache_hits_total") or 0.0

        snap["model"] = {
            "name": model_name,
            "served": model_name,
            "engine": "vllm",
            "version": None,
            "owned_by": "vllm",
        }
        snap["requests"] = {
            "running": int(m.value("vllm:num_requests_running") or 0),
            "waiting": int(m.value("vllm:num_requests_waiting") or 0),
            "paused": 0,
            "prefilling": 0,
            "decode_ready": int(m.value("vllm:num_requests_running") or 0),
            "replaying": 0,
            "preemptions": int(m.value("vllm:num_preemptions_total") or 0),
            "waiting_by_reason": {
                s.labels.get("reason"): int(s.value)
                for s in m.samples("vllm:num_requests_waiting_by_reason")
            },
        }
        snap["tokens"] = {
            "prompt": int(prompt or 0),
            "prompt_cached": int(cached or 0),
            "prefill": int(local_compute if local_compute is not None else (prompt or 0)),
            "generation": int(generation or 0),
            "decode": int(generation or 0),
            "replayed": 0,
        }
        blocks = int(cfg_labels.get("num_gpu_blocks") or 0)
        kv_pct = round(100.0 * (m.value("vllm:kv_cache_usage_perc") or 0.0), 2)
        snap["cache"] = {
            "kv_pct": kv_pct,
            # vLLM only exposes a usage ratio; derive the page count so the UI
            # does not show "N/N pages" (which reads as 100% full).
            "kv_used_pages": int(round(kv_pct / 100.0 * blocks)),
            "kv_capacity_pages": blocks,
            "prefix_queries": int(queries),
            "prefix_hits": int(hits),
            "hit_rate": round(hits / queries, 4) if queries else 0.0,
            "host_used_bytes": 0,
            "host_capacity_bytes": 0,
            "pools": [],
            "source": {
                s.labels.get("source"): int(s.value)
                for s in m.samples("vllm:prompt_tokens_by_source_total")
            },
        }
        snap["speed"] = {
            "decode_tps": _round(st.rates.rate("generation", now), 2),
            "prefill_tps": _round(st.rates.rate("prefill", now), 2),
            "ttft_p50_ms": _ms(m.quantile("vllm:time_to_first_token_seconds", 0.5)),
            "ttft_p90_ms": _ms(m.quantile("vllm:time_to_first_token_seconds", 0.9)),
            "ttft_p99_ms": _ms(m.quantile("vllm:time_to_first_token_seconds", 0.99)),
            "queue_p50_ms": _ms(m.quantile("vllm:request_queue_time_seconds", 0.5)),
            "e2e_p50_ms": _ms(m.quantile("vllm:e2e_request_latency_seconds", 0.5)),
            "e2e_p90_ms": _ms(m.quantile("vllm:e2e_request_latency_seconds", 0.9)),
            "itl_p50_ms": _ms(m.quantile("vllm:inter_token_latency_seconds", 0.5)),
            "prefill_mean_ms": _ms(_avg_hist(m, "vllm:request_prefill_time_seconds")),
            "decode_mean_ms": _ms(_avg_hist(m, "vllm:request_decode_time_seconds")),
            "prompt_tokens_avg": _round(_avg_hist(m, "vllm:request_prompt_tokens"), 1),
        }
        drafted = m.value("vllm:spec_decode_num_draft_tokens_total") or 0.0
        accepted = m.value("vllm:spec_decode_num_accepted_tokens_total") or 0.0
        snap["spec"] = {
            "backend": cfg_labels.get("speculative_backend") or ("mtp" if drafted else None),
            "draft_window": _f(m.value("vllm:spec_decode_num_draft_tokens_per_pos_count")),
            "rounds": int(m.value("vllm:spec_decode_num_draft_tokens_total") or 0),
            "drafted": int(drafted),
            "accepted": int(accepted),
            "accept_rate": round(accepted / drafted, 4) if drafted else 0.0,
            "fallback": 0,
        }
        snap["config"] = {
            "kv_dtype": cfg_labels.get("cache_dtype"),
            "block_size": _f(cfg_labels.get("block_size")),
            "kv_tokens": _f(cfg_labels.get("kv_cache_size_tokens")),
            "gpu_blocks": _f(cfg_labels.get("num_gpu_blocks")),
            "gpu_mem_util": _f(cfg_labels.get("gpu_memory_utilization")),
            "prefix_caching": cfg_labels.get("enable_prefix_caching"),
            "mamba_ssm_cache_dtype": cfg_labels.get("mamba_ssm_cache_dtype"),
            "max_concurrency": _f(cfg_labels.get("kv_cache_max_concurrency")),
            "sleep_state": next(
                (s.labels.get("sleep_state") for s in m.samples("vllm:engine_sleep_state") if s.value >= 1),
                None,
            ),
            "finished": {
                s.labels.get("finished_reason"): int(s.value)
                for s in m.samples("vllm:request_success_total")
            },
        }
        snap["ok"] = True
        snap["scraped_at"] = now
        return snap


class NinferAdapter(EngineAdapter):
    kind = "ninfer"

    def detect(self, text: str) -> bool:
        return "ninfer_" in text

    def normalize(self, m: Metrics, now: float) -> dict:
        snap = blank()
        st = self.state
        fam = m.families.get("ninfer_model_info")
        labels = fam.samples[0].labels if fam and fam.samples else {}
        model_name = labels.get("model_name")

        prompt = m.value("ninfer_prompt_tokens_total")
        cached = m.value("ninfer_prompt_tokens_cached_total")
        prefill = m.value("ninfer_prefill_tokens_total")
        generation = m.value("ninfer_generation_tokens_total")
        decode = m.value("ninfer_decode_tokens_total")
        st.rates.push("generation", generation, now)
        st.rates.push("prefill", prefill, now)
        st.rates.push("decode", decode, now)

        pools = []
        for sample in m.samples("ninfer_device_kv_used_page_groups"):
            pool = sample.labels.get("pool_id")
            cap = m.value("ninfer_device_kv_capacity_page_groups", pool_id=pool) or 0
            pools.append(
                {
                    "id": pool,
                    "used": int(sample.value),
                    "capacity": int(cap),
                    "pct": round(100.0 * sample.value / cap, 2) if cap else 0.0,
                }
            )
        used_pages = sum(p["used"] for p in pools)
        cap_pages = sum(p["capacity"] for p in pools)

        drafted = m.value("ninfer_spec_decode_draft_tokens_total") or 0.0
        accepted = m.value("ninfer_spec_decode_accepted_tokens_total") or 0.0
        backend = labels.get("speculative_backend")
        if backend in ("none", ""):
            backend = None

        snap["model"] = {
            "name": model_name,
            "served": model_name,
            "engine": "ninfer",
            "version": labels.get("version"),
            "owned_by": "ninfer",
        }
        snap["requests"] = {
            "running": int(m.value("ninfer_requests_running") or 0),
            "waiting": int(m.value("ninfer_requests_waiting") or 0),
            "paused": int(m.value("ninfer_requests_paused") or 0),
            "prefilling": int(m.value("ninfer_requests_prefilling") or 0),
            "decode_ready": int(m.value("ninfer_requests_decode_ready") or 0),
            "replaying": int(m.value("ninfer_requests_replaying") or 0),
            "preemptions": int(m.value("ninfer_preemptions_total") or 0),
        }
        snap["tokens"] = {
            "prompt": int(prompt or 0),
            "prompt_cached": int(cached or 0),
            "prefill": int(prefill or 0),
            "generation": int(generation or 0),
            "decode": int(decode or 0),
            "replayed": int(m.value("ninfer_replayed_tokens_total") or 0),
        }
        snap["cache"] = {
            "kv_pct": round(100.0 * used_pages / cap_pages, 2) if cap_pages else 0.0,
            "kv_used_pages": used_pages,
            "kv_capacity_pages": cap_pages,
            "prefix_queries": int(prompt or 0),
            "prefix_hits": int(cached or 0),
            "hit_rate": round(cached / prompt, 4) if prompt else 0.0,
            "host_used_bytes": int(m.value("ninfer_host_context_used_bytes") or 0),
            "host_capacity_bytes": int(m.value("ninfer_host_context_capacity_bytes") or 0),
            "host_peak_bytes": int(m.value("ninfer_host_context_peak_bytes") or 0),
            "pools": pools,
            "state_slots": {
                "used": int(m.value("ninfer_device_state_used_slots") or 0),
                "capacity": int(m.value("ninfer_device_state_capacity_slots") or 0),
            },
        }
        snap["speed"] = {
            "decode_tps": _round(st.rates.rate("decode", now), 2),
            "prefill_tps": _round(st.rates.rate("prefill", now), 2),
            "ttft_p50_ms": _ms(m.quantile("ninfer_time_to_first_token_seconds", 0.5)),
            "ttft_p90_ms": _ms(m.quantile("ninfer_time_to_first_token_seconds", 0.9)),
            "ttft_p99_ms": _ms(m.quantile("ninfer_time_to_first_token_seconds", 0.99)),
            "queue_p50_ms": _ms(m.quantile("ninfer_request_queue_seconds", 0.5)),
            "e2e_p50_ms": _ms(m.quantile("ninfer_request_duration_seconds", 0.5)),
            "e2e_p90_ms": _ms(m.quantile("ninfer_request_duration_seconds", 0.9)),
            "itl_p50_ms": _itl(st.rates.rate("decode", now), now),
            "prefill_mean_ms": None,
            "decode_mean_ms": None,
            "prompt_tokens_avg": None,
        }
        snap["spec"] = {
            "backend": backend,
            "draft_window": _f(m.value("ninfer_spec_decode_draft_window")),
            "rounds": int(m.value("ninfer_spec_decode_rounds_total") or 0),
            "drafted": int(drafted),
            "accepted": int(accepted),
            "accept_rate": round(accepted / drafted, 4) if drafted else 0.0,
            "fallback": int(m.value("ninfer_spec_decode_fallback_steps_total") or 0),
        }
        finished = {
            s.labels.get("outcome"): int(s.value) for s in m.samples("ninfer_requests_total")
        }
        snap["config"] = {
            "max_concurrency": _f(m.value("ninfer_max_concurrency")),
            "max_context": _f(m.value("ninfer_max_context_tokens")),
            "draft_window": _f(m.value("ninfer_spec_decode_draft_window")),
            "state_slots": _f(m.value("ninfer_device_state_capacity_slots")),
            "host_capacity_bytes": _f(m.value("ninfer_host_context_capacity_bytes")),
            "engine_ready": _f(m.value("ninfer_engine_ready")),
            "finished": finished,
            "response_failures": int(m.value("ninfer_response_failures_total") or 0),
        }
        snap["ok"] = bool(m.value("ninfer_engine_ready") or 0)
        snap["scraped_at"] = now
        return snap


def _itl(decode_tps: float | None, now: float) -> float | None:
    if not decode_tps or decode_tps <= 0:
        return None
    return round(1000.0 / decode_tps, 2)


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


ADAPTERS: dict[str, type[EngineAdapter]] = {"vllm": VllmAdapter, "ninfer": NinferAdapter}


def adapter_for(kind: str | None, state: EngineStore) -> EngineAdapter:
    cls = ADAPTERS.get(kind or "", EngineAdapter)
    return cls(state)


def detect_kind(text: str) -> str | None:
    if "ninfer_" in text:
        return "ninfer"
    if "vllm:" in text:
        return "vllm"
    return None
