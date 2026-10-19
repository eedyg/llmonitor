"""What the recorded tokens would have cost, and what the GPU burned to make them.

Two independent numbers, both read-only and both approximations:

**API equivalent.** DeepSeek publishes ``deepseek-flash`` in CNY per 1M tokens
(peak price; off-peak is half):

================  ========  ========
                  高峰      空闲
================  ========  ========
输入·缓存命中      0.04 元   0.02 元
输入·缓存未命中    2.00 元   1.00 元
输出              8.00 元   4.00 元
================  ========  ========

Peak = Beijing time Mon-Fri 09:00-12:00 and 14:00-18:00 (Chinese public holidays
are *not* modelled — ~10 days a year, so the number is off by at most a few
percent). We split the request rows by that predicate in SQL, so the mixed cost
is exact for the rate card:

    cost = k·P·total + (1-k)·P·peak        (k = offpeak_factor = 0.5)

**Electricity.** GPU power (``nvidia-smi`` / NVML) integrated over wall time,
per local day. Only the card's own draw — no CPU/PSU losses, no chassis fans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PER_MILLION = 1_000_000.0


@dataclass(frozen=True)
class DeepSeekPrice:
    cache_hit: float
    cache_miss: float
    output: float
    offpeak_factor: float = 0.5

    def money(self, total: float, peak: float, unit_price: float) -> float:
        """Cost of ``total`` tokens of which ``peak`` ran inside peak hours."""
        k = self.offpeak_factor
        return unit_price * (k * total + (1.0 - k) * peak) / PER_MILLION


def _price(cfg: Any) -> DeepSeekPrice:
    return DeepSeekPrice(
        cache_hit=float(cfg.cache_hit),
        cache_miss=float(cfg.cache_miss),
        output=float(cfg.output),
        offpeak_factor=float(cfg.offpeak_factor),
    )


def token_cost(price: DeepSeekPrice, scope: dict) -> dict:
    """DeepSeek-flash equivalent cost for one token scope."""
    peak = scope.get("peak") or {}
    hit = float(scope.get("prompt_cached") or 0)
    miss = float(scope.get("prompt_miss") or 0)
    out = float(scope.get("output") or 0)
    peak_hit = float(peak.get("prompt_cached") or 0)
    peak_miss = float(peak.get("prompt_miss") or 0)
    peak_out = float(peak.get("output") or 0)

    cost = (
        price.money(hit, peak_hit, price.cache_hit)
        + price.money(miss, peak_miss, price.cache_miss)
        + price.money(out, peak_out, price.output)
    )
    peak_price = (hit * price.cache_hit + miss * price.cache_miss + out * price.output) / PER_MILLION
    offpeak_price = peak_price * price.offpeak_factor
    peak_tokens = peak_hit + peak_miss + peak_out
    total_tokens = hit + miss + out
    return {
        "cost": round(cost, 4),
        "peak_price": round(peak_price, 4),        # if everything ran at peak rate
        "offpeak_price": round(offpeak_price, 4),  # if everything ran off-peak
        "peak_surcharge": round(cost - offpeak_price, 4),
        "peak_token_pct": round(100.0 * peak_tokens / total_tokens, 1) if total_tokens else None,
        "cache_unknown": bool(scope.get("cache_unknown")),
    }


def energy_cost(cny_per_kwh: float, energy: dict | None) -> dict:
    kwh = float((energy or {}).get("kwh") or 0.0)
    return {
        "kwh": round(kwh, 4),
        "cost": round(kwh * float(cny_per_kwh), 4),
        "seconds": (energy or {}).get("seconds"),
        "since": (energy or {}).get("since"),
    }


def scope_report(billing_cfg: Any, scope: dict, energy: dict | None) -> dict:
    price = _price(billing_cfg.deepseek)
    ds = token_cost(price, scope)
    power = energy_cost(float(billing_cfg.electricity.cny_per_kwh), energy)
    return {
        "deepseek": ds,
        "energy": power,
        "total": round(ds["cost"] + power["cost"], 4),
    }


def price_block(billing_cfg: Any) -> dict:
    cfg = billing_cfg.deepseek
    return {
        "currency": "CNY",
        "model": str(cfg.model),
        "per_million": {
            "cache_hit": float(cfg.cache_hit),
            "cache_miss": float(cfg.cache_miss),
            "output": float(cfg.output),
        },
        "offpeak_factor": float(cfg.offpeak_factor),
        "peak_hours": [int(h) for h in cfg.peak_hours],
        "peak_timezone": "Asia/Shanghai (UTC+8)",
        "cny_per_kwh": float(billing_cfg.electricity.cny_per_kwh),
    }


def snapshot_report(billing_cfg: Any, tokens: dict, energy: dict) -> dict:
    """Cost per token scope for the WS/REST snapshot."""
    return {
        "price": price_block(billing_cfg),
        "session": scope_report(billing_cfg, tokens.get("session") or {},
                                (energy or {}).get("session")),
        "history": scope_report(billing_cfg, tokens.get("history") or {},
                                (energy or {}).get("history")),
    }


def range_report(billing_cfg: Any, data: dict, kwh_by_day: dict[str, float]) -> dict:
    """Cost for a date range: one scope + one figure per day."""
    daily = data.get("daily") or []
    total_kwh = 0.0
    for row in daily:
        kwh = float(kwh_by_day.get(row["date"]) or 0.0)
        total_kwh += kwh
        report = scope_report(billing_cfg, row, {"kwh": kwh})
        row["billing"] = report
    scope = scope_report(billing_cfg, data.get("scope") or {}, {"kwh": total_kwh})
    return {"price": price_block(billing_cfg), "scope": scope}
