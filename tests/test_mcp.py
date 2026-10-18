"""The MCP layer is a thin adapter: check the shapes it hands to a client."""
import monitor.mcp_server as srv


def _snapshot():
    scope = {
        "requests": 3, "total": 1000, "prompt": 800, "prompt_cached": 600,
        "prompt_miss": 200, "output": 200, "thinking": 150, "content": 50,
        "cache_hit_pct": 75.0, "cache_estimated": True,
        "first_at": 1.0, "last_at": 2.0,
    }
    money = {
        "deepseek": {"cost": 0.6, "peak_price": 1.2, "offpeak_price": 0.6,
                     "peak_surcharge": 0.0, "peak_token_pct": 0.0},
        "energy": {"kwh": 0.5, "cost": 0.3, "seconds": 3600, "since": "2026-10-19"},
        "total": 0.9,
    }
    return {
        "controller": {"state": "healthy", "engine": "vllm"},
        "engine": {"model": {"served": "m"}, "requests": {}, "tokens": {"prompt": 1},
                   "cache": {}, "speed": {"decode_tps": 1.0, "decode_tps_60s": 2.0,
                                          "prefill_tps_60s": 3.0},
                   "spec": {}},
        "gpu": {"gpus": [{"name": "RTX", "power_w": 250}]},
        "live": [],
        "tokens": {"session": dict(scope, requests=1), "history": scope},
        "billing": {"price": {"currency": "CNY", "model": "deepseek-flash"},
                    "session": money, "history": money},
    }


def test_status_carries_ledger_and_money(monkeypatch):
    monkeypatch.setattr(srv, "_get", lambda path, **kw: _snapshot())
    status = srv.llm_status()
    ledger = status["ledger"]
    assert ledger["total_tokens"] == 1000
    assert ledger["session_total_tokens"] == 1000
    assert ledger["deepseek_cny"] == 0.6
    assert ledger["energy_kwh"] == 0.5
    assert ledger["electricity_cny"] == 0.3
    # the 1 min rates ride along with the engine speed block
    assert status["speed"]["decode_tps_60s"] == 2.0
    assert status["speed"]["prefill_tps_60s"] == 3.0


def test_ledger_without_range_uses_the_snapshot(monkeypatch):
    monkeypatch.setattr(srv, "_get", lambda path, **kw: _snapshot())
    out = srv.llm_ledger()
    assert out["history"]["total"] == 1000
    assert out["session"]["requests"] == 1
    assert out["billing"]["deepseek"]["cost"] == 0.6


def test_ledger_and_cost_with_a_range(monkeypatch):
    def fake_get(path, **params):
        assert path == "/api/tokens"
        assert params == {"start": "2026-10-17", "end": "2026-10-19"}
        row = {
            "date": "2026-10-18", "requests": 2, "total": 500, "prompt": 400,
            "prompt_cached": 300, "prompt_miss": 100, "output": 100,
            "thinking": 60, "content": 40,
            "billing": {
                "deepseek": {"cost": 0.25}, "energy": {"kwh": 0.4, "cost": 0.24},
                "total": 0.49,
            },
        }
        return {
            "start": "2026-10-17", "end": "2026-10-19", "days": 1,
            "scope": {"total": 500}, "daily": [row],
            "billing": {"price": {"model": "deepseek-flash"},
                        "scope": {"deepseek": {"cost": 0.25},
                                  "energy": {"kwh": 0.4, "cost": 0.24},
                                  "total": 0.49}},
        }

    monkeypatch.setattr(srv, "_get", fake_get)
    ledger = srv.llm_ledger(start="2026-10-17", end="2026-10-19")
    assert ledger["days"] == 1
    assert ledger["scope"]["total"] == 500
    assert ledger["daily"][0]["deepseek_cny"] == 0.25
    assert ledger["daily"][0]["electricity_cny"] == 0.24

    cost = srv.llm_cost(start="2026-10-17", end="2026-10-19")
    assert cost["cost"]["total"] == 0.49
    assert cost["per_day"][0]["energy_cny"] == 0.24


def test_metrics_sections(monkeypatch):
    monkeypatch.setattr(srv, "_get", lambda path, **kw: _snapshot())
    assert srv.llm_metrics("ledger")["ledger"]["history"]["total"] == 1000
    assert srv.llm_metrics("billing")["billing"]["history"]["total"] == 0.9
    unknown = srv.llm_metrics("nope")
    assert "ledger" in unknown["available"] and "billing" in unknown["available"]
