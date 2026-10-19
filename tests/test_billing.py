from datetime import datetime, timedelta, timezone

import pytest

from monitor import billing
from monitor.config import Config
from monitor.store import Store
from monitor.tap import CHAT, Record

BJ = timezone(timedelta(hours=8))


def cfg():
    return Config("/nonexistent/monitor-test-config.yaml").root.billing


def scope(hit=0, miss=0, out=0, peak_hit=0, peak_miss=0, peak_out=0):
    return {
        "prompt_cached": hit,
        "prompt_miss": miss,
        "output": out,
        "peak": {"prompt_cached": peak_hit, "prompt_miss": peak_miss,
                 "output": peak_out},
    }


def test_offpeak_is_half_of_peak():
    price = billing.DeepSeekPrice(cache_hit=0.04, cache_miss=2.0, output=8.0)
    # 1M cached input + 1M output, all off-peak
    off = billing.token_cost(price, scope(hit=1_000_000, out=1_000_000))
    assert off["cost"] == pytest.approx(0.02 + 4.0)
    assert off["peak_price"] == pytest.approx(0.04 + 8.0)
    assert off["peak_surcharge"] == pytest.approx(0.0)
    # same tokens entirely inside peak hours
    peak = billing.token_cost(price, scope(hit=1_000_000, out=1_000_000,
                                          peak_hit=1_000_000, peak_out=1_000_000))
    assert peak["cost"] == pytest.approx(off["peak_price"])
    assert peak["peak_token_pct"] == 100.0
    # half and half
    half = billing.token_cost(price, scope(hit=1_000_000, out=1_000_000,
                                          peak_hit=500_000, peak_out=500_000))
    assert half["cost"] == pytest.approx((off["cost"] + peak["cost"]) / 2)


def test_miss_is_charged_at_the_miss_rate():
    price = billing.DeepSeekPrice(cache_hit=0.04, cache_miss=2.0, output=8.0)
    cost = billing.token_cost(price, scope(miss=1_000_000))
    # all off-peak -> half of the 2 元/M peak rate
    assert cost["cost"] == pytest.approx(1.0)
    assert cost["peak_token_pct"] == 0.0


def test_store_splits_peak_hours_in_beijing_time(tmp_path):
    store = Store(tmp_path / "m.db")
    monday = datetime(2026, 10, 5, tzinfo=BJ)          # a Monday
    assert monday.weekday() == 0

    def add(prompt, at):
        rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
        rec.started_at = at.timestamp()
        rec.status = 200
        rec.prompt_tokens = prompt
        rec.content_tokens = 10
        store.add(rec)

    add(100, monday.replace(hour=10, minute=30))       # peak (Mon 10:30 北京)
    add(200, monday.replace(hour=20, minute=30))       # off-peak (Mon 20:30)
    add(400, monday.replace(hour=10, minute=30) + timedelta(days=5))  # Saturday
    add(800, monday.replace(hour=13, minute=0))        # lunch break: off-peak

    totals = store.token_totals()
    history = totals["history"]
    assert history["prompt"] == 1500
    assert history["peak"]["prompt"] == 100
    assert history["peak"]["output"] == 10
    # the mixed cost obeys peak_price > cost > offpeak_price
    price = billing.DeepSeekPrice(cache_hit=0.04, cache_miss=2.0, output=8.0)
    cost = billing.token_cost(price, history)
    assert cost["offpeak_price"] < cost["cost"] < cost["peak_price"]

    # the same split comes back per day for the range view
    day = monday.date().isoformat()
    daily = store.token_range(monday.timestamp(), (monday + timedelta(days=1)).timestamp())["daily"]
    assert daily[0]["date"] == day
    assert daily[0]["peak"]["prompt"] == 100


def test_energy_ledger_integrates_gpu_power(tmp_path):
    store = Store(tmp_path / "m.db")
    for _ in range(10):
        store.record_energy(300.0, 1.0)                 # 300 W for 10 s
    session = store.energy_totals()["session"]
    assert session["kwh"] == pytest.approx(300.0 * 10 / 3600 / 1000, abs=2e-6)
    assert session["seconds"] == pytest.approx(10.0)

    # garbage is ignored, absurd gaps are clamped to 10 s
    store.record_energy(None, 1.0)
    store.record_energy(-5.0, 1.0)
    store.record_energy(300.0, 0.0)
    before = store.energy_totals()["session"]["kwh"]
    store.record_energy(1000.0, 86400.0)
    after = store.energy_totals()["session"]["kwh"]
    assert after - before == pytest.approx(1000.0 * 10 / 3600 / 1000, abs=2e-6)

    history = store.energy_totals()["history"]
    assert history["kwh"] > 0 and history["days"] == 1
    today = datetime.now().date().isoformat()
    assert store.energy_by_day(today, today).get(today, 0) > 0
    assert store.energy_by_day("1999-01-01", "1999-01-02") == {}


def test_range_report_adds_money_per_day(tmp_path):
    store = Store(tmp_path / "m.db")
    monday = datetime(2026, 10, 5, tzinfo=BJ)
    rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
    rec.started_at = monday.replace(hour=10, minute=30).timestamp()
    rec.status = 200
    rec.prompt_tokens = 1_000_000
    rec.content_tokens = 1_000_000
    store.add(rec)
    # energy for a day inside the range (record_energy always books "today")
    with store._lock:
        store.db.execute("INSERT INTO gpu_energy (day, wh, seconds) VALUES ('2026-10-05',250.0,3600)")
        store.db.commit()

    data = store.token_range(monday.timestamp(),
                             (monday + timedelta(days=2)).timestamp())
    report = billing.range_report(cfg(), data, store.energy_by_day("2026-10-01", "2026-10-31"))
    money = report["scope"]
    assert money["deepseek"]["cost"] > 0
    assert money["energy"]["kwh"] == pytest.approx(0.25, rel=1e-6)
    assert money["energy"]["cost"] == pytest.approx(0.15, rel=1e-6)
    assert money["total"] == pytest.approx(money["deepseek"]["cost"] + 0.15, rel=1e-6)
    assert data["daily"][0]["billing"]["deepseek"]["cost"] > 0


def test_snapshot_report_shape(tmp_path):
    store = Store(tmp_path / "m.db")
    tokens = store.token_totals()
    energy = store.energy_totals()
    report = billing.snapshot_report(cfg(), tokens, energy)
    assert report["price"]["currency"] == "CNY"
    assert report["price"]["model"] == "deepseek-flash"
    for key in ("session", "history"):
        assert set(report[key]) == {"deepseek", "energy", "total"}
    assert report["history"]["total"] == 0


def test_unknown_cache_ratio_is_flagged_in_the_cost():
    price = billing.DeepSeekPrice(cache_hit=0.04, cache_miss=2.0, output=8.0)
    cost = billing.token_cost(price, {"prompt_miss": 1_000_000, "cache_unknown": True})
    assert cost["cache_unknown"] is True
    assert cost["cost"] == pytest.approx(1.0)          # upper bound, off-peak
