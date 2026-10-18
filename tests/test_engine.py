from monitor.engine import EngineStore, Rates, blank


def test_rates_min_span_suppresses_early_extrapolation():
    rates = Rates(60.0, min_span_s=10.0)
    base = 1000.0
    rates.push("gen", 0, base)
    assert rates.rate("gen", base) is None
    rates.push("gen", 100, base + 5)
    assert rates.rate("gen", base + 5) is None   # span < 10 s
    rates.push("gen", 200, base + 20)
    assert round(rates.rate("gen", base + 20), 2) == 10.0


def test_engine_store_keeps_a_one_minute_window():
    store = EngineStore(window_s=10.0)
    base = 1000.0
    for i in range(31):                          # 1 s apart, 0..30 s
        store.push_rate("generation", i * 100, base + i)
        store.push_rate("prefill", i * 40, base + i)
    assert len(store.rates.series["generation"]) <= 12
    assert len(store.rates_60.series["generation"]) == 31
    assert round(store.rates_60.rate("generation", base + 30), 1) == 100.0
    # prefill rides the same 60 s window
    assert len(store.rates_60.series["prefill"]) == 31
    assert round(store.rates_60.rate("prefill", base + 30), 1) == 40.0


def test_snapshot_exposes_both_one_minute_rates():
    speed = blank()["speed"]
    assert speed["decode_tps_60s"] is None
    assert speed["prefill_tps_60s"] is None


def test_cache_hit_ratio():
    store = EngineStore()
    assert store.cache_hit_ratio() is None
    store.snapshot["tokens"] = {"prompt": 1000, "prompt_cached": 250}
    assert store.cache_hit_ratio() == 0.25
