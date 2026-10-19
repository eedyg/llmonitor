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


def _vllm_metrics(kv_pct: float, prompt_tokens: int = 0) -> "Metrics":
    """Minimal vLLM /metrics text with a live KV gauge and a frozen counter."""
    from monitor.prom import Metrics

    return Metrics(f"""
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{{model_name="qwen3.8-27b",num_gpu_blocks="202",block_size="1600",kv_cache_size_tokens="272954",cache_dtype="fp8"}} 1
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{{model_name="qwen3.8-27b"}} {kv_pct}
# TYPE vllm:prompt_tokens_by_source counter
vllm:prompt_tokens_by_source_total{{model_name="qwen3.8-27b",source="local_compute"}} {prompt_tokens}
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total{{model_name="qwen3.8-27b"}} 0
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model_name="qwen3.8-27b"}} 1
""")


def test_prefill_rate_comes_from_kv_growth_while_the_counter_is_frozen():
    """prompt_tokens only lands when a request finishes; the KPI must not read 0."""
    from monitor.engine import VllmAdapter

    store = EngineStore(window_s=10.0)
    adapter = VllmAdapter(store)
    # 120k prefill in flight: the accounted counter stays at 0 the whole time,
    # the KV gauge climbs 0 -> 10 % -> 20 % of a 272,954-token pool.
    adapter.normalize(_vllm_metrics(0.00, 0), now=1000.0)
    adapter.normalize(_vllm_metrics(0.10, 0), now=1010.0)
    snap = adapter.normalize(_vllm_metrics(0.20, 0), now=1020.0)
    speed = snap["speed"]
    assert speed["prefill_source"] == "live"
    # 0.10 * 272954 tokens over 10 s
    assert 2600 < speed["prefill_live_tps"] < 2800
    assert speed["prefill_tps"] == speed["prefill_live_tps"]


def test_prefill_rate_prefers_the_accounted_spike_when_the_request_lands():
    from monitor.engine import VllmAdapter

    store = EngineStore(window_s=10.0)
    adapter = VllmAdapter(store)
    # KV already full and flat; only the accounted counter moves when it lands
    adapter.normalize(_vllm_metrics(0.44, 0), now=2000.0)
    snap = adapter.normalize(_vllm_metrics(0.44, 120000), now=2010.0)
    assert snap["speed"]["prefill_source"] == "accounted"
    assert snap["speed"]["prefill_tps"] == 12000.0
