import math

from monitor.prom import Metrics, parse


TEXT = """
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="qwen3.8-27b"} 3.0
# HELP vllm:prefix_cache_hits_total Prefix cache hits
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="qwen3.8-27b"} 120.0
vllm:prefix_cache_queries_total{engine="0",model_name="qwen3.8-27b"} 150.0
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1"} 5
vllm:time_to_first_token_seconds_bucket{le="0.5"} 15
vllm:time_to_first_token_seconds_bucket{le="1.0"} 18
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 20
vllm:time_to_first_token_seconds_sum 8.0
vllm:time_to_first_token_seconds_count 20
"""


def test_parse_and_query():
    m = Metrics(TEXT)
    assert m.value("vllm:num_requests_running", model_name="qwen3.8-27b") == 3.0
    assert m.value("vllm:num_requests_running", model_name="nope") is None
    assert m.total("vllm:prefix_cache_hits_total") == 120.0
    assert m.has("vllm:time_to_first_token_seconds")


def test_histogram_quantile_interpolates():
    m = Metrics(TEXT)
    q50 = m.quantile("vllm:time_to_first_token_seconds", 0.5)
    assert q50 is not None
    # rank = 10 -> inside the 0.1..0.5 bucket (5 -> 15)
    assert 0.1 < q50 < 0.5
    assert m.quantile("vllm:time_to_first_token_seconds", 0.99) is not None


def test_histogram_mean():
    m = Metrics(TEXT)
    assert m.mean("vllm:time_to_first_token_seconds") == 8.0 / 20


def test_label_escaping():
    families = parse('foo{a="x\\"y",b="z"} 1\n')
    sample = families["foo"].samples[0]
    assert sample.labels == {"a": 'x"y', "b": "z"}


def test_ninfer_snapshot_is_parsed():
    from monitor.engine import EngineStore, NinferAdapter

    text = """
# TYPE ninfer_requests_running gauge
ninfer_requests_running 2
ninfer_requests_prefilling 1
ninfer_device_kv_used_page_groups{pool_id="0"} 100
ninfer_device_kv_capacity_page_groups{pool_id="0"} 400
ninfer_prompt_tokens_total 1000
ninfer_prompt_tokens_cached_total 400
ninfer_generation_tokens_total 500
ninfer_decode_tokens_total 480
ninfer_engine_ready 1
ninfer_max_concurrency 2
ninfer_max_context_tokens 262144
"""
    store = EngineStore()
    snap = NinferAdapter(store).normalize(Metrics(text), 1000.0)
    assert snap["requests"]["running"] == 2
    assert snap["requests"]["prefilling"] == 1
    assert snap["cache"]["kv_pct"] == 25.0
    assert snap["cache"]["hit_rate"] == 0.4
    assert snap["ok"] is True
    assert snap["config"]["max_context"] == 262144.0
