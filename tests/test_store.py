import time

import pytest

from monitor.store import Store
from monitor.tap import CHAT, Record


def make_record(i: int, text: str = "hello") -> Record:
    rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
    rec.started_at = time.time() - i
    rec.finished_at = rec.started_at + 0.5
    rec.status = 200
    rec.prompt_tokens = 10
    rec.content_tokens = 5
    rec.content_text = text
    return rec


def test_history_and_text_pruning(tmp_path):
    store = Store(tmp_path / "m.db", text_keep=2)
    for i in range(5):
        store.add(make_record(i, f"answer-{i}"))
    rows = store.history(limit=10)["items"]
    assert len(rows) == 5
    # only the newest two keep the text
    with_text = [r for r in rows if r["has_text"]]
    assert len(with_text) == 2
    newest = rows[0]["id"]
    assert store.get(newest)["content_text"] == "answer-0"
    oldest = rows[-1]["id"]
    assert store.get(oldest)["content_text"] in (None, "")


def test_engine_record_merge(tmp_path):
    store = Store(tmp_path / "m.db")
    rec = make_record(0)
    rec.engine = "ninfer"
    rec.engine_request_id = "req_abc"
    store.add(rec)
    ok = store.attach_engine_record(
        "req_abc",
        {
            "event": "request_done",
            "result": {"prompt_tokens": 55, "completion_tokens": 23, "model_thinking_tokens": 7,
                       "prefix_cache_hit_tokens": 32},
            "timings_seconds": {"ttft": 0.061, "decode": 0.187},
        },
    )
    assert ok
    row = store.get(rec.id)
    assert row["prompt_tokens"] == 55
    assert row["prompt_cached"] == 32
    assert row["thinking_tokens"] == 7
    assert row["content_tokens"] == 16
    assert row["ttft_ms"] == 61.0
    assert row["engine_record"]["event"] == "request_done"


def test_series_ring(tmp_path):
    store = Store(tmp_path / "m.db")
    now = time.time()
    for i in range(3):
        store.push_sample({"ts": now + i, "decode_tps": i, "kv_pct": 1.0})
    data = store.series(since=now - 1)
    assert data["ts"] == [round(now, 3), round(now + 1, 3), round(now + 2, 3)]
    assert data["decode_tps"] == [0, 1, 2]


def test_token_totals_ledger(tmp_path):
    store = Store(tmp_path / "m.db")

    def add(prompt, cached, thinking, content, started_at):
        rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
        rec.started_at = started_at
        rec.finished_at = started_at + 0.2
        rec.status = 200
        rec.prompt_tokens = prompt
        rec.prompt_cached = cached
        rec.thinking_tokens = thinking
        rec.content_tokens = content
        store.add(rec)

    now = time.time()
    add(100, 60, 30, 70, now)
    add(50, 0, 0, 25, now + 1)
    add(10, 0, 0, 5, store.started_at - 100)  # predates this process
    store.started_at -= 60  # pretend the process has been up for a minute

    totals = store.token_totals()
    session, history = totals["session"], totals["history"]
    assert session["requests"] == 2                      # session = since start
    assert (session["prompt"], session["prompt_cached"], session["prompt_miss"]) == (150, 60, 90)
    assert (session["output"], session["thinking"], session["content"]) == (125, 30, 95)
    assert session["total"] == 275                       # in + out
    assert session["cache_hit_pct"] == 40.0
    assert session["cache_estimated"] is False
    assert session["uptime_s"] >= 59 and session["tps"] == 4.58
    assert history["requests"] == 3                      # everything retained
    assert history["total"] == 290
    assert history["cache_hit_pct"] == 37.5


def test_token_totals_estimate_cache_when_engine_omits_it(tmp_path):
    store = Store(tmp_path / "m.db")
    rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
    rec.started_at = time.time()
    rec.status = 200
    rec.prompt_tokens = 1000
    rec.prompt_cached = None  # vLLM did not report cached_tokens
    rec.content_tokens = 10
    store.add(rec)

    scope = store.token_totals(cache_ratio=0.8)["session"]
    assert scope["prompt_cached"] == 800
    assert scope["prompt_miss"] == 200
    assert scope["cache_estimated"] is True

    scope = store.token_totals(cache_ratio=None)["session"]
    assert scope["prompt_cached"] == 0
    assert scope["cache_estimated"] is False


def test_token_range_per_day(tmp_path):
    from datetime import date, timedelta

    from monitor.store import day_window

    store = Store(tmp_path / "m.db")
    day1 = date(2026, 10, 1)
    day2 = day1 + timedelta(days=1)
    day4 = day1 + timedelta(days=3)
    s1, _ = day_window(day1.isoformat(), day1.isoformat())
    s2, _ = day_window(day2.isoformat(), day2.isoformat())
    s4, _ = day_window(day4.isoformat(), day4.isoformat())
    since, until = day_window("2026-10-01", "2026-10-03")

    def add(prompt, thinking, content, started_at):
        rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
        rec.started_at = started_at
        rec.finished_at = started_at + 0.1
        rec.status = 200
        rec.prompt_tokens = prompt
        rec.thinking_tokens = thinking
        rec.content_tokens = content
        store.add(rec)

    add(100, 10, 20, s1 + 3600)          # 10-01 midday
    add(300, 30, 40, s2 + 7200)          # 10-02
    add(999, 99, 99, s4 + 3600)          # 10-04: outside the window

    data = store.token_range(since, until)
    assert [d["date"] for d in data["daily"]] == ["2026-10-01", "2026-10-02", "2026-10-03"]
    assert data["daily"][0]["total"] == 130
    assert data["daily"][1]["total"] == 370
    assert data["daily"][2]["total"] == 0        # zero-filled quiet day
    assert data["scope"]["requests"] == 2
    assert data["scope"]["total"] == 500
    assert data["scope"]["prompt"] == 400
    assert data["scope"]["output"] == 100


def test_token_range_estimates_cache_per_day(tmp_path):
    from datetime import date

    from monitor.store import day_window

    store = Store(tmp_path / "m.db")
    since, until = day_window("2026-10-05", "2026-10-06")
    for day in (0, 1):
        rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
        rec.started_at = since + day * 86400 + 60
        rec.status = 200
        rec.prompt_tokens = 1000
        rec.content_tokens = 10
        store.add(rec)

    data = store.token_range(since, until, cache_ratio=0.75)
    assert data["scope"]["cache_estimated"] is True
    assert data["scope"]["prompt_cached"] == 1500
    assert [d["prompt_cached"] for d in data["daily"]] == [750, 750]
    # no ratio -> reported zeros, not a guess
    plain = store.token_range(since, until)
    assert plain["scope"]["prompt_cached"] == 0
    assert plain["scope"]["cache_estimated"] is False


def test_cache_ratio_is_persisted_and_survives_restart(tmp_path):
    """A reboot zeroes the engine counters; the ledger must not move."""
    from monitor.store import day_window

    store = Store(tmp_path / "m.db")
    now = time.time()
    rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
    rec.started_at = now - 60
    rec.status = 200
    rec.prompt_tokens = 1000
    rec.prompt_cached = None      # vLLM never fills cached_tokens per request
    rec.content_tokens = 10
    store.add(rec)

    # what the metrics loop saw while the engine was warm: 900 of 1000 cached
    store.record_cache("vllm", 1000, 900, now)
    history = store.token_totals()["history"]
    assert history["prompt_cached"] == 900
    assert history["cache_estimated"] is True
    assert history["cache_unknown"] is False

    # engine restart: counters are back at 0, then a cold request arrives.
    # Deltas now accumulate, so the 10 cold tokens *do* dilute the ratio
    # (900/1010) instead of being silently dropped.
    store.record_cache("vllm", 0, 0, now + 1)
    store.record_cache("vllm", 10, 0, now + 2)
    assert store.token_totals()["history"]["prompt_cached"] == 891

    data = store.token_range(*day_window(None, None))
    assert data["scope"]["prompt_cached"] == 891
    assert data["daily"][-1]["prompt_cached"] == 891


def test_token_range_uses_each_day_cache_ratio(tmp_path):
    from monitor.store import day_window

    store = Store(tmp_path / "m.db")
    since, until = day_window("2026-10-01", "2026-10-02")
    for day in (0, 1):
        rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
        rec.started_at = since + day * 86400 + 60
        rec.status = 200
        rec.prompt_tokens = 1000
        rec.content_tokens = 10
        store.add(rec)
    store.db.execute(
        "INSERT INTO cache_daily (day, engine, queries, hits) VALUES"
        " ('2026-10-01','vllm',1000,800), ('2026-10-02','vllm',1000,200)"
    )
    store.db.commit()

    data = store.token_range(since, until)
    assert [d["prompt_cached"] for d in data["daily"]] == [800, 200]
    assert data["scope"]["prompt_cached"] == 1000      # weighted, not averaged
    assert data["scope"]["cache_estimated"] is True


def test_missing_cache_ratio_is_unknown_not_zero(tmp_path):
    store = Store(tmp_path / "m.db")
    rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
    rec.started_at = time.time()
    rec.status = 200
    rec.prompt_tokens = 1000
    rec.content_tokens = 10
    store.add(rec)

    scope = store.token_totals(cache_ratio=None)["history"]
    assert scope["cache_unknown"] is True
    assert scope["cache_hit_pct"] is None
    assert scope["prompt_cached"] == 0


def test_tiny_cache_sample_does_not_reprice_a_day(tmp_path):
    """A post-restart sample must not rewrite a day it barely covers."""
    from monitor.store import day_window

    store = Store(tmp_path / "m.db")
    since, until = day_window("2026-10-05", "2026-10-05")
    for _ in range(10):
        rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
        rec.started_at = since + 60
        rec.status = 200
        rec.prompt_tokens = 100_000
        rec.content_tokens = 10
        store.add(rec)
    # a full-day sample from the previous day, then a 1k-token restart sample
    store.db.execute(
        "INSERT INTO cache_daily (day, engine, queries, hits) VALUES"
        " ('2026-10-04','vllm',200000,200000), ('2026-10-05','vllm',1000,0)"
    )
    store.db.commit()

    data = store.token_range(since, until)
    day = data["daily"][0]
    assert day["prompt"] == 1_000_000
    # 1k of 1M is 0.1% coverage -> fall back to the all-time ratio, not 0%
    assert day["cache_hit_pct"] > 99.0
    assert day["prompt_cached"] == data["scope"]["prompt_cached"]


def test_record_cache_accumulates_deltas_across_engine_restarts(tmp_path):
    """MAX() of two independently-resetting counters used to scramble the ratio."""
    store = Store(tmp_path / "m.db")
    now = time.time()
    store.record_cache("vllm", 100, 40, now)          # first sample
    store.record_cache("vllm", 200, 90, now + 1)      # +100 / +50
    store.record_cache("vllm", 50, 10, now + 2)       # restart -> +50 / +10
    store.record_cache("vllm", 50, 10, now + 3)       # idle repeat -> +0
    row = store.db.execute("SELECT queries, hits FROM cache_daily").fetchone()
    assert tuple(row) == (250, 100)
    # a monitor restart must not lose the baseline (cache_state is persisted)
    store2 = Store(tmp_path / "m.db")
    store2.record_cache("vllm", 60, 20, now + 4)      # +10 / +10
    row = store2.db.execute("SELECT queries, hits FROM cache_daily").fetchone()
    assert tuple(row) == (260, 110)


def test_hit_ratio_survives_a_benchmark_with_zero_reuse(tmp_path):
    """A day with 60 % hits stays near 60 % after a huge zero-hit run + restart."""
    store = Store(tmp_path / "m.db")
    now = time.time()
    store.record_cache("vllm", 20_000, 12_000, now)        # 60 %
    store.record_cache("vllm", 2_500_000, 0, now + 1)      # benchmark, restart
    q, h = store.db.execute("SELECT queries, hits FROM cache_daily").fetchone()
    assert q == 2_500_000
    assert h == 12_000
    assert abs(h / q - 0.0048) < 0.001     # diluted, but not scrambled


def test_record_cache_clamps_hits_to_queries(tmp_path):
    store = Store(tmp_path / "m.db")
    store.record_cache("vllm", 100, 500, time.time())
    q, h = store.db.execute("SELECT queries, hits FROM cache_daily").fetchone()
    assert (q, h) == (100, 100)


def test_partial_per_request_cache_falls_back_to_the_engine_ratio():
    """启用 cached_tokens 那天不能把「部分覆盖」当成全天命中率。

    真实事故：10-19 只有 3.4% 的请求带 cached_tokens，于是当天被算成 2.9%，
    而引擎计数器（cache_daily）覆盖全量、真实是 77.6%。
    """
    from monitor.store import Store

    raw = {
        "requests": 10, "prompt": 100_000, "prompt_cached": 9_000,
        "cached_rows": 1, "prompt_with_cached": 10_000,
        "thinking": 0, "content": 0, "peak": {},
    }
    out = Store._finish_scope(raw, cache_ratio=0.8)
    assert out["prompt_cached"] == 80_000        # 引擎比率，而不是那 9,000
    assert out["cache_estimated"] is True
    assert out["cache_hit_pct"] == 80.0


def test_full_per_request_cache_coverage_wins():
    """覆盖率足够时用逐请求真值（比引擎的全局比率更准）。"""
    from monitor.store import Store

    raw = {
        "requests": 10, "prompt": 100_000, "prompt_cached": 61_000,
        "cached_rows": 10, "prompt_with_cached": 100_000,
        "thinking": 0, "content": 0, "peak": {},
    }
    out = Store._finish_scope(raw, cache_ratio=0.8)
    assert out["prompt_cached"] == 61_000
    assert out["cache_estimated"] is False
    assert out["cache_hit_pct"] == 61.0


def test_partial_coverage_window_is_not_diluted():
    """10-20 的真实回归：51% 覆盖率把 95% 的命中算成了 48.5%。

    规则必须是「引擎比率 / 抽样外推」，绝不能用「部分和 ÷ 全量 prompt」。
    """
    from monitor.store import Store

    raw = {
        "requests": 2000,
        "prompt": 273_224_384,
        "prompt_cached": 132_637_448,     # 只覆盖前 51% 的 prompt
        "cached_rows": 1093,
        "prompt_with_cached": 139_700_000,
        "thinking": 0, "content": 0, "peak": {},
    }
    out = Store._finish_scope(raw, cache_ratio=0.872)
    assert out["cache_hit_pct"] == pytest.approx(87.2, abs=0.1)   # 不是 48.5
    assert out["cache_estimated"] is True


def test_partial_coverage_extrapolates_from_the_sample_without_an_engine_ratio():
    """没有引擎计数器时，用覆盖率内的真值当样本外推，而不是把部分和当全量。"""
    from monitor.store import Store

    raw = {
        "requests": 10, "prompt": 1_000,
        "prompt_cached": 300, "cached_rows": 4, "prompt_with_cached": 400,
        "thinking": 0, "content": 0, "peak": {},
    }
    out = Store._finish_scope(raw, cache_ratio=None)
    assert out["prompt_cached"] == 750       # 300/400 = 75% × 1000
    assert out["cache_estimated"] is True
    assert out["cache_unknown"] is False
