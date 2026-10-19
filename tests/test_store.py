import time

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

    # engine restart: counters are back at 0, then a cold request arrives
    store.record_cache("vllm", 0, 0, now + 1)
    store.record_cache("vllm", 10, 0, now + 2)
    assert store.token_totals()["history"]["prompt_cached"] == 900

    data = store.token_range(*day_window(None, None))
    assert data["scope"]["prompt_cached"] == 900
    assert data["daily"][-1]["prompt_cached"] == 900


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
