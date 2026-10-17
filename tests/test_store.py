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
