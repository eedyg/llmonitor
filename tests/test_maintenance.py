import os

from monitor import maintenance
from monitor.config import Config
from monitor.store import Store


def test_tail_lines_reads_only_the_tail(tmp_path):
    path = tmp_path / "engine.log"
    path.write_text("\n".join(f"line-{i}" for i in range(5000)) + "\n")
    assert maintenance.tail_lines(path, 3) == ["line-4997", "line-4998", "line-4999"]
    assert maintenance.tail_lines(tmp_path / "missing.log", 5) == []


def test_trim_keeps_the_inode_so_an_appending_writer_survives(tmp_path):
    path = tmp_path / "engine.log"
    path.write_text("head\n" + "x" * (3 * 1024 * 1024) + "\ntail-marker\n")
    inode = path.stat().st_ino
    removed = maintenance.trim_log_in_place(path, 2 * 1024 * 1024, 512 * 1024)
    assert removed > 0
    assert path.stat().st_ino == inode            # same file, not a rotated copy
    assert path.stat().st_size <= 512 * 1024 + 128
    assert path.read_text().endswith("tail-marker\n")
    # under the line and tiny files are left alone
    assert maintenance.trim_log_in_place(path, 8 * 1024 * 1024, 1024) == 0


def test_trim_noop_when_disabled(tmp_path):
    path = tmp_path / "x.log"
    path.write_text("y" * (1024 * 1024))
    assert maintenance.trim_log_in_place(path, 0, 0) == 0
    assert path.stat().st_size == 1024 * 1024


class _FakeRt:
    def __init__(self, cfg, store):
        self.cfg = cfg
        self.store = store
        self.events = []

    def publish(self, event):
        self.events.append(event)


def test_maintain_trims_every_log_it_owns(tmp_path):
    cfg = Config(str(tmp_path / "config.yaml"))
    cfg.data["paths"]["run_dir"] = "run"
    cfg.data["paths"]["state_dir"] = "state"
    cfg.data["engine"]["ninfer_jsonl_path"] = "run/ninfer-requests.jsonl"
    cfg.data["logs"].update({"max_log_mb": 0.5, "keep_tail_mb": 0.25,
                             "lifecycle_max_mb": 0.5, "jsonl_max_mb": 0.5})
    for key in ("run", "state"):
        (tmp_path / key).mkdir(parents=True, exist_ok=True)
    store = Store(tmp_path / "state" / "monitor.db")
    rt = _FakeRt(cfg, store)

    big = "z" * (3 * 1024 * 1024) + "\nEND\n"
    engine_log = tmp_path / "run" / "ninfer-qwen38.log"
    engine_log.write_text(big)
    self_log = tmp_path / "run" / "monitor.log"
    self_log.write_text(big)
    life = tmp_path / "state" / "lifecycle.log"
    life.write_text(big)
    jsonl = tmp_path / "run" / "ninfer-requests.jsonl"
    jsonl.write_text(big)

    done = maintenance.maintain(rt)
    assert set(done) == {"ninfer-qwen38.log", "monitor.log", "lifecycle.log",
                         "ninfer-requests.jsonl"}
    for path in (engine_log, self_log, life, jsonl):
        assert path.stat().st_size < 1024 * 1024
        assert path.read_text().endswith("END\n")
        os.stat(path)  # still there, still writable
        assert os.access(path, os.W_OK)
    assert rt.events and rt.events[-1]["type"] == "maintenance"


def test_enforce_max_bytes_drops_oldest_rows(tmp_path):
    import time

    from monitor.tap import CHAT, Record

    store = Store(tmp_path / "m.db")
    base = time.time() - 3600
    for i in range(400):
        rec = Record(protocol=CHAT, engine="vllm", endpoint="/v1/chat/completions")
        rec.started_at = base + i
        rec.status = 200
        rec.content_text = "x" * 2048
        store.add(rec)
    assert store.history(limit=1)["total"] == 400
    before = store._db_bytes()
    removed = store.enforce_max_bytes(before // 2)
    assert removed > 0
    left = store.history(limit=1)["total"]
    assert 0 < left < 400
    # the newest rows survive, the oldest go first
    newest = store.history(limit=1)["items"][0]
    assert newest["started_at"] > base + 300
    assert store._db_bytes() <= before // 2 or left <= 100
