"""Request history (SQLite) + in-memory time-series ring."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    engine TEXT,
    endpoint TEXT,
    protocol TEXT,
    stream INTEGER,
    model TEXT,
    started_at REAL,
    started_iso TEXT,
    finished_at REAL,
    elapsed_ms REAL,
    ttft_ms REAL,
    ttfc_ms REAL,
    decode_ms REAL,
    decode_tps REAL,
    prompt_tokens INTEGER,
    prompt_cached INTEGER,
    prompt_tokens_est INTEGER,
    prompt_chars INTEGER,
    prompt_messages INTEGER,
    request_bytes INTEGER,
    has_media INTEGER,
    tools INTEGER,
    thinking_tokens INTEGER,
    content_tokens INTEGER,
    thinking_chars INTEGER,
    content_chars INTEGER,
    finish_reason TEXT,
    status INTEGER,
    error TEXT,
    engine_request_id TEXT,
    max_tokens INTEGER,
    sampling_json TEXT,
    usage_json TEXT,
    client_ip TEXT,
    client_ua TEXT,
    thinking_text TEXT,
    content_text TEXT,
    engine_record_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_started ON requests(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_requests_engine_id ON requests(engine_request_id);
CREATE TABLE IF NOT EXISTS docker_images (
    id TEXT PRIMARY KEY,
    refs TEXT,
    first_seen REAL,
    superseded_at REAL,
    removed_at REAL,
    source TEXT
);
"""

LIST_COLUMNS = (
    "id, engine, endpoint, protocol, stream, model, started_at, started_iso, finished_at,"
    " elapsed_ms, ttft_ms, ttfc_ms, decode_ms, decode_tps, prompt_tokens, prompt_cached,"
    " prompt_tokens_est, prompt_chars, prompt_messages, request_bytes, has_media, tools,"
    " thinking_tokens, content_tokens, thinking_chars, content_chars, finish_reason, status,"
    " error, max_tokens, sampling_json, client_ip, client_ua, engine_request_id,"
    " (thinking_text IS NOT NULL OR content_text IS NOT NULL) AS has_text"
)


class Store:
    def __init__(self, db_path: Path, ring_seconds: int = 7200, text_keep: int = 200,
                 text_cap: int = 2 * 1024 * 1024, retention_days: int = 30) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.text_keep = max(0, int(text_keep))
        self.text_cap = max(0, int(text_cap))
        self.retention_days = max(1, int(retention_days))
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(requests)")}
        if "engine_request_id" not in columns:
            self.db.execute("ALTER TABLE requests ADD COLUMN engine_request_id TEXT")
        self.db.commit()
        self.ring: deque[dict] = deque(maxlen=max(60, int(ring_seconds)))
        self.counters = {"requests": 0, "errors": 0, "bytes_in": 0, "bytes_out": 0}
        self.started_at = time.time()

    # -- time series --------------------------------------------------------
    def push_sample(self, point: dict) -> None:
        self.ring.append(point)

    def series(self, since: float | None = None, keys: list[str] | None = None) -> dict:
        now = time.time()
        if since is None:
            since = now - 3600
        points = [p for p in self.ring if p.get("ts", 0) >= since]
        if not points:
            return {"ts": [], "keys": []}
        available = [k for k in points[-1].keys() if k != "ts"]
        use = [k for k in (keys or available) if k in available]
        out: dict[str, Any] = {"ts": [round(p["ts"], 3) for p in points], "keys": use}
        for key in use:
            out[key] = [p.get(key) for p in points]
        return out

    # -- request history ----------------------------------------------------
    def add(self, record) -> None:
        row = record.to_row()
        if self.text_cap:
            row["thinking_text"] = (row.get("thinking_text") or "")[: self.text_cap] or None
            row["content_text"] = (row.get("content_text") or "")[: self.text_cap] or None
        if not self.text_keep:
            row["thinking_text"] = None
            row["content_text"] = None
        columns = [
            "id", "engine", "endpoint", "protocol", "stream", "model", "started_at",
            "started_iso", "finished_at", "elapsed_ms", "ttft_ms", "ttfc_ms", "decode_ms",
            "decode_tps", "prompt_tokens", "prompt_cached", "prompt_tokens_est",
            "prompt_chars", "prompt_messages", "request_bytes", "has_media", "tools",
            "thinking_tokens", "content_tokens", "thinking_chars", "content_chars",
            "finish_reason", "status", "error", "max_tokens", "sampling_json", "usage_json",
            "engine_request_id",
            "client_ip", "client_ua", "thinking_text", "content_text", "engine_record_json",
        ]
        prepared = []
        for column in columns:
            value = row.get(column)
            if column == "stream":
                value = 1 if value else 0
            elif column == "has_media":
                value = 1 if value else 0
            elif column in ("sampling_json", "usage_json") and not isinstance(value, str):
                value = json.dumps(value or {}, ensure_ascii=False)
            prepared.append(value)
        placeholders = ",".join("?" for _ in columns)
        with self._lock:
            self.db.execute(
                f"INSERT OR REPLACE INTO requests ({','.join(columns)}) VALUES ({placeholders})",
                prepared,
            )
            self.db.commit()
            self.counters["requests"] += 1
            if row.get("status") and int(row["status"]) >= 400:
                self.counters["errors"] += 1
            self._prune_locked()

    def _prune_locked(self) -> None:
        if self.text_keep:
            self.db.execute(
                "UPDATE requests SET thinking_text=NULL, content_text=NULL,"
                " engine_record_json=NULL WHERE id IN ("
                "  SELECT id FROM requests ORDER BY started_at DESC LIMIT -1 OFFSET ?)",
                (self.text_keep,),
            )
        cutoff = time.time() - self.retention_days * 86400
        self.db.execute("DELETE FROM requests WHERE started_at < ?", (cutoff,))
        self.db.commit()

    def history(self, limit: int = 100, offset: int = 0, engine: str | None = None,
                status: str | None = None, since: float | None = None,
                search: str | None = None) -> dict:
        where = []
        args: list[Any] = []
        if engine:
            where.append("engine = ?")
            args.append(engine)
        if since:
            where.append("started_at >= ?")
            args.append(since)
        if status == "error":
            where.append("(status >= 400 OR error IS NOT NULL)")
        elif status == "ok":
            where.append("status < 400 AND error IS NULL")
        if search:
            where.append("(id LIKE ? OR model LIKE ? OR endpoint LIKE ?)")
            like = f"%{search}%"
            args.extend([like, like, like])
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            total = self.db.execute(f"SELECT COUNT(*) FROM requests{clause}", args).fetchone()[0]
            rows = self.db.execute(
                f"SELECT {LIST_COLUMNS} FROM requests{clause}"
                " ORDER BY started_at DESC LIMIT ? OFFSET ?",
                [*args, max(1, min(limit, 500)), max(0, offset)],
            ).fetchall()
        return {"total": total, "items": [dict(r) for r in rows]}

    def get(self, request_id: str) -> dict | None:
        with self._lock:
            row = self.db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        for key in ("sampling_json", "usage_json", "engine_record_json"):
            raw = data.pop(key, None)
            data[key[:-5]] = json.loads(raw) if raw else None
        for key in ("stream", "has_media"):
            data[key] = bool(data.get(key))
        return data

    def attach_engine_record(self, engine_request_id: str, record: dict) -> bool:
        """Merge an NInfer JSONL request_done record into the tapped row."""
        result = record.get("result") or {}
        timings = record.get("timings_seconds") or {}
        speculative = record.get("speculative") or {}
        prompt_tokens = result.get("prompt_tokens")
        cached = result.get("prefix_cache_hit_tokens")
        thinking = result.get("model_thinking_tokens")
        completion = result.get("completion_tokens")
        ttft_ms = round(float(timings.get("ttft") or 0) * 1000.0, 2) or None
        decode_s = timings.get("decode")
        decode_tps = None
        if decode_s and completion:
            decode_tps = round(max(0, int(completion) - 1) / float(decode_s), 2)
        with self._lock:
            row = self.db.execute(
                "SELECT id FROM requests WHERE engine_request_id = ?", (engine_request_id,)
            ).fetchone()
            if not row:
                return False
            self.db.execute(
                """UPDATE requests SET
                     engine_record_json = ?,
                     prompt_tokens = COALESCE(?, prompt_tokens),
                     prompt_cached = COALESCE(?, prompt_cached),
                     thinking_tokens = COALESCE(?, thinking_tokens),
                     content_tokens = COALESCE(?, content_tokens),
                     ttft_ms = COALESCE(?, ttft_ms),
                     decode_tps = COALESCE(?, decode_tps)
                   WHERE id = ?""",
                (
                    json.dumps(record, ensure_ascii=False),
                    prompt_tokens,
                    cached,
                    thinking,
                    (max(0, int(completion) - int(thinking)) if completion is not None and thinking is not None else None),
                    ttft_ms,
                    decode_tps,
                    row[0],
                ),
            )
            self.db.commit()
        return True

    def summary(self) -> dict:
        with self._lock:
            total = self.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            errors = self.db.execute(
                "SELECT COUNT(*) FROM requests WHERE status >= 400 OR error IS NOT NULL"
            ).fetchone()[0]
            by_engine = {
                r[0] or "?": r[1]
                for r in self.db.execute(
                    "SELECT engine, COUNT(*) FROM requests GROUP BY engine"
                ).fetchall()
            }
            recent = self.db.execute(
                "SELECT AVG(decode_tps), AVG(ttft_ms), AVG(thinking_tokens), AVG(content_tokens),"
                " AVG(prompt_tokens) FROM requests WHERE started_at > ? AND decode_tps IS NOT NULL",
                (time.time() - 3600,),
            ).fetchone()
        return {
            "total": total,
            "errors": errors,
            "by_engine": by_engine,
            "hour": {
                "decode_tps_avg": _r(recent[0], 2),
                "ttft_ms_avg": _r(recent[1], 1),
                "thinking_tokens_avg": _r(recent[2], 1),
                "content_tokens_avg": _r(recent[3], 1),
                "prompt_tokens_avg": _r(recent[4], 1),
            },
            "counters": dict(self.counters),
            "uptime_s": round(time.time() - self.started_at, 1),
        }

    # -- docker image ledger ------------------------------------------------
    def track_image(self, image_id: str, refs: list[str], source: str = "pull") -> None:
        with self._lock:
            self.db.execute(
                """INSERT INTO docker_images (id, refs, first_seen, source) VALUES (?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     refs = excluded.refs,
                     removed_at = NULL,
                     source = excluded.source""",
                (image_id, json.dumps(refs, ensure_ascii=False), time.time(), source),
            )
            self.db.commit()

    def mark_superseded(self, image_id: str) -> None:
        with self._lock:
            self.db.execute(
                """INSERT INTO docker_images (id, refs, first_seen, superseded_at, source)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET superseded_at = excluded.superseded_at""",
                (image_id, "[]", time.time(), time.time(), "superseded"),
            )
            self.db.commit()

    def mark_image_removed(self, image_id: str) -> None:
        with self._lock:
            self.db.execute("UPDATE docker_images SET removed_at = ? WHERE id = ?",
                            (time.time(), image_id))
            self.db.commit()

    def tracked_image_ids(self) -> list[str]:
        with self._lock:
            rows = self.db.execute(
                "SELECT id FROM docker_images WHERE removed_at IS NULL"
            ).fetchall()
        return [r[0] for r in rows]

    def docker_ledger(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.db.execute(
                "SELECT id, refs, first_seen, superseded_at, removed_at, source"
                " FROM docker_images ORDER BY first_seen DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear(self, keep_last: int = 0) -> int:
        with self._lock:
            if keep_last:
                cur = self.db.execute(
                    "DELETE FROM requests WHERE id NOT IN ("
                    " SELECT id FROM requests ORDER BY started_at DESC LIMIT ?)",
                    (keep_last,),
                )
            else:
                cur = self.db.execute("DELETE FROM requests")
            self.db.commit()
        return cur.rowcount


def _r(value: Any, digits: int) -> float | None:
    try:
        return None if value is None else round(float(value), digits)
    except (TypeError, ValueError):
        return None
