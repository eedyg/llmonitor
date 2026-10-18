"""Request history (SQLite) + in-memory time-series ring."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta
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
CREATE TABLE IF NOT EXISTS gpu_energy (
    day TEXT PRIMARY KEY,          -- local calendar day, YYYY-MM-DD
    wh REAL NOT NULL DEFAULT 0,    -- integrated GPU power
    seconds REAL NOT NULL DEFAULT 0
);
"""

# Cumulative token ledger. prompt falls back to the estimate when usage never
# arrived; a 0 prompt is meaningless, hence NULLIF.
PROMPT_EXPR = "COALESCE(NULLIF(prompt_tokens, 0), prompt_tokens_est, 0)"

TOKEN_FIELDS = f"""
       COUNT(*),
       COALESCE(SUM({PROMPT_EXPR}), 0),
       COALESCE(SUM(COALESCE(prompt_cached, 0)), 0),
       COALESCE(SUM(COALESCE(thinking_tokens, 0)), 0),
       COALESCE(SUM(COALESCE(content_tokens, 0)), 0),
       COUNT(prompt_cached),
       MIN(started_at),
       MAX(started_at)"""

TOKEN_SQL = "SELECT" + TOKEN_FIELDS + "\n  FROM requests"

# Peak billing hours are Beijing time (UTC+8), Mon-Fri. Chinese public holidays
# are not modelled: at most a handful of days a year land in the wrong bucket.
PEAK_SQL = (
    "(CAST(strftime('%w', started_at, 'unixepoch', '+8 hours') AS INTEGER) BETWEEN 1 AND 5"
    " AND CAST(strftime('%H', started_at, 'unixepoch', '+8 hours') AS INTEGER) IN ({hours}))"
)

DAILY_TOKEN_SQL = """
SELECT strftime('%Y-%m-%d', started_at, 'unixepoch', 'localtime') AS day,
       COUNT(*),
       COALESCE(SUM({prompt}), 0),
       COALESCE(SUM(COALESCE(prompt_cached, 0)), 0),
       COALESCE(SUM(COALESCE(thinking_tokens, 0)), 0),
       COALESCE(SUM(COALESCE(content_tokens, 0)), 0),
       COUNT(prompt_cached),
       COALESCE(SUM(CASE WHEN {peak} THEN {prompt} ELSE 0 END), 0),
       COALESCE(SUM(CASE WHEN {peak} THEN COALESCE(prompt_cached, 0) ELSE 0 END), 0),
       COALESCE(SUM(CASE WHEN {peak} THEN COALESCE(thinking_tokens, 0) ELSE 0 END), 0),
       COALESCE(SUM(CASE WHEN {peak} THEN COALESCE(content_tokens, 0) ELSE 0 END), 0)
  FROM requests
 WHERE started_at >= ? AND started_at < ?
 GROUP BY day
 ORDER BY day
"""

LIST_COLUMNS = (    "id, engine, endpoint, protocol, stream, model, started_at, started_iso, finished_at,"
    " elapsed_ms, ttft_ms, ttfc_ms, decode_ms, decode_tps, prompt_tokens, prompt_cached,"
    " prompt_tokens_est, prompt_chars, prompt_messages, request_bytes, has_media, tools,"
    " thinking_tokens, content_tokens, thinking_chars, content_chars, finish_reason, status,"
    " error, max_tokens, sampling_json, client_ip, client_ua, engine_request_id,"
    " (thinking_text IS NOT NULL OR content_text IS NOT NULL) AS has_text"
)


def day_window(start: Any = None, end: Any = None, today: date | None = None,
               max_days: int = 400) -> tuple[float, float]:
    """[since, until) epoch bounds for two *local* calendar days, both inclusive.

    Accepts ``YYYY-MM-DD`` strings or epoch values; empty means today. The day
    range is capped so a typo cannot ask SQLite to zero-fill 100 years.
    """
    today = today or date.today()
    first = _parse_day(start, today)
    last = _parse_day(end, today)
    if last < first:
        first, last = last, first
    if (last - first).days + 1 > max_days:
        first = last - timedelta(days=max_days - 1)
    since = time.mktime(datetime.combine(first, datetime.min.time()).timetuple())
    until = time.mktime(datetime.combine(last + timedelta(days=1), datetime.min.time()).timetuple())
    return since, until


def _parse_day(value: Any, today: date) -> date:
    if value is None or value == "":
        return today
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value)).date()
    text = str(value).strip()
    if text.isdigit():
        return datetime.fromtimestamp(float(text)).date()
    return date.fromisoformat(text[:10])


class Store:
    def __init__(self, db_path: Path, ring_seconds: int = 7200, text_keep: int = 200,
                 text_cap: int = 2 * 1024 * 1024, retention_days: int = 30,
                 peak_hours: tuple[int, ...] = (9, 10, 11, 14, 15, 16, 17)) -> None:
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
        self._enable_incremental_vacuum()
        self.ring: deque[dict] = deque(maxlen=max(60, int(ring_seconds)))
        self.counters = {"requests": 0, "errors": 0, "bytes_in": 0, "bytes_out": 0}
        self.started_at = time.time()
        self._tokens_cache: dict | None = None
        hours = tuple(sorted({int(h) for h in peak_hours if 0 <= int(h) <= 23}))
        self.peak_sql = PEAK_SQL.format(hours=",".join(str(h) for h in hours)) if hours else "0"
        # GPU energy: in-memory session counter + a small per-day table
        self._energy_session_wh = 0.0
        self._energy_session_s = 0.0
        self._energy_pending_wh = 0.0
        self._energy_pending_s = 0.0
        self._energy_day = date.today().isoformat()
        self._energy_flush_at = time.time()

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

    # -- disk guard ---------------------------------------------------------
    def _db_bytes(self) -> int:
        page_count = self.db.execute("PRAGMA page_count").fetchone()[0]
        page_size = self.db.execute("PRAGMA page_size").fetchone()[0]
        return int(page_count) * int(page_size)

    def _db_live_bytes(self) -> int:
        """Bytes that will remain after a vacuum (free pages excluded)."""
        pages = self.db.execute("PRAGMA page_count").fetchone()[0]
        free = self.db.execute("PRAGMA freelist_count").fetchone()[0]
        page_size = self.db.execute("PRAGMA page_size").fetchone()[0]
        return max(0, int(pages) - int(free)) * int(page_size)

    def _enable_incremental_vacuum(self) -> None:
        """Let prune() hand pages back without a full VACUUM.

        VACUUM needs a second copy of the database on disk, which is precisely
        what we cannot afford once the cap is hit. Switching the mode needs one
        VACUUM, so do it only while the file is small enough to be cheap.
        """
        try:
            mode = self.db.execute("PRAGMA auto_vacuum").fetchone()[0]
            if mode == 2:  # already INCREMENTAL
                return
            self.db.execute("PRAGMA auto_vacuum=INCREMENTAL")
            if self._db_bytes() <= 256 * 1024 * 1024:
                self.db.execute("VACUUM")
            self.db.commit()
        except sqlite3.Error:  # pragma: no cover - the cap still works without it
            pass

    def enforce_max_bytes(self, max_bytes: int) -> int:
        """Hard cap: drop the oldest rows until the database file fits again.

        ``retention_days`` already bounds the age of a row; this bounds the
        size, which matters at request rates high enough to out-grow the disk
        inside that window. Returns the number of rows dropped.
        """
        if max_bytes <= 0:
            return 0
        removed = 0
        with self._lock:
            for _ in range(32):  # bounded: 32 rounds of >=100 rows
                if self._db_live_bytes() <= max_bytes:
                    break
                rows = self.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
                if not rows:
                    break
                batch = max(100, int(rows) // 10)
                cur = self.db.execute(
                    "DELETE FROM requests WHERE id IN ("
                    " SELECT id FROM requests ORDER BY started_at ASC LIMIT ?)",
                    (batch,),
                )
                removed += max(0, cur.rowcount)
                self.db.commit()
                if cur.rowcount <= 0:
                    break
            if removed:
                try:
                    self.db.execute("PRAGMA incremental_vacuum")
                    self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    self.db.commit()
                except sqlite3.Error:  # pragma: no cover
                    pass
                self._tokens_cache = None
        return removed

    # -- cumulative token ledger --------------------------------------------
    @staticmethod
    def _rows_to_raw(row: tuple, offset: int = 0) -> dict:
        return {
            "requests": int(row[offset] or 0),
            "prompt": int(row[offset + 1] or 0),
            "prompt_cached": int(row[offset + 2] or 0),
            "thinking": int(row[offset + 3] or 0),
            "content": int(row[offset + 4] or 0),
            "cached_rows": int(row[offset + 5] or 0),
            "first_at": row[offset + 6] if len(row) > offset + 6 else None,
            "last_at": row[offset + 7] if len(row) > offset + 7 else None,
        }

    def _token_scope(self, since: float | None, until: float | None = None) -> dict:
        """Raw aggregate of the token columns (caller must hold the lock)."""
        where: list[str] = []
        args: list[Any] = []
        if since is not None:
            where.append("started_at >= ?")
            args.append(since)
        if until is not None:
            where.append("started_at < ?")
            args.append(until)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        raw = self._rows_to_raw(self.db.execute(TOKEN_SQL + clause, args).fetchone())
        raw["prompt"] = int(raw["prompt"] or 0)
        raw["prompt_cached"] = min(raw["prompt_cached"], raw["prompt"])
        peak_clause = (clause + " AND " + self.peak_sql) if clause else (" WHERE " + self.peak_sql)
        peak = self._rows_to_raw(self.db.execute(TOKEN_SQL + peak_clause, args).fetchone())
        peak["prompt_cached"] = min(peak["prompt_cached"], peak["prompt"])
        raw["peak"] = peak
        return raw

    @staticmethod
    def _finish_scope(raw: dict, cache_ratio: float | None = None) -> dict:
        """Derive the displayed numbers from one raw aggregate."""
        prompt = raw["prompt"]
        cached = raw["prompt_cached"]
        thinking = raw["thinking"]
        content = raw["content"]
        peak = raw.get("peak") or {}
        peak_prompt = int(peak.get("prompt") or 0)
        peak_cached = min(int(peak.get("prompt_cached") or 0), peak_prompt)
        peak_thinking = int(peak.get("thinking") or 0)
        peak_content = int(peak.get("content") or 0)
        estimated = False
        if not raw["cached_rows"] and cache_ratio and prompt:
            cached = min(prompt, int(round(prompt * cache_ratio)))
            # same ratio for the peak subset: the engine only knows a global one
            peak_cached = min(peak_prompt, int(round(peak_prompt * cache_ratio)))
            estimated = True
        output = thinking + content
        peak_output = peak_thinking + peak_content
        return {
            "requests": raw["requests"],
            "total": prompt + output,
            "prompt": prompt,
            "prompt_cached": cached,
            "prompt_miss": prompt - cached,
            "output": output,
            "thinking": thinking,
            "content": content,
            "cache_hit_pct": round(100.0 * cached / prompt, 1) if prompt else None,
            "cache_estimated": estimated,
            "first_at": raw.get("first_at"),
            "last_at": raw.get("last_at"),
            "peak": {
                "prompt": peak_prompt,
                "prompt_cached": peak_cached,
                "prompt_miss": peak_prompt - peak_cached,
                "output": peak_output,
                "thinking": peak_thinking,
                "content": peak_content,
                "tokens": peak_prompt + peak_output,
            },
        }

    def token_totals(self, cache_ratio: float | None = None, cache_s: float = 2.0) -> dict:
        """Read-only cumulative token ledger for the overview.

        ``session`` covers every request recorded since this process started
        (so it only grows while the monitor runs); ``history`` covers every
        row still retained in SQLite. Both are plain SQL aggregates over the
        request rows: nothing is written, nothing has to be kept in sync.

        Some engines (vLLM builds, for one) do not report
        ``prompt_tokens_details.cached_tokens`` per request. When no row has
        that column we fall back to the engine-wide hit ratio, flag the result
        as estimated instead of pretending the cache never hit.
        """
        now = time.time()
        with self._lock:
            hit = self._tokens_cache
            if hit and now - hit["at"] < cache_s and hit["ratio"] == cache_ratio:
                return hit["data"]
            session = self._finish_scope(self._token_scope(self.started_at), cache_ratio)
            history = self._finish_scope(self._token_scope(None), cache_ratio)
            uptime = max(0.0, now - self.started_at)
            session["uptime_s"] = round(uptime, 1)
            session["tps"] = round(session["total"] / uptime, 2) if uptime > 0.5 else None
            data = {
                "ts": round(now, 3),
                "session": session,
                "history": history,
            }
            self._tokens_cache = {"at": now, "ratio": cache_ratio, "data": data}
            return data

    def token_range(self, since: float, until: float,
                    cache_ratio: float | None = None) -> dict:
        """Ledger for an explicit [since, until) epoch window, per local day.

        Used by the date-range picker: one aggregate for the whole window plus
        one row per calendar day (zero-filled, so a quiet day still shows up).
        """
        with self._lock:
            scope = self._finish_scope(self._token_scope(since, until), cache_ratio)
            rows = self.db.execute(
                DAILY_TOKEN_SQL.format(prompt=PROMPT_EXPR, peak=self.peak_sql),
                (since, until),
            ).fetchall()
        by_day: dict[str, dict] = {}
        for row in rows:
            raw = self._rows_to_raw(row, 1)
            raw.pop("first_at", None)
            raw.pop("last_at", None)
            raw["peak"] = {
                "prompt": int(row[7] or 0),
                "prompt_cached": min(int(row[8] or 0), int(row[7] or 0)),
                "thinking": int(row[9] or 0),
                "content": int(row[10] or 0),
            }
            entry = self._finish_scope(raw, cache_ratio)
            entry.pop("first_at", None)
            entry.pop("last_at", None)
            entry["date"] = row[0]
            by_day[row[0]] = entry
        empty = self._finish_scope({"requests": 0, "prompt": 0, "prompt_cached": 0,
                                    "thinking": 0, "content": 0, "cached_rows": 0,
                                    "first_at": None, "last_at": None, "peak": {}})
        empty.pop("first_at", None)
        empty.pop("last_at", None)
        daily: list[dict] = []
        cursor = datetime.fromtimestamp(since).date()
        last = datetime.fromtimestamp(max(since, until - 1)).date()
        while cursor <= last and len(daily) < 400:
            key = cursor.isoformat()
            entry = dict(by_day.get(key) or empty)
            entry["date"] = key
            daily.append(entry)
            cursor += timedelta(days=1)
        return {"scope": scope, "daily": daily}

    # -- GPU energy ---------------------------------------------------------
    def record_energy(self, power_w: Any, dt_s: float, now: float | None = None) -> None:
        """Integrate one GPU power sample into the daily kWh ledger.

        Called at ~1 Hz from the series loop, so the write is batched: we keep
        the pending watt-hours in memory and flush every 5 Wh / 5 s.
        """
        try:
            watts = float(power_w)
            dt = min(float(dt_s), 10.0)  # a suspend/resume gap must not invent energy
        except (TypeError, ValueError):
            return
        if watts <= 0 or dt <= 0:
            return
        now = now if now is not None else time.time()
        wh = watts * dt / 3600.0
        today = date.today().isoformat()
        if today != self._energy_day:
            self._flush_energy()
            self._energy_day = today
        self._energy_session_wh += wh
        self._energy_session_s += dt
        self._energy_pending_wh += wh
        self._energy_pending_s += dt
        if self._energy_pending_wh >= 5.0 or now - self._energy_flush_at >= 5.0:
            self._flush_energy()

    def _flush_energy(self) -> None:
        if self._energy_pending_wh <= 0:
            self._energy_flush_at = time.time()
            return
        wh, seconds = self._energy_pending_wh, self._energy_pending_s
        self._energy_pending_wh = 0.0
        self._energy_pending_s = 0.0
        self._energy_flush_at = time.time()
        with self._lock:
            self.db.execute(
                "INSERT INTO gpu_energy (day, wh, seconds) VALUES (?,?,?)"
                " ON CONFLICT(day) DO UPDATE SET"
                "   wh = wh + excluded.wh, seconds = seconds + excluded.seconds",
                (self._energy_day, wh, seconds),
            )
            self.db.commit()

    def energy_totals(self) -> dict:
        """Energy for this process (memory) and everything retained (table)."""
        with self._lock:
            row = self.db.execute(
                "SELECT COALESCE(SUM(wh), 0), COALESCE(SUM(seconds), 0),"
                " MIN(day), COUNT(*) FROM gpu_energy"
            ).fetchone()
            days = int(row[3] or 0)
            since = row[2]
            if self._energy_pending_wh > 0:
                if since is None or self._energy_day < since:
                    since = self._energy_day
                # the pending bucket may be a day that has no row yet
                if not self.db.execute(
                    "SELECT 1 FROM gpu_energy WHERE day = ?", (self._energy_day,)
                ).fetchone():
                    days += 1
        history_wh = float(row[0] or 0.0) + self._energy_pending_wh
        history_s = float(row[1] or 0.0) + self._energy_pending_s
        return {
            "session": {
                "kwh": round(self._energy_session_wh / 1000.0, 6),
                "seconds": round(self._energy_session_s, 1),
            },
            "history": {
                "kwh": round(history_wh / 1000.0, 6),
                "seconds": round(history_s, 1),
                "since": since,
                "days": days,
            },
        }

    def energy_by_day(self, start_day: str, end_day: str) -> dict[str, float]:
        """kWh per local day in ``[start_day, end_day]`` (inclusive, ISO dates)."""
        with self._lock:
            rows = self.db.execute(
                "SELECT day, wh FROM gpu_energy WHERE day >= ? AND day <= ?",
                (start_day, end_day),
            ).fetchall()
        out = {str(r[0]): float(r[1] or 0.0) / 1000.0 for r in rows}
        if start_day <= self._energy_day <= end_day:
            out[self._energy_day] = out.get(self._energy_day, 0.0) + self._energy_pending_wh / 1000.0
        return out

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
