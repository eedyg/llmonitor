"""Bounded disk usage.

The monitor is allowed to write in three places, and all three would grow
forever if left alone:

``run/<engine>.log``        engine stdout. We open it with ``ab`` (O_APPEND) and
                            hand the fd to the child, so the engine owns the
                            writes and never rotates anything.
``run/ninfer-requests.jsonl``  NInfer's per-request JSONL, written because we
                            pass ``--request-log-jsonl``.
``state/lifecycle.log``     our own docker/start/stop transcripts.
``state/monitor.db``        request history: bounded by ``storage.retention_days``
                            and the hard ``storage.max_db_mb`` cap below.

Log files are trimmed **in place** (keep the tail, truncate the rest) instead
of being rotated by rename: the writer holds an O_APPEND fd, and renaming the
file would leave it writing into an unlinked inode until the engine restarts —
disk never freed, log invisible. Both engines in this repo open their logs with
append semantics (``ab`` for the child stdout, ``ios::out|ios::app`` in NInfer),
so truncation is safe.

Docker container logs are *not* ours to trim (the file lives under
``/var/lib/docker``); configure the daemon or the container with
``--log-opt max-size=... --log-opt max-file=...``. See README.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("monitor.maintenance")

MB = 1024 * 1024
# Never trim a log just because it crossed the line by a few bytes: rewriting a
# file for no reason only creates churn.
MIN_TRIM_BYTES = 256 * 1024
# Upper bound for a single tail read, so a runaway log cannot be loaded whole.
TAIL_READ_LIMIT = 32 * MB


def tail_lines(path: str | Path, count: int) -> list[str]:
    """Last ``count`` non-empty lines, without reading the whole file."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size <= 0:
        return []
    want = min(size, TAIL_READ_LIMIT)
    try:
        with path.open("rb") as handle:
            handle.seek(size - want)
            blob = handle.read()
    except OSError:
        return []
    if want < size:  # drop the partial first line
        cut = blob.find(b"\n")
        if cut >= 0:
            blob = blob[cut + 1 :]
    lines = [line for line in blob.decode("utf-8", "replace").splitlines() if line.strip()]
    return lines[-max(1, int(count)) :]


def trim_log_in_place(path: str | Path, max_bytes: int, keep_bytes: int) -> int:
    """Truncate ``path`` to its last ``keep_bytes`` once it exceeds ``max_bytes``.

    Returns the number of bytes reclaimed (0 when nothing was done). The file is
    rewritten through the same inode, so an O_APPEND writer keeps appending to
    the trimmed file instead of leaking a deleted one.
    """
    if max_bytes <= 0:
        return 0
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if size <= max_bytes or size <= MIN_TRIM_BYTES:
        return 0
    keep = max(MIN_TRIM_BYTES, min(int(keep_bytes or 0), max_bytes))
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, size - keep))
            tail = handle.read()
        cut = tail.find(b"\n")  # start on a whole line
        if 0 <= cut < len(tail) - 1:
            tail = tail[cut + 1 :]
        with path.open("r+b") as handle:
            handle.seek(0)
            handle.write(tail)
            handle.truncate(len(tail))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:  # pragma: no cover - log rotation is best effort
        log.debug("could not trim %s: %s", path, exc)
        return 0
    return max(0, size - len(tail))


def maintain(rt) -> dict:  # noqa: ANN001 - runtime duck type, avoids a cycle
    """One pass of disk hygiene. Runs in a worker thread."""
    cfg = rt.cfg.root
    logs = cfg.logs
    done: dict[str, int] = {}

    max_log = int(float(logs.max_log_mb or 0) * MB)
    keep_log = int(float(logs.keep_tail_mb or 0) * MB)
    if max_log > 0:
        run_dir = rt.cfg.p("paths.run_dir")
        names = {
            str(cfg.engine.container_names.vllm),
            str(cfg.engine.container_names.ninfer),
        }
        for name in names:
            removed = trim_log_in_place(run_dir / f"{name}.log", max_log, keep_log)
            if removed:
                done[f"{name}.log"] = removed
        # our own stdout, when the monitor is started as `... >> run/monitor.log`
        own = trim_log_in_place(run_dir / "monitor.log", max_log, keep_log)
        if own:
            done["monitor.log"] = own
        life = trim_log_in_place(
            rt.cfg.p("paths.state_dir") / "lifecycle.log",
            int(float(logs.lifecycle_max_mb or 0) * MB),
            keep_log,
        )
        if life:
            done["lifecycle.log"] = life

    if bool(cfg.engine.ninfer_jsonl):
        jsonl = rt.cfg.p("engine.ninfer_jsonl_path")
        removed = trim_log_in_place(
            jsonl,
            int(float(logs.jsonl_max_mb or 0) * MB),
            keep_log,
        )
        if removed:
            done[jsonl.name] = removed

    dropped = rt.store.enforce_max_bytes(int(float(cfg.storage.max_db_mb or 0) * MB))
    if dropped:
        done["monitor.db rows"] = dropped

    if done:
        log.info("disk hygiene: %s", ", ".join(f"{k} -{v}" for k, v in done.items()))
        rt.publish({"type": "maintenance", "trimmed": done})
    return done
