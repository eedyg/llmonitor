"""Tests for the KV-tier collector (monitor/kvcache.py)."""

from __future__ import annotations

import json
import os

import pytest

from monitor import kvcache
from monitor.prom import Metrics

METRICS = """
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{model_name="qwen3.8-27b",num_gpu_blocks="250",block_size="16",cache_dtype="fp8"} 1
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="qwen3.8-27b"} 0.25
# TYPE vllm:kv_offload_cpu_cache_usage_perc gauge
vllm:kv_offload_cpu_cache_usage_perc 0.4
# TYPE vllm:kv_offload_cpu_cache_write_usage_perc gauge
vllm:kv_offload_cpu_cache_write_usage_perc 0.1
# TYPE vllm:kv_offload_tiering_read_bytes_total counter
vllm:kv_offload_tiering_read_bytes_total{tier="1:fs"} 1048576
# TYPE vllm:kv_offload_tiering_write_bytes_total counter
vllm:kv_offload_tiering_write_bytes_total{tier="1:fs"} 2097152
# TYPE vllm:kv_offload_tiering_chunk_queries_total counter
vllm:kv_offload_tiering_chunk_queries_total{tier="1:fs"} 40
# TYPE vllm:kv_offload_tiering_chunk_hits_total counter
vllm:kv_offload_tiering_chunk_hits_total{tier="1:fs"} 10
# TYPE vllm:kv_offload_tiering_promotion_job_failures_total counter
vllm:kv_offload_tiering_promotion_job_failures_total{tier="1:fs"} 2
# TYPE vllm:kv_offload_tiering_backpressure_stores_dropped_total counter
vllm:kv_offload_tiering_backpressure_stores_dropped_total{tier="1:fs"} 3
"""


def make_monitor(tmp_path, **kwargs):
    opts = dict(
        enabled=True,
        directory=tmp_path,
        cpu_budget_bytes=12 * 1024 ** 3,
        disk_limit_bytes=0,
        device="/dev/nvme1n1",
        interval_s=30.0,
        health_interval_s=120.0,
    )
    opts.update(kwargs)
    return kvcache.KvCacheMonitor(**opts)


def test_parse_size_accepts_human_units():
    assert kvcache.parse_size("12G") == 12884901888
    assert kvcache.parse_size("8192m") == 8589934592
    assert kvcache.parse_size(1024) == 1024
    assert kvcache.parse_size("") == 0
    assert kvcache.parse_size("nonsense") == 0


def test_collect_engine_parses_all_three_tiers(tmp_path):
    mon = make_monitor(tmp_path)
    mon.collect_engine(Metrics(METRICS), now=1000.0)
    snap = mon.view()
    assert snap["gpu"]["pct"] == 25.0
    assert snap["gpu"]["blocks"] == 250
    assert snap["gpu"]["block_size"] == 16
    # CPU tier: vLLM exposes "pinned by in-flight transfers", reported as active
    assert snap["cpu"]["active_pct"] == 40.0
    assert snap["cpu"]["write_pct"] == 10.0
    disk = snap["disk"]
    assert disk["tier"] == "1:fs"
    assert disk["queries"] == 40
    assert disk["hits"] == 10
    assert disk["hit_rate"] == 0.25
    assert disk["promo_fail"] == 2
    assert disk["dropped_stores"] == 3


def test_disk_io_rate_comes_from_tier_counters(tmp_path):
    mon = make_monitor(tmp_path)
    first = METRICS.replace("1048576", "0").replace("2097152", "0")
    mon.collect_engine(Metrics(first), now=0.0)
    second = METRICS.replace("1048576", str(10 * 1024 ** 2)).replace(
        "2097152", str(20 * 1024 ** 2)
    )
    mon.collect_engine(Metrics(second), now=10.0)
    disk = mon.view()["disk"]
    assert disk["read_bps"] == pytest.approx(1024 ** 2)
    assert disk["write_bps"] == pytest.approx(2 * 1024 ** 2)


def test_dir_usage_walks_subdirectories(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.bin").write_bytes(b"x" * 100)
    (tmp_path / "two.bin").write_bytes(b"y" * 50)
    total, files = kvcache.dir_usage(tmp_path)
    assert total == 150
    assert files == 2


def test_collect_disk_reports_usage_and_capacity(tmp_path):
    (tmp_path / "blk.bin").write_bytes(b"z" * 2048)
    mon = make_monitor(tmp_path)
    mon.collect_disk(now=5.0)
    disk = mon.view()["disk"]
    assert disk["used_bytes"] == 2048
    assert disk["files"] == 1
    assert disk["capacity_bytes"] > 0  # falls back to the filesystem size


def test_reclaim_deletes_oldest_files_until_under_limit(tmp_path):
    paths = []
    for i in range(10):
        path = tmp_path / f"blk{i:02d}.bin"
        path.write_bytes(b"q" * 1000)
        os.utime(path, (1_000_000 + i, 1_000_000 + i))
        paths.append(path)
    mon = make_monitor(tmp_path, disk_limit_bytes=5_000)
    mon.collect_disk(now=9.0)
    disk = mon.view()["disk"]
    # 90 % watermark of 5_000 = 4_500 -> keep 4 files
    assert disk["used_bytes"] <= 4_500
    assert disk["files"] == 4
    assert not paths[0].exists()
    assert paths[-1].exists()
    assert mon.view()["reclaim"]["last_files"] == 6


def test_no_reclaim_when_limit_is_zero(tmp_path):
    for i in range(5):
        (tmp_path / f"blk{i}.bin").write_bytes(b"q" * 1000)
    mon = make_monitor(tmp_path, disk_limit_bytes=0)
    mon.collect_disk(now=1.0)
    assert mon.view()["disk"]["files"] == 5
    assert mon.view()["reclaim"]["last_at"] == 0.0


def test_parse_smart_normalizes_nvme_json():
    payload = {
        "critical_warning": 0,
        "temperature": 331,
        "avail_spare": 100,
        "percent_used": 2,
        "data_units_written": 737311,
        "data_units_read": 55791,
        "media_errors": 0,
        "power_on_hours": 1,
        "unsafe_shutdowns": 1,
    }
    health = kvcache.parse_smart(payload)
    assert health["ok"] is True
    assert health["temp_c"] == pytest.approx(57.9, abs=0.1)
    assert health["pct_used"] == 2
    assert health["written_gb"] == pytest.approx(
        737311 * 1000 * 512 / 1024 ** 3, abs=0.1
    )


def test_parse_smart_flags_critical_warning():
    health = kvcache.parse_smart({"critical_warning": 4, "temperature": 300})
    assert health["ok"] is False


def test_collect_health_reports_error_when_sudo_missing(tmp_path, monkeypatch):
    mon = make_monitor(tmp_path)

    def boom(*_args, **_kwargs):
        raise FileNotFoundError("sudo")

    monkeypatch.setattr(kvcache.subprocess, "run", boom)
    mon.collect_health(now=1.0)
    health = mon.view()["health"]
    assert health["ok"] is False
    assert "sudo" in (health["error"] or "")


def test_collect_health_parses_nvme_output(tmp_path, monkeypatch):
    payload = {"critical_warning": 0, "temperature": 300, "percent_used": 3,
               "avail_spare": 99, "media_errors": 0, "data_units_written": 1000,
               "data_units_read": 500, "power_on_hours": 12, "unsafe_shutdowns": 0}

    class Proc:
        returncode = 0
        stdout = json.dumps(payload)
        stderr = ""

    monkeypatch.setattr(kvcache.subprocess, "run", lambda *a, **k: Proc())
    mon = make_monitor(tmp_path)
    mon.collect_health(now=1.0)
    health = mon.view()["health"]
    assert health["ok"] is True
    assert health["pct_used"] == 3
    assert health["device"] == "/dev/nvme1n1"


def test_health_is_throttled(tmp_path):
    mon = make_monitor(tmp_path, health_interval_s=120.0)
    assert mon.health_due(1000.0) is True
    mon._last_health = 1000.0
    assert mon.health_due(1100.0) is False
    assert mon.health_due(1121.0) is True


def test_disabled_monitor_reports_not_enabled(tmp_path):
    mon = make_monitor(tmp_path, enabled=False)
    assert mon.view()["enabled"] is False


def test_counter_suffix_both_spellings(tmp_path):
    """vLLM 0.31 exports counters as ``*_total``; the reader accepts either."""
    legacy = METRICS.replace("_bytes_total{", "_bytes{").replace(
        "chunk_queries_total{", "chunk_queries{")
    mon = make_monitor(tmp_path)
    mon.collect_engine(Metrics(legacy), now=1.0)
    assert mon.view()["disk"]["queries"] == 40
    assert mon.view()["disk"]["tier"] == "1:fs"


def test_reclaim_falls_back_to_sudo_when_unprivileged(tmp_path, monkeypatch):
    """Engine files are root-owned; an unprivileged monitor must still shrink."""
    for i in range(6):
        path = tmp_path / f"blk{i}.bin"
        path.write_bytes(b"q" * 1000)
        os.utime(path, (1_000_000 + i, 1_000_000 + i))
    mon = make_monitor(tmp_path, disk_limit_bytes=2_000)

    def denied(_path):
        raise PermissionError(13, "Permission denied")

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    real_unlink = os.unlink  # kvcache.os is the os module, so keep the original

    def fake_run(cmd, **_kwargs):
        assert cmd[:4] == ["sudo", "-n", "rm", "-f"]
        for path in [c for c in cmd[4:] if c != "--"]:
            real_unlink(path)  # simulate root deleting the file
        return Proc()

    monkeypatch.setattr(kvcache.os, "unlink", denied)
    monkeypatch.setattr(kvcache.subprocess, "run", fake_run)
    mon.collect_disk(now=7.0)
    reclaim = mon.view()["reclaim"]
    assert reclaim["error"] is None
    assert reclaim["last_files"] > 0
    assert mon.view()["disk"]["used_bytes"] <= 1_800


def test_reclaim_reports_sudo_failure(tmp_path, monkeypatch):
    for i in range(4):
        (tmp_path / f"blk{i}.bin").write_bytes(b"q" * 1000)
    mon = make_monitor(tmp_path, disk_limit_bytes=1_000)
    monkeypatch.setattr(kvcache.os, "unlink", lambda _p: (_ for _ in ()).throw(
        PermissionError(13, "Permission denied")))

    class Proc:
        returncode = 1
        stdout = ""
        stderr = "sudo: a password is required"

    monkeypatch.setattr(kvcache.subprocess, "run", lambda *a, **k: Proc())
    mon.collect_disk(now=8.0)
    assert "password" in (mon.view()["reclaim"]["error"] or "")


def test_apply_launch_retargets_collector(tmp_path):
    """A single launch can override dir/budget/limit; the collector must follow."""
    mon = make_monitor(tmp_path)
    target = tmp_path / "kvoffload"
    target.mkdir()
    mon.apply_launch({
        "offload_enabled": "1",
        "kvoffload_dir": str(target),
        "cpu_bytes_to_use": "4G",
        "kv_disk_limit_gb": "8",
    })
    snap = mon.view()
    assert snap["enabled"] is True
    assert snap["dir"] == str(target)
    assert snap["cpu"]["budget_bytes"] == 4 * 1024 ** 3
    assert snap["disk"]["limit_bytes"] == 8 * 1024 ** 3


def test_apply_launch_can_disable_offload(tmp_path):
    mon = make_monitor(tmp_path)
    mon.apply_launch({"offload_enabled": "0"})
    assert mon.view()["enabled"] is False
