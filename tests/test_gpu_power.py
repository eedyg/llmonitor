"""GPU 功耗上限的校验与持久化逻辑（monitor/gpu_ctl.py）。"""

from __future__ import annotations

import json

from monitor import gpu_ctl


def test_validate_accepts_the_driver_range():
    assert gpu_ctl.validate(270, 250, 300) == (270.0, None)
    assert gpu_ctl.validate("250", 250, 300) == (250.0, None)
    assert gpu_ctl.validate(300, 250, 300) == (300.0, None)


def test_validate_rejects_out_of_range():
    value, error = gpu_ctl.validate(200, 250, 300)
    assert value is None and "最小值" in error
    value, error = gpu_ctl.validate(350, 250, 300)
    assert value is None and "最大值" in error


def test_validate_rejects_junk():
    for bad in (None, "", "abc", 0, -5):
        value, error = gpu_ctl.validate(bad, 250, 300)
        assert value is None and error


def test_validate_without_a_known_range_still_checks_positivity():
    assert gpu_ctl.validate(275, None, None) == (275.0, None)


def test_apply_power_limit_refuses_before_touching_the_gpu(monkeypatch):
    """越界请求必须在调用 nvidia-smi -pl 之前就被拒绝。"""
    called = []
    monkeypatch.setattr(gpu_ctl, "query_limits", lambda: {
        "ok": True, "limit_w": 270.0, "min_w": 250.0, "max_w": 300.0,
        "default_w": 300.0, "enforced_w": 270.0, "draw_w": 100.0, "persisted_w": 270.0})
    monkeypatch.setattr(gpu_ctl, "_smi", lambda *a: called.append(a) or (0, ""))
    out = gpu_ctl.apply_power_limit(999)
    assert out["ok"] is False and "最大值" in out["error"]
    assert called == [], "越界时不应调用 nvidia-smi"


def test_apply_power_limit_persists_for_the_boot_unit(monkeypatch):
    written = {}

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        if cmd[0] == "sudo" and cmd[1] == "-n" and cmd[2] == "tee":
            written["path"] = cmd[3]
            written["value"] = kwargs.get("input")
            return Proc()
        raise AssertionError(f"unexpected run: {cmd}")

    monkeypatch.setattr(gpu_ctl, "query_limits", lambda: {
        "ok": True, "min_w": 250.0, "max_w": 300.0, "enforced_w": 275.0,
        "persisted_w": 270.0, "limit_w": 275.0, "default_w": 300.0, "draw_w": 1.0})
    monkeypatch.setattr(gpu_ctl, "_smi", lambda *a: (0, "was set to 275.00 W"))
    monkeypatch.setattr(gpu_ctl.subprocess, "run", fake_run)
    out = gpu_ctl.apply_power_limit(275)
    assert out["ok"] is True and out["persisted"] is True
    assert str(gpu_ctl.WATTS_FILE) == written["path"]
    assert written["value"].strip() == "275"
