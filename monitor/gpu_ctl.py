"""GPU power-limit control (`nvidia-smi -pl`) with persistence.

The value the operator picks is written to a tiny file that the systemd unit reads
on boot and after every driver reload, so a change made here survives a reboot
without rewriting unit files:

    nvidia-power-limit.service:
        ExecStart=/bin/bash -c 'W=$(cat /etc/nvidia-power-limit.watts ...); nvidia-smi -i 0 -pl "$W"'

The driver only accepts values inside [power.min_limit, power.max_limit]; anything
else exits 2, so the range check happens here and is surfaced to the UI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

WATTS_FILE = Path("/etc/nvidia-power-limit.watts")
DEFAULT_WATTS = 270.0
TIMEOUT_S = 20


def validate(watts: float | str | None, min_w: float | None,
             max_w: float | None) -> tuple[float | None, str | None]:
    """Return (value, error). Rejects non-numeric and out-of-range requests."""
    try:
        value = float(watts)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, f"不是数字：{watts!r}"
    if value <= 0:
        return None, "功耗必须是正数"
    if min_w is not None and value < min_w:
        return None, f"低于驱动允许的最小值 {min_w:.0f} W"
    if max_w is not None and value > max_w:
        return None, f"高于驱动允许的最大值 {max_w:.0f} W"
    return round(value, 1), None


def _smi(*args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(["sudo", "-n", "nvidia-smi", *args],
                              capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    out = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode, out


def persisted_watts() -> float | None:
    """Value the boot unit will apply (the file we own)."""
    try:
        return float(WATTS_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def query_limits() -> dict:
    """Current / min / max / default + persisted, straight from the driver."""
    code, out = _smi(
        "--query-gpu=power.limit,power.min_limit,power.max_limit,power.default_limit,"
        "enforced.power.limit,power.draw",
        "--format=csv,noheader,nounits",
    )
    data: dict = {"ok": code == 0, "error": None if code == 0 else out,
                  "persisted_w": persisted_watts()}
    if code != 0:
        return data
    try:
        limit, min_w, max_w, default_w, enforced, draw = [
            float(x.strip()) for x in out.splitlines()[0].split(",")]
    except (ValueError, IndexError):
        return {**data, "ok": False, "error": f"无法解析 nvidia-smi 输出：{out[:120]}"}
    return {**data, "limit_w": limit, "min_w": min_w, "max_w": max_w,
            "default_w": default_w, "enforced_w": enforced, "draw_w": draw}


def apply_power_limit(watts: float | str) -> dict:
    """Set the limit live and persist it for the boot unit. Never raises."""
    limits = query_limits()
    if not limits.get("ok"):
        return {"ok": False, "error": limits.get("error") or "nvidia-smi 不可用", **limits}
    value, error = validate(watts, limits.get("min_w"), limits.get("max_w"))
    if error:
        return {"ok": False, "error": error, "min_w": limits.get("min_w"),
                "max_w": limits.get("max_w")}
    code, out = _smi("-i", "0", "-pl", f"{value:g}")
    if code != 0:
        return {"ok": False, "error": out or f"nvidia-smi 退出码 {code}",
                "min_w": limits.get("min_w"), "max_w": limits.get("max_w")}
    # Persist for boot / driver reload: the unit reads this file, so we never have
    # to rewrite a systemd file from the web API.
    try:
        proc = subprocess.run(
            ["sudo", "-n", "tee", str(WATTS_FILE)], input=f"{value:g}\n",
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        persisted_ok = proc.returncode == 0
        persist_error = None if persisted_ok else (proc.stderr or "").strip() or "写入失败"
    except (OSError, subprocess.SubprocessError) as exc:
        persisted_ok, persist_error = False, f"{type(exc).__name__}: {exc}"
    # Echo the *configured* limit (power.limit), not enforced.power.limit: NVML
    # only refreshes the enforced figure on its own power-monitoring cycle, which
    # lagged far beyond any reasonable poll (measured: limit=260 while
    # enforced still read 270 several seconds later). The UI wants the setpoint.
    after = query_limits()
    return {"ok": True, "requested_w": value,
            "applied_w": after.get("limit_w", after.get("enforced_w")),
            "out": out, "persisted": persisted_ok, "persist_error": persist_error,
            **after}
