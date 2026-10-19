"""GPU telemetry via NVML (ctypes, no extra dependency) with nvidia-smi fallback."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
from pathlib import Path


def _cpu_name(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return "?"


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _ProcessV1(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("used_gpu_memory", ctypes.c_ulonglong)]


class _ProcessV2(ctypes.Structure):
    # nvmlProcessInfo_v2_t: usedGpuMemory comes BEFORE the instance ids.
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


class _ProcessV3(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


# nvmlClocksThrottleReasons bits (nvml.h)
THROTTLE_REASONS = {
    0x0000000000000001: "idle",
    0x0000000000000002: "app_clocks",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000010: "sync_boost",
    0x0000000000000020: "sw_thermal",
    0x0000000000000040: "hw_thermal",
    0x0000000000000080: "hw_power_brake",
    0x0000000000000100: "display_clock",
}


class NvmlGpu:
    """One GPU handle."""

    def __init__(self, lib: ctypes.CDLL, index: int, handle: ctypes.c_void_p) -> None:
        self.lib = lib
        self.index = index
        self.handle = handle

    def _str(self, fn_name: str, size: int = 96) -> str:
        buf = ctypes.create_string_buffer(size)
        rc = getattr(self.lib, fn_name)(self.handle, buf, ctypes.c_uint(size))
        return buf.value.decode("utf-8", "replace") if rc == 0 else ""

    def _uint(self, fn_name: str, *args) -> float | None:
        out = ctypes.c_uint()
        rc = getattr(self.lib, fn_name)(self.handle, *args, ctypes.byref(out))
        return float(out.value) if rc == 0 else None

    def query(self) -> dict:
        lib = self.lib
        info: dict = {"index": self.index}

        info["name"] = self._str("nvmlDeviceGetName")
        mem = _Memory()
        if lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem)) == 0:
            info["mem_total_mib"] = round(mem.total / 1048576)
            info["mem_used_mib"] = round(mem.used / 1048576)
            info["mem_free_mib"] = round(mem.free / 1048576)
            info["mem_used_pct"] = round(100.0 * mem.used / mem.total, 1) if mem.total else 0.0

        util = _Utilization()
        if lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util)) == 0:
            info["util_pct"] = float(util.gpu)
            info["mem_util_pct"] = float(util.memory)

        info["temp_c"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))
        info["temp_mem_c"] = None
        if hasattr(lib, "nvmlDeviceGetTemperatureThreshold"):
            thr = ctypes.c_uint()
            if lib.nvmlDeviceGetTemperatureThreshold(self.handle, ctypes.c_uint(0), ctypes.byref(thr)) == 0:
                info["temp_limit_c"] = float(thr.value)

        power_mw = self._uint("nvmlDeviceGetPowerUsage")
        info["power_w"] = round(power_mw / 1000.0, 1) if power_mw is not None else None
        # power_limit_w = the configured setpoint (what the operator set with -pl);
        # power_enforced_w = what the hardware is enforcing right now, which NVML
        # only refreshes on its own power-monitoring cycle (can lag for seconds).
        setpoint_mw = self._uint("nvmlDeviceGetPowerManagementLimit")
        info["power_limit_w"] = round(setpoint_mw / 1000.0, 1) if setpoint_mw is not None else None
        enforced_mw = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        info["power_enforced_w"] = (round(enforced_mw / 1000.0, 1)
                                    if enforced_mw is not None else None)

        info["fan_pct"] = self._uint("nvmlDeviceGetFanSpeed")
        info["sm_clock_mhz"] = self._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(0))
        info["mem_clock_mhz"] = self._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(1))
        info["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")
        info["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")

        reasons = ctypes.c_ulonglong()
        active: list[str] = []
        if lib.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle, ctypes.byref(reasons)) == 0:
            for bit, label in THROTTLE_REASONS.items():
                if reasons.value & bit and label != "idle":
                    active.append(label)
        info["throttle"] = active

        info["procs"] = self._procs()
        return info

    def _procs(self) -> list[dict]:
        lib = self.lib
        for fn_name, struct in (
            ("nvmlDeviceGetComputeRunningProcesses_v3", _ProcessV3),
            ("nvmlDeviceGetComputeRunningProcesses_v2", _ProcessV2),
            ("nvmlDeviceGetComputeRunningProcesses", _ProcessV1),
        ):
            fn = getattr(lib, fn_name, None)
            if fn is None:
                continue
            count = ctypes.c_uint(64)
            arr = (struct * 64)()
            rc = fn(self.handle, ctypes.byref(count), arr)
            if rc == 7:  # INSUFFICIENT_SIZE
                count = ctypes.c_uint(256)
                arr = (struct * 256)()
                rc = fn(self.handle, ctypes.byref(count), arr)
            if rc != 0:
                continue
            out = []
            for i in range(min(count.value, len(arr))):
                proc = arr[i]
                pid = int(proc.pid)
                out.append(
                    {
                        "pid": pid,
                        "name": _cpu_name(pid),
                        "mem_mib": round(proc.used_gpu_memory / 1048576),
                    }
                )
            return out
        return []


class GpuSampler:
    def __init__(self, fallback_csv: bool = True) -> None:
        self.fallback_csv = fallback_csv
        self.mode = "off"
        self.lib: ctypes.CDLL | None = None
        self.gpus: list[NvmlGpu] = []
        self.error: str | None = None
        self._init()

    # -- init ---------------------------------------------------------------
    def _init(self) -> None:
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            lib.nvmlInit_v2.restype = ctypes.c_int
            if lib.nvmlInit_v2() != 0:
                raise RuntimeError("nvmlInit failed")
            count = ctypes.c_uint()
            lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
            handles: list[NvmlGpu] = []
            for i in range(count.value):
                h = ctypes.c_void_p()
                if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(h)) == 0:
                    handles.append(NvmlGpu(lib, i, h))
            self.lib = lib
            self.gpus = handles
            self.mode = "nvml"
        except Exception as exc:  # pragma: no cover - depends on host
            self.error = str(exc)
            if self.fallback_csv and shutil.which("nvidia-smi"):
                self.mode = "nvidia-smi"

    # -- sample -------------------------------------------------------------
    def sample(self) -> list[dict]:
        if self.mode == "nvml":
            out = []
            for gpu in self.gpus:
                try:
                    out.append(gpu.query())
                except Exception as exc:  # pragma: no cover
                    out.append({"index": gpu.index, "error": str(exc)})
            return out
        if self.mode == "nvidia-smi":
            return self._sample_csv()
        return []

    def _sample_csv(self) -> list[dict]:
        fields = (
            "index,name,memory.total,memory.used,utilization.gpu,utilization.memory,"
            "temperature.gpu,power.draw,power.limit,fan.speed,clocks.sm,clocks.mem,"
            "pcie.link.gen.current,pcie.link.width.current"
        )
        try:
            raw = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
        except Exception:
            return []
        out = []
        for line in raw.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 14:
                continue
            try:
                out.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "mem_total_mib": int(float(parts[2])),
                        "mem_used_mib": int(float(parts[3])),
                        "mem_used_pct": round(100 * float(parts[3]) / float(parts[2]), 1),
                        "util_pct": float(parts[4]),
                        "mem_util_pct": float(parts[5]),
                        "temp_c": float(parts[6]),
                        "power_w": float(parts[7]),
                        "power_limit_w": float(parts[8]),
                        "fan_pct": float(parts[9]) if parts[9] not in ("[N/A]", "N/A") else None,
                        "sm_clock_mhz": float(parts[10]),
                        "mem_clock_mhz": float(parts[11]),
                        "pcie_gen": float(parts[12]),
                        "pcie_width": float(parts[13]),
                        "throttle": [],
                        "procs": [],
                    }
                )
            except ValueError:
                continue
        return out


class GpuCollector:
    """Samples at a fixed interval and keeps the last N points for sparklines."""

    def __init__(self, interval_s: float = 1.0, fallback_csv: bool = True) -> None:
        self.sampler = GpuSampler(fallback_csv=fallback_csv)
        self.interval_s = interval_s
        self.latest: list[dict] = []
        self.updated_at: float = 0.0
        self.history_len = max(120, int(600 / max(interval_s, 0.2)))
        self.history: list[dict] = []

    @property
    def first(self) -> dict:
        return self.latest[0] if self.latest else {}

    def sample(self) -> list[dict]:
        data = self.sampler.sample()
        if data:
            self.latest = data
            self.updated_at = time.time()
            self.history.append(
                {
                    "ts": self.updated_at,
                    "util": data[0].get("util_pct"),
                    "vram_pct": data[0].get("mem_used_pct"),
                    "power": data[0].get("power_w"),
                    "temp": data[0].get("temp_c"),
                    "fan": data[0].get("fan_pct"),
                }
            )
            if len(self.history) > self.history_len:
                del self.history[: len(self.history) - self.history_len]
        return self.latest

    def info(self) -> dict:
        return {
            "mode": self.sampler.mode,
            "error": self.sampler.error,
            "gpus": self.latest,
            "updated_at": self.updated_at,
        }

    def used_mib(self) -> int:
        return int(sum(g.get("mem_used_mib") or 0 for g in self.latest))
