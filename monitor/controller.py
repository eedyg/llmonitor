"""Engine lifecycle: start / stop / restart / switch / probe.

P2b: the launch command is built natively from engines/*.yaml (launcher.py).
Set `engine.launcher: false` (or MONITOR_ENGINE_LAUNCHER=0) to fall back to the
old `../run_llm.sh` path.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from . import launcher
from .launcher import LaunchPlan
from .runtime import Runtime


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def match_startup(patterns: list[dict], lines: list[str]) -> tuple[int, str, str, bool]:
    """Pick the most advanced startup stage visible in the log tail.

    Returns (stage_index, phase, matching_line, is_error). Error markers win over
    progress markers; otherwise the highest index wins (patterns are ordered).
    """
    best_score = -1
    phase = ""
    detail = ""
    is_error = False
    for index, pattern in enumerate(patterns):
        try:
            rx = re.compile(str(pattern.get("match")), re.I)
        except re.error:
            continue
        for line in lines:
            clean = _clean(line)
            if not rx.search(clean):
                continue
            score = 10_000 if pattern.get("error") else index
            if score > best_score:
                best_score = score
                phase = str(pattern.get("phase"))
                detail = clean
                is_error = bool(pattern.get("error"))
    return max(best_score, -1), phase, detail, is_error


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Controller:
    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self.cfg = rt.cfg
        self.state = "unknown"
        self.kind: str | None = None
        self.detail: str | None = None
        self.since = time.time()
        self.health_ms: float | None = None
        self.last_action: dict | None = None
        self.start_command: list[str] | None = None
        self.startup: dict | None = None
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._process_cache: tuple[float, list[str]] = (0.0, [])
        self._gpu_free: int | None = None
        self._start_at: float = 0.0
        self._log_offset: int = 0
        self._child: subprocess.Popen | None = None

    # -- paths / helpers ----------------------------------------------------
    @property
    def run_llm(self) -> Path:
        return self.cfg.p("paths.run_llm")

    @property
    def project_dir(self) -> Path:
        return self.run_llm.parent

    @property
    def run_dir(self) -> Path:
        return self.cfg.p("paths.run_dir")

    @property
    def engine_port(self) -> int:
        return int(self.cfg.root.server.engine_port)

    @property
    def gateway_port(self) -> int:
        return int(self.cfg.root.server.gateway_port)

    @property
    def use_launcher(self) -> bool:
        return bool(self.cfg.root.engine.launcher)

    def _set(self, state: str, detail: str | None = None, kind: str | None = None) -> None:
        changed = state != self.state or detail != self.detail
        self.state = state
        self.detail = detail
        if kind is not None:
            self.kind = kind
        self.since = time.time()
        if changed:
            self.rt.publish(
                {"type": "engine_state", "state": state, "engine": self.kind, "detail": detail,
                 "ts": time.time()}
            )

    def info(self) -> dict:
        return {
            "state": self.state,
            "engine": self.kind,
            "detail": self.detail,
            "since": self.since,
            "since_iso": datetime.fromtimestamp(self.since).isoformat(timespec="seconds"),
            "health_ms": self.health_ms,
            "last_action": self.last_action,
            "start_command": self.start_command,
            "engine_port": self.engine_port,
            "gateway_port": self.gateway_port,
            "gpu_free_mib": self._gpu_free,
            "run_llm": str(self.run_llm),
            "launcher": self.use_launcher,
            "plan_source": "launcher" if self.use_launcher else "run_llm.sh",
            "startup": self.startup,
            "image": getattr(self.rt, "image_override", None),
            "profiles": list(self.cfg.root.engine.profiles),
            "defaults": {
                "engine": self.cfg.root.engine.default,
                "profiles": list(self.cfg.root.engine.profiles),
                "ctx": dict(self.cfg.root.engine.default_ctx),
            },
        }

    # -- process detection --------------------------------------------------
    def _containers_running(self) -> list[str]:
        now = time.time()
        stamp, cached = self._process_cache
        if now - stamp < 2.0:
            return cached
        names: list[str] = []
        try:
            out = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            names = [line.strip() for line in out.splitlines() if line.strip()]
        except Exception:
            names = []
        self._process_cache = (now, names)
        return names

    def _ninfer_pid(self) -> int | None:
        name = self.cfg.root.engine.container_names.ninfer
        pid_file = self.run_dir / f"{name}.pid"
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return pid
        except Exception:
            return None

    def detect_engine(self) -> str | None:
        running = set(self._containers_running())
        names = self.cfg.root.engine.container_names
        if names.vllm in running:
            return "vllm"
        if names.ninfer in running:
            return "ninfer"
        if self._ninfer_pid() is not None:
            return "ninfer"
        # NOTE: no fallback to the metrics snapshot here — it lags a stop by a
        # few seconds and would make start() think a stale engine still exists.
        return None

    def process_alive(self) -> bool:
        return self.detect_engine() is not None

    # -- health -------------------------------------------------------------
    async def health(self, timeout: float = 2.0) -> bool:
        url = f"http://127.0.0.1:{self.engine_port}/health"
        started = time.time()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
            self.health_ms = round((time.time() - started) * 1000.0, 1)
            if resp.status_code != 200:
                return False
            try:
                data = resp.json()
            except Exception:
                return True
            return data.get("status") in (None, "ok", "healthy")
        except Exception:
            self.health_ms = None
            return False

    async def refresh(self) -> None:
        healthy = await self.health()
        live_kind = self.detect_engine()
        if live_kind:
            self.kind = live_kind
        elif healthy and not self.kind:
            self.kind = self.rt.engine.kind
        if healthy:
            if self.state != "healthy":
                self._finish_startup()
                self._set("healthy", "engine is serving", live_kind)
            else:
                self.state = "healthy"
            return
        if self.state == "starting":
            return
        if self.state == "stopping":
            if not live_kind:
                self._set("stopped", "engine stopped", self.kind)
            return
        if live_kind:
            if self.state != "unhealthy":
                self._set("unhealthy", "process is up but /health is not OK", live_kind)
        elif self.state not in ("stopped", "failed"):
            self._set("stopped", "engine is not running")

    # -- command / plan building -------------------------------------------
    def build_command(self, opts: dict, dry_run: bool = False) -> tuple[list[str], dict[str, str]]:
        """Legacy path: shell out to ../run_llm.sh (launcher disabled)."""
        engine = (opts.get("engine") or self.cfg.root.engine.default).lower()
        if engine not in ("vllm", "ninfer"):
            raise ValueError(f"unknown engine '{engine}'")
        profile = opts.get("profile") or "stage1"
        if profile not in list(self.cfg.root.engine.profiles):
            raise ValueError(f"unknown profile '{profile}'")
        ctx = opts.get("ctx") or dict(self.cfg.root.engine.default_ctx).get(engine, "128k")

        cmd: list[str] = [
            str(self.run_llm),
            "--engine", engine,
            "--profile", str(profile),
            "--host", str(self.cfg.root.engine.bind_host),
            "--ctx", str(ctx),
        ]
        if opts.get("kv_capacity"):
            cmd += ["--kv-capacity", str(opts["kv_capacity"])]
        if opts.get("spec_tokens"):
            cmd += ["--spec-tokens", str(opts["spec_tokens"])]
        if opts.get("model_dir"):
            cmd += ["--model-dir", str(opts["model_dir"])]
        if opts.get("served_name"):
            cmd += ["--served-name", str(opts["served_name"])]
        if dry_run:
            cmd.append("--dry-run")

        extra = [str(a) for a in (opts.get("extra_args") or [])]
        if engine == "ninfer" and bool(self.cfg.root.engine.ninfer_jsonl):
            jsonl = self.cfg.root.engine.ninfer_jsonl_path
            jsonl_path = self.cfg.p("paths.run_dir") / Path(str(jsonl)).name
            extra = [
                "--request-log-jsonl", str(jsonl_path),
                "--log-stats-interval-ms", str(int(self.cfg.root.engine.ninfer_stats_interval_ms)),
                *extra,
            ]
        if extra:
            cmd += ["--"] + extra

        env = os.environ.copy()
        env.update(
            {
                "HOST_PORT": str(self.engine_port),
                "COMPAT": "0",           # the monitor gateway is the public port now
                "RUN_DIR": str(self.run_dir),
                "ENGINE": engine,
            }
        )
        for key, value in dict(self.cfg.root.engine.extra_env or {}).items():
            env[str(key)] = str(value)
        if engine == "vllm" and opts.get("image"):
            env["VLLM_IMAGE"] = str(opts["image"])
        return cmd, env

    def build_plan(self, opts: dict) -> LaunchPlan:
        engine = (opts.get("engine") or self.cfg.root.engine.default).lower()
        merged = dict(opts)
        merged.setdefault("host", str(self.cfg.root.engine.bind_host))
        merged["port"] = self.engine_port
        merged.setdefault("profile", "stage1")
        if not merged.get("ctx"):
            merged["ctx"] = dict(self.cfg.root.engine.default_ctx).get(engine, "128k")
        if engine == "ninfer" and bool(self.cfg.root.engine.ninfer_jsonl):
            extra = [str(a) for a in (merged.get("extra_args") or [])]
            jsonl_path = self.run_dir / Path(str(self.cfg.root.engine.ninfer_jsonl_path)).name
            merged["extra_args"] = [
                "--request-log-jsonl", str(jsonl_path),
                "--log-stats-interval-ms", str(int(self.cfg.root.engine.ninfer_stats_interval_ms)),
                *extra,
            ]
        for key, value in dict(self.cfg.root.engine.extra_env or {}).items():
            os.environ.setdefault(str(key), str(value))
        plan = launcher.resolve(
            engine,
            merged,
            env=os.environ,
            project_dir=self.project_dir,
            run_dir=self.run_dir,
            check_docker=True,
            check_files=True,
        )
        if opts.get("image") and plan.ok:
            plan.image = str(opts["image"])
        return plan

    # -- actions ------------------------------------------------------------
    async def start(self, opts: dict) -> dict:
        if self._lock.locked():
            return {"ok": False, "error": "another lifecycle action is in progress"}
        async with self._lock:
            healthy = await self.health()
            if healthy:
                return {"ok": False, "error": "engine already running / healthy", "state": self.state}
            self._process_cache = (0.0, [])   # never trust a cached docker ps here
            if self.process_alive():
                return {
                    "ok": False,
                    "error": "a stale engine process/container exists; stop it first (or use switch)",
                    "state": self.state,
                }
            if self.use_launcher:
                return await self._start_launcher(opts)
            return await self._start_script(opts)

    async def _start_script(self, opts: dict) -> dict:
        cmd, env = self.build_command(opts)
        if opts.get("dry_run"):
            code, out = await self._run(cmd, env, timeout=120)
            return {"ok": code == 0, "command": cmd, "output": out, "dry_run": True}
        engine = opts.get("engine") or self.cfg.root.engine.default
        self.kind = engine
        self.rt.engine_api_key = str(opts.get("api_key") or "") or None
        self.start_command = cmd
        self._set("starting", f"launching {engine}…", engine)
        code, out = await self._run(cmd, env, timeout=600, log=True)
        if code != 0:
            self._set("failed", f"run_llm.sh exited {code}")
            return {"ok": False, "error": f"run_llm.sh exited {code}", "command": cmd,
                    "output": out[-4000:]}
        self.last_action = {"action": "start", "engine": engine, "at": _now_iso(),
                            "source": "run_llm.sh"}
        deadline = time.time() + float(self.cfg.root.engine.health_timeout_s)
        self._task = asyncio.create_task(self._await_health(deadline, None))
        return {"ok": True, "state": self.state, "command": cmd, "output": out[-2000:],
                "source": "run_llm.sh"}

    async def _start_launcher(self, opts: dict) -> dict:
        plan = await asyncio.to_thread(self.build_plan, opts)
        if opts.get("dry_run"):
            return {"ok": plan.ok, "dry_run": True, "command": plan.argv,
                    "command_text": plan.command(), "plan": plan.public(),
                    "output": "\n".join(plan.errors)}
        if not plan.ok:
            detail = "; ".join(plan.errors)
            self._set("failed", detail)
            return {"ok": False, "error": plan.errors, "detail": detail, "plan": plan.public()}

        self.kind = plan.engine
        self.rt.engine_api_key = str(opts.get("api_key") or "") or None
        self.start_command = plan.argv
        self._start_at = time.time()
        self._log_offset = 0
        if plan.logfile:
            try:
                self._log_offset = Path(plan.logfile).stat().st_size
            except OSError:
                self._log_offset = 0
        # A stopped engine can leave the port bound (or the container name taken).
        await self._cleanup_before_start(plan)
        if self._port_listening(self.engine_port):
            await self._wait_port_free(self.engine_port, 90)
        if plan.kind == "process":
            # NInfer does not set SO_REUSEADDR: a TIME-WAIT socket on this port
            # blocks the bind for ~60s. run_llm.sh waited for it; so do we.
            waited = await self._wait_time_wait(self.engine_port)
            if waited:
                self.rt.publish({"type": "engine_state", "state": "starting",
                                 "detail": f"waited {waited}s for TIME-WAIT on :{self.engine_port}"})
        self.startup = self._new_startup(plan)
        self._set("starting", f"launching {plan.engine} · {plan.profile}…", plan.engine)
        self.rt.publish({"type": "startup", "startup": self.startup})

        code, out = await self._spawn(plan)
        if code != 0:
            tail = await self._engine_log_tail(40)
            message = (tail[-1] if tail else out.strip()[-400:]) or f"spawn exited {code}"
            self._fail_startup(message, tail)
            return {"ok": False, "error": message, "output": out[-4000:], "plan": plan.public()}

        self.last_action = {"action": "start", "engine": plan.engine, "at": _now_iso(),
                            "source": "launcher", "profile": plan.profile}
        deadline = time.time() + float(self.cfg.root.engine.health_timeout_s)
        self._task = asyncio.create_task(self._await_health(deadline, plan))
        return {"ok": True, "state": self.state, "command": plan.argv,
                "command_text": plan.command(), "plan": plan.public(), "source": "launcher",
                "output": out[-1000:]}

    def _port_listening(self, port: int) -> bool:
        try:
            out = subprocess.run(
                ["ss", "-ltnH", f"sport = :{port}"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            return any(line.strip() for line in out.splitlines())
        except Exception:
            return False

    async def _wait_port_free(self, port: int, timeout: float = 90.0) -> float:
        started = time.time()
        self._set("starting", f"waiting for :{port} to be released…")
        while time.time() - started < timeout:
            if not self._port_listening(port):
                return round(time.time() - started, 1)
            await asyncio.sleep(1.0)
        return round(time.time() - started, 1)

    async def _cleanup_before_start(self, plan: LaunchPlan) -> None:
        """Remove a stale stopped container / dead pidfile with the same identity."""
        if plan.kind == "container":
            name = plan.container_name or self.cfg.root.engine.container_names.vllm
            code, out, _err = await asyncio.to_thread(
                self.rt.docker.run, ["ps", "-a", "--filter", f"name=^{name}$",
                                     "--format", "{{.State}}"], 15)
            state = out.strip().splitlines()[0] if code == 0 and out.strip() else ""
            if state and state != "running":
                self.rt.publish({"type": "engine_state", "state": "starting",
                                 "detail": f"removing stale container {name} ({state})"})
                await asyncio.to_thread(self.rt.docker.run, ["rm", name], 60)
            return
        name = plan.container_name or self.cfg.root.engine.container_names.ninfer
        pid_file = self.run_dir / f"{name}.pid"
        pid = self._ninfer_pid()
        if pid is not None:
            return
        pid_file.unlink(missing_ok=True)

    def _time_wait(self, port: int) -> bool:
        try:
            out = subprocess.run(
                ["ss", "-tanH", "state", "time-wait", f"sport = :{port}"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            return any(line.strip() for line in out.splitlines())
        except Exception:
            return False

    async def _wait_time_wait(self, port: int, timeout: float = 75.0) -> float:
        if not self._time_wait(port):
            return 0.0
        started = time.time()
        self._set("starting", f"waiting for TIME-WAIT sockets on :{port} (NInfer cannot rebind)…")
        while time.time() - started < timeout:
            if not self._time_wait(port):
                break
            await asyncio.sleep(1.0)
        return round(time.time() - started, 1)

    async def _spawn(self, plan: LaunchPlan) -> tuple[int, str]:
        if plan.kind == "container":
            proc = await asyncio.create_subprocess_exec(
                *plan.argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
            except asyncio.TimeoutError:
                proc.kill()
                return 124, "docker run timed out after 180s"
            return proc.returncode or 0, (out or b"").decode("utf-8", "replace")

        logfile = Path(plan.logfile) if plan.logfile else self.run_dir / f"{plan.container_name}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handle = logfile.open("ab")
        try:
            child = subprocess.Popen(
                plan.argv,
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(self.project_dir),
            )
        finally:
            handle.close()
        self._child = child
        if plan.pidfile:
            Path(plan.pidfile).parent.mkdir(parents=True, exist_ok=True)
            Path(plan.pidfile).write_text(str(child.pid))
        await asyncio.sleep(0.5)
        code = child.poll()
        if code is not None:
            return code, f"{plan.binary} exited immediately with {code}"
        return 0, f"pid {child.pid} · log {logfile}"

    def _new_startup(self, plan: LaunchPlan) -> dict:
        try:
            spec = launcher.load_spec(plan.engine)
        except Exception:
            spec = {}
        patterns = spec.get("startup") or []
        phases = [str(p.get("phase")) for p in patterns if not p.get("error")]
        return {
            "phase": phases[0] if phases else "启动",
            "index": 0,
            "total": len(phases),
            "phases": phases,
            "detail": "",
            "tail": [],
            "started_at": time.time(),
            "error": False,
        }

    def _finish_startup(self) -> None:
        if not self.startup:
            return
        self.startup.update(
            {"phase": "就绪", "index": self.startup.get("total", 0), "detail": "ready",
             "finished_at": time.time()}
        )
        self.rt.publish({"type": "startup", "startup": self.startup})

    def _fail_startup(self, message: str, tail: list[str] | None = None) -> None:
        if self.startup is None:
            self.startup = {"phase": "错误", "index": 0, "total": 0, "phases": [],
                            "detail": message, "tail": [], "started_at": time.time()}
        self.startup.update({"phase": "错误", "error": True, "detail": message,
                             "tail": (tail or [])[-8:], "finished_at": time.time()})
        self.rt.publish({"type": "startup", "startup": self.startup})
        self._set("failed", message)

    async def _engine_log_tail(self, lines: int = 40) -> list[str]:
        kind = self.kind
        if kind == "vllm":
            name = self.cfg.root.engine.container_names.vllm
            code, out = await self._run(
                ["docker", "logs", "--since", str(int(self._start_at or time.time() - 120)),
                 "--tail", str(lines), name],
                os.environ.copy(), timeout=10,
            )
            return [line for line in out.splitlines() if line.strip()][-lines:]
        name = self.cfg.root.engine.container_names.ninfer
        path = self.run_dir / f"{name}.log"
        try:
            with path.open("rb") as handle:
                if self._log_offset:
                    handle.seek(self._log_offset)
                blob = handle.read()
        except OSError:
            return []
        text = blob.decode("utf-8", "replace")
        return [line for line in text.splitlines() if line.strip()][-lines:]

    async def _watch_startup(self) -> None:
        if not self.startup:
            return
        try:
            spec = launcher.load_spec(self.kind or "vllm")
        except Exception:
            return
        patterns = spec.get("startup") or []
        while True:
            lines = await self._engine_log_tail(400)
            score, phase, detail, is_error = match_startup(patterns, lines)
            if score >= 0 and detail:
                phases = self.startup.get("phases") or []
                current = int(self.startup.get("index", 0))
                new_index = len(phases) if is_error else min(max(score, 0), max(len(phases) - 1, 0))
                if is_error or new_index != current or detail != self.startup.get("detail"):
                    self.startup.update(
                        {
                            "phase": "错误" if is_error else (phase or "启动"),
                            "index": new_index,
                            "detail": detail[:300],
                            "error": bool(self.startup.get("error")) or is_error,
                            "tail": [line[-200:] for line in lines[-6:]],
                        }
                    )
                    self.rt.publish({"type": "startup", "startup": self.startup})
            await asyncio.sleep(1.5)

    async def _await_health(self, deadline: float, plan: LaunchPlan | None) -> None:
        poll = float(self.cfg.root.engine.health_poll_s)
        watcher = asyncio.create_task(self._watch_startup())
        try:
            while time.time() < deadline:
                if await self.health():
                    self._finish_startup()
                    self._set("healthy", "engine is serving")
                    return
                if self.state == "starting" and not self.process_alive():
                    await asyncio.sleep(3)
                    if not self.process_alive():
                        tail = await self._engine_log_tail(40)
                        message = tail[-1][:300] if tail else "engine exited during startup"
                        self._fail_startup(message, tail)
                        return
                await asyncio.sleep(poll)
            tail = await self._engine_log_tail(40)
            self._fail_startup("timed out waiting for /health", tail)
        finally:
            watcher.cancel()

    async def stop(self, force: bool = False) -> dict:
        if self._lock.locked():
            return {"ok": False, "error": "another lifecycle action is in progress"}
        async with self._lock:
            kind = self.detect_engine() or self.kind or self.cfg.root.engine.default
            self._set("stopping", f"stopping {kind}…", kind)
            if kind == "vllm":
                result = await self._stop_container()
            else:
                result = await self._stop_process()
            self._process_cache = (0.0, [])
            self.last_action = {"action": "stop", "engine": kind, "at": _now_iso(),
                                "ok": result.get("ok")}
            if not result.get("ok") and not result.get("already"):
                self._set("failed", result.get("error") or "stop failed")
                return result
            await self._await_stopped()
            self.startup = None
            return {"ok": True, "state": self.state, **result}

    async def _stop_container(self) -> dict:
        name = self.cfg.root.engine.container_names.vllm
        grace = int(self.cfg.root.engine.stop_grace_s or 25)
        code, out, err = await asyncio.to_thread(self.rt.docker.run,
                                                 ["stop", "-t", str(grace), name], 120)
        missing = "no such container" in (err or "").lower() or "no such container" in (out or "").lower()
        if code != 0 and not missing:
            return {"ok": False, "error": f"docker stop failed: {(err or out).strip()[:300]}"}
        code, out, err = await asyncio.to_thread(self.rt.docker.run, ["rm", name], 60)
        missing = "no such container" in (err or "").lower()
        if code != 0 and not missing:
            return {"ok": False, "error": f"docker rm failed: {(err or out).strip()[:300]}"}
        return {"ok": True, "already": missing, "output": (out or err).strip()}

    async def _stop_process(self) -> dict:
        name = self.cfg.root.engine.container_names.ninfer
        pid = self._ninfer_pid()
        pid_file = self.run_dir / f"{name}.pid"
        if pid is None:
            pid_file.unlink(missing_ok=True)
            return {"ok": True, "already": True}
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pid_file.unlink(missing_ok=True)
            return {"ok": True, "already": True}
        deadline = time.time() + float(self.cfg.root.engine.stop_grace_s or 25)
        while time.time() < deadline:
            if not self._process_exists(pid) and not self._port_listening(self.engine_port):
                break
            await asyncio.sleep(0.5)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            kill_deadline = time.time() + 15
            while time.time() < kill_deadline and self._port_listening(self.engine_port):
                await asyncio.sleep(0.5)
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass
        pid_file.unlink(missing_ok=True)
        self._child = None
        return {"ok": True, "pid": pid}

    @staticmethod
    def _process_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    async def _await_stopped(self, timeout: float = 90.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not await self.health():
                self._set("stopped", "engine stopped", self.kind)
                await self._wait_gpu_free()
                return
            await asyncio.sleep(0.5)
        self._set("failed", "engine still answers /health after stop")

    async def _wait_gpu_free(self, timeout: float = 90.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.rt.gpu.sample()
            used = self.rt.gpu.used_mib()
            total = sum(g.get("mem_total_mib") or 0 for g in self.rt.gpu.latest)
            self._gpu_free = max(0, int(total - used))
            if used < 2048:
                return
            await asyncio.sleep(1.0)

    async def restart(self, opts: dict) -> dict:
        if self._lock.locked():
            return {"ok": False, "error": "another lifecycle action is in progress"}
        await self.stop()
        for _ in range(120):
            if self.state in ("stopped", "failed"):
                break
            await asyncio.sleep(0.5)
        return await self.start(opts)

    async def switch(self, opts: dict) -> dict:
        if self._lock.locked():
            return {"ok": False, "error": "another lifecycle action is in progress"}
        target = (opts.get("engine") or "").lower()
        if not target:
            return {"ok": False, "error": "engine is required"}
        if self.state == "healthy" and self.kind == target and not opts.get("force"):
            return {"ok": False, "error": f"{target} is already running (pass force=true to recreate)"}
        await self.stop()
        for _ in range(240):
            if self.state in ("stopped", "failed"):
                break
            await asyncio.sleep(0.5)
        if self.state == "failed" and self.detail and "still answers" in self.detail:
            return {"ok": False, "error": self.detail}
        return await self.start(opts)

    # -- probe --------------------------------------------------------------
    async def probe(self, extended: bool = False) -> dict:
        model = (self.rt.engine.snapshot.get("model") or {}).get("served") or "qwen3.8-27b"
        base = f"http://127.0.0.1:{self.gateway_port}"
        headers = {"x-monitor-probe": "1", "content-type": "application/json"}
        tests = [
            ("health", "GET", "/health", None),
            ("models", "GET", "/v1/models", None),
            (
                "chat",
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
                 "max_tokens": 24, "stream": False, "reasoning_effort": "none"},
            ),
        ]
        if extended:
            tests += [
                (
                    "responses",
                    "POST",
                    "/v1/responses",
                    {"model": model, "input": "ping", "max_output_tokens": 16},
                ),
                (
                    "messages",
                    "POST",
                    "/v1/messages",
                    {"model": model, "max_tokens": 16,
                     "messages": [{"role": "user", "content": "ping"}]},
                ),
            ]

        async def one(name: str, method: str, path: str, payload: dict | None) -> dict:
            started = time.time()
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    if method == "GET":
                        resp = await client.get(base + path, headers=headers)
                    else:
                        resp = await client.post(base + path, headers=headers, json=payload)
                ms = round((time.time() - started) * 1000.0, 1)
                detail = ""
                if resp.status_code >= 400:
                    detail = resp.text[:300]
                elif name == "chat":
                    data = resp.json()
                    detail = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")[:80]
                return {"name": name, "path": path, "ok": resp.status_code < 400,
                        "status": resp.status_code, "ms": ms, "detail": detail}
            except Exception as exc:
                return {"name": name, "path": path, "ok": False, "status": None,
                        "ms": round((time.time() - started) * 1000.0, 1),
                        "detail": f"{type(exc).__name__}: {exc}"}

        results = await asyncio.gather(*(one(*t) for t in tests))
        alive = await self.health()
        payload = {
            "ok": all(item["ok"] for item in results),
            "extended": bool(extended),
            "model": model,
            "tests": list(results),
            "engine_alive_after": alive,
            "ts": time.time(),
        }
        if not alive:
            payload["warning"] = (
                "engine is not healthy after the probe — an extended-protocol request can "
                "trip engine-side bugs (NInfer on this build does). Restart the engine."
            )
        return payload

    # -- logs ---------------------------------------------------------------
    async def logs(self, tail: int = 200) -> dict:
        kind = self.detect_engine() or self.kind
        tail = max(10, min(int(tail), 2000))
        if kind == "vllm":
            name = self.cfg.root.engine.container_names.vllm
            code, out = await self._run(
                ["docker", "logs", "--tail", str(tail), name], os.environ.copy(), timeout=15
            )
            return {"engine": kind, "source": f"docker logs {name}", "lines": out.splitlines()}
        if kind == "ninfer":
            name = self.cfg.root.engine.container_names.ninfer
            path = self.run_dir / f"{name}.log"
            if not path.exists():
                return {"engine": kind, "source": str(path), "lines": [], "error": "no log file"}
            lines = path.read_text(errors="replace").splitlines()[-tail:]
            return {"engine": kind, "source": str(path), "lines": lines}
        return {"engine": None, "source": None, "lines": [], "error": "engine is not running"}

    # -- subprocess ---------------------------------------------------------
    async def _run(self, cmd: list[str], env: dict, timeout: float, log: bool = False) -> tuple[int, str]:
        def _exec() -> tuple[int, str]:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
                return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
            except subprocess.TimeoutExpired as exc:
                out = ""
                if exc.stdout:
                    out += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(errors="replace")
                if exc.stderr:
                    out += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(errors="replace")
                return 124, out + f"\n[timeout after {timeout}s]"
            except FileNotFoundError as exc:
                return 127, str(exc)

        code, out = await asyncio.to_thread(_exec)
        if log:
            log_path = self.cfg.p("paths.state_dir") / "lifecycle.log"
            try:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"\n[{_now_iso()}] $ {' '.join(cmd)}\n"
                        f"[exit {code}]\n{out}\n"
                    )
            except OSError:
                pass
        return code, out
