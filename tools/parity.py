#!/usr/bin/env python3
"""Parity harness: prove launcher.py == run_llm.sh, byte for byte.

How it works
------------
`run_llm.sh --dry-run` prints its command one argument per line, but multi-line
arguments (the kv3 JSON) make that text ambiguous. So instead we put a fake
`docker` (and a fake ninfer binary) first on PATH that records the exact argv
(NUL separated) and exits 0. run_llm.sh runs its real launch path, the shims
capture what it *would* have executed, and we diff that against LaunchPlan.argv.

Usage:
    python tools/parity.py              # full matrix
    python tools/parity.py -v           # show every case
    python tools/parity.py --case mtp   # filter by substring
    python tools/parity.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # monitor/
PROJECT = ROOT.parent                                   # workspace/llm
RUN_LLM = PROJECT / "run_llm.sh"
PORT = 18765

sys.path.insert(0, str(ROOT))

from monitor import launcher  # noqa: E402

DOCKER_SHIM = """#!/usr/bin/env bash
if [ "$1" = "run" ]; then
  { printf '%s\\0' "$@"; } > "${DOCKER_CAPTURE}"
fi
exit 0
"""

NINFER_SHIM = """#!/usr/bin/env bash
{ printf '%s\\0' "$@"; } > "${NINFER_CAPTURE}"
exit 0
"""


CASES: list[dict] = [
    # vLLM matrix
    {"engine": "vllm", "profile": "stage1", "ctx": "128k"},
    {"engine": "vllm", "profile": "stage1", "ctx": "256k"},
    {"engine": "vllm", "profile": "stage1", "ctx": "256k", "env": {"KV_CACHE_DTYPE": "auto"}},
    {"engine": "vllm", "profile": "mtp", "ctx": "128k"},
    {"engine": "vllm", "profile": "mtp", "ctx": "256k", "env": {"SPEC_TOKENS": "5"}},
    {"engine": "vllm", "profile": "mtp", "ctx": "128k",
     "env": {"VLLM_REASONING_PARSER": "", "VLLM_TOOL_CALL_PARSER": ""}},
    {"engine": "vllm", "profile": "dflash", "ctx": "128k"},
    {"engine": "vllm", "profile": "kv2", "ctx": "128k"},
    {"engine": "vllm", "profile": "kv3", "ctx": "128k"},
    {"engine": "vllm", "profile": "kv3", "ctx": "128k", "env": {"CPU_BYTES_TO_USE": "4294967296"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k",
     "env": {"MAX_NUM_SEQS": "128", "GPU_MEM_UTIL": "0.9"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"API_KEY": "sk-test"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"CORS": "0"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"USE_PRIVILEGED": "1"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"MAMBA_SSM_CACHE_DTYPE": "bfloat16"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k",
     "env": {"VLLM_IMAGE": "vllm/vllm-openai:v0.31.0"}},
    {"engine": "vllm", "profile": "dflash2", "ctx": "128k"},  # must be refused by both
    # NInfer matrix
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k"},
    {"engine": "ninfer", "profile": "stage1", "ctx": "256k"},
    {"engine": "ninfer", "profile": "mtp", "ctx": "128k"},
    {"engine": "ninfer", "profile": "mtp", "ctx": "256k", "env": {"SPEC_TOKENS": "5"}},
    {"engine": "ninfer", "profile": "dflash", "ctx": "128k"},
    {"engine": "ninfer", "profile": "dflash2", "ctx": "128k"},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k", "env": {"NINFER_KV_DTYPE": "bf16"}},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k", "env": {"NINFER_KV_CAPACITY": "64k"}},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k",
     "env": {"NINFER_MAX_CONCURRENCY": "4", "NINFER_DEVICE_STATE_SLOTS": "4"}},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k", "env": {"API_KEY": "sk-test", "CORS": "1"}},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k", "env": {"NINFER_VISION": "1"}},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k", "env": {"NINFER_LM_HEAD_DRAFT": "0"}},
    {"engine": "ninfer", "profile": "mtp", "ctx": "128k", "env": {"NINFER_HOST_CONTEXT_MIB": "4096"}},
    {"engine": "ninfer", "profile": "kv2", "ctx": "128k"},  # must be refused by both
]


def _read_argv(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    blob = path.read_bytes()
    if not blob:
        return None
    parts = blob.split(b"\0")
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    return [p.decode("utf-8", "replace") for p in parts]


def run_bash(case: dict, workdir: Path) -> tuple[int, list[str] | None, str, dict]:
    shim_dir = workdir / "shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    docker_shim = shim_dir / "docker"
    docker_shim.write_text(DOCKER_SHIM)
    docker_shim.chmod(0o755)
    ninfer_shim = shim_dir / "ninfer-serve"
    ninfer_shim.write_text(NINFER_SHIM)
    ninfer_shim.chmod(0o755)

    docker_capture = workdir / "docker.argv"
    ninfer_capture = workdir / "ninfer.argv"
    for stale in (docker_capture, ninfer_capture):
        stale.unlink(missing_ok=True)

    env = {
        "HOME": os.environ.get("HOME", "/root"),
        "PATH": f"{shim_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "LANG": "C.UTF-8",
        "HOST_PORT": str(PORT),
        "COMPAT": "0",
        "RUN_DIR": str(workdir / "run"),
        "DOCKER_CAPTURE": str(docker_capture),
        "NINFER_CAPTURE": str(ninfer_capture),
        "NINFER_BIN": str(ninfer_shim),
    }
    env.update(case.get("env") or {})

    cmd = [str(RUN_LLM), "--engine", case["engine"], "--profile", case["profile"],
           "--host", "127.0.0.1", "--ctx", case["ctx"]]
    cmd += case.get("args") or []
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    argv = _read_argv(docker_capture) if case["engine"] == "vllm" else _read_argv(ninfer_capture)
    # the shims capture "$@" (argv[1:]); put argv[0] back so the lists are comparable
    if argv is not None:
        argv = (["docker"] + argv) if case["engine"] == "vllm" else ([str(ninfer_shim)] + argv)
    if argv is None and proc.returncode == 0:
        # the shim may still be flushing when the script returns (nohup path)
        for _ in range(20):
            argv = _read_argv(docker_capture) if case["engine"] == "vllm" else _read_argv(ninfer_capture)
            if argv is not None:
                argv = (["docker"] + argv) if case["engine"] == "vllm" else ([str(ninfer_shim)] + argv)
                break
            import time

            time.sleep(0.05)
    return proc.returncode, argv, (proc.stderr or "").strip(), env


def strip_labels(argv: list[str]) -> tuple[list[str], list[str]]:
    out: list[str] = []
    labels: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--label" and i + 1 < len(argv):
            labels.append(argv[i + 1])
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out, labels


def diff_lists(a: list[str], b: list[str], label_a: str, label_b: str) -> list[str]:
    problems: list[str] = []
    for i in range(max(len(a), len(b))):
        left = a[i] if i < len(a) else "<missing>"
        right = b[i] if i < len(b) else "<extra>"
        if left != right:
            problems.append(f"  arg[{i}] {label_a}={left!r}  !=  {label_b}={right!r}")
    return problems


def compare(case: dict, bash_argv: list[str], plan) -> list[str]:
    problems: list[str] = []
    if plan.errors:
        problems.append(f"  launcher errors: {plan.errors}")

    if plan.kind == "container":
        bash_head, _ = strip_labels(bash_argv)
        plan_head, plan_labels = strip_labels(plan.argv)
        image = plan.image
        try:
            b_idx = bash_head.index(image)
        except ValueError:
            return [f"  image {image!r} missing from run_llm.sh argv"]
        try:
            p_idx = plan_head.index(image)
        except ValueError:
            return [f"  image {image!r} missing from launcher argv"]
        problems += diff_lists(bash_head[: b_idx + 1], plan_head[: p_idx + 1], "bash", "plan")
        problems += diff_lists(bash_head[b_idx + 1:], plan_head[p_idx + 1:], "bash", "plan")
        if not plan_labels:
            problems.append("  launcher did not add the managed label")
    else:
        # argv[0] is the binary (same shim on both sides), everything else must match
        problems += diff_lists(bash_argv, plan.argv, "bash", "plan")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="substring filter on engine/profile/ctx")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", help="write a machine-readable report here")
    args = parser.parse_args()

    if not RUN_LLM.exists():
        print(f"run_llm.sh not found at {RUN_LLM}", file=sys.stderr)
        return 2

    cases = CASES
    if args.case:
        needle = args.case.lower()
        cases = [c for c in cases
                 if needle in f"{c['engine']} {c['profile']} {c['ctx']} {' '.join((c.get('env') or {}).keys())}".lower()]

    report: list[dict] = []
    failures = 0
    with tempfile.TemporaryDirectory(prefix="parity-") as tmp:
        for index, case in enumerate(cases):
            workdir = Path(tmp) / f"case{index}"
            workdir.mkdir(parents=True, exist_ok=True)
            env_over = case.get("env") or {}
            opts = {"profile": case["profile"], "ctx": case["ctx"], "host": "127.0.0.1",
                    "port": PORT}
            if case.get("args"):
                opts["extra_args"] = case["args"]

            rc, bash_argv, stderr, env = run_bash(case, workdir)
            # lane the launcher through the same env run_llm.sh saw
            plan = launcher.resolve(
                case["engine"], opts, env={**os.environ, **env},
                project_dir=PROJECT, run_dir=workdir / "run",
                check_docker=False,
            )

            name = f"{case['engine']}/{case['profile']}/{case['ctx']}" + (
                f" {','.join(f'{k}={v}' for k, v in env_over.items())}" if env_over else ""
            )

            if rc != 0:
                # both sides must refuse
                ok = bool(plan.errors)
                status = "REFUSE-OK" if ok else "REFUSE-MISMATCH"
                problems = [] if ok else [f"  run_llm.sh refused (rc={rc}) but launcher produced a plan"]
                detail = stderr.splitlines()[-1] if stderr else ""
                if args.verbose or not ok:
                    print(f"[{status}] {name}  ({detail[:100]})")
                report.append({"case": name, "status": status, "rc": rc,
                               "plan_errors": plan.errors, "problems": problems})
                if not ok:
                    failures += 1
                continue

            if bash_argv is None:
                print(f"[NO-CAPTURE] {name}: run_llm.sh succeeded but no argv was captured")
                report.append({"case": name, "status": "NO-CAPTURE", "rc": rc})
                failures += 1
                continue

            problems = compare(case, bash_argv, plan)
            status = "PASS" if not problems else "FAIL"
            if problems:
                failures += 1
            if args.verbose or problems:
                print(f"[{status}] {name}")
                for line in problems:
                    print(line)
            report.append({"case": name, "status": status, "rc": rc,
                           "bash_argv": bash_argv, "plan_argv": plan.argv,
                           "problems": problems})

    total = len(report)
    passed = sum(1 for r in report if r["status"] in ("PASS", "REFUSE-OK"))
    print(f"\n{passed}/{total} cases match  ({failures} failing)")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"report: {args.json}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
