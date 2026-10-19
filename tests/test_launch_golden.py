"""Golden launch-argv contract test.

run_llm.sh is retired, so `tools/parity.py` (which diffed the yaml specs against
`run_llm.sh --dry-run`) can no longer run. This test freezes the argv it used to
prove — the full engine × profile × ctx matrix, byte for byte — so a yaml edit
that would have changed the launched command still fails loudly.

The golden file was generated while parity was still green (32/32), so it *is*
the run_llm.sh behaviour, captured once.

    # regenerate after an intentional launch change
    MONITOR_UPDATE_GOLDEN=1 ../.venv/bin/python -m pytest -q tests/test_launch_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from monitor import launcher

PROJECT = Path("/home/eedy/workspace/llm")
PORT = 8001
GOLDEN = Path(__file__).parent / "golden" / "launch_argv.json"

# Same matrix tools/parity.py used to drive against run_llm.sh.
CASES: list[dict] = [
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
    {"engine": "vllm", "profile": "kv3", "ctx": "128k",
     "env": {"CPU_BYTES_TO_USE": "4294967296"}},
    # the orthogonal offload layer (config.yaml → kvcache.enabled)
    {"engine": "vllm", "profile": "mtp", "ctx": "128k", "env": {"KVCACHE_OFFLOAD": "1"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"MAMBA_SSM_CACHE_DTYPE": "bfloat16"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"GPU_MEM_UTIL": "0.85"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"MAX_NUM_SEQS": "64"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k",
     "env": {"MAX_NUM_BATCHED_TOKENS": "8192"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"API_KEY": "sk-test"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"CORS": "0"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"USE_PRIVILEGED": "1"}},
    {"engine": "vllm", "profile": "stage1", "ctx": "128k", "env": {"VLLM_IMAGE": "vllm/vllm-openai:v0.31.0"}},
    {"engine": "vllm", "profile": "kv3", "ctx": "128k", "env": {"KVOFFLOAD_DIR": "/mnt/kvcache/kvoffload"}},
    {"engine": "vllm", "profile": "dflash2", "ctx": "128k"},          # refused by both sides
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k"},
    {"engine": "ninfer", "profile": "stage1", "ctx": "256k"},
    {"engine": "ninfer", "profile": "mtp", "ctx": "128k"},
    {"engine": "ninfer", "profile": "kv2", "ctx": "128k"},            # refused by both sides
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k",
     "env": {"NINFER_HOST_CONTEXT_MIB": "8192"}},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k", "args": ["--foo", "--bar=1"]},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k", "env": {"NINFER_VISION": "0"}},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k", "env": {"NINFER_LM_HEAD_DRAFT": "1"}},
    {"engine": "ninfer", "profile": "stage1", "ctx": "256k", "env": {"NINFER_KV_DTYPE": "auto"}},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k", "env": {"NINFER_DEVICE_STATE_SLOTS": "8"}},
    {"engine": "ninfer", "profile": "stage1", "ctx": "128k", "env": {"NINFER_PRESERVE_THINKING": "1"}},
]


def case_name(case: dict) -> str:
    env_over = case.get("env") or {}
    name = f"{case['engine']}/{case['profile']}/{case['ctx']}"
    if env_over:
        name += " " + ",".join(f"{k}={v}" for k, v in env_over.items())
    if case.get("args"):
        name += " args=" + " ".join(case["args"])
    return name


def plan_for(case: dict):
    opts = {"profile": case["profile"], "ctx": case["ctx"], "host": "127.0.0.1", "port": PORT}
    if case.get("args"):
        opts["extra_args"] = case["args"]
    return launcher.resolve(case["engine"], opts, env=dict(case.get("env") or {}),
                            project_dir=PROJECT, run_dir=PROJECT / "run",
                            check_docker=False, check_files=False)


def _golden() -> dict[str, dict]:
    return {row["name"]: row for row in json.loads(GOLDEN.read_text())}


def test_golden_covers_the_matrix():
    golden = _golden()
    assert set(golden) == {case_name(c) for c in CASES}, (
        "the case matrix and tests/golden/launch_argv.json disagree")


@pytest.mark.parametrize("case", CASES, ids=case_name)
def test_launch_argv_matches_golden(case):
    plan = plan_for(case)
    want = _golden()[case_name(case)]
    assert plan.ok == want["ok"], f"ok changed: {plan.errors}"
    if not want["ok"]:
        assert plan.errors, "expected a refusal"
        return
    assert plan.argv == want["argv"], (
        "launch argv drifted from the frozen run_llm.sh contract;\n"
        f"  got  {plan.argv}\n  want {want['argv']}")


@pytest.mark.skipif(not os.environ.get("MONITOR_UPDATE_GOLDEN"), reason="opt-in")
def test_update_golden():  # pragma: no cover - maintenance helper
    rows = []
    for case in CASES:
        plan = plan_for(case)
        rows.append({"name": case_name(case), "ok": plan.ok,
                     "errors": plan.errors, "argv": plan.argv})
    GOLDEN.write_text(json.dumps(rows, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
