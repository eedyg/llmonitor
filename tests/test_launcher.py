import json

import pytest

from monitor import launcher


def resolve(engine="vllm", **opts):
    opts.setdefault("ctx", "128k")
    opts.setdefault("host", "127.0.0.1")
    opts.setdefault("port", 8001)
    return launcher.resolve(
        engine, opts, env={}, check_docker=False, check_files=False,
        project_dir="/home/eedy/workspace/llm",
    )


def test_vllm_stage1_args():
    plan = resolve(profile="stage1")
    assert plan.ok
    assert plan.kind == "container"
    assert plan.argv[:3] == ["docker", "run", "-d"]
    assert "--enable-prefix-caching" in plan.argv
    assert plan.engine_argv[:2] == ["--model", "/models/Qwen3.8-27B-FP8"]
    # labels make the container recognisable for the docker scope
    assert plan.labels["llm-monitor.managed"] == "1"


def test_ctx_normalisation_and_max():
    assert resolve(profile="stage1", ctx="64k").ctx == 65536
    assert resolve(profile="stage1", ctx="max").ctx == 262144
    assert resolve(profile="stage1", ctx=32768).ctx == 32768
    with pytest.raises(launcher.SpecError):
        launcher.resolve("vllm", {"profile": "stage1", "ctx": "999k"}, env={},
                         check_docker=False, check_files=False, strict=True)


def test_ctx_beyond_window_is_refused():
    plan = resolve(profile="stage1", ctx="512k")
    assert not plan.ok
    assert any("exceeds this model's max window" in e for e in plan.errors)


def test_256k_kv_warnings():
    auto = resolve(profile="stage1", ctx="256k", kv_cache_dtype="auto")
    assert any("does NOT fit" in w for w in auto.warnings)
    fp8 = resolve(profile="stage1", ctx="256k")
    assert any("KV-tight" in w for w in fp8.warnings)


def test_unsupported_profiles():
    assert not resolve(profile="dflash2").ok
    assert not resolve("ninfer", profile="kv2").ok
    assert not resolve("ninfer", profile="kv3").ok
    assert any("unknown profile" in e for e in resolve(profile="nope").errors)


def test_vllm_mtp_speculative_config():
    plan = resolve(profile="mtp")
    cfg_idx = plan.engine_argv.index("--speculative-config")
    assert plan.engine_argv[cfg_idx + 1] == '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
    assert "--per-request-spec-decode-metrics" in plan.engine_argv


def test_spec_tokens_override():
    plan = resolve(profile="mtp", spec_tokens="5")
    cfg_idx = plan.engine_argv.index("--speculative-config")
    assert '"num_speculative_tokens":5' in plan.engine_argv[cfg_idx + 1]


def test_ninfer_lm_head_draft_only_for_spec_profiles():
    stage1 = resolve("ninfer", profile="stage1")
    assert "--lm-head-draft" not in stage1.argv
    mtp = resolve("ninfer", profile="mtp")
    assert mtp.argv[-1] == "--lm-head-draft"
    assert "--spec" in mtp.argv and "mtp" in mtp.argv


def test_ninfer_kv_capacity_follows_ctx():
    plan = resolve("ninfer", profile="stage1", ctx="128k")
    idx = plan.argv.index("--kv-capacity")
    assert plan.argv[idx + 1] == "131072"
    plan = resolve("ninfer", profile="stage1", kv_capacity="64k")
    idx = plan.argv.index("--kv-capacity")
    assert plan.argv[idx + 1] == "65536"
    assert any("smaller than the context" in w for w in plan.warnings)
    plan = resolve("ninfer", profile="stage1", kv_capacity="auto")
    idx = plan.argv.index("--kv-capacity")
    assert plan.argv[idx + 1] == "auto"


def test_ninfer_kv_dtype_follows_kv_cache_dtype():
    plan = resolve("ninfer", profile="stage1", kv_cache_dtype="bf16")
    idx = plan.argv.index("--kv-dtype")
    assert plan.argv[idx + 1] == "bf16"
    plan = resolve("ninfer", profile="stage1", kv_cache_dtype="auto")
    idx = plan.argv.index("--kv-dtype")
    assert plan.argv[idx + 1] == "fp8"


def test_cors_and_api_key_conditionals():
    vllm = resolve(profile="stage1", cors="0", api_key="sk-1")
    assert "--allowed-origins" in vllm.argv
    idx = vllm.argv.index("--api-key")
    assert vllm.argv[idx + 1] == "sk-1"
    ninfer = resolve("ninfer", profile="stage1", cors="0")
    assert "--cors" not in ninfer.argv
    ninfer = resolve("ninfer", profile="stage1", cors="1")
    assert "--cors" in ninfer.argv


def test_privileged_replaces_device():
    plan = resolve(profile="stage1", privileged="1")
    assert "--privileged" in plan.argv
    assert "nvidia.com/gpu=all" not in plan.argv


def test_kv3_multiline_json_is_one_argument():
    plan = resolve(profile="kv3")
    idx = plan.engine_argv.index("--kv-transfer-config")
    blob = plan.engine_argv[idx + 1]
    assert blob.startswith('{"kv_connector":"OffloadingConnector"')
    # Multiline JSON must survive as exactly one argument, and must stay valid.
    parsed = json.loads(blob)
    assert parsed["kv_connector"] == "OffloadingConnector"
    extra = parsed["kv_connector_extra_config"]
    assert extra["spec_name"] == "TieringOffloadingSpec"
    tier = extra["secondary_tiers"][0]
    assert tier["type"] == "fs"
    assert tier["root_dir"] == "/kvoffload/Qwen3.8-27B-FP8"
    # vLLM 0.31 contract: backpressure_cls is a *detector*, policy_cls the policy.
    assert tier["backpressure"]["backpressure_cls"] == "EMABackpressureDetector"
    assert tier["backpressure"]["policy_cls"] == "DropStorePolicy"
    assert "/home/eedy/kvoffload:/kvoffload" in plan.mounts


def test_offload_layer_is_orthogonal_to_profile():
    """config.yaml → kvcache.enabled drives a layer that any profile can carry."""
    plan = resolve(profile="mtp", offload_enabled="1", cpu_bytes_to_use="12G")
    idx = plan.engine_argv.index("--kv-transfer-config")
    parsed = json.loads(plan.engine_argv[idx + 1])
    assert parsed["kv_connector_extra_config"]["cpu_bytes_to_use"] == 12884901888
    # mtp keeps its speculative args and gains the offload layer.
    assert "--speculative-config" in plan.engine_argv
    assert "/home/eedy/kvoffload:/kvoffload" in plan.mounts
    # The layer must come after the profile args, not before.
    assert idx > plan.engine_argv.index("--speculative-config")


def test_offload_disabled_by_default():
    plan = resolve(profile="mtp")
    assert "--kv-transfer-config" not in plan.engine_argv
    assert not any("kvoffload" in mount for mount in plan.mounts)


def test_offload_layer_skipped_for_legacy_profiles():
    """kv2/kv3 carry their own offload args; the layer must not double them up."""
    plan = resolve(profile="kv3", offload_enabled="1")
    assert plan.engine_argv.count("--kv-transfer-config") == 1
    # exactly one mount, not one from the profile plus one from the layer
    assert plan.mounts.count("/home/eedy/kvoffload:/kvoffload") == 1
    assert plan.argv.count("/home/eedy/kvoffload:/kvoffload") == 1


def test_size_vars_are_normalized():
    plan = resolve(profile="mtp", offload_enabled="1", cpu_bytes_to_use="8192m")
    idx = plan.engine_argv.index("--kv-transfer-config")
    parsed = json.loads(plan.engine_argv[idx + 1])
    assert parsed["kv_connector_extra_config"]["cpu_bytes_to_use"] == 8589934592


def test_default_cpu_budget_is_12g():
    plan = resolve(profile="kv3")
    idx = plan.engine_argv.index("--kv-transfer-config")
    parsed = json.loads(plan.engine_argv[idx + 1])
    assert parsed["kv_connector_extra_config"]["cpu_bytes_to_use"] == 12884901888


def test_env_overrides_still_work():
    plan = launcher.resolve(
        "vllm", {"profile": "stage1", "ctx": "128k"},
        env={"MAX_NUM_SEQS": "64", "GPU_MEM_UTIL": "0.5", "VLLM_IMAGE": "vllm/vllm-openai:v0.31.0"},
        check_docker=False, check_files=False, project_dir="/home/eedy/workspace/llm",
    )
    assert "64" in plan.argv and "0.5" in plan.argv
    assert plan.image == "vllm/vllm-openai:v0.31.0"


def test_image_override_changes_argv():
    plan = resolve(profile="stage1", image="vllm/vllm-openai:v0.31.0")
    assert plan.image == "vllm/vllm-openai:v0.31.0"
    assert "vllm/vllm-openai:v0.31.0" in plan.argv
    assert "vllm/vllm-openai:cu130-sm120" not in plan.argv


def test_engine_schema_options():
    vllm = launcher.engine_schema("vllm")
    ninfer = launcher.engine_schema("ninfer")
    vkeys = [o["key"] for o in vllm["options"]]
    nkeys = [o["key"] for o in ninfer["options"]]
    assert "kv_cache_dtype" in vkeys and "kv_dtype" in nkeys
    kv_v = next(o for o in vllm["options"] if o["key"] == "kv_cache_dtype")
    assert kv_v["default"] == "fp8" and "fp8" in kv_v["choices"] and "auto" in kv_v["choices"]
    kv_n = next(o for o in ninfer["options"] if o["key"] == "kv_dtype")
    assert kv_n["default"] == "fp8" and "bf16" in kv_n["choices"]
    assert set(vllm["profiles"]) == {"stage1", "mtp", "dflash", "kv2", "kv3"}
    assert set(ninfer["profiles"]) == {"stage1", "mtp", "dflash", "dflash2"}
    # spec_tokens is per-profile, so it must not expose a static default
    assert next(o for o in vllm["options"] if o["key"] == "spec_tokens")["default"] is None


def test_kv_type_choices_reach_argv():
    plan = resolve(profile="stage1", kv_cache_dtype="fp8_e5m2")
    idx = plan.engine_argv.index("--kv-cache-dtype")
    assert plan.engine_argv[idx + 1] == "fp8_e5m2"

    auto = resolve(profile="stage1", kv_cache_dtype="auto")
    assert "--kv-cache-dtype" not in auto.engine_argv

    ninfer = resolve("ninfer", profile="stage1", kv_dtype="bf16")
    idx = ninfer.argv.index("--kv-dtype")
    assert ninfer.argv[idx + 1] == "bf16"


def test_schema_keys_are_known_variables():
    for engine in ("vllm", "ninfer"):
        spec = launcher.load_spec(engine)
        known = set((spec.get("defaults") or {}).keys()) | {"api_key"}
        for opt in spec.get("options") or []:
            assert opt["key"] in known, f"{engine}: unknown option key {opt['key']}"
