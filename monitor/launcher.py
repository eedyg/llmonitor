"""Declarative engine launcher: engines/*.yaml + options -> LaunchPlan.

This is the replacement for run_llm.sh's argument building. It is intentionally
table-driven: profiles, mounts, env, conditionals and preflight rules live in
YAML so adding a profile never means touching Python.

P2a status: nothing calls this yet except tools/parity.py, which proves the
resolved argv is byte-identical to `run_llm.sh --dry-run` for the whole matrix.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Iterable, Mapping

import yaml

ENGINE_DIR = Path(__file__).resolve().parent / "engines"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
WINDOW_RE = re.compile(r'"max_position_embeddings"\s*:\s*(\d+)')

# run_llm.sh compatibility: environment variables still win over spec defaults.
ENV_OPTS: dict[str, str] = {
    "API_KEY": "api_key",
    "CORS": "cors",
    "USE_PRIVILEGED": "privileged",
    "HF_HUB_OFFLINE": "hf_hub_offline",
    "MODELS_DIR": "models_dir",
    "MODEL_DIR_NAME": "model_dir_name",
    "SERVED_MODEL_NAME": "served_model_name",
    "CONTAINER_NAME": "container_name",
    "VLLM_IMAGE": "image",
    "VLLM_CACHE_DIR": "cache_dir",
    "VLLM_LOGGING_LEVEL": "log_level",
    "VLLM_REASONING_PARSER": "reasoning_parser",
    "VLLM_TOOL_CALL_PARSER": "tool_call_parser",
    "MAMBA_SSM_CACHE_DTYPE": "mamba_ssm_cache_dtype",
    "GPU_MEM_UTIL": "gpu_mem_util",
    "MAX_NUM_SEQS": "max_num_seqs",
    "MAX_NUM_BATCHED_TOKENS": "max_num_batched_tokens",
    "KV_CACHE_DTYPE": "kv_cache_dtype",
    "KV_OFFLOAD_GB": "kv_offload_gb",
    "CPU_BYTES_TO_USE": "cpu_bytes_to_use",
    "KVCACHE_OFFLOAD": "offload_enabled",
    "KVOFFLOAD_DIR": "kvoffload_dir",
    "RESTART_POLICY": "restart_policy",
    "SPEC_TOKENS": "spec_tokens",
    "SPEC_METRICS": "spec_metrics",
    "NINFER_BIN": "binary",
    "NINFER_ARTIFACT": "artifact",
    "NINFER_MAX_CONCURRENCY": "max_concurrency",
    "NINFER_KV_DTYPE": "kv_dtype",
    "NINFER_KV_CAPACITY": "kv_capacity",
    "NINFER_DEVICE_STATE_SLOTS": "device_state_slots",
    "NINFER_HOST_CONTEXT_MIB": "host_context_mib",
    "NINFER_PRESERVE_THINKING": "preserve_thinking",
    "NINFER_VISION": "vision",
    "NINFER_LM_HEAD_DRAFT": "lm_head_draft",
}


class SpecError(ValueError):
    pass


def _truthy(value: Any) -> bool:
    """Explicit boolean flags: only 1/true/yes/on are on (run_llm.sh semantics)."""
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _bool(value: Any) -> bool:
    """General truthiness for `when:` conditions: non-empty and not 0/false/off/none."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in ("", "0", "false", "no", "off", "none")


def parse_size(raw: Any) -> int:
    """65536 | 64k | 128K | 1m -> int."""
    text = str(raw).strip().lower().replace(" ", "").replace("_", "")
    if not text:
        raise ValueError("empty size")
    mult = 1
    if text[-1] in ("k", "m", "g"):
        mult = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[text[-1]]
        text = text[:-1]
    return int(float(text) * mult)


def normalize_ctx(raw: Any, model_max_context: int) -> int:
    text = str(raw).strip().lower().replace(" ", "")
    if text in ("max", "model", "full"):
        return int(model_max_context)
    return parse_size(text)


def load_spec(engine: str, engine_dir: Path | None = None) -> dict:
    path = (engine_dir or ENGINE_DIR) / f"{engine}.yaml"
    if not path.exists():
        raise SpecError(f"no spec for engine '{engine}' ({path})")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["_path"] = str(path)
    return data


def engine_schema(engine: str, engine_dir: Path | None = None) -> dict:
    """UI/API description of one engine: profiles, context default, option fields."""
    spec = load_spec(engine, engine_dir)
    defaults = spec.get("defaults") or {}
    options: list[dict] = []
    for raw in spec.get("options") or []:
        item = dict(raw)
        if item.get("default") is None:
            value = defaults.get(item.get("key"))
            if not isinstance(value, (dict, list)):
                item["default"] = value
        options.append(item)
    return {
        "engine": engine,
        "kind": spec.get("kind"),
        "profiles": sorted((spec.get("profiles") or {}).keys()),
        "unsupported_profiles": sorted((spec.get("unsupported_profiles") or {}).keys()),
        "default_ctx": spec.get("default_ctx"),
        "options": options,
    }


def available_engines(engine_dir: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (engine_dir or ENGINE_DIR).glob("*.yaml"))


def model_dirs(engine: str, engine_dir: Path | None = None,
               project_dir: Path | str | None = None) -> list[str]:
    """Model directories this engine can actually load right now.

    A directory counts when it holds a `config.json`, which is exactly the test
    `resolve()` uses (`models.dir_candidates` + `dir_name`). The UI renders these
    as the `model_dir_name` choices, so switching models is a dropdown instead of
    editing engines/vllm.yaml — and a half-downloaded directory never shows up.
    """
    spec = load_spec(engine, engine_dir)
    project = Path(project_dir) if project_dir else _default_project_dir()
    variables = {
        "project_dir": str(project),
        "home": str(Path.home()),
        "monitor_dir": str(Path(__file__).resolve().parent.parent),
        "engine": engine,
    }
    names: list[str] = []
    seen: set[str] = set()
    for candidate in (spec.get("models") or {}).get("dir_candidates") or []:
        try:
            root = Path(str(_render(candidate, variables, "models.dir_candidates")))
        except Exception:  # pragma: no cover - a broken spec must not break the UI
            continue
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in seen:
                continue
            if not (child / "config.json").exists():
                continue
            # A downloader's control file means the shards are still moving:
            # listing it would let the UI launch a half-written checkpoint.
            if any(child.glob("*.aria2")) or any(child.glob("*.part")):
                continue
            # A GGUF-only directory has a config.json but nothing vLLM can load.
            if not any(child.glob("*.safetensors")):
                continue
            seen.add(child.name)
            names.append(child.name)
    return names


@dataclass
class LaunchPlan:
    engine: str
    kind: str                      # container | process
    profile: str
    argv: list[str]
    engine_argv: list[str]         # args handed to the engine binary/serve command
    host: str
    port: int
    ctx: int
    served_model_name: str
    container_name: str | None = None
    image: str | None = None
    binary: str | None = None
    artifact: str | None = None
    models_dir: str | None = None
    mounts: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    pidfile: str | None = None
    logfile: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    vars: dict[str, Any] = field(default_factory=dict)
    spec_path: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def command(self, pretty: bool = True) -> str:
        if not pretty:
            return " ".join(shlex.quote(a) for a in self.argv)
        return " \\\n  ".join(shlex.quote(a) if (" " in a or "\n" in a) else a for a in self.argv)

    def public(self) -> dict:
        return {
            "engine": self.engine,
            "kind": self.kind,
            "profile": self.profile,
            "ok": self.ok,
            "command": self.command(),
            "argv": self.argv,
            "engine_argv": self.engine_argv,
            "host": self.host,
            "port": self.port,
            "ctx": self.ctx,
            "served_model_name": self.served_model_name,
            "container_name": self.container_name,
            "image": self.image,
            "binary": self.binary,
            "artifact": self.artifact,
            "models_dir": self.models_dir,
            "mounts": self.mounts,
            "env": self.env,
            "labels": self.labels,
            "pidfile": self.pidfile,
            "logfile": self.logfile,
            "warnings": self.warnings,
            "errors": self.errors,
            "spec": self.spec_path,
        }


def _render(value: Any, variables: Mapping[str, Any], where: str) -> Any:
    if isinstance(value, str):
        try:
            return Template(value).substitute(variables)
        except KeyError as exc:
            raise SpecError(f"{where}: unknown template variable ${{{exc.args[0]}}}") from exc
    if isinstance(value, list):
        return [_render(v, variables, where) for v in value]
    if isinstance(value, dict):
        return {k: _render(v, variables, where) for k, v in value.items()}
    return value


def _render_list(items: Iterable[Any] | None, variables: Mapping[str, Any], where: str) -> list[str]:
    return [str(_render(item, variables, where)) for item in (items or [])]


def _cond_ok(value: Any, cond: Any) -> bool:
    if isinstance(cond, bool):
        return _bool(value) == cond
    if isinstance(cond, dict):
        for op, arg in cond.items():
            if op == "truthy":
                if _bool(value) != _bool(arg):
                    return False
            elif op == "falsy":
                if _bool(value) != (not _bool(arg)):
                    return False
            elif op == "eq":
                if str(value if value is not None else "") != str(arg):
                    return False
            elif op == "ne":
                if str(value if value is not None else "") == str(arg):
                    return False
            elif op == "present":
                if bool(value) != bool(arg):
                    return False
            elif op == "in":
                if str(value if value is not None else "") not in [str(a) for a in arg]:
                    return False
            elif op == "not_in":
                if str(value if value is not None else "") in [str(a) for a in arg]:
                    return False
            else:
                raise SpecError(f"unknown condition operator '{op}'")
        return True
    return str(value) == str(cond)


def _when_ok(when: dict | None, variables: Mapping[str, Any]) -> bool:
    for key, cond in (when or {}).items():
        if not _cond_ok(variables.get(key), cond):
            return False
    return True


def _default_project_dir() -> Path:
    # monitor/monitor/launcher.py -> monitor/monitor -> monitor -> <project>
    return Path(__file__).resolve().parent.parent.parent


def _model_window(model_path: Path, fallback: int) -> int:
    try:
        text = (model_path / "config.json").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback
    match = WINDOW_RE.search(text)
    if match:
        try:
            value = int(match.group(1))
            return value if value > 0 else fallback
        except ValueError:
            return fallback
    return fallback


def _docker_image_present(image: str) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=20, check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def resolve(
    engine: str,
    opts: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    project_dir: Path | str | None = None,
    run_dir: Path | str | None = None,
    engine_dir: Path | None = None,
    check_docker: bool = True,
    check_files: bool = True,
    strict: bool = False,
) -> LaunchPlan:
    opts = dict(opts or {})
    env = dict(os.environ if env is None else env)
    spec = load_spec(engine, engine_dir)
    project = Path(project_dir) if project_dir else _default_project_dir()
    run = Path(run_dir) if run_dir else project / "run"

    # ── variables: defaults < env < explicit options ───────────────────────
    variables: dict[str, Any] = {}
    variables.update(spec.get("defaults") or {})
    variables.update(
        {
            "project_dir": str(project),
            "home": str(Path.home()),
            "monitor_dir": str(Path(__file__).resolve().parent.parent),
            "engine": engine,
            "kind": spec.get("kind", "process"),
            "container_name": spec.get("container_name", f"{engine}-server"),
            "image": spec.get("image", ""),
            "binary": spec.get("binary", ""),
            "artifact": spec.get("artifact", ""),
            "served_model_name": spec.get("served_model_name", ""),
            "model_dir_name": (spec.get("models") or {}).get("dir_name", ""),
            "dflash_dir_name": (spec.get("models") or {}).get("dflash_dir_name", ""),
            "api_key": "",
            "models_dir": "",
            "profile": str(opts.get("profile") or "stage1"),
        }
    )
    for env_key, target in ENV_OPTS.items():
        if env_key in env:
            variables[target] = env[env_key]
    for key, value in opts.items():
        if value is not None:
            variables[key] = value
    # spec-level strings may reference each other (${artifact} -> ${project_dir}/...)
    for _ in range(4):
        changed = False
        for key in ("binary", "artifact", "image", "container_name", "served_model_name",
                    "model_dir_name", "dflash_dir_name", "cache_dir", "kvoffload_dir"):
            value = variables.get(key)
            if isinstance(value, str) and "${" in value:
                rendered = _render(value, variables, key)
                if rendered != value:
                    variables[key] = rendered
                    changed = True
        if not changed:
            break

    # ── size-typed variables ("12G" -> 12884901888) ─────────────────────────
    # Sizes must be rendered into JSON/mount strings as plain bytes, so they are
    # normalized once here instead of in every spec that references them.
    warnings: list[str] = []
    errors: list[str] = []
    for key in spec.get("size_vars") or []:
        raw = variables.get(key)
        if raw in (None, ""):
            continue
        try:
            variables[key] = str(parse_size(raw))
        except (TypeError, ValueError):
            errors.append(
                f"invalid size for {key}: {raw!r} (use 12884901888, 12g, 1024m)"
            )

    profile = str(variables["profile"])
    profiles = spec.get("profiles") or {}
    unsupported = spec.get("unsupported_profiles") or {}
    if profile in unsupported:
        errors.append(str(unsupported[profile]))
    elif profile not in profiles:
        errors.append(
            f"unknown profile '{profile}' for {engine} "
            f"(available: {', '.join(sorted(profiles))})"
        )
    profile_spec = profiles.get(profile) or {}

    # ── model window + context ─────────────────────────────────────────────
    model_max = int(spec.get("model_max_context") or 0)
    models_cfg = spec.get("models") or {}
    if models_cfg:
        if not variables.get("models_dir"):
            for candidate in models_cfg.get("dir_candidates") or []:
                path = Path(_render(candidate, variables, "models.dir_candidates"))
                if (path / str(variables["model_dir_name"]) / "config.json").exists():
                    variables["models_dir"] = str(path)
                    break
        if not variables.get("models_dir"):
            candidates = ", ".join(str(c) for c in models_cfg.get("dir_candidates") or [])
            errors.append(f"could not find '{variables['model_dir_name']}' under: {candidates}")
            variables["models_dir"] = candidates
        model_path = Path(str(variables["models_dir"])) / str(variables["model_dir_name"])
        if model_max:
            model_max = _model_window(model_path, model_max)
        variables["model_max_context"] = model_max

    ctx_raw = variables.get("ctx")
    ctx = 0
    if ctx_raw in (None, ""):
        ctx = normalize_ctx(spec.get("default_ctx") or "128k", model_max or 262144)
    else:
        try:
            ctx = normalize_ctx(ctx_raw, model_max or 262144)
        except ValueError:
            errors.append(f"invalid context size '{ctx_raw}' (use 65536, 64k, 128k, 1m, max)")
    if ctx <= 0:
        errors.append("context size must be > 0")
    if model_max and ctx > model_max:
        errors.append(
            f"context {ctx} exceeds this model's max window ({model_max} tokens). "
            f"Use --ctx {model_max} or less."
        )
    variables["ctx"] = str(ctx)

    host = str(variables.get("host") or "0.0.0.0")
    port = int(variables.get("port") or 8000)
    variables["host"] = host
    variables["port"] = str(port)

    kv_cache_dtype = str(variables.get("kv_cache_dtype") or "fp8")

    # ── engine specific preflight ──────────────────────────────────────────
    if engine == "vllm":
        if ctx >= 262144:
            if kv_cache_dtype == "auto":
                warnings.append(
                    f"Context {ctx} with bf16 KV (KV_CACHE_DTYPE=auto) does NOT fit on this GPU: "
                    "bf16 KV needs 16.32 GiB/token-set but only ~12.11 GiB is free (measured). "
                    "Use the default fp8 KV, or --engine ninfer."
                )
            else:
                warnings.append(
                    f"Context {ctx} on vLLM is KV-tight at {kv_cache_dtype} KV: the pool holds "
                    "295,827 tokens = 1.13x a full-window request. vLLM admits a request against "
                    "max_model_len (not its actual prompt), so at 256k only one long request is "
                    "resident at a time (measured running=1 for 2x128k). Prefill is chunked but "
                    "never parallel in 0.31 (max_num_partial_prefills was removed), so concurrency "
                    "buys decode overlap only — lower ctx if you need several long requests to fit."
                )
        if profile == "dflash":
            draft = Path(str(variables["models_dir"])) / str(variables["dflash_dir_name"]) / "config.json"
            if check_files and not draft.exists():
                errors.append(
                    f"Profile 'dflash' needs the draft model at {draft.parent} (not present). "
                    "Download z-lab/Qwen3.8-27B-DFlash2 (~3.85 GB) or use --profile mtp."
                )
        if check_docker and variables.get("image") and not _docker_image_present(str(variables["image"])):
            errors.append(
                f"Image not present locally: {variables['image']} (docker pull it first)"
            )
        if variables.get("privileged") and _truthy(variables["privileged"]):
            warnings.append("USE_PRIVILEGED=1 -> using --privileged instead of CDI.")
    else:
        if check_files and not Path(str(variables["binary"])).is_file():
            errors.append(f"NInfer binary not found: {variables['binary']}")
        elif check_files and not os.access(str(variables["binary"]), os.X_OK):
            errors.append(f"NInfer binary is not executable: {variables['binary']}")
        if check_files and not Path(str(variables["artifact"])).is_file():
            errors.append(f"NInfer artifact not found: {variables['artifact']}")

        capacity = variables.get("kv_capacity")
        if capacity in (None, ""):
            capacity = str(ctx)
        elif str(capacity) != "auto":
            try:
                capacity = str(parse_size(capacity))
            except ValueError:
                errors.append(f"invalid --kv-capacity '{capacity}' (use 131072, 128k, 1m or auto)")
                capacity = str(ctx)
        variables["kv_capacity"] = str(capacity)
        if str(capacity) != "auto" and capacity and int(capacity) < ctx:
            warnings.append(
                f"KV pool ({capacity} tokens) is smaller than the context ceiling ({ctx}); "
                "long single requests will be paused or truncated. Raise --kv-capacity."
            )
        declared = variables.get("kv_dtype")
        if declared:
            variables["kv_dtype"] = str(declared)
        elif variables.get("kv_cache_dtype") and str(variables["kv_cache_dtype"]) != "auto":
            variables["kv_dtype"] = str(variables["kv_cache_dtype"])
        else:
            variables["kv_dtype"] = "fp8"

    # ── spec tokens (per profile) ──────────────────────────────────────────
    per_profile = (spec.get("defaults") or {}).get("spec_tokens") or {}
    if isinstance(variables.get("spec_tokens"), dict) or variables.get("spec_tokens") in (None, ""):
        variables["spec_tokens"] = per_profile.get(profile)
    if "${spec_tokens}" in str(profile_spec) or any(
        "spec_tokens" in str(item) for item in (profile_spec.get("args") or [])
    ):
        if variables.get("spec_tokens") in (None, ""):
            errors.append(f"profile '{profile}' needs --spec-tokens (no default)")

    # ── extra files / mount preparation ────────────────────────────────────
    if profile_spec.get("requires"):
        for required in _render_list(profile_spec.get("requires"), variables, "profile.requires"):
            if check_files and not Path(required).exists():
                errors.append(f"profile '{profile}' requires {required} (not present)")

    # ── build argv ─────────────────────────────────────────────────────────
    engine_args: list[str] = []
    engine_args += _render_list(spec.get("engine_args"), variables, "engine_args")
    for rule in spec.get("conditional_args") or []:
        if _when_ok(rule.get("when"), variables):
            engine_args += _render_list(rule.get("args"), variables, "conditional_args")
    engine_args += _render_list(profile_spec.get("args"), variables, "profile.args")

    # ── layers: orthogonal to the profile ──────────────────────────────────
    # A layer adds args/mounts whenever its `when` condition holds, so a feature
    # such as KV offload can be combined with any profile (mtp, dflash, …)
    # instead of being baked into a mutually-exclusive profile.
    layer_mounts: list[str] = []
    for layer in spec.get("layers") or []:
        if not _when_ok(layer.get("when"), variables):
            continue
        layer_args = _render_list(layer.get("args"), variables, "layers.args")
        engine_args += layer_args
        layer_mounts += _render_list(layer.get("mounts"), variables, "layers.mounts")

    flags_cfg = spec.get("profile_flags") or {}
    profile_flags: list[str] = []
    for name, rule in flags_cfg.items():
        # a flag needs the profile to opt in AND the global switch to be on
        enabled = _bool(profile_spec.get(name)) and _bool(variables.get(name))
        if enabled and _when_ok(rule.get("when"), {**variables, name: "1"}):
            profile_flags += _render_list(rule.get("args"), variables, "profile_flags")
    engine_args += profile_flags

    extra = [str(a) for a in (variables.get("extra_args") or [])]
    engine_args += extra
    if variables.get("extra"):
        engine_args += shlex.split(str(variables["extra"]))

    mounts: list[str] = []
    env_vars: dict[str, str] = {}
    labels: dict[str, str] = {}

    if spec.get("kind") == "container":
        container = spec.get("container") or {}
        argv: list[str] = ["docker"] + _render_list(container.get("flags"), variables, "container.flags")
        labels = {
            str(k): str(_render(v, variables, "container.labels"))
            for k, v in (container.get("labels") or {}).items()
        }
        for key, value in labels.items():
            argv += ["--label", f"{key}={value}"]
        if _truthy(variables.get("privileged")):
            argv.append(str(container.get("privileged_flag") or "--privileged"))
        else:
            argv += ["--device", str(container.get("device"))]
        mounts = _render_list(container.get("mounts"), variables, "container.mounts")
        for mount in mounts:
            argv += ["-v", mount]
        for key, value in (container.get("env") or {}).items():
            env_vars[str(key)] = str(_render(value, variables, "container.env"))
            argv += ["-e", f"{key}={env_vars[str(key)]}"]
        for mount in _render_list(profile_spec.get("mounts"), variables, "profile.mounts"):
            if mount not in mounts:
                mounts.append(mount)
                argv += ["-v", mount]
        for mount in layer_mounts:
            if mount not in mounts:
                mounts.append(mount)
                argv += ["-v", mount]
        image = str(variables.get("image"))
        argv.append(image)
        argv += engine_args
        plan = LaunchPlan(
            engine=engine, kind="container", profile=profile, argv=argv, engine_argv=engine_args,
            host=host, port=port, ctx=ctx, served_model_name=str(variables["served_model_name"]),
            container_name=str(variables["container_name"]), image=image,
            models_dir=str(variables.get("models_dir") or "") or None,
            mounts=mounts, env=env_vars, labels=labels,
            warnings=warnings, errors=errors, vars=variables, spec_path=spec["_path"],
        )
    else:
        binary = str(variables.get("binary"))
        argv = [binary] + engine_args
        plan = LaunchPlan(
            engine=engine, kind="process", profile=profile, argv=argv,
            engine_argv=engine_args[1:] if engine_args else [],
            host=host, port=port, ctx=ctx, served_model_name=str(variables["served_model_name"]),
            container_name=str(variables["container_name"]),
            binary=binary, artifact=str(variables.get("artifact") or "") or None,
            pidfile=str(run / f"{variables['container_name']}.pid"),
            logfile=str(run / f"{variables['container_name']}.log"),
            warnings=warnings, errors=errors, vars=variables, spec_path=spec["_path"],
        )

    if strict and plan.errors:
        raise SpecError("; ".join(plan.errors))
    return plan


def resolve_all(engine: str, **kwargs: Any) -> list[LaunchPlan]:
    spec = load_spec(engine, kwargs.get("engine_dir"))
    return [resolve(engine, {"profile": p}, **kwargs) for p in sorted(spec.get("profiles") or {})]


def main(argv: list[str] | None = None) -> int:
    """CLI: print the plan that the monitor would execute (no side effects)."""
    import argparse
    import json as _json
    import sys

    parser = argparse.ArgumentParser(prog="monitor.launcher", description="Resolve an engine LaunchPlan")
    parser.add_argument("--engine", default="vllm", choices=available_engines())
    parser.add_argument("--profile", default="stage1")
    parser.add_argument("--ctx", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--kv-capacity", default=None)
    parser.add_argument("--spec-tokens", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--quiet", "-q", action="store_true", help="only print the command")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    opts = {
        "profile": args.profile,
        "ctx": args.ctx,
        "host": args.host,
        "port": args.port,
        "kv_capacity": args.kv_capacity,
        "spec_tokens": args.spec_tokens,
        "api_key": args.api_key,
        "image": args.image,
    }
    plan = resolve(args.engine, opts, project_dir=args.project_dir)
    if args.json:
        print(_json.dumps(plan.public(), indent=2, ensure_ascii=False))
    elif args.quiet:
        if not plan.ok:
            for error in plan.errors:
                print(f"ERROR {error}", file=sys.stderr)
            return 1
        print(plan.command(pretty=False))
    else:
        for warning in plan.warnings:
            print(f"WARN  {warning}")
        for error in plan.errors:
            print(f"ERROR {error}")
        print(plan.command())
        if not plan.ok:
            print(f"\n{len(plan.errors)} error(s): this plan would not launch.")
    return 0 if plan.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
