"""Docker management, strictly scoped to the vLLM engine.

Two rules make this safe on a shared machine:

1. **Reads are global, writes are scoped.** We enumerate *all* containers and
   images so we can refuse to delete an image that any other project uses, but
   we only ever mutate resources whose name/repo-tag is in the allowlist.
2. **Verb allowlist + per-object assertions.** There is no code path that can
   emit `docker system prune`, `image prune`, `container prune`, `volume rm`,
   `network rm`, or `rmi`/`rm` on an object outside the scope. A blocked verb
   raises before `docker` is executed.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

ALLOWED_VERBS: set[tuple[str, ...]] = {
    ("version",),
    ("info",),
    ("ps",),
    ("inspect",),
    ("logs",),
    ("image", "ls"),
    ("image", "inspect"),
    ("pull",),
    ("rmi",),
    ("rm",),
    ("stop",),
    ("start",),
    ("system", "df"),
}

# Belt and braces: these must never run, even if a future caller tries.
FORBIDDEN_SUBSTRINGS = (
    "system prune",
    "image prune",
    "container prune",
    "volume prune",
    "network prune",
    "volume rm",
    "network rm",
    "builder prune",
    "image rm -f",
)


class DockerError(RuntimeError):
    pass


class ScopeViolation(DockerError):
    pass


class ResourceInUse(DockerError):
    pass


def ref_repo(ref: str) -> str:
    """'dockerproxy.net:5000/vllm/vllm-openai:tag' -> 'dockerproxy.net:5000/vllm/vllm-openai'."""
    ref = ref.split("@", 1)[0]
    slash = ref.rfind("/")
    colon = ref.rfind(":")
    return ref[:colon] if colon > slash else ref


def parse_labels(blob: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for part in (blob or "").split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            labels[key.strip()] = value.strip()
    return labels


@dataclass
class Scope:
    image_repos: tuple[str, ...] = ("vllm/vllm-openai",)
    containers: tuple[str, ...] = ("vllm-qwen38",)
    label: str = "llm-monitor.managed"

    def image_ref_allowed(self, ref: str) -> bool:
        repo = ref_repo(ref)
        return any(repo == allowed or repo.startswith(allowed + "/") for allowed in self.image_repos)

    def container_allowed(self, name: str, labels: dict[str, str] | None = None) -> bool:
        if name in self.containers:
            return True
        return (labels or {}).get(self.label) == "1"

    def as_dict(self) -> dict:
        return {
            "image_repos": list(self.image_repos),
            "containers": list(self.containers),
            "label": self.label,
        }


class Docker:
    """Thin, guard-railed wrapper around the docker CLI."""

    def __init__(self, scope: Scope, runner=None, timeout: float = 60.0) -> None:
        self.scope = scope
        self.timeout = timeout
        self._runner = runner  # tests inject a fake (argv) -> (rc, out, err)

    # -- execution ----------------------------------------------------------
    def _assert_command(self, argv: Iterable[str]) -> list[str]:
        args = [str(a) for a in argv]
        joined = " ".join(args)
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in joined:
                raise ScopeViolation(f"refusing forbidden docker command: docker {joined}")
        if not args:
            raise ScopeViolation("empty docker command")
        # find the verb: skip global options
        verb: list[str] = []
        i = 0
        while i < len(args) and args[i].startswith("-"):
            i += 1
        if i < len(args):
            verb = [args[i]]
            if args[i] in ("image", "system", "manifest", "container") and i + 1 < len(args):
                verb.append(args[i + 1])
        if tuple(verb) not in ALLOWED_VERBS:
            raise ScopeViolation(f"refusing docker verb: docker {joined}")
        return args

    def run(self, argv: Iterable[str], timeout: float | None = None) -> tuple[int, str, str]:
        args = self._assert_command(argv)
        if self._runner is not None:
            return self._runner(args)
        try:
            proc = subprocess.run(
                ["docker", *args], capture_output=True, text=True,
                timeout=timeout or self.timeout, check=False,
            )
            return proc.returncode, proc.stdout or "", proc.stderr or ""
        except FileNotFoundError as exc:
            return 127, "", str(exc)
        except subprocess.TimeoutExpired:
            return 124, "", f"docker {' '.join(args)} timed out"

    def available(self) -> tuple[bool, str | None]:
        code, _out, err = self.run(["version", "--format", "{{.Server.Version}}"], timeout=10)
        return code == 0, (None if code == 0 else (err.strip() or "docker unreachable"))

    # -- read ---------------------------------------------------------------
    def containers(self) -> list[dict]:
        code, out, err = self.run(["ps", "-a", "--no-trunc", "--format", "{{json .}}"])
        if code != 0:
            raise DockerError(err.strip() or "docker ps failed")
        rows = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            labels = parse_labels(raw.get("Labels", ""))
            name = (raw.get("Names") or "").split(",")[0]
            rows.append(
                {
                    "id": raw.get("ID", ""),
                    "name": name,
                    "image_ref": raw.get("Image", ""),
                    "state": raw.get("State", ""),
                    "status": raw.get("Status", ""),
                    "created": raw.get("CreatedAt", ""),
                    "size": raw.get("Size", ""),
                    "labels": labels,
                    "in_scope": self.scope.container_allowed(name, labels),
                }
            )
        return rows

    def image_usage(self) -> dict[str, list[str]]:
        """image id -> container names, across ALL containers (read-only)."""
        names = [row["name"] for row in self.containers() if row["name"]]
        usage: dict[str, list[str]] = {}
        for chunk in _chunks(names, 40):
            code, out, _err = self.run(
                ["inspect", "--format", "{{.Name}}\t{{.Image}}", *chunk], timeout=30
            )
            if code != 0:
                continue
            for line in out.splitlines():
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                name, image_id = parts[0].lstrip("/"), parts[1]
                usage.setdefault(image_id, []).append(name)
        return usage

    def image_id(self, ref: str) -> str | None:
        code, out, _err = self.run(["image", "inspect", "--format", "{{.Id}}", ref], timeout=20)
        return out.strip() if code == 0 and out.strip() else None

    def images(self, tracked: Iterable[str] = ()) -> list[dict]:
        ids: set[str] = set()
        for repo in self.scope.image_repos:
            code, out, _err = self.run(
                ["image", "ls", "--no-trunc", "--quiet", "--filter", f"reference={repo}"]
            )
            if code == 0:
                ids.update(out.split())
        tracked = set(tracked)
        ids.update(tracked)
        if not ids:
            return []
        usage = self.image_usage()
        inspected: list[dict] = []
        for chunk in _chunks(sorted(ids), 25):
            code, out, _err = self.run(["image", "inspect", *chunk], timeout=60)
            if code == 0:
                try:
                    inspected.extend(json.loads(out))
                except json.JSONDecodeError:
                    pass
                continue
            # one of them vanished; fall back to one-by-one
            for image_id in chunk:
                code1, out1, _e1 = self.run(["image", "inspect", image_id], timeout=30)
                if code1 == 0:
                    try:
                        inspected.extend(json.loads(out1))
                    except json.JSONDecodeError:
                        pass

        rows: list[dict] = []
        for img in inspected:
            image_id = img.get("Id", "")
            tags = [t for t in (img.get("RepoTags") or []) if t and t != "<none>:<none>"]
            in_scope_tags = [t for t in tags if self.scope.image_ref_allowed(t)]
            out_tags = [t for t in tags if not self.scope.image_ref_allowed(t)]
            in_use_by = usage.get(image_id, [])
            dangling = not tags
            is_tracked = image_id in tracked
            removable = False
            blocked = None
            if in_use_by:
                blocked = "被容器使用：" + ", ".join(in_use_by)
            elif in_scope_tags:
                removable = True
            elif dangling and is_tracked:
                removable = True
            elif out_tags:
                blocked = "只有 scope 外的 tag，拒绝操作：" + ", ".join(out_tags)
            else:
                blocked = "不在 scope 内（无 in-scope tag，也不是 monitor 拉取留下的旧镜像）"
            rows.append(
                {
                    "id": image_id,
                    "short": image_id.replace("sha256:", "")[:12],
                    "tags": tags,
                    "in_scope_tags": in_scope_tags,
                    "out_of_scope_tags": out_tags,
                    "size_mb": round((img.get("Size") or 0) / 1048576, 1),
                    "created": img.get("Created", ""),
                    "dangling": dangling,
                    "tracked": is_tracked,
                    "in_use_by": in_use_by,
                    "in_use_here": any(
                        self.scope.container_allowed(n) for n in in_use_by
                    ),
                    "removable": removable,
                    "blocked_reason": blocked,
                }
            )
        rows.sort(key=lambda r: (-(r["size_mb"] or 0), r["short"]))
        return rows

    # -- write --------------------------------------------------------------
    def remove_image(self, image_id: str, tracked: Iterable[str] = (), dry_run: bool = False) -> dict:
        rows = {row["id"]: row for row in self.images(tracked=tracked)}
        row = rows.get(image_id)
        if row is None:
            raise ScopeViolation("image is not visible in the vLLM scope")
        if not row["removable"]:
            reason = row["blocked_reason"] or "image is not removable"
            if row["in_use_by"]:
                raise ResourceInUse(reason)
            raise ScopeViolation(reason)
        if row["in_scope_tags"]:
            argv = ["rmi", *row["in_scope_tags"]]
            note = None
            if row["out_of_scope_tags"]:
                note = (
                    "该镜像还有 scope 外的 tag，只删除 scope 内 tag："
                    + ", ".join(row["out_of_scope_tags"])
                )
            if dry_run:
                return {"command": ["docker", *argv], "note": note}
            code, out, err = self.run(argv, timeout=120)
            if code != 0:
                raise DockerError((err or out).strip() or "docker rmi failed")
            return {"command": ["docker", *argv], "note": note, "output": (out or err).strip()}

        argv = ["rmi", image_id]
        if dry_run:
            return {"command": ["docker", *argv]}
        code, out, err = self.run(argv, timeout=120)
        if code != 0:
            raise DockerError((err or out).strip() or "docker rmi failed")
        return {"command": ["docker", *argv], "output": (out or err).strip()}

    def remove_container(self, name: str, dry_run: bool = False) -> dict:
        row = next((c for c in self.containers() if c["name"] == name), None)
        if row is None:
            raise DockerError(f"no container named {name}")
        if not row["in_scope"]:
            raise ScopeViolation(f"container {name} is outside the vLLM scope")
        if row["state"] == "running":
            raise DockerError("container is running; stop the engine first")
        argv = ["rm", name]
        if dry_run:
            return {"command": ["docker", *argv]}
        code, out, err = self.run(argv, timeout=60)
        if code != 0:
            raise DockerError((err or out).strip() or "docker rm failed")
        return {"command": ["docker", *argv], "output": (out or err).strip()}

    # -- pull (async, streamed) --------------------------------------------
    async def pull_stream(self, ref: str, on_proc=None):
        if not self.scope.image_ref_allowed(ref):
            raise ScopeViolation(f"image '{ref}' is outside the vLLM scope")
        self._assert_command(["pull", ref])
        proc = await asyncio.create_subprocess_exec(
            "docker", "pull", ref,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        if on_proc is not None:
            on_proc(proc)
        assert proc.stdout is not None
        buf = b""
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                positions = [i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0]
                if not positions:
                    break
                cut = min(positions)
                line, buf = buf[:cut], buf[cut + 1:]
                text = line.decode("utf-8", "replace").strip()
                if text:
                    yield text
        if buf.strip():
            yield buf.decode("utf-8", "replace").strip()
        code = await proc.wait()
        if code != 0:
            raise DockerError(f"docker pull exited {code}")

    def summary(self, tracked: Iterable[str] = ()) -> dict:
        available, error = self.available()
        if not available:
            return {"available": False, "error": error, "scope": self.scope.as_dict()}
        containers = self.containers()
        images = self.images(tracked=tracked)
        in_scope = [c for c in containers if c["in_scope"]]
        total_mb = round(sum(i["size_mb"] or 0 for i in images), 1)
        reclaim_mb = round(sum(i["size_mb"] or 0 for i in images if i["removable"]), 1)
        return {
            "available": True,
            "scope": self.scope.as_dict(),
            "containers": in_scope,
            "other_containers": len(containers) - len(in_scope),
            "images": images,
            "disk": {
                "images_mb": total_mb,
                "reclaimable_mb": reclaim_mb,
                "image_count": len(images),
            },
        }


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
