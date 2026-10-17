import asyncio
import json

import pytest

from monitor.docker_scope import Docker, DockerError, ResourceInUse, Scope, ScopeViolation, ref_repo

VLLM_ID = "sha256:" + "a" * 64
OLD_ID = "sha256:" + "b" * 64
OTHER_ID = "sha256:" + "c" * 64

PS_ROWS = [
    {"ID": "1" * 64, "Names": "vllm-qwen38", "Image": "vllm/vllm-openai:cu130-sm120",
     "State": "running", "Status": "Up 10 minutes", "CreatedAt": "2026-10-17 22:25:15 +0800 CST",
     "Size": "544MB (virtual 23.4GB)", "Labels": "maintainer=x"},
    {"ID": "2" * 64, "Names": "mx-test_llm", "Image": "ymatrix/matrixdb-dev:7.0",
     "State": "running", "Status": "Up 1 hour", "CreatedAt": "2026-10-17 21:00:00 +0800 CST",
     "Size": "1MB (virtual 2GB)", "Labels": "maintainer=y"},
    {"ID": "3" * 64, "Names": "vllm-old", "Image": "vllm/vllm-openai:cu130-sm120",
     "State": "exited", "Status": "Exited (0) 1 hour ago", "CreatedAt": "2026-10-16 10:00:00 +0800 CST",
     "Size": "0B (virtual 23.4GB)", "Labels": "llm-monitor.managed=1"},
    {"ID": "4" * 64, "Names": "old-vllm", "Image": "vllm/vllm-openai:cu130-sm120",
     "State": "exited", "Status": "Exited (0) 2 hours ago", "CreatedAt": "2026-10-15 10:00:00 +0800 CST",
     "Size": "0B (virtual 23.4GB)", "Labels": ""},
]

IMAGES = [
    {
        "Id": VLLM_ID,
        "RepoTags": ["vllm/vllm-openai:cu130-sm120", "vllm/vllm-openai:v0.31.0",
                     "dockerproxy.net/vllm/vllm-openai:v0.31.0"],
        "Size": 22842960920,
        "Created": "2026-10-03T08:06:44Z",
    },
    {
        "Id": OLD_ID,
        "RepoTags": None,          # dangling, produced by our own pull
        "Size": 1000000000,
        "Created": "2026-10-01T08:00:00Z",
    },
]


def fake_runner(argv):
    if argv[:2] == ["ps", "-a"]:
        return 0, "\n".join(json.dumps(r) for r in PS_ROWS), ""
    if argv[:2] == ["inspect", "--format"]:  # image usage map
        return 0, (
            "/vllm-qwen38\t" + VLLM_ID + "\n"
            "/mx-test_llm\t" + OTHER_ID + "\n"
            "/vllm-old\t" + VLLM_ID + "\n"
        ), ""
    if argv[:2] == ["image", "ls"]:
        return 0, VLLM_ID + "\n" + VLLM_ID + "\n" + OLD_ID + "\n", ""
    if argv[:2] == ["image", "inspect"]:
        ids = argv[2:]
        data = [img for img in IMAGES if img["Id"] in ids]
        return 0, json.dumps(data), ""
    if argv[:1] == ["rmi"]:
        return 0, "Untagged: x\n", ""
    if argv[:1] == ["rm"]:
        return 0, "old-vllm\n", ""
    if argv[:1] == ["version"]:
        return 0, "27.0.0\n", ""
    return 0, "", ""


def make_docker(tracked=(OLD_ID,)):
    scope = Scope(image_repos=("vllm/vllm-openai",), containers=("vllm-qwen38",))
    return Docker(scope, runner=fake_runner), list(tracked)


def test_ref_repo():
    assert ref_repo("vllm/vllm-openai:cu130-sm120") == "vllm/vllm-openai"
    assert ref_repo("localhost:5000/vllm/vllm-openai:tag") == "localhost:5000/vllm/vllm-openai"
    assert ref_repo("repo@sha256:abc") == "repo"


def test_scope_rules():
    scope = Scope(image_repos=("vllm/vllm-openai",), containers=("vllm-qwen38",))
    assert scope.image_ref_allowed("vllm/vllm-openai:cu130-sm120")
    assert not scope.image_ref_allowed("dockerproxy.net/vllm/vllm-openai:v0.31.0")
    assert not scope.image_ref_allowed("timescale/timescaledb:2.29.2-pg16")
    assert scope.container_allowed("vllm-qwen38")
    assert not scope.container_allowed("mx-test_llm")
    assert scope.container_allowed("whatever", {"llm-monitor.managed": "1"})


def test_forbidden_verbs_never_execute():
    docker, _ = make_docker()
    with pytest.raises(ScopeViolation):
        docker.run(["system", "prune", "-af"])
    with pytest.raises(ScopeViolation):
        docker.run(["image", "prune", "-a"])
    with pytest.raises(ScopeViolation):
        docker.run(["volume", "rm", "x"])
    with pytest.raises(ScopeViolation):
        docker.run(["network", "rm", "x"])


def test_images_flags_usage_and_scope():
    docker, tracked = make_docker()
    rows = {row["id"]: row for row in docker.images(tracked=tracked)}

    vllm = rows[VLLM_ID]
    assert vllm["removable"] is False
    assert "vllm-qwen38" in vllm["in_use_by"]
    assert vllm["in_scope_tags"] == ["vllm/vllm-openai:cu130-sm120", "vllm/vllm-openai:v0.31.0"]
    assert vllm["out_of_scope_tags"] == ["dockerproxy.net/vllm/vllm-openai:v0.31.0"]

    old = rows[OLD_ID]
    assert old["dangling"] is True
    assert old["tracked"] is True
    assert old["removable"] is True


def test_remove_image_refuses_in_use_and_out_of_scope():
    docker, tracked = make_docker()
    with pytest.raises(ResourceInUse):
        docker.remove_image(VLLM_ID, tracked=tracked)
    with pytest.raises(ScopeViolation):
        docker.remove_image(OTHER_ID, tracked=tracked)


def test_remove_dangling_tracked_image_uses_id():
    docker, tracked = make_docker()
    preview = docker.remove_image(OLD_ID, tracked=tracked, dry_run=True)
    assert preview["command"] == ["docker", "rmi", OLD_ID]


def test_remove_tagged_image_only_touches_in_scope_tags():
    docker = Docker(
        Scope(image_repos=("vllm/vllm-openai",), containers=("vllm-qwen38",)),
        runner=fake_runner,
    )
    # pretend nothing uses the image so it becomes removable
    rows = docker.images(tracked=())
    target = next(r for r in rows if r["id"] == VLLM_ID)
    assert target["removable"] is False  # still used by vllm-qwen38 / old-vllm


def test_container_guards():
    docker, _ = make_docker()
    with pytest.raises(ScopeViolation):
        docker.remove_container("mx-test_llm")
    with pytest.raises(ScopeViolation):
        docker.remove_container("old-vllm")       # unknown name and no managed label
    with pytest.raises(DockerError):
        docker.remove_container("vllm-qwen38")    # running
    assert docker.remove_container("vllm-old", dry_run=True)["command"] == ["docker", "rm", "vllm-old"]


def test_pull_rejects_out_of_scope_ref():
    docker, _ = make_docker()

    async def go():
        async for _ in docker.pull_stream("timescale/timescaledb:latest"):
            pass

    with pytest.raises(ScopeViolation):
        asyncio.run(go())


def test_summary_shape():
    docker, tracked = make_docker()
    summary = docker.summary(tracked=tracked)
    assert summary["available"] is True
    assert [c["name"] for c in summary["containers"]] == ["vllm-qwen38", "vllm-old"]
    assert summary["other_containers"] == 2
    assert summary["disk"]["image_count"] == 2
    assert summary["disk"]["reclaimable_mb"] > 900
