from monitor import launcher
from monitor.controller import match_startup


def patterns(engine):
    return launcher.load_spec(engine).get("startup") or []


def test_vllm_phases_advance():
    p = patterns("vllm")
    score, phase, detail, error = match_startup(
        p, ["(EngineCore) INFO [core.py:129] Initializing a V1 LLM engine (v0.31.0)"])
    assert phase == "初始化引擎" and not error and score == 0

    score, phase, _detail, error = match_startup(
        p, ["INFO [default_loader.py:484] Loading weights took 13.47 seconds"])
    assert phase == "加载权重" and not error and score == 1

    score, phase, _detail, error = match_startup(
        p, ["INFO [model_runner.py:1087] Graph capturing finished in 11 secs, took 1.05 GiB"])
    assert phase == "捕获 CUDA Graph" and not error and score == 3

    score, phase, _detail, error = match_startup(
        p, ["INFO:     Application startup complete."])
    assert phase == "就绪" and not error


def test_vllm_error_marker_wins():
    p = patterns("vllm")
    lines = [
        "INFO [default_loader.py:484] Loading weights took 13.47 seconds",
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
    ]
    score, phase, detail, error = match_startup(p, lines)
    assert error is True
    assert phase == "错误"
    assert "CUDA out of memory" in detail


def test_ninfer_phases():
    p = patterns("ninfer")
    lines = [
        "2026-10-17 10:02:25.130  INFO  starting engine",
        "2026-10-17 10:02:25.669  INFO  loading weights | 19.7 GiB",
        "2026-10-17 10:02:28.824  INFO  weights ready | 19.7 GiB | 3.2s",
    ]
    score, phase, _d, error = match_startup(p, lines)
    assert phase == "权重就绪" and not error
    assert score == 2

    score, phase, _d, error = match_startup(
        p, ["2026-10-17 10:02:31.616  INFO  listening on http://0.0.0.0:8000"])
    assert phase == "监听中"


def test_unknown_lines_do_not_advance():
    score, phase, detail, error = match_startup(patterns("vllm"), ["nothing to see here"])
    assert score == -1 and phase == "" and detail == "" and error is False
