#!/usr/bin/env python3
"""OpenAI-compatible load generator, useful for validating the monitor.

Runs N concurrent streaming chat requests against the monitor gateway (so every
request goes through the tap) for a fixed duration, then prints client-side
numbers you can compare with the monitor's history.

    python tools/bench.py --concurrency 8 --duration 300
    python tools/bench.py --url http://127.0.0.1:8000 --model qwen3.8-27b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
import uuid

import httpx

WORDS = (
    "speculative decoding prefetch kv cache attention kernel tensor parallel pipeline "
    "latency throughput batching scheduler quantization expert routing memory bandwidth "
    "prefill decode tokenizer sampling temperature nucleus context window checkpoint "
    "inference server gateway monitor observability telemetry histogram percentile "
).split()


def make_prompt(rng: random.Random, target_tokens: int) -> str:
    nonce = uuid.uuid4().hex[:8]
    words = [rng.choice(WORDS) for _ in range(max(10, target_tokens))]
    body = " ".join(words)
    return (
        f"[{nonce}] You are given a technical note. Summarize the key trade-offs in 3 bullet "
        f"points, then answer: which one matters most for a single-GPU deployment?\n\n{body}"
    )


async def one_request(client: httpx.AsyncClient, args, rng: random.Random, index: int) -> dict:
    prompt_tokens = rng.choice(args.prompt_tokens)
    max_tokens = rng.choice(args.max_tokens)
    thinking = index % 4 == 0  # 1 in 4 workers exercises the thinking path
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": make_prompt(rng, prompt_tokens)}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not thinking:
        payload["reasoning_effort"] = "none"

    started = time.time()
    first = None
    reasoning_chunks = 0
    content_chunks = 0
    usage: dict = {}
    status = None
    error = None
    try:
        async with client.stream("POST", f"{args.url}/v1/chat/completions", json=payload) as resp:
            status = resp.status_code
            if resp.status_code >= 400:
                error = (await resp.aread()).decode("utf-8", "replace")[:200]
            else:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning") or delta.get("reasoning_content"):
                            reasoning_chunks += 1
                        elif delta.get("content"):
                            content_chunks += 1
                        if first is None and (delta.get("content") or delta.get("reasoning")
                                              or delta.get("reasoning_content")):
                            first = time.time()
    except Exception as exc:  # network / disconnect
        error = f"{type(exc).__name__}: {exc}"

    finished = time.time()
    return {
        "ok": error is None and (status or 0) < 400,
        "status": status,
        "error": error,
        "ttft": (first - started) if first else None,
        "total": finished - started,
        "prompt_tokens": (usage.get("prompt_tokens") or 0),
        "completion_tokens": (usage.get("completion_tokens") or (reasoning_chunks + content_chunks)),
        "reasoning_tokens": ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or reasoning_chunks),
        "thinking": thinking,
        "max_tokens": max_tokens,
        "prompt_target": prompt_tokens,
    }


async def worker(client, args, deadline, index, results, counters):
    rng = random.Random(1000 + index)
    while time.time() < deadline:
        counters["inflight"] += 1
        try:
            results.append(await one_request(client, args, rng, index))
        finally:
            counters["inflight"] -= 1
        if args.pause_ms:
            await asyncio.sleep(rng.uniform(0, args.pause_ms / 1000.0))


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return ordered[k]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=300, help="seconds")
    parser.add_argument("--prompt-tokens", default="200,600,1500,3000",
                        help="comma list of target prompt word counts")
    parser.add_argument("--max-tokens", default="128,256,384,512")
    parser.add_argument("--pause-ms", type=float, default=0, help="random pause between requests")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    args.prompt_tokens = [int(x) for x in str(args.prompt_tokens).split(",")]
    args.max_tokens = [int(x) for x in str(args.max_tokens).split(",")]

    results: list[dict] = []
    counters = {"inflight": 0}
    deadline = time.time() + args.duration
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), limits=limits) as client:
        tasks = [asyncio.create_task(worker(client, args, deadline, i, results, counters))
                 for i in range(args.concurrency)]
        started = time.time()
        while any(not task.done() for task in tasks):
            await asyncio.sleep(15)
            done = len(results)
            elapsed = time.time() - started
            ok = sum(1 for r in results if r["ok"])
            out_tokens = sum(r["completion_tokens"] for r in results)
            print(
                f"[{elapsed:5.0f}s] done={done:4d} ok={ok:4d} inflight={counters['inflight']:2d} "
                f"out_tok/s={out_tokens / max(elapsed, 1):6.1f}",
                flush=True,
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    ttfts = [r["ttft"] * 1000 for r in ok if r["ttft"]]
    totals = [r["total"] * 1000 for r in ok]
    decodes = [
        (r["completion_tokens"] - 1) / max(r["total"] - (r["ttft"] or 0), 0.05)
        for r in ok if r["completion_tokens"] > 1
    ]
    span = max((r["total"] for r in ok), default=1)
    prompt_tokens = sum(r["prompt_tokens"] for r in ok)
    completion_tokens = sum(r["completion_tokens"] for r in ok)
    reasoning_tokens = sum(r["reasoning_tokens"] for r in ok)

    print("\n=== client side ===")
    print(f"requests      : {len(results)}  ok={len(ok)} failed={len(failed)}")
    print(f"prompt tokens : {prompt_tokens:,}")
    print(f"output tokens : {completion_tokens:,}  (thinking {reasoning_tokens:,})")
    print(f"ttft ms       : p50={pct(ttfts, .5):.0f} p90={pct(ttfts, .9):.0f} p99={pct(ttfts, .99):.0f}")
    print(f"total ms      : p50={pct(totals, .5):.0f} p90={pct(totals, .9):.0f}")
    print(f"decode tok/s  : avg={statistics.mean(decodes) if decodes else 0:.1f} "
          f"p50={pct(decodes, .5) or 0:.1f} p10={pct(decodes, .1) or 0:.1f}")
    if failed:
        print("first errors  :")
        for row in failed[:3]:
            print(f"  {row['status']} {row['error']}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
