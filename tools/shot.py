#!/usr/bin/env python3
"""Screenshot a URL through the Chrome DevTools Protocol (waits for real network).

usage: shot.py URL OUT.png [--width 1600] [--height 3400] [--wait 3]
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import sys
import time

import httpx
import websockets

CHROME = shutil.which("chromium") or "/snap/bin/chromium"


async def capture(url: str, out: str, width: int, height: int, wait: float, port: int) -> None:
    proc = subprocess.Popen(
        [
            CHROME, "--headless=new", "--no-sandbox", "--disable-gpu",
            "--hide-scrollbars", "--force-prefers-reduced-motion",
            f"--remote-debugging-port={port}",
            f"--window-size={width},{height}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 20
        target = None
        while time.time() < deadline:
            try:
                resp = httpx.get(f"http://127.0.0.1:{port}/json", timeout=1.0)
                pages = [t for t in resp.json() if t.get("type") == "page"]
                if pages:
                    target = pages[0]
                    break
            except Exception:
                pass
            await asyncio.sleep(0.2)
        if not target:
            raise SystemExit("chrome did not expose a page target")

        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024) as ws:
            counter = 0

            async def call(method: str, params: dict | None = None) -> dict:
                nonlocal counter
                counter += 1
                await ws.send(json.dumps({"id": counter, "method": method, "params": params or {}}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == counter:
                        return msg

            await call("Page.enable")
            await call("Emulation.setDeviceMetricsOverride", {
                "width": width, "height": height, "deviceScaleFactor": 1, "mobile": False,
            })
            await call("Page.navigate", {"url": url})
            await asyncio.sleep(wait)
            shot = await call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
            data = shot.get("result", {}).get("data")
            if not data:
                raise SystemExit(f"capture failed: {shot}")
            with open(out, "wb") as handle:
                handle.write(base64.b64decode(data))
            print(f"wrote {out}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    url = sys.argv[1]
    out = sys.argv[2]
    width = height = None
    wait = 3.0
    args = sys.argv[3:]
    for i, arg in enumerate(args):
        if arg == "--width":
            width = int(args[i + 1])
        elif arg == "--height":
            height = int(args[i + 1])
        elif arg == "--wait":
            wait = float(args[i + 1])
    asyncio.run(capture(url, out, width or 1600, height or 3400, wait, 9223))


if __name__ == "__main__":
    main()
