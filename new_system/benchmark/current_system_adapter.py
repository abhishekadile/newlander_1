#!/usr/bin/env python3
"""
current_system_adapter.py
==========================
Benchmark adapter for the EXISTING OpenCFU-based colony detection pipeline.

Calls the production system EXACTLY as server.js does — via HTTP POST to the
running /detect endpoint. Falls back to direct subprocess invocation of the
colonyDetector.js flow if the server is not running.

Returns:
    {
        "count": int,
        "colonies": list[dict],   # per-colony data if available
        "latency_ms": float,
    }
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parent
NEW_SYSTEM_ROOT = BENCHMARK_DIR.parent
REPO_ROOT = NEW_SYSTEM_ROOT.parent   # d:/work/cell/

SERVER_URL = "http://localhost:3000"
DETECT_ENDPOINT = f"{SERVER_URL}/detect"

# ──────────────────────────────────────────────────────────────────────────────


def _run_via_http(image_path: Path) -> dict:
    """
    Call the existing /detect HTTP endpoint.
    The server must already be running (node server.js).
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("pip install requests")

    t0 = time.perf_counter()
    with open(image_path, "rb") as f:
        resp = requests.post(
            DETECT_ENDPOINT,
            files={"image": (image_path.name, f, "image/jpeg")},
            timeout=120,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    resp.raise_for_status()
    data = resp.json()

    if not data.get("success"):
        raise RuntimeError(f"Server returned error: {data.get('error', data)}")

    return {
        "count": data.get("colonyCount", 0),
        "colonies": data.get("colonies", []),
        "latency_ms": latency_ms,
    }


def _server_is_running() -> bool:
    try:
        import requests
        r = requests.get(f"{SERVER_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _run_via_nodejs_subprocess(image_path: Path) -> dict:
    """
    Direct fallback: run a small Node.js inline script that instantiates
    ColonyDetector and runs detectColonies, then prints JSON to stdout.
    Mirrors what server.js does in the /detect handler.
    """
    js_code = f"""
const ColonyDetector = require({json.dumps(str(REPO_ROOT / 'colonyDetector.js').replace(os.sep, '/'))});
const path = require('path');

(async () => {{
    const detector = new ColonyDetector();
    const imagePath = {json.dumps(str(image_path).replace(os.sep, '/'))};
    const params = {{
        threshold_type: 'regular',
        threshold_value: 15,
        min_radius: 3,
        max_radius: 50,
        enable_color_grouping: false,
        coarseness: 10.0,
        neighbours: 10
    }};
    const t0 = Date.now();
    const result = await detector.detectColonies(imagePath, params);
    const latency_ms = Date.now() - t0;
    console.log(JSON.stringify({{ ...result, latency_ms }}));
}})();
"""
    t0 = time.perf_counter()

    # Check if node is available
    try:
        node_result = subprocess.run(
            ["node", "-e", js_code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
    except FileNotFoundError:
        raise RuntimeError("node not found on PATH. Install Node.js or start server.js.")

    if node_result.returncode != 0:
        raise RuntimeError(
            f"Node.js subprocess failed:\n{node_result.stderr[-500:]}"
        )

    try:
        data = json.loads(node_result.stdout.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse node output: {e}\nOutput: {node_result.stdout[:500]}"
        )

    if not data.get("success"):
        raise RuntimeError(f"ColonyDetector returned error: {data.get('error', data)}")

    # Use the subprocess wall-clock time as latency if not embedded in result
    lat = data.pop("latency_ms", latency_ms)
    return {
        "count": data.get("colonyCount", 0),
        "colonies": data.get("colonies", []),
        "latency_ms": lat,
    }


def run(image_path: Path) -> dict:
    """
    Run the current (OpenCFU-based) system on image_path.
    Tries HTTP endpoint first; falls back to direct Node.js subprocess.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if _server_is_running():
        return _run_via_http(image_path)
    else:
        return _run_via_nodejs_subprocess(image_path)


# ── CLI test entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to test image")
    args = parser.parse_args()
    result = run(Path(args.image))
    print(json.dumps(result, indent=2))
