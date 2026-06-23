"""
current_system_adapter.py

Adapter for the existing OpenCFU-based colony detection pipeline.
Calls the production system EXACTLY as server.js's /detect route does —
no modifications to the underlying system.

Two execution modes (tried in order):
  1. HTTP POST to http://localhost:3000/detect  (requires the Express server to be running)
  2. Direct Node.js subprocess: spawns a minimal wrapper that calls ColonyDetector

Returns:
    {"count": int, "latency_ms": float, "raw_response": dict, "error": str | None}
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL     = "http://localhost:3000/detect"
SERVER_TIMEOUT = 60       # seconds
REPO_ROOT      = Path(__file__).parent.parent.parent   # d:/work/cell


# ---------------------------------------------------------------------------
# Mode 1: HTTP
# ---------------------------------------------------------------------------

def _detect_via_http(image_path: str) -> dict:
    import urllib.request
    import urllib.parse

    image_path = str(Path(image_path).resolve())
    payload = json.dumps({"imagePath": image_path}).encode()
    req = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=SERVER_TIMEOUT) as resp:
        body = json.loads(resp.read().decode())
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return body, latency_ms


def _server_is_up() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:3000/health", timeout=3)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Mode 2: Direct Node.js subprocess
# ---------------------------------------------------------------------------

_NODE_WRAPPER = """
const ColonyDetector = require('./colonyDetector');
const path = require('path');
const detector = new ColonyDetector();
const imagePath = process.argv[2];
detector.detectColonies(imagePath, {})
  .then(result => {
    process.stdout.write(JSON.stringify(result));
    process.exit(0);
  })
  .catch(err => {
    process.stdout.write(JSON.stringify({success: false, error: String(err)}));
    process.exit(1);
  });
"""


def _detect_via_node(image_path: str) -> dict:
    image_path = str(Path(image_path).resolve())
    t0 = time.perf_counter()
    result = subprocess.run(
        ["node", "-e", _NODE_WRAPPER, image_path],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(
            f"Node process returned no output (exit {result.returncode}). "
            f"stderr: {result.stderr[:300]}"
        )
    body = json.loads(stdout)
    return body, latency_ms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(image_path: str) -> dict:
    """
    Detect colonies in image_path using the current (OpenCFU) system.
    Tries HTTP first; falls back to direct Node.js subprocess.

    Returns:
        {
            "count":       int,      # detected colony count
            "latency_ms":  float,    # wall-clock time including preprocessing
            "raw_response": dict,    # full JSON response from the system
            "error":       str|None, # non-None if detection failed
            "mode":        str,      # "http" | "node"
        }
    """
    if _server_is_up():
        try:
            body, latency_ms = _detect_via_http(image_path)
            count = body.get("colonyCount", body.get("count", -1))
            return {
                "count":        int(count) if count is not None else -1,
                "latency_ms":   latency_ms,
                "raw_response": body,
                "error":        None if body.get("success", True) else body.get("error"),
                "mode":         "http",
            }
        except Exception as exc:
            # Fall through to node mode
            pass

    # Node subprocess fallback
    try:
        body, latency_ms = _detect_via_node(image_path)
        count = body.get("colonyCount", body.get("count", -1))
        return {
            "count":        int(count) if count is not None else -1,
            "latency_ms":   latency_ms,
            "raw_response": body,
            "error":        None if body.get("success", True) else body.get("error"),
            "mode":         "node",
        }
    except Exception as exc:
        return {
            "count":        -1,
            "latency_ms":   float("nan"),
            "raw_response": {},
            "error":        str(exc),
            "mode":         "node",
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python current_system_adapter.py <image_path>")
        sys.exit(1)
    result = run(sys.argv[1])
    print(json.dumps(result, indent=2))
