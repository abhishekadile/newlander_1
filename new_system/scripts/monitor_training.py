"""
monitor_training.py — runs LOCALLY during Colab training.

Polls the remote Colab VM's results.csv (written by Ultralytics after each epoch)
and prints a live single-line progress update. Also tracks session age and warns
before approaching the Colab tier time limit.

Usage:
    python new_system/scripts/monitor_training.py [--tier free|pro|pro_plus] [--interval 60]

Run this in a separate terminal while run_colab_training.sh is active.
"""

import argparse
import csv
import io
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Tier session budget assumptions (hours).
# These are rough estimates — actual limits vary and can change.
# The script accepts --tier as a user input so they can set their own expectation.
# If the Colab CLI exposes a quota/remaining-time command, that real value
# is preferred over these estimates.
# ---------------------------------------------------------------------------

TIER_HOURS = {
    "free":      12.0,
    "pro":       24.0,
    "pro_plus":  48.0,
}

SESSION_WARN_FRACTION = 0.80   # warn at 80 % of budget

SESSION_FILE = Path(__file__).parent.parent / "runs" / ".colab_session_started_at"

# Ultralytics prefixes the task type (detect/) to the project path
REMOTE_RESULTS_CSV = "runs/detect/runs/yolo26n_colony/results.csv"
REMOTE_TRAIN_LOG   = "train.log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[monitor {ts}] {msg}", flush=True)


def warn(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[WARNING {ts}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def run_colab_download(remote: str, local: str) -> tuple[int, str]:
    """Download a file from Colab VM; return (returncode, output)."""
    try:
        result = subprocess.run(
            ["colab", "download", remote, local],
            capture_output=True, timeout=60,
        )
        out = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        return result.returncode, out
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except FileNotFoundError:
        return -1, "colab CLI not found"


_TMP_RESULTS = "/tmp/_monitor_results.csv"
_TMP_LOG     = "/tmp/_monitor_train.log"


def try_get_quota() -> str | None:
    """Attempt to read real quota from CLI. Returns string or None."""
    return None  # No quota subcommand in current Colab CLI version


def fetch_remote_results() -> str | None:
    """Download results.csv from VM and return its last data line."""
    rc, _ = run_colab_download(
        f"/content/{REMOTE_RESULTS_CSV}",
        _TMP_RESULTS
    )
    if rc != 0:
        return None
    try:
        import os
        if not os.path.exists(_TMP_RESULTS):
            return None
        with open(_TMP_RESULTS, encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) < 2:
            return None
        # Return header + last data line so parse_results_row can use them
        return lines[0] + "\n" + lines[-1]
    except Exception:
        return None


def parse_results_row(row_text: str) -> dict | None:
    """
    Parse header+last-row format returned by fetch_remote_results().
    Ultralytics results.csv columns (may vary by version):
      epoch, train/box_loss, train/cls_loss, train/dfl_loss,
      metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
      val/box_loss, val/cls_loss, val/dfl_loss, time
    """
    lines = [l.strip() for l in row_text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return None

    header_line = lines[0]
    data_line   = lines[-1]

    # Use DictReader with the actual header
    try:
        reader = csv.DictReader(io.StringIO(header_line + "\n" + data_line))
        rows = list(reader)
        if not rows:
            return None
        row = rows[0]
    except Exception:
        return None

    result: dict = {}

    # Find epoch column (usually first, named 'epoch' or '                   epoch')
    for k, v in row.items():
        if "epoch" in k.lower():
            try:
                result["epoch"] = int(float(v.strip()))
            except (ValueError, AttributeError):
                pass

    if "epoch" not in result:
        return None

    # Map column name fragments to result keys
    col_map = {
        "box_loss":  "box_loss",
        "precision": "precision",
        "recall":    "recall",
        "map50-95":  "mAP5095",
        "map50":     "mAP50",   # must come after map50-95 to avoid partial match
    }
    for k, v in row.items():
        k_lower = k.lower().strip()
        for fragment, result_key in col_map.items():
            if fragment in k_lower and result_key not in result:
                try:
                    result[result_key] = float(v.strip())
                except (ValueError, AttributeError):
                    pass

    return result if len(result) > 1 else None


def fetch_full_results_csv() -> list[dict]:
    """Read the locally-cached results.csv (downloaded by fetch_remote_results)."""
    import os
    if not os.path.exists(_TMP_RESULTS):
        return []
    try:
        with open(_TMP_RESULTS, encoding="utf-8", errors="replace") as f:
            content = f.read()
        lines = [l for l in content.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return []
        header = [h.strip() for h in lines[0].split(",")]
        rows = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            d: dict = {}
            for i, h in enumerate(header):
                try:
                    d[h] = float(parts[i])
                except (IndexError, ValueError):
                    d[h] = parts[i] if i < len(parts) else ""
            rows.append(d)
        return rows
    except Exception:
        return []


def compute_eta(rows: list[dict], total_epochs: int) -> tuple[float | None, float | None]:
    """
    Returns (avg_epoch_seconds, eta_seconds).
    Uses the 'time' column if present (Ultralytics writes elapsed time per epoch),
    otherwise estimates from wall-clock deltas.
    """
    if not rows:
        return None, None

    time_col = next((k for k in rows[0] if "time" in k.lower()), None)
    if time_col:
        times = []
        for r in rows:
            try:
                times.append(float(r[time_col]))
            except (ValueError, TypeError):
                pass
        if len(times) >= 2:
            avg_epoch_sec = sum(times) / len(times)
            completed = len(times)
            remaining = total_epochs - completed
            eta = avg_epoch_sec * remaining
            return avg_epoch_sec, eta

    return None, None


# ---------------------------------------------------------------------------
# Session-age tracking
# ---------------------------------------------------------------------------

def session_age_seconds() -> float | None:
    if not SESSION_FILE.exists():
        return None
    try:
        ts_str = SESSION_FILE.read_text().strip()
        # Parse ISO 8601 (with or without timezone)
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f%z"):
            try:
                ts = datetime.strptime(ts_str[:26], fmt[:len(fmt)])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return (datetime.now(timezone.utc) - ts).total_seconds()
            except ValueError:
                continue
    except Exception:
        pass
    return None


def check_session_age(tier: str, budget_hours: float, warned: set) -> None:
    age = session_age_seconds()
    if age is None:
        return

    # Try real quota first
    quota_info = try_get_quota()
    if quota_info:
        log(f"Colab quota info: {quota_info}")
        return

    budget_sec = budget_hours * 3600
    fraction = age / budget_sec

    if fraction >= SESSION_WARN_FRACTION and "80pct" not in warned:
        warn(
            f"Session age {fmt_duration(age)} — approaching assumed {tier.upper()} tier "
            f"limit (~{budget_hours:.0f}h). "
            "Expect possible disconnect soon. Ensure sync_checkpoints.py is running."
        )
        warned.add("80pct")

    if fraction >= 0.95 and "95pct" not in warned:
        warn(
            f"Session age {fmt_duration(age)} — at 95 % of assumed {tier.upper()} "
            f"tier limit. Disconnect likely imminent."
        )
        warned.add("95pct")


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------

def monitor(tier: str, interval: int, total_epochs: int) -> None:
    budget_hours = TIER_HOURS.get(tier, TIER_HOURS["free"])
    warned: set = set()

    log(f"Starting monitor: tier={tier}, budget={budget_hours}h, "
        f"poll_interval={interval}s, total_epochs={total_epochs}")
    log(f"Session start file: {SESSION_FILE}")
    log("Press Ctrl-C to stop monitoring.\n")

    last_epoch_printed = -1

    while True:
        try:
            # --- Session age check ---
            check_session_age(tier, budget_hours, warned)

            # --- Fetch latest epoch row ---
            row_text = fetch_remote_results()
            if not row_text:
                log("Waiting for results.csv to appear on VM …")
            else:
                row = parse_results_row(row_text)
                if row and row.get("epoch", -1) != last_epoch_printed:
                    last_epoch_printed = row.get("epoch", -1)

                    # Elapsed session time
                    age = session_age_seconds()
                    elapsed_str = fmt_duration(age) if age else "unknown"

                    # ETA
                    rows = fetch_full_results_csv()
                    avg_sec, eta_sec = compute_eta(rows, total_epochs)
                    eta_str = fmt_duration(eta_sec) if eta_sec else "?"

                    # Build status line
                    epoch    = row.get("epoch", "?")
                    box_loss = f"{row['box_loss']:.3f}" if "box_loss" in row else "?"
                    map50    = f"{row['mAP50']:.3f}"    if "mAP50"    in row else "?"
                    map5095  = f"{row['mAP5095']:.3f}"  if "mAP5095"  in row else "?"
                    recall   = f"{row['recall']:.3f}"   if "recall"   in row else "?"

                    print(
                        f"Epoch {epoch}/{total_epochs} | "
                        f"box_loss {box_loss} | "
                        f"recall {recall} | "
                        f"mAP50 {map50} | "
                        f"mAP50-95 {map5095} | "
                        f"elapsed {elapsed_str} | "
                        f"ETA {eta_str}",
                        flush=True,
                    )
                elif row is None:
                    log(f"Raw result line (could not parse epoch): {row_text[:80]}")

            time.sleep(interval)

        except KeyboardInterrupt:
            log("Monitor stopped by user.")
            sys.exit(0)
        except Exception as exc:
            log(f"Monitor error: {exc} — will retry in {interval}s")
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live monitor for YOLO26n colony detection training on Colab."
    )
    parser.add_argument(
        "--tier",
        choices=["free", "pro", "pro_plus"],
        default="free",
        help="Colab tier — used to estimate session time budget (default: free ≈ 12h). "
             "Set to your actual tier. If the CLI exposes a quota command, that real "
             "value will be used instead of this estimate.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Poll interval in seconds (default: 60).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Total training epochs (default: 200, matches train_colony_model.py).",
    )
    args = parser.parse_args()
    monitor(args.tier, args.interval, args.epochs)


if __name__ == "__main__":
    main()
