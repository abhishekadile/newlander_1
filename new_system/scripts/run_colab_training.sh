#!/usr/bin/env bash
# run_colab_training.sh — LOCAL orchestration script.
#
# Runs the full training pipeline on a Google Colab T4 GPU via the official
# Google Colab CLI (https://github.com/googlecolab/google-colab-cli).
#
# Usage:
#   bash new_system/scripts/run_colab_training.sh
#
# Prerequisites:
#   - google-colab-cli installed: pip install google-colab-cli
#   - First run requires interactive Google OAuth in your browser.
#     The script will pause and prompt you to complete the login.
#
# This script MUST be run from the repository root (d:/work/cell).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NEW_SYSTEM="$REPO_ROOT/new_system"
RUNS_DIR="$NEW_SYSTEM/runs"
WEIGHTS_DIR="$NEW_SYSTEM/weights"
SCRIPTS_DIR="$NEW_SYSTEM/scripts"

mkdir -p "$RUNS_DIR" "$WEIGHTS_DIR"

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

log() { echo "[colab] $(date '+%H:%M:%S') $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Trap: always stop the Colab session, even on failure
# ---------------------------------------------------------------------------

SESSION_STARTED=false

cleanup() {
    if $SESSION_STARTED; then
        log "Stopping Colab session …"
        colab stop 2>/dev/null || true
        log "Session stopped."
    fi
}
trap cleanup EXIT ERR INT TERM

# ---------------------------------------------------------------------------
# Step 0: Verify Colab CLI is installed + read actual help text
# ---------------------------------------------------------------------------

log "=== Step 0: Colab CLI verification ==="
if ! command -v colab &>/dev/null; then
    die "colab CLI not found. Install with: pip install google-colab-cli"
fi

log "--- colab --help ---"
colab --help || true
echo ""

log "--- colab exec --help ---"
colab exec --help || true
echo ""

log "--- colab new --help ---"
colab new --help || true
echo ""

# ---------------------------------------------------------------------------
# Step 1: Provision a T4 session
# ---------------------------------------------------------------------------

log "=== Step 1: Provisioning T4 session ==="
log "You may be prompted to complete Google OAuth in your browser."
log "Complete the browser login, then return here and press Enter to continue."
echo ""

# Provision — the CLI may open a browser tab for OAuth on first use.
# Capture output to verify T4 assignment.
PROVISION_OUTPUT="$(colab new --gpu T4 2>&1)" || die "colab new failed."
echo "$PROVISION_OUTPUT"

# Check that we actually got a T4
if ! echo "$PROVISION_OUTPUT" | grep -qi "T4\|t4"; then
    log "WARNING: Could not confirm T4 in provisioning output."
    log "Output was:"
    echo "$PROVISION_OUTPUT"
    read -rp "[colab] Press Enter to continue anyway, or Ctrl-C to abort: "
fi

SESSION_STARTED=true

# Record session start time for the monitor script
SESSION_START_FILE="$RUNS_DIR/.colab_session_started_at"
date --iso-8601=seconds > "$SESSION_START_FILE" 2>/dev/null \
    || python3 -c "from datetime import datetime; print(datetime.utcnow().isoformat())" > "$SESSION_START_FILE"
log "Session start time written to $SESSION_START_FILE"

# ---------------------------------------------------------------------------
# Step 2: Mount Google Drive (backup in case of VM disk loss)
# ---------------------------------------------------------------------------

log "=== Step 2: Mounting Google Drive ==="
colab drivemount || log "WARNING: drivemount failed or not supported — continuing without Drive backup."

# ---------------------------------------------------------------------------
# Step 3: Install remote dependencies
# ---------------------------------------------------------------------------

log "=== Step 3: Installing remote packages ==="
colab install ultralytics openpyxl scikit-learn || die "Package installation failed."

# ---------------------------------------------------------------------------
# Step 4: Upload remote scripts (NO data files — data fetched on VM in Step 5)
# ---------------------------------------------------------------------------

log "=== Step 4: Uploading scripts to Colab VM ==="
colab upload "$SCRIPTS_DIR/remote_dataset_setup.py" remote_dataset_setup.py \
    || die "Failed to upload remote_dataset_setup.py"
colab upload "$SCRIPTS_DIR/train_colony_model.py" train_colony_model.py \
    || die "Failed to upload train_colony_model.py"
colab upload "$SCRIPTS_DIR/export_model.py" export_model.py \
    || die "Failed to upload export_model.py"
log "Scripts uploaded."

# ---------------------------------------------------------------------------
# Step 5: Run dataset setup remotely
# ---------------------------------------------------------------------------

log "=== Step 5: Remote dataset setup (download + convert on VM) ==="
log "This fetches ~612 MB directly to the Colab VM disk — not routed through local machine."
colab exec -f "$SCRIPTS_DIR/remote_dataset_setup.py" \
    || die "remote_dataset_setup.py failed. See output above."

# Pull the generated data.yaml back locally for reference
log "Pulling data.yaml to local machine …"
colab download /content/configs/data.yaml "$NEW_SYSTEM/configs/data.yaml" \
    || log "WARNING: Could not pull data.yaml — continuing."

# ---------------------------------------------------------------------------
# Step 6: Launch training (detached so we can monitor without blocking)
# ---------------------------------------------------------------------------

log "=== Step 6: Launching training (detached) ==="

# Detect whether the CLI has a native detach/background flag from the help text.
# We captured exec --help above; check for a detach flag.
DETACH_FLAG=""
if colab exec --help 2>&1 | grep -qi -- "--detach\|--background\|-d "; then
    log "CLI has a native detach flag — using it."
    DETACH_FLAG="--detach"
fi

if [ -n "$DETACH_FLAG" ]; then
    colab exec $DETACH_FLAG -c "python train_colony_model.py > train.log 2>&1" \
        || die "Failed to launch training."
else
    log "No native detach flag found — using nohup workaround."
    colab exec -c "nohup python train_colony_model.py > train.log 2>&1 & echo TRAINING_PID=\$! && disown" \
        || die "Failed to launch training."
fi

log "Training launched in background on Colab VM."
log ""
log "=== Next steps — run in separate terminals ==="
log "  Monitor progress:    python new_system/scripts/monitor_training.py --tier free"
log "  Sync checkpoints:    python new_system/scripts/sync_checkpoints.py"
log ""
log "Press Enter here when training is complete to proceed with final downloads."
read -rp "[colab] Training complete? Press Enter to continue: "

# ---------------------------------------------------------------------------
# Step 7: Final artifact downloads
# ---------------------------------------------------------------------------

log "=== Step 7: Final downloads ==="

# best.pt
colab download runs/yolo26n_colony/weights/best.pt "$WEIGHTS_DIR/best.pt" \
    && log "Downloaded best.pt" \
    || log "WARNING: Could not download best.pt"

# last.pt
colab download runs/yolo26n_colony/weights/last.pt "$WEIGHTS_DIR/last.pt" \
    && log "Downloaded last.pt" \
    || log "WARNING: Could not download last.pt"

# results.csv
colab download runs/yolo26n_colony/results.csv "$RUNS_DIR/results.csv" \
    && log "Downloaded results.csv" \
    || log "WARNING: Could not download results.csv"

# KNOWN_LIMITATIONS.md
colab download runs/yolo26n_colony/KNOWN_LIMITATIONS.md "$RUNS_DIR/KNOWN_LIMITATIONS.md" \
    && log "Downloaded KNOWN_LIMITATIONS.md" \
    || log "WARNING: Could not download KNOWN_LIMITATIONS.md"

# ---------------------------------------------------------------------------
# Step 8: Run export (ONNX + OpenVINO) on the VM, then download
# ---------------------------------------------------------------------------

log "=== Step 8: Remote export (ONNX + OpenVINO) ==="
colab exec -f "$SCRIPTS_DIR/export_model.py" \
    || log "WARNING: export_model.py failed — weights may still be usable as .pt"

# Download exported models
for ext in onnx; do
    colab download "runs/yolo26n_colony/weights/best.${ext}" "$WEIGHTS_DIR/best.${ext}" \
        && log "Downloaded best.${ext}" \
        || log "WARNING: Could not download best.${ext}"
done

# OpenVINO exports to a directory
colab download "runs/yolo26n_colony/weights/best_openvino_model" "$WEIGHTS_DIR/best_openvino_model" \
    && log "Downloaded OpenVINO model directory" \
    || log "WARNING: Could not download OpenVINO model — try manual colab download."

# ---------------------------------------------------------------------------
# Step 9: Save session log
# ---------------------------------------------------------------------------

log "=== Step 9: Saving session log ==="
colab log --output "$RUNS_DIR/colab_session_log.ipynb" \
    && log "Session log saved." \
    || log "WARNING: Could not save session log."

log ""
log "=== All done. Colab session will now be stopped. ==="
# cleanup() via EXIT trap will call `colab stop`
