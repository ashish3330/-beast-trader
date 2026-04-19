#!/bin/bash
# Weekly retrain script for Dragon/Beast trader
# Runs cache refresh, ML training, then restarts the trader.

set -euo pipefail

BASE_DIR="/Users/ashish/Documents/beast-trader"
LOG_FILE="${BASE_DIR}/logs/retrain.log"
PYTHON="python3 -B"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$LOG_FILE"
}

mkdir -p "${BASE_DIR}/logs"

log "========== WEEKLY RETRAIN START =========="

log "Step 1: Refreshing cache..."
if $PYTHON "${BASE_DIR}/scripts/refresh_cache.py" >> "$LOG_FILE" 2>&1; then
    log "Cache refresh completed successfully."
else
    log "ERROR: Cache refresh failed (exit code $?)."
    exit 1
fi

log "Step 2: Training meta-labels..."
if $PYTHON "${BASE_DIR}/train_meta_labels.py" >> "$LOG_FILE" 2>&1; then
    log "ML training completed successfully."
else
    log "ERROR: ML training failed (exit code $?)."
    exit 1
fi

log "Step 3: Restarting dragon trader..."
launchctl stop com.dragon.trader
sleep 2
launchctl start com.dragon.trader
log "Dragon trader restarted."

log "========== WEEKLY RETRAIN COMPLETE =========="
