#!/bin/bash
# Hard-tune pipeline orchestrator — runs the full tuner suite end-to-end.
# Started 2026-05-04. Production-ready synthesis + validation + deploy.
#
# Stages:
#   1. Wait for pass1 to complete (TUNE_DAYS=180 must already be running)
#   2. Pass2 fine grid (depends on pass1)
#   3. Direction bias + Trail sweep IN PARALLEL (independent, both depend on pass2)
#   4. Rescue losers + Toxic hours IN PARALLEL (independent)
#   5. Walk-forward overfit detector (sequential)
#   6. Synthesize auto_tuned.py (merges everything)
#   7. Validate (last-mile sanity gate)
#   8. Restart trader so it loads the new auto_tuned.py
#
# Runs WITH the live trader. Each stage logs to logs/hard_tune_*.log
# and updates a status file at logs/hard_tune_status.txt so a watcher
# can stream progress.
set -uo pipefail

cd /Users/ashish/Documents/beast-trader
LOG_DIR="logs"
STATUS="$LOG_DIR/hard_tune_status.txt"
RES="backtest/results"
PASS1_JSON="$RES/tune_180d_pass1.json"

mark() {
  printf '%s  %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$STATUS"
}

run_stage() {
  local name="$1"; shift
  mark "START $name"
  if "$@" > "$LOG_DIR/hard_tune_$name.log" 2>&1; then
    mark "DONE  $name"
    return 0
  fi
  mark "FAIL  $name (see $LOG_DIR/hard_tune_$name.log)"
  return 1
}

# ── 1. Wait for pass1 ──────────────────────────────────────────────
mark "WAIT  pass1 (looking for $PASS1_JSON mtime > 5/3 23:03)"
while :; do
  if [[ -f "$PASS1_JSON" ]]; then
    # The pre-existing pass1 was at May 2 23:03. Anything after May 4 is fresh.
    mtime=$(stat -f %m "$PASS1_JSON")
    # Anything after 1777939200 (= 2026-05-04 00:00 UTC) is current run.
    if [[ "$mtime" -gt 1777939200 ]]; then break; fi
  fi
  sleep 30
done
mark "DONE  pass1 detected (fresh JSON written)"

# ── 2. Pass2 (sequential) ──────────────────────────────────────────
TUNE_DAYS=180 TUNE_WORKERS=6 \
  run_stage pass2 python3 -B scripts/tune_pass2_fine.py

# ── 3. Direction bias + Trail sweep in parallel ────────────────────
mark "START dir_bias + trail_sweep (parallel)"
TUNE_DAYS=180 \
  python3 -B scripts/tune_direction_bias.py > "$LOG_DIR/hard_tune_dir_bias.log" 2>&1 &
DB_PID=$!
TUNE_DAYS=180 \
  python3 -B scripts/sweep_trails.py > "$LOG_DIR/hard_tune_trail.log" 2>&1 &
TR_PID=$!
wait $DB_PID && mark "DONE  dir_bias" || mark "FAIL  dir_bias"
wait $TR_PID && mark "DONE  trail_sweep" || mark "FAIL  trail_sweep"

# ── 4. Rescue + toxic hours in parallel ────────────────────────────
mark "START rescue + toxic_hours (parallel)"
python3 -B scripts/rescue_losers.py > "$LOG_DIR/hard_tune_rescue.log" 2>&1 &
RE_PID=$!
python3 -B scripts/profile_toxic_hours.py > "$LOG_DIR/hard_tune_toxic.log" 2>&1 &
TX_PID=$!
wait $RE_PID && mark "DONE  rescue_losers" || mark "FAIL  rescue_losers"
wait $TX_PID && mark "DONE  toxic_hours" || mark "FAIL  toxic_hours"

# ── 5. Walk-forward overfit detector ───────────────────────────────
WF_WORKERS=4 run_stage walk_forward python3 -B scripts/walk_forward.py

# ── 6. Synthesize ──────────────────────────────────────────────────
run_stage synthesize python3 -B scripts/synthesize_auto_tuned.py

# ── 7. Validate ────────────────────────────────────────────────────
run_stage validate python3 -B scripts/validate_tuned.py

# ── 8. Restart trader to load new auto_tuned.py ────────────────────
mark "RESTART live trader (com.dragon.trader)"
launchctl kickstart -k "gui/$(id -u)/com.dragon.trader"
sleep 4
if pgrep -f "beast-trader/run.py" > /dev/null; then
  mark "DONE  trader restarted"
else
  mark "FAIL  trader did not restart"
fi

mark "ALL STAGES COMPLETE"
