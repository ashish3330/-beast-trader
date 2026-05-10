#!/bin/bash
# Sequential trail-lock sweep. 4 parallel max.
# stderr to /dev/null (sklearn warnings only).
ROOT=/Users/ashish/Documents/beast-trader
OUT=$ROOT/backtest/results/trail_sweep_2026_05_10
mkdir -p "$OUT"
cd "$ROOT"

# Each spec = "var=value:label". Inline env-var-then-command form so the
# variable is exported into the python child's environment.
launch_batch() {
  local pids=()
  for spec in "$@"; do
    var_pair="${spec%%:*}"   # e.g. DRAGON_TRAIL_LOCK_AT_15R=0.5
    label="${spec#*:}"
    bash -c "$var_pair python3 -B backtest/v5_backtest.py --days 180 --all-symbols --with-slippage --with-commission --with-swap > $OUT/${label}.log 2>/dev/null" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "BATCH DONE"
}

echo "Batch 1/4: lock15R 0.5..0.8"
launch_batch \
  "DRAGON_TRAIL_LOCK_AT_15R=0.5:lock15R_0.5" \
  "DRAGON_TRAIL_LOCK_AT_15R=0.6:lock15R_0.6" \
  "DRAGON_TRAIL_LOCK_AT_15R=0.7:lock15R_0.7" \
  "DRAGON_TRAIL_LOCK_AT_15R=0.8:lock15R_0.8"

echo "Batch 2/4: lock15R 0.9 + lock10R 0.2..0.4"
launch_batch \
  "DRAGON_TRAIL_LOCK_AT_15R=0.9:lock15R_0.9" \
  "DRAGON_TRAIL_LOCK_AT_10R=0.2:lock10R_0.2" \
  "DRAGON_TRAIL_LOCK_AT_10R=0.3:lock10R_0.3" \
  "DRAGON_TRAIL_LOCK_AT_10R=0.4:lock10R_0.4"

echo "Batch 3/4: lock10R 0.5..0.6 + lock07R 0.05..0.1"
launch_batch \
  "DRAGON_TRAIL_LOCK_AT_10R=0.5:lock10R_0.5" \
  "DRAGON_TRAIL_LOCK_AT_10R=0.6:lock10R_0.6" \
  "DRAGON_TRAIL_LOCK_AT_07R=0.05:lock07R_0.05" \
  "DRAGON_TRAIL_LOCK_AT_07R=0.1:lock07R_0.1"

echo "Batch 4/4: lock07R 0.2..0.4"
launch_batch \
  "DRAGON_TRAIL_LOCK_AT_07R=0.2:lock07R_0.2" \
  "DRAGON_TRAIL_LOCK_AT_07R=0.3:lock07R_0.3" \
  "DRAGON_TRAIL_LOCK_AT_07R=0.4:lock07R_0.4"

echo "ALL 15 SWEEP RUNS COMPLETE"
