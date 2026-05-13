#!/usr/bin/env python3
"""Daily live-vs-backtest divergence monitor.

Runs scripts/live_vs_backtest.py, parses output, logs WARNINGs for divergence
on live-universe symbols only. Writes timestamped status to logs/live_drift.log.

Trigger reasons (logs WARNING):
  - aggregate live $/day < 50% of backtest $/day expectation (live underperforming)
  - aggregate live $/day > 200% of backtest $/day (live too good — possible reporting bug)
  - per-symbol divergence_flag=YES for any live-universe symbol with >=10 trades

Output:
  logs/live_drift.log     — append-only status with timestamp
  logs/live_drift.json    — latest snapshot (overwritten each run)

Schedule via launchd plist com.dragon.live-drift.plist daily at 04:30 UTC.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
LIVE_VS_BT_SCRIPT = ROOT / "scripts" / "live_vs_backtest.py"
RESULTS_JSON = ROOT / "backtest" / "results" / "live_vs_backtest.json"
LOG_DIR = ROOT / "logs"
DRIFT_LOG = LOG_DIR / "live_drift.log"
DRIFT_JSON = LOG_DIR / "live_drift.json"

# Per-symbol thresholds — only flag if at least this many live trades observed
MIN_TRADES_FOR_FLAG = 10
# Aggregate thresholds (ratio = live_per_day / backtest_per_day)
AGG_UNDERPERFORM_RATIO = 0.5   # live < 50% backtest = WARN
AGG_OVERPERFORM_RATIO = 2.0    # live > 200% backtest = WARN (reporting bug?)


def _setup_log() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("dragon.drift")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.FileHandler(DRIFT_LOG, mode="a")
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(h)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        log.addHandler(sh)
    return log


def _live_universe() -> set:
    sys.path.insert(0, str(ROOT))
    import config
    return set(config.SYMBOLS.keys())


def main() -> int:
    log = _setup_log()
    log.info("=== live-drift check starting ===")

    # Run the comparison
    try:
        result = subprocess.run(
            [sys.executable, "-B", str(LIVE_VS_BT_SCRIPT)],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        log.error("live_vs_backtest.py timed out after 120s")
        return 1
    if result.returncode != 0:
        log.error("live_vs_backtest.py failed (rc=%d): %s", result.returncode, result.stderr[-500:])
        return 1

    if not RESULTS_JSON.exists():
        log.error("live_vs_backtest.json not produced")
        return 1

    payload = json.loads(RESULTS_JSON.read_text())
    rows = payload.get("rows", [])
    totals = payload.get("totals", {})
    live_trades_30d = totals.get("live_trades_30d", 0)
    live_pnl_30d = totals.get("live_pnl_30d", 0.0)

    LIVE = _live_universe()
    live_rows = [r for r in rows if r["sym"] in LIVE]

    # Aggregate live vs backtest expectation across the live universe
    live_per_day_total = sum(r.get("live_per_day", 0) or 0 for r in live_rows)
    bt_per_day_total = sum(r.get("backtest_per_day", 0) or 0 for r in live_rows)
    agg_ratio = (live_per_day_total / bt_per_day_total) if bt_per_day_total > 0 else None

    flags = []
    # Aggregate flag
    if live_trades_30d < 5:
        log.info("Aggregate: live_trades_30d=%d <5, too few to compare meaningfully", live_trades_30d)
    elif agg_ratio is None:
        log.info("Aggregate: backtest baseline 0 — no comparison possible")
    elif agg_ratio < AGG_UNDERPERFORM_RATIO:
        msg = (f"AGG UNDERPERFORM: live ${live_per_day_total:.2f}/day vs backtest "
               f"${bt_per_day_total:.2f}/day (ratio {agg_ratio:.2%}) — live materially below backtest")
        log.warning(msg); flags.append(msg)
    elif agg_ratio > AGG_OVERPERFORM_RATIO:
        msg = (f"AGG OVERPERFORM: live ${live_per_day_total:.2f}/day vs backtest "
               f"${bt_per_day_total:.2f}/day (ratio {agg_ratio:.2%}) — verify reporting")
        log.warning(msg); flags.append(msg)
    else:
        log.info("Aggregate: live $%.2f/day vs backtest $%.2f/day (ratio %.0f%%) — within tolerance",
                 live_per_day_total, bt_per_day_total, (agg_ratio or 0) * 100)

    # Per-symbol flags (only with enough data)
    per_sym_flags = []
    for r in live_rows:
        if r["live_30d_trades"] >= MIN_TRADES_FOR_FLAG and r["divergence_flag"]:
            sym = r["sym"]
            ratio = r.get("ratio")
            ratio_str = f"{ratio:.2f}" if ratio is not None else "inf"
            msg = (f"SYMBOL DIVERGE: {sym} live ${r['live_30d_pnl']:.0f}/30d vs "
                   f"backtest ratio={ratio_str} (n_live={r['live_30d_trades']})")
            log.warning(msg)
            per_sym_flags.append({"sym": sym, "ratio": ratio, "live_pnl": r["live_30d_pnl"]})

    # Summary
    log.info("Summary: live_trades_30d=%d live_pnl_30d=$%.2f live_universe=%d "
             "aggregate_flag=%s symbol_flags=%d",
             live_trades_30d, live_pnl_30d, len(LIVE),
             "YES" if flags else "no", len(per_sym_flags))

    # Snapshot
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "live_universe_count": len(LIVE),
        "live_trades_30d": live_trades_30d,
        "live_pnl_30d": live_pnl_30d,
        "agg_ratio": agg_ratio,
        "agg_live_per_day": live_per_day_total,
        "agg_backtest_per_day": bt_per_day_total,
        "aggregate_warnings": flags,
        "symbol_warnings": per_sym_flags,
    }
    DRIFT_JSON.write_text(json.dumps(snapshot, indent=2))
    log.info("=== live-drift check done (warnings: agg=%d sym=%d) ===",
             len(flags), len(per_sym_flags))
    return 0


if __name__ == "__main__":
    sys.exit(main())
