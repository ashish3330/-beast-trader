#!/usr/bin/env python3
"""Weekly walk-forward re-validation — catches per-symbol param decay.

Runs k-fold walk-forward (5 folds, 7d embargo, 540d span) on the candidate
universe, then compares the verdict for each LIVE symbol vs the prior week's
result. Alerts on:
  - Any live symbol newly classified OVERFIT (was ROBUST/WEAK last week)
  - Live aggregate test_pnl_total dropped >25% week-over-week
  - Any live symbol's test_pf_mean dropped below 1.0 (was >=1.5 last week)

Output:
  logs/weekly_walkforward.log            — append-only history
  logs/weekly_walkforward_latest.json    — latest snapshot
  backtest/results/walk_forward.json     — overwritten by underlying script

Schedule: Sunday 06:30 UTC (after weekly_retrain.sh which finishes ~05:35).

Why this matters: per feedback_strategy_changes.md, per-symbol params drift —
v4→v6.1 went 60%→35% WR over 2-3 months. Catching that before a $1K bleed
session costs nothing relative to the loss it prevents.
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
WF_SCRIPT = ROOT / "scripts" / "walk_forward.py"
WF_OUTPUT = ROOT / "backtest" / "results" / "walk_forward.json"
LOG_DIR = ROOT / "logs"
SUMMARY_LOG = LOG_DIR / "weekly_walkforward.log"
LATEST_JSON = LOG_DIR / "weekly_walkforward_latest.json"

# Verdict thresholds (must match walk_forward.py logic)
ROBUST_PF = 1.5
OVERFIT_PF = 1.0


def _setup_log():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("dragon.wf")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.FileHandler(SUMMARY_LOG, mode="a")
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(h)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        log.addHandler(sh)
    return log


def _verdict(test_pf, test_pnl, test_trades):
    if not test_trades or test_trades < 50:
        return "TINY"
    if test_pf is None:
        return "TINY"
    if test_pf >= ROBUST_PF and test_pnl > 0:
        return "ROBUST"
    if test_pf >= OVERFIT_PF and test_pnl > 0:
        return "WEAK"
    return "OVERFIT"


def _live_universe():
    sys.path.insert(0, str(ROOT))
    import config
    return set(config.SYMBOLS.keys())


def _aggregate_for(payload, sym):
    res = payload.get("results", {}).get(sym)
    if not res:
        return None
    agg = res.get("aggregate", {})
    return {
        "test_pf": agg.get("test_pf_mean", 0),
        "test_pnl": agg.get("test_pnl_total", 0),
        "test_trades": agg.get("test_trades_total", 0),
    }


def main() -> int:
    log = _setup_log()
    log.info("=== weekly walk-forward starting ===")

    # Snapshot prior result before overwrite
    prior = None
    if LATEST_JSON.exists():
        try:
            prior = json.loads(LATEST_JSON.read_text())
        except Exception as e:
            log.warning("could not parse prior latest: %s", e)
            prior = None

    # Run k-fold walk-forward
    cmd = [sys.executable, "-B", str(WF_SCRIPT),
           "--kfold", "5", "--embargo-days", "7", "--top", "60", "--total-days", "540"]
    log.info("running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                                cwd=str(ROOT), env={"WF_WORKERS": "6", "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"})
    except subprocess.TimeoutExpired:
        log.error("walk_forward.py timed out after 900s")
        return 1
    if result.returncode != 0:
        log.error("walk_forward.py failed (rc=%d): %s",
                  result.returncode, (result.stderr or "")[-500:])
        return 1
    if not WF_OUTPUT.exists():
        log.error("walk_forward.json missing after run")
        return 1

    payload = json.loads(WF_OUTPUT.read_text())
    LIVE = _live_universe()

    # Compute current verdicts
    current = {}
    agg_pnl_total = 0
    for sym in LIVE:
        a = _aggregate_for(payload, sym)
        if a is None:
            current[sym] = {"verdict": "MISSING", "test_pf": 0, "test_pnl": 0, "test_trades": 0}
            continue
        v = _verdict(a["test_pf"], a["test_pnl"], a["test_trades"])
        current[sym] = {"verdict": v, **a}
        agg_pnl_total += a["test_pnl"]

    # Compare to prior week
    alerts = []
    deltas = []
    if prior:
        prior_results = prior.get("per_symbol", {})
        prior_agg = prior.get("aggregate_pnl_total", 0)
        for sym, c in current.items():
            p = prior_results.get(sym)
            if not p:
                continue
            # Verdict shift detection
            if c["verdict"] == "OVERFIT" and p["verdict"] in ("ROBUST", "WEAK"):
                alerts.append(f"DECAY: {sym} verdict {p['verdict']}→OVERFIT (test_pf "
                              f"{p['test_pf']:.2f}→{c['test_pf']:.2f})")
            elif c["test_pf"] < OVERFIT_PF and p["test_pf"] >= ROBUST_PF:
                alerts.append(f"DECAY: {sym} test_pf {p['test_pf']:.2f}→{c['test_pf']:.2f} "
                              f"(crossed below {OVERFIT_PF})")
            # Per-symbol PnL change
            d = c["test_pnl"] - p["test_pnl"]
            deltas.append((sym, d, p["test_pnl"], c["test_pnl"]))

        # Aggregate change
        if prior_agg > 0:
            agg_delta_pct = (agg_pnl_total - prior_agg) / prior_agg * 100
            if agg_delta_pct < -25:
                alerts.append(f"DECAY: aggregate test_pnl ${prior_agg:,.0f}→${agg_pnl_total:,.0f} "
                              f"({agg_delta_pct:+.1f}%) — universe-wide decay")

    # Counts
    counts = {"ROBUST": 0, "WEAK": 0, "OVERFIT": 0, "TINY": 0, "MISSING": 0}
    for c in current.values():
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1

    # Log results
    log.info("Live universe (%d symbols): ROBUST=%d WEAK=%d OVERFIT=%d TINY=%d MISSING=%d",
             len(LIVE), counts["ROBUST"], counts["WEAK"], counts["OVERFIT"],
             counts["TINY"], counts["MISSING"])
    log.info("Aggregate test_pnl: $%.0f", agg_pnl_total)

    for a in alerts:
        log.warning(a)

    # Top decaying symbols (worst PnL deltas)
    if deltas:
        deltas.sort(key=lambda x: x[1])  # worst first
        worst = [d for d in deltas[:5] if d[1] < -200]  # only flag >$200 decay
        if worst:
            log.warning("Top decaying: %s",
                        ", ".join(f"{s}({pp:.0f}→{cc:.0f}, Δ${dd:+.0f})"
                                  for s, dd, pp, cc in worst))

    # Persist snapshot for next-week comparison
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "live_universe_count": len(LIVE),
        "verdict_counts": counts,
        "aggregate_pnl_total": agg_pnl_total,
        "alerts": alerts,
        "per_symbol": current,
    }
    LATEST_JSON.write_text(json.dumps(snapshot, indent=2))

    log.info("=== weekly walk-forward done (alerts: %d) ===", len(alerts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
