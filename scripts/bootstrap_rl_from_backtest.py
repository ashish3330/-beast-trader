#!/usr/bin/env python3 -B
"""
Bootstrap rl_learner.db with trades from a 180d backtest.

Why: the demo's rl_learner.db is currently 0 bytes — RLLearner has never seen
a real trade outcome, so its trail multipliers are all defaults (1.0). This
script runs a 180d backtest, then feeds every trade through
RLLearner.record_outcome() so the learning loop fires (_maybe_update_weights,
_maybe_update_exits) and writes per-symbol trail adjustments to disk.

After running, the live agent (after restart) will pick up the bootstrapped
trail_adjustments table and start trading with learned multipliers instead of
the hardcoded TRAIL_STEPS table.

Usage:
    python3 -B scripts/bootstrap_rl_from_backtest.py --days 180
    python3 -B scripts/bootstrap_rl_from_backtest.py --days 180 --reset  # wipe DB first
"""
import sys, argparse, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, ALL_SYMBOLS
from agent.rl_learner import RLLearner, RL_DB


def reset_db():
    """Wipe rl_learner.db so we start clean."""
    if Path(RL_DB).exists():
        Path(RL_DB).unlink()
        print(f"  Deleted {RL_DB}")


def bootstrap(symbols, days=180):
    from types import SimpleNamespace
    rl = RLLearner(SimpleNamespace())  # creates schema if absent; state unused at this scope
    print(f"\n  Bootstrap RL from {days}d backtest — {len(symbols)} symbols\n")

    total_fed = 0
    per_sym_summary = []

    for sym in symbols:
        r = backtest_symbol(sym, days, verbose=True)
        if not r or r["trades"] == 0:
            continue

        # Feed each trade into RL
        for t in r["details"]:
            direction = "LONG" if t["direction"] == 1 else "SHORT"
            rl.record_outcome(
                symbol=sym,
                direction=direction,
                pnl=t["pnl"],
                r_multiple=t["pnl_r"],
                score=t["quality"],
                regime=t["regime"],
                exit_reason=t["exit_reason"],
                score_components=None,  # backtest doesn't track per-component
                peak_r=t["peak_r"],
            )
            total_fed += 1

        # Pull learned adjustments
        adj = rl.get_trail_adjustments(sym)
        per_sym_summary.append((sym, r["trades"], adj))

    print(f"\n  Fed {total_fed} trade outcomes to RL")

    # Show learned multipliers
    print(f"\n  Learned trail multipliers per symbol:")
    print(f"  {'symbol':12s} {'trades':>7s}  lock×    be×      tight×")
    for sym, n, adj in per_sym_summary:
        lock_m = adj.get("lock_threshold_mult", 1.0)
        be_m = adj.get("be_threshold_mult", 1.0)
        tight_m = adj.get("trail_tightness_mult", 1.0)
        marker = " *" if (lock_m != 1.0 or be_m != 1.0 or tight_m != 1.0) else ""
        print(f"  {sym:12s} {n:7d}  {lock_m:.3f}   {be_m:.3f}    {tight_m:.3f}{marker}")

    # Verify DB persistence
    conn = sqlite3.connect(str(RL_DB))
    n_outcomes = conn.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]
    n_trail = conn.execute("SELECT COUNT(*) FROM trail_adjustments").fetchone()[0]
    n_weights = conn.execute("SELECT COUNT(*) FROM score_weights").fetchone()[0]
    n_audit = conn.execute("SELECT COUNT(*) FROM rl_audit_log").fetchone()[0]
    conn.close()
    print(f"\n  Persisted to {RL_DB}:")
    print(f"    trade_outcomes:    {n_outcomes}")
    print(f"    trail_adjustments: {n_trail}")
    print(f"    score_weights:     {n_weights}")
    print(f"    rl_audit_log:      {n_audit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--reset", action="store_true", help="Wipe rl_learner.db before bootstrap")
    ap.add_argument("--all-symbols", action="store_true", help="Use full ALL_SYMBOLS list")
    args = ap.parse_args()

    if args.reset:
        reset_db()

    if args.all_symbols:
        symbols = list(ALL_SYMBOLS.keys())
    else:
        symbols = ["XAUUSD", "XAGUSD", "BTCUSD", "NAS100.r", "JPN225ft", "USDCAD"]

    bootstrap(symbols, days=args.days)


if __name__ == "__main__":
    main()
