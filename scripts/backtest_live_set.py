#!/usr/bin/env python3 -B
"""
Run mirror backtest on live agent's exact 8-symbol set.

Why: live trades XAUUSD, XAGUSD, NAS100.r, SP500.r, GER40.r, USDCAD, EURUSD, GBPJPY
but post_fix_validation.json only validated 6 symbols (and 2 of those — BTCUSD,
JPN225ft — aren't even in the live SYMBOLS list). This script closes that gap.

Settings mirror live execution:
- ML meta-label gate ON (matches live veto threshold)
- RL trail multipliers from rl_learner.db (matches live executor)
- Same DEFAULT_PARAMS, SL_OVERRIDE, MIN_QUALITY as live config

Usage:
    python3 -B scripts/backtest_live_set.py --days 30
    python3 -B scripts/backtest_live_set.py --days 30 --no-ml-gate --no-rl-trail  # ablation
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol
from agent.rl_learner import RL_DB
from config import SYMBOLS as LIVE_SYMBOLS


def load_rl_adjustments():
    """Read trail multipliers directly from DB so symbols not in current SYMBOLS still
    surface (consistent with the v5_backtest --rl-trail path)."""
    by_sym = {}
    if not Path(RL_DB).exists():
        return by_sym
    conn = sqlite3.connect(str(RL_DB), timeout=5.0)
    try:
        for sym, lock_m, be_m, tight_m in conn.execute(
            "SELECT symbol, lock_threshold_mult, be_threshold_mult, trail_tightness_mult "
            "FROM trail_adjustments"
        ).fetchall():
            by_sym[sym] = {
                "lock_threshold_mult": float(lock_m),
                "be_threshold_mult": float(be_m),
                "trail_tightness_mult": float(tight_m),
            }
    finally:
        conn.close()
    return by_sym


def load_meta_models(symbols):
    try:
        from models.signal_model import SignalModel
        m = SignalModel()
        loaded = 0
        for s in symbols:
            try:
                m.load(s)
                if m.has_model(s):
                    loaded += 1
            except Exception:
                pass
        print(f"  [ML-GATE] {loaded}/{len(symbols)} meta-label models loaded")
        return m if loaded else None
    except Exception as e:
        print(f"  [ML-GATE] disabled — {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30,
                    help="Backtest window in days (30 = smallest common cache window)")
    ap.add_argument("--no-ml-gate", action="store_true")
    ap.add_argument("--no-rl-trail", action="store_true")
    ap.add_argument("--out", type=str, default="backtest/live_set_validation.json")
    args = ap.parse_args()

    symbols = list(LIVE_SYMBOLS.keys()) if isinstance(LIVE_SYMBOLS, dict) else list(LIVE_SYMBOLS)

    print("=" * 70)
    print(f"  MIRROR BACKTEST — live agent's {len(symbols)}-symbol set")
    print(f"  Symbols: {symbols}")
    print(f"  Period: {args.days}d")
    print(f"  ML gate: {'OFF' if args.no_ml_gate else 'ON'}")
    print(f"  RL trail: {'OFF' if args.no_rl_trail else 'ON'}")
    print("=" * 70 + "\n")

    meta_model = None if args.no_ml_gate else load_meta_models(symbols)
    rl_by_sym = {} if args.no_rl_trail else load_rl_adjustments()
    if rl_by_sym:
        print(f"  [RL-TRAIL] Adjustments for {len(rl_by_sym)} symbol(s):")
        for sym, adj in sorted(rl_by_sym.items()):
            if any(v != 1.0 for v in adj.values()):
                print(f"    {sym}: lock×{adj['lock_threshold_mult']:.2f} "
                      f"be×{adj['be_threshold_mult']:.2f} "
                      f"tight×{adj['trail_tightness_mult']:.2f}")
    print()

    per_symbol = {}
    total_pnl = 0.0
    total_trades = 0
    skipped = []

    for sym in symbols:
        params = {}
        if sym in rl_by_sym:
            params["rl_adj"] = rl_by_sym[sym]
        if meta_model is not None:
            params["_meta_model"] = meta_model
        r = backtest_symbol(sym, args.days, params or None, verbose=True)
        if not r:
            skipped.append(sym)
            continue
        per_symbol[sym] = {
            "trades": r["trades"],
            "pnl": round(r["pnl"], 2),
            "pf": round(r.get("pf", 0), 2),
            "wr_pct": round(r.get("wr", 0), 1),       # backtest_symbol returns wr in percent
            "dd_pct": round(r.get("dd", 0), 1),       # backtest_symbol returns dd in percent
            "avg_r": round(r.get("avg_r", 0), 2),
            "avg_peak_r": round(r.get("avg_peak_r", 0), 2),
            "avg_giveback": round(r.get("avg_giveback", 0), 2),
            "equity": round(r.get("equity", 0), 2),
        }
        total_pnl += r["pnl"]
        total_trades += r["trades"]

    print("\n" + "=" * 70)
    print("  RESULTS — sorted by PnL")
    print("=" * 70)
    print(f"  {'symbol':10s} {'trd':>4s} {'pf':>6s} {'wr':>6s} {'dd':>6s} {'pnl':>10s} {'avg_r':>7s} {'gb_r':>6s}  verdict")
    print("  " + "-" * 70)
    ranked = sorted(per_symbol.items(), key=lambda kv: -kv[1]["pnl"])
    decisions = {}
    for sym, s in ranked:
        # Verdict heuristic: PF >= 1.5 = keep, 1.2-1.5 = monitor, <1.2 = kill
        if s["pf"] >= 1.5:
            v = "KEEP"
        elif s["pf"] >= 1.2:
            v = "MONITOR"
        else:
            v = "KILL"
        decisions[sym] = v
        print(f"  {sym:10s} {s['trades']:4d} {s['pf']:6.2f} {s['wr_pct']:5.1f}% "
              f"{s['dd_pct']:5.1f}% ${s['pnl']:8.2f} {s['avg_r']:6.2f}R {s['avg_giveback']:5.2f}R  {v}")

    print("  " + "-" * 70)
    print(f"  TOTAL  trades={total_trades}  pnl=${total_pnl:.2f}")
    if skipped:
        print(f"\n  SKIPPED (no data or zero trades): {skipped}")

    out = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "config": {
            "days": args.days,
            "symbols": symbols,
            "ml_gate": not args.no_ml_gate,
            "rl_trail": not args.no_rl_trail,
        },
        "totals": {"trades": total_trades, "pnl": round(total_pnl, 2)},
        "per_symbol": per_symbol,
        "decisions": decisions,
        "skipped": skipped,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
