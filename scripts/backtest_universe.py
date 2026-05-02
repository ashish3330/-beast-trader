#!/usr/bin/env python3 -B
"""
Backtest every symbol with usable cached data.

Goal: discover which Vantage symbols actually have edge under current post-fix
config (1.5-2.0x ATR SL, ML-gate ON when model exists, RL-trail ON, MIN_SCORE
6.0). Output ranked list to drive the SYMBOLS = {...} decision.

Two passes:
  PASS 1 (default params): all 18+ symbols with H1 data >= 60d, current global
    SL multiplier (1.5x, or per-symbol override if exists), 90d window. Filter
    to PF >= 1.2.

  PASS 2 (lightweight tune): for each PASS-1 winner, run backtest_symbol with
    a small param sweep over (sl_atr_mult ∈ {1.5, 1.8, 2.0, 2.5}). Pick best
    PF. Cap at 90d.

Output: backtest/universe_validation.json + console summary.

Usage:
    python3 -B scripts/backtest_universe.py --days 90
    python3 -B scripts/backtest_universe.py --days 90 --tune
"""
import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, ALL_SYMBOLS, SL_OVERRIDE  # noqa: E402
from agent.rl_learner import RL_DB  # noqa: E402

# Symbols to attempt (only those with cache files we know about + 60+ days)
UNIVERSE = [
    "AUDJPY", "AUDUSD", "EURAUD", "EURCHF", "EURGBP", "EURJPY", "EURUSD",
    "GBPJPY", "GBPUSD", "GER40.r", "NZDUSD", "SP500.r", "UK100.r", "USDCAD",
    "USDCHF", "USDJPY",
    # Truncated/short windows — will return thin or skip
    "XAUUSD", "XAGUSD", "NAS100.r",
    # Crypto
    "BTCUSD", "ETHUSD",
    # Commodities
    "COPPER-CR", "UKOUSD",
]


def load_rl_adjustments():
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
        return m if loaded else None, loaded
    except Exception:
        return None, 0


def run_pass1(symbols, days, meta_model, rl_by_sym):
    print("=" * 80)
    print(f"  PASS 1 — default params, {days}d window, {len(symbols)} symbols")
    print("=" * 80)
    print(f"  {'symbol':12s} {'trd':>4s} {'pf':>6s} {'wr':>6s} {'dd':>6s} {'pnl':>10s} "
          f"{'avg_r':>7s} {'gb_r':>6s}  verdict")
    print("  " + "-" * 75)

    results = {}
    for sym in symbols:
        if sym not in ALL_SYMBOLS:
            continue
        params = {}
        if meta_model is not None:
            params["_meta_model"] = meta_model
        if sym in rl_by_sym:
            params["rl_adj"] = rl_by_sym[sym]
        try:
            r = backtest_symbol(sym, days, params or None, verbose=False)
        except Exception as e:
            print(f"  {sym:12s}  ERROR: {e}")
            continue
        if not r or r.get("trades", 0) < 5:
            continue
        pf = r.get("pf", 0)
        verdict = "STRONG" if pf >= 2.0 else ("KEEP" if pf >= 1.5 else
                                              ("MARGINAL" if pf >= 1.2 else "DROP"))
        print(f"  {sym:12s} {r['trades']:4d} {pf:6.2f} {r.get('wr', 0):5.1f}% "
              f"{r.get('dd', 0):5.1f}% ${r['pnl']:8.2f} {r.get('avg_r', 0):6.2f}R "
              f"{r.get('avg_giveback', 0):5.2f}R  {verdict}")
        results[sym] = {
            "trades": r["trades"],
            "pf": round(pf, 2),
            "wr_pct": round(r.get("wr", 0), 1),
            "dd_pct": round(r.get("dd", 0), 1),
            "pnl": round(r["pnl"], 2),
            "avg_r": round(r.get("avg_r", 0), 2),
            "avg_giveback": round(r.get("avg_giveback", 0), 2),
            "verdict": verdict,
            "sl_mult": SL_OVERRIDE.get(sym, 1.5),
        }
    return results


def run_pass2_tune(winners, days, meta_model, rl_by_sym):
    """Lightweight SL sweep on PASS-1 winners."""
    print()
    print("=" * 80)
    print(f"  PASS 2 — SL multiplier sweep on {len(winners)} winner(s)")
    print("=" * 80)
    print(f"  {'symbol':12s}  {'sweep results (sl_mult: pf)':<60s}  best")
    print("  " + "-" * 80)

    sweep_mults = [1.2, 1.5, 1.8, 2.0, 2.5]
    out = {}
    for sym in winners:
        sym_results = {}
        for m in sweep_mults:
            params = {"sl_atr_mult": m}
            if meta_model is not None:
                params["_meta_model"] = meta_model
            if sym in rl_by_sym:
                params["rl_adj"] = rl_by_sym[sym]
            try:
                r = backtest_symbol(sym, days, params, verbose=False)
            except Exception:
                continue
            if not r or r.get("trades", 0) < 5:
                continue
            sym_results[m] = (r["pf"], r["pnl"], r["trades"])
        if not sym_results:
            print(f"  {sym:12s}  no results")
            continue
        best_m, best_data = max(sym_results.items(), key=lambda kv: kv[1][0])
        results_str = " | ".join(f"{m:.1f}:{d[0]:.2f}" for m, d in sorted(sym_results.items()))
        print(f"  {sym:12s}  {results_str:60s}  ★ {best_m:.1f} (PF {best_data[0]:.2f})")
        out[sym] = {"sweep": {str(m): {"pf": d[0], "pnl": d[1], "trades": d[2]}
                              for m, d in sym_results.items()},
                    "best_sl_mult": best_m,
                    "best_pf": best_data[0],
                    "best_pnl": best_data[1]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--tune", action="store_true",
                    help="Run PASS-2 SL sweep on PASS-1 winners")
    ap.add_argument("--out", type=str, default="backtest/universe_validation.json")
    args = ap.parse_args()

    universe = [s for s in UNIVERSE if s in ALL_SYMBOLS]
    print(f"Universe: {len(universe)} symbols")
    print(f"  {universe}")
    print()

    meta_model, n_models = load_meta_models(universe)
    print(f"  ML models loaded: {n_models}/{len(universe)}")
    rl_by_sym = load_rl_adjustments()
    print(f"  RL trail adjustments: {len(rl_by_sym)} symbol(s)")
    print()

    t0 = time.time()
    pass1 = run_pass1(universe, args.days, meta_model, rl_by_sym)
    pass1_secs = time.time() - t0

    pass2 = {}
    if args.tune:
        winners = [s for s, r in pass1.items() if r["pf"] >= 1.2]
        if winners:
            t1 = time.time()
            pass2 = run_pass2_tune(winners, args.days, meta_model, rl_by_sym)
            pass2_secs = time.time() - t1

    out = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"days": args.days, "tune": args.tune,
                   "n_universe": len(universe), "n_ml_models": n_models},
        "pass1": pass1,
        "pass2": pass2,
        "summary": {
            "strong": sorted([s for s, r in pass1.items() if r["verdict"] == "STRONG"],
                             key=lambda s: -pass1[s]["pf"]),
            "keep":   sorted([s for s, r in pass1.items() if r["verdict"] == "KEEP"],
                             key=lambda s: -pass1[s]["pf"]),
            "marginal": [s for s, r in pass1.items() if r["verdict"] == "MARGINAL"],
            "drop":   [s for s, r in pass1.items() if r["verdict"] == "DROP"],
            "pass1_seconds": round(pass1_secs, 1),
        },
    }

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  STRONG (PF≥2.0): {out['summary']['strong']}")
    print(f"  KEEP   (PF≥1.5): {out['summary']['keep']}")
    print(f"  MARGIN (PF≥1.2): {out['summary']['marginal']}")
    print(f"  DROP            : {out['summary']['drop']}")
    print(f"  Pass-1 wall time: {pass1_secs:.1f}s")

    out_path = ROOT / args.out
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
