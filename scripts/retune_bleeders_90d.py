"""
Targeted 90d retune for symbols that bled in the 90d validation despite
180d-tuned auto_tuned params. Per-symbol exhaustive grid:

  SL_ATR     0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0
  mQ_t       35, 37, 40, 45, 50, 55, 60
  mQ_r       40, 45, 50, 55
  ratchet    (0.2,0.5), (0.3,0.6), (0.3,0.7), (0.4,0.7), (0.4,0.8), (0.5,0.9)
  direction  LONG, SHORT, BOTH                 [via DIRECTION_BIAS override]

Score = pnl * pf / sqrt(dd) with constraints: trades >= 20, dd <= 12, pf >= 1.10.
Best per symbol written to backtest/results/retune_bleeders_90d.json + auto-patched
into auto_tuned.py.
"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, SL_OVERRIDE
import config

SYMBOLS = ["ETHUSD", "GBPUSD", "EURCHF", "USDCAD"]
DAYS = 90
WORKERS = 4

# Only knobs that actually export to live via auto_tuned.py.
# Ratchets live in TRAIL_OVERRIDE (per-symbol trail profile) — varying them
# in-process does NOT transfer to live. Fix at default (0.2,0.5) so the grid
# reflects what's actually deployable.
SL_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
MQ_T = [35, 37, 40, 45, 50, 55, 60]
MQ_R = [40, 45, 50, 55]
RATCHETS = [(0.2, 0.5)]   # fixed — not exportable
DIRECTIONS = ["LONG", "SHORT", "BOTH"]

# Hard constraints — fail-fast filters
MIN_TRADES = 20
MAX_DD = 12.0
MIN_PF = 1.10


def _slim(r):
    if not r: return r
    return {k: v for k, v in r.items() if k != "details"}


def _bt_one(args):
    sym, sl, t, r_mq, r1, r2, dirn = args
    # Override DIRECTION_BIAS for this run
    old_dir = config.DIRECTION_BIAS.get(sym)
    if dirn == "BOTH":
        config.DIRECTION_BIAS.pop(sym, None)
    else:
        config.DIRECTION_BIAS[sym] = dirn
    # Override SL
    old_sl = SL_OVERRIDE.get(sym)
    SL_OVERRIDE[sym] = sl
    try:
        params = {
            "min_quality": {"trending": t, "ranging": r_mq, "volatile": t, "low_vol": t},
            "sl_atr_mult": sl,
            "ratchet_1r": r1,
            "ratchet_2r": r2,
        }
        r = backtest_symbol(sym, DAYS, params, verbose=False)
    finally:
        if old_sl is not None: SL_OVERRIDE[sym] = old_sl
        else: SL_OVERRIDE.pop(sym, None)
        if old_dir is not None: config.DIRECTION_BIAS[sym] = old_dir
        elif sym in config.DIRECTION_BIAS: config.DIRECTION_BIAS.pop(sym, None)

    if not r: return None
    return {
        "sym": sym, "sl": sl, "mq_t": t, "mq_r": r_mq, "r1": r1, "r2": r2, "dir": dirn,
        "pf": r.get("pf", 0), "pnl": r.get("pnl", 0), "dd": r.get("dd", 100),
        "trades": r.get("trades", 0), "wr": r.get("wr_pct", 0),
    }


def main():
    grid = [(s, sl, t, r_mq, r1, r2, d)
            for s in SYMBOLS
            for sl in SL_GRID
            for t in MQ_T
            for r_mq in MQ_R
            for (r1, r2) in RATCHETS
            for d in DIRECTIONS]

    print(f"Sweeping {len(SYMBOLS)} symbols × {len(grid)//len(SYMBOLS)} combos = {len(grid)} backtests, {WORKERS} workers, {DAYS}d window")
    t0 = time.time()
    results = []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_bt_one, grid), 1):
            if r and r["trades"] >= MIN_TRADES and r["dd"] <= MAX_DD and r["pf"] >= MIN_PF:
                results.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(grid)}  ({(time.time()-t0)/60:.1f} min)")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min  |  {len(results)} viable / {len(grid)} tested")

    best_per_sym = {}
    for r in results:
        s = r["sym"]
        score = r["pnl"] * r["pf"] / max(r["dd"], 1) ** 0.5
        if s not in best_per_sym or score > best_per_sym[s]["score"]:
            best_per_sym[s] = {**r, "score": score}

    print("\n--- BEST PARAMS (90d, profitable + DD ≤ 12%) ---")
    for s in SYMBOLS:
        b = best_per_sym.get(s)
        if not b:
            print(f"  {s:<10} NO VIABLE — falling back to defensive (SL=4.0, mQ=60)")
            best_per_sym[s] = {
                "sym": s, "sl": 4.0, "mq_t": 60, "mq_r": 60,
                "r1": 0.4, "r2": 0.8, "dir": "BOTH",
                "pf": 0, "pnl": 0, "dd": 0, "trades": 0, "wr": 0,
                "score": -1, "fallback": True,
            }
            continue
        tag = " ★" if b["pnl"] > 100 else ""
        print(f"  {s:<10} sl={b['sl']:>4}  mQ={b['mq_t']}/{b['mq_r']}  r={b['r1']}/{b['r2']}  dir={b['dir']:<5}  PF {b['pf']:.2f}  PnL ${b['pnl']:>5.0f}  WR {b['wr']:.1f}%  DD {b['dd']:.1f}%  n={b['trades']}{tag}")

    out = ROOT / "backtest" / "results" / "retune_bleeders_90d.json"
    json.dump({"days": DAYS, "best": best_per_sym, "all_viable": results}, open(out, "w"), indent=2, default=str)
    print(f"\nWrote {out}")
    return best_per_sym


if __name__ == "__main__":
    main()
