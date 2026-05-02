#!/usr/bin/env python3 -B
"""
rescue_losers.py — extended sweep for the 6 symbols that failed pass-1 tuning.

Sweep grid per symbol (4 SL × 4 mQ × 3 force_dir = 48 combos × 6 symbols = 288 runs):
  sl_atr_mult     ∈ {2.5, 3.0, 3.5, 4.0}
  min_quality     ∈ {50, 55, 60, 65}      (applied to all 4 regimes)
  force_direction ∈ {LONG, SHORT, BOTH}

Score = pnl × pf / sqrt(max(dd, 1))

Pass threshold: PF ≥ 1.30 AND trades ≥ 30 AND DD ≤ 18%.

Outputs:
  backtest/results/rescue_losers.json           — full sweep result for every combo
  backtest/results/rescue_losers_auto_dict.py   — RESCUE_AUTO dict (only rescued symbols)

Does NOT touch config.py or auto_tuned.py.

Usage:
    python3 -B scripts/rescue_losers.py
"""
import sys
import json
import math
import time
from pathlib import Path
from itertools import product
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import after sys.path tweak — also forces v5_backtest to load with live config
from backtest.v5_backtest import backtest_symbol, SL_OVERRIDE, ALL_SYMBOLS

DAYS = 180
LOSERS = ["BCHUSD", "EURGBP", "EURNZD", "NZDJPY", "NZDUSD", "UKOUSD"]
SL_GRID = [2.5, 3.0, 3.5, 4.0]
MQ_GRID = [50, 55, 60, 65]
DIR_GRID = ["LONG", "SHORT", "BOTH"]

# Pass thresholds
PF_MIN = 1.30
TRADES_MIN = 30
DD_MAX = 18.0

OUT_DIR = ROOT / "backtest" / "results"
OUT_JSON = OUT_DIR / "rescue_losers.json"
OUT_DICT = OUT_DIR / "rescue_losers_auto_dict.py"


def score_fn(pnl, pf, dd):
    """Score = pnl × pf / sqrt(max(dd, 1)). Negative PnL → negative score."""
    return pnl * pf / math.sqrt(max(dd, 1.0))


def run_one(args):
    """Worker: run a single (symbol, sl, mq, fd) combo on 180d backtest."""
    symbol, sl, mq, fd = args

    # Mutate global SL_OVERRIDE for this worker's process (each worker is its own process).
    # Done unconditionally because we always want the sweep SL, not whatever live config has.
    SL_OVERRIDE[symbol] = sl

    params = {
        "sl_atr_mult": sl,
        "min_quality": {"trending": mq, "ranging": mq, "volatile": mq, "low_vol": mq},
        "force_direction": fd,
        # ratchet: keep pass-1 winning pair so we don't double-explode the grid
        "ratchet_1r": 0.2,
        "ratchet_2r": 0.5,
    }

    try:
        r = backtest_symbol(symbol, DAYS, params, verbose=False)
    except Exception as e:
        return {
            "symbol": symbol, "sl": sl, "mq": mq, "fd": fd,
            "error": f"{type(e).__name__}: {e}",
            "trades": 0, "pf": 0, "wr": 0, "pnl": 0, "dd": 0, "score": -1e9,
        }

    if r is None:
        return {
            "symbol": symbol, "sl": sl, "mq": mq, "fd": fd,
            "trades": 0, "pf": 0, "wr": 0, "pnl": 0, "dd": 0, "score": -1e9,
        }

    pf = r.get("pf", 0) or 0
    pnl = r.get("pnl", 0) or 0
    dd = r.get("dd", 0) or 0
    trades = r.get("trades", 0) or 0
    wr = r.get("wr", 0) or 0

    return {
        "symbol": symbol, "sl": sl, "mq": mq, "fd": fd,
        "trades": trades, "wins": r.get("wins", 0),
        "pf": pf, "wr": wr, "pnl": pnl, "dd": dd,
        "avg_r": r.get("avg_r", 0), "avg_peak_r": r.get("avg_peak_r", 0),
        "score": score_fn(pnl, pf, dd),
    }


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = [(s, sl, mq, fd) for s in LOSERS for sl, mq, fd in product(SL_GRID, MQ_GRID, DIR_GRID)]
    print(f"[rescue] {len(LOSERS)} symbols × {len(SL_GRID)*len(MQ_GRID)*len(DIR_GRID)} combos = {len(combos)} runs over {DAYS}d")
    print(f"[rescue] grid: SL={SL_GRID} mQ={MQ_GRID} fd={DIR_GRID}")
    print(f"[rescue] thresholds: PF>={PF_MIN}, trades>={TRADES_MIN}, DD<={DD_MAX}%")

    with Pool(6) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_one, combos), start=1):
            results.append(r)
            if i % 24 == 0 or i == len(combos):
                print(f"  [rescue] {i}/{len(combos)} ({(i/len(combos))*100:.0f}%)")

    # Group by symbol and find best per symbol
    by_sym = {}
    for r in results:
        by_sym.setdefault(r["symbol"], []).append(r)

    summary = {}
    rescued = {}
    print("\n" + "="*80)
    print("  RESCUE SWEEP RESULTS")
    print("="*80)
    for sym in LOSERS:
        rs = sorted(by_sym.get(sym, []), key=lambda x: x["score"], reverse=True)
        if not rs:
            print(f"  {sym}: NO RESULTS")
            continue
        best = rs[0]
        # Find best result that meets thresholds (best valid)
        valid = [r for r in rs if r["pf"] >= PF_MIN and r["trades"] >= TRADES_MIN and r["dd"] <= DD_MAX]
        valid_best = valid[0] if valid else None

        summary[sym] = {
            "best_by_score": best,
            "best_valid": valid_best,
            "valid_count": len(valid),
            "tested": len(rs),
        }

        if valid_best:
            rescued[sym] = {
                "sl_atr_mult": valid_best["sl"],
                "min_quality": valid_best["mq"],
                "direction": valid_best["fd"],
                "_pf": valid_best["pf"],
                "_trades": valid_best["trades"],
                "_dd": valid_best["dd"],
                "_pnl": valid_best["pnl"],
                "_wr": valid_best["wr"],
            }
            print(f"  {sym:8s} RESCUED: PF={valid_best['pf']:.2f} WR={valid_best['wr']:.1f}% "
                  f"DD={valid_best['dd']:.1f}% n={valid_best['trades']} "
                  f"PnL=${valid_best['pnl']:.2f} | SL={valid_best['sl']} mQ={valid_best['mq']} dir={valid_best['fd']}")
        else:
            print(f"  {sym:8s} DEAD   : best PF={best['pf']:.2f} WR={best['wr']:.1f}% "
                  f"DD={best['dd']:.1f}% n={best['trades']} "
                  f"PnL=${best['pnl']:.2f} | SL={best['sl']} mQ={best['mq']} dir={best['fd']} "
                  f"({len(valid)} valid combos)")

    elapsed = time.time() - t0
    print(f"\n[rescue] elapsed {elapsed:.1f}s · rescued {len(rescued)}/{len(LOSERS)}")

    # ── Write full sweep JSON ──
    payload = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "days": DAYS,
        "thresholds": {"pf_min": PF_MIN, "trades_min": TRADES_MIN, "dd_max": DD_MAX},
        "grid": {"sl": SL_GRID, "mq": MQ_GRID, "fd": DIR_GRID},
        "elapsed_s": round(elapsed, 1),
        "rescued": rescued,
        "summary": summary,
        "all_results": results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[rescue] wrote {OUT_JSON}")

    # ── Write RESCUE_AUTO dict for synthesis (only rescued symbols) ──
    lines = [
        "# AUTO-GENERATED by scripts/rescue_losers.py — DO NOT EDIT BY HAND.",
        f"# Captured {time.strftime('%Y-%m-%dT%H:%M:%S')} · {DAYS}d backtest",
        f"# Thresholds: PF>={PF_MIN}, trades>={TRADES_MIN}, DD<={DD_MAX}%",
        "",
        "RESCUE_AUTO = {",
    ]
    for sym, cfg in rescued.items():
        lines.append(
            f"    {sym!r}: {{'sl_atr_mult': {cfg['sl_atr_mult']}, "
            f"'min_quality': {cfg['min_quality']}, "
            f"'direction': {cfg['direction']!r}}},  "
            f"# PF={cfg['_pf']:.2f} n={cfg['_trades']} DD={cfg['_dd']:.1f}% PnL=${cfg['_pnl']:.2f}"
        )
    lines.append("}")
    lines.append("")
    OUT_DICT.write_text("\n".join(lines))
    print(f"[rescue] wrote {OUT_DICT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
