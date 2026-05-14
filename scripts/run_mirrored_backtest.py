#!/usr/bin/env python3 -B
"""
Run mirrored backtest with audit-fix gates ON.

Mirrors what live brain does at entry: MIN_EDGE filter, EV-gate,
A+ bypass (quality >= 75% skips both), proven-edge whitelist with
relaxed EV threshold (-0.30R vs +0.10R for unknowns).

Usage:
    python3 -B scripts/run_mirrored_backtest.py [--days 180] [--symbols SYM1,SYM2,...]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS, ALL_SYMBOLS  # noqa: E402

OUT_DIR = ROOT / "backtest" / "results" / "mirrored_audit_fix_2026_05_13"


def run_one(args):
    symbol, days = args
    params = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
    }
    # 2026-05-14: mirror live brain by reading auto_tuned per-symbol filters
    try:
        import auto_tuned as _at  # type: ignore
        # Phase 5 range filter
        rfp = getattr(_at, "RANGE_FILTER_PARAMS_AUTO", {}).get(symbol)
        if rfp:
            params["range_filter_enabled"] = True
            params["range_lookback"] = rfp.get("lookback", 48)
            params["range_buffer_atr"] = rfp.get("buffer_atr", 0.5)
        # Phase 6 fib filter
        fp = getattr(_at, "FIB_PARAMS_AUTO", {}).get(symbol)
        if fp:
            params["fib_enabled"] = True
            params["fib_swing_lookback"] = fp.get("lookback", 50)
            params["fib_zone_lo"] = fp.get("zone_lo", 0.5)
            params["fib_zone_hi"] = fp.get("zone_hi", 0.618)
            params["fib_as_filter"] = fp.get("as_filter", True)
    except Exception:
        pass
    t0 = time.time()
    try:
        r = backtest_symbol(symbol, days=days, params=params, verbose=False)
        if r:
            r["wall_time_s"] = round(time.time() - t0, 1)
            return symbol, r
    except Exception as e:
        return symbol, {"error": str(e)}
    return symbol, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--symbols", default=None, help="comma-separated")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = list(ALL_SYMBOLS.keys())

    print(f"\n{'='*70}")
    print(f"  MIRRORED BACKTEST (audit-fix gates ON)")
    print(f"  Days: {args.days} | Symbols: {len(symbols)} | Workers: {args.workers}")
    print(f"{'='*70}\n")

    with Pool(args.workers) as pool:
        results = pool.map(run_one, [(s, args.days) for s in symbols])

    # Aggregate
    rows = []
    total_trades = 0
    total_pnl = 0.0
    for sym, r in results:
        if r and r.get("trades", 0) > 0:
            rows.append(r)
            total_trades += r["trades"]
            total_pnl += r.get("pnl", 0)

    rows.sort(key=lambda x: -x.get("pnl", 0))

    print(f"\n{'SYM':<12} {'N':>4} {'WR':>5} {'PF':>6} {'PnL':>9} {'avgR':>6} {'DD':>5}")
    print("-" * 60)
    for r in rows:
        pf_str = f"{r['pf']:.2f}" if r["pf"] < 99 else "  ∞"
        print(f"{r['symbol']:<12} {r['trades']:>4} {r['wr']:>4.0f}% {pf_str:>6} "
              f"${r.get('pnl', 0):>+8.2f} {r.get('avg_r', 0):>+5.2f} {r.get('dd', 0):>4.1f}%")

    print(f"\n  TOTAL: {total_trades} trades | PnL ${total_pnl:+.2f}\n")

    # Save
    out = {
        "days": args.days,
        "audit_fix_gates": True,
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "results": {r["symbol"]: r for r in rows},
    }
    out_file = OUT_DIR / f"mirrored_{args.days}d.json"
    out_file.write_text(json.dumps(out, indent=2, default=str))
    print(f"  Saved: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
