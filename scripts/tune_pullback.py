"""
Per-symbol PULLBACK ENTRY-TIMING sweep on the live universe.

The backtest's pullback fill is now parameterized (pullback_atr_retrace,
pullback_max_wait). This sweeps retrace depth x wait window per symbol over a
long H1 window and picks the best combo under an anti-overfit rule:

  - candidate must keep >= max(20, TRADE_FLOOR_FRAC * baseline_trades) trades
  - candidate DD must be <= baseline_dd * DD_CAP_MULT
  - candidate PF must beat baseline PF by >= MIN_PF_GAIN_FRAC (material gain)
  - otherwise KEEP the global default (0.2, 1) — don't change what isn't better

Writes backtest/results/pullback_sweep_<tag>.json with per-symbol winner + grid.

Usage:
  python3 -B scripts/tune_pullback.py --days 1095 --tag 3yr --workers 5
"""
import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RETRACE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
WAIT_GRID = [1, 2, 3]
DEFAULT = (0.20, 1)

TRADE_FLOOR_FRAC = 0.60   # candidate must keep >=60% of baseline trades
DD_CAP_MULT = 1.50        # candidate DD <= 1.5x baseline DD
MIN_PF_GAIN_FRAC = 0.05   # require >=5% PF improvement to override default
RES = Path(__file__).resolve().parent.parent / "backtest" / "results"


def _sweep_one(symbol, days):
    """Run the full retrace x wait grid for one symbol. Returns dict or None."""
    from backtest.v5_backtest import backtest_symbol, ALL_SYMBOLS
    if symbol not in ALL_SYMBOLS:
        return symbol, None
    mm = None
    try:
        from models.signal_model import SignalModel
        mm = SignalModel(); mm.load(symbol)
        if not mm.has_model(symbol):
            mm = None
    except Exception:
        mm = None

    def run(retrace, wait):
        p = {"pullback_atr_retrace": retrace, "pullback_max_wait": wait}
        if mm is not None:
            p["_meta_model"] = mm
        r = backtest_symbol(symbol, days, p, verbose=False)
        return r

    base = run(*DEFAULT)
    if not base or base["trades"] == 0:
        return symbol, {"baseline": base, "grid": [], "winner": DEFAULT, "reason": "no-baseline-trades"}

    grid = []
    for rt in RETRACE_GRID:
        for w in WAIT_GRID:
            r = run(rt, w)
            if r:
                grid.append({"retrace": rt, "wait": w, "pf": r["pf"], "wr": r["wr"],
                             "dd": r["dd"], "trades": r["trades"], "pnl": r["pnl"]})

    base_pf, base_tr, base_dd = base["pf"], base["trades"], max(base["dd"], 0.1)
    trade_floor = max(20, int(TRADE_FLOOR_FRAC * base_tr))
    eligible = [g for g in grid
                if g["trades"] >= trade_floor and g["dd"] <= base_dd * DD_CAP_MULT]
    winner, reason = DEFAULT, "kept-default"
    if eligible:
        best = max(eligible, key=lambda g: (g["pf"], g["trades"]))
        if best["pf"] >= base_pf * (1.0 + MIN_PF_GAIN_FRAC) and (best["retrace"], best["wait"]) != DEFAULT:
            winner = (best["retrace"], best["wait"])
            reason = f"PF {base_pf:.2f}->{best['pf']:.2f}"
    return symbol, {
        "baseline": {"retrace": DEFAULT[0], "wait": DEFAULT[1], "pf": base_pf,
                     "wr": base["wr"], "dd": base["dd"], "trades": base_tr, "pnl": base["pnl"]},
        "grid": grid, "winner": list(winner), "reason": reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1095)
    ap.add_argument("--tag", default="3yr")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    from config import SYMBOLS
    syms = list(SYMBOLS.keys())
    print(f"Pullback sweep: {len(syms)} symbols x {len(RETRACE_GRID)*len(WAIT_GRID)} combos "
          f"@ {args.days}d, {args.workers} workers\n")

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_sweep_one, s, args.days): s for s in syms}
        for f in as_completed(futs):
            s, res = f.result()
            results[s] = res
            if res is None:
                print(f"  {s:12s} SKIP (not in ALL_SYMBOLS)"); continue
            b = res.get("baseline", {})
            w = res["winner"]
            print(f"  {s:12s} base PF={b.get('pf',0):.2f} n={b.get('trades',0):<4} "
                  f"-> winner retrace={w[0]} wait={w[1]}  [{res['reason']}]")

    out = RES / f"pullback_sweep_{args.tag}.json"
    json.dump({"days": args.days, "results": results}, open(out, "w"), indent=2)

    changed = {s: r["winner"] for s, r in results.items()
               if r and tuple(r["winner"]) != DEFAULT}
    print("\n" + "=" * 60)
    print(f"  CHANGED ({len(changed)}/{len(syms)}):")
    for s, w in changed.items():
        print(f"    {s:12s} retrace={w[0]} wait={w[1]}  [{results[s]['reason']}]")
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
