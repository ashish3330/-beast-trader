"""
Run the live trading universe (config.SYMBOLS) through v5_backtest at a given
lookback and dump structured per-symbol metrics to JSON. ML gate ON for live
parity; RL-neutral by default (no learned trail adj) so it reflects the
post-RL-reset starting point. Reused for baseline and post-tune validation.

Usage:
  python3 -B scripts/bt_live_universe.py --days 1095 --tag baseline_3yr
"""
import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOLS
from backtest.v5_backtest import backtest_symbol, ALL_SYMBOLS

RES = Path(__file__).resolve().parent.parent / "backtest" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1095)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--no-ml-gate", action="store_true")
    args = ap.parse_args()

    meta_model = None
    if not args.no_ml_gate:
        try:
            from models.signal_model import SignalModel
            meta_model = SignalModel()
            for s in SYMBOLS:
                try: meta_model.load(s)
                except Exception: pass
            loaded = sum(1 for s in SYMBOLS if meta_model.has_model(s))
            print(f"[ML-GATE] {loaded}/{len(SYMBOLS)} models loaded")
        except Exception as e:
            print(f"[ML-GATE] disabled: {e}")

    live = [s for s in SYMBOLS if s in ALL_SYMBOLS]
    missing = [s for s in SYMBOLS if s not in ALL_SYMBOLS]
    if missing:
        print(f"[WARN] not in ALL_SYMBOLS (skipped): {missing}")

    rows, tot_pnl, tot_trades = [], 0.0, 0
    for s in live:
        params = {}
        if meta_model is not None:
            params["_meta_model"] = meta_model
        r = backtest_symbol(s, args.days, params or None, verbose=False)
        if not r:
            print(f"  {s:12s} NO RESULT"); continue
        rows.append(r)
        tot_pnl += r["pnl"]; tot_trades += r["trades"]

    rows.sort(key=lambda r: r["pf"], reverse=True)
    print("\n" + "=" * 64)
    print(f"  3yr-H1 universe @ {args.days}d  [tag={args.tag}]")
    print("=" * 64)
    print(f"  {'SYM':12s} {'PF':>6} {'WR%':>6} {'DD%':>6} {'n':>5} {'PnL$':>12}")
    for r in rows:
        print(f"  {r['symbol']:12s} {r['pf']:>6.2f} {r['wr']:>6.1f} {r['dd']:>6.1f} "
              f"{r['trades']:>5d} {r['pnl']:>12.2f}")
    pfs = [r["pf"] for r in rows if r["trades"] >= 15]
    avg_pf = sum(pfs) / len(pfs) if pfs else 0
    print("-" * 64)
    print(f"  symbols={len(rows)}  trades={tot_trades}  avgPF(n>=15)={avg_pf:.2f}  PnL=${tot_pnl:.2f}")

    out = RES / f"bt_universe_{args.tag}.json"
    json.dump({"days": args.days, "tag": args.tag, "total_pnl": round(tot_pnl, 2),
               "total_trades": tot_trades, "avg_pf": round(avg_pf, 2),
               "rows": rows}, open(out, "w"), indent=2)
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
