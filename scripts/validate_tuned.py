"""
Re-run --all-symbols 180d backtest with current config (auto_tuned applied)
and compare to the prior baseline. Reports per-symbol delta PnL and total lift.

Outputs:
  backtest/results/validate_<days>d_<pass>.json
"""
import sys, json, os, time, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import ALL_SYMBOLS, backtest_symbol

DAYS = int(os.environ.get("TUNE_DAYS", "180"))
PASS = os.environ.get("TUNE_PASS", "pass1")

RESULTS = ROOT / "backtest" / "results"
TUNE_PATH = RESULTS / f"tune_{DAYS}d_{PASS}.json"
OUT_PATH = RESULTS / f"validate_{DAYS}d_{PASS}.json"


def _slim(r):
    if not r: return r
    return {k: v for k, v in r.items() if k != "details"}


def main():
    base_pnl = {}
    if TUNE_PATH.exists():
        tdata = json.load(open(TUNE_PATH))
        for sym, r in tdata.get("results", {}).items():
            b = r.get("baseline") or {}
            base_pnl[sym] = b.get("pnl", 0.0)

    print(f"Validating {len(ALL_SYMBOLS)} symbols × {DAYS}d with auto_tuned applied")
    print(f"{'SYM':<14} {'PF':>5} {'WR':>5} {'TR':>5} {'PnL':>9} {'DD':>5}  {'BASE':>9}  {'Δ':>8}")
    print("-" * 78)

    out = {}
    total_pnl = 0.0
    total_lift = 0.0
    t0 = time.time()
    for sym in sorted(ALL_SYMBOLS.keys()):
        r = backtest_symbol(sym, DAYS, None, verbose=False)
        if not r: continue
        out[sym] = _slim(r)
        bp = base_pnl.get(sym, 0.0)
        delta = r["pnl"] - bp
        tag = "↑" if delta > 5 else ("↓" if delta < -5 else "·")
        print(f"{sym:<14} {r['pf']:>5.2f} {r['wr']:>5.1f} {r['trades']:>5} "
              f"${r['pnl']:>8.0f} {r['dd']:>4.1f}%  ${bp:>8.0f}  {delta:>+7.0f} {tag}")
        total_pnl += r["pnl"]
        total_lift += delta

    elapsed = time.time() - t0
    print("-" * 78)
    print(f"  TOTAL PnL ${total_pnl:.0f}   Δ vs baseline ${total_lift:+.0f}   ({elapsed:.0f}s)")

    json.dump({
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "days": DAYS,
        "pass": PASS,
        "elapsed_s": round(elapsed, 1),
        "total_pnl": round(total_pnl, 2),
        "total_lift_vs_baseline": round(total_lift, 2),
        "per_symbol": out,
    }, open(OUT_PATH, "w"), indent=2, default=str)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
