"""
Spread stress test for tuned symbols.

Pass 1 backtests use the broker-quoted spread (`symbol_meta.json`). Real-world
fills can run 1.5-2× this during news / illiquid periods. This script reruns
each accepted symbol with normal AND doubled spread, then flags any whose edge
collapses (negative PnL or PF < 1.10).

Inputs:
  backtest/results/tune_180d_pass1.json   (only the 30 accepted symbols are stressed)

Outputs:
  backtest/results/spread_stress.json
  backtest/results/spread_fragile_symbols.json

Pool size: 6 workers. Each task doubles ALL_SYMBOLS[sym]["spread"] inside the
worker process before calling backtest_symbol() — auto_tuned params are applied
automatically via config.py import.
"""
import sys, os, json, time, copy
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import ALL_SYMBOLS, backtest_symbol  # noqa: E402

DAYS = int(os.environ.get("STRESS_DAYS", "180"))
RESULTS = ROOT / "backtest" / "results"
TUNE_PATH = RESULTS / "tune_180d_pass1.json"
OUT_PATH = RESULTS / "spread_stress.json"
FRAGILE_PATH = RESULTS / "spread_fragile_symbols.json"

# Acceptance rule (mirrors scripts/apply_tuned_params.py)
def _is_accepted(sym_entry):
    best = sym_entry.get("best") or {}
    res = best.get("result") or {}
    base = sym_entry.get("baseline") or {}
    if not res:
        return False
    if res.get("pf", 0) < 1.10: return False
    if res.get("trades", 0) < 20: return False
    if res.get("dd", 0) > 20: return False
    if res.get("pnl", 0) <= base.get("pnl", 0) + 5: return False
    return True


def _slim(r):
    if not r:
        return None
    return {"pnl": round(float(r.get("pnl", 0.0)), 2),
            "pf": round(float(r.get("pf", 0.0)), 2),
            "wr": round(float(r.get("wr", 0.0)), 1),
            "trades": int(r.get("trades", 0)),
            "dd": round(float(r.get("dd", 0.0)), 1)}


def _run_one(args):
    sym, days, multiplier = args
    # Mutate this worker's copy of ALL_SYMBOLS so backtest_symbol picks up the new spread.
    if multiplier != 1.0:
        meta = dict(ALL_SYMBOLS[sym])
        meta["spread"] = float(meta["spread"]) * multiplier
        ALL_SYMBOLS[sym] = meta
    r = backtest_symbol(sym, days, None, verbose=False)
    return sym, multiplier, _slim(r)


def main():
    if not TUNE_PATH.exists():
        print(f"NOT FOUND: {TUNE_PATH}")
        sys.exit(1)
    tune = json.load(open(TUNE_PATH))
    accepted = sorted(s for s, e in tune["results"].items() if _is_accepted(e))
    print(f"Stress-testing {len(accepted)} accepted symbols × {DAYS}d (1× and 2× spread)")
    print(f"Workers: 6")

    tasks = []
    for sym in accepted:
        tasks.append((sym, DAYS, 1.0))
        tasks.append((sym, DAYS, 2.0))

    t0 = time.time()
    out = {sym: {"normal": None, "double": None, "fragile": False} for sym in accepted}
    with Pool(processes=6) as pool:
        for sym, mult, res in pool.imap_unordered(_run_one, tasks):
            slot = "normal" if mult == 1.0 else "double"
            out[sym][slot] = res
            tag = "1x" if mult == 1.0 else "2x"
            if res:
                print(f"  {sym:<14} {tag}  pnl=${res['pnl']:>8.0f}  pf={res['pf']:.2f}  n={res['trades']}")
            else:
                print(f"  {sym:<14} {tag}  (no result)")

    fragile = []
    total_normal = 0.0
    total_double = 0.0
    for sym, rec in out.items():
        n = rec["normal"]; d = rec["double"]
        if n: total_normal += n["pnl"]
        if d: total_double += d["pnl"]
        is_fragile = False
        if d is None:
            is_fragile = True
        else:
            if d["pnl"] < 0 or d["pf"] < 1.10:
                is_fragile = True
        rec["fragile"] = bool(is_fragile)
        if is_fragile:
            fragile.append(sym)

    elapsed = time.time() - t0
    payload = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "days": DAYS,
        "elapsed_s": round(elapsed, 1),
        "accepted_count": len(accepted),
        "fragile_count": len(fragile),
        "total_pnl_normal": round(total_normal, 2),
        "total_pnl_double": round(total_double, 2),
        "total_pnl_impact": round(total_double - total_normal, 2),
        "results": out,
    }
    json.dump(payload, open(OUT_PATH, "w"), indent=2)
    json.dump(sorted(fragile), open(FRAGILE_PATH, "w"), indent=2)

    print()
    print("-" * 78)
    print(f"  total PnL  1x ${total_normal:,.0f}   2x ${total_double:,.0f}   "
          f"Δ ${total_double - total_normal:+,.0f}   ({elapsed:.0f}s)")
    print(f"  fragile: {len(fragile)} / {len(accepted)}")
    for s in sorted(fragile):
        d = out[s]["double"]
        n = out[s]["normal"]
        if d and n:
            print(f"    {s:<14} 1x pnl=${n['pnl']:>7.0f} pf={n['pf']:.2f}   "
                  f"2x pnl=${d['pnl']:>7.0f} pf={d['pf']:.2f}")
        else:
            print(f"    {s:<14} (missing run)")
    print(f"\nWrote {OUT_PATH}")
    print(f"Wrote {FRAGILE_PATH}")


if __name__ == "__main__":
    main()
