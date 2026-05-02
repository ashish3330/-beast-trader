"""
Trail-profile sweep for the top-6 winners from tune_180d_pass1.json.

For each symbol x trail profile (DEFAULT, TIGHT, LOOSE, AGGR_RUN), runs a 180d
backtest re-using each symbol's tuned params (min_quality, sl_atr_mult, ratchet_*)
and forcing the chosen trail profile via params["force_trail"] (which bypasses
TRAIL_OVERRIDE in v5_backtest.backtest_symbol).

Score = pnl * pf / sqrt(max(dd, 1)).

Outputs:
  backtest/results/trail_sweep.json          (full per-symbol per-profile metrics)
  backtest/results/trail_overrides_auto_dict.py  (TRAIL_OVERRIDE_AUTO mapping for
    symbols where a non-default profile beats DEFAULT by >= $50 in PnL)

Run: python3 -B scripts/sweep_trails.py
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, SL_OVERRIDE  # noqa: E402

RESULTS = ROOT / "backtest" / "results"
TUNE_FILE = RESULTS / "tune_180d_pass1.json"
OUT_JSON = RESULTS / "trail_sweep.json"
OUT_PY = RESULTS / "trail_overrides_auto_dict.py"

DAYS = 180
WORKERS = 4
PNL_DELTA_GATE = 50.0  # only override DEFAULT if non-default profile wins by this much

# DEFAULT mirrors DEFAULT_PARAMS["trail"]. Kept here so report shows what was
# tested even if v5_backtest defaults change later.
DEFAULT = [
    (8.0, 0.3, "trail"), (4.0, 0.5, "trail"), (2.0, 0.8, "trail"),
    (1.5, 0.7, "lock"),  (1.0, 0.4, "lock"),  (0.5, 0.0, "be"),
]
TIGHT = [
    (6.0, 0.4, "trail"), (3.0, 0.6, "trail"),
    (1.5, 0.9, "lock"),  (1.0, 0.5, "lock"),  (0.5, 0.0, "be"),
]
LOOSE = [
    (10.0, 0.2, "trail"), (5.0, 0.4, "trail"), (2.5, 0.6, "trail"),
    (1.5, 0.5, "lock"),   (0.7, 0.0, "be"),
]
AGGR_RUN = [
    (15.0, 0.3, "trail"), (8.0, 0.5, "trail"),
    (3.0, 0.5, "lock"),   (1.0, 0.0, "be"),
]
PROFILES = {
    "DEFAULT":  DEFAULT,
    "TIGHT":    TIGHT,
    "LOOSE":    LOOSE,
    "AGGR_RUN": AGGR_RUN,
}


def _score(r):
    if not r or r.get("trades", 0) < 5:
        return -1e9
    pnl = r.get("pnl", 0.0)
    pf = r.get("pf", 0.0)
    dd = max(r.get("dd", 0.0), 1.0)
    return pnl * pf / math.sqrt(dd)


def _slim(r):
    if not r:
        return r
    return {k: v for k, v in r.items() if k != "details"}


def _job(args):
    symbol, profile_name, profile, tuned_params = args
    t0 = time.time()
    # Tuned params (sl_atr_mult / min_quality / ratchet_*) plus forced trail.
    params = dict(tuned_params or {})
    params["force_trail"] = profile

    sl = params.get("sl_atr_mult")
    old = SL_OVERRIDE.get(symbol)
    if sl is not None:
        SL_OVERRIDE[symbol] = sl
    try:
        r = backtest_symbol(symbol, DAYS, params, verbose=False)
    finally:
        if sl is not None:
            if old is not None:
                SL_OVERRIDE[symbol] = old
            else:
                SL_OVERRIDE.pop(symbol, None)

    return {
        "symbol": symbol,
        "profile": profile_name,
        "elapsed_s": round(time.time() - t0, 1),
        "result": _slim(r),
        "score": _score(r) if r else -1e9,
    }


def _format_profile_py(p):
    parts = []
    for r, lock, kind in p:
        parts.append(f"({r}, {lock}, {kind!r})")
    return "[" + ", ".join(parts) + "]"


def main():
    if not TUNE_FILE.exists():
        sys.exit(f"Missing {TUNE_FILE}")

    with open(TUNE_FILE) as f:
        tune = json.load(f)

    ranked = sorted(
        tune["results"].items(),
        key=lambda kv: kv[1]["best"]["result"]["pnl"] if kv[1].get("best", {}).get("result") else -1e9,
        reverse=True,
    )
    top6 = [s for s, _ in ranked[:6]]
    print(f"Top 6 by post-tune PnL: {top6}")

    # Build the sweep job list
    jobs = []
    for sym in top6:
        tuned = tune["results"][sym]["best"]["params"]
        for pname, profile in PROFILES.items():
            jobs.append((sym, pname, profile, tuned))

    print(f"Running {len(jobs)} backtests across {WORKERS} workers ({DAYS}d each)")
    t0 = time.time()
    with Pool(WORKERS) as pool:
        out = pool.map(_job, jobs)
    print(f"Done in {time.time() - t0:.1f}s")

    # Group by symbol
    by_sym = {sym: {} for sym in top6}
    for row in out:
        by_sym[row["symbol"]][row["profile"]] = row

    # Pick winner per symbol + diff vs DEFAULT
    summary = {}
    overrides_auto = {}
    total_delta = 0.0
    print("\n" + "=" * 78)
    print(f"  {'SYMBOL':<10} {'BEST':<10} {'PnL':>10} {'PF':>6} {'DD':>6} {'TRD':>5}  Δ$ vs DEFAULT")
    print("=" * 78)
    for sym in top6:
        rows = by_sym[sym]
        default_row = rows.get("DEFAULT") or {}
        default_pnl = (default_row.get("result") or {}).get("pnl", 0.0)
        # Pick highest score across all 4 profiles
        best_name, best_row = max(
            rows.items(), key=lambda kv: kv[1].get("score", -1e9)
        )
        best_pnl = (best_row.get("result") or {}).get("pnl", 0.0)
        delta_pnl = best_pnl - default_pnl

        r = best_row.get("result") or {}
        print(
            f"  {sym:<10} {best_name:<10} {r.get('pnl', 0.0):10.2f} "
            f"{r.get('pf', 0.0):6.2f} {r.get('dd', 0.0):6.2f} {r.get('trades', 0):5d}  "
            f"{delta_pnl:+.2f}"
        )

        summary[sym] = {
            "winner": best_name,
            "winner_score": best_row.get("score"),
            "winner_pnl": best_pnl,
            "default_pnl": default_pnl,
            "delta_pnl": delta_pnl,
            "profiles": {pn: rows[pn] for pn in PROFILES if pn in rows},
        }

        if best_name != "DEFAULT" and delta_pnl >= PNL_DELTA_GATE:
            overrides_auto[sym] = PROFILES[best_name]
            total_delta += delta_pnl

    print("=" * 78)
    print(f"  Symbols with override (≥ ${PNL_DELTA_GATE:.0f} gain): {list(overrides_auto.keys())}")
    print(f"  Total Δpnl from overrides: ${total_delta:+.2f}")

    # Write JSON
    payload = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "days": DAYS,
        "workers": WORKERS,
        "top6": top6,
        "profiles_tested": {k: v for k, v in PROFILES.items()},
        "delta_gate": PNL_DELTA_GATE,
        "summary": summary,
        "overrides_auto": {sym: PROFILES[summary[sym]["winner"]]
                           for sym in overrides_auto},
        "total_delta_pnl": total_delta,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")

    # Write Python override dict
    lines = [
        '"""Auto-generated by scripts/sweep_trails.py — DO NOT EDIT BY HAND."""',
        "",
        "TRAIL_OVERRIDE_AUTO = {",
    ]
    for sym, profile in overrides_auto.items():
        lines.append(f"    {sym!r}: {_format_profile_py(profile)},")
    lines.append("}")
    lines.append("")
    OUT_PY.write_text("\n".join(lines))
    print(f"Wrote {OUT_PY}")


if __name__ == "__main__":
    main()
