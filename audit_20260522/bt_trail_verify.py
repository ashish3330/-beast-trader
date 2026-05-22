#!/usr/bin/env python3 -B
"""
BT TRAIL verifier — confirm backtest computes the same per-(sym, regime) trail
profile as the live executor for the 5 active symbols.

LIVE resolution order (executor._resolve_trail_steps):
  0. MARGINAL_TRAIL when entry_score < SCORE_TIER_THRESHOLD (skipped for BT cmp)
  1. SYMBOL_REGIME_TRAIL_OVERRIDE[sym][regime]
  2. SYMBOL_TRAIL_OVERRIDE[sym]
  3. REGIME_TRAIL_DEFAULTS[regime]
  4. TRAIL_STEPS  (global)

BT resolution order (v5_backtest.py:930-948 + 513):
  A. TRAIL_OVERRIDE_REGIME[sym][regime]      (mirrors live #1)
  B. trail_steps = TRAIL_OVERRIDE.get(sym, p['trail'])
       where TRAIL_OVERRIDE mirrors live SYMBOL_TRAIL_OVERRIDE,
       and p['trail'] mirrors live TRAIL_STEPS.
  ─ MISSING: REGIME_TRAIL_DEFAULTS layer between B-symbol and B-default.

This script prints per-(sym, regime) `BT_trail` vs `LIVE_trail` and `match=YES/NO`,
and writes the human report to bt_trail_verify.md.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Live config
import config as live
# BT config
from backtest import v5_backtest as bt

REGIMES = ("trending", "ranging", "volatile", "low_vol")
SYMBOLS = ["SP500.r", "UKOUSD", "US2000.r", "DJ30.r", "USOUSD"]


def live_resolve(symbol, regime):
    """Mirror executor._resolve_trail_steps (ignoring marginal-tier path)."""
    cell = live.SYMBOL_REGIME_TRAIL_OVERRIDE.get(symbol, {}).get(regime)
    if cell:
        return cell, "SYMBOL_REGIME_TRAIL_OVERRIDE"
    sym_t = live.SYMBOL_TRAIL_OVERRIDE.get(symbol)
    if sym_t:
        return sym_t, "SYMBOL_TRAIL_OVERRIDE"
    reg_d = live.REGIME_TRAIL_DEFAULTS.get(regime)
    if reg_d:
        return reg_d, "REGIME_TRAIL_DEFAULTS"
    return live.TRAIL_STEPS, "TRAIL_STEPS"


def bt_resolve(symbol, regime):
    """Mirror BT trail resolution at v5_backtest.py:930 + 513."""
    cell = bt.TRAIL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
    if cell is not None:
        return cell, "TRAIL_OVERRIDE_REGIME"
    sym_t = bt.TRAIL_OVERRIDE.get(symbol)
    if sym_t is not None:
        return sym_t, "TRAIL_OVERRIDE"
    return bt.DEFAULT_PARAMS["trail"], "DEFAULT_PARAMS[trail]"


def to_canon(steps, fmt):
    """Normalize to (R, type, param) for comparison.
    fmt='live'  → already (R, type, param)
    fmt='bt'    → stored as (R, param, type)
    """
    out = []
    for tup in steps:
        if len(tup) != 3:
            return None
        if fmt == "live":
            r, t, p = tup
        else:
            r, p, t = tup
        # Round to avoid float precision noise.
        out.append((round(float(r), 6), str(t), round(float(p), 6)))
    return out


def main():
    rows = []
    mismatches = []
    print(f"BT TRAIL verifier — {len(SYMBOLS)} symbols × {len(REGIMES)} regimes\n")
    # Header
    fmt = "{sym:>9}  {regime:<9}  match={match:>3}  bt_src={bs:<30} live_src={ls}"
    for sym in SYMBOLS:
        for reg in REGIMES:
            live_steps, live_src = live_resolve(sym, reg)
            bt_steps,   bt_src   = bt_resolve(sym, reg)
            live_c = to_canon(live_steps, "live")
            bt_c   = to_canon(bt_steps,   "bt")
            match = (live_c == bt_c)
            rows.append({
                "symbol": sym, "regime": reg,
                "match": match,
                "bt_source": bt_src,
                "live_source": live_src,
                "bt_trail": bt_c,
                "live_trail": live_c,
            })
            print(f"[{sym:>9} {reg:<9}] match={'YES' if match else 'NO ':3}  "
                  f"BT_src={bt_src:<28} LIVE_src={live_src}")
            if not match:
                print(f"    BT_trail   = {bt_c}")
                print(f"    LIVE_trail = {live_c}")
                mismatches.append((sym, reg))

    # Summary
    total = len(rows)
    ok = sum(1 for r in rows if r["match"])
    print(f"\n=== SUMMARY: {ok}/{total} match ===")
    if mismatches:
        print(f"Mismatched cells: {mismatches}")

    # Now actually run backtest_symbol() per symbol to make sure no
    # divergence between the resolver above and what BT actually loads
    # when wired up end-to-end. We just need it to not crash AND we
    # observe trail step counts via trades produced.
    print("\n=== End-to-end BT smoke runs (10d) ===")
    smoke_rows = []
    for sym in SYMBOLS:
        try:
            # 10 days returns None (too little data after warmup) — use 60d.
            res = bt.backtest_symbol(sym, days=60, verbose=False)
            if res is None:
                print(f"[{sym}] backtest returned None (likely no cache or too little data)")
                smoke_rows.append({"symbol": sym, "ok": False, "reason": "None"})
                continue
            trades = res.get("trades", 0)
            pnl = res.get("pnl", 0.0)
            pf  = res.get("pf", 0.0)
            print(f"[{sym}] trades={trades} pnl={pnl:.2f} pf={pf:.2f}")
            smoke_rows.append({"symbol": sym, "ok": True, "trades": trades, "pnl": pnl, "pf": pf})
        except Exception as e:
            print(f"[{sym}] ERROR: {e!r}")
            smoke_rows.append({"symbol": sym, "ok": False, "reason": repr(e)})

    return rows, mismatches, smoke_rows


if __name__ == "__main__":
    rows, mismatches, smoke = main()
    out = {
        "rows": rows,
        "mismatches": mismatches,
        "smoke": smoke,
    }
    Path(__file__).parent.joinpath("bt_trail_verify.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote bt_trail_verify.json")
