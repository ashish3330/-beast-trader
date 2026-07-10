"""XAUUSD hard-tune 2026-06-19 — coordinate descent on 4 BT-measurable axes.

Axes:
  1. SL_ATR_MULT: [0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
  2. SIGNAL_QUALITY per regime: ±10 from baseline
  3. DIRECTION_BIAS per regime: LONG / SHORT / BOTH
  4. TOXIC_HOURS_PER_SYMBOL: derived from journal worst hours

Hard 25-min budget. Coordinate descent locking best at each axis.
"""
import sys, os, time, json
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import backtest.v5_backtest as bt

SYM = "XAUUSD"
DAYS = 180

# === Baseline ===
BASELINE = {
    "sl": 1.5,
    "quality": {"trending": 55, "ranging": 60, "volatile": 55, "low_vol": 60},
    "dir_bias": "SHORT",
    "toxic_extra": set(),
}

def run_bt(sl_mult, quality_dict, dir_bias_str, toxic_extra):
    """Run a single BT with overrides applied."""
    # Save original state
    orig_sl = bt.SL_OVERRIDE.get(SYM)
    orig_dir = bt.DIR_BIAS.get(SYM)
    orig_toxic = bt.TOXIC_HOURS_PER_SYMBOL.get(SYM, set()).copy()
    orig_quality = None
    try:
        import config
        orig_quality = config.SIGNAL_QUALITY_SYMBOL.get(SYM, {}).copy() if config.SIGNAL_QUALITY_SYMBOL.get(SYM) else None
    except Exception:
        pass

    # Apply overrides
    bt.SL_OVERRIDE[SYM] = sl_mult
    bt.DIR_BIAS[SYM] = {"LONG": 1, "SHORT": -1, "BOTH": 0}[dir_bias_str]
    bt.TOXIC_HOURS_PER_SYMBOL[SYM] = set(toxic_extra)
    try:
        import config
        config.SIGNAL_QUALITY_SYMBOL[SYM] = dict(quality_dict)
    except Exception:
        pass

    # Build params — use force_direction since DIR_BIAS only has limited LONG/SHORT support
    params = {
        "force_direction": dir_bias_str,
    }
    try:
        r = bt.backtest_symbol(SYM, DAYS, params, verbose=False)
    finally:
        # Restore
        if orig_sl is not None:
            bt.SL_OVERRIDE[SYM] = orig_sl
        if orig_dir is not None:
            bt.DIR_BIAS[SYM] = orig_dir
        bt.TOXIC_HOURS_PER_SYMBOL[SYM] = orig_toxic
        if orig_quality is not None:
            try:
                import config
                config.SIGNAL_QUALITY_SYMBOL[SYM] = orig_quality
            except Exception:
                pass
    return r

def fmt(r):
    if not r or r.get("trades", 0) == 0:
        return "n=0 PF=0.00 PnL=$0 DD=0%"
    return f"n={r['trades']:3d} PF={r['pf']:5.2f} WR={r['wr']:4.1f}% PnL=${r['pnl']:+7.2f} DD={r['dd']:4.1f}%"

def main():
    t0 = time.time()
    results = {}

    # ── Baseline ──
    print("=" * 70)
    print(f"XAUUSD HARDTUNE — 4-axis coordinate descent (180d BT)")
    print("=" * 70)
    baseline = run_bt(BASELINE["sl"], BASELINE["quality"], BASELINE["dir_bias"], BASELINE["toxic_extra"])
    print(f"\nBASELINE: SL={BASELINE['sl']} Q={BASELINE['quality']} DIR={BASELINE['dir_bias']} TOX={sorted(BASELINE['toxic_extra'])}")
    print(f"  → {fmt(baseline)}")
    results["baseline"] = baseline

    min_trades_floor = max(int(baseline["trades"] * 0.7), 10) if baseline.get("trades") else 10

    # ── Axis 1: SL_ATR_MULT ──
    print("\n" + "=" * 70)
    print("AXIS 1: SL_ATR_MULT")
    print("=" * 70)
    sl_results = {}
    for sl in [0.7, 1.0, 1.5, 2.0, 2.5, 3.0]:
        r = run_bt(sl, BASELINE["quality"], BASELINE["dir_bias"], BASELINE["toxic_extra"])
        sl_results[sl] = r
        marker = " ✓" if r and r.get("trades", 0) >= min_trades_floor else ""
        print(f"  SL={sl:.1f}: {fmt(r)}{marker}")
    # Pick best PF where trades >= 0.7 × baseline
    viable = [(sl, r) for sl, r in sl_results.items() if r and r.get("trades", 0) >= min_trades_floor]
    if not viable:
        # Fallback: best by PF regardless
        viable = [(sl, r) for sl, r in sl_results.items() if r and r.get("trades", 0) >= 5]
    best_sl, best_sl_r = max(viable, key=lambda x: x[1]["pf"]) if viable else (BASELINE["sl"], baseline)
    print(f"\n  WINNER axis 1: SL={best_sl} → {fmt(best_sl_r)}")
    results["axis1_sl"] = {"winner": best_sl, "result": best_sl_r, "all": sl_results}

    locked_sl = best_sl
    print(f"\n  [budget: {(time.time()-t0)/60:.1f} min]")

    # ── Axis 2: SIGNAL_QUALITY per regime ──
    print("\n" + "=" * 70)
    print("AXIS 2: SIGNAL_QUALITY per regime (±10 from baseline)")
    print("=" * 70)
    # Baseline q = trending:55, ranging:60, volatile:55, low_vol:60
    # Use 3 levels per regime, but vary one regime at a time then pick best combo
    # Keep manageable: vary all regimes uniformly first, then refine
    quality_results = {}
    quality_grids = [
        # Symmetric: uniform shift
        {"trending": 45, "ranging": 50, "volatile": 45, "low_vol": 50},   # -10
        {"trending": 50, "ranging": 55, "volatile": 50, "low_vol": 55},   # -5
        {"trending": 55, "ranging": 60, "volatile": 55, "low_vol": 60},   # baseline
        {"trending": 60, "ranging": 65, "volatile": 60, "low_vol": 65},   # +5
        {"trending": 65, "ranging": 70, "volatile": 65, "low_vol": 70},   # +10
        # Asymmetric: tighten trending only (XAU trending is strongest regime per note line 1712)
        {"trending": 45, "ranging": 60, "volatile": 55, "low_vol": 60},
        {"trending": 65, "ranging": 60, "volatile": 55, "low_vol": 60},
        # Asymmetric: loosen ranging
        {"trending": 55, "ranging": 50, "volatile": 55, "low_vol": 60},
        # Strict: high floor
        {"trending": 70, "ranging": 70, "volatile": 70, "low_vol": 70},
    ]
    for q in quality_grids:
        r = run_bt(locked_sl, q, BASELINE["dir_bias"], BASELINE["toxic_extra"])
        key = json.dumps(q, sort_keys=True)
        quality_results[key] = r
        marker = " ✓" if r and r.get("trades", 0) >= min_trades_floor else ""
        print(f"  Q={q}: {fmt(r)}{marker}")
    viable = [(k, r) for k, r in quality_results.items() if r and r.get("trades", 0) >= min_trades_floor]
    if not viable:
        viable = [(k, r) for k, r in quality_results.items() if r and r.get("trades", 0) >= 5]
    best_q_key, best_q_r = max(viable, key=lambda x: x[1]["pf"]) if viable else (json.dumps(BASELINE["quality"], sort_keys=True), baseline)
    best_q = json.loads(best_q_key)
    print(f"\n  WINNER axis 2: Q={best_q} → {fmt(best_q_r)}")
    results["axis2_quality"] = {"winner": best_q, "result": best_q_r}

    locked_q = best_q
    print(f"\n  [budget: {(time.time()-t0)/60:.1f} min]")

    # ── Axis 3: DIRECTION_BIAS ──
    print("\n" + "=" * 70)
    print("AXIS 3: DIRECTION_BIAS")
    print("=" * 70)
    dir_results = {}
    for d in ["LONG", "SHORT", "BOTH"]:
        r = run_bt(locked_sl, locked_q, d, BASELINE["toxic_extra"])
        dir_results[d] = r
        marker = " ✓" if r and r.get("trades", 0) >= min_trades_floor else ""
        print(f"  DIR={d}: {fmt(r)}{marker}")
    viable = [(d, r) for d, r in dir_results.items() if r and r.get("trades", 0) >= min_trades_floor]
    if not viable:
        viable = [(d, r) for d, r in dir_results.items() if r and r.get("trades", 0) >= 5]
    best_d, best_d_r = max(viable, key=lambda x: x[1]["pf"]) if viable else (BASELINE["dir_bias"], baseline)
    print(f"\n  WINNER axis 3: DIR={best_d} → {fmt(best_d_r)}")
    results["axis3_dir"] = {"winner": best_d, "result": best_d_r}

    locked_d = best_d
    print(f"\n  [budget: {(time.time()-t0)/60:.1f} min]")

    # ── Axis 4: TOXIC_HOURS_PER_SYMBOL ──
    print("\n" + "=" * 70)
    print("AXIS 4: TOXIC_HOURS_PER_SYMBOL (from journal worst hours)")
    print("=" * 70)
    # Journal data shows XAUUSD bleeders: h10 (-$18, 20% WR), h18 (-$23, 33%), h20 (-$67, 25%)
    # Also h14 (-$3, 50%) borderline. h3/h4 already in global TOXIC_HOURS_UTC.
    toxic_results = {}
    toxic_candidates = [
        set(),                # baseline (no extras)
        {20},                 # worst single hour
        {18, 20},             # two worst
        {10, 18, 20},         # three worst
        {10, 14, 18, 20},     # full bleed set
        {18, 20, 23},         # alt
    ]
    for tox in toxic_candidates:
        r = run_bt(locked_sl, locked_q, locked_d, tox)
        key = ",".join(str(h) for h in sorted(tox)) or "(none)"
        toxic_results[key] = r
        marker = " ✓" if r and r.get("trades", 0) >= min_trades_floor else ""
        print(f"  TOX={{{key}}}: {fmt(r)}{marker}")
    viable = [(k, r) for k, r in toxic_results.items() if r and r.get("trades", 0) >= min_trades_floor]
    if not viable:
        viable = [(k, r) for k, r in toxic_results.items() if r and r.get("trades", 0) >= 5]
    best_t_key, best_t_r = max(viable, key=lambda x: x[1]["pf"]) if viable else ("(none)", baseline)
    best_t = set(int(x) for x in best_t_key.split(",")) if best_t_key != "(none)" else set()
    print(f"\n  WINNER axis 4: TOX={{{best_t_key}}} → {fmt(best_t_r)}")
    results["axis4_toxic"] = {"winner": sorted(best_t), "result": best_t_r}

    locked_t = best_t

    # ── Final confirm ──
    print("\n" + "=" * 70)
    print("FINAL CONFIRM — full winner_config")
    print("=" * 70)
    final = run_bt(locked_sl, locked_q, locked_d, locked_t)
    print(f"  SL={locked_sl} Q={locked_q} DIR={locked_d} TOX={sorted(locked_t)}")
    print(f"  → {fmt(final)}")

    # Decision
    threshold_pf = max(1.5, baseline["pf"] * 1.10)
    dd_cap = baseline["dd"] * 1.30 if baseline.get("dd") else 999
    decision = "ADVANCE_TO_WF" if (
        final and final.get("pf", 0) >= threshold_pf and
        final.get("trades", 0) >= 30 and
        final.get("dd", 999) <= dd_cap
    ) else "NULL"

    print(f"\n  Threshold: PF >= {threshold_pf:.2f}, trades >= 30, DD <= {dd_cap:.2f}%")
    print(f"  Decision: {decision}")
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")

    # Print structured summary
    print("\n" + "=" * 70)
    print("STRUCTURED RESULT")
    print("=" * 70)
    summary = {
        "sym": SYM,
        "baseline_pf": baseline["pf"],
        "baseline_pnl": baseline["pnl"],
        "baseline_trades": baseline["trades"],
        "baseline_dd": baseline["dd"],
        "winner_config": {
            "SL_ATR_MULT": locked_sl,
            "SIGNAL_QUALITY": locked_q,
            "DIRECTION_BIAS": locked_d,
            "TOXIC_HOURS_PER_SYMBOL": sorted(locked_t),
        },
        "winner_pf": final["pf"] if final else 0,
        "winner_pnl": final["pnl"] if final else 0,
        "winner_trades": final["trades"] if final else 0,
        "winner_dd": final["dd"] if final else 0,
        "decision": decision,
    }
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    main()
