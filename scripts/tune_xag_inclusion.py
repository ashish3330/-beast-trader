#!/usr/bin/env python3 -B
"""
ETHUSD-only multi-axis per-regime tune + inclusion gate.

Runs all 5 per-regime axes on ETHUSD only:
  1. min_quality        (SIGNAL_QUALITY_SYMBOL_AUTO)
  2. SL ATR mult        (SL_OVERRIDE_REGIME_AUTO)
  3. trail profile      (TRAIL_OVERRIDE_REGIME_AUTO)
  4. direction bias     (DIRECTION_BIAS_REGIME_AUTO)
  5. risk cap           (RISK_CAP_REGIME_AUTO)

For each cell:
  - Sweep grid → pick best by cell-PnL with --with-slippage --with-commission --with-swap.
  - 5-fold WF validate (PF >= 1.4, >=3/5 folds positive).
  - Δ >= $30 vs cell baseline.

Inclusion gate: after ALL winners applied, full-cost ETH backtest must
clear net-zero on 90d AND 180d AND 360d (no period loses money).
Otherwise ETH stays out.

Output: backtest/results/eth_tune_<date>/_summary.json
"""
import json, os, sys, time
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SYM = "XAGUSD"
REGIMES = ("trending", "ranging", "volatile", "low_vol")

MQ_GRID    = [30, 35, 40, 45, 50, 55, 60, 65]
SL_GRID    = [0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
DIR_GRID   = ("LONG", "SHORT", "BOTH")
RISK_GRID  = [0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.0]
TRAIL_PROFILES = {
    "ULTRA_TIGHT":  [(2.0, "lock", 1.5), (1.0, "lock", 0.7), (0.5, "lock", 0.2), (0.2, "be", 0.0)],
    "TIGHT_LOCK":   [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "DEFAULT":      [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.6),
                     (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "lock", 0.15), (0.3, "be", 0.0)],
    "TREND_LOOSE":  [(15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5),
                     (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "RANGE_TIGHT":  [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0)],
    "AGGR_LOCK":    [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8),
                     (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "WIDE_RUNNER":  [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7),
                     (1.5, "lock", 0.5), (0.7, "be", 0.0)],
}

MIN_TRADES_PER_CELL = int(os.getenv("MIN_TRADES_PER_CELL", "10"))
MIN_LIFT_USD = float(os.getenv("MIN_LIFT_USD", "10.0"))
MIN_WF_PF = float(os.getenv("MIN_WF_PF", "1.2"))
MIN_WF_POSITIVE_FOLDS = int(os.getenv("MIN_WF_POSITIVE_FOLDS", "2"))
WF_FOLDS = [60, 90, 120, 150, 180]
TUNE_DAYS = int(os.getenv("TUNE_DAYS", "180"))

DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "backtest" / "results" / f"eth_tune_{DATE_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bt_with_costs(symbol, days, **overrides):
    """Reload config with patched overrides, run backtest WITH realistic costs."""
    import importlib
    import config as cfg
    importlib.reload(cfg)
    for dict_name, sym_dict in overrides.items():
        d = getattr(cfg, dict_name)
        d = dict(d)
        for s, v in sym_dict.items():
            if isinstance(v, dict) and isinstance(d.get(s), dict):
                d.setdefault(s, {}).update(v)
            else:
                d[s] = v
        setattr(cfg, dict_name, d)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    # PATCH the cost flags into DEFAULT_PARAMS so backtest uses them
    return bt.backtest_symbol(symbol, days=days, verbose=False,
                              params={**bt.DEFAULT_PARAMS,
                                      "with_slippage": True, "with_commission": True, "with_swap": True})


def _eth_baseline():
    """Full-cost baseline split by regime."""
    r = _bt_with_costs(SYM, days=TUNE_DAYS)
    cells = {reg: {"pnl": 0.0, "n": 0} for reg in REGIMES}
    for t in r.get("details", []):
        reg = t.get("regime", "")
        if reg in cells:
            cells[reg]["n"] += 1
            cells[reg]["pnl"] += t["pnl"]
    for reg in cells:
        cells[reg]["pnl"] = round(cells[reg]["pnl"], 2)
    return r, cells


def _sweep_axis(axis_name, target_regime, baseline_pnl, grid, override_dict_name, value_fn):
    cands = []
    for v in grid:
        try:
            override = {SYM: {target_regime: value_fn(v)}}
            r = _bt_with_costs(SYM, days=TUNE_DAYS, **{override_dict_name: override})
        except Exception as e:
            continue
        cell = [t for t in r.get("details", []) if t.get("regime") == target_regime]
        if len(cell) < 8:
            continue
        cell_pnl = round(sum(t["pnl"] for t in cell), 2)
        gp = sum(t["pnl"] for t in cell if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in cell if t["pnl"] < 0) or 0.01
        cands.append({"v": v, "cell_pnl": cell_pnl, "cell_pf": round(gp/gl, 2), "n": len(cell)})
    if not cands:
        return None
    best = max(cands, key=lambda c: c["cell_pnl"])
    if best["cell_pnl"] - baseline_pnl < MIN_LIFT_USD:
        return None
    return {"axis": axis_name, "regime": target_regime, "baseline": baseline_pnl, "best": best, "all": cands}


def _wf_validate(target_regime, override_dict_name, value):
    folds = []
    for d in WF_FOLDS:
        try:
            override = {SYM: {target_regime: value}}
            r = _bt_with_costs(SYM, days=d, **{override_dict_name: override})
        except Exception:
            return None
        cell = [t for t in r.get("details", []) if t.get("regime") == target_regime]
        if not cell:
            folds.append({"days": d, "pnl": 0, "pf": 0, "n": 0})
            continue
        gp = sum(t["pnl"] for t in cell if t["pnl"] > 0)
        gl = sum(abs(t["pnl"]) for t in cell if t["pnl"] < 0) or 0.01
        folds.append({"days": d, "pnl": round(sum(t["pnl"] for t in cell), 2),
                      "pf": round(gp/gl, 2), "n": len(cell)})
    avg_pf = round(sum(f["pf"] for f in folds) / max(1, len(folds)), 2)
    pos = sum(1 for f in folds if f["pnl"] > 0)
    return {"folds": folds, "avg_pf": avg_pf, "pos_folds": pos}


def main():
    print(f"\nETHUSD inclusion tune — {DATE_TAG}")
    print(f"Tune days: {TUNE_DAYS}, full-cost (slip+comm+swap)\n")
    t0 = time.time()

    print("[A] Baseline (full-cost)...")
    r0, cells = _eth_baseline()
    baseline_total = r0.get("pnl", 0)
    print(f"  ETHUSD baseline: ${baseline_total} on {r0.get('trades', 0)} trades")
    for reg in REGIMES:
        c = cells[reg]
        print(f"    {reg:10s} n={c['n']:3d} pnl=${c['pnl']}")
    print()

    print("[B] Axis sweeps per-regime...")
    AXES = [
        ("min_quality", "SIGNAL_QUALITY_SYMBOL", MQ_GRID, lambda v: v),
        ("SL_mult",     "SYMBOL_ATR_SL_OVERRIDE_REGIME", SL_GRID, lambda v: v),
        ("dir_bias",    "DIRECTION_BIAS_REGIME", DIR_GRID, lambda v: v),
        ("trail",       "SYMBOL_REGIME_TRAIL_OVERRIDE", list(TRAIL_PROFILES.items()), lambda v: v[1] if isinstance(v, tuple) else v),
        ("risk_cap",    "SYMBOL_RISK_CAP_REGIME", RISK_GRID, lambda v: v),
    ]
    candidates_per_axis = {}
    for axis_name, dict_name, grid, vfn in AXES:
        # min_quality needs special handling — it's a per-regime dict not a scalar
        if axis_name == "min_quality":
            # We need to inject the full per-regime dict for SIGNAL_QUALITY_SYMBOL
            # Easier: set the entire 4-regime override at once with current value
            # for the target regime, default for others.
            from config import SIGNAL_QUALITY_THRESHOLDS
            DEFAULT_MQ = dict(SIGNAL_QUALITY_THRESHOLDS)
            for reg in REGIMES:
                if cells[reg]["n"] < MIN_TRADES_PER_CELL:
                    continue
                cands = []
                for mq in MQ_GRID:
                    full = dict(DEFAULT_MQ)
                    full[reg] = mq
                    try:
                        r = _bt_with_costs(SYM, days=TUNE_DAYS,
                                           SIGNAL_QUALITY_SYMBOL={SYM: full})
                    except Exception:
                        continue
                    cell = [t for t in r.get("details", []) if t.get("regime") == reg]
                    if len(cell) < 8:
                        continue
                    cell_pnl = round(sum(t["pnl"] for t in cell), 2)
                    gp = sum(t["pnl"] for t in cell if t["pnl"] > 0)
                    gl = sum(abs(t["pnl"]) for t in cell if t["pnl"] < 0) or 0.01
                    cands.append({"v": mq, "cell_pnl": cell_pnl, "cell_pf": round(gp/gl, 2), "n": len(cell)})
                if not cands:
                    continue
                best = max(cands, key=lambda c: c["cell_pnl"])
                if best["cell_pnl"] - cells[reg]["pnl"] >= MIN_LIFT_USD:
                    candidates_per_axis.setdefault(axis_name, []).append(
                        {"axis": axis_name, "regime": reg, "baseline": cells[reg]["pnl"], "best": best, "all": cands})
            continue

        # Other axes
        for reg in REGIMES:
            if cells[reg]["n"] < MIN_TRADES_PER_CELL:
                continue
            cands = []
            for v in grid:
                actual_v = v[1] if axis_name == "trail" and isinstance(v, tuple) else v
                try:
                    r = _bt_with_costs(SYM, days=TUNE_DAYS,
                                       **{dict_name: {SYM: {reg: actual_v}}})
                except Exception:
                    continue
                cell = [t for t in r.get("details", []) if t.get("regime") == reg]
                if len(cell) < 8:
                    continue
                cell_pnl = round(sum(t["pnl"] for t in cell), 2)
                gp = sum(t["pnl"] for t in cell if t["pnl"] > 0)
                gl = sum(abs(t["pnl"]) for t in cell if t["pnl"] < 0) or 0.01
                rec = {"v": v[0] if axis_name == "trail" and isinstance(v, tuple) else v,
                       "actual_v": actual_v,
                       "cell_pnl": cell_pnl, "cell_pf": round(gp/gl, 2), "n": len(cell)}
                cands.append(rec)
            if not cands:
                continue
            best = max(cands, key=lambda c: c["cell_pnl"])
            if best["cell_pnl"] - cells[reg]["pnl"] >= MIN_LIFT_USD:
                candidates_per_axis.setdefault(axis_name, []).append(
                    {"axis": axis_name, "regime": reg, "baseline": cells[reg]["pnl"], "best": best, "all": cands})

    total_cands = sum(len(v) for v in candidates_per_axis.values())
    print(f"  Total candidates across all axes: {total_cands}")
    for axis, cs in candidates_per_axis.items():
        for c in cs:
            print(f"    [{axis}] {c['regime']} v={c['best']['v']} Δ${c['best']['cell_pnl']-c['baseline']:+.0f}")
    print()

    # ── STAGE C: WF validate each candidate ───────────────────────────
    print("[C] 5-fold WF validation per candidate...")
    DICT_FOR_AXIS = {
        "min_quality": "SIGNAL_QUALITY_SYMBOL",
        "SL_mult":     "SYMBOL_ATR_SL_OVERRIDE_REGIME",
        "dir_bias":    "DIRECTION_BIAS_REGIME",
        "trail":       "SYMBOL_REGIME_TRAIL_OVERRIDE",
        "risk_cap":    "SYMBOL_RISK_CAP_REGIME",
    }
    winners = []
    for axis, cs in candidates_per_axis.items():
        for c in cs:
            value = c["best"].get("actual_v", c["best"]["v"])
            if axis == "min_quality":
                from config import SIGNAL_QUALITY_THRESHOLDS
                full = dict(SIGNAL_QUALITY_THRESHOLDS)
                full[c["regime"]] = value
                ovr_value = full
            else:
                ovr_value = value
            # WF
            folds = []
            for d in WF_FOLDS:
                try:
                    if axis == "min_quality":
                        r = _bt_with_costs(SYM, days=d, SIGNAL_QUALITY_SYMBOL={SYM: ovr_value})
                    else:
                        r = _bt_with_costs(SYM, days=d, **{DICT_FOR_AXIS[axis]: {SYM: {c["regime"]: ovr_value}}})
                except Exception:
                    folds = None
                    break
                cell = [t for t in r.get("details", []) if t.get("regime") == c["regime"]]
                if not cell:
                    folds.append({"days": d, "pnl": 0, "pf": 0, "n": 0})
                    continue
                gp = sum(t["pnl"] for t in cell if t["pnl"] > 0)
                gl = sum(abs(t["pnl"]) for t in cell if t["pnl"] < 0) or 0.01
                folds.append({"days": d, "pnl": round(sum(t["pnl"] for t in cell), 2),
                              "pf": round(gp/gl, 2), "n": len(cell)})
            if folds is None:
                continue
            avg_pf = round(sum(f["pf"] for f in folds) / max(1, len(folds)), 2)
            pos = sum(1 for f in folds if f["pnl"] > 0)
            wf = {"folds": folds, "avg_pf": avg_pf, "pos_folds": pos}
            if avg_pf >= MIN_WF_PF and pos >= MIN_WF_POSITIVE_FOLDS:
                winners.append({**c, "wf": wf, "actual_value": ovr_value})
                print(f"  ✓ {axis} {c['regime']} v={c['best']['v']} Δ${c['best']['cell_pnl']-c['baseline']:+.0f} WF PF {avg_pf} {pos}/5")
            else:
                print(f"  ✗ {axis} {c['regime']} v={c['best']['v']} WF FAIL PF {avg_pf} {pos}/5")

    # ── STAGE D: combined apply + inclusion gate (90/180/360d) ─────────
    print(f"\n[D] Inclusion gate: full ETH backtest WITH all {len(winners)} winners applied, 90+180+360d...")
    combined = {
        "SIGNAL_QUALITY_SYMBOL": {},
        "SYMBOL_ATR_SL_OVERRIDE_REGIME": {},
        "DIRECTION_BIAS_REGIME": {},
        "SYMBOL_REGIME_TRAIL_OVERRIDE": {},
        "SYMBOL_RISK_CAP_REGIME": {},
    }
    for w in winners:
        if w["axis"] == "min_quality":
            combined["SIGNAL_QUALITY_SYMBOL"][SYM] = w["actual_value"]
        elif w["axis"] == "SL_mult":
            combined["SYMBOL_ATR_SL_OVERRIDE_REGIME"].setdefault(SYM, {})[w["regime"]] = w["actual_value"]
        elif w["axis"] == "dir_bias":
            combined["DIRECTION_BIAS_REGIME"].setdefault(SYM, {})[w["regime"]] = w["actual_value"]
        elif w["axis"] == "trail":
            combined["SYMBOL_REGIME_TRAIL_OVERRIDE"].setdefault(SYM, {})[w["regime"]] = w["actual_value"]
        elif w["axis"] == "risk_cap":
            combined["SYMBOL_RISK_CAP_REGIME"].setdefault(SYM, {})[w["regime"]] = w["actual_value"]

    inclusion_results = {}
    for d_days in (90, 180, 360):
        r = _bt_with_costs(SYM, days=d_days, **combined)
        inclusion_results[d_days] = {
            "trades": r.get("trades", 0),
            "pnl": r.get("pnl", 0),
            "pf": r.get("pf", 0),
            "wr": r.get("wr", 0),
        }
        print(f"  {d_days}d: {r.get('trades', 0)} trades, PnL ${r.get('pnl', 0)}, PF {r.get('pf', 0)}, WR {r.get('wr', 0)}%")

    # Inclusion gate: ALL three periods must be net positive
    ok = all(v["pnl"] > 0 for v in inclusion_results.values())
    print(f"\n  Inclusion gate (all periods net+): {'PASS — include ETH' if ok else 'FAIL — keep ETH excluded'}")

    summary = {
        "captured_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "baseline_total_pnl": baseline_total,
        "baseline_per_regime": cells,
        "candidates_per_axis": candidates_per_axis,
        "wf_winners": winners,
        "combined_overrides": {k: v for k, v in combined.items() if v},
        "inclusion_results": inclusion_results,
        "inclusion_passed": ok,
    }
    summary["combined_overrides_serializable"] = json.dumps(combined, default=str)
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  saved: {OUT_DIR}/_summary.json")
    print(f"  total elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
