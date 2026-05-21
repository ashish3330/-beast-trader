#!/usr/bin/env python3 -B
"""
Forex SL × Trail Cartesian tune (12 symbols, 25 configs each).

Mirrors tune_swi20_optimal.py methodology — monkey-patch config + reload
v5_backtest so SL and trail can be forced across all regimes.

Output: tune_session_20260521/forex_sl_trail.json
"""
import os
import sys
import json
import time
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "tune_session_20260521" / "forex_sl_trail.json"

FOREX_SYMS = [
    "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "EURJPY", "USDCAD",
    "AUDJPY", "EURAUD", "GBPCHF", "CADJPY", "AUDUSD", "NZDUSD",
]

SL_GRID = [0.5, 1.0, 1.5, 2.0, 2.5]

# Trail profiles in LIVE format (R, type, param). Will be converted to
# backtest (R, param, type) when applied via monkey-patch.
TRAIL_PROFILES_LIVE = {
    "_TIGHT_LOCK":    [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_WIDE_RUNNER":   [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7),
                       (1.5, "lock", 0.5), (0.7, "be", 0.0)],
    "_RANGE_TIGHT":   [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0)],
    "_AGGR_LOCK":     [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8),
                       (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "_RUNNER_NO_BE":  [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5),
                       (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)],
}

WF_FOLDS = [60, 90, 120, 150, 180]

MIN_TRADES = 30   # forex requires >=30
MIN_PF = 1.8
MIN_WF_PF = 1.3
MIN_WF_POS_FOLDS = 3
MIN_DELTA_PNL = 30.0  # >= $30 improvement


def _bt_isolated(symbol, days, sl_mult=None, trail_live=None):
    """Run a backtest with SL/trail monkey-patched at config level.

    sl_mult=None  → keep config default (used for baseline).
    trail_live=None → keep config default.

    Reloads config + backtest inside the worker so this is parallel-safe
    when invoked via multiprocessing.Pool.
    """
    import importlib
    import config as cfg
    importlib.reload(cfg)

    if sl_mult is not None:
        cfg.SYMBOL_ATR_SL_OVERRIDE = dict(cfg.SYMBOL_ATR_SL_OVERRIDE)
        cfg.SYMBOL_ATR_SL_OVERRIDE[symbol] = sl_mult
        # Clear per-regime SL so it doesn't override per-symbol value
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME = dict(cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME)
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME[symbol] = {}

    if trail_live is not None:
        cfg.SYMBOL_TRAIL_OVERRIDE = dict(cfg.SYMBOL_TRAIL_OVERRIDE)
        cfg.SYMBOL_TRAIL_OVERRIDE[symbol] = trail_live
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE = dict(cfg.SYMBOL_REGIME_TRAIL_OVERRIDE)
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE[symbol] = {
            "trending": trail_live, "ranging": trail_live,
            "volatile": trail_live, "low_vol":  trail_live,
        }

    import backtest.v5_backtest as bt
    importlib.reload(bt)
    return bt.backtest_symbol(symbol, days=days, verbose=False)


def _safe_bt(symbol, days, sl_mult=None, trail_live=None):
    try:
        r = _bt_isolated(symbol, days, sl_mult, trail_live)
        if r is None:
            return None
        return {
            "trades": int(r.get("trades", 0)),
            "wr":     float(r.get("wr", 0) or 0),
            "pf":     float(r.get("pf", 0) or 0),
            "pnl":    float(r.get("pnl", 0) or 0),
            "dd":     float(r.get("dd", 0) or 0),
            "avg_r":  float(r.get("avg_r", 0) or 0),
        }
    except Exception:
        return None


# ── Pool workers ──────────────────────────────────────────────────────────

def _sweep_one(args):
    symbol, sl, name, trail_live, days = args
    r = _safe_bt(symbol, days, sl, trail_live)
    return {
        "symbol": symbol,
        "sl_mult": sl,
        "trail_name": name,
        "result": r,
    }


def _wf_one(args):
    symbol, sl, trail_live, days = args
    r = _safe_bt(symbol, days, sl, trail_live)
    return {"symbol": symbol, "sl_mult": sl, "days": days, "result": r}


def _baseline_one(args):
    symbol, days = args
    r = _safe_bt(symbol, days)  # no SL/trail override = use current config
    return {"symbol": symbol, "result": r}


# ── Main pipeline ─────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "session": "tune_session_20260521",
        "scope": "forex_sl_trail",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "symbols": FOREX_SYMS,
        "sl_grid": SL_GRID,
        "trail_profiles": {k: v for k, v in TRAIL_PROFILES_LIVE.items()},
        "wf_folds_days": WF_FOLDS,
        "gates": {
            "min_trades": MIN_TRADES,
            "min_pf": MIN_PF,
            "min_wf_pf": MIN_WF_PF,
            "min_wf_pos_folds": MIN_WF_POS_FOLDS,
            "min_delta_pnl": MIN_DELTA_PNL,
        },
        "baselines": {},
        "sweep": {},
        "winners": {},
        "summary": {},
    }

    workers = max(2, min(8, os.cpu_count() or 4))
    print(f"[forex tune] workers={workers}")

    # ── 1) Baselines (current config, 180d) ───────────────────────────────
    print(f"\n[A] Baselines 180d  ({len(FOREX_SYMS)} symbols)")
    with Pool(workers) as pool:
        for res in pool.imap_unordered(_baseline_one,
                                       [(s, 180) for s in FOREX_SYMS]):
            sym = res["symbol"]; r = res["result"]
            out["baselines"][sym] = r
            if r is None:
                print(f"  {sym:7s}: NO DATA")
            else:
                print(f"  {sym:7s}: n={r['trades']:3d} pf={r['pf']:5.2f} "
                      f"pnl=${r['pnl']:+8.0f} wr={r['wr']:5.1f}%")

    # Determine which symbols have sufficient data to tune
    tunable = []
    insufficient = []
    for s in FOREX_SYMS:
        r = out["baselines"].get(s)
        if r is None or r.get("trades", 0) < MIN_TRADES:
            insufficient.append(s)
        else:
            tunable.append(s)
    print(f"\n  tunable: {tunable}")
    print(f"  insufficient_data: {insufficient}")

    # ── 2) Cartesian sweep ────────────────────────────────────────────────
    print(f"\n[B] Sweep {len(tunable)} sym × {len(SL_GRID)} SL × "
          f"{len(TRAIL_PROFILES_LIVE)} trail = "
          f"{len(tunable) * len(SL_GRID) * len(TRAIL_PROFILES_LIVE)} BTs")
    jobs = []
    for sym in tunable:
        for sl in SL_GRID:
            for name, trail in TRAIL_PROFILES_LIVE.items():
                jobs.append((sym, sl, name, trail, 180))

    sweep_by_sym = {s: [] for s in tunable}
    t_sweep = time.time()
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_sweep_one, jobs), 1):
            sym = res["symbol"]
            r = res["result"]
            if r is None:
                continue
            sweep_by_sym[sym].append({
                "sl_mult": res["sl_mult"],
                "trail_name": res["trail_name"],
                **r,
            })
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i:4d}/{len(jobs)} ({time.time() - t_sweep:.0f}s)")
    out["sweep"] = sweep_by_sym
    print(f"  [B] sweep done in {time.time() - t_sweep:.0f}s")

    # ── 3) Pick candidates per symbol ─────────────────────────────────────
    print(f"\n[C] Candidates (PF>={MIN_PF}, trades>={MIN_TRADES}, "
          f"Δ>=${MIN_DELTA_PNL:.0f}):")
    candidates_by_sym = {}
    for sym in tunable:
        base = out["baselines"].get(sym, {})
        base_pnl = base.get("pnl", 0.0)
        rows = sweep_by_sym.get(sym, [])
        good = [r for r in rows
                if r["pf"] >= MIN_PF
                and r["trades"] >= MIN_TRADES
                and (r["pnl"] - base_pnl) >= MIN_DELTA_PNL]
        good.sort(key=lambda x: -x["pnl"])
        candidates_by_sym[sym] = good[:5]  # top 5 by PnL
        if good:
            top = good[0]
            print(f"  {sym:7s}: {len(good):2d} pass  top SL={top['sl_mult']} "
                  f"{top['trail_name']:14s} pnl=${top['pnl']:+8.0f} pf={top['pf']:5.2f} "
                  f"Δ=${top['pnl'] - base_pnl:+7.0f}")
        else:
            print(f"  {sym:7s}: 0 pass (baseline pnl=${base_pnl:+.0f})")

    # ── 4) Walk-forward validation on top candidates ─────────────────────
    print(f"\n[D] WF validation — {WF_FOLDS} folds")
    sub_jobs = []
    for sym, cands in candidates_by_sym.items():
        for c in cands:
            trail = TRAIL_PROFILES_LIVE[c["trail_name"]]
            for d in WF_FOLDS:
                sub_jobs.append((sym, c["sl_mult"], c["trail_name"], trail, d))
    print(f"  total WF BTs: {len(sub_jobs)}")
    results_named = []
    t_wf2 = time.time()
    if sub_jobs:
        with Pool(workers) as pool:
            for i, tup in enumerate(pool.imap_unordered(_wf_one_named, sub_jobs), 1):
                results_named.append(tup)
                if i % 50 == 0 or i == len(sub_jobs):
                    print(f"  {i:4d}/{len(sub_jobs)} ({time.time() - t_wf2:.0f}s)")

    # Aggregate
    wf_results_by_cand = {}
    bucket = {}
    for (sym, sl, name, days, r) in results_named:
        key = (sym, sl, name)
        bucket.setdefault(key, []).append({"days": days, "result": r})
    for key, folds in bucket.items():
        folds.sort(key=lambda x: x["days"])
        pfs = [f["result"]["pf"] for f in folds if f["result"] is not None]
        pos = sum(1 for f in folds
                  if f["result"] is not None and f["result"]["pnl"] > 0)
        n_valid = len(pfs)
        avg_pf = round(sum(pfs) / max(1, n_valid), 3) if n_valid else 0.0
        wf_results_by_cand[key] = {
            "avg_pf": avg_pf,
            "pos_folds": pos,
            "n_folds_valid": n_valid,
            "folds": folds,
        }

    # ── 5) Pick WF winners per symbol ─────────────────────────────────────
    print(f"\n[E] WF winners (avg_pf>={MIN_WF_PF}, pos>={MIN_WF_POS_FOLDS}/5):")
    winners = {}
    for sym, cands in candidates_by_sym.items():
        base_pnl = out["baselines"].get(sym, {}).get("pnl", 0.0)
        passed = []
        for c in cands:
            key = (sym, c["sl_mult"], c["trail_name"])
            wf = wf_results_by_cand.get(key)
            if wf is None:
                continue
            ok = (wf["avg_pf"] >= MIN_WF_PF
                  and wf["pos_folds"] >= MIN_WF_POS_FOLDS
                  and (c["pnl"] - base_pnl) >= MIN_DELTA_PNL)
            entry = {**c, "wf": wf, "delta_pnl": round(c["pnl"] - base_pnl, 2),
                     "wf_pass": ok}
            passed.append(entry)
        passed.sort(key=lambda x: (-(x["wf_pass"]), -(x["delta_pnl"])))
        winners[sym] = passed
        wf_passing = [p for p in passed if p["wf_pass"]]
        if wf_passing:
            best = wf_passing[0]
            print(f"  {sym:7s}: WINNER SL={best['sl_mult']} {best['trail_name']:14s} "
                  f"Δ=${best['delta_pnl']:+8.0f} 180d-pf={best['pf']:5.2f} "
                  f"WF-pf={best['wf']['avg_pf']:5.2f} pos={best['wf']['pos_folds']}/5")
        else:
            print(f"  {sym:7s}: no WF-pass (baseline pnl=${base_pnl:+.0f})")
    out["winners"] = winners

    # ── 6) Summary ───────────────────────────────────────────────────────
    summary = {
        "tunable": tunable,
        "insufficient_data": insufficient,
        "retune_recommended": {},
        "already_optimal": [],
        "no_winner": [],
    }
    for sym, passed in winners.items():
        wf_passing = [p for p in passed if p["wf_pass"]]
        base_pnl = out["baselines"].get(sym, {}).get("pnl", 0.0)
        if wf_passing:
            best = wf_passing[0]
            summary["retune_recommended"][sym] = {
                "current_baseline_pnl": base_pnl,
                "new_sl_mult": best["sl_mult"],
                "new_trail_name": best["trail_name"],
                "new_trail_steps": TRAIL_PROFILES_LIVE[best["trail_name"]],
                "new_pnl": best["pnl"],
                "delta_pnl": best["delta_pnl"],
                "new_pf": best["pf"],
                "new_wf_pf": best["wf"]["avg_pf"],
                "wf_pos_folds": best["wf"]["pos_folds"],
                "trades": best["trades"],
            }
        else:
            summary["no_winner"].append(sym)
    # Mark "already_optimal" if baseline is healthy (pf >= 1.8, trades >= 30)
    # AND no candidate beat it by $30
    for sym in summary["no_winner"][:]:
        b = out["baselines"].get(sym, {})
        if b and b.get("pf", 0) >= MIN_PF and b.get("trades", 0) >= MIN_TRADES \
           and b.get("pnl", 0) > 0:
            summary["already_optimal"].append(sym)
    out["summary"] = summary
    out["elapsed_sec"] = round(time.time() - t0, 1)
    out["finished_at"] = datetime.utcnow().isoformat() + "Z"

    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[done] wrote {OUT_PATH}")
    print(f"  retune: {list(summary['retune_recommended'].keys())}")
    print(f"  already_optimal: {summary['already_optimal']}")
    print(f"  no_winner (other): "
          f"{[s for s in summary['no_winner'] if s not in summary['already_optimal']]}")
    print(f"  insufficient_data: {summary['insufficient_data']}")
    print(f"  total elapsed: {out['elapsed_sec']}s")


# Top-level WF worker so it's picklable for Pool
def _wf_one_named(args):
    sym, sl, name, trail, days = args
    r = _safe_bt(sym, days, sl, trail)
    return (sym, sl, name, days, r)


if __name__ == "__main__":
    main()
