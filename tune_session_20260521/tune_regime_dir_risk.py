"""Per-(symbol, regime) DIRECTION BIAS + RISK CAP tuner.

Read-only on repo source files. Uses in-memory monkey-patching of
backtest.v5_backtest globals (_DIR_BIAS_REGIME_STR, RISK_CAP_REGIME).

Output: tune_session_20260521/regime_dir_risk.json

Worker pattern: each subprocess does its own import and patches state
locally so parent process state is never mutated.
"""
from __future__ import annotations

import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
OUT_DIR = ROOT / "tune_session_20260521"
OUT_DIR.mkdir(exist_ok=True)
OUT_JSON = OUT_DIR / "regime_dir_risk.json"

sys.path.insert(0, str(ROOT))

REGIMES = ("trending", "ranging", "volatile", "low_vol")
RISK_GRID = (0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
DIR_CHOICES = ("LONG", "SHORT", "BOTH")  # BOTH = open both sides (skip dir filter)
DAYS = 180
WF_FOLDS = 5
MIN_TRADES = 10


# ─────────────────────────────────────────────────────────────────────────────
# Worker: runs ONE backtest variant under a patched config.
# ─────────────────────────────────────────────────────────────────────────────
def _run_one(args):
    """args = (mode, sym, payload)

    mode='dir' : payload = {'force': 'LONG'|'SHORT'|'BOTH'|None}
                Returns trade journal (regime-tagged).
    mode='risk': payload = {'regime': r, 'risk': float}
                Returns trade journal restricted to that regime only.
    """
    mode, sym, payload = args
    try:
        # Fresh import in subprocess (mp uses spawn-or-fork; either way our
        # patches live only here).
        from backtest import v5_backtest as bt

        # For direction sweep we must temporarily remove this symbol's
        # regime-bias overlay so the forced direction actually wins.
        _orig_dir = bt._DIR_BIAS_REGIME_STR.pop(sym, None)
        _orig_risk = None
        params = {}

        if mode == "dir":
            force = payload.get("force")
            if force is not None:
                params["force_direction"] = force
            # else: baseline (use current per-symbol DIR_BIAS as-is, but with
            # regime overlay removed so it's per-symbol bias only)
            # Actually: baseline should match LIVE truth which INCLUDES regime
            # overlay. Re-add for baseline runs.
            if payload.get("baseline") and _orig_dir is not None:
                bt._DIR_BIAS_REGIME_STR[sym] = _orig_dir

        elif mode == "risk":
            regime = payload["regime"]
            risk = payload["risk"]
            # Snapshot + patch RISK_CAP_REGIME
            _orig_risk = bt.RISK_CAP_REGIME.get(sym)
            cur = dict(_orig_risk) if _orig_risk else {}
            cur[regime] = risk
            bt.RISK_CAP_REGIME[sym] = cur
            # Re-apply original dir bias overlay (we don't want to change dir
            # logic during risk sweep)
            if _orig_dir is not None:
                bt._DIR_BIAS_REGIME_STR[sym] = _orig_dir
            # also apply baseline force from payload if asked
            if payload.get("force_direction"):
                params["force_direction"] = payload["force_direction"]

        res = bt.backtest_symbol(sym, days=DAYS, params=params, verbose=False)

        # Restore (paranoia — subprocess will die anyway)
        if _orig_dir is not None:
            bt._DIR_BIAS_REGIME_STR[sym] = _orig_dir
        if mode == "risk":
            if _orig_risk is None:
                bt.RISK_CAP_REGIME.pop(sym, None)
            else:
                bt.RISK_CAP_REGIME[sym] = _orig_risk

        if res is None or not res.get("details"):
            return (mode, sym, payload, None)

        # Pack only what we need (entry_bar for WF, regime, direction, pnl, pnl_r)
        trades = [
            {
                "eb": t["entry_bar"],
                "reg": t["regime"],
                "dir": t["direction"],
                "pnl": t["pnl"],
                "r": t["pnl_r"],
            }
            for t in res["details"]
        ]
        return (mode, sym, payload, trades)
    except Exception as e:
        import traceback
        return (mode, sym, payload, {"error": repr(e), "tb": traceback.format_exc()})


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────
def _stats(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    wr = 100.0 * len(wins) / len(trades) if trades else 0.0
    # equity curve DD
    eq = [0.0]
    for t in trades:
        eq.append(eq[-1] + t["pnl"])
    peak = eq[0]
    max_dd = 0.0
    for e in eq:
        peak = max(peak, e)
        if peak > 0:
            dd = peak - e
            max_dd = max(max_dd, dd)
    return {
        "n": len(trades),
        "pnl": round(sum(t["pnl"] for t in trades), 2),
        "pf": round(pf, 2),
        "wr": round(wr, 1),
        "dd": round(max_dd, 2),
    }


def _filter_regime(trades, regime):
    return [t for t in trades if t["reg"] == regime]


def _filter_dir(trades, direction):
    """direction: 'LONG' (+1) | 'SHORT' (-1)"""
    d = 1 if direction == "LONG" else -1
    return [t for t in trades if t["dir"] == d]


def _wf_split(trades, k=WF_FOLDS):
    """Split trade list into k contiguous folds by entry_bar order."""
    if len(trades) < k:
        return [trades]
    # Sort by entry_bar
    s = sorted(trades, key=lambda t: t["eb"])
    sz = len(s) // k
    folds = []
    for i in range(k):
        lo = i * sz
        hi = (i + 1) * sz if i < k - 1 else len(s)
        folds.append(s[lo:hi])
    return folds


def _score(stats):
    """Sharpe-ish heuristic: PnL × PF / max(DD, 1)."""
    return stats["pnl"] * stats["pf"] / max(stats["dd"], 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Direction analysis (Part A)
# ─────────────────────────────────────────────────────────────────────────────
def analyse_direction(sym, results_by_dir):
    """results_by_dir: {'BOTH': trades, 'LONG': trades, 'SHORT': trades}
    Returns {regime: {dir_bias dict}} for this symbol.
    """
    out = {}
    both = results_by_dir.get("BOTH") or []
    long_only = results_by_dir.get("LONG") or []
    short_only = results_by_dir.get("SHORT") or []

    for regime in REGIMES:
        both_r = _filter_regime(both, regime)
        if len(both_r) < MIN_TRADES:
            continue
        # From BOTH backtest, split by trade direction.
        long_in_both = _filter_dir(both_r, "LONG")
        short_in_both = _filter_dir(both_r, "SHORT")
        ls = _stats(long_in_both)
        ss = _stats(short_in_both)
        bs = _stats(both_r)

        # Dominance test: one side >= 2× PnL AND >= 2× PF
        rec = None
        if ls["n"] >= MIN_TRADES and ss["n"] >= MIN_TRADES:
            if ls["pnl"] >= 2 * max(ss["pnl"], 0.01) and ls["pf"] >= 2 * max(ss["pf"], 0.01) and ls["pnl"] > 0:
                rec = "LONG"
            elif ss["pnl"] >= 2 * max(ls["pnl"], 0.01) and ss["pf"] >= 2 * max(ls["pf"], 0.01) and ss["pnl"] > 0:
                rec = "SHORT"
        elif ls["n"] >= MIN_TRADES and ls["pnl"] > 0 and ss["pnl"] < 0:
            rec = "LONG"
        elif ss["n"] >= MIN_TRADES and ss["pnl"] > 0 and ls["pnl"] < 0:
            rec = "SHORT"

        # WF check using BOTH journal split into folds, then per-fold check
        wf_passed = None
        if rec:
            folds = _wf_split(both_r)
            ok = 0
            for fold in folds:
                fl = _filter_dir(fold, rec)
                fs = _filter_dir(fold, "SHORT" if rec == "LONG" else "LONG")
                fl_st = _stats(fl)
                fs_st = _stats(fs)
                if fl_st["n"] >= 2 and fl_st["pnl"] > fs_st["pnl"] and fl_st["pf"] >= 1.0:
                    ok += 1
            wf_passed = ok >= 3

        out[regime] = {
            "both": bs,
            "long": ls,
            "short": ss,
            "recommend": rec,
            "wf_passed": wf_passed,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Risk analysis (Part B)
# ─────────────────────────────────────────────────────────────────────────────
def analyse_risk(sym, regime, baseline_trades, risk_results, current_cap):
    """risk_results: {risk_val: trades}
    Returns {risk_cap dict} for this (sym, regime) cell.
    """
    # Score each candidate using regime-filtered trades
    scored = {}
    for r, trades in risk_results.items():
        rtr = _filter_regime(trades, regime)
        st = _stats(rtr)
        scored[r] = {"stats": st, "score": _score(st) if st["n"] >= MIN_TRADES else -1e9}

    if not scored:
        return None

    # Best score, but if multiple risks tie (within 1% — caps that don't bind
    # produce identical trades), prefer the HIGHEST risk value (least
    # constraining = closest to "no cap"). This avoids spurious "lower risk!"
    # recommendations driven by floating-point dust.
    top_score = max(s["score"] for s in scored.values())
    tied = [k for k, v in scored.items() if v["score"] >= top_score * 0.99 and v["stats"]["n"] >= MIN_TRADES]
    if not tied:
        return None
    best_r = max(tied)  # highest risk among tied scores
    best_st = scored[best_r]["stats"]

    # WF check: split best_r's regime trades into 5 folds, require >=3 PF>1.3
    best_trades = _filter_regime(risk_results[best_r], regime)
    folds = _wf_split(best_trades)
    ok = 0
    for fold in folds:
        st = _stats(fold)
        if st["n"] >= 2 and st["pf"] >= 1.3:
            ok += 1
    wf_passed = ok >= 3

    # Compute score for current cap (if set) for comparison
    cur_score = None
    if current_cap in scored:
        cur_score = scored[current_cap]["score"]
    elif current_cap is None:
        # When no cap is set, behavior matches the highest risk run (cap is
        # non-binding), so use that as the reference score.
        cur_score = scored[max(scored.keys())]["score"]

    return {
        "current": current_cap,
        "recommend": best_r,
        "score": round(scored[best_r]["score"], 2),
        "current_score": round(cur_score, 2) if cur_score is not None else None,
        "pnl": best_st["pnl"],
        "pf": best_st["pf"],
        "trades": best_st["n"],
        "wf_passed": wf_passed,
        "all_scores": {str(k): round(v["score"], 2) for k, v in scored.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from config import SYMBOLS
    # Load current direction/risk cap state for diffing
    try:
        import auto_tuned
        cur_dir_bias = getattr(auto_tuned, "DIRECTION_BIAS_REGIME_AUTO", {})
        cur_risk_cap = getattr(auto_tuned, "RISK_CAP_REGIME_AUTO", {})
    except Exception:
        cur_dir_bias = {}
        cur_risk_cap = {}

    symbols = list(SYMBOLS.keys())
    print(f"[{time.strftime('%H:%M:%S')}] tuning {len(symbols)} symbols × {len(REGIMES)} regimes")
    print(f"  symbols: {symbols}")
    t_start = time.time()

    # ── Phase 1: direction backtests ──────────────────────────────────
    # Per symbol: 3 variants (LONG, SHORT, BOTH). BOTH gives us the data
    # for both direction split + later risk analysis baseline.
    dir_jobs = []
    for sym in symbols:
        for d in DIR_CHOICES:
            dir_jobs.append(("dir", sym, {"force": d}))

    print(f"\n[{time.strftime('%H:%M:%S')}] PHASE 1: {len(dir_jobs)} direction backtests")
    t0 = time.time()
    dir_data = {}  # {sym: {dir: trades}}
    with mp.Pool(processes=8) as pool:
        for mode, sym, payload, trades in pool.imap_unordered(_run_one, dir_jobs):
            if isinstance(trades, dict) and "error" in trades:
                print(f"  ERR {sym}/{payload}: {trades['error']}")
                continue
            dir_data.setdefault(sym, {})[payload["force"]] = trades or []
    print(f"  done in {round(time.time()-t0, 1)}s")

    # ── Phase 1 analysis ──────────────────────────────────────────────
    dir_analysis = {}  # {sym: {regime: dir_bias_dict}}
    for sym in symbols:
        dir_analysis[sym] = analyse_direction(sym, dir_data.get(sym, {}))

    # ── Phase 2: risk cap backtests ───────────────────────────────────
    # Only test cells where (current cap exists) OR (direction bias chosen
    # in this session) OR (regime has >=10 trades in BOTH backtest).
    risk_jobs = []
    risk_cells = []  # list of (sym, regime) to evaluate
    for sym in symbols:
        for regime in REGIMES:
            has_cur = bool(cur_risk_cap.get(sym, {}).get(regime))
            dir_rec = dir_analysis.get(sym, {}).get(regime, {}).get("recommend")
            both_r = _filter_regime(dir_data.get(sym, {}).get("BOTH", []), regime)
            if not (has_cur or dir_rec or len(both_r) >= MIN_TRADES):
                continue
            risk_cells.append((sym, regime))
            for rv in RISK_GRID:
                risk_jobs.append(("risk", sym, {"regime": regime, "risk": rv}))

    print(f"\n[{time.strftime('%H:%M:%S')}] PHASE 2: {len(risk_jobs)} risk backtests over {len(risk_cells)} cells")
    t0 = time.time()
    risk_data = {}  # {(sym, regime): {risk: trades}}
    with mp.Pool(processes=8) as pool:
        for mode, sym, payload, trades in pool.imap_unordered(_run_one, risk_jobs):
            if isinstance(trades, dict) and "error" in trades:
                print(f"  ERR {sym}/{payload}: {trades['error']}")
                continue
            key = (sym, payload["regime"])
            risk_data.setdefault(key, {})[payload["risk"]] = trades or []
    print(f"  done in {round(time.time()-t0, 1)}s")

    # ── Phase 2 analysis ──────────────────────────────────────────────
    risk_analysis = {}  # {sym: {regime: risk_cap_dict}}
    for (sym, regime) in risk_cells:
        cur_cap = cur_risk_cap.get(sym, {}).get(regime)
        baseline_trades = dir_data.get(sym, {}).get("BOTH", [])
        res = analyse_risk(sym, regime, baseline_trades, risk_data.get((sym, regime), {}), cur_cap)
        if res is None:
            continue
        risk_analysis.setdefault(sym, {})[regime] = res

    # ── Merge into final JSON ─────────────────────────────────────────
    out = {}
    for sym in symbols:
        sym_out = {}
        for regime in REGIMES:
            d = dir_analysis.get(sym, {}).get(regime)
            r = risk_analysis.get(sym, {}).get(regime)
            cell = {}
            if d:
                cell["dir_bias"] = {
                    "current": cur_dir_bias.get(sym, {}).get(regime),
                    "recommend": d.get("recommend"),
                    "long_pnl": d["long"]["pnl"],
                    "short_pnl": d["short"]["pnl"],
                    "long_pf": d["long"]["pf"],
                    "short_pf": d["short"]["pf"],
                    "long_n": d["long"]["n"],
                    "short_n": d["short"]["n"],
                    "wf_passed": d.get("wf_passed"),
                }
            if r:
                cell["risk_cap"] = {
                    "current": r["current"],
                    "recommend": r["recommend"],
                    "score": r["score"],
                    "current_score": r.get("current_score"),
                    "pnl": r["pnl"],
                    "pf": r["pf"],
                    "trades": r["trades"],
                    "wf_passed": r["wf_passed"],
                    "all_scores": r["all_scores"],
                }
            if cell:
                sym_out[regime] = cell
        if sym_out:
            out[sym] = sym_out

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n[{time.strftime('%H:%M:%S')}] wrote {OUT_JSON}")
    print(f"total elapsed: {round(time.time() - t_start, 1)}s")

    # Brief summary of recommendations
    print("\n── DIRECTION BIAS RECOMMENDATIONS ──")
    n_dir = 0
    for sym, regs in out.items():
        for regime, cell in regs.items():
            db = cell.get("dir_bias", {})
            rec = db.get("recommend")
            if rec and rec != db.get("current"):
                n_dir += 1
                print(f"  {sym}/{regime}: {db.get('current')} → {rec}  (L={db['long_pnl']:+.0f}/P{db['long_pf']:.2f}  S={db['short_pnl']:+.0f}/P{db['short_pf']:.2f}  WF={db.get('wf_passed')})")
    print(f"  total: {n_dir} dir changes")

    print("\n── RISK CAP RECOMMENDATIONS (≥10% score gain over current, WF-passed) ──")
    BASE_RISK = 0.8  # symbol default risk_pct in BT
    n_risk = 0
    for sym, regs in out.items():
        for regime, cell in regs.items():
            rc = cell.get("risk_cap", {})
            cur = rc.get("current")
            rec = rc.get("recommend")
            if rec is None or not rc.get("wf_passed"):
                continue
            # Skip when current=None and recommended >= base risk (= no behavior change).
            if cur is None and rec >= BASE_RISK:
                continue
            if rec == cur:
                continue
            # Require score improvement >= 10% over current
            cur_score = rc.get("current_score") or 0.0
            new_score = rc.get("score") or 0.0
            if cur_score > 0 and new_score < cur_score * 1.10:
                continue
            n_risk += 1
            print(f"  {sym}/{regime}: {cur} → {rec}  (score {cur_score:.0f} → {new_score:.0f} pnl={rc['pnl']:+.0f} pf={rc['pf']:.2f} n={rc['trades']})")
    print(f"  total: {n_risk} risk changes (WF-passed, ≥10% score gain)")


if __name__ == "__main__":
    main()
