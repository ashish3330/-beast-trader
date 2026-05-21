#!/usr/bin/env python3 -B
"""
METALS / CRYPTO / ENERGY SL × trail tune — 2026-05-21.

Per-symbol Cartesian sweep of:
  SL ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
  Trail ∈ {_TIGHT_LOCK, _WIDE_RUNNER, _AGGR_LOCK, _RUNNER_NO_BE, _WIDE_RUNNER_BE07}

Symbols: XAUUSD, XAGUSD, BTCUSD, ETHUSD, BCHUSD, XPTUSD (→ XPTUSD.r), UKOUSD, USOIL (→ USOUSD).

Winner criteria (high-vol assets, looser than equity):
  PF >= 1.5 AND trades >= 20
  Δ$ vs baseline >= +$40
  WALK-FORWARD 5 folds (trailing 60/90/120/150/180 days): >=3/5 positive AND avg PF > 1.2

Output: tune_session_20260521/metals_crypto_sl_trail.json
"""
import json
import os
import sys
import time
import importlib
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "tune_session_20260521" / "metals_crypto_sl_trail.json"

# --- Trail profiles (verbatim from auto_tuned.py) ---
TRAIL_PROFILES = {
    "_TIGHT_LOCK":       [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_WIDE_RUNNER":      [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)],
    "_AGGR_LOCK":        [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8), (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "_RUNNER_NO_BE":     [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5), (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)],
    "_WIDE_RUNNER_BE07": [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)],
}

SL_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
WF_FOLDS = [60, 90, 120, 150, 180]
TUNE_DAYS = 180

# Symbols (task → cache symbol mapping)
SYMBOLS = {
    "XAUUSD":   "XAUUSD",
    "XAGUSD":   "XAGUSD",
    "BTCUSD":   "BTCUSD",
    "ETHUSD":   "ETHUSD",
    "BCHUSD":   "BCHUSD",
    "XPTUSD":   "XPTUSD.r",
    "UKOUSD":   "UKOUSD",
    "USOIL":    "USOUSD",  # USOUSD is WTI crude in this universe
}

# Winner gates
MIN_TRADES = 20
MIN_PF = 1.5
MIN_WF_PF = 1.2
MIN_WF_POS = 3
MIN_DELTA = 40.0


def _bt(sym_cache, days, sl_mult, trail):
    """Backtest with SL & trail overridden across all regimes."""
    import config as cfg
    importlib.reload(cfg)
    cfg.SYMBOL_ATR_SL_OVERRIDE = dict(cfg.SYMBOL_ATR_SL_OVERRIDE)
    cfg.SYMBOL_ATR_SL_OVERRIDE[sym_cache] = sl_mult
    # Clear per-regime SL so per-symbol value wins
    cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME = dict(cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME)
    cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME[sym_cache] = {}
    cfg.SYMBOL_REGIME_TRAIL_OVERRIDE = dict(cfg.SYMBOL_REGIME_TRAIL_OVERRIDE)
    cfg.SYMBOL_REGIME_TRAIL_OVERRIDE[sym_cache] = {
        "trending": trail, "ranging": trail, "volatile": trail, "low_vol": trail,
    }
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    return bt.backtest_symbol(sym_cache, days=days, verbose=False)


def _baseline(sym_cache):
    """Baseline = current live config (no overrides)."""
    import config as cfg
    importlib.reload(cfg)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    return bt.backtest_symbol(sym_cache, days=TUNE_DAYS, verbose=False)


def _sweep_one(args):
    sym_key, sym_cache, sl, name = args
    trail = TRAIL_PROFILES[name]
    try:
        r = _bt(sym_cache, TUNE_DAYS, sl, trail)
    except Exception as e:
        return {"sym_key": sym_key, "sl_mult": sl, "trail_name": name,
                "error": str(e), "ok": False}
    if not r:
        return {"sym_key": sym_key, "sl_mult": sl, "trail_name": name,
                "error": "no_data", "ok": False}
    return {
        "sym_key": sym_key,
        "sl_mult": sl,
        "trail_name": name,
        "pnl": float(r.get("pnl", 0)),
        "pf": float(r.get("pf", 0)),
        "wr": float(r.get("wr", 0)),
        "n": int(r.get("trades", 0)),
        "dd": float(r.get("dd", 0)),
        "avg_r": float(r.get("avg_r", 0)),
        "avg_peak_r": float(r.get("avg_peak_r", 0)),
        "avg_giveback": float(r.get("avg_giveback", 0)),
        "ok": True,
    }


def _wf_one(args):
    sym_key, sym_cache, sl, name = args
    trail = TRAIL_PROFILES[name]
    folds = []
    for d in WF_FOLDS:
        try:
            r = _bt(sym_cache, d, sl, trail)
        except Exception as e:
            return {"sym_key": sym_key, "sl_mult": sl, "trail_name": name,
                    "wf_error": str(e), "wf": None}
        if not r:
            folds.append({"days": d, "pnl": 0.0, "pf": 0.0, "n": 0})
            continue
        folds.append({
            "days": d,
            "pnl": float(r.get("pnl", 0)),
            "pf": float(r.get("pf", 0)),
            "n": int(r.get("trades", 0)),
        })
    avg_pf = round(sum(f["pf"] for f in folds) / max(1, len(folds)), 3)
    pos = sum(1 for f in folds if f["pnl"] > 0)
    return {
        "sym_key": sym_key,
        "sl_mult": sl,
        "trail_name": name,
        "wf": {"folds": folds, "avg_pf": avg_pf, "pos_folds": pos},
    }


def main():
    t0 = time.time()
    print(f"\nMETALS/CRYPTO/ENERGY SL × Trail tune  —  {datetime.now().isoformat(timespec='seconds')}")
    print(f"Symbols: {list(SYMBOLS)}")
    print(f"SL grid: {SL_GRID}")
    print(f"Trail profiles: {list(TRAIL_PROFILES)}")
    print(f"WF folds (trailing days): {WF_FOLDS}")
    print(f"Output: {OUT_PATH}\n")

    # ---- 1. Baseline per symbol ----
    print("[A] Baselines (current live config):")
    baselines = {}
    for sym_key, sym_cache in SYMBOLS.items():
        r = _baseline(sym_cache)
        if not r:
            print(f"  {sym_key:8s} ({sym_cache}): NO DATA")
            baselines[sym_key] = {"symbol": sym_cache, "trades": 0, "status": "INSUFFICIENT_DATA"}
            continue
        n = int(r.get("trades", 0))
        pf = float(r.get("pf", 0))
        pnl = float(r.get("pnl", 0))
        wr = float(r.get("wr", 0))
        dd = float(r.get("dd", 0))
        status = "OK" if n >= MIN_TRADES else "INSUFFICIENT_DATA"
        baselines[sym_key] = {
            "symbol": sym_cache,
            "trades": n,
            "pf": pf,
            "pnl": pnl,
            "wr": wr,
            "dd": dd,
            "status": status,
        }
        print(f"  {sym_key:8s} ({sym_cache:9s}): n={n:3d}  PF={pf:5.2f}  PnL=${pnl:+8.2f}  WR={wr:5.1f}%  DD={dd:4.1f}%  [{status}]")
    print()

    # ---- 2. Cartesian sweep — only for symbols with enough baseline trades ----
    sym_to_tune = [k for k, b in baselines.items() if b.get("status") == "OK"]
    sym_no_tune = [k for k, b in baselines.items() if b.get("status") != "OK"]
    if sym_no_tune:
        print(f"[!] Skipping (insufficient baseline trades): {sym_no_tune}\n")

    jobs = []
    for sym_key in sym_to_tune:
        sym_cache = SYMBOLS[sym_key]
        for sl in SL_GRID:
            for name in TRAIL_PROFILES:
                jobs.append((sym_key, sym_cache, sl, name))
    print(f"[B] Cartesian sweep: {len(jobs)} backtests "
          f"({len(sym_to_tune)} syms × {len(SL_GRID)} SL × {len(TRAIL_PROFILES)} trails) ...")

    workers = max(2, min(8, cpu_count() or 4))
    sweep_results = []
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_sweep_one, jobs), 1):
            sweep_results.append(res)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i:3d}/{len(jobs)}  ({time.time() - t0:.0f}s)")
    print()

    # Group results by symbol
    by_sym = {k: [] for k in sym_to_tune}
    for r in sweep_results:
        if r.get("ok"):
            by_sym[r["sym_key"]].append(r)

    # ---- 3. Per-symbol top picks + WF ----
    print("[C] Per-symbol top candidates (sorted by PnL desc, applying winner gates):\n")
    wf_jobs = []
    sym_top10 = {}
    for sym_key in sym_to_tune:
        rs = by_sym[sym_key]
        # winner gates pre-WF: trades >= 20, PF >= 1.5
        good = [r for r in rs if r["n"] >= MIN_TRADES and r["pf"] >= MIN_PF]
        good.sort(key=lambda x: -x["pnl"])
        top = good[:8]  # WF-test top 8
        sym_top10[sym_key] = top
        print(f"  {sym_key:8s}: {len(rs)} cfg ran, {len(good)} pass PF/n gates, top {len(top)} -> WF")
        for r in top[:5]:
            print(f"      SL={r['sl_mult']:.2f}  {r['trail_name']:18s}  n={r['n']:3d}  PF={r['pf']:5.2f}  PnL=${r['pnl']:+8.2f}  DD={r['dd']:4.1f}%")
        for r in top:
            wf_jobs.append((sym_key, SYMBOLS[sym_key], r["sl_mult"], r["trail_name"]))
    print()

    # ---- 4. WF validation ----
    print(f"[D] WF validation: {len(wf_jobs)} configs across trailing 60/90/120/150/180d ...")
    wf_results = {}
    with Pool(workers) as pool:
        for i, w in enumerate(pool.imap_unordered(_wf_one, wf_jobs), 1):
            key = (w["sym_key"], w["sl_mult"], w["trail_name"])
            wf_results[key] = w["wf"]
            if i % 10 == 0 or i == len(wf_jobs):
                print(f"  {i:3d}/{len(wf_jobs)}  ({time.time() - t0:.0f}s)")
    print()

    # ---- 5. Pick winners ----
    print("[E] Apply WF gates + delta vs baseline:\n")
    winners = {}
    per_sym_details = {}
    for sym_key in sym_to_tune:
        base_pnl = baselines[sym_key]["pnl"]
        candidates = []
        for r in sym_top10[sym_key]:
            wf = wf_results.get((sym_key, r["sl_mult"], r["trail_name"]))
            if wf is None:
                continue
            delta = r["pnl"] - base_pnl
            passes_wf = wf["avg_pf"] >= MIN_WF_PF and wf["pos_folds"] >= MIN_WF_POS
            passes_delta = delta >= MIN_DELTA
            cand = {
                **r, "wf": wf, "delta": round(delta, 2),
                "wf_ok": bool(passes_wf),
                "delta_ok": bool(passes_delta),
                "winner": bool(passes_wf and passes_delta),
            }
            candidates.append(cand)
        candidates.sort(key=lambda x: -x["pnl"])
        per_sym_details[sym_key] = candidates
        winner = next((c for c in candidates if c["winner"]), None)
        if winner:
            winners[sym_key] = {
                "best_sl": winner["sl_mult"],
                "best_trail_name": winner["trail_name"],
                "best_trail_profile": TRAIL_PROFILES[winner["trail_name"]],
                "baseline_pnl": round(base_pnl, 2),
                "best_pnl": round(winner["pnl"], 2),
                "delta": winner["delta"],
                "pf": winner["pf"],
                "wr": winner["wr"],
                "n": winner["n"],
                "dd": winner["dd"],
                "wf_avg_pf": winner["wf"]["avg_pf"],
                "wf_pos_folds": winner["wf"]["pos_folds"],
                "wf_folds": winner["wf"]["folds"],
                "status": "WINNER",
            }
            print(f"  {sym_key:8s} WINNER  SL={winner['sl_mult']:.2f} {winner['trail_name']:18s} "
                  f"PnL=${winner['pnl']:+8.2f} (Δ${winner['delta']:+.2f}) PF={winner['pf']:5.2f} "
                  f"WF avg_pf={winner['wf']['avg_pf']:.2f} {winner['wf']['pos_folds']}/5")
        else:
            print(f"  {sym_key:8s} NO_WINNER (top: ", end="")
            if candidates:
                c = candidates[0]
                wf_str = f"WF {c['wf']['avg_pf']:.2f} {c['wf']['pos_folds']}/5"
                print(f"SL={c['sl_mult']:.2f} {c['trail_name']} Δ${c['delta']:+.2f} {wf_str})")
            else:
                print("none passed gates)")
    print()

    # ---- 6. Save JSON ----
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tune_days": TUNE_DAYS,
        "wf_folds_days": WF_FOLDS,
        "sl_grid": SL_GRID,
        "trail_profiles": {k: v for k, v in TRAIL_PROFILES.items()},
        "criteria": {
            "min_trades": MIN_TRADES,
            "min_pf": MIN_PF,
            "min_wf_avg_pf": MIN_WF_PF,
            "min_wf_pos_folds": MIN_WF_POS,
            "min_delta_pnl": MIN_DELTA,
        },
        "symbol_mapping": SYMBOLS,
        "baselines": baselines,
        "winners": winners,
        "per_symbol_details": {
            sym_key: {
                "candidates": [
                    {
                        "sl_mult": c["sl_mult"], "trail_name": c["trail_name"],
                        "pnl": round(c["pnl"], 2), "pf": c["pf"], "wr": c["wr"],
                        "n": c["n"], "dd": c["dd"],
                        "delta": c["delta"],
                        "wf_avg_pf": c["wf"]["avg_pf"], "wf_pos_folds": c["wf"]["pos_folds"],
                        "wf_folds": c["wf"]["folds"],
                        "wf_ok": c["wf_ok"], "delta_ok": c["delta_ok"],
                        "winner": c["winner"],
                    }
                    for c in per_sym_details[sym_key]
                ],
                "all_grid_results": [
                    {k: v for k, v in r.items() if k not in ("sym_key",)}
                    for r in by_sym[sym_key]
                ],
            }
            for sym_key in sym_to_tune
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"[F] Saved -> {OUT_PATH}  ({time.time() - t0:.0f}s total)")
    return out


if __name__ == "__main__":
    main()
