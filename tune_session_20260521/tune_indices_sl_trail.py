#!/usr/bin/env python3 -B
"""INDEX symbols SL × trail tune — 2026-05-21.

Sweeps SL ∈ {1.5, 2.0, 2.5, 3.0, 3.5} × trail ∈ {_TIGHT_LOCK, _WIDE_RUNNER,
_RANGE_TIGHT, _AGGR_LOCK, _RUNNER_NO_BE} = 25 configs per symbol over 9 index
symbols. Selects per-symbol winners (PF≥2.0, ≥20 trades, max PnL), then
walk-forward validates on 5 ~36d folds (≥3/5 positive + avg PF > 1.5). Only
keeps winners that beat baseline by ≥ $50.

Mutates `config.SYMBOL_ATR_SL_OVERRIDE`, `config.SYMBOL_ATR_SL_OVERRIDE_REGIME`,
`config.SYMBOL_REGIME_TRAIL_OVERRIDE`, then reloads `backtest.v5_backtest`
so `force_trail` is the only active trail. Each worker process holds its
own module state — safe under multiprocessing.

Output JSON: /Users/ashish/Documents/beast-trader/tune_session_20260521/indices_sl_trail.json
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

SYMBOLS = [
    "SWI20.r", "DJ30.r", "NAS100.r", "SP500.r", "JPN225ft",
    "GER40.r", "US2000.r", "UK100.r", "HK50.r",
]

SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]

# Trail profile names → step lists. Pulled from auto_tuned.py at module load.
TRAIL_NAMES = ["_TIGHT_LOCK", "_WIDE_RUNNER", "_RANGE_TIGHT", "_AGGR_LOCK", "_RUNNER_NO_BE"]

DAYS = 180
# Disjoint walk-forward windows: 5 sequential ~36d folds spanning the most
# recent 180d. Fold N covers days [180 - 36*N, 180 - 36*(N-1)] before
# `df["time"].max()`. Fold 5 ends at the most-recent bar, fold 1 is the
# oldest.
WF_NUM_FOLDS = 5
WF_FOLD_DAYS = 36

MIN_TRADES = 20
MIN_PF = 2.0
MIN_DELTA = 50.0
WF_MIN_POS = 3
WF_MIN_AVG_PF = 1.5

OUT_FILE = ROOT / "tune_session_20260521" / "indices_sl_trail.json"
LOG_FILE = ROOT / "tune_session_20260521" / "tune_indices_progress.log"


def _load_trail_profiles():
    """Load trail profile lists from auto_tuned by name."""
    import auto_tuned as _at
    return {name: getattr(_at, name) for name in TRAIL_NAMES}


def _bt_worker(args):
    """Worker: run a single backtest with forced SL + trail across all regimes.

    args = (symbol, days, sl_mult, trail_name, trail_steps, fold_id)
      fold_id = None for in-sample full-window. If fold_id is an int 1..5,
      we slice df to a disjoint 36d window before backtest, fold 5 being most
      recent. We do this by monkey-patching backtest.v5_backtest.load_data.
    Returns (symbol, days, sl_mult, trail_name, fold_id, result_dict or None, err_str or None)
    """
    symbol, days, sl_mult, trail_name, trail_steps, fold_id = args
    try:
        import importlib
        import pandas as pd
        import config as cfg
        importlib.reload(cfg)
        # Force per-symbol SL across all regimes.
        cfg.SYMBOL_ATR_SL_OVERRIDE = dict(cfg.SYMBOL_ATR_SL_OVERRIDE)
        cfg.SYMBOL_ATR_SL_OVERRIDE[symbol] = sl_mult
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME = dict(cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME)
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME[symbol] = {}
        # Force trail profile across all regimes.
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE = dict(cfg.SYMBOL_REGIME_TRAIL_OVERRIDE)
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE[symbol] = {
            "trending": trail_steps,
            "ranging": trail_steps,
            "volatile": trail_steps,
            "low_vol": trail_steps,
        }
        import backtest.v5_backtest as bt
        importlib.reload(bt)

        # Disjoint WF fold: monkey-patch load_data so the BT sees only this
        # fold's window. Fold 5 ends at df.time.max(); fold 1 is the oldest.
        # Fold N covers [max - 36*(NUM-N+1), max - 36*(NUM-N)] days.
        if fold_id is not None:
            orig_load = bt.load_data
            fold_n = fold_id
            num = WF_NUM_FOLDS
            fold_d = WF_FOLD_DAYS

            def load_data_fold(sym, _ignored_days=None):
                df = orig_load(sym, days=None)  # full data
                if df is None or df.empty:
                    return df
                end = df["time"].max()
                # fold_id=5 → window (end - 36d, end]
                # fold_id=4 → window (end - 72d, end - 36d]
                offset_end = (num - fold_n) * fold_d  # days before "end"
                offset_start = offset_end + fold_d
                t_end = end - pd.Timedelta(days=offset_end)
                t_start = end - pd.Timedelta(days=offset_start)
                df = df[(df["time"] > t_start) & (df["time"] <= t_end)].reset_index(drop=True)
                return df

            bt.load_data = load_data_fold
            r = bt.backtest_symbol(symbol, days=None, verbose=False)
        else:
            r = bt.backtest_symbol(symbol, days=days, verbose=False)
        if r is None:
            return (symbol, days, sl_mult, trail_name, fold_id, None, "result_none")
        out = {
            "trades": int(r.get("trades", 0)),
            "pf": float(r.get("pf", 0)),
            "wr": float(r.get("wr", 0)),
            "pnl": float(r.get("pnl", 0)),
            "dd": float(r.get("dd", 0)),
        }
        return (symbol, days, sl_mult, trail_name, fold_id, out, None)
    except Exception as e:
        return (symbol, days, sl_mult, trail_name, fold_id, None,
                f"{type(e).__name__}: {e}\n{traceback.format_exc()[:300]}")


def _baseline_worker(symbol):
    """Run a baseline backtest with the current live config (no overrides).
    Also returns the actual data span in days so we can detect symbols with
    insufficient history for a 180d WF.
    """
    try:
        import importlib
        import config as cfg
        importlib.reload(cfg)
        import backtest.v5_backtest as bt
        importlib.reload(bt)
        df = bt.load_data(symbol, days=None)
        span_days = float((df["time"].max() - df["time"].min()).total_seconds() / 86400) if df is not None and not df.empty else 0.0
        r = bt.backtest_symbol(symbol, days=DAYS, verbose=False)
        if r is None:
            return (symbol, None, span_days, "result_none")
        return (symbol, {
            "trades": int(r.get("trades", 0)),
            "pf": float(r.get("pf", 0)),
            "wr": float(r.get("wr", 0)),
            "pnl": float(r.get("pnl", 0)),
            "dd": float(r.get("dd", 0)),
        }, span_days, None)
    except Exception as e:
        return (symbol, None, 0.0, f"{type(e).__name__}: {e}")


def _log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")

    profiles = _load_trail_profiles()
    _log(f"Loaded {len(profiles)} trail profiles: {list(profiles)}")

    workers = max(2, (os.cpu_count() or 4) - 2)
    _log(f"Pool workers: {workers}")

    t0 = time.time()

    # ── Step 1: baselines ──
    _log("Step 1: baselines (9 BTs)")
    baselines = {}
    spans = {}
    with Pool(workers) as pool:
        for sym, res, span, err in pool.imap_unordered(_baseline_worker, SYMBOLS):
            spans[sym] = span
            if err:
                _log(f"  {sym}: ERR {err}")
                baselines[sym] = None
            else:
                baselines[sym] = res
                _log(f"  {sym}: span={span:5.1f}d  trades={res['trades']:3d} "
                     f"PF={res['pf']:.2f} PnL=${res['pnl']:+8.0f}")
    _log(f"  baselines done in {time.time() - t0:.0f}s\n")

    # ── Step 2: sweep ──
    _log("Step 2: 9 syms × 25 configs = 225 BTs")
    jobs = []
    SPAN_MIN = WF_NUM_FOLDS * WF_FOLD_DAYS  # 180d needed for full WF
    for sym in SYMBOLS:
        if baselines.get(sym) is None:
            continue
        if baselines[sym]["trades"] < MIN_TRADES:
            _log(f"  {sym}: INSUFFICIENT_DATA (baseline {baselines[sym]['trades']} trades)")
            continue
        if spans.get(sym, 0) < SPAN_MIN:
            _log(f"  {sym}: INSUFFICIENT_DATA (span {spans[sym]:.0f}d < {SPAN_MIN}d "
                 f"required for 5x{WF_FOLD_DAYS}d WF)")
            continue
        for sl in SL_GRID:
            for tname in TRAIL_NAMES:
                jobs.append((sym, DAYS, sl, tname, profiles[tname]))
    _log(f"  scheduled {len(jobs)} sweep jobs")

    # Add fold_id=None for in-sample sweep jobs
    jobs = [(s, d, sl, tn, ts, None) for (s, d, sl, tn, ts) in jobs]

    sweep_results = {}  # sym → list of result dicts
    done = 0
    t1 = time.time()
    with Pool(workers) as pool:
        for sym, days, sl, tname, fold, res, err in pool.imap_unordered(_bt_worker, jobs):
            done += 1
            if err:
                _log(f"  [{done}/{len(jobs)}] {sym} SL={sl} {tname}: ERR {err[:120]}")
                continue
            sweep_results.setdefault(sym, []).append({
                "sl": sl, "trail": tname, **res,
            })
            if done % 25 == 0 or done == len(jobs):
                _log(f"  [{done}/{len(jobs)}] elapsed {time.time() - t1:.0f}s")
    _log(f"  sweep done in {time.time() - t1:.0f}s\n")

    # ── Step 3: pick winners ──
    _log("Step 3: pick winners per symbol")
    winners = {}
    for sym in SYMBOLS:
        results = sweep_results.get(sym, [])
        if not results:
            winners[sym] = None
            continue
        # Filter: PF >= MIN_PF, trades >= MIN_TRADES
        eligible = [r for r in results if r["pf"] >= MIN_PF and r["trades"] >= MIN_TRADES]
        if not eligible:
            _log(f"  {sym}: no eligible (PF≥{MIN_PF}, trades≥{MIN_TRADES})")
            winners[sym] = None
            continue
        # Max PnL
        best = max(eligible, key=lambda r: r["pnl"])
        _log(f"  {sym} winner: SL={best['sl']} {best['trail']:14} "
             f"trades={best['trades']} PF={best['pf']:.2f} PnL=${best['pnl']:+.0f}")
        winners[sym] = best
    _log("")

    # ── Step 4: WF validate winners (disjoint 5×36d folds) ──
    _log("Step 4: walk-forward validation (5 disjoint ~36d folds)")
    wf_jobs = []
    for sym, w in winners.items():
        if w is None:
            continue
        for fold in range(1, WF_NUM_FOLDS + 1):
            wf_jobs.append((sym, None, w["sl"], w["trail"], profiles[w["trail"]], fold))
    _log(f"  scheduled {len(wf_jobs)} WF jobs")

    wf_by_sym = {}  # sym → list[{fold, pf, pnl, trades}]
    done = 0
    t2 = time.time()
    with Pool(workers) as pool:
        for sym, days, sl, tname, fold, res, err in pool.imap_unordered(_bt_worker, wf_jobs):
            done += 1
            if err:
                _log(f"  [{done}/{len(wf_jobs)}] {sym} fold={fold} ERR {err[:120]}")
                continue
            wf_by_sym.setdefault(sym, []).append({
                "fold": fold, "pf": res["pf"], "pnl": res["pnl"],
                "trades": res["trades"], "wr": res["wr"],
            })
            if done % 10 == 0 or done == len(wf_jobs):
                _log(f"  [{done}/{len(wf_jobs)}] elapsed {time.time() - t2:.0f}s")
    _log(f"  WF done in {time.time() - t2:.0f}s\n")

    # ── Step 5: assemble output ──
    _log("Step 5: assemble final output")
    out = {}
    for sym in SYMBOLS:
        base = baselines.get(sym)
        if base is None:
            out[sym] = {"status": "NO_BASELINE", "data_span_days": round(spans.get(sym, 0), 1)}
            continue
        if base["trades"] < MIN_TRADES:
            out[sym] = {
                "status": "INSUFFICIENT_DATA",
                "reason": f"baseline only {base['trades']} trades",
                "baseline": base,
                "data_span_days": round(spans.get(sym, 0), 1),
            }
            continue
        if spans.get(sym, 0) < SPAN_MIN:
            out[sym] = {
                "status": "INSUFFICIENT_DATA",
                "reason": f"data span {spans[sym]:.0f}d < {SPAN_MIN}d required",
                "baseline": base,
                "data_span_days": round(spans.get(sym, 0), 1),
            }
            continue
        win = winners.get(sym)
        if win is None:
            out[sym] = {
                "status": "NO_WINNER",
                "baseline": base,
            }
            continue
        folds = sorted(wf_by_sym.get(sym, []), key=lambda x: x["fold"])
        wf_folds_out = [{"fold": f["fold"], "pf": round(f["pf"], 2),
                         "pnl": round(f["pnl"], 2), "trades": f["trades"],
                         "wr": round(f["wr"], 1)} for f in folds]
        pos_folds = sum(1 for f in folds if f["pnl"] > 0)
        avg_pf = round(sum(f["pf"] for f in folds) / max(1, len(folds)), 2)
        # Concentration risk: how much of WF PnL came from the most-recent fold.
        sum_wf_pnl = sum(f["pnl"] for f in folds)
        last_fold_pnl = folds[-1]["pnl"] if folds else 0.0
        fold5_share = round(last_fold_pnl / sum_wf_pnl * 100, 1) if sum_wf_pnl else 0.0
        # Sanity gates beyond the spec'd ones:
        #   • fold5 PnL must be >= 0 (last 36d not bleeding)
        #   • single-fold concentration <= 60% of total (not one-window edge)
        last_fold_positive = last_fold_pnl >= 0
        not_concentrated = abs(fold5_share) <= 60.0
        wf_passed = (
            (pos_folds >= WF_MIN_POS)
            and (avg_pf > WF_MIN_AVG_PF)
            and last_fold_positive
            and not_concentrated
        )
        delta = round(win["pnl"] - base["pnl"], 2)
        recommend = bool(wf_passed and delta >= MIN_DELTA)
        out[sym] = {
            "status": "OK",
            "data_span_days": round(spans.get(sym, 0), 1),
            "baseline": {k: round(base[k], 2) if isinstance(base[k], float) else base[k]
                         for k in base},
            "winner": {
                "sl": win["sl"], "trail": win["trail"],
                "trades": win["trades"], "pf": round(win["pf"], 2),
                "wr": round(win["wr"], 1), "pnl": round(win["pnl"], 2),
                "dd": round(win["dd"], 2),
            },
            "wf_folds": wf_folds_out,
            "wf_pos_folds": pos_folds,
            "wf_avg_pf": avg_pf,
            "wf_last_fold_pnl": round(last_fold_pnl, 2),
            "wf_last_fold_positive": last_fold_positive,
            "wf_fold5_pnl_share_pct": fold5_share,
            "wf_concentrated": not not_concentrated,
            "wf_passed": wf_passed,
            "delta_pnl": delta,
            "recommend_ship": recommend,
        }
        _log(f"  {sym}: WF {pos_folds}/5 avg_pf={avg_pf} "
             f"fold5_share={fold5_share:+.1f}% Δ=${delta:+.0f} ship={recommend}")

    # Also embed all sweep results & some metadata for traceability.
    final = {
        "_meta": {
            "ts": datetime.now().isoformat(),
            "days": DAYS,
            "wf_num_folds": WF_NUM_FOLDS,
            "wf_fold_days": WF_FOLD_DAYS,
            "sl_grid": SL_GRID,
            "trail_names": TRAIL_NAMES,
            "filters": {
                "min_trades": MIN_TRADES, "min_pf": MIN_PF,
                "min_delta": MIN_DELTA,
                "wf_min_positive_folds": WF_MIN_POS,
                "wf_min_avg_pf": WF_MIN_AVG_PF,
                "wf_max_fold5_share_pct": 60.0,
                "wf_require_last_fold_positive": True,
            },
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "symbols": out,
        "sweep_full": {sym: sweep_results.get(sym, []) for sym in SYMBOLS},
    }
    OUT_FILE.write_text(json.dumps(final, indent=2))
    _log(f"\nDone in {time.time() - t0:.0f}s → {OUT_FILE}")


if __name__ == "__main__":
    main()
