#!/usr/bin/env python3 -B
"""SP500.r per-symbol full-knob tuner — 2026-05-22.

Coordinate-descent search across 7 dimensions, anchored on the live baseline:

  1. SL ATR mult       ∈ {0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0}
  2. Trail profile     ∈ {_TIGHT_LOCK, _WIDE_RUNNER, _RANGE_TIGHT, _TREND_LOOSE,
                          _AGGR_LOCK, _RUNNER_NO_BE, _WIDE_RUNNER_BE07}
  3. Pullback ATR      ∈ {0.4, 0.5, 0.6, 0.8, 1.0, 1.2}
  4. Pullback wait     ∈ {3, 4, 5, 6}
  5. VWAP buffer ATR   ∈ {disabled, 0.3, 0.5, 0.7, 1.0}
  6. min_quality (all regimes) ∈ {28, 30, 33, 35, 38, 40}
  7. Toxic hours       ∈ [{14}, {}, {14, 16}]

Strategy:
  Phase A — full SL × trail grid (9 × 7 = 63 BTs). Top-5 → Phase B.
  Phase B — coordinate descent: for each top-5 (SL, trail) winner, tune
            (PB-ATR × PB-wait) [24 BTs], then VWAP × minQ [30 BTs], then
            toxic-hours [3 BTs]. Top-1 of each B chain → Phase C.
  Phase C — top-3 finalists × 5-fold disjoint WF (15 BTs).

Ship rule:
  - Δ ≥ $50 vs baseline
  - WF ≥ 3/5 folds positive

READ-ONLY: never writes auto_tuned/config; outputs only to
/Users/ashish/Documents/beast-trader/per_symbol_tune_20260522/SP500.r.json
and SP500.r.md
"""
import json
import os
import sys
import time
import traceback
import types
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

SYMBOL = "SP500.r"
DAYS = 180
OUT_JSON = ROOT / "per_symbol_tune_20260522" / "SP500.r.json"
OUT_MD = ROOT / "per_symbol_tune_20260522" / "SP500.r.md"
LOG_FILE = ROOT / "per_symbol_tune_20260522" / "tune_sp500_r.log"

# ── Dimensions ──
SL_GRID = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
TRAIL_NAMES = [
    "_TIGHT_LOCK", "_WIDE_RUNNER", "_RANGE_TIGHT", "_TREND_LOOSE",
    "_AGGR_LOCK", "_RUNNER_NO_BE", "_WIDE_RUNNER_BE07",
]
PB_ATR_GRID = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_GRID = [3, 4, 5, 6]
VWAP_GRID = [
    ("disabled", 9999.0),  # huge buffer => filter never triggers
    ("0.3", 0.3),
    ("0.5", 0.5),
    ("0.7", 0.7),
    ("1.0", 1.0),
]
MIN_Q_GRID = [28, 30, 33, 35, 38, 40]
TOXIC_GRID = [
    ("h14", frozenset({14})),
    ("none", frozenset()),
    ("h14_16", frozenset({14, 16})),
]

# Ship gates
MIN_DELTA = 50.0
WF_MIN_POS = 3
WF_NUM_FOLDS = 5
WF_FOLD_DAYS = 36

# Phase A picks
PHASE_A_TOP_K = 5
PHASE_B_TOP_K = 3


# ────────────────────────────────────────────────────────────────────
# Module-clone caching: one cloned `bt` per VWAP buffer literal.
# Each process holds its own cache (multiprocessing-safe).
# ────────────────────────────────────────────────────────────────────
_BT_CACHE = {}
_BT_BASE_SRC = None
_BT_BASE_FILE = None


def _get_bt(vwap_buf):
    """Return a cloned backtest.v5_backtest module whose hard-coded VWAP
    buffer (0.5) is rewritten to `vwap_buf`.
    """
    global _BT_BASE_SRC, _BT_BASE_FILE
    key = round(vwap_buf, 4)
    if key in _BT_CACHE:
        return _BT_CACHE[key]
    if _BT_BASE_SRC is None:
        import backtest.v5_backtest as _ref
        _BT_BASE_FILE = _ref.__file__
        with open(_BT_BASE_FILE) as f:
            _BT_BASE_SRC = f.read()
        ORIG = 'atr_buf = float(ind["at"][bi]) * 0.5'
        if _BT_BASE_SRC.count(ORIG) != 1:
            raise RuntimeError(
                f"VWAP buffer literal '{ORIG}' not unique in v5_backtest.py "
                f"(count={_BT_BASE_SRC.count(ORIG)})"
            )
    src = _BT_BASE_SRC.replace(
        'atr_buf = float(ind["at"][bi]) * 0.5',
        f'atr_buf = float(ind["at"][bi]) * {vwap_buf}',
    )
    mod = types.ModuleType(f"bt_vwap_{key}")
    mod.__file__ = _BT_BASE_FILE
    exec(compile(src, _BT_BASE_FILE, "exec"), mod.__dict__)
    _BT_CACHE[key] = mod
    return mod


def _load_trail_profile(name):
    """Return live-format trail profile (R, type, param) for a given name."""
    import auto_tuned as _at
    if not hasattr(_at, name):
        raise RuntimeError(f"Trail profile {name} missing from auto_tuned")
    return getattr(_at, name)


def _live_to_bt_trail(steps):
    """Convert auto_tuned (R, type, param) → backtest (R, param, type)."""
    out = []
    for tup in steps:
        if len(tup) == 3:
            r, t, p = tup
            out.append((r, p, t))
    return out


# ────────────────────────────────────────────────────────────────────
# Single-BT worker. Applies all 7 knobs.
# ────────────────────────────────────────────────────────────────────
def _bt_one(args):
    """Run one backtest with the supplied 7-knob config + optional fold_id.
    Returns dict with metrics or {'err': str}.
    """
    (sl, trail_name, pb_atr, pb_wait, vwap_label, vwap_buf,
     min_q, toxic_label, toxic_set, fold_id) = args
    try:
        # ── 1. Clone config + reload bt with patched VWAP literal ──
        import config as cfg
        # Per-process: mutate config attributes BEFORE bt import (pullback
        # constants are imported inside the entry loop each iteration).
        cfg.PULLBACK_ATR_RETRACE = float(pb_atr)
        cfg.PULLBACK_MAX_WAIT_BARS = int(pb_wait)

        bt = _get_bt(vwap_buf)

        # ── 2. Apply per-symbol overrides on the cloned bt module ──
        # SL: per-symbol override (force, regardless of regime)
        bt.SL_OVERRIDE = dict(bt.SL_OVERRIDE)
        bt.SL_OVERRIDE[SYMBOL] = float(sl)
        bt.SL_OVERRIDE_REGIME = dict(bt.SL_OVERRIDE_REGIME)
        bt.SL_OVERRIDE_REGIME[SYMBOL] = {}  # clear regime-specific overrides

        # Trail: clear regime-specific, set per-symbol (force_trail is fallback)
        bt.TRAIL_OVERRIDE = dict(bt.TRAIL_OVERRIDE)
        steps_live = _load_trail_profile(trail_name)
        steps_bt = _live_to_bt_trail(steps_live)
        bt.TRAIL_OVERRIDE[SYMBOL] = steps_bt
        bt.TRAIL_OVERRIDE_REGIME = dict(bt.TRAIL_OVERRIDE_REGIME)
        bt.TRAIL_OVERRIDE_REGIME[SYMBOL] = {
            "trending": steps_bt, "ranging": steps_bt,
            "volatile": steps_bt, "low_vol": steps_bt,
        }

        # Toxic hours: BT reads module-level TOXIC_HOURS. Union live's
        # {1,2,3,4} with the requested per-symbol set since the BT engine
        # doesn't separately load TOXIC_HOURS_PER_SYMBOL.
        base_toxic = {1, 2, 3, 4}
        bt.TOXIC_HOURS = set(base_toxic) | set(toxic_set)
        # SP500.r is not in TOXIC_EXEMPT (only crypto + JPN225ft are), so we
        # don't need to clear that.

        # ── 3. Build params with min_quality override ──
        p = {
            "min_quality": {
                "trending": int(min_q),
                "ranging": int(min_q),
                "volatile": int(min_q),
                "low_vol": int(min_q),
            },
        }

        # ── 4. Optional fold-slicing ──
        if fold_id is not None:
            import pandas as pd
            orig_load = bt.load_data
            fold_n = int(fold_id)
            num = WF_NUM_FOLDS
            fold_d = WF_FOLD_DAYS

            def load_data_fold(sym, _ignored_days=None):
                df = orig_load(sym, days=None)
                if df is None or df.empty:
                    return df
                end = df["time"].max()
                offset_end = (num - fold_n) * fold_d
                offset_start = offset_end + fold_d
                t_end = end - pd.Timedelta(days=offset_end)
                t_start = end - pd.Timedelta(days=offset_start)
                df = df[(df["time"] > t_start) & (df["time"] <= t_end)].reset_index(drop=True)
                return df

            bt.load_data = load_data_fold
            r = bt.backtest_symbol(SYMBOL, days=None, params=p, verbose=False)
        else:
            r = bt.backtest_symbol(SYMBOL, days=DAYS, params=p, verbose=False)

        if r is None:
            return {"err": "result_none"}
        return {
            "trades": int(r.get("trades", 0)),
            "pf":     float(r.get("pf", 0)),
            "wr":     float(r.get("wr", 0)),
            "pnl":    float(r.get("pnl", 0)),
            "dd":     float(r.get("dd", 0)),
        }
    except Exception as e:
        return {"err": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:400]}"}


def _bt_baseline(args=None):
    """Live baseline: read auto_tuned/config as-is, no overrides."""
    try:
        import importlib
        import config as cfg
        importlib.reload(cfg)
        import backtest.v5_backtest as bt
        importlib.reload(bt)
        r = bt.backtest_symbol(SYMBOL, days=DAYS, verbose=False)
        if r is None:
            return {"err": "result_none"}
        return {
            "trades": int(r.get("trades", 0)),
            "pf":     float(r.get("pf", 0)),
            "wr":     float(r.get("wr", 0)),
            "pnl":    float(r.get("pnl", 0)),
            "dd":     float(r.get("dd", 0)),
        }
    except Exception as e:
        return {"err": f"{type(e).__name__}: {e}"}


def _log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _fmt_cfg(c):
    return (f"SL={c['sl']} {c['trail']:18} PB=({c['pb_atr']},{c['pb_wait']}) "
            f"VWAP={c['vwap_label']:8} mQ={c['min_q']} tx={c['toxic_label']}")


# ────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────
def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")

    t0 = time.time()
    workers = max(2, (os.cpu_count() or 4) - 2)
    _log(f"SP500.r tuner — workers={workers}")

    # ── Anchor (live-config defaults that all phases use unless tuned) ──
    # Baseline pullback per live config: PULLBACK_ATR_RETRACE=0.8, MAX_WAIT=5
    # Baseline VWAP buffer in BT: 0.5 (hardcoded)
    # Baseline min_q for SP500.r (live config SIGNAL_QUALITY_SYMBOL):
    #   trending=40, ranging=45, volatile=35, low_vol=40 (mean ≈ 40)
    # Toxic per-symbol from live config: {14} (but BT doesn't auto-apply it)
    ANCHOR = {
        "sl": 2.0,           # current live SP500.r SL
        "trail": "_TIGHT_LOCK",  # current live SP500.r trail (volatile)
        "pb_atr": 0.8,
        "pb_wait": 5,
        "vwap_label": "0.5",
        "vwap_buf": 0.5,
        "min_q": 35,         # mid of live SIGNAL_QUALITY_SYMBOL
        "toxic_label": "h14",
        "toxic_set": frozenset({14}),
    }

    # ── Phase 0: baseline (live config, no overrides) ──
    _log("Phase 0: live-config baseline")
    with Pool(1) as pool:
        baseline = pool.apply(_bt_baseline)
    if "err" in baseline:
        _log(f"  baseline FAILED: {baseline['err']}")
        sys.exit(1)
    _log(f"  baseline: trades={baseline['trades']} PF={baseline['pf']:.2f} "
         f"PnL=${baseline['pnl']:+.0f} DD={baseline['dd']:.2f}%")

    # ── Phase A: SL × trail (anchor for everything else) ──
    _log(f"\nPhase A: SL × trail grid ({len(SL_GRID)}×{len(TRAIL_NAMES)} = "
         f"{len(SL_GRID)*len(TRAIL_NAMES)} BTs)")
    a_jobs = []
    for sl in SL_GRID:
        for tname in TRAIL_NAMES:
            a_jobs.append((
                sl, tname,
                ANCHOR["pb_atr"], ANCHOR["pb_wait"],
                ANCHOR["vwap_label"], ANCHOR["vwap_buf"],
                ANCHOR["min_q"],
                ANCHOR["toxic_label"], ANCHOR["toxic_set"],
                None,  # full window
            ))
    a_results = []
    t_a = time.time()
    with Pool(workers) as pool:
        a_raw = list(pool.imap(_bt_one, a_jobs))
    for job, res in zip(a_jobs, a_raw):
        (sl, tname, pb_atr, pb_wait, vlab, vbuf, mq, tlab, tset, _) = job
        if "err" in res:
            _log(f"  SL={sl} {tname}: ERR {res['err'][:120]}")
            continue
        a_results.append({
            "sl": sl, "trail": tname,
            "pb_atr": pb_atr, "pb_wait": pb_wait,
            "vwap_label": vlab, "vwap_buf": vbuf,
            "min_q": mq,
            "toxic_label": tlab, "toxic_set": sorted(tset),
            "trades": res["trades"], "pf": res["pf"],
            "wr": res["wr"], "pnl": res["pnl"], "dd": res["dd"],
            "delta": res["pnl"] - baseline["pnl"],
        })
    _log(f"  Phase A done in {time.time() - t_a:.1f}s  "
         f"({len(a_results)}/{len(a_jobs)} OK)")

    # Sort by PnL desc; require trades >= 20 to avoid zero-trade configs
    a_results_sorted = sorted(
        [r for r in a_results if r["trades"] >= 20],
        key=lambda x: x["pnl"], reverse=True,
    )
    top_a = a_results_sorted[:PHASE_A_TOP_K]
    _log(f"  Top-{len(top_a)} Phase A:")
    for r in top_a:
        _log(f"    {_fmt_cfg(r)}  trades={r['trades']:3d} PF={r['pf']:5.2f} "
             f"PnL=${r['pnl']:+7.0f}  Δ=${r['delta']:+.0f}")

    # ── Phase B: coordinate descent on each top-A winner ──
    _log(f"\nPhase B: coordinate descent on top-{len(top_a)} Phase-A winners")
    b_chains = []  # one per top-A seed
    for seed_idx, seed in enumerate(top_a):
        _log(f"  ─── Seed {seed_idx+1}/{len(top_a)}: {_fmt_cfg(seed)} ───")
        cur = dict(seed)

        # B1: pullback ATR × wait
        b1_jobs = []
        for pa in PB_ATR_GRID:
            for pw in PB_WAIT_GRID:
                b1_jobs.append((
                    cur["sl"], cur["trail"], pa, pw,
                    cur["vwap_label"], cur["vwap_buf"],
                    cur["min_q"], cur["toxic_label"], frozenset(cur["toxic_set"]),
                    None,
                ))
        t_b = time.time()
        with Pool(workers) as pool:
            b1_raw = list(pool.imap(_bt_one, b1_jobs))
        b1_res = []
        for job, res in zip(b1_jobs, b1_raw):
            if "err" in res or res["trades"] < 20:
                continue
            b1_res.append({
                "pb_atr": job[2], "pb_wait": job[3],
                "trades": res["trades"], "pf": res["pf"],
                "pnl": res["pnl"], "dd": res["dd"],
            })
        b1_res.sort(key=lambda x: x["pnl"], reverse=True)
        if b1_res:
            best_b1 = b1_res[0]
            cur["pb_atr"] = best_b1["pb_atr"]
            cur["pb_wait"] = best_b1["pb_wait"]
            cur["trades"] = best_b1["trades"]
            cur["pf"] = best_b1["pf"]
            cur["pnl"] = best_b1["pnl"]
            cur["dd"] = best_b1["dd"]
            cur["delta"] = best_b1["pnl"] - baseline["pnl"]
            _log(f"    B1 best: PB=({best_b1['pb_atr']},{best_b1['pb_wait']}) "
                 f"PnL=${best_b1['pnl']:+.0f}  ({time.time()-t_b:.1f}s, {len(b1_res)} OK)")

        # B2: VWAP × min_q
        b2_jobs = []
        for (vlab, vbuf) in VWAP_GRID:
            for mq in MIN_Q_GRID:
                b2_jobs.append((
                    cur["sl"], cur["trail"], cur["pb_atr"], cur["pb_wait"],
                    vlab, vbuf, mq, cur["toxic_label"],
                    frozenset(cur["toxic_set"]), None,
                ))
        t_b = time.time()
        with Pool(workers) as pool:
            b2_raw = list(pool.imap(_bt_one, b2_jobs))
        b2_res = []
        for job, res in zip(b2_jobs, b2_raw):
            if "err" in res or res["trades"] < 20:
                continue
            b2_res.append({
                "vwap_label": job[4], "vwap_buf": job[5],
                "min_q": job[6],
                "trades": res["trades"], "pf": res["pf"],
                "pnl": res["pnl"], "dd": res["dd"],
            })
        b2_res.sort(key=lambda x: x["pnl"], reverse=True)
        if b2_res:
            best_b2 = b2_res[0]
            cur["vwap_label"] = best_b2["vwap_label"]
            cur["vwap_buf"] = best_b2["vwap_buf"]
            cur["min_q"] = best_b2["min_q"]
            cur["trades"] = best_b2["trades"]
            cur["pf"] = best_b2["pf"]
            cur["pnl"] = best_b2["pnl"]
            cur["dd"] = best_b2["dd"]
            cur["delta"] = best_b2["pnl"] - baseline["pnl"]
            _log(f"    B2 best: VWAP={best_b2['vwap_label']} mQ={best_b2['min_q']} "
                 f"PnL=${best_b2['pnl']:+.0f}  ({time.time()-t_b:.1f}s, {len(b2_res)} OK)")

        # B3: toxic hours
        b3_jobs = []
        for (tlab, tset) in TOXIC_GRID:
            b3_jobs.append((
                cur["sl"], cur["trail"], cur["pb_atr"], cur["pb_wait"],
                cur["vwap_label"], cur["vwap_buf"], cur["min_q"],
                tlab, tset, None,
            ))
        t_b = time.time()
        with Pool(workers) as pool:
            b3_raw = list(pool.imap(_bt_one, b3_jobs))
        b3_res = []
        for job, res in zip(b3_jobs, b3_raw):
            if "err" in res or res["trades"] < 20:
                continue
            b3_res.append({
                "toxic_label": job[7], "toxic_set": sorted(job[8]),
                "trades": res["trades"], "pf": res["pf"],
                "pnl": res["pnl"], "dd": res["dd"],
            })
        b3_res.sort(key=lambda x: x["pnl"], reverse=True)
        if b3_res:
            best_b3 = b3_res[0]
            cur["toxic_label"] = best_b3["toxic_label"]
            cur["toxic_set"] = best_b3["toxic_set"]
            cur["trades"] = best_b3["trades"]
            cur["pf"] = best_b3["pf"]
            cur["pnl"] = best_b3["pnl"]
            cur["dd"] = best_b3["dd"]
            cur["delta"] = best_b3["pnl"] - baseline["pnl"]
            _log(f"    B3 best: toxic={best_b3['toxic_label']} "
                 f"PnL=${best_b3['pnl']:+.0f}  ({time.time()-t_b:.1f}s, {len(b3_res)} OK)")

        b_chains.append({
            "seed": seed,
            "tuned": cur,
            "b1_results": b1_res,
            "b2_results": b2_res,
            "b3_results": b3_res,
        })

    # ── Pick top-3 finalists by tuned PnL ──
    finalists = sorted(b_chains, key=lambda x: x["tuned"]["pnl"], reverse=True)
    finalists = finalists[:PHASE_B_TOP_K]
    _log(f"\nTop-{len(finalists)} tuned finalists:")
    for i, c in enumerate(finalists):
        t = c["tuned"]
        _log(f"  #{i+1} {_fmt_cfg(t)}  PnL=${t['pnl']:+.0f}  Δ=${t['delta']:+.0f}")

    # ── Phase C: walk-forward validation ──
    _log(f"\nPhase C: walk-forward validation ({WF_NUM_FOLDS} disjoint "
         f"{WF_FOLD_DAYS}d folds) on top-{len(finalists)} finalists")
    wf_jobs = []
    for f_idx, c in enumerate(finalists):
        t = c["tuned"]
        for fold in range(1, WF_NUM_FOLDS + 1):
            wf_jobs.append((
                t["sl"], t["trail"], t["pb_atr"], t["pb_wait"],
                t["vwap_label"], t["vwap_buf"], t["min_q"],
                t["toxic_label"], frozenset(t["toxic_set"]), fold,
            ))
    t_c = time.time()
    with Pool(workers) as pool:
        wf_raw = list(pool.imap(_bt_one, wf_jobs))
    _log(f"  WF done in {time.time() - t_c:.1f}s")

    wf_by_finalist = [[] for _ in finalists]
    for f_idx, c in enumerate(finalists):
        for fold in range(1, WF_NUM_FOLDS + 1):
            job_idx = f_idx * WF_NUM_FOLDS + (fold - 1)
            res = wf_raw[job_idx]
            if "err" in res:
                wf_by_finalist[f_idx].append(
                    {"fold": fold, "err": res["err"][:120],
                     "pf": 0, "pnl": 0, "trades": 0, "wr": 0}
                )
                continue
            wf_by_finalist[f_idx].append({
                "fold": fold,
                "trades": res["trades"],
                "pf": round(res["pf"], 2),
                "pnl": round(res["pnl"], 2),
                "wr": round(res["wr"], 1),
            })

    # ── Final assembly ──
    final = {
        "_meta": {
            "ts": datetime.now().isoformat(),
            "symbol": SYMBOL,
            "days": DAYS,
            "phase_a_grid_size": len(SL_GRID) * len(TRAIL_NAMES),
            "phase_a_top_k": PHASE_A_TOP_K,
            "phase_b_top_k": PHASE_B_TOP_K,
            "wf_num_folds": WF_NUM_FOLDS,
            "wf_fold_days": WF_FOLD_DAYS,
            "ship_rule": {
                "min_delta_usd": MIN_DELTA,
                "wf_min_positive_folds": WF_MIN_POS,
            },
            "dimensions": {
                "sl_grid": SL_GRID,
                "trail_names": TRAIL_NAMES,
                "pb_atr_grid": PB_ATR_GRID,
                "pb_wait_grid": PB_WAIT_GRID,
                "vwap_grid": [v[0] for v in VWAP_GRID],
                "min_q_grid": MIN_Q_GRID,
                "toxic_grid": [t[0] for t in TOXIC_GRID],
            },
            "anchor": {k: (sorted(v) if isinstance(v, frozenset) else v)
                       for k, v in ANCHOR.items()},
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "baseline": baseline,
        "phase_a_all": a_results,
        "phase_a_top": top_a,
        "phase_b_chains": [
            {
                "seed": {k: (sorted(v) if isinstance(v, frozenset) else v)
                         for k, v in c["seed"].items()},
                "tuned": {k: (sorted(v) if isinstance(v, frozenset) else v)
                          for k, v in c["tuned"].items()},
                "b1_top5": c["b1_results"][:5],
                "b2_top5": c["b2_results"][:5],
                "b3_top5": c["b3_results"][:5],
            }
            for c in b_chains
        ],
        "finalists": [],
    }

    # WF analysis per finalist
    best_ship = None
    best_ship_pnl = -1e18
    for f_idx, c in enumerate(finalists):
        t = c["tuned"]
        folds = wf_by_finalist[f_idx]
        pos = sum(1 for f in folds if f.get("pnl", 0) > 0)
        avg_pf = round(sum(f.get("pf", 0) for f in folds) / max(1, len(folds)), 2)
        total_wf_pnl = round(sum(f.get("pnl", 0) for f in folds), 2)
        wf_ok = pos >= WF_MIN_POS
        delta_ok = (t["pnl"] - baseline["pnl"]) >= MIN_DELTA
        ship = bool(wf_ok and delta_ok)
        rec = {
            "rank": f_idx + 1,
            "config": {
                "sl": t["sl"], "trail": t["trail"],
                "pullback_atr_retrace": t["pb_atr"],
                "pullback_max_wait_bars": t["pb_wait"],
                "vwap_buffer_atr": t["vwap_buf"],
                "vwap_buffer_label": t["vwap_label"],
                "min_quality_all_regimes": t["min_q"],
                "toxic_hours_extra": sorted(t["toxic_set"]),
                "toxic_label": t["toxic_label"],
            },
            "in_sample": {
                "trades": t["trades"],
                "pf": round(t["pf"], 2),
                "wr": round(t.get("wr", 0), 1) if "wr" in t else None,
                "pnl": round(t["pnl"], 2),
                "dd": round(t["dd"], 2),
            },
            "delta_pnl": round(t["pnl"] - baseline["pnl"], 2),
            "wf_folds": folds,
            "wf_pos_folds": pos,
            "wf_avg_pf": avg_pf,
            "wf_total_pnl": total_wf_pnl,
            "wf_passed": wf_ok,
            "delta_passed": delta_ok,
            "recommend_ship": ship,
        }
        final["finalists"].append(rec)
        if ship and t["pnl"] > best_ship_pnl:
            best_ship_pnl = t["pnl"]
            best_ship = rec
        _log(f"  Finalist #{f_idx+1}: in-sample Δ=${t['pnl']-baseline['pnl']:+.0f} "
             f"WF {pos}/{WF_NUM_FOLDS} avg_pf={avg_pf:.2f} ship={ship}")

    final["winner"] = best_ship
    final["ship_recommend"] = best_ship is not None

    OUT_JSON.write_text(json.dumps(final, indent=2, default=str))
    _log(f"\nJSON written: {OUT_JSON}")

    # ── Markdown summary ──
    md = []
    md.append(f"# SP500.r Per-Symbol Tune — {datetime.now().strftime('%Y-%m-%d')}\n")
    md.append("## Baseline (live config)\n")
    md.append(f"- trades: **{baseline['trades']}**  PF: **{baseline['pf']:.2f}**  "
              f"WR: **{baseline['wr']:.1f}%**  "
              f"PnL: **${baseline['pnl']:+,.0f}**  DD: **{baseline['dd']:.2f}%**\n")
    md.append("## Phase A — Top SL × Trail\n")
    md.append("| Rank | SL | Trail | Trades | PF | WR | PnL | Δ |\n|---:|---:|:--|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(top_a, 1):
        md.append(f"| {i} | {r['sl']} | {r['trail']} | {r['trades']} | "
                  f"{r['pf']:.2f} | {r['wr']:.1f}% | ${r['pnl']:+,.0f} | ${r['delta']:+,.0f} |\n")

    md.append("\n## Phase B/C — Tuned Finalists\n")
    for rec in final["finalists"]:
        cfg = rec["config"]
        md.append(f"\n### Finalist #{rec['rank']}\n")
        md.append(f"- SL: **{cfg['sl']}**\n")
        md.append(f"- Trail: **{cfg['trail']}**\n")
        md.append(f"- Pullback: ATR={cfg['pullback_atr_retrace']}, "
                  f"wait={cfg['pullback_max_wait_bars']} bars\n")
        md.append(f"- VWAP buffer: {cfg['vwap_buffer_label']} ATR "
                  f"({cfg['vwap_buffer_atr']})\n")
        md.append(f"- min_quality (all regimes): **{cfg['min_quality_all_regimes']}**\n")
        md.append(f"- Toxic extra hours: {cfg['toxic_hours_extra'] or 'none'} "
                  f"({cfg['toxic_label']})\n")
        ins = rec["in_sample"]
        md.append(f"- **In-sample 180d**: trades={ins['trades']} PF={ins['pf']} "
                  f"PnL=${ins['pnl']:+,.0f} DD={ins['dd']}% Δ=${rec['delta_pnl']:+,.0f}\n")
        md.append(f"- **WF**: {rec['wf_pos_folds']}/{WF_NUM_FOLDS} positive, "
                  f"avg_pf={rec['wf_avg_pf']:.2f}, total=${rec['wf_total_pnl']:+,.0f}\n")
        md.append("\n  | Fold | Trades | PF | WR | PnL |\n  |---:|---:|---:|---:|---:|\n")
        for f in rec["wf_folds"]:
            md.append(f"  | {f.get('fold','?')} | {f.get('trades','?')} | "
                      f"{f.get('pf','?')} | {f.get('wr','?')}% | "
                      f"${f.get('pnl',0):+,.0f} |\n")
        md.append(f"- **Ship**: {'YES' if rec['recommend_ship'] else 'NO'} "
                  f"(Δ≥${MIN_DELTA}: {rec['delta_passed']}, "
                  f"WF≥{WF_MIN_POS}/5: {rec['wf_passed']})\n")

    md.append("\n## Verdict\n")
    if final["ship_recommend"]:
        w = final["winner"]
        cfg = w["config"]
        md.append(f"- **SHIP**: Finalist #{w['rank']}\n")
        md.append(f"- Cfg: SL={cfg['sl']}, trail={cfg['trail']}, "
                  f"PB=({cfg['pullback_atr_retrace']},{cfg['pullback_max_wait_bars']}), "
                  f"VWAP={cfg['vwap_buffer_label']}, mQ={cfg['min_quality_all_regimes']}, "
                  f"toxic={cfg['toxic_hours_extra']}\n")
        md.append(f"- Δ=${w['delta_pnl']:+,.0f}  WF {w['wf_pos_folds']}/5  "
                  f"avg_pf={w['wf_avg_pf']}\n")
    else:
        md.append("- **NO-SHIP**: no finalist passed Δ ≥ $50 AND WF ≥ 3/5.\n")

    md.append(f"\n_Tune ran {final['_meta']['elapsed_sec']:.0f}s "
              f"({final['_meta']['phase_a_grid_size']} Phase-A BTs)._\n")

    OUT_MD.write_text("".join(md))
    _log(f"MD written: {OUT_MD}")
    _log(f"\nALL DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
