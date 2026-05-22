#!/usr/bin/env python3 -B
"""BTCUSD per-symbol full-knob tuner — 2026-05-22.

Coordinate-descent + Cartesian search across 7 dimensions, anchored on the
live baseline:

  1. SL ATR mult       ∈ {0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5}
  2. Trail profile     ∈ {_TIGHT_LOCK, _WIDE_RUNNER, _RANGE_TIGHT, _TREND_LOOSE,
                          _AGGR_LOCK, _RUNNER_NO_BE, _WIDE_RUNNER_BE07}
  3. Pullback ATR      ∈ {0.4, 0.6, 0.8, 1.0, 1.2}
  4. Pullback wait     ∈ {3, 5, 7}
  5. VWAP buffer ATR   ∈ {disabled, 0.3, 0.5, 0.7, 1.0}
  6. min_quality       ∈ {28, 30, 33, 35, 38, 40, 43}
  7. Direction bias per regime — trending × volatile × (LONG/SHORT/BOTH)

Strategy:
  Phase A — full SL × trail grid (10 × 7 = 70 BTs). Top-5 → Phase B.
  Phase B — coordinate descent on each top-A seed:
            B1) PB-ATR × PB-wait    (5 × 3 = 15 BTs)
            B2) VWAP × min_quality  (5 × 7 = 35 BTs)
            B3) dir_trending × dir_volatile (3 × 3 = 9 BTs)
            → tuned config per seed. Top-3 → Phase C.
  Phase C — top-3 finalists × 5 disjoint walk-forward folds
            (cache only 20d so folds are 4-day rolling). 15 BTs.

Ship rule:
  - Δ ≥ $30 vs baseline
  - WF ≥ 3/5 folds positive

READ-ONLY: never writes auto_tuned.py / config.py / backtest/v5_backtest.py.
All overrides applied in-memory inside worker processes via module clone +
attribute mutation. Outputs only to:
    /Users/ashish/Documents/beast-trader/per_symbol_tune_20260522/BTCUSD.json
    /Users/ashish/Documents/beast-trader/per_symbol_tune_20260522/BTCUSD.md
    /Users/ashish/Documents/beast-trader/per_symbol_tune_20260522/tune_btcusd.log

Notes on BTCUSD specifically:
  - Crypto, 24/7. TOXIC_EXEMPT covers BTC for hours {1,2,3,4} → leave alone.
  - Live config: SL=1.5 (volatile-only), TRAIL=very dense lock-ladder,
    DIRECTION_BIAS_REGIME={volatile: 'SHORT'}, mQ={tr:30,rg:32,vo:35,lv:30},
    risk_cap 0.4%.
  - Cache only 20 days deep — backtest_symbol(days=180) collapses to that
    window. WF uses disjoint 4-day chunks to get true OOS feedback within
    the available data.
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

SYMBOL = "BTCUSD"
DAYS = 180  # falls back to cache span (~20d) inside load_data
OUT_JSON = ROOT / "per_symbol_tune_20260522" / "BTCUSD.json"
OUT_MD = ROOT / "per_symbol_tune_20260522" / "BTCUSD.md"
LOG_FILE = ROOT / "per_symbol_tune_20260522" / "tune_btcusd.log"

# ── Dimensions (per user spec) ──
SL_GRID = [0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5]
TRAIL_NAMES = [
    "_TIGHT_LOCK", "_WIDE_RUNNER", "_RANGE_TIGHT", "_TREND_LOOSE",
    "_AGGR_LOCK", "_RUNNER_NO_BE", "_WIDE_RUNNER_BE07",
]
PB_ATR_GRID = [0.4, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_GRID = [3, 5, 7]
VWAP_GRID = [
    ("disabled", 9999.0),  # huge buffer => filter never triggers
    ("0.3", 0.3),
    ("0.5", 0.5),
    ("0.7", 0.7),
    ("1.0", 1.0),
]
MIN_Q_GRID = [28, 30, 33, 35, 38, 40, 43]
DIR_GRID = ["LONG", "SHORT", "BOTH"]  # per regime

# Ship gates (per user spec)
MIN_DELTA = 30.0
WF_MIN_POS = 3
WF_NUM_FOLDS = 5
WF_FOLD_DAYS = 4  # 20d cache / 5 folds

# Top-K per phase
PHASE_A_TOP_K = 5
PHASE_B_TOP_K = 3


# ────────────────────────────────────────────────────────────────────
# Module-clone caching: one cloned `bt` per VWAP buffer literal.
# ────────────────────────────────────────────────────────────────────
_BT_CACHE = {}
_BT_BASE_SRC = None
_BT_BASE_FILE = None


def _get_bt(vwap_buf):
    """Return a cloned backtest.v5_backtest module whose hard-coded VWAP
    buffer (0.5) is rewritten to `vwap_buf`."""
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
# Single-BT worker.
# ────────────────────────────────────────────────────────────────────
def _bt_one(args):
    """Run one backtest with the supplied 7-knob config + optional fold_id.
    Returns dict with metrics or {'err': str}.
    """
    (sl, trail_name, pb_atr, pb_wait, vwap_label, vwap_buf,
     min_q, dir_trending, dir_volatile, fold_id) = args
    try:
        # 1. Mutate config attributes (pullback constants are imported inside
        #    backtest_symbol's entry loop, so we patch them on the live module).
        import config as cfg
        cfg.PULLBACK_ATR_RETRACE = float(pb_atr)
        cfg.PULLBACK_MAX_WAIT_BARS = int(pb_wait)

        bt = _get_bt(vwap_buf)

        # 2. Per-symbol overrides on the cloned bt module
        # SL (per-symbol; clear per-regime override so sl_atr_mult sweeps cleanly)
        bt.SL_OVERRIDE = dict(bt.SL_OVERRIDE)
        bt.SL_OVERRIDE[SYMBOL] = float(sl)
        bt.SL_OVERRIDE_REGIME = dict(bt.SL_OVERRIDE_REGIME)
        bt.SL_OVERRIDE_REGIME[SYMBOL] = {}

        # Trail: clear per-(sym, regime) trail override so per-symbol wins
        bt.TRAIL_OVERRIDE = dict(bt.TRAIL_OVERRIDE)
        steps_live = _load_trail_profile(trail_name)
        steps_bt = _live_to_bt_trail(steps_live)
        bt.TRAIL_OVERRIDE[SYMBOL] = steps_bt
        bt.TRAIL_OVERRIDE_REGIME = dict(bt.TRAIL_OVERRIDE_REGIME)
        bt.TRAIL_OVERRIDE_REGIME[SYMBOL] = {}  # let TRAIL_OVERRIDE[sym] win

        # Direction bias per regime — patch _DIR_BIAS_REGIME_STR (string form,
        # converted inside _dir_bias_for_regime to int). User wants
        # trending and volatile dimensions tunable; ranging and low_vol
        # default to BOTH (no bias).
        bt._DIR_BIAS_REGIME_STR = dict(bt._DIR_BIAS_REGIME_STR)
        bt._DIR_BIAS_REGIME_STR[SYMBOL] = {
            "trending": dir_trending,
            "volatile": dir_volatile,
            "ranging":  "BOTH",
            "low_vol":  "BOTH",
        }
        # Also blank the per-symbol fallback so per-regime is sole source
        if SYMBOL in bt.DIR_BIAS:
            bt.DIR_BIAS = {k: v for k, v in bt.DIR_BIAS.items() if k != SYMBOL}

        # 3. Params with uniform min_quality across regimes
        params = {
            "min_quality": {
                "trending": int(min_q),
                "ranging": int(min_q),
                "volatile": int(min_q),
                "low_vol": int(min_q),
            },
        }

        # 4. Optional disjoint-WF slicing.
        # 20d cache → 5 disjoint 4-day folds rolling backward from end.
        if fold_id is not None:
            import pandas as pd
            orig_load = bt.load_data
            fold_n = int(fold_id)  # 1..5
            num = WF_NUM_FOLDS
            fold_d = WF_FOLD_DAYS

            def load_data_fold(sym, days=None):
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
            r = bt.backtest_symbol(SYMBOL, days=None, params=params, verbose=False)
        else:
            r = bt.backtest_symbol(SYMBOL, days=DAYS, params=params, verbose=False)

        if r is None:
            return {"err": "result_none"}
        return {
            "trades": int(r.get("trades", 0)),
            "pf":     float(r.get("pf", 0)),
            "wr":     float(r.get("wr", 0)),
            "pnl":    float(r.get("pnl", 0)),
            "dd":     float(r.get("dd", 0)),
            "avg_r":  float(r.get("avg_r", 0) or 0),
            "avg_peak_r":  float(r.get("avg_peak_r", 0) or 0),
            "avg_giveback":  float(r.get("avg_giveback", 0) or 0),
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
            "avg_r":  float(r.get("avg_r", 0) or 0),
            "avg_peak_r":  float(r.get("avg_peak_r", 0) or 0),
            "avg_giveback":  float(r.get("avg_giveback", 0) or 0),
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
            f"VWAP={c['vwap_label']:8} mQ={c['min_q']} "
            f"dir(tr/vo)={c['dir_trending']}/{c['dir_volatile']}")


# ────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────
def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")

    t0 = time.time()
    workers = max(2, (os.cpu_count() or 4) - 2)
    _log(f"BTCUSD tuner — workers={workers}")
    _log(f"Grids: SL={len(SL_GRID)} Trail={len(TRAIL_NAMES)} "
         f"PB_ATR={len(PB_ATR_GRID)} PB_WAIT={len(PB_WAIT_GRID)} "
         f"VWAP={len(VWAP_GRID)} mQ={len(MIN_Q_GRID)} DIR={len(DIR_GRID)}")

    # ── Anchor (live config baseline) ──
    # Live BTCUSD: SL 1.5, dense lock trail (closest match: _TIGHT_LOCK),
    # pullback default 0.8/5, VWAP 0.5 hardcoded, mQ avg ≈ 30 (live spread:
    # tr=30 rg=32 vo=35 lv=30), DIR={volatile:SHORT}.
    ANCHOR = {
        "sl": 1.5,
        "trail": "_TIGHT_LOCK",  # closest profile to live's dense lock-ladder
        "pb_atr": 0.8,
        "pb_wait": 5,
        "vwap_label": "0.5",
        "vwap_buf": 0.5,
        "min_q": 30,
        "dir_trending": "BOTH",
        "dir_volatile": "SHORT",
    }

    # ── Phase 0: live-config baseline ──
    _log("\nPhase 0: live-config baseline")
    with Pool(1) as pool:
        baseline = pool.apply(_bt_baseline)
    if "err" in baseline:
        _log(f"  baseline FAILED: {baseline['err']}")
        sys.exit(1)
    _log(f"  baseline: trades={baseline['trades']} PF={baseline['pf']:.2f} "
         f"WR={baseline['wr']:.1f}% PnL=${baseline['pnl']:+.2f} "
         f"DD={baseline['dd']:.2f}% avg_peak={baseline['avg_peak_r']:.2f}R "
         f"give={baseline['avg_giveback']:.2f}R")

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
                ANCHOR["dir_trending"], ANCHOR["dir_volatile"],
                None,  # full window
            ))
    a_results = []
    t_a = time.time()
    with Pool(workers) as pool:
        a_raw = list(pool.imap(_bt_one, a_jobs))
    for job, res in zip(a_jobs, a_raw):
        (sl, tname, pb_atr, pb_wait, vlab, vbuf, mq, dtr, dvo, _) = job
        if "err" in res:
            _log(f"  SL={sl} {tname}: ERR {res['err'][:120]}")
            continue
        a_results.append({
            "sl": sl, "trail": tname,
            "pb_atr": pb_atr, "pb_wait": pb_wait,
            "vwap_label": vlab, "vwap_buf": vbuf,
            "min_q": mq,
            "dir_trending": dtr, "dir_volatile": dvo,
            "trades": res["trades"], "pf": res["pf"],
            "wr": res["wr"], "pnl": res["pnl"], "dd": res["dd"],
            "avg_r": res["avg_r"], "avg_peak_r": res["avg_peak_r"],
            "avg_giveback": res["avg_giveback"],
            "delta": res["pnl"] - baseline["pnl"],
        })
    _log(f"  Phase A done in {time.time() - t_a:.1f}s "
         f"({len(a_results)}/{len(a_jobs)} OK)")

    # Sort by PnL desc; require trades >= 10 (BTC is high-freq so still plenty)
    MIN_TRADES = 10
    a_sorted = sorted(
        [r for r in a_results if r["trades"] >= MIN_TRADES],
        key=lambda x: x["pnl"], reverse=True,
    )
    top_a = a_sorted[:PHASE_A_TOP_K]
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
                    cur["min_q"],
                    cur["dir_trending"], cur["dir_volatile"],
                    None,
                ))
        t_b = time.time()
        with Pool(workers) as pool:
            b1_raw = list(pool.imap(_bt_one, b1_jobs))
        b1_res = []
        for job, res in zip(b1_jobs, b1_raw):
            if "err" in res or res["trades"] < MIN_TRADES:
                continue
            b1_res.append({
                "pb_atr": job[2], "pb_wait": job[3],
                "trades": res["trades"], "pf": res["pf"], "wr": res["wr"],
                "pnl": res["pnl"], "dd": res["dd"],
            })
        b1_res.sort(key=lambda x: x["pnl"], reverse=True)
        if b1_res:
            best = b1_res[0]
            cur["pb_atr"] = best["pb_atr"]
            cur["pb_wait"] = best["pb_wait"]
            cur["trades"] = best["trades"]
            cur["pf"] = best["pf"]
            cur["wr"] = best["wr"]
            cur["pnl"] = best["pnl"]
            cur["dd"] = best["dd"]
            cur["delta"] = best["pnl"] - baseline["pnl"]
            _log(f"    B1 best: PB=({best['pb_atr']},{best['pb_wait']}) "
                 f"PnL=${best['pnl']:+.0f}  ({time.time()-t_b:.1f}s, "
                 f"{len(b1_res)} OK)")

        # B2: VWAP × min_q
        b2_jobs = []
        for (vlab, vbuf) in VWAP_GRID:
            for mq in MIN_Q_GRID:
                b2_jobs.append((
                    cur["sl"], cur["trail"], cur["pb_atr"], cur["pb_wait"],
                    vlab, vbuf, mq,
                    cur["dir_trending"], cur["dir_volatile"],
                    None,
                ))
        t_b = time.time()
        with Pool(workers) as pool:
            b2_raw = list(pool.imap(_bt_one, b2_jobs))
        b2_res = []
        for job, res in zip(b2_jobs, b2_raw):
            if "err" in res or res["trades"] < MIN_TRADES:
                continue
            b2_res.append({
                "vwap_label": job[4], "vwap_buf": job[5],
                "min_q": job[6],
                "trades": res["trades"], "pf": res["pf"], "wr": res["wr"],
                "pnl": res["pnl"], "dd": res["dd"],
            })
        b2_res.sort(key=lambda x: x["pnl"], reverse=True)
        if b2_res:
            best = b2_res[0]
            cur["vwap_label"] = best["vwap_label"]
            cur["vwap_buf"] = best["vwap_buf"]
            cur["min_q"] = best["min_q"]
            cur["trades"] = best["trades"]
            cur["pf"] = best["pf"]
            cur["wr"] = best["wr"]
            cur["pnl"] = best["pnl"]
            cur["dd"] = best["dd"]
            cur["delta"] = best["pnl"] - baseline["pnl"]
            _log(f"    B2 best: VWAP={best['vwap_label']} mQ={best['min_q']} "
                 f"PnL=${best['pnl']:+.0f}  ({time.time()-t_b:.1f}s, "
                 f"{len(b2_res)} OK)")

        # B3: dir_trending × dir_volatile
        b3_jobs = []
        for dtr in DIR_GRID:
            for dvo in DIR_GRID:
                b3_jobs.append((
                    cur["sl"], cur["trail"], cur["pb_atr"], cur["pb_wait"],
                    cur["vwap_label"], cur["vwap_buf"], cur["min_q"],
                    dtr, dvo, None,
                ))
        t_b = time.time()
        with Pool(workers) as pool:
            b3_raw = list(pool.imap(_bt_one, b3_jobs))
        b3_res = []
        for job, res in zip(b3_jobs, b3_raw):
            if "err" in res or res["trades"] < MIN_TRADES:
                continue
            b3_res.append({
                "dir_trending": job[7], "dir_volatile": job[8],
                "trades": res["trades"], "pf": res["pf"], "wr": res["wr"],
                "pnl": res["pnl"], "dd": res["dd"],
            })
        b3_res.sort(key=lambda x: x["pnl"], reverse=True)
        if b3_res:
            best = b3_res[0]
            cur["dir_trending"] = best["dir_trending"]
            cur["dir_volatile"] = best["dir_volatile"]
            cur["trades"] = best["trades"]
            cur["pf"] = best["pf"]
            cur["wr"] = best["wr"]
            cur["pnl"] = best["pnl"]
            cur["dd"] = best["dd"]
            cur["delta"] = best["pnl"] - baseline["pnl"]
            _log(f"    B3 best: dir(tr/vo)={best['dir_trending']}/"
                 f"{best['dir_volatile']} "
                 f"PnL=${best['pnl']:+.0f}  ({time.time()-t_b:.1f}s, "
                 f"{len(b3_res)} OK)")

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
        _log(f"  #{i+1} {_fmt_cfg(t)} PnL=${t['pnl']:+.0f}  Δ=${t['delta']:+.0f}")

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
                t["dir_trending"], t["dir_volatile"], fold,
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
                wf_by_finalist[f_idx].append({
                    "fold": fold, "err": res["err"][:120],
                    "pf": 0, "pnl": 0, "trades": 0, "wr": 0,
                })
                continue
            wf_by_finalist[f_idx].append({
                "fold": fold,
                "trades": res["trades"],
                "pf": round(res["pf"], 2),
                "pnl": round(res["pnl"], 2),
                "wr": round(res["wr"], 1),
            })

    # ── Final assembly ──
    total_bts = (
        1
        + len(SL_GRID) * len(TRAIL_NAMES)
        + len(finalists) * (
            len(PB_ATR_GRID) * len(PB_WAIT_GRID)
            + len(VWAP_GRID) * len(MIN_Q_GRID)
            + len(DIR_GRID) * len(DIR_GRID)
        )
        + len(finalists) * WF_NUM_FOLDS
    )
    # b_chains has len(top_a), not len(finalists); recompute true total
    total_bts_actual = (
        1
        + len(SL_GRID) * len(TRAIL_NAMES)
        + len(b_chains) * (
            len(PB_ATR_GRID) * len(PB_WAIT_GRID)
            + len(VWAP_GRID) * len(MIN_Q_GRID)
            + len(DIR_GRID) * len(DIR_GRID)
        )
        + len(finalists) * WF_NUM_FOLDS
    )
    final = {
        "_meta": {
            "ts": datetime.now().isoformat(),
            "symbol": SYMBOL,
            "days_requested": DAYS,
            "cache_window_days": "~20",
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
                "dir_grid": DIR_GRID,
            },
            "anchor": ANCHOR,
            "total_backtests_run": total_bts_actual,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "baseline": baseline,
        "phase_a_all": a_results,
        "phase_a_top": top_a,
        "phase_b_chains": [
            {
                "seed": c["seed"],
                "tuned": c["tuned"],
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
                "sl_atr_mult": t["sl"],
                "trail_profile": t["trail"],
                "pullback_atr_retrace": t["pb_atr"],
                "pullback_max_wait_bars": t["pb_wait"],
                "vwap_buffer_atr_label": t["vwap_label"],
                "vwap_buffer_atr": t["vwap_buf"],
                "min_quality_all_regimes": t["min_q"],
                "direction_bias_trending": t["dir_trending"],
                "direction_bias_volatile": t["dir_volatile"],
            },
            "in_sample": {
                "trades": t["trades"],
                "pf": round(t["pf"], 2),
                "wr": round(t.get("wr", 0), 1),
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
        _log(f"  Finalist #{f_idx+1}: Δ=${t['pnl']-baseline['pnl']:+.0f} "
             f"WF {pos}/{WF_NUM_FOLDS} avg_pf={avg_pf:.2f} "
             f"total=${total_wf_pnl:+.0f} ship={ship}")

    final["winner"] = best_ship
    final["ship_recommend"] = best_ship is not None

    OUT_JSON.write_text(json.dumps(final, indent=2, default=str))
    _log(f"\nJSON written: {OUT_JSON}")

    # ── Markdown summary ──
    md = []
    md.append(f"# BTCUSD Per-Symbol Tune — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    md.append("Cache window ~20d (live data depth). WF uses disjoint "
              f"{WF_FOLD_DAYS}d rolling folds.\n\n")
    md.append("## Baseline (live config)\n")
    md.append(f"- trades: **{baseline['trades']}**  PF: **{baseline['pf']:.2f}**  "
              f"WR: **{baseline['wr']:.1f}%**  "
              f"PnL: **${baseline['pnl']:+,.0f}**  DD: **{baseline['dd']:.2f}%**  "
              f"avg_peak: **{baseline['avg_peak_r']:.2f}R**  "
              f"giveback: **{baseline['avg_giveback']:.2f}R**\n")
    md.append("- Live note: 30d live PnL = -$15.66 (small bleed). "
              "Backtest 180d (capped to ~20d cache) baseline ≈ $148.\n")

    md.append("\n## Phase A — Top SL × Trail\n")
    md.append("| Rank | SL | Trail | Trades | PF | WR | PnL | Δ |\n"
              "|---:|---:|:--|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(top_a, 1):
        md.append(f"| {i} | {r['sl']} | {r['trail']} | {r['trades']} | "
                  f"{r['pf']:.2f} | {r['wr']:.1f}% | ${r['pnl']:+,.0f} | "
                  f"${r['delta']:+,.0f} |\n")

    md.append("\n## Phase B/C — Tuned Finalists\n")
    for rec in final["finalists"]:
        cfg = rec["config"]
        md.append(f"\n### Finalist #{rec['rank']}\n")
        md.append(f"- SL ATR mult: **{cfg['sl_atr_mult']}**\n")
        md.append(f"- Trail: **{cfg['trail_profile']}**\n")
        md.append(f"- Pullback: ATR=**{cfg['pullback_atr_retrace']}**, "
                  f"wait=**{cfg['pullback_max_wait_bars']}** bars\n")
        md.append(f"- VWAP buffer: **{cfg['vwap_buffer_atr_label']}** ATR "
                  f"(`{cfg['vwap_buffer_atr']}`)\n")
        md.append(f"- min_quality (all regimes): **{cfg['min_quality_all_regimes']}**\n")
        md.append(f"- Direction bias: trending=**{cfg['direction_bias_trending']}**, "
                  f"volatile=**{cfg['direction_bias_volatile']}**\n")
        ins = rec["in_sample"]
        md.append(f"- **In-sample 180d**: trades={ins['trades']} PF={ins['pf']} "
                  f"WR={ins['wr']}% PnL=${ins['pnl']:+,.0f} DD={ins['dd']}% "
                  f"Δ=${rec['delta_pnl']:+,.0f}\n")
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
        md.append(f"- Cfg: SL={cfg['sl_atr_mult']}, "
                  f"trail={cfg['trail_profile']}, "
                  f"PB=({cfg['pullback_atr_retrace']},{cfg['pullback_max_wait_bars']}), "
                  f"VWAP={cfg['vwap_buffer_atr_label']}, "
                  f"mQ={cfg['min_quality_all_regimes']}, "
                  f"dir(tr/vo)={cfg['direction_bias_trending']}/"
                  f"{cfg['direction_bias_volatile']}\n")
        md.append(f"- Δ=${w['delta_pnl']:+,.0f}  WF {w['wf_pos_folds']}/5 "
                  f"avg_pf={w['wf_avg_pf']}\n")
    else:
        md.append(f"- **NO-SHIP**: no finalist passed Δ ≥ ${MIN_DELTA} AND "
                  f"WF ≥ {WF_MIN_POS}/5.\n")

    md.append(f"\n_Tune ran {final['_meta']['elapsed_sec']:.0f}s "
              f"({final['_meta']['total_backtests_run']} backtests total)._\n")

    OUT_MD.write_text("".join(md))
    _log(f"MD written: {OUT_MD}")
    _log(f"\nALL DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
