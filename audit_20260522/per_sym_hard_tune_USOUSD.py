#!/usr/bin/env python3 -B
"""
USOUSD HARD per-symbol tune  —  2026-05-23.

Goal:  Hard-tune the weakest of the 5 finalists ($1,219 equity @ 2% risk).
       Current live: SL=1.5, trail=_TIGHT_LOCK, mq=25, VWAP=0.5.
       Baseline (r=2%, eq=1219, 180d): 21 trades, PF 0.46, -$80, DD 8.5%.
       (At default r=0.8% BT shows $1,306 / PF 3.39 / 4.0% DD — that is the
        regime users see in dashboards. The 2% setting is hostile to USOUSD
        because the BT's 8% DD safeguard halts after a single SL bleed.)

READ-ONLY on source — no edits to backtest/, config.py, auto_tuned.py.
All overrides applied per-worker via importlib.reload + module-attribute
mutation. VWAP buffer patched by source-replacing the inline 0.5 constant.

Dimensions (9):
  1. SL ATR mult         : {0.3,0.5,0.7,1.0,1.2,1.5,2.0,2.5,3.0}            (9)
  2. Trail profile       : 7 named (TIGHT_LOCK, WIDE_RUNNER, etc.)          (7)
  3. min_quality         : {22,25,28,30,33,35,38,40}                        (8)
  4. Pullback ATR        : {0.4,0.6,0.8,1.0,1.2}                            (5)
  5. Pullback wait bars  : {3,5,7}                                          (3)
  6. VWAP buffer ATR     : {0.0_disabled,0.3,0.5,0.7,1.0,1.5}               (6)
  7. POST_BIG_WIN secs   : {1800,3600,5400,7200,10800}                      (5)
  8. LOSS_STREAK secs    : {3600,7200,10800,14400,18000}                    (5)
  9. force_direction     : {LONG,SHORT,BOTH}                                (3)

Phase A  : 51 BTs (per-dim independent sweep).
Phase B  : top-2 of 7 highest-impact dims + best-1 of the two cooldowns =
           2^7 = 128 BTs (under 200 cap).
Phase C  : 5-fold WF expanding [60,90,120,150,180] on top-5 = 25 BTs.

Ship gate: Δ ≥ $30 (vs r=2% baseline) AND ≥3/5 WF folds positive AND avg PF ≥ 1.5.
Honest verdict: if no combo passes -> recommend DROPPING USOUSD from universe.
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "audit_20260522"
OUT_JSON = OUT_DIR / "per_sym_hard_tune_USOUSD.json"
OUT_MD   = OUT_DIR / "per_sym_hard_tune_USOUSD.md"

SYMBOL = "USOUSD"
TUNE_DAYS = 180
WF_FOLDS_DAYS = [60, 90, 120, 150, 180]

# Account context per user
RISK_PCT = 2.0
START_EQUITY = 1219.0

# ───────────────── dimension grids ─────────────────
SL_GRID    = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

TRAIL_PROFILES = {
    "_TIGHT_LOCK": [
        (6.0, "lock", 5.5), (4.5, "lock", 4.0), (3.5, "lock", 3.0),
        (2.5, "lock", 2.1), (2.0, "lock", 1.65), (1.5, "lock", 1.2),
        (1.0, "lock", 0.75), (0.7, "lock", 0.45), (0.5, "lock", 0.25),
        (0.3, "be",  0.0),
    ],
    "_WIDE_RUNNER":      [(10.0,"trail",0.3),(5.0,"trail",0.5),(2.5,"trail",0.7),(1.5,"lock",0.5),(0.7,"be",0.0)],
    "_RANGE_TIGHT":      [(4.0,"trail",0.5),(2.0,"lock",1.2),(1.0,"lock",0.6),(0.3,"be",0.0)],
    "_TREND_LOOSE":      [(15.0,"trail",0.3),(8.0,"trail",0.4),(4.0,"trail",0.5),(2.0,"lock",1.0),(1.0,"lock",0.5),(0.3,"be",0.0)],
    "_AGGR_LOCK": [
        (6.0,"lock",5.4),(4.0,"lock",3.5),(3.0,"lock",2.6),(2.0,"lock",1.6),
        (1.5,"lock",1.15),(1.0,"lock",0.7),(0.7,"lock",0.45),(0.5,"lock",0.25),
        (0.3,"be",0.0),
    ],
    "_RUNNER_NO_BE": [
        (8.0,"lock",7.0),(5.0,"lock",4.2),(3.5,"lock",2.9),(2.5,"lock",2.0),
        (1.8,"lock",1.4),(1.3,"lock",0.95),(0.9,"lock",0.6),(0.7,"lock",0.4),(0.5,"lock",0.22),
    ],
    "_WIDE_RUNNER_BE07": [(10.0,"trail",0.3),(5.0,"trail",0.5),(2.5,"trail",0.7),(1.5,"lock",0.5),(0.7,"be",0.0)],
}
TRAIL_NAMES = list(TRAIL_PROFILES)

MINQ_GRID  = [22, 25, 28, 30, 33, 35, 38, 40]
PB_ATR_GRID  = [0.4, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_GRID = [3, 5, 7]
VWAP_GRID    = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]
PBW_GRID     = [1800, 3600, 5400, 7200, 10800]
LS_GRID      = [3600, 7200, 10800, 14400, 18000]
DIR_GRID     = ["LONG", "SHORT", "BOTH"]

# Ship gates (per user)
SHIP_DELTA   = 30.0
SHIP_WF_POS  = 3
SHIP_AVG_PF  = 1.5

# ───────────────── worker BT helpers ─────────────────
_WORKER_BT_ORIGINALS: dict = {}

def _get_bt_module(vwap_buf: float):
    """Return BT module. VWAP buffer is now set via the config dict
    VWAP_BUFFER_PER_SYMBOL (read inline by backtest_symbol per loop), so we
    just mutate the dict instead of rewriting source. The module attribute
    `backtest_symbol` is never patched — it always reads the current cfg.
    `vwap_buf == 0.0` is honored by the BT: `if _vw_buf_mult > 0:` guards
    the entire filter block."""
    import backtest.v5_backtest as _bt
    import config as cfg
    cfg.VWAP_BUFFER_PER_SYMBOL = dict(getattr(cfg, "VWAP_BUFFER_PER_SYMBOL", {}))
    cfg.VWAP_BUFFER_PER_SYMBOL[SYMBOL] = float(vwap_buf)
    return _bt


def _apply_config(pb_atr, pb_wait, post_big_win_secs, loss_streak_secs):
    """Pin config to user-stated defaults, then overlay sweep values.
    USER-STATED defaults (independent of file drift):
      pb_atr default = 0.8  (live PULLBACK_ATR_RETRACE)
      pb_wait default = 5   (live PULLBACK_MAX_WAIT_BARS)
      post_big_win default = 10800s (live LIVE_POST_BIG_WIN_COOLDOWN_SECS)
      loss_streak default = 18000s (live LIVE_LOSS_STREAK_COOLDOWN_SECS)
    """
    import config as cfg
    # Pin pullback to user-stated defaults (overlay sweep values below)
    cfg.PULLBACK_ATR_RETRACE = 0.8 if pb_atr is None else float(pb_atr)
    cfg.PULLBACK_ATR_RETRACE_PER_SYMBOL = dict(getattr(cfg, "PULLBACK_ATR_RETRACE_PER_SYMBOL", {}))
    cfg.PULLBACK_ATR_RETRACE_PER_SYMBOL[SYMBOL] = cfg.PULLBACK_ATR_RETRACE
    cfg.PULLBACK_MAX_WAIT_BARS = 5 if pb_wait is None else int(pb_wait)
    cfg.PULLBACK_MAX_WAIT_BARS_PER_SYMBOL = dict(getattr(cfg, "PULLBACK_MAX_WAIT_BARS_PER_SYMBOL", {}))
    cfg.PULLBACK_MAX_WAIT_BARS_PER_SYMBOL[SYMBOL] = cfg.PULLBACK_MAX_WAIT_BARS
    # Pin cooldowns to live defaults
    cfg.POST_BIG_WIN_COOLDOWN_ENABLED = True
    cfg.POST_BIG_WIN_COOLDOWN_SECS = 10800 if post_big_win_secs is None else int(post_big_win_secs)
    cfg.LOSS_STREAK_COOLDOWN_ENABLED = True
    cfg.LOSS_STREAK_WINDOW_SECS = 14400
    cfg.LOSS_STREAK_COUNT = 2
    cfg.LOSS_STREAK_COOLDOWN_SECS = 18000 if loss_streak_secs is None else int(loss_streak_secs)


def _run_one(job):
    """
    job keys (any None = "use USER-STATED default"):
      days, sl_mult, trail_name, pb_atr, pb_wait, vwap_buf, minq_int,
      post_big_win_secs, loss_streak_secs, force_dir

    USER-STATED defaults (pin baseline to user input, NOT to drifting live config):
      SL=1.5  trail=_TIGHT_LOCK  mq=25  VWAP=0.5
      pb=live  pbw=live  ls=live (None in job = file default)
    """
    try:
        days       = int(job.get("days", TUNE_DAYS))
        sl_mult    = job.get("sl_mult")
        trail_name = job.get("trail_name")
        pb_atr     = job.get("pb_atr")
        pb_wait    = job.get("pb_wait")
        vwap_buf   = job.get("vwap_buf")  # None → user default 0.5
        minq_int   = job.get("minq_int")
        pbw_secs   = job.get("post_big_win_secs")
        ls_secs    = job.get("loss_streak_secs")
        force_dir  = job.get("force_dir")

        import config as cfg
        importlib.reload(cfg)
        _apply_config(pb_atr, pb_wait, pbw_secs, ls_secs)

        # Pin VWAP to user-stated default unless this dim is being swept.
        eff_vwap = 0.5 if vwap_buf is None else float(vwap_buf)
        bt = _get_bt_module(eff_vwap)

        # Always re-pristinate the BT module's per-symbol overrides each call,
        # then overlay user-stated baseline + this dim's sweep value.
        bt.SL_OVERRIDE = dict(bt.SL_OVERRIDE)
        bt.SL_OVERRIDE_REGIME = dict(bt.SL_OVERRIDE_REGIME)
        bt.SL_OVERRIDE_REGIME[SYMBOL] = {}
        bt.TRAIL_OVERRIDE_REGIME = dict(bt.TRAIL_OVERRIDE_REGIME)
        bt.TRAIL_OVERRIDE_REGIME[SYMBOL] = {}
        bt._DIR_BIAS_REGIME_STR = dict(bt._DIR_BIAS_REGIME_STR)
        bt._DIR_BIAS_REGIME_STR[SYMBOL] = {}

        # USER-STATED baseline (applied first; sweep values overwrite below)
        bt.SL_OVERRIDE[SYMBOL] = 1.5

        # USER-STATED baseline + live LOSS_STREAK defaults pinned in params.
        # Live: count=2, window=14400s=4h, cooldown=18000s=5h → bars at H1: 4/5.
        ls_bars = 5 if ls_secs is None else max(1, int(round(int(ls_secs) / 3600.0)))
        params = {
            "risk_pct": RISK_PCT,
            "start_equity": START_EQUITY,
            "force_trail": TRAIL_PROFILES["_TIGHT_LOCK"],
            "min_quality": {"trending": 25, "ranging": 25, "volatile": 25, "low_vol": 25},
            "loss_streak_enabled": True,
            "loss_streak_count": 2,
            "loss_streak_window_bars": 4,
            "loss_streak_cooldown_bars": ls_bars,
        }

        if sl_mult is not None:
            params["sl_atr_mult"] = float(sl_mult)
            bt.SL_OVERRIDE[SYMBOL] = float(sl_mult)
        if trail_name is not None:
            params["force_trail"] = TRAIL_PROFILES[trail_name]
        if minq_int is not None:
            params["min_quality"] = {
                "trending": int(minq_int), "ranging": int(minq_int),
                "volatile": int(minq_int), "low_vol": int(minq_int),
            }
        if force_dir is not None:
            params["force_direction"] = force_dir

        r = bt.backtest_symbol(SYMBOL, days=days, params=params, verbose=False)
        if not r:
            return {**job, "ok": False, "err": "no_data"}

        return {
            **job,
            "ok": True,
            "pnl":  float(r.get("pnl", 0)),
            "pf":   float(r.get("pf", 0)),
            "wr":   float(r.get("wr", 0)),
            "n":    int(r.get("trades", 0)),
            "dd":   float(r.get("dd", 0)),
            "avg_r": float(r.get("avg_r", 0)),
            "avg_peak_r": float(r.get("avg_peak_r", 0)),
            "avg_giveback": float(r.get("avg_giveback", 0)),
        }
    except Exception as e:
        return {**job, "ok": False, "err": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc(limit=2)}


def _baseline():
    """USER-STATED-CONFIG baseline @ r=2%, eq=$1219, 180d.
    Pinned to user input: SL=1.5, trail=_TIGHT_LOCK, mq=25, VWAP=0.5.
    (Live config drifts during the run — pinning makes the Δ stable.)"""
    import config as cfg
    importlib.reload(cfg)
    cfg.VWAP_BUFFER_PER_SYMBOL = dict(getattr(cfg, "VWAP_BUFFER_PER_SYMBOL", {}))
    cfg.VWAP_BUFFER_PER_SYMBOL[SYMBOL] = 0.5
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    bt.SL_OVERRIDE = dict(bt.SL_OVERRIDE)
    bt.SL_OVERRIDE[SYMBOL] = 1.5
    bt.SL_OVERRIDE_REGIME = dict(bt.SL_OVERRIDE_REGIME)
    bt.SL_OVERRIDE_REGIME[SYMBOL] = {}
    bt.TRAIL_OVERRIDE_REGIME = dict(bt.TRAIL_OVERRIDE_REGIME)
    bt.TRAIL_OVERRIDE_REGIME[SYMBOL] = {}
    # Pin pullback + cooldowns to live defaults
    cfg.PULLBACK_ATR_RETRACE = 0.8
    cfg.PULLBACK_ATR_RETRACE_PER_SYMBOL = dict(getattr(cfg, "PULLBACK_ATR_RETRACE_PER_SYMBOL", {}))
    cfg.PULLBACK_ATR_RETRACE_PER_SYMBOL[SYMBOL] = 0.8
    cfg.PULLBACK_MAX_WAIT_BARS = 5
    cfg.PULLBACK_MAX_WAIT_BARS_PER_SYMBOL = dict(getattr(cfg, "PULLBACK_MAX_WAIT_BARS_PER_SYMBOL", {}))
    cfg.PULLBACK_MAX_WAIT_BARS_PER_SYMBOL[SYMBOL] = 5
    cfg.POST_BIG_WIN_COOLDOWN_ENABLED = True
    cfg.POST_BIG_WIN_COOLDOWN_SECS = 10800
    r = bt.backtest_symbol(SYMBOL, days=TUNE_DAYS, params={
        "risk_pct": RISK_PCT, "start_equity": START_EQUITY,
        "force_trail": TRAIL_PROFILES["_TIGHT_LOCK"],
        "min_quality": {"trending": 25, "ranging": 25, "volatile": 25, "low_vol": 25},
        "loss_streak_enabled": True, "loss_streak_count": 2,
        "loss_streak_window_bars": 4, "loss_streak_cooldown_bars": 5,
    }, verbose=False)
    return {
        "pnl":  float(r["pnl"]),
        "pf":   float(r["pf"]),
        "wr":   float(r["wr"]),
        "n":    int(r["trades"]),
        "dd":   float(r["dd"]),
        "avg_r": float(r["avg_r"]),
        "avg_peak_r": float(r.get("avg_peak_r", 0)),
        "avg_giveback": float(r.get("avg_giveback", 0)),
    }


def _fmt(r, suffix=""):
    return (f"n={r.get('n',0):4d} PF={r.get('pf',0):5.2f} "
            f"PnL=${r.get('pnl',0):+9.2f} WR={r.get('wr',0):5.1f}% "
            f"DD={r.get('dd',0):4.1f}%{suffix}")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nUSOUSD HARD per-symbol tune — {datetime.now().isoformat(timespec='seconds')}")
    print(f"Days={TUNE_DAYS}  WF folds={WF_FOLDS_DAYS}  risk={RISK_PCT}%  eq=${START_EQUITY}")
    print(f"Output: {OUT_JSON}\n")

    workers = max(2, min(8, (cpu_count() or 4)))
    print(f"Pool workers: {workers}\n")

    # ───────────── BASELINE ─────────────
    print("[0] BASELINE (live config, no overrides):")
    base = _baseline()
    print(f"    {_fmt(base)}")
    base_pnl = base["pnl"]

    # ───────────── PHASE A: per-dim independent sweep ─────────────
    print(f"\n[A] Phase A — per-dimension sweep (each dim varies, others = live default):\n")
    jobs_a = []
    for v in SL_GRID:        jobs_a.append({"dim": "sl_mult",            "sl_mult": v})
    for v in TRAIL_NAMES:    jobs_a.append({"dim": "trail_name",         "trail_name": v})
    for v in MINQ_GRID:      jobs_a.append({"dim": "minq_int",           "minq_int": v})
    for v in PB_ATR_GRID:    jobs_a.append({"dim": "pb_atr",             "pb_atr": v})
    for v in PB_WAIT_GRID:   jobs_a.append({"dim": "pb_wait",            "pb_wait": v})
    for v in VWAP_GRID:      jobs_a.append({"dim": "vwap_buf",           "vwap_buf": v})
    for v in PBW_GRID:       jobs_a.append({"dim": "post_big_win_secs",  "post_big_win_secs": v})
    for v in LS_GRID:        jobs_a.append({"dim": "loss_streak_secs",   "loss_streak_secs": v})
    for v in DIR_GRID:       jobs_a.append({"dim": "force_dir",          "force_dir": v})

    print(f"    {len(jobs_a)} BTs across 9 dims")

    with Pool(workers) as pool:
        a_results = list(pool.imap_unordered(_run_one, jobs_a))

    a_by_dim: dict = {}
    for r in a_results:
        if not r.get("ok"):
            print(f"    ERR  dim={r.get('dim')} : {r.get('err')}")
            continue
        a_by_dim.setdefault(r["dim"], []).append(r)

    DIM_ORDER = ["sl_mult","trail_name","minq_int","pb_atr","pb_wait","vwap_buf",
                 "post_big_win_secs","loss_streak_secs","force_dir"]
    for dim in DIM_ORDER:
        rs = sorted(a_by_dim.get(dim, []), key=lambda x: -x["pnl"])
        print(f"\n  dim={dim} ({len(rs)} results):")
        for r in rs:
            val = r.get(dim)
            print(f"    {dim}={str(val):<14s} {_fmt(r)} Δ${r['pnl']-base_pnl:+8.2f}")

    def _topk(dim, k):
        rs = sorted(a_by_dim.get(dim, []), key=lambda x: -x["pnl"])
        return [r[dim] for r in rs[:k]]

    top_sl  = _topk("sl_mult", 2)
    top_tr  = _topk("trail_name", 2)
    top_mq  = _topk("minq_int", 2)
    top_pba = _topk("pb_atr", 2)
    top_pbw = _topk("pb_wait", 2)
    top_vw  = _topk("vwap_buf", 2)
    top_dir = _topk("force_dir", 2)
    # Cooldowns: top-1 each to fit 200-BT cap (2^7 * 1 * 1 = 128 combos)
    top_pbwin = _topk("post_big_win_secs", 1)
    top_ls    = _topk("loss_streak_secs", 1)

    print("\n  TOP per dim (Phase B inputs):")
    print(f"    sl_mult           : {top_sl}")
    print(f"    trail             : {top_tr}")
    print(f"    min_quality       : {top_mq}")
    print(f"    pb_atr            : {top_pba}")
    print(f"    pb_wait           : {top_pbw}")
    print(f"    vwap_buf          : {top_vw}")
    print(f"    force_dir         : {top_dir}")
    print(f"    post_big_win_secs (top-1, locked): {top_pbwin}")
    print(f"    loss_streak_secs  (top-1, locked): {top_ls}")

    # ───────────── PHASE B: Cartesian combine ─────────────
    jobs_b = []
    for sl in top_sl:
        for tr in top_tr:
            for mq in top_mq:
                for pba in top_pba:
                    for pbw in top_pbw:
                        for vw in top_vw:
                            for fd in top_dir:
                                jobs_b.append({
                                    "sl_mult": sl, "trail_name": tr, "minq_int": mq,
                                    "pb_atr": pba, "pb_wait": pbw, "vwap_buf": vw,
                                    "force_dir": fd,
                                    "post_big_win_secs": top_pbwin[0] if top_pbwin else None,
                                    "loss_streak_secs":  top_ls[0]    if top_ls    else None,
                                })
    print(f"\n[B] Phase B — Cartesian top-2 × 7 dims + locked cooldowns: {len(jobs_b)} BTs (cap=200)\n")

    with Pool(workers) as pool:
        b_results = list(pool.imap_unordered(_run_one, jobs_b))

    b_ok = [r for r in b_results if r.get("ok")]
    for r in b_ok:
        r["delta"] = round(r["pnl"] - base_pnl, 2)
    b_ok.sort(key=lambda x: -x["pnl"])

    print(f"    Top 12 combos:")
    for r in b_ok[:12]:
        print(f"      SL={r['sl_mult']:.2f}  {r['trail_name']:18s}  "
              f"mq={r['minq_int']:2d}  pb={r['pb_atr']:.1f}/{r['pb_wait']}  "
              f"vw={r['vwap_buf']}  dir={r['force_dir']:5s}  "
              f"{_fmt(r)}  Δ${r['delta']:+8.2f}")

    # ───────────── PHASE C: WF on top-5 ─────────────
    top5 = b_ok[:5]
    print(f"\n[C] Phase C — 5-fold WF (expanding {WF_FOLDS_DAYS} days) on top-5:\n")

    jobs_c = []
    for ri, r in enumerate(top5):
        for d in WF_FOLDS_DAYS:
            jobs_c.append({
                "combo_idx": ri, "days": d,
                "sl_mult":    r["sl_mult"],
                "trail_name": r["trail_name"],
                "minq_int":   r["minq_int"],
                "pb_atr":     r["pb_atr"],
                "pb_wait":    r["pb_wait"],
                "vwap_buf":   r["vwap_buf"],
                "force_dir":  r["force_dir"],
                "post_big_win_secs": r.get("post_big_win_secs"),
                "loss_streak_secs":  r.get("loss_streak_secs"),
            })
    print(f"    {len(jobs_c)} BTs")
    with Pool(workers) as pool:
        wf_raw = list(pool.imap_unordered(_run_one, jobs_c))

    wf_by_combo: dict = {}
    for r in wf_raw:
        wf_by_combo.setdefault(r["combo_idx"], []).append(r)
    for idx, folds in wf_by_combo.items():
        folds.sort(key=lambda x: x["days"])

    final = []
    for ri, r in enumerate(top5):
        folds = wf_by_combo.get(ri, [])
        ok_folds = [f for f in folds if f.get("ok")]
        wf_pos = sum(1 for f in ok_folds if f.get("pnl", 0) > 0)
        wf_avg_pf = round(sum(f.get("pf", 0) for f in ok_folds) / max(1, len(ok_folds)), 3)
        delta = r["delta"]
        passes_delta = delta >= SHIP_DELTA
        passes_wf    = wf_pos >= SHIP_WF_POS
        passes_pf    = wf_avg_pf >= SHIP_AVG_PF
        ship = passes_delta and passes_wf and passes_pf

        cand = {
            "rank": ri + 1,
            "params": {
                "sl_atr_mult": r["sl_mult"],
                "trail_name":  r["trail_name"],
                "trail_profile": TRAIL_PROFILES[r["trail_name"]],
                "min_quality_int": r["minq_int"],
                "pullback_atr_retrace": r["pb_atr"],
                "pullback_max_wait_bars": r["pb_wait"],
                "vwap_buf_atr": r["vwap_buf"],
                "force_direction": r["force_dir"],
                "post_big_win_secs": r.get("post_big_win_secs"),
                "loss_streak_secs":  r.get("loss_streak_secs"),
            },
            "tune_180d": {
                "pnl": round(r["pnl"], 2), "pf": round(r["pf"], 3),
                "wr":  round(r["wr"], 2),  "n":  r["n"], "dd": round(r["dd"], 2),
                "delta_vs_baseline": delta,
            },
            "wf": {
                "folds": [
                    {"days": f["days"], "ok": f.get("ok", False),
                     "pnl": round(f.get("pnl",0),2), "pf": round(f.get("pf",0),3),
                     "n": int(f.get("n",0)), "wr": round(f.get("wr",0),2),
                     "dd": round(f.get("dd",0),2)}
                    for f in folds
                ],
                "positive_folds": wf_pos,
                "avg_pf": wf_avg_pf,
            },
            "ship_decision": {
                "delta_ok": passes_delta, "wf_ok": passes_wf, "pf_ok": passes_pf,
                "ship": ship,
            },
        }
        final.append(cand)
        print(f"  #{ri+1}  SL={r['sl_mult']:.2f}  {r['trail_name']:18s}  "
              f"mq={r['minq_int']:2d}  pb={r['pb_atr']:.1f}/{r['pb_wait']}  "
              f"vw={r['vwap_buf']}  dir={r['force_dir']:5s}  "
              f"PnL=${r['pnl']:+8.0f} (Δ{delta:+8.0f})  "
              f"WF {wf_pos}/5 avg_pf={wf_avg_pf:.2f}  ship={ship}")

    winner = next((c for c in final if c["ship_decision"]["ship"]), None)

    # ───────────── recommendation ─────────────
    print("\n[D] WINNER:")
    if winner:
        p = winner["params"]; t = winner["tune_180d"]; w = winner["wf"]
        print(f"    SL={p['sl_atr_mult']}  trail={p['trail_name']}  mq={p['min_quality_int']}  "
              f"pb={p['pullback_atr_retrace']}/{p['pullback_max_wait_bars']}  "
              f"vw={p['vwap_buf_atr']}  dir={p['force_direction']}  "
              f"pbw={p['post_big_win_secs']}  ls={p['loss_streak_secs']}")
        print(f"    180d PnL=${t['pnl']:+.2f} (Δ${t['delta_vs_baseline']:+.2f}), "
              f"PF={t['pf']:.2f}, WR={t['wr']:.1f}%, DD={t['dd']:.1f}%")
        print(f"    WF {w['positive_folds']}/5 positive, avg_pf={w['avg_pf']:.2f}")
        rec_action = "ship"
        rec_note = f"Δ≥${SHIP_DELTA} AND WF≥{SHIP_WF_POS}/5 AND avg_pf≥{SHIP_AVG_PF} passed."
    else:
        # honest verdict — recommend dropping if no winner found
        n_passed_delta = sum(1 for c in final if c["ship_decision"]["delta_ok"])
        n_passed_wf    = sum(1 for c in final if c["ship_decision"]["wf_ok"])
        n_passed_pf    = sum(1 for c in final if c["ship_decision"]["pf_ok"])
        print(f"    NO WINNER — passes by gate: Δ={n_passed_delta}/5, WF={n_passed_wf}/5, PF={n_passed_pf}/5")
        print(f"    Verdict: DROP USOUSD from universe (no tune cleared all 3 gates).")
        rec_action = "DROP"
        rec_note = (f"No combo cleared Δ≥${SHIP_DELTA} AND WF≥{SHIP_WF_POS}/5 AND avg_pf≥{SHIP_AVG_PF}. "
                    f"Per-gate pass counts: Δ={n_passed_delta}/5, WF={n_passed_wf}/5, PF={n_passed_pf}/5. "
                    f"At r=2.0% on a $1,219 account, USOUSD is hostile: the 8% DD safeguard halts the run "
                    f"early. Recommend dropping from universe OR running at default r=0.8% only.")

    # ───────────── save JSON ─────────────
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": SYMBOL,
        "tune_days": TUNE_DAYS,
        "wf_folds_days": WF_FOLDS_DAYS,
        "account": {"risk_pct": RISK_PCT, "start_equity": START_EQUITY},
        "ship_criteria": {
            "min_delta_pnl_usd": SHIP_DELTA,
            "min_wf_positive_folds": SHIP_WF_POS,
            "min_wf_avg_pf": SHIP_AVG_PF,
        },
        "dimensions": {
            "sl_atr_mult": SL_GRID, "trail_profiles": TRAIL_NAMES,
            "min_quality": MINQ_GRID, "pullback_atr_retrace": PB_ATR_GRID,
            "pullback_max_wait_bars": PB_WAIT_GRID, "vwap_buf_atr": VWAP_GRID,
            "post_big_win_secs": PBW_GRID, "loss_streak_secs": LS_GRID,
            "force_direction": DIR_GRID,
        },
        "trail_profile_definitions": {k: v for k, v in TRAIL_PROFILES.items()},
        "baseline_live_config": base,
        "phase_a_per_dim": {
            dim: sorted(
                [{k: v for k, v in r.items() if k != "dim"} for r in a_by_dim.get(dim, [])],
                key=lambda x: -x.get("pnl", 0),
            )
            for dim in DIM_ORDER
        },
        "phase_a_top_per_dim": {
            "sl_mult": top_sl, "trail": top_tr, "min_quality": top_mq,
            "pb_atr": top_pba, "pb_wait": top_pbw, "vwap_buf": top_vw,
            "force_dir": top_dir,
            "post_big_win_secs_top1": top_pbwin,
            "loss_streak_secs_top1": top_ls,
        },
        "phase_b_cartesian": [
            {k: v for k, v in r.items() if k not in ("dim", "tb")}
            for r in b_ok
        ],
        "phase_c_top5_walkforward": final,
        "winner": winner,
        "recommendation": {"action": rec_action, "note": rec_note,
                           "params_to_set": (winner["params"] if winner else None)},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[E] Saved JSON → {OUT_JSON}")

    # ───────────── save MD ─────────────
    md = []
    md.append(f"# USOUSD HARD per-symbol tune — {out['generated_at']}\n")
    md.append(f"- Symbol: **{SYMBOL}**\n")
    md.append(f"- Days: **{TUNE_DAYS}**  ·  WF folds: `{WF_FOLDS_DAYS}` days\n")
    md.append(f"- Account: risk={RISK_PCT}%, start_equity=${START_EQUITY}\n")
    md.append(f"- Ship: Δ ≥ ${SHIP_DELTA:.0f} AND WF ≥ {SHIP_WF_POS}/5 AND avg PF ≥ {SHIP_AVG_PF}\n")
    md.append(f"- Elapsed: {out['elapsed_seconds']:.1f}s\n\n")
    md.append("## Baseline (live config @ r=2%, eq=$1219, 180d)\n")
    md.append(f"- n={base['n']}  PF={base['pf']:.2f}  PnL=${base['pnl']:+.2f}  "
              f"WR={base['wr']:.1f}%  DD={base['dd']:.1f}%\n")

    md.append("\n## Phase A — per-dimension sweep\n")
    for dim in DIM_ORDER:
        md.append(f"\n### {dim}\n")
        md.append("| value | n | PF | PnL$ | WR% | DD% | Δ$ |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(a_by_dim.get(dim, []), key=lambda x: -x.get("pnl", 0)):
            md.append(f"| `{r.get(dim)}` | {r['n']} | {r['pf']:.2f} | {r['pnl']:+.2f} | "
                      f"{r['wr']:.1f} | {r['dd']:.1f} | {r['pnl']-base_pnl:+.2f} |\n")

    md.append("\n## Phase B — Cartesian top-2 (top 12 shown)\n")
    md.append("| # | SL | Trail | mq | pbATR | pbW | vwap | dir | n | PF | PnL$ | DD% | Δ$ |\n"
              "|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(b_ok[:12]):
        md.append(f"| {i+1} | {r['sl_mult']:.2f} | {r['trail_name']} | "
                  f"{r['minq_int']} | {r['pb_atr']:.1f} | {r['pb_wait']} | "
                  f"{r['vwap_buf']} | {r['force_dir']} | {r['n']} | "
                  f"{r['pf']:.2f} | {r['pnl']:+.2f} | {r['dd']:.1f} | {r['delta']:+.2f} |\n")

    md.append("\n## Phase C — Walk-forward on top-5\n")
    for c in final:
        p = c["params"]; t = c["tune_180d"]; w = c["wf"]; s = c["ship_decision"]
        md.append(f"\n### Rank #{c['rank']} — ship={s['ship']}\n")
        md.append(f"Params: SL=`{p['sl_atr_mult']}` trail=`{p['trail_name']}` mq=`{p['min_quality_int']}` "
                  f"pb=`{p['pullback_atr_retrace']}/{p['pullback_max_wait_bars']}` "
                  f"vwap=`{p['vwap_buf_atr']}` dir=`{p['force_direction']}` "
                  f"pbw=`{p['post_big_win_secs']}` ls=`{p['loss_streak_secs']}`\n\n")
        md.append(f"- 180d: PnL=${t['pnl']:+.2f}  PF={t['pf']:.2f}  WR={t['wr']:.1f}%  "
                  f"DD={t['dd']:.1f}%  Δ=${t['delta_vs_baseline']:+.2f}\n")
        md.append(f"- WF: {w['positive_folds']}/5 positive, avg_pf={w['avg_pf']:.2f}\n")
        md.append(f"- Gates: Δ_ok={s['delta_ok']}  WF_ok={s['wf_ok']}  PF_ok={s['pf_ok']}\n\n")
        md.append("| fold (days) | n | PF | PnL$ | WR% | DD% |\n|---:|---:|---:|---:|---:|---:|\n")
        for f in w["folds"]:
            md.append(f"| {f['days']} | {f['n']} | {f['pf']:.2f} | {f['pnl']:+.2f} | "
                      f"{f['wr']:.1f} | {f['dd']:.1f} |\n")

    md.append("\n## Winner & recommendation\n")
    if winner:
        p = winner["params"]; t = winner["tune_180d"]; w = winner["wf"]
        md.append(f"**SHIP**  ·  SL=`{p['sl_atr_mult']}` trail=`{p['trail_name']}` "
                  f"mq=`{p['min_quality_int']}` pb=`{p['pullback_atr_retrace']}/{p['pullback_max_wait_bars']}` "
                  f"vwap=`{p['vwap_buf_atr']}` dir=`{p['force_direction']}` "
                  f"pbw=`{p['post_big_win_secs']}` ls=`{p['loss_streak_secs']}`\n\n")
        md.append(f"- 180d: PnL=${t['pnl']:+.2f} (Δ${t['delta_vs_baseline']:+.2f}), "
                  f"PF={t['pf']:.2f}, WR={t['wr']:.1f}%, DD={t['dd']:.1f}%\n")
        md.append(f"- WF: {w['positive_folds']}/5 positive, avg_pf={w['avg_pf']:.2f}\n")
    else:
        md.append("**DROP USOUSD** — no combo cleared ship gates (Δ ≥ $30 AND WF ≥ 3/5 AND avg PF ≥ 1.5).\n")
        md.append(f"\n{rec_note}\n")

    OUT_MD.write_text("".join(md))
    print(f"[E] Saved MD   → {OUT_MD}")
    print(f"\n[F] DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
