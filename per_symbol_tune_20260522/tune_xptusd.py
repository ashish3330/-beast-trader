#!/usr/bin/env python3 -B
"""
XPTUSD.r per-symbol fine-tune  —  2026-05-22.

READ-ONLY (no source edits to backtest/, config.py, auto_tuned.py).
Constants are mutated at runtime via importlib.reload + attribute mutation.
VWAP buffer is patched by rebuilding `backtest_symbol` from source with the
0.5 constant swapped (per-worker, in-memory only).

Phases:
  A) per-dimension sweep over 7 axes
  B) top-2 per dim Cartesian combine (128 BTs)
  C) WF 5-fold on top-5 (trailing 60/90/120/150/180 days)

Ship criteria: Δ ≥ $100 AND WF >= 3/5.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "per_symbol_tune_20260522"
OUT_JSON = OUT_DIR / "XPTUSD.r.json"
OUT_MD   = OUT_DIR / "XPTUSD.r.md"

SYMBOL = "XPTUSD.r"
TUNE_DAYS = 180
WF_FOLDS_DAYS = [60, 90, 120, 150, 180]

# ───────────────── dimension grids ─────────────────
SL_GRID   = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

TRAIL_PROFILES = {
    "_TIGHT_LOCK":       [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_WIDE_RUNNER":      [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)],
    "_RANGE_TIGHT":      [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0)],
    "_TREND_LOOSE":      [(15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5), (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_AGGR_LOCK":        [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8), (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "_RUNNER_NO_BE":     [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5), (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)],
    "_WIDE_RUNNER_BE07": [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)],
}
TRAIL_NAMES = list(TRAIL_PROFILES)

PB_ATR_GRID  = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_GRID = [3, 4, 5, 6, 8]
VWAP_GRID    = [0.0, 0.3, 0.5, 0.7, 1.0]   # 0.0 = disabled (NaN vwap series)
MINQ_GRID    = [28, 30, 33, 35, 38, 40]
DIR_GRID     = ["LONG", "SHORT", "BOTH"]

SHIP_DELTA = 100.0
SHIP_WF_POS = 3   # >=3/5 folds positive

# Live-baseline regime min_quality for XPTUSD.r (from auto_tuned.py line 163)
LIVE_MINQ_BY_REGIME = {"trending": 37, "ranging": 42, "volatile": 30, "low_vol": 37}

# ──────────────────────────────────────────────────
# helpers to drive a backtest with overrides — runs INSIDE each worker.
# Reload of config + bt happens once per worker process (Pool initializer)
# so we re-patch state cheaply across many jobs.
# ──────────────────────────────────────────────────

_WORKER_BT_ORIGINALS: dict = {}


def _get_bt_module(vwap_buf: float):
    """Return a backtest module whose backtest_symbol has the inline VWAP
    0.5 multiplier replaced by `vwap_buf`. vwap_buf == 0.0 -> disabled.
    Re-applies the patch every call so we never inherit stale state from a
    previous job in the same worker."""
    key = round(float(vwap_buf), 4)
    import backtest.v5_backtest as _bt
    # Cache the truly-pristine references the first time we see the module,
    # then reset to them on each call.
    if "fn" not in _WORKER_BT_ORIGINALS:
        _WORKER_BT_ORIGINALS["fn"] = _bt.backtest_symbol
        _WORKER_BT_ORIGINALS["compute"] = _bt._compute_indicators
    _bt.backtest_symbol = _WORKER_BT_ORIGINALS["fn"]
    _bt._compute_indicators = _WORKER_BT_ORIGINALS["compute"]

    if key == 0.5:
        return _bt

    if key <= 0.0:
        # Disable VWAP filter by neutering the indicator series at runtime.
        # Patch the BT module's _compute_indicators binding (NOT the source
        # module's), since `from signals.momentum_scorer import _compute_indicators`
        # creates a local name in bt's namespace that doesn't see edits to
        # momentum_scorer.
        orig_bt_fn = _WORKER_BT_ORIGINALS["fn"]
        orig_compute = _WORKER_BT_ORIGINALS["compute"]
        import numpy as _np

        def _patched_compute(df, icfg):
            ind = orig_compute(df, icfg)
            if ind is not None and "vwap" in ind and ind["vwap"] is not None:
                arr = _np.asarray(ind["vwap"], dtype=float).copy()
                arr[:] = _np.nan
                ind["vwap"] = arr
            return ind

        def _bt_disabled(symbol, days=90, params=None, verbose=True, **kw):
            _bt._compute_indicators = _patched_compute
            try:
                return orig_bt_fn(symbol, days=days, params=params, verbose=verbose, **kw)
            finally:
                _bt._compute_indicators = orig_compute

        _bt.backtest_symbol = _bt_disabled
        return _bt

    # Replace inline VWAP buffer constant via source patch + exec.
    src = inspect.getsource(_WORKER_BT_ORIGINALS["fn"])
    target = 'atr_buf = float(ind["at"][bi]) * 0.5'
    replacement = f'atr_buf = float(ind["at"][bi]) * {key}'
    if target not in src:
        raise RuntimeError("VWAP-buf inline constant not found; aborting.")
    new_src = src.replace(target, replacement, 1)

    # Compile new function inside the module's globals so all helpers resolve.
    g = _bt.__dict__
    loc = {}
    exec(compile(new_src, f"<vwap_buf={key}>", "exec"), g, loc)
    _bt.backtest_symbol = loc["backtest_symbol"]
    return _bt


def _apply_config_overrides(pullback_atr: float | None, pullback_wait: int | None):
    """Mutate live config attributes so the inline `from config import ...`
    inside backtest_symbol picks up our values."""
    import config as cfg
    if pullback_atr is not None:
        cfg.PULLBACK_ATR_RETRACE = float(pullback_atr)
    if pullback_wait is not None:
        cfg.PULLBACK_MAX_WAIT_BARS = int(pullback_wait)


def _run_one(job):
    """
    job = dict with keys:
      days, sl_mult, trail_name, pb_atr, pb_wait, vwap_buf, minq_int, force_dir
    Any None = "use baseline/live default for that dim".
    """
    try:
        days       = int(job.get("days", TUNE_DAYS))
        sl_mult    = job.get("sl_mult")
        trail_name = job.get("trail_name")
        pb_atr     = job.get("pb_atr")
        pb_wait    = job.get("pb_wait")
        vwap_buf   = job.get("vwap_buf", 0.5)
        minq_int   = job.get("minq_int")
        force_dir  = job.get("force_dir")

        # Reload config so we have a fresh copy each job, then mutate.
        import config as cfg
        importlib.reload(cfg)
        _apply_config_overrides(pb_atr, pb_wait)

        # Get a fresh BT module bound to the right VWAP constant.
        # NB: vwap_buf == None -> baseline 0.5.
        eff_vwap = 0.5 if vwap_buf is None else float(vwap_buf)
        bt = _get_bt_module(eff_vwap)

        # Build params dict — start from live params, override per-dim.
        params = {}
        if sl_mult is not None:
            params["sl_atr_mult"] = float(sl_mult)
            # Defeat per-symbol AND per-(sym, regime) SL_OVERRIDE for this run
            # by mutating module-level dicts on the chosen BT module.
            bt.SL_OVERRIDE = dict(bt.SL_OVERRIDE)
            bt.SL_OVERRIDE[SYMBOL] = float(sl_mult)
            bt.SL_OVERRIDE_REGIME = dict(bt.SL_OVERRIDE_REGIME)
            bt.SL_OVERRIDE_REGIME[SYMBOL] = {}
        if trail_name is not None:
            params["force_trail"] = TRAIL_PROFILES[trail_name]
            # also blank per-(sym, regime) trail override so force_trail wins
            bt.TRAIL_OVERRIDE_REGIME = dict(bt.TRAIL_OVERRIDE_REGIME)
            bt.TRAIL_OVERRIDE_REGIME[SYMBOL] = {}
        if minq_int is not None:
            # Single int → apply to ALL regimes
            mq = {"trending": int(minq_int), "ranging": int(minq_int),
                  "volatile": int(minq_int), "low_vol": int(minq_int)}
            params["min_quality"] = mq
        if force_dir is not None:
            params["force_direction"] = force_dir
            # Also blank live's per-(sym, regime) DIR bias so force_direction is sole driver
            bt._DIR_BIAS_REGIME_STR = dict(bt._DIR_BIAS_REGIME_STR)
            bt._DIR_BIAS_REGIME_STR[SYMBOL] = {}

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
        import traceback
        return {**job, "ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc(limit=2)}


def _baseline():
    """Live-config baseline (no per-dim overrides)."""
    import config as cfg
    importlib.reload(cfg)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    r = bt.backtest_symbol(SYMBOL, days=TUNE_DAYS, verbose=False)
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


# ──────────────────────────────────────────────────
def _fmt(r, k_extra=""):
    return (f"n={r.get('n',0):4d} PF={r.get('pf',0):5.2f} "
            f"PnL=${r.get('pnl',0):+9.2f} WR={r.get('wr',0):5.1f}% "
            f"DD={r.get('dd',0):4.1f}%{k_extra}")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nXPTUSD.r per-symbol fine tune  —  {datetime.now().isoformat(timespec='seconds')}")
    print(f"Days={TUNE_DAYS}  WF folds={WF_FOLDS_DAYS}  Output={OUT_JSON}\n")

    workers = max(2, min(8, cpu_count() or 4))
    print(f"Pool workers: {workers}\n")

    # ───────────── BASELINE ─────────────
    print("[0] BASELINE (live config, no overrides):")
    base = _baseline()
    print(f"    {_fmt(base)}")

    base_pnl = base["pnl"]

    # ───────────── PHASE A: per-dim sweeps ─────────────
    print(f"\n[A] Phase A — per-dimension sweep (each dim varies, others = baseline):\n")
    phase_a_jobs = []
    # 1) SL
    for v in SL_GRID:
        phase_a_jobs.append({"dim": "sl_mult",    "sl_mult": v})
    # 2) Trail
    for v in TRAIL_NAMES:
        phase_a_jobs.append({"dim": "trail_name", "trail_name": v})
    # 3) Pullback ATR
    for v in PB_ATR_GRID:
        phase_a_jobs.append({"dim": "pb_atr",     "pb_atr": v})
    # 4) Pullback wait
    for v in PB_WAIT_GRID:
        phase_a_jobs.append({"dim": "pb_wait",    "pb_wait": v})
    # 5) VWAP buf
    for v in VWAP_GRID:
        phase_a_jobs.append({"dim": "vwap_buf",   "vwap_buf": v})
    # 6) min_quality (single int)
    for v in MINQ_GRID:
        phase_a_jobs.append({"dim": "minq_int",   "minq_int": v})
    # 7) Direction
    for v in DIR_GRID:
        phase_a_jobs.append({"dim": "force_dir",  "force_dir": v})

    print(f"    {len(phase_a_jobs)} BTs across 7 dims")

    with Pool(workers) as pool:
        a_results = list(pool.imap_unordered(_run_one, phase_a_jobs))

    # group by dim
    a_by_dim: dict[str, list[dict]] = {}
    for r in a_results:
        if not r.get("ok"):
            print(f"    ERR  dim={r.get('dim')} : {r.get('err')}")
            continue
        a_by_dim.setdefault(r["dim"], []).append(r)
    for dim in ["sl_mult", "trail_name", "pb_atr", "pb_wait", "vwap_buf", "minq_int", "force_dir"]:
        rs = sorted(a_by_dim.get(dim, []), key=lambda x: -x["pnl"])
        print(f"\n  dim={dim} ({len(rs)} results):")
        for r in rs:
            val_key = dim
            val = r.get(val_key)
            print(f"    {dim}={str(val):<18s} {_fmt(r)} Δ${r['pnl']-base_pnl:+8.2f}")

    # Pick TOP-2 per dim (ranked by PnL; trades>=10 floor to filter degenerate)
    def _top2(dim: str, key_field: str):
        rs = [r for r in a_by_dim.get(dim, []) if r.get("n", 0) >= 10]
        rs.sort(key=lambda x: -x["pnl"])
        if len(rs) < 2:
            rs2 = sorted(a_by_dim.get(dim, []), key=lambda x: -x["pnl"])
            rs = rs2[:2]
        return [r[key_field] for r in rs[:2]]

    top_sl   = _top2("sl_mult", "sl_mult")
    top_tr   = _top2("trail_name", "trail_name")
    top_pba  = _top2("pb_atr", "pb_atr")
    top_pbw  = _top2("pb_wait", "pb_wait")
    top_vw   = _top2("vwap_buf", "vwap_buf")
    top_mq   = _top2("minq_int", "minq_int")
    top_dir  = _top2("force_dir", "force_dir")

    print("\n  TOP-2 per dim:")
    print(f"    sl_mult   : {top_sl}")
    print(f"    trail     : {top_tr}")
    print(f"    pb_atr    : {top_pba}")
    print(f"    pb_wait   : {top_pbw}")
    print(f"    vwap_buf  : {top_vw}")
    print(f"    minq      : {top_mq}")
    print(f"    force_dir : {top_dir}")

    # ───────────── PHASE B: Cartesian combine ─────────────
    phase_b_jobs = []
    for sl in top_sl:
        for tr in top_tr:
            for pba in top_pba:
                for pbw in top_pbw:
                    for vw in top_vw:
                        for mq in top_mq:
                            for fd in top_dir:
                                phase_b_jobs.append({
                                    "sl_mult": sl,
                                    "trail_name": tr,
                                    "pb_atr": pba,
                                    "pb_wait": pbw,
                                    "vwap_buf": vw,
                                    "minq_int": mq,
                                    "force_dir": fd,
                                })
    print(f"\n[B] Phase B — Cartesian combine of TOP-2 per dim: {len(phase_b_jobs)} BTs")
    print(f"    (expected 2^7 = 128)")

    with Pool(workers) as pool:
        b_results = list(pool.imap_unordered(_run_one, phase_b_jobs))

    b_ok = [r for r in b_results if r.get("ok")]
    b_ok.sort(key=lambda x: -x["pnl"])

    # Δ vs baseline + filter
    for r in b_ok:
        r["delta"] = round(r["pnl"] - base_pnl, 2)

    print(f"\n    Top 12 combos:")
    for r in b_ok[:12]:
        print(f"      SL={r['sl_mult']:.2f}  {r['trail_name']:18s}  "
              f"pbATR={r['pb_atr']:.2f}  pbW={r['pb_wait']}  "
              f"vw={r['vwap_buf']}  mq={r['minq_int']}  dir={r['force_dir']:5s}  "
              f"{_fmt(r)}  Δ${r['delta']:+8.2f}")

    # ───────────── PHASE C: WF on top-5 ─────────────
    top5 = b_ok[:5]
    print(f"\n[C] Phase C — Walk-forward 5-fold on top-5 combos "
          f"(days = {WF_FOLDS_DAYS}):\n")

    wf_jobs = []
    for ri, r in enumerate(top5):
        for d in WF_FOLDS_DAYS:
            wf_jobs.append({
                "combo_idx": ri,
                "days": d,
                "sl_mult":    r["sl_mult"],
                "trail_name": r["trail_name"],
                "pb_atr":     r["pb_atr"],
                "pb_wait":    r["pb_wait"],
                "vwap_buf":   r["vwap_buf"],
                "minq_int":   r["minq_int"],
                "force_dir":  r["force_dir"],
            })
    print(f"    {len(wf_jobs)} BTs")
    with Pool(workers) as pool:
        wf_raw = list(pool.imap_unordered(_run_one, wf_jobs))
    wf_by_combo: dict[int, list[dict]] = {}
    for r in wf_raw:
        wf_by_combo.setdefault(r["combo_idx"], []).append(r)

    # Sort each by days
    for idx, folds in wf_by_combo.items():
        folds.sort(key=lambda x: x["days"])

    # Print + evaluate winner gates
    final_candidates = []
    for ri, r in enumerate(top5):
        folds = wf_by_combo.get(ri, [])
        wf_pos = sum(1 for f in folds if f.get("ok") and f.get("pnl", 0) > 0)
        wf_avg_pf = round(
            sum(f.get("pf", 0) for f in folds if f.get("ok")) / max(1, sum(1 for f in folds if f.get("ok"))),
            3,
        )
        delta = r["delta"]
        passes_delta = delta >= SHIP_DELTA
        passes_wf    = wf_pos >= SHIP_WF_POS
        ship = passes_delta and passes_wf

        cand = {
            "rank": ri + 1,
            "params": {
                "sl_atr_mult":   r["sl_mult"],
                "trail_name":    r["trail_name"],
                "trail_profile": TRAIL_PROFILES[r["trail_name"]],
                "pullback_atr_retrace": r["pb_atr"],
                "pullback_max_wait_bars": r["pb_wait"],
                "vwap_buf_atr":  r["vwap_buf"],
                "min_quality_int": r["minq_int"],
                "force_direction": r["force_dir"],
            },
            "tune_180d": {
                "pnl": round(r["pnl"], 2),
                "pf":  round(r["pf"], 3),
                "wr":  round(r["wr"], 2),
                "n":   r["n"],
                "dd":  round(r["dd"], 2),
                "delta_vs_baseline": delta,
            },
            "wf": {
                "folds": [
                    {
                        "days": f["days"],
                        "ok":  f.get("ok", False),
                        "pnl": round(f.get("pnl", 0), 2),
                        "pf":  round(f.get("pf", 0), 3),
                        "n":   int(f.get("n", 0)),
                        "wr":  round(f.get("wr", 0), 2),
                        "dd":  round(f.get("dd", 0), 2),
                    } for f in folds
                ],
                "positive_folds": wf_pos,
                "avg_pf": wf_avg_pf,
            },
            "ship_decision": {
                "delta_ok": passes_delta,
                "wf_ok":    passes_wf,
                "ship":     ship,
            },
        }
        final_candidates.append(cand)
        print(f"  #{ri+1}  SL={r['sl_mult']:.2f}  {r['trail_name']:18s}  "
              f"pb={r['pb_atr']:.2f}/{r['pb_wait']}  vw={r['vwap_buf']}  "
              f"mq={r['minq_int']}  dir={r['force_dir']:5s}  "
              f"PnL=${r['pnl']:+8.0f} (Δ{delta:+8.0f})  "
              f"WF {wf_pos}/5 avg_pf={wf_avg_pf:.2f}  ship={ship}")

    # ───────────── pick winner ─────────────
    winner = next((c for c in final_candidates if c["ship_decision"]["ship"]), None)

    print("\n[D] WINNER:")
    if winner:
        p = winner["params"]
        print(f"    SL={p['sl_atr_mult']}  trail={p['trail_name']}  "
              f"pb={p['pullback_atr_retrace']}/{p['pullback_max_wait_bars']}  "
              f"vw={p['vwap_buf_atr']}  mq={p['min_quality_int']}  dir={p['force_direction']}")
        t = winner["tune_180d"]
        w = winner["wf"]
        print(f"    180d PnL=${t['pnl']:+.2f} (Δ${t['delta_vs_baseline']:+.2f})  PF={t['pf']:.2f}  WR={t['wr']:.1f}%  DD={t['dd']:.1f}%")
        print(f"    WF {w['positive_folds']}/5 positive, avg_pf={w['avg_pf']:.2f}")
    else:
        print(f"    NO WINNER — no combo met Δ ≥ ${SHIP_DELTA} AND WF ≥ {SHIP_WF_POS}/5")

    # ───────────── save artifacts ─────────────
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": SYMBOL,
        "tune_days": TUNE_DAYS,
        "wf_folds_days": WF_FOLDS_DAYS,
        "ship_criteria": {
            "min_delta_pnl_usd": SHIP_DELTA,
            "min_wf_positive_folds": SHIP_WF_POS,
        },
        "dimensions": {
            "sl_atr_mult": SL_GRID,
            "trail_profiles": TRAIL_NAMES,
            "pullback_atr_retrace": PB_ATR_GRID,
            "pullback_max_wait_bars": PB_WAIT_GRID,
            "vwap_buf_atr": VWAP_GRID,
            "min_quality": MINQ_GRID,
            "force_direction": DIR_GRID,
        },
        "trail_profile_definitions": TRAIL_PROFILES,
        "live_min_quality_by_regime": LIVE_MINQ_BY_REGIME,
        "baseline": base,
        "phase_a_per_dim": {
            dim: sorted(
                [{k: v for k, v in r.items() if k not in ("dim",)} for r in a_by_dim.get(dim, [])],
                key=lambda x: -x.get("pnl", 0),
            )
            for dim in ["sl_mult", "trail_name", "pb_atr", "pb_wait", "vwap_buf", "minq_int", "force_dir"]
        },
        "phase_a_top2_per_dim": {
            "sl_mult":   top_sl,
            "trail":     top_tr,
            "pb_atr":    top_pba,
            "pb_wait":   top_pbw,
            "vwap_buf":  top_vw,
            "min_quality": top_mq,
            "force_dir": top_dir,
        },
        "phase_b_cartesian": [
            {k: v for k, v in r.items() if k not in ("dim", "tb")}
            for r in b_ok
        ],
        "phase_c_top5_walkforward": final_candidates,
        "winner": winner,
        "recommendation": (
            {
                "action": "ship",
                "params_to_set": winner["params"],
                "note": "Δ ≥ $100 and WF ≥ 3/5 passed.",
            } if winner else
            {
                "action": "hold",
                "note": "No combo cleared ship gates; keep current live config.",
            }
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[E] Saved JSON → {OUT_JSON}")

    # ───────────── write markdown report ─────────────
    md = []
    md.append(f"# XPTUSD.r per-symbol fine-tune — {out['generated_at']}\n")
    md.append(f"- Days: **{TUNE_DAYS}**  ·  WF folds: `{WF_FOLDS_DAYS}` days\n")
    md.append(f"- Ship: Δ ≥ ${SHIP_DELTA:.0f} AND WF ≥ {SHIP_WF_POS}/5\n")
    md.append(f"- Elapsed: {out['elapsed_seconds']:.1f}s\n")
    md.append(f"\n## Baseline (live config)\n")
    md.append(f"- n={base['n']}  PF={base['pf']:.2f}  PnL=${base['pnl']:+.2f}  WR={base['wr']:.1f}%  DD={base['dd']:.1f}%\n")

    md.append("\n## Phase A — per-dimension sweep\n")
    for dim in ["sl_mult", "trail_name", "pb_atr", "pb_wait", "vwap_buf", "minq_int", "force_dir"]:
        md.append(f"\n### {dim}\n")
        md.append("| value | n | PF | PnL$ | WR% | DD% | Δ$ |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(a_by_dim.get(dim, []), key=lambda x: -x.get("pnl", 0)):
            md.append(f"| `{r.get(dim)}` | {r['n']} | {r['pf']:.2f} | {r['pnl']:+.2f} | "
                      f"{r['wr']:.1f} | {r['dd']:.1f} | {r['pnl']-base_pnl:+.2f} |\n")

    md.append("\n## Phase B — Cartesian top-2 per dim (top 12 shown)\n")
    md.append("| # | SL | Trail | pb_atr | pb_wait | vwap | mq | dir | PF | PnL$ | WR% | DD% | Δ$ |\n"
              "|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(b_ok[:12]):
        md.append(f"| {i+1} | {r['sl_mult']:.2f} | {r['trail_name']} | "
                  f"{r['pb_atr']:.2f} | {r['pb_wait']} | {r['vwap_buf']} | "
                  f"{r['minq_int']} | {r['force_dir']} | "
                  f"{r['pf']:.2f} | {r['pnl']:+.2f} | {r['wr']:.1f} | {r['dd']:.1f} | "
                  f"{r['delta']:+.2f} |\n")

    md.append("\n## Phase C — Walk-forward 5-fold on top-5\n")
    for c in final_candidates:
        p = c["params"]; t = c["tune_180d"]; w = c["wf"]; s = c["ship_decision"]
        md.append(f"\n### Rank #{c['rank']} — ship={s['ship']}\n")
        md.append(f"Params: SL=`{p['sl_atr_mult']}`  trail=`{p['trail_name']}`  "
                  f"pb=`{p['pullback_atr_retrace']}/{p['pullback_max_wait_bars']}`  "
                  f"vwap=`{p['vwap_buf_atr']}`  mq=`{p['min_quality_int']}`  dir=`{p['force_direction']}`\n\n")
        md.append(f"- 180d: PnL=${t['pnl']:+.2f}  PF={t['pf']:.2f}  WR={t['wr']:.1f}%  DD={t['dd']:.1f}%  Δ=${t['delta_vs_baseline']:+.2f}\n")
        md.append(f"- WF: {w['positive_folds']}/5 positive, avg_pf={w['avg_pf']:.2f}\n\n")
        md.append("| fold (days) | n | PF | PnL$ | WR% | DD% |\n|---:|---:|---:|---:|---:|---:|\n")
        for f in w["folds"]:
            md.append(f"| {f['days']} | {f['n']} | {f['pf']:.2f} | {f['pnl']:+.2f} | {f['wr']:.1f} | {f['dd']:.1f} |\n")

    md.append("\n## Winner & recommendation\n")
    if winner:
        p = winner["params"]; t = winner["tune_180d"]; w = winner["wf"]
        md.append(f"**SHIP**  ·  SL=`{p['sl_atr_mult']}` trail=`{p['trail_name']}` pb=`{p['pullback_atr_retrace']}/{p['pullback_max_wait_bars']}` "
                  f"vwap=`{p['vwap_buf_atr']}` mq=`{p['min_quality_int']}` dir=`{p['force_direction']}`\n\n")
        md.append(f"- 180d: PnL=${t['pnl']:+.2f} (Δ${t['delta_vs_baseline']:+.2f}), PF={t['pf']:.2f}, WR={t['wr']:.1f}%, DD={t['dd']:.1f}%\n")
        md.append(f"- WF: {w['positive_folds']}/5 positive, avg_pf={w['avg_pf']:.2f}\n")
    else:
        md.append("**HOLD** — no combo met ship gates (Δ ≥ $100 AND WF ≥ 3/5). Keep current live config.\n")

    OUT_MD.write_text("".join(md))
    print(f"[E] Saved MD   → {OUT_MD}")
    print(f"\n[F] DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
