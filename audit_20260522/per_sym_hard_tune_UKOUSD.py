#!/usr/bin/env python3 -B
"""UKOUSD per-symbol HARD-TUNE — mirror-aware BT — 2026-05-23.

Same protocol shape as the SP500.r per-symbol tune that ran on 2026-05-22, but
extended for the 9 dimensions requested in the brief:

  1. SL ATR mult        : {0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0}        (9)
  2. Trail profile      : 7 named profiles                                     (7)
  3. min_quality (all-regime flat) : {22, 25, 28, 30, 33, 35, 38, 40}          (8)
  4. Pullback ATR       : {0.4, 0.6, 0.8, 1.0, 1.2}                            (5)
  5. Pullback max wait  : {3, 5, 7}                                            (3)
  6. VWAP buffer ATR    : {0.0_disabled, 0.3, 0.5, 0.7, 1.0, 1.5}              (6)
  7. POST_BIG_WIN cooldown secs : {1800, 3600, 5400, 7200, 10800}              (5)
  8. LOSS_STREAK cooldown secs  : {3600, 7200, 10800, 14400, 18000}            (5)
  9. Direction bias     : {LONG, SHORT, BOTH}                                  (3)

Anchor (live config as of 2026-05-23, from auto_tuned.py):
  SL=0.2  trail=_TIGHT_LOCK  pb_atr=0.8  pb_wait=5  vwap=0.5
  min_q=25 (all regimes)  PBW=10800s  LS=18000s  dir_bias=LONG

Strategy (cap 200 Phase-B BTs):
  Phase A : full SL × trail grid (9 × 7 = 63 BTs). Top-2 seeds by PnL.
  Phase B : coordinate descent on each top-2 seed:
              B1: pullback ATR × wait        (5×3 = 15)
              B2: VWAP × min_q               (6×8 = 48)
              B3: PBW × LS cooldown          (5×5 = 25)
              B4: dir_bias                   (3)
            Per seed = 91 BTs;  2 seeds = 182 BTs (≤ 200 cap).
  Phase C : top-3 finalists × 5-fold disjoint walk-forward (15 BTs).

Ship rule (per brief):
  Δ ≥ +$30  AND  ≥ 3/5 WF folds positive  AND  WF avg PF ≥ 1.5

Account equity: $1,219  →  risk_pct = 2.0  →  start_equity = 1219 in BT.

READ-ONLY on source files. python3 -B. multiprocessing.Pool. Cap 2h.

Outputs:
  /Users/ashish/Documents/beast-trader/audit_20260522/per_sym_hard_tune_UKOUSD.json
  /Users/ashish/Documents/beast-trader/audit_20260522/per_sym_hard_tune_UKOUSD.md
"""
import json
import os
import sys
import time
import traceback
import types
import importlib
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

SYMBOL    = "UKOUSD"
DAYS      = 180
OUT_DIR   = ROOT / "audit_20260522"
OUT_JSON  = OUT_DIR / "per_sym_hard_tune_UKOUSD.json"
OUT_MD    = OUT_DIR / "per_sym_hard_tune_UKOUSD.md"
LOG_FILE  = OUT_DIR / "per_sym_hard_tune_UKOUSD.log"

# Account equity & risk
START_EQUITY = 1219.0
RISK_PCT     = 2.0

# ── Dimensions ──
SL_GRID         = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
TRAIL_NAMES     = ["_TIGHT_LOCK", "_WIDE_RUNNER", "_RANGE_TIGHT", "_TREND_LOOSE",
                   "_AGGR_LOCK", "_RUNNER_NO_BE", "_WIDE_RUNNER_BE07"]
MIN_Q_GRID      = [22, 25, 28, 30, 33, 35, 38, 40]
PB_ATR_GRID     = [0.4, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_GRID    = [3, 5, 7]
VWAP_GRID       = [("0.0_disabled", 0.0), ("0.3", 0.3), ("0.5", 0.5),
                   ("0.7", 0.7), ("1.0", 1.0), ("1.5", 1.5)]
PBW_SECS_GRID   = [1800, 3600, 5400, 7200, 10800]
LS_SECS_GRID    = [3600, 7200, 10800, 14400, 18000]
DIR_BIAS_GRID   = ["LONG", "SHORT", "BOTH"]

# Ship gates
MIN_DELTA   = 30.0
WF_MIN_POS  = 3
WF_MIN_AVG_PF = 1.5
WF_NUM_FOLDS  = 5
WF_FOLD_DAYS  = 36   # 5 × 36 = 180d disjoint

# Phase-A picks / phase-B
PHASE_A_TOP_K = 2     # brief: top-2 combine
PHASE_C_TOP_K = 3     # top-3 finalists into WF

# Phase-B BT cap (hard limit per brief)
PHASE_B_CAP   = 200

# Live anchor (used to seed Phase B and as defaults inside Phase A)
ANCHOR = {
    "sl":         0.2,
    "trail":      "_TIGHT_LOCK",
    "pb_atr":     0.8,
    "pb_wait":    5,
    "vwap_label": "0.5",
    "vwap_buf":   0.5,
    "min_q":      25,
    "pbw_secs":   10800,
    "ls_secs":    18000,
    "dir_bias":   "LONG",
}

# Time budget (cap 2h)
TIME_CAP_SECS = 2 * 3600


# ════════════════════════════════════════════════════════════════════════════
# v5_backtest now reads `VWAP_BUFFER_PER_SYMBOL` from config inside the entry
# loop (each iteration via `from config import ...`), so we just patch the
# per-symbol VWAP buffer value in `config` before each BT.  0.0 = disable.
# No need to clone the BT module by VWAP literal.
# ════════════════════════════════════════════════════════════════════════════
_BT_MOD = None


def _get_bt():
    """Return (and cache per-process) the v5_backtest module."""
    global _BT_MOD
    if _BT_MOD is None:
        import backtest.v5_backtest as _bt
        _BT_MOD = _bt
    return _BT_MOD


def _load_trail_profile(name):
    """Return live-format trail profile [(R, type, param), ...] for `name`."""
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


# ════════════════════════════════════════════════════════════════════════════
# Single-BT worker. Applies all 9 knobs.
# ════════════════════════════════════════════════════════════════════════════
def _bt_one(args):
    """Run one backtest with the supplied 9-knob config + optional fold_id.
    Returns dict with metrics or {'err': str}.
    """
    (sl, trail_name, pb_atr, pb_wait, vwap_label, vwap_buf,
     min_q, pbw_secs, ls_secs, dir_bias, fold_id) = args
    try:
        # ── 1. Mutate config-level constants BEFORE the cloned bt is loaded ──
        # so they are picked up via `from config import ...` inside bt fns.
        import config as cfg
        cfg.PULLBACK_ATR_RETRACE = float(pb_atr)
        cfg.PULLBACK_MAX_WAIT_BARS = int(pb_wait)
        # POST_BIG_WIN secs is re-imported inside backtest_symbol on each call:
        cfg.POST_BIG_WIN_COOLDOWN_ENABLED = True
        cfg.POST_BIG_WIN_COOLDOWN_SECS = int(pbw_secs)
        # PBW thresholds untouched (R=10, $=15, block_both=False per live)
        # VWAP buffer per-symbol — read fresh from config inside BT entry loop
        # (each iteration via `from config import VWAP_BUFFER_PER_SYMBOL`):
        if not hasattr(cfg, "VWAP_BUFFER_PER_SYMBOL"):
            cfg.VWAP_BUFFER_PER_SYMBOL = {}
        cfg.VWAP_BUFFER_PER_SYMBOL = dict(cfg.VWAP_BUFFER_PER_SYMBOL)
        cfg.VWAP_BUFFER_PER_SYMBOL[SYMBOL] = float(vwap_buf)

        # ── 2. Get the bt module ──
        bt = _get_bt()

        # ── 3. Per-symbol overrides on the cloned bt module's globals ──
        bt.SL_OVERRIDE = dict(bt.SL_OVERRIDE)
        bt.SL_OVERRIDE[SYMBOL] = float(sl)
        bt.SL_OVERRIDE_REGIME = dict(bt.SL_OVERRIDE_REGIME)
        bt.SL_OVERRIDE_REGIME[SYMBOL] = {}  # clear per-regime so SL takes effect

        bt.TRAIL_OVERRIDE = dict(bt.TRAIL_OVERRIDE)
        steps_live = _load_trail_profile(trail_name)
        steps_bt = _live_to_bt_trail(steps_live)
        bt.TRAIL_OVERRIDE[SYMBOL] = steps_bt
        bt.TRAIL_OVERRIDE_REGIME = dict(bt.TRAIL_OVERRIDE_REGIME)
        bt.TRAIL_OVERRIDE_REGIME[SYMBOL] = {
            "trending": steps_bt, "ranging": steps_bt,
            "volatile": steps_bt, "low_vol": steps_bt,
        }

        # Direction bias (LONG / SHORT / BOTH).
        # bt.DIR_BIAS is symbol → int; force_direction in `p` overrides it
        # but we mutate DIR_BIAS too for documentary clarity.
        bt.DIR_BIAS = dict(bt.DIR_BIAS)
        if dir_bias == "LONG":
            bt.DIR_BIAS[SYMBOL] = 1
        elif dir_bias == "SHORT":
            bt.DIR_BIAS[SYMBOL] = -1
        else:  # BOTH
            bt.DIR_BIAS.pop(SYMBOL, None)

        # Toxic-hours: UKOUSD has a per-symbol whitelist {8,9,10,11,15,21} via
        # TOXIC_HOURS_PER_SYMBOL that the live agent honors; v5_backtest reads
        # TOXIC_HOURS_UTC only. Mirror live's per-symbol restriction by
        # unioning the *blocking* hours into bt.TOXIC_HOURS for this run.
        # That makes UKOUSD entries only fire during {8,9,10,11,15,21} UTC.
        try:
            from config import TOXIC_HOURS_PER_SYMBOL as _LIVE_TOX_PER
            if SYMBOL in _LIVE_TOX_PER:
                bt.TOXIC_HOURS = set(bt.TOXIC_HOURS) | set(_LIVE_TOX_PER[SYMBOL])
        except Exception:
            pass

        # ── 4. Build params dict with min_quality + loss_streak + risk + dir ──
        p = {
            "min_quality": {
                "trending": int(min_q),
                "ranging":  int(min_q),
                "volatile": int(min_q),
                "low_vol":  int(min_q),
            },
            # LOSS_STREAK bars from secs (BT is H1):
            "loss_streak_enabled": True,
            "loss_streak_count": 2,
            "loss_streak_window_bars": 4,
            "loss_streak_cooldown_bars": max(1, int(round(float(ls_secs) / 3600.0))),
            # Direction bias (BT reads p["force_direction"] before bt.DIR_BIAS):
            "force_direction": dir_bias,
            # Risk + equity (per brief):
            "risk_pct": float(RISK_PCT),
            "start_equity": float(START_EQUITY),
        }

        # ── 5. Optional fold-slicing (disjoint 36d folds for WF) ──
        # IMPORTANT: stash the *true* unpatched load_data in a module attr the
        # first time we touch it; later fold calls in the same worker must use
        # that ORIGINAL load_data, NOT the prior fold's patched closure.
        if fold_id is not None:
            import pandas as pd
            if not hasattr(bt, "_TUNE_ORIG_LOAD_DATA"):
                bt._TUNE_ORIG_LOAD_DATA = bt.load_data
            orig_load = bt._TUNE_ORIG_LOAD_DATA
            fold_n = int(fold_id)
            num = WF_NUM_FOLDS
            fold_d = WF_FOLD_DAYS

            def load_data_fold(sym, _ignored_days=None, _orig=orig_load, _n=fold_n,
                               _num=num, _fd=fold_d):
                df = _orig(sym, days=None)
                if df is None or df.empty:
                    return df
                end = df["time"].max()
                offset_end = (_num - _n) * _fd
                offset_start = offset_end + _fd
                t_end = end - pd.Timedelta(days=offset_end)
                t_start = end - pd.Timedelta(days=offset_start)
                df = df[(df["time"] > t_start) & (df["time"] <= t_end)].reset_index(drop=True)
                return df

            bt.load_data = load_data_fold
            try:
                r = bt.backtest_symbol(SYMBOL, days=None, params=p, verbose=False)
            finally:
                # Restore so subsequent calls in this worker see the original loader.
                bt.load_data = orig_load
        else:
            # Make sure load_data is the original unpatched function before a
            # full-window call in a worker that previously ran a WF cell.
            if hasattr(bt, "_TUNE_ORIG_LOAD_DATA"):
                bt.load_data = bt._TUNE_ORIG_LOAD_DATA
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
        return {"err": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}"}


def _bt_baseline(_args=None):
    """Live baseline: anchor knobs (= current live config), with the same risk
    and equity overrides as every other run so results are comparable.
    """
    args = (ANCHOR["sl"], ANCHOR["trail"], ANCHOR["pb_atr"], ANCHOR["pb_wait"],
            ANCHOR["vwap_label"], ANCHOR["vwap_buf"], ANCHOR["min_q"],
            ANCHOR["pbw_secs"], ANCHOR["ls_secs"], ANCHOR["dir_bias"], None)
    return _bt_one(args)


def _log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _fmt_cfg(c):
    return (f"SL={c['sl']} {c['trail']:18} PB=({c['pb_atr']},{c['pb_wait']}) "
            f"VWAP={c['vwap_label']:13} mQ={c['min_q']} "
            f"PBW={c['pbw_secs']:5d} LS={c['ls_secs']:5d} DIR={c['dir_bias']}")


def _row_from_seed_and_res(seed, res):
    """Merge a seed config dict + a _bt_one result into a single row."""
    out = dict(seed)
    out.update({
        "trades": res["trades"], "pf": res["pf"], "wr": res["wr"],
        "pnl": res["pnl"], "dd": res["dd"],
    })
    return out


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")

    t0 = time.time()
    workers = max(2, (os.cpu_count() or 4) - 2)
    _log(f"UKOUSD hard-tune — workers={workers}  cap={TIME_CAP_SECS}s")
    _log(f"Anchor: { {k: v for k, v in ANCHOR.items()} }")
    _log(f"Risk={RISK_PCT}%  start_equity=${START_EQUITY}")

    # ── Phase 0: baseline (live anchor under same risk/equity) ──
    _log("\nPhase 0: anchor baseline")
    with Pool(1) as pool:
        baseline = pool.apply(_bt_baseline)
    if "err" in baseline:
        _log(f"  baseline FAILED: {baseline['err']}")
        sys.exit(1)
    _log(f"  baseline: trades={baseline['trades']} PF={baseline['pf']:.2f} "
         f"PnL=${baseline['pnl']:+.0f} WR={baseline['wr']:.1f}% DD={baseline['dd']:.2f}%")

    # ── Phase A: full SL × Trail grid ──
    _log(f"\nPhase A: SL × Trail grid ({len(SL_GRID)} × {len(TRAIL_NAMES)} = "
         f"{len(SL_GRID)*len(TRAIL_NAMES)} BTs)")
    a_jobs = []
    a_seeds = []
    for sl in SL_GRID:
        for tname in TRAIL_NAMES:
            seed = {
                "sl": sl, "trail": tname,
                "pb_atr": ANCHOR["pb_atr"], "pb_wait": ANCHOR["pb_wait"],
                "vwap_label": ANCHOR["vwap_label"], "vwap_buf": ANCHOR["vwap_buf"],
                "min_q": ANCHOR["min_q"],
                "pbw_secs": ANCHOR["pbw_secs"], "ls_secs": ANCHOR["ls_secs"],
                "dir_bias": ANCHOR["dir_bias"],
            }
            a_seeds.append(seed)
            a_jobs.append((sl, tname, seed["pb_atr"], seed["pb_wait"],
                           seed["vwap_label"], seed["vwap_buf"], seed["min_q"],
                           seed["pbw_secs"], seed["ls_secs"], seed["dir_bias"], None))
    t_a = time.time()
    with Pool(workers) as pool:
        a_raw = list(pool.imap(_bt_one, a_jobs))
    a_results = []
    for seed, res in zip(a_seeds, a_raw):
        if "err" in res:
            _log(f"  SL={seed['sl']} {seed['trail']}: ERR {res['err'][:120]}")
            continue
        row = _row_from_seed_and_res(seed, res)
        row["delta"] = row["pnl"] - baseline["pnl"]
        a_results.append(row)
    _log(f"  Phase A done in {time.time()-t_a:.1f}s  "
         f"({len(a_results)}/{len(a_jobs)} OK)")

    # Keep rows with at least 10 trades (UKOUSD whitelist limits volume).
    a_filtered = [r for r in a_results if r["trades"] >= 10]
    a_sorted = sorted(a_filtered, key=lambda x: x["pnl"], reverse=True)
    top_a = a_sorted[:PHASE_A_TOP_K]
    _log(f"  Top-{len(top_a)} Phase A (trades≥10):")
    for r in top_a:
        _log(f"    {_fmt_cfg(r)}  n={r['trades']:3d} PF={r['pf']:5.2f} "
             f"PnL=${r['pnl']:+8.0f}  Δ=${r['delta']:+.0f}")

    if not top_a:
        _log("  No viable Phase-A seed — aborting.")
        sys.exit(1)

    # ── Phase B: coordinate descent on each top-2 seed ──
    _log(f"\nPhase B: coordinate descent on top-{len(top_a)} Phase-A seeds")
    b_chains = []
    b_bt_count = 0

    for seed_idx, seed in enumerate(top_a):
        if time.time() - t0 > TIME_CAP_SECS * 0.8:
            _log(f"  TIME CAP — skipping remaining seeds")
            break
        _log(f"\n  ─── Seed {seed_idx+1}/{len(top_a)}: {_fmt_cfg(seed)} ───")
        cur = dict(seed)

        # B1: pullback ATR × wait
        b1_jobs, b1_seeds = [], []
        for pa in PB_ATR_GRID:
            for pw in PB_WAIT_GRID:
                s = dict(cur); s["pb_atr"] = pa; s["pb_wait"] = pw
                b1_seeds.append(s)
                b1_jobs.append((cur["sl"], cur["trail"], pa, pw,
                                cur["vwap_label"], cur["vwap_buf"], cur["min_q"],
                                cur["pbw_secs"], cur["ls_secs"], cur["dir_bias"], None))
        t_b = time.time()
        with Pool(workers) as pool:
            b1_raw = list(pool.imap(_bt_one, b1_jobs))
        b_bt_count += len(b1_jobs)
        b1_res = []
        for s, res in zip(b1_seeds, b1_raw):
            if "err" in res or res["trades"] < 10:
                continue
            row = _row_from_seed_and_res(s, res)
            row["delta"] = row["pnl"] - baseline["pnl"]
            b1_res.append(row)
        b1_res.sort(key=lambda x: x["pnl"], reverse=True)
        if b1_res:
            best = b1_res[0]
            cur.update({k: best[k] for k in ("pb_atr", "pb_wait", "trades", "pf",
                                              "wr", "pnl", "dd")})
            cur["delta"] = cur["pnl"] - baseline["pnl"]
            _log(f"    B1 best: PB=({cur['pb_atr']},{cur['pb_wait']}) "
                 f"PnL=${cur['pnl']:+.0f}  Δ=${cur['delta']:+.0f}  "
                 f"({time.time()-t_b:.1f}s, {len(b1_res)} OK)")

        # B2: VWAP × min_q
        b2_jobs, b2_seeds = [], []
        for (vlab, vbuf) in VWAP_GRID:
            for mq in MIN_Q_GRID:
                s = dict(cur); s["vwap_label"] = vlab; s["vwap_buf"] = vbuf; s["min_q"] = mq
                b2_seeds.append(s)
                b2_jobs.append((cur["sl"], cur["trail"], cur["pb_atr"], cur["pb_wait"],
                                vlab, vbuf, mq, cur["pbw_secs"], cur["ls_secs"],
                                cur["dir_bias"], None))
        t_b = time.time()
        with Pool(workers) as pool:
            b2_raw = list(pool.imap(_bt_one, b2_jobs))
        b_bt_count += len(b2_jobs)
        b2_res = []
        for s, res in zip(b2_seeds, b2_raw):
            if "err" in res or res["trades"] < 10:
                continue
            row = _row_from_seed_and_res(s, res)
            row["delta"] = row["pnl"] - baseline["pnl"]
            b2_res.append(row)
        b2_res.sort(key=lambda x: x["pnl"], reverse=True)
        if b2_res:
            best = b2_res[0]
            cur.update({k: best[k] for k in ("vwap_label", "vwap_buf", "min_q",
                                              "trades", "pf", "wr", "pnl", "dd")})
            cur["delta"] = cur["pnl"] - baseline["pnl"]
            _log(f"    B2 best: VWAP={cur['vwap_label']:13} mQ={cur['min_q']} "
                 f"PnL=${cur['pnl']:+.0f}  Δ=${cur['delta']:+.0f}  "
                 f"({time.time()-t_b:.1f}s, {len(b2_res)} OK)")

        # B3: POST_BIG_WIN × LOSS_STREAK cooldowns
        b3_jobs, b3_seeds = [], []
        for pbw in PBW_SECS_GRID:
            for ls in LS_SECS_GRID:
                s = dict(cur); s["pbw_secs"] = pbw; s["ls_secs"] = ls
                b3_seeds.append(s)
                b3_jobs.append((cur["sl"], cur["trail"], cur["pb_atr"], cur["pb_wait"],
                                cur["vwap_label"], cur["vwap_buf"], cur["min_q"],
                                pbw, ls, cur["dir_bias"], None))
        t_b = time.time()
        with Pool(workers) as pool:
            b3_raw = list(pool.imap(_bt_one, b3_jobs))
        b_bt_count += len(b3_jobs)
        b3_res = []
        for s, res in zip(b3_seeds, b3_raw):
            if "err" in res or res["trades"] < 10:
                continue
            row = _row_from_seed_and_res(s, res)
            row["delta"] = row["pnl"] - baseline["pnl"]
            b3_res.append(row)
        b3_res.sort(key=lambda x: x["pnl"], reverse=True)
        if b3_res:
            best = b3_res[0]
            cur.update({k: best[k] for k in ("pbw_secs", "ls_secs", "trades", "pf",
                                              "wr", "pnl", "dd")})
            cur["delta"] = cur["pnl"] - baseline["pnl"]
            _log(f"    B3 best: PBW={cur['pbw_secs']} LS={cur['ls_secs']} "
                 f"PnL=${cur['pnl']:+.0f}  Δ=${cur['delta']:+.0f}  "
                 f"({time.time()-t_b:.1f}s, {len(b3_res)} OK)")

        # B4: Direction bias
        b4_jobs, b4_seeds = [], []
        for db in DIR_BIAS_GRID:
            s = dict(cur); s["dir_bias"] = db
            b4_seeds.append(s)
            b4_jobs.append((cur["sl"], cur["trail"], cur["pb_atr"], cur["pb_wait"],
                            cur["vwap_label"], cur["vwap_buf"], cur["min_q"],
                            cur["pbw_secs"], cur["ls_secs"], db, None))
        t_b = time.time()
        with Pool(workers) as pool:
            b4_raw = list(pool.imap(_bt_one, b4_jobs))
        b_bt_count += len(b4_jobs)
        b4_res = []
        for s, res in zip(b4_seeds, b4_raw):
            if "err" in res or res["trades"] < 10:
                continue
            row = _row_from_seed_and_res(s, res)
            row["delta"] = row["pnl"] - baseline["pnl"]
            b4_res.append(row)
        b4_res.sort(key=lambda x: x["pnl"], reverse=True)
        if b4_res:
            best = b4_res[0]
            cur.update({k: best[k] for k in ("dir_bias", "trades", "pf", "wr",
                                              "pnl", "dd")})
            cur["delta"] = cur["pnl"] - baseline["pnl"]
            _log(f"    B4 best: DIR={cur['dir_bias']} "
                 f"PnL=${cur['pnl']:+.0f}  Δ=${cur['delta']:+.0f}  "
                 f"({time.time()-t_b:.1f}s, {len(b4_res)} OK)")

        b_chains.append({
            "seed": seed,
            "tuned": cur,
            "b1_top5": b1_res[:5],
            "b2_top5": b2_res[:5],
            "b3_top5": b3_res[:5],
            "b4_all":  b4_res,
        })
    _log(f"\n  Phase B used {b_bt_count} BTs (cap={PHASE_B_CAP}).")

    # ── Top-3 finalists by tuned PnL ──
    chains_sorted = sorted(b_chains, key=lambda x: x["tuned"]["pnl"], reverse=True)
    finalists = chains_sorted[:PHASE_C_TOP_K]
    _log(f"\nTop-{len(finalists)} tuned finalists:")
    for i, c in enumerate(finalists):
        t = c["tuned"]
        _log(f"  #{i+1} {_fmt_cfg(t)}  PnL=${t['pnl']:+.0f}  Δ=${t['delta']:+.0f}")

    # ── Phase C: walk-forward validation (5 disjoint folds × top-3) ──
    _log(f"\nPhase C: walk-forward {WF_NUM_FOLDS} disjoint {WF_FOLD_DAYS}d folds "
         f"on top-{len(finalists)} finalists")
    wf_jobs = []
    for c in finalists:
        t = c["tuned"]
        for fold in range(1, WF_NUM_FOLDS + 1):
            wf_jobs.append((t["sl"], t["trail"], t["pb_atr"], t["pb_wait"],
                            t["vwap_label"], t["vwap_buf"], t["min_q"],
                            t["pbw_secs"], t["ls_secs"], t["dir_bias"], fold))
    t_c = time.time()
    with Pool(workers) as pool:
        wf_raw = list(pool.imap(_bt_one, wf_jobs))
    _log(f"  WF done in {time.time()-t_c:.1f}s")

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
                "pf":  round(res["pf"], 2),
                "pnl": round(res["pnl"], 2),
                "wr":  round(res["wr"], 1),
            })

    # ── Final assembly ──
    final = {
        "_meta": {
            "ts": datetime.now().isoformat(),
            "symbol": SYMBOL,
            "days": DAYS,
            "start_equity": START_EQUITY,
            "risk_pct": RISK_PCT,
            "phase_a_grid_size": len(SL_GRID) * len(TRAIL_NAMES),
            "phase_a_top_k": PHASE_A_TOP_K,
            "phase_b_bts": b_bt_count,
            "phase_b_cap": PHASE_B_CAP,
            "phase_c_top_k": PHASE_C_TOP_K,
            "wf_num_folds": WF_NUM_FOLDS,
            "wf_fold_days": WF_FOLD_DAYS,
            "ship_rule": {
                "min_delta_usd": MIN_DELTA,
                "wf_min_pos_folds": WF_MIN_POS,
                "wf_min_avg_pf": WF_MIN_AVG_PF,
            },
            "dimensions": {
                "sl_grid": SL_GRID,
                "trail_names": TRAIL_NAMES,
                "min_q_grid": MIN_Q_GRID,
                "pb_atr_grid": PB_ATR_GRID,
                "pb_wait_grid": PB_WAIT_GRID,
                "vwap_grid": [v[0] for v in VWAP_GRID],
                "pbw_secs_grid": PBW_SECS_GRID,
                "ls_secs_grid": LS_SECS_GRID,
                "dir_bias_grid": DIR_BIAS_GRID,
            },
            "anchor": ANCHOR,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "baseline": baseline,
        "phase_a_all": a_results,
        "phase_a_top": top_a,
        "phase_b_chains": b_chains,
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
        wf_pos_ok = pos >= WF_MIN_POS
        wf_pf_ok  = avg_pf >= WF_MIN_AVG_PF
        delta_ok  = (t["pnl"] - baseline["pnl"]) >= MIN_DELTA
        ship = bool(wf_pos_ok and wf_pf_ok and delta_ok)
        rec = {
            "rank": f_idx + 1,
            "config": {
                "sl": t["sl"],
                "trail": t["trail"],
                "pullback_atr_retrace":   t["pb_atr"],
                "pullback_max_wait_bars": t["pb_wait"],
                "vwap_buffer_atr":   t["vwap_buf"],
                "vwap_buffer_label": t["vwap_label"],
                "min_quality_all_regimes": t["min_q"],
                "post_big_win_cooldown_secs": t["pbw_secs"],
                "loss_streak_cooldown_secs": t["ls_secs"],
                "direction_bias": t["dir_bias"],
            },
            "in_sample": {
                "trades": t["trades"],
                "pf": round(t["pf"], 2),
                "wr": round(t.get("wr", 0), 1) if "wr" in t else None,
                "pnl": round(t["pnl"], 2),
                "dd": round(t["dd"], 2),
            },
            "delta_pnl": round(t["pnl"] - baseline["pnl"], 2),
            "wf_folds":   folds,
            "wf_pos_folds": pos,
            "wf_avg_pf":  avg_pf,
            "wf_total_pnl": total_wf_pnl,
            "wf_pos_passed": wf_pos_ok,
            "wf_pf_passed":  wf_pf_ok,
            "delta_passed":  delta_ok,
            "recommend_ship": ship,
        }
        final["finalists"].append(rec)
        if ship and t["pnl"] > best_ship_pnl:
            best_ship_pnl = t["pnl"]
            best_ship = rec
        _log(f"  Finalist #{f_idx+1}: Δ=${t['pnl']-baseline['pnl']:+.0f}  "
             f"WF {pos}/{WF_NUM_FOLDS} avg_pf={avg_pf:.2f} ship={ship}")

    final["winner"] = best_ship
    final["ship_recommend"] = best_ship is not None

    # Persist JSON
    OUT_JSON.write_text(json.dumps(final, indent=2, default=str))
    _log(f"\nJSON written: {OUT_JSON}")

    # ── Markdown summary ──
    md = []
    md.append(f"# UKOUSD Per-Symbol Hard-Tune — {datetime.now().strftime('%Y-%m-%d')}\n")
    md.append(f"**Symbol:** {SYMBOL} | **Days:** {DAYS} | **Equity:** ${START_EQUITY} | "
              f"**Risk:** {RISK_PCT}% | **Workers:** {workers} | "
              f"**Elapsed:** {round(time.time()-t0,1)}s\n")

    md.append("\n## Anchor (live config)\n")
    md.append(f"- SL={ANCHOR['sl']}  trail={ANCHOR['trail']}  PB=({ANCHOR['pb_atr']},{ANCHOR['pb_wait']})  "
              f"VWAP={ANCHOR['vwap_label']}  mQ={ANCHOR['min_q']}  "
              f"PBW={ANCHOR['pbw_secs']}s  LS={ANCHOR['ls_secs']}s  DIR={ANCHOR['dir_bias']}\n")

    md.append("\n## Baseline (anchor under risk=2.0%, equity=$1219)\n")
    md.append(f"- trades: **{baseline['trades']}**  PF: **{baseline['pf']:.2f}**  "
              f"WR: **{baseline['wr']:.1f}%**  PnL: **${baseline['pnl']:+,.0f}**  "
              f"DD: **{baseline['dd']:.2f}%**\n")

    md.append("\n## Phase A — Top SL × Trail (anchor knobs fixed)\n")
    md.append("| Rank | SL | Trail | Trades | PF | WR | PnL | Δ |\n"
              "|---:|---:|:--|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(top_a, 1):
        md.append(f"| {i} | {r['sl']} | {r['trail']} | {r['trades']} | "
                  f"{r['pf']:.2f} | {r['wr']:.1f}% | "
                  f"${r['pnl']:+,.0f} | ${r['delta']:+,.0f} |\n")

    md.append("\n## Phase B/C — Tuned Finalists\n")
    for rec in final["finalists"]:
        cfg_ = rec["config"]
        md.append(f"\n### Finalist #{rec['rank']}\n")
        md.append(f"- SL: **{cfg_['sl']}**\n")
        md.append(f"- Trail: **{cfg_['trail']}**\n")
        md.append(f"- Pullback: ATR={cfg_['pullback_atr_retrace']}, "
                  f"wait={cfg_['pullback_max_wait_bars']} bars\n")
        md.append(f"- VWAP buffer: {cfg_['vwap_buffer_label']} "
                  f"({cfg_['vwap_buffer_atr']})\n")
        md.append(f"- min_quality (all regimes): **{cfg_['min_quality_all_regimes']}**\n")
        md.append(f"- POST_BIG_WIN cooldown: **{cfg_['post_big_win_cooldown_secs']}s**\n")
        md.append(f"- LOSS_STREAK cooldown: **{cfg_['loss_streak_cooldown_secs']}s**\n")
        md.append(f"- Direction bias: **{cfg_['direction_bias']}**\n")
        ins = rec["in_sample"]
        md.append(f"- **In-sample {DAYS}d**: trades={ins['trades']} PF={ins['pf']} "
                  f"WR={ins['wr']}% PnL=${ins['pnl']:+,.0f} DD={ins['dd']}% "
                  f"Δ=${rec['delta_pnl']:+,.0f}\n")
        md.append(f"- **WF**: {rec['wf_pos_folds']}/{WF_NUM_FOLDS} positive, "
                  f"avg_pf={rec['wf_avg_pf']:.2f}, "
                  f"total=${rec['wf_total_pnl']:+,.0f}\n")
        md.append("\n  | Fold | Trades | PF | WR | PnL |\n  |---:|---:|---:|---:|---:|\n")
        for f in rec["wf_folds"]:
            md.append(f"  | {f.get('fold','?')} | {f.get('trades','?')} | "
                      f"{f.get('pf','?')} | {f.get('wr','?')}% | "
                      f"${f.get('pnl',0):+,.0f} |\n")
        md.append(f"- **Ship**: {'YES' if rec['recommend_ship'] else 'NO'} "
                  f"(Δ≥${MIN_DELTA}: {rec['delta_passed']}, "
                  f"WF pos≥{WF_MIN_POS}/{WF_NUM_FOLDS}: {rec['wf_pos_passed']}, "
                  f"avg PF≥{WF_MIN_AVG_PF}: {rec['wf_pf_passed']})\n")

    md.append("\n## Verdict\n")
    if final["ship_recommend"]:
        w = final["winner"]
        cfg_ = w["config"]
        md.append(f"- **SHIP**: Finalist #{w['rank']}\n")
        md.append(f"- Cfg: SL={cfg_['sl']}, trail={cfg_['trail']}, "
                  f"PB=({cfg_['pullback_atr_retrace']},{cfg_['pullback_max_wait_bars']}), "
                  f"VWAP={cfg_['vwap_buffer_label']}, "
                  f"mQ={cfg_['min_quality_all_regimes']}, "
                  f"PBW={cfg_['post_big_win_cooldown_secs']}s, "
                  f"LS={cfg_['loss_streak_cooldown_secs']}s, "
                  f"DIR={cfg_['direction_bias']}\n")
        md.append(f"- Δ=${w['delta_pnl']:+,.0f}  "
                  f"WF {w['wf_pos_folds']}/{WF_NUM_FOLDS} pos  "
                  f"avg_pf={w['wf_avg_pf']}\n")
    else:
        md.append(f"- **NO-SHIP**: no finalist passed all three ship gates "
                  f"(Δ≥${MIN_DELTA}, WF≥{WF_MIN_POS}/{WF_NUM_FOLDS} pos, "
                  f"avg PF≥{WF_MIN_AVG_PF}).\n")

    md.append(f"\n_Tune ran {final['_meta']['elapsed_sec']:.0f}s "
              f"(Phase-A {final['_meta']['phase_a_grid_size']} BTs, "
              f"Phase-B {b_bt_count} BTs)._\n")

    OUT_MD.write_text("".join(md))
    _log(f"MD written: {OUT_MD}")
    _log(f"\nALL DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
