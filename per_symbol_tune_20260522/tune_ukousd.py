#!/usr/bin/env python3 -B
"""
UKOUSD per-symbol fine-tune  —  2026-05-22.

7-dimension sweep:
  1. SL          : sl_atr_mult ∈ {0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0,3.5}   (14)
  2. Trail       : 7 named profiles                                                          (7)
  3. PB_ATR      : PULLBACK_ATR_RETRACE  ∈ {0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2}                 (8)
  4. PB_WAIT     : PULLBACK_MAX_WAIT_BARS ∈ {3,4,5,6,8}                                      (5)
  5. VWAP_BUF    : {'0.0_disabled',0.3,0.5,0.7,1.0,1.5}  (source-patched into BT module)     (6)
  6. min_quality : flat-X ∈ {28,30,33,35,38,40,43}                                           (7)
  7. TOXIC       : {full, partial, narrow, single, off} 5 subsets                            (5)

Phase A: ~52 BTs independent
Phase B: 2^7 = 128 combos (top-2 per dim)
Phase C: top-5 walk-forward 5-fold

Ship rule: Δ ≥ +$100/180d AND WF ≥3/5 positive.

READ-ONLY: no source files modified. VWAP buffer applied as in-memory text patch
to a worker-local copy of v5_backtest.py.
"""
import json, time, os, sys, importlib, copy
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Optional

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

SYMBOL    = "UKOUSD"
OUT_DIR   = ROOT / "per_symbol_tune_20260522"
OUT_JSON  = OUT_DIR / "UKOUSD.json"
OUT_MD    = OUT_DIR / "UKOUSD.md"
TUNE_DAYS = 180

# ────────────────────────────────────────────────────────────────────────────
# Trail profiles (verbatim from auto_tuned.py)
# ────────────────────────────────────────────────────────────────────────────
TRAIL_PROFILES = {
    "_TIGHT_LOCK":       [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_WIDE_RUNNER":      [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)],
    "_RANGE_TIGHT":      [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0)],
    "_TREND_LOOSE":      [(15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5), (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_AGGR_LOCK":        [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8), (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "_RUNNER_NO_BE":     [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5), (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)],
    "_WIDE_RUNNER_BE07": [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)],
}

# Sweep dimensions ──────────────────────────────────────────────────────────
SL_GRID         = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5]
TRAIL_NAMES     = list(TRAIL_PROFILES.keys())
PB_ATR_GRID     = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
PB_WAIT_GRID    = [3, 4, 5, 6, 8]
VWAP_BUF_GRID   = ["0.0_disabled", 0.3, 0.5, 0.7, 1.0, 1.5]
MIN_QUALITY_GRID = [28, 30, 33, 35, 38, 40, 43]

# UKOUSD current toxic whitelist {8,9,10,11,15,21} (kept ALL hours not in this set blocked)
# Subsets to sweep (the elements ALLOWED — everything else blocked):
TOXIC_VARIANTS = {
    "current_{8,9,10,11,15,21}":   {8, 9, 10, 11, 15, 21},
    "tight_{15,21}":                {15, 21},
    "midtight_{10,11,15,21}":       {10, 11, 15, 21},
    "narrow_{15}":                  {15},
    "narrow_{21}":                  {21},
    "none_(no_whitelist_restriction)": None,  # special: only base TOXIC_HOURS_UTC {1,2,3,4}
}

# Walk-forward folds (trailing) — same shape as tune_session_20260521
WF_FOLDS_DAYS   = [60, 90, 120, 150, 180]

WORKERS         = max(2, min(8, (cpu_count() or 4) - 2))

# Ship gates
MIN_DELTA_PNL   = 100.0
MIN_WF_POS      = 3   # of 5

# ────────────────────────────────────────────────────────────────────────────
# Worker: in-memory patched v5_backtest loader.
#
# Each backtest is parameterised by 7 knobs. Per-worker we:
#   1. Patch config.py module globals (PULLBACK_ATR_RETRACE/PULLBACK_MAX_WAIT_BARS,
#      TOXIC_HOURS_PER_SYMBOL[UKOUSD], SIGNAL_QUALITY_SYMBOL[UKOUSD],
#      SYMBOL_ATR_SL_OVERRIDE[UKOUSD], SYMBOL_TRAIL_OVERRIDE[UKOUSD],
#      SYMBOL_REGIME_TRAIL_OVERRIDE[UKOUSD]).
#   2. Reload backtest.v5_backtest fresh.
#   3. Source-patch the VWAP buffer literal (`* 0.5`) in the reloaded module's
#      backtest_symbol bytecode by re-executing a textually patched copy of the
#      source under a fresh module name. (Avoids touching the on-disk file.)
#   4. Run backtest_symbol(SYMBOL, days, verbose=False).
#
# All sweep loops use this single helper for parity.
# ────────────────────────────────────────────────────────────────────────────
V5_SRC_TEXT: Optional[str] = None  # lazy-loaded


def _read_v5_src() -> str:
    """Read v5_backtest.py source once per worker (cached)."""
    global V5_SRC_TEXT
    if V5_SRC_TEXT is None:
        V5_SRC_TEXT = (ROOT / "backtest" / "v5_backtest.py").read_text()
    return V5_SRC_TEXT


def _patch_vwap_src(src: str, vwap_buf) -> str:
    """Patch VWAP block in v5_backtest source.

    vwap_buf == "0.0_disabled" → replace block with pass (gate disabled).
    Numeric                    → replace literal `* 0.5` with `* <num>`.
    """
    needle = 'atr_buf = float(ind["at"][bi]) * 0.5'
    assert needle in src, "VWAP literal not found — v5_backtest changed?"
    if vwap_buf == "0.0_disabled":
        # Replace the buf line with one that forces buf so large the gate
        # never fires (effectively disabled — VWAP - 1e9 < anything).
        return src.replace(needle, 'atr_buf = float(ind["at"][bi]) * 1e9')
    return src.replace(needle, f'atr_buf = float(ind["at"][bi]) * {float(vwap_buf)}')


def _load_patched_bt(vwap_buf):
    """Build a fresh, isolated v5_backtest module with patched VWAP buffer."""
    src = _patch_vwap_src(_read_v5_src(), vwap_buf)
    import types
    mod_name = f"v5_bt_patched_{os.getpid()}_{hash(str(vwap_buf)) & 0xFFFF}"
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(ROOT / "backtest" / "v5_backtest.py")
    mod.__package__ = "backtest"
    sys.modules[mod_name] = mod
    code = compile(src, mod.__file__, "exec")
    exec(code, mod.__dict__)
    return mod


def _apply_config_overrides(sl, trail, pb_atr, pb_wait, min_quality, toxic_set):
    """Mutate live `config` module so reloaded v5_backtest picks up overrides."""
    import config as cfg
    importlib.reload(cfg)

    # also reload auto_tuned so its dicts are pristine and don't bleed
    # state across sweeps
    try:
        import auto_tuned as _at
        importlib.reload(_at)
    except Exception:
        pass

    # Re-trigger config's bottom-of-file auto_tuned merge by re-executing it.
    # (Reloading config above already runs the try-import block once; if it
    # raises ImportError or has side-effects we missed, this is a no-op since
    # importlib.reload already invoked __main__ logic.)

    # SL
    cfg.SYMBOL_ATR_SL_OVERRIDE = dict(cfg.SYMBOL_ATR_SL_OVERRIDE)
    cfg.SYMBOL_ATR_SL_OVERRIDE[SYMBOL] = sl
    cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME = dict(cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME)
    cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME[SYMBOL] = {}  # clear per-regime

    # Trail (all 4 regimes use the swept profile)
    cfg.SYMBOL_TRAIL_OVERRIDE = dict(cfg.SYMBOL_TRAIL_OVERRIDE)
    cfg.SYMBOL_TRAIL_OVERRIDE[SYMBOL] = trail
    cfg.SYMBOL_REGIME_TRAIL_OVERRIDE = dict(cfg.SYMBOL_REGIME_TRAIL_OVERRIDE)
    cfg.SYMBOL_REGIME_TRAIL_OVERRIDE[SYMBOL] = {
        "trending": trail, "ranging": trail, "volatile": trail, "low_vol": trail,
    }

    # Pullback
    cfg.PULLBACK_ATR_RETRACE = float(pb_atr)
    cfg.PULLBACK_MAX_WAIT_BARS = int(pb_wait)

    # min_quality flat across regimes
    cfg.SIGNAL_QUALITY_SYMBOL = dict(cfg.SIGNAL_QUALITY_SYMBOL)
    cfg.SIGNAL_QUALITY_SYMBOL[SYMBOL] = {
        "trending": int(min_quality), "ranging": int(min_quality),
        "volatile": int(min_quality), "low_vol": int(min_quality),
    }

    # Toxic-hours whitelist for UKOUSD: blocking-set = all hours NOT in whitelist.
    # `None` toxic_set = no whitelist tightening, only the global {1,2,3,4} applies.
    cfg.TOXIC_HOURS_PER_SYMBOL = dict(cfg.TOXIC_HOURS_PER_SYMBOL)
    if toxic_set is None:
        cfg.TOXIC_HOURS_PER_SYMBOL.pop(SYMBOL, None)
    else:
        cfg.TOXIC_HOURS_PER_SYMBOL[SYMBOL] = set(range(24)) - set(toxic_set)

    # NOTE: v5_backtest reads TOXIC_HOURS from TOXIC_HOURS_UTC only (not
    # TOXIC_HOURS_PER_SYMBOL). We need to UNION the per-symbol entries into
    # TOXIC_HOURS_UTC for this symbol. v5_backtest:248 uses set(TOXIC_HOURS_UTC).
    # We simulate live's `TOXIC_HOURS_PER_SYMBOL` by injecting it into UTC.
    # SAFETY: only adds for the run; we reload cfg next call.
    if SYMBOL in cfg.TOXIC_HOURS_PER_SYMBOL:
        cfg.TOXIC_HOURS_UTC = (set(cfg.TOXIC_HOURS_UTC) |
                                set(cfg.TOXIC_HOURS_PER_SYMBOL[SYMBOL]))


# ────────────────────────────────────────────────────────────────────────────
# Single-BT runner
# ────────────────────────────────────────────────────────────────────────────
def _run_one(args):
    """Execute a single backtest with the given knob settings.

    Returns dict with keys (sweep config + result metrics).
    On error: {"error": str, "ok": False, **args}.
    """
    (label, days, sl, trail_name, pb_atr, pb_wait, vwap_buf, min_q, toxic_label) = args
    try:
        trail = TRAIL_PROFILES[trail_name]
        toxic_set = TOXIC_VARIANTS[toxic_label]
        _apply_config_overrides(sl, trail, pb_atr, pb_wait, min_q, toxic_set)
        bt = _load_patched_bt(vwap_buf)
        r = bt.backtest_symbol(SYMBOL, days=days, verbose=False)
    except Exception as e:
        return {"label": label, "ok": False, "error": f"{type(e).__name__}: {e}",
                "sl": sl, "trail": trail_name, "pb_atr": pb_atr, "pb_wait": pb_wait,
                "vwap_buf": vwap_buf, "min_q": min_q, "toxic": toxic_label, "days": days}
    if not r:
        return {"label": label, "ok": False, "error": "no_data",
                "sl": sl, "trail": trail_name, "pb_atr": pb_atr, "pb_wait": pb_wait,
                "vwap_buf": vwap_buf, "min_q": min_q, "toxic": toxic_label, "days": days}
    return {
        "label": label, "ok": True, "days": days,
        "sl": sl, "trail": trail_name, "pb_atr": pb_atr, "pb_wait": pb_wait,
        "vwap_buf": vwap_buf, "min_q": min_q, "toxic": toxic_label,
        "trades": int(r.get("trades", 0)),
        "pf":     float(r.get("pf", 0)),
        "pnl":    float(r.get("pnl", 0)),
        "wr":     float(r.get("wr", 0)),
        "dd":     float(r.get("dd", 0)),
        "avg_r":  float(r.get("avg_r", 0)),
        "avg_peak_r":   float(r.get("avg_peak_r", 0)),
        "avg_giveback": float(r.get("avg_giveback", 0)),
    }


# ────────────────────────────────────────────────────────────────────────────
# Baseline (current live config — touch no knobs)
# ────────────────────────────────────────────────────────────────────────────
BASELINE_CFG = {
    "sl":            0.5,
    "trail":         "_RUNNER_NO_BE",
    "pb_atr":        0.8,
    "pb_wait":       5,
    "vwap_buf":      0.5,
    "min_q":         48,    # config.py line 1074: UKOUSD 48/48/48/48
    "toxic":         "current_{8,9,10,11,15,21}",
}


def _baseline_args(label="baseline", days=TUNE_DAYS):
    b = BASELINE_CFG
    return (label, days, b["sl"], b["trail"], b["pb_atr"], b["pb_wait"],
            b["vwap_buf"], b["min_q"], b["toxic"])


def _make_args(label, days, overrides):
    """Build a _run_one args tuple from baseline + overrides dict."""
    cfg = {**BASELINE_CFG, **overrides}
    return (label, days, cfg["sl"], cfg["trail"], cfg["pb_atr"], cfg["pb_wait"],
            cfg["vwap_buf"], cfg["min_q"], cfg["toxic"])


# ────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ────────────────────────────────────────────────────────────────────────────
def _fmt_row(r):
    if not r.get("ok"):
        return f"  {r['label']:<32} ERROR: {r.get('error','?')}"
    return (f"  {r['label']:<32} n={r['trades']:3d}  PF={r['pf']:5.2f}  "
            f"PnL=${r['pnl']:+9.2f}  WR={r['wr']:5.1f}%  DD={r['dd']:4.1f}%")


# ════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print(f"\n{'='*78}")
    print(f" UKOUSD per-symbol fine-tune   {datetime.now().isoformat(timespec='seconds')}")
    print(f" Workers: {WORKERS}    Days: {TUNE_DAYS}    Output: {OUT_JSON}")
    print(f"{'='*78}\n")

    # ────────────────────────────────────────────────────────────────────
    # 0) Baseline
    # ────────────────────────────────────────────────────────────────────
    print("[0] Baseline — current live config:")
    baseline = _run_one(_baseline_args())
    print(_fmt_row(baseline))
    if not baseline.get("ok"):
        raise RuntimeError("Baseline failed — aborting.")
    BASE_PNL = baseline["pnl"]
    print()

    # ────────────────────────────────────────────────────────────────────
    # PHASE A — independent dimension sweeps, top-2 per dim
    # ────────────────────────────────────────────────────────────────────
    print("[A] Phase A — independent sweeps")
    phase_a_jobs = []
    # (1) SL
    for sl in SL_GRID:
        if sl == BASELINE_CFG["sl"]:
            continue
        phase_a_jobs.append(_make_args(f"sl={sl}", TUNE_DAYS, {"sl": sl}))
    # (2) Trail
    for tn in TRAIL_NAMES:
        if tn == BASELINE_CFG["trail"]:
            continue
        phase_a_jobs.append(_make_args(f"trail={tn}", TUNE_DAYS, {"trail": tn}))
    # (3) PB_ATR
    for v in PB_ATR_GRID:
        if v == BASELINE_CFG["pb_atr"]:
            continue
        phase_a_jobs.append(_make_args(f"pb_atr={v}", TUNE_DAYS, {"pb_atr": v}))
    # (4) PB_WAIT
    for v in PB_WAIT_GRID:
        if v == BASELINE_CFG["pb_wait"]:
            continue
        phase_a_jobs.append(_make_args(f"pb_wait={v}", TUNE_DAYS, {"pb_wait": v}))
    # (5) VWAP_BUF
    for v in VWAP_BUF_GRID:
        if v == BASELINE_CFG["vwap_buf"]:
            continue
        phase_a_jobs.append(_make_args(f"vwap_buf={v}", TUNE_DAYS, {"vwap_buf": v}))
    # (6) min_quality
    for v in MIN_QUALITY_GRID:
        if v == BASELINE_CFG["min_q"]:
            continue
        phase_a_jobs.append(_make_args(f"min_q={v}", TUNE_DAYS, {"min_q": v}))
    # (7) Toxic
    for k in TOXIC_VARIANTS:
        if k == BASELINE_CFG["toxic"]:
            continue
        phase_a_jobs.append(_make_args(f"toxic={k}", TUNE_DAYS, {"toxic": k}))

    print(f"    {len(phase_a_jobs)} backtests across 7 dims, {WORKERS} workers ...")
    a_results = []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, phase_a_jobs), 1):
            a_results.append(r)
            if i % 10 == 0 or i == len(phase_a_jobs):
                print(f"      {i:3d}/{len(phase_a_jobs)}   ({time.time()-t0:.0f}s)")
    print()

    # Bucket by dim, sort by pnl, pick top-3 (we'll later pick top-2 for B)
    def _dim_of(r):
        return r["label"].split("=")[0]

    by_dim = {}
    for r in a_results:
        d = _dim_of(r)
        by_dim.setdefault(d, []).append(r)
    # Include baseline value as a candidate (so dims where baseline is best
    # don't lose by being excluded from Phase B).
    base_per_dim = {
        "sl":       {"label": f"sl={BASELINE_CFG['sl']}",       "val": BASELINE_CFG["sl"]},
        "trail":    {"label": f"trail={BASELINE_CFG['trail']}", "val": BASELINE_CFG["trail"]},
        "pb_atr":   {"label": f"pb_atr={BASELINE_CFG['pb_atr']}", "val": BASELINE_CFG["pb_atr"]},
        "pb_wait":  {"label": f"pb_wait={BASELINE_CFG['pb_wait']}", "val": BASELINE_CFG["pb_wait"]},
        "vwap_buf": {"label": f"vwap_buf={BASELINE_CFG['vwap_buf']}", "val": BASELINE_CFG["vwap_buf"]},
        "min_q":    {"label": f"min_q={BASELINE_CFG['min_q']}", "val": BASELINE_CFG["min_q"]},
        "toxic":    {"label": f"toxic={BASELINE_CFG['toxic']}", "val": BASELINE_CFG["toxic"]},
    }
    # Inject baseline into each dim (same pnl as the global baseline)
    for d in by_dim:
        by_dim[d].append({
            "label": base_per_dim[d]["label"],
            "ok": True, "days": TUNE_DAYS,
            **{k: v for k, v in baseline.items() if k not in ("label",)},
        })

    print("[A.1] Per-dim ranking (top-3 by PnL, pre-WF gate):\n")
    top_per_dim = {}
    for d in ["sl", "trail", "pb_atr", "pb_wait", "vwap_buf", "min_q", "toxic"]:
        sorted_r = sorted([r for r in by_dim[d] if r.get("ok")],
                          key=lambda x: -x["pnl"])[:3]
        top_per_dim[d] = sorted_r
        print(f"  {d}:")
        for r in sorted_r:
            print(_fmt_row(r))
        print()

    # ────────────────────────────────────────────────────────────────────
    # PHASE B — combinatorial (top-2 per dim) = 2^7 = 128 combos
    # ────────────────────────────────────────────────────────────────────
    print("[B] Phase B — 2^7 = 128 combos (top-2 per dim)")
    DIM_FIELD = {
        "sl":       ("sl", "sl"),
        "trail":    ("trail", "trail"),
        "pb_atr":   ("pb_atr", "pb_atr"),
        "pb_wait":  ("pb_wait", "pb_wait"),
        "vwap_buf": ("vwap_buf", "vwap_buf"),
        "min_q":    ("min_q", "min_q"),
        "toxic":    ("toxic", "toxic"),
    }
    # Extract top-2 *values* per dim
    top2_vals = {}
    for d, rows in top_per_dim.items():
        field = DIM_FIELD[d][1]
        seen, vals = set(), []
        for r in rows:
            v = r.get(field)
            key = str(v)
            if key not in seen:
                seen.add(key); vals.append(v)
            if len(vals) >= 2:
                break
        top2_vals[d] = vals
        print(f"    {d}: {vals}")

    import itertools
    combo_jobs = []
    combos = list(itertools.product(
        top2_vals["sl"], top2_vals["trail"], top2_vals["pb_atr"],
        top2_vals["pb_wait"], top2_vals["vwap_buf"], top2_vals["min_q"],
        top2_vals["toxic"]
    ))
    for (sl, tr, pa, pw, vb, mq, tx) in combos:
        label = f"sl={sl}|tr={tr}|pa={pa}|pw={pw}|vb={vb}|mq={mq}|tx={tx}"
        combo_jobs.append((label, TUNE_DAYS, sl, tr, pa, pw, vb, mq, tx))
    print(f"\n    {len(combo_jobs)} combos ...")
    b_results = []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, combo_jobs), 1):
            b_results.append(r)
            if i % 16 == 0 or i == len(combo_jobs):
                print(f"      {i:3d}/{len(combo_jobs)}   ({time.time()-t0:.0f}s)")
    print()

    # ────────────────────────────────────────────────────────────────────
    # PHASE C — WF-validate top-5
    # ────────────────────────────────────────────────────────────────────
    print("[C] Phase C — walk-forward top-5")
    b_ok = [r for r in b_results if r.get("ok") and r["trades"] >= 30]
    b_ok.sort(key=lambda x: -x["pnl"])
    top5 = b_ok[:5]
    print(f"    Top-5 Phase B candidates (pre-WF):\n")
    for r in top5:
        print(_fmt_row({**r, "label": r["label"][:32]}))
    print()

    wf_jobs = []
    for r in top5:
        for d in WF_FOLDS_DAYS:
            label = f"WF{d:3d}|{r['label'][:24]}"
            wf_jobs.append((label, d, r["sl"], r["trail"], r["pb_atr"],
                            r["pb_wait"], r["vwap_buf"], r["min_q"], r["toxic"]))
    # Also WF the baseline at each fold
    for d in WF_FOLDS_DAYS:
        wf_jobs.append(_baseline_args(label=f"WF{d:3d}|baseline", days=d))

    print(f"    {len(wf_jobs)} WF backtests ...")
    wf_results = []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, wf_jobs), 1):
            wf_results.append(r)
            if i % 10 == 0 or i == len(wf_jobs):
                print(f"      {i:3d}/{len(wf_jobs)}   ({time.time()-t0:.0f}s)")
    print()

    # Group WF results by (sl, trail, pb_atr, pb_wait, vwap_buf, min_q, toxic)
    def _knob_key(r):
        return (r["sl"], r["trail"], r["pb_atr"], r["pb_wait"],
                str(r["vwap_buf"]), r["min_q"], r["toxic"])

    base_key = _knob_key({"sl": BASELINE_CFG["sl"], "trail": BASELINE_CFG["trail"],
                          "pb_atr": BASELINE_CFG["pb_atr"], "pb_wait": BASELINE_CFG["pb_wait"],
                          "vwap_buf": BASELINE_CFG["vwap_buf"], "min_q": BASELINE_CFG["min_q"],
                          "toxic": BASELINE_CFG["toxic"]})

    wf_by_key = {}
    for r in wf_results:
        if not r.get("ok"):
            continue
        k = _knob_key(r)
        wf_by_key.setdefault(k, []).append(r)

    # Build per-candidate WF summary
    print("[C.1] WF summary per candidate (folds 60/90/120/150/180):\n")
    candidate_summaries = []
    base_folds = sorted(wf_by_key.get(base_key, []), key=lambda x: x["days"])
    base_fold_pnl = {f["days"]: f["pnl"] for f in base_folds}

    for r in top5:
        k = _knob_key(r)
        folds = sorted(wf_by_key.get(k, []), key=lambda x: x["days"])
        if not folds:
            print(f"    {r['label']:<60}  WF MISSING")
            continue
        fold_arr = []
        pos = 0
        delta_pos = 0
        for f in folds:
            base_p = base_fold_pnl.get(f["days"], 0.0)
            delta = f["pnl"] - base_p
            fold_arr.append({"days": f["days"], "pnl": f["pnl"], "pf": f["pf"],
                             "n": f["trades"], "delta_vs_base": round(delta, 2)})
            if f["pnl"] > 0:
                pos += 1
            if delta > 0:
                delta_pos += 1
        avg_pf = round(sum(f["pf"] for f in folds) / max(1, len(folds)), 3)
        full_delta = r["pnl"] - BASE_PNL
        summary = {
            "label":            r["label"],
            "sl":               r["sl"],
            "trail":            r["trail"],
            "pb_atr":           r["pb_atr"],
            "pb_wait":          r["pb_wait"],
            "vwap_buf":         r["vwap_buf"],
            "min_q":            r["min_q"],
            "toxic":            r["toxic"],
            "full_180d_pnl":    round(r["pnl"], 2),
            "full_180d_pf":     r["pf"],
            "full_180d_trades": r["trades"],
            "full_180d_wr":     r["wr"],
            "full_180d_dd":     r["dd"],
            "delta_vs_base":    round(full_delta, 2),
            "wf_folds":         fold_arr,
            "wf_avg_pf":        avg_pf,
            "wf_pos_folds":     pos,
            "wf_delta_pos_folds": delta_pos,
            "passes_ship":      bool(full_delta >= MIN_DELTA_PNL and delta_pos >= MIN_WF_POS),
        }
        candidate_summaries.append(summary)
        print(f"    {r['label'][:62]:<62}")
        print(f"      pnl=${r['pnl']:+9.2f}  PF={r['pf']:5.2f}  n={r['trades']:3d}  "
              f"WF avg_pf={avg_pf:5.2f}  WF +PnL {pos}/5  WF +Δ {delta_pos}/5  "
              f"Δ=${full_delta:+.2f}   ship={summary['passes_ship']}")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Pick winner
    # ────────────────────────────────────────────────────────────────────
    candidate_summaries.sort(key=lambda x: (-int(x["passes_ship"]), -x["delta_vs_base"]))
    winner = candidate_summaries[0] if (candidate_summaries
                                        and candidate_summaries[0]["passes_ship"]) else None

    verdict = "SHIP" if winner else "HOLD — no candidate clears Δ+$100 + WF ≥3/5 vs baseline"

    print(f"[D] VERDICT: {verdict}")
    if winner:
        print(f"    Winner: {winner['label']}")
        print(f"    Δ vs baseline: ${winner['delta_vs_base']:+.2f}")
        print(f"    WF +PnL folds: {winner['wf_pos_folds']}/5  "
              f"WF +Δ folds: {winner['wf_delta_pos_folds']}/5")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Save JSON
    # ────────────────────────────────────────────────────────────────────
    out = {
        "symbol":        SYMBOL,
        "generated_at":  datetime.now().isoformat(timespec="seconds"),
        "tune_days":     TUNE_DAYS,
        "wf_folds_days": WF_FOLDS_DAYS,
        "workers":       WORKERS,
        "elapsed_seconds": round(time.time() - t0, 1),
        "baseline_config":  BASELINE_CFG,
        "baseline_result":  {k: baseline[k] for k in
                             ("trades", "pf", "pnl", "wr", "dd", "avg_r",
                              "avg_peak_r", "avg_giveback")},
        "ship_criteria": {
            "min_delta_pnl_full":   MIN_DELTA_PNL,
            "min_wf_delta_pos_folds": MIN_WF_POS,
            "wf_folds_used_for_delta": "delta_vs_baseline (per-fold)",
        },
        "verdict":     verdict,
        "winner":      winner,
        "candidates":  candidate_summaries,
        "phase_a": {
            "top_per_dim":      {
                d: [{k: v for k, v in r.items() if k != "label"} | {"label": r["label"]}
                    for r in rows] for d, rows in top_per_dim.items()
            },
            "all_results": [
                {k: v for k, v in r.items()} for r in a_results
            ],
        },
        "phase_b": {
            "top2_per_dim": {d: [str(v) for v in vs] for d, vs in top2_vals.items()},
            "n_combos":     len(combo_jobs),
            "results":      [{k: v for k, v in r.items()} for r in b_results],
        },
        "trail_profiles_used": {k: v for k, v in TRAIL_PROFILES.items()},
        "toxic_variants_used": {k: (sorted(list(v)) if v else None)
                                for k, v in TOXIC_VARIANTS.items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"[E] Saved JSON  -> {OUT_JSON}")

    # Markdown summary
    md = []
    md.append(f"# UKOUSD per-symbol fine-tune  —  {datetime.now().isoformat(timespec='seconds')}\n")
    md.append(f"**Symbol:** {SYMBOL}  |  **Days:** {TUNE_DAYS}  |  "
              f"**Workers:** {WORKERS}  |  **Elapsed:** {round(time.time()-t0,1)}s\n")
    md.append("## Baseline (current live config)\n")
    md.append(f"- Config: `{BASELINE_CFG}`")
    md.append(f"- Result: n={baseline['trades']}  PF={baseline['pf']:.2f}  "
              f"PnL=${baseline['pnl']:.2f}  WR={baseline['wr']:.1f}%  DD={baseline['dd']:.1f}%\n")
    md.append(f"## Verdict: **{verdict}**\n")
    if winner:
        md.append(f"- Winner config: `sl={winner['sl']}  trail={winner['trail']}  "
                  f"pb_atr={winner['pb_atr']}  pb_wait={winner['pb_wait']}  "
                  f"vwap_buf={winner['vwap_buf']}  min_q={winner['min_q']}  "
                  f"toxic={winner['toxic']}`")
        md.append(f"- Δ vs baseline: **${winner['delta_vs_base']:+.2f}**")
        md.append(f"- Full-180d: n={winner['full_180d_trades']}  PF={winner['full_180d_pf']:.2f}  "
                  f"WR={winner['full_180d_wr']:.1f}%  DD={winner['full_180d_dd']:.1f}%")
        md.append(f"- WF +PnL: {winner['wf_pos_folds']}/5  WF +Δ vs base: {winner['wf_delta_pos_folds']}/5")
        md.append(f"- Folds: {winner['wf_folds']}\n")
    md.append("## Phase A — per-dim ranking (top-3 by PnL)\n")
    for d, rows in top_per_dim.items():
        md.append(f"### {d}\n")
        md.append("| value | PnL | PF | n | WR% | DD% |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            v = r.get(DIM_FIELD[d][1])
            md.append(f"| `{v}` | ${r['pnl']:+.2f} | {r['pf']:.2f} | {r['trades']} | "
                      f"{r['wr']:.1f} | {r['dd']:.1f} |")
        md.append("")
    md.append("## Phase B — top-5 candidates by full-180d PnL (pre-WF)\n")
    md.append("| sl | trail | pb_atr | pb_wait | vwap | min_q | toxic | PnL | PF | n |")
    md.append("|---|---|---:|---:|---|---:|---|---:|---:|---:|")
    for r in top5:
        md.append(f"| {r['sl']} | {r['trail']} | {r['pb_atr']} | {r['pb_wait']} | "
                  f"{r['vwap_buf']} | {r['min_q']} | {r['toxic']} | "
                  f"${r['pnl']:+.2f} | {r['pf']:.2f} | {r['trades']} |")
    md.append("\n## Phase C — WF folds per candidate (vs baseline)\n")
    md.append("| Candidate | f60 | f90 | f120 | f150 | f180 | +PnL/5 | +Δ/5 | Ship |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in candidate_summaries:
        f = {x["days"]: x for x in c["wf_folds"]}
        cells = "|".join(
            (f"${f[d]['pnl']:+.0f}" if d in f else "—") for d in WF_FOLDS_DAYS
        )
        md.append(f"| `{c['label'][:32]}` | " + cells + f" | "
                  f"{c['wf_pos_folds']}/5 | {c['wf_delta_pos_folds']}/5 | "
                  f"{'YES' if c['passes_ship'] else 'NO'} |")
    md.append("")
    OUT_MD.write_text("\n".join(md))
    print(f"[E] Saved MD    -> {OUT_MD}\n")

    print(f"DONE — total elapsed {round(time.time()-t0,1)}s")
    return out


if __name__ == "__main__":
    main()
