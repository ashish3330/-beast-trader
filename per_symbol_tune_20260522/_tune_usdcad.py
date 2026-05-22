#!/usr/bin/env python3 -B
"""Per-symbol fine-tune for USDCAD (2026-05-22 session).

Dimensions tuned (7):
  1. SL ATR mult                 ∈ {0.3, 0.4, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0}
  2. Trail profile               7 named profiles
  3. Pullback ATR retrace        ∈ {0.4, 0.5, 0.6, 0.8, 1.0, 1.2}
  4. Pullback max wait bars      ∈ {3, 4, 5, 6, 8}
  5. VWAP buffer (× ATR)         ∈ {0.0_disabled, 0.3, 0.5, 0.7, 1.0}
  6. Min-quality uniform         ∈ {28, 30, 33, 35, 38, 40, 43}
  7. Toxic-hour variants         ∈ ({15,16}, {15}, {16}, set(), {15,16,17})

Pipeline:
  Phase A — anchored axis sweep   (~46 BTs, find best on each axis vs baseline)
  Phase B — top combinations      (≤128 cartesian samples from top-axis tiers)
  Phase C — walk-forward          (top 5 by Δ on folds {60,90,120,150,180})

Ship rule  Δ ≥ $50  AND  WF positive folds ≥ 3 / 5.

READ-ONLY: writes JSON+MD to per_symbol_tune_20260522/ only. Does NOT
mutate config.py / auto_tuned.py / v5_backtest.py on disk. Live config is
monkey-patched inside worker processes and reloaded each iteration.
"""
from __future__ import annotations
import os
import sys
import json
import time
import random
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "per_symbol_tune_20260522"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "USDCAD.json"
OUT_MD = OUT_DIR / "USDCAD.md"

SYMBOL = "USDCAD"
DAYS = 180
WF_FOLDS = [60, 90, 120, 150, 180]

SHIP_MIN_DELTA = 50.0
SHIP_MIN_WF_POS = 3

# ── Dimensions ─────────────────────────────────────────────────────────
SL_GRID = [0.3, 0.4, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

# Trail profiles in LIVE (R, type, param) format. Backtest converts to
# (R, param, type) internally via SYMBOL_TRAIL_OVERRIDE pipeline.
TRAIL_PROFILES = {
    "_TIGHT_LOCK":      [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_WIDE_RUNNER":     [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7),
                         (1.5, "lock", 0.5), (0.7, "be", 0.0)],
    "_RANGE_TIGHT":     [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0)],
    "_TREND_LOOSE":     [(15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5),
                         (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_AGGR_LOCK":       [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8),
                         (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "_RUNNER_NO_BE":    [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5),
                         (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)],
    "_WIDE_RUNNER_BE07":[(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7),
                         (1.5, "lock", 0.5), (0.7, "be", 0.0)],
}
TRAIL_NAMES = list(TRAIL_PROFILES.keys())

PB_ATR_GRID = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_GRID = [3, 4, 5, 6, 8]

# VWAP buffer values; -1 sentinel = disable filter entirely
VWAP_GRID = [-1, 0.3, 0.5, 0.7, 1.0]

MIN_QUALITY_GRID = [28, 30, 33, 35, 38, 40, 43]

TOXIC_VARIANTS = {
    "both":      {15, 16},        # baseline
    "h15_only":  {15},
    "h16_only":  {16},
    "none":      set(),
    "h15_16_17": {15, 16, 17},
}

# Baseline anchor (current live config) for Phase A
BASE = {
    "sl": 2.5,
    "trail": "_WIDE_RUNNER",
    "pb_atr": 0.8,
    "pb_wait": 5,
    "vwap": 0.5,
    "min_q": "default",           # use SIGNAL_QUALITY_SYMBOL_AUTO (per-regime)
    "toxic": "both",
}


# ─── Worker: run isolated backtest with monkey-patched config ──────────
def _bt_isolated(symbol, days, *, sl=None, trail_name=None, pb_atr=None,
                 pb_wait=None, vwap_buf=None, min_q=None, toxic_set=None):
    """Reload config + backtest then run one BT with given overrides.

    Each call is self-contained → safe across multiprocessing workers.
    `min_q` semantics:
        None           → keep current live SIGNAL_QUALITY_SYMBOL
        "default"      → same as None
        int            → broadcast that int to all 4 regimes
        dict           → apply as-is
    `toxic_set` is the per-symbol EXTRA set merged into TOXIC_HOURS_PER_SYMBOL.
    `vwap_buf` semantics:
        None           → backtest applies its hardcoded 0.5×ATR buffer
        -1             → disable VWAP filter (monkey-patch to no-op)
        float          → use this ATR multiplier (monkey-patch)
    """
    import importlib
    import config as cfg
    importlib.reload(cfg)

    # 1) SL override
    if sl is not None:
        cfg.SYMBOL_ATR_SL_OVERRIDE = dict(cfg.SYMBOL_ATR_SL_OVERRIDE)
        cfg.SYMBOL_ATR_SL_OVERRIDE[symbol] = float(sl)
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME = dict(cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME)
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME[symbol] = {}

    # 2) Trail profile
    if trail_name is not None:
        trail = TRAIL_PROFILES[trail_name]
        cfg.SYMBOL_TRAIL_OVERRIDE = dict(cfg.SYMBOL_TRAIL_OVERRIDE)
        cfg.SYMBOL_TRAIL_OVERRIDE[symbol] = trail
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE = dict(cfg.SYMBOL_REGIME_TRAIL_OVERRIDE)
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE[symbol] = {
            "trending": trail, "ranging": trail,
            "volatile": trail, "low_vol":  trail,
        }

    # 3) Pullback
    if pb_atr is not None:
        cfg.PULLBACK_ATR_RETRACE = float(pb_atr)
    if pb_wait is not None:
        cfg.PULLBACK_MAX_WAIT_BARS = int(pb_wait)

    # 4) min_quality
    if min_q is not None and min_q != "default":
        cfg.SIGNAL_QUALITY_SYMBOL = dict(cfg.SIGNAL_QUALITY_SYMBOL)
        if isinstance(min_q, int):
            cfg.SIGNAL_QUALITY_SYMBOL[symbol] = {
                "trending": min_q, "ranging": min_q,
                "volatile": min_q, "low_vol":  min_q,
            }
        elif isinstance(min_q, dict):
            cfg.SIGNAL_QUALITY_SYMBOL[symbol] = dict(min_q)

    # 5) Toxic hours
    if toxic_set is not None:
        cfg.TOXIC_HOURS_PER_SYMBOL = dict(cfg.TOXIC_HOURS_PER_SYMBOL)
        cfg.TOXIC_HOURS_PER_SYMBOL[symbol] = set(toxic_set)

    # Reload backtest module so it picks up patched config.
    # Reload momentum_scorer too so any prior monkey-patch is wiped.
    from signals import momentum_scorer as _ms
    importlib.reload(_ms)
    import backtest.v5_backtest as bt
    importlib.reload(bt)

    # Inject toxic-hour set into BT module (BT reads TOXIC_HOURS_UTC only —
    # per-symbol extras need a runtime merge inside the worker).
    bt.TOXIC_HOURS = set(cfg.TOXIC_HOURS_UTC) | set(
        cfg.TOXIC_HOURS_PER_SYMBOL.get(symbol, set()))
    # Recompute exempt for completeness (no symbol-specific change here).
    bt.TOXIC_EXEMPT = {k: set(v) for k, v in cfg.TOXIC_HOUR_EXEMPT.items()}

    # 6) VWAP buffer override — when vwap_buf is None we leave BT's
    # hardcoded 0.5×ATR gate alone. Otherwise we monkey-patch:
    #   - vwap_buf == -1  → disable entirely (NaN-fill vwap array)
    #   - vwap_buf >=0    → wrap _score_with_components to apply our
    #                       buffer AND also disable BT's inline gate
    #                       (NaN-fill) so only our buffer applies.
    if vwap_buf is not None:
        buf_mult = float(vwap_buf)
        import numpy as _np
        _orig_comp = _ms._compute_indicators

        if buf_mult < 0:
            # Disabled — nuke vwap array so BT's inline 0.5×ATR sees NaN
            # and falls through.
            def _nuke_vwap(df, icfg, _o=_orig_comp):
                ind2 = _o(df, icfg)
                if isinstance(ind2, dict) and "vwap" in ind2:
                    ind2["vwap"] = _np.full(len(ind2["vwap"]), _np.nan, dtype=float)
                return ind2
            bt._compute_indicators = _nuke_vwap
            _ms._compute_indicators = _nuke_vwap
        else:
            # Custom buffer — apply via _score_with_components wrapper.
            # Also disable BT's inline 0.5×ATR by nuking vwap (so only our
            # wrapper sees a non-NaN vwap, stashed under _vwap_orig).
            _orig_score = _ms._score_with_components

            def _vwap_filtered_score(ind, i, weights=None):
                ls, ss, cl, cs = _orig_score(ind, i, weights=weights)
                vw_arr = ind.get("_vwap_orig") if isinstance(ind, dict) else None
                at_arr = ind.get("at") if isinstance(ind, dict) else None
                if vw_arr is None or at_arr is None or i >= len(vw_arr):
                    return ls, ss, cl, cs
                vw = float(vw_arr[i])
                atr_v = float(at_arr[i])
                price = float(ind["c"][i])
                if vw != vw or atr_v <= 0:  # NaN / invalid
                    return ls, ss, cl, cs
                buf = atr_v * buf_mult
                if ls >= ss:  # LONG candidate
                    if price <= (vw - buf):
                        return 0.0, 0.0, cl, cs
                else:        # SHORT candidate
                    if price >= (vw + buf):
                        return 0.0, 0.0, cl, cs
                return ls, ss, cl, cs

            def _stash_and_nuke(df, icfg, _o=_orig_comp):
                ind2 = _o(df, icfg)
                if isinstance(ind2, dict) and "vwap" in ind2:
                    ind2["_vwap_orig"] = ind2["vwap"]
                    ind2["vwap"] = _np.full(len(ind2["vwap"]), _np.nan, dtype=float)
                return ind2

            bt._compute_indicators = _stash_and_nuke
            _ms._compute_indicators = _stash_and_nuke
            bt._score_with_components = _vwap_filtered_score
            _ms._score_with_components = _vwap_filtered_score

    # Run BT
    return bt.backtest_symbol(symbol, days=days, verbose=False)


def _safe_bt(spec):
    """spec is dict with keys (sl, trail, pb_atr, pb_wait, vwap, min_q, toxic)"""
    try:
        r = _bt_isolated(
            SYMBOL, DAYS,
            sl=spec["sl"], trail_name=spec["trail"],
            pb_atr=spec["pb_atr"], pb_wait=spec["pb_wait"],
            vwap_buf=spec["vwap"], min_q=spec["min_q"],
            toxic_set=TOXIC_VARIANTS[spec["toxic"]],
        )
        if r is None:
            return {"spec": spec, "error": "no_data"}
        return {
            "spec": spec,
            "trades": int(r.get("trades", 0)),
            "wr": round(float(r.get("wr", 0) or 0), 1),
            "pf": round(float(r.get("pf", 0) or 0), 2),
            "pnl": round(float(r.get("pnl", 0) or 0), 2),
            "dd": round(float(r.get("dd", 0) or 0), 2),
            "avg_r": round(float(r.get("avg_r", 0) or 0), 3),
        }
    except Exception as e:
        return {"spec": spec, "error": repr(e), "tb": traceback.format_exc()}


def _safe_bt_days(args):
    """For WF: (spec, days)"""
    spec, days = args
    try:
        r = _bt_isolated(
            SYMBOL, days,
            sl=spec["sl"], trail_name=spec["trail"],
            pb_atr=spec["pb_atr"], pb_wait=spec["pb_wait"],
            vwap_buf=spec["vwap"], min_q=spec["min_q"],
            toxic_set=TOXIC_VARIANTS[spec["toxic"]],
        )
        if r is None:
            return {"spec": spec, "days": days, "error": "no_data"}
        return {
            "spec": spec, "days": days,
            "trades": int(r.get("trades", 0)),
            "wr": round(float(r.get("wr", 0) or 0), 1),
            "pf": round(float(r.get("pf", 0) or 0), 2),
            "pnl": round(float(r.get("pnl", 0) or 0), 2),
            "dd": round(float(r.get("dd", 0) or 0), 2),
            "avg_r": round(float(r.get("avg_r", 0) or 0), 3),
        }
    except Exception as e:
        return {"spec": spec, "days": days, "error": repr(e)}


def _spec_to_key(spec):
    return (spec["sl"], spec["trail"], spec["pb_atr"], spec["pb_wait"],
            spec["vwap"], spec["min_q"], spec["toxic"])


def _baseline_spec():
    return {
        "sl": BASE["sl"], "trail": BASE["trail"],
        "pb_atr": BASE["pb_atr"], "pb_wait": BASE["pb_wait"],
        "vwap": BASE["vwap"], "min_q": BASE["min_q"],
        "toxic": BASE["toxic"],
    }


def phase_a(pool, baseline_pnl):
    """Anchored axis sweep — vary each axis holding others fixed at BASE."""
    jobs = []
    sweeps = {
        "sl": [], "trail": [], "pb_atr": [],
        "pb_wait": [], "vwap": [], "min_q": [], "toxic": [],
    }

    base = _baseline_spec()
    # SL axis
    for v in SL_GRID:
        s = dict(base); s["sl"] = v
        jobs.append(("sl", v, s))
    # Trail axis
    for v in TRAIL_NAMES:
        s = dict(base); s["trail"] = v
        jobs.append(("trail", v, s))
    # PB atr
    for v in PB_ATR_GRID:
        s = dict(base); s["pb_atr"] = v
        jobs.append(("pb_atr", v, s))
    # PB wait
    for v in PB_WAIT_GRID:
        s = dict(base); s["pb_wait"] = v
        jobs.append(("pb_wait", v, s))
    # VWAP
    for v in VWAP_GRID:
        s = dict(base); s["vwap"] = v
        jobs.append(("vwap", v, s))
    # min_q
    for v in MIN_QUALITY_GRID:
        s = dict(base); s["min_q"] = v
        jobs.append(("min_q", v, s))
    # toxic
    for v in TOXIC_VARIANTS.keys():
        s = dict(base); s["toxic"] = v
        jobs.append(("toxic", v, s))

    print(f"[A] axis sweep {len(jobs)} BTs")
    specs = [j[2] for j in jobs]
    t0 = time.time()
    results = []
    for i, r in enumerate(pool.imap(_safe_bt, specs), 1):
        results.append(r)
        if i % 10 == 0 or i == len(specs):
            print(f"  [A] {i:3d}/{len(specs)} ({time.time()-t0:.0f}s)")
    # Stitch axis/value
    for (axis, val, _s), r in zip(jobs, results):
        sweeps[axis].append({"value": str(val), "result": r})
    return sweeps


def _top_by_pnl(rows, k):
    good = [r for r in rows if "pnl" in r.get("result", {})]
    good.sort(key=lambda r: -(r["result"].get("pnl", -1e9)))
    return good[:k]


def phase_b(pool, sweeps, baseline_pnl, budget=128, seed=42):
    """Top-tier combinations: take top values per axis, sample combos."""
    # Top picks per axis (best by 180d PnL)
    top = {}
    sizes = {
        "sl": 3, "trail": 3, "pb_atr": 2, "pb_wait": 2,
        "vwap": 2, "min_q": 2, "toxic": 2,
    }
    for axis, k in sizes.items():
        rows = sweeps[axis]
        # Normalise value to native dtype
        rows_sorted = []
        for r in rows:
            res = r.get("result", {})
            pnl = res.get("pnl") if isinstance(res, dict) else None
            rows_sorted.append((pnl if pnl is not None else -1e9, r["value"], r))
        rows_sorted.sort(key=lambda t: -t[0])
        top[axis] = [r["value"] for _, _, r in rows_sorted[:k]]
    print(f"[B] top picks per axis: {top}")

    # Cartesian product size
    total = 1
    for v in top.values():
        total *= len(v)
    print(f"[B] full cross-product = {total} combos; budget = {budget}")

    # Build full set, dedupe, sample if needed
    combos = []
    seen = set()
    for sl in top["sl"]:
        for tr in top["trail"]:
            for pa in top["pb_atr"]:
                for pw in top["pb_wait"]:
                    for vw in top["vwap"]:
                        for mq in top["min_q"]:
                            for tx in top["toxic"]:
                                # Restore correct native types
                                sl_f = float(sl)
                                pa_f = float(pa)
                                pw_i = int(pw)
                                vw_f = float(vw)
                                mq_v = mq if mq == "default" else int(mq)
                                tx_s = tx
                                spec = {
                                    "sl": sl_f, "trail": tr,
                                    "pb_atr": pa_f, "pb_wait": pw_i,
                                    "vwap": vw_f, "min_q": mq_v,
                                    "toxic": tx_s,
                                }
                                k = _spec_to_key(spec)
                                if k in seen:
                                    continue
                                seen.add(k)
                                combos.append(spec)
    if len(combos) > budget:
        random.seed(seed)
        combos = random.sample(combos, budget)
    print(f"[B] running {len(combos)} BTs")

    t0 = time.time()
    results = []
    for i, r in enumerate(pool.imap(_safe_bt, combos), 1):
        results.append(r)
        if i % 16 == 0 or i == len(combos):
            print(f"  [B] {i:3d}/{len(combos)} ({time.time()-t0:.0f}s)")
    # Filter Δ ≥ ship min, sort by PnL
    enriched = []
    for r in results:
        res = r if "pnl" in r else None
        if res is None or "error" in r:
            continue
        delta = r["pnl"] - baseline_pnl
        enriched.append({**r, "delta_pnl": round(delta, 2)})
    enriched.sort(key=lambda x: -x["pnl"])
    return enriched


def phase_c(pool, top_combos, baseline_pnl, baseline_per_fold):
    """Walk-forward validate top-5 candidates on 5 folds each."""
    cands = top_combos[:5]
    print(f"[C] WF validate top {len(cands)} candidates × {len(WF_FOLDS)} folds = "
          f"{len(cands) * len(WF_FOLDS)} BTs")
    jobs = []
    for c in cands:
        for d in WF_FOLDS:
            jobs.append((c["spec"], d))
    t0 = time.time()
    results = []
    for i, r in enumerate(pool.imap(_safe_bt_days, jobs), 1):
        results.append(r)
        if i % 5 == 0 or i == len(jobs):
            print(f"  [C] {i:3d}/{len(jobs)} ({time.time()-t0:.0f}s)")

    # Aggregate per candidate
    by_key = {}
    for c in cands:
        by_key[_spec_to_key(c["spec"])] = {"spec": c["spec"], "candidate": c, "folds": []}
    for r in results:
        k = _spec_to_key(r["spec"])
        if k in by_key:
            by_key[k]["folds"].append(r)

    # Score each
    summary = []
    for k, v in by_key.items():
        folds = sorted(v["folds"], key=lambda x: x["days"])
        pos = 0
        pfs = []
        deltas = []
        for f in folds:
            if "pnl" not in f:
                continue
            if f["pnl"] > 0:
                pos += 1
            if f["pf"]:
                pfs.append(f["pf"])
            base_pnl_fold = baseline_per_fold.get(f["days"], 0)
            deltas.append(round(f["pnl"] - base_pnl_fold, 2))
        avg_pf = round(sum(pfs) / max(1, len(pfs)), 3) if pfs else 0.0
        cand = v["candidate"]
        delta_180 = cand["delta_pnl"]
        ship_ok = (delta_180 >= SHIP_MIN_DELTA) and (pos >= SHIP_MIN_WF_POS)
        summary.append({
            "spec": v["spec"],
            "n180": cand["trades"],
            "pf180": cand["pf"],
            "pnl180": cand["pnl"],
            "delta_180": delta_180,
            "wf_pos_folds": pos,
            "wf_avg_pf": avg_pf,
            "wf_folds_pnl": [(f["days"], f.get("pnl"), f.get("pf"))
                             for f in folds],
            "wf_folds_delta": deltas,
            "ship": ship_ok,
        })
    summary.sort(key=lambda x: (-int(x["ship"]), -x["delta_180"]))
    return summary


def main():
    t0 = time.time()
    workers = max(2, min(8, os.cpu_count() or 4))
    print(f"USDCAD per-symbol tune — workers={workers}")

    out = {
        "session": "per_symbol_tune_20260522",
        "symbol": SYMBOL,
        "days": DAYS,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "dimensions": {
            "sl_grid": SL_GRID,
            "trail_names": TRAIL_NAMES,
            "pb_atr_grid": PB_ATR_GRID,
            "pb_wait_grid": PB_WAIT_GRID,
            "vwap_grid": VWAP_GRID,
            "min_quality_grid": MIN_QUALITY_GRID,
            "toxic_variants": {k: sorted(list(v)) for k, v in TOXIC_VARIANTS.items()},
        },
        "baseline_anchor": BASE,
        "ship_gate": {
            "min_delta_pnl_180d": SHIP_MIN_DELTA,
            "min_wf_positive_folds": SHIP_MIN_WF_POS,
            "wf_folds_days": WF_FOLDS,
        },
    }

    with Pool(workers) as pool:
        # ── Baseline (current live config, no overrides) ──
        print("\n[0] Baseline (current live config)")
        base_res = _bt_isolated(SYMBOL, DAYS)  # no overrides = current live
        base_pnl = float(base_res.get("pnl", 0))
        out["baseline"] = {
            "trades": int(base_res.get("trades", 0)),
            "wr": round(float(base_res.get("wr", 0)), 1),
            "pf": round(float(base_res.get("pf", 0)), 2),
            "pnl": round(base_pnl, 2),
            "dd": round(float(base_res.get("dd", 0)), 2),
            "avg_r": round(float(base_res.get("avg_r", 0) or 0), 3),
        }
        print(f"  baseline 180d: n={out['baseline']['trades']} pf={out['baseline']['pf']:.2f} "
              f"pnl=${out['baseline']['pnl']:+.0f} wr={out['baseline']['wr']:.1f}%")

        # Per-fold baseline for delta calculation in Phase C
        print("[0b] Baseline per-fold (for Δ calc)")
        base_fold_jobs = [(_baseline_spec(), d) for d in WF_FOLDS]
        baseline_per_fold = {}
        for r in pool.imap(_safe_bt_days, base_fold_jobs):
            if "pnl" in r:
                baseline_per_fold[r["days"]] = r["pnl"]
                print(f"  base {r['days']:3d}d: pnl=${r['pnl']:+.0f} pf={r['pf']:.2f} n={r['trades']}")
        out["baseline_per_fold"] = baseline_per_fold

        # ── Phase A ──
        print("\n[A] Axis sweep")
        sweeps = phase_a(pool, base_pnl)
        out["phase_a"] = sweeps
        # Print best per axis
        print("\n[A] best per axis (by Δ vs baseline):")
        for axis, rows in sweeps.items():
            scored = []
            for r in rows:
                res = r.get("result", {})
                if isinstance(res, dict) and "pnl" in res:
                    scored.append((res["pnl"] - base_pnl, r["value"], res))
            scored.sort(key=lambda t: -t[0])
            if scored:
                d, v, res = scored[0]
                print(f"  {axis:10s}: best={v:<20s}  Δ=${d:+8.0f}  pnl=${res['pnl']:+8.0f}  "
                      f"pf={res['pf']:.2f}  n={res['trades']}")

        # ── Phase B ──
        print("\n[B] Top-combo cartesian sweep")
        top_combos = phase_b(pool, sweeps, base_pnl, budget=128)
        out["phase_b_top10"] = top_combos[:10]
        print("\n[B] top 10:")
        for i, c in enumerate(top_combos[:10], 1):
            s = c["spec"]
            print(f"  {i:2d}. SL={s['sl']:<4} {s['trail']:<18} pb={s['pb_atr']}×{s['pb_wait']}b "
                  f"vw={s['vwap']:<4} q={s['min_q']:<8} tox={s['toxic']:<10} "
                  f"pnl=${c['pnl']:+8.0f} Δ=${c['delta_pnl']:+7.0f} pf={c['pf']:.2f} n={c['trades']}")

        # ── Phase C ──
        print("\n[C] Walk-forward validation")
        wf_summary = phase_c(pool, top_combos, base_pnl, baseline_per_fold)
        out["phase_c"] = wf_summary
        print("\n[C] WF summary:")
        for i, w in enumerate(wf_summary, 1):
            s = w["spec"]
            mark = "✓SHIP" if w["ship"] else "—"
            print(f"  {i}. {mark} SL={s['sl']} {s['trail']:<18} "
                  f"pb={s['pb_atr']}×{s['pb_wait']}b vw={s['vwap']} q={s['min_q']} "
                  f"tox={s['toxic']:<10} Δ180=${w['delta_180']:+7.0f} "
                  f"WF pos={w['wf_pos_folds']}/5 avg_pf={w['wf_avg_pf']:.2f}")

        # Pick winner
        winner = next((w for w in wf_summary if w["ship"]), None)
        out["winner"] = winner
        if winner:
            s = winner["spec"]
            print(f"\nWINNER: SL={s['sl']} {s['trail']} pb={s['pb_atr']}×{s['pb_wait']}b "
                  f"vw={s['vwap']} q={s['min_q']} tox={s['toxic']}")
            print(f"  Δ=${winner['delta_180']:+.0f} 180d-pf={winner['pf180']:.2f} "
                  f"WF {winner['wf_pos_folds']}/5 pos")
        else:
            print("\nNo WF-pass candidate — keep current config.")

    out["elapsed_sec"] = round(time.time() - t0, 1)
    out["finished_at"] = datetime.utcnow().isoformat() + "Z"
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[done] wrote {OUT_JSON}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
