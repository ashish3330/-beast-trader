#!/usr/bin/env python3 -B
"""
ETHUSD per-symbol fine-tune — 2026-05-22.

Dimensions (per task spec):
  1. SL ATR mult: {0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5}
  2. Trail profile: 7 named profiles
  3. Pullback ATR retrace: {0.4, 0.6, 0.8, 1.0, 1.2}
  4. Pullback wait bars: {3, 5, 7}
  5. VWAP buffer: {0.0_disabled, 0.5_default}   (note: live source hardcodes
     0.5 inline at v5_backtest.py:769; READ-ONLY constraint allows binary
     toggle only. Other 3 levels {0.3,0.7,1.0} not testable without source edit.)
  6. min_quality (uniform across regimes): {28, 30, 33, 35, 38}
  7. Direction bias per regime — trending ∈ {LONG, SHORT, BOTH},
     volatile ∈ {LONG, SHORT, BOTH}. ranging/low_vol left default ("BOTH").

Phases:
  A) Coarse search: ~80 random combos (sklearn-free LHS-flavoured)
  B) Compose top-15 candidates with small neighborhood perturbations
  C) Walk-forward (60/90/120/150/180 days) on top-5 from A+B

Ship rule: Δ >= +$50 AND WF >= 3/5 positive folds AND WF avg_pf >= 1.2.

READ-ONLY. python3 -B. Cap 2h.
Output: per_symbol_tune_20260522/ETHUSD.json + .md
"""
import json
import os
import sys
import time
import random
import importlib
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

SYMBOL = "ETHUSD"
TUNE_DAYS = 180
WF_FOLDS = [60, 90, 120, 150, 180]
OUT_JSON = ROOT / "per_symbol_tune_20260522" / "ETHUSD.json"
OUT_MD   = ROOT / "per_symbol_tune_20260522" / "ETHUSD.md"

RNG = random.Random(20260522)

# ── Dimension grids ───────────────────────────────────────────────────────
SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5]

# 7 named trail profiles. First 5 match the prior metals_crypto tuner.
# Two new profiles cover crypto-style aggressive lock + tight runner.
TRAIL_PROFILES = {
    "_TIGHT_LOCK":          [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_WIDE_RUNNER":         [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)],
    "_AGGR_LOCK":           [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8), (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "_RUNNER_NO_BE":        [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5), (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)],
    "_WIDE_RUNNER_BE07":    [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)],
    # 8-step dense commodity-style lock — current live ETHUSD profile baseline approximator.
    "_COMMODITY_AGGRESSIVE": [(2.0, "lock", 1.5), (1.0, "lock", 0.7), (0.7, "lock", 0.5), (0.5, "lock", 0.35),
                              (0.35, "lock", 0.25), (0.25, "lock", 0.17), (0.18, "lock", 0.12), (0.12, "lock", 0.07),
                              (0.08, "lock", 0.03), (0.05, "be", 0.0)],
    # 7-step ETH-tuned profile from live SYMBOL_TRAIL_OVERRIDE["ETHUSD"].
    "_ETH_LIVE":            [(6.0, "trail", 0.3), (4.0, "trail", 0.5), (2.5, "trail", 0.8),
                             (1.5, "lock", 0.5), (1.0, "lock", 0.3), (0.7, "lock", 0.15), (0.4, "be", 0.0)],
}

PB_ATR_GRID  = [0.4, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_GRID = [3, 5, 7]
VWAP_GRID    = ["0.0_disabled", "0.5_default"]   # see header note
MQ_GRID      = [28, 30, 33, 35, 38]
DIR_GRID     = ["LONG", "SHORT", "BOTH"]

# Winner criteria
MIN_TRADES   = 20
MIN_PF       = 1.5
MIN_WF_PF    = 1.2
MIN_WF_POS   = 3
MIN_DELTA    = 50.0  # task spec: ship Δ >= $50


# ── Worker helpers ────────────────────────────────────────────────────────
def _apply_overrides_and_bt(cfg_overrides, params_extra, days):
    """In-process: mutate config, reload v5_backtest, run backtest_symbol.

    cfg_overrides dict supports keys:
      sl_mult     -> SYMBOL_ATR_SL_OVERRIDE[ETHUSD] = value (also clears regime SL)
      trail_steps -> SYMBOL_REGIME_TRAIL_OVERRIDE[ETHUSD][all regimes] = value (live tuple form)
                     also SYMBOL_TRAIL_OVERRIDE[ETHUSD] = value
      pb_atr      -> PULLBACK_ATR_RETRACE = value
      pb_wait     -> PULLBACK_MAX_WAIT_BARS = value
      vwap_disable-> if True, monkey-patch _compute_indicators to NaN vwap
      dir_regime  -> {regime: 'LONG'|'SHORT'|'BOTH'} for trending/volatile
    params_extra -> dict merged into backtest_symbol(params=...)
    """
    import config as cfg
    importlib.reload(cfg)

    sym = SYMBOL
    if "sl_mult" in cfg_overrides:
        cfg.SYMBOL_ATR_SL_OVERRIDE = dict(cfg.SYMBOL_ATR_SL_OVERRIDE)
        cfg.SYMBOL_ATR_SL_OVERRIDE[sym] = float(cfg_overrides["sl_mult"])
        # Clear per-regime SL so per-symbol value wins
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME = dict(cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME)
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME[sym] = {}

    if "trail_steps" in cfg_overrides:
        steps = cfg_overrides["trail_steps"]
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE = dict(cfg.SYMBOL_REGIME_TRAIL_OVERRIDE)
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE[sym] = {
            r: steps for r in ("trending", "ranging", "volatile", "low_vol")
        }
        cfg.SYMBOL_TRAIL_OVERRIDE = dict(cfg.SYMBOL_TRAIL_OVERRIDE)
        cfg.SYMBOL_TRAIL_OVERRIDE[sym] = steps

    if "pb_atr" in cfg_overrides:
        cfg.PULLBACK_ATR_RETRACE = float(cfg_overrides["pb_atr"])
    if "pb_wait" in cfg_overrides:
        cfg.PULLBACK_MAX_WAIT_BARS = int(cfg_overrides["pb_wait"])

    if "dir_regime" in cfg_overrides:
        cfg.DIRECTION_BIAS_REGIME = dict(cfg.DIRECTION_BIAS_REGIME)
        cur = dict(cfg.DIRECTION_BIAS_REGIME.get(sym, {}))
        for r, v in cfg_overrides["dir_regime"].items():
            cur[r] = v   # 'LONG'|'SHORT'|'BOTH'
        cfg.DIRECTION_BIAS_REGIME[sym] = cur

    # Now reload backtest module so it picks up new config
    import backtest.v5_backtest as bt
    importlib.reload(bt)

    # Optional VWAP disable monkey-patch (after reload)
    if cfg_overrides.get("vwap_disable"):
        import numpy as _np
        orig_compute = bt._compute_indicators
        def _wrap(df, icfg):
            out = orig_compute(df, icfg)
            if out is not None and "vwap" in out:
                out["vwap"] = _np.full(len(out["vwap"]), _np.nan, dtype=_np.float64)
            return out
        bt._compute_indicators = _wrap

    params = dict(params_extra or {})
    try:
        r = bt.backtest_symbol(sym, days=days, params=params, verbose=False)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[:400]}
    if not r:
        return {"ok": False, "error": "no_data"}
    return {
        "ok": True,
        "pnl": float(r.get("pnl", 0)),
        "pf": float(r.get("pf", 0)),
        "wr": float(r.get("wr", 0)),
        "n": int(r.get("trades", 0)),
        "dd": float(r.get("dd", 0)),
        "avg_r": float(r.get("avg_r", 0)),
        "avg_peak_r": float(r.get("avg_peak_r", 0)),
        "avg_giveback": float(r.get("avg_giveback", 0)),
    }


def _baseline_only():
    """Run live-config baseline (no overrides), 180d."""
    import config as cfg
    importlib.reload(cfg)
    import backtest.v5_backtest as bt
    importlib.reload(bt)
    r = bt.backtest_symbol(SYMBOL, days=TUNE_DAYS, verbose=False)
    if not r:
        return None
    return {
        "pnl": float(r.get("pnl", 0)),
        "pf": float(r.get("pf", 0)),
        "wr": float(r.get("wr", 0)),
        "n": int(r.get("trades", 0)),
        "dd": float(r.get("dd", 0)),
    }


# ── Sampling ──────────────────────────────────────────────────────────────
def _trial_to_overrides(trial):
    """Convert a trial dict (dim choices) to (cfg_overrides, params_extra)."""
    cfg_o = {
        "sl_mult": trial["sl"],
        "trail_steps": TRAIL_PROFILES[trial["trail"]],
        "pb_atr": trial["pb_atr"],
        "pb_wait": trial["pb_wait"],
        "vwap_disable": (trial["vwap"] == "0.0_disabled"),
        "dir_regime": {
            "trending": trial["dir_trend"],
            "volatile": trial["dir_vol"],
            # ranging/low_vol left as default ('BOTH' fallback)
            "ranging": "BOTH",
            "low_vol": "BOTH",
        },
    }
    mq = trial["mq"]
    params = {"min_quality": {"trending": mq, "ranging": mq, "volatile": mq, "low_vol": mq}}
    return cfg_o, params


def _job_runner(args):
    """Multiprocessing worker. args = (trial_id, trial_dict, days)."""
    trial_id, trial, days = args
    t0 = time.time()
    cfg_o, params = _trial_to_overrides(trial)
    res = _apply_overrides_and_bt(cfg_o, params, days)
    res["trial_id"] = trial_id
    res["trial"] = trial
    res["secs"] = round(time.time() - t0, 2)
    return res


def _wf_runner(args):
    """Walk-forward worker. args = (trial_id, trial_dict)."""
    trial_id, trial = args
    cfg_o, params = _trial_to_overrides(trial)
    folds = []
    for d in WF_FOLDS:
        r = _apply_overrides_and_bt(cfg_o, params, d)
        if not r.get("ok"):
            folds.append({"days": d, "pnl": 0.0, "pf": 0.0, "n": 0, "error": r.get("error")})
            continue
        folds.append({"days": d, "pnl": round(r["pnl"], 2), "pf": round(r["pf"], 2), "n": r["n"]})
    avg_pf = round(sum(f["pf"] for f in folds) / max(1, len(folds)), 3)
    pos = sum(1 for f in folds if f["pnl"] > 0)
    return {"trial_id": trial_id, "trial": trial,
            "wf": {"folds": folds, "avg_pf": avg_pf, "pos_folds": pos}}


def _sample_trial():
    """One uniformly-sampled trial."""
    return {
        "sl":         RNG.choice(SL_GRID),
        "trail":      RNG.choice(list(TRAIL_PROFILES)),
        "pb_atr":     RNG.choice(PB_ATR_GRID),
        "pb_wait":    RNG.choice(PB_WAIT_GRID),
        "vwap":       RNG.choice(VWAP_GRID),
        "mq":         RNG.choice(MQ_GRID),
        "dir_trend":  RNG.choice(DIR_GRID),
        "dir_vol":    RNG.choice(DIR_GRID),
    }


def _dedupe(trials):
    seen = set()
    out = []
    for t in trials:
        key = (t["sl"], t["trail"], t["pb_atr"], t["pb_wait"], t["vwap"], t["mq"],
               t["dir_trend"], t["dir_vol"])
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _neighborhood(base, k=8):
    """Generate k neighbour trials by perturbing 1-2 dims at a time."""
    out = []
    dims = ["sl", "trail", "pb_atr", "pb_wait", "vwap", "mq", "dir_trend", "dir_vol"]
    grids = {
        "sl": SL_GRID, "trail": list(TRAIL_PROFILES),
        "pb_atr": PB_ATR_GRID, "pb_wait": PB_WAIT_GRID,
        "vwap": VWAP_GRID, "mq": MQ_GRID,
        "dir_trend": DIR_GRID, "dir_vol": DIR_GRID,
    }
    for _ in range(k):
        cand = dict(base)
        # perturb 1 or 2 dims
        n_perturb = RNG.choice([1, 1, 2])
        for d in RNG.sample(dims, n_perturb):
            choices = [c for c in grids[d] if c != cand[d]]
            if choices:
                cand[d] = RNG.choice(choices)
        out.append(cand)
    return out


def main():
    t0 = time.time()
    print(f"\nETHUSD per-symbol tune  —  {datetime.now().isoformat(timespec='seconds')}")
    print(f"  Phase A target: 80 trials   Phase B: ~24   Phase C (WF): top-5")
    print(f"  Days: {TUNE_DAYS}    WF folds: {WF_FOLDS}")
    print(f"  Output: {OUT_JSON}")

    # ── 1. Baseline ────────────────────────────────────────────────
    print("\n[A0] Live-config baseline ...")
    base = _baseline_only()
    if not base:
        print("  ! Could not get baseline. Aborting.")
        return
    print(f"  baseline: n={base['n']}  PF={base['pf']:.2f}  PnL=${base['pnl']:+.2f}  WR={base['wr']:.1f}%  DD={base['dd']:.1f}%")

    workers = max(2, min(8, cpu_count() or 4))

    # ── 2. Phase A — coarse random sweep ───────────────────────────
    print("\n[A] Coarse random sweep ...")
    trials_a = _dedupe([_sample_trial() for _ in range(95)])[:80]
    print(f"  trials_a: {len(trials_a)}")
    jobs_a = [(i, t, TUNE_DAYS) for i, t in enumerate(trials_a)]
    res_a = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job_runner, jobs_a), 1):
            res_a.append(r)
            if i % 10 == 0 or i == len(jobs_a):
                pnl = r.get("pnl", 0) if r.get("ok") else 0
                print(f"  A {i:3d}/{len(jobs_a)} ({time.time()-t0:.0f}s)  last trial -> PnL=${pnl:+.2f}")

    ok_a = [r for r in res_a if r.get("ok") and r["n"] >= MIN_TRADES]
    ok_a.sort(key=lambda x: -x["pnl"])
    print(f"  Phase A: {len(res_a)} runs, {len(ok_a)} pass n>={MIN_TRADES}, top PnL=${ok_a[0]['pnl']:+.2f}" if ok_a else "  Phase A: none passed gates")

    # ── 3. Phase B — neighbourhood compose around top-10 ───────────
    print("\n[B] Neighborhood compose around top-10 of Phase A ...")
    top10 = ok_a[:10]
    trials_b = []
    for r in top10:
        trials_b += _neighborhood(r["trial"], k=3)
    # filter dupes vs phase-A set
    seen_keys = set()
    for r in res_a:
        t = r["trial"]
        seen_keys.add((t["sl"], t["trail"], t["pb_atr"], t["pb_wait"], t["vwap"], t["mq"],
                       t["dir_trend"], t["dir_vol"]))
    trials_b_filt = []
    for t in _dedupe(trials_b):
        k = (t["sl"], t["trail"], t["pb_atr"], t["pb_wait"], t["vwap"], t["mq"],
             t["dir_trend"], t["dir_vol"])
        if k in seen_keys:
            continue
        seen_keys.add(k)
        trials_b_filt.append(t)
    trials_b_filt = trials_b_filt[:24]
    print(f"  trials_b: {len(trials_b_filt)}")
    jobs_b = [(1000 + i, t, TUNE_DAYS) for i, t in enumerate(trials_b_filt)]
    res_b = []
    if jobs_b:
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_job_runner, jobs_b), 1):
                res_b.append(r)
                if i % 8 == 0 or i == len(jobs_b):
                    pnl = r.get("pnl", 0) if r.get("ok") else 0
                    print(f"  B {i:3d}/{len(jobs_b)} ({time.time()-t0:.0f}s)  last trial -> PnL=${pnl:+.2f}")

    # ── 4. Pick top-5 for WF ───────────────────────────────────────
    all_ok = [r for r in (res_a + res_b) if r.get("ok") and r["n"] >= MIN_TRADES and r["pf"] >= MIN_PF]
    all_ok.sort(key=lambda x: -x["pnl"])
    top5 = all_ok[:5]
    print(f"\n[C] WF top-5 (post gates: n>={MIN_TRADES}, PF>={MIN_PF}):")
    for i, r in enumerate(top5):
        t = r["trial"]
        print(f"  #{i+1}: PnL=${r['pnl']:+.2f}  PF={r['pf']:.2f}  n={r['n']}  DD={r['dd']:.1f}%   "
              f"SL={t['sl']} trail={t['trail']} pb={t['pb_atr']}/{t['pb_wait']} vwap={t['vwap']} mq={t['mq']} "
              f"dir(t/v)={t['dir_trend']}/{t['dir_vol']}")
    if not top5:
        print("  ! No candidates passed Phase A+B gates. Skipping WF.")

    wf_jobs = [(r["trial_id"], r["trial"]) for r in top5]
    wf_res = {}
    if wf_jobs:
        print(f"\n  WF: {len(wf_jobs)} candidates x {len(WF_FOLDS)} folds")
        with Pool(workers) as pool:
            for i, w in enumerate(pool.imap_unordered(_wf_runner, wf_jobs), 1):
                wf_res[w["trial_id"]] = w
                print(f"  WF {i}/{len(wf_jobs)} ({time.time()-t0:.0f}s)  "
                      f"avg_pf={w['wf']['avg_pf']:.2f}  pos={w['wf']['pos_folds']}/5")

    # ── 5. Apply gates → winner ────────────────────────────────────
    print("\n[D] Apply ship gates  (Δ >= $50, WF pos >= 3, WF avg_pf >= 1.2):")
    final = []
    for r in top5:
        wf = wf_res.get(r["trial_id"], {}).get("wf", None)
        if wf is None:
            continue
        delta = r["pnl"] - base["pnl"]
        wf_ok = wf["avg_pf"] >= MIN_WF_PF and wf["pos_folds"] >= MIN_WF_POS
        delta_ok = delta >= MIN_DELTA
        winner = bool(wf_ok and delta_ok)
        final.append({
            **r, "wf": wf, "delta": round(delta, 2),
            "wf_ok": bool(wf_ok), "delta_ok": bool(delta_ok), "winner": winner,
        })
        flag = "WINNER " if winner else "       "
        print(f"  {flag} Δ=${delta:+8.2f}  WF avg_pf={wf['avg_pf']:.2f} pos={wf['pos_folds']}/5  "
              f"SL={r['trial']['sl']} {r['trial']['trail']}  pb={r['trial']['pb_atr']}/{r['trial']['pb_wait']}  "
              f"vwap={r['trial']['vwap']}  mq={r['trial']['mq']}  dir={r['trial']['dir_trend']}/{r['trial']['dir_vol']}")

    final.sort(key=lambda x: -x["pnl"])
    winner = next((c for c in final if c["winner"]), None)

    # ── 6. JSON output ─────────────────────────────────────────────
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": SYMBOL,
        "tune_days": TUNE_DAYS,
        "wf_folds_days": WF_FOLDS,
        "rng_seed": 20260522,
        "dimensions": {
            "sl_grid": SL_GRID,
            "trail_profiles": list(TRAIL_PROFILES),
            "pb_atr_grid": PB_ATR_GRID,
            "pb_wait_grid": PB_WAIT_GRID,
            "vwap_grid": VWAP_GRID,
            "vwap_note": ("Live source v5_backtest.py:769 hardcodes the 0.5 multiplier. "
                          "READ-ONLY constraint means only binary toggle {disabled, default-0.5} is testable."),
            "mq_grid": MQ_GRID,
            "dir_grid_per_regime": DIR_GRID,
            "regimes_with_dir_bias": ["trending", "volatile"],
        },
        "trail_profile_steps": {k: v for k, v in TRAIL_PROFILES.items()},
        "ship_rule": {
            "min_delta_pnl": MIN_DELTA,
            "min_wf_avg_pf": MIN_WF_PF,
            "min_wf_pos_folds": MIN_WF_POS,
            "pre_wf_filter": {"min_trades": MIN_TRADES, "min_pf": MIN_PF},
        },
        "baseline": base,
        "phase_a": {
            "n_trials": len(res_a),
            "n_pass_n": len([r for r in res_a if r.get("ok") and r["n"] >= MIN_TRADES]),
            "trials": [
                {k: v for k, v in r.items() if k != "tb"} for r in res_a
            ],
        },
        "phase_b": {
            "n_trials": len(res_b),
            "trials": [
                {k: v for k, v in r.items() if k != "tb"} for r in res_b
            ],
        },
        "phase_c_wf": [
            {
                **{k: v for k, v in r.items() if k not in ("tb",)},
                "wf": r["wf"], "delta": r["delta"],
                "wf_ok": r["wf_ok"], "delta_ok": r["delta_ok"], "winner": r["winner"],
            }
            for r in final
        ],
        "winner": (
            {
                "sl_mult": winner["trial"]["sl"],
                "trail_name": winner["trial"]["trail"],
                "trail_steps": TRAIL_PROFILES[winner["trial"]["trail"]],
                "pb_atr_retrace": winner["trial"]["pb_atr"],
                "pb_max_wait_bars": winner["trial"]["pb_wait"],
                "vwap": winner["trial"]["vwap"],
                "min_quality_uniform": winner["trial"]["mq"],
                "direction_bias_regime": {
                    "trending": winner["trial"]["dir_trend"],
                    "volatile": winner["trial"]["dir_vol"],
                },
                "pnl": winner["pnl"],
                "delta_vs_baseline": winner["delta"],
                "pf": winner["pf"], "wr": winner["wr"], "n": winner["n"], "dd": winner["dd"],
                "wf_avg_pf": winner["wf"]["avg_pf"],
                "wf_pos_folds": winner["wf"]["pos_folds"],
                "wf_folds": winner["wf"]["folds"],
                "status": "WINNER",
            } if winner else
            {"status": "NO_WINNER",
             "reason": (
                 "Phase A+B produced 0 candidates passing pre-WF gates (n>=20, PF>=1.5)"
                 if not top5 else
                 "None of top-5 passed both ship gates (delta>=$50 AND WF pos>=3 AND WF avg_pf>=1.2)"
             ),
             "best_pnl_attempt": (top5[0]["pnl"] if top5 else None),
             "best_pnl_delta": (round(top5[0]["pnl"] - base["pnl"], 2) if top5 else None)}
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[E] Saved -> {OUT_JSON}  ({time.time() - t0:.0f}s total)")

    # ── 7. Markdown summary ────────────────────────────────────────
    md = [f"# ETHUSD per-symbol tune — {datetime.now().isoformat(timespec='seconds')}\n"]
    md.append(f"- Source repo: `{ROOT}`")
    md.append(f"- Backtest: `backtest_symbol('ETHUSD', days=180, params=P)`  (READ-ONLY)")
    md.append(f"- Tune days: **{TUNE_DAYS}**  •  WF folds: **{WF_FOLDS}**  •  RNG seed: 20260522")
    md.append(f"- Workers: {workers}  •  Elapsed: **{out['elapsed_seconds']}s**\n")
    md.append("## Dimensions explored")
    md.append(f"1. **SL ATR mult** ∈ {SL_GRID}")
    md.append(f"2. **Trail profile** ∈ {list(TRAIL_PROFILES)}")
    md.append(f"3. **Pullback ATR retrace** ∈ {PB_ATR_GRID}")
    md.append(f"4. **Pullback wait bars** ∈ {PB_WAIT_GRID}")
    md.append(f"5. **VWAP buffer** ∈ {VWAP_GRID}  *(see VWAP note in JSON)*")
    md.append(f"6. **min_quality** (uniform per regime) ∈ {MQ_GRID}")
    md.append(f"7. **direction_bias per regime** (trending/volatile) ∈ {DIR_GRID}\n")
    md.append("## Baseline (live config)")
    md.append(f"- n={base['n']}  PF={base['pf']:.2f}  PnL=${base['pnl']:+.2f}  WR={base['wr']:.1f}%  DD={base['dd']:.1f}%\n")
    md.append(f"## Phase A — coarse random sweep ({len(res_a)} trials)")
    if ok_a:
        md.append("Top 10 (by PnL):")
        md.append("")
        md.append("|  # |   PnL    |  PF  |  WR  | n  |  DD  | SL  | Trail | pb_atr/wait | VWAP | mq | dir(t/v) |")
        md.append("|----|----------|------|------|----|------|-----|-------|--------------|------|----|----------|")
        for i, r in enumerate(ok_a[:10]):
            t = r["trial"]
            md.append(f"| {i+1} | ${r['pnl']:+.2f} | {r['pf']:.2f} | {r['wr']:.1f}% | {r['n']} | {r['dd']:.1f}% | "
                      f"{t['sl']} | {t['trail']} | {t['pb_atr']}/{t['pb_wait']} | {t['vwap']} | {t['mq']} | "
                      f"{t['dir_trend']}/{t['dir_vol']} |")
        md.append("")
    md.append(f"## Phase B — neighborhood compose ({len(res_b)} trials)")
    ok_b = [r for r in res_b if r.get("ok") and r["n"] >= MIN_TRADES]
    ok_b.sort(key=lambda x: -x["pnl"])
    if ok_b:
        md.append("Top 5 (by PnL):")
        md.append("")
        md.append("|  # |   PnL    |  PF  |  WR  | n  |  DD  | SL  | Trail | pb_atr/wait | VWAP | mq | dir(t/v) |")
        md.append("|----|----------|------|------|----|------|-----|-------|--------------|------|----|----------|")
        for i, r in enumerate(ok_b[:5]):
            t = r["trial"]
            md.append(f"| {i+1} | ${r['pnl']:+.2f} | {r['pf']:.2f} | {r['wr']:.1f}% | {r['n']} | {r['dd']:.1f}% | "
                      f"{t['sl']} | {t['trail']} | {t['pb_atr']}/{t['pb_wait']} | {t['vwap']} | {t['mq']} | "
                      f"{t['dir_trend']}/{t['dir_vol']} |")
        md.append("")
    md.append(f"## Phase C — walk-forward (top-5)")
    md.append("")
    md.append("|  # |   PnL    |   Δ    |  PF  | n  | WF avg_pf | WF pos | delta_ok | wf_ok | WINNER |")
    md.append("|----|----------|--------|------|----|-----------|--------|----------|-------|--------|")
    for i, r in enumerate(final):
        flag = "YES" if r["winner"] else ""
        md.append(f"| {i+1} | ${r['pnl']:+.2f} | ${r['delta']:+.2f} | {r['pf']:.2f} | {r['n']} | "
                  f"{r['wf']['avg_pf']:.2f} | {r['wf']['pos_folds']}/5 | {r['delta_ok']} | {r['wf_ok']} | {flag} |")
    md.append("")
    md.append("## Winner")
    if winner:
        t = winner["trial"]
        md.append(f"- **SL ATR mult**: {t['sl']}")
        md.append(f"- **Trail profile**: {t['trail']}  →  steps {TRAIL_PROFILES[t['trail']]}")
        md.append(f"- **Pullback**: ATR={t['pb_atr']}  wait_bars={t['pb_wait']}")
        md.append(f"- **VWAP**: {t['vwap']}")
        md.append(f"- **min_quality** (uniform): {t['mq']}")
        md.append(f"- **direction_bias_regime**: trending={t['dir_trend']}, volatile={t['dir_vol']}, ranging/low_vol=BOTH")
        md.append(f"- **PnL**: ${winner['pnl']:+.2f}   Δvs baseline: ${winner['delta']:+.2f}")
        md.append(f"- **PF**: {winner['pf']:.2f}  WR: {winner['wr']:.1f}%  n: {winner['n']}  DD: {winner['dd']:.1f}%")
        md.append(f"- **WF**: avg_pf={winner['wf']['avg_pf']:.2f}  pos={winner['wf']['pos_folds']}/5")
        md.append(f"- Folds: {winner['wf']['folds']}")
    else:
        md.append(f"- **NO_WINNER** — {out['winner'].get('reason')}")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[F] MD     -> {OUT_MD}")


if __name__ == "__main__":
    main()
