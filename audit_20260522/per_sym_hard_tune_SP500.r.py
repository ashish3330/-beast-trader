#!/usr/bin/env python3 -B
"""Per-symbol HARD-TUNE — SP500.r ONLY — 2026-05-23.

Mirror-aware BT (POST_BIG_WIN / LOSS_STREAK / PEAK_GIVEBACK / EarlyLossCut /
score-tier) — DEEP coordinate-descent sweep.

Baseline (SP500.r, live):
  SL=0.2, trail=_TIGHT_LOCK (all 4 regimes), mq={t:25,r:25,v:25,lv:25},
  pullback ATR=0.6 / wait=4, vwap buf=1.0, range filter (96, 1.0),
  dir_bias_regime={volatile: LONG}, toxic +{11},
  POST_BIG_WIN_COOLDOWN_SECS=10800, LOSS_STREAK_COOLDOWN_SECS=18000.

Phase A (independent dims):
  1. SL multiplier: {0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0}     — 9 BTs
  2. Trail profile: {_TIGHT_LOCK, _AGGR_LOCK, _RUNNER_NO_BE,
                    _WIDE_RUNNER, _RANGE_TIGHT, _TREND_LOOSE,
                    _WIDE_RUNNER_BE07}                                  — 7
  3. min_quality (all-regime uniform): {22,25,28,30,33,35,38,40}      — 8
  4. Pullback ATR: {0.4, 0.6, 0.8, 1.0, 1.2}                          — 5
  5. Pullback wait: {3, 5, 7}                                         — 3
  6. VWAP buffer: {0.0_off, 0.3, 0.5, 0.7, 1.0, 1.5}                  — 6
  7. POST_BIG_WIN_COOLDOWN_SECS: {1800, 3600, 5400, 7200, 10800}      — 5
     LOSS_STREAK_COOLDOWN_SECS:  {3600, 7200, 10800, 14400, 18000}    — 5
  8. Direction bias regime: per-regime {None, LONG, SHORT, BOTH}      — 4×4=16
                                                                       —————
                                                                Phase A: ≤72 BTs

Phase B: cartesian top-2 of each dim (cap 200) — best 10 by 180d PnL.

Phase C: 5-fold WF on top 3 from Phase B.

Ship rule: Δ ≥ $30 AND ≥3/5 folds positive AND avg PF ≥ 1.5.

Output:
  audit_20260522/per_sym_hard_tune_SP500.r.json
  audit_20260522/per_sym_hard_tune_SP500.r.md
"""

import json
import os
import sys
import time
import traceback
import itertools
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "audit_20260522"
OUT_JSON = OUT_DIR / "per_sym_hard_tune_SP500.r.json"
OUT_MD   = OUT_DIR / "per_sym_hard_tune_SP500.r.md"
LOG_FILE = OUT_DIR / "per_sym_hard_tune_SP500.r.log"

SYMBOL = "SP500.r"
DAYS = 180
WF_NUM_FOLDS = 5
WF_FOLD_DAYS = 36  # disjoint 36d folds for SP500.r (15y cache available)

# Ship gates
MIN_DELTA = 30.0
WF_MIN_POS = 3
WF_AVG_PF_MIN = 1.5

PHASE_B_MAX = 200
PHASE_C_TOP = 8           # expand WF set to top-8 combos
PHASE_C_SINGLES = 5       # WF top-5 single-knob winners too (for ship-as-1-knob)

# Risk override — user wants live to match 2%
RISK_PCT = 2.0

# ─────────────────────────────────────────────────────────────────────
# SWEEP GRIDS
# ─────────────────────────────────────────────────────────────────────
SL_GRID        = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
TRAIL_PROFILES = [
    "_TIGHT_LOCK",
    "_AGGR_LOCK",
    "_RUNNER_NO_BE",
    "_WIDE_RUNNER",
    "_RANGE_TIGHT",
    "_TREND_LOOSE",
    "_WIDE_RUNNER_BE07",
]
MIN_Q_GRID     = [22, 25, 28, 30, 33, 35, 38, 40]
PB_ATR_GRID    = [0.4, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_GRID   = [3, 5, 7]
VWAP_GRID      = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]
POST_BIG_WIN_GRID = [1800, 3600, 5400, 7200, 10800]
LOSS_STREAK_GRID  = [3600, 7200, 10800, 14400, 18000]

# Per-regime direction biases — 4 settings × 4 regimes
REGIMES = ["trending", "ranging", "volatile", "low_vol"]
DIR_BIAS_VALS = [None, "LONG", "SHORT", "BOTH"]

# Range filter sweep (optional, but live already has (96, 1.0); offer disable + alternates)
RF_GRID = [None, "live", (48, 0.5), (72, 0.7), (96, 1.0), (120, 1.5)]


# ─────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────
def _log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────────────────────────────
# WORKER — single backtest with in-process overlays.
# ─────────────────────────────────────────────────────────────────────
def _bt_one(args):
    """
    args = dict with keys:
      sl, trail_name, min_q (int|None=keep baseline-all-regime),
      pb_atr, pb_wait, vwap_buf,
      post_big_win_secs, loss_streak_secs,
      dir_bias (dict {regime: 'LONG'|'SHORT'|'BOTH'} or None=keep baseline),
      range_filter (None=keep baseline 'live', False=disable, tuple=override),
      fold_id (1..5 or None)
    Returns dict {trades, pf, wr, pnl, dd, err?}.
    """
    try:
        import importlib as _il
        import auto_tuned as _at
        _il.reload(_at)
        import config as _cfg
        _il.reload(_cfg)

        # COOLDOWN overrides BEFORE bt import (BT imports config inside function)
        if args.get("post_big_win_secs") is not None:
            _cfg.POST_BIG_WIN_COOLDOWN_SECS = int(args["post_big_win_secs"])

        # Pullback / VWAP overrides — patch live config so BT reads them
        if args.get("pb_atr") is not None:
            d = dict(getattr(_cfg, "PULLBACK_ATR_RETRACE_PER_SYMBOL", {}))
            d[SYMBOL] = float(args["pb_atr"])
            _cfg.PULLBACK_ATR_RETRACE_PER_SYMBOL = d
        if args.get("pb_wait") is not None:
            d = dict(getattr(_cfg, "PULLBACK_MAX_WAIT_BARS_PER_SYMBOL", {}))
            d[SYMBOL] = int(args["pb_wait"])
            _cfg.PULLBACK_MAX_WAIT_BARS_PER_SYMBOL = d
        if args.get("vwap_buf") is not None:
            d = dict(getattr(_cfg, "VWAP_BUFFER_PER_SYMBOL", {}))
            d[SYMBOL] = float(args["vwap_buf"])
            _cfg.VWAP_BUFFER_PER_SYMBOL = d

        # Now reload BT — it imports config at function-call time, but module-
        # level globals (TRAIL_OVERRIDE_REGIME, SL_OVERRIDE, _DIR_BIAS_REGIME_STR,
        # TOXIC_HOURS) are read at import. Reload so it picks up our patches.
        import backtest.v5_backtest as _bt
        _il.reload(_bt)

        # SL overlay
        if args.get("sl") is not None:
            _bt.SL_OVERRIDE[SYMBOL] = float(args["sl"])
            # Clear any per-regime SL override that would shadow our setting.
            if SYMBOL in _bt.SL_OVERRIDE_REGIME:
                _bt.SL_OVERRIDE_REGIME.pop(SYMBOL, None)

        # Trail profile overlay (across all regimes — TRAIL_OVERRIDE_REGIME
        # beats TRAIL_OVERRIDE in BT trade-time code).
        if args.get("trail_name") is not None:
            tname = args["trail_name"]
            tprof = getattr(_at, tname, None)
            if tprof is None:
                return {"err": f"trail_profile_not_found:{tname}"}
            _bt.TRAIL_OVERRIDE_REGIME[SYMBOL] = {r: tprof for r in REGIMES}

        # min_quality (uniform across regimes)
        if args.get("min_q") is not None:
            mq = int(args["min_q"])
            cur = dict(_cfg.SIGNAL_QUALITY_SYMBOL.get(SYMBOL, {}))
            for r in REGIMES:
                cur[r] = mq
            _cfg.SIGNAL_QUALITY_SYMBOL[SYMBOL] = cur

        # Direction bias overlay — full replace at regime level
        if args.get("dir_bias") is not None:
            # `dir_bias` is dict {regime: 'LONG'|'SHORT'|'BOTH'} (BOTH = no bias cell)
            # We REPLACE the entire SP500.r entry (not merge) so explicit None=remove.
            new_cell = {}
            for r, v in args["dir_bias"].items():
                if v is None:
                    continue
                # Use 'BOTH' to explicitly allow both directions in regime.
                new_cell[r] = v
            if new_cell:
                _bt._DIR_BIAS_REGIME_STR[SYMBOL] = new_cell
            elif SYMBOL in _bt._DIR_BIAS_REGIME_STR:
                _bt._DIR_BIAS_REGIME_STR.pop(SYMBOL, None)

        # Toxic hours — keep live baseline (+{11}). Not swept here per task.
        # (Toxic hour was a 'NEW' input but baseline is single-element; we
        # cover it implicitly by keeping live behaviour.)

        # Range filter
        rf = args.get("range_filter", "live")
        params = {"risk_pct": RISK_PCT}
        if rf is None or rf == "live":
            # Keep live behaviour — auto-inject from auto_tuned.RANGE_FILTER_PARAMS_AUTO
            pass
        elif rf is False:
            params["range_filter_enabled"] = False
        elif isinstance(rf, (tuple, list)):
            params["range_filter_enabled"] = True
            params["range_lookback"] = int(rf[0])
            params["range_buffer_atr"] = float(rf[1])

        # Loss-streak cooldown — patched via params dict (BT reads p["loss_streak_cooldown_bars"])
        if args.get("loss_streak_secs") is not None:
            bars = max(1, int(round(float(args["loss_streak_secs"]) / 3600.0)))
            params["loss_streak_cooldown_bars"] = bars

        # 5. Fold slicing for WF — disjoint 36d folds counting from end
        fold_id = args.get("fold_id")
        if fold_id is not None:
            import pandas as pd
            orig_load = _bt.load_data
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
            _bt.load_data = load_data_fold
            r = _bt.backtest_symbol(SYMBOL, days=None, params=params, verbose=False)
        else:
            r = _bt.backtest_symbol(SYMBOL, days=DAYS, params=params, verbose=False)

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


def _baseline_args():
    """Live-config baseline — no overlays, just risk_pct=2.0."""
    return {
        "sl": None,
        "trail_name": None,
        "min_q": None,
        "pb_atr": None,
        "pb_wait": None,
        "vwap_buf": None,
        "post_big_win_secs": None,
        "loss_streak_secs": None,
        "dir_bias": None,
        "range_filter": "live",
        "fold_id": None,
    }


# ─────────────────────────────────────────────────────────────────────
# PHASE A: single-dim sweeps
# ─────────────────────────────────────────────────────────────────────
def _phase_a(workers, baseline):
    """Run each dim's sweep independently. Return dict {dim: [results]}."""
    jobs_meta = []  # list of (dim, label, args)

    # Dim 1: SL
    for sl in SL_GRID:
        a = _baseline_args(); a["sl"] = sl
        jobs_meta.append(("sl", f"sl={sl}", a))

    # Dim 2: trail
    for tn in TRAIL_PROFILES:
        a = _baseline_args(); a["trail_name"] = tn
        jobs_meta.append(("trail", f"trail={tn}", a))

    # Dim 3: min_q
    for mq in MIN_Q_GRID:
        a = _baseline_args(); a["min_q"] = mq
        jobs_meta.append(("min_q", f"min_q={mq}", a))

    # Dim 4: pullback ATR
    for pb in PB_ATR_GRID:
        a = _baseline_args(); a["pb_atr"] = pb
        jobs_meta.append(("pb_atr", f"pb_atr={pb}", a))

    # Dim 5: pullback wait
    for pw in PB_WAIT_GRID:
        a = _baseline_args(); a["pb_wait"] = pw
        jobs_meta.append(("pb_wait", f"pb_wait={pw}", a))

    # Dim 6: VWAP buf
    for vw in VWAP_GRID:
        a = _baseline_args(); a["vwap_buf"] = vw
        jobs_meta.append(("vwap", f"vwap={vw}", a))

    # Dim 7a: POST_BIG_WIN cooldown
    for cd in POST_BIG_WIN_GRID:
        a = _baseline_args(); a["post_big_win_secs"] = cd
        jobs_meta.append(("post_big_win", f"pbw_cd={cd}", a))

    # Dim 7b: LOSS_STREAK cooldown
    for cd in LOSS_STREAK_GRID:
        a = _baseline_args(); a["loss_streak_secs"] = cd
        jobs_meta.append(("loss_streak", f"ls_cd={cd}", a))

    # Dim 8: per-regime direction bias — each regime, each value (only non-baseline)
    # Baseline is {volatile:LONG}; we test variants per regime.
    for regime in REGIMES:
        for val in DIR_BIAS_VALS:
            # Build full bias cell: start from baseline {volatile: LONG} then override regime
            cell = {"volatile": "LONG"}  # mirror live default
            if val is None:
                # Remove this regime's bias
                cell.pop(regime, None)
            else:
                cell[regime] = val
            a = _baseline_args(); a["dir_bias"] = cell
            jobs_meta.append(("dir_bias", f"dir_bias[{regime}]={val}", a))

    _log(f"Phase A: {len(jobs_meta)} single-dim candidates")
    t0 = time.time()
    with Pool(workers) as pool:
        raw = list(pool.imap(_bt_one, [m[2] for m in jobs_meta]))
    _log(f"Phase A done in {time.time()-t0:.1f}s")

    by_dim = {}
    all_results = []
    for (dim, label, args), res in zip(jobs_meta, raw):
        if "err" in res:
            _log(f"  ERR {label}: {res['err'][:120]}")
            continue
        rec = {
            "dim": dim, "label": label, "args": args,
            "trades": res["trades"], "pf": res["pf"], "wr": res["wr"],
            "pnl": res["pnl"], "dd": res["dd"],
            "delta": res["pnl"] - baseline["pnl"],
        }
        by_dim.setdefault(dim, []).append(rec)
        all_results.append(rec)
    for dim in by_dim:
        by_dim[dim].sort(key=lambda x: x["pnl"], reverse=True)

    return by_dim, all_results


def _phase_b_combos(by_dim, baseline, max_combos=PHASE_B_MAX, top_per_dim=3):
    """Cartesian top-N of each dim that has positive-delta winners."""
    picks = {}
    for dim, lst in by_dim.items():
        positives = [r for r in lst if r["delta"] > 0]
        if positives:
            picks[dim] = positives[:top_per_dim]
        else:
            picks[dim] = [None]  # 'None' = no overlay for this dim

    dims = list(picks.keys())
    combos = []
    for combo in itertools.product(*[picks[d] for d in dims]):
        if all(c is None for c in combo):
            continue
        # Merge args
        merged = _baseline_args()
        label_parts = []
        for c in combo:
            if c is None:
                continue
            for k, v in c["args"].items():
                if v is not None and merged.get(k) is None:
                    merged[k] = v
                elif v is not None and merged.get(k) is not None and k == "dir_bias":
                    # Merge dir_bias cells
                    if isinstance(merged[k], dict) and isinstance(v, dict):
                        new = dict(merged[k])
                        new.update(v)
                        merged[k] = new
            label_parts.append(c["label"])
        combos.append({"label": " + ".join(label_parts), "args": merged})

    # Cap
    if len(combos) > max_combos:
        _log(f"  Phase B: {len(combos)} combos > cap {max_combos}; truncating.")
        combos = combos[:max_combos]
    return combos


def _phase_b_run(combos, workers, baseline):
    if not combos:
        return []
    _log(f"Phase B: running {len(combos)} combos")
    t0 = time.time()
    with Pool(workers) as pool:
        raw = list(pool.imap(_bt_one, [c["args"] for c in combos]))
    _log(f"Phase B done in {time.time()-t0:.1f}s")
    out = []
    for c, res in zip(combos, raw):
        if "err" in res:
            continue
        out.append({
            "label": c["label"], "args": c["args"],
            "trades": res["trades"], "pf": res["pf"], "wr": res["wr"],
            "pnl": res["pnl"], "dd": res["dd"],
            "delta": res["pnl"] - baseline["pnl"],
        })
    out.sort(key=lambda x: x["pnl"], reverse=True)
    return out


def _phase_c_wf(top_n, workers, baseline):
    if not top_n:
        return []
    _log(f"Phase C: WF on top-{len(top_n)} candidates")
    jobs = []
    for cand in top_n:
        for fold in range(1, WF_NUM_FOLDS + 1):
            a = dict(cand["args"])
            a["fold_id"] = fold
            jobs.append(a)
    t0 = time.time()
    with Pool(workers) as pool:
        raw = list(pool.imap(_bt_one, jobs))
    _log(f"Phase C done in {time.time()-t0:.1f}s")

    out = []
    for i, cand in enumerate(top_n):
        folds = []
        for fold in range(1, WF_NUM_FOLDS + 1):
            r = raw[i * WF_NUM_FOLDS + (fold - 1)]
            if "err" in r:
                folds.append({"fold": fold, "trades": 0, "pf": 0, "pnl": 0, "wr": 0,
                              "err": r["err"][:120]})
            else:
                folds.append({
                    "fold": fold,
                    "trades": r["trades"],
                    "pf":     round(r["pf"], 2),
                    "pnl":    round(r["pnl"], 2),
                    "wr":     round(r["wr"], 1),
                })
        pos = sum(1 for fr in folds if fr.get("pnl", 0) > 0)
        avg_pf = round(sum(fr.get("pf", 0) for fr in folds) / max(1, len(folds)), 2)
        total = round(sum(fr.get("pnl", 0) for fr in folds), 2)
        out.append({
            **cand,
            "wf_folds": folds,
            "wf_pos":    pos,
            "wf_avg_pf": avg_pf,
            "wf_total":  total,
        })
    return out


def _serial_args(a):
    """Convert args dict to JSON-safe form (range_filter tuple → list)."""
    if a is None:
        return None
    out = {}
    for k, v in a.items():
        if isinstance(v, (set, frozenset)):
            out[k] = sorted(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _write_md(final):
    lines = []
    lines.append(f"# SP500.r hard-tune — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    base = final["baseline"]
    lines.append("## Baseline (live config, risk_pct=2.0, 180d)")
    lines.append(f"- trades={base['trades']}  PF={base['pf']:.2f}  PnL=${base['pnl']:+,.0f}  "
                 f"DD={base['dd']:.2f}%  WR={base['wr']:.1f}%")
    lines.append("")

    lines.append("## Phase A — single-dim top-3 per dim")
    by_dim = final["phase_a_by_dim"]
    for dim in sorted(by_dim.keys()):
        lines.append(f"### {dim}")
        lines.append("| label | trades | PF | PnL | Δ | DD |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in by_dim[dim][:3]:
            lines.append(f"| {r['label']} | {r['trades']} | {r['pf']:.2f} | "
                         f"${r['pnl']:+,.0f} | ${r['delta']:+,.0f} | {r['dd']:.2f}% |")
        lines.append("")

    lines.append("## Phase B — top-10 combos")
    lines.append("| label | trades | PF | PnL | Δ | DD |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in final["phase_b_top10"]:
        lines.append(f"| {r['label']} | {r['trades']} | {r['pf']:.2f} | "
                     f"${r['pnl']:+,.0f} | ${r['delta']:+,.0f} | {r['dd']:.2f}% |")
    lines.append("")

    lines.append("## Phase C — 5-fold WF on top-3")
    for r in final["phase_c_wf"]:
        lines.append(f"### {r['label']}")
        lines.append(f"- Full 180d: trades={r['trades']} PF={r['pf']:.2f} PnL=${r['pnl']:+,.0f} Δ=${r['delta']:+,.0f}")
        lines.append(f"- WF pos folds: {r['wf_pos']}/{WF_NUM_FOLDS}  avg_pf={r['wf_avg_pf']}  total_wf_pnl=${r['wf_total']:+,.0f}")
        lines.append("| fold | trades | PF | PnL | WR |")
        lines.append("|---|---:|---:|---:|---:|")
        for fr in r["wf_folds"]:
            err = fr.get("err", "")
            lines.append(f"| {fr['fold']} | {fr['trades']} | {fr['pf']} | "
                         f"${fr['pnl']:+,.0f} | {fr['wr']}% |"
                         + (f"  err: {err}" if err else ""))
        lines.append("")

    lines.append("## Ship decision")
    sd = final["ship_decision"]
    lines.append(f"- **{sd['verdict']}**")
    lines.append(f"- gates: Δ≥${MIN_DELTA}, WF pos ≥{WF_MIN_POS}/{WF_NUM_FOLDS}, avg PF ≥{WF_AVG_PF_MIN}")
    if sd.get("winner"):
        w = sd["winner"]
        lines.append(f"- winner: `{w['label']}`")
        lines.append(f"- Δ=${w['delta']:+,.0f}  WF {w['wf_pos']}/{WF_NUM_FOLDS}  avg_pf={w['wf_avg_pf']}")
        lines.append("")
        lines.append("### Winner overlay (apply to auto_tuned.py / config.py)")
        lines.append("```json")
        lines.append(json.dumps(_serial_args(w["args"]), indent=2))
        lines.append("```")
    else:
        lines.append("- no candidate passed gates → KEEP BASELINE")
    lines.append("")
    lines.append(f"_elapsed: {final['_meta']['elapsed_sec']}s  total BTs: {final['_meta']['total_bts']}_")

    OUT_MD.write_text("\n".join(lines))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")
    t_start = time.time()
    workers = max(2, (os.cpu_count() or 4) - 2)
    _log(f"SP500.r hard-tune starting — workers={workers}")

    bt_count = 0

    # Baseline
    _log("Running baseline (live config, risk_pct=2.0, 180d)")
    with Pool(1) as pool:
        baseline_res = pool.apply(_bt_one, (_baseline_args(),))
    if "err" in baseline_res:
        _log(f"BASELINE FAILED: {baseline_res['err']}")
        return
    bt_count += 1
    baseline = baseline_res
    _log(f"  baseline: trades={baseline['trades']} PF={baseline['pf']:.2f} "
         f"PnL=${baseline['pnl']:+,.0f} DD={baseline['dd']:.2f}%  WR={baseline['wr']:.1f}%")

    # Phase A
    by_dim, all_results = _phase_a(workers, baseline)
    bt_count += sum(len(v) for v in by_dim.values())
    _log("Phase A top-3 per dim:")
    for dim in sorted(by_dim.keys()):
        for r in by_dim[dim][:3]:
            _log(f"  [{dim:14s}] {r['label']:35s} trades={r['trades']:3d} "
                 f"PF={r['pf']:5.2f} PnL=${r['pnl']:+9,.0f}  Δ=${r['delta']:+9,.0f}")

    # Phase B
    combos = _phase_b_combos(by_dim, baseline)
    _log(f"Phase B: built {len(combos)} combos")
    phase_b_results = _phase_b_run(combos, workers, baseline)
    bt_count += len(phase_b_results)
    _log("Phase B top-10:")
    for r in phase_b_results[:10]:
        _log(f"  {r['label']:80s} trades={r['trades']:3d} PF={r['pf']:5.2f} "
             f"PnL=${r['pnl']:+9,.0f}  Δ=${r['delta']:+9,.0f}")

    # Phase C — WF on top-N from Phase B + top single-knob winners (so a 1-knob change can ship)
    top_for_wf = list(phase_b_results[:PHASE_C_TOP]) if phase_b_results else []
    # Add top single-knob winners (deduped by label)
    seen_labels = {t["label"] for t in top_for_wf}
    if all_results:
        all_results.sort(key=lambda x: x["pnl"], reverse=True)
        added_singles = 0
        for r in all_results:
            if r["delta"] <= 0 or added_singles >= PHASE_C_SINGLES:
                continue
            if r["label"] in seen_labels:
                continue
            top_for_wf.append({"label": r["label"], "args": r["args"],
                               "trades": r["trades"], "pf": r["pf"], "wr": r["wr"],
                               "pnl": r["pnl"], "dd": r["dd"], "delta": r["delta"]})
            seen_labels.add(r["label"])
            added_singles += 1

    wf_results = _phase_c_wf(top_for_wf, workers, baseline) if top_for_wf else []
    bt_count += len(wf_results) * WF_NUM_FOLDS

    # Ship decision
    candidates_for_ship = []
    for r in wf_results:
        passes = (r["delta"] >= MIN_DELTA
                  and r["wf_pos"] >= WF_MIN_POS
                  and r["wf_avg_pf"] >= WF_AVG_PF_MIN)
        r["ship"] = passes
        if passes:
            candidates_for_ship.append(r)
    candidates_for_ship.sort(key=lambda x: (x["wf_pos"], x["pnl"]), reverse=True)
    winner = candidates_for_ship[0] if candidates_for_ship else None

    # Assemble JSON
    final = {
        "_meta": {
            "ts": datetime.now().isoformat(),
            "symbol": SYMBOL,
            "days": DAYS,
            "wf_num_folds": WF_NUM_FOLDS,
            "wf_fold_days": WF_FOLD_DAYS,
            "ship_min_delta": MIN_DELTA,
            "ship_wf_min_pos": WF_MIN_POS,
            "ship_avg_pf_min": WF_AVG_PF_MIN,
            "risk_pct": RISK_PCT,
            "elapsed_sec": round(time.time() - t_start, 1),
            "total_bts": bt_count,
            "phase_b_cap": PHASE_B_MAX,
            "phase_c_top": PHASE_C_TOP,
        },
        "baseline": baseline,
        "phase_a_by_dim": {
            dim: [{**{k: v for k, v in r.items() if k != "args"},
                   "args": _serial_args(r["args"])}
                  for r in lst]
            for dim, lst in by_dim.items()
        },
        "phase_b_top10": [{**{k: v for k, v in r.items() if k != "args"},
                           "args": _serial_args(r["args"])}
                          for r in phase_b_results[:10]],
        "phase_c_wf": [{**{k: v for k, v in r.items() if k != "args"},
                        "args": _serial_args(r["args"])}
                       for r in wf_results],
        "ship_decision": {
            "verdict": "SHIP" if winner else "KEEP_BASELINE",
            "winner": ({**{k: v for k, v in winner.items() if k != "args"},
                        "args": _serial_args(winner["args"])} if winner else None),
            "rule": (f"Δ ≥ ${MIN_DELTA} AND WF pos ≥ {WF_MIN_POS}/{WF_NUM_FOLDS} "
                     f"AND avg PF ≥ {WF_AVG_PF_MIN}"),
        },
    }

    OUT_JSON.write_text(json.dumps(final, indent=2, default=str))
    _write_md(final)
    _log(f"JSON: {OUT_JSON}")
    _log(f"MD:   {OUT_MD}")
    _log(f"ALL DONE — {bt_count} BTs in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
