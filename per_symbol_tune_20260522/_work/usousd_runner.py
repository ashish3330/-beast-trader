"""USOUSD fine-tune runner — 2026-05-22.

Spec:
  Symbol: USOUSD (WTI crude oil)
  Baseline: ~$14,798 / 180d (PF 7.18, n=350, WR 75%, DD 5.3%)
  Dimensions:
    1) SL ATR mult:   {0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}
    2) Trail profile: 7 named (TIGHT_LOCK, WIDE_RUNNER, RANGE_TIGHT,
                                TREND_LOOSE, AGGR_LOCK, RUNNER_NO_BE,
                                WIDE_RUNNER_BE07)
    3) Pullback ATR:  {0.4, 0.5, 0.6, 0.8, 1.0, 1.2}
    4) Pullback wait: {3, 4, 5, 6, 8}
    5) VWAP buffer:   {disabled, 0.3, 0.5, 0.7, 1.0}
    6) min_quality:   {28, 30, 33, 35, 38, 40}

Phases:
  A — independent 1D sweeps     (10+7+6+5+5+6 = 39 BTs)
  B — top-2 × top-2 × top-2 × top-2 × top-2 × top-4 = 128 BTs
  C — walk-forward on top-5 with 5 disjoint 36d folds (5×5 = 25 BTs)
      + per-fold baseline (5 BTs)

Ship criterion: Δ_PnL ≥ +$100 AND WF positive folds ≥ 3/5.

Read-only: no repo files mutated. All overrides via in-memory monkey-patching
(config, v5_backtest module attrs, in-memory source recompile for VWAP).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT = Path("/Users/ashish/Documents/beast-trader")
OUT_DIR = ROOT / "per_symbol_tune_20260522"
OUT_JSON = OUT_DIR / "USOUSD.json"
OUT_MD   = OUT_DIR / "USOUSD.md"

SYMBOL = "USOUSD"
DAYS = 180

# Disjoint walk-forward: 5 sequential 36d folds spanning the most recent 180d.
WF_NUM_FOLDS = 5
WF_FOLD_DAYS = 36


def _bt_worker(args):
    """One backtest evaluation, all overrides in-memory.

    args = (tag, fold_id_or_none, overrides)
      fold_id_or_none: int 1..5 for WF window (1=oldest, 5=most recent), else None
      overrides keys (all optional):
        sl_mult:     float
        force_trail: list[(R,param,type)] backtest-format
        pb_atr:      float (config.PULLBACK_ATR_RETRACE)
        pb_wait:     int   (config.PULLBACK_MAX_WAIT_BARS)
        vwap_buf:    'disabled' or float (in-memory source rewrite)
        min_quality: int   (applied uniformly across regimes)
    """
    tag, fold_id, ovr = args
    sys.path.insert(0, str(ROOT))

    import importlib
    import importlib.util

    import config as _cfg

    if ovr.get("pb_atr") is not None:
        _cfg.PULLBACK_ATR_RETRACE = float(ovr["pb_atr"])
    if ovr.get("pb_wait") is not None:
        _cfg.PULLBACK_MAX_WAIT_BARS = int(ovr["pb_wait"])
    if ovr.get("min_quality") is not None:
        mq = int(ovr["min_quality"])
        _cfg.SIGNAL_QUALITY_SYMBOL[SYMBOL] = {
            "trending": mq, "ranging": mq, "volatile": mq, "low_vol": mq,
        }

    vwap_buf = ovr.get("vwap_buf", None)

    if vwap_buf is not None:
        src_path = ROOT / "backtest" / "v5_backtest.py"
        src = src_path.read_text()
        old = 'atr_buf = float(ind["at"][bi]) * 0.5'
        if vwap_buf == "disabled":
            # Sentinel makes the LONG/SHORT condition impossible, so the
            # filter becomes inert without breaking control flow.
            new = 'atr_buf = float("inf")  # vwap disabled'
        else:
            new = f'atr_buf = float(ind["at"][bi]) * {float(vwap_buf)}'
        if old not in src:
            return tag, {"error": "vwap literal not found", "fold": fold_id}
        src2 = src.replace(old, new)
        modname = f"_v5_patched_{os.getpid()}_{abs(hash(src2)) % 100000}"
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(modname, loader=None))
        mod.__file__ = str(src_path)
        sys.modules[modname] = mod
        code = compile(src2, str(src_path), "exec")
        try:
            exec(code, mod.__dict__)
        except Exception as e:
            return tag, {"error": f"exec patched src: {e}\n{traceback.format_exc()[:300]}",
                         "fold": fold_id}
        v5 = mod
    else:
        import backtest.v5_backtest as v5
        importlib.reload(v5)  # pick up any config patches done in this worker

    # SL override patch (set or clear)
    if ovr.get("sl_mult") is not None:
        v5.SL_OVERRIDE[SYMBOL] = float(ovr["sl_mult"])
        # Clear regime SL override so the global SL drives everything
        if SYMBOL in v5.SL_OVERRIDE_REGIME:
            v5.SL_OVERRIDE_REGIME.pop(SYMBOL, None)

    if ovr.get("force_trail") is not None:
        # Clear regime trail override so force_trail dominates all regimes
        if SYMBOL in v5.TRAIL_OVERRIDE_REGIME:
            v5.TRAIL_OVERRIDE_REGIME.pop(SYMBOL, None)

    params = {}
    if ovr.get("force_trail") is not None:
        params["force_trail"] = ovr["force_trail"]

    # Walk-forward window: monkey-patch load_data
    if fold_id is not None:
        import pandas as pd
        orig_load = v5.load_data
        fold_n = int(fold_id)
        num = WF_NUM_FOLDS
        fold_d = WF_FOLD_DAYS

        def load_data_fold(sym, _ignored_days=None):
            df = orig_load(sym, days=None)  # full data
            if df is None or df.empty:
                return df
            end = df["time"].max()
            # fold 5 → (end - 36d, end]; fold 1 → oldest
            offset_end = (num - fold_n) * fold_d
            offset_start = offset_end + fold_d
            t_end = end - pd.Timedelta(days=offset_end)
            t_start = end - pd.Timedelta(days=offset_start)
            df2 = df[(df["time"] > t_start) & (df["time"] <= t_end)].reset_index(drop=True)
            return df2

        v5.load_data = load_data_fold

    t0 = time.time()
    try:
        r = v5.backtest_symbol(SYMBOL, days=(None if fold_id else DAYS),
                               params=params, verbose=False)
    except Exception as e:
        return tag, {"error": str(e), "fold": fold_id, "took": time.time() - t0,
                     "tb": traceback.format_exc()[:300]}
    if r is None:
        return tag, {"error": "no result", "fold": fold_id, "took": time.time() - t0}
    out = {
        "tag": tag,
        "fold": fold_id,
        "trades": int(r.get("trades", 0)),
        "pf":     float(r.get("pf", 0.0) or 0.0),
        "pnl":    float(r.get("pnl", 0.0) or 0.0),
        "wr":     float(r.get("wr", 0.0) or 0.0),
        "dd":     float(r.get("dd", 0.0) or 0.0),
        "avg_r":  float(r.get("avg_r", 0.0) or 0.0),
        "took":   round(time.time() - t0, 2),
    }
    return tag, out


# ── Trail profiles — mirror auto_tuned.py (live → backtest tuple-order) ──
_TIGHT_LOCK   = [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)]
_WIDE_RUNNER  = [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)]
_RANGE_TIGHT  = [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0)]
_TREND_LOOSE  = [(15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5), (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0)]
_AGGR_LOCK    = [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8), (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)]
_RUNNER_NO_BE = [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5), (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)]
_WIDE_RUNNER_BE07 = [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)]


def _to_bt(steps):
    """Convert (R, type, param) → (R, param, type)."""
    out = []
    for tup in steps:
        if len(tup) == 3:
            r, t, p = tup
            out.append((r, p, t))
    return out


TRAIL_PROFILES = {
    "TIGHT_LOCK":       _to_bt(_TIGHT_LOCK),
    "WIDE_RUNNER":      _to_bt(_WIDE_RUNNER),
    "RANGE_TIGHT":      _to_bt(_RANGE_TIGHT),
    "TREND_LOOSE":      _to_bt(_TREND_LOOSE),
    "AGGR_LOCK":        _to_bt(_AGGR_LOCK),
    "RUNNER_NO_BE":     _to_bt(_RUNNER_NO_BE),
    "WIDE_RUNNER_BE07": _to_bt(_WIDE_RUNNER_BE07),
}


# ── Grids (user spec verbatim) ──
SL_VALUES      = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
TRAIL_NAMES    = list(TRAIL_PROFILES)  # 7 named profiles
PB_ATR_VALUES  = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_VALUES = [3, 4, 5, 6, 8]
VWAP_VALUES    = ["disabled", 0.3, 0.5, 0.7, 1.0]
MQ_VALUES      = [28, 30, 33, 35, 38, 40]


def run_pool(tasks, nproc, label=""):
    """Run (tag, fold, overrides) tasks via Pool, return dict tag→result."""
    print(f"  [{label}] dispatching {len(tasks)} tasks across {nproc} workers")
    t0 = time.time()
    with Pool(processes=nproc) as pool:
        results = {}
        done = 0
        for tag, out in pool.imap_unordered(_bt_worker, tasks, chunksize=1):
            results[tag] = out
            done += 1
            if "error" in out:
                print(f"    [{done}/{len(tasks)}] {tag}: ERR {out['error'][:80]}")
            elif done <= 5 or done % 20 == 0 or done == len(tasks):
                print(f"    [{done}/{len(tasks)}] {tag}: pnl=${out.get('pnl',0):+.0f} "
                      f"pf={out.get('pf',0):.2f} n={out.get('trades',0)} "
                      f"t={out.get('took','?')}s")
    print(f"  [{label}] done in {time.time()-t0:.1f}s")
    return results


def _serialize_ovr(ovr):
    """Make overrides JSON-friendly (trail list of tuples → name lookup)."""
    out = {}
    for k, v in ovr.items():
        if k == "force_trail":
            name = None
            for pn, pv in TRAIL_PROFILES.items():
                if pv == v:
                    name = pn
                    break
            out[k] = name or [list(t) for t in v]
        else:
            out[k] = v
    return out


def main():
    t_start = time.time()
    nproc = max(2, min(8, (cpu_count() or 4) - 2))
    print(f"USOUSD fine-tune. nproc={nproc}. Budget 2h.")
    print(f"Output: {OUT_JSON}")

    # ── Baseline (live config, no overrides) ──
    print("\n=== BASELINE ===")
    base_res = run_pool([("baseline", None, {})], 1, label="baseline")
    base = base_res["baseline"]
    if "error" in base:
        print("BASELINE FAILED:", base)
        return
    baseline_pnl = base["pnl"]
    print(f"Baseline 180d: pnl=${baseline_pnl:+.0f}  pf={base['pf']:.2f}  "
          f"n={base['trades']}  wr={base['wr']:.1f}%  dd={base['dd']:.2f}%")

    # ── Phase A: independent 1D sweeps ──
    print("\n=== PHASE A — INDEPENDENT 1D SWEEPS ===")

    # Each dim swept while OTHER dims stay at current live config defaults.
    # SL
    sl_tasks = [(f"sl_{x}", None, {"sl_mult": x}) for x in SL_VALUES]
    sl_res = run_pool(sl_tasks, nproc, label="A:SL")
    sl_ok = {k: v for k, v in sl_res.items() if "error" not in v}
    sl_sorted = sorted(sl_ok.items(), key=lambda kv: -kv[1]["pnl"])
    print(f"  SL top-3: " + ", ".join(f"{k}=${v['pnl']:+.0f}" for k, v in sl_sorted[:3]))

    # Trail
    trail_tasks = [(f"trail_{n}", None, {"force_trail": p}) for n, p in TRAIL_PROFILES.items()]
    trail_res = run_pool(trail_tasks, nproc, label="A:Trail")
    trail_ok = {k: v for k, v in trail_res.items() if "error" not in v}
    trail_sorted = sorted(trail_ok.items(), key=lambda kv: -kv[1]["pnl"])
    print(f"  Trail top-3: " + ", ".join(f"{k}=${v['pnl']:+.0f}" for k, v in trail_sorted[:3]))

    # Pullback ATR
    pb_atr_tasks = [(f"pb_atr_{x}", None, {"pb_atr": x}) for x in PB_ATR_VALUES]
    pb_atr_res = run_pool(pb_atr_tasks, nproc, label="A:PBatr")
    pb_atr_ok = {k: v for k, v in pb_atr_res.items() if "error" not in v}
    pb_atr_sorted = sorted(pb_atr_ok.items(), key=lambda kv: -kv[1]["pnl"])
    print(f"  PB ATR top-3: " + ", ".join(f"{k}=${v['pnl']:+.0f}" for k, v in pb_atr_sorted[:3]))

    # Pullback wait
    pb_wait_tasks = [(f"pb_wait_{x}", None, {"pb_wait": x}) for x in PB_WAIT_VALUES]
    pb_wait_res = run_pool(pb_wait_tasks, nproc, label="A:PBwait")
    pb_wait_ok = {k: v for k, v in pb_wait_res.items() if "error" not in v}
    pb_wait_sorted = sorted(pb_wait_ok.items(), key=lambda kv: -kv[1]["pnl"])
    print(f"  PB Wait top-3: " + ", ".join(f"{k}=${v['pnl']:+.0f}" for k, v in pb_wait_sorted[:3]))

    # VWAP buffer
    vwap_tasks = [(f"vwap_{x}", None, {"vwap_buf": x}) for x in VWAP_VALUES]
    vwap_res = run_pool(vwap_tasks, nproc, label="A:VWAP")
    vwap_ok = {k: v for k, v in vwap_res.items() if "error" not in v}
    vwap_sorted = sorted(vwap_ok.items(), key=lambda kv: -kv[1]["pnl"])
    print(f"  VWAP top-3: " + ", ".join(f"{k}=${v['pnl']:+.0f}" for k, v in vwap_sorted[:3]))

    # min_quality
    mq_tasks = [(f"mq_{x}", None, {"min_quality": x}) for x in MQ_VALUES]
    mq_res = run_pool(mq_tasks, nproc, label="A:MQ")
    mq_ok = {k: v for k, v in mq_res.items() if "error" not in v}
    mq_sorted = sorted(mq_ok.items(), key=lambda kv: -kv[1]["pnl"])
    print(f"  MQ top-3: " + ", ".join(f"{k}=${v['pnl']:+.0f}" for k, v in mq_sorted[:3]))

    phase_a = {
        "sl":      [{"tag": k, **v} for k, v in sl_sorted[:5]],
        "trail":   [{"tag": k, **v} for k, v in trail_sorted[:5]],
        "pb_atr":  [{"tag": k, **v} for k, v in pb_atr_sorted[:5]],
        "pb_wait": [{"tag": k, **v} for k, v in pb_wait_sorted[:5]],
        "vwap":    [{"tag": k, **v} for k, v in vwap_sorted[:5]],
        "mq":      [{"tag": k, **v} for k, v in mq_sorted[:5]],
    }

    # ── Phase B: 128 combos = top-4 × top-2 × top-2 × top-2 × top-2 × top-2 ──
    # SL gets top-4 (most-impactful dim historically). 4·2·2·2·2·2 = 128.
    print("\n=== PHASE B — COMBINATORIAL (128 BTs) ===")
    sl_top    = [x[0] for x in sl_sorted[:4]]
    trail_top = [x[0] for x in trail_sorted[:2]]
    pa_top    = [x[0] for x in pb_atr_sorted[:2]]
    pw_top    = [x[0] for x in pb_wait_sorted[:2]]
    vw_top    = [x[0] for x in vwap_sorted[:2]]
    mq_top    = [x[0] for x in mq_sorted[:2]]
    print(f"  Phase B grid: SL{len(sl_top)} × Trail{len(trail_top)} × PBatr{len(pa_top)} "
          f"× PBwait{len(pw_top)} × VWAP{len(vw_top)} × MQ{len(mq_top)} "
          f"= {len(sl_top)*len(trail_top)*len(pa_top)*len(pw_top)*len(vw_top)*len(mq_top)}")

    combo_tasks = []
    combo_ovr_by_tag = {}
    for sl_tag in sl_top:
        sl_v = float(sl_tag.split("_")[1])
        for trail_tag in trail_top:
            tr_n = trail_tag.split("_", 1)[1]
            for pa_tag in pa_top:
                pa_v = float(pa_tag.split("_")[2])
                for pw_tag in pw_top:
                    pw_v = int(pw_tag.split("_")[2])
                    for vw_tag in vw_top:
                        vw_raw = vw_tag.split("_", 1)[1]
                        vw_v = vw_raw if vw_raw == "disabled" else float(vw_raw)
                        for mq_tag in mq_top:
                            mq_v = int(mq_tag.split("_")[1])
                            tag = (f"sl{sl_v}_tr{tr_n}_pa{pa_v}_pw{pw_v}"
                                   f"_vw{vw_v}_mq{mq_v}")
                            ovr = {
                                "sl_mult":     sl_v,
                                "force_trail": TRAIL_PROFILES[tr_n],
                                "pb_atr":      pa_v,
                                "pb_wait":     pw_v,
                                "vwap_buf":    vw_v,
                                "min_quality": mq_v,
                            }
                            combo_tasks.append((tag, None, ovr))
                            combo_ovr_by_tag[tag] = ovr

    combo_res = run_pool(combo_tasks, nproc, label="B:combo")
    combo_ok = {k: v for k, v in combo_res.items() if "error" not in v}
    combo_sorted = sorted(combo_ok.items(), key=lambda kv: -kv[1]["pnl"])

    print("\n  Phase B top-10:")
    for k, v in combo_sorted[:10]:
        delta = v["pnl"] - baseline_pnl
        print(f"    {k}  pnl=${v['pnl']:+.0f} (Δ${delta:+.0f})  pf={v['pf']:.2f}  "
              f"n={v['trades']}  dd={v['dd']:.2f}%")

    phase_b_top10 = [{"tag": k, **v, "delta": v["pnl"] - baseline_pnl,
                      "overrides": _serialize_ovr(combo_ovr_by_tag[k])}
                     for k, v in combo_sorted[:10]]

    # ── Phase C: walk-forward on top-5 with 5 disjoint 36d folds ──
    print("\n=== PHASE C — WALK-FORWARD (top-5 × 5 disjoint folds) ===")
    top5_tags = [k for k, _ in combo_sorted[:5]]
    wf_tasks = []
    for tag in top5_tags:
        ovr = combo_ovr_by_tag[tag]
        for fold in range(1, WF_NUM_FOLDS + 1):
            wf_tasks.append((f"{tag}@fold{fold}", fold, ovr))
    # Per-fold baseline for delta context
    for fold in range(1, WF_NUM_FOLDS + 1):
        wf_tasks.append((f"baseline@fold{fold}", fold, {}))

    wf_res = run_pool(wf_tasks, nproc, label="C:WF")

    # Assemble WF summary
    wf_summary = []
    for tag in top5_tags:
        folds = []
        for f in range(1, WF_NUM_FOLDS + 1):
            cand = wf_res.get(f"{tag}@fold{f}", {})
            base_f = wf_res.get(f"baseline@fold{f}", {})
            folds.append({
                "fold": f,
                "cand_pnl": cand.get("pnl"),
                "cand_pf":  cand.get("pf"),
                "cand_n":   cand.get("trades"),
                "base_pnl": base_f.get("pnl"),
                "base_pf":  base_f.get("pf"),
                "delta":    (cand.get("pnl", 0) - base_f.get("pnl", 0))
                            if (cand.get("pnl") is not None and base_f.get("pnl") is not None) else None,
            })
        # User spec: WF ≥ 3/5 positive — count folds with cand_pnl > 0
        positive = sum(1 for fr in folds if fr.get("cand_pnl") is not None and fr["cand_pnl"] > 0)
        # Also track "beats baseline" for context
        beats_base = sum(1 for fr in folds if fr.get("delta") is not None and fr["delta"] > 0)
        full_180d = combo_ok[tag]["pnl"]
        wf_summary.append({
            "tag": tag,
            "overrides": _serialize_ovr(combo_ovr_by_tag[tag]),
            "folds": folds,
            "positive_folds": positive,
            "beats_baseline_folds": beats_base,
            "full_180d_pnl": full_180d,
            "full_180d_delta": full_180d - baseline_pnl,
            "full_180d_pf":   combo_ok[tag]["pf"],
            "full_180d_n":    combo_ok[tag]["trades"],
            "full_180d_dd":   combo_ok[tag]["dd"],
        })

    print("\n  WF results:")
    for s in wf_summary:
        deltas = ", ".join(f"{f['cand_pnl']:+.0f}" if f.get("cand_pnl") is not None else "ERR"
                           for f in s["folds"])
        print(f"    {s['tag']}  WF +{s['positive_folds']}/5  beats={s['beats_baseline_folds']}/5  "
              f"180d=${s['full_180d_pnl']:+.0f} (Δ${s['full_180d_delta']:+.0f})  folds_pnl=[{deltas}]")

    # ── Ship decision ──
    SHIP_DELTA = 100.0
    SHIP_WF    = 3
    eligible = [s for s in wf_summary
                if s["full_180d_delta"] >= SHIP_DELTA and s["positive_folds"] >= SHIP_WF]
    if eligible:
        winner = max(eligible, key=lambda s: (s["full_180d_delta"], s["positive_folds"]))
        recommendation = "SHIP"
    else:
        winner = max(wf_summary, key=lambda s: (s["positive_folds"], s["full_180d_delta"]))
        if winner["positive_folds"] < SHIP_WF:
            recommendation = "NO_IMPROVEMENT"
        else:
            recommendation = "HOLD"

    elapsed = time.time() - t_start
    total_bts = 1 + len(sl_tasks) + len(trail_tasks) + len(pb_atr_tasks) + \
                len(pb_wait_tasks) + len(vwap_tasks) + len(mq_tasks) + \
                len(combo_tasks) + len(wf_tasks)

    # ── Save JSON ──
    out = {
        "_meta": {
            "symbol": SYMBOL,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "days": DAYS,
            "wf_num_folds": WF_NUM_FOLDS,
            "wf_fold_days": WF_FOLD_DAYS,
            "total_backtests": total_bts,
            "runtime_sec": round(elapsed, 1),
            "ship_criterion": {"min_delta_pnl": SHIP_DELTA, "min_wf_positive_folds": SHIP_WF},
            "grids": {
                "sl": SL_VALUES,
                "trail": list(TRAIL_PROFILES),
                "pb_atr": PB_ATR_VALUES,
                "pb_wait": PB_WAIT_VALUES,
                "vwap": VWAP_VALUES,
                "min_quality": MQ_VALUES,
            },
        },
        "baseline": {
            "trades": base["trades"], "pf": base["pf"], "pnl": base["pnl"],
            "wr": base["wr"], "dd": base["dd"], "avg_r": base["avg_r"],
        },
        "phase_a_top5_per_dim": phase_a,
        "phase_b_top10": phase_b_top10,
        "phase_c_wf_top5": wf_summary,
        "winner": {
            "tag": winner["tag"],
            "overrides": winner["overrides"],
            "wf_folds": winner["folds"],
            "positive_folds": winner["positive_folds"],
            "expected_180d_pnl": winner["full_180d_pnl"],
            "delta_vs_baseline": winner["full_180d_delta"],
            "full_180d_pf": winner["full_180d_pf"],
            "full_180d_n":  winner["full_180d_n"],
            "full_180d_dd": winner["full_180d_dd"],
        },
        "recommendation": recommendation,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")

    # ── Markdown summary ──
    md_lines = [
        f"# USOUSD fine-tune — 2026-05-22",
        "",
        f"**Symbol:** USOUSD (WTI crude oil)  ",
        f"**Baseline (live config, 180d):** trades={base['trades']}, "
        f"PF={base['pf']:.2f}, PnL=${base['pnl']:+.0f}, WR={base['wr']:.1f}%, "
        f"DD={base['dd']:.2f}%.",
        f"**Total backtests:** {total_bts} in {elapsed/60:.1f} min "
        f"({nproc} workers, in-memory monkey-patch).",
        "",
        f"## Recommendation: **{recommendation}**",
        "",
        f"- Winner tag: `{winner['tag']}`",
        f"- Overrides: `{winner['overrides']}`",
        f"- WF positive folds: {winner['positive_folds']}/5  (beats baseline {winner['beats_baseline_folds']}/5)",
        f"- Full-180d PnL: ${winner['full_180d_pnl']:+.0f} (Δ ${winner['full_180d_delta']:+.0f} vs baseline)",
        f"- Full-180d PF: {winner['full_180d_pf']:.2f}, n={winner['full_180d_n']}, DD={winner['full_180d_dd']:.2f}%",
        "",
        f"Ship criteria: Δ_PnL ≥ +${int(SHIP_DELTA)} AND positive WF folds ≥ {SHIP_WF}/5.",
        "",
        "## Phase A — independent 1D sweeps (top per dim)",
        "",
        f"- SL top-3: {[(k, round(v['pnl'])) for k, v in sl_sorted[:3]]}",
        f"- Trail top-3: {[(k, round(v['pnl'])) for k, v in trail_sorted[:3]]}",
        f"- PB ATR top-3: {[(k, round(v['pnl'])) for k, v in pb_atr_sorted[:3]]}",
        f"- PB Wait top-3: {[(k, round(v['pnl'])) for k, v in pb_wait_sorted[:3]]}",
        f"- VWAP top-3: {[(k, round(v['pnl'])) for k, v in vwap_sorted[:3]]}",
        f"- MQ top-3: {[(k, round(v['pnl'])) for k, v in mq_sorted[:3]]}",
        "",
        "## Phase B — top-5 combos (full 180d)",
        "",
        "| tag | pnl | Δ | pf | n | dd% |",
        "|-----|----:|--:|---:|--:|----:|",
    ]
    for k, v in combo_sorted[:5]:
        md_lines.append(
            f"| `{k}` | ${v['pnl']:+.0f} | ${v['pnl']-baseline_pnl:+.0f} | "
            f"{v['pf']:.2f} | {v['trades']} | {v['dd']:.2f} |"
        )
    md_lines.extend([
        "",
        "## Phase C — walk-forward on top-5 (5 disjoint 36d folds, fold-5 most recent)",
        "",
    ])
    for s in wf_summary:
        md_lines.append(f"### `{s['tag']}` — WF +{s['positive_folds']}/5, "
                        f"180d Δ ${s['full_180d_delta']:+.0f}")
        md_lines.append("")
        md_lines.append("| fold | cand_pnl | base_pnl | delta | pf | n |")
        md_lines.append("|-----:|---------:|---------:|------:|---:|--:|")
        for fr in s["folds"]:
            cp = fr.get("cand_pnl"); bp = fr.get("base_pnl"); dl = fr.get("delta")
            cpf = fr.get("cand_pf"); cn = fr.get("cand_n")
            md_lines.append(
                f"| {fr['fold']} | "
                f"{'$%+.0f'%cp if cp is not None else 'ERR'} | "
                f"{'$%+.0f'%bp if bp is not None else 'ERR'} | "
                f"{'$%+.0f'%dl if dl is not None else 'ERR'} | "
                f"{cpf:.2f if cpf else 0:.2f} | "
                f"{cn if cn is not None else '?'} |"
            )
        md_lines.append("")

    OUT_MD.write_text("\n".join(md_lines))
    print(f"Wrote {OUT_MD}")
    print(f"\nTotal: {total_bts} backtests in {elapsed/60:.1f} min. Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
