"""US2000.r fine-tune runner.

Phase A (independent dim sweeps) + Phase B (top-2 combinatorial 64) + Phase C
(walk-forward validation on top-5). Read-only on source — all overrides via
in-memory patching of config / v5_backtest module / auto_tuned in worker.

Run: python3 -B us2000_runner.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT = Path("/Users/ashish/Documents/beast-trader")
OUT_DIR = ROOT / "per_symbol_tune_20260522"
OUT_JSON = OUT_DIR / "US2000.r.json"
OUT_MD   = OUT_DIR / "US2000.r.md"

SYMBOL = "US2000.r"
DAYS = 180

# Need to import only inside worker subprocess to keep parent fast.


def _bt_worker(args):
    """One backtest evaluation.

    args = (tag, days, overrides)
    overrides keys (all optional, all in-memory only):
      sl_mult:       float  → patches v5_backtest.SL_OVERRIDE[symbol]
      force_trail:   list   → passed via params['force_trail'] AND clears
                              v5_backtest.TRAIL_OVERRIDE_REGIME[symbol]
      pb_atr:        float  → config.PULLBACK_ATR_RETRACE
      pb_wait:       int    → config.PULLBACK_MAX_WAIT_BARS
      vwap_buf:      float  → patches in-memory v5_backtest source 0.5 literal
                              (None or 'disabled' to disable VWAP filter entirely)
      min_quality:   int    → patches config.SIGNAL_QUALITY_SYMBOL[symbol]
                              (uniform across regimes)
    """
    tag, days, ovr = args
    sys.path.insert(0, str(ROOT))

    # Patch config BEFORE importing v5_backtest if vwap requires source rewrite.
    import importlib, importlib.util

    import config as _cfg

    if "pb_atr" in ovr and ovr["pb_atr"] is not None:
        _cfg.PULLBACK_ATR_RETRACE = float(ovr["pb_atr"])
    if "pb_wait" in ovr and ovr["pb_wait"] is not None:
        _cfg.PULLBACK_MAX_WAIT_BARS = int(ovr["pb_wait"])
    if "min_quality" in ovr and ovr["min_quality"] is not None:
        mq = int(ovr["min_quality"])
        _cfg.SIGNAL_QUALITY_SYMBOL[SYMBOL] = {
            "trending": mq, "ranging": mq, "volatile": mq, "low_vol": mq,
        }

    vwap_buf = ovr.get("vwap_buf", None)

    # If VWAP buffer override given (or 'disabled') we need to load a patched
    # copy of v5_backtest.py source with the literal swapped. Default 0.5 — if
    # vwap_buf is None we just import normally.
    if vwap_buf is not None:
        src_path = ROOT / "backtest" / "v5_backtest.py"
        src = src_path.read_text()
        old = 'atr_buf = float(ind["at"][bi]) * 0.5'
        if vwap_buf == "disabled":
            new = 'atr_buf = float("inf")  # vwap disabled'
        else:
            new = f'atr_buf = float(ind["at"][bi]) * {float(vwap_buf)}'
        assert old in src, "VWAP literal not found"
        src2 = src.replace(old, new)
        # Force a fresh module so prior import doesn't shadow.
        spec = importlib.util.spec_from_file_location(
            f"_v5_patched_{os.getpid()}_{id(src2)}", src_path)
        mod = importlib.util.module_from_spec(spec)
        # Replace source via compile/exec rather than spec.loader to inject src2
        code = compile(src2, str(src_path), "exec")
        sys.modules[mod.__name__] = mod
        mod.__dict__["__file__"] = str(src_path)
        exec(code, mod.__dict__)
        v5 = mod
    else:
        import backtest.v5_backtest as v5
        # If we previously patched, reimport fresh would be expensive — but
        # within one process we want consistency. Pool=fresh process per task
        # if maxtasksperchild=1; but for speed we keep workers. So patches
        # accumulate; reset SL/regime explicitly.

    # SL override patch
    if "sl_mult" in ovr and ovr["sl_mult"] is not None:
        v5.SL_OVERRIDE[SYMBOL] = float(ovr["sl_mult"])

    # Clear regime trail override for this symbol so force_trail dominates.
    # Also clear regime SL override so sl_mult sweep is the only signal.
    if "force_trail" in ovr and ovr["force_trail"] is not None:
        if SYMBOL in v5.TRAIL_OVERRIDE_REGIME:
            v5.TRAIL_OVERRIDE_REGIME.pop(SYMBOL, None)
    if "sl_mult" in ovr and ovr["sl_mult"] is not None:
        v5.SL_OVERRIDE_REGIME.pop(SYMBOL, None)

    params = {}
    if "force_trail" in ovr and ovr["force_trail"] is not None:
        params["force_trail"] = ovr["force_trail"]

    t0 = time.time()
    try:
        r = v5.backtest_symbol(SYMBOL, days=days, params=params, verbose=False)
    except Exception as e:
        return tag, {"error": str(e), "took": time.time() - t0}
    if r is None:
        return tag, {"error": "no result", "took": time.time() - t0}
    out = {
        "tag": tag,
        "days": days,
        "trades": int(r.get("trades", 0)),
        "pf":  float(r.get("pf", 0.0) or 0.0),
        "pnl": float(r.get("pnl", 0.0) or 0.0),
        "wr":  float(r.get("wr", 0.0) or 0.0),
        "dd":  float(r.get("dd", 0.0) or 0.0),
        "avg_r": float(r.get("avg_r", 0.0) or 0.0),
        "took": round(time.time() - t0, 2),
    }
    return tag, out


# Trail profiles (mirror auto_tuned.py exactly)
_TIGHT_LOCK   = [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)]
_WIDE_RUNNER  = [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)]
_RANGE_TIGHT  = [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0)]
_TREND_LOOSE  = [(15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5), (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0)]
_AGGR_LOCK    = [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8), (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)]
_RUNNER_NO_BE = [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5), (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)]
_WIDE_RUNNER_BE07 = [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)]

# v5_backtest expects (R, param, type) tuples. auto_tuned.py uses (R, type, param) —
# we mirror exactly the auto_tuned format here. Use the live→bt converter inline.
def _to_bt(steps):
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


SL_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5]
PB_ATR_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
PB_WAIT_VALUES = [3, 4, 5, 6, 8]
VWAP_VALUES = ["disabled", 0.3, 0.5, 0.7, 1.0, 1.5]
MQ_VALUES = [28, 30, 33, 35, 38, 40, 43]


def run_pool(tasks, nproc):
    """Run a list of (tag, days, overrides) tasks via Pool.

    maxtasksperchild=1 to prevent in-process patch contamination — every BT
    runs in a fresh subprocess so SL_OVERRIDE / config / VWAP source patches
    can never leak between tasks.
    """
    with Pool(processes=nproc, maxtasksperchild=1) as pool:
        results = []
        for tag, out in pool.imap_unordered(_bt_worker, tasks, chunksize=1):
            results.append((tag, out))
            print(f"  [{tag}] pnl={out.get('pnl', 'ERR')} pf={out.get('pf', 'ERR')} n={out.get('trades', 'ERR')} t={out.get('took', '?')}s")
    return dict(results)


def main():
    t_start = time.time()
    nproc = max(1, min(8, cpu_count() - 2))
    print(f"Using {nproc} workers. Time budget 2h.")

    # ── Baseline ──
    print("\n=== BASELINE ===")
    base = run_pool([("baseline", DAYS, {})], 1)["baseline"]
    print(f"Baseline: pnl=${base['pnl']:.0f} pf={base['pf']:.2f} n={base['trades']}")
    baseline_pnl = base["pnl"]

    # ── Phase A: independent dim sweeps ──
    print("\n=== PHASE A — INDEPENDENT SWEEPS ===")

    # Dim 1: SL
    print("\n-- Dim 1: SL multiplier (16 values) --")
    sl_tasks = [(f"sl_{x}", DAYS, {"sl_mult": x}) for x in SL_VALUES]
    sl_res = run_pool(sl_tasks, nproc)
    sl_sorted = sorted(sl_res.items(), key=lambda kv: -kv[1].get("pnl", -1e9))
    sl_top3 = sl_sorted[:3]
    print(f"  SL top-3: {[(k, v['pnl']) for k, v in sl_top3]}")

    # Dim 2: Trail (alone) — uses current SL
    print("\n-- Dim 2: Trail profile (7) --")
    trail_tasks = [(f"trail_{n}", DAYS, {"force_trail": p}) for n, p in TRAIL_PROFILES.items()]
    trail_res = run_pool(trail_tasks, nproc)
    trail_sorted = sorted(trail_res.items(), key=lambda kv: -kv[1].get("pnl", -1e9))
    print(f"  Trail top-3: {[(k, v['pnl']) for k, v in trail_sorted[:3]]}")

    # Dim 3: Pullback ATR
    print("\n-- Dim 3: Pullback ATR (8) --")
    pb_atr_tasks = [(f"pb_atr_{x}", DAYS, {"pb_atr": x}) for x in PB_ATR_VALUES]
    pb_atr_res = run_pool(pb_atr_tasks, nproc)
    pb_atr_sorted = sorted(pb_atr_res.items(), key=lambda kv: -kv[1].get("pnl", -1e9))
    print(f"  PB ATR top-3: {[(k, v['pnl']) for k, v in pb_atr_sorted[:3]]}")

    # Dim 4: Pullback wait
    print("\n-- Dim 4: Pullback wait bars (5) --")
    pb_wait_tasks = [(f"pb_wait_{x}", DAYS, {"pb_wait": x}) for x in PB_WAIT_VALUES]
    pb_wait_res = run_pool(pb_wait_tasks, nproc)
    pb_wait_sorted = sorted(pb_wait_res.items(), key=lambda kv: -kv[1].get("pnl", -1e9))
    print(f"  PB Wait top-3: {[(k, v['pnl']) for k, v in pb_wait_sorted[:3]]}")

    # Dim 5: VWAP buffer
    print("\n-- Dim 5: VWAP buffer (6) --")
    vwap_tasks = [(f"vwap_{x}", DAYS, {"vwap_buf": x}) for x in VWAP_VALUES]
    vwap_res = run_pool(vwap_tasks, nproc)
    vwap_sorted = sorted(vwap_res.items(), key=lambda kv: -kv[1].get("pnl", -1e9))
    print(f"  VWAP top-3: {[(k, v['pnl']) for k, v in vwap_sorted[:3]]}")

    # Dim 6: min_quality
    print("\n-- Dim 6: min_quality (7) --")
    mq_tasks = [(f"mq_{x}", DAYS, {"min_quality": x}) for x in MQ_VALUES]
    mq_res = run_pool(mq_tasks, nproc)
    mq_sorted = sorted(mq_res.items(), key=lambda kv: -kv[1].get("pnl", -1e9))
    print(f"  MQ top-3: {[(k, v['pnl']) for k, v in mq_sorted[:3]]}")

    # Phase A summary
    phase_a = {
        "sl":      [{"tag": k, **v} for k, v in sl_sorted[:3]],
        "trail":   [{"tag": k, **v} for k, v in trail_sorted[:3]],
        "pb_atr":  [{"tag": k, **v} for k, v in pb_atr_sorted[:3]],
        "pb_wait": [{"tag": k, **v} for k, v in pb_wait_sorted[:3]],
        "vwap":    [{"tag": k, **v} for k, v in vwap_sorted[:3]],
        "mq":      [{"tag": k, **v} for k, v in mq_sorted[:3]],
    }

    # ── Phase B: top-2 from each dim → 2^6 = 64 combos ──
    print("\n=== PHASE B — TOP-2 COMBINATORIAL (64 BTs) ===")
    sl_top2  = [x[0] for x in sl_sorted[:2]]
    trail_top2 = [x[0] for x in trail_sorted[:2]]
    pb_atr_top2 = [x[0] for x in pb_atr_sorted[:2]]
    pb_wait_top2 = [x[0] for x in pb_wait_sorted[:2]]
    vwap_top2 = [x[0] for x in vwap_sorted[:2]]
    mq_top2 = [x[0] for x in mq_sorted[:2]]

    def _val(tag, prefix, cast):
        return cast(tag[len(prefix):])

    combo_tasks = []
    for sl_tag in sl_top2:
        for trail_tag in trail_top2:
            for pb_atr_tag in pb_atr_top2:
                for pb_wait_tag in pb_wait_top2:
                    for vwap_tag in vwap_top2:
                        for mq_tag in mq_top2:
                            sl_v = float(sl_tag.split("_")[1])
                            tr_v = trail_tag.split("_", 1)[1]
                            pa = float(pb_atr_tag.split("_")[2])
                            pw = int(pb_wait_tag.split("_")[2])
                            vw_raw = vwap_tag.split("_", 1)[1]
                            vw = vw_raw if vw_raw == "disabled" else float(vw_raw)
                            mq = int(mq_tag.split("_")[1])
                            tag = f"sl{sl_v}_tr{tr_v}_pa{pa}_pw{pw}_vw{vw}_mq{mq}"
                            combo_tasks.append((tag, DAYS, {
                                "sl_mult": sl_v,
                                "force_trail": TRAIL_PROFILES[tr_v],
                                "pb_atr": pa,
                                "pb_wait": pw,
                                "vwap_buf": vw,
                                "min_quality": mq,
                            }))
    print(f"Built {len(combo_tasks)} combo tasks")
    combo_res = run_pool(combo_tasks, nproc)
    combo_sorted = sorted(combo_res.items(), key=lambda kv: -kv[1].get("pnl", -1e9))
    print("\nTop 10 combos:")
    for k, v in combo_sorted[:10]:
        print(f"  {k}  pnl=${v['pnl']:.0f}  pf={v['pf']:.2f}  n={v['trades']}  delta=${v['pnl']-baseline_pnl:+.0f}")

    phase_b_top5 = []
    for tag, v in combo_sorted[:5]:
        # parse tag back to params
        parts = tag.split("_")
        cfg = {"tag": tag, "pnl": v["pnl"], "pf": v["pf"], "trades": v["trades"], "delta": v["pnl"] - baseline_pnl}
        phase_b_top5.append(cfg)

    # ── Phase C: WF validation top-5 across 5 windows ──
    print("\n=== PHASE C — WALK-FORWARD VALIDATION (top 5 × 5 folds) ===")
    wf_days = [60, 90, 120, 150, 180]
    wf_tasks = []
    # Rebuild override dicts for top-5
    top5_configs = []
    for tag, _v in combo_sorted[:5]:
        # parse
        # format: sl{v}_tr{name}_pa{v}_pw{v}_vw{v}_mq{v}
        # use the matching task overrides from combo_tasks (lookup by tag)
        ovr_for_tag = next(t[2] for t in combo_tasks if t[0] == tag)
        top5_configs.append((tag, ovr_for_tag))
        for d in wf_days:
            wf_tasks.append((f"{tag}@{d}d", d, ovr_for_tag))

    # Also baseline at each WF window for delta context
    for d in wf_days:
        wf_tasks.append((f"baseline@{d}d", d, {}))

    wf_res = run_pool(wf_tasks, nproc)

    # Compute folds positive: top5 pnl > baseline pnl at same window
    wf_summary = []
    for tag, ovr in top5_configs:
        folds = []
        for d in wf_days:
            cand = wf_res.get(f"{tag}@{d}d", {})
            base_d = wf_res.get(f"baseline@{d}d", {})
            c_pnl = cand.get("pnl", float("nan"))
            b_pnl = base_d.get("pnl", float("nan"))
            delta = c_pnl - b_pnl
            folds.append({"days": d, "cand_pnl": c_pnl, "base_pnl": b_pnl, "delta": delta,
                          "pf": cand.get("pf"), "n": cand.get("trades")})
        positive = sum(1 for f in folds if f["delta"] > 0)
        wf_summary.append({"tag": tag, "overrides": _serialize_ovr(ovr), "folds": folds,
                           "positive_folds": positive,
                           "full_180d_pnl": combo_res[tag]["pnl"],
                           "full_180d_delta": combo_res[tag]["pnl"] - baseline_pnl})

    print("\nWF results:")
    for s in wf_summary:
        deltas = [f"{f['delta']:+.0f}" for f in s["folds"]]
        print(f"  {s['tag']}  positive={s['positive_folds']}/5  deltas={deltas}  Δ180d=${s['full_180d_delta']:+.0f}")

    # Ship eligibility: Δ ≥ $100 AND positive folds ≥ 3
    eligible = [s for s in wf_summary
                if s["full_180d_delta"] >= 100 and s["positive_folds"] >= 3]
    if eligible:
        # pick the one with highest 180d delta + ties broken by folds
        winner = max(eligible, key=lambda s: (s["full_180d_delta"], s["positive_folds"]))
        recommendation = "SHIP"
    else:
        # pick best for hold/no-improvement framing
        winner = max(wf_summary, key=lambda s: (s["positive_folds"], s["full_180d_delta"]))
        # If best WF folds < 3, NO_IMPROVEMENT; else HOLD
        if winner["positive_folds"] < 3:
            recommendation = "NO_IMPROVEMENT"
        else:
            recommendation = "HOLD"

    # ── Save JSON ──
    out = {
        "symbol": SYMBOL,
        "baseline": {"trades": base["trades"], "pf": base["pf"], "pnl": base["pnl"]},
        "phase_a_winners": phase_a,
        "phase_b_top5": phase_b_top5,
        "phase_c_winner": {
            "tag": winner["tag"],
            "overrides": winner["overrides"],
            "wf_folds": winner["folds"],
            "positive_folds": winner["positive_folds"],
            "expected_180d_pnl": winner["full_180d_pnl"],
            "delta_vs_baseline": winner["full_180d_delta"],
        },
        "phase_c_all_top5": wf_summary,
        "recommendation": recommendation,
        "total_runtime_s": round(time.time() - t_start, 1),
        "total_backtests": 1 + 16 + 7 + 8 + 5 + 6 + 7 + len(combo_tasks) + len(wf_tasks),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}  (recommendation={recommendation})")

    # ── Markdown summary ──
    md_lines = [
        f"# US2000.r fine-tune — 2026-05-22",
        "",
        f"**Baseline:** {base['trades']} trades, PF {base['pf']:.2f}, PnL ${base['pnl']:.0f} / 180d.",
        f"**Best combo (Phase B 180d):** {combo_sorted[0][0]} → ${combo_sorted[0][1]['pnl']:.0f} (Δ ${combo_sorted[0][1]['pnl']-baseline_pnl:+.0f}).",
        f"**Phase C winner:** {winner['tag']} — positive folds {winner['positive_folds']}/5, expected 180d ${winner['full_180d_pnl']:.0f} (Δ ${winner['full_180d_delta']:+.0f}).",
        f"**Recommendation: {recommendation}**.",
        "",
        f"Total runtime {round((time.time()-t_start)/60,1)} min. Baseline 180d PnL is dominated by a Q1-window winning streak — recent 90d gives ${wf_res.get('baseline@90d',{}).get('pnl','?')}. WF validation uses 5 windows {wf_days} to cut overfit risk.",
        "",
    ]
    OUT_MD.write_text("\n".join(md_lines))
    print(f"Wrote {OUT_MD}")


def _serialize_ovr(ovr):
    """Make overrides JSON-friendly (trail list of tuples → list of lists)."""
    out = {}
    for k, v in ovr.items():
        if k == "force_trail":
            # Save profile name if matches one of TRAIL_PROFILES
            name = None
            for pn, pv in TRAIL_PROFILES.items():
                if pv == v:
                    name = pn
                    break
            out[k] = name or [list(t) for t in v]
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
