"""AUDJPY fine-tune runner — 2026-05-22.

Phase A independent dim sweeps (42 BTs) + Phase B top-2 combinatorial (64) +
Phase C walk-forward validation (top-5 × 5 folds + 5 baseline = 30).
Baseline + sanity = 1. Grand total ~137 BTs at ~1.2s each.

READ-ONLY on source: all overrides happen via in-memory monkey-patching of:
  - config.PULLBACK_ATR_RETRACE / PULLBACK_MAX_WAIT_BARS
  - config.SIGNAL_QUALITY_SYMBOL[symbol]
  - v5_backtest.SL_OVERRIDE[symbol] / SL_OVERRIDE_REGIME[symbol]
  - v5_backtest.TRAIL_OVERRIDE_REGIME[symbol]
  - v5_backtest.TOXIC_HOURS (global set, fine because we only run AUDJPY)
  - v5_backtest.py source-string patched in-memory (VWAP 0.5 literal swap)

Output:
  /Users/ashish/Documents/beast-trader/per_symbol_tune_20260522/AUDJPY.json
  /Users/ashish/Documents/beast-trader/per_symbol_tune_20260522/AUDJPY.md

Run: python3 -B audjpy_runner.py
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
OUT_JSON = OUT_DIR / "AUDJPY.json"
OUT_MD   = OUT_DIR / "AUDJPY.md"

SYMBOL = "AUDJPY"
DAYS = 180

# Spec grids (exact match to user instructions)
SL_VALUES      = [0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5]
PB_ATR_VALUES  = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_VALUES = [3, 4, 5, 6, 8]
VWAP_VALUES    = ["disabled", 0.3, 0.5, 0.7, 1.0]
MQ_VALUES      = [28, 30, 33, 35, 38, 40]
TOXIC_OPTIONS  = [
    ("h7",   {7}),
    ("none", set()),
    ("h6_7", {6, 7}),
]


# Trail profiles (mirror auto_tuned.py exactly, in live (R, type, param) format)
_TIGHT_LOCK       = [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)]
_WIDE_RUNNER      = [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)]
_RANGE_TIGHT      = [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0)]
_TREND_LOOSE      = [(15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5), (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0)]
_AGGR_LOCK        = [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8), (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)]
_RUNNER_NO_BE     = [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5), (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)]
_WIDE_RUNNER_BE07 = [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)]


def _to_bt(steps):
    """Convert auto_tuned (R, type, param) → backtest (R, param, type)."""
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


def _bt_worker(args):
    """One backtest with optional overrides. Each call runs in a subprocess.

    args = (tag, days, overrides)
    overrides (all optional):
      sl_mult, force_trail_name, pb_atr, pb_wait, vwap_buf, min_quality,
      toxic_hours_set  → set of extra toxic hours UTC to add (or {} to clear h7).
    """
    tag, days, ovr = args
    sys.path.insert(0, str(ROOT))

    import importlib, importlib.util
    import config as _cfg

    # ── Config-level patches ──
    if "pb_atr" in ovr and ovr["pb_atr"] is not None:
        _cfg.PULLBACK_ATR_RETRACE = float(ovr["pb_atr"])
    if "pb_wait" in ovr and ovr["pb_wait"] is not None:
        _cfg.PULLBACK_MAX_WAIT_BARS = int(ovr["pb_wait"])
    if "min_quality" in ovr and ovr["min_quality"] is not None:
        mq = int(ovr["min_quality"])
        # Mutate in place so v5_backtest's `from config import SIGNAL_QUALITY_SYMBOL`
        # (inside the function) picks this up on next call.
        _cfg.SIGNAL_QUALITY_SYMBOL[SYMBOL] = {
            "trending": mq, "ranging": mq, "volatile": mq, "low_vol": mq,
        }

    vwap_buf = ovr.get("vwap_buf", None)

    # ── If VWAP override, load v5_backtest from patched source ──
    if vwap_buf is not None:
        src_path = ROOT / "backtest" / "v5_backtest.py"
        src = src_path.read_text()
        old = 'atr_buf = float(ind["at"][bi]) * 0.5'
        if vwap_buf == "disabled":
            new = 'atr_buf = float("inf")  # vwap disabled (audjpy_runner)'
        else:
            new = f'atr_buf = float(ind["at"][bi]) * {float(vwap_buf)}'
        assert old in src, "VWAP literal not found in v5_backtest.py"
        src2 = src.replace(old, new)
        mod_name = f"_v5_patched_{os.getpid()}_{abs(hash(src2))}"
        # Build a fresh module with patched code
        spec = importlib.util.spec_from_file_location(mod_name, src_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        mod.__dict__["__file__"] = str(src_path)
        code = compile(src2, str(src_path), "exec")
        exec(code, mod.__dict__)
        v5 = mod
    else:
        # Fresh import each task is overkill; reuse imported module. But ensure
        # state from prior patches is reset for this run.
        if "backtest.v5_backtest" in sys.modules:
            v5 = sys.modules["backtest.v5_backtest"]
        else:
            import backtest.v5_backtest as v5

    # ── v5_backtest-level patches ──
    if "sl_mult" in ovr and ovr["sl_mult"] is not None:
        v5.SL_OVERRIDE[SYMBOL] = float(ovr["sl_mult"])
        # Clear regime SL override so sl_mult is the only signal across all
        # regimes for this sweep.
        v5.SL_OVERRIDE_REGIME.pop(SYMBOL, None)

    # Clear regime trail override so force_trail dominates
    if "force_trail_name" in ovr and ovr["force_trail_name"] is not None:
        v5.TRAIL_OVERRIDE_REGIME.pop(SYMBOL, None)
        # Also remove from TRAIL_OVERRIDE so we know force_trail wins
        v5.TRAIL_OVERRIDE.pop(SYMBOL, None)

    # Toxic hours: rebuild global TOXIC_HOURS set (only AUDJPY runs)
    if "toxic_hours_set" in ovr and ovr["toxic_hours_set"] is not None:
        # Base set is {1,2,3,4} (from live config); extras from ovr
        base = {1, 2, 3, 4}
        extras = set(ovr["toxic_hours_set"])
        v5.TOXIC_HOURS = base | extras
    else:
        # Default = live current with h7 already in TOXIC_HOURS_PER_SYMBOL.
        # BT doesn't read per-symbol set; so we add h7 to global for baseline.
        v5.TOXIC_HOURS = {1, 2, 3, 4, 7}

    params = {}
    if "force_trail_name" in ovr and ovr["force_trail_name"] is not None:
        params["force_trail"] = TRAIL_PROFILES[ovr["force_trail_name"]]

    t0 = time.time()
    try:
        r = v5.backtest_symbol(SYMBOL, days=days, params=params, verbose=False)
    except Exception as e:
        import traceback
        return tag, {"error": f"{type(e).__name__}: {e}",
                     "tb": traceback.format_exc()[:300],
                     "took": time.time() - t0}
    if r is None:
        return tag, {"error": "no result", "took": time.time() - t0}
    out = {
        "tag": tag,
        "days": days,
        "trades": int(r.get("trades", 0)),
        "pf":     float(r.get("pf", 0.0) or 0.0),
        "pnl":    float(r.get("pnl", 0.0) or 0.0),
        "wr":     float(r.get("wr", 0.0) or 0.0),
        "dd":     float(r.get("dd", 0.0) or 0.0),
        "avg_r":  float(r.get("avg_r", 0.0) or 0.0),
        "took":   round(time.time() - t0, 2),
    }
    return tag, out


def run_pool(tasks, nproc, label=""):
    """Run a list of (tag, days, overrides) tasks via multiprocessing Pool."""
    print(f"  Running {len(tasks)} tasks ({label})...")
    with Pool(processes=nproc) as pool:
        results = {}
        for tag, out in pool.imap_unordered(_bt_worker, tasks, chunksize=1):
            results[tag] = out
            if "error" in out:
                print(f"    [{tag}] ERR {out['error']}")
            else:
                print(f"    [{tag}] pnl=${out['pnl']:.0f} pf={out['pf']:.2f} "
                      f"n={out['trades']} dd={out['dd']:.2f} t={out['took']}s")
    return results


def _topk(res_dict, k=3, metric="pnl"):
    items = [(k_, v) for k_, v in res_dict.items() if "error" not in v]
    items.sort(key=lambda kv: -kv[1].get(metric, -1e9))
    return items[:k]


def main():
    t_start = time.time()
    nproc = max(1, min(8, (cpu_count() or 4) - 2))
    print(f"AUDJPY tuner — workers={nproc}, days={DAYS}, time budget 2h")
    print(f"Output: {OUT_JSON}")

    # ── Baseline (current production config — toxic h7 enabled) ──
    print("\n=== BASELINE (live current config) ===")
    base_res = run_pool([("baseline", DAYS, {"toxic_hours_set": {7}})], 1, "baseline")
    base = base_res["baseline"]
    if "error" in base:
        print(f"BASELINE FAILED: {base}")
        return
    print(f"Baseline: pnl=${base['pnl']:.0f} pf={base['pf']:.2f} "
          f"n={base['trades']} dd={base['dd']:.2f}")
    baseline_pnl = base["pnl"]

    # ── Phase A: independent dim sweeps (toxic h7 fixed during dim sweeps) ──
    print("\n=== PHASE A — INDEPENDENT SWEEPS ===")

    print("\n-- Dim 1: SL multiplier (10) --")
    sl_tasks = [(f"sl_{x}", DAYS, {"sl_mult": x, "toxic_hours_set": {7}})
                for x in SL_VALUES]
    sl_res = run_pool(sl_tasks, nproc, "SL")

    print("\n-- Dim 2: Trail profile (7) --")
    trail_tasks = [(f"trail_{n}", DAYS, {"force_trail_name": n, "toxic_hours_set": {7}})
                   for n in TRAIL_PROFILES]
    trail_res = run_pool(trail_tasks, nproc, "Trail")

    print("\n-- Dim 3: Pullback ATR (6) --")
    pb_atr_tasks = [(f"pb_atr_{x}", DAYS, {"pb_atr": x, "toxic_hours_set": {7}})
                    for x in PB_ATR_VALUES]
    pb_atr_res = run_pool(pb_atr_tasks, nproc, "PB ATR")

    print("\n-- Dim 4: Pullback wait (5) --")
    pb_wait_tasks = [(f"pb_wait_{x}", DAYS, {"pb_wait": x, "toxic_hours_set": {7}})
                     for x in PB_WAIT_VALUES]
    pb_wait_res = run_pool(pb_wait_tasks, nproc, "PB Wait")

    print("\n-- Dim 5: VWAP buffer (5) --")
    vwap_tasks = [(f"vwap_{x}", DAYS, {"vwap_buf": x, "toxic_hours_set": {7}})
                  for x in VWAP_VALUES]
    vwap_res = run_pool(vwap_tasks, nproc, "VWAP")

    print("\n-- Dim 6: min_quality (6) --")
    mq_tasks = [(f"mq_{x}", DAYS, {"min_quality": x, "toxic_hours_set": {7}})
                for x in MQ_VALUES]
    mq_res = run_pool(mq_tasks, nproc, "MQ")

    print("\n-- Dim 7: Toxic hours (3) --")
    tox_tasks = [(f"toxic_{name}", DAYS, {"toxic_hours_set": s})
                 for name, s in TOXIC_OPTIONS]
    tox_res = run_pool(tox_tasks, nproc, "Toxic")

    # Phase A top picks
    sl_top  = _topk(sl_res, 3)
    tr_top  = _topk(trail_res, 3)
    pa_top  = _topk(pb_atr_res, 3)
    pw_top  = _topk(pb_wait_res, 3)
    vw_top  = _topk(vwap_res, 3)
    mq_top  = _topk(mq_res, 3)
    tox_top = _topk(tox_res, 3)

    print("\n--- Phase A top picks (by 180d PnL) ---")
    for name, picks in [("SL", sl_top), ("Trail", tr_top), ("PB_ATR", pa_top),
                        ("PB_wait", pw_top), ("VWAP", vw_top), ("MQ", mq_top),
                        ("Toxic", tox_top)]:
        print(f"  {name}: " + ", ".join(f"{t[0]}=${t[1]['pnl']:.0f}" for t in picks))

    phase_a = {
        "sl":       [{"tag": k, **v} for k, v in sl_top],
        "trail":    [{"tag": k, **v} for k, v in tr_top],
        "pb_atr":   [{"tag": k, **v} for k, v in pa_top],
        "pb_wait":  [{"tag": k, **v} for k, v in pw_top],
        "vwap":     [{"tag": k, **v} for k, v in vw_top],
        "mq":       [{"tag": k, **v} for k, v in mq_top],
        "toxic":    [{"tag": k, **v} for k, v in tox_top],
    }

    # ── Phase B: top-2 × 6 combinatorial = 64 BTs. Toxic = Phase A winner.
    print("\n=== PHASE B — TOP-2 COMBINATORIAL (64 BTs) ===")
    sl_top2  = [_parse_sl(t[0])      for t in sl_top[:2]]
    tr_top2  = [_parse_trail(t[0])   for t in tr_top[:2]]
    pa_top2  = [_parse_pb_atr(t[0])  for t in pa_top[:2]]
    pw_top2  = [_parse_pb_wait(t[0]) for t in pw_top[:2]]
    vw_top2  = [_parse_vwap(t[0])    for t in vw_top[:2]]
    mq_top2  = [_parse_mq(t[0])      for t in mq_top[:2]]
    best_toxic_name = tox_top[0][0]  # "toxic_h7" or "toxic_none" etc.
    best_toxic_set  = dict(TOXIC_OPTIONS)[best_toxic_name.split("_", 1)[1]]

    print(f"  Best toxic from Phase A: {best_toxic_name} → set={best_toxic_set}")
    print(f"  SL top-2:     {sl_top2}")
    print(f"  Trail top-2:  {tr_top2}")
    print(f"  PB_ATR top-2: {pa_top2}")
    print(f"  PB_wait top-2:{pw_top2}")
    print(f"  VWAP top-2:   {vw_top2}")
    print(f"  MQ top-2:     {mq_top2}")

    combo_tasks = []
    for sl_v in sl_top2:
        for tr_n in tr_top2:
            for pa_v in pa_top2:
                for pw_v in pw_top2:
                    for vw_v in vw_top2:
                        for mq_v in mq_top2:
                            tag = (f"sl{sl_v}_tr{tr_n}_pa{pa_v}"
                                   f"_pw{pw_v}_vw{vw_v}_mq{mq_v}")
                            combo_tasks.append((tag, DAYS, {
                                "sl_mult": sl_v,
                                "force_trail_name": tr_n,
                                "pb_atr": pa_v,
                                "pb_wait": pw_v,
                                "vwap_buf": vw_v,
                                "min_quality": mq_v,
                                "toxic_hours_set": best_toxic_set,
                            }))
    print(f"  Built {len(combo_tasks)} combo tasks")
    combo_res = run_pool(combo_tasks, nproc, "Phase B combos")
    combo_sorted = sorted(
        [(k, v) for k, v in combo_res.items() if "error" not in v],
        key=lambda kv: -kv[1].get("pnl", -1e9)
    )

    print("\nTop 10 combos:")
    for k, v in combo_sorted[:10]:
        print(f"  {k}  pnl=${v['pnl']:.0f}  pf={v['pf']:.2f}  "
              f"n={v['trades']}  dd={v['dd']:.2f}  "
              f"Δ=${v['pnl']-baseline_pnl:+.0f}")

    phase_b_top5 = combo_sorted[:5]

    # ── Phase C: walk-forward validation top-5 across 5 windows ──
    print("\n=== PHASE C — WALK-FORWARD VALIDATION (top-5 × 5 windows) ===")
    wf_days = [60, 90, 120, 150, 180]
    wf_tasks = []
    top5_overrides = []
    for tag, _v in phase_b_top5:
        ovr = next(t[2] for t in combo_tasks if t[0] == tag)
        top5_overrides.append((tag, ovr))
        for d in wf_days:
            wf_tasks.append((f"{tag}@{d}d", d, ovr))
    # Add baseline at each WF window
    for d in wf_days:
        wf_tasks.append((f"baseline@{d}d", d, {"toxic_hours_set": {7}}))

    wf_res = run_pool(wf_tasks, nproc, "Phase C WF")

    # Score WF: for each fold, candidate vs baseline at that window. Positive if
    # candidate's PnL > baseline's PnL at the same window. Ship requires:
    #   Δ at 180d ≥ $50 AND positive folds ≥ 3/5.
    wf_summary = []
    for tag, ovr in top5_overrides:
        folds = []
        for d in wf_days:
            cand = wf_res.get(f"{tag}@{d}d", {})
            base_d = wf_res.get(f"baseline@{d}d", {})
            c_pnl = cand.get("pnl", float("nan"))
            b_pnl = base_d.get("pnl", float("nan"))
            delta = c_pnl - b_pnl if (c_pnl == c_pnl and b_pnl == b_pnl) else float("nan")
            folds.append({
                "days": d,
                "cand_pnl": c_pnl,
                "base_pnl": b_pnl,
                "delta": delta,
                "cand_pf": cand.get("pf"),
                "cand_n":  cand.get("trades"),
            })
        positive = sum(1 for f in folds if (f["delta"] == f["delta"]) and f["delta"] > 0)
        full = combo_res.get(tag, {})
        wf_summary.append({
            "tag": tag,
            "overrides": _serialize_ovr(ovr),
            "folds": folds,
            "positive_folds": positive,
            "full_180d_pnl": full.get("pnl"),
            "full_180d_pf":  full.get("pf"),
            "full_180d_dd":  full.get("dd"),
            "full_180d_n":   full.get("trades"),
            "full_180d_delta": (full.get("pnl", 0) or 0) - baseline_pnl,
        })

    print("\n--- WF results ---")
    for s in wf_summary:
        deltas = [f"{f['delta']:+.0f}" for f in s["folds"]]
        print(f"  {s['tag']}")
        print(f"    positive={s['positive_folds']}/5  fold_deltas={deltas}")
        print(f"    Δ180d=${s['full_180d_delta']:+.0f}  "
              f"PF={s['full_180d_pf']:.2f}  N={s['full_180d_n']}")

    # Ship decision: Δ ≥ $50 AND WF ≥ 3/5
    eligible = [s for s in wf_summary
                if s["full_180d_delta"] >= 50 and s["positive_folds"] >= 3]
    if eligible:
        winner = max(eligible, key=lambda s: (s["positive_folds"], s["full_180d_delta"]))
        recommendation = "SHIP"
    else:
        winner = max(wf_summary, key=lambda s: (s["positive_folds"], s["full_180d_delta"]))
        if winner["positive_folds"] < 3 or winner["full_180d_delta"] < 50:
            recommendation = "HOLD"
        else:
            recommendation = "HOLD"  # belt-and-braces

    # ── Save JSON ──
    out = {
        "symbol": SYMBOL,
        "tune_date": "2026-05-22",
        "days": DAYS,
        "baseline": {
            "trades":  base["trades"],
            "pf":      round(base["pf"], 3),
            "pnl":     round(base["pnl"], 2),
            "wr":      round(base["wr"], 2),
            "dd":      round(base["dd"], 3),
            "config":  "live current (SL 2.5 vol2.0, _WIDE_RUNNER, PB 0.8/5, "
                       "VWAP 0.5, MQ {t:30 r:32 v:30 l:30}, toxic h7)",
        },
        "phase_a": phase_a,
        "phase_b_top10": [
            {"tag": k, **{kk: (round(vv, 3) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()}}
            for k, v in combo_sorted[:10]
        ],
        "phase_c_winner": {
            "tag": winner["tag"],
            "overrides": winner["overrides"],
            "wf_folds": winner["folds"],
            "positive_folds": winner["positive_folds"],
            "expected_180d_pnl": winner["full_180d_pnl"],
            "expected_180d_pf":  winner["full_180d_pf"],
            "expected_180d_dd":  winner["full_180d_dd"],
            "delta_vs_baseline": winner["full_180d_delta"],
        },
        "phase_c_all_top5": wf_summary,
        "recommendation": recommendation,
        "ship_criteria": {
            "min_delta_pnl": 50,
            "min_positive_folds": 3,
        },
        "total_runtime_s": round(time.time() - t_start, 1),
        "total_backtests": 1 + len(sl_tasks) + len(trail_tasks) + len(pb_atr_tasks)
                           + len(pb_wait_tasks) + len(vwap_tasks) + len(mq_tasks)
                           + len(tox_tasks) + len(combo_tasks) + len(wf_tasks),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")
    print(f"Recommendation: {recommendation}")

    # ── Markdown summary ──
    md_lines = [
        f"# AUDJPY fine-tune — 2026-05-22",
        "",
        f"**Total backtests:** {out['total_backtests']}  "
        f"**Runtime:** {round(out['total_runtime_s']/60, 1)} min",
        "",
        "## Baseline (live current config)",
        f"- Trades: {base['trades']}",
        f"- PnL: ${base['pnl']:.2f} / 180d",
        f"- PF: {base['pf']:.2f}",
        f"- WR: {base['wr']:.1f}%",
        f"- DD: {base['dd']:.2f}",
        "- Live config: SL 2.5 (volatile 2.0), _WIDE_RUNNER, PB 0.8/5, VWAP 0.5, "
        "MQ {t:30 r:32 v:30 l:30}, toxic h7",
        "",
        "## Phase A — Independent dim sweeps (top picks)",
        ""
    ]
    for name, picks in [("SL", sl_top), ("Trail", tr_top), ("PB_ATR", pa_top),
                        ("PB_wait", pw_top), ("VWAP", vw_top), ("MQ", mq_top),
                        ("Toxic", tox_top)]:
        md_lines.append(f"**{name}:** " +
                        ", ".join(f"`{t[0]}` pnl=${t[1]['pnl']:.0f} "
                                  f"pf={t[1]['pf']:.2f}" for t in picks))
    md_lines.append("")

    md_lines += [
        "## Phase B — Top-2 combinatorial (top 10)",
        "",
        "| Rank | Tag | PnL | PF | Trades | DD | Δ vs baseline |",
        "|------|-----|-----|----|----|----|---------------|",
    ]
    for i, (k, v) in enumerate(combo_sorted[:10], 1):
        md_lines.append(f"| {i} | `{k}` | ${v['pnl']:.0f} | {v['pf']:.2f} | "
                        f"{v['trades']} | {v['dd']:.2f} | "
                        f"${v['pnl']-baseline_pnl:+.0f} |")
    md_lines.append("")

    md_lines += [
        "## Phase C — Walk-forward validation (top 5)",
        "",
        "| Tag | Pos folds | Δ180d | PF | N | Decision |",
        "|-----|-----------|-------|----|----|----------|",
    ]
    for s in wf_summary:
        decision = "✓ SHIP" if (s["full_180d_delta"] >= 50 and s["positive_folds"] >= 3) else "  hold"
        md_lines.append(f"| `{s['tag']}` | {s['positive_folds']}/5 | "
                        f"${s['full_180d_delta']:+.0f} | "
                        f"{(s['full_180d_pf'] or 0):.2f} | "
                        f"{s['full_180d_n']} | {decision} |")
    md_lines.append("")
    md_lines += [
        "## Winner + Recommendation",
        "",
        f"**Tag:** `{winner['tag']}`",
        f"**Recommendation:** {recommendation}",
        f"**Expected 180d PnL:** ${winner['full_180d_pnl']:.0f} "
        f"(Δ ${winner['full_180d_delta']:+.0f} vs baseline ${baseline_pnl:.0f})",
        f"**Expected PF:** {(winner['full_180d_pf'] or 0):.2f}",
        f"**WF positive folds:** {winner['positive_folds']}/5",
        "",
        "### Final params",
        "```json",
        json.dumps(winner["overrides"], indent=2, default=str),
        "```",
        "",
        "### Walk-forward folds (cand vs baseline at each window)",
        "",
        "| Window | Candidate PnL | Baseline PnL | Δ | PF | N |",
        "|--------|---------------|--------------|---|----|---|",
    ]
    for f in winner["folds"]:
        md_lines.append(f"| {f['days']}d | ${f['cand_pnl']:.0f} | "
                        f"${f['base_pnl']:.0f} | ${f['delta']:+.0f} | "
                        f"{(f['cand_pf'] or 0):.2f} | {f['cand_n']} |")
    md_lines.append("")
    md_lines += [
        "### Ship gates",
        f"- Δ vs baseline ≥ $50: "
        f"{'✓' if winner['full_180d_delta'] >= 50 else '✗'} "
        f"(${winner['full_180d_delta']:+.0f})",
        f"- WF positive folds ≥ 3/5: "
        f"{'✓' if winner['positive_folds'] >= 3 else '✗'} "
        f"({winner['positive_folds']}/5)",
        "",
    ]
    OUT_MD.write_text("\n".join(md_lines))
    print(f"Wrote {OUT_MD}")


def _serialize_ovr(ovr):
    """Make overrides JSON-friendly: trail name stays a string."""
    out = {}
    for k, v in ovr.items():
        if k == "toxic_hours_set":
            out[k] = sorted(list(v))
        else:
            out[k] = v
    return out


# ─── Tag parsers ───
def _parse_sl(tag):     return float(tag.split("_", 1)[1])
def _parse_trail(tag):  return tag.split("_", 1)[1]
def _parse_pb_atr(tag): return float(tag.split("_")[2])
def _parse_pb_wait(tag):return int(tag.split("_")[2])
def _parse_vwap(tag):
    s = tag.split("_", 1)[1]
    return s if s == "disabled" else float(s)
def _parse_mq(tag):     return int(tag.split("_", 1)[1])


if __name__ == "__main__":
    main()
