#!/usr/bin/env python3 -B
"""Iter-2 refinement: tune the winning VWAP_SIDE variants.

Iter-1 results (180d portfolio):
  vwap_side_filter           +$4,586  | EURJPY -$853 (only regression)
  vwap_side_boost_1.15       +$6,477  | GBPUSD flips to -$83, EURJPY -$352
  tick_ratio_boost_1.15      +$559    | GBPUSD flips to -$82

Iter-2 tests:
  A. VWAP filter with whitelist (exclude EURJPY)         — robustness check
  B. VWAP boost @ 1.05/1.10/1.20                          — tune the boost magnitude
  C. VWAP filter with ATR-band buffer                     — soften the strict side
  D. VWAP filter + tick_ratio combo (both must agree)     — composite

Output appended to 03_volume.json under 'iter2'.
"""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path
import multiprocessing as mp
import numpy as np

ROOT = Path("/Users/ashish/Documents/beast-trader")
OUT_DIR = ROOT / "entry_research_20260522"
OUT_JSON = OUT_DIR / "03_volume.json"

sys.path.insert(0, str(ROOT))

SYMBOLS = ["ETHUSD", "GBPUSD", "EURUSD", "GBPJPY", "EURJPY",
           "USDCAD", "GER40.r", "SP500.r"]
WHITELIST_NO_EURJPY = [s for s in SYMBOLS if s != "EURJPY"]

DAYS = 180
WF_FOLDS = 5


def _run_variant(args):
    variant, sym, cfg = args
    try:
        from backtest import v5_backtest as bt
        from signals import momentum_scorer as ms

        mode = cfg["mode"]
        strategy = cfg.get("strategy")
        threshold = float(cfg.get("threshold", 0))
        boost_mult = float(cfg.get("boost_score_mult", 1.0))
        atr_band = float(cfg.get("atr_band", 0))
        tick_ratio_extra = float(cfg.get("tick_ratio_extra", 0))

        orig_score = ms._score_with_components

        def _vwap_ok(ind, i, long_s, short_s, band=0.0):
            vw = ind["vwap"][i]
            if np.isnan(vw):
                return False
            price = float(ind["c"][i])
            atr_i = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0.0
            buf = atr_i * band
            if long_s >= short_s:
                return price > (vw - buf)
            else:
                return price < (vw + buf)

        def _tick_ok(ind, i, thr):
            vs = float(ind["vol_sma"][i]) if not np.isnan(ind["vol_sma"][i]) else 0.0
            vv = float(ind["vol"][i])
            return vs > 0 and (vv / vs) >= thr

        def patched(ind, i, weights=None):
            long_s, short_s, comp_l, comp_s = orig_score(ind, i, weights=weights)
            if strategy is None:
                return long_s, short_s, comp_l, comp_s

            ok = True
            if strategy == "vwap_band":
                ok = _vwap_ok(ind, i, long_s, short_s, band=atr_band)
            elif strategy == "vwap_and_tick":
                ok = _vwap_ok(ind, i, long_s, short_s, band=atr_band) and _tick_ok(ind, i, tick_ratio_extra)
            elif strategy == "vwap_or_tick":
                ok = _vwap_ok(ind, i, long_s, short_s, band=atr_band) or _tick_ok(ind, i, tick_ratio_extra)

            if mode == "filter":
                if not ok:
                    return 0.0, 0.0, comp_l, comp_s
                return long_s, short_s, comp_l, comp_s
            elif mode == "boost":
                if ok:
                    return long_s * boost_mult, short_s * boost_mult, comp_l, comp_s
                return long_s, short_s, comp_l, comp_s
            return long_s, short_s, comp_l, comp_s

        bt._score_with_components = patched
        ms._score_with_components = patched

        res = bt.backtest_symbol(sym, days=DAYS, params={}, verbose=False)

        ms._score_with_components = orig_score
        bt._score_with_components = orig_score

        if res is None or not res.get("details"):
            return (variant, sym, cfg, None)
        return (variant, sym, cfg, [
            {"eb": t["entry_bar"], "pnl": t["pnl"], "r": t["pnl_r"]}
            for t in res["details"]
        ])
    except Exception as e:
        return (variant, sym, cfg, {"error": repr(e), "tb": traceback.format_exc()})


def _stats(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = gw / gl if gl > 0 else (999.0 if gw > 0 else 0.0)
    wr = 100.0 * len(wins) / len(trades)
    eq = [0.0]
    for t in trades:
        eq.append(eq[-1] + t["pnl"])
    peak = eq[0]; mdd = 0.0
    for e in eq:
        peak = max(peak, e)
        mdd = max(mdd, peak - e)
    return {"n": len(trades), "pnl": round(sum(t["pnl"] for t in trades), 2),
            "pf": round(pf, 2), "wr": round(wr, 1), "dd": round(mdd, 2)}


def _wf_metrics(trades, k=5):
    if len(trades) < k:
        return {"n_folds": 0, "avg_pf": 0.0, "positive_folds": 0, "fold_pnl": []}
    s = sorted(trades, key=lambda t: t["eb"])
    sz = len(s) // k
    folds = [s[i*sz:(i+1)*sz if i < k-1 else len(s)] for i in range(k)]
    fp = []; pp = []; pos = 0
    for f in folds:
        st = _stats(f)
        fp.append(st["pnl"]); pp.append(st["pf"])
        if st["pnl"] > 0: pos += 1
    return {"n_folds": len(folds), "avg_pf": round(sum(pp)/len(pp), 2),
            "positive_folds": pos, "fold_pnl": [round(x, 2) for x in fp],
            "fold_pf": [round(x, 2) for x in pp]}


VARIANTS = [
    # A. VWAP filter with ATR-band buffer (softer)
    ("vwap_filter_band_0.1atr", {"mode": "filter", "strategy": "vwap_band", "atr_band": 0.1}),
    ("vwap_filter_band_0.25atr", {"mode": "filter", "strategy": "vwap_band", "atr_band": 0.25}),
    ("vwap_filter_band_0.5atr", {"mode": "filter", "strategy": "vwap_band", "atr_band": 0.5}),

    # B. VWAP boost — different multipliers
    ("vwap_boost_1.05", {"mode": "boost", "strategy": "vwap_band", "atr_band": 0, "boost_score_mult": 1.05}),
    ("vwap_boost_1.10", {"mode": "boost", "strategy": "vwap_band", "atr_band": 0, "boost_score_mult": 1.10}),
    ("vwap_boost_1.20", {"mode": "boost", "strategy": "vwap_band", "atr_band": 0, "boost_score_mult": 1.20}),

    # C. VWAP + tick_ratio combo (BOTH must pass) — strictest
    ("vwap_AND_tick1.2_filter", {"mode": "filter", "strategy": "vwap_and_tick", "atr_band": 0, "tick_ratio_extra": 1.2}),
    ("vwap_AND_tick1.5_filter", {"mode": "filter", "strategy": "vwap_and_tick", "atr_band": 0, "tick_ratio_extra": 1.5}),

    # D. VWAP OR tick (EITHER passes) — softest
    ("vwap_OR_tick1.5_filter", {"mode": "filter", "strategy": "vwap_or_tick", "atr_band": 0, "tick_ratio_extra": 1.5}),
]


def main():
    t0 = time.time()
    tasks = [(v, s, cfg) for v, cfg in VARIANTS for s in SYMBOLS]
    print(f"Iter-2: {len(tasks)} backtests ({len(VARIANTS)} variants × {len(SYMBOLS)} syms)", flush=True)

    raw = {}
    with mp.Pool(min(8, os.cpu_count() or 4)) as pool:
        for ret in pool.imap_unordered(_run_variant, tasks, chunksize=1):
            v, s, cfg, t = ret
            raw.setdefault(v, {})[s] = t

    # Load baseline from iter-1 JSON
    with open(OUT_JSON) as f:
        prev = json.load(f)
    base_port = prev["summary"]["baseline"]["portfolio"]
    base_per_sym = {s: prev["summary"]["baseline"]["per_sym"][s] for s in SYMBOLS}

    iter2 = {}
    for v, cfg in VARIANTS:
        per_sym = {}
        all_t = []
        all_t_no_eurjpy = []
        for s in SYMBOLS:
            t = raw[v].get(s)
            if isinstance(t, dict) or t is None:
                per_sym[s] = {"error": "no_data"}
                continue
            st = _stats(t)
            per_sym[s] = {
                "stats": st,
                "delta_pnl": round(st["pnl"] - base_per_sym[s]["pnl"], 2),
            }
            all_t.extend(t)
            if s != "EURJPY":
                all_t_no_eurjpy.extend(t)
        port_st = _stats(all_t)
        port_wf = _wf_metrics(all_t)
        port_st_no_eurjpy = _stats(all_t_no_eurjpy)
        # Build baseline-no-eurjpy comparator
        base_no_eurjpy = sum(base_per_sym[s]["pnl"] for s in SYMBOLS if s != "EURJPY")

        ship_full = (
            (port_st["pnl"] - base_port["pnl"]) >= 30
            and port_wf["avg_pf"] > 1.5
            and port_wf["positive_folds"] >= 3
        )
        regressions = sum(1 for s in SYMBOLS if isinstance(raw[v].get(s), list)
                          and per_sym[s]["delta_pnl"] < -200)
        sign_flips = sum(
            1 for s in SYMBOLS if isinstance(raw[v].get(s), list)
            and base_per_sym[s]["pnl"] > 0 and per_sym[s]["stats"]["pnl"] < 0
        )

        iter2[v] = {
            "cfg": cfg,
            "per_sym": per_sym,
            "portfolio": port_st,
            "portfolio_wf": port_wf,
            "portfolio_no_eurjpy": port_st_no_eurjpy,
            "delta_full": round(port_st["pnl"] - base_port["pnl"], 2),
            "delta_no_eurjpy": round(port_st_no_eurjpy["pnl"] - base_no_eurjpy, 2),
            "ship_full": ship_full,
            "regressions_gt_200": regressions,
            "sign_flips": sign_flips,
        }

    prev["iter2"] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "variants": iter2,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(prev, f, indent=2, default=str)

    # Summary print
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"\n{'Variant':38s} {'Δ$':>10s} {'PF':>6s} {'WF-PF':>6s} {'Regr':>5s} {'Flip':>5s} {'SHIP':>5s}")
    print("-" * 90)
    for v, _cfg in VARIANTS:
        r = iter2[v]
        print(f"{v:38s} {r['delta_full']:>+10.2f} {r['portfolio']['pf']:>6.2f} "
              f"{r['portfolio_wf']['avg_pf']:>6.2f} {r['regressions_gt_200']:>5d} "
              f"{r['sign_flips']:>5d} {'YES' if r['ship_full'] else 'NO':>5s}")


if __name__ == "__main__":
    main()
