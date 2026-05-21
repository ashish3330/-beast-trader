#!/usr/bin/env python3 -B
"""
STATISTICAL EDGE MAP across (symbol, hour, day_of_week).

For each (sym, hour, dow) cell compute n / wr / pf / avg_r from a 360d
backtest. Then two POST-HOC variants:

  WL: WHITELIST_HIGH_CONFIDENCE — allow only cells with n>=15 AND PF>=2.0
  BL: BLACKLIST_NEGATIVE_EV     — block only cells with n>=15 AND PF<1.0

Walk-forward 5-fold: build cell map on 4/5 of trades (chronological),
evaluate the variant on the 5th. Ship rule: Δ>=$30 AND avg WF PF>1.5
AND >=3/5 folds positive.

READ-ONLY: post-hoc trade-list filtering, no edits to v5_backtest.py.
"""
import sys
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, load_data  # noqa: E402

SYMBOLS = [
    "DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY",
    "EURUSD", "US2000.r", "UKOUSD", "JPN225ft",
]
DAYS = 360

# Cell confidence thresholds
WL_N_MIN  = 15
WL_PF_MIN = 2.0
BL_N_MIN  = 15
BL_PF_MAX = 1.0
HIGH_CONF_N_MIN  = 20
HIGH_CONF_PF_MIN = 2.0
HIGH_CONF_WR_MIN = 55.0

OUT_DIR = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────
def run_backtest_with_dow(sym, days):
    """Run backtest_symbol; reload data once to map entry_bar→(hour,dow).
    Returns (result_dict, trades_with_dow) or (None, [])."""
    r = backtest_symbol(sym, days=days, params=None, verbose=False)
    if r is None or r.get("trades", 0) == 0:
        return None, []
    df = load_data(sym, days)
    if df is None:
        return r, []
    times = df["time"].values
    n = len(times)
    trades_out = []
    for t in r.get("_trades", []) or []:  # may not be present in default return
        eb = t.get("entry_bar")
        if eb is None or eb >= n:
            continue
        ts = pd.Timestamp(times[eb])
        t2 = dict(t)
        t2["hour"] = ts.hour
        t2["dow"]  = ts.dayofweek  # 0=Mon
        t2["ts"]   = ts
        trades_out.append(t2)
    return r, trades_out


def trades_metrics(trades):
    if not trades:
        return {"n": 0, "wins": 0, "wr": 0.0, "pf": 0.0,
                "avg_r": 0.0, "pnl": 0.0, "dd": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = gw / gl if gl > 0 else (999.0 if gw > 0 else 0.0)
    wr = len(wins) / len(trades) * 100
    avg_r = float(np.mean([t.get("pnl_r", 0.0) for t in trades]))
    pnl = sum(t["pnl"] for t in trades)
    # Max DD walk
    eq = 0.0; peak = 0.0; max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.get("ts", 0)):
        eq += t["pnl"]
        peak = max(peak, eq)
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return {"n": len(trades), "wins": len(wins),
            "wr": round(wr, 1), "pf": round(pf, 2),
            "avg_r": round(avg_r, 2), "pnl": round(pnl, 2),
            "dd_usd": round(max_dd, 2)}


def build_cell_map(trades):
    """Return {(hour, dow): metrics_dict}."""
    cells = defaultdict(list)
    for t in trades:
        cells[(t["hour"], t["dow"])].append(t)
    out = {}
    for k, ts in cells.items():
        m = trades_metrics(ts)
        m["recommend"] = classify_cell(m)
        out[k] = m
    return out


def classify_cell(m):
    n = m["n"]; pf = m["pf"]; wr = m["wr"]
    if n >= HIGH_CONF_N_MIN and pf >= HIGH_CONF_PF_MIN and wr >= HIGH_CONF_WR_MIN:
        return "HIGH_CONFIDENCE"
    if n >= WL_N_MIN and pf >= WL_PF_MIN:
        return "WHITELIST_OK"
    if n >= BL_N_MIN and pf < BL_PF_MAX:
        return "BLACKLIST_NEG_EV"
    if n < WL_N_MIN:
        return "LOW_SAMPLE"
    return "NEUTRAL"


def apply_whitelist(trades, cell_map):
    """Allow only cells classified WHITELIST_OK or HIGH_CONFIDENCE."""
    keep = []
    for t in trades:
        c = cell_map.get((t["hour"], t["dow"]))
        if c is None:
            continue  # unknown cell → block (conservative)
        if c["recommend"] in ("WHITELIST_OK", "HIGH_CONFIDENCE"):
            keep.append(t)
    return keep


def apply_blacklist(trades, cell_map):
    """Block only BLACKLIST_NEG_EV cells; otherwise allow."""
    keep = []
    for t in trades:
        c = cell_map.get((t["hour"], t["dow"]))
        if c is None:
            keep.append(t)  # unknown cell → allow (only block proven bad)
            continue
        if c["recommend"] == "BLACKLIST_NEG_EV":
            continue
        keep.append(t)
    return keep


def walk_forward(trades, n_folds=5):
    """Build cell map on first (n_folds-1)/n_folds of trades, evaluate variants
    on the remaining 1/n_folds slice. Rolling fold = expanding window per step.

    Folds 1..n_folds: train on trades[0:k*(N/n_folds)] eval on next chunk.
    """
    trades_sorted = sorted(trades, key=lambda x: x["ts"])
    N = len(trades_sorted)
    if N < n_folds * 3:
        return None
    folds = []
    chunk = N // n_folds
    for k in range(1, n_folds):
        train = trades_sorted[: k * chunk]
        test  = trades_sorted[k * chunk: (k + 1) * chunk]
        if not test:
            continue
        cell_map = build_cell_map(train)
        base_m = trades_metrics(test)
        wl     = apply_whitelist(test, cell_map)
        bl     = apply_blacklist(test, cell_map)
        wl_m   = trades_metrics(wl)
        bl_m   = trades_metrics(bl)
        folds.append({
            "fold": k,
            "train_n": len(train), "test_n": len(test),
            "base": base_m, "wl": wl_m, "bl": bl_m,
            "delta_wl": round(wl_m["pnl"] - base_m["pnl"], 2),
            "delta_bl": round(bl_m["pnl"] - base_m["pnl"], 2),
        })
    return folds


def summarize_wf(folds):
    if not folds:
        return None
    deltas_wl = [f["delta_wl"] for f in folds]
    deltas_bl = [f["delta_bl"] for f in folds]
    pf_wl = [f["wl"]["pf"] for f in folds if f["wl"]["n"] > 0]
    pf_bl = [f["bl"]["pf"] for f in folds if f["bl"]["n"] > 0]
    return {
        "wl": {
            "avg_delta": round(float(np.mean(deltas_wl)), 2),
            "sum_delta": round(float(np.sum(deltas_wl)), 2),
            "avg_pf":    round(float(np.mean(pf_wl)) if pf_wl else 0.0, 2),
            "pos_folds": sum(1 for d in deltas_wl if d > 0),
            "n_folds":   len(deltas_wl),
        },
        "bl": {
            "avg_delta": round(float(np.mean(deltas_bl)), 2),
            "sum_delta": round(float(np.sum(deltas_bl)), 2),
            "avg_pf":    round(float(np.mean(pf_bl)) if pf_bl else 0.0, 2),
            "pos_folds": sum(1 for d in deltas_bl if d > 0),
            "n_folds":   len(deltas_bl),
        },
    }


def ship_decision(wf_summary, full_delta):
    """SHIP iff Δ≥$30 AND WF avg PF>1.5 AND ≥3/5 folds positive."""
    rules = []
    ok_delta = full_delta >= 30
    ok_pf    = wf_summary["avg_pf"] > 1.5
    ok_pos   = wf_summary["pos_folds"] >= max(3, math.ceil(0.6 * wf_summary["n_folds"]))
    rules.append(("delta>=30",   ok_delta))
    rules.append(("wf_pf>1.5",   ok_pf))
    rules.append(("pos_folds>=3", ok_pos))
    return all(x for _, x in rules), rules


# ─────────────────────────────────────────────────────────────────────
def main():
    # backtest_symbol doesn't return trades list by default — monkey-patch to
    # capture. We do this by RE-implementing the trade replay using the public
    # backtest output augmented via a small wrapper. The cleanest path is to
    # patch result to include trades via a thread-local hack. Easier: rerun
    # using internal hook.
    #
    # Inspection shows backtest_symbol returns a dict with metrics but NOT the
    # full trades. We need the trades. Simplest fix without editing module:
    # monkey-patch backtest_symbol's trades.append via a wrapper. Even simpler:
    # use a side-channel by patching trades.append within v5_backtest module.
    import backtest.v5_backtest as v5

    sym_trades = {}

    orig_backtest = v5.backtest_symbol
    captured = {"trades": None}

    # Capture trades by patching list-append... too fragile. Instead, patch
    # the module's `pickle.dump`/internal — even simpler, redefine: we patch
    # the function to return trades. Cheapest: use module-level
    # _LAST_TRADES global. v5 doesn't expose it but we can monkeypatch by
    # wrapping list class. Skip — simpler: re-run with a wrapper that
    # intercepts.
    #
    # Cleanest path: patch `trades.append` is hard. Use ast/exec trick? Way
    # easier: temporarily inject result["_trades"] = trades into the function
    # via copying source and adding one line.
    #
    # We'll patch via a wrapper: rerun backtest_symbol but inject capture via
    # a sentinel — we override `pd.Timestamp` no, that's silly.
    #
    # FINAL APPROACH: redefine backtest_symbol's behavior by appending a small
    # post-hook: read source from module and exec a modified copy with a
    # capture variable. Done via inspect.

    import inspect, types, re as _re
    src = inspect.getsource(v5.backtest_symbol)
    # Replace the final result-build line by injecting trades into result.
    new_src = src.replace(
        "    return result",
        "    result['_trades'] = trades\n    return result",
        1,
    )
    # Place into a local namespace with the same globals
    ns = dict(v5.__dict__)
    exec(new_src, ns)
    patched = ns["backtest_symbol"]

    print("=== STATISTICAL EDGE MAP — 8 symbols × 360d ===\n")
    all_sym_data = {}
    for sym in SYMBOLS:
        r = patched(sym, days=DAYS, params=None, verbose=False)
        if r is None or r.get("trades", 0) == 0:
            print(f"  {sym:12s} — no trades")
            continue
        df = load_data(sym, DAYS)
        if df is None:
            print(f"  {sym:12s} — no data"); continue
        times = df["time"].values
        n = len(times)
        trades = []
        for t in r["_trades"]:
            eb = t.get("entry_bar")
            if eb is None or eb >= n:
                continue
            ts = pd.Timestamp(times[eb])
            t2 = dict(t)
            t2["hour"] = ts.hour
            t2["dow"]  = int(ts.dayofweek)
            t2["ts"]   = ts
            trades.append(t2)
        sym_trades[sym] = trades
        base_m = trades_metrics(trades)
        cell_map = build_cell_map(trades)
        wl_trades = apply_whitelist(trades, cell_map)
        bl_trades = apply_blacklist(trades, cell_map)
        wl_m = trades_metrics(wl_trades)
        bl_m = trades_metrics(bl_trades)
        wf   = walk_forward(trades, n_folds=5)
        wf_s = summarize_wf(wf) if wf else None
        ship_wl, rules_wl = (False, [])
        ship_bl, rules_bl = (False, [])
        if wf_s:
            ship_wl, rules_wl = ship_decision(
                wf_s["wl"], wl_m["pnl"] - base_m["pnl"])
            ship_bl, rules_bl = ship_decision(
                wf_s["bl"], bl_m["pnl"] - base_m["pnl"])

        # Pretty cell map for output
        out_cells = {}
        # high-conf and blacklist counts
        n_hc = n_wl = n_bl = n_low = n_neu = 0
        for (h, d), m in cell_map.items():
            key = f"{h:02d}_{d}"
            out_cells[key] = m
            r2 = m["recommend"]
            if   r2 == "HIGH_CONFIDENCE":   n_hc += 1
            elif r2 == "WHITELIST_OK":      n_wl += 1
            elif r2 == "BLACKLIST_NEG_EV":  n_bl += 1
            elif r2 == "LOW_SAMPLE":        n_low += 1
            else:                           n_neu += 1

        all_sym_data[sym] = {
            "baseline":      base_m,
            "whitelist":     wl_m,
            "blacklist":     bl_m,
            "delta_wl":      round(wl_m["pnl"] - base_m["pnl"], 2),
            "delta_bl":      round(bl_m["pnl"] - base_m["pnl"], 2),
            "cell_counts":   {"high_confidence": n_hc, "whitelist_ok": n_wl,
                              "blacklist_neg_ev": n_bl, "low_sample": n_low,
                              "neutral": n_neu, "total": len(out_cells)},
            "wf_summary":    wf_s,
            "wf_folds":      wf,
            "ship_wl":       ship_wl, "ship_wl_rules": rules_wl,
            "ship_bl":       ship_bl, "ship_bl_rules": rules_bl,
            "cells":         out_cells,
        }

        print(f"  {sym:12s} n={base_m['n']:4d}  PF={base_m['pf']:5.2f}  "
              f"WR={base_m['wr']:5.1f}%  PnL=${base_m['pnl']:>9.2f}")
        print(f"             cells: HC={n_hc}  WL={n_wl}  BL={n_bl}  "
              f"LowN={n_low}  Neu={n_neu}  ({len(out_cells)} total)")
        print(f"             WL    n={wl_m['n']:4d} PF={wl_m['pf']:5.2f} "
              f"WR={wl_m['wr']:5.1f}% PnL=${wl_m['pnl']:>9.2f}  "
              f"Δ=${wl_m['pnl']-base_m['pnl']:>8.2f}  ship={ship_wl}")
        print(f"             BL    n={bl_m['n']:4d} PF={bl_m['pf']:5.2f} "
              f"WR={bl_m['wr']:5.1f}% PnL=${bl_m['pnl']:>9.2f}  "
              f"Δ=${bl_m['pnl']-base_m['pnl']:>8.2f}  ship={ship_bl}")
        if wf_s:
            print(f"             WF WL: avgΔ=${wf_s['wl']['avg_delta']:>7.2f} "
                  f"avgPF={wf_s['wl']['avg_pf']:.2f} pos={wf_s['wl']['pos_folds']}/{wf_s['wl']['n_folds']}")
            print(f"             WF BL: avgΔ=${wf_s['bl']['avg_delta']:>7.2f} "
                  f"avgPF={wf_s['bl']['avg_pf']:.2f} pos={wf_s['bl']['pos_folds']}/{wf_s['bl']['n_folds']}")
        print()

    # Portfolio-level summary
    tot_base = sum(d["baseline"]["pnl"] for d in all_sym_data.values())
    tot_wl   = sum(d["whitelist"]["pnl"] for d in all_sym_data.values())
    tot_bl   = sum(d["blacklist"]["pnl"] for d in all_sym_data.values())
    print(f"PORTFOLIO  base=${tot_base:.2f}  WL=${tot_wl:.2f} Δ=${tot_wl-tot_base:+.2f}  "
          f"BL=${tot_bl:.2f} Δ=${tot_bl-tot_base:+.2f}")

    payload = {
        "config": {
            "days": DAYS, "symbols": SYMBOLS,
            "wl_n_min": WL_N_MIN, "wl_pf_min": WL_PF_MIN,
            "bl_n_min": BL_N_MIN, "bl_pf_max": BL_PF_MAX,
            "high_conf_n_min": HIGH_CONF_N_MIN,
            "high_conf_pf_min": HIGH_CONF_PF_MIN,
            "high_conf_wr_min": HIGH_CONF_WR_MIN,
            "ship_rules": "Δ>=$30 AND wf_avg_pf>1.5 AND pos_folds>=3/5",
        },
        "portfolio": {
            "baseline_pnl":  round(tot_base, 2),
            "whitelist_pnl": round(tot_wl, 2),
            "blacklist_pnl": round(tot_bl, 2),
            "delta_wl":      round(tot_wl - tot_base, 2),
            "delta_bl":      round(tot_bl - tot_base, 2),
        },
        "symbols": all_sym_data,
    }
    out_json = OUT_DIR / "10_stat_edge.json"

    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        return str(o)
    json.dump(payload, open(out_json, "w"), indent=2, default=_json_default)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
