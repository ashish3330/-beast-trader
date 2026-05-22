#!/usr/bin/env python3 -B
"""Per-symbol ENTRY-quality tuner — XAUUSD / SP500.r / USDJPY — 2026-05-22.

Targets: 3 worst-divergence symbols from parity_reaudit.md
  - XAUUSD  -32R divergence (SHORT-in-low_vol bleeds)
  - SP500.r -30R divergence (LONG-in-low_vol bleeds, hr14/15/16)
  - USDJPY  -19R divergence (LONG-in-ranging bleeds, hr 9/10)

Knobs (all in auto_tuned.py; brain.py untouched):
  1. DIRECTION_BIAS_REGIME_AUTO[symbol][regime] ∈ {None, 'LONG', 'SHORT', 'BOTH'}
  2. SIGNAL_QUALITY_SYMBOL_AUTO[symbol][regime] ∈ {28, 32, 38, 45, 55, 65}
  3. TOXIC_HOURS_PER_SYMBOL_AUTO[symbol] (extra) ∈ set candidates
  4. RANGE_FILTER_PARAMS_AUTO[symbol] (ranging only) ∈ {(48,0.3), (72,0.5), (96,1.0)}

Workflow:
  Phase 0: live-config baseline (per symbol, 180d)
  Phase A: for each symbol, sweep the 4 knobs as coordinate descent:
           - Try each (regime, dir_bias_setting) for each regime where live had losers
           - For each min_quality candidate per regime
           - For each toxic-hour candidate set
           - For each range filter (if regime ranging is loser)
  Phase B: top-3 winners by full-window PnL → 5-fold disjoint WF
  Phase C: ship-eligible (Δ >= $30 AND WF >= 3/5 positive)
  Phase D: stack all winners → BT verify no negative interaction

Output:
  /Users/ashish/Documents/beast-trader/audit_20260522/per_sym_entry_xau_sp_jpy.md
  /Users/ashish/Documents/beast-trader/audit_20260522/per_sym_entry_xau_sp_jpy.json
"""

import json
import os
import sys
import time
import traceback
import importlib
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "audit_20260522"
OUT_JSON = OUT_DIR / "per_sym_entry_xau_sp_jpy.json"
OUT_MD   = OUT_DIR / "per_sym_entry_xau_sp_jpy.md"
LOG_FILE = OUT_DIR / "per_sym_entry_tune.log"

DAYS = 180

# Ship gates
MIN_DELTA = 30.0
WF_MIN_POS = 3
WF_NUM_FOLDS = 5

# Per-symbol WF window — XAU/JPY have shallow caches (~30d), SP500 has 15y.
# BT needs len(df) >= 200 bars (~8.3 days of H1). For shallow-cache symbols
# we use OVERLAPPING sliding folds instead of disjoint ones.
PER_SYMBOL_DAYS = {
    "XAUUSD":  29,
    "SP500.r": 180,
    "USDJPY":  28,
}
PER_SYMBOL_FOLD_DAYS = {
    "XAUUSD":  15,  # sliding 15d windows (~360 H1 bars, > 200 min)
    "SP500.r": 36,  # disjoint 36d folds
    "USDJPY":  15,
}
# When sliding folds: start of fold i = i * SLIDE_STEP, length = FOLD_DAYS
PER_SYMBOL_FOLD_MODE = {
    "XAUUSD":  "sliding",
    "SP500.r": "disjoint",
    "USDJPY":  "sliding",
}
PER_SYMBOL_SLIDE_STEP = {
    "XAUUSD":  3,   # 5 folds × 3d slide = covers 15+12=27d of 29d cache
    "USDJPY":  3,
}

SYMBOLS = ["XAUUSD", "SP500.r", "USDJPY"]

# ── Per-symbol candidate knob grids (informed by BT-side trade forensics) ──
# IMPORTANT: BT's regime classifier ≠ live's. BT classifies the same bars as
# 'volatile'/'ranging' where live records 'low_vol'/'ranging'. We tune to BT
# reality because the BT is what predicts ship eligibility.
#
# BT-side breakdown (180d, current auto_tuned):
#   XAUUSD : all (dir, regime) buckets positive — no obvious BT loser.
#            Live journal SHORT-in-low_vol bleeds — but BT doesn't reproduce.
#            Focus: BT hour 18 (-$14.86 / 1 trade); generally fine.
#   SP500.r: LONG-ranging +$35K, LONG-volatile +$1.27M, SHORT-ranging +$151
#            Hours 11/12 -$12K/-$14K. Tighten hours 11/12.
#   USDJPY : LONG-ranging WINS (+$33.62/91% WR). LONG-volatile LOSES (-$15/58%).
#            Hours 6, 8, 9, 10 bad.
CANDIDATES = {
    "XAUUSD": {
        # Live journal said SHORT-in-low_vol bleeds. BT shows SHORTs are fine
        # in both volatile and ranging. We test removing SHORTs from each
        # regime to see if any improves BT.
        "dir_bias_sweep": [
            {},  # baseline (no extra cell)
            {"volatile": "LONG"},
            {"ranging":  "LONG"},
            {"volatile": "SHORT"},  # opposite — sanity check
            {"low_vol":  "LONG"},   # journal-side hypothesis (BT may have low_vol bars too)
            {"low_vol":  "BOTH"},
            {"trending": "LONG"},
        ],
        # Tighten min_q per regime
        "min_q_sweep": [
            None,  # current: 40/40/40/40
            {"volatile": 45},
            {"volatile": 50},
            {"volatile": 55},
            {"ranging":  45},
            {"ranging":  50},
            {"low_vol":  50},
            {"low_vol":  60},
        ],
        "toxic_sweep": [
            (set(), "none"),
            ({18},        "h18"),
            ({17, 18},    "h17_18"),
            ({16, 17, 18}, "h16_18"),
            ({18, 21},    "h18_21"),
        ],
        # Range-filter (lookback, buffer_atr) — only in RANGING regime
        "range_filter_sweep": [
            None,
            (48, 0.5),
            (72, 0.7),
            (96, 1.0),
        ],
    },
    "SP500.r": {
        "dir_bias_sweep": [
            {},
            {"ranging":  "LONG"},   # SHORT-ranging is the weak side (PF~1)
            {"volatile": "LONG"},   # already in auto_tuned as 'volatile':LONG
            {"low_vol":  "SHORT"},  # journal-side hypothesis
            {"low_vol":  "LONG"},
        ],
        "min_q_sweep": [
            None,  # current: all 28
            {"low_vol":  35},
            {"low_vol":  40},
            {"volatile": 35},
            {"volatile": 38},
            {"ranging":  35},
        ],
        "toxic_sweep": [
            (set(), "none"),
            ({11},          "h11"),
            ({12},          "h12"),
            ({11, 12},      "h11_12"),
            ({11, 12, 14},  "h11_14"),  # live: 14 was bad, BT: 11/12 worst
        ],
        "range_filter_sweep": [
            None,
            (48, 0.5),
            (72, 0.7),
            (96, 1.0),
        ],
    },
    "USDJPY": {
        # BT-side LONG-volatile is loser (-$15, 58% WR), LONG-ranging wins.
        # We test BOTH the BT pattern AND the live hypothesis (ranging-LONG bleeds).
        "dir_bias_sweep": [
            {},  # baseline
            {"volatile": "SHORT"},  # block LONG-volatile (BT loser)
            {"volatile": "LONG"},   # current auto_tuned (already set)
            {"volatile": "BOTH"},   # remove the LONG cell
            {"ranging":  "SHORT"},  # journal-side hypothesis (block LONG-ranging)
            {"low_vol":  "SHORT"},  # journal-side
        ],
        "min_q_sweep": [
            None,
            {"volatile": 45},
            {"volatile": 55},
            {"volatile": 60},
            {"ranging":  50},
            {"ranging":  55},
            {"low_vol":  50},
        ],
        "toxic_sweep": [
            (set(), "none"),
            ({6, 8, 9, 10},     "h6_10"),
            ({9, 10},           "h9_10"),
            ({6, 7, 8, 9, 10},  "h6_7_10"),
            ({7, 9, 10},        "h7_9_10"),
        ],
        "range_filter_sweep": [
            None,
            (48, 0.5),
            (72, 0.7),
            (96, 1.0),
        ],
    },
}


def _log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────
# WORKER — single backtest with knob overlays applied in-process.
# Returns {trades, pf, wr, pnl, dd, err?}
# ──────────────────────────────────────────────────────────────────────
def _bt_one(args):
    """
    args: (symbol, dir_bias_cell, min_q_cell, toxic_set, range_filter, fold_id)
      dir_bias_cell: dict {regime: 'LONG'|'SHORT'|'BOTH'}
      min_q_cell: dict {regime: int} or None
      toxic_set: set of extra hours (unioned with global)
      range_filter: tuple (lookback, buffer_atr) or None
      fold_id: 1..5 or None (full window)
    """
    (symbol, dir_bias_cell, min_q_cell,
     toxic_set, range_filter, fold_id) = args
    try:
        # Fresh per-process reload so overlays don't leak across BTs.
        # IMPORTANT: auto_tuned's inner dicts are shared by reference with
        # config's merged dicts (config does .update(AUTO), which copies refs
        # not values). We must reload auto_tuned FIRST to reset its literals,
        # THEN reload config to refresh references, THEN reload bt.
        import importlib as _il
        import auto_tuned as _at
        _il.reload(_at)
        import config as _cfg
        _il.reload(_cfg)
        import backtest.v5_backtest as _bt
        _il.reload(_bt)

        # 1. Direction bias overlay (in-process)
        if dir_bias_cell:
            existing = _bt._DIR_BIAS_REGIME_STR.get(symbol, {})
            new_cell = dict(existing)
            for r, side in dir_bias_cell.items():
                new_cell[r] = side
            _bt._DIR_BIAS_REGIME_STR[symbol] = new_cell

        # 2. Min-quality overlay (we patch config.SIGNAL_QUALITY_SYMBOL which
        #    BT reads inside backtest_symbol via `from config import ...`).
        #    CRITICAL: the inner dict is shared by reference with
        #    auto_tuned.SIGNAL_QUALITY_SYMBOL_AUTO (config does
        #    SIGNAL_QUALITY_SYMBOL.update(SIGNAL_QUALITY_SYMBOL_AUTO) which
        #    copies references, not deep-copies). Mutating in-place WILL leak
        #    across reloads because auto_tuned is cached. Always copy before
        #    overlay.
        if min_q_cell is not None:
            cur = dict(_cfg.SIGNAL_QUALITY_SYMBOL.get(symbol, {}))
            cur.update(min_q_cell)
            _cfg.SIGNAL_QUALITY_SYMBOL[symbol] = cur

        # 3. Toxic hours overlay — BT reads module-level TOXIC_HOURS as a set;
        #    since BT does not key TOXIC_HOURS per symbol, we union into the
        #    module set for THIS process's lifetime. This is correct because
        #    only the target symbol gets backtested here. We also pad
        #    TOXIC_EXEMPT for other symbols to avoid cross-contamination.
        if toxic_set:
            _bt.TOXIC_HOURS = set(_bt.TOXIC_HOURS) | set(toxic_set)

        # 4. Range filter params — BT auto-injects on backtest_symbol() entry
        #    via `auto_tuned.RANGE_FILTER_PARAMS_AUTO`. We can override by
        #    passing range_lookback / range_buffer_atr / range_filter_enabled
        #    in params dict. Build the params dict now.
        params = {}
        if range_filter is not None:
            params["range_filter_enabled"] = True
            params["range_lookback"] = int(range_filter[0])
            params["range_buffer_atr"] = float(range_filter[1])
        # If candidate explicitly chose "None" (disabled), force-disable so
        # the auto-inject in backtest_symbol doesn't re-enable it.
        else:
            params["range_filter_enabled"] = False

        # 5. Fold slicing for WF
        sym_days = PER_SYMBOL_DAYS.get(symbol, DAYS)
        sym_fold_d = PER_SYMBOL_FOLD_DAYS.get(symbol, 36)
        sym_mode = PER_SYMBOL_FOLD_MODE.get(symbol, "disjoint")
        slide_step = PER_SYMBOL_SLIDE_STEP.get(symbol, sym_fold_d)
        if fold_id is not None:
            import pandas as pd
            orig_load = _bt.load_data
            fold_n = int(fold_id)
            num = WF_NUM_FOLDS
            fold_d = sym_fold_d

            if sym_mode == "sliding":
                # Sliding: fold i covers [start_offset .. start_offset + fold_d]
                # where start_offset = (fold_n - 1) * slide_step from oldest bar.
                def load_data_fold(sym, _ignored_days=None):
                    df = orig_load(sym, days=None)
                    if df is None or df.empty:
                        return df
                    t_min = df["time"].min()
                    t_start = t_min + pd.Timedelta(days=(fold_n - 1) * slide_step)
                    t_end   = t_start + pd.Timedelta(days=fold_d)
                    df = df[(df["time"] >= t_start) & (df["time"] < t_end)].reset_index(drop=True)
                    return df
            else:
                # Disjoint: counting from end, num folds × fold_d days each.
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
            r = _bt.backtest_symbol(symbol, days=None, params=params, verbose=False)
        else:
            r = _bt.backtest_symbol(symbol, days=sym_days, params=params, verbose=False)

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


def _bt_baseline(symbol):
    """Live-config baseline for a single symbol."""
    try:
        import importlib as _il
        import auto_tuned as _at
        _il.reload(_at)
        import config as _cfg
        _il.reload(_cfg)
        import backtest.v5_backtest as _bt
        _il.reload(_bt)
        sym_days = PER_SYMBOL_DAYS.get(symbol, DAYS)
        r = _bt.backtest_symbol(symbol, days=sym_days, verbose=False)
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
        return {"err": f"{type(e).__name__}: {e}"}


# ──────────────────────────────────────────────────────────────────────
# Iterate candidates per symbol — generate (cfg, args) pairs
# ──────────────────────────────────────────────────────────────────────
def _gen_candidates(symbol):
    grid = CANDIDATES[symbol]
    out = []
    # Phase A1: dir_bias only (with everything else at baseline)
    for db in grid["dir_bias_sweep"]:
        out.append({
            "tag": f"db={db or 'none'}",
            "dir_bias_cell": db,
            "min_q_cell": None,
            "toxic_set": set(),
            "range_filter": None,
        })
    # Phase A2: min_q only
    for mq in grid["min_q_sweep"]:
        if mq is None:
            continue
        out.append({
            "tag": f"mq={mq}",
            "dir_bias_cell": {},
            "min_q_cell": mq,
            "toxic_set": set(),
            "range_filter": None,
        })
    # Phase A3: toxic only
    for (tset, tlabel) in grid["toxic_sweep"]:
        if not tset:
            continue
        out.append({
            "tag": f"toxic={tlabel}",
            "dir_bias_cell": {},
            "min_q_cell": None,
            "toxic_set": set(tset),
            "range_filter": None,
        })
    # Phase A4: range filter only
    for rf in grid["range_filter_sweep"]:
        if rf is None:
            continue
        out.append({
            "tag": f"rf=({rf[0]},{rf[1]})",
            "dir_bias_cell": {},
            "min_q_cell": None,
            "toxic_set": set(),
            "range_filter": rf,
        })
    return out


def _phase_a_for_symbol(symbol, workers):
    """Run Phase A coordinate-style single-knob sweep. Return baseline + sorted results."""
    cands = _gen_candidates(symbol)
    _log(f"[{symbol}] Phase A: {len(cands)} single-knob candidates")

    # Baseline (live config)
    with Pool(1) as pool:
        baseline = pool.apply(_bt_baseline, (symbol,))
    if "err" in baseline:
        _log(f"  [{symbol}] baseline FAILED: {baseline['err']}")
        return baseline, [], cands
    _log(f"  [{symbol}] baseline: trades={baseline['trades']} PF={baseline['pf']:.2f} "
         f"PnL=${baseline['pnl']:+.0f} DD={baseline['dd']:.2f}%")

    # Build job tuples
    jobs = [
        (symbol, c["dir_bias_cell"], c["min_q_cell"],
         c["toxic_set"], c["range_filter"], None)
        for c in cands
    ]
    t0 = time.time()
    with Pool(workers) as pool:
        raw = list(pool.imap(_bt_one, jobs))
    _log(f"  [{symbol}] Phase A done in {time.time()-t0:.1f}s")
    out = []
    for c, res in zip(cands, raw):
        if "err" in res:
            _log(f"    [{symbol}] {c['tag']}: ERR {res['err'][:140]}")
            continue
        out.append({
            **c,
            "trades": res["trades"],
            "pf":     res["pf"],
            "wr":     res["wr"],
            "pnl":    res["pnl"],
            "dd":     res["dd"],
            "delta":  res["pnl"] - baseline["pnl"],
        })
    out.sort(key=lambda x: x["pnl"], reverse=True)
    return baseline, out, cands


def _build_combo_candidates(symbol, top_a, baseline, k_top=3):
    """Combine top-A single knobs into combo candidates (each top-k cross-product)."""
    # Take top-k by delta from each knob "type" (dir_bias, min_q, toxic, rf).
    by_type = {"db": [], "mq": [], "tx": [], "rf": []}
    for r in top_a:
        if r["dir_bias_cell"]:    by_type["db"].append(r)
        elif r["min_q_cell"]:     by_type["mq"].append(r)
        elif r["toxic_set"]:      by_type["tx"].append(r)
        elif r["range_filter"]:   by_type["rf"].append(r)
    # Only keep positive-delta ones
    for k in by_type:
        by_type[k] = [r for r in by_type[k] if r["delta"] > 0][:k_top]
        if not by_type[k]:
            by_type[k] = [None]
    combos = []
    for db in by_type["db"]:
        for mq in by_type["mq"]:
            for tx in by_type["tx"]:
                for rf in by_type["rf"]:
                    if db is None and mq is None and tx is None and rf is None:
                        continue
                    combos.append({
                        "tag": "+".join([x["tag"] for x in (db, mq, tx, rf) if x]),
                        "dir_bias_cell": db["dir_bias_cell"] if db else {},
                        "min_q_cell":    mq["min_q_cell"]    if mq else None,
                        "toxic_set":     tx["toxic_set"]     if tx else set(),
                        "range_filter":  rf["range_filter"]  if rf else None,
                    })
    return combos


def _wf_for_finalists(symbol, finalists, workers):
    """Run 5-fold WF on top-N finalists. Returns list of {finalist, folds[]}."""
    jobs = []
    for f in finalists:
        for fold in range(1, WF_NUM_FOLDS + 1):
            jobs.append((
                symbol, f["dir_bias_cell"], f["min_q_cell"],
                f["toxic_set"], f["range_filter"], fold,
            ))
    t0 = time.time()
    with Pool(workers) as pool:
        raw = list(pool.imap(_bt_one, jobs))
    _log(f"  [{symbol}] WF done in {time.time()-t0:.1f}s")
    out = []
    for i, finalist in enumerate(finalists):
        folds = []
        for fold in range(1, WF_NUM_FOLDS + 1):
            r = raw[i * WF_NUM_FOLDS + (fold - 1)]
            if "err" in r:
                folds.append({"fold": fold, "trades": 0, "pf": 0, "pnl": 0, "wr": 0,
                              "err": r["err"][:140]})
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
        out.append({**finalist, "wf_folds": folds, "wf_pos": pos, "wf_avg_pf": avg_pf,
                    "wf_total_pnl": round(sum(fr.get("pnl", 0) for fr in folds), 2)})
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")
    t_start = time.time()
    workers = max(2, (os.cpu_count() or 4) - 2)
    _log(f"Per-symbol entry tuner — workers={workers}")

    results = {}
    for symbol in SYMBOLS:
        _log(f"\n========= {symbol} =========")
        baseline, top_a, cands = _phase_a_for_symbol(symbol, workers)
        if "err" in baseline:
            results[symbol] = {"err": baseline["err"]}
            continue
        # Print top-A
        _log(f"  [{symbol}] Top single-knob deltas:")
        for r in top_a[:8]:
            _log(f"    {r['tag']:38s}  trades={r['trades']:3d} "
                 f"PF={r['pf']:5.2f} PnL=${r['pnl']:+7.0f}  Δ=${r['delta']:+.0f}")

        # Phase B: combo candidates
        combos = _build_combo_candidates(symbol, top_a, baseline, k_top=2)
        # Always include single-knob top-3 winners too so a 1-knob change can ship
        single_winners = [r for r in top_a if r["delta"] > 0][:3]
        all_finalists = single_winners + combos
        _log(f"  [{symbol}] Combo+singles: {len(all_finalists)} finalists pre-WF")

        # Run finalists at full window (re-confirm)
        f_jobs = [
            (symbol, f["dir_bias_cell"], f["min_q_cell"],
             f["toxic_set"], f["range_filter"], None)
            for f in all_finalists
        ]
        with Pool(workers) as pool:
            f_raw = list(pool.imap(_bt_one, f_jobs))
        scored_finalists = []
        for f, res in zip(all_finalists, f_raw):
            if "err" in res:
                continue
            scored_finalists.append({
                **f,
                "trades": res["trades"], "pf": res["pf"],
                "wr": res["wr"], "pnl": res["pnl"], "dd": res["dd"],
                "delta": res["pnl"] - baseline["pnl"],
            })
        scored_finalists.sort(key=lambda x: x["pnl"], reverse=True)
        top_finalists = scored_finalists[:5]
        _log(f"  [{symbol}] Top-{len(top_finalists)} finalists (full window):")
        for r in top_finalists:
            _log(f"    {r['tag']:50s}  trades={r['trades']:3d} "
                 f"PF={r['pf']:5.2f} PnL=${r['pnl']:+7.0f}  Δ=${r['delta']:+.0f}")

        # Phase C: walk-forward
        wf_results = _wf_for_finalists(symbol, top_finalists, workers)
        # Print WF results
        ship_eligible = []
        for r in wf_results:
            ship = (r["delta"] >= MIN_DELTA) and (r["wf_pos"] >= WF_MIN_POS)
            r["ship"] = bool(ship)
            if ship:
                ship_eligible.append(r)
            _log(f"  [{symbol}] {r['tag']:50s}  Δ=${r['delta']:+.0f}  "
                 f"WF {r['wf_pos']}/{WF_NUM_FOLDS} avg_pf={r['wf_avg_pf']}  "
                 f"ship={ship}")

        winner = max(ship_eligible, key=lambda x: x["pnl"]) if ship_eligible else None
        results[symbol] = {
            "baseline": baseline,
            "phase_a_top": top_a[:12],
            "finalists_wf": wf_results,
            "winner": winner,
        }
        if winner:
            _log(f"  [{symbol}] WINNER: {winner['tag']}  Δ=${winner['delta']:+.0f}  "
                 f"WF {winner['wf_pos']}/{WF_NUM_FOLDS}")
        else:
            _log(f"  [{symbol}] NO winner — keeping baseline.")

    # Phase D: stack winners — backtest all 3 symbols together
    _log("\n\n========= STACK D — verify no negative interaction =========")
    stack_winners = {s: results[s]["winner"]
                     for s in SYMBOLS if results.get(s, {}).get("winner")}
    if not stack_winners:
        _log("  No winners to stack.")
    else:
        # For stacking, we must apply ALL winners simultaneously across all 3 syms.
        # Run a single BT per symbol with all 3 overlays applied.
        # This is conservative — overlays affect only their own symbol's BT.
        for symbol in SYMBOLS:
            # Build per-symbol stacked args
            w = stack_winners.get(symbol)
            if w is None:
                # Use baseline (no overlay) — verify other syms' overlays don't leak
                args = (symbol, {}, None, set(), None, None)
            else:
                args = (symbol, w["dir_bias_cell"], w["min_q_cell"],
                        w["toxic_set"], w["range_filter"], None)
            with Pool(1) as pool:
                res = pool.apply(_bt_one, (args,))
            base = results[symbol]["baseline"]
            if "err" in res:
                _log(f"  [{symbol}] STACK ERR: {res['err'][:140]}")
                continue
            delta = res["pnl"] - base["pnl"]
            _log(f"  [{symbol}] STACK: trades={res['trades']:3d} PF={res['pf']:5.2f} "
                 f"PnL=${res['pnl']:+7.0f}  Δ=${delta:+.0f}  vs baseline=${base['pnl']:+.0f}")
            results.setdefault("stack", {})[symbol] = {
                "baseline_pnl": base["pnl"], "stacked_pnl": res["pnl"],
                "delta": delta, "stack_pf": res["pf"], "stack_trades": res["trades"],
            }

    # Final assembly
    final = {
        "_meta": {
            "ts": datetime.now().isoformat(),
            "symbols": SYMBOLS,
            "days": DAYS,
            "wf_num_folds": WF_NUM_FOLDS,
            "per_symbol_wf_fold_days": PER_SYMBOL_FOLD_DAYS,
            "per_symbol_fold_mode": PER_SYMBOL_FOLD_MODE,
            "ship_min_delta": MIN_DELTA,
            "wf_min_pos": WF_MIN_POS,
            "elapsed_sec": round(time.time() - t_start, 1),
        },
        "results": {},
    }
    for s in SYMBOLS:
        sr = results[s]
        if "err" in sr:
            final["results"][s] = {"err": sr["err"]}
            continue
        # Strip non-serializable bits
        def _serial(rec):
            return {k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
                    for k, v in rec.items()}
        final["results"][s] = {
            "baseline": sr["baseline"],
            "phase_a_top": [_serial(r) for r in sr["phase_a_top"]],
            "finalists_wf": [_serial(r) for r in sr["finalists_wf"]],
            "winner": _serial(sr["winner"]) if sr["winner"] else None,
        }
    if "stack" in results:
        final["stack_verify"] = results["stack"]

    OUT_JSON.write_text(json.dumps(final, indent=2, default=str))
    _log(f"\nJSON written: {OUT_JSON}")
    _log(f"ALL DONE in {time.time()-t_start:.0f}s")
    return final


if __name__ == "__main__":
    main()
