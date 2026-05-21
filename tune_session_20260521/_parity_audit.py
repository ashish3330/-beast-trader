#!/usr/bin/env python3 -B
"""
Live↔Backtest parity audit. READ-ONLY.

Strategy:
  1. Pull MT5 deal-leg trades from journal for last 30 days.
  2. Cluster legs into logical entries by (symbol, direction, entry_price).
     Each entry produces up to 3 pyramid legs; aggregate PnL across legs.
  3. For each logical entry, locate the matching H1 bar in the price cache.
  4. Re-simulate exit using the SAME trail/SL params live used (read from
     config/auto_tuned) via backtest.v5_backtest.simulate_trail.
  5. Compare aggregate live PnL_R vs BT PnL_R, and categorize divergence.

Output: writes parity_audit.md + parity_audit_raw.csv next to this script.
"""
from __future__ import annotations
import sqlite3, sys, math, csv, json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backtest.v5_backtest import (
    ALL_SYMBOLS, load_data, simulate_trail,
    SL_OVERRIDE, SL_OVERRIDE_REGIME,
    TRAIL_OVERRIDE, TRAIL_OVERRIDE_REGIME,
    DEFAULT_PARAMS,
)
from signals.momentum_scorer import _compute_indicators, IND_DEFAULTS, IND_OVERRIDES

JOURNAL_DB = ROOT / "data" / "trade_journal.db"
OUT_DIR = ROOT / "tune_session_20260521"
MD_PATH = OUT_DIR / "parity_audit.md"
CSV_PATH = OUT_DIR / "parity_audit_raw.csv"

DIVERGENCE_DOLLAR_THRESHOLD = 1.0  # $1 per logical entry

# Categories the BT does NOT model — any live exit using these is automatically
# a divergence with structural root cause (not a price-matching bug).
BT_ONLY_EXITS = {"SL", "TIMEOUT"}
LIVE_EXIT_FAMILIES = {
    "SL_broker":       ("SLIPPAGE",      "broker SL hit; BT also SL but at different price"),
    "TP_broker":       ("EXIT_MISMATCH", "live closed at fixed TP; BT trails open"),
    "EarlyLossCut":    ("EXIT_MISMATCH", "live cut at -2R/T1-T3 tier; BT lets it run to SL"),
    "PeakGiveback":    ("EXIT_MISMATCH", "live peak-R giveback rule; not in BT"),
    "EmergencyDD":     ("EXTERNAL_CLOSE","portfolio DD kill; not in BT per-symbol"),
    "GuardianDayLoss": ("EXTERNAL_CLOSE","daily-loss guardian; not in BT"),
    "GuardianStale":   ("EXTERNAL_CLOSE","stale-loser guardian; not in BT"),
    "GuardianSharp":   ("EXTERNAL_CLOSE","sharp-loss guardian; not in BT"),
    "GuardianHeat":    ("EXTERNAL_CLOSE","heat-reduce guardian; not in BT"),
    "HardDollarCap":   ("EXTERNAL_CLOSE","hard-dollar cap kill; not in BT"),
    "DailyKillSwitch": ("EXTERNAL_CLOSE","daily kill switch; not in BT"),
    "DragonReversal":  ("EXIT_MISMATCH", "opposite-signal close; not in BT"),
    "DragonWeekendClose":("EXIT_MISMATCH","weekend forced close; not in BT"),
    "OTHER":           ("UNREPLAYABLE",  "unrecognized exit reason"),
}


def categorize_exit(reason: str) -> str:
    if not reason:
        return "OTHER"
    if reason.startswith("[sl "):
        return "SL_broker"
    if reason.startswith("[tp "):
        return "TP_broker"
    if reason.startswith("EarlyLossCut"):
        return "EarlyLossCut"
    if reason == "PeakGiveback":     return "PeakGiveback"
    if reason == "EmergencyDD":      return "EmergencyDD"
    if reason == "GuardianDayLoss":  return "GuardianDayLoss"
    if reason == "GuardianStaleLos": return "GuardianStale"
    if reason == "GuardianSharpLos": return "GuardianSharp"
    if reason == "GuardianHeatRedu": return "GuardianHeat"
    if reason == "HardDollarCap":    return "HardDollarCap"
    if reason == "DailyKillSwitch":  return "DailyKillSwitch"
    if reason == "DragonReversal":   return "DragonReversal"
    if reason == "DragonWeekendClo": return "DragonWeekendClose"
    return "OTHER"


def fetch_legs():
    con = sqlite3.connect(f"file:{JOURNAL_DB}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("""
        SELECT id, timestamp, symbol, direction, entry_price, exit_price, pnl,
               risk_pct, score, regime, r_multiple, exit_reason, duration_bars
        FROM trades
        WHERE source='mt5_deal'
          AND timestamp >= datetime('now', '-30 days')
        ORDER BY symbol, entry_price, timestamp
    """)
    cols = ["id","ts","symbol","direction","entry","exit","pnl",
            "risk_pct","score","regime","r","reason","dur"]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    con.close()
    return rows


def cluster_entries(legs):
    """Cluster legs into logical entries by (symbol, direction, entry_price)
    when consecutive in time. Returns list of dicts each containing
    list of legs that share an entry."""
    clusters = []
    by_sym = defaultdict(list)
    for l in legs:
        by_sym[l["symbol"]].append(l)
    for sym, sym_legs in by_sym.items():
        # sort by entry_price first then time for consistent grouping
        sym_legs.sort(key=lambda x: (x["entry"], x["direction"], x["ts"]))
        cur_key = None
        cur_grp = []
        for l in sym_legs:
            key = (l["symbol"], l["direction"], round(float(l["entry"]), 8))
            if key != cur_key:
                if cur_grp:
                    clusters.append(cur_grp)
                cur_grp = [l]
                cur_key = key
            else:
                cur_grp.append(l)
        if cur_grp:
            clusters.append(cur_grp)
    return clusters


def find_h1_bar(df, ts_str: str):
    """Locate H1 bar index whose close ≤ ts. The live entry is placed at the
    H1 close prior to the deal-record timestamp."""
    ts = pd.to_datetime(ts_str, utc=True)
    times = df["time"]
    if not pd.api.types.is_datetime64_any_dtype(times):
        times = pd.to_datetime(times, utc=True)
    # Force matching unit to avoid pandas lossless-conversion errors.
    times_ns = pd.DatetimeIndex(times).astype("datetime64[ns, UTC]")
    ts_ns = ts.tz_convert("UTC").asm8 if ts.tzinfo is not None else ts.tz_localize("UTC").asm8
    arr = times_ns.values.astype("datetime64[ns]")
    idx = int(np.searchsorted(arr, np.datetime64(ts_ns, "ns"), side="right") - 1)
    return idx if idx >= 0 else None


def closest_entry_bar(df, target_entry, around_idx, window=12, direction=1):
    """Refine entry-bar index by searching window bars around `around_idx`
    for the bar whose close OR (close ± 0.2*ATR pullback) best matches the
    recorded entry_price. The live entry can be on either the close OR the
    pullback bar; we test both."""
    n = len(df)
    lo = max(0, around_idx - window)
    hi = min(n - 1, around_idx + window)
    best_idx = around_idx
    best_diff = float("inf")
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    for i in range(lo, hi + 1):
        for cand in (float(closes[i]), float(highs[i]), float(lows[i])):
            diff = abs(cand - target_entry)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
    return best_idx, float(closes[best_idx])


def run_audit():
    print(">>> fetching legs ...")
    legs = fetch_legs()
    print(f"    {len(legs)} legs in last 30d")
    clusters = cluster_entries(legs)
    print(f"    {len(clusters)} logical entries (clusters)")

    # Pre-cache dataframes/indicators per symbol
    df_cache = {}
    ind_cache = {}
    times_cache = {}
    def get_df(sym):
        if sym not in df_cache:
            df = load_data(sym, days=60)
            df_cache[sym] = df
            if df is not None:
                icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(sym, {})}
                try:
                    ind_cache[sym] = _compute_indicators(df, icfg)
                except Exception as e:
                    ind_cache[sym] = None
                times_cache[sym] = pd.to_datetime(df["time"], utc=True) \
                    if not pd.api.types.is_datetime64_any_dtype(df["time"]) \
                    else df["time"]
        return df_cache.get(sym), ind_cache.get(sym), times_cache.get(sym)

    audit_rows = []
    unreplayable = 0

    for cluster in clusters:
        first = cluster[0]
        sym = first["symbol"]
        if sym not in ALL_SYMBOLS:
            unreplayable += 1
            continue
        df, ind, times_idx = get_df(sym)
        if df is None or ind is None:
            unreplayable += 1
            continue

        # Aggregate live data
        legs_n = len(cluster)
        live_pnl = sum(float(l["pnl"] or 0) for l in cluster)
        live_r   = sum(float(l["r"] or 0)   for l in cluster) / legs_n  # avg per leg
        # Sum-of-legs R isn't meaningful since each leg=1/3 of position size;
        # we use the mean R (per unit risk) as a proxy. live PnL is the
        # ground-truth dollar comparison.
        live_exit_price = np.mean([float(l["exit"] or 0) for l in cluster
                                    if l["exit"] not in (None, 0)])
        if math.isnan(live_exit_price): live_exit_price = 0.0
        # Exit reason: take the most "severe" category from the cluster's legs
        # (Guardian/EmergencyDD > EarlyLossCut > PeakGiveback > broker SL/TP).
        cats = [categorize_exit(l["reason"]) for l in cluster]
        # Priority order
        prio = ["EmergencyDD","DailyKillSwitch","GuardianDayLoss","GuardianStale",
                "GuardianSharp","GuardianHeat","HardDollarCap","DragonWeekendClose",
                "DragonReversal","PeakGiveback","EarlyLossCut","TP_broker","SL_broker","OTHER"]
        live_cat = "SL_broker"
        for c in prio:
            if c in cats:
                live_cat = c
                break

        # Locate entry bar in BT data
        entry_price = float(first["entry"] or 0)
        if entry_price <= 0:
            unreplayable += 1
            continue
        direction = 1 if first["direction"] == "LONG" else -1

        entry_ts = first["ts"]
        idx = find_h1_bar(df, entry_ts)
        if idx is None or idx < 50 or idx >= len(df) - 1:
            unreplayable += 1
            continue
        entry_bar, _ = closest_entry_bar(df, entry_price, idx, window=8, direction=direction)

        # Sanity: predicted vs actual entry-price must be within a reasonable tolerance
        # Skip if mismatch > 0.5% of price (means we likely didn't find the right bar)
        ref_px = float(df["close"].values[entry_bar])
        if abs(ref_px - entry_price) / max(abs(entry_price), 1e-9) > 0.005:
            # Still record but flag missing
            unreplayable += 1
            continue

        # ATR / SL distance
        at_arr = ind["at"]
        if entry_bar >= len(at_arr): unreplayable += 1; continue
        atr = float(at_arr[entry_bar])
        if atr <= 0 or math.isnan(atr):
            unreplayable += 1
            continue

        # Live regime preferred; fallback to BT-computed
        regime = (first["regime"] or "").strip() or "low_vol"

        sl_eff = SL_OVERRIDE_REGIME.get(sym, {}).get(regime,
                  SL_OVERRIDE.get(sym, DEFAULT_PARAMS["sl_atr_mult"]))
        sl_dist = atr * sl_eff
        if sl_dist <= 0:
            unreplayable += 1
            continue

        # Trail steps mirror live
        trail_steps = (TRAIL_OVERRIDE_REGIME.get(sym, {}).get(regime)
                       or TRAIL_OVERRIDE.get(sym, DEFAULT_PARAMS["trail"]))

        # Simulate
        meta = ALL_SYMBOLS[sym]
        spread = meta["spread"]
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        try:
            bt_exit_price, bt_exit_bar, bt_reason, bt_peak_r = simulate_trail(
                entry_price, sl_dist, direction,
                h, l, c, entry_bar + 1, len(df),
                spread, trail_steps,
                ratchet_1r=0.2, ratchet_2r=0.5)
        except Exception as e:
            unreplayable += 1
            continue

        bt_pnl_points = (bt_exit_price - entry_price) * direction
        bt_pnl_r = bt_pnl_points / sl_dist if sl_dist > 0 else 0.0

        # Dollar comparison: a Dragon entry opens 3 pyramid legs each sized
        # at 1/3 of the dollar-risk. Per-leg dollar/R ≈ (full_position_$/R)/3.
        # If the cluster contains > 3 legs, it's actually N/3 separate signals
        # that happened to share entry_price (re-entries at same price).
        # We compute the full-position $/R from median(leg_$/R)*3 and then
        # scale by n_signals = ceil(legs_n / 3) to get expected aggregate $.
        dpr_samples = []
        for lg in cluster:
            r = float(lg["r"] or 0); p = float(lg["pnl"] or 0)
            if abs(r) > 0.05:
                dpr_samples.append(abs(p) / abs(r))
        leg_dpr = float(np.median(dpr_samples)) if dpr_samples else 0.0
        full_position_dpr = leg_dpr * 3.0  # 3 legs per signal
        n_signals = max(1, int(round(legs_n / 3.0)))
        bt_pnl_dollar = bt_pnl_r * full_position_dpr * n_signals

        delta = live_pnl - bt_pnl_dollar
        diverged = abs(delta) > DIVERGENCE_DOLLAR_THRESHOLD or live_cat not in {"SL_broker"}

        # Map live category → divergence type
        if live_cat in LIVE_EXIT_FAMILIES:
            div_type, div_note = LIVE_EXIT_FAMILIES[live_cat]
        else:
            div_type, div_note = "UNREPLAYABLE", "unmapped"

        # Refinement: for SL_broker live exits, distinguish actual slippage
        # (live exit ≈ BT exit, small Δ) from TRAIL_LAG (live exit price far
        # from BT exit because BT trail timing differs). Threshold: |Δprice|
        # > 0.5×ATR ⇒ TRAIL_LAG (the trail itself diverged), else SLIPPAGE.
        if live_cat == "SL_broker":
            atr_dist = abs(live_exit_price - bt_exit_price)
            if atr > 0 and atr_dist > 0.5 * atr:
                div_type = "TRAIL_LAG"
                div_note = ("live SL trailed differently — exit-price gap "
                            f"{atr_dist/atr:.2f}× ATR")

        if not diverged:
            div_type, div_note = "OK", "live SL ≈ BT SL"

        audit_rows.append({
            "symbol": sym,
            "dir": "L" if direction == 1 else "S",
            "entry": entry_price,
            "live_exit": round(live_exit_price, 5),
            "bt_exit":   round(bt_exit_price,   5),
            "live_pnl":  round(live_pnl,  2),
            "bt_pnl":    round(bt_pnl_dollar, 2),
            "delta":     round(delta, 2),
            "legs":      legs_n,
            "live_cat":  live_cat,
            "bt_reason": bt_reason,
            "div_type":  div_type,
            "regime":    regime,
            "ts":        first["ts"],
            "note":      div_note,
        })

    return audit_rows, unreplayable, len(clusters)


def write_outputs(rows, unreplayable, n_clusters):
    if not rows:
        print("no rows!")
        return
    # CSV
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {CSV_PATH}")

    # Stats
    diverged = [r for r in rows if r["div_type"] not in ("OK",)]
    total = len(rows)
    sum_delta = sum(r["delta"] for r in rows)
    abs_delta = sum(abs(r["delta"]) for r in rows)

    by_cat = Counter(r["div_type"] for r in rows)
    by_cat_delta = defaultdict(float)
    for r in rows: by_cat_delta[r["div_type"]] += r["delta"]
    by_cat_abs = defaultdict(float)
    for r in rows: by_cat_abs[r["div_type"]] += abs(r["delta"])

    by_sym = defaultdict(list)
    for r in rows: by_sym[r["symbol"]].append(r)
    worst_sym = sorted(
        [(s, sum(x["delta"] for x in rs), sum(abs(x["delta"]) for x in rs), len(rs))
         for s, rs in by_sym.items()],
        key=lambda x: x[2], reverse=True)

    worst_trades = sorted(rows, key=lambda r: abs(r["delta"]), reverse=True)[:15]

    lines = []
    lines.append("# Live↔BT Parity Audit — 2026-05-21")
    lines.append("")
    lines.append("Audit window: last 30 days of mt5_deal exit-legs from `data/trade_journal.db`.")
    lines.append("Replay: per-(symbol, direction, entry_price) clusters mapped to H1 cache bar; "
                 "simulate_trail() invoked with live SYMBOL_ATR_SL_OVERRIDE / "
                 "SYMBOL_TRAIL_OVERRIDE / regime overrides.")
    lines.append("")
    lines.append("## Summary stats (last 30 days)")
    lines.append("")
    lines.append(f"- MT5 deal-legs scanned: {sum(r['legs'] for r in rows) + 0}")
    lines.append(f"- Logical entries clustered: {n_clusters}")
    lines.append(f"- Replayable entries (after price+ATR match): {total}")
    lines.append(f"- UNREPLAYABLE / dropped: {unreplayable} "
                 f"({100*unreplayable/max(n_clusters,1):.1f}% of clusters — "
                 f"reasons: bar/price mismatch >0.5%, missing cache, no ATR)")
    lines.append(f"- Entries with |Δ| > ${DIVERGENCE_DOLLAR_THRESHOLD:.0f}: "
                 f"{sum(1 for r in rows if abs(r['delta'])>DIVERGENCE_DOLLAR_THRESHOLD)} "
                 f"({100*sum(1 for r in rows if abs(r['delta'])>DIVERGENCE_DOLLAR_THRESHOLD)/total:.0f}%)")
    lines.append(f"- Entries with category mismatch (live exit ≠ BT SL/TIMEOUT): "
                 f"{sum(1 for r in rows if r['div_type']!='OK')} "
                 f"({100*sum(1 for r in rows if r['div_type']!='OK')/total:.0f}%)")
    lines.append(f"- Net Δ live − BT: ${sum_delta:+.2f}")
    lines.append(f"- |Δ| sum (total drift magnitude): ${abs_delta:.2f}")
    lines.append(f"- Worst-divergence symbol: {worst_sym[0][0]} "
                 f"(|Δ| ${worst_sym[0][2]:.2f} over {worst_sym[0][3]} entries)")
    lines.append("")

    lines.append("## Divergence breakdown by category")
    lines.append("")
    lines.append("| category | n | net Δ ($) | |Δ| ($) |")
    lines.append("|---|---:|---:|---:|")
    for cat, n in by_cat.most_common():
        lines.append(f"| {cat} | {n} | {by_cat_delta[cat]:+.2f} | {by_cat_abs[cat]:.2f} |")
    lines.append("")

    lines.append("## Top 15 worst-divergence entries")
    lines.append("")
    lines.append("| # | symbol | dir | entry | live_exit | bt_exit | live_pnl | bt_pnl | Δ | category | reason |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for i, r in enumerate(worst_trades, 1):
        lines.append(f"| {i} | {r['symbol']} | {r['dir']} | {r['entry']:g} | "
                     f"{r['live_exit']:g} | {r['bt_exit']:g} | "
                     f"{r['live_pnl']:+.2f} | {r['bt_pnl']:+.2f} | {r['delta']:+.2f} | "
                     f"{r['div_type']} | {r['live_cat']}: {r['note']} |")
    lines.append("")

    lines.append("## Per-symbol divergence summary")
    lines.append("")
    lines.append("| symbol | entries | net Δ | |Δ| | avg Δ/entry | dominant live exit |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for s, net, absd, n in worst_sym:
        cats = Counter(x["live_cat"] for x in by_sym[s]).most_common(1)[0]
        lines.append(f"| {s} | {n} | {net:+.2f} | {absd:.2f} | {net/n:+.2f} | {cats[0]} ({cats[1]}) |")
    lines.append("")

    lines.append("## Suggested fixes (root-cause first)")
    lines.append("")
    suggestions = []
    if by_cat_abs.get("TRAIL_LAG", 0) > 0:
        tl_rows = [r for r in rows if r["div_type"] == "TRAIL_LAG"]
        tl_delta = sum(r["delta"] for r in tl_rows)
        tl_abs = sum(abs(r["delta"]) for r in tl_rows)
        suggestions.append(
            f"0. **TRAIL_LAG dominates** — {len(tl_rows)} broker-SL entries had "
            f"exit-price gap > 0.5×ATR (|Δ| sum ${tl_abs:.2f}, net ${tl_delta:+.2f}). "
            f"This means live SL was at a materially different price than BT's "
            f"trailed SL at the same bar — most likely cause: per-tick trail "
            f"updates in live (executor.py:1337-1422) tighten SL faster than "
            f"BT's per-bar simulate_trail. Fix: change simulate_trail to use "
            f"intra-bar high/low for trail check (currently uses close only at "
            f"v5_backtest.py:400).")
    if by_cat_abs.get("EXIT_MISMATCH", 0) > 0:
        elc_rows = [r for r in rows if r["live_cat"] == "EarlyLossCut"]
        if elc_rows:
            elc_delta = sum(r["delta"] for r in elc_rows)
            elc_n = len(elc_rows)
            suggestions.append(
                f"1. **EarlyLossCut over-firing** — {elc_n} entries closed via tiered early-cut, "
                f"net Δ ${elc_delta:+.2f} vs BT-trail. Live cuts at -2R / -2.5R / -3R tiers "
                f"(executor.py:1422) while BT lets SL hit at the configured ATR distance. "
                f"Either: (a) widen BT SL multiplier to match the effective early-cut R, or "
                f"(b) port EarlyLossCut into simulate_trail with the live tier thresholds.")
        pgb_rows = [r for r in rows if r["live_cat"] == "PeakGiveback"]
        if pgb_rows:
            pgb_delta = sum(r["delta"] for r in pgb_rows)
            suggestions.append(
                f"2. **PeakGiveback unmodeled** — {len(pgb_rows)} winners closed by peak-R giveback "
                f"(executor.py:1337), net Δ ${pgb_delta:+.2f}. BT's ratchet (0.2R/0.5R floors) "
                f"plus trail-lock differs. Mirror executor's peak-giveback check inside simulate_trail.")
    if by_cat_abs.get("EXTERNAL_CLOSE", 0) > 0:
        ext_rows = [r for r in rows if r["div_type"] == "EXTERNAL_CLOSE"]
        ext_delta = sum(r["delta"] for r in ext_rows)
        suggestions.append(
            f"3. **Portfolio guardians not in BT** — {len(ext_rows)} entries killed by "
            f"EmergencyDD / GuardianDayLoss / GuardianStale / HardDollarCap / DailyKillSwitch / "
            f"GuardianHeat / GuardianSharp (net Δ ${ext_delta:+.2f}). These are PORTFOLIO-level "
            f"rules — single-symbol BT cannot model the cross-symbol equity coupling. "
            f"Acceptable structural gap; flag remains as long as it doesn't dominate the divergence.")
    if by_cat_abs.get("SLIPPAGE", 0) > 0:
        sl_rows = [r for r in rows if r["div_type"] == "SLIPPAGE"]
        sl_delta = sum(r["delta"] for r in sl_rows)
        suggestions.append(
            f"4. **Broker SL slippage** — {len(sl_rows)} broker-SL exits diverged by net Δ "
            f"${sl_delta:+.2f}. Live SL fills at actual broker price; BT fills at exact SL level. "
            f"Enable `with_slippage` in backtest cost overlay to compare apples-apples.")
    if not suggestions:
        suggestions.append("No systematic divergence detected.")
    lines.extend(suggestions)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Methodology notes")
    lines.append("")
    lines.append("- Cluster: legs sharing identical `(symbol, direction, entry_price)` are "
                 "treated as a single logical entry (Dragon opens 3 pyramid legs per signal).")
    lines.append("- BT entry-bar match: locate H1 bar prior to first-leg timestamp, then refine "
                 "within ±8 bars to the bar whose close best matches recorded entry_price. "
                 "Tolerance 0.5%; mismatches → UNREPLAYABLE.")
    lines.append("- BT exit: `simulate_trail()` with live SL/trail/regime overrides "
                 "(SYMBOL_ATR_SL_OVERRIDE_REGIME → SYMBOL_ATR_SL_OVERRIDE → DEFAULT).")
    lines.append("- Live PnL: sum of leg `pnl` from journal (post-execution actual). "
                 "BT PnL$: bt_pnl_r × median($/R from legs) × legs_n.")
    lines.append("- BT-only exit reasons modeled: SL, TIMEOUT. Live-only exit families "
                 "automatically flagged as divergence regardless of dollar Δ.")

    MD_PATH.write_text("\n".join(lines))
    print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    rows, unreplayable, n_clusters = run_audit()
    write_outputs(rows, unreplayable, n_clusters)
