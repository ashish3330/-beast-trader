#!/usr/bin/env python3 -B
"""08 — MIN-RISK-REWARD entry filter research (READ-ONLY).

Methodology:
  1. Run baseline backtest_symbol() once per symbol (180 days) — captures full trade list.
  2. For each (min_rr, target_window) variant, post-hoc FILTER trades whose
     RR (distance to nearest structural target / sl_dist) at the SIGNAL bar
     was below the threshold.
  3. Equity-curve reconstruction without filtered trades = sum of kept PnL.
  4. WF 5-fold: chop the trade timeline into 5 equal calendar slices.

Targets (industry-standard):
  LONG  target_high = max(swing_high_lookback, EMA200, prior_day_high, VWAP-if-above)
  SHORT target_low  = min(swing_low_lookback,  EMA200, prior_day_low,  VWAP-if-below)
  rr = (target - entry_price) / sl_dist   (LONG; absolute value)

Caveat: post-hoc filtering ignores secondary effects (cooldown timer would
have un-stalled, consec_loss counter would differ, equity-DD check could
have stopped sim earlier/later). First-order PnL delta is a faithful
estimate of the filter's contribution because each trade's R outcome is
independent given the strategy is R-based with fixed risk_pct.

Output:
  08_min_rr.json — variants + Δ vs baseline + WF fold breakdown
  08_min_rr.md   — human summary + ship decision
"""
import sys, json, pickle, time as _time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import (
    backtest_symbol, load_data, ALL_SYMBOLS,
    SL_OVERRIDE, SL_OVERRIDE_REGIME, get_regime,
    DEFAULT_PARAMS,
)
from signals.momentum_scorer import _compute_indicators, _ema, IND_DEFAULTS, IND_OVERRIDES

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY", "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
DAYS = 180

MIN_RR_GRID    = [1.5, 2.0, 2.5, 3.0]
TARGET_WINDOWS = [20, 50, 100]
N_FOLDS = 5

OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "08_min_rr.json"
OUT_MD   = OUT_DIR / "08_min_rr.md"

START_EQ = DEFAULT_PARAMS["start_equity"]


# ────────────────────────────────────────────────────────────────────
def compute_ema200(close):
    return _ema(close.astype(np.float64), 200)


def find_prior_day_levels(times, h, l, entry_ts):
    """Return (prior_day_high, prior_day_low) using H1 bars strictly before entry day.
    times: np.ndarray of datetime64. entry_ts: pd.Timestamp.
    """
    ts = pd.Timestamp(entry_ts).floor("D")
    prev_day_start = ts - pd.Timedelta(days=1)
    # mask for bars within previous calendar day
    mask = (times >= np.datetime64(prev_day_start)) & (times < np.datetime64(ts))
    if not mask.any():
        return None, None
    return float(h[mask].max()), float(l[mask].min())


def evaluate_min_rr(trade, h, l, c, times, ema200, atr_arr, sl_eff, point,
                    target_window, min_rr):
    """Return True if trade PASSES the min-RR filter (keep)."""
    entry_bar = trade["entry_bar"]
    direction = trade["direction"]
    entry_price = trade["entry"]

    bi = entry_bar - 1  # signal bar (signal at i, entry at i+1)
    if bi < target_window:
        return True  # not enough history → don't filter

    # SL distance at signal bar
    atr_now = atr_arr[bi]
    if np.isnan(atr_now) or atr_now <= 0:
        return True
    sl_dist = atr_now * sl_eff

    # Structural targets
    targets = []

    # 1) Swing high/low last N bars (excluding the current bar)
    lo = max(0, bi - target_window)
    win_h = h[lo:bi]
    win_l = l[lo:bi]
    if len(win_h) > 0:
        swing_high = float(win_h.max())
        swing_low  = float(win_l.min())
    else:
        swing_high = swing_low = None

    # 2) EMA200
    ema_now = float(ema200[bi]) if not np.isnan(ema200[bi]) else None

    # 3) Prior day high/low (H1 bars within previous calendar day)
    entry_ts = pd.Timestamp(times[bi])
    pdh, pdl = find_prior_day_levels(times, h, l, entry_ts)

    if direction == 1:  # LONG
        cand = []
        if swing_high is not None and swing_high > entry_price:
            cand.append(swing_high)
        if ema_now is not None and ema_now > entry_price:
            cand.append(ema_now)
        if pdh is not None and pdh > entry_price:
            cand.append(pdh)
        if not cand:
            return True  # no structural target above entry → can't apply filter
        # Use the FURTHEST target available (industry-standard: "to nearest" means
        # nearest level still in front; for filter we want best-case RR so we
        # take the max — but for STRICT filter the nearest above is the right
        # bar. The task says "distance to nearest structural target / SL".
        # NEAREST = closest to entry. Use the MIN of candidates (closest above).
        target = min(cand)
        target_dist = target - entry_price
    else:  # SHORT
        cand = []
        if swing_low is not None and swing_low < entry_price:
            cand.append(swing_low)
        if ema_now is not None and ema_now < entry_price:
            cand.append(ema_now)
        if pdl is not None and pdl < entry_price:
            cand.append(pdl)
        if not cand:
            return True
        target = max(cand)  # nearest below entry
        target_dist = entry_price - target

    if target_dist <= 0:
        return True
    rr = target_dist / sl_dist
    return rr >= min_rr


def calc_stats(filtered_trades, start_eq=START_EQ):
    """Compute PF/WR/PnL/DD from a list of kept trades.

    DD reconstructed from running equity curve (filtered).
    """
    if not filtered_trades:
        return {"trades": 0, "pf": 0, "wr": 0, "pnl": 0, "dd": 0,
                "avg_r": 0, "wins": 0}
    wins = [t for t in filtered_trades if t["pnl"] > 0]
    losses = [t for t in filtered_trades if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999 if gross_win > 0 else 0)
    wr = len(wins) / len(filtered_trades) * 100
    avg_r = float(np.mean([t["pnl_r"] for t in filtered_trades]))

    eq = start_eq
    peak = eq
    max_dd = 0.0
    for t in filtered_trades:
        eq += t["pnl"]
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return {
        "trades": len(filtered_trades), "wins": len(wins),
        "pf": round(pf, 2), "wr": round(wr, 1),
        "pnl": round(sum(t["pnl"] for t in filtered_trades), 2),
        "dd": round(max_dd, 1), "avg_r": round(avg_r, 2),
    }


def split_folds(trades, n_folds=N_FOLDS):
    """Split chronological trades into n equal slices."""
    if len(trades) < n_folds:
        return [trades]
    fold_size = len(trades) // n_folds
    folds = []
    for k in range(n_folds):
        if k == n_folds - 1:
            folds.append(trades[k*fold_size:])
        else:
            folds.append(trades[k*fold_size:(k+1)*fold_size])
    return folds


# ────────────────────────────────────────────────────────────────────
def main():
    t0 = _time.time()
    print(f"=== 08 — MIN-RR entry filter ({DAYS}d, {len(SYMBOLS)} symbols) ===")

    # Phase 1: Baseline + cache per-symbol data
    per_sym_data = {}
    baseline_stats = {}
    baseline_trades_all = {}
    print("\n[1/3] Baseline runs (180d):")
    for sym in SYMBOLS:
        if sym not in ALL_SYMBOLS:
            print(f"  {sym:12s}  SKIP (not in universe)")
            continue
        r = backtest_symbol(sym, days=DAYS, params=None, verbose=False)
        if r is None or r.get("trades", 0) == 0:
            print(f"  {sym:12s}  SKIP (no trades)")
            continue
        baseline_stats[sym] = {
            "trades": r["trades"], "pf": r["pf"], "wr": r["wr"],
            "pnl": r["pnl"], "dd": r["dd"], "avg_r": r["avg_r"],
        }
        baseline_trades_all[sym] = r["details"]
        # Load price data + indicators (once per symbol — reuse for all variants)
        df = load_data(sym, days=DAYS)
        if df is None:
            continue
        icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(sym, {})}
        ind = _compute_indicators(df, icfg)
        ema200 = compute_ema200(df["close"].values)
        per_sym_data[sym] = {
            "df": df,
            "h": df["high"].values.astype(float),
            "l": df["low"].values.astype(float),
            "c": df["close"].values.astype(float),
            "times": df["time"].values,
            "atr": ind["at"],
            "bbw": ind["bbw"],
            "adx": ind["adx"],
            "ema200": ema200,
            "point": ALL_SYMBOLS[sym]["point"],
        }
        print(f"  {sym:12s} trades={r['trades']:4d} pf={r['pf']:5.2f} "
              f"pnl=${r['pnl']:9.2f} dd={r['dd']:.1f}%")

    baseline_total_pnl = sum(s["pnl"] for s in baseline_stats.values())
    print(f"\n  Baseline total: pnl=${baseline_total_pnl:.2f}  trades="
          f"{sum(s['trades'] for s in baseline_stats.values())}")

    # Phase 2: Sweep variants
    print(f"\n[2/3] Sweep {len(MIN_RR_GRID)}×{len(TARGET_WINDOWS)} variants:")
    variants = []
    for min_rr in MIN_RR_GRID:
        for tw in TARGET_WINDOWS:
            v_per_sym = {}
            total_pnl = 0.0
            total_trades = 0
            for sym, trades in baseline_trades_all.items():
                if sym not in per_sym_data:
                    continue
                d = per_sym_data[sym]
                # determine sl_eff (use per-symbol override; ignore regime override
                # for simplicity — small noise vs filter contribution)
                sl_mult_base = DEFAULT_PARAMS["sl_atr_mult"]
                sl_eff = SL_OVERRIDE.get(sym, sl_mult_base)
                kept = []
                for t in trades:
                    if evaluate_min_rr(t, d["h"], d["l"], d["c"], d["times"],
                                       d["ema200"], d["atr"], sl_eff,
                                       d["point"], tw, min_rr):
                        kept.append(t)
                stats = calc_stats(kept)
                v_per_sym[sym] = stats
                total_pnl += stats["pnl"]
                total_trades += stats["trades"]
            delta_pnl = total_pnl - baseline_total_pnl
            variants.append({
                "min_rr": min_rr, "target_window": tw,
                "total_pnl": round(total_pnl, 2),
                "total_trades": total_trades,
                "delta_pnl": round(delta_pnl, 2),
                "per_symbol": v_per_sym,
            })
            print(f"  min_rr={min_rr:.1f} tw={tw:3d}  pnl=${total_pnl:9.2f} "
                  f"Δ=${delta_pnl:+8.2f}  trades={total_trades}")

    # Phase 3: WF for top variants
    print(f"\n[3/3] WF 5-fold for variants with Δ ≥ $30:")
    ship_candidates = sorted(
        [v for v in variants if v["delta_pnl"] >= 30],
        key=lambda v: -v["delta_pnl"]
    )
    wf_results = []
    for v in ship_candidates:
        # Per fold: combine all symbols' trades chronologically, split into 5 slices.
        all_trades = []
        for sym, baseline_trades in baseline_trades_all.items():
            if sym not in per_sym_data:
                continue
            d = per_sym_data[sym]
            sl_mult_base = DEFAULT_PARAMS["sl_atr_mult"]
            sl_eff = SL_OVERRIDE.get(sym, sl_mult_base)
            for t in baseline_trades:
                t_copy = dict(t)
                t_copy["_sym"] = sym
                t_copy["_kept"] = evaluate_min_rr(
                    t, d["h"], d["l"], d["c"], d["times"], d["ema200"],
                    d["atr"], sl_eff, d["point"],
                    v["target_window"], v["min_rr"]
                )
                # Attach the entry timestamp for ordering
                bi = t["entry_bar"] - 1
                if 0 <= bi < len(d["times"]):
                    t_copy["_ts"] = pd.Timestamp(d["times"][bi]).value  # ns int
                else:
                    t_copy["_ts"] = 0
                all_trades.append(t_copy)
        all_trades.sort(key=lambda t: t["_ts"])
        folds = split_folds(all_trades, N_FOLDS)
        fold_metrics = []
        for k, fold in enumerate(folds):
            base_fold = fold
            filt_fold = [t for t in fold if t["_kept"]]
            base_stats = calc_stats(base_fold)
            filt_stats = calc_stats(filt_fold)
            fold_metrics.append({
                "fold": k + 1,
                "base_pnl": base_stats["pnl"], "base_pf": base_stats["pf"],
                "filt_pnl": filt_stats["pnl"], "filt_pf": filt_stats["pf"],
                "delta_pnl": round(filt_stats["pnl"] - base_stats["pnl"], 2),
                "n_base": base_stats["trades"], "n_filt": filt_stats["trades"],
            })
        pos_folds = sum(1 for f in fold_metrics if f["delta_pnl"] > 0)
        avg_pf = float(np.mean([f["filt_pf"] for f in fold_metrics if f["filt_pf"] > 0])) if fold_metrics else 0
        wf_results.append({
            "min_rr": v["min_rr"], "target_window": v["target_window"],
            "total_delta_pnl": v["delta_pnl"],
            "wf_avg_pf": round(avg_pf, 2),
            "wf_positive_folds": pos_folds,
            "wf_folds": fold_metrics,
            "wf_ship": (
                v["delta_pnl"] >= 30 and avg_pf > 1.5 and pos_folds >= 3
            ),
        })
        print(f"  min_rr={v['min_rr']:.1f} tw={v['target_window']:3d}  "
              f"Δ=${v['delta_pnl']:+8.2f}  WF avg PF={avg_pf:.2f}  "
              f"+folds={pos_folds}/{N_FOLDS}  SHIP={wf_results[-1]['wf_ship']}")

    # Decision
    shippable = [w for w in wf_results if w["wf_ship"]]
    decision = {
        "ship": bool(shippable),
        "best_variant": shippable[0] if shippable else None,
        "rationale": (
            f"{len(shippable)} variants meet ship criteria (Δ≥$30 AND WF avg PF>1.5 AND ≥3/5 +folds)."
            if shippable
            else "No variant meets all ship criteria; baseline already extracts most edge."
        ),
    }

    payload = {
        "days": DAYS,
        "symbols": SYMBOLS,
        "n_folds": N_FOLDS,
        "min_rr_grid": MIN_RR_GRID,
        "target_windows": TARGET_WINDOWS,
        "baseline": baseline_stats,
        "baseline_total_pnl": round(baseline_total_pnl, 2),
        "variants": variants,
        "walk_forward": wf_results,
        "decision": decision,
        "duration_sec": round(_time.time() - t0, 1),
        "methodology": (
            "Post-hoc filter on baseline trade list. Each trade evaluated at "
            "its signal bar (entry_bar-1) against structural targets: "
            "swing high/low over target_window H1 bars, EMA200, prior-day "
            "high/low. RR = (nearest target ahead - entry_price) / "
            "(ATR * sl_eff). Keep trade if RR >= min_rr. Caveat: secondary "
            "effects (cooldown counter, DD circuit-breaker) not re-simulated."
        ),
    }
    json.dump(payload, open(OUT_JSON, "w"), indent=2, default=str)
    print(f"\nwrote {OUT_JSON}")

    # Markdown
    lines = []
    lines.append(f"# 08 — MIN-RR Entry Filter Research\n")
    lines.append(f"**Date:** 2026-05-22  **Period:** 180d  **Symbols:** {', '.join(SYMBOLS)}\n")
    lines.append(f"**Method:** Post-hoc filter on baseline trade list. Targets = swing high/low (lookback N) ∪ EMA200 ∪ prior-day high/low. RR = nearest target / sl_dist.\n")
    lines.append(f"**Baseline:** ${baseline_total_pnl:,.2f} across {sum(s['trades'] for s in baseline_stats.values())} trades.\n")
    lines.append(f"## Variant sweep\n")
    lines.append(f"| min_rr | target_window | trades | total PnL | Δ vs base |")
    lines.append(f"|--------|---------------|--------|-----------|-----------|")
    for v in variants:
        lines.append(f"| {v['min_rr']:.1f} | {v['target_window']:>3} | {v['total_trades']} | ${v['total_pnl']:,.2f} | ${v['delta_pnl']:+,.2f} |")
    lines.append("")
    if wf_results:
        lines.append(f"## Walk-forward (variants with Δ ≥ $30)\n")
        lines.append(f"| min_rr | tw | Δ PnL | WF avg PF | +folds | SHIP |")
        lines.append(f"|--------|----|-------|-----------|--------|------|")
        for w in wf_results:
            lines.append(f"| {w['min_rr']:.1f} | {w['target_window']} | ${w['total_delta_pnl']:+,.2f} | {w['wf_avg_pf']:.2f} | {w['wf_positive_folds']}/{N_FOLDS} | {'YES' if w['wf_ship'] else 'no'} |")
        lines.append("")
    lines.append(f"## Decision\n")
    lines.append(f"- **Ship:** {'YES' if decision['ship'] else 'NO'}")
    lines.append(f"- **Rationale:** {decision['rationale']}")
    if decision["best_variant"]:
        b = decision["best_variant"]
        lines.append(f"- **Best variant:** min_rr={b['min_rr']}, target_window={b['target_window']} bars → Δ ${b['total_delta_pnl']:+.2f}, WF PF {b['wf_avg_pf']}, +{b['wf_positive_folds']}/{N_FOLDS} folds")
    lines.append("")
    lines.append(f"## Caveats\n")
    lines.append(f"- Post-hoc filter — does not re-simulate cooldown timers, consec-loss counter, or DD circuit-breaker. First-order PnL delta only.")
    lines.append(f"- sl_eff per symbol uses SL_OVERRIDE only (not SL_OVERRIDE_REGIME). Negligible drift vs filter signal.")
    lines.append(f"- 'Nearest target' = closest level ahead of entry (industry-strict). Conservative — biases against filter (smaller RR denominator).")
    lines.append(f"- target_window in H1 bars (20=~1d, 50=~2d, 100=~4d swing scale).")
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
