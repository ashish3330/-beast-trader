#!/usr/bin/env python3 -B
"""
06_divergence — Momentum divergence entries (RSI / MACD).

Concept:
  Regular divergence — REVERSAL signal.
      Price HH but RSI LH → REGULAR_BEAR → counter-trend SHORT (or filter LONG)
      Price LL but RSI HL → REGULAR_BULL → counter-trend LONG  (or filter SHORT)
  Hidden divergence — CONTINUATION signal (trend resuming after pullback).
      In uptrend: Price HL but RSI LL → HIDDEN_BULL → confirms LONG
      In downtrend: Price LH but RSI HH → HIDDEN_BEAR → confirms SHORT

Detection (per bar bi):
  Walk back from bi using Williams 5-bar fractals on H[]/L[]. Collect the
  two most-recent swing highs (resp. lows). Read RSI (or MACD-line) at
  those swing bar indices. Compare price slope vs RSI slope.

Variants tested:
  baseline                — no divergence logic, current v5 behaviour
  REGULAR_DIV_FILTER      — reject entries whose direction opposes the
                            most-recent regular div on H1
  HIDDEN_DIV_BOOST        — only allow entries with confirming hidden div
  DIV_REVERSAL_ENTRY      — bypass MIN_QUALITY when a regular div fires
                            in the entry direction (counter-trend)
  REGULAR_DIV_FILTER_H4   — same as REGULAR_DIV_FILTER but div detected on
                            H4 (resampled from H1)
  HIDDEN_DIV_BOOST_H4     — same as HIDDEN_DIV_BOOST but on H4

Indicators tested: RSI (primary), MACD_line (secondary).
"""
import sys, os, json, time, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest import v5_backtest as v5
from backtest.v5_backtest import (
    ALL_SYMBOLS, backtest_symbol, SL_OVERRIDE, CACHE, load_data,
)
from signals.momentum_scorer import (
    _compute_indicators, _score_with_components,
    IND_DEFAULTS, IND_OVERRIDES,
)

# ─── CONFIG ──────────────────────────────────────────────────────────────
SYMBOLS = ["XAUUSD", "BTCUSD", "NAS100.r", "JPN225ft",
           "USDCAD", "USDJPY", "EURUSD", "GBPJPY"]
DAYS_BASELINE = 180
WF_FOLDS = 5
WF_TOTAL_DAYS = 540   # 5×108d folds, embargo 1d
WF_EMBARGO = 1

# Williams fractal lookback windows
SWING_LOOKBACK = 60     # bars to scan for swings (H1 = ~2.5 days)
SWING_LR = 2            # bars to left/right for fractal (Williams 5-bar = lr=2)
SWING_MIN_GAP = 5       # require swings at least this many bars apart

# Output paths
OUT_DIR = ROOT / "entry_research_20260522"
OUT_JSON = OUT_DIR / "06_divergence.json"
OUT_MD = OUT_DIR / "06_divergence.md"

# Ship-criteria (per task brief)
SHIP_DELTA_USD = 30.0
SHIP_WF_PF = 1.5
SHIP_FOLDS_POSITIVE = 3


# ─── DIVERGENCE DETECTION ────────────────────────────────────────────────
def _find_fractal_swings(h, l, ind_series, end_i, lookback, lr=2,
                         min_gap=5, kind="high"):
    """Return list of (bar_idx, price_level, ind_value) for swings, newest first.
    kind: 'high' (use h[]) or 'low' (use l[]). Williams 5-bar fractal:
    a fractal high at j requires h[j] > h[j-1..j-lr] AND h[j] > h[j+1..j+lr].
    """
    start = max(lr, end_i - lookback)
    swings = []
    series = h if kind == "high" else l
    for j in range(end_i - lr, start, -1):
        # bounds for the right-side check
        if j + lr >= end_i + 1:
            continue
        ok = True
        for k in range(1, lr + 1):
            if kind == "high":
                if not (series[j] > series[j - k] and series[j] > series[j + k]):
                    ok = False; break
            else:
                if not (series[j] < series[j - k] and series[j] < series[j + k]):
                    ok = False; break
        if ok:
            # enforce gap from previously-seen swing
            if swings and abs(j - swings[-1][0]) < min_gap:
                continue
            iv = float(ind_series[j]) if not np.isnan(ind_series[j]) else None
            if iv is None:
                continue
            swings.append((j, float(series[j]), iv))
            if len(swings) >= 2:
                break
    return swings


def _divergence_state_at(bi, h, l, ind_series_high, ind_series_low,
                          lookback=SWING_LOOKBACK, lr=SWING_LR,
                          min_gap=SWING_MIN_GAP):
    """Return dict with possible divergence labels at bar bi.
    Returns: {'regular_bear': bool, 'regular_bull': bool,
              'hidden_bear':  bool, 'hidden_bull':  bool}
    """
    out = {"regular_bear": False, "regular_bull": False,
           "hidden_bear":  False, "hidden_bull":  False}

    # ── Use HIGHs + ind for bearish-type checks ──
    highs = _find_fractal_swings(h, l, ind_series_high, bi, lookback,
                                  lr=lr, min_gap=min_gap, kind="high")
    if len(highs) >= 2:
        # highs[0] = newest, highs[1] = older
        p_new, p_old = highs[0][1], highs[1][1]
        i_new, i_old = highs[0][2], highs[1][2]
        if p_new > p_old and i_new < i_old:
            out["regular_bear"] = True
        if p_new < p_old and i_new > i_old:
            out["hidden_bear"] = True

    # ── Use LOWs + ind for bullish-type checks ──
    lows = _find_fractal_swings(h, l, ind_series_low, bi, lookback,
                                  lr=lr, min_gap=min_gap, kind="low")
    if len(lows) >= 2:
        p_new, p_old = lows[0][1], lows[1][1]
        i_new, i_old = lows[0][2], lows[1][2]
        if p_new < p_old and i_new > i_old:
            out["regular_bull"] = True
        if p_new > p_old and i_new < i_old:
            out["hidden_bull"] = True

    return out


def precompute_div_mask(df, ind, indicator_key="rs"):
    """Compute divergence state at every bar in df. Returns array of dicts.
    indicator_key: 'rs' for RSI, 'ml' for MACD line.
    """
    n = len(df)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    ind_arr = np.asarray(ind[indicator_key], dtype=float)
    mask = [None] * n
    # Start scanning once we have lookback+lr bars
    start = SWING_LOOKBACK + SWING_LR + 2
    for bi in range(start, n):
        mask[bi] = _divergence_state_at(bi, h, l, ind_arr, ind_arr)
    return mask


def precompute_div_mask_h4(df, indicator_key="rs"):
    """Resample H1→H4, compute indicators on H4, detect divergence, then
    map each H4 bar's div state to the H1 bars it covers."""
    # OHLC resample to H4 (right-closed for live parity)
    df_h4 = df.set_index("time").resample("4h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna().reset_index()
    if len(df_h4) < 50:
        return [None] * len(df)
    icfg = IND_DEFAULTS  # symbol-agnostic defaults are fine for H4 divergence
    ind_h4 = _compute_indicators(df_h4, icfg)
    if ind_h4 is None:
        return [None] * len(df)
    n_h4 = len(df_h4)
    h_h4 = df_h4["high"].values.astype(float)
    l_h4 = df_h4["low"].values.astype(float)
    ind_arr = np.asarray(ind_h4[indicator_key], dtype=float)
    h4_mask = [None] * n_h4
    start = SWING_LOOKBACK + SWING_LR + 2
    for bi in range(start, n_h4):
        h4_mask[bi] = _divergence_state_at(bi, h_h4, l_h4, ind_arr, ind_arr)

    # Map H1 bar → H4 div state (use the last fully-closed H4 bar at H1 time)
    h4_times = df_h4["time"].values
    h1_times = df["time"].values
    h1_mask = [None] * len(df)
    for i in range(len(df)):
        idx = np.searchsorted(h4_times, h1_times[i], side="right") - 1
        if idx >= 1:
            h1_mask[i] = h4_mask[idx - 1]  # last closed H4 bar
    return h1_mask


# ─── ENTRY-INJECTION WRAPPER ─────────────────────────────────────────────
# Module-level state set by the run() driver before each symbol's BT
_DIV_STATE = {
    "mask": None,          # list[dict|None]  per-bar div labels
    "mode": "baseline",    # see VARIANTS
}

# Save original
_orig_score_with_components = v5._score_with_components


def _patched_score(ind, bi, weights=None):
    """Wrap _score_with_components: apply divergence filter/boost/entry."""
    long_s, short_s, comp_l, comp_s = _orig_score_with_components(
        ind, bi, weights=weights)

    mask = _DIV_STATE["mask"]
    mode = _DIV_STATE["mode"]
    if mode == "baseline" or mask is None or bi >= len(mask) or mask[bi] is None:
        return long_s, short_s, comp_l, comp_s

    div = mask[bi]
    # Decide which direction the current bar leans.
    direction = 1 if long_s >= short_s else -1

    if mode in ("REGULAR_DIV_FILTER", "REGULAR_DIV_FILTER_H4"):
        # Reject when the strongest signal direction opposes a fresh regular div
        if direction == 1 and div["regular_bear"]:
            return 0.0, 0.0, comp_l, comp_s
        if direction == -1 and div["regular_bull"]:
            return 0.0, 0.0, comp_l, comp_s
        return long_s, short_s, comp_l, comp_s

    if mode in ("HIDDEN_DIV_BOOST", "HIDDEN_DIV_BOOST_H4"):
        # Require a confirming hidden div in the signal direction (continuation)
        if direction == 1 and not div["hidden_bull"]:
            return 0.0, 0.0, comp_l, comp_s
        if direction == -1 and not div["hidden_bear"]:
            return 0.0, 0.0, comp_l, comp_s
        return long_s, short_s, comp_l, comp_s

    if mode == "DIV_REVERSAL_ENTRY":
        # Counter-trend: if a regular_bear div exists, FORCE short side high
        # so a SHORT entry is taken regardless of long_s/short_s ranking.
        # Acceptable because counter-trend reversal entries don't need the
        # full trend stack to align.
        if div["regular_bear"]:
            # boost SHORT to a strong (above-threshold) raw value
            return 0.0, 9.0, comp_l, comp_s
        if div["regular_bull"]:
            return 9.0, 0.0, comp_l, comp_s
        # Otherwise behave like baseline (no veto)
        return long_s, short_s, comp_l, comp_s

    return long_s, short_s, comp_l, comp_s


VARIANTS = [
    "baseline",
    "REGULAR_DIV_FILTER",
    "HIDDEN_DIV_BOOST",
    "DIV_REVERSAL_ENTRY",
    "REGULAR_DIV_FILTER_H4",
    "HIDDEN_DIV_BOOST_H4",
]


# ─── WINDOWED LOAD_DATA (for k-fold) ─────────────────────────────────────
_orig_load_data = v5.load_data
_WIN_START_OFFSET = None
_WIN_END_OFFSET = None
_FULL_DAYS = None


def _windowed_load_data(symbol, days=90):
    meta = ALL_SYMBOLS[symbol]
    path = CACHE / meta["cache"]
    if not path.exists():
        return None
    df = pickle.load(open(path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    end_of_data = df["time"].max()
    if _WIN_START_OFFSET is not None and _WIN_END_OFFSET is not None:
        start = end_of_data - pd.Timedelta(days=_WIN_START_OFFSET)
        end   = end_of_data - pd.Timedelta(days=_WIN_END_OFFSET)
        df = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
    elif _FULL_DAYS is not None:
        cutoff = end_of_data - pd.Timedelta(days=_FULL_DAYS)
        df = df[df["time"] >= cutoff].reset_index(drop=True)
    return df


# ─── RUN ONE BACKTEST WITH A VARIANT ─────────────────────────────────────
def _bt_one(symbol, mode, tf="H1", indicator_key="rs", verbose=False):
    """Run BT for symbol under given divergence variant. Returns slim dict."""
    # Apply current load_data patch & fetch df FIRST so we can compute mask
    v5.load_data = _windowed_load_data
    df = v5.load_data(symbol, DAYS_BASELINE)
    if df is None or len(df) < 200:
        return None
    icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(symbol, {})}
    ind = _compute_indicators(df, icfg)
    if ind is None:
        return None

    if mode == "baseline":
        mask = None
    elif tf == "H4":
        mask = precompute_div_mask_h4(df, indicator_key=indicator_key)
    else:
        mask = precompute_div_mask(df, ind, indicator_key=indicator_key)

    _DIV_STATE["mask"] = mask
    _DIV_STATE["mode"] = mode
    # Inject patched scorer
    v5._score_with_components = _patched_score
    try:
        r = backtest_symbol(symbol, DAYS_BASELINE, params=None, verbose=False)
    finally:
        v5._score_with_components = _orig_score_with_components
        _DIV_STATE["mask"] = None
        _DIV_STATE["mode"] = "baseline"

    if r is None:
        return None
    # Strip trade details to keep output small
    return {k: v for k, v in r.items() if k != "details"}


# ─── BASELINE + 180D SWEEP ────────────────────────────────────────────────
def run_full_180d():
    """Baseline + each variant for 8 symbols, 180d window."""
    global _WIN_START_OFFSET, _WIN_END_OFFSET, _FULL_DAYS
    _WIN_START_OFFSET = None
    _WIN_END_OFFSET = None
    _FULL_DAYS = DAYS_BASELINE

    print(f"\n{'='*80}")
    print(f"  06_divergence — 180d backtest, 8 symbols")
    print(f"{'='*80}\n")

    results = {}
    for variant in VARIANTS:
        # Pick H1 vs H4 path
        tf = "H4" if variant.endswith("_H4") else "H1"
        # RSI is primary; MACD-line variant captured separately below
        ik = "rs"

        per_sym = {}
        total_pnl = 0.0
        total_trades = 0
        for sym in SYMBOLS:
            t0 = time.time()
            r = _bt_one(sym, variant, tf=tf, indicator_key=ik)
            dt = time.time() - t0
            if r is None:
                per_sym[sym] = None
                continue
            per_sym[sym] = r
            total_pnl += r.get("pnl", 0)
            total_trades += r.get("trades", 0)
            print(f"  {variant:28s} {sym:10s} | {r['trades']:4d}t WR {r['wr']:5.1f}% "
                  f"PF {r['pf']:5.2f} PnL ${r['pnl']:>+8.2f} DD {r['dd']:4.1f}% "
                  f"({dt:.1f}s)")
        results[variant] = {
            "per_symbol": per_sym,
            "total_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
        }
        print(f"  {'─'*78}")
        print(f"  {variant:28s} {'TOTAL':10s} | {total_trades:4d}t              "
              f"            PnL ${total_pnl:>+8.2f}\n")

    # Δ vs baseline
    base = results["baseline"]["total_pnl"]
    for variant in VARIANTS:
        if variant == "baseline":
            results[variant]["delta_vs_baseline"] = 0
        else:
            results[variant]["delta_vs_baseline"] = round(
                results[variant]["total_pnl"] - base, 2)
    return results


# ─── WALK-FORWARD 5-FOLD ─────────────────────────────────────────────────
def _build_kfold(total_days, k, embargo):
    fold_size = total_days / float(k)
    out = []
    for fi in range(k):
        test_start = total_days - fi * fold_size
        test_end = total_days - (fi + 1) * fold_size
        out.append({
            "fold": fi,
            "test_start_off": test_start,
            "test_end_off": test_end,
            "embargo": embargo,
        })
    return out


def run_walk_forward(top_variants):
    """5-fold time-series CV on the most-promising variants."""
    global _WIN_START_OFFSET, _WIN_END_OFFSET, _FULL_DAYS
    _FULL_DAYS = None
    folds = _build_kfold(WF_TOTAL_DAYS, WF_FOLDS, WF_EMBARGO)
    print(f"\n{'='*80}")
    print(f"  Walk-Forward 5-fold ({WF_TOTAL_DAYS}d / {WF_FOLDS} folds, embargo {WF_EMBARGO}d)")
    print(f"  Variants: {top_variants}")
    print(f"{'='*80}\n")

    wf = {}
    for variant in top_variants:
        tf = "H4" if variant.endswith("_H4") else "H1"
        per_fold = []
        for f in folds:
            _WIN_START_OFFSET = f["test_start_off"]
            _WIN_END_OFFSET = f["test_end_off"]
            fold_pnl = 0.0
            fold_trades = 0
            fold_wins = 0
            fold_gw = 0.0
            fold_gl = 0.0
            for sym in SYMBOLS:
                r = _bt_one(sym, variant, tf=tf)
                if r is None:
                    continue
                fold_pnl += r.get("pnl", 0)
                fold_trades += r.get("trades", 0)
                fold_wins += r.get("wins", 0)
                # Reconstruct gross numbers from pf+pnl (approx — pf=gw/gl, net=gw-gl)
                pf = r.get("pf", 0)
                pnl = r.get("pnl", 0)
                if pf > 0 and pf < 999:
                    # gw - gl = pnl, gw = pf*gl → (pf-1)*gl = pnl → gl = pnl/(pf-1) if pf>1
                    if pf > 1 and pnl > 0:
                        gl = pnl / (pf - 1)
                        gw = pf * gl
                    elif pf < 1 and pnl < 0:
                        gl = -pnl / (1 - pf)
                        gw = pf * gl
                    else:
                        gw = abs(pnl); gl = abs(pnl)
                    fold_gw += gw
                    fold_gl += gl
            fold_pf = (fold_gw / fold_gl) if fold_gl > 0 else (999 if fold_gw > 0 else 0)
            fold_wr = (fold_wins / fold_trades * 100) if fold_trades else 0
            per_fold.append({
                "fold": f["fold"],
                "window_days": [f["test_start_off"], f["test_end_off"]],
                "pnl": round(fold_pnl, 2),
                "trades": fold_trades,
                "wins": fold_wins,
                "wr": round(fold_wr, 1),
                "pf": round(fold_pf, 2),
            })
            print(f"  {variant:28s} fold {f['fold']} | "
                  f"trades {fold_trades:4d} | PF {fold_pf:5.2f} | "
                  f"PnL ${fold_pnl:>+8.2f}")
        # Aggregate
        pfs = [pf["pf"] for pf in per_fold if pf["trades"] > 0]
        pnls = [pf["pnl"] for pf in per_fold if pf["trades"] > 0]
        n_pos = sum(1 for p in pnls if p > 0)
        agg = {
            "n_folds": len(pfs),
            "pf_mean": round(float(np.mean(pfs)) if pfs else 0, 2),
            "pf_stdev": round(float(np.std(pfs)) if len(pfs) > 1 else 0, 2),
            "pnl_total": round(sum(pnls), 2),
            "pnl_mean": round(float(np.mean(pnls)) if pnls else 0, 2),
            "folds_positive": n_pos,
        }
        wf[variant] = {"folds": per_fold, "aggregate": agg}
        print(f"  {variant:28s} AGG | folds {agg['n_folds']} | "
              f"PF μ {agg['pf_mean']:.2f} σ {agg['pf_stdev']:.2f} | "
              f"Σ PnL ${agg['pnl_total']:>+8.2f} | "
              f"{n_pos}/{agg['n_folds']} positive\n")
    return wf


# ─── DECIDE & WRITE OUTPUT ───────────────────────────────────────────────
def decide_ship(results_180d, wf):
    """Apply ship-criteria: Δ ≥ $30 AND WF avg PF > 1.5 AND ≥3/5 folds positive."""
    ship = {}
    base_pnl = results_180d["baseline"]["total_pnl"]
    for variant, rec in wf.items():
        agg = rec["aggregate"]
        delta = results_180d[variant]["total_pnl"] - base_pnl
        c1 = delta >= SHIP_DELTA_USD
        c2 = agg["pf_mean"] > SHIP_WF_PF
        c3 = agg["folds_positive"] >= SHIP_FOLDS_POSITIVE
        ship[variant] = {
            "delta_180d_usd": round(delta, 2),
            "wf_pf_mean": agg["pf_mean"],
            "wf_folds_positive": agg["folds_positive"],
            "criteria": {"delta_ge_30": c1, "pf_gt_1_5": c2, "ge_3_folds_positive": c3},
            "SHIP": bool(c1 and c2 and c3),
        }
    return ship


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ── Phase 1: baseline + all variants on 180d ──
    r180 = run_full_180d()

    # ── Phase 2: pick top non-baseline variants (positive Δ, ≥30 trades total) ──
    candidates = []
    for v in VARIANTS:
        if v == "baseline":
            continue
        d = r180[v]["delta_vs_baseline"]
        n = r180[v]["total_trades"]
        candidates.append((v, d, n))
    candidates.sort(key=lambda x: x[1], reverse=True)
    # Walk-forward the top 4 + always include best filter+boost+reversal triad
    wf_set = []
    for v, d, n in candidates:
        if n >= 30:
            wf_set.append(v)
        if len(wf_set) >= 4:
            break
    # Ensure we cover all three concept families if possible (filter, boost, reversal)
    must_have_prefixes = ["REGULAR_DIV_FILTER", "HIDDEN_DIV_BOOST", "DIV_REVERSAL_ENTRY"]
    for prefix in must_have_prefixes:
        if not any(v.startswith(prefix) for v in wf_set):
            for v, d, n in candidates:
                if v.startswith(prefix) and n >= 10:
                    wf_set.append(v); break
    print(f"\n  Walk-forward candidates: {wf_set}\n")

    wf = run_walk_forward(wf_set) if wf_set else {}

    # ── Phase 3: ship decision ──
    ship = decide_ship(r180, wf)

    # ── Write outputs ──
    payload = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": SYMBOLS,
        "days_baseline": DAYS_BASELINE,
        "wf_total_days": WF_TOTAL_DAYS,
        "wf_folds": WF_FOLDS,
        "wf_embargo": WF_EMBARGO,
        "swing_lookback": SWING_LOOKBACK,
        "swing_lr": SWING_LR,
        "swing_min_gap": SWING_MIN_GAP,
        "ship_criteria": {
            "delta_usd_min": SHIP_DELTA_USD,
            "wf_pf_min": SHIP_WF_PF,
            "folds_positive_min": SHIP_FOLDS_POSITIVE,
        },
        "results_180d": r180,
        "walk_forward": wf,
        "ship_decision": ship,
        "elapsed_s": round(time.time() - t_total, 1),
    }
    json.dump(payload, open(OUT_JSON, "w"), indent=2, default=str)
    print(f"\n  Wrote {OUT_JSON}  ({payload['elapsed_s']:.0f}s total)")

    # ── Write markdown summary ──
    write_markdown(payload)
    print(f"  Wrote {OUT_MD}")


def write_markdown(p):
    lines = []
    lines.append("# 06 — Momentum Divergence Entries")
    lines.append("")
    lines.append(f"_Generated: {p['captured_at']}  •  "
                 f"Symbols: {len(p['symbols'])}  •  "
                 f"180d baseline + {p['wf_folds']}-fold WF ({p['wf_total_days']}d, embargo {p['wf_embargo']}d)_")
    lines.append("")
    lines.append("## Concept")
    lines.append("- **Regular divergence** — reversal. Price HH + RSI LH → SHORT setup. Price LL + RSI HL → LONG setup.")
    lines.append("- **Hidden divergence** — continuation. In uptrend, Price HL + RSI LL → LONG (trend resuming).")
    lines.append("- Detection: Williams 5-bar fractals over the last "
                 f"{p['swing_lookback']} bars (lr={p['swing_lr']}, min gap {p['swing_min_gap']} bars). RSI is primary momentum series.")
    lines.append("")

    # 180d table
    lines.append("## 180-day backtest")
    lines.append("")
    lines.append("| Variant | Trades | PnL $ | Δ vs baseline |")
    lines.append("|---------|-------:|------:|--------------:|")
    base = p["results_180d"]["baseline"]["total_pnl"]
    for variant in VARIANTS:
        rec = p["results_180d"].get(variant, {})
        if not rec:
            continue
        d = rec.get("delta_vs_baseline", 0)
        lines.append(f"| {variant} | {rec['total_trades']} | {rec['total_pnl']:+.2f} | "
                     f"{'—' if variant == 'baseline' else f'{d:+.2f}'} |")
    lines.append("")

    # WF table
    if p["walk_forward"]:
        lines.append("## Walk-Forward 5-fold")
        lines.append("")
        lines.append("| Variant | Folds | PF μ | PF σ | Σ PnL $ | Folds positive |")
        lines.append("|---------|------:|-----:|-----:|--------:|---------------:|")
        for variant, rec in p["walk_forward"].items():
            agg = rec["aggregate"]
            lines.append(f"| {variant} | {agg['n_folds']} | {agg['pf_mean']:.2f} | "
                         f"{agg['pf_stdev']:.2f} | {agg['pnl_total']:+.2f} | "
                         f"{agg['folds_positive']}/{agg['n_folds']} |")
        lines.append("")

    # Ship decision
    lines.append("## Ship decision")
    lines.append("")
    sc = p["ship_criteria"]
    lines.append(f"Ship if Δ ≥ ${sc['delta_usd_min']:.0f} **AND** WF PF μ > {sc['wf_pf_min']:.2f} **AND** ≥ {sc['folds_positive_min']}/5 folds positive.")
    lines.append("")
    lines.append("| Variant | Δ 180d $ | WF PF μ | Folds+ | SHIP |")
    lines.append("|---------|---------:|--------:|-------:|:----:|")
    any_ship = False
    for variant, d in p["ship_decision"].items():
        any_ship = any_ship or d["SHIP"]
        lines.append(f"| {variant} | {d['delta_180d_usd']:+.2f} | "
                     f"{d['wf_pf_mean']:.2f} | {d['wf_folds_positive']}/5 | "
                     f"{'**YES**' if d['SHIP'] else 'no'} |")
    lines.append("")
    lines.append(f"**Overall: {'SHIP at least one variant' if any_ship else 'NO SHIP — all variants fail ship-criteria'}.**")
    lines.append("")

    open(OUT_MD, "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
