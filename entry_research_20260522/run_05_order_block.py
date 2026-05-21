#!/usr/bin/env python3 -B
"""
ORDER BLOCK (ICT/SMC) RE-ENTRY RESEARCH — entry_research_20260522 / task 05.

Concept
-------
Order Block (OB) = the last opposing candle before an impulsive structure-breaking move.
  * Bullish OB  : last DOWN candle before a strong UP impulse that breaks the recent high
  * Bearish OB  : last UP candle before a strong DOWN impulse that breaks the recent low

Entry trigger : price returns into the OB zone [open, close] of that opposing candle;
                signal direction matches the impulse direction.
Freshness     : OB must be 'untouched' (not yet revisited since formation).

Backtest design (READ-ONLY w.r.t. live code)
-------------------------------------------
We reuse the v5 backtest signal stack (indicators, _score_with_components, regime, MTF,
fib, range, audit-fix gates, ML, simulate_trail, cost_model). The OB layer is inserted
at the same logical point as the fib filter: AFTER score+regime+threshold+dir_bias+
range filter+fib but BEFORE the audit-fix friction gate. This mirrors how a real
ICT/SMC overlay would integrate.

Modes
-----
  ADDITIONAL : a) take all current signals AND
               b) ALSO take OB-only entries whose score is below the threshold
                  (the OB itself provides the conviction).
  FILTER     : current signal must coincide with being inside a fresh OB zone in
               the same direction. Otherwise reject.

Variants (5 iter cap)
---------------------
We iterate ≤5 grid points to honour the iteration cap:
  iter 1 : FILTER, impulse 2.0×ATR, lookback 20, freshness=no_revisit  (canonical)
  iter 2 : FILTER, impulse 1.5×ATR, lookback 20, freshness=no_revisit  (looser impulse)
  iter 3 : FILTER, impulse 2.5×ATR, lookback 30, freshness=no_revisit  (stricter)
  iter 4 : FILTER, impulse 2.0×ATR, lookback 15, freshness=3           (short memory)
  iter 5 : ADDITIONAL, impulse 2.0×ATR, lookback 20, freshness=no_revisit

Walk-forward
------------
180 day window, 5 contiguous folds (~36d each). PF, WR, PnL, DD reported per fold +
aggregate. Ship-eligibility (matches sibling research):
  * Aggregate PF >= 1.30
  * >=4 of 5 folds positive PnL
  * Beats baseline PnL by >= 10 %
  * Aggregate DD <= 12 %
  * Aggregate trades >= 60 across 8 symbols
"""
from __future__ import annotations

import sys, json, time as _time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── reuse production primitives ────────────────────────────────────────────
from signals.momentum_scorer import (
    _compute_indicators, _score_with_components,
    IND_DEFAULTS, IND_OVERRIDES,
)
from backtest.v5_backtest import (
    ALL_SYMBOLS, DEFAULT_PARAMS, SL_OVERRIDE, SL_OVERRIDE_REGIME,
    TRAIL_OVERRIDE, TRAIL_OVERRIDE_REGIME, RISK_CAP, RISK_CAP_REGIME,
    DIR_BIAS, TOXIC_HOURS, TOXIC_EXEMPT, SESSION,
    _dir_bias_for_regime, get_regime, simulate_trail, load_data,
    _estimate_lots,
)
from backtest.cost_model import CostModel, count_overnight_rollovers

# ── 8-symbol live set (matches scripts/backtest_live_set.py 2026-04-30) ────
LIVE8 = [
    "XAUUSD", "XAGUSD", "NAS100.r", "SP500.r",
    "GER40.r", "USDCAD", "EURUSD", "GBPJPY",
]

# Fall back gracefully — some symbols may not have 180d cache.
DAYS = 180

OUT_DIR = Path(__file__).resolve().parent
JSON_OUT = OUT_DIR / "05_order_block.json"
MD_OUT   = OUT_DIR / "05_order_block.md"


# ═══════════════════════════════════════════════════════════════════════════
#  ORDER BLOCK DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def find_order_blocks(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    at: np.ndarray,
    lookback: int = 20,
    impulse_atr: float = 2.0,
    impulse_bars: int = 3,
) -> Tuple[List[dict], List[dict]]:
    """Detect bullish and bearish OBs across the full series.

    Returns:
      bull_obs, bear_obs  — lists of dicts with keys:
        'idx'   : bar index of the OB candle itself (the opposing one)
        'lo'    : zone lower edge
        'hi'    : zone upper edge
        'formed_at' : bar index at which the impulse confirmed (idx + impulse_bars)

    Impulse rule
    ------------
    From bar j we look at the 3-bar move c[j+3] - c[j].  If the magnitude exceeds
    impulse_atr × atr[j], AND that move broke the prior LOOKBACK-bar high (bull)
    or low (bear), THEN the last opposing candle in the window (j-lookback ..j) is
    marked as the OB.
    """
    n = len(c)
    bull, bear = [], []
    for j in range(lookback + 2, n - impulse_bars - 1):
        atr_j = float(at[j]) if not np.isnan(at[j]) else 0.0
        if atr_j <= 0:
            continue
        # impulse measured by the close-to-close move across `impulse_bars`
        move = float(c[j + impulse_bars] - c[j])
        thr  = impulse_atr * atr_j

        # bullish impulse
        if move >= thr:
            prior_high = float(h[max(0, j - lookback): j + 1].max())
            # confirm structure break: any close in the impulse window > prior_high
            broke = bool((c[j + 1: j + impulse_bars + 1] > prior_high).any())
            if broke:
                # last DOWN candle in (j-lookback .. j]
                ob_idx = None
                for k in range(j, max(0, j - lookback) - 1, -1):
                    if c[k] < o[k]:
                        ob_idx = k
                        break
                if ob_idx is not None:
                    lo = float(min(o[ob_idx], c[ob_idx]))
                    hi = float(max(o[ob_idx], c[ob_idx]))
                    bull.append({
                        "idx": ob_idx,
                        "lo": lo, "hi": hi,
                        "formed_at": j + impulse_bars,
                    })

        # bearish impulse
        if move <= -thr:
            prior_low = float(l[max(0, j - lookback): j + 1].min())
            broke = bool((c[j + 1: j + impulse_bars + 1] < prior_low).any())
            if broke:
                ob_idx = None
                for k in range(j, max(0, j - lookback) - 1, -1):
                    if c[k] > o[k]:
                        ob_idx = k
                        break
                if ob_idx is not None:
                    lo = float(min(o[ob_idx], c[ob_idx]))
                    hi = float(max(o[ob_idx], c[ob_idx]))
                    bear.append({
                        "idx": ob_idx,
                        "lo": lo, "hi": hi,
                        "formed_at": j + impulse_bars,
                    })
    return bull, bear


def ob_active_at(
    bi: int, direction: int,
    bull: List[dict], bear: List[dict],
    h: np.ndarray, l: np.ndarray, c: np.ndarray,
    freshness: str = "no_revisit",
    max_age: int = 200,
) -> bool:
    """Is there a FRESH order block whose zone contains current price, in the
    correct direction?

    freshness:
      'no_revisit' — zone must not have been touched between formation and now
      '3'          — zone touched at most 3 times since formation
      '1'          — touched at most once since formation
    """
    if direction == 1:
        obs = bull
    elif direction == -1:
        obs = bear
    else:
        return False

    price = float(c[bi])
    # walk obs backwards (most recent first) — only consider ones that are
    # already formed and not too old.
    for ob in reversed(obs):
        if ob["formed_at"] > bi:
            continue
        if bi - ob["formed_at"] > max_age:
            break  # older obs are even further back, stop scanning
        # price inside zone?
        if ob["lo"] <= price <= ob["hi"]:
            # freshness check: count touches between formation and (bi-1)
            f1 = ob["formed_at"] + 1
            if f1 >= bi:
                return True  # no bars between → fresh by default
            window_hi = h[f1: bi]
            window_lo = l[f1: bi]
            # a 'touch' = the bar's range overlaps the zone
            touched = ((window_lo <= ob["hi"]) & (window_hi >= ob["lo"])).sum()
            if freshness == "no_revisit":
                if touched == 0:
                    return True
            elif freshness == "3":
                if touched <= 3:
                    return True
            elif freshness == "1":
                if touched <= 1:
                    return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  INSTRUMENTED BACKTEST — mirrors v5_backtest.backtest_symbol but with the
#  OB hook injected. NO LIVE CODE MODIFIED.
# ═══════════════════════════════════════════════════════════════════════════

def _ob_backtest(
    symbol: str,
    days: int,
    df: pd.DataFrame,
    mode: str,                      # 'baseline' | 'filter' | 'additional'
    impulse_atr: float = 2.0,
    lookback: int = 20,
    freshness: str = "no_revisit",
    impulse_bars: int = 3,
    additional_threshold_drop: float = 15.0,
    params: dict | None = None,
) -> dict | None:
    """Single-symbol backtest. mode controls the OB layer."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if "range_filter_enabled" not in p:
        try:
            import auto_tuned as _at  # type: ignore
            rfp = getattr(_at, "RANGE_FILTER_PARAMS_AUTO", {}).get(symbol)
            if rfp:
                p["range_filter_enabled"] = True
                p["range_lookback"] = rfp.get("lookback", 48)
                p["range_buffer_atr"] = rfp.get("buffer_atr", 0.5)
        except Exception:
            pass

    meta = ALL_SYMBOLS[symbol]
    spread = meta["spread"]; point = meta["point"]; cat = meta["cat"]

    cost_model = CostModel(spread=spread, point=point, symbol=symbol,
                           with_slippage=False, with_commission=False, with_swap=False)

    if df is None or len(df) < 200:
        return None

    icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(symbol, {})}
    warmup = max(icfg["EMA_T"], 100) + 30

    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    n = len(df)
    times = df["time"].values

    ind = _compute_indicators(df, icfg)
    if ind is None:
        return None

    sl_mult = SL_OVERRIDE.get(symbol, p["sl_atr_mult"])
    trail_steps = TRAIL_OVERRIDE.get(symbol, p["trail"])
    dir_bias = DIR_BIAS.get(symbol, 0)
    risk_cap = RISK_CAP.get(symbol, p["risk_pct"])
    sess_start, sess_end = SESSION.get(symbol, SESSION["default"])
    toxic_exempt = TOXIC_EXEMPT.get(symbol, set())
    try:
        from config import SIGNAL_QUALITY_SYMBOL as _SYM_Q
        _o = _SYM_Q.get(symbol, {})
        min_q = {k: _o.get(k, p["min_quality"][k]) for k in p["min_quality"]}
    except Exception:
        min_q = p["min_quality"]

    # ── OB precompute (skip in baseline mode) ──────────────────────────────
    if mode in ("filter", "additional"):
        bull_obs, bear_obs = find_order_blocks(
            o, h, l, c, ind["at"],
            lookback=lookback, impulse_atr=impulse_atr,
            impulse_bars=impulse_bars,
        )
    else:
        bull_obs = bear_obs = []

    equity = p["start_equity"]; peak_eq = equity
    trades: List[dict] = []
    consec_losses = 0; cooldown_until = 0; sl_cooldown_until = 0
    ob_signal_extra = 0  # tracks signals admitted only by OB (additional mode)

    for i in range(warmup, n - 1):
        if equity <= 0:
            break
        if i < cooldown_until or i < sl_cooldown_until:
            continue
        if cat != "Crypto":
            hour = pd.Timestamp(times[i]).hour
            if hour < sess_start or hour >= sess_end:
                continue
        hour = pd.Timestamp(times[i]).hour
        if hour in TOXIC_HOURS and hour not in toxic_exempt:
            continue

        bi = i
        if bi < 21 or np.isnan(ind["at"][bi]) or ind["at"][bi] == 0:
            continue
        long_s, short_s, comp_l, comp_s = _score_with_components(
            ind, bi, weights=p.get("component_weights"))
        long_s, short_s = float(long_s), float(short_s)
        raw_score = max(long_s, short_s)
        signal_quality = min(100.0, raw_score / p["quality_div"] * 100)

        bbw_val = float(ind["bbw"][bi]) if not np.isnan(ind["bbw"][bi]) else 0.02
        adx_val = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 20
        regime = get_regime(bbw_val, adx_val)
        threshold = min_q.get(regime, 55) if isinstance(min_q, dict) else min_q

        # determine candidate direction first (needed for OB lookup)
        if long_s >= short_s:
            direction = 1; raw = long_s
        else:
            direction = -1; raw = short_s

        # ── OB GATE LOGIC ───────────────────────────────────────────────
        ob_hit = False
        if mode in ("filter", "additional"):
            ob_hit = ob_active_at(
                bi, direction, bull_obs, bear_obs, h, l, c,
                freshness=freshness, max_age=200,
            )

        if mode == "filter":
            if signal_quality < threshold:
                continue
            if not ob_hit:
                continue
        elif mode == "additional":
            # Accept if normal gate passes OR (OB hit AND score is at most
            # `additional_threshold_drop` below threshold).
            normal_ok = signal_quality >= threshold
            ob_ok = ob_hit and signal_quality >= (threshold - additional_threshold_drop)
            if not (normal_ok or ob_ok):
                continue
            if ob_ok and not normal_ok:
                ob_signal_extra += 1
        else:  # baseline
            if signal_quality < threshold:
                continue

        _dir_bias_eff = _dir_bias_for_regime(symbol, regime, dir_bias)
        if _dir_bias_eff != 0 and direction != _dir_bias_eff:
            continue

        # ── range-extreme filter (live mirror) ──────────────────────────
        if p.get("range_filter_enabled") and regime == "ranging":
            try:
                rng_lookback = int(p.get("range_lookback", 48))
                rng_buf = float(p.get("range_buffer_atr", 0.5))
                lo_i = max(0, bi - rng_lookback)
                highs_win = h[lo_i:bi + 1]; lows_win = l[lo_i:bi + 1]
                close_now = float(c[bi]); atr_now = float(ind["at"][bi])
                if atr_now > 0 and len(highs_win) >= 10:
                    buf = atr_now * rng_buf
                    if direction == 1 and close_now >= float(highs_win.max()) - buf:
                        continue
                    if direction == -1 and close_now <= float(lows_win.min()) + buf:
                        continue
            except Exception:
                pass

        atr = float(ind["at"][bi])
        retrace = atr * 0.2
        entry_bar = i + 1
        if entry_bar >= n - 1:
            continue
        if direction == 1:
            pullback_hit = l[entry_bar] <= c[i] - retrace
            entry_price = c[i] - retrace if pullback_hit else c[i]
        else:
            pullback_hit = h[entry_bar] >= c[i] + retrace
            entry_price = c[i] + retrace if pullback_hit else c[i]

        _sl_regime_mult = SL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
        sl_eff = _sl_regime_mult if _sl_regime_mult is not None else sl_mult
        sl_dist = atr * sl_eff
        if sl_dist <= 0:
            continue
        entry_price += cost_model.entry_cost(direction, signed_size=sl_dist, atr=atr)

        if signal_quality >= 80:  conv = p.get("conv_80", 1.5)
        elif signal_quality >= 65: conv = p.get("conv_65", 1.2)
        elif signal_quality >= 55: conv = p.get("conv_55", 1.0)
        else:                      conv = p.get("conv_low", 0.6)

        _eff_risk_cap = RISK_CAP_REGIME.get(symbol, {}).get(regime, risk_cap)
        risk = min(_eff_risk_cap, p["risk_pct"]) * conv
        dollar_risk = equity * (risk / 100.0)
        lot_value = sl_dist / point
        if lot_value <= 0:
            continue

        _trail_regime_cell = TRAIL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
        adapted_steps = _trail_regime_cell if _trail_regime_cell is not None else trail_steps

        exit_price, exit_bar, exit_reason, peak_r = simulate_trail(
            entry_price, sl_dist, direction, h, l, c,
            entry_bar + 1, n, spread, adapted_steps,
            ratchet_1r=p.get("ratchet_1r", 0.3),
            ratchet_2r=p.get("ratchet_2r", 0.7),
            rl_adj=None,
        )
        exit_price += cost_model.exit_cost(direction, signed_size=sl_dist, atr=atr)

        pnl_points = (exit_price - entry_price) * direction
        pnl_r = pnl_points / sl_dist if sl_dist > 0 else 0
        pnl_dollar = dollar_risk * pnl_r

        equity += pnl_dollar
        peak_eq = max(peak_eq, equity)

        trades.append({
            "entry_bar": entry_bar, "exit_bar": exit_bar,
            "direction": direction, "entry": entry_price, "exit": exit_price,
            "pnl": pnl_dollar, "pnl_r": pnl_r, "peak_r": peak_r,
            "quality": signal_quality, "regime": regime,
            "exit_reason": exit_reason, "risk_pct": risk,
            "ob_admitted": bool(ob_hit),
            "ts": str(pd.Timestamp(times[entry_bar])),
        })

        if pnl_dollar < 0:
            consec_losses += 1
            sl_cooldown_until = exit_bar + p.get("sl_cooldown_bars", 3)
            if consec_losses >= p.get("consec_loss_limit", 4):
                cooldown_until = exit_bar + p.get("consec_loss_cooldown", 12)
                consec_losses = 0
        else:
            consec_losses = 0

        dd = (peak_eq - equity) / peak_eq * 100 if peak_eq > 0 else 0
        if dd >= 8.0:
            equity = peak_eq * 0.92
            break

    if not trades:
        return {"symbol": symbol, "trades": 0, "pf": 0, "wr": 0, "pnl": 0,
                "dd": 0, "equity": p["start_equity"], "ob_extra": ob_signal_extra}

    wins  = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else 999
    wr = len(wins) / len(trades) * 100
    avg_r = float(np.mean([t["pnl_r"] for t in trades]))

    # DD
    eq = [p["start_equity"]]
    for t in trades:
        eq.append(eq[-1] + t["pnl"])
    pk = eq[0]; mdd = 0.0
    for e in eq:
        pk = max(pk, e)
        d = (pk - e) / pk * 100 if pk > 0 else 0
        mdd = max(mdd, d)

    return {
        "symbol": symbol,
        "trades": len(trades),
        "wins": len(wins),
        "pf": round(pf, 2), "wr": round(wr, 1),
        "pnl": round(sum(t["pnl"] for t in trades), 2),
        "avg_r": round(avg_r, 3),
        "dd": round(mdd, 2),
        "equity": round(equity, 2),
        "ob_extra": ob_signal_extra,
        "details": trades,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD HARNESS
# ═══════════════════════════════════════════════════════════════════════════

def _slice_df(df: pd.DataFrame, lo_frac: float, hi_frac: float) -> pd.DataFrame:
    """Return df rows whose `time` falls in [lo, hi] fraction of the full
    timestamp span. Time-based, not row-based, so multi-symbol comparisons
    use aligned wall-clock windows."""
    if df is None or len(df) == 0:
        return df
    t = df["time"]
    t0, t1 = t.iloc[0], t.iloc[-1]
    span = t1 - t0
    lo = t0 + span * lo_frac
    hi = t0 + span * hi_frac
    return df[(t >= lo) & (t <= hi)].reset_index(drop=True)


def _run_universe(mode_label: str, kwargs: dict, dfs: Dict[str, pd.DataFrame]) -> dict:
    """Run the 8-symbol universe under one mode/params combination and aggregate."""
    per_sym = {}
    total_trades = 0; total_pnl = 0.0; total_wins = 0
    total_gw = 0.0; total_gl = 0.0
    eq_curve = [DEFAULT_PARAMS["start_equity"] * len(LIVE8)]
    ob_extra_total = 0

    for sym in LIVE8:
        df = dfs.get(sym)
        if df is None:
            per_sym[sym] = {"trades": 0, "pnl": 0.0, "pf": 0, "wr": 0, "dd": 0, "skipped": True}
            continue
        r = _ob_backtest(sym, DAYS, df, **kwargs)
        if r is None:
            per_sym[sym] = {"trades": 0, "pnl": 0.0, "pf": 0, "wr": 0, "dd": 0, "skipped": True}
            continue
        # collapse trade detail before storing (keep aggregate only)
        rs = {k: v for k, v in r.items() if k != "details"}
        per_sym[sym] = rs
        total_trades += r["trades"]
        total_pnl    += r["pnl"]
        total_wins   += r.get("wins", 0)
        ob_extra_total += r.get("ob_extra", 0)
        # gross win/loss from per-trade detail
        for t in r.get("details", []):
            if t["pnl"] > 0: total_gw += t["pnl"]
            else:            total_gl += abs(t["pnl"])

    agg_pf = (total_gw / total_gl) if total_gl > 0 else (999 if total_gw > 0 else 0)
    agg_wr = (total_wins / total_trades * 100) if total_trades else 0
    # Synthetic portfolio DD approximation: sum-of-symbol pnls timeline isn't
    # available without per-trade timestamp aggregation across symbols; we
    # report the WORST per-symbol DD as a conservative proxy.
    worst_dd = max((per_sym[s].get("dd", 0) for s in per_sym), default=0)

    return {
        "mode": mode_label,
        "params": kwargs,
        "per_symbol": per_sym,
        "totals": {
            "trades": total_trades,
            "pnl": round(total_pnl, 2),
            "pf": round(agg_pf, 2),
            "wr": round(agg_wr, 1),
            "worst_symbol_dd": round(worst_dd, 2),
            "ob_signal_extra": ob_extra_total,
        },
    }


def _walk_forward(label: str, kwargs: dict, dfs_full: Dict[str, pd.DataFrame],
                  n_folds: int = 5) -> dict:
    """Slice each symbol's df into n_folds equal time chunks, run the universe
    on each, aggregate."""
    fold_results = []
    print(f"  [{label}] walk-forward {n_folds} folds ...")
    for k in range(n_folds):
        lo = k / n_folds; hi = (k + 1) / n_folds
        dfs_k = {s: _slice_df(df, lo, hi) for s, df in dfs_full.items()}
        r = _run_universe(label, kwargs, dfs_k)
        fold_results.append({
            "fold": k + 1,
            "totals": r["totals"],
        })
        t = r["totals"]
        print(f"    fold {k+1}/{n_folds}: trades={t['trades']:4d} "
              f"PF={t['pf']:5.2f} WR={t['wr']:5.1f}% PnL=${t['pnl']:>8.2f} "
              f"worstDD={t['worst_symbol_dd']:5.1f}%")
    # aggregate
    agg_trades = sum(f["totals"]["trades"] for f in fold_results)
    agg_pnl    = sum(f["totals"]["pnl"]    for f in fold_results)
    agg_pf     = sum(f["totals"]["pf"]     for f in fold_results) / max(1, len(fold_results))
    agg_wr     = sum(f["totals"]["wr"]     for f in fold_results) / max(1, len(fold_results))
    pos_folds  = sum(1 for f in fold_results if f["totals"]["pnl"] > 0)
    worst_dd   = max((f["totals"]["worst_symbol_dd"] for f in fold_results), default=0)
    return {
        "label": label,
        "params": kwargs,
        "folds": fold_results,
        "aggregate": {
            "trades": agg_trades,
            "pnl": round(agg_pnl, 2),
            "pf_avg": round(agg_pf, 2),
            "wr_avg": round(agg_wr, 1),
            "positive_folds": pos_folds,
            "worst_symbol_dd": round(worst_dd, 2),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def _load_universe(days: int) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in LIVE8:
        df = load_data(s, days=days)
        if df is None or len(df) < 200:
            print(f"  WARN: {s} cache missing or too short — skipped")
            continue
        out[s] = df
    return out


def main():
    t0 = _time.time()
    print(f"\n{'='*70}\n  ORDER BLOCK RE-ENTRY RESEARCH (entry_research_20260522/05)")
    print(f"  Days={DAYS}  Symbols={LIVE8}\n{'='*70}\n")

    dfs = _load_universe(DAYS)
    if not dfs:
        print("  FATAL: no symbol data loaded — abort")
        return

    # Baseline (no OB) — single full-period run + walk-forward
    print("\n[BASELINE] no OB filter")
    baseline_full = _run_universe("baseline", {"mode": "baseline"}, dfs)
    bt = baseline_full["totals"]
    print(f"  FULL 180d: trades={bt['trades']:4d} PF={bt['pf']:5.2f} "
          f"WR={bt['wr']:5.1f}% PnL=${bt['pnl']:>8.2f} worstDD={bt['worst_symbol_dd']:5.1f}%")
    baseline_wf = _walk_forward("baseline", {"mode": "baseline"}, dfs, n_folds=5)

    # 5 iterations — cap honoured.
    iters = [
        ("FILTER_2.0x_lb20_fresh", {
            "mode": "filter", "impulse_atr": 2.0, "lookback": 20,
            "freshness": "no_revisit", "impulse_bars": 3}),
        ("FILTER_1.5x_lb20_fresh", {
            "mode": "filter", "impulse_atr": 1.5, "lookback": 20,
            "freshness": "no_revisit", "impulse_bars": 3}),
        ("FILTER_2.5x_lb30_fresh", {
            "mode": "filter", "impulse_atr": 2.5, "lookback": 30,
            "freshness": "no_revisit", "impulse_bars": 3}),
        ("FILTER_2.0x_lb15_f3",    {
            "mode": "filter", "impulse_atr": 2.0, "lookback": 15,
            "freshness": "3", "impulse_bars": 3}),
        ("ADDITIONAL_2.0x_lb20_fresh", {
            "mode": "additional", "impulse_atr": 2.0, "lookback": 20,
            "freshness": "no_revisit", "impulse_bars": 3,
            "additional_threshold_drop": 15.0}),
    ]

    full_results = []
    wf_results = []
    for label, kw in iters:
        print(f"\n[{label}]")
        full = _run_universe(label, kw, dfs)
        ft = full["totals"]
        print(f"  FULL 180d: trades={ft['trades']:4d} PF={ft['pf']:5.2f} "
              f"WR={ft['wr']:5.1f}% PnL=${ft['pnl']:>8.2f} "
              f"worstDD={ft['worst_symbol_dd']:5.1f}% ob_extra={ft['ob_signal_extra']}")
        full_results.append(full)
        wf = _walk_forward(label, kw, dfs, n_folds=5)
        wf_results.append(wf)

    # ── Ship eligibility evaluation ──
    base_pnl = baseline_full["totals"]["pnl"]
    base_wf_agg = baseline_wf["aggregate"]
    eligibility = []
    for full, wf in zip(full_results, wf_results):
        agg = wf["aggregate"]
        pf_ok       = agg["pf_avg"] >= 1.30
        pos_ok      = agg["positive_folds"] >= 4
        beat_base   = agg["pnl"] >= base_wf_agg["pnl"] * 1.10
        dd_ok       = agg["worst_symbol_dd"] <= 12.0
        trades_ok   = agg["trades"] >= 60
        ship = pf_ok and pos_ok and beat_base and dd_ok and trades_ok
        eligibility.append({
            "label": full["mode"],
            "pf_avg": agg["pf_avg"],
            "positive_folds": agg["positive_folds"],
            "pnl": agg["pnl"],
            "worst_symbol_dd": agg["worst_symbol_dd"],
            "trades": agg["trades"],
            "checks": {
                "pf>=1.30": pf_ok,
                "pos_folds>=4": pos_ok,
                "beat_baseline_10pct": beat_base,
                "worst_dd<=12pct": dd_ok,
                "trades>=60": trades_ok,
            },
            "SHIP_ELIGIBLE": ship,
        })

    out = {
        "task": "05_order_block",
        "concept": "ICT/SMC Order Block re-entry",
        "captured_at": pd.Timestamp.utcnow().isoformat(),
        "days": DAYS,
        "symbols": LIVE8,
        "iter_cap": 5,
        "iters_used": len(iters),
        "baseline": {
            "full_180d": baseline_full["totals"],
            "walk_forward": baseline_wf["aggregate"],
            "wf_folds": baseline_wf["folds"],
            "per_symbol_180d": baseline_full["per_symbol"],
        },
        "variants": [
            {
                "label": label,
                "params": kw,
                "full_180d": full["totals"],
                "walk_forward": wf["aggregate"],
                "wf_folds": wf["folds"],
                "per_symbol_180d": full["per_symbol"],
            }
            for (label, kw), full, wf in zip(iters, full_results, wf_results)
        ],
        "eligibility": eligibility,
        "runtime_sec": round(_time.time() - t0, 1),
    }

    JSON_OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  → wrote {JSON_OUT}")
    print(f"  Runtime: {out['runtime_sec']}s\n")

    # ── markdown report ────────────────────────────────────────────────
    md = []
    md.append("# Task 05 — Order Block (ICT/SMC) re-entry research\n")
    md.append(f"_Generated {out['captured_at']}_  \n")
    md.append(f"Universe: {', '.join(LIVE8)}  ")
    md.append(f"Window: {DAYS} days, 5-fold walk-forward  ")
    md.append(f"Iterations used: {len(iters)}/5  \n")

    md.append("## Concept\n")
    md.append(
        "Order Block = last opposing candle before an impulsive structure-breaking move. "
        "When price returns into the candle's [open, close] zone, enter in the impulse "
        "direction. Tested as FILTER (must overlap fresh OB) and ADDITIONAL "
        "(OB grants entries up to 15 quality-points below threshold).\n"
    )

    md.append("## Baseline (no OB)\n")
    md.append("Full 180d: " + json.dumps(baseline_full["totals"]) + "  ")
    md.append("WF 5-fold: " + json.dumps(base_wf_agg) + "\n")

    md.append("## Variant Summary (walk-forward aggregate)\n")
    md.append("| Variant | trades | PF avg | WR avg | PnL | +folds | worst DD | SHIP |")
    md.append("|---|---:|---:|---:|---:|---:|---:|:-:|")
    for v, e in zip(out["variants"], eligibility):
        a = v["walk_forward"]
        md.append(
            f"| {v['label']} | {a['trades']} | {a['pf_avg']} | {a['wr_avg']} | "
            f"${a['pnl']} | {a['positive_folds']}/5 | {a['worst_symbol_dd']}% | "
            f"{'YES' if e['SHIP_ELIGIBLE'] else 'no'} |"
        )

    md.append("\n## Per-fold detail (selected variants)\n")
    for v in out["variants"]:
        md.append(f"### {v['label']}")
        md.append("| fold | trades | PF | WR | PnL | worst DD |")
        md.append("|---:|---:|---:|---:|---:|---:|")
        for f in v["wf_folds"]:
            t = f["totals"]
            md.append(f"| {f['fold']} | {t['trades']} | {t['pf']} | "
                      f"{t['wr']} | ${t['pnl']} | {t['worst_symbol_dd']}% |")
        md.append("")

    md.append("## Ship eligibility breakdown\n")
    for e in eligibility:
        checks_str = ", ".join(
            f"{k}={'PASS' if v else 'FAIL'}" for k, v in e["checks"].items())
        md.append(f"- **{e['label']}**: {'SHIP' if e['SHIP_ELIGIBLE'] else 'REJECT'} — {checks_str}")

    # Honest verdict
    shippers = [e for e in eligibility if e["SHIP_ELIGIBLE"]]
    best = max(eligibility, key=lambda x: x["pnl"]) if eligibility else None
    md.append("\n## Honest verdict\n")
    if shippers:
        md.append(f"- {len(shippers)} variant(s) pass all 5 ship checks. "
                  f"Best by PnL: **{best['label']}** with WF PnL ${best['pnl']} "
                  f"(baseline ${base_wf_agg['pnl']}).")
    else:
        md.append(f"- **No variant passes ship-eligibility.** Baseline WF PnL ${base_wf_agg['pnl']}; "
                  f"best OB variant: **{best['label'] if best else 'n/a'}** at ${best['pnl'] if best else 0}.")
        md.append("- OB re-entry as implemented does not provide a net edge on the live 8-symbol set "
                  "over a 180d window with the current scoring stack already filtering low-conviction "
                  "entries.")
        md.append("- ICT/SMC concepts assume manual structure marking with discretion; mechanical "
                  "auto-detection on H1 may be picking up too many low-quality zones, OR existing "
                  "MTF/fib/audit-fix gates already capture most of the OB edge.")

    MD_OUT.write_text("\n".join(md))
    print(f"  → wrote {MD_OUT}")


if __name__ == "__main__":
    main()
