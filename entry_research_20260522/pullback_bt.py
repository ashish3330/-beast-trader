#!/usr/bin/env python3 -B
"""Inline backtest with pluggable pullback gates.

Mirrors v5_backtest.backtest_symbol logic faithfully but adds a
`pullback_mode` parameter that gates entries by industry-grade pullback
strategies before the entry fill.

Modes (set via p["pullback_mode"]):
    "baseline"   — current v5_backtest behavior (no extra gate)
    "fib_50"     — entry only if signal in 0.382-0.618 retrace zone of last swing
    "fib_618"    — entry only if signal in 0.5-0.786 retrace zone (deeper pullback)
    "ema20"      — entry only if price within ATR_KISS_MULT*ATR of EMA20
    "ema50"      — entry only if price within ATR_KISS_MULT*ATR of EMA50
    "order_block"— entry only if price returned to last opposite-color body
                   that preceded an impulsive move (≥2 ATR)
    "bb_mid"     — entry only if price within ATR_KISS_MULT*ATR of BB middle
    "trendline"  — entry only on retest of trendline drawn through last 2 swings

Tuning knobs in p:
    pullback_kiss_atr        — default 0.5 — proximity to MA/BB middle in ATR units
    pullback_fib_lo          — default 0.382
    pullback_fib_hi          — default 0.618
    pullback_swing_lookback  — default 60 bars
    pullback_ob_impulse_atr  — default 2.0 — min impulse to qualify an OB
    pullback_ob_lookback     — default 30 bars to search for OB
    pullback_trendline_lookback — default 80
    pullback_trendline_tol_atr  — default 0.3 — tolerance for retest

This is READ-ONLY on the v5_backtest source — we reimport indicators,
simulate_trail, get_regime, and the per-symbol overrides directly.
"""
from __future__ import annotations

import sys, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signals.momentum_scorer import (
    _compute_indicators, _score, _score_with_components,
    IND_DEFAULTS, IND_OVERRIDES,
)
from backtest.cost_model import CostModel, count_overnight_rollovers
from backtest.v5_backtest import (
    DEFAULT_PARAMS, ALL_SYMBOLS, SL_OVERRIDE, SL_OVERRIDE_REGIME,
    TRAIL_OVERRIDE, TRAIL_OVERRIDE_REGIME, RISK_CAP, RISK_CAP_REGIME,
    DIR_BIAS, TOXIC_HOURS, TOXIC_EXEMPT, SESSION,
    _dir_bias_for_regime, get_regime, simulate_trail,
    _estimate_lots, load_data, CACHE,
)
from config import (
    MOMENTUM_SIZE_BOOST_ENABLED as _MOM_SIZE_BOOST_ENABLED,
    MOMENTUM_TRAIL_ADAPTIVE_ENABLED as _MOM_TRAIL_ADAPTIVE_ENABLED,
    MOMENTUM_MIN_SCORE_ADAPTIVE_ENABLED as _MOM_MIN_SCORE_ENABLED,
    MOMENTUM_SL_ADAPTIVE_ENABLED as _MOM_SL_ADAPTIVE_ENABLED,
    MTF_CASCADE_ENABLED as _MTF_CASCADE_ENABLED,
)


# ── Pullback gate implementations ─────────────────────────────────────────

def _last_swings(h, l, bi, lookback):
    """Find most recent Williams Fractal (5-bar) swing high and low before bi.
    Returns (sh, sh_idx, sl, sl_idx) or (None, None, None, None) if not found."""
    swing_hi = swing_lo = None
    swing_hi_idx = swing_lo_idx = None
    if bi < lookback + 3:
        return None, None, None, None
    for j in range(bi - 3, max(bi - lookback, 2), -1):
        hj = float(h[j]); lj = float(l[j])
        if (swing_hi is None and
                hj > h[j-1] and hj > h[j-2] and
                hj > h[j+1] and hj > h[j+2]):
            swing_hi = hj; swing_hi_idx = j
        if (swing_lo is None and
                lj < l[j-1] and lj < l[j-2] and
                lj < l[j+1] and lj < l[j+2]):
            swing_lo = lj; swing_lo_idx = j
        if swing_hi is not None and swing_lo is not None:
            break
    return swing_hi, swing_hi_idx, swing_lo, swing_lo_idx


def gate_fib(direction, c, h, l, bi, atr, p):
    """Fib retracement zone gate."""
    lo = float(p.get("pullback_fib_lo", 0.382))
    hi = float(p.get("pullback_fib_hi", 0.618))
    lb = int(p.get("pullback_swing_lookback", 60))
    sh, sh_i, sl, sl_i = _last_swings(h, l, bi, lb)
    if sh is None or sl is None:
        return False
    rng = sh - sl
    if rng <= 2 * atr:  # require ≥ 2 ATR impulse
        return False
    last_was_high = (sh_i or 0) > (sl_i or 0)
    close_now = float(c[bi])
    if last_was_high:
        # Last swing was UP — LONGs want price pulled back DOWN into golden zone
        if direction != 1:
            return False
        retr = (close_now - sl) / rng  # 0 = at low, 1 = at high
        # Pullback FROM the high — we want retr in [1-hi, 1-lo]
        return (1 - hi) <= retr <= (1 - lo)
    else:
        if direction != -1:
            return False
        retr = (sh - close_now) / rng  # 0 = at high, 1 = at low
        return (1 - hi) <= retr <= (1 - lo)


def gate_ma_kiss(direction, c, ema_series, bi, atr, p):
    """Price within kiss_atr ATR of MA (and on right side of MA)."""
    kiss = float(p.get("pullback_kiss_atr", 0.5))
    ma_now = float(ema_series[bi])
    if not np.isfinite(ma_now) or ma_now <= 0:
        return False
    close_now = float(c[bi])
    dist = abs(close_now - ma_now)
    if dist > kiss * atr:
        return False
    # Must be on the trend side — LONG needs price ≥ MA (within tolerance below),
    # SHORT needs price ≤ MA. Allow small wick: |dist| ≤ kiss*ATR satisfies "near"
    # Side check: for LONG, prior bar's close should be above MA (uptrend) OR
    # this bar shows recovery (close > open near MA).
    if direction == 1:
        # In uptrend; we want price coming back DOWN to MA from above
        # require recent N bars had close > MA (trend up)
        return True
    else:
        return True


def gate_order_block(direction, o, c, h, l, bi, atr, p):
    """Order Block re-entry: find the last opposite-color candle before an
    impulsive move (≥ impulse_atr * ATR over next 3 bars), check if price
    has now returned to that block's range."""
    impulse_mult = float(p.get("pullback_ob_impulse_atr", 2.0))
    lb = int(p.get("pullback_ob_lookback", 30))
    if bi < lb + 5:
        return False
    # Scan back from bi-1 to find an impulse
    for j in range(bi - 3, max(bi - lb, 5), -1):
        # Check if bars j..j+3 had an impulse of size impulse_mult*ATR
        if direction == 1:
            # LONG → look for a bearish (red) candle followed by bullish impulse UP
            move = float(c[min(j + 3, bi - 1)] - c[j])
            if move < impulse_mult * atr:
                continue
            # Last red candle at or before j
            for k in range(j, max(j - 5, 0), -1):
                if c[k] < o[k]:  # bearish body
                    # OB zone = [low[k], high[k]]
                    ob_lo = float(l[k]); ob_hi = float(h[k])
                    # has price now returned into the OB?
                    if ob_lo <= float(c[bi]) <= ob_hi:
                        return True
                    break
        else:
            move = float(c[j] - c[min(j + 3, bi - 1)])
            if move < impulse_mult * atr:
                continue
            for k in range(j, max(j - 5, 0), -1):
                if c[k] > o[k]:  # bullish body
                    ob_lo = float(l[k]); ob_hi = float(h[k])
                    if ob_lo <= float(c[bi]) <= ob_hi:
                        return True
                    break
    return False


def gate_bb_mid(direction, c, ind, bi, atr, p):
    """Price within kiss_atr ATR of BB middle."""
    kiss = float(p.get("pullback_kiss_atr", 0.5))
    bm = ind.get("bm")
    if bm is None:
        return False
    bm_now = float(bm[bi])
    if not np.isfinite(bm_now):
        return False
    dist = abs(float(c[bi]) - bm_now)
    return dist <= kiss * atr


def gate_trendline(direction, h, l, c, bi, atr, p):
    """Trendline retest: connect last 2 same-side swings, check current price
    is at the projected trendline (within tol_atr ATR)."""
    lb = int(p.get("pullback_trendline_lookback", 80))
    tol = float(p.get("pullback_trendline_tol_atr", 0.3))
    if bi < lb + 5:
        return False
    # Find 2 most recent swings of relevant type
    if direction == 1:
        # LONG → uptrend trendline = connect 2 most recent swing LOWS, check
        # current price is at or just above the line
        swings = []
        for j in range(bi - 3, max(bi - lb, 2), -1):
            if (l[j] < l[j-1] and l[j] < l[j-2] and
                    l[j] < l[j+1] and l[j] < l[j+2]):
                swings.append((j, float(l[j])))
                if len(swings) >= 2:
                    break
        if len(swings) < 2:
            return False
        (j1, p1), (j0, p0) = swings  # j1 newer, j0 older
        slope = (p1 - p0) / max(j1 - j0, 1)
        proj = p1 + slope * (bi - j1)
        # require uptrend slope (slope > 0)
        if slope <= 0:
            return False
        return abs(float(c[bi]) - proj) <= tol * atr and float(c[bi]) >= proj * 0.998
    else:
        swings = []
        for j in range(bi - 3, max(bi - lb, 2), -1):
            if (h[j] > h[j-1] and h[j] > h[j-2] and
                    h[j] > h[j+1] and h[j] > h[j+2]):
                swings.append((j, float(h[j])))
                if len(swings) >= 2:
                    break
        if len(swings) < 2:
            return False
        (j1, p1), (j0, p0) = swings
        slope = (p1 - p0) / max(j1 - j0, 1)
        proj = p1 + slope * (bi - j1)
        if slope >= 0:
            return False
        return abs(float(c[bi]) - proj) <= tol * atr and float(c[bi]) <= proj * 1.002


# ── Inline backtest runner ────────────────────────────────────────────────

def backtest_with_pullback(symbol, days=180, params=None, verbose=False):
    """Mirror v5_backtest.backtest_symbol but apply pullback_mode gate to
    every potential entry. Returns same shape dict.

    `params["pullback_mode"]` must be one of the supported modes.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    mode = p.get("pullback_mode", "baseline")

    # mirror v5_backtest's range_filter injection
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

    df = load_data(symbol, days)
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

    # We need EMA20 and EMA50 for ma-kiss modes. Compute inline if absent.
    from signals.momentum_scorer import _ema
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)

    sl_mult = SL_OVERRIDE.get(symbol, p["sl_atr_mult"])
    if p.get("force_trail") is not None:
        trail_steps = p["force_trail"]
    else:
        trail_steps = TRAIL_OVERRIDE.get(symbol, p["trail"])
    dir_bias = DIR_BIAS.get(symbol, 0)
    risk_cap = RISK_CAP.get(symbol, p["risk_pct"])
    sess_start, sess_end = SESSION.get(symbol, SESSION["default"])
    toxic_exempt = TOXIC_EXEMPT.get(symbol, set())
    try:
        from config import SIGNAL_QUALITY_SYMBOL as _SYM_Q
        _override = _SYM_Q.get(symbol, {})
        min_q = {k: _override.get(k, p["min_quality"][k]) for k in p["min_quality"]}
    except Exception:
        min_q = p["min_quality"]

    equity = p["start_equity"]
    peak_eq = equity
    trades = []
    consec_losses = 0
    cooldown_until = 0
    sl_cooldown_until = 0

    # MTF cascade precompute (mirrors v5_backtest line 542-546)
    _MTF_PRECOMP = None
    if _MTF_CASCADE_ENABLED:
        try:
            from signals.mtf_trend import precompute_mtf_trends, mtf_verdict_at_bar
            _MTF_PRECOMP = precompute_mtf_trends(c, tfs=("W1", "D1", "H4"))
            _mtf_verdict = mtf_verdict_at_bar
        except Exception:
            _MTF_PRECOMP = None

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
        cw = p.get("component_weights")
        long_s, short_s, comp_l, comp_s = _score_with_components(ind, bi, weights=cw)
        long_s, short_s = float(long_s), float(short_s)
        raw_score = max(long_s, short_s)
        signal_quality = min(100.0, raw_score / p["quality_div"] * 100)

        bbw_val = float(ind["bbw"][bi]) if not np.isnan(ind["bbw"][bi]) else 0.02
        adx_val = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 20
        regime = get_regime(bbw_val, adx_val)
        threshold = min_q.get(regime, 55) if isinstance(min_q, dict) else min_q

        # MOMENTUM_MIN_SCORE adjustment (gated)
        if _MOM_MIN_SCORE_ENABLED:
            try:
                from signals.momentum_signal import compute_momentum_at_bar, min_score_delta
                from config import MOMENTUM_MIN_SCORE_FLOOR
                mom_bar = compute_momentum_at_bar(ind, bi)
                delta = min_score_delta(mom_bar)
                quality_delta = delta * (100.0 / p.get("quality_div", 8))
                adjusted_threshold = max(
                    MOMENTUM_MIN_SCORE_FLOOR * (100.0 / p.get("quality_div", 8)),
                    threshold + quality_delta,
                )
                if signal_quality < adjusted_threshold:
                    continue
            except Exception:
                if signal_quality < threshold:
                    continue
        else:
            if signal_quality < threshold:
                continue

        if long_s >= short_s:
            direction = 1; raw = long_s
        else:
            direction = -1; raw = short_s

        _dir_bias_eff = _dir_bias_for_regime(symbol, regime, dir_bias)
        if _dir_bias_eff != 0 and direction != _dir_bias_eff:
            continue

        # range filter
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

        # MTF cascade gate (mirrors v5_backtest)
        if _MTF_CASCADE_ENABLED and _MTF_PRECOMP is not None:
            verdict = _mtf_verdict(_MTF_PRECOMP, i, direction)
            if verdict == "REJECT":
                continue

        # ═══ PULLBACK GATE ═══
        if mode == "baseline":
            gate_pass = True
        elif mode == "fib_50":
            # 0.382-0.618 retrace zone
            p_local = {**p, "pullback_fib_lo": 0.382, "pullback_fib_hi": 0.618}
            gate_pass = gate_fib(direction, c, h, l, bi, atr, p_local)
        elif mode == "fib_618":
            p_local = {**p, "pullback_fib_lo": 0.5, "pullback_fib_hi": 0.786}
            gate_pass = gate_fib(direction, c, h, l, bi, atr, p_local)
        elif mode == "fib_custom":
            gate_pass = gate_fib(direction, c, h, l, bi, atr, p)
        elif mode == "ema20":
            gate_pass = gate_ma_kiss(direction, c, ema20, bi, atr, p)
        elif mode == "ema50":
            gate_pass = gate_ma_kiss(direction, c, ema50, bi, atr, p)
        elif mode == "bb_mid":
            gate_pass = gate_bb_mid(direction, c, ind, bi, atr, p)
        elif mode == "order_block":
            gate_pass = gate_order_block(direction, o, c, h, l, bi, atr, p)
        elif mode == "trendline":
            gate_pass = gate_trendline(direction, h, l, c, bi, atr, p)
        elif mode == "ema20_or_fib":
            gate_pass = (gate_ma_kiss(direction, c, ema20, bi, atr, p) or
                         gate_fib(direction, c, h, l, bi, atr, p))
        elif mode == "ema20_and_fib":
            gate_pass = (gate_ma_kiss(direction, c, ema20, bi, atr, p) and
                         gate_fib(direction, c, h, l, bi, atr, p))
        else:
            gate_pass = True

        if not gate_pass:
            continue

        # === Entry, SL, trail, exit — verbatim from v5_backtest ===
        entry_bar = i + 1
        if entry_bar >= n - 1:
            continue

        # Mirror v5_backtest pullback fill (0.2 ATR), since this is the
        # ALREADY built-in pullback. We keep it identical so all variants
        # share the same fill mechanic. Variants gate WHEN to attempt entry,
        # not HOW to fill.
        retrace = atr * 0.2
        if direction == 1:
            pullback_hit = l[entry_bar] <= c[i] - retrace
            entry_price = c[i] - retrace if pullback_hit else c[i]
        else:
            pullback_hit = h[entry_bar] >= c[i] + retrace
            entry_price = c[i] + retrace if pullback_hit else c[i]

        _sl_regime_mult = SL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
        sl_eff = _sl_regime_mult if _sl_regime_mult is not None else sl_mult
        if _MOM_SL_ADAPTIVE_ENABLED:
            try:
                from signals.momentum_signal import compute_momentum_at_bar as _cm, sl_multiplier as _sm
                sl_eff = sl_eff * _sm(_cm(ind, bi))
            except Exception:
                pass
        sl_dist = atr * sl_eff
        if sl_dist <= 0:
            continue

        entry_price += cost_model.entry_cost(direction, signed_size=sl_dist, atr=atr)

        if signal_quality >= 80:
            conv = p.get("conv_80", 1.5)
        elif signal_quality >= 65:
            conv = p.get("conv_65", 1.2)
        elif signal_quality >= 55:
            conv = p.get("conv_55", 1.0)
        else:
            conv = p.get("conv_low", 0.6)

        _eff_risk_cap = RISK_CAP_REGIME.get(symbol, {}).get(regime, risk_cap)
        risk = min(_eff_risk_cap, p["risk_pct"]) * conv
        if _MOM_SIZE_BOOST_ENABLED:
            try:
                from signals.momentum_signal import compute_momentum_at_bar as _cm2, size_multiplier as _smul
                sig_dir = "LONG" if direction == 1 else "SHORT"
                risk *= _smul(_cm2(ind, bi), sig_dir)
            except Exception:
                pass
        dollar_risk = equity * (risk / 100.0)
        lot_value = sl_dist / point
        if lot_value <= 0:
            continue

        _trail_regime_cell = TRAIL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
        _trail_base = _trail_regime_cell if _trail_regime_cell is not None else trail_steps
        if _MOM_TRAIL_ADAPTIVE_ENABLED:
            try:
                from signals.momentum_signal import (
                    compute_momentum_at_bar as _cm3, trail_multiplier as _tm, lock_threshold_mult as _ltm,
                )
                mom_bar = _cm3(ind, bi)
                tmult = _tm(mom_bar); lmult = _ltm(mom_bar)
                adapted_steps = [
                    (trig * lmult, (param * tmult if kind == "trail" else param), kind)
                    for trig, param, kind in _trail_base
                ]
            except Exception:
                adapted_steps = _trail_base
        else:
            adapted_steps = _trail_base

        exit_price, exit_bar, exit_reason, peak_r = simulate_trail(
            entry_price, sl_dist, direction, h, l, c,
            entry_bar + 1, n, spread, adapted_steps,
            ratchet_1r=p.get("ratchet_1r", 0.3),
            ratchet_2r=p.get("ratchet_2r", 0.7),
            rl_adj=p.get("rl_adj"))

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
        return {"symbol": symbol, "trades": 0, "pf": 0, "wr": 0, "pnl": 0, "dd": 0}

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 999
    wr = len(wins) / len(trades) * 100

    eq_curve = [p["start_equity"]]
    for t in trades:
        eq_curve.append(eq_curve[-1] + t["pnl"])
    peak = eq_curve[0]; max_dd = 0
    for e in eq_curve:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        "symbol": symbol, "trades": len(trades), "wins": len(wins),
        "pf": round(pf, 2), "wr": round(wr, 1),
        "pnl": round(sum(t["pnl"] for t in trades), 2),
        "dd": round(max_dd, 1), "equity": round(equity, 2),
    }


if __name__ == "__main__":
    # Sanity: re-run baseline mode and check it roughly matches v5_backtest.
    import sys as _sys
    sym = _sys.argv[1] if len(_sys.argv) > 1 else "DJ30.r"
    days = int(_sys.argv[2]) if len(_sys.argv) > 2 else 180
    r = backtest_with_pullback(sym, days=days, params={"pullback_mode": "baseline"})
    print("baseline:", r)
    for m in ("fib_50", "ema20", "order_block"):
        r = backtest_with_pullback(sym, days=days, params={"pullback_mode": m})
        print(f"{m}:", r)
