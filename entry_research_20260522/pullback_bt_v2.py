#!/usr/bin/env python3 -B
"""Pullback BT v2 — pullback as FILL improvement (not signal gate).

Modes:
    "baseline"            — current 0.2 ATR / 1-bar pullback (parity with live)
    "deep_05_3bar"        — 0.5 ATR / 3-bar wait, fallback to direct entry
    "deep_07_5bar"        — 0.7 ATR / 5-bar wait, fallback
    "deep_10_5bar"        — 1.0 ATR / 5-bar wait, fallback
    "fib50_fill"          — try fib 38-50% zone fill, 5 bar wait, fallback
    "ema20_fill"          — try EMA20 touch fill, 5 bar wait, fallback
    "pullback_size_boost" — same fill, but BOOST size 1.3x if filled (true pullback)

This file only differs from pullback_bt.py in the entry block.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signals.momentum_scorer import (
    _compute_indicators, _score_with_components,
    IND_DEFAULTS, IND_OVERRIDES, _ema,
)
from backtest.cost_model import CostModel
from backtest.v5_backtest import (
    DEFAULT_PARAMS, ALL_SYMBOLS, SL_OVERRIDE, SL_OVERRIDE_REGIME,
    TRAIL_OVERRIDE, TRAIL_OVERRIDE_REGIME, RISK_CAP, RISK_CAP_REGIME,
    DIR_BIAS, TOXIC_HOURS, TOXIC_EXEMPT, SESSION,
    _dir_bias_for_regime, get_regime, simulate_trail, load_data,
)
from config import (
    MOMENTUM_SIZE_BOOST_ENABLED as _MOM_SIZE_BOOST_ENABLED,
    MOMENTUM_TRAIL_ADAPTIVE_ENABLED as _MOM_TRAIL_ADAPTIVE_ENABLED,
    MOMENTUM_MIN_SCORE_ADAPTIVE_ENABLED as _MOM_MIN_SCORE_ENABLED,
    MOMENTUM_SL_ADAPTIVE_ENABLED as _MOM_SL_ADAPTIVE_ENABLED,
    MTF_CASCADE_ENABLED as _MTF_CASCADE_ENABLED,
)


def _last_swing_pre(h, l, bi, lookback):
    """Return (sh, sh_idx, sl, sl_idx) most recent Williams Fractal swing."""
    swing_hi = swing_lo = None
    swing_hi_idx = swing_lo_idx = None
    if bi < lookback + 3:
        return None, None, None, None
    for j in range(bi - 3, max(bi - lookback, 2), -1):
        hj = float(h[j]); lj = float(l[j])
        if (swing_hi is None and hj > h[j-1] and hj > h[j-2] and
                hj > h[j+1] and hj > h[j+2]):
            swing_hi = hj; swing_hi_idx = j
        if (swing_lo is None and lj < l[j-1] and lj < l[j-2] and
                lj < l[j+1] and lj < l[j+2]):
            swing_lo = lj; swing_lo_idx = j
        if swing_hi is not None and swing_lo is not None:
            break
    return swing_hi, swing_hi_idx, swing_lo, swing_lo_idx


def _entry_target(mode, direction, c, h, l, bi, ema20_s, ema50_s, ind, atr, p):
    """Compute the desired fill price for the given pullback mode.
    Returns None if no valid target (caller should fallback)."""
    close_now = float(c[bi])
    if mode in ("deep_05_3bar", "deep_05_5bar", "deep_05_8bar"):
        retr = atr * 0.5
        return close_now - retr if direction == 1 else close_now + retr
    if mode in ("deep_07_5bar", "deep_07_8bar"):
        retr = atr * 0.7
        return close_now - retr if direction == 1 else close_now + retr
    if mode in ("deep_10_5bar", "deep_10_8bar"):
        retr = atr * 1.0
        return close_now - retr if direction == 1 else close_now + retr
    if mode == "deep_03_3bar":
        retr = atr * 0.3
        return close_now - retr if direction == 1 else close_now + retr
    if mode in ("pullback_size_boost", "pullback_size_boost_hi",
                "no_fallback_05_5b"):
        retr = atr * 0.5
        return close_now - retr if direction == 1 else close_now + retr
    if mode == "no_fallback_07_5b":
        retr = atr * 0.7
        return close_now - retr if direction == 1 else close_now + retr
    if mode in ("fib50_fill", "fib50_fill_5b"):
        sh, sh_i, sl, sl_i = _last_swing_pre(h, l, bi, 60)
        if sh is None or sl is None:
            return None
        rng = sh - sl
        if rng <= 2 * atr:
            return None
        # 50% retrace of the most recent swing range
        target_lo = sl + 0.5 * rng
        target_hi = sh - 0.5 * rng
        if direction == 1:
            # Want LONG fill at 50% retrace from the top (price falls into it)
            return target_lo
        else:
            return target_hi
    if mode in ("ema20_fill", "ema20_fill_5b"):
        e = float(ema20_s[bi])
        if not np.isfinite(e):
            return None
        return e
    return None


def _try_fill_pullback(direction, target, h, l, c, entry_bar, n, max_wait):
    """Walk forward max_wait bars; return (filled_bar, fill_price) or (None, None)."""
    for k in range(max_wait):
        bar = entry_bar + k
        if bar >= n - 1:
            break
        if direction == 1:
            if l[bar] <= target:
                return bar, target
        else:
            if h[bar] >= target:
                return bar, target
    return None, None


# Mode → (max_wait_bars, fallback_allowed, size_mult_if_pullback)
MODE_CONFIG = {
    "baseline":              (1, True, 1.0),    # mirror live
    "deep_03_3bar":          (3, True, 1.0),
    "deep_05_3bar":          (3, True, 1.0),
    "deep_05_5bar":          (5, True, 1.0),
    "deep_05_8bar":          (8, True, 1.0),
    "deep_07_5bar":          (5, True, 1.0),
    "deep_07_8bar":          (8, True, 1.0),
    "deep_10_5bar":          (5, True, 1.0),
    "deep_10_8bar":          (8, True, 1.0),
    "fib50_fill":            (3, True, 1.0),
    "fib50_fill_5b":         (5, True, 1.0),
    "ema20_fill":            (3, True, 1.0),
    "ema20_fill_5b":         (5, True, 1.0),
    "pullback_size_boost":   (3, True, 1.3),   # 0.5 ATR / 3 bar / 1.3x size if filled
    "pullback_size_boost_hi":(5, True, 1.5),   # 0.5 ATR / 5 bar / 1.5x size
    "no_fallback_05_5b":     (5, False, 1.0),  # SKIP if not filled (no fallback)
    "no_fallback_07_5b":     (5, False, 1.0),
}


def backtest_v2(symbol, days=180, params=None, verbose=False):
    p = {**DEFAULT_PARAMS, **(params or {})}
    mode = p.get("pullback_mode", "baseline")
    max_wait, allow_fallback, size_mult = MODE_CONFIG.get(mode, (1, True, 1.0))

    # For "deep_*" use deep target; for baseline use 0.2 ATR
    if mode == "baseline":
        def _tgt(direction, c_, h_, l_, bi, e20, e50, ind, atr):
            close_now = float(c_[bi])
            retr = atr * 0.2
            return close_now - retr if direction == 1 else close_now + retr
    else:
        def _tgt(direction, c_, h_, l_, bi, e20, e50, ind, atr):
            return _entry_target(mode, direction, c_, h_, l_, bi, e20, e50, ind, atr, p)

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
    ema20 = _ema(c, 20); ema50 = _ema(c, 50)

    sl_mult = SL_OVERRIDE.get(symbol, p["sl_atr_mult"])
    trail_steps = (p["force_trail"] if p.get("force_trail") is not None
                   else TRAIL_OVERRIDE.get(symbol, p["trail"]))
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

    equity = p["start_equity"]; peak_eq = equity
    trades = []
    consec_losses = 0; cooldown_until = 0; sl_cooldown_until = 0

    _MTF_PRECOMP = None
    if _MTF_CASCADE_ENABLED:
        try:
            from signals.mtf_trend import precompute_mtf_trends, mtf_verdict_at_bar
            _MTF_PRECOMP = precompute_mtf_trends(c, tfs=("W1", "D1", "H4"))
            _mtf_verdict = mtf_verdict_at_bar
        except Exception:
            _MTF_PRECOMP = None

    for i in range(warmup, n - 1):
        if equity <= 0: break
        if i < cooldown_until or i < sl_cooldown_until: continue
        if cat != "Crypto":
            hour = pd.Timestamp(times[i]).hour
            if hour < sess_start or hour >= sess_end: continue
        hour = pd.Timestamp(times[i]).hour
        if hour in TOXIC_HOURS and hour not in toxic_exempt: continue
        bi = i
        if bi < 21 or np.isnan(ind["at"][bi]) or ind["at"][bi] == 0: continue
        cw = p.get("component_weights")
        long_s, short_s, comp_l, comp_s = _score_with_components(ind, bi, weights=cw)
        long_s, short_s = float(long_s), float(short_s)
        raw_score = max(long_s, short_s)
        signal_quality = min(100.0, raw_score / p["quality_div"] * 100)
        bbw_val = float(ind["bbw"][bi]) if not np.isnan(ind["bbw"][bi]) else 0.02
        adx_val = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 20
        regime = get_regime(bbw_val, adx_val)
        threshold = min_q.get(regime, 55) if isinstance(min_q, dict) else min_q

        if _MOM_MIN_SCORE_ENABLED:
            try:
                from signals.momentum_signal import compute_momentum_at_bar, min_score_delta
                from config import MOMENTUM_MIN_SCORE_FLOOR
                mom_bar = compute_momentum_at_bar(ind, bi)
                delta = min_score_delta(mom_bar)
                quality_delta = delta * (100.0 / p.get("quality_div", 8))
                adjusted_threshold = max(
                    MOMENTUM_MIN_SCORE_FLOOR * (100.0 / p.get("quality_div", 8)),
                    threshold + quality_delta)
                if signal_quality < adjusted_threshold: continue
            except Exception:
                if signal_quality < threshold: continue
        else:
            if signal_quality < threshold: continue

        if long_s >= short_s:
            direction = 1; raw = long_s
        else:
            direction = -1; raw = short_s

        _dir_bias_eff = _dir_bias_for_regime(symbol, regime, dir_bias)
        if _dir_bias_eff != 0 and direction != _dir_bias_eff: continue

        if p.get("range_filter_enabled") and regime == "ranging":
            try:
                rng_lookback = int(p.get("range_lookback", 48))
                rng_buf = float(p.get("range_buffer_atr", 0.5))
                lo_i = max(0, bi - rng_lookback)
                highs_win = h[lo_i:bi + 1]; lows_win = l[lo_i:bi + 1]
                close_now = float(c[bi]); atr_now = float(ind["at"][bi])
                if atr_now > 0 and len(highs_win) >= 10:
                    buf = atr_now * rng_buf
                    if direction == 1 and close_now >= float(highs_win.max()) - buf: continue
                    if direction == -1 and close_now <= float(lows_win.min()) + buf: continue
            except Exception: pass

        atr = float(ind["at"][bi])

        if _MTF_CASCADE_ENABLED and _MTF_PRECOMP is not None:
            verdict = _mtf_verdict(_MTF_PRECOMP, i, direction)
            if verdict == "REJECT": continue

        entry_bar = i + 1
        if entry_bar >= n - 1: continue

        # ── PULLBACK FILL ──
        target = _tgt(direction, c, h, l, bi, ema20, ema50, ind, atr)
        filled_at = None; fill_price = None
        if target is not None and (
                (direction == 1 and target < float(c[bi])) or
                (direction == -1 and target > float(c[bi]))):
            filled_at, fill_price = _try_fill_pullback(direction, target, h, l, c,
                                                       entry_bar, n, max_wait)

        if filled_at is not None:
            entry_price = fill_price
            actual_entry_bar = filled_at
            is_pullback = True
        else:
            if not allow_fallback:
                continue  # SKIP if pullback not hit
            # Fallback fill mode. Two options:
            #   - p["fallback"] = "stale_close" (default; matches v5_backtest):
            #     use c[i]. Slight look-ahead since by entry_bar price has moved.
            #   - p["fallback"] = "next_open": use o[entry_bar]. Realistic for
            #     live (market order fills at next bar open after signal close).
            if p.get("fallback", "stale_close") == "next_open":
                entry_price = float(o[entry_bar])
            else:
                entry_price = float(c[i])
            actual_entry_bar = entry_bar
            is_pullback = False

        _sl_regime_mult = SL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
        sl_eff = _sl_regime_mult if _sl_regime_mult is not None else sl_mult
        if _MOM_SL_ADAPTIVE_ENABLED:
            try:
                from signals.momentum_signal import compute_momentum_at_bar as _cm, sl_multiplier as _sm
                sl_eff = sl_eff * _sm(_cm(ind, bi))
            except Exception: pass
        sl_dist = atr * sl_eff
        if sl_dist <= 0: continue

        entry_price += cost_model.entry_cost(direction, signed_size=sl_dist, atr=atr)

        if signal_quality >= 80: conv = p.get("conv_80", 1.5)
        elif signal_quality >= 65: conv = p.get("conv_65", 1.2)
        elif signal_quality >= 55: conv = p.get("conv_55", 1.0)
        else: conv = p.get("conv_low", 0.6)

        _eff_risk_cap = RISK_CAP_REGIME.get(symbol, {}).get(regime, risk_cap)
        risk = min(_eff_risk_cap, p["risk_pct"]) * conv
        if _MOM_SIZE_BOOST_ENABLED:
            try:
                from signals.momentum_signal import compute_momentum_at_bar as _cm2, size_multiplier as _smul
                sig_dir = "LONG" if direction == 1 else "SHORT"
                risk *= _smul(_cm2(ind, bi), sig_dir)
            except Exception: pass
        # Apply pullback size boost only if actually filled at pullback
        if is_pullback and size_mult != 1.0:
            risk *= size_mult
        dollar_risk = equity * (risk / 100.0)
        lot_value = sl_dist / point
        if lot_value <= 0: continue

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
                    for trig, param, kind in _trail_base]
            except Exception: adapted_steps = _trail_base
        else: adapted_steps = _trail_base

        exit_price, exit_bar, exit_reason, peak_r = simulate_trail(
            entry_price, sl_dist, direction, h, l, c,
            actual_entry_bar + 1, n, spread, adapted_steps,
            ratchet_1r=p.get("ratchet_1r", 0.3),
            ratchet_2r=p.get("ratchet_2r", 0.7),
            rl_adj=p.get("rl_adj"))

        exit_price += cost_model.exit_cost(direction, signed_size=sl_dist, atr=atr)
        pnl_points = (exit_price - entry_price) * direction
        pnl_r = pnl_points / sl_dist if sl_dist > 0 else 0
        pnl_dollar = dollar_risk * pnl_r
        equity += pnl_dollar
        peak_eq = max(peak_eq, equity)

        trades.append({"direction": direction, "pnl": pnl_dollar, "pnl_r": pnl_r,
                       "peak_r": peak_r, "regime": regime,
                       "is_pullback": is_pullback, "quality": signal_quality})

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
            equity = peak_eq * 0.92; break

    if not trades:
        return {"symbol": symbol, "trades": 0, "pf": 0, "wr": 0, "pnl": 0, "dd": 0, "pb_rate": 0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 999
    wr = len(wins) / len(trades) * 100
    pb_count = sum(1 for t in trades if t["is_pullback"])
    pb_rate = pb_count / len(trades) * 100

    eq_curve = [p["start_equity"]]
    for t in trades: eq_curve.append(eq_curve[-1] + t["pnl"])
    peak = eq_curve[0]; max_dd = 0
    for e in eq_curve:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        "symbol": symbol, "trades": len(trades), "wins": len(wins),
        "pf": round(float(pf), 2), "wr": round(float(wr), 1),
        "pnl": round(float(sum(t["pnl"] for t in trades)), 2),
        "dd": round(float(max_dd), 1),
        "pb_rate": round(float(pb_rate), 1),
    }


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "DJ30.r"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    for m in ("baseline", "deep_05_3bar", "deep_07_5bar", "fib50_fill_5b", "ema20_fill_5b", "pullback_size_boost"):
        r = backtest_v2(sym, days=days, params={"pullback_mode": m})
        print(f"{m:25s}", r)
