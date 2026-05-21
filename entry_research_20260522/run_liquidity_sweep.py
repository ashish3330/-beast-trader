#!/usr/bin/env python3 -B
"""
SMC Liquidity-Sweep / Stop-Hunt Entry Research
==============================================
Implements N-bar sweep detection on top of the v5 backtest pipeline.

LONG entry: price wicked below low-N-bar low (max wick = X×ATR),
           closes back above (low + Y×ATR), price now trending up (close>EMA20)
SHORT entry: symmetric on highs.

Variants:
  - N ∈ {10, 20, 30}, X ∈ {0.2, 0.4, 0.6}, Y ∈ {0.3, 0.5}
  - STRICT (require BOS) vs LOOSE
  - FILTER (must be sweep) vs ADDITIONAL (sweep OR normal signal)

Walk-forward 5-fold.

Output: 02_liquidity_sweep.json + 02_liquidity_sweep.md
"""
import sys
import json
import time as _time
from pathlib import Path
from itertools import product
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from signals.momentum_scorer import (
    _compute_indicators, _score_with_components, _ema,
    IND_DEFAULTS, IND_OVERRIDES,
)
from backtest.v5_backtest import (
    backtest_symbol,
    ALL_SYMBOLS, SL_OVERRIDE, SL_OVERRIDE_REGIME,
    TRAIL_OVERRIDE, TRAIL_OVERRIDE_REGIME, DIR_BIAS,
    _dir_bias_for_regime, RISK_CAP, RISK_CAP_REGIME,
    SESSION, TOXIC_HOURS, TOXIC_EXEMPT,
    get_regime, simulate_trail,
    load_data, _estimate_lots,
    DEFAULT_PARAMS,
)
from backtest.cost_model import CostModel, count_overnight_rollovers

SYMBOLS = ["DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY",
           "EURUSD", "US2000.r", "UKOUSD", "JPN225ft"]
DAYS = 180
OUT_DIR = Path("/Users/ashish/Documents/beast-trader/entry_research_20260522")


def detect_sweep(h, l, c, ema20, at, bi, N, X, Y, require_bos):
    """Detect liquidity sweep at bar bi.

    Returns (direction, kind) or (0, None):
       direction = +1 (LONG-side sweep, took out low → reverse up)
                 = -1 (SHORT-side sweep, took out high → reverse down)
       kind = "loose" or "strict-bos"
    """
    if bi < N + 5:
        return 0, None
    atr_now = float(at[bi])
    if atr_now <= 0 or np.isnan(atr_now):
        return 0, None

    # N-bar swing low/high computed on the window ENDING bi-1 (exclude
    # current bar to avoid trivially detecting current-bar low as swing).
    win_lo = float(np.min(l[bi - N:bi]))
    win_hi = float(np.max(h[bi - N:bi]))

    cur_l = float(l[bi])
    cur_h = float(h[bi])
    cur_c = float(c[bi])
    cur_ema = float(ema20[bi])
    if np.isnan(cur_ema):
        return 0, None

    # LONG-side sweep: low pierced win_lo, close recovered above win_lo + Y*atr,
    # wick depth no more than X*atr, price now > EMA20 (trending up).
    long_sweep = False
    if cur_l < win_lo:
        wick_depth = win_lo - cur_l
        if wick_depth <= X * atr_now:
            recovery = cur_c - win_lo
            if recovery > Y * atr_now and cur_c > cur_ema:
                long_sweep = True

    # SHORT-side sweep: symmetric.
    short_sweep = False
    if cur_h > win_hi:
        wick_depth = cur_h - win_hi
        if wick_depth <= X * atr_now:
            recovery = win_hi - cur_c
            if recovery > Y * atr_now and cur_c < cur_ema:
                short_sweep = True

    if not long_sweep and not short_sweep:
        return 0, None

    if require_bos:
        # Break of structure: recent high (last N bars before sweep) must have
        # been broken DOWN before this sweep (LONG case) or UP (SHORT case).
        # Operational definition: in the N bars before bi, there should be at
        # least one close that breaks the prior swing.
        lo2 = float(np.min(l[bi - 2 * N:bi - N])) if bi >= 2 * N else None
        hi2 = float(np.max(h[bi - 2 * N:bi - N])) if bi >= 2 * N else None
        if long_sweep and lo2 is not None:
            # BOS DOWN: in window [bi-N, bi), some close < lo2 (prior structure broken low)
            recent_lows = np.min(c[bi - N:bi])
            if recent_lows >= lo2:
                long_sweep = False
        if short_sweep and hi2 is not None:
            recent_highs = np.max(c[bi - N:bi])
            if recent_highs <= hi2:
                short_sweep = False

    if long_sweep and not short_sweep:
        return 1, ("strict-bos" if require_bos else "loose")
    if short_sweep and not long_sweep:
        return -1, ("strict-bos" if require_bos else "loose")
    return 0, None


def backtest_with_sweep(symbol, days, sweep_cfg, mode, params=None, verbose=False):
    """Run v5 backtest with sweep logic active.

    sweep_cfg: {"N": N, "X": X, "Y": Y, "require_bos": bool}
    mode:
      - "baseline":   no sweep involvement, mirrors backtest_symbol
      - "filter":     normal signal must ALSO have sweep alignment
      - "additional": fires on sweep OR normal signal
    """
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

    with_slippage = bool(p.get("with_slippage", False))
    with_commission = bool(p.get("with_commission", False))
    with_swap = bool(p.get("with_swap", False))
    cost_model = CostModel(spread=spread, point=point, symbol=symbol,
                           with_slippage=with_slippage,
                           with_commission=with_commission,
                           with_swap=with_swap)

    df = load_data(symbol, days)
    if df is None or len(df) < 200:
        return None

    icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(symbol, {})}
    warmup = max(icfg["EMA_T"], 100) + 30

    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    n = len(df)
    times = df["time"].values

    ind = _compute_indicators(df, icfg)
    if ind is None:
        return None

    # Precompute EMA20 for sweep filter
    ema20 = _ema(c, 20)
    at = ind["at"]

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

    N = sweep_cfg["N"]; X = sweep_cfg["X"]; Y = sweep_cfg["Y"]
    require_bos = sweep_cfg.get("require_bos", False)

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
        if bi < 21 or np.isnan(at[bi]) or at[bi] == 0:
            continue

        # --- Sweep detection ---
        sweep_dir, sweep_kind = detect_sweep(h, l, c, ema20, at, bi,
                                              N=N, X=X, Y=Y,
                                              require_bos=require_bos)

        # --- Normal scoring (always compute for comparison) ---
        cw = p.get("component_weights")
        long_s, short_s, comp_l, comp_s = _score_with_components(ind, bi, weights=cw)
        long_s, short_s = float(long_s), float(short_s)
        raw_score = max(long_s, short_s)
        signal_quality = min(100.0, raw_score / p["quality_div"] * 100)

        bbw_val = float(ind["bbw"][bi]) if not np.isnan(ind["bbw"][bi]) else 0.02
        adx_val = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 20
        regime = get_regime(bbw_val, adx_val)
        threshold = min_q.get(regime, 55) if isinstance(min_q, dict) else min_q

        normal_pass = signal_quality >= threshold
        if long_s >= short_s:
            normal_dir = 1
        else:
            normal_dir = -1

        # Determine entry direction by mode
        direction = 0
        if mode == "baseline":
            if not normal_pass:
                continue
            direction = normal_dir
        elif mode == "filter":
            # Need normal pass AND sweep AND sweep aligns with normal
            if not normal_pass:
                continue
            if sweep_dir == 0 or sweep_dir != normal_dir:
                continue
            direction = normal_dir
        elif mode == "additional":
            # Sweep alone OR normal signal
            if normal_pass:
                direction = normal_dir
            elif sweep_dir != 0:
                direction = sweep_dir
                # Use threshold as floor — sweep entries get implied quality = threshold
                signal_quality = max(signal_quality, threshold)
            else:
                continue
        else:
            raise ValueError(f"bad mode {mode}")

        # Apply per-(sym,regime) direction bias gate
        _dir_bias_eff = _dir_bias_for_regime(symbol, regime, dir_bias)
        if _dir_bias_eff != 0 and direction != _dir_bias_eff:
            continue

        # Range-extreme filter (same as baseline)
        if p.get("range_filter_enabled") and regime == "ranging":
            try:
                rng_lookback = int(p.get("range_lookback", 48))
                rng_buf = float(p.get("range_buffer_atr", 0.5))
                lo_i = max(0, bi - rng_lookback)
                highs_win = h[lo_i:bi + 1]
                lows_win = l[lo_i:bi + 1]
                close_now = float(c[bi])
                atr_now = float(at[bi])
                if atr_now > 0 and len(highs_win) >= 10:
                    buf = atr_now * rng_buf
                    if direction == 1 and close_now >= float(highs_win.max()) - buf:
                        continue
                    if direction == -1 and close_now <= float(lows_win.min()) + buf:
                        continue
            except Exception:
                pass

        # ATR + SL
        atr = float(at[bi])
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

        # Conviction sizing
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
        dollar_risk = equity * (risk / 100.0)
        lot_value = sl_dist / point
        if lot_value <= 0:
            continue

        _trail_regime_cell = TRAIL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
        _trail_base = _trail_regime_cell if _trail_regime_cell is not None else trail_steps
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

        est_lots = _estimate_lots(dollar_risk, sl_dist, point, cat)
        commission_usd = cost_model.commission_charge(est_lots) if with_commission else 0.0
        swap_usd = 0.0
        if with_swap and est_lots > 0:
            entry_ts = times[entry_bar] if entry_bar < n else times[i]
            exit_ts = times[min(exit_bar, n - 1)]
            n_roll, n_wed = count_overnight_rollovers(entry_ts, exit_ts)
            triple_wed = n_wed if cat == "Forex" else 0
            swap_usd = cost_model.swap_charge(direction, est_lots, n_roll, triple_wed)
        pnl_dollar_net = pnl_dollar - commission_usd + swap_usd

        equity += pnl_dollar_net
        peak_eq = max(peak_eq, equity)
        trades.append({
            "entry_bar": entry_bar, "exit_bar": exit_bar,
            "direction": direction, "entry": entry_price, "exit": exit_price,
            "pnl": pnl_dollar_net, "pnl_r": pnl_r, "peak_r": peak_r,
            "quality": signal_quality, "regime": regime,
            "exit_reason": exit_reason,
            "is_sweep": bool(sweep_dir != 0 and (mode != "baseline")),
            "is_normal_pass": bool(normal_pass),
        })

        if pnl_dollar_net < 0:
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
    peak = eq_curve[0]
    max_dd = 0
    for e in eq_curve:
        peak = max(peak, e)
        ddv = (peak - e) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, ddv)
    pnl_total = round(sum(t["pnl"] for t in trades), 2)

    sweep_trades = [t for t in trades if t.get("is_sweep")]
    return {
        "symbol": symbol, "trades": len(trades), "wins": len(wins),
        "pf": round(pf, 2), "wr": round(wr, 1),
        "pnl": pnl_total, "dd": round(max_dd, 1),
        "equity": round(equity, 2),
        "sweep_trade_count": len(sweep_trades),
        "n_bars": n,
    }


def walk_forward_5fold(symbol, sweep_cfg, mode, total_days=180):
    """Split total_days into 5 contiguous folds; report per-fold + aggregate."""
    fold_days = total_days // 5  # 36
    # Single backtest, then split trades by entry_bar percentile (proxy for time)
    # Simpler: run the full backtest, split trades into 5 equal time-quintile buckets.
    res = backtest_with_sweep(symbol, total_days, sweep_cfg, mode, verbose=False)
    if res is None or res.get("trades", 0) == 0:
        return None

    n_bars = res.get("n_bars", 0)
    if n_bars == 0:
        return None
    # Re-run to capture trades list (need it for fold split)
    # Quick re-implementation: collect trades by entry_bar -> bucket = entry_bar // (n_bars//5)
    # We don't have trades in result by default — collect via second pass with verbose flag
    # We need trades. Modify backtest_with_sweep to optionally return trades.
    raise NotImplementedError  # use trade-level walk-forward via _wf_run instead


def _wf_run(symbol, sweep_cfg, mode, total_days=180):
    """Run sweep backtest and return trade-level walk-forward by entry_bar quintiles."""
    # Replicate backtest_with_sweep logic but keep trade list. To avoid code dup,
    # tweak: monkey-patch backtest_with_sweep to optionally return trades.
    res = _backtest_collect(symbol, total_days, sweep_cfg, mode)
    if res is None:
        return None
    trades = res["trades_list"]
    n_bars = res["n_bars"]
    if n_bars == 0 or not trades:
        return {
            "symbol": symbol, "trades": 0, "folds": [],
            "fold_pnls": [], "fold_pfs": [], "n_pos_folds": 0,
            "avg_pf": 0.0, "total_pnl": 0.0,
            "overall_pf": 0.0, "overall_pnl": 0.0,
            "overall_wr": 0.0, "overall_dd": 0.0,
        }
    fold_size = max(1, n_bars // 5)
    folds = [[] for _ in range(5)]
    for t in trades:
        fi = min(4, t["entry_bar"] // fold_size)
        folds[fi].append(t)
    fold_metrics = []
    for fi, ftrades in enumerate(folds):
        if not ftrades:
            fold_metrics.append({"fold": fi, "trades": 0, "pf": 0.0, "pnl": 0.0, "wr": 0.0})
            continue
        wins = [t for t in ftrades if t["pnl"] > 0]
        losses = [t for t in ftrades if t["pnl"] <= 0]
        gw = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        # Cap PF at 10 when gross_loss=0 (treat as "perfect fold" not 999 outlier
        # that distorts averages).
        pf = gw / gl if gl > 0 else (10.0 if gw > 0 else 0.0)
        pnl = sum(t["pnl"] for t in ftrades)
        wr = len(wins) / len(ftrades) * 100
        fold_metrics.append({
            "fold": fi, "trades": len(ftrades),
            "pf": round(pf, 2), "pnl": round(pnl, 2), "wr": round(wr, 1),
        })
    pfs = [fm["pf"] for fm in fold_metrics if fm["trades"] > 0]
    pnls = [fm["pnl"] for fm in fold_metrics if fm["trades"] > 0]
    n_pos = sum(1 for p in pnls if p > 0)
    return {
        "symbol": symbol,
        "trades": len(trades),
        "folds": fold_metrics,
        "fold_pnls": pnls,
        "fold_pfs": pfs,
        "n_pos_folds": n_pos,
        "avg_pf": round(float(np.mean(pfs)) if pfs else 0.0, 2),
        "total_pnl": round(float(sum(pnls)), 2),
        "overall_pf": res["overall"]["pf"],
        "overall_pnl": res["overall"]["pnl"],
        "overall_wr": res["overall"]["wr"],
        "overall_dd": res["overall"]["dd"],
    }


def _backtest_collect(symbol, days, sweep_cfg, mode, params=None):
    """Same as backtest_with_sweep but ALSO returns trades list + n_bars."""
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
    with_slippage = bool(p.get("with_slippage", False))
    with_commission = bool(p.get("with_commission", False))
    with_swap = bool(p.get("with_swap", False))
    cost_model = CostModel(spread=spread, point=point, symbol=symbol,
                           with_slippage=with_slippage,
                           with_commission=with_commission,
                           with_swap=with_swap)
    df = load_data(symbol, days)
    if df is None or len(df) < 200:
        return None
    icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(symbol, {})}
    warmup = max(icfg["EMA_T"], 100) + 30
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    n = len(df)
    times = df["time"].values
    ind = _compute_indicators(df, icfg)
    if ind is None:
        return None
    ema20 = _ema(c, 20)
    at = ind["at"]
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
    N = sweep_cfg["N"]; X = sweep_cfg["X"]; Y = sweep_cfg["Y"]
    require_bos = sweep_cfg.get("require_bos", False)

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
        if bi < 21 or np.isnan(at[bi]) or at[bi] == 0:
            continue
        sweep_dir, sweep_kind = detect_sweep(h, l, c, ema20, at, bi,
                                              N=N, X=X, Y=Y,
                                              require_bos=require_bos)
        cw = p.get("component_weights")
        long_s, short_s, comp_l, comp_s = _score_with_components(ind, bi, weights=cw)
        long_s, short_s = float(long_s), float(short_s)
        raw_score = max(long_s, short_s)
        signal_quality = min(100.0, raw_score / p["quality_div"] * 100)
        bbw_val = float(ind["bbw"][bi]) if not np.isnan(ind["bbw"][bi]) else 0.02
        adx_val = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 20
        regime = get_regime(bbw_val, adx_val)
        threshold = min_q.get(regime, 55) if isinstance(min_q, dict) else min_q
        normal_pass = signal_quality >= threshold
        if long_s >= short_s:
            normal_dir = 1
        else:
            normal_dir = -1
        direction = 0
        if mode == "baseline":
            if not normal_pass:
                continue
            direction = normal_dir
        elif mode == "filter":
            if not normal_pass:
                continue
            if sweep_dir == 0 or sweep_dir != normal_dir:
                continue
            direction = normal_dir
        elif mode == "additional":
            if normal_pass:
                direction = normal_dir
            elif sweep_dir != 0:
                direction = sweep_dir
                signal_quality = max(signal_quality, threshold)
            else:
                continue
        else:
            raise ValueError(f"bad mode {mode}")
        _dir_bias_eff = _dir_bias_for_regime(symbol, regime, dir_bias)
        if _dir_bias_eff != 0 and direction != _dir_bias_eff:
            continue
        if p.get("range_filter_enabled") and regime == "ranging":
            try:
                rng_lookback = int(p.get("range_lookback", 48))
                rng_buf = float(p.get("range_buffer_atr", 0.5))
                lo_i = max(0, bi - rng_lookback)
                highs_win = h[lo_i:bi + 1]
                lows_win = l[lo_i:bi + 1]
                close_now = float(c[bi])
                atr_now = float(at[bi])
                if atr_now > 0 and len(highs_win) >= 10:
                    buf = atr_now * rng_buf
                    if direction == 1 and close_now >= float(highs_win.max()) - buf:
                        continue
                    if direction == -1 and close_now <= float(lows_win.min()) + buf:
                        continue
            except Exception:
                pass
        atr = float(at[bi])
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
        dollar_risk = equity * (risk / 100.0)
        lot_value = sl_dist / point
        if lot_value <= 0:
            continue
        _trail_regime_cell = TRAIL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
        _trail_base = _trail_regime_cell if _trail_regime_cell is not None else trail_steps
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
        est_lots = _estimate_lots(dollar_risk, sl_dist, point, cat)
        commission_usd = cost_model.commission_charge(est_lots) if with_commission else 0.0
        swap_usd = 0.0
        if with_swap and est_lots > 0:
            entry_ts = times[entry_bar] if entry_bar < n else times[i]
            exit_ts = times[min(exit_bar, n - 1)]
            n_roll, n_wed = count_overnight_rollovers(entry_ts, exit_ts)
            triple_wed = n_wed if cat == "Forex" else 0
            swap_usd = cost_model.swap_charge(direction, est_lots, n_roll, triple_wed)
        pnl_dollar_net = pnl_dollar - commission_usd + swap_usd
        equity += pnl_dollar_net
        peak_eq = max(peak_eq, equity)
        trades.append({
            "entry_bar": entry_bar, "exit_bar": exit_bar,
            "direction": direction,
            "pnl": pnl_dollar_net, "pnl_r": pnl_r, "peak_r": peak_r,
            "quality": signal_quality, "regime": regime,
            "exit_reason": exit_reason,
            "is_sweep": bool(sweep_dir != 0 and (mode != "baseline")),
            "is_normal_pass": bool(normal_pass),
        })
        if pnl_dollar_net < 0:
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

    # Aggregate overall metrics
    if not trades:
        return {
            "trades_list": [],
            "n_bars": n,
            "overall": {"trades": 0, "pf": 0, "wr": 0, "pnl": 0, "dd": 0, "equity": equity},
        }
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 999
    wr = len(wins) / len(trades) * 100
    eq_curve = [p["start_equity"]]
    for t in trades:
        eq_curve.append(eq_curve[-1] + t["pnl"])
    peak = eq_curve[0]
    max_dd = 0
    for e in eq_curve:
        peak = max(peak, e)
        ddv = (peak - e) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, ddv)
    pnl_total = round(sum(t["pnl"] for t in trades), 2)
    return {
        "trades_list": trades,
        "n_bars": n,
        "overall": {
            "trades": len(trades), "wins": len(wins),
            "pf": round(pf, 2), "wr": round(wr, 1),
            "pnl": pnl_total, "dd": round(max_dd, 1),
            "equity": round(equity, 2),
            "sweep_trade_count": sum(1 for t in trades if t["is_sweep"]),
        },
    }


def run_variant(label, sweep_cfg, mode):
    """Run all 8 symbols on a variant. Returns dict of symbol -> WF results."""
    print(f"\n=== Variant: {label} (mode={mode}, cfg={sweep_cfg}) ===")
    out = {}
    for sym in SYMBOLS:
        t0 = _time.time()
        wf = _wf_run(sym, sweep_cfg, mode, total_days=DAYS)
        dt = _time.time() - t0
        if wf is None:
            print(f"  {sym:12s} | NO DATA")
            out[sym] = None
            continue
        sweep_n = wf.get("overall_pf", 0)
        n_pos = wf.get("n_pos_folds", 0)
        sweep_ct = 0
        if wf.get("trades", 0):
            # Get sweep count separately via _backtest_collect light call
            pass
        print(
            f"  {sym:12s} | T {wf['trades']:4d} | overall PF {wf['overall_pf']:5.2f} "
            f"WR {wf['overall_wr']:5.1f}% | PnL ${wf['overall_pnl']:>8.2f} | "
            f"DD {wf['overall_dd']:4.1f}% | WF avgPF {wf['avg_pf']:.2f} "
            f"posFolds {n_pos}/5 | {dt:4.1f}s"
        )
        out[sym] = wf
    return out


def main():
    started = _time.time()
    print("="*78)
    print(" SMC LIQUIDITY-SWEEP ENTRY RESEARCH — 180d, 8 symbols")
    print("="*78)

    # ── Baseline ──
    print("\n[BASELINE] no sweep involvement (sanity vs backtest_symbol)")
    baseline_cfg = {"N": 20, "X": 0.4, "Y": 0.5, "require_bos": False}
    baseline = run_variant("baseline", baseline_cfg, mode="baseline")

    # Save baseline aggregate
    baseline_summary = {}
    for sym, r in baseline.items():
        if r is None:
            baseline_summary[sym] = None
            continue
        baseline_summary[sym] = {
            "trades": r["trades"], "pf": r["overall_pf"], "wr": r["overall_wr"],
            "pnl": r["overall_pnl"], "dd": r["overall_dd"],
            "avg_pf": r["avg_pf"], "n_pos_folds": r["n_pos_folds"],
            "fold_pnls": r["fold_pnls"],
        }

    # ── Iteration 1: ADDITIONAL — modest sweep, loose (most permissive) ──
    iter_results = []

    cfg1 = {"N": 20, "X": 0.4, "Y": 0.5, "require_bos": False}
    r1 = run_variant("iter1_additional_N20_X0.4_Y0.5_loose", cfg1, mode="additional")
    iter_results.append({"label": "iter1_additional_N20_X0.4_Y0.5_loose",
                          "mode": "additional", "cfg": cfg1, "results": r1})

    # ── Iteration 2: FILTER — same params, sweep-as-filter ──
    cfg2 = {"N": 20, "X": 0.4, "Y": 0.5, "require_bos": False}
    r2 = run_variant("iter2_filter_N20_X0.4_Y0.5_loose", cfg2, mode="filter")
    iter_results.append({"label": "iter2_filter_N20_X0.4_Y0.5_loose",
                          "mode": "filter", "cfg": cfg2, "results": r2})

    # ── Iteration 3: ADDITIONAL strict (require BOS) ──
    cfg3 = {"N": 20, "X": 0.4, "Y": 0.5, "require_bos": True}
    r3 = run_variant("iter3_additional_N20_X0.4_Y0.5_strict_bos", cfg3, mode="additional")
    iter_results.append({"label": "iter3_additional_N20_X0.4_Y0.5_strict_bos",
                          "mode": "additional", "cfg": cfg3, "results": r3})

    # ── Iteration 4: ADDITIONAL — wider N=30, more recovery (Y=0.5) ──
    cfg4 = {"N": 30, "X": 0.6, "Y": 0.5, "require_bos": False}
    r4 = run_variant("iter4_additional_N30_X0.6_Y0.5_loose", cfg4, mode="additional")
    iter_results.append({"label": "iter4_additional_N30_X0.6_Y0.5_loose",
                          "mode": "additional", "cfg": cfg4, "results": r4})

    # ── Iteration 5: ADDITIONAL — tighter wick (X=0.2), small recovery (Y=0.3), N=10 ──
    cfg5 = {"N": 10, "X": 0.2, "Y": 0.3, "require_bos": False}
    r5 = run_variant("iter5_additional_N10_X0.2_Y0.3_loose", cfg5, mode="additional")
    iter_results.append({"label": "iter5_additional_N10_X0.2_Y0.3_loose",
                          "mode": "additional", "cfg": cfg5, "results": r5})

    # ── Scoring: per-symbol Δ-PnL vs baseline, ship criteria ──
    print("\n" + "="*78)
    print(" SCORING — winner search")
    print("="*78)
    print("  Ship criteria (per symbol): ΔPnL ≥ $30 AND avg WF PF > 1.5 AND ≥3/5 positive folds")

    winners = []  # (variant_label, sym, delta_pnl, avg_pf, n_pos, var_data)
    variant_summaries = []
    for v in iter_results:
        v_results = v["results"]
        per_sym_meets = {}
        total_delta = 0.0
        ships = []
        for sym in SYMBOLS:
            b = baseline.get(sym)
            vr = v_results.get(sym)
            if b is None or vr is None:
                per_sym_meets[sym] = None
                continue
            delta_pnl = vr["overall_pnl"] - b["overall_pnl"]
            ship = (delta_pnl >= 30.0 and
                    vr["avg_pf"] > 1.5 and
                    vr["n_pos_folds"] >= 3)
            per_sym_meets[sym] = {
                "delta_pnl": round(delta_pnl, 2),
                "avg_pf": vr["avg_pf"],
                "n_pos_folds": vr["n_pos_folds"],
                "v_trades": vr["trades"],
                "v_pnl": vr["overall_pnl"],
                "v_pf": vr["overall_pf"],
                "ship": ship,
            }
            total_delta += delta_pnl
            if ship:
                ships.append(sym)
        variant_summaries.append({
            "label": v["label"],
            "mode": v["mode"], "cfg": v["cfg"],
            "per_sym": per_sym_meets,
            "total_delta_pnl": round(total_delta, 2),
            "ships": ships,
            "ship_count": len(ships),
        })
        print(f"\n  {v['label']}")
        print(f"    total ΔPnL {total_delta:+.2f}, ships {len(ships)}/{len(SYMBOLS)} "
              f"({ships if ships else 'none'})")
        for sym, m in per_sym_meets.items():
            if m is None:
                continue
            tag = "SHIP" if m["ship"] else "    "
            print(f"      {tag} {sym:12s} ΔPnL ${m['delta_pnl']:+8.2f} "
                  f"avgPF {m['avg_pf']:5.2f} pos {m['n_pos_folds']}/5 T={m['v_trades']}")

    # Winner = variant with highest ship_count then total_delta_pnl
    variant_summaries.sort(key=lambda v: (v["ship_count"], v["total_delta_pnl"]),
                            reverse=True)
    best = variant_summaries[0] if variant_summaries else None

    print("\n" + "="*78)
    if best and best["ship_count"] >= 1:
        print(f" WINNER: {best['label']}")
        print(f"   mode={best['mode']} cfg={best['cfg']}")
        print(f"   ships {best['ship_count']}/8, total ΔPnL {best['total_delta_pnl']:+.2f}")
    else:
        print(" NO WINNER — no variant met ship criteria for any symbol "
              "(or insufficient).")

    out_path = OUT_DIR / "02_liquidity_sweep.json"
    payload = {
        "research": "SMC liquidity sweep / stop hunt",
        "date": "2026-05-22",
        "days": DAYS,
        "symbols": SYMBOLS,
        "ship_criteria": {
            "delta_pnl_per_symbol_min": 30.0,
            "wf_avg_pf_min": 1.5,
            "wf_pos_folds_min": 3,
        },
        "baseline": baseline_summary,
        "iterations": variant_summaries,
        "winner": best,
        "winner_ships": best["ships"] if (best and best["ship_count"] >= 1) else [],
        "elapsed_sec": round(_time.time() - started, 1),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  saved -> {out_path}")
    return payload


if __name__ == "__main__":
    main()
