#!/usr/bin/env python3 -B
"""
ICT Liquidity-Sweep + FVG-Retest backtester (experiment/new-entry-exit-logic).

Strategy (exactly as specified):
  Bias    : Daily — price above/below 200 EMA → long/short bias only.
  Liquidity: 4H — price sweeps a prior swing high/low (false breakout).
  Trigger : 15M — a 3-candle FVG (imbalance) forms in the reversal direction;
            price retraces into the gap; enter at market on the gap MIDPOINT.

  Entry LONG : sweep BELOW a 4H swing low, reverse up, bullish FVG
               (low[c3] > high[c1]); enter when price retraces to gap midpoint.
  Entry SHORT: sweep ABOVE a 4H swing high, reverse down, bearish FVG
               (high[c3] < low[c1]); enter when price retraces to gap midpoint.

  Exits:
    SL  = opposite side of the swept liquidity level (the sweep extreme).
    TP1 = 1.5 x stop distance  (close 50%)
    TP2 = 3.0 x stop distance  (close remaining 50%)
    Time stop: if TP1 not hit within 4H, close entire position at market.
    Entry-candle breach: if price breaches the entry candle's low (long) /
                         high (short) before TP1, exit immediately.

No indicators beyond the Daily 200 EMA bias. Pure price action.

Usage:
    python3 -B backtest/ict_fvg_backtest.py --symbol USDCAD --base m15
    python3 -B backtest/ict_fvg_backtest.py --all --base h1   # H1-proxy universe
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")

# Per-asset spread (price units) for cost modelling — mirrors v5_backtest values.
SPREAD = {
    "EURUSD": 0.00015, "GBPUSD": 0.00020, "USDJPY": 0.015, "USDCAD": 0.00020,
    "USDCHF": 0.00020, "AUDJPY": 0.020, "EURJPY": 0.020, "GBPJPY": 0.025,
    "XAUUSD": 0.30, "XAGUSD": 0.030, "BTCUSD": 30.0, "ETHUSD": 2.0,
    "NAS100.r": 1.50, "SP500.r": 0.50, "DJ30.r": 2.0, "US2000.r": 0.50,
    "GER40.r": 2.0, "UK100.r": 2.0, "JPN225ft": 10.0, "SPI200.r": 2.0,
    "SWI20.r": 3.0, "XPTUSD.r": 1.0, "USOUSD": 0.05,
}

# Swing pivot half-window (bars each side) on the 4H frame.
# When True, the "entry-candle breach before TP1" hard stop is disabled —
# trades then rely only on the swept-level SL + 4H time stop. Used to isolate
# how much the (very tight) breach rule helps vs hurts.
DISABLE_BREACH = False

SWING_LOOKBACK = 3
# How many recent 4H swing levels to keep "live" as sweep targets.
SWING_MEMORY = 20
# Max bars to wait on 15M for the FVG retrace fill after a sweep before the
# setup expires.
SETUP_EXPIRY_BARS_15M = 24      # ~6h on M15
# Time stop: close entire position if TP1 not hit within this many hours.
TIME_STOP_HOURS = 4.0
# TP structure (R multiples): TP1 closes 50%, TP2 the runner 50%.
TP1_R = 1.5
TP2_R = 3.0
# If >0, only tally trades in the last N base bars (indicators still warm up on
# full history). Used to report a clean trailing window e.g. 180d M15 = 17280.
MEASURE_LAST_N = 0


def _ema(arr, period):
    out = np.empty(len(arr))
    if len(arr) == 0:
        return out
    k = 2.0 / (period + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _load(symbol, base):
    # map symbol -> cache filename
    fn_variants = [
        f"raw_{base}_{symbol}.pkl",
        f"raw_{base}_{symbol.replace('.', '_')}.pkl",
        f"raw_{base}_{symbol.lower()}.pkl",
    ]
    for v in fn_variants:
        p = CACHE / v
        if p.exists():
            df = pickle.load(open(p, "rb"))
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            # Force ns resolution so DatetimeIndex.view("int64") and
            # Timestamp.value agree (pandas 2.x can store second-resolution
            # datetimes → view gives seconds, .value gives ns → key mismatch).
            try:
                df.index = df.index.as_unit("ns")
            except Exception:
                pass
            cols = ["open", "high", "low", "close"]
            if "tick_volume" in df.columns:
                cols.append("tick_volume")
            return df[cols]
    return None


def _resample(df, rule):
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    return out


def _find_swings(h4):
    """Return arrays of (idx_time, level) for confirmed swing highs and lows
    on the 4H frame using a symmetric fractal of SWING_LOOKBACK bars."""
    highs, lows = [], []
    H = h4["high"].values
    L = h4["low"].values
    T = h4.index
    n = len(h4)
    w = SWING_LOOKBACK
    for i in range(w, n - w):
        seg_h = H[i - w:i + w + 1]
        seg_l = L[i - w:i + w + 1]
        # store time as int64 nanoseconds to avoid tz-aware/naive comparison
        if H[i] == seg_h.max() and (seg_h == H[i]).sum() == 1:
            highs.append((T[i].value, H[i]))
        if L[i] == seg_l.min() and (seg_l == L[i]).sum() == 1:
            lows.append((T[i].value, L[i]))
    return highs, lows


def backtest(symbol, base="m15", verbose=True):
    df = _load(symbol, base)
    if df is None or len(df) < 500:
        if verbose:
            print(f"{symbol}: no/short {base} data")
        return None

    spread = SPREAD.get(symbol, 0.0002)

    # Build higher-timeframe frames from the base.
    h4 = _resample(df, "4h")
    d1 = _resample(df, "1D")
    if len(h4) < 50 or len(d1) < 200:
        if verbose:
            print(f"{symbol}: insufficient 4H/D1 depth ({len(h4)}/{len(d1)})")
        return None

    # Daily 200 EMA bias, mapped forward onto each base bar.
    d1 = d1.copy()
    d1["ema200"] = _ema(d1["close"].values, 200)
    d1["bias"] = np.where(d1["close"] > d1["ema200"], 1, -1)
    bias_series = d1["bias"].reindex(df.index, method="ffill")

    # Confirmed 4H swings.
    swing_highs, swing_lows = _find_swings(h4)

    base_t = df.index.view("int64")  # int64 ns — matches swing time keys
    base_o = df["open"].values
    base_h = df["high"].values
    base_l = df["low"].values
    base_c = df["close"].values
    n = len(df)

    # State machine: scan base bars. Maintain recently-confirmed 4H swing
    # levels; detect sweeps; then look for an FVG in the reversal direction;
    # then fill on midpoint retrace.
    sh_idx = 0  # pointer into swing_highs
    sl_idx = 0
    live_highs = []   # list of recent swing-high levels available to sweep
    live_lows = []

    # Sequential setup tracking. After a sweep we open a "window" on that side
    # and look for an FVG in the reversal direction within SWEEP_TO_FVG_BARS;
    # once an FVG forms we wait for a midpoint retrace fill within
    # SETUP_EXPIRY_BARS_15M. Both sides tracked independently.
    SWEEP_TO_FVG_BARS = 12        # ~3h on M15 to form the reversal FVG
    long_sweep = None             # {"swept_level", "sweep_i"} awaiting bullish FVG
    short_sweep = None
    long_fvg = None               # {"mid","top","bot","swept_level","fvg_i"} awaiting fill
    short_fvg = None
    trades = []

    bar_minutes = {"m15": 15, "m5": 5, "h1": 60}.get(base, 15)
    time_stop_bars = int(TIME_STOP_HOURS * 60 / bar_minutes)

    i = 2
    while i < n - 1:
        t = base_t[i]

        # Promote 4H swings confirmed up to now into the live sweep-target sets.
        while sh_idx < len(swing_highs) and swing_highs[sh_idx][0] <= t:
            live_highs.append(swing_highs[sh_idx][1]); sh_idx += 1
        while sl_idx < len(swing_lows) and swing_lows[sl_idx][0] <= t:
            live_lows.append(swing_lows[sl_idx][1]); sl_idx += 1
        live_highs = live_highs[-SWING_MEMORY:]
        live_lows = live_lows[-SWING_MEMORY:]

        bias = bias_series.iloc[i] if i < len(bias_series) else 0

        # ======================= LONG side =======================
        if bias == 1:
            # (1) FILL: pending bullish FVG → enter on retrace to midpoint.
            if long_fvg is not None:
                if i - long_fvg["fvg_i"] > SETUP_EXPIRY_BARS_15M:
                    long_fvg = None
                elif base_l[i] <= long_fvg["mid"]:   # price retraced into gap
                    tr = _simulate(i, 1, long_fvg["mid"], long_fvg["swept_level"],
                                   spread, base_t, base_o, base_h, base_l, base_c,
                                   n, time_stop_bars)
                    if tr:
                        trades.append(tr)
                    long_fvg = None
            # (2) FVG FORM: after a sweep, first bullish 3-candle gap.
            if long_fvg is None and long_sweep is not None:
                if i - long_sweep["sweep_i"] > SWEEP_TO_FVG_BARS:
                    long_sweep = None
                elif base_l[i] > base_h[i - 2]:      # bullish FVG: low3 > high1
                    top, bot = base_l[i], base_h[i - 2]
                    long_fvg = {"mid": (top + bot) / 2.0, "top": top, "bot": bot,
                                "swept_level": long_sweep["swept_level"], "fvg_i": i}
                    long_sweep = None
            # (3) SWEEP: low pierces a swing low then closes back above it.
            if long_sweep is None and long_fvg is None and live_lows:
                below = [L for L in live_lows if base_l[i] < L <= base_c[i]]
                if below:
                    swept_level = max(below)   # the highest swing low we swept
                    long_sweep = {"swept_level": swept_level, "sweep_i": i}

        # ======================= SHORT side =======================
        if bias == -1:
            if short_fvg is not None:
                if i - short_fvg["fvg_i"] > SETUP_EXPIRY_BARS_15M:
                    short_fvg = None
                elif base_h[i] >= short_fvg["mid"]:
                    tr = _simulate(i, -1, short_fvg["mid"], short_fvg["swept_level"],
                                   spread, base_t, base_o, base_h, base_l, base_c,
                                   n, time_stop_bars)
                    if tr:
                        trades.append(tr)
                    short_fvg = None
            if short_fvg is None and short_sweep is not None:
                if i - short_sweep["sweep_i"] > SWEEP_TO_FVG_BARS:
                    short_sweep = None
                elif base_h[i] < base_l[i - 2]:      # bearish FVG: high3 < low1
                    top, bot = base_l[i - 2], base_h[i]
                    short_fvg = {"mid": (top + bot) / 2.0, "top": top, "bot": bot,
                                 "swept_level": short_sweep["swept_level"], "fvg_i": i}
                    short_sweep = None
            if short_sweep is None and short_fvg is None and live_highs:
                above = [H for H in live_highs if base_h[i] > H >= base_c[i]]
                if above:
                    swept_level = min(above)
                    short_sweep = {"swept_level": swept_level, "sweep_i": i}

        i += 1

    # Optional measurement window: warm up indicators on full history but only
    # tally trades whose entry falls in the last MEASURE_LAST_N bars. Lets us
    # report e.g. "last 180 days" while keeping the 200-bar daily-EMA warmup.
    if MEASURE_LAST_N and n > MEASURE_LAST_N:
        cutoff = n - MEASURE_LAST_N
        trades = [t for t in trades if t and t.get("entry_i", 0) >= cutoff]
    return _summarize(symbol, trades)


def _simulate(entry_i, direction, entry_px, swept_level, spread,
              T, O, H, L, C, n, time_stop_bars):
    """Walk forward from entry bar; apply SL / TP1(1.5x,50%) / TP2(3x,50%) /
    entry-candle-breach / 4H time stop. Returns a trade dict (R-based pnl)."""
    # SL at the opposite side of the swept level (the sweep extreme).
    if direction == 1:
        sl = swept_level - spread          # just beyond the swept low
        stop_dist = entry_px - sl
    else:
        sl = swept_level + spread
        stop_dist = sl - entry_px
    # Guard: reject degenerate stops (entry ~= swept level → R explodes).
    # Require the stop to be at least ~5bp of price AND > 3x spread.
    min_stop = max(3.0 * spread, 0.0005 * entry_px)
    if stop_dist < min_stop:
        return None
    tp1 = entry_px + direction * TP1_R * stop_dist
    tp2 = entry_px + direction * TP2_R * stop_dist
    entry_low = L[entry_i]
    entry_high = H[entry_i]

    half = 0.5
    realized_R = 0.0
    tp1_hit = False
    remaining = 1.0

    end = min(n, entry_i + 1 + time_stop_bars + 200)
    for j in range(entry_i + 1, end):
        bars_held = j - entry_i
        hi, lo = H[j], L[j]
        if direction == 1:
            # entry-candle breach (before TP1): exit all at entry_low
            if (not DISABLE_BREACH) and not tp1_hit and lo < entry_low:
                realized_R += remaining * ((entry_low - entry_px) / stop_dist)
                remaining = 0.0; break
            # SL
            if lo <= sl:
                realized_R += remaining * (-1.0)
                remaining = 0.0; break
            # TP1
            if not tp1_hit and hi >= tp1:
                realized_R += half * TP1_R
                remaining -= half; tp1_hit = True
            # TP2
            if tp1_hit and hi >= tp2:
                realized_R += remaining * TP2_R
                remaining = 0.0; break
        else:
            if (not DISABLE_BREACH) and not tp1_hit and hi > entry_high:
                realized_R += remaining * ((entry_px - entry_high) / stop_dist)
                remaining = 0.0; break
            if hi >= sl:
                realized_R += remaining * (-1.0)
                remaining = 0.0; break
            if not tp1_hit and lo <= tp1:
                realized_R += half * TP1_R
                remaining -= half; tp1_hit = True
            if tp1_hit and lo <= tp2:
                realized_R += remaining * TP2_R
                remaining = 0.0; break
        # 4H time stop: if TP1 not hit within window, close all at this close
        if not tp1_hit and bars_held >= time_stop_bars:
            close_px = C[j]
            realized_R += remaining * (direction * (close_px - entry_px) / stop_dist)
            remaining = 0.0; break
    else:
        # ran out of data — mark to last close
        if remaining > 0:
            close_px = C[end - 1]
            realized_R += remaining * (direction * (close_px - entry_px) / stop_dist)

    return {"dir": direction, "entry_i": entry_i, "R": realized_R,
            "tp1_hit": tp1_hit, "stop_dist": stop_dist}


def _summarize(symbol, trades):
    trades = [t for t in trades if t]
    n = len(trades)
    if n == 0:
        return {"symbol": symbol, "trades": 0, "wr": 0, "pf": 0, "total_R": 0,
                "avg_R": 0, "tp1_rate": 0}
    Rs = np.array([t["R"] for t in trades])
    wins = Rs[Rs > 0]; losses = Rs[Rs <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    return {
        "symbol": symbol, "trades": n,
        "wr": (Rs > 0).mean(),
        "pf": pf,
        "total_R": Rs.sum(),
        "avg_R": Rs.mean(),
        "tp1_rate": np.mean([t["tp1_hit"] for t in trades]),
        "longs": sum(1 for t in trades if t["dir"] == 1),
        "shorts": sum(1 for t in trades if t["dir"] == -1),
    }


# ════════════════════════════════════════════════════════════════════════
#  STRATEGY 2 — Previous-Day-Liquidity Sweep + Delta Divergence
#  (CVD approximated by tick-volume signed by candle direction — NOT true
#   order-flow delta. Treat as a directional read, not authoritative.)
#  Entry: price pierces PDH/PDL by sweep_pips then closes back inside, AND
#         price makes LL/HH vs 20-bar swing while CVD makes HL/LH (divergence).
#  Exit : fixed-pip SL, TP = 2.5R, 2h time stop. 1% risk (R-based here).
# ════════════════════════════════════════════════════════════════════════

# price units per "pip" for the fixed-pip SL.
PIP = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01, "USDCAD": 0.0001,
    "USDCHF": 0.0001, "AUDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01,
    "CHFJPY": 0.01, "XAUUSD": 0.10, "XAGUSD": 0.01, "USOUSD": 0.01,
    # indices: 1 "pip" = 1 point
    "NAS100.r": 1.0, "SP500.r": 1.0, "DJ30.r": 1.0, "US2000.r": 1.0,
    "JPN225ft": 1.0, "SPI200.r": 1.0, "SWI20.r": 1.0, "UK100.r": 1.0,
    "BTCUSD": 1.0, "ETHUSD": 1.0, "XPTUSD.r": 0.10,
}
SL_PIPS = {  # spec: 10 forex, 15 gold; sensible defaults elsewhere
    "XAUUSD": 15, "XPTUSD.r": 15, "XAGUSD": 15,
    "NAS100.r": 30, "SP500.r": 15, "DJ30.r": 30, "US2000.r": 15,
    "JPN225ft": 30, "SPI200.r": 15, "SWI20.r": 15, "UK100.r": 20,
    "BTCUSD": 200, "ETHUSD": 20, "USOUSD": 20,
}
DELTA_SWING_LB = 20
DELTA_SWEEP_PIPS = 7        # pierce PDH/PDL by 5-10 pips
DELTA_TP_R = 2.5
DELTA_TIME_STOP_H = 2.0


def _est_hour(ts_ns):
    # crude EST = UTC-5 (ignores DST; fine for a noon-block filter)
    import datetime as _dt
    utc = _dt.datetime.utcfromtimestamp(ts_ns / 1e9)
    return (utc.hour - 5) % 24


def backtest_delta(symbol, base="m15", verbose=True):
    df = _load(symbol, base)
    if df is None or len(df) < 500 or "tick_volume" not in df.columns:
        if verbose:
            print(f"{symbol}: no data / no tick_volume")
        return None
    pip = PIP.get(symbol)
    if pip is None:
        return None
    sl_pips = SL_PIPS.get(symbol, 10)
    sl_dist = sl_pips * pip

    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    vol = df["tick_volume"].values.astype(float)
    t_ns = df.index.view("int64")
    n = len(df)

    # CVD proxy: signed tick volume cumulative
    signed = np.where(c > o, vol, np.where(c < o, -vol, 0.0))
    cvd = np.cumsum(signed)

    # previous-day high/low mapped onto each bar
    d1 = _resample(df, "1D")
    pdh = d1["high"].shift(1).reindex(df.index, method="ffill").values
    pdl = d1["low"].shift(1).reindex(df.index, method="ffill").values

    bar_min = {"m15": 15, "m5": 5, "h1": 60}.get(base, 15)
    time_stop_bars = int(DELTA_TIME_STOP_H * 60 / bar_min)
    sweep_band = DELTA_SWEEP_PIPS * pip

    trades = []
    lb = DELTA_SWING_LB
    i = lb + 1
    while i < n - 1:
        # session filter: block 12:00-13:00 EST
        if _est_hour(t_ns[i]) == 12:
            i += 1; continue
        if np.isnan(pdl[i]) or np.isnan(pdh[i]):
            i += 1; continue

        prior_low = l[i - lb:i].min()
        prior_high = h[i - lb:i].max()
        prior_cvd_low = cvd[i - lb:i].min()
        prior_cvd_high = cvd[i - lb:i].max()

        # LONG: sweep below PDL + bullish delta divergence
        bull_sweep = (l[i] < pdl[i] - sweep_band) and (c[i] > pdl[i])
        price_ll = l[i] < prior_low
        delta_hl = cvd[i] > prior_cvd_low
        if bull_sweep and price_ll and delta_hl:
            entry = c[i]
            tr = _simulate_delta(i, 1, entry, sl_dist, c, h, l, n, time_stop_bars)
            if tr: trades.append(tr)
            i += time_stop_bars; continue

        # SHORT: sweep above PDH + bearish delta divergence
        bear_sweep = (h[i] > pdh[i] + sweep_band) and (c[i] < pdh[i])
        price_hh = h[i] > prior_high
        delta_lh = cvd[i] < prior_cvd_high
        if bear_sweep and price_hh and delta_lh:
            entry = c[i]
            tr = _simulate_delta(i, -1, entry, sl_dist, c, h, l, n, time_stop_bars)
            if tr: trades.append(tr)
            i += time_stop_bars; continue

        i += 1

    return _summarize(symbol, trades)


def _simulate_delta(entry_i, direction, entry, sl_dist, C, H, L, n, time_stop_bars):
    if sl_dist <= 0:
        return None
    sl = entry - direction * sl_dist
    tp = entry + direction * DELTA_TP_R * sl_dist
    end = min(n, entry_i + 1 + time_stop_bars + 50)
    for j in range(entry_i + 1, end):
        hi, lo = H[j], L[j]
        if direction == 1:
            if lo <= sl:
                return {"dir": 1, "entry_i": entry_i, "R": -1.0, "tp1_hit": False}
            if hi >= tp:
                return {"dir": 1, "entry_i": entry_i, "R": DELTA_TP_R, "tp1_hit": True}
        else:
            if hi >= sl:
                return {"dir": -1, "entry_i": entry_i, "R": -1.0, "tp1_hit": False}
            if lo <= tp:
                return {"dir": -1, "entry_i": entry_i, "R": DELTA_TP_R, "tp1_hit": True}
        if j - entry_i >= time_stop_bars:
            r = direction * (C[j] - entry) / sl_dist
            return {"dir": direction, "entry_i": entry_i, "R": r, "tp1_hit": False}
    r = direction * (C[end - 1] - entry) / sl_dist
    return {"dir": direction, "entry_i": entry_i, "R": r, "tp1_hit": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--base", default="m15", choices=["m15", "m5", "h1"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--strategy", default="fvg", choices=["fvg", "delta"],
                    help="fvg = sweep+FVG retest; delta = PDH/PDL sweep + CVD-proxy divergence")
    args = ap.parse_args()

    if args.all:
        syms = sorted(SPREAD.keys())
    elif args.symbol:
        syms = [args.symbol]
    else:
        syms = ["USDJPY", "XAUUSD"] if args.strategy == "delta" else ["USDCAD"]

    runner = backtest_delta if args.strategy == "delta" else backtest
    if args.strategy == "delta":
        print(f"\nPDH/PDL Sweep + Delta-Divergence (CVD=tick-vol proxy) | base={args.base} | "
              f"fixed-pip SL, TP=2.5R, 2h time-stop, noon-EST block\n")
    else:
        print(f"\nICT Liquidity-Sweep + FVG-Retest | base={args.base} | SL=swept level, "
              f"TP1=1.5R(50%), TP2=3R(50%), 4H time-stop, entry-candle breach\n")
    print(f"{'Symbol':<10} {'Trades':>7} {'L/S':>9} {'WR':>6} {'TP1%':>6} "
          f"{'PF':>7} {'TotalR':>9} {'avgR':>7}")
    print("-" * 70)
    agg_R = 0.0; agg_n = 0
    for s in syms:
        r = runner(s, base=args.base, verbose=False)
        if not r or r["trades"] == 0:
            continue
        print(f"{r['symbol']:<10} {r['trades']:>7} "
              f"{r.get('longs',0):>4}/{r.get('shorts',0):<4} "
              f"{r['wr']*100:>5.1f}% {r['tp1_rate']*100:>5.0f}% "
              f"{r['pf']:>7.2f} {r['total_R']:>9.2f} {r['avg_R']:>7.3f}")
        agg_R += r["total_R"]; agg_n += r["trades"]
    print("-" * 70)
    print(f"{'TOTAL':<10} {agg_n:>7} {'':>9} {'':>6} {'':>6} {'':>7} {agg_R:>9.2f} R")
    print(f"\nNote: results in R-multiples (1R = stop distance). Spread cost is\n"
          f"baked into SL placement; slippage not modelled. M15 base only has\n"
          f"full history for USDCAD (50k bars); other symbols use whatever base\n"
          f"data exists.\n")


if __name__ == "__main__":
    main()
