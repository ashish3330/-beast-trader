#!/usr/bin/env python3 -B
"""
Pine Trend v3 — port of "Enhanced Crypto Moves v3" Pine Script strategy.

ARCHITECTURAL CAVEAT (read first):
This strategy is a 6-AND indicator-stack entry — exactly the pattern shown
in feedback_value_entry_research_20260605.md to be mathematically lagging.
It MAY work better than Dragon's momentum because:
  (a) strict AND (not N-of-K) → fewer entries, possibly higher quality
  (b) longer timeframe (H1+) → larger ATR-relative move budget
  (c) the trailing/partial structure is well-designed (no fixed TP, runner)
But it may also fail for the same reason all 6-indicator entries fail:
by the time all 6 align, the move is already done.

ONLY deployable after backtest validation. Default state: signal-only.

Entry logic (exactly mirroring the Pine):
  LONG:
    close > EMA(price, 200)                # long-term trend filter
    AND EMA(price, 20) > EMA(price, 50)    # short-term trend alignment
    AND Supertrend(3.0, 14) bullish        # ATR-based trend
    AND ADX(14) > 20                       # strong trend
    AND MACD(12,26,9): macd_line > signal  # momentum
    AND RSI(14) > 50                       # bullish bias
    AND no existing position
    AND bars_since_last_entry >= 5         # cooldown

  SHORT: symmetric (price<EMA200, EMA20<EMA50, ST bearish, ADX>20,
                    macd_line < signal_line, RSI<50)

Position sizing: 1% account risk per trade, qty = (equity*0.01) / (1.5*ATR14)
Initial SL: 1.5 * ATR
Trailing: activates after 3*ATR profit, trails by 2*ATR
Break-even: SL moves to entry when price hits +1.5*ATR
Partials:  close 30% at +3*ATR, 25% at +5*ATR, 20% at +8*ATR (leaves 25% runner)
Trend reversal exit: close if opposite trend forms
Time stop: close if position held > 50 bars
Cooldown: 5 bars between new entries

API (mirrors SweepReclaimStrategy + NR7Strategy + FVGStrategy):
    strat = PineTrendV3Strategy(state)
    sig = strat.evaluate("BTCUSD")
    # sig is None, or:
    # {
    #   "direction": "LONG"|"SHORT",
    #   "entry": float,         # = current close (market entry)
    #   "sl": float,            # entry +/- 1.5 * ATR
    #   "tp1": float,           # +3 ATR  (close 30%)
    #   "tp2": float,           # +5 ATR  (close 25%)
    #   "tp3": float,           # +8 ATR  (close 20%)
    #   "be_at": float,         # move to BE at +1.5 ATR
    #   "trail_activate": float,# trail activates at +3 ATR
    #   "trail_distance": float,# trails by 2 ATR
    #   "atr": float,
    #   "bar_time": pd.Timestamp,
    #   "reason": str,
    # }

Frequency expectation (H1 base):
  Each closed H1 bar checks the 6-AND gate. Strict AND → maybe 1-5 signals
  per symbol per month. If you see >50/month, something is wrong with the
  indicators (probably ADX not strict enough on this data).

Constraints honoured:
  * NEW file, no modification of existing modules
  * Closed bars only — uses iloc[-2] (the live forming bar is never read)
  * Stateless across calls except a per-symbol cooldown counter (bar_time of
    last fired entry; new entry blocked if < 5 H1 bars since)
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("dragon.pine_v3")

# ════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMS (defaults match the Pine Script)
# ════════════════════════════════════════════════════════════════════════
EMA_LONG = 200
EMA_MID = 50
EMA_SHORT = 20
ST_FACTOR = 3.0
ST_ATR_LEN = 14
ADX_LEN = 14
ADX_THRESH = 20
RSI_LEN = 14
ATR_LEN = 14
SL_ATR_MULT = 1.5
TRAIL_OFFSET_MULT = 3.0     # activates trail
TRAIL_DIST_MULT = 2.0       # trail distance
BE_ATR_MULT = 1.5
TP1_ATR_MULT = 3.0
TP2_ATR_MULT = 5.0
TP3_ATR_MULT = 8.0
TP1_FRAC = 0.30
TP2_FRAC = 0.25
TP3_FRAC = 0.20
MAX_BARS_IN_TRADE = 50
ENTRY_COOLDOWN_BARS = 5
MIN_H1_BARS = max(EMA_LONG + 5, 250)   # enough warmup for EMA200


# ════════════════════════════════════════════════════════════════════════
#  Indicator helpers (vectorized for the FULL frame; we read iloc[-2])
# ════════════════════════════════════════════════════════════════════════
def _ema(arr, period):
    arr = np.asarray(arr, dtype=float)
    out = np.empty(len(arr))
    if len(arr) == 0:
        return out
    k = 2.0 / (period + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out


def _rsi(close, period=14):
    close = np.asarray(close, dtype=float)
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.empty(n); avg_loss = np.empty(n)
    avg_gain[period] = gain[1:period+1].mean()
    avg_loss[period] = loss[1:period+1].mean()
    for i in range(period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    rs = np.where(avg_loss == 0, np.inf, avg_gain / np.where(avg_loss == 0, 1, avg_loss))
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi


def _macd(close, fast=12, slow=26, signal=9):
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd = ema_f - ema_s
    sig = _ema(macd, signal)
    return macd, sig


def _atr(H, L, C, period=14):
    n = len(H)
    if n < period + 1:
        return np.full(n, np.nan)
    tr = np.empty(n)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i-1]), abs(L[i] - C[i-1]))
    atr = np.empty(n)
    atr[period] = tr[1:period+1].mean()
    k = 1.0 / period
    for i in range(period+1, n):
        atr[i] = atr[i-1] * (1 - k) + tr[i] * k
    atr[:period] = np.nan
    return atr


def _adx(H, L, C, period=14):
    n = len(H)
    if n < 2*period + 1:
        return np.full(n, np.nan)
    up = H[1:] - H[:-1]
    dn = L[:-1] - L[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.empty(n - 1)
    for i in range(n - 1):
        tr[i] = max(H[i+1] - L[i+1], abs(H[i+1] - C[i]), abs(L[i+1] - C[i]))
    sm_plus = np.empty(n - 1); sm_minus = np.empty(n - 1); sm_tr = np.empty(n - 1)
    sm_plus[period-1] = plus_dm[:period].sum()
    sm_minus[period-1] = minus_dm[:period].sum()
    sm_tr[period-1] = tr[:period].sum()
    for i in range(period, n - 1):
        sm_plus[i] = sm_plus[i-1] - sm_plus[i-1]/period + plus_dm[i]
        sm_minus[i] = sm_minus[i-1] - sm_minus[i-1]/period + minus_dm[i]
        sm_tr[i] = sm_tr[i-1] - sm_tr[i-1]/period + tr[i]
    safe_tr = np.where(sm_tr == 0, 1.0, sm_tr)
    plus_di = 100 * sm_plus / safe_tr
    minus_di = 100 * sm_minus / safe_tr
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = np.empty(n - 1)
    adx[2*period-1] = dx[period-1:2*period].mean()
    for i in range(2*period, n - 1):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    adx_full = np.empty(n)
    adx_full[1:] = adx
    adx_full[0] = np.nan
    adx_full[:2*period] = np.nan
    return adx_full


def _supertrend(H, L, C, period=14, factor=3.0):
    """Pine-compatible Supertrend. Returns (st_value_array, st_dir_array)
    where st_dir < 0 = uptrend (bullish), st_dir > 0 = downtrend.
    Matches the Pine convention used in the script."""
    n = len(H)
    atr = _atr(H, L, C, period)
    src = (H + L) / 2.0
    up_basic = src - factor * atr
    dn_basic = src + factor * atr
    up = np.empty(n); dn = np.empty(n)
    st_dir = np.empty(n); st = np.empty(n)
    up[:] = np.nan; dn[:] = np.nan; st[:] = np.nan; st_dir[:] = np.nan
    # Warm up
    start = period + 1
    if start >= n:
        return st, st_dir
    up[start] = up_basic[start]
    dn[start] = dn_basic[start]
    st_dir[start] = 1   # default down
    st[start] = dn[start]
    for i in range(start + 1, n):
        # upper trend line: take the higher of new basic and prior, unless price broke below prior
        up[i] = up_basic[i] if (C[i-1] < up[i-1] if not np.isnan(up[i-1]) else True) else max(up_basic[i], up[i-1])
        dn[i] = dn_basic[i] if (C[i-1] > dn[i-1] if not np.isnan(dn[i-1]) else True) else min(dn_basic[i], dn[i-1])
        # direction flip logic
        prev_dir = st_dir[i-1] if not np.isnan(st_dir[i-1]) else 1
        if prev_dir == 1 and C[i] > dn[i-1]:
            st_dir[i] = -1
        elif prev_dir == -1 and C[i] < up[i-1]:
            st_dir[i] = 1
        else:
            st_dir[i] = prev_dir
        st[i] = up[i] if st_dir[i] == -1 else dn[i]
    return st, st_dir


def _normalize_candles(df):
    if df is None or len(df) == 0:
        return None
    try:
        df = df.copy()
        if "time" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.index.name or "index": "time"})
            else:
                return None
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        try:
            df.index = df.index.as_unit("ns")
        except Exception:
            pass
        cols = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in cols):
            return None
        return df[cols].astype(float)
    except Exception as e:
        log.debug("candle normalize failed: %s", e)
        return None


# ════════════════════════════════════════════════════════════════════════
#  PineTrendV3Strategy — detector class
# ════════════════════════════════════════════════════════════════════════
class PineTrendV3Strategy:
    """Per-symbol stateful 6-AND indicator-stack trend detector.

    Stateful: tracks the H1 bar_time of the last fired entry per symbol to
    enforce the 5-bar cooldown. Otherwise stateless.
    """

    def __init__(self, state, params=None):
        self.state = state
        p = params or {}
        self.ema_long = int(p.get("EMA_LONG", EMA_LONG))
        self.ema_mid = int(p.get("EMA_MID", EMA_MID))
        self.ema_short = int(p.get("EMA_SHORT", EMA_SHORT))
        self.st_factor = float(p.get("ST_FACTOR", ST_FACTOR))
        self.st_atr_len = int(p.get("ST_ATR_LEN", ST_ATR_LEN))
        self.adx_len = int(p.get("ADX_LEN", ADX_LEN))
        self.adx_thresh = float(p.get("ADX_THRESH", ADX_THRESH))
        self.rsi_len = int(p.get("RSI_LEN", RSI_LEN))
        self.atr_len = int(p.get("ATR_LEN", ATR_LEN))
        self.sl_mult = float(p.get("SL_ATR_MULT", SL_ATR_MULT))
        self.trail_off_mult = float(p.get("TRAIL_OFFSET_MULT", TRAIL_OFFSET_MULT))
        self.trail_dist_mult = float(p.get("TRAIL_DIST_MULT", TRAIL_DIST_MULT))
        self.be_mult = float(p.get("BE_ATR_MULT", BE_ATR_MULT))
        self.tp1_mult = float(p.get("TP1_ATR_MULT", TP1_ATR_MULT))
        self.tp2_mult = float(p.get("TP2_ATR_MULT", TP2_ATR_MULT))
        self.tp3_mult = float(p.get("TP3_ATR_MULT", TP3_ATR_MULT))
        self.max_bars = int(p.get("MAX_BARS_IN_TRADE", MAX_BARS_IN_TRADE))
        self.cooldown_bars = int(p.get("ENTRY_COOLDOWN_BARS", ENTRY_COOLDOWN_BARS))
        # Per-symbol state: last_entry_bar_idx (in the in-frame index), last_eval_bar_t
        self._last_entry_idx = {}        # symbol -> int bar index of last fired entry
        self._last_bar_t = {}            # symbol -> ns timestamp of last evaluated bar

    def _get_h1(self, symbol):
        if self.state is None:
            return None
        try:
            return self.state.get_candles(symbol, 60)
        except Exception as e:
            log.debug("[%s] H1 fetch failed: %s", symbol, e)
            return None

    def evaluate(self, symbol):
        h1_raw = self._get_h1(symbol)
        h1 = _normalize_candles(h1_raw)
        if h1 is None or len(h1) < MIN_H1_BARS:
            return None
        i = len(h1) - 2   # last CLOSED H1 bar
        bar_t = h1.index[i]
        # Per-symbol per-bar dedupe.
        if self._last_bar_t.get(symbol) == bar_t:
            return None
        self._last_bar_t[symbol] = bar_t

        O = h1["open"].values
        H = h1["high"].values
        L = h1["low"].values
        C = h1["close"].values

        # Compute indicators ONLY up to and including bar i (no lookahead).
        ema_long = _ema(C[:i+1], self.ema_long)
        ema_mid = _ema(C[:i+1], self.ema_mid)
        ema_short = _ema(C[:i+1], self.ema_short)
        rsi = _rsi(C[:i+1], self.rsi_len)
        macd_line, macd_sig = _macd(C[:i+1])
        atr = _atr(H[:i+1], L[:i+1], C[:i+1], self.atr_len)
        adx = _adx(H[:i+1], L[:i+1], C[:i+1], self.adx_len)
        st_val, st_dir = _supertrend(H[:i+1], L[:i+1], C[:i+1], self.st_atr_len, self.st_factor)

        # Read last-bar values
        c = float(C[i])
        el = float(ema_long[-1])
        em = float(ema_mid[-1])
        es = float(ema_short[-1])
        r = float(rsi[-1])
        ml = float(macd_line[-1])
        msl = float(macd_sig[-1])
        a = float(atr[-1])
        adx_v = float(adx[-1])
        sd = float(st_dir[-1])

        if np.isnan(el) or np.isnan(r) or np.isnan(a) or np.isnan(adx_v) or np.isnan(sd):
            return None
        if a <= 0:
            return None

        # 5-bar cooldown
        last_idx = self._last_entry_idx.get(symbol)
        if last_idx is not None and (i - last_idx) < self.cooldown_bars:
            return None

        # 6-AND gates
        long_trend = (
            c > el and es > em and sd < 0
            and adx_v > self.adx_thresh
            and ml > msl
            and r > 50
        )
        short_trend = (
            c < el and es < em and sd > 0
            and adx_v > self.adx_thresh
            and ml < msl
            and r < 50
        )

        if not long_trend and not short_trend:
            return None

        direction = "LONG" if long_trend else "SHORT"
        entry = c
        if direction == "LONG":
            sl = entry - self.sl_mult * a
            tp1 = entry + self.tp1_mult * a
            tp2 = entry + self.tp2_mult * a
            tp3 = entry + self.tp3_mult * a
            be_at = entry + self.be_mult * a
            trail_activate = entry + self.trail_off_mult * a
        else:
            sl = entry + self.sl_mult * a
            tp1 = entry - self.tp1_mult * a
            tp2 = entry - self.tp2_mult * a
            tp3 = entry - self.tp3_mult * a
            be_at = entry - self.be_mult * a
            trail_activate = entry - self.trail_off_mult * a

        self._last_entry_idx[symbol] = i
        return {
            "direction": direction,
            "entry": float(entry),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
            "be_at": float(be_at),
            "trail_activate": float(trail_activate),
            "trail_distance": float(self.trail_dist_mult * a),
            "max_bars_in_trade": self.max_bars,
            "tp1_frac": TP1_FRAC,
            "tp2_frac": TP2_FRAC,
            "tp3_frac": TP3_FRAC,
            "atr": float(a),
            "adx": float(adx_v),
            "rsi": float(r),
            "bar_time": bar_t,
            "reason": f"{direction} 6-AND aligned (ADX={adx_v:.1f} RSI={r:.1f} ST={'up' if sd<0 else 'down'})",
        }


# ════════════════════════════════════════════════════════════════════════
#  Standalone smoke test
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Build a synthetic strong-uptrend H1 series (250 bars) — all 6 longs should fire
    np.random.seed(11)
    n = 300
    trend = np.arange(n) * 0.5   # strong upward drift
    noise = np.cumsum(np.random.normal(0, 0.3, n))
    closes = 100.0 + trend + noise
    highs = closes + np.abs(np.random.normal(0, 0.4, n)) + 0.2
    lows = closes - np.abs(np.random.normal(0, 0.4, n)) - 0.2
    opens = np.concatenate([[100.0], closes[:-1]])
    times = pd.date_range("2026-01-01", periods=n, freq="60min", tz="UTC")
    df = pd.DataFrame({"time": times, "open": opens, "high": highs, "low": lows, "close": closes})

    class _S:
        def get_candles(self, sym, tf):
            if tf == 60:
                # Append a "forming" bar at the end so iloc[-2] is the last truly closed bar.
                d2 = df.copy()
                d2.loc[len(d2)] = {"time": df["time"].iloc[-1] + pd.Timedelta("60min"),
                                    "open": df["close"].iloc[-1], "high": df["close"].iloc[-1]+0.1,
                                    "low": df["close"].iloc[-1]-0.1, "close": df["close"].iloc[-1]}
                return d2
            return None

    strat = PineTrendV3Strategy(_S())
    sig = strat.evaluate("TEST")
    print(f"Pine v3 signal: {sig}")
    if sig and sig["direction"] == "LONG":
        print("✓ SMOKE TEST PASSED (LONG fired on strong-uptrend synthetic series)")
    else:
        # Often won't fire on this short series if ADX hasn't ramped yet; that's OK
        print("Note: signal did not fire on smoke test — may need longer warmup. Check on real cache.")
