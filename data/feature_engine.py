"""
Beast Trader — Feature Engine.
Generates ML features from tick/candle data for LightGBM model.
Output: normalized numpy array ready for prediction.
"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from collections import deque

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOLS, TIMEFRAMES, PREDICTION_HORIZON_TF

log = logging.getLogger("beast.features")

# Feature names — order must match training and inference
FEATURE_NAMES = [
    # Price returns
    "ret_1m", "ret_5m", "ret_15m", "ret_1h",
    # Volatility
    "vol_1m", "vol_5m", "vol_15m",
    # Momentum
    "momentum_5m", "momentum_15m", "momentum_1h",
    # Volume features
    "tick_count_1m", "volume_delta_1m", "vwap_deviation",
    # Microstructure
    "spread_abs", "spread_pct", "spread_percentile", "tick_direction",
    # Technical indicators
    "rsi", "macd_hist", "bb_position", "supertrend_dist", "supertrend_dir",
    "ema20_dist", "ema50_dist", "ema200_dist",
    # Regime
    "vol_percentile", "adx", "trend_strength",
    # ATR-normalized
    "atr_norm",
]

NUM_FEATURES = len(FEATURE_NAMES)


class FeatureEngine:
    """Generate ML features from shared state."""

    def __init__(self, state):
        self.state = state
        # Rolling window of raw feature vectors for z-score normalization
        self._rolling_buf = {}  # symbol -> deque of np.ndarray
        self._window = 500

    def generate(self, symbol) -> np.ndarray:
        """
        Generate feature vector for a symbol.
        Returns numpy array of shape (NUM_FEATURES,) or None if insufficient data.
        """
        tick = self.state.get_tick(symbol)
        if tick is None:
            return None

        ind = self.state.get_indicators(symbol)
        if not ind:
            return None

        # Get candle data for multiple timeframes
        m1 = self.state.get_candles(symbol, 1)
        m5 = self.state.get_candles(symbol, 5)
        m15 = self.state.get_candles(symbol, 15)
        h1 = self.state.get_candles(symbol, 60)

        if m1 is None or m5 is None or m15 is None:
            return None
        if len(m1) < 20 or len(m5) < 20 or len(m15) < 20:
            return None

        mid = (tick.bid + tick.ask) / 2.0
        features = np.zeros(NUM_FEATURES, dtype=np.float64)

        try:
            # ═══ PRICE RETURNS ═══
            features[0] = self._return(m1, 1)    # ret_1m
            features[1] = self._return(m5, 1)    # ret_5m
            features[2] = self._return(m15, 1)   # ret_15m
            features[3] = self._return(h1, 1) if h1 is not None and len(h1) > 1 else 0.0  # ret_1h

            # ═══ VOLATILITY ═══
            features[4] = self._volatility(m1, 20)   # vol_1m
            features[5] = self._volatility(m5, 20)   # vol_5m
            features[6] = self._volatility(m15, 20)  # vol_15m

            # ═══ MOMENTUM ═══
            features[7] = self._momentum(m5, 5)     # momentum_5m
            features[8] = self._momentum(m15, 5)    # momentum_15m
            features[9] = self._momentum(h1, 5) if h1 is not None and len(h1) > 5 else 0.0  # momentum_1h

            # ═══ VOLUME FEATURES ═══
            tick_hist = self.state.get_tick_history(symbol, 120)
            features[10] = float(len(tick_hist))  # tick_count_1m (approx)

            # Volume delta: up ticks vs down ticks
            if len(tick_hist) > 1:
                ups = sum(1 for i in range(1, len(tick_hist)) if tick_hist[i].bid > tick_hist[i-1].bid)
                downs = sum(1 for i in range(1, len(tick_hist)) if tick_hist[i].bid < tick_hist[i-1].bid)
                total = ups + downs
                features[11] = (ups - downs) / total if total > 0 else 0.0  # volume_delta_1m
            else:
                features[11] = 0.0

            # VWAP deviation
            vwap = ind.get("vwap", mid)
            features[12] = (mid - vwap) / vwap * 100 if vwap > 0 else 0.0  # vwap_deviation

            # ═══ MICROSTRUCTURE ═══
            spread = tick.ask - tick.bid
            features[13] = spread  # spread_abs
            features[14] = spread / mid * 10000 if mid > 0 else 0.0  # spread_pct (in pips)

            # Spread percentile from recent ticks
            if len(tick_hist) > 10:
                spreads = [t.ask - t.bid for t in tick_hist]
                features[15] = float(np.searchsorted(np.sort(spreads), spread)) / len(spreads)
            else:
                features[15] = 0.5  # spread_percentile

            # Tick direction (last N ticks)
            if len(tick_hist) >= 5:
                last5 = tick_hist[-5:]
                dirs = [1 if last5[i].bid > last5[i-1].bid else (-1 if last5[i].bid < last5[i-1].bid else 0)
                        for i in range(1, len(last5))]
                features[16] = float(np.mean(dirs))  # tick_direction
            else:
                features[16] = 0.0

            # ═══ TECHNICAL INDICATORS ═══
            features[17] = ind.get("rsi", 50.0)  # rsi (already 0-100)
            features[18] = ind.get("macd_hist", 0.0)  # macd_hist

            # BB position: where price is in BB range (0=lower, 1=upper)
            bb_upper = ind.get("bb_upper", mid)
            bb_lower = ind.get("bb_lower", mid)
            bb_range = bb_upper - bb_lower
            features[19] = (mid - bb_lower) / bb_range if bb_range > 0 else 0.5  # bb_position

            # SuperTrend distance (normalized by ATR)
            st = ind.get("supertrend", mid)
            atr = ind.get("atr", 1.0)
            features[20] = (mid - st) / atr if atr > 0 else 0.0  # supertrend_dist
            features[21] = float(ind.get("supertrend_dir", 0))   # supertrend_dir

            # EMA distances (normalized by ATR)
            features[22] = (mid - ind.get("ema20", mid)) / atr if atr > 0 else 0.0   # ema20_dist
            features[23] = (mid - ind.get("ema50", mid)) / atr if atr > 0 else 0.0   # ema50_dist
            features[24] = (mid - ind.get("ema200", mid)) / atr if atr > 0 else 0.0  # ema200_dist

            # ═══ REGIME ═══
            # Volatility percentile (current vol vs rolling)
            if len(m15) >= 50:
                vol_series = m15["close"].pct_change().rolling(14).std().dropna().values
                if len(vol_series) > 0:
                    current_vol = vol_series[-1]
                    features[25] = float(np.searchsorted(np.sort(vol_series), current_vol)) / len(vol_series)
                else:
                    features[25] = 0.5
            else:
                features[25] = 0.5  # vol_percentile

            features[26] = ind.get("adx", 25.0)  # adx

            # Trend strength: EMA alignment score
            ema20 = ind.get("ema20", mid)
            ema50 = ind.get("ema50", mid)
            ema200 = ind.get("ema200", mid)
            bullish = (1 if ema20 > ema50 else -1) + (1 if ema50 > ema200 else -1) + (1 if mid > ema20 else -1)
            features[27] = float(bullish) / 3.0  # trend_strength (-1 to 1)

            # ATR normalized by price
            features[28] = atr / mid * 100 if mid > 0 else 0.0  # atr_norm

            # ═══ Z-SCORE NORMALIZATION ═══
            features = self._normalize(symbol, features)

        except Exception as e:
            log.warning("[%s] Feature generation error: %s", symbol, e)
            return None

        return features

    def generate_batch(self, symbol, candles_df, horizon=None):
        """
        Generate features + labels from historical candle data for training.
        Returns (X, y) where y is forward return sign at horizon.
        horizon defaults to PREDICTION_HORIZON_TF from config.
        """
        if horizon is None:
            horizon = PREDICTION_HORIZON_TF

        if candles_df is None or len(candles_df) < 200:
            return None, None

        close = candles_df["close"].values.astype(float)
        high = candles_df["high"].values.astype(float)
        low = candles_df["low"].values.astype(float)
        n = len(close)

        # Pre-compute indicators over full series
        from data.tick_streamer import TickStreamer
        ema20 = TickStreamer._ema(close, 20)
        ema50 = TickStreamer._ema(close, 50)
        ema200 = TickStreamer._ema(close, min(200, n))
        atr_raw = TickStreamer._atr(high, low, close, 14)
        rsi_raw = TickStreamer._rsi(close, 14)
        macd_line, macd_signal, macd_hist = TickStreamer._macd(close)
        st, st_dir = TickStreamer._supertrend(high, low, close)
        bb_upper, bb_mid, bb_lower = TickStreamer._bollinger(close)
        adx_raw = TickStreamer._adx(high, low, close)

        # Pad indicator arrays to match close length
        def pad(arr, target_len):
            if len(arr) >= target_len:
                return arr[-target_len:]
            return np.concatenate([np.full(target_len - len(arr), arr[0] if len(arr) > 0 else 0), arr])

        atr = pad(atr_raw, n)
        rsi = pad(rsi_raw, n)
        adx = pad(adx_raw, n)

        vol_1m = pd.Series(close).pct_change().rolling(20).std().fillna(0).values
        vol_5m = pd.Series(close).pct_change(5).rolling(20).std().fillna(0).values
        vol_15m = pd.Series(close).pct_change(15).rolling(20).std().fillna(0).values

        # Pre-compute tick_volume and spread arrays once (not inside the loop)
        tv = candles_df["tick_volume"].values.astype(float) if "tick_volume" in candles_df.columns else np.ones(n, dtype=float)

        # Spread: use column if available, otherwise approximate from high-low range
        if "spread" in candles_df.columns:
            spread_vals = candles_df["spread"].values.astype(float)
        else:
            # Proxy: high - low minus abs(open - close) captures non-body range
            spread_vals = np.maximum(0.0, (high - low) - np.abs(candles_df["open"].values.astype(float) - close))

        # VWAP proxy: rolling 20-bar volume-weighted average price
        vwap = np.full(n, np.nan, dtype=np.float64)
        for vi in range(20, n):
            pv = np.sum(close[vi-20:vi] * tv[vi-20:vi])
            sv = np.sum(tv[vi-20:vi])
            vwap[vi] = pv / sv if sv > 0 else close[vi]

        # Forward returns for labels
        forward_returns = np.zeros(n)
        for i in range(n - horizon):
            forward_returns[i] = (close[i + horizon] - close[i]) / close[i]

        # Build feature matrix
        start_idx = 200  # need enough history
        end_idx = n - horizon  # need forward return
        if start_idx >= end_idx:
            return None, None

        X = np.zeros((end_idx - start_idx, NUM_FEATURES))
        y = np.zeros(end_idx - start_idx)

        for j, i in enumerate(range(start_idx, end_idx)):
            mid = close[i]
            a = atr[i] if atr[i] > 0 else 1.0

            # Price returns
            X[j, 0] = (close[i] - close[i-1]) / close[i-1] if close[i-1] > 0 else 0
            X[j, 1] = (close[i] - close[i-5]) / close[i-5] if close[i-5] > 0 else 0
            X[j, 2] = (close[i] - close[i-15]) / close[i-15] if close[i-15] > 0 else 0
            X[j, 3] = (close[i] - close[i-60]) / close[i-60] if i >= 60 and close[i-60] > 0 else 0

            # Volatility
            X[j, 4] = vol_1m[i]
            X[j, 5] = vol_5m[i]
            X[j, 6] = vol_15m[i]

            # Momentum (rate of change)
            X[j, 7] = (close[i] - close[i-5]) / (a * 5) if a > 0 else 0
            X[j, 8] = (close[i] - close[i-15]) / (a * 15) if a > 0 else 0
            X[j, 9] = (close[i] - close[i-60]) / (a * 60) if i >= 60 and a > 0 else 0

            # Volume features (from candle data)
            X[j, 10] = float(tv[i])
            vol_window = tv[max(0, i-20):i]
            X[j, 11] = (float(tv[i]) - float(np.mean(vol_window))) / (float(np.std(vol_window)) + 1e-8) if len(vol_window) > 0 else 0.0

            # VWAP deviation — computed from rolling VWAP
            if not np.isnan(vwap[i]) and vwap[i] > 0:
                X[j, 12] = (mid - vwap[i]) / vwap[i] * 100
            else:
                X[j, 12] = 0.0

            # Microstructure
            X[j, 13] = float(spread_vals[i])
            X[j, 14] = float(spread_vals[i]) / mid * 10000 if mid > 0 else 0

            # Spread percentile — rolling 100-bar window
            sp_window = spread_vals[max(0, i-100):i+1]
            if len(sp_window) > 5:
                X[j, 15] = float(np.searchsorted(np.sort(sp_window), spread_vals[i])) / len(sp_window)
            else:
                X[j, 15] = 0.5

            X[j, 16] = 1.0 if close[i] > close[i-1] else (-1.0 if close[i] < close[i-1] else 0.0)

            # Technical
            X[j, 17] = rsi[i]
            X[j, 18] = macd_hist[i]
            bb_range = bb_upper[i] - bb_lower[i]
            X[j, 19] = (mid - bb_lower[i]) / bb_range if bb_range > 0 else 0.5
            X[j, 20] = (mid - st[i]) / a if a > 0 else 0
            X[j, 21] = float(st_dir[i])
            X[j, 22] = (mid - ema20[i]) / a if a > 0 else 0
            X[j, 23] = (mid - ema50[i]) / a if a > 0 else 0
            X[j, 24] = (mid - ema200[i]) / a if a > 0 else 0

            # Regime
            vol_window = vol_15m[max(0, i-100):i+1]
            if len(vol_window) > 0 and vol_15m[i] > 0:
                X[j, 25] = float(np.searchsorted(np.sort(vol_window), vol_15m[i])) / len(vol_window)
            else:
                X[j, 25] = 0.5
            X[j, 26] = adx[i]
            bullish = (1 if ema20[i] > ema50[i] else -1) + (1 if ema50[i] > ema200[i] else -1) + (1 if mid > ema20[i] else -1)
            X[j, 27] = float(bullish) / 3.0
            X[j, 28] = a / mid * 100 if mid > 0 else 0

            # Label: forward return sign
            y[j] = 1.0 if forward_returns[i] > 0 else (0.0 if forward_returns[i] < 0 else 0.5)

        return X, y

    def _return(self, df, periods=1):
        """Calculate return over N candles."""
        if df is None or len(df) < periods + 1:
            return 0.0
        close = df["close"].values
        return float((close[-1] - close[-1 - periods]) / close[-1 - periods]) if close[-1 - periods] > 0 else 0.0

    def _volatility(self, df, window=20):
        """Rolling standard deviation of returns."""
        if df is None or len(df) < window:
            return 0.0
        returns = np.diff(df["close"].values[-window - 1:]) / df["close"].values[-window - 1:-1]
        return float(np.std(returns)) if len(returns) > 0 else 0.0

    def _momentum(self, df, periods=5):
        """Rate of change over N candles."""
        if df is None or len(df) < periods + 1:
            return 0.0
        close = df["close"].values
        return float((close[-1] - close[-1 - periods]) / close[-1 - periods]) if close[-1 - periods] > 0 else 0.0

    def _normalize(self, symbol, features):
        """Z-score normalization using a rolling deque of the last 500 values."""
        if symbol not in self._rolling_buf:
            self._rolling_buf[symbol] = deque(maxlen=self._window)

        buf = self._rolling_buf[symbol]
        buf.append(features.copy())

        if len(buf) < 10:
            # Not enough data for normalization — return raw
            return features

        arr = np.array(buf)  # (N, NUM_FEATURES)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        # Don't normalize RSI (already bounded 0-100), supertrend_dir (categorical), bb_position (0-1)
        skip_indices = {17, 19, 21}  # rsi, bb_position, supertrend_dir
        normalized = np.copy(features)
        for i in range(NUM_FEATURES):
            if i not in skip_indices and std[i] > 1e-8:
                normalized[i] = (features[i] - mean[i]) / std[i]
                normalized[i] = np.clip(normalized[i], -5.0, 5.0)  # clip outliers

        return normalized

    def get_feature_names(self):
        return FEATURE_NAMES.copy()
