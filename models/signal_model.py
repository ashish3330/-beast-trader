"""
Beast Trader — Meta-Labeling Signal Model.
Instead of predicting direction (coin flip), predicts whether a scored signal
will be profitable. Uses the proven momentum scoring system as the base signal
generator, then trains LightGBM to filter winners from losers.

Walk-forward: train 70%, validate 15%, test 15%.

Ensemble stacking (v2): 3 diverse models per symbol, AUC-weighted averaging.
Feature drift detection: warns + reduces confidence when live features diverge.
"""
import os
import time
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_DIR, SYMBOLS, CONFIDENCE_THRESHOLD, ATR_SL_MULTIPLIER,
    TRAIL_STEPS,
)
from signals.momentum_scorer import (
    _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, REGIME_PARAMS,
    MIN_SCORE,
)

log = logging.getLogger("beast.model")

# ── Per-symbol tuned LightGBM hyperparameters ──
# Grid-searched 2026-04-17: Phase 1 (80 combos lr x leaves x depth),
# Phase 2 (81 combos regularization) per symbol.
TUNED_LGB_PARAMS = {
    # AUC=0.7763  PF=2.16  Prec@conf=0.750
    "XAUUSD": {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.7,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
    },
    # AUC=0.8033  PF=2.28  Prec@conf=0.740
    "XAGUSD": {
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
    },
}
# Default params for symbols without per-symbol tuning
DEFAULT_LGB_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
}

# ── Ensemble Model B: high-regularization LightGBM (diversity via constraint) ──
ENSEMBLE_B_LGB_PARAMS = {
    "learning_rate": 0.01,
    "num_leaves": 15,
    "max_depth": 4,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "lambda_l1": 3.0,
    "lambda_l2": 5.0,
}

# ── Ensemble Model C: ExtraTreesClassifier params ──
ENSEMBLE_C_ET_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "max_features": 0.7,
    "n_jobs": -1,
    "random_state": 42,
}

# ── Feature drift config ──
DRIFT_THRESHOLD = 0.30   # fraction of features drifting to trigger warning
DRIFT_STD_MULT  = 3.0    # number of std devs to count as drifting
DRIFT_PENALTY   = 0.80   # multiply confidence when drift detected

# Meta-label feature names
META_FEATURE_NAMES = [
    # Score sub-components (from _score)
    "long_score", "short_score", "chosen_score",
    # Direction encoding
    "direction",  # +1 long, -1 short
    # Indicator context
    "adx", "bb_width", "atr_percentile",
    "rsi", "supertrend_dist", "ema_alignment",
    "macd_hist_norm",
    # Regime
    "vol_percentile", "trend_structure",
    # Time features
    "hour_of_day_sin", "hour_of_day_cos",
    "day_of_week_sin", "day_of_week_cos",
    # Microstructure
    "spread_atr_ratio",
    # Recent performance
    "recent_win_streak",
    # Score quality
    "score_margin",  # chosen_score - opposite_score
    "score_vs_threshold",  # how far above MIN_SCORE
    # ── Multi-timeframe features (real M15/M5 data) ──
    "m15_rsi",             # M15 RSI (real 14-period)
    "m15_ema_align",       # M15 EMA 8/21 alignment with direction
    "m15_atr_ratio",       # M15 ATR / H1 ATR (volatility expansion)
    "m15_macd_hist",       # M15 MACD histogram (normalized by ATR)
    "m15_adx",             # M15 ADX (trend strength on lower TF)
    "m5_rsi",              # M5 RSI (micro momentum)
    "m5_ema_align",        # M5 EMA 8/21 alignment with direction
    "m5_atr_ratio",        # M5 ATR / H1 ATR
    "m5_momentum",         # M5 price change over last 6 bars (30min)
    "mtf_agreement",       # How many TFs agree on direction (-1 to +1)
    "m15_bb_position",     # M15 price position within Bollinger Bands (0-1)
    "m5_consec_candles",   # M5 consecutive same-direction candles
    # ── Momentum persistence (33-36) ──
    "ret_1bar",            # 1-bar return (immediate momentum)
    "ret_3bar",            # 3-bar return
    "ret_5bar",            # 5-bar return
    "consec_candles",      # consecutive same-direction candles
    # ── Cross-asset / macro (37-38) ──
    "atr_change",          # ATR expansion/contraction (atr / atr_5ago)
    "bb_squeeze",          # 1 if BB width < 20th percentile (breakout setup)
    # ── Reversal detection (39-41) ──
    "rsi_divergence",      # price making new high but RSI lower (bearish divergence)
    "dist_from_high_20",   # distance from 20-bar high (% of ATR)
    "dist_from_low_20",    # distance from 20-bar low (% of ATR)
]
NUM_META_FEATURES = len(META_FEATURE_NAMES)


def _compute_tf_indicators(df):
    """Compute basic indicators on any timeframe DataFrame.
    Returns dict with: rsi, ema8, ema21, adx, atr, macd_hist, bbw, bb_upper, bb_lower, close, high, low, consec.
    """
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    n = len(c)

    # EMA helper
    def _ema(arr, period):
        out = np.full_like(arr, np.nan)
        if len(arr) < period:
            return out
        out[period - 1] = np.mean(arr[:period])
        m = 2.0 / (period + 1)
        for i in range(period, len(arr)):
            out[i] = arr[i] * m + out[i - 1] * (1 - m)
        return out

    # RSI
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gain, 14)
    avg_loss = _ema(loss, 14)
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    # EMAs
    ema8 = _ema(c, 8)
    ema21 = _ema(c, 21)

    # ATR
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = _ema(tr, 14)

    # ADX (simplified)
    up = h - np.roll(h, 1); up[0] = 0
    dn = np.roll(l, 1) - l; dn[0] = 0
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm_s = _ema(pdm, 14)
    ndm_s = _ema(ndm, 14)
    pdi = np.where(atr > 0, 100.0 * pdm_s / atr, 0.0)
    ndi = np.where(atr > 0, 100.0 * ndm_s / atr, 0.0)
    dx = np.where((pdi + ndi) > 0, 100.0 * np.abs(pdi - ndi) / (pdi + ndi), 0.0)
    adx = _ema(dx, 14)

    # MACD
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal

    # Bollinger Bands
    bb_mid = _ema(c, 20)
    bb_std = np.full(n, np.nan)
    for i in range(19, n):
        bb_std[i] = np.std(c[i - 19:i + 1])
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # Consecutive candles
    consec = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i - 1]:
            consec[i] = max(consec[i - 1] + 1, 1)
        elif c[i] < c[i - 1]:
            consec[i] = min(consec[i - 1] - 1, -1)

    return {
        "rsi": rsi, "ema8": ema8, "ema21": ema21, "adx": adx, "atr": atr,
        "macd_hist": macd_hist, "bb_upper": bb_upper, "bb_lower": bb_lower,
        "close": c, "high": h, "low": l, "consec": consec, "n": n,
        "pdi": pdi, "ndi": ndi,
    }


def _align_ltf_to_h1(h1_times, ltf_df, ltf_ind):
    """For each H1 bar, find the corresponding lower-TF bar index.
    Returns array of LTF indices aligned to H1 bars.
    Uses the last LTF bar that starts before each H1 bar's close time.
    """
    ltf_times = ltf_df["time"].values
    h1_t = h1_times.values if hasattr(h1_times, 'values') else np.array(h1_times)
    aligned = np.zeros(len(h1_t), dtype=np.int64)
    ltf_idx = 0
    for i in range(len(h1_t)):
        # Find last LTF bar <= H1 bar time
        while ltf_idx < len(ltf_times) - 1 and ltf_times[ltf_idx + 1] <= h1_t[i]:
            ltf_idx += 1
        aligned[i] = min(ltf_idx, ltf_ind["n"] - 1)
    return aligned


def _fill_mtf_features(X, j, bar_i, direction, h1_atr,
                       m15_ind, m15_aligned, m5_ind, m5_aligned,
                       h1_close, h1_high, h1_low):
    """Fill MTF feature columns 21-32 in feature matrix X.
    Uses real M15/M5 indicators when available, falls back to H1 approximation.
    """
    # Feature indices: 21=m15_rsi, 22=m15_ema_align, 23=m15_atr_ratio,
    # 24=m15_macd_hist, 25=m15_adx, 26=m5_rsi, 27=m5_ema_align,
    # 28=m5_atr_ratio, 29=m5_momentum, 30=mtf_agreement,
    # 31=m15_bb_position, 32=m5_consec_candles

    a = h1_atr if h1_atr > 0 else 1.0

    # --- M15 features ---
    if m15_ind is not None and m15_aligned is not None:
        mi = int(m15_aligned[bar_i])
        mi = min(mi, m15_ind["n"] - 1)

        # M15 RSI (real)
        m15_rsi = m15_ind["rsi"][mi]
        X[j, 21] = m15_rsi if not np.isnan(m15_rsi) else 50.0

        # M15 EMA alignment with trade direction
        e8 = m15_ind["ema8"][mi]
        e21 = m15_ind["ema21"][mi]
        if not np.isnan(e8) and not np.isnan(e21):
            X[j, 22] = (1.0 if e8 > e21 else -1.0) * direction
        else:
            X[j, 22] = 0.0

        # M15 ATR / H1 ATR
        m15_atr = m15_ind["atr"][mi]
        X[j, 23] = (m15_atr / a) if (not np.isnan(m15_atr) and a > 0) else 1.0

        # M15 MACD histogram normalized by H1 ATR
        m15_mh = m15_ind["macd_hist"][mi]
        X[j, 24] = (m15_mh / a) if (not np.isnan(m15_mh) and a > 0) else 0.0

        # M15 ADX
        m15_adx = m15_ind["adx"][mi]
        X[j, 25] = m15_adx if not np.isnan(m15_adx) else 25.0

        # M15 Bollinger Band position (0=at lower, 1=at upper)
        bbu = m15_ind["bb_upper"][mi]
        bbl = m15_ind["bb_lower"][mi]
        m15_c = m15_ind["close"][mi]
        if not np.isnan(bbu) and not np.isnan(bbl) and (bbu - bbl) > 0:
            X[j, 31] = (m15_c - bbl) / (bbu - bbl)
        else:
            X[j, 31] = 0.5
    else:
        # Fallback: approximate from H1
        if bar_i >= 5:
            m15_c = h1_close[max(0, bar_i - 4):bar_i + 1]
            g = np.maximum(0, np.diff(m15_c))
            l = np.maximum(0, -np.diff(m15_c))
            ag = np.mean(g) if len(g) > 0 else 0
            al = np.mean(l) if len(l) > 0 else 0.001
            X[j, 21] = 100 - 100 / (1 + ag / al) if al > 0 else 50.0
            ema5 = np.mean(h1_close[bar_i - 4:bar_i + 1])
            ema10 = np.mean(h1_close[max(0, bar_i - 9):bar_i + 1])
            X[j, 22] = (1.0 if ema5 > ema10 else -1.0) * direction
            recent_atr = np.mean(h1_high[bar_i - 4:bar_i + 1] - h1_low[bar_i - 4:bar_i + 1])
            X[j, 23] = recent_atr / a if a > 0 else 1.0
        else:
            X[j, 21] = 50.0; X[j, 22] = 0.0; X[j, 23] = 1.0
        X[j, 24] = 0.0; X[j, 25] = 25.0; X[j, 31] = 0.5

    # --- M5 features ---
    if m5_ind is not None and m5_aligned is not None:
        si = int(m5_aligned[bar_i])
        si = min(si, m5_ind["n"] - 1)

        # M5 RSI
        m5_rsi = m5_ind["rsi"][si]
        X[j, 26] = m5_rsi if not np.isnan(m5_rsi) else 50.0

        # M5 EMA alignment
        e8 = m5_ind["ema8"][si]
        e21 = m5_ind["ema21"][si]
        if not np.isnan(e8) and not np.isnan(e21):
            X[j, 27] = (1.0 if e8 > e21 else -1.0) * direction
        else:
            X[j, 27] = 0.0

        # M5 ATR / H1 ATR
        m5_atr = m5_ind["atr"][si]
        X[j, 28] = (m5_atr / a) if (not np.isnan(m5_atr) and a > 0) else 1.0

        # M5 momentum (6-bar = 30 min price change / H1 ATR)
        if si >= 6:
            X[j, 29] = (m5_ind["close"][si] - m5_ind["close"][si - 6]) / a * direction
        else:
            X[j, 29] = 0.0

        # M5 consecutive candles
        X[j, 32] = float(m5_ind["consec"][si])
    else:
        # Fallback
        X[j, 26] = 50.0; X[j, 27] = 0.0; X[j, 28] = 1.0
        X[j, 29] = 0.0; X[j, 32] = 0.0

    # --- Cross-TF agreement ---
    # H1 direction from EMA: already in X (ema_alignment at index 9)
    h1_agree = 1 if X[j, 9] > 0 else (-1 if X[j, 9] < 0 else 0)
    m15_agree = 1 if X[j, 22] > 0 else (-1 if X[j, 22] < 0 else 0)
    m5_agree = 1 if X[j, 27] > 0 else (-1 if X[j, 27] < 0 else 0)
    X[j, 30] = (h1_agree + m15_agree + m5_agree) / 3.0  # -1 to +1


class SignalModel:
    """LightGBM meta-labeling classifier: predicts if a scored signal will be profitable."""

    def __init__(self):
        self.models = {}              # symbol -> lgb.Booster (legacy single model)
        self.ensembles = {}           # symbol -> dict of ensemble data
        self.feature_importance = {}  # symbol -> dict of feature -> importance
        self._train_metrics = {}      # symbol -> metrics dict
        self._feat_stats = {}         # symbol -> {"mean": np.array, "std": np.array}

    def load(self, symbol=None):
        """Load saved model(s) from disk. Prefers ensemble, falls back to legacy single model."""
        symbols = [symbol] if symbol else list(SYMBOLS.keys())
        for sym in symbols:
            # Try ensemble first
            ens_path = MODEL_DIR / f"{sym.replace('.', '_')}_meta_lgb_ensemble.pkl"
            if ens_path.exists():
                try:
                    with open(ens_path, "rb") as f:
                        data = pickle.load(f)
                    self.ensembles[sym] = data["ensemble"]
                    self.models[sym] = data["ensemble"]["model_a"]  # keep compat
                    self.feature_importance[sym] = data.get("importance", {})
                    self._train_metrics[sym] = data.get("metrics", {})
                    self._feat_stats[sym] = data.get("feat_stats", {})
                    log.info("[%s] Ensemble model loaded from %s", sym, ens_path)
                    continue
                except Exception as e:
                    log.error("[%s] Ensemble load failed, trying legacy: %s", sym, e)

            # Legacy single model
            model_path = MODEL_DIR / f"{sym.replace('.', '_')}_meta_lgb.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        data = pickle.load(f)
                    self.models[sym] = data["model"]
                    self.feature_importance[sym] = data.get("importance", {})
                    self._train_metrics[sym] = data.get("metrics", {})
                    # Load feat_stats if present in legacy file
                    if "feat_stats" in data:
                        self._feat_stats[sym] = data["feat_stats"]
                    log.info("[%s] Legacy meta-model loaded from %s", sym, model_path)
                except Exception as e:
                    log.error("[%s] Model load failed: %s", sym, e)

    def save(self, symbol):
        """Save model to disk. Saves both legacy single model and ensemble."""
        if symbol not in self.models:
            return

        feat_stats = self._feat_stats.get(symbol, {})

        # Legacy single-model file (backward compat)
        model_path = MODEL_DIR / f"{symbol.replace('.', '_')}_meta_lgb.pkl"
        try:
            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": self.models[symbol],
                    "importance": self.feature_importance.get(symbol, {}),
                    "metrics": self._train_metrics.get(symbol, {}),
                    "feat_stats": feat_stats,
                    "timestamp": time.time(),
                }, f)
            log.info("[%s] Legacy meta-model saved to %s", symbol, model_path)
        except Exception as e:
            log.error("[%s] Legacy model save failed: %s", symbol, e)

        # Ensemble file
        if symbol in self.ensembles:
            ens_path = MODEL_DIR / f"{symbol.replace('.', '_')}_meta_lgb_ensemble.pkl"
            try:
                with open(ens_path, "wb") as f:
                    pickle.dump({
                        "ensemble": self.ensembles[symbol],
                        "importance": self.feature_importance.get(symbol, {}),
                        "metrics": self._train_metrics.get(symbol, {}),
                        "feat_stats": feat_stats,
                        "timestamp": time.time(),
                    }, f)
                log.info("[%s] Ensemble model saved to %s", symbol, ens_path)
            except Exception as e:
                log.error("[%s] Ensemble save failed: %s", symbol, e)

    def train(self, symbol, mt5_conn, feature_engine):
        """
        Train meta-labeling model for a symbol.

        Steps:
        1. Fetch H1 candles
        2. Run _compute_indicators and _score on each bar
        3. Generate scored signals (score >= MIN_SCORE)
        4. Simulate forward outcomes (ATR-based SL + trailing)
        5. Build meta-features
        6. Train LightGBM walk-forward
        7. Return metrics dict
        """
        log.info("[%s] Starting meta-labeling training...", symbol)

        # ═══ 1. FETCH H1 + M15 + M5 CANDLES ═══
        MT5_H1 = 16385
        MT5_M15 = 15
        MT5_M5 = 5
        rates = mt5_conn.copy_rates_from_pos(symbol, MT5_H1, 0, 50000)

        if rates is None or len(rates) < 1000:
            log.warning("[%s] Not enough H1 data (%s candles)", symbol,
                        len(rates) if rates is not None else 0)
            return {"status": "error", "reason": "insufficient_data",
                    "candles": len(rates) if rates is not None else 0}

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        log.info("[%s] Got %d H1 candles for meta-labeling", symbol, len(df))

        # Fetch M15 and M5 candles for real MTF features
        m15_df, m5_df = None, None
        m15_ind, m5_ind = None, None
        m15_aligned, m5_aligned = None, None
        try:
            m15_rates = mt5_conn.copy_rates_from_pos(symbol, MT5_M15, 0, 50000)
            if m15_rates is not None and len(m15_rates) >= 500:
                m15_df = pd.DataFrame(m15_rates)
                m15_df["time"] = pd.to_datetime(m15_df["time"], unit="s", utc=True)
                m15_ind = _compute_tf_indicators(m15_df)
                m15_aligned = _align_ltf_to_h1(df["time"], m15_df, m15_ind)
                log.info("[%s] Got %d M15 candles for MTF features", symbol, len(m15_df))
        except Exception as e:
            log.warning("[%s] M15 fetch failed: %s — using H1 approximation", symbol, e)

        try:
            m5_rates = mt5_conn.copy_rates_from_pos(symbol, MT5_M5, 0, 50000)
            if m5_rates is not None and len(m5_rates) >= 500:
                m5_df = pd.DataFrame(m5_rates)
                m5_df["time"] = pd.to_datetime(m5_df["time"], unit="s", utc=True)
                m5_ind = _compute_tf_indicators(m5_df)
                m5_aligned = _align_ltf_to_h1(df["time"], m5_df, m5_ind)
                log.info("[%s] Got %d M5 candles for MTF features", symbol, len(m5_df))
        except Exception as e:
            log.warning("[%s] M5 fetch failed: %s — using H1 approximation", symbol, e)

        # ═══ 2. COMPUTE INDICATORS ═══
        icfg = dict(IND_DEFAULTS)
        icfg.update(IND_OVERRIDES.get(symbol, {}))
        ind = _compute_indicators(df, icfg)
        n = ind["n"]

        # ═══ 3. SCORE EVERY BAR, COLLECT SIGNALS ═══
        signals = []  # list of (bar_index, direction, long_score, short_score)
        # Start from bar 100 to ensure indicators are warmed up
        start_bar = max(100, 21)

        for i in range(start_bar, n - 1):  # -1 because we score completed bars
            if np.isnan(ind["at"][i]) or ind["at"][i] == 0:
                continue

            long_score, short_score = _score(ind, i)

            buy = long_score >= MIN_SCORE
            sell = short_score >= MIN_SCORE

            if not buy and not sell:
                continue

            if buy and (not sell or long_score >= short_score):
                direction = 1  # long
                chosen_score = long_score
            else:
                direction = -1  # short
                chosen_score = short_score

            signals.append((i, direction, long_score, short_score, chosen_score))

        if len(signals) < 50:
            log.warning("[%s] Only %d scored signals — need at least 50", symbol, len(signals))
            return {"status": "error", "reason": "insufficient_signals",
                    "n_signals": len(signals)}

        log.info("[%s] Found %d scored signals (%.1f%% of bars)",
                 symbol, len(signals), len(signals) / (n - start_bar) * 100)

        # ═══ 4. SIMULATE FORWARD OUTCOMES ═══
        close = ind["c"]
        high = ind["h"]
        low = ind["l"]
        atr = ind["at"]

        labels = []  # 1 = profitable, 0 = loss
        for sig_idx, (bar_i, direction, ls, ss, cs) in enumerate(signals):
            outcome = self._simulate_trade(
                bar_i, direction, close, high, low, atr, n, symbol=symbol
            )
            labels.append(outcome)

        labels = np.array(labels, dtype=np.float64)
        win_rate = float(np.mean(labels))
        log.info("[%s] Simulated outcomes: %.1f%% win rate (%d wins / %d total)",
                 symbol, win_rate * 100, int(np.sum(labels)), len(labels))

        # ═══ 5. BUILD META-FEATURES ═══
        X = np.zeros((len(signals), NUM_META_FEATURES), dtype=np.float64)

        # Pre-compute ATR percentile array (rolling 200-bar)
        atr_pctile = np.full(n, 0.5, dtype=np.float64)
        for i in range(200, n):
            window = atr[max(0, i - 200):i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                atr_pctile[i] = float(np.searchsorted(np.sort(valid), atr[i])) / len(valid)

        # Pre-compute vol percentile (rolling 200-bar of 14-period close vol)
        vol_series = pd.Series(close).pct_change().rolling(14).std().fillna(0).values
        vol_pctile = np.full(n, 0.5, dtype=np.float64)
        for i in range(200, n):
            window = vol_series[max(0, i - 200):i + 1]
            if len(window) > 0 and vol_series[i] > 0:
                vol_pctile[i] = float(np.searchsorted(np.sort(window), vol_series[i])) / len(window)

        # Spread proxy from candle data (high - low - |open - close|)
        spread_proxy = np.maximum(0.0, (high - low) - np.abs(ind["o"] - close))

        # Track win streak for training data
        win_streak = 0

        for j, (bar_i, direction, ls, ss, cs) in enumerate(signals):
            a = atr[bar_i] if not np.isnan(atr[bar_i]) and atr[bar_i] > 0 else 1.0
            mid = close[bar_i]

            # Score sub-components
            X[j, 0] = ls   # long_score
            X[j, 1] = ss   # short_score
            X[j, 2] = cs   # chosen_score
            X[j, 3] = float(direction)

            # Indicator context
            X[j, 4] = ind["adx"][bar_i] if not np.isnan(ind["adx"][bar_i]) else 25.0
            X[j, 5] = ind["bbw"][bar_i] if not np.isnan(ind["bbw"][bar_i]) else 2.0
            X[j, 6] = atr_pctile[bar_i]

            rsi_val = ind["rs"][bar_i]
            X[j, 7] = rsi_val if not np.isnan(rsi_val) else 50.0

            st_val = ind["stl"][bar_i]
            if not np.isnan(st_val) and a > 0:
                X[j, 8] = (mid - st_val) / a * direction  # positive = ST confirms direction
            else:
                X[j, 8] = 0.0

            # EMA alignment: how many EMAs support the direction
            es, el, et = ind["es"][bar_i], ind["el"][bar_i], ind["et"][bar_i]
            if direction == 1:
                align = (1 if mid > es else -1) + (1 if es > el else -1) + (1 if el > et else -1)
            else:
                align = (1 if mid < es else -1) + (1 if es < el else -1) + (1 if el < et else -1)
            X[j, 9] = float(align) / 3.0

            # MACD histogram normalized by ATR
            X[j, 10] = ind["mh"][bar_i] / a if a > 0 else 0.0

            # Regime
            X[j, 11] = vol_pctile[bar_i]
            X[j, 12] = float(ind["st"][bar_i]) * direction  # positive = structure agrees

            # Time features (cyclical encoding)
            dt = df["time"].iloc[bar_i]
            hour = dt.hour
            dow = dt.dayofweek
            X[j, 13] = np.sin(2 * np.pi * hour / 24.0)  # hour_sin
            X[j, 14] = np.cos(2 * np.pi * hour / 24.0)  # hour_cos
            X[j, 15] = np.sin(2 * np.pi * dow / 5.0)    # dow_sin
            X[j, 16] = np.cos(2 * np.pi * dow / 5.0)    # dow_cos

            # Spread relative to ATR
            X[j, 17] = spread_proxy[bar_i] / a if a > 0 else 0.0

            # Recent win streak (from training labels so far)
            X[j, 18] = float(win_streak)
            if j > 0:
                if labels[j - 1] == 1:
                    win_streak = max(win_streak + 1, 1)
                else:
                    win_streak = min(win_streak - 1, -1)

            # Score quality
            opposite_score = ss if direction == 1 else ls
            X[j, 19] = cs - opposite_score  # score_margin
            X[j, 20] = cs - MIN_SCORE       # score_vs_threshold

            # ── Real Multi-timeframe features (M15 + M5) ──
            _fill_mtf_features(X, j, bar_i, direction, a,
                               m15_ind, m15_aligned, m5_ind, m5_aligned,
                               close, high, low)

            # ── Momentum persistence (indices 33-36) ──
            if bar_i >= 5:
                c_bi = close[bar_i]; c_1 = close[bar_i-1]; c_3 = close[bar_i-3]; c_5 = close[bar_i-5]
                X[j, 33] = (c_bi - c_1) / a if (a > 0 and np.isfinite(c_bi) and np.isfinite(c_1)) else 0.0
                X[j, 34] = (c_bi - c_3) / a if (a > 0 and np.isfinite(c_bi) and np.isfinite(c_3)) else 0.0
                X[j, 35] = (c_bi - c_5) / a if (a > 0 and np.isfinite(c_bi) and np.isfinite(c_5)) else 0.0
            else:
                X[j, 33] = X[j, 34] = X[j, 35] = 0.0

            # Consecutive candles
            X[j, 36] = float(ind["consec"][bar_i]) if np.isfinite(ind["consec"][bar_i]) else 0.0

            # ── Cross-asset / macro (indices 37-38) ──
            if bar_i >= 5 and np.isfinite(atr[bar_i]) and np.isfinite(atr[bar_i-5]) and atr[bar_i-5] > 0:
                X[j, 37] = atr[bar_i] / atr[bar_i-5]
            else:
                X[j, 37] = 1.0

            # BB squeeze
            bbw = ind["bbw"][bar_i] if not np.isnan(ind["bbw"][bar_i]) else 2.0
            if bar_i >= 200:
                bbw_window = ind["bbw"][bar_i-200:bar_i+1]
                bbw_valid = bbw_window[~np.isnan(bbw_window)]
                X[j, 38] = 1.0 if len(bbw_valid) > 0 and bbw <= np.percentile(bbw_valid, 20) else 0.0
            else:
                X[j, 38] = 0.0

            # ── Reversal detection (indices 39-41) ──
            if bar_i >= 10:
                price_higher = close[bar_i] > np.max(close[bar_i-10:bar_i])
                rsi_lower = ind["rs"][bar_i] < np.nanmax(ind["rs"][bar_i-10:bar_i]) if not np.isnan(ind["rs"][bar_i]) else False
                price_lower = close[bar_i] < np.min(close[bar_i-10:bar_i])
                rsi_higher = ind["rs"][bar_i] > np.nanmin(ind["rs"][bar_i-10:bar_i]) if not np.isnan(ind["rs"][bar_i]) else False
                if direction == 1:
                    X[j, 39] = -1.0 if (price_higher and rsi_lower) else (1.0 if (price_lower and rsi_higher) else 0.0)
                else:
                    X[j, 39] = 1.0 if (price_lower and rsi_higher) else (-1.0 if (price_higher and rsi_lower) else 0.0)
            else:
                X[j, 39] = 0.0

            # Distance from 20-bar high/low
            if bar_i >= 20 and a > 0:
                high_20 = np.nanmax(high[bar_i-20:bar_i+1])
                low_20 = np.nanmin(low[bar_i-20:bar_i+1])
                if np.isfinite(high_20) and np.isfinite(low_20) and np.isfinite(close[bar_i]):
                    X[j, 40] = (high_20 - close[bar_i]) / a
                    X[j, 41] = (close[bar_i] - low_20) / a
                else:
                    X[j, 40] = X[j, 41] = 0.0
            else:
                X[j, 40] = X[j, 41] = 0.0

        y = labels

        # ═══ 6. WALK-FORWARD TRAINING ═══
        n_samples = len(X)
        train_end = int(n_samples * 0.70)
        val_end = int(n_samples * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        if len(X_train) < 30 or len(X_val) < 10 or len(X_test) < 10:
            log.warning("[%s] Not enough samples per split (train=%d val=%d test=%d)",
                        symbol, len(X_train), len(X_val), len(X_test))
            return {"status": "error", "reason": "insufficient_split_sizes",
                    "train": len(X_train), "val": len(X_val), "test": len(X_test)}

        # ── Feature drift stats: save training distribution ──
        feat_mean = np.nanmean(X_train, axis=0)
        feat_std = np.nanstd(X_train, axis=0)
        # Avoid division by zero for constant features
        feat_std[feat_std < 1e-10] = 1.0
        self._feat_stats[symbol] = {"mean": feat_mean, "std": feat_std}

        # Handle class balance
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale = n_neg / n_pos if n_pos > 0 else 1.0

        from sklearn.metrics import roc_auc_score

        # ── Model A: tuned LightGBM (existing) ──
        sym_hp = TUNED_LGB_PARAMS.get(symbol, DEFAULT_LGB_PARAMS)
        params_a = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "boosting_type": "gbdt",
            "num_leaves": sym_hp["num_leaves"],
            "learning_rate": sym_hp["learning_rate"],
            "max_depth": sym_hp["max_depth"],
            "feature_fraction": sym_hp["feature_fraction"],
            "bagging_fraction": sym_hp["bagging_fraction"],
            "bagging_freq": 5,
            "scale_pos_weight": scale,
            "min_child_samples": max(5, int(len(X_train) * 0.01)),
            "lambda_l1": sym_hp["lambda_l1"],
            "lambda_l2": sym_hp["lambda_l2"],
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=META_FEATURE_NAMES)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=META_FEATURE_NAMES, reference=dtrain)

        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model_a = lgb.train(
            params_a, dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        val_preds_a = model_a.predict(X_val)
        try:
            auc_a = float(roc_auc_score(y_val, val_preds_a))
        except ValueError:
            auc_a = 0.5
        log.info("[%s] Model A (tuned LGB): val AUC=%.4f trees=%d", symbol, auc_a, model_a.num_trees())

        # ── Model B: high-regularization LightGBM ──
        hp_b = ENSEMBLE_B_LGB_PARAMS
        params_b = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "boosting_type": "gbdt",
            "num_leaves": hp_b["num_leaves"],
            "learning_rate": hp_b["learning_rate"],
            "max_depth": hp_b["max_depth"],
            "feature_fraction": hp_b["feature_fraction"],
            "bagging_fraction": hp_b["bagging_fraction"],
            "bagging_freq": 5,
            "scale_pos_weight": scale,
            "min_child_samples": max(5, int(len(X_train) * 0.02)),
            "lambda_l1": hp_b["lambda_l1"],
            "lambda_l2": hp_b["lambda_l2"],
            "verbose": -1,
            "n_jobs": -1,
            "seed": 123,
        }

        dtrain_b = lgb.Dataset(X_train, label=y_train, feature_name=META_FEATURE_NAMES)
        dval_b = lgb.Dataset(X_val, label=y_val, feature_name=META_FEATURE_NAMES, reference=dtrain_b)

        model_b = lgb.train(
            params_b, dtrain_b,
            num_boost_round=1500,
            valid_sets=[dval_b],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
        )

        val_preds_b = model_b.predict(X_val)
        try:
            auc_b = float(roc_auc_score(y_val, val_preds_b))
        except ValueError:
            auc_b = 0.5
        log.info("[%s] Model B (high-reg LGB): val AUC=%.4f trees=%d", symbol, auc_b, model_b.num_trees())

        # ── Model C: ExtraTreesClassifier (algorithm diversity) ──
        et_params = ENSEMBLE_C_ET_PARAMS.copy()
        et_params["class_weight"] = {0: 1.0, 1: scale}
        model_c = ExtraTreesClassifier(**et_params)
        model_c.fit(X_train, y_train.astype(int))

        val_preds_c = model_c.predict_proba(X_val)[:, 1]
        try:
            auc_c = float(roc_auc_score(y_val, val_preds_c))
        except ValueError:
            auc_c = 0.5
        log.info("[%s] Model C (ExtraTrees): val AUC=%.4f", symbol, auc_c)

        # ── AUC-weighted ensemble ──
        # Weights proportional to (auc - 0.5) so random models get zero weight
        raw_w = np.array([max(0, auc_a - 0.5), max(0, auc_b - 0.5), max(0, auc_c - 0.5)])
        if raw_w.sum() < 1e-10:
            raw_w = np.array([1.0, 1.0, 1.0])
        weights = raw_w / raw_w.sum()
        log.info("[%s] Ensemble weights: A=%.3f B=%.3f C=%.3f", symbol, weights[0], weights[1], weights[2])

        # Store ensemble
        self.ensembles[symbol] = {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
            "weights": weights,
            "auc_a": auc_a,
            "auc_b": auc_b,
            "auc_c": auc_c,
        }

        # ═══ 7. EVALUATE (ensemble) ═══
        # Validation — ensemble
        val_preds = weights[0] * val_preds_a + weights[1] * val_preds_b + weights[2] * val_preds_c
        val_class = (val_preds > 0.5).astype(int)
        try:
            val_auc = float(roc_auc_score(y_val, val_preds))
        except ValueError:
            val_auc = 0.5
        val_acc = float(np.mean(val_class == y_val))

        # Test — ensemble
        test_preds_a = model_a.predict(X_test)
        test_preds_b = model_b.predict(X_test)
        test_preds_c = model_c.predict_proba(X_test)[:, 1]
        test_preds = weights[0] * test_preds_a + weights[1] * test_preds_b + weights[2] * test_preds_c
        test_class = (test_preds > 0.5).astype(int)
        try:
            test_auc = float(roc_auc_score(y_test, test_preds))
        except ValueError:
            test_auc = 0.5
        test_acc = float(np.mean(test_class == y_test))

        # Precision at high confidence (the metric that matters for meta-labeling)
        high_conf_mask = test_preds > CONFIDENCE_THRESHOLD
        n_high_conf = int(np.sum(high_conf_mask))
        if n_high_conf > 0:
            precision_at_conf = float(np.mean(y_test[high_conf_mask] == 1))
            recall_at_conf = float(n_high_conf / len(y_test))
        else:
            precision_at_conf = 0.0
            recall_at_conf = 0.0

        # Profit factor simulation on test set
        test_wins = float(np.sum((test_class == 1) & (y_test == 1)))
        test_losses = float(np.sum((test_class == 1) & (y_test == 0)))
        pf_filtered = test_wins / test_losses if test_losses > 0 else float("inf")

        log.info("[%s] Ensemble meta-label results:", symbol)
        log.info("  Val:  AUC=%.3f (A=%.3f B=%.3f C=%.3f)  Acc=%.3f",
                 val_auc, auc_a, auc_b, auc_c, val_acc)
        log.info("  Test: AUC=%.3f  Acc=%.3f", test_auc, test_acc)
        log.info("  Precision@%.0f%%=%.3f  (pass rate=%.1f%%)",
                 CONFIDENCE_THRESHOLD * 100, precision_at_conf, recall_at_conf * 100)
        log.info("  Filtered PF=%.2f", pf_filtered)
        log.info("  Base win rate=%.1f%%  Signals=%d", win_rate * 100, n_samples)

        self.models[symbol] = model_a  # primary model for backward compat

        # Feature importance (from Model A — the tuned one)
        importance = model_a.feature_importance(importance_type="gain")
        imp_dict = {}
        for i, name in enumerate(META_FEATURE_NAMES):
            if i < len(importance):
                imp_dict[name] = float(importance[i])
        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
        self.feature_importance[symbol] = imp_dict

        metrics = {
            "status": "ok",
            "val_auc": val_auc,
            "val_auc_a": auc_a,
            "val_auc_b": auc_b,
            "val_auc_c": auc_c,
            "val_accuracy": val_acc,
            "test_auc": test_auc,
            "test_accuracy": test_acc,
            "precision_at_conf": precision_at_conf,
            "recall_at_conf": recall_at_conf,
            "filtered_pf": pf_filtered,
            "base_win_rate": win_rate,
            "n_signals": n_samples,
            "n_candles": n,
            "n_trees": model_a.num_trees(),
            "ensemble_weights": weights.tolist(),
            "timestamp": time.time(),
        }
        self._train_metrics[symbol] = metrics

        self.save(symbol)
        return metrics

    def _simulate_trade(self, bar_i, direction, close, high, low, atr, n,
                        max_hold=30, symbol=None):
        """
        Simulate a single trade outcome from bar_i forward.
        Uses ATR-based SL and trailing stop logic.

        Returns: 1.0 if profitable, 0.0 if loss.
        """
        a = atr[bar_i] if not np.isnan(atr[bar_i]) and atr[bar_i] > 0 else 1.0
        entry = close[bar_i]

        from config import SYMBOL_ATR_SL_OVERRIDE
        sym_sl = SYMBOL_ATR_SL_OVERRIDE.get(symbol, ATR_SL_MULTIPLIER)
        sl_dist = max(a * sym_sl, a * 1.0)  # per-symbol SL, floor 1.0x ATR
        tp_dist = sl_dist * 2.0              # 2R TP target

        if direction == 1:  # long
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:  # short
            sl = entry + sl_dist
            tp = entry - tp_dist

        best_price = entry
        current_sl = sl

        for k in range(1, min(max_hold, n - bar_i)):
            bar_k = bar_i + k
            h_k = high[bar_k]
            l_k = low[bar_k]
            c_k = close[bar_k]

            if direction == 1:  # long
                # Check SL hit (low touches SL)
                if l_k <= current_sl:
                    pnl = current_sl - entry
                    return 1.0 if pnl > 0 else 0.0

                # Check TP hit
                if h_k >= tp:
                    return 1.0

                # Update trailing stop
                if c_k > best_price:
                    best_price = c_k
                    profit_r = (best_price - entry) / sl_dist
                    # Apply trailing from TRAIL_STEPS
                    for threshold_r, action, param in TRAIL_STEPS:
                        if profit_r >= threshold_r:
                            if action == "trail":
                                new_sl = best_price - param * a
                            elif action == "lock":
                                new_sl = entry + param * sl_dist
                            elif action == "be":
                                new_sl = entry
                            else:
                                continue
                            if new_sl > current_sl:
                                current_sl = new_sl
                            break

            else:  # short
                # Check SL hit (high touches SL)
                if h_k >= current_sl:
                    pnl = entry - current_sl
                    return 1.0 if pnl > 0 else 0.0

                # Check TP hit
                if l_k <= tp:
                    return 1.0

                # Update trailing stop
                if c_k < best_price:
                    best_price = c_k
                    profit_r = (entry - best_price) / sl_dist
                    for threshold_r, action, param in TRAIL_STEPS:
                        if profit_r >= threshold_r:
                            if action == "trail":
                                new_sl = best_price + param * a
                            elif action == "lock":
                                new_sl = entry - param * sl_dist
                            elif action == "be":
                                new_sl = entry
                            else:
                                continue
                            if new_sl < current_sl:
                                current_sl = new_sl
                            break

        # Max hold reached — mark to market
        final_price = close[min(bar_i + max_hold, n - 1)]
        if direction == 1:
            return 1.0 if final_price > entry else 0.0
        else:
            return 1.0 if final_price < entry else 0.0

    def _check_feature_drift(self, symbol, X):
        """
        Check if live features have drifted from training distribution.
        Returns (drift_score, drifting_features) where drift_score is fraction
        of features beyond DRIFT_STD_MULT standard deviations.
        """
        if symbol not in self._feat_stats:
            return 0.0, []

        stats = self._feat_stats[symbol]
        mean = stats["mean"]
        std = stats["std"]

        if len(mean) != X.shape[1]:
            return 0.0, []

        deviations = np.abs(X[0] - mean) / std
        drifting_mask = deviations > DRIFT_STD_MULT
        drift_score = float(np.sum(drifting_mask)) / len(mean)

        drifting_names = []
        if drift_score > 0:
            for i, is_drift in enumerate(drifting_mask):
                if is_drift and i < len(META_FEATURE_NAMES):
                    drifting_names.append(META_FEATURE_NAMES[i])

        return drift_score, drifting_names

    def predict(self, symbol, score_components: dict) -> dict:
        """
        Predict whether a scored signal will be profitable.
        Uses ensemble (3 models, AUC-weighted) if available, falls back to single model.
        Applies feature drift detection to reduce confidence when out-of-distribution.

        Args:
            symbol: trading symbol
            score_components: dict with keys matching META_FEATURE_NAMES

        Returns: {
            "confidence": float (0-1),
            "take_trade": bool,
            "raw_prob": float,
            "drift_score": float (0-1, fraction of drifting features),
            "drift_penalty": bool,
        }
        """
        result = {
            "confidence": 0.0,
            "take_trade": False,
            "raw_prob": 0.5,
            "drift_score": 0.0,
            "drift_penalty": False,
        }

        if symbol not in self.models:
            return result

        try:
            X = np.zeros((1, NUM_META_FEATURES), dtype=np.float64)
            for i, name in enumerate(META_FEATURE_NAMES):
                X[0, i] = float(score_components.get(name, 0.0))

            # ── Ensemble prediction ──
            if symbol in self.ensembles:
                ens = self.ensembles[symbol]
                w = ens["weights"]
                pred_a = float(ens["model_a"].predict(X)[0])
                pred_b = float(ens["model_b"].predict(X)[0])
                pred_c = float(ens["model_c"].predict_proba(X)[0, 1])
                prob = w[0] * pred_a + w[1] * pred_b + w[2] * pred_c
            else:
                # Legacy single model
                prob = float(self.models[symbol].predict(X)[0])

            # ── Feature drift detection ──
            drift_score, drifting_names = self._check_feature_drift(symbol, X)
            drift_penalty = False
            if drift_score > DRIFT_THRESHOLD:
                log.warning("[%s] Feature drift %.0f%% (%d/%d features) — reducing confidence. Drifting: %s",
                            symbol, drift_score * 100,
                            int(drift_score * NUM_META_FEATURES), NUM_META_FEATURES,
                            ", ".join(drifting_names[:5]))
                prob *= DRIFT_PENALTY
                drift_penalty = True

            result = {
                "confidence": prob,
                "take_trade": prob >= CONFIDENCE_THRESHOLD,
                "raw_prob": prob,
                "drift_score": drift_score,
                "drift_penalty": drift_penalty,
            }

        except Exception as e:
            log.warning("[%s] Meta-prediction error: %s", symbol, e)

        return result

    def build_predict_features(self, symbol, long_score, short_score, direction,
                               ind, bar_i, df, recent_win_streak=0,
                               m15_df=None, m5_df=None):
        """
        Build the meta-feature dict for predict() from live scoring data.

        Args:
            symbol: trading symbol
            long_score, short_score: from _score()
            direction: +1 or -1
            ind: dict from _compute_indicators()
            bar_i: index in ind arrays
            df: DataFrame with 'time' column
            recent_win_streak: from trade history

        Returns: dict ready for predict()
        """
        chosen_score = long_score if direction == 1 else short_score
        opposite_score = short_score if direction == 1 else long_score
        a = ind["at"][bar_i] if not np.isnan(ind["at"][bar_i]) and ind["at"][bar_i] > 0 else 1.0
        mid = ind["c"][bar_i]

        # EMA alignment
        es, el, et = ind["es"][bar_i], ind["el"][bar_i], ind["et"][bar_i]
        if direction == 1:
            align = (1 if mid > es else -1) + (1 if es > el else -1) + (1 if el > et else -1)
        else:
            align = (1 if mid < es else -1) + (1 if es < el else -1) + (1 if el < et else -1)

        # SuperTrend distance (signed by direction)
        st_val = ind["stl"][bar_i]
        if not np.isnan(st_val) and a > 0:
            st_dist = (mid - st_val) / a * direction
        else:
            st_dist = 0.0

        # ATR percentile (approximate from available data)
        at_arr = ind["at"]
        n = ind["n"]
        window = at_arr[max(0, bar_i - 200):bar_i + 1]
        valid = window[~np.isnan(window)]
        atr_pctile = float(np.searchsorted(np.sort(valid), a)) / len(valid) if len(valid) > 0 else 0.5

        # Vol percentile
        c = ind["c"]
        vol_ser = pd.Series(c[:bar_i + 1]).pct_change().rolling(14).std().fillna(0).values
        if len(vol_ser) > 200:
            vol_win = vol_ser[max(0, len(vol_ser) - 200):]
            cur_vol = vol_ser[-1]
            vol_pctile = float(np.searchsorted(np.sort(vol_win), cur_vol)) / len(vol_win) if cur_vol > 0 else 0.5
        else:
            vol_pctile = 0.5

        # Spread proxy from candle
        spread_proxy = max(0.0, (ind["h"][bar_i] - ind["l"][bar_i]) - abs(ind["o"][bar_i] - mid))

        # Time features
        dt = df["time"].iloc[bar_i]
        hour = dt.hour
        dow = dt.dayofweek

        features = {
            "long_score": long_score,
            "short_score": short_score,
            "chosen_score": chosen_score,
            "direction": float(direction),
            "adx": ind["adx"][bar_i] if not np.isnan(ind["adx"][bar_i]) else 25.0,
            "bb_width": ind["bbw"][bar_i] if not np.isnan(ind["bbw"][bar_i]) else 2.0,
            "atr_percentile": atr_pctile,
            "rsi": ind["rs"][bar_i] if not np.isnan(ind["rs"][bar_i]) else 50.0,
            "supertrend_dist": st_dist,
            "ema_alignment": float(align) / 3.0,
            "macd_hist_norm": ind["mh"][bar_i] / a if a > 0 else 0.0,
            "vol_percentile": vol_pctile,
            "trend_structure": float(ind["st"][bar_i]) * direction,
            "hour_of_day_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_of_day_cos": np.cos(2 * np.pi * hour / 24.0),
            "day_of_week_sin": np.sin(2 * np.pi * dow / 5.0),
            "day_of_week_cos": np.cos(2 * np.pi * dow / 5.0),
            "spread_atr_ratio": spread_proxy / a if a > 0 else 0.0,
            "recent_win_streak": float(recent_win_streak),
            "score_margin": chosen_score - opposite_score,
            "score_vs_threshold": chosen_score - MIN_SCORE,
        }

        # ── Real Multi-timeframe features (M15 + M5) ──
        close_arr = ind["c"]
        high_arr = ind["h"]
        low_arr = ind["l"]
        atr_arr = ind["at"]

        # Compute M15/M5 indicators from live candle data
        _m15_ind, _m5_ind = None, None
        if m15_df is not None and len(m15_df) >= 30:
            try:
                _m15_ind = _compute_tf_indicators(m15_df)
            except Exception:
                pass
        if m5_df is not None and len(m5_df) >= 30:
            try:
                _m5_ind = _compute_tf_indicators(m5_df)
            except Exception:
                pass

        # M15 features
        if _m15_ind is not None:
            mi = _m15_ind["n"] - 2  # last completed bar
            mi = max(0, mi)
            m15_rsi = _m15_ind["rsi"][mi]
            features["m15_rsi"] = m15_rsi if not np.isnan(m15_rsi) else 50.0
            e8 = _m15_ind["ema8"][mi]; e21 = _m15_ind["ema21"][mi]
            features["m15_ema_align"] = ((1.0 if e8 > e21 else -1.0) * direction) if (not np.isnan(e8) and not np.isnan(e21)) else 0.0
            m15_atr = _m15_ind["atr"][mi]
            features["m15_atr_ratio"] = (m15_atr / a) if (not np.isnan(m15_atr) and a > 0) else 1.0
            m15_mh = _m15_ind["macd_hist"][mi]
            features["m15_macd_hist"] = (m15_mh / a) if (not np.isnan(m15_mh) and a > 0) else 0.0
            m15_adx = _m15_ind["adx"][mi]
            features["m15_adx"] = m15_adx if not np.isnan(m15_adx) else 25.0
            bbu = _m15_ind["bb_upper"][mi]; bbl = _m15_ind["bb_lower"][mi]
            m15_c = _m15_ind["close"][mi]
            features["m15_bb_position"] = ((m15_c - bbl) / (bbu - bbl)) if (not np.isnan(bbu) and not np.isnan(bbl) and (bbu - bbl) > 0) else 0.5
        else:
            # Fallback: approximate from H1
            if bar_i >= 5:
                m15_c = close_arr[max(0, bar_i - 4):bar_i + 1]
                g = np.maximum(0, np.diff(m15_c)); l = np.maximum(0, -np.diff(m15_c))
                ag = np.mean(g) if len(g) > 0 else 0; al = np.mean(l) if len(l) > 0 else 0.001
                features["m15_rsi"] = 100 - 100 / (1 + ag / al) if al > 0 else 50.0
                ema5 = np.mean(close_arr[bar_i - 4:bar_i + 1])
                ema10 = np.mean(close_arr[max(0, bar_i - 9):bar_i + 1])
                features["m15_ema_align"] = (1.0 if ema5 > ema10 else -1.0) * direction
                recent_atr = np.mean(high_arr[bar_i - 4:bar_i + 1] - low_arr[bar_i - 4:bar_i + 1])
                features["m15_atr_ratio"] = recent_atr / a if a > 0 else 1.0
            else:
                features["m15_rsi"] = 50.0; features["m15_ema_align"] = 0.0; features["m15_atr_ratio"] = 1.0
            features["m15_macd_hist"] = 0.0; features["m15_adx"] = 25.0; features["m15_bb_position"] = 0.5

        # M5 features
        if _m5_ind is not None:
            si = _m5_ind["n"] - 2
            si = max(0, si)
            m5_rsi = _m5_ind["rsi"][si]
            features["m5_rsi"] = m5_rsi if not np.isnan(m5_rsi) else 50.0
            e8 = _m5_ind["ema8"][si]; e21 = _m5_ind["ema21"][si]
            features["m5_ema_align"] = ((1.0 if e8 > e21 else -1.0) * direction) if (not np.isnan(e8) and not np.isnan(e21)) else 0.0
            m5_atr = _m5_ind["atr"][si]
            features["m5_atr_ratio"] = (m5_atr / a) if (not np.isnan(m5_atr) and a > 0) else 1.0
            if si >= 6:
                features["m5_momentum"] = (_m5_ind["close"][si] - _m5_ind["close"][si - 6]) / a * direction
            else:
                features["m5_momentum"] = 0.0
            features["m5_consec_candles"] = float(_m5_ind["consec"][si])
        else:
            features["m5_rsi"] = 50.0; features["m5_ema_align"] = 0.0
            features["m5_atr_ratio"] = 1.0; features["m5_momentum"] = 0.0
            features["m5_consec_candles"] = 0.0

        # MTF agreement
        h1_agree = 1 if features["ema_alignment"] > 0 else (-1 if features["ema_alignment"] < 0 else 0)
        m15_agree = 1 if features["m15_ema_align"] > 0 else (-1 if features["m15_ema_align"] < 0 else 0)
        m5_agree = 1 if features["m5_ema_align"] > 0 else (-1 if features["m5_ema_align"] < 0 else 0)
        features["mtf_agreement"] = (h1_agree + m15_agree + m5_agree) / 3.0

        # ── Momentum persistence ──
        if bar_i >= 5:
            features["ret_1bar"] = (close_arr[bar_i] - close_arr[bar_i-1]) / a if a > 0 else 0
            features["ret_3bar"] = (close_arr[bar_i] - close_arr[bar_i-3]) / a if a > 0 else 0
            features["ret_5bar"] = (close_arr[bar_i] - close_arr[bar_i-5]) / a if a > 0 else 0
        else:
            features["ret_1bar"] = features["ret_3bar"] = features["ret_5bar"] = 0.0

        features["consec_candles"] = float(ind["consec"][bar_i]) if not np.isnan(ind["consec"][bar_i]) else 0.0

        # ── Cross-asset / macro ──
        if bar_i >= 5 and not np.isnan(atr_arr[bar_i-5]) and atr_arr[bar_i-5] > 0:
            features["atr_change"] = float(atr_arr[bar_i]) / float(atr_arr[bar_i-5])
        else:
            features["atr_change"] = 1.0

        bbw = ind["bbw"][bar_i] if not np.isnan(ind["bbw"][bar_i]) else 2.0
        if bar_i >= 200:
            bbw_win = ind["bbw"][bar_i-200:bar_i+1]
            bbw_valid = bbw_win[~np.isnan(bbw_win)]
            features["bb_squeeze"] = 1.0 if len(bbw_valid) > 0 and bbw <= np.percentile(bbw_valid, 20) else 0.0
        else:
            features["bb_squeeze"] = 0.0

        # ── Reversal detection ──
        if bar_i >= 10:
            price_higher = close_arr[bar_i] > np.max(close_arr[bar_i-10:bar_i])
            rsi_val = ind["rs"][bar_i]
            rsi_lower = rsi_val < np.nanmax(ind["rs"][bar_i-10:bar_i]) if not np.isnan(rsi_val) else False
            price_lower = close_arr[bar_i] < np.min(close_arr[bar_i-10:bar_i])
            rsi_higher = rsi_val > np.nanmin(ind["rs"][bar_i-10:bar_i]) if not np.isnan(rsi_val) else False
            if direction == 1:
                features["rsi_divergence"] = -1.0 if (price_higher and rsi_lower) else (1.0 if (price_lower and rsi_higher) else 0.0)
            else:
                features["rsi_divergence"] = 1.0 if (price_lower and rsi_higher) else (-1.0 if (price_higher and rsi_lower) else 0.0)
        else:
            features["rsi_divergence"] = 0.0

        if bar_i >= 20 and a > 0:
            features["dist_from_high_20"] = (np.max(high_arr[bar_i-20:bar_i+1]) - close_arr[bar_i]) / a
            features["dist_from_low_20"] = (close_arr[bar_i] - np.min(low_arr[bar_i-20:bar_i+1])) / a
        else:
            features["dist_from_high_20"] = features["dist_from_low_20"] = 0.0

        return features

    def train_all(self, mt5_conn, feature_engine):
        """Train meta-models for all symbols."""
        results = {}
        for symbol in SYMBOLS:
            try:
                results[symbol] = self.train(symbol, mt5_conn, feature_engine)
            except Exception as e:
                log.error("[%s] Training failed: %s", symbol, e)
                results[symbol] = {"status": "error", "reason": str(e)}
        return results

    def load_all(self):
        """Load all saved models."""
        for symbol in SYMBOLS:
            self.load(symbol)

    def has_model(self, symbol) -> bool:
        return symbol in self.models

    def get_importance(self, symbol) -> dict:
        return self.feature_importance.get(symbol, {})

    def get_metrics(self, symbol) -> dict:
        return self._train_metrics.get(symbol, {})
