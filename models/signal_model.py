"""
Beast Trader — Meta-Labeling Signal Model.
Instead of predicting direction (coin flip), predicts whether a scored signal
will be profitable. Uses the proven momentum scoring system as the base signal
generator, then trains LightGBM to filter winners from losers.

Walk-forward: train 70%, validate 15%, test 15%.
"""
import os
import time
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

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
    # ── NEW: Multi-timeframe features ──
    "m15_rsi",             # M15 RSI (lower TF momentum)
    "m15_ema_align",       # M15 EMA alignment with direction
    "m15_atr_ratio",       # M15 ATR / H1 ATR (volatility expansion)
    # ── NEW: Momentum persistence ──
    "ret_1bar",            # 1-bar return (immediate momentum)
    "ret_3bar",            # 3-bar return
    "ret_5bar",            # 5-bar return
    "consec_candles",      # consecutive same-direction candles
    # ── NEW: Cross-asset / macro ──
    "atr_change",          # ATR expansion/contraction (atr / atr_5ago)
    "bb_squeeze",          # 1 if BB width < 20th percentile (breakout setup)
    # ── NEW: Reversal detection ──
    "rsi_divergence",      # price making new high but RSI lower (bearish divergence)
    "dist_from_high_20",   # distance from 20-bar high (% of ATR)
    "dist_from_low_20",    # distance from 20-bar low (% of ATR)
]
NUM_META_FEATURES = len(META_FEATURE_NAMES)


class SignalModel:
    """LightGBM meta-labeling classifier: predicts if a scored signal will be profitable."""

    def __init__(self):
        self.models = {}              # symbol -> lgb.Booster
        self.feature_importance = {}  # symbol -> dict of feature -> importance
        self._train_metrics = {}      # symbol -> metrics dict

    def load(self, symbol=None):
        """Load saved model(s) from disk."""
        symbols = [symbol] if symbol else list(SYMBOLS.keys())
        for sym in symbols:
            model_path = MODEL_DIR / f"{sym.replace('.', '_')}_meta_lgb.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        data = pickle.load(f)
                    self.models[sym] = data["model"]
                    self.feature_importance[sym] = data.get("importance", {})
                    self._train_metrics[sym] = data.get("metrics", {})
                    log.info("[%s] Meta-model loaded from %s", sym, model_path)
                except Exception as e:
                    log.error("[%s] Model load failed: %s", sym, e)

    def save(self, symbol):
        """Save model to disk."""
        if symbol not in self.models:
            return
        model_path = MODEL_DIR / f"{symbol.replace('.', '_')}_meta_lgb.pkl"
        try:
            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": self.models[symbol],
                    "importance": self.feature_importance.get(symbol, {}),
                    "metrics": self._train_metrics.get(symbol, {}),
                    "timestamp": time.time(),
                }, f)
            log.info("[%s] Meta-model saved to %s", symbol, model_path)
        except Exception as e:
            log.error("[%s] Model save failed: %s", symbol, e)

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

        # ═══ 1. FETCH H1 CANDLES ═══
        MT5_H1 = 16385
        rates = mt5_conn.copy_rates_from_pos(symbol, MT5_H1, 0, 50000)

        if rates is None or len(rates) < 1000:
            log.warning("[%s] Not enough H1 data (%s candles)", symbol,
                        len(rates) if rates is not None else 0)
            return {"status": "error", "reason": "insufficient_data",
                    "candles": len(rates) if rates is not None else 0}

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        log.info("[%s] Got %d H1 candles for meta-labeling", symbol, len(df))

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
                bar_i, direction, close, high, low, atr, n
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

            # ── NEW: Multi-timeframe (M15 approximated from H1) ──
            # M15 RSI approximated: use shorter RSI lookback on H1
            if bar_i >= 5:
                m15_closes = close[max(0, bar_i-4):bar_i+1]
                m15_gains = np.maximum(0, np.diff(m15_closes))
                m15_losses = np.maximum(0, -np.diff(m15_closes))
                avg_g = np.mean(m15_gains) if len(m15_gains) > 0 else 0
                avg_l = np.mean(m15_losses) if len(m15_losses) > 0 else 0.001
                m15_rsi = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 50
            else:
                m15_rsi = 50.0
            X[j, 21] = m15_rsi

            # M15 EMA alignment (5-bar vs 10-bar EMA direction match)
            if bar_i >= 10:
                ema5 = np.mean(close[bar_i-4:bar_i+1])
                ema10 = np.mean(close[bar_i-9:bar_i+1])
                m15_align = (1.0 if ema5 > ema10 else -1.0) * direction
            else:
                m15_align = 0.0
            X[j, 22] = m15_align

            # M15 ATR ratio (recent vs historical volatility)
            if bar_i >= 5:
                recent_atr = np.mean(high[bar_i-4:bar_i+1] - low[bar_i-4:bar_i+1])
                X[j, 23] = recent_atr / a if a > 0 else 1.0
            else:
                X[j, 23] = 1.0

            # ── NEW: Momentum persistence ──
            if bar_i >= 5:
                X[j, 24] = (close[bar_i] - close[bar_i-1]) / a if a > 0 else 0  # ret_1bar
                X[j, 25] = (close[bar_i] - close[bar_i-3]) / a if a > 0 else 0  # ret_3bar
                X[j, 26] = (close[bar_i] - close[bar_i-5]) / a if a > 0 else 0  # ret_5bar
            else:
                X[j, 24] = X[j, 25] = X[j, 26] = 0.0

            # Consecutive candles
            X[j, 27] = float(ind["consec"][bar_i]) if not np.isnan(ind["consec"][bar_i]) else 0.0

            # ── NEW: Cross-asset / macro ──
            # ATR change (expansion = trending, contraction = ranging)
            if bar_i >= 5 and not np.isnan(atr[bar_i-5]) and atr[bar_i-5] > 0:
                X[j, 28] = atr[bar_i] / atr[bar_i-5]
            else:
                X[j, 28] = 1.0

            # BB squeeze (width below 20th percentile = breakout setup)
            bbw = ind["bbw"][bar_i] if not np.isnan(ind["bbw"][bar_i]) else 2.0
            if bar_i >= 200:
                bbw_window = ind["bbw"][bar_i-200:bar_i+1]
                bbw_valid = bbw_window[~np.isnan(bbw_window)]
                X[j, 29] = 1.0 if len(bbw_valid) > 0 and bbw <= np.percentile(bbw_valid, 20) else 0.0
            else:
                X[j, 29] = 0.0

            # ── NEW: Reversal detection ──
            # RSI divergence: price making new high but RSI lower
            if bar_i >= 10:
                price_higher = close[bar_i] > np.max(close[bar_i-10:bar_i])
                rsi_lower = ind["rs"][bar_i] < np.nanmax(ind["rs"][bar_i-10:bar_i]) if not np.isnan(ind["rs"][bar_i]) else False
                price_lower = close[bar_i] < np.min(close[bar_i-10:bar_i])
                rsi_higher = ind["rs"][bar_i] > np.nanmin(ind["rs"][bar_i-10:bar_i]) if not np.isnan(ind["rs"][bar_i]) else False
                if direction == 1:
                    X[j, 30] = -1.0 if (price_higher and rsi_lower) else (1.0 if (price_lower and rsi_higher) else 0.0)
                else:
                    X[j, 30] = 1.0 if (price_lower and rsi_higher) else (-1.0 if (price_higher and rsi_lower) else 0.0)
            else:
                X[j, 30] = 0.0

            # Distance from 20-bar high/low (mean reversion signal)
            if bar_i >= 20 and a > 0:
                high_20 = np.max(high[bar_i-20:bar_i+1])
                low_20 = np.min(low[bar_i-20:bar_i+1])
                X[j, 31] = (high_20 - close[bar_i]) / a  # dist from high
                X[j, 32] = (close[bar_i] - low_20) / a   # dist from low
            else:
                X[j, 31] = X[j, 32] = 0.0

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

        # Handle class balance
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale = n_neg / n_pos if n_pos > 0 else 1.0

        params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.03,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "scale_pos_weight": scale,
            "min_child_samples": max(5, int(len(X_train) * 0.01)),
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
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

        model = lgb.train(
            params, dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        # ═══ 7. EVALUATE ═══
        from sklearn.metrics import roc_auc_score, precision_score, recall_score

        # Validation metrics
        val_preds = model.predict(X_val)
        val_class = (val_preds > 0.5).astype(int)
        try:
            val_auc = float(roc_auc_score(y_val, val_preds))
        except ValueError:
            val_auc = 0.5
        val_acc = float(np.mean(val_class == y_val))

        # Test metrics (out-of-sample)
        test_preds = model.predict(X_test)
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

        log.info("[%s] Meta-label results:", symbol)
        log.info("  Val:  AUC=%.3f  Acc=%.3f", val_auc, val_acc)
        log.info("  Test: AUC=%.3f  Acc=%.3f", test_auc, test_acc)
        log.info("  Precision@%.0f%%=%.3f  (pass rate=%.1f%%)",
                 CONFIDENCE_THRESHOLD * 100, precision_at_conf, recall_at_conf * 100)
        log.info("  Filtered PF=%.2f  Trees=%d", pf_filtered, model.num_trees())
        log.info("  Base win rate=%.1f%%  Signals=%d", win_rate * 100, n_samples)

        self.models[symbol] = model

        # Feature importance
        importance = model.feature_importance(importance_type="gain")
        imp_dict = {}
        for i, name in enumerate(META_FEATURE_NAMES):
            if i < len(importance):
                imp_dict[name] = float(importance[i])
        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
        self.feature_importance[symbol] = imp_dict

        metrics = {
            "status": "ok",
            "val_auc": val_auc,
            "val_accuracy": val_acc,
            "test_auc": test_auc,
            "test_accuracy": test_acc,
            "precision_at_conf": precision_at_conf,
            "recall_at_conf": recall_at_conf,
            "filtered_pf": pf_filtered,
            "base_win_rate": win_rate,
            "n_signals": n_samples,
            "n_candles": n,
            "n_trees": model.num_trees(),
            "timestamp": time.time(),
        }
        self._train_metrics[symbol] = metrics

        self.save(symbol)
        return metrics

    def _simulate_trade(self, bar_i, direction, close, high, low, atr, n,
                        max_hold=30):
        """
        Simulate a single trade outcome from bar_i forward.
        Uses ATR-based SL and trailing stop logic.

        Returns: 1.0 if profitable, 0.0 if loss.
        """
        a = atr[bar_i] if not np.isnan(atr[bar_i]) and atr[bar_i] > 0 else 1.0
        entry = close[bar_i]

        sl_dist = a * ATR_SL_MULTIPLIER  # 3x ATR SL
        tp_dist = sl_dist * 2.5          # 2.5R TP target (middle regime)

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

    def predict(self, symbol, score_components: dict) -> dict:
        """
        Predict whether a scored signal will be profitable.

        Args:
            symbol: trading symbol
            score_components: dict with keys matching META_FEATURE_NAMES, including:
                - long_score, short_score, chosen_score, direction
                - adx, bb_width, atr_percentile, rsi, supertrend_dist
                - ema_alignment, macd_hist_norm, vol_percentile, trend_structure
                - hour_of_day_sin/cos, day_of_week_sin/cos
                - spread_atr_ratio, recent_win_streak
                - score_margin, score_vs_threshold

        Returns: {
            "confidence": float (0-1),
            "take_trade": bool,
            "raw_prob": float,
        }
        """
        result = {
            "confidence": 0.0,
            "take_trade": False,
            "raw_prob": 0.5,
        }

        if symbol not in self.models:
            return result

        model = self.models[symbol]

        try:
            X = np.zeros((1, NUM_META_FEATURES), dtype=np.float64)
            for i, name in enumerate(META_FEATURE_NAMES):
                X[0, i] = float(score_components.get(name, 0.0))

            prob = float(model.predict(X)[0])

            result = {
                "confidence": prob,
                "take_trade": prob >= CONFIDENCE_THRESHOLD,
                "raw_prob": prob,
            }

        except Exception as e:
            log.warning("[%s] Meta-prediction error: %s", symbol, e)

        return result

    def build_predict_features(self, symbol, long_score, short_score, direction,
                               ind, bar_i, df, recent_win_streak=0):
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

        # ── NEW: Multi-timeframe features ──
        close_arr = ind["c"]
        high_arr = ind["h"]
        low_arr = ind["l"]
        atr_arr = ind["at"]

        # M15 RSI (short lookback on H1)
        if bar_i >= 5:
            m15_c = close_arr[bar_i-4:bar_i+1]
            m15_g = np.maximum(0, np.diff(m15_c))
            m15_l = np.maximum(0, -np.diff(m15_c))
            avg_g = np.mean(m15_g) if len(m15_g) > 0 else 0
            avg_l = np.mean(m15_l) if len(m15_l) > 0 else 0.001
            features["m15_rsi"] = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 50.0
        else:
            features["m15_rsi"] = 50.0

        # M15 EMA alignment
        if bar_i >= 10:
            ema5 = np.mean(close_arr[bar_i-4:bar_i+1])
            ema10 = np.mean(close_arr[bar_i-9:bar_i+1])
            features["m15_ema_align"] = (1.0 if ema5 > ema10 else -1.0) * direction
        else:
            features["m15_ema_align"] = 0.0

        # M15 ATR ratio
        if bar_i >= 5:
            recent_atr = np.mean(high_arr[bar_i-4:bar_i+1] - low_arr[bar_i-4:bar_i+1])
            features["m15_atr_ratio"] = recent_atr / a if a > 0 else 1.0
        else:
            features["m15_atr_ratio"] = 1.0

        # ── NEW: Momentum persistence ──
        if bar_i >= 5:
            features["ret_1bar"] = (close_arr[bar_i] - close_arr[bar_i-1]) / a if a > 0 else 0
            features["ret_3bar"] = (close_arr[bar_i] - close_arr[bar_i-3]) / a if a > 0 else 0
            features["ret_5bar"] = (close_arr[bar_i] - close_arr[bar_i-5]) / a if a > 0 else 0
        else:
            features["ret_1bar"] = features["ret_3bar"] = features["ret_5bar"] = 0.0

        features["consec_candles"] = float(ind["consec"][bar_i]) if not np.isnan(ind["consec"][bar_i]) else 0.0

        # ── NEW: Cross-asset / macro ──
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

        # ── NEW: Reversal detection ──
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
