"""
Beast Trader — LightGBM Signal Model.
Direction prediction (LONG/SHORT/FLAT) using tick-level features.
Walk-forward validation with rolling windows.
"""
import os
import time
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, SYMBOLS, PREDICTION_HORIZON_TF, CONFIDENCE_THRESHOLD
from data.feature_engine import FeatureEngine, FEATURE_NAMES, NUM_FEATURES

log = logging.getLogger("beast.model")


class SignalModel:
    """LightGBM classifier for direction prediction."""

    def __init__(self):
        self.models = {}           # symbol -> lgb.Booster
        self.feature_importance = {}  # symbol -> dict of feature -> importance
        self._train_metrics = {}   # symbol -> {accuracy, auc, etc.}

    def load(self, symbol=None):
        """Load saved model(s) from disk."""
        symbols = [symbol] if symbol else list(SYMBOLS.keys())
        for sym in symbols:
            model_path = MODEL_DIR / f"{sym.replace('.', '_')}_lgb.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        data = pickle.load(f)
                    self.models[sym] = data["model"]
                    self.feature_importance[sym] = data.get("importance", {})
                    self._train_metrics[sym] = data.get("metrics", {})
                    log.info("[%s] Model loaded from %s", sym, model_path)
                except Exception as e:
                    log.error("[%s] Model load failed: %s", sym, e)

    def save(self, symbol):
        """Save model to disk."""
        if symbol not in self.models:
            return
        model_path = MODEL_DIR / f"{symbol.replace('.', '_')}_lgb.pkl"
        try:
            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": self.models[symbol],
                    "importance": self.feature_importance.get(symbol, {}),
                    "metrics": self._train_metrics.get(symbol, {}),
                    "timestamp": time.time(),
                }, f)
            log.info("[%s] Model saved to %s", symbol, model_path)
        except Exception as e:
            log.error("[%s] Model save failed: %s", symbol, e)

    def train(self, symbol, mt5_conn, feature_engine: FeatureEngine):
        """
        Train model for a symbol using historical MT5 data.
        Walk-forward: train on 6 months, test on 1 month, roll forward.
        """
        log.info("[%s] Starting model training...", symbol)

        # Fetch historical M15 candles (max available)
        tf_map = {15: 15}
        mt5_tf = tf_map.get(PREDICTION_HORIZON_TF, PREDICTION_HORIZON_TF)
        rates = mt5_conn.copy_rates_from_pos(symbol, mt5_tf, 0, 50000)

        if rates is None or len(rates) < 1000:
            log.warning("[%s] Not enough historical data (%s candles)", symbol,
                        len(rates) if rates is not None else 0)
            return False

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        log.info("[%s] Got %d M15 candles for training", symbol, len(df))

        # Generate features and labels
        X, y = feature_engine.generate_batch(symbol, df, horizon=1)
        if X is None or len(X) < 500:
            log.warning("[%s] Not enough features generated (%s samples)", symbol,
                        len(X) if X is not None else 0)
            return False

        log.info("[%s] Feature matrix: %d samples x %d features", symbol, X.shape[0], X.shape[1])

        # Walk-forward cross-validation
        # Train on 6 months worth (~6*20*24*4 = ~11520 M15 candles), test on 1 month
        train_size = min(int(len(X) * 0.7), 10000)
        test_size = min(int(len(X) * 0.15), 3000)

        best_model = None
        best_auc = 0.0
        all_accuracies = []

        # Use 3 walk-forward folds
        n_folds = 3
        fold_size = (len(X) - train_size) // n_folds if n_folds > 1 else test_size

        for fold in range(n_folds):
            train_end = train_size + fold * fold_size
            test_end = min(train_end + fold_size, len(X))
            if train_end >= len(X) or test_end <= train_end:
                break

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]

            # Handle class balance
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            scale = n_neg / n_pos if n_pos > 0 else 1.0

            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "scale_pos_weight": scale,
                "verbose": -1,
                "n_jobs": -1,
                "seed": 42,
            }

            dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
            dval = lgb.Dataset(X_test, label=y_test, feature_name=FEATURE_NAMES, reference=dtrain)

            callbacks = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ]

            model = lgb.train(
                params, dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=callbacks,
            )

            # Evaluate
            preds = model.predict(X_test)
            pred_class = (preds > 0.5).astype(int)
            accuracy = float(np.mean(pred_class == y_test))
            all_accuracies.append(accuracy)

            # AUC
            from sklearn.metrics import roc_auc_score
            try:
                auc = float(roc_auc_score(y_test, preds))
            except ValueError:
                auc = 0.5

            log.info("[%s] Fold %d: accuracy=%.3f, AUC=%.3f, trees=%d",
                     symbol, fold + 1, accuracy, auc, model.num_trees())

            if auc > best_auc:
                best_auc = auc
                best_model = model

        if best_model is None:
            log.warning("[%s] Training produced no valid model", symbol)
            return False

        self.models[symbol] = best_model

        # Feature importance
        importance = best_model.feature_importance(importance_type="gain")
        imp_dict = {}
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(importance):
                imp_dict[name] = float(importance[i])
        # Sort by importance
        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
        self.feature_importance[symbol] = imp_dict

        # Metrics
        self._train_metrics[symbol] = {
            "accuracy": float(np.mean(all_accuracies)),
            "best_auc": best_auc,
            "n_samples": len(X),
            "n_trees": best_model.num_trees(),
            "timestamp": time.time(),
        }

        log.info("[%s] Training complete: accuracy=%.3f, AUC=%.3f, trees=%d",
                 symbol, np.mean(all_accuracies), best_auc, best_model.num_trees())

        # Save
        self.save(symbol)
        return True

    def predict(self, symbol, features: np.ndarray):
        """
        Predict direction for a symbol.
        Returns: {
            "direction": "LONG" | "SHORT" | "FLAT",
            "prob_up": float,
            "prob_down": float,
            "confidence": float,
        }
        """
        result = {
            "direction": "FLAT",
            "prob_up": 0.5,
            "prob_down": 0.5,
            "confidence": 0.0,
        }

        if symbol not in self.models:
            return result

        model = self.models[symbol]

        try:
            # Reshape for single prediction
            X = features.reshape(1, -1)
            prob_up = float(model.predict(X)[0])
            prob_down = 1.0 - prob_up

            confidence = abs(prob_up - 0.5) * 2.0  # 0 to 1 scale

            if prob_up > (0.5 + CONFIDENCE_THRESHOLD / 2):
                direction = "LONG"
            elif prob_down > (0.5 + CONFIDENCE_THRESHOLD / 2):
                direction = "SHORT"
            else:
                direction = "FLAT"

            result = {
                "direction": direction,
                "prob_up": prob_up,
                "prob_down": prob_down,
                "confidence": confidence,
            }

        except Exception as e:
            log.warning("[%s] Prediction error: %s", symbol, e)

        return result

    def train_all(self, mt5_conn, feature_engine: FeatureEngine):
        """Train models for all symbols."""
        for symbol in SYMBOLS:
            try:
                self.train(symbol, mt5_conn, feature_engine)
            except Exception as e:
                log.error("[%s] Training failed: %s", symbol, e)

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
