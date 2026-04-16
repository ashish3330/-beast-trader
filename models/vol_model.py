"""
Beast Trader — Volatility Prediction Model.
LightGBM regressor: predicts ATR_next_4bars / ATR_current (H1 timeframe).
Walk-forward validation. Train/save/load like signal_model.
"""
import os
import time
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Tuple

log = logging.getLogger("beast.vol_model")

# Paths
MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")

CACHE_FILES = {
    "XAUUSD":   "raw_h1_xauusd.pkl",
    "BTCUSD":   "raw_h1_BTCUSD.pkl",
    "NAS100.r": "raw_h1_NAS100_r.pkl",
    "GER40.r":  "raw_h1_GER40_r.pkl",
}

# Feature names for the model
FEATURE_NAMES = [
    "atr_pctl_20",       # ATR percentile over 20-bar lookback
    "atr_pctl_50",       # ATR percentile over 50-bar lookback
    "atr_pctl_100",      # ATR percentile over 100-bar lookback
    "hour_sin",          # hour of day (sine encoding)
    "hour_cos",          # hour of day (cosine encoding)
    "dow_sin",           # day of week (sine encoding)
    "dow_cos",           # day of week (cosine encoding)
    "bb_width",          # Bollinger Band width (%)
    "bb_width_pctl_20",  # BB width percentile over 20-bar lookback
    "range_ratio_5",     # avg(high-low last 5 bars) / ATR
    "range_ratio_10",    # avg(high-low last 10 bars) / ATR
    "hl_range_1",        # high-low range of current bar / ATR
    "hl_range_2",        # high-low range of bar -1 / ATR
    "hl_range_3",        # high-low range of bar -2 / ATR
    "hl_range_4",        # high-low range of bar -3 / ATR
    "hl_range_5",        # high-low range of bar -4 / ATR
    "atr_change_5",      # ATR now / ATR 5 bars ago
    "atr_change_10",     # ATR now / ATR 10 bars ago
    "close_vs_bb_upper", # (close - BB_upper) / ATR
    "close_vs_bb_lower", # (close - BB_lower) / ATR
    "vol_ratio",         # tick_volume / SMA(tick_volume, 20)
    "body_range_ratio",  # |close-open| / (high-low)
]

NUM_FEATURES = len(FEATURE_NAMES)


# ═══════════════════════════════════════════════════════════
# NUMPY HELPERS
# ═══════════════════════════════════════════════════════════

def _ema(a, p):
    al = 2 / (p + 1)
    o = np.empty_like(a, dtype=np.float64)
    o[0] = a[0]
    for i in range(1, len(a)):
        o[i] = al * a[i] + (1 - al) * o[i - 1]
    return o


def _atr(h, l, c, p):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return _ema(tr, p)


def _bb(c, ln=20, ns=2.0):
    n = len(c)
    bbu = np.full(n, np.nan, dtype=np.float64)
    bbl = np.full(n, np.nan, dtype=np.float64)
    bbw = np.full(n, np.nan, dtype=np.float64)
    for i in range(ln - 1, n):
        win = c[i - ln + 1:i + 1]
        m = np.mean(win)
        s = np.std(win)
        bbu[i] = m + ns * s
        bbl[i] = m - ns * s
        bbw[i] = (bbu[i] - bbl[i]) / m * 100 if m > 0 else 0
    return bbu, bbl, bbw


def _percentile_rank(arr, i, lookback):
    """Percentile rank of arr[i] within arr[i-lookback+1:i+1]."""
    if i < lookback - 1:
        return 0.5
    window = arr[i - lookback + 1:i + 1]
    valid = window[~np.isnan(window)]
    if len(valid) < 2:
        return 0.5
    return float(np.sum(valid <= arr[i]) / len(valid))


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features and labels from H1 dataframe.

    Returns:
        X: (n_samples, NUM_FEATURES) feature matrix
        y: (n_samples,) target = ATR_next_4bars / ATR_current
        valid_mask: boolean array marking valid rows
    """
    c = df["close"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    n = len(c)

    vol = df["tick_volume"].values.astype(np.float64) if "tick_volume" in df.columns else np.ones(n)

    # Timestamps
    if pd.api.types.is_datetime64_any_dtype(df["time"]):
        hours = df["time"].dt.hour.values.astype(np.float64)
        dows = df["time"].dt.dayofweek.values.astype(np.float64)
    else:
        times = pd.to_datetime(df["time"], unit="s", utc=True)
        hours = times.dt.hour.values.astype(np.float64)
        dows = times.dt.dayofweek.values.astype(np.float64)

    # Core indicators
    atr_14 = _atr(h, l, c, 14)
    bbu, bbl, bbw = _bb(c, 20, 2.0)

    # Volume SMA
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    for j in range(20, n):
        vol_sma[j] = np.mean(vol[j - 20:j])

    # High-low ranges
    hl_range = h - l

    # Forward ATR (target): average ATR of next 4 bars
    fwd_atr = np.full(n, np.nan, dtype=np.float64)
    for j in range(n - 4):
        fwd_atr[j] = np.mean(atr_14[j + 1:j + 5])

    # Build feature matrix
    min_lookback = 100  # need 100 bars for percentile calculations
    X = np.full((n, NUM_FEATURES), np.nan, dtype=np.float64)
    y = np.full(n, np.nan, dtype=np.float64)

    for i in range(min_lookback, n):
        atr_now = atr_14[i]
        if atr_now <= 0 or np.isnan(atr_now):
            continue

        feat = np.zeros(NUM_FEATURES, dtype=np.float64)

        # ATR percentiles
        feat[0] = _percentile_rank(atr_14, i, 20)
        feat[1] = _percentile_rank(atr_14, i, 50)
        feat[2] = _percentile_rank(atr_14, i, 100)

        # Time encoding (cyclical)
        feat[3] = np.sin(2 * np.pi * hours[i] / 24.0)
        feat[4] = np.cos(2 * np.pi * hours[i] / 24.0)
        feat[5] = np.sin(2 * np.pi * dows[i] / 7.0)
        feat[6] = np.cos(2 * np.pi * dows[i] / 7.0)

        # BB width
        feat[7] = bbw[i] if not np.isnan(bbw[i]) else 0.0
        feat[8] = _percentile_rank(bbw, i, 20)

        # Range ratios
        if i >= 5:
            feat[9] = np.mean(hl_range[i - 4:i + 1]) / atr_now
        if i >= 10:
            feat[10] = np.mean(hl_range[i - 9:i + 1]) / atr_now

        # Individual bar HL ranges (last 5 bars) normalized by ATR
        for k in range(5):
            if i - k >= 0:
                feat[11 + k] = hl_range[i - k] / atr_now

        # ATR change ratios
        if i >= 5 and atr_14[i - 5] > 0:
            feat[16] = atr_now / atr_14[i - 5]
        if i >= 10 and atr_14[i - 10] > 0:
            feat[17] = atr_now / atr_14[i - 10]

        # Close vs BB bands
        if not np.isnan(bbu[i]):
            feat[18] = (c[i] - bbu[i]) / atr_now
        if not np.isnan(bbl[i]):
            feat[19] = (c[i] - bbl[i]) / atr_now

        # Volume ratio
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 0:
            feat[20] = vol[i] / vol_sma[i]

        # Body/range ratio
        if hl_range[i] > 0:
            feat[21] = abs(c[i] - o[i]) / hl_range[i]

        X[i] = feat

        # Target: vol multiplier = ATR_next_4bars / ATR_current
        if not np.isnan(fwd_atr[i]) and atr_now > 0:
            y[i] = fwd_atr[i] / atr_now

    # Valid mask: both features and target present
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)

    return X[valid], y[valid], valid


# ═══════════════════════════════════════════════════════════
# VOLATILITY MODEL
# ═══════════════════════════════════════════════════════════

class VolatilityModel:
    """LightGBM regressor for volatility prediction."""

    def __init__(self):
        self.models = {}                # symbol -> lgb.Booster
        self.feature_importance = {}    # symbol -> dict
        self._train_metrics = {}        # symbol -> dict

    def load(self, symbol: str = None):
        """Load saved model(s) from disk."""
        symbols = [symbol] if symbol else list(CACHE_FILES.keys())
        for sym in symbols:
            model_path = MODEL_DIR / f"{sym.replace('.', '_')}_vol_lgb.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        data = pickle.load(f)
                    self.models[sym] = data["model"]
                    self.feature_importance[sym] = data.get("importance", {})
                    self._train_metrics[sym] = data.get("metrics", {})
                    log.info("[%s] Vol model loaded from %s", sym, model_path)
                except Exception as e:
                    log.error("[%s] Vol model load failed: %s", sym, e)

    def save(self, symbol: str):
        """Save model to disk."""
        if symbol not in self.models:
            return
        model_path = MODEL_DIR / f"{symbol.replace('.', '_')}_vol_lgb.pkl"
        try:
            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": self.models[symbol],
                    "importance": self.feature_importance.get(symbol, {}),
                    "metrics": self._train_metrics.get(symbol, {}),
                    "timestamp": time.time(),
                }, f)
            log.info("[%s] Vol model saved to %s", symbol, model_path)
        except Exception as e:
            log.error("[%s] Vol model save failed: %s", symbol, e)

    def train(self, symbol: str, df: pd.DataFrame = None) -> bool:
        """
        Train volatility model for a symbol using H1 data.
        Walk-forward validation: train on 70%, validate on rolling 15% chunks.

        Args:
            symbol: symbol name
            df: optional DataFrame; if None, loads from cache
        """
        log.info("[%s] Training volatility model...", symbol)

        if df is None:
            cache_file = CACHE_FILES.get(symbol)
            if not cache_file:
                log.warning("[%s] No cache file configured", symbol)
                return False
            cache_path = CACHE_DIR / cache_file
            if not cache_path.exists():
                log.warning("[%s] Cache file not found: %s", symbol, cache_path)
                return False
            df = pickle.load(open(cache_path, "rb"))

        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # Extract features
        X, y, valid_mask = extract_features(df)

        if len(X) < 500:
            log.warning("[%s] Not enough samples: %d", symbol, len(X))
            return False

        log.info("[%s] Feature matrix: %d samples x %d features", symbol, X.shape[0], X.shape[1])

        # Walk-forward cross-validation
        train_size = int(len(X) * 0.6)
        n_folds = 3
        fold_size = (len(X) - train_size) // n_folds

        best_model = None
        best_mae = float("inf")
        all_maes = []
        all_r2s = []

        for fold in range(n_folds):
            train_end = train_size + fold * fold_size
            test_end = min(train_end + fold_size, len(X))
            if train_end >= len(X) or test_end <= train_end:
                break

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]

            params = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "n_jobs": -1,
                "seed": 42,
                "min_child_samples": 20,
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
            mae = float(np.mean(np.abs(preds - y_test)))
            ss_res = np.sum((y_test - preds) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            all_maes.append(mae)
            all_r2s.append(r2)

            log.info("[%s] Fold %d: MAE=%.4f, R2=%.4f, trees=%d",
                     symbol, fold + 1, mae, r2, model.num_trees())

            if mae < best_mae:
                best_mae = mae
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
        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
        self.feature_importance[symbol] = imp_dict

        # Metrics
        self._train_metrics[symbol] = {
            "mae": float(np.mean(all_maes)),
            "best_mae": float(best_mae),
            "r2": float(np.mean(all_r2s)),
            "n_samples": len(X),
            "n_trees": best_model.num_trees(),
            "timestamp": time.time(),
        }

        log.info("[%s] Vol model trained: MAE=%.4f, R2=%.4f, trees=%d",
                 symbol, np.mean(all_maes), np.mean(all_r2s), best_model.num_trees())

        self.save(symbol)
        return True

    def predict(self, symbol: str, features: np.ndarray) -> float:
        """
        Predict volatility multiplier for a symbol.

        Args:
            symbol: symbol name
            features: (NUM_FEATURES,) feature vector

        Returns:
            Volatility multiplier (e.g. 1.3 = expect 30% vol increase).
            Returns 1.0 (neutral) if no model available.
        """
        if symbol not in self.models:
            return 1.0

        try:
            X = features.reshape(1, -1)
            pred = float(self.models[symbol].predict(X)[0])
            # Clamp to reasonable range [0.3, 3.0]
            pred = max(0.3, min(3.0, pred))
            return pred
        except Exception as e:
            log.warning("[%s] Vol prediction error: %s", symbol, e)
            return 1.0

    def predict_from_df(self, symbol: str, df: pd.DataFrame, bar_idx: int = -1) -> float:
        """
        Convenience: extract features from a DataFrame row and predict.

        Args:
            symbol: symbol name
            df: H1 DataFrame
            bar_idx: which bar to predict for (-1 = last valid)

        Returns:
            Volatility multiplier.
        """
        if symbol not in self.models:
            return 1.0

        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        X, _, valid = extract_features(df)
        if len(X) == 0:
            return 1.0

        if bar_idx == -1:
            features = X[-1]
        else:
            # Map bar_idx to valid index
            valid_indices = np.where(valid)[0]
            target = valid_indices[valid_indices <= bar_idx]
            if len(target) == 0:
                return 1.0
            features = X[len(target) - 1]

        return self.predict(symbol, features)

    def train_all(self) -> Dict:
        """Train models for all configured symbols."""
        results = {}
        for symbol in CACHE_FILES:
            try:
                ok = self.train(symbol)
                results[symbol] = "OK" if ok else "FAIL"
            except Exception as e:
                log.error("[%s] Vol training failed: %s", symbol, e)
                results[symbol] = f"ERROR: {e}"
        return results

    def load_all(self):
        """Load all saved models."""
        for symbol in CACHE_FILES:
            self.load(symbol)

    def has_model(self, symbol: str) -> bool:
        return symbol in self.models

    def get_importance(self, symbol: str) -> dict:
        return self.feature_importance.get(symbol, {})

    def get_metrics(self, symbol: str) -> dict:
        return self._train_metrics.get(symbol, {})


# ═══════════════════════════════════════════════════════════
# MAIN — train and evaluate all symbols
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Beast Trader Volatility Model")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to train")
    parser.add_argument("--load", action="store_true", help="Load and evaluate instead of train")
    args = parser.parse_args()

    vm = VolatilityModel()

    if args.load:
        if args.symbol:
            vm.load(args.symbol)
        else:
            vm.load_all()

        print("\n" + "=" * 70)
        print("  VOLATILITY MODEL — Loaded Models")
        print("=" * 70)
        for sym in CACHE_FILES:
            if vm.has_model(sym):
                m = vm.get_metrics(sym)
                print(f"  {sym:<12} MAE={m.get('mae', '?'):.4f}  R2={m.get('r2', '?'):.4f}  "
                      f"trees={m.get('n_trees', '?')}")
            else:
                print(f"  {sym:<12} (no model)")
        print()

    else:
        print("\n" + "=" * 70)
        print("  VOLATILITY MODEL — Training")
        print("=" * 70)

        t0 = time.time()

        if args.symbol:
            ok = vm.train(args.symbol)
            print(f"\n  {args.symbol}: {'OK' if ok else 'FAIL'}")
        else:
            results = vm.train_all()
            for sym, status in results.items():
                m = vm.get_metrics(sym)
                if m:
                    print(f"  {sym:<12} {status:<6} MAE={m.get('mae', 0):.4f}  "
                          f"R2={m.get('r2', 0):.4f}  trees={m.get('n_trees', 0)}")
                else:
                    print(f"  {sym:<12} {status}")

        elapsed = time.time() - t0
        print(f"\n  Elapsed: {elapsed:.1f}s")

        # Show top features for each symbol
        print("\n" + "-" * 70)
        print("  TOP 5 FEATURES BY SYMBOL")
        print("-" * 70)
        for sym in CACHE_FILES:
            imp = vm.get_importance(sym)
            if imp:
                top5 = list(imp.items())[:5]
                feats = ", ".join(f"{k}={v:.0f}" for k, v in top5)
                print(f"  {sym:<12} {feats}")

        # Quick prediction test
        print("\n" + "-" * 70)
        print("  SAMPLE PREDICTIONS (last bar of each symbol)")
        print("-" * 70)
        for sym, cache_file in CACHE_FILES.items():
            if not vm.has_model(sym):
                continue
            cache_path = CACHE_DIR / cache_file
            if not cache_path.exists():
                continue
            df = pickle.load(open(cache_path, "rb"))
            if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            pred = vm.predict_from_df(sym, df)
            label = "EXPANSION" if pred > 1.1 else ("CONTRACTION" if pred < 0.9 else "STABLE")
            print(f"  {sym:<12} vol_mult={pred:.3f}  ({label})")

        print("\n" + "=" * 70)
