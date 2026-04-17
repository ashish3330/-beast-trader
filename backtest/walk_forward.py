"""
WALK-FORWARD VALIDATION — Dragon Trader gold-standard OOS test.

Method:
  1. Load full H1 data (~2.5 years) per symbol from cache.
  2. Split into overlapping 6-month windows (4 mo train + 2 mo test).
  3. Slide window forward by 2 months each step.
  4. On each window:
     - Train LightGBM meta-label model on TRAIN period only.
     - Run full Dragon backtest on TEST period with that model.
  5. Report PF / WR / DD / trades per window per symbol.
  6. Strategy is robust if PF > 1.0 in ALL test windows.

python3 -B backtest/walk_forward.py
"""
import sys, pickle, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.momentum_scorer import (
    _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES,
    REGIME_PARAMS, DEFAULT_PARAMS, MIN_SCORE,
)
from config import (
    SYMBOLS, SYMBOL_TRAIL_OVERRIDE, TRAIL_STEPS,
    SYMBOL_ATR_SL_OVERRIDE as CFG_ATR_SL_OVERRIDE,
    ATR_SL_MULTIPLIER,
    DRAGON_SYMBOL_MIN_SCORE, DRAGON_ML_ENABLED,
    SYMBOL_SESSION_OVERRIDE as CFG_SESSION_OVERRIDE,
)
from models.signal_model import (
    SignalModel, META_FEATURE_NAMES, NUM_META_FEATURES,
    TUNED_LGB_PARAMS, DEFAULT_LGB_PARAMS,
)

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed. pip install lightgbm")
    sys.exit(1)

# ═══ CONSTANTS ═══
CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
START_EQ = 1000.0
SLIP = 0.0
RISK_PCT = 0.008                 # 0.8% risk per trade (matches dragon_backtest)
DAILY_LOSS_LIMIT = 0.01
CONSEC_LOSS_COOLDOWN = 24
TRAIN_MONTHS = 4
TEST_MONTHS = 2
SLIDE_MONTHS = 2
CONFIDENCE_THRESHOLD = 0.56      # from config DRAGON_CONFIDENCE_FLOOR

# Dragon 6 symbols + their cache/point/tv/spread/lot
DRAGON_SYMBOLS = {
    "XAUUSD":   {"cache": "raw_h1_xauusd.pkl",   "point": 0.01,   "tv": 1.0,    "spread": 0.33,  "lot": 0.01, "cat": "Gold"},
    "XAGUSD":   {"cache": "raw_h1_XAGUSD.pkl",   "point": 0.001,  "tv": 5.0,    "spread": 0.035, "lot": 0.01, "cat": "Gold"},
    "BTCUSD":   {"cache": "raw_h1_BTCUSD.pkl",   "point": 0.01,   "tv": 0.01,   "spread": 17.0,  "lot": 0.01, "cat": "Crypto"},
    "NAS100.r": {"cache": "raw_h1_NAS100_r.pkl",  "point": 0.01,   "tv": 0.01,   "spread": 1.80,  "lot": 0.10, "cat": "Index"},
    "JPN225ft": {"cache": "raw_h1_JPN225ft.pkl",  "point": 0.01,   "tv": 0.0063, "spread": 10.0,  "lot": 1.00, "cat": "Index"},
    "USDJPY":   {"cache": "raw_h1_USDJPY.pkl",    "point": 0.001,  "tv": 0.63,   "spread": 0.018, "lot": 0.20, "cat": "Forex"},
}

# Session overrides (default 06-22 UTC for non-crypto)
SESSION_OVERRIDE = {
    "JPN225ft": (0, 22),
}
SESSION_OVERRIDE.update(CFG_SESSION_OVERRIDE)


# ═══ REGIME / SCORE HELPERS (same as dragon_backtest) ═══
def get_regime(ind, bi):
    if bi < 21 or np.isnan(ind["bbw"][bi]):
        return "unknown"
    bbw = float(ind["bbw"][bi])
    adx = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 0
    if bbw < 1.5 and adx < 20:
        return "ranging"
    if 1.5 <= bbw < 3.0 and adx > 25:
        return "trending"
    if bbw >= 3.0:
        return "volatile"
    return "low_vol"


def get_adaptive_min_score(regime, symbol=None):
    if symbol and symbol in DRAGON_SYMBOL_MIN_SCORE:
        sym_scores = DRAGON_SYMBOL_MIN_SCORE[symbol]
        if regime in sym_scores:
            return sym_scores[regime]
    return {"trending": 6.0, "ranging": 8.0, "volatile": 7.0, "low_vol": 7.0}.get(regime, 7.0)


# ═══ ML META-FEATURE BUILDER (from SignalModel.train — extracted) ═══
def build_meta_features_batch(ind, df, signals, labels):
    """Build meta-feature matrix for a list of (bar_i, direction, ls, ss, cs) tuples."""
    n = ind["n"]
    close = ind["c"]; high = ind["h"]; low = ind["l"]; atr = ind["at"]

    # Pre-compute ATR percentile
    atr_pctile = np.full(n, 0.5, dtype=np.float64)
    for i in range(200, n):
        window = atr[max(0, i - 200):i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            atr_pctile[i] = float(np.searchsorted(np.sort(valid), atr[i])) / len(valid)

    # Vol percentile
    vol_series = pd.Series(close).pct_change().rolling(14).std().fillna(0).values
    vol_pctile = np.full(n, 0.5, dtype=np.float64)
    for i in range(200, n):
        window = vol_series[max(0, i - 200):i + 1]
        if len(window) > 0 and vol_series[i] > 0:
            vol_pctile[i] = float(np.searchsorted(np.sort(window), vol_series[i])) / len(window)

    spread_proxy = np.maximum(0.0, (high - low) - np.abs(ind["o"] - close))

    X = np.zeros((len(signals), NUM_META_FEATURES), dtype=np.float64)
    win_streak = 0

    for j, (bar_i, direction, ls, ss, cs) in enumerate(signals):
        a = atr[bar_i] if not np.isnan(atr[bar_i]) and atr[bar_i] > 0 else 1.0
        mid = close[bar_i]

        X[j, 0] = ls
        X[j, 1] = ss
        X[j, 2] = cs
        X[j, 3] = float(direction)
        X[j, 4] = ind["adx"][bar_i] if not np.isnan(ind["adx"][bar_i]) else 25.0
        X[j, 5] = ind["bbw"][bar_i] if not np.isnan(ind["bbw"][bar_i]) else 2.0
        X[j, 6] = atr_pctile[bar_i]
        rsi_val = ind["rs"][bar_i]
        X[j, 7] = rsi_val if not np.isnan(rsi_val) else 50.0

        st_val = ind["stl"][bar_i]
        if not np.isnan(st_val) and a > 0:
            X[j, 8] = (mid - st_val) / a * direction
        else:
            X[j, 8] = 0.0

        es, el, et = ind["es"][bar_i], ind["el"][bar_i], ind["et"][bar_i]
        if direction == 1:
            align = (1 if mid > es else -1) + (1 if es > el else -1) + (1 if el > et else -1)
        else:
            align = (1 if mid < es else -1) + (1 if es < el else -1) + (1 if el < et else -1)
        X[j, 9] = float(align) / 3.0
        X[j, 10] = ind["mh"][bar_i] / a if a > 0 else 0.0
        X[j, 11] = vol_pctile[bar_i]
        X[j, 12] = float(ind["st"][bar_i]) * direction

        dt = df["time"].iloc[bar_i]
        hour = dt.hour; dow = dt.dayofweek
        X[j, 13] = np.sin(2 * np.pi * hour / 24.0)
        X[j, 14] = np.cos(2 * np.pi * hour / 24.0)
        X[j, 15] = np.sin(2 * np.pi * dow / 5.0)
        X[j, 16] = np.cos(2 * np.pi * dow / 5.0)
        X[j, 17] = spread_proxy[bar_i] / a if a > 0 else 0.0
        X[j, 18] = float(win_streak)
        if j > 0 and labels is not None and j - 1 < len(labels):
            if labels[j - 1] == 1:
                win_streak = max(win_streak + 1, 1)
            else:
                win_streak = min(win_streak - 1, -1)

        opposite_score = ss if direction == 1 else ls
        X[j, 19] = cs - opposite_score
        X[j, 20] = cs - MIN_SCORE

        # Multi-timeframe
        if bar_i >= 5:
            m15_c = close[max(0, bar_i - 4):bar_i + 1]
            m15_g = np.maximum(0, np.diff(m15_c))
            m15_l = np.maximum(0, -np.diff(m15_c))
            avg_g = np.mean(m15_g) if len(m15_g) > 0 else 0
            avg_l = np.mean(m15_l) if len(m15_l) > 0 else 0.001
            X[j, 21] = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 50.0
        else:
            X[j, 21] = 50.0

        if bar_i >= 10:
            ema5 = np.mean(close[bar_i - 4:bar_i + 1])
            ema10 = np.mean(close[bar_i - 9:bar_i + 1])
            X[j, 22] = (1.0 if ema5 > ema10 else -1.0) * direction
        else:
            X[j, 22] = 0.0

        if bar_i >= 5:
            recent_atr = np.mean(high[bar_i - 4:bar_i + 1] - low[bar_i - 4:bar_i + 1])
            X[j, 23] = recent_atr / a if a > 0 else 1.0
        else:
            X[j, 23] = 1.0

        # Momentum persistence
        if bar_i >= 5:
            X[j, 24] = (close[bar_i] - close[bar_i - 1]) / a if a > 0 else 0
            X[j, 25] = (close[bar_i] - close[bar_i - 3]) / a if a > 0 else 0
            X[j, 26] = (close[bar_i] - close[bar_i - 5]) / a if a > 0 else 0
        X[j, 27] = float(ind["consec"][bar_i]) if not np.isnan(ind["consec"][bar_i]) else 0.0

        # ATR change
        if bar_i >= 5 and not np.isnan(atr[bar_i - 5]) and atr[bar_i - 5] > 0:
            X[j, 28] = atr[bar_i] / atr[bar_i - 5]
        else:
            X[j, 28] = 1.0

        # BB squeeze
        bbw = ind["bbw"][bar_i] if not np.isnan(ind["bbw"][bar_i]) else 2.0
        if bar_i >= 200:
            bbw_win = ind["bbw"][bar_i - 200:bar_i + 1]
            bbw_valid = bbw_win[~np.isnan(bbw_win)]
            X[j, 29] = 1.0 if len(bbw_valid) > 0 and bbw <= np.percentile(bbw_valid, 20) else 0.0

        # Reversal detection
        if bar_i >= 10:
            price_higher = close[bar_i] > np.max(close[bar_i - 10:bar_i])
            rsi_v = ind["rs"][bar_i]
            rsi_lower = rsi_v < np.nanmax(ind["rs"][bar_i - 10:bar_i]) if not np.isnan(rsi_v) else False
            price_lower = close[bar_i] < np.min(close[bar_i - 10:bar_i])
            rsi_higher = rsi_v > np.nanmin(ind["rs"][bar_i - 10:bar_i]) if not np.isnan(rsi_v) else False
            if direction == 1:
                X[j, 30] = -1.0 if (price_higher and rsi_lower) else (1.0 if (price_lower and rsi_higher) else 0.0)
            else:
                X[j, 30] = 1.0 if (price_lower and rsi_higher) else (-1.0 if (price_higher and rsi_lower) else 0.0)

        if bar_i >= 20 and a > 0:
            X[j, 31] = (np.max(high[bar_i - 20:bar_i + 1]) - close[bar_i]) / a
            X[j, 32] = (close[bar_i] - np.min(low[bar_i - 20:bar_i + 1])) / a

    return X


# ═══ TRADE SIMULATION FOR ML LABELS (same as SignalModel._simulate_trade) ═══
def simulate_trade_outcome(bar_i, direction, close, high, low, atr, n,
                           trail_steps, sl_mult=1.5, max_hold=30):
    """Returns 1.0 if profitable, 0.0 if loss."""
    a = atr[bar_i] if not np.isnan(atr[bar_i]) and atr[bar_i] > 0 else 1.0
    entry = close[bar_i]
    sl_dist = a * sl_mult
    tp_dist = sl_dist * 2.5

    if direction == 1:
        sl = entry - sl_dist; tp = entry + tp_dist
    else:
        sl = entry + sl_dist; tp = entry - tp_dist

    best_price = entry; current_sl = sl

    for k in range(1, min(max_hold, n - bar_i)):
        bk = bar_i + k
        h_k = high[bk]; l_k = low[bk]; c_k = close[bk]

        if direction == 1:
            if l_k <= current_sl:
                return 1.0 if current_sl > entry else 0.0
            if h_k >= tp:
                return 1.0
            if c_k > best_price:
                best_price = c_k
                profit_r = (best_price - entry) / sl_dist
                for thr, action, param in trail_steps:
                    if profit_r >= thr:
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
        else:
            if h_k >= current_sl:
                return 1.0 if current_sl < entry else 0.0
            if l_k <= tp:
                return 1.0
            if c_k < best_price:
                best_price = c_k
                profit_r = (entry - best_price) / sl_dist
                for thr, action, param in trail_steps:
                    if profit_r >= thr:
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

    final = close[min(bar_i + max_hold, n - 1)]
    if direction == 1:
        return 1.0 if final > entry else 0.0
    else:
        return 1.0 if final < entry else 0.0


# ═══ TRAIN LGB ON TRAIN-PERIOD SIGNALS ═══
def train_ml_model(symbol, ind, df, train_start_idx, train_end_idx):
    """Train LightGBM meta-label model on signals in [train_start_idx, train_end_idx)."""
    n = ind["n"]
    close = ind["c"]; high = ind["h"]; low = ind["l"]; atr = ind["at"]

    trail_steps = SYMBOL_TRAIL_OVERRIDE.get(symbol, TRAIL_STEPS)
    sl_mult = CFG_ATR_SL_OVERRIDE.get(symbol, ATR_SL_MULTIPLIER)

    # Collect scored signals in train range
    signals = []
    start_bar = max(train_start_idx, 100)
    for i in range(start_bar, min(train_end_idx, n - 1)):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        ls, ss = _score(ind, i)
        buy = ls >= MIN_SCORE
        sell = ss >= MIN_SCORE
        if not buy and not sell:
            continue
        if buy and (not sell or ls >= ss):
            direction = 1; cs = ls
        else:
            direction = -1; cs = ss
        signals.append((i, direction, ls, ss, cs))

    if len(signals) < 40:
        return None  # not enough signals to train

    # Simulate trade outcomes for labels
    labels = np.array([
        simulate_trade_outcome(s[0], s[1], close, high, low, atr, n, trail_steps, sl_mult)
        for s in signals
    ], dtype=np.float64)

    # Build meta-features
    X = build_meta_features_batch(ind, df, signals, labels)
    y = labels

    # Class balance
    n_pos = np.sum(y == 1); n_neg = np.sum(y == 0)
    scale = n_neg / n_pos if n_pos > 0 else 1.0

    sym_hp = TUNED_LGB_PARAMS.get(symbol, DEFAULT_LGB_PARAMS)
    params = {
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
        "min_child_samples": max(5, int(len(X) * 0.01)),
        "lambda_l1": sym_hp["lambda_l1"],
        "lambda_l2": sym_hp["lambda_l2"],
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # Split train into 80/20 for early stopping
    split = int(len(X) * 0.80)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    if len(X_tr) < 20 or len(X_val) < 5:
        return None

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=META_FEATURE_NAMES)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=META_FEATURE_NAMES, reference=dtrain)

    callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)]

    model = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)
    return model


# ═══ BUILD META-FEATURES FOR A SINGLE BAR (for test-period prediction) ═══
def build_single_meta_features(ind, df, bar_i, direction, ls, ss, cs,
                               atr_pctile_arr, vol_pctile_arr, spread_proxy_arr,
                               win_streak):
    """Build 1-row meta-feature array for model.predict()."""
    X = np.zeros((1, NUM_META_FEATURES), dtype=np.float64)
    close = ind["c"]; high = ind["h"]; low = ind["l"]; atr = ind["at"]
    a = atr[bar_i] if not np.isnan(atr[bar_i]) and atr[bar_i] > 0 else 1.0
    mid = close[bar_i]

    X[0, 0] = ls; X[0, 1] = ss; X[0, 2] = cs; X[0, 3] = float(direction)
    X[0, 4] = ind["adx"][bar_i] if not np.isnan(ind["adx"][bar_i]) else 25.0
    X[0, 5] = ind["bbw"][bar_i] if not np.isnan(ind["bbw"][bar_i]) else 2.0
    X[0, 6] = atr_pctile_arr[bar_i]
    rsi_val = ind["rs"][bar_i]
    X[0, 7] = rsi_val if not np.isnan(rsi_val) else 50.0

    st_val = ind["stl"][bar_i]
    if not np.isnan(st_val) and a > 0:
        X[0, 8] = (mid - st_val) / a * direction

    es, el, et = ind["es"][bar_i], ind["el"][bar_i], ind["et"][bar_i]
    if direction == 1:
        align = (1 if mid > es else -1) + (1 if es > el else -1) + (1 if el > et else -1)
    else:
        align = (1 if mid < es else -1) + (1 if es < el else -1) + (1 if el < et else -1)
    X[0, 9] = float(align) / 3.0
    X[0, 10] = ind["mh"][bar_i] / a if a > 0 else 0.0
    X[0, 11] = vol_pctile_arr[bar_i]
    X[0, 12] = float(ind["st"][bar_i]) * direction

    dt = df["time"].iloc[bar_i]
    hour = dt.hour; dow = dt.dayofweek
    X[0, 13] = np.sin(2 * np.pi * hour / 24.0)
    X[0, 14] = np.cos(2 * np.pi * hour / 24.0)
    X[0, 15] = np.sin(2 * np.pi * dow / 5.0)
    X[0, 16] = np.cos(2 * np.pi * dow / 5.0)
    X[0, 17] = spread_proxy_arr[bar_i] / a if a > 0 else 0.0
    X[0, 18] = float(win_streak)

    opposite_score = ss if direction == 1 else ls
    X[0, 19] = cs - opposite_score
    X[0, 20] = cs - MIN_SCORE

    if bar_i >= 5:
        m15_c = close[max(0, bar_i - 4):bar_i + 1]
        m15_g = np.maximum(0, np.diff(m15_c))
        m15_l = np.maximum(0, -np.diff(m15_c))
        avg_g = np.mean(m15_g); avg_l = np.mean(m15_l) if np.mean(m15_l) > 0 else 0.001
        X[0, 21] = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 50.0

    if bar_i >= 10:
        ema5 = np.mean(close[bar_i - 4:bar_i + 1])
        ema10 = np.mean(close[bar_i - 9:bar_i + 1])
        X[0, 22] = (1.0 if ema5 > ema10 else -1.0) * direction

    if bar_i >= 5:
        recent_atr = np.mean(high[bar_i - 4:bar_i + 1] - low[bar_i - 4:bar_i + 1])
        X[0, 23] = recent_atr / a if a > 0 else 1.0

    if bar_i >= 5:
        X[0, 24] = (close[bar_i] - close[bar_i - 1]) / a if a > 0 else 0
        X[0, 25] = (close[bar_i] - close[bar_i - 3]) / a if a > 0 else 0
        X[0, 26] = (close[bar_i] - close[bar_i - 5]) / a if a > 0 else 0
    X[0, 27] = float(ind["consec"][bar_i]) if not np.isnan(ind["consec"][bar_i]) else 0.0

    if bar_i >= 5 and not np.isnan(atr[bar_i - 5]) and atr[bar_i - 5] > 0:
        X[0, 28] = atr[bar_i] / atr[bar_i - 5]
    else:
        X[0, 28] = 1.0

    bbw = ind["bbw"][bar_i] if not np.isnan(ind["bbw"][bar_i]) else 2.0
    if bar_i >= 200:
        bbw_win = ind["bbw"][bar_i - 200:bar_i + 1]
        bbw_valid = bbw_win[~np.isnan(bbw_win)]
        X[0, 29] = 1.0 if len(bbw_valid) > 0 and bbw <= np.percentile(bbw_valid, 20) else 0.0

    if bar_i >= 10:
        price_higher = close[bar_i] > np.max(close[bar_i - 10:bar_i])
        rv = ind["rs"][bar_i]
        rsi_lower = rv < np.nanmax(ind["rs"][bar_i - 10:bar_i]) if not np.isnan(rv) else False
        price_lower = close[bar_i] < np.min(close[bar_i - 10:bar_i])
        rsi_higher = rv > np.nanmin(ind["rs"][bar_i - 10:bar_i]) if not np.isnan(rv) else False
        if direction == 1:
            X[0, 30] = -1.0 if (price_higher and rsi_lower) else (1.0 if (price_lower and rsi_higher) else 0.0)
        else:
            X[0, 30] = 1.0 if (price_lower and rsi_higher) else (-1.0 if (price_higher and rsi_lower) else 0.0)

    if bar_i >= 20 and a > 0:
        X[0, 31] = (np.max(high[bar_i - 20:bar_i + 1]) - close[bar_i]) / a
        X[0, 32] = (close[bar_i] - np.min(low[bar_i - 20:bar_i + 1])) / a

    return X


# ═══ PRE-COMPUTE HELPER ARRAYS FOR TEST PERIOD ═══
def precompute_helper_arrays(ind):
    """Pre-compute ATR percentile, vol percentile, spread proxy for the full data."""
    n = ind["n"]
    close = ind["c"]; high = ind["h"]; low = ind["l"]; atr = ind["at"]

    atr_pctile = np.full(n, 0.5, dtype=np.float64)
    for i in range(200, n):
        window = atr[max(0, i - 200):i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            atr_pctile[i] = float(np.searchsorted(np.sort(valid), atr[i])) / len(valid)

    vol_series = pd.Series(close).pct_change().rolling(14).std().fillna(0).values
    vol_pctile = np.full(n, 0.5, dtype=np.float64)
    for i in range(200, n):
        window = vol_series[max(0, i - 200):i + 1]
        if len(window) > 0 and vol_series[i] > 0:
            vol_pctile[i] = float(np.searchsorted(np.sort(window), vol_series[i])) / len(window)

    spread_proxy = np.maximum(0.0, (high - low) - np.abs(ind["o"] - close))

    return atr_pctile, vol_pctile, spread_proxy


# ═══ BACKTEST ONE TEST WINDOW ═══
def backtest_window(symbol, scfg, ind, df, test_start_idx, test_end_idx, ml_model=None):
    """
    Run Dragon backtest on [test_start_idx, test_end_idx) using optional ML model.
    Returns metrics dict or None.
    """
    pt = scfg["point"]; tv = scfg["tv"]; spread = scfg["spread"]
    cat = scfg["cat"]
    sl_cap = 5000 * pt

    trail_steps = SYMBOL_TRAIL_OVERRIDE.get(symbol, TRAIL_STEPS)
    sym_sl_mult = CFG_ATR_SL_OVERRIDE.get(symbol, ATR_SL_MULTIPLIER)
    use_ml = DRAGON_ML_ENABLED.get(symbol, False) and ml_model is not None

    # Pre-compute helper arrays for ML prediction
    atr_pctile_arr = vol_pctile_arr = spread_proxy_arr = None
    if use_ml:
        atr_pctile_arr, vol_pctile_arr, spread_proxy_arr = precompute_helper_arrays(ind)

    n = ind["n"]
    eq = START_EQ; peak = START_EQ; max_dd = 0
    n_trades = 0; wins = 0; gross_p = 0; gross_l = 0
    in_trade = False; d = 0; entry = 0; pos_sl = 0; sl_dist = 0
    trade_lot = 0.0
    consec_losses = 0; cooldown_until = 0
    daily_pnl = 0.0; current_day = None; day_eq_start = START_EQ; day_stopped = False
    ml_win_streak = 0

    for i in range(max(test_start_idx, 100), min(test_end_idx, n)):
        atr_val = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0
        if atr_val == 0:
            continue

        bar_time = df["time"].iloc[i]
        bar_hour = bar_time.hour if hasattr(bar_time, "hour") else 12
        sess_start, sess_end = SESSION_OVERRIDE.get(symbol, (6, 22))
        if cat != "Crypto" and (bar_hour >= sess_end or bar_hour < sess_start):
            continue

        bar_date = bar_time.date() if hasattr(bar_time, "date") else None
        if bar_date and bar_date != current_day:
            current_day = bar_date; day_eq_start = eq; daily_pnl = 0.0; day_stopped = False

        # MANAGE trailing SL
        if in_trade:
            if (d == 1 and ind["l"][i] <= pos_sl) or (d == -1 and ind["h"][i] >= pos_sl):
                exit_cost = spread + SLIP * pt
                pnl = d * (pos_sl - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
                eq += pnl; daily_pnl += pnl
                if pnl > 0:
                    gross_p += pnl; wins += 1; consec_losses = 0
                else:
                    gross_l += abs(pnl); consec_losses += 1
                    if consec_losses >= 3:
                        cooldown_until = i + CONSEC_LOSS_COOLDOWN; consec_losses = 0
                n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
                in_trade = False
                if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                    day_stopped = True
                continue

            # Trailing SL update using per-symbol trail profile
            cur = float(ind["c"][i])
            profit_r = ((cur - entry) * d) / sl_dist if sl_dist > 0 else 0
            new_sl = None
            for thr, action, param in trail_steps:
                if profit_r >= thr:
                    if action == "trail":
                        new_sl = cur - param * atr_val * d
                    elif action == "lock":
                        new_sl = entry + param * sl_dist * d
                    elif action == "be":
                        new_sl = entry + 2 * pt * d
                    break
            if new_sl is not None:
                if d == 1 and new_sl > pos_sl:
                    pos_sl = new_sl
                elif d == -1 and new_sl < pos_sl:
                    pos_sl = new_sl

        if day_stopped:
            continue
        if i < cooldown_until:
            continue

        # SCORE
        bi = i - 1
        if bi < 21:
            continue
        ls, ss = _score(ind, bi)

        regime = get_regime(ind, bi)
        adaptive_min = get_adaptive_min_score(regime, symbol=symbol)

        buy = ls >= adaptive_min
        sell = ss >= adaptive_min
        if not buy and not sell:
            continue
        new_dir = 1 if (buy and (not sell or ls >= ss)) else -1
        chosen_score = ls if new_dir == 1 else ss

        # ML filter (proper walk-forward: model trained on past data only)
        if use_ml:
            X_pred = build_single_meta_features(
                ind, df, bi, new_dir, ls, ss, chosen_score,
                atr_pctile_arr, vol_pctile_arr, spread_proxy_arr, ml_win_streak
            )
            prob = float(ml_model.predict(X_pred)[0])
            if prob < CONFIDENCE_THRESHOLD:
                continue
        elif DRAGON_ML_ENABLED.get(symbol, False):
            # ML enabled but no model available -- use probabilistic filter like dragon_backtest
            np.random.seed(i)  # deterministic per bar
            pass_prob = min(1.0, max(ls, ss) / 10.0)
            if np.random.random() > pass_prob:
                continue

        # REVERSAL
        if in_trade and new_dir != d:
            exit_cost = spread + SLIP * pt
            pnl = d * (float(ind["c"][i]) - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
            eq += pnl; daily_pnl += pnl
            if pnl > 0:
                gross_p += pnl; wins += 1; consec_losses = 0; ml_win_streak = max(ml_win_streak + 1, 1)
            else:
                gross_l += abs(pnl); consec_losses += 1; ml_win_streak = min(ml_win_streak - 1, -1)
                if consec_losses >= 3:
                    cooldown_until = i + CONSEC_LOSS_COOLDOWN; consec_losses = 0
            n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
            in_trade = False
            if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                day_stopped = True; continue

        # ENTRY
        if not in_trade:
            d = new_dir
            sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
            sl_dist = max(atr_val * sl_m, atr_val * sym_sl_mult)
            sl_dist = min(sl_dist, sl_cap)
            risk_amount = eq * RISK_PCT
            pip_value_per_lot = (sl_dist / pt) * tv
            if pip_value_per_lot > 0:
                trade_lot = max(risk_amount / pip_value_per_lot, 0.01)
            else:
                trade_lot = 0.01
            entry_cost = spread + SLIP * pt
            entry = float(ind["o"][i]) + entry_cost / 2 * d
            pos_sl = entry - sl_dist * d; in_trade = True

    # Close open trade at end
    if in_trade:
        pnl = d * (float(ind["c"][min(test_end_idx - 1, n - 1)]) - entry) / pt * tv * trade_lot
        eq += pnl
        if pnl > 0:
            gross_p += pnl; wins += 1
        else:
            gross_l += abs(pnl)
        n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)

    if n_trades == 0:
        return None

    pf = gross_p / gross_l if gross_l > 0 else (999 if gross_p > 0 else 0)
    dd = max_dd / peak * 100 if peak else 0
    ret = (eq - START_EQ) / START_EQ * 100
    wr = wins / n_trades * 100

    return {
        "trades": n_trades,
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "ret": round(ret, 1),
        "dd": round(dd, 1),
        "eq": round(eq, 2),
    }


# ═══ GENERATE WALK-FORWARD WINDOWS ═══
def generate_windows(df):
    """
    Generate (train_start, train_end, test_start, test_end) index tuples.
    Each window: 4 months train + 2 months test, sliding by 2 months.
    """
    times = df["time"]
    t_min = times.min(); t_max = times.max()

    windows = []
    cursor = t_min
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=TRAIN_MONTHS)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)

        if test_end > t_max:
            break

        # Convert to index positions
        tr_s = df.index[times >= train_start].min()
        tr_e = df.index[times >= train_end].min()
        te_s = tr_e
        te_e = df.index[times >= test_end].min()

        if pd.isna(tr_s) or pd.isna(tr_e) or pd.isna(te_e):
            cursor += pd.DateOffset(months=SLIDE_MONTHS)
            continue

        windows.append({
            "train_start_idx": int(tr_s),
            "train_end_idx": int(tr_e),
            "test_start_idx": int(te_s),
            "test_end_idx": int(te_e),
            "train_label": f"{train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}",
            "test_label": f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
        })

        cursor += pd.DateOffset(months=SLIDE_MONTHS)

    return windows


# ═══ RUN WALK-FORWARD FOR ONE SYMBOL ═══
def run_symbol(symbol):
    """Run full walk-forward validation for a symbol. Returns list of window results."""
    scfg = DRAGON_SYMBOLS[symbol]
    cache_path = CACHE / scfg["cache"]
    if not cache_path.exists():
        print(f"  [SKIP] {symbol}: cache not found at {cache_path}")
        return []

    df = pickle.load(open(cache_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.reset_index(drop=True)

    # Compute indicators on full data (needed for all windows)
    icfg = dict(IND_DEFAULTS); icfg.update(IND_OVERRIDES.get(symbol, {}))
    ind = _compute_indicators(df, icfg)

    windows = generate_windows(df)
    if not windows:
        print(f"  [SKIP] {symbol}: not enough data for walk-forward windows")
        return []

    results = []
    for w_idx, w in enumerate(windows):
        t0 = time.time()

        # Train ML on train period (proper walk-forward -- no lookahead)
        ml_model = None
        if DRAGON_ML_ENABLED.get(symbol, False):
            ml_model = train_ml_model(symbol, ind, df, w["train_start_idx"], w["train_end_idx"])

        # Backtest on test period
        r = backtest_window(
            symbol, scfg, ind, df,
            w["test_start_idx"], w["test_end_idx"],
            ml_model=ml_model,
        )

        elapsed = time.time() - t0
        ml_tag = "ML" if ml_model else ("rnd" if DRAGON_ML_ENABLED.get(symbol, False) else "off")

        if r:
            r["window"] = w_idx + 1
            r["test_period"] = w["test_label"]
            r["train_period"] = w["train_label"]
            r["ml"] = ml_tag
            results.append(r)
            pf_color = "\033[92m" if r["pf"] >= 1.0 else "\033[91m"
            reset = "\033[0m"
            print(f"    W{w_idx+1:02d}  Test {w['test_label']}  "
                  f"N={r['trades']:>4}  WR={r['wr']:>5.1f}%  "
                  f"{pf_color}PF={r['pf']:>6.2f}{reset}  "
                  f"Ret={r['ret']:>+7.1f}%  DD={r['dd']:>5.1f}%  "
                  f"[{ml_tag}] ({elapsed:.1f}s)")
        else:
            print(f"    W{w_idx+1:02d}  Test {w['test_label']}  "
                  f"NO TRADES ({elapsed:.1f}s)")

    return results


# ═══ MAIN ═══
def main():
    print("=" * 110)
    print("  WALK-FORWARD VALIDATION -- Dragon Trader")
    print(f"  Windows: {TRAIN_MONTHS}mo train + {TEST_MONTHS}mo test, sliding {SLIDE_MONTHS}mo")
    print(f"  ML meta-label: per-symbol (train on each window, predict on OOS)")
    print(f"  Risk: {RISK_PCT*100:.1f}% | Spread-only (no slippage) | Dragon scoring + per-symbol trail")
    print("=" * 110)

    all_results = {}
    for symbol in sorted(DRAGON_SYMBOLS.keys()):
        print(f"\n{'─' * 80}")
        print(f"  {symbol}  (ML={'ON' if DRAGON_ML_ENABLED.get(symbol, False) else 'OFF'})")
        print(f"{'─' * 80}")
        results = run_symbol(symbol)
        all_results[symbol] = results

    # ═══ SUMMARY TABLE ═══
    print(f"\n\n{'=' * 110}")
    print("  WALK-FORWARD SUMMARY")
    print(f"{'=' * 110}")
    print(f"\n{'Symbol':<12} {'Windows':>8} {'AllPF>1':>8} {'AvgPF':>7} {'AvgWR':>7} {'AvgDD':>7} {'AvgRet':>8} {'MinPF':>7} {'MaxPF':>7} {'Verdict':>10}")
    print("-" * 95)

    portfolio_pass = True
    for symbol in sorted(DRAGON_SYMBOLS.keys()):
        results = all_results.get(symbol, [])
        if not results:
            print(f"{symbol:<12} {'N/A':>8} {'--':>8} {'--':>7} {'--':>7} {'--':>7} {'--':>8} {'--':>7} {'--':>7} {'NO DATA':>10}")
            portfolio_pass = False
            continue

        n_win = len(results)
        pfs = [r["pf"] for r in results]
        wrs = [r["wr"] for r in results]
        dds = [r["dd"] for r in results]
        rets = [r["ret"] for r in results]
        all_above_1 = all(pf >= 1.0 for pf in pfs)
        avg_pf = np.mean(pfs)
        min_pf = min(pfs)
        max_pf = max(pfs)

        if not all_above_1:
            portfolio_pass = False

        verdict = "ROBUST" if all_above_1 else "FRAGILE"
        v_color = "\033[92m" if all_above_1 else "\033[91m"
        reset = "\033[0m"

        n_pass = sum(1 for pf in pfs if pf >= 1.0)
        pass_label = f"{n_pass}/{n_win}"

        print(f"{symbol:<12} {n_win:>8} {pass_label:>8} {avg_pf:>7.2f} {np.mean(wrs):>6.1f}% {np.mean(dds):>6.1f}% {np.mean(rets):>+7.1f}% {min_pf:>7.2f} {max_pf:>7.2f} {v_color}{verdict:>10}{reset}")

    print("-" * 95)

    # Per-window cross-symbol view
    all_windows_flat = []
    for sym, results in all_results.items():
        for r in results:
            all_windows_flat.append(r)

    if all_windows_flat:
        total_trades = sum(r["trades"] for r in all_windows_flat)
        total_pfs = [r["pf"] for r in all_windows_flat]
        total_wrs = [r["wr"] for r in all_windows_flat]
        total_dds = [r["dd"] for r in all_windows_flat]
        n_pass_total = sum(1 for pf in total_pfs if pf >= 1.0)
        print(f"{'TOTAL':<12} {len(all_windows_flat):>8} {n_pass_total}/{len(all_windows_flat):>5} {np.mean(total_pfs):>7.2f} {np.mean(total_wrs):>6.1f}% {np.mean(total_dds):>6.1f}%")

    # Final verdict
    print(f"\n{'=' * 110}")
    if portfolio_pass:
        print("  VERDICT: STRATEGY IS ROBUST -- PF > 1.0 in ALL walk-forward windows for ALL symbols.")
    else:
        fragile = [s for s in sorted(DRAGON_SYMBOLS.keys())
                   if not all(r["pf"] >= 1.0 for r in all_results.get(s, [{}]))]
        print(f"  VERDICT: STRATEGY HAS WEAK SPOTS -- not all windows profitable.")
        print(f"  Fragile symbols: {', '.join(fragile)}")
        for sym in fragile:
            results = all_results.get(sym, [])
            failing = [r for r in results if r["pf"] < 1.0]
            for r in failing:
                print(f"    {sym} W{r['window']:02d} {r['test_period']}: PF={r['pf']:.2f} WR={r['wr']:.1f}% DD={r['dd']:.1f}%")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
