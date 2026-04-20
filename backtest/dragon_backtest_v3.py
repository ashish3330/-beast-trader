"""
DRAGON BACKTEST V3 — World-Class Multi-Strategy AI Agent.

ARCHITECTURE:
- Strategy 1: MOMENTUM (proven, from v2 with industry gates)
- Strategy 2: MEAN-REVERSION (new, fires when momentum is flat)
- FVG Integration: Entry boost, SL/TP targets from fair value gaps
- ML Simulation: Deterministic quality scoring (replaces random filter)
- Compound Sizing: Risk as % of CURRENT equity, Sharpe-adjusted
- Industry Gates: Soft quality scoring (from v2, per-symbol config)

TARGET: 5-8 trades/day (vs current 3.7) for exponential compound growth.
- Momentum alone: ~3.7/day (as before)
- Mean-reversion adds: ~2-4/day in ranging/low-vol periods
- Combined: ~5-8/day → 25-35%/month compound growth

SYMBOLS: 7 (removed EURJPY PF 0.71, USDCHF PF 0.60)
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.momentum_scorer import (
    _compute_indicators, _score, _ema, _atr,
    IND_DEFAULTS, IND_OVERRIDES, REGIME_PARAMS, DEFAULT_PARAMS
)

# M15 indicator parameters: ~3x H1 periods to cover same time horizon
# H1 EMA(15) = 15 hours ≈ M15 EMA(45) = 11.25 hours (close enough)
IND_M15_DEFAULTS = {
    "EMA_S": 45, "EMA_L": 120, "EMA_T": 240,
    "ST_F": 2.5, "ST_ATR": 30,
    "MACD_F": 24, "MACD_SL": 63, "MACD_SIG": 21,
    "ATR_LEN": 42,
}
# Per-symbol M15 overrides (scaled from H1 IND_OVERRIDES)
IND_M15_OVERRIDES = {
    "XAUUSD":   {"EMA_S": 45, "EMA_L": 90, "EMA_T": 180, "ST_F": 2.0, "ST_ATR": 21,
                 "MACD_F": 15, "MACD_SL": 78, "MACD_SIG": 12, "ATR_LEN": 21},
    "NAS100.r": {"EMA_S": 60, "EMA_L": 120, "EMA_T": 240, "ST_F": 3.0, "ST_ATR": 30,
                 "MACD_F": 15, "MACD_SL": 63, "MACD_SIG": 21, "ATR_LEN": 30},
    "XAGUSD":   {"EMA_S": 24, "EMA_L": 120, "EMA_T": 240, "ST_F": 3.5, "ST_ATR": 42,
                 "MACD_F": 36, "MACD_SL": 63, "MACD_SIG": 27, "ATR_LEN": 42},
    "BTCUSD":   {"EMA_S": 60, "EMA_L": 150, "EMA_T": 180, "ST_F": 2.5, "ST_ATR": 30,
                 "MACD_F": 15, "MACD_SL": 78, "MACD_SIG": 27, "ATR_LEN": 30},
    "JPN225ft": {"EMA_S": 45, "EMA_L": 120, "EMA_T": 240, "ST_F": 2.5, "ST_ATR": 30,
                 "MACD_F": 24, "MACD_SL": 63, "MACD_SIG": 21, "ATR_LEN": 30},
    "USDJPY":   {"EMA_S": 24, "EMA_L": 120, "EMA_T": 240, "ST_F": 2.5, "ST_ATR": 30,
                 "MACD_F": 15, "MACD_SL": 63, "MACD_SIG": 21, "ATR_LEN": 30},
    "USDCAD":   {"EMA_S": 45, "EMA_L": 120, "EMA_T": 240, "ST_F": 3.0, "ST_ATR": 30,
                 "MACD_F": 24, "MACD_SL": 63, "MACD_SIG": 21, "ATR_LEN": 30},
}
from backtest.fvg_vectorized import detect_fvg
from backtest.engine import mean_reversion_score

from config import (RL_ENABLED_SYMBOLS, RL_SYMBOL_PARAMS, SYMBOL_ATR_SL_OVERRIDE,
                    DRAGON_SYMBOL_MIN_SCORE as SYMBOL_MIN_SCORE_OVERRIDE,
                    SYMBOL_TRAIL_OVERRIDE, TRAIL_STEPS, SYMBOL_SESSION_OVERRIDE)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
START_EQ = 1000.0
SLIP = 0.0

# ═══ COMPOUND SIZING PARAMS ═══
BASE_RISK_PCT = 0.012           # 1.2% base risk per trade
MIN_RISK_PCT = 0.006            # 0.6% floor (drawdown protection)
MAX_RISK_PCT = 0.020            # 2.0% ceiling (hot streak boost)
SHARPE_LOOKBACK = 20            # trades for rolling Sharpe
DAILY_LOSS_LIMIT = 0.02         # 2% daily loss → stop
CONSEC_LOSS_COOLDOWN = 12       # 12 bars cooldown after 3 losses (was 24)

# ═══ MEAN REVERSION PARAMS ═══
MR_MIN_SCORE = 3.0              # H1-calibrated: BB touch(1) + RSI near extreme(1) + EMA dist(1) = 3
MR_SL_MULT = 1.0               # Tighter SL for MR (1.0 ATR vs 1.5-2.5 for momentum)
MR_TP_MULT = 1.5               # Take profit at 1.5R (reversion targets are closer)
MR_RISK_DISCOUNT = 0.7         # MR trades risk 70% of momentum trades (less conviction)

# ═══ FVG PARAMS ═══
FVG_ENTRY_BOOST = 1.3           # 30% risk boost when entering at FVG
FVG_SL_PROTECTION = True        # Use FVG boundary as SL support
FVG_TP_TARGET = True            # Use opposite FVG as TP target

USE_RL = True
RL_REDUCE_MIN = 0.6

# ═══ 7 SYMBOLS — H1 data (for baseline comparison) ═══
ALL_SYMBOLS = {
    "XAUUSD":    {"cache": "raw_h1_xauusd.pkl",   "point": 0.01,    "tv": 1.0,     "spread": 0.33,   "cat": "Gold"},
    "XAGUSD":    {"cache": "raw_h1_XAGUSD.pkl",   "point": 0.001,   "tv": 5.0,     "spread": 0.035,  "cat": "Gold"},
    "BTCUSD":    {"cache": "raw_h1_BTCUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 17.0,   "cat": "Crypto"},
    "NAS100.r":  {"cache": "raw_h1_NAS100_r.pkl", "point": 0.01,    "tv": 0.01,    "spread": 1.80,   "cat": "Index"},
    "JPN225ft":  {"cache": "raw_h1_JPN225ft.pkl", "point": 0.01,    "tv": 0.0063,  "spread": 10.0,   "cat": "Index"},
    "USDJPY":    {"cache": "raw_h1_USDJPY.pkl",   "point": 0.001,   "tv": 0.63,    "spread": 0.018,  "cat": "Forex"},
    "USDCAD":    {"cache": "raw_h1_USDCAD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015,"cat": "Forex"},
}

# ═══ 7 SYMBOLS — M15 data (primary TF for live) ═══
M15_SYMBOLS = {
    "XAUUSD":    {"cache": "raw_m15_xauusd.pkl",   "point": 0.01,    "tv": 1.0,     "spread": 0.33,   "cat": "Gold"},
    "XAGUSD":    {"cache": "raw_m15_XAGUSD.pkl",   "point": 0.001,   "tv": 5.0,     "spread": 0.035,  "cat": "Gold"},
    "BTCUSD":    {"cache": "raw_m15_BTCUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 17.0,   "cat": "Crypto"},
    "NAS100.r":  {"cache": "raw_m15_NAS100_r.pkl", "point": 0.01,    "tv": 0.01,    "spread": 1.80,   "cat": "Index"},
    "JPN225ft":  {"cache": "raw_m15_JPN225ft.pkl", "point": 0.01,    "tv": 0.0063,  "spread": 10.0,   "cat": "Index"},
    "USDJPY":    {"cache": "raw_m15_USDJPY.pkl",   "point": 0.001,   "tv": 0.63,    "spread": 0.018,  "cat": "Forex"},
    "USDCAD":    {"cache": "raw_m15_USDCAD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015,"cat": "Forex"},
}

# Per-symbol industry gates (from v2 backtest validation)
SYMBOL_GATES_ENABLED = {
    "XAUUSD":   True,   # F→A with gates
    "XAGUSD":   False,  # Already A+
    "BTCUSD":   False,  # Already A+
    "NAS100.r": False,  # Gates hurt via cascade
    "JPN225ft": True,   # +19% PF with gates
    "USDJPY":   True,   # Lower DD with gates
    "USDCAD":   False,  # Gates slightly hurt
}

# Per-symbol mean-reversion enable (only where ranging periods exist)
SYMBOL_MR_ENABLED = {
    "XAUUSD":   True,   # Gold ranges significantly
    "XAGUSD":   True,   # Silver ranges
    "BTCUSD":   False,  # BTC trends, rarely mean-reverts cleanly
    "NAS100.r": True,   # Indices have range days
    "JPN225ft": True,   # Asian session ranges
    "USDJPY":   True,   # Forex ranges a lot
    "USDCAD":   True,   # Forex ranges a lot
}


# ═══════════════════════════════════════════════════════════════════════════════
# INDUSTRY GATES (from v2 — soft quality scoring)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_industry_extras(ind, close, high, low):
    """Compute all industry-grade indicators for gate evaluation."""
    n = len(close)

    # Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(10, n):
        direction = abs(close[i] - close[i - 10])
        volatility = sum(abs(close[j] - close[j-1]) for j in range(i - 9, i + 1))
        er[i] = direction / volatility if volatility > 0 else 0

    # Choppiness Index
    chop = np.full(n, np.nan)
    atr1 = np.zeros(n)
    for i in range(1, n):
        atr1[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr1[0] = high[0] - low[0]
    for i in range(14, n):
        atr_sum = np.sum(atr1[i - 13:i + 1])
        hi = np.max(high[i - 13:i + 1])
        lo = np.min(low[i - 13:i + 1])
        hl_range = hi - lo
        if hl_range > 0:
            chop[i] = 100 * np.log10(atr_sum / hl_range) / np.log10(14)

    # Z-score
    zscore = np.full(n, np.nan)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] if close[i-1] > 0 else 0
    for i in range(100, n):
        win = returns[i - 99:i + 1]
        mu = np.mean(win); sigma = np.std(win)
        if sigma > 0: zscore[i] = (returns[i] - mu) / sigma

    # Dual ROC
    roc_fast = np.full(n, np.nan)
    roc_slow = np.full(n, np.nan)
    for i in range(12, n):
        if close[i-12] > 0: roc_fast[i] = (close[i] - close[i-12]) / close[i-12] * 100
    for i in range(21, n):
        if close[i-21] > 0: roc_slow[i] = (close[i] - close[i-21]) / close[i-21] * 100

    # Elder Impulse
    ema13 = _ema(close, 13)
    from signals.momentum_scorer import _macd
    _, _, macd_hist = _macd(close, 12, 26, 9)
    impulse = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if ema13[i] > ema13[i-1] and macd_hist[i] > macd_hist[i-1]: impulse[i] = 1
        elif ema13[i] < ema13[i-1] and macd_hist[i] < macd_hist[i-1]: impulse[i] = -1

    # Regime composite
    regime_comp = np.zeros(n)
    for i in range(1, n):
        s = 0
        if not np.isnan(ind["adx"][i]) and ind["adx"][i] > 25: s += 1
        if not np.isnan(chop[i]) and chop[i] < 50: s += 1
        if not np.isnan(er[i]) and er[i] > 0.4: s += 1
        if not np.isnan(ind["bbw"][i]) and 1.5 <= ind["bbw"][i] < 4.0: s += 1
        regime_comp[i] = s

    return {
        "er": er, "zscore": zscore, "roc_fast": roc_fast, "roc_slow": roc_slow,
        "impulse": impulse, "regime_comp": regime_comp, "chop": chop,
    }


def evaluate_quality(i, direction, score, ind, extras, symbol):
    """Soft quality gate. Returns quality_ratio 0.0-1.0."""
    er = extras["er"]
    zscore = extras["zscore"]
    roc_fast = extras["roc_fast"]
    roc_slow = extras["roc_slow"]
    impulse = extras["impulse"]
    regime_comp = extras["regime_comp"]

    cat = ALL_SYMBOLS.get(symbol, {}).get("cat", "Forex")
    is_trending = cat in ("Crypto", "Index", "Gold")

    # Hard gate: Elder 2+ consecutive opposing
    if i > 1 and score < 9.0:
        if direction == 1 and impulse[i] == -1 and impulse[i-1] == -1:
            return 0.0
        if direction == -1 and impulse[i] == 1 and impulse[i-1] == 1:
            return 0.0

    quality = 0.0; mx = 0.0

    # ER (0-2)
    mx += 2.0
    if not np.isnan(er[i]):
        if er[i] >= 0.5: quality += 2.0
        elif er[i] >= 0.3: quality += 1.5
        elif er[i] >= 0.15: quality += 1.0
        elif er[i] >= 0.08: quality += 0.5

    # Z-score (0-2)
    mx += 2.0
    if not np.isnan(zscore[i]):
        abs_z = abs(zscore[i])
        z_chasing = (direction == 1 and zscore[i] > 0) or (direction == -1 and zscore[i] < 0)
        if z_chasing:
            if is_trending:
                if abs_z < 2.0: quality += 2.0
                elif abs_z < 2.5: quality += 1.5
                elif abs_z < 3.0: quality += 1.0
                else: quality += 0.5
            else:
                if abs_z < 1.5: quality += 2.0
                elif abs_z < 2.0: quality += 1.5
                elif abs_z < 2.5: quality += 0.75
        else:
            quality += 2.0

    # Dual ROC (0-2)
    mx += 2.0
    if not np.isnan(roc_fast[i]) and not np.isnan(roc_slow[i]):
        confirms = 0
        if direction == 1:
            if roc_fast[i] > 0: confirms += 1
            if roc_slow[i] > 0: confirms += 1
        else:
            if roc_fast[i] < 0: confirms += 1
            if roc_slow[i] < 0: confirms += 1
        quality += confirms

    # Fresh momentum (0-2)
    mx += 2.0
    rsi = ind["rs"][i] if not np.isnan(ind["rs"][i]) else 50
    if i > 1:
        if direction == 1 and ind["mh"][i] > ind["mh"][i-1]: quality += 1.0
        if direction == -1 and ind["mh"][i] < ind["mh"][i-1]: quality += 1.0
    if direction == 1 and 35 < rsi < 75: quality += 0.5
    if direction == -1 and 25 < rsi < 65: quality += 0.5
    if not np.isnan(ind["vol_sma"][i]) and ind["vol_sma"][i] > 0:
        if ind["vol"][i] > 1.3 * ind["vol_sma"][i]: quality += 0.5

    # Regime composite (0-2)
    mx += 2.0
    quality += min(2.0, regime_comp[i] * 0.5)

    ratio = quality / mx if mx > 0 else 0
    return ratio


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_regime(ind, bi, chop_arr):
    if bi < 21 or np.isnan(ind["bbw"][bi]): return "unknown"
    bbw = float(ind["bbw"][bi])
    adx = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 0
    chop_val = float(chop_arr[bi]) if not np.isnan(chop_arr[bi]) else 50
    if bbw < 1.5 and adx < 20 and chop_val > 55: return "ranging"
    if 1.5 <= bbw < 3.0 and adx > 25 and chop_val < 50: return "trending"
    if bbw >= 3.0: return "volatile"
    if chop_val < 38.2 and adx > 20: return "trending"
    if chop_val > 61.8: return "ranging"
    return "low_vol"


def get_adaptive_min_score(regime, symbol=None):
    if symbol and symbol in SYMBOL_MIN_SCORE_OVERRIDE:
        sym_scores = SYMBOL_MIN_SCORE_OVERRIDE[symbol]
        if regime in sym_scores: return sym_scores[regime]
    return {"trending": 6.0, "ranging": 8.0, "volatile": 7.0, "low_vol": 7.0}.get(regime, 7.0)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOUND SIZING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_risk_pct(r_multiples, base=BASE_RISK_PCT):
    """Sharpe-adjusted compound risk sizing."""
    if len(r_multiples) < SHARPE_LOOKBACK:
        return base
    recent = r_multiples[-SHARPE_LOOKBACK:]
    avg_r = np.mean(recent)
    std_r = np.std(recent)
    if std_r == 0: return base
    sharpe = avg_r / std_r

    if sharpe > 2.0:
        risk = base * 1.5
    elif sharpe > 1.0:
        risk = base * 1.2
    elif sharpe > 0.5:
        risk = base
    elif sharpe > 0:
        risk = base * 0.8
    else:
        risk = base * 0.5  # negative sharpe = protect capital

    return max(MIN_RISK_PCT, min(MAX_RISK_PCT, risk))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def get_m15_adaptive_min_score(regime, symbol=None):
    """M15-calibrated thresholds: slightly lower than H1 but not reckless."""
    # Raised from 5.5 to 6.5 after M15 backtest showed 33% WR at 5.5
    m15_scores = {
        "XAUUSD":   {"trending": 6.5, "ranging": 7.0, "volatile": 6.5, "low_vol": 6.5},
        "XAGUSD":   {"trending": 6.5, "ranging": 7.0, "volatile": 7.0, "low_vol": 7.0},
        "BTCUSD":   {"trending": 6.0, "ranging": 6.5, "volatile": 6.0, "low_vol": 6.0},
        "NAS100.r": {"trending": 6.5, "ranging": 6.5, "volatile": 6.5, "low_vol": 6.5},
        "JPN225ft": {"trending": 6.5, "ranging": 6.5, "volatile": 7.0, "low_vol": 7.0},
        "USDJPY":   {"trending": 6.5, "ranging": 7.0, "volatile": 7.0, "low_vol": 7.0},
        "USDCAD":   {"trending": 6.5, "ranging": 6.5, "volatile": 6.5, "low_vol": 6.5},
    }
    if symbol and symbol in m15_scores:
        sym_scores = m15_scores[symbol]
        if regime in sym_scores: return sym_scores[regime]
    return {"trending": 6.5, "ranging": 7.0, "volatile": 6.5, "low_vol": 6.5}.get(regime, 6.5)


def run(symbol, days=365, use_m15=False):
    scfg = (M15_SYMBOLS if use_m15 else ALL_SYMBOLS).get(symbol)
    if not scfg: scfg = ALL_SYMBOLS[symbol]
    cache_path = CACHE / scfg["cache"]
    if not cache_path.exists(): return None
    df = pickle.load(open(cache_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    pt = scfg["point"]; tv = scfg["tv"]; spread = scfg["spread"]
    cat = scfg["cat"]; sl_cap = 5000 * pt
    if use_m15:
        icfg = dict(IND_M15_DEFAULTS); icfg.update(IND_M15_OVERRIDES.get(symbol, {}))
    else:
        icfg = dict(IND_DEFAULTS); icfg.update(IND_OVERRIDES.get(symbol, {}))
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    start_idx = max(int(df[df["time"] >= cutoff].index[0]), icfg["EMA_T"] + 30)
    ind = _compute_indicators(df, icfg)
    n = ind["n"]
    close = ind["c"]; high = ind["h"]; low = ind["l"]; open_arr = ind["o"]

    # Compute extras
    extras = compute_industry_extras(ind, close, high, low)
    use_gates = SYMBOL_GATES_ENABLED.get(symbol, False)
    use_mr = SYMBOL_MR_ENABLED.get(symbol, False)

    # Compute FVG
    fvg = detect_fvg(open_arr, high, low, close)

    # State
    eq = START_EQ; peak = START_EQ; max_dd = 0
    n_trades = 0; wins = 0; gross_p = 0; gross_l = 0
    in_trade = False; d = 0; entry = 0; pos_sl = 0; sl_dist = 0
    trade_lot = 0.0; trade_strategy = "momentum"
    consec_losses = 0; cooldown_until = 0
    daily_pnl = 0.0; current_day = None; day_eq_start = START_EQ; day_stopped = False
    r_multiples = []; max_consec_loss = 0; current_streak = 0

    # RL state
    rl_trades = []; rl_regime_wr = {}; rl_hour_wr = {}
    entry_regime = "unknown"; entry_hour = 12; entry_score = 0.0

    # Strategy tracking
    mom_trades = 0; mr_trades = 0; mom_wins = 0; mr_wins = 0
    fvg_boost_count = 0

    for i in range(start_idx, n):
        atr_val = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0
        if atr_val == 0: continue

        bar_time = df["time"].iloc[i]
        bar_hour = bar_time.hour if hasattr(bar_time, "hour") else 12
        sess_start, sess_end = SYMBOL_SESSION_OVERRIDE.get(symbol, (6, 22))
        if cat != "Crypto" and (bar_hour >= sess_end or bar_hour < sess_start): continue

        bar_date = bar_time.date() if hasattr(bar_time, "date") else None
        if bar_date and bar_date != current_day:
            current_day = bar_date; day_eq_start = eq; daily_pnl = 0.0; day_stopped = False

        # ─── MANAGE OPEN POSITION ───
        if in_trade:
            hit_sl = (d == 1 and ind["l"][i] <= pos_sl) or (d == -1 and ind["h"][i] >= pos_sl)
            if hit_sl:
                exit_cost = spread + SLIP * pt
                pnl = d * (pos_sl - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
                eq += pnl; daily_pnl += pnl
                r_val = pnl / (BASE_RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
                r_multiples.append(r_val)

                # RL recording
                rl_trades.append({"pnl": pnl, "regime": entry_regime, "hour": entry_hour,
                                  "dir": d, "score": entry_score, "won": pnl > 0,
                                  "strategy": trade_strategy})
                if len(rl_trades) > 100: rl_trades = rl_trades[-100:]

                sym_rl = RL_SYMBOL_PARAMS.get(symbol, {}); rl_lookback = sym_rl.get("lookback", 20)
                if len(rl_trades) >= rl_lookback:
                    recent_rl = rl_trades[-rl_lookback:]
                    for r in set(t["regime"] for t in recent_rl):
                        rr = [t for t in recent_rl if t["regime"] == r]
                        if len(rr) >= 3: rl_regime_wr[r] = sum(1 for t in rr if t["won"]) / len(rr)
                    for h in set(t["hour"] for t in recent_rl):
                        hh = [t for t in recent_rl if t["hour"] == h]
                        if len(hh) >= 3: rl_hour_wr[h] = sum(1 for t in hh if t["won"]) / len(hh)

                if pnl > 0:
                    gross_p += pnl; wins += 1; consec_losses = 0; current_streak = 0
                    if trade_strategy == "momentum": mom_wins += 1
                    else: mr_wins += 1
                else:
                    gross_l += abs(pnl); consec_losses += 1; current_streak += 1
                    max_consec_loss = max(max_consec_loss, current_streak)
                    if consec_losses >= 3:
                        cooldown_until = i + CONSEC_LOSS_COOLDOWN; consec_losses = 0

                n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
                in_trade = False
                if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                    day_stopped = True
                continue

            # Trailing SL
            cur = float(ind["c"][i])
            profit_r = ((cur - entry) * d) / sl_dist if sl_dist > 0 else 0

            if trade_strategy == "mean_reversion":
                # MR uses simpler trailing: lock at 0.8R, trail at 1.5R
                new_sl = None
                if profit_r >= 1.5:
                    new_sl = cur - 0.5 * atr_val * d
                elif profit_r >= 0.8:
                    new_sl = entry + 0.3 * sl_dist * d
                elif profit_r >= 0.4:
                    new_sl = entry + 2 * pt * d  # breakeven
            else:
                # Momentum: use per-symbol trail profile
                trail = SYMBOL_TRAIL_OVERRIDE.get(symbol, TRAIL_STEPS)
                new_sl = None
                for th, ac, pa in trail:
                    if profit_r >= th:
                        if ac == "trail": new_sl = cur - pa * atr_val * d
                        elif ac == "lock": new_sl = entry + pa * sl_dist * d
                        elif ac == "be": new_sl = entry + 2 * pt * d
                        break

            if new_sl is not None:
                if d == 1 and new_sl > pos_sl: pos_sl = new_sl
                elif d == -1 and new_sl < pos_sl: pos_sl = new_sl

        if day_stopped or i < cooldown_until: continue

        # ─── SIGNAL EVALUATION ───
        bi = i - 1
        if bi < 21: continue

        regime = get_regime(ind, bi, extras["chop"])
        signal_found = False
        new_dir = 0
        best_score = 0
        selected_strategy = "momentum"

        # === STRATEGY 1: MOMENTUM ===
        ls, ss = _score(ind, bi)
        adaptive_min = get_m15_adaptive_min_score(regime, symbol=symbol) if use_m15 else get_adaptive_min_score(regime, symbol=symbol)
        mom_buy = ls >= adaptive_min
        mom_sell = ss >= adaptive_min

        if mom_buy or mom_sell:
            new_dir = 1 if (mom_buy and (not mom_sell or ls >= ss)) else -1
            best_score = max(ls, ss)
            selected_strategy = "momentum"
            signal_found = True

            # Industry gates (if enabled for this symbol)
            if use_gates:
                quality = evaluate_quality(bi, new_dir, best_score, ind, extras, symbol)
                threshold = 0.50 if best_score >= 9.0 else 0.55 if best_score >= 8.0 else 0.60 if best_score >= 7.5 else 0.65
                if quality < threshold:
                    signal_found = False

        # === STRATEGY 2: MEAN REVERSION (only if momentum didn't fire) ===
        if not signal_found and use_mr and not in_trade:
            mr_ls, mr_ss = mean_reversion_score(ind, bi)
            if mr_ls >= MR_MIN_SCORE or mr_ss >= MR_MIN_SCORE:
                new_dir = 1 if (mr_ls >= MR_MIN_SCORE and (mr_ss < MR_MIN_SCORE or mr_ls >= mr_ss)) else -1
                best_score = max(mr_ls, mr_ss)
                selected_strategy = "mean_reversion"
                signal_found = True

                # MR-specific gate: don't mean-revert in strong trend
                if not np.isnan(ind["adx"][bi]) and ind["adx"][bi] > 35:
                    signal_found = False

                # MR-specific gate: FVG must support reversion direction
                if signal_found and FVG_SL_PROTECTION:
                    if new_dir == 1 and fvg["inside_bear_fvg"][bi]:
                        signal_found = False  # Don't go long inside bearish FVG
                    if new_dir == -1 and fvg["inside_bull_fvg"][bi]:
                        signal_found = False  # Don't go short inside bullish FVG

        if not signal_found: continue

        # ─── FVG ENHANCEMENT ───
        fvg_boost = 1.0
        fvg_sl_level = None
        fvg_tp_level = None

        if new_dir == 1:
            # Long: bullish FVG below = support for SL, bearish FVG above = TP target
            if fvg["inside_bull_fvg"][bi]:
                fvg_boost = FVG_ENTRY_BOOST  # Entering inside a bullish gap = strong
                fvg_boost_count += 1
            if FVG_TP_TARGET and fvg["dist_nearest_bear"][bi] > 0:
                fvg_tp_level = float(ind["c"][bi]) + fvg["dist_nearest_bear"][bi]
        else:
            if fvg["inside_bear_fvg"][bi]:
                fvg_boost = FVG_ENTRY_BOOST
                fvg_boost_count += 1
            if FVG_TP_TARGET and fvg["dist_nearest_bull"][bi] > 0:
                fvg_tp_level = float(ind["c"][bi]) - fvg["dist_nearest_bull"][bi]

        # ─── REVERSAL HANDLING ───
        if in_trade and new_dir != d:
            exit_cost = spread + SLIP * pt
            pnl = d * (float(ind["c"][i]) - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
            eq += pnl; daily_pnl += pnl
            r_val = pnl / (BASE_RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
            r_multiples.append(r_val)
            rl_trades.append({"pnl": pnl, "regime": entry_regime, "hour": entry_hour,
                              "dir": d, "score": entry_score, "won": pnl > 0,
                              "strategy": trade_strategy})
            if pnl > 0:
                gross_p += pnl; wins += 1; consec_losses = 0; current_streak = 0
            else:
                gross_l += abs(pnl); consec_losses += 1; current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
                if consec_losses >= 3:
                    cooldown_until = i + CONSEC_LOSS_COOLDOWN; consec_losses = 0
            n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
            in_trade = False
            if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                day_stopped = True; continue

        # ─── ENTRY ───
        if not in_trade:
            d = new_dir
            entry_regime = regime; entry_hour = bar_hour; entry_score = best_score
            trade_strategy = selected_strategy

            # SL distance
            if selected_strategy == "mean_reversion":
                sl_dist = atr_val * MR_SL_MULT
            else:
                sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
                sym_sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, 1.5)
                sl_dist = max(atr_val * sl_m, atr_val * sym_sl_mult)
            sl_dist = min(sl_dist, sl_cap)

            # Compound sizing
            risk_pct = compute_risk_pct(r_multiples)
            if selected_strategy == "mean_reversion":
                risk_pct *= MR_RISK_DISCOUNT  # MR gets less risk

            # FVG boost
            risk_pct *= fvg_boost

            # Cap risk
            risk_pct = min(risk_pct, MAX_RISK_PCT)
            risk_amount = eq * risk_pct

            # RL intelligence
            sym_rl = RL_SYMBOL_PARAMS.get(symbol, {})
            rl_lookback = sym_rl.get("lookback", 20)
            rl_boost_max = sym_rl.get("boost_max", 1.4)
            if USE_RL and symbol in RL_ENABLED_SYMBOLS and len(rl_trades) >= rl_lookback:
                r_wr = rl_regime_wr.get(regime, 0.5)
                if r_wr < 0.25 and len([t for t in rl_trades if t["regime"] == regime]) >= 5:
                    continue
                h_wr = rl_hour_wr.get(bar_hour, 0.5)
                if h_wr < 0.20 and len([t for t in rl_trades if t["hour"] == bar_hour]) >= 5:
                    continue
                rd_trades = [t for t in rl_trades[-40:] if t["regime"] == regime and t["dir"] == d]
                if len(rd_trades) >= 5:
                    rd_wr = sum(1 for t in rd_trades if t["won"]) / len(rd_trades)
                    if rd_wr < 0.20: continue

                rl_mult = 1.0
                if r_wr > 0.55: rl_mult *= 1.0 + (r_wr - 0.5) * 1.0
                elif r_wr < 0.40: rl_mult *= max(0.5, 1.0 - (0.5 - r_wr) * 1.0)
                if h_wr > 0.55: rl_mult *= 1.0 + (h_wr - 0.5) * 0.8
                elif h_wr < 0.40: rl_mult *= max(0.6, 1.0 - (0.5 - h_wr) * 0.6)
                recent_rl = rl_trades[-rl_lookback:]
                gp = sum(t["pnl"] for t in recent_rl if t["pnl"] > 0)
                gl = sum(abs(t["pnl"]) for t in recent_rl if t["pnl"] < 0) or 0.01
                rpf = gp / gl
                if rpf < 0.7: rl_mult *= 0.5
                elif rpf > 2.5: rl_mult *= 1.2
                rl_mult = max(RL_REDUCE_MIN, min(rl_boost_max, rl_mult))
                risk_amount *= rl_mult

            pip_value_per_lot = (sl_dist / pt) * tv
            if pip_value_per_lot > 0:
                trade_lot = risk_amount / pip_value_per_lot
                trade_lot = max(trade_lot, 0.01)
            else:
                trade_lot = 0.01

            entry_cost = spread + SLIP * pt
            entry = float(ind["o"][i]) + entry_cost / 2 * d
            pos_sl = entry - sl_dist * d
            in_trade = True

            if selected_strategy == "momentum": mom_trades += 1
            else: mr_trades += 1

    # Close open trade at end
    if in_trade:
        pnl = d * (float(ind["c"][n-1]) - entry) / pt * tv * trade_lot
        eq += pnl
        r_val = pnl / (BASE_RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
        r_multiples.append(r_val)
        if pnl > 0: gross_p += pnl; wins += 1
        else: gross_l += abs(pnl)
        n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)

    pf = gross_p / gross_l if gross_l > 0 else (999 if gross_p > 0 else 0)
    dd = max_dd / peak * 100 if peak else 0
    ret = (eq - START_EQ) / START_EQ * 100
    wr = wins / n_trades * 100 if n_trades else 0
    avg_r = np.mean(r_multiples) if r_multiples else 0
    std_r = np.std(r_multiples) if len(r_multiples) > 1 else 1
    sharpe = (avg_r / std_r) * np.sqrt(252) if std_r > 0 else 0
    trades_per_day = n_trades / days if days > 0 else 0

    return {
        "sym": symbol, "trades": n_trades, "wr": round(wr, 1), "pf": round(pf, 2),
        "ret": round(ret, 1), "dd": round(dd, 1), "eq": round(eq, 2),
        "sharpe": round(sharpe, 2), "trades_day": round(trades_per_day, 2),
        "mom_trades": mom_trades, "mr_trades": mr_trades,
        "mom_wr": round(mom_wins/mom_trades*100, 1) if mom_trades > 0 else 0,
        "mr_wr": round(mr_wins/mr_trades*100, 1) if mr_trades > 0 else 0,
        "fvg_boosts": fvg_boost_count, "max_consec_loss": max_consec_loss,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--m15", action="store_true", help="Use M15 data (primary TF)")
    args = parser.parse_args()

    sym_dict = M15_SYMBOLS if args.m15 else ALL_SYMBOLS
    symbols = [args.symbol] if args.symbol else sorted(sym_dict.keys())
    tf_label = "M15" if args.m15 else "H1"

    print("=" * 140)
    print(f"  DRAGON V3 — Multi-Strategy AI Agent | {tf_label} Primary | Momentum + MR + FVG + Gates + Compound")
    print(f"  7 Symbols | {args.days}d | Target: {'8-15' if args.m15 else '5-8'} trades/day for exponential compound growth")
    print("=" * 140)
    print(f"\n{'Symbol':<12} {'Trades':>7} {'T/Day':>6} {'WR%':>6} {'PF':>6} {'Ret%':>9} {'DD%':>6} {'Final$':>10} {'Sharpe':>7} {'Mom':>5} {'MR':>4} {'MomWR':>6} {'MR_WR':>6} {'FVG':>4} {'Grade':>6}")
    print("-" * 140)

    results = []
    for sym in symbols:
        if sym not in sym_dict:
            print(f"  {sym}: not configured"); continue
        r = run(sym, args.days, use_m15=args.m15)
        if r:
            results.append(r)
            grade = "A+" if r["pf"] >= 2.0 else "A" if r["pf"] >= 1.5 else "B" if r["pf"] >= 1.2 else "C" if r["pf"] >= 1.0 else "F"
            print(f"{r['sym']:<12} {r['trades']:>7} {r['trades_day']:>6.2f} {r['wr']:>5.1f}% {r['pf']:>6.2f} {r['ret']:>8.1f}% {r['dd']:>5.1f}% ${r['eq']:>9.2f} {r['sharpe']:>7.2f} {r['mom_trades']:>5} {r['mr_trades']:>4} {r['mom_wr']:>5.1f}% {r['mr_wr']:>5.1f}% {r['fvg_boosts']:>4} {grade:>6}")

    print("-" * 140)
    if results:
        total_trades = sum(r["trades"] for r in results)
        total_per_day = sum(r["trades_day"] for r in results)
        total_mom = sum(r["mom_trades"] for r in results)
        total_mr = sum(r["mr_trades"] for r in results)
        gp = sum(r.get("eq", START_EQ) - START_EQ for r in results if r.get("eq", START_EQ) > START_EQ)
        gl = sum(START_EQ - r.get("eq", START_EQ) for r in results if r.get("eq", START_EQ) < START_EQ)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_dd = np.mean([r["dd"] for r in results])
        portfolio_pf = (gp + gl + gp) / (gl + 0.01) if gl > 0 else 999

        print(f"{'PORTFOLIO':<12} {total_trades:>7} {total_per_day:>6.2f} {'':>6} {'':>6} {'':>9} {avg_dd:>5.1f}% {'':>10} {avg_sharpe:>7.2f} {total_mom:>5} {total_mr:>4}")
        print(f"\n  Total trades/day: {total_per_day:.1f} | Momentum: {total_mom} | Mean-Reversion: {total_mr}")
        print(f"  FVG boosts applied: {sum(r['fvg_boosts'] for r in results)}")

        profitable = [r for r in results if r["pf"] >= 1.2]
        losing = [r for r in results if r["pf"] < 1.0]
        print(f"  Profitable (PF>=1.2): {len(profitable)}/7 | Losing (PF<1.0): {len(losing)}/7")

        # Monthly compound projection
        if total_per_day > 0 and len(results) > 0:
            avg_wr = np.mean([r["wr"] for r in results if r["trades"] > 10]) / 100
            # Approximate monthly compound: (1 + expectancy * risk)^trades_per_month
            expectancy = avg_wr * 1.5 - (1 - avg_wr) * 1.0  # assuming 1.5:1 RR
            monthly_trades = total_per_day * 22
            monthly_return = (1 + expectancy * BASE_RISK_PCT) ** monthly_trades - 1
            print(f"\n  COMPOUND PROJECTION (22 trading days/month):")
            print(f"  {total_per_day:.1f} trades/day × 22 days = {monthly_trades:.0f} trades/month")
            print(f"  Avg WR: {avg_wr*100:.1f}% | Expectancy: {expectancy:.3f}R per trade")
            print(f"  Monthly compound return: {monthly_return*100:.1f}%")
            eq_proj = START_EQ
            for m in range(1, 7):
                eq_proj *= (1 + monthly_return)
                print(f"    Month {m}: ${eq_proj:.0f}")

    print("=" * 140)
