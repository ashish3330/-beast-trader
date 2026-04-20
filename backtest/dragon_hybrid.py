"""
DRAGON HYBRID BACKTEST — Best of institutional + Dragon per-symbol tuning.

Key insight from comparison:
- Institutional (vol-scaled, conviction sizing) is BETTER for: Crypto, Index, Gold
  BTCUSD: +262K% vs +140K%, JPN225ft: +154% vs +7%, XAUUSD: +105% vs +55%
- Dragon (tighter MIN_SCORE, fixed risk) is BETTER for: Forex
  XAGUSD: PF 2.50 vs 1.44, USDCHF: PF 1.34 vs 1.00

HYBRID approach:
1. Conviction sizing for ALL — scales position by score excess (institutional win)
2. Vol-scaled threshold for Crypto/Index/Gold (institutional win)
3. Per-symbol tuned MIN_SCORE for Forex (Dragon win — chop needs tighter gates)
4. Soft ML scaler for ALL — no random 50% hard gate (institutional win)
5. Relaxed RL skip: 8+ trades at <15% WR (institutional win)
6. 4-loss circuit breaker, 12-bar cooldown (institutional win)
7. NO win cooldown (anti-pattern per research)
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.momentum_scorer import (
    _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, REGIME_PARAMS, DEFAULT_PARAMS
)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
START_EQ = 1000.0
SLIP = 0.0
BASE_RISK_PCT = 0.008
MAX_RISK_PCT = 0.015
MIN_RISK_PCT = 0.003
DAILY_LOSS_LIMIT = 0.015
CONSEC_LOSS_COOLDOWN = 12
CONSEC_LOSS_TRIGGER = 4
USE_RL = True
RL_REDUCE_MIN = 0.7

from config import RL_ENABLED_SYMBOLS, RL_SYMBOL_PARAMS
from config import SYMBOL_ATR_SL_OVERRIDE, SYMBOL_TRAIL_OVERRIDE, TRAIL_STEPS

# Vol-scaling constants
VOL_FAST = 20
VOL_SLOW = 120
BASE_MIN_SCORE = 6.5
SCORE_CLAMP_LOW = 5.0
SCORE_CLAMP_HIGH = 8.5

# Per-symbol approach selection: "institutional" (vol-scaled) or "tuned" (per-symbol fixed)
# Institutional won for Crypto, Index, Gold, EURJPY
# Tuned won for most Forex pairs and XAGUSD
SYMBOL_APPROACH = {
    "XAUUSD":   "institutional",
    "XAGUSD":   "tuned",       # PF 2.50 vs 1.44 — tuned wins big
    "BTCUSD":   "institutional",
    "ETHUSD":   "institutional",
    "NAS100.r": "institutional",
    "SP500.r":  "institutional",
    "GER40.r":  "tuned",          # PF 1.24 tuned vs 1.03 institutional — chop-prone
    "UK100.r":  "tuned",          # similar to GER40
    "JPN225ft": "institutional",
    "USDJPY":   "tuned",
    "EURUSD":   "tuned",
    "GBPUSD":   "tuned",
    "GBPJPY":   "tuned",
    "EURJPY":   "institutional",  # went from PF 0.98 to 1.21
    "AUDJPY":   "tuned",
    "AUDUSD":   "tuned",
    "NZDUSD":   "tuned",
    "EURAUD":   "tuned",
    "USDCAD":   "tuned",
    "USDCHF":   "tuned",
    "EURGBP":   "tuned",
    "EURCHF":   "tuned",
    "COPPER-Cr":"institutional",
    "UKOUSD":   "institutional",
}

# Tuned MIN_SCORE for forex/silver (from config.py — our relaxed version)
TUNED_MIN_SCORE = {
    "XAUUSD":   {"trending": 7.0, "ranging": 7.5, "volatile": 7.0, "low_vol": 7.0},
    "XAGUSD":   {"trending": 7.0, "ranging": 7.0, "volatile": 7.5, "low_vol": 7.5},
    "BTCUSD":   {"trending": 6.0, "ranging": 7.0, "volatile": 6.5, "low_vol": 6.5},
    "NAS100.r": {"trending": 7.0, "ranging": 7.0, "volatile": 7.0, "low_vol": 7.0},
    "JPN225ft": {"trending": 7.0, "ranging": 7.0, "volatile": 7.5, "low_vol": 7.5},
    "USDJPY":   {"trending": 7.0, "ranging": 7.5, "volatile": 7.5, "low_vol": 7.5},
    "USDCHF":   {"trending": 7.0, "ranging": 7.0, "volatile": 7.5, "low_vol": 7.5},
    "USDCAD":   {"trending": 7.0, "ranging": 7.0, "volatile": 7.0, "low_vol": 7.0},
    "EURJPY":   {"trending": 6.5, "ranging": 7.0, "volatile": 7.0, "low_vol": 7.0},
}

# Default tuned score for symbols not in the map
DEFAULT_TUNED = {"trending": 7.0, "ranging": 7.5, "volatile": 7.5, "low_vol": 7.5}

SYMBOL_SESSION_OVERRIDE = {
    "JPN225ft": (0, 22),
}

ALL_SYMBOLS = {
    "XAUUSD":    {"cache": "raw_h1_xauusd.pkl",   "point": 0.01,    "tv": 1.0,     "spread": 0.33,   "lot": 0.01,  "cat": "Gold"},
    "XAGUSD":    {"cache": "raw_h1_XAGUSD.pkl",   "point": 0.001,   "tv": 5.0,     "spread": 0.035,  "lot": 0.01,  "cat": "Gold"},
    "BTCUSD":    {"cache": "raw_h1_BTCUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 17.0,   "lot": 0.01,  "cat": "Crypto"},
    "ETHUSD":    {"cache": "raw_h1_ETHUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 2.0,    "lot": 0.01,  "cat": "Crypto"},
    "NAS100.r":  {"cache": "raw_h1_NAS100_r.pkl", "point": 0.01,    "tv": 0.01,    "spread": 1.80,   "lot": 0.10,  "cat": "Index"},
    "SP500.r":   {"cache": "raw_h1_SP500_r.pkl",  "point": 0.01,    "tv": 0.01,    "spread": 1.50,   "lot": 0.10,  "cat": "Index"},
    "GER40.r":   {"cache": "raw_h1_GER40_r.pkl",  "point": 0.01,    "tv": 0.0117,  "spread": 2.45,   "lot": 0.10,  "cat": "Index"},
    "UK100.r":   {"cache": "raw_h1_UK100_r.pkl",  "point": 0.01,    "tv": 0.0134,  "spread": 2.0,    "lot": 0.10,  "cat": "Index"},
    "JPN225ft":  {"cache": "raw_h1_JPN225ft.pkl", "point": 0.01,    "tv": 0.0063,  "spread": 10.0,   "lot": 1.00,  "cat": "Index"},
    "USDJPY":    {"cache": "raw_h1_USDJPY.pkl",   "point": 0.001,   "tv": 0.63,    "spread": 0.018,  "lot": 0.20,  "cat": "Forex"},
    "EURUSD":    {"cache": "raw_h1_EURUSD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00012,"lot": 0.20,  "cat": "Forex"},
    "GBPUSD":    {"cache": "raw_h1_GBPUSD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015,"lot": 0.20,  "cat": "Forex"},
    "GBPJPY":    {"cache": "raw_h1_GBPJPY.pkl",   "point": 0.001,   "tv": 0.63,    "spread": 0.025,  "lot": 0.05,  "cat": "Forex"},
    "EURJPY":    {"cache": "raw_h1_EURJPY.pkl",   "point": 0.001,   "tv": 0.63,    "spread": 0.020,  "lot": 0.05,  "cat": "Forex"},
    "AUDJPY":    {"cache": "raw_h1_AUDJPY.pkl",   "point": 0.001,   "tv": 0.63,    "spread": 0.020,  "lot": 0.05,  "cat": "Forex"},
    "AUDUSD":    {"cache": "raw_h1_AUDUSD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00012,"lot": 0.05,  "cat": "Forex"},
    "NZDUSD":    {"cache": "raw_h1_NZDUSD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015,"lot": 0.05,  "cat": "Forex"},
    "EURAUD":    {"cache": "raw_h1_EURAUD.pkl",    "point": 0.00001, "tv": 1.0,     "spread": 0.00020,"lot": 0.05,  "cat": "Forex"},
    "USDCAD":    {"cache": "raw_h1_USDCAD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015,"lot": 0.05,  "cat": "Forex"},
    "USDCHF":    {"cache": "raw_h1_USDCHF.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015,"lot": 0.05,  "cat": "Forex"},
    "EURGBP":    {"cache": "raw_h1_EURGBP.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015,"lot": 0.05,  "cat": "Forex"},
    "EURCHF":    {"cache": "raw_h1_EURCHF.pkl",   "point": 0.00001, "tv": 1.2762,  "spread": 0.00015,"lot": 0.05,  "cat": "Forex"},
    "COPPER-Cr": {"cache": "raw_h1_COPPER-Cr.pkl","point": 0.01,    "tv": 0.01,    "spread": 0.50,   "lot": 0.10,  "cat": "Commodity"},
    "UKOUSD":    {"cache": "raw_h1_UKOUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 0.50,   "lot": 0.01,  "cat": "Commodity"},
}


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


def get_min_score(symbol, regime, ind, bi):
    """Hybrid threshold: vol-scaled for institutional, per-symbol tuned for forex."""
    approach = SYMBOL_APPROACH.get(symbol, "tuned")

    if approach == "institutional":
        # Vol-scaled dynamic threshold (Man AHL approach)
        if bi < VOL_SLOW + 5:
            return BASE_MIN_SCORE
        atr = ind["at"]
        fast_atr = np.nanmean(atr[bi - VOL_FAST:bi + 1])
        slow_atr = np.nanmean(atr[bi - VOL_SLOW:bi + 1])
        if slow_atr <= 0 or np.isnan(fast_atr) or np.isnan(slow_atr):
            return BASE_MIN_SCORE
        vol_ratio = fast_atr / slow_atr
        scaled = BASE_MIN_SCORE * vol_ratio
        # Regime boost: trending easier, ranging harder
        regime_adj = {"trending": -0.5, "ranging": +0.5, "volatile": 0.0,
                      "low_vol": 0.0, "unknown": 0.0}.get(regime, 0.0)
        return max(SCORE_CLAMP_LOW, min(SCORE_CLAMP_HIGH, scaled + regime_adj))
    else:
        # Per-symbol tuned threshold (Dragon approach)
        sym_scores = TUNED_MIN_SCORE.get(symbol, DEFAULT_TUNED)
        return sym_scores.get(regime, 7.0)


def score_to_conviction(score, min_score):
    """Continuous score → conviction multiplier (0.5 to 1.5x)."""
    if score < min_score:
        return 0.0
    excess = score - min_score
    conviction = 0.6 + (excess / 3.0) * 0.9
    return min(1.5, max(0.5, conviction))


def ml_soft_scaler(score):
    """Soft ML meta-label: higher score = higher risk allocation."""
    return min(1.0, score / 10.0)


def run(symbol, days=365):
    scfg = ALL_SYMBOLS[symbol]
    cache_path = CACHE / scfg["cache"]
    if not cache_path.exists():
        return None
    df = pickle.load(open(cache_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    pt = scfg["point"]
    tv = scfg["tv"]
    spread = scfg["spread"]
    cat = scfg["cat"]
    sl_cap = 5000 * pt
    icfg = dict(IND_DEFAULTS)
    icfg.update(IND_OVERRIDES.get(symbol, {}))
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    start_idx = max(int(df[df["time"] >= cutoff].index[0]), max(icfg["EMA_T"] + 30, VOL_SLOW + 10))
    ind = _compute_indicators(df, icfg)
    n = ind["n"]

    eq = START_EQ
    peak = START_EQ
    max_dd = 0
    n_trades = 0
    wins = 0
    gross_p = 0
    gross_l = 0
    in_trade = False
    d = 0
    entry = 0
    pos_sl = 0
    sl_dist = 0
    trade_lot = 0.0
    entry_regime = "unknown"
    consec_losses = 0
    cooldown_until = 0
    daily_pnl = 0.0
    current_day = None
    day_eq_start = START_EQ
    day_stopped = False
    r_multiples = []
    max_consec_loss = 0
    current_streak = 0
    rl_trades = []
    rl_regime_wr = {}
    rl_hour_wr = {}
    entry_hour = 12
    entry_score = 0.0

    def record_outcome(pnl):
        nonlocal wins, gross_p, gross_l, consec_losses, current_streak, max_consec_loss
        nonlocal n_trades, peak, max_dd, cooldown_until, day_stopped

        r_val = pnl / (BASE_RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
        r_multiples.append(r_val)

        sym_rl = RL_SYMBOL_PARAMS.get(symbol, {})
        rl_lookback = sym_rl.get("lookback", 20)
        rl_trades.append({"pnl": pnl, "regime": entry_regime, "hour": entry_hour,
                          "dir": d, "score": entry_score, "won": pnl > 0})
        if len(rl_trades) > 100:
            del rl_trades[:-100]
        if len(rl_trades) >= rl_lookback:
            recent_rl = rl_trades[-rl_lookback:]
            for r in set(t["regime"] for t in recent_rl):
                rr = [t for t in recent_rl if t["regime"] == r]
                if len(rr) >= 3:
                    rl_regime_wr[r] = sum(1 for t in rr if t["won"]) / len(rr)
            for h in set(t["hour"] for t in recent_rl):
                hh = [t for t in recent_rl if t["hour"] == h]
                if len(hh) >= 3:
                    rl_hour_wr[h] = sum(1 for t in hh if t["won"]) / len(hh)

        if pnl > 0:
            gross_p += pnl
            wins += 1
            consec_losses = 0
            current_streak = 0
        else:
            gross_l += abs(pnl)
            consec_losses += 1
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
            if consec_losses >= CONSEC_LOSS_TRIGGER:
                cooldown_until = i + CONSEC_LOSS_COOLDOWN
                consec_losses = 0

        n_trades += 1
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    for i in range(start_idx, n):
        atr_val = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0
        if atr_val == 0:
            continue

        bar_time = df["time"].iloc[i]
        bar_hour = bar_time.hour if hasattr(bar_time, "hour") else 12
        sess_start, sess_end = SYMBOL_SESSION_OVERRIDE.get(symbol, (6, 22))
        if cat != "Crypto" and (bar_hour >= sess_end or bar_hour < sess_start):
            continue

        bar_date = bar_time.date() if hasattr(bar_time, "date") else None
        if bar_date and bar_date != current_day:
            current_day = bar_date
            day_eq_start = eq
            daily_pnl = 0.0
            day_stopped = False

        # MANAGE: trailing SL
        if in_trade:
            if (d == 1 and ind["l"][i] <= pos_sl) or (d == -1 and ind["h"][i] >= pos_sl):
                exit_cost = (spread + SLIP * pt)
                pnl = d * (pos_sl - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
                eq += pnl
                daily_pnl += pnl
                record_outcome(pnl)
                in_trade = False
                if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                    day_stopped = True
                continue

            cur = float(ind["c"][i])
            profit_r = ((cur - entry) * d) / sl_dist if sl_dist > 0 else 0
            new_sl = None
            trail = SYMBOL_TRAIL_OVERRIDE.get(symbol, TRAIL_STEPS)
            for th, ac, pa in trail:
                if profit_r >= th:
                    if ac == "trail":
                        new_sl = cur - pa * atr_val * d
                    elif ac == "lock":
                        new_sl = entry + pa * sl_dist * d
                    elif ac == "be":
                        new_sl = entry + 2 * pt * d
                    elif ac == "reduce_sl":
                        new_sl = entry - pa * sl_dist * d
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

        bi = i - 1
        if bi < VOL_SLOW + 5:
            continue
        ls, ss = _score(ind, bi)

        regime = get_regime(ind, bi)
        adaptive_min = get_min_score(symbol, regime, ind, bi)

        buy = ls >= adaptive_min
        sell = ss >= adaptive_min
        if not buy and not sell:
            continue
        new_dir = 1 if (buy and (not sell or ls >= ss)) else -1
        best_score = max(ls, ss)

        # Conviction sizing + soft ML
        ml_scale = ml_soft_scaler(best_score)
        conviction = score_to_conviction(best_score, adaptive_min)
        if conviction <= 0:
            continue

        # REVERSAL
        if in_trade and new_dir != d:
            exit_cost = (spread + SLIP * pt)
            pnl = d * (float(ind["c"][i]) - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
            eq += pnl
            daily_pnl += pnl
            record_outcome(pnl)
            in_trade = False
            if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                day_stopped = True
                continue

        # ENTRY
        if not in_trade:
            d = new_dir
            entry_regime = regime
            entry_hour = bar_hour
            entry_score = best_score

            sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
            sym_sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, 1.5)
            sl_dist = max(atr_val * sl_m, atr_val * sym_sl_mult)
            sl_dist = min(sl_dist, sl_cap)

            # Conviction-scaled position sizing
            risk_pct = BASE_RISK_PCT * conviction * ml_scale
            risk_pct = max(MIN_RISK_PCT, min(MAX_RISK_PCT, risk_pct))

            # RL: only skip truly toxic, scale rest
            sym_rl = RL_SYMBOL_PARAMS.get(symbol, {})
            rl_lookback = sym_rl.get("lookback", 20)
            rl_boost_max = sym_rl.get("boost_max", 1.4)
            if USE_RL and symbol in RL_ENABLED_SYMBOLS and len(rl_trades) >= rl_lookback:
                r_wr = rl_regime_wr.get(regime, 0.5)
                regime_count = len([t for t in rl_trades if t["regime"] == regime])
                if r_wr < 0.15 and regime_count >= 8:
                    continue
                h_wr = rl_hour_wr.get(bar_hour, 0.5)
                hour_count = len([t for t in rl_trades if t["hour"] == bar_hour])
                if h_wr < 0.15 and hour_count >= 8:
                    continue
                rd_trades = [t for t in rl_trades[-40:] if t["regime"] == regime and t["dir"] == d]
                if len(rd_trades) >= 8:
                    rd_wr = sum(1 for t in rd_trades if t["won"]) / len(rd_trades)
                    if rd_wr < 0.15:
                        continue

                rl_mult = 1.0
                if r_wr > 0.55:
                    rl_mult *= 1.0 + (r_wr - 0.5) * 0.8
                elif r_wr < 0.35:
                    rl_mult *= max(0.6, 1.0 - (0.5 - r_wr) * 0.8)
                if h_wr > 0.55:
                    rl_mult *= 1.0 + (h_wr - 0.5) * 0.6
                elif h_wr < 0.35:
                    rl_mult *= max(0.7, 1.0 - (0.5 - h_wr) * 0.5)

                recent_rl = rl_trades[-rl_lookback:]
                gp = sum(t["pnl"] for t in recent_rl if t["pnl"] > 0)
                gl = sum(abs(t["pnl"]) for t in recent_rl if t["pnl"] < 0) or 0.01
                rpf = gp / gl
                if rpf < 0.7:
                    rl_mult *= 0.6
                elif rpf > 2.5:
                    rl_mult *= 1.15
                rl_mult = max(RL_REDUCE_MIN, min(rl_boost_max, rl_mult))
                risk_pct *= rl_mult

            risk_pct = max(MIN_RISK_PCT, min(MAX_RISK_PCT, risk_pct))
            risk_amount = eq * risk_pct

            pip_value_per_lot = (sl_dist / pt) * tv
            if pip_value_per_lot > 0:
                trade_lot = risk_amount / pip_value_per_lot
                trade_lot = max(trade_lot, 0.01)
            else:
                trade_lot = 0.01

            entry_cost = (spread + SLIP * pt)
            entry = float(ind["o"][i]) + entry_cost / 2 * d
            pos_sl = entry - sl_dist * d
            in_trade = True

    if in_trade:
        pnl = d * (float(ind["c"][n - 1]) - entry) / pt * tv * trade_lot
        eq += pnl
        r_val = pnl / (BASE_RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
        r_multiples.append(r_val)
        if pnl > 0:
            gross_p += pnl
            wins += 1
        else:
            gross_l += abs(pnl)
        n_trades += 1
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    pf = gross_p / gross_l if gross_l > 0 else (999 if gross_p > 0 else 0)
    dd = max_dd / peak * 100 if peak else 0
    ret = (eq - START_EQ) / START_EQ * 100
    wr = wins / n_trades * 100 if n_trades else 0
    avg_r = np.mean(r_multiples) if r_multiples else 0
    std_r = np.std(r_multiples) if len(r_multiples) > 1 else 1
    sharpe = (avg_r / std_r) * np.sqrt(252) if std_r > 0 else 0

    return {"sym": symbol, "trades": n_trades, "wr": round(wr, 1), "pf": round(pf, 2),
            "ret": round(ret, 1), "dd": round(dd, 1), "eq": round(eq, 2),
            "gross_p": round(gross_p, 2), "gross_l": round(gross_l, 2),
            "max_consec_loss": max_consec_loss, "avg_r": round(avg_r, 3),
            "sharpe": round(sharpe, 2)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    print("=" * 115)
    print(f"  DRAGON HYBRID: Institutional for Crypto/Index/Gold + Tuned for Forex | Conviction sizing | Soft ML")
    print(f"  $1,000 | {args.days}d | Per-symbol approach | Score→Size | {CONSEC_LOSS_TRIGGER} loss breaker | No win cooldown")
    print("=" * 115)
    print(f"\n{'Symbol':<12} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Return%':>9} {'DD%':>7} {'Final$':>10} {'MaxConsLoss':>12} {'AvgR':>7} {'Grade':>6}")
    print("-" * 95)

    results = []
    for sym in sorted(ALL_SYMBOLS.keys()):
        r = run(sym, args.days)
        if r:
            results.append(r)
            grade = "A+" if r["pf"] >= 2.0 else "A" if r["pf"] >= 1.5 else "B" if r["pf"] >= 1.2 else "C" if r["pf"] >= 1.0 else "F"
            approach = SYMBOL_APPROACH.get(r["sym"], "tuned")
            tag = "I" if approach == "institutional" else "T"
            print(f"{r['sym']:<12} {r['trades']:>7} {r['wr']:>6.1f}% {r['pf']:>7.2f} {r['ret']:>8.1f}% {r['dd']:>6.1f}% ${r['eq']:>9.2f} {r['max_consec_loss']:>12} {r['avg_r']:>+7.3f} {grade:>5}{tag}")

    print("-" * 95)
    profitable = sorted([r for r in results if r["pf"] >= 1.2], key=lambda x: x["pf"], reverse=True)
    marginal = [r for r in results if 1.0 <= r["pf"] < 1.2]
    losing = [r for r in results if r["pf"] < 1.0]
    gp = sum(r["gross_p"] for r in results)
    gl = sum(r["gross_l"] for r in results)
    total_ret = sum(r["ret"] for r in results)
    avg_sharpe = np.mean([r["sharpe"] for r in results]) if results else 0
    print(f"{'PORTFOLIO':<12} {'':>7} {'':>7} {gp / gl if gl else 0:>7.2f} {total_ret / len(results) if results else 0:>8.1f}% {'':>7} {'':>10} {'':>12} {'':>7} {'':>6}")
    print(f"\nA+/A (PF>=1.5): {len([r for r in profitable if r['pf'] >= 1.5])} | B (1.2-1.5): {len([r for r in profitable if r['pf'] < 1.5])} | C (1.0-1.2): {len(marginal)} | F (<1.0): {len(losing)}")
    print(f"Avg Sharpe: {avg_sharpe:.2f}")
    if profitable:
        print("\n  RECOMMENDED FOR LIVE (Hybrid-grade):")
        for r in profitable:
            approach = SYMBOL_APPROACH.get(r["sym"], "tuned")
            print(f"    {r['sym']:<12} PF={r['pf']:.2f}  WR={r['wr']:.1f}%  Ret={r['ret']:+.1f}%  DD={r['dd']:.1f}%  Sharpe={r['sharpe']:.2f}  [{approach}]")
    print("=" * 115)
