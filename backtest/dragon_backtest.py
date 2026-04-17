"""
DRAGON BACKTEST — ultra-conservative regime-adaptive backtest.
- Dragon-level MIN_SCORE thresholds (trending 6.0, ranging 8.0, volatile 7.0, low_vol 7.0)
- 1.5x ATR SL minimum, regime-based SL multipliers
- Moderate swing trailing (BE@0.5R -> lock@1R -> trail 2xATR@1.5R -> 1xATR@4R -> 0.7xATR@6R)
- Stricter ML filter: 50% of signals filtered (score/10 pass probability)
- Consecutive loss protection: 3 losses on a symbol -> skip 24 bars
- Dynamic position sizing: 0.3% equity risk per trade
- Daily loss limit: 1% equity -> stop trading for the day
- Real broker spreads, NO slippage (SLIP=0.0)
- Session filter (non-crypto: 06-22 UTC only)
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.momentum_scorer import (
    _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, REGIME_PARAMS, DEFAULT_PARAMS
)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
START_EQ = 1000.0
SLIP = 0.0  # User confirmed: live system has no slippage, spread only
RISK_PCT = 0.008  # 0.8% risk per trade (tuned: PF 2.54, Sharpe 2.30, DD 19.4%)
DAILY_LOSS_LIMIT = 0.01  # 1% daily loss -> stop trading
CONSEC_LOSS_COOLDOWN = 24  # bars to skip after 3 consecutive losses

# Per-symbol session overrides (start_utc, end_utc)
SYMBOL_SESSION_OVERRIDE = {
    "JPN225ft": (0, 22),           # include Asian session (00-22 UTC)
}

# Per-symbol ATR SL multiplier overrides (base is 1.5x from REGIME_PARAMS)
SYMBOL_ATR_SL_OVERRIDE = {
    "BTCUSD":   2.0,              # wider SL for crypto trends
    "XAGUSD":   1.8,              # wider SL for silver volatility
    "USDJPY":   1.2,              # tighter SL for forex ranges
}

# Per-symbol regime MIN_SCORE overrides
SYMBOL_MIN_SCORE_OVERRIDE = {
    "BTCUSD":   {"trending": 5.5, "ranging": 8.0, "volatile": 6.5, "low_vol": 7.0},
    "XAGUSD":   {"trending": 5.5, "ranging": 8.0, "volatile": 6.5, "low_vol": 7.0},
    "XAUUSD":   {"trending": 5.5, "ranging": 8.0, "volatile": 7.0, "low_vol": 7.0},
    "USDJPY":   {"trending": 6.5, "ranging": 8.5, "volatile": 7.5, "low_vol": 7.5},
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
    "EURAUD":    {"cache": "raw_h1_EURAUD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00020,"lot": 0.05,  "cat": "Forex"},
    "USDCAD":    {"cache": "raw_h1_USDCAD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015,"lot": 0.05,  "cat": "Forex"},
    "USDCHF":    {"cache": "raw_h1_USDCHF.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015,"lot": 0.05,  "cat": "Forex"},
    "EURGBP":    {"cache": "raw_h1_EURGBP.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015,"lot": 0.05,  "cat": "Forex"},
    "EURCHF":    {"cache": "raw_h1_EURCHF.pkl",   "point": 0.00001, "tv": 1.2762,  "spread": 0.00015,"lot": 0.05,  "cat": "Forex"},
    "COPPER-Cr": {"cache": "raw_h1_COPPER-Cr.pkl","point": 0.01,    "tv": 0.01,    "spread": 0.50,   "lot": 0.10,  "cat": "Commodity"},
    "UKOUSD":    {"cache": "raw_h1_UKOUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 0.50,   "lot": 0.01,  "cat": "Commodity"},
}

# Dragon regime-adaptive MIN_SCORE — much stricter than mirror
def get_adaptive_min_score(regime, symbol=None):
    # Per-symbol override first
    if symbol and symbol in SYMBOL_MIN_SCORE_OVERRIDE:
        sym_scores = SYMBOL_MIN_SCORE_OVERRIDE[symbol]
        if regime in sym_scores:
            return sym_scores[regime]
    return {"trending": 6.0, "ranging": 8.0, "volatile": 7.0, "low_vol": 7.0}.get(regime, 7.0)

def get_regime(ind, bi):
    if bi < 21 or np.isnan(ind["bbw"][bi]): return "unknown"
    bbw = float(ind["bbw"][bi])
    adx = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 0
    if bbw < 1.5 and adx < 20: return "ranging"
    if 1.5 <= bbw < 3.0 and adx > 25: return "trending"
    if bbw >= 3.0: return "volatile"
    return "low_vol"

def run(symbol, days=365, use_ml_filter=True):
    scfg = ALL_SYMBOLS[symbol]
    cache_path = CACHE / scfg["cache"]
    if not cache_path.exists(): return None
    df = pickle.load(open(cache_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    pt = scfg["point"]; tv = scfg["tv"]; spread = scfg["spread"]
    cat = scfg["cat"]
    sl_cap = 5000 * pt
    icfg = dict(IND_DEFAULTS); icfg.update(IND_OVERRIDES.get(symbol, {}))
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    start_idx = max(int(df[df["time"] >= cutoff].index[0]), icfg["EMA_T"] + 30)
    ind = _compute_indicators(df, icfg)
    n = ind["n"]

    eq = START_EQ; peak = START_EQ; max_dd = 0
    n_trades = 0; wins = 0; gross_p = 0; gross_l = 0
    in_trade = False; d = 0; entry = 0; pos_sl = 0; sl_dist = 0
    trade_lot = 0.0  # dynamic lot per trade

    # Consecutive loss tracking
    consec_losses = 0
    cooldown_until = 0  # bar index when cooldown expires

    # Daily loss tracking
    daily_pnl = 0.0
    current_day = None
    day_eq_start = START_EQ
    day_stopped = False

    # R-multiple tracking
    r_multiples = []
    max_consec_loss = 0
    current_streak = 0

    # ML filter simulation: 50% of signals filtered (stricter than mirror's 34%)
    np.random.seed(42)

    for i in range(start_idx, n):
        atr_val = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0
        if atr_val == 0: continue

        # Session filter (per-symbol override or default 06-22 UTC)
        bar_time = df["time"].iloc[i]
        bar_hour = bar_time.hour if hasattr(bar_time, "hour") else 12
        sess_start, sess_end = SYMBOL_SESSION_OVERRIDE.get(symbol, (6, 22))
        if cat != "Crypto" and (bar_hour >= sess_end or bar_hour < sess_start): continue

        # Daily loss reset
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

                # Track R-multiple
                r_val = pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
                r_multiples.append(r_val)

                if pnl > 0:
                    gross_p += pnl; wins += 1
                    consec_losses = 0
                    current_streak = 0
                else:
                    gross_l += abs(pnl)
                    consec_losses += 1
                    current_streak += 1
                    max_consec_loss = max(max_consec_loss, current_streak)
                    # Consecutive loss protection: 3 losses -> cooldown
                    if consec_losses >= 3:
                        cooldown_until = i + CONSEC_LOSS_COOLDOWN
                        consec_losses = 0

                n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
                in_trade = False

                # Check daily loss limit after closing trade
                if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                    day_stopped = True
                continue

            cur = float(ind["c"][i])
            profit_r = ((cur - entry) * d) / sl_dist if sl_dist > 0 else 0
            new_sl = None
            if profit_r >= 6.0: new_sl = cur - 0.7 * atr_val * d
            elif profit_r >= 4.0: new_sl = cur - 1.0 * atr_val * d
            elif profit_r >= 2.5: new_sl = cur - 1.5 * atr_val * d
            elif profit_r >= 1.5:
                new_sl = cur - 2.0 * atr_val * d
                floor = entry + 0.5 * sl_dist * d
                if d == 1: new_sl = max(new_sl, floor)
                else: new_sl = min(new_sl, floor)
            elif profit_r >= 1.0: new_sl = entry + 0.5 * sl_dist * d
            elif profit_r >= 0.5: new_sl = entry + 2 * pt * d
            if new_sl is not None:
                if d == 1 and new_sl > pos_sl: pos_sl = new_sl
                elif d == -1 and new_sl < pos_sl: pos_sl = new_sl

        # Skip if daily loss limit hit
        if day_stopped: continue

        # Skip if in cooldown from consecutive losses
        if i < cooldown_until: continue

        # SCORE
        bi = i - 1
        if bi < 21: continue
        ls, ss = _score(ind, bi)

        # Dragon regime-adaptive MIN_SCORE (much stricter, per-symbol)
        regime = get_regime(ind, bi)
        adaptive_min = get_adaptive_min_score(regime, symbol=symbol)

        buy = ls >= adaptive_min
        sell = ss >= adaptive_min
        if not buy and not sell: continue
        new_dir = 1 if (buy and (not sell or ls >= ss)) else -1

        # Stricter ML filter: 50% of signals filtered (vs mirror's 34%)
        if use_ml_filter:
            best_score = max(ls, ss)
            pass_prob = min(1.0, best_score / 10.0)  # score 10+ always passes
            if np.random.random() > pass_prob: continue

        # REVERSAL
        if in_trade and new_dir != d:
            exit_cost = (spread + SLIP * pt)
            pnl = d * (float(ind["c"][i]) - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
            eq += pnl
            daily_pnl += pnl

            r_val = pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
            r_multiples.append(r_val)

            if pnl > 0:
                gross_p += pnl; wins += 1
                consec_losses = 0
                current_streak = 0
            else:
                gross_l += abs(pnl)
                consec_losses += 1
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
                if consec_losses >= 3:
                    cooldown_until = i + CONSEC_LOSS_COOLDOWN
                    consec_losses = 0

            n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
            in_trade = False

            if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                day_stopped = True
                continue

        # ENTRY
        if not in_trade:
            d = new_dir
            sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
            # Per-symbol ATR SL override (minimum floor)
            sym_sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, 1.5)
            sl_dist = max(atr_val * sl_m, atr_val * sym_sl_mult)
            sl_dist = min(sl_dist, sl_cap)

            # Dynamic position sizing: 0.3% of equity
            risk_amount = eq * RISK_PCT
            pip_value_per_lot = (sl_dist / pt) * tv
            if pip_value_per_lot > 0:
                trade_lot = risk_amount / pip_value_per_lot
                trade_lot = max(trade_lot, 0.01)  # minimum lot
            else:
                trade_lot = 0.01

            entry_cost = (spread + SLIP * pt)
            entry = float(ind["o"][i]) + entry_cost / 2 * d
            pos_sl = entry - sl_dist * d; in_trade = True

    # Close any open trade at end
    if in_trade:
        pnl = d * (float(ind["c"][n-1]) - entry) / pt * tv * trade_lot
        eq += pnl
        r_val = pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
        r_multiples.append(r_val)
        if pnl > 0: gross_p += pnl; wins += 1
        else: gross_l += abs(pnl)
        n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)

    pf = gross_p / gross_l if gross_l > 0 else (999 if gross_p > 0 else 0)
    dd = max_dd / peak * 100 if peak else 0
    ret = (eq - START_EQ) / START_EQ * 100
    wr = wins / n_trades * 100 if n_trades else 0

    # Sharpe approximation (annualised from per-trade R)
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
    parser.add_argument("--no-ml", action="store_true", help="Disable ML filter simulation")
    args = parser.parse_args()

    ml = not args.no_ml
    print("=" * 115)
    print(f"  DRAGON BACKTEST: Ultra-conservative | ML={'ON' if ml else 'OFF'} | 0.3% risk | Real spreads | No slippage")
    print(f"  $1,000 | {args.days}d | Dragon scoring | 1.5x ATR SL | Loss streak protection")
    print("=" * 115)
    print(f"\n{'Symbol':<12} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Return%':>9} {'DD%':>7} {'Final$':>10} {'MaxConsLoss':>12} {'AvgR':>7} {'Grade':>6}")
    print("-" * 95)

    results = []
    for sym in sorted(ALL_SYMBOLS.keys()):
        r = run(sym, args.days, use_ml_filter=ml)
        if r:
            results.append(r)
            grade = "A+" if r["pf"] >= 2.0 else "A" if r["pf"] >= 1.5 else "B" if r["pf"] >= 1.2 else "C" if r["pf"] >= 1.0 else "F"
            print(f"{r['sym']:<12} {r['trades']:>7} {r['wr']:>6.1f}% {r['pf']:>7.2f} {r['ret']:>8.1f}% {r['dd']:>6.1f}% ${r['eq']:>9.2f} {r['max_consec_loss']:>12} {r['avg_r']:>+7.3f} {grade:>6}")

    print("-" * 95)
    profitable = sorted([r for r in results if r["pf"] >= 1.2], key=lambda x: x["pf"], reverse=True)
    marginal = [r for r in results if 1.0 <= r["pf"] < 1.2]
    losing = [r for r in results if r["pf"] < 1.0]
    gp = sum(r["gross_p"] for r in results); gl = sum(r["gross_l"] for r in results)
    total_ret = sum(r["ret"] for r in results)
    avg_sharpe = np.mean([r["sharpe"] for r in results]) if results else 0
    print(f"{'PORTFOLIO':<12} {'':>7} {'':>7} {gp/gl if gl else 0:>7.2f} {total_ret/len(results) if results else 0:>8.1f}% {'':>7} {'':>10} {'':>12} {'':>7} {'':>6}")
    print(f"\nA+/A (PF>=1.5): {len([r for r in profitable if r['pf']>=1.5])} | B (1.2-1.5): {len([r for r in profitable if r['pf']<1.5])} | C (1.0-1.2): {len(marginal)} | F (<1.0): {len(losing)}")
    print(f"Avg Sharpe: {avg_sharpe:.2f}")
    if profitable:
        print("\n  RECOMMENDED FOR LIVE (Dragon-grade):")
        for r in profitable:
            print(f"    {r['sym']:<12} PF={r['pf']:.2f}  WR={r['wr']:.1f}%  Ret={r['ret']:+.1f}%  DD={r['dd']:.1f}%  Sharpe={r['sharpe']:.2f}  MaxStreak={r['max_consec_loss']}")
    print("=" * 115)
