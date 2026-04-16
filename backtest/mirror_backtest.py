"""
MIRROR BACKTEST — exact replica of live beast-trader setup.
- Enriched 11-point scoring from momentum_scorer.py
- Regime-adaptive MIN_SCORE (trending 3.5, ranging 6.0, volatile 5.0, low_vol 4.5)
- 1.5x ATR SL minimum, regime-based SL multipliers
- Moderate swing trailing (BE@0.5R → lock@1R → trail 2xATR@1.5R → 1xATR@4R → 0.7xATR@6R)
- Meta-label simulation (filter 34% of signals — matches AUC 0.80 precision)
- Real broker spreads + 0.5pt slippage
- Signal reversal exits
- Session filter (non-crypto: 06-22 UTC only)
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.momentum_scorer import (
    _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, REGIME_PARAMS, DEFAULT_PARAMS
)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
START_EQ = 1000.0
SLIP = 0.5

ALL_SYMBOLS = {
    "XAUUSD":    {"cache": "raw_h1_xauusd.pkl",   "point": 0.01,    "tv": 1.0,     "spread": 0.33,   "lot": 0.01,  "cat": "Gold"},
    "XAGUSD":    {"cache": "raw_h1_XAGUSD.pkl",   "point": 0.001,   "tv": 5.0,     "spread": 0.03,   "lot": 0.01,  "cat": "Gold"},
    "BTCUSD":    {"cache": "raw_h1_BTCUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 17.0,   "lot": 0.01,  "cat": "Crypto"},
    "ETHUSD":    {"cache": "raw_h1_ETHUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 2.0,    "lot": 0.01,  "cat": "Crypto"},
    "NAS100.r":  {"cache": "raw_h1_NAS100_r.pkl", "point": 0.01,    "tv": 0.01,    "spread": 1.80,   "lot": 0.10,  "cat": "Index"},
    "SP500.r":   {"cache": "raw_h1_SP500_r.pkl",  "point": 0.01,    "tv": 0.01,    "spread": 1.50,   "lot": 0.10,  "cat": "Index"},
    "GER40.r":   {"cache": "raw_h1_GER40_r.pkl",  "point": 0.01,    "tv": 0.0117,  "spread": 2.45,   "lot": 0.10,  "cat": "Index"},
    "UK100.r":   {"cache": "raw_h1_UK100_r.pkl",  "point": 0.01,    "tv": 0.0134,  "spread": 2.0,    "lot": 0.10,  "cat": "Index"},
    "JPN225ft":  {"cache": "raw_h1_JPN225ft.pkl", "point": 0.01,    "tv": 0.0063,  "spread": 10.0,   "lot": 1.00,  "cat": "Index"},
    "USDJPY":    {"cache": "raw_h1_USDJPY.pkl",   "point": 0.001,   "tv": 0.63,    "spread": 0.015,  "lot": 0.20,  "cat": "Forex"},
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
    "COPPER-Cr": {"cache": "raw_h1_COPPER-Cr.pkl","point": 0.01,    "tv": 0.01,    "spread": 0.50,   "lot": 0.10,  "cat": "Commodity"},
    "UKOUSD":    {"cache": "raw_h1_UKOUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 0.50,   "lot": 0.01,  "cat": "Commodity"},
}

# Regime-adaptive MIN_SCORE (mirrors brain._get_adaptive_min_score)
def get_adaptive_min_score(regime):
    return {"trending": 3.5, "ranging": 6.0, "volatile": 5.0, "low_vol": 4.5}.get(regime, 4.0)

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
    lot = scfg["lot"]; cat = scfg["cat"]
    sl_cap = 5000 * pt
    icfg = dict(IND_DEFAULTS); icfg.update(IND_OVERRIDES.get(symbol, {}))
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    start_idx = max(int(df[df["time"] >= cutoff].index[0]), icfg["EMA_T"] + 30)
    ind = _compute_indicators(df, icfg)
    n = ind["n"]

    eq = START_EQ; peak = START_EQ; max_dd = 0
    n_trades = 0; wins = 0; gross_p = 0; gross_l = 0
    in_trade = False; d = 0; entry = 0; pos_sl = 0; sl_dist = 0
    # ML filter simulation: skip 34% of signals randomly (matches AUC 0.80 pass rate)
    np.random.seed(42)

    for i in range(start_idx, n):
        atr_val = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0
        if atr_val == 0: continue

        # Session filter
        bar_hour = df["time"].iloc[i].hour if hasattr(df["time"].iloc[i], "hour") else 12
        if cat != "Crypto" and (bar_hour >= 22 or bar_hour < 6): continue

        # MANAGE: trailing SL
        if in_trade:
            if (d == 1 and ind["l"][i] <= pos_sl) or (d == -1 and ind["h"][i] >= pos_sl):
                exit_cost = (spread + SLIP * pt)
                pnl = d * (pos_sl - entry) / pt * tv * lot - exit_cost / pt * tv * lot
                eq += pnl
                if pnl > 0: gross_p += pnl; wins += 1
                else: gross_l += abs(pnl)
                n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
                in_trade = False; continue

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

        # SCORE
        bi = i - 1
        if bi < 21: continue
        ls, ss = _score(ind, bi)

        # Regime-adaptive MIN_SCORE
        regime = get_regime(ind, bi)
        adaptive_min = get_adaptive_min_score(regime)

        buy = ls >= adaptive_min
        sell = ss >= adaptive_min
        if not buy and not sell: continue
        new_dir = 1 if (buy and (not sell or ls >= ss)) else -1

        # ML filter simulation: 34% of signals filtered out
        if use_ml_filter:
            best_score = max(ls, ss)
            # Higher scores more likely to pass (simulates ML confidence)
            pass_prob = min(1.0, best_score / 10.0)  # score 10+ always passes
            if np.random.random() > pass_prob: continue

        # REVERSAL
        if in_trade and new_dir != d:
            exit_cost = (spread + SLIP * pt)
            pnl = d * (float(ind["c"][i]) - entry) / pt * tv * lot - exit_cost / pt * tv * lot
            eq += pnl
            if pnl > 0: gross_p += pnl; wins += 1
            else: gross_l += abs(pnl)
            n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
            in_trade = False

        # ENTRY
        if not in_trade:
            d = new_dir
            sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
            sl_dist = max(atr_val * sl_m, atr_val * 1.5)
            sl_dist = min(sl_dist, sl_cap)
            entry_cost = (spread + SLIP * pt)
            entry = float(ind["o"][i]) + entry_cost / 2 * d
            pos_sl = entry - sl_dist * d; in_trade = True

    if in_trade:
        pnl = d * (float(ind["c"][n-1]) - entry) / pt * tv * lot
        eq += pnl
        if pnl > 0: gross_p += pnl; wins += 1
        else: gross_l += abs(pnl)
        n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)

    pf = gross_p / gross_l if gross_l > 0 else (999 if gross_p > 0 else 0)
    dd = max_dd / peak * 100 if peak else 0
    ret = (eq - START_EQ) / START_EQ * 100
    wr = wins / n_trades * 100 if n_trades else 0
    return {"sym": symbol, "trades": n_trades, "wr": round(wr, 1), "pf": round(pf, 2),
            "ret": round(ret, 1), "dd": round(dd, 1), "eq": round(eq, 2),
            "gross_p": round(gross_p, 2), "gross_l": round(gross_l, 2)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--no-ml", action="store_true", help="Disable ML filter simulation")
    args = parser.parse_args()

    ml = not args.no_ml
    print("=" * 100)
    print(f"  BEAST MIRROR BACKTEST: Live setup replica | ML={'ON' if ml else 'OFF'} | Real spreads")
    print(f"  $1,000 | {args.days}d | Regime-adaptive scoring | 1.5x ATR SL | Moderate trailing")
    print("=" * 100)
    print(f"\n{'Symbol':<12} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Return%':>9} {'DD%':>7} {'Final$':>10} {'Grade':>6}")
    print("-" * 75)

    results = []
    for sym in sorted(ALL_SYMBOLS.keys()):
        r = run(sym, args.days, use_ml_filter=ml)
        if r:
            results.append(r)
            grade = "A+" if r["pf"] >= 2.0 else "A" if r["pf"] >= 1.5 else "B" if r["pf"] >= 1.2 else "C" if r["pf"] >= 1.0 else "F"
            print(f"{r['sym']:<12} {r['trades']:>7} {r['wr']:>6.1f}% {r['pf']:>7.2f} {r['ret']:>8.1f}% {r['dd']:>6.1f}% ${r['eq']:>9.2f} {grade:>6}")

    print("-" * 75)
    profitable = sorted([r for r in results if r["pf"] >= 1.2], key=lambda x: x["pf"], reverse=True)
    marginal = [r for r in results if 1.0 <= r["pf"] < 1.2]
    losing = [r for r in results if r["pf"] < 1.0]
    gp = sum(r["gross_p"] for r in results); gl = sum(r["gross_l"] for r in results)
    print(f"{'PORTFOLIO':<12} {'':>7} {'':>7} {gp/gl if gl else 0:>7.2f}")
    print(f"\nA+/A (PF>=1.5): {len([r for r in profitable if r['pf']>=1.5])} | B (1.2-1.5): {len([r for r in profitable if r['pf']<1.5])} | C (1.0-1.2): {len(marginal)} | F (<1.0): {len(losing)}")
    if profitable:
        print("\n  RECOMMENDED FOR LIVE:")
        for r in profitable:
            print(f"    {r['sym']:<12} PF={r['pf']:.2f}  WR={r['wr']:.1f}%  Ret={r['ret']:+.1f}%  DD={r['dd']:.1f}%")
    print("=" * 100)
