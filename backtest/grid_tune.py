"""
Grid tuner — finds optimal (SL_mult, trail_profile, MIN_SCORE) per symbol.
OPTIMIZED: precomputes indicators + scores once, reuses across all combos.
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
from itertools import product
import time as _time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.momentum_scorer import (
    _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, REGIME_PARAMS, DEFAULT_PARAMS
)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
START_EQ = 1000.0
SLIP = 0.0
RISK_PCT = 0.008
DAILY_LOSS_LIMIT = 0.01
CONSEC_LOSS_COOLDOWN = 24

# Grid dimensions
SL_MULTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
TRAIL_PROFILES = {
    "orig": [
        (6.0, "trail", 0.7), (4.0, "trail", 1.0), (2.5, "trail", 1.5),
        (1.5, "trail", 2.0), (1.0, "lock", 0.5), (0.5, "be", 0.0),
    ],
    "aggr": [
        (8.0, "trail", 0.5), (4.0, "trail", 0.7), (2.0, "trail", 1.0),
        (1.5, "trail", 1.5), (0.8, "lock", 0.2),
    ],
    "prog": [
        (6.0, "trail", 0.5), (4.0, "trail", 0.7), (2.0, "trail", 1.0),
        (1.5, "trail", 1.5), (1.0, "lock", 0.33), (0.6, "lock", 0.20),
        (0.3, "lock", 0.10),
    ],
    "tight": [
        (4.0, "trail", 0.5), (2.0, "trail", 0.7), (1.5, "trail", 1.0),
        (1.0, "lock", 0.4), (0.5, "lock", 0.15), (0.3, "be", 0.0),
    ],
}
TREND_MINS = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
RANGE_MINS = [6.5, 7.0, 7.5, 8.0, 8.5]

ALL_SYMBOLS = {
    "EURUSD":    {"cache": "raw_h1_EURUSD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00012, "cat": "Forex"},
    "GBPUSD":    {"cache": "raw_h1_GBPUSD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015, "cat": "Forex"},
    "GBPJPY":    {"cache": "raw_h1_GBPJPY.pkl",   "point": 0.001,   "tv": 0.63,    "spread": 0.025,   "cat": "Forex"},
    "EURJPY":    {"cache": "raw_h1_EURJPY.pkl",    "point": 0.001,   "tv": 0.63,    "spread": 0.020,   "cat": "Forex"},
    "AUDJPY":    {"cache": "raw_h1_AUDJPY.pkl",    "point": 0.001,   "tv": 0.63,    "spread": 0.020,   "cat": "Forex"},
    "AUDUSD":    {"cache": "raw_h1_AUDUSD.pkl",    "point": 0.00001, "tv": 1.0,     "spread": 0.00012, "cat": "Forex"},
    "NZDUSD":    {"cache": "raw_h1_NZDUSD.pkl",    "point": 0.00001, "tv": 1.0,     "spread": 0.00015, "cat": "Forex"},
    "EURAUD":    {"cache": "raw_h1_EURAUD.pkl",    "point": 0.00001, "tv": 1.0,     "spread": 0.00020, "cat": "Forex"},
    "USDCAD":    {"cache": "raw_h1_USDCAD.pkl",    "point": 0.00001, "tv": 1.0,     "spread": 0.00015, "cat": "Forex"},
    "USDCHF":    {"cache": "raw_h1_USDCHF.pkl",    "point": 0.00001, "tv": 1.0,     "spread": 0.00015, "cat": "Forex"},
    "EURGBP":    {"cache": "raw_h1_EURGBP.pkl",    "point": 0.00001, "tv": 1.0,     "spread": 0.00015, "cat": "Forex"},
    "EURCHF":    {"cache": "raw_h1_EURCHF.pkl",    "point": 0.00001, "tv": 1.2762,  "spread": 0.00015, "cat": "Forex"},
}


def get_regime(bbw, adx):
    if bbw < 1.5 and adx < 20: return "ranging"
    if 1.5 <= bbw < 3.0 and adx > 25: return "trending"
    if bbw >= 3.0: return "volatile"
    return "low_vol"


def precompute(symbol, days=365):
    """Precompute indicators, scores, regimes — done ONCE per symbol."""
    scfg = ALL_SYMBOLS[symbol]
    cache_path = CACHE / scfg["cache"]
    if not cache_path.exists():
        return None

    df = pickle.load(open(cache_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    pt = scfg["point"]; tv = scfg["tv"]; spread = scfg["spread"]
    sl_cap = 5000 * pt
    icfg = dict(IND_DEFAULTS); icfg.update(IND_OVERRIDES.get(symbol, {}))
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    start_idx = max(int(df[df["time"] >= cutoff].index[0]), icfg["EMA_T"] + 30)
    ind = _compute_indicators(df, icfg)
    n = ind["n"]

    # Precompute per-bar data
    bars = []
    for i in range(start_idx, n):
        atr_val = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0
        if atr_val == 0: continue

        bar_time = df["time"].iloc[i]
        bar_hour = bar_time.hour if hasattr(bar_time, "hour") else 12
        if bar_hour >= 22 or bar_hour < 6: continue

        bi = i - 1
        if bi < 21: continue

        bbw = float(ind["bbw"][bi]) if not np.isnan(ind["bbw"][bi]) else 0
        adx = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 0
        regime = get_regime(bbw, adx)
        ls, ss = _score(ind, bi)

        bar_date = bar_time.date() if hasattr(bar_time, "date") else None

        bars.append({
            "i": i, "atr": atr_val, "regime": regime,
            "ls": ls, "ss": ss, "date": bar_date,
            "o": float(ind["o"][i]), "h": float(ind["h"][i]),
            "l": float(ind["l"][i]), "c": float(ind["c"][i]),
            "sl_cap": sl_cap,
            # ML filter (precomputed random)
            "ml_rand": np.random.random(),
        })

    return bars, pt, tv, spread


def run_combo(bars, pt, tv, spread, sl_mult, trail, trend_min, range_min):
    """Run backtest with precomputed bars — FAST."""
    min_scores = {"trending": trend_min, "ranging": range_min,
                  "volatile": trend_min + 0.5, "low_vol": trend_min + 0.5}

    eq = START_EQ; peak = START_EQ; max_dd = 0
    n_trades = 0; wins = 0; gross_p = 0; gross_l = 0
    in_trade = False; d = 0; entry = 0.0; pos_sl = 0.0; sl_dist = 0.0
    trade_lot = 0.0
    consec_losses = 0; cooldown_idx = 0
    daily_pnl = 0.0; current_day = None; day_eq_start = START_EQ; day_stopped = False

    for bar in bars:
        i = bar["i"]; atr_val = bar["atr"]
        bar_date = bar["date"]

        if bar_date and bar_date != current_day:
            current_day = bar_date; day_eq_start = eq; daily_pnl = 0.0; day_stopped = False

        # MANAGE
        if in_trade:
            hit_sl = (d == 1 and bar["l"] <= pos_sl) or (d == -1 and bar["h"] >= pos_sl)
            if hit_sl:
                pnl = d * (pos_sl - entry) / pt * tv * trade_lot - spread / pt * tv * trade_lot
                eq += pnl; daily_pnl += pnl
                if pnl > 0: gross_p += pnl; wins += 1; consec_losses = 0
                else:
                    gross_l += abs(pnl); consec_losses += 1
                    if consec_losses >= 3: cooldown_idx = i + CONSEC_LOSS_COOLDOWN; consec_losses = 0
                n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
                in_trade = False
                if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start): day_stopped = True
                continue

            cur = bar["c"]
            profit_r = ((cur - entry) * d) / sl_dist if sl_dist > 0 else 0
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

        if day_stopped: continue
        if i < cooldown_idx: continue

        ls = bar["ls"]; ss = bar["ss"]
        regime = bar["regime"]
        adaptive_min = min_scores.get(regime, 7.0)

        buy = ls >= adaptive_min
        sell = ss >= adaptive_min
        if not buy and not sell: continue
        new_dir = 1 if (buy and (not sell or ls >= ss)) else -1

        # ML filter
        best_score = max(ls, ss)
        pass_prob = min(1.0, best_score / 10.0)
        if bar["ml_rand"] > pass_prob: continue

        # REVERSAL
        if in_trade and new_dir != d:
            pnl = d * (bar["c"] - entry) / pt * tv * trade_lot - spread / pt * tv * trade_lot
            eq += pnl; daily_pnl += pnl
            if pnl > 0: gross_p += pnl; wins += 1; consec_losses = 0
            else:
                gross_l += abs(pnl); consec_losses += 1
                if consec_losses >= 3: cooldown_idx = i + CONSEC_LOSS_COOLDOWN; consec_losses = 0
            n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
            in_trade = False
            if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start): day_stopped = True; continue

        # ENTRY
        if not in_trade:
            d = new_dir
            sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
            sl_dist = min(max(atr_val * sl_m, atr_val * sl_mult), bar["sl_cap"])

            risk_amount = eq * RISK_PCT
            pip_value_per_lot = (sl_dist / pt) * tv
            trade_lot = max(risk_amount / pip_value_per_lot, 0.01) if pip_value_per_lot > 0 else 0.01

            entry = bar["o"] + spread / 2 * d
            pos_sl = entry - sl_dist * d; in_trade = True

    # Close open
    if in_trade and bars:
        pnl = d * (bars[-1]["c"] - entry) / pt * tv * trade_lot
        eq += pnl
        if pnl > 0: gross_p += pnl; wins += 1
        else: gross_l += abs(pnl)
        n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)

    pf = gross_p / gross_l if gross_l > 0 else (999 if gross_p > 0 else 0)
    dd = max_dd / peak * 100 if peak else 0
    wr = wins / n_trades * 100 if n_trades else 0
    ret = (eq - START_EQ) / START_EQ * 100
    return {"trades": n_trades, "wr": round(wr, 1), "pf": round(pf, 2),
            "ret": round(ret, 1), "dd": round(dd, 1), "eq": round(eq, 2)}


def tune_symbol(symbol, days=365):
    t0 = _time.time()
    print(f"\n  TUNING: {symbol}...", end=" ", flush=True)

    np.random.seed(42)
    pre = precompute(symbol, days)
    if pre is None:
        print("NO CACHE")
        return symbol, None, None

    bars, pt, tv, spread = pre
    print(f"({len(bars)} bars)", end=" ", flush=True)

    best = None
    best_params = None
    tested = 0

    for sl_m in SL_MULTS:
        for trail_name, trail in TRAIL_PROFILES.items():
            for t_min in TREND_MINS:
                for r_min in RANGE_MINS:
                    if r_min < t_min: continue
                    r = run_combo(bars, pt, tv, spread, sl_m, trail, t_min, r_min)
                    tested += 1
                    if r["trades"] < 15: continue
                    score = r["pf"] - r["dd"] * 0.02 + r["trades"] * 0.001
                    if best is None or score > best["_score"]:
                        r["_score"] = score
                        best = r
                        best_params = {"sl": sl_m, "trail": trail_name,
                                       "trend_min": t_min, "range_min": r_min}

    elapsed = _time.time() - t0
    if best:
        p = best_params
        print(f"→ SL={p['sl']}x {p['trail']} T={p['trend_min']} R={p['range_min']} | "
              f"PF={best['pf']:.2f} WR={best['wr']:.1f}% Ret={best['ret']:+.1f}% DD={best['dd']:.1f}% "
              f"({best['trades']}t) [{elapsed:.0f}s]", flush=True)
        return symbol, best_params, best
    else:
        print(f"NO viable config [{elapsed:.0f}s]", flush=True)
        return symbol, None, None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    print("=" * 90)
    print(f"  DRAGON GRID TUNER — {args.days}d | {len(SL_MULTS)}xSL × {len(TRAIL_PROFILES)}xTrail × scores")
    print("=" * 90)

    results = []
    symbols = [args.symbol] if args.symbol else sorted(ALL_SYMBOLS.keys())
    for sym in symbols:
        sym, params, r = tune_symbol(sym, args.days)
        if params:
            results.append((sym, params, r))

    print(f"\n{'='*90}")
    print(f"  TUNED RESULTS")
    print(f"{'='*90}")
    print(f"{'Symbol':<10} {'SL':>4} {'Trail':<6} {'T_min':>5} {'R_min':>5} | {'PF':>6} {'WR%':>6} {'Ret%':>8} {'DD%':>5} {'Trades':>6} {'Grade':>5}")
    print("-" * 80)
    for sym, p, r in sorted(results, key=lambda x: x[2]["pf"], reverse=True):
        grade = "A+" if r["pf"] >= 2.0 else "A" if r["pf"] >= 1.5 else "B" if r["pf"] >= 1.2 else "C" if r["pf"] >= 1.0 else "F"
        print(f"{sym:<10} {p['sl']:>4} {p['trail']:<6} {p['trend_min']:>5} {p['range_min']:>5} | "
              f"{r['pf']:>6.2f} {r['wr']:>5.1f}% {r['ret']:>+7.1f}% {r['dd']:>4.1f}% {r['trades']:>6} {grade:>5}")

    profitable = [x for x in results if x[2]["pf"] >= 1.2]
    print(f"\n  Profitable (PF >= 1.2): {len(profitable)}/{len(results)}")
    if profitable:
        print("\n  CONFIG FOR config.py:")
        for sym, p, r in sorted(profitable, key=lambda x: x[2]["pf"], reverse=True):
            print(f'    "{sym}": {p["sl"]},  # tuned: PF={r["pf"]:.2f} trail={p["trail"]} T={p["trend_min"]} R={p["range_min"]}')
