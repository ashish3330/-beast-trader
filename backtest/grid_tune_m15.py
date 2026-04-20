"""
M15 GRID TUNER — finds optimal indicator params + SL + trail + thresholds per symbol.

Unlike H1 grid_tune.py which only tunes SL/trail/scores (indicators are fixed),
M15 requires tuning the INDICATOR PARAMETERS because H1-tuned periods don't transfer.

Grid dimensions:
- EMA_S: [20, 30, 45, 60]         (4 options — time horizon 5h to 15h)
- EMA_L: [80, 120, 160]           (3 options)
- MACD_F: [12, 18, 24, 36]        (4 options)
- MACD_SL: [42, 63, 78]           (3 options)
- ST_F: [2.0, 2.5, 3.0, 3.5]     (4 options)
- SL_MULT: [0.5, 1.0, 1.5, 2.0, 2.5]  (5 options)
- TRAIL: [orig, aggr, tight]       (3 options)
- MIN_SCORE: [5.5, 6.0, 6.5, 7.0] (4 options)

Total: 4×3×4×3×4×5×3×4 = 34,560 combos per symbol.
With precomputation trick: compute indicators once per (EMA_S, EMA_L, MACD_F, MACD_SL, ST_F),
then reuse across SL/trail/score combos.

Indicator combos: 4×3×4×3×4 = 576
Per-indicator combos: 5×3×4 = 60
Total with precompute: 576 indicator passes × 60 fast combos = manageable.
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
from itertools import product
import time as _time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.momentum_scorer import (
    _compute_indicators, _score, REGIME_PARAMS, DEFAULT_PARAMS
)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
START_EQ = 1000.0
SLIP = 0.0
RISK_PCT = 0.010  # 1.0% for M15 (slightly lower than H1's 1.2% due to more trades)
DAILY_LOSS_LIMIT = 0.02
CONSEC_LOSS_COOLDOWN = 48  # 48 M15 bars = 12 hours

# ═══ INDICATOR GRID ═══
EMA_S_GRID = [20, 30, 45, 60]
EMA_L_GRID = [80, 120, 160]
MACD_F_GRID = [12, 18, 24, 36]
MACD_SL_GRID = [42, 63, 78]
ST_F_GRID = [2.0, 2.5, 3.0, 3.5]

# ═══ STRATEGY GRID ═══
SL_MULTS = [0.5, 1.0, 1.5, 2.0, 2.5]
TRAIL_PROFILES = {
    "orig": [
        (6.0, "trail", 0.7), (4.0, "trail", 1.0), (2.5, "trail", 1.5),
        (1.5, "trail", 2.0), (1.0, "lock", 0.5), (0.5, "be", 0.0),
    ],
    "aggr": [
        (8.0, "trail", 0.5), (4.0, "trail", 0.7), (2.0, "trail", 1.0),
        (1.5, "trail", 1.5), (0.8, "lock", 0.2),
    ],
    "tight": [
        (4.0, "trail", 0.5), (2.0, "trail", 0.7), (1.5, "trail", 1.0),
        (1.0, "lock", 0.4), (0.5, "lock", 0.15), (0.3, "be", 0.0),
    ],
}
MIN_SCORES = [5.5, 6.0, 6.5, 7.0]

# ═══ SYMBOLS (M15 cache) ═══
M15_SYMBOLS = {
    "XAUUSD":    {"cache": "raw_m15_xauusd.pkl",   "point": 0.01,    "tv": 1.0,     "spread": 0.33,    "cat": "Gold",   "sess": (6, 22)},
    "XAGUSD":    {"cache": "raw_m15_XAGUSD.pkl",   "point": 0.001,   "tv": 5.0,     "spread": 0.035,   "cat": "Gold",   "sess": (6, 22)},
    "BTCUSD":    {"cache": "raw_m15_BTCUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 17.0,    "cat": "Crypto", "sess": (0, 24)},
    "NAS100.r":  {"cache": "raw_m15_NAS100_r.pkl", "point": 0.01,    "tv": 0.01,    "spread": 1.80,    "cat": "Index",  "sess": (6, 22)},
    "JPN225ft":  {"cache": "raw_m15_JPN225ft.pkl", "point": 0.01,    "tv": 0.0063,  "spread": 10.0,    "cat": "Index",  "sess": (0, 22)},
    "USDJPY":    {"cache": "raw_m15_USDJPY.pkl",   "point": 0.001,   "tv": 0.63,    "spread": 0.018,   "cat": "Forex",  "sess": (6, 22)},
    "USDCAD":    {"cache": "raw_m15_USDCAD.pkl",   "point": 0.00001, "tv": 1.0,     "spread": 0.00015, "cat": "Forex",  "sess": (6, 22)},
}


def get_regime(bbw, adx):
    if bbw < 1.5 and adx < 20: return "ranging"
    if 1.5 <= bbw < 3.0 and adx > 25: return "trending"
    if bbw >= 3.0: return "volatile"
    return "low_vol"


def precompute_bars(df, ind, scfg, start_idx):
    """Precompute per-bar data from indicators."""
    n = ind["n"]
    pt = scfg["point"]; sl_cap = 5000 * pt
    cat = scfg["cat"]
    sess_start, sess_end = scfg.get("sess", (6, 22))

    bars = []
    for i in range(start_idx, n):
        atr_val = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0
        if atr_val == 0: continue

        bar_time = df["time"].iloc[i]
        bar_hour = bar_time.hour if hasattr(bar_time, "hour") else 12
        if sess_end != 24 and (bar_hour >= sess_end or bar_hour < sess_start):
            continue

        bi = i - 1
        if bi < 50: continue  # Need more history for M15

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
        })
    return bars


def run_combo(bars, pt, tv, spread, sl_mult, trail, min_score):
    """Fast backtest with precomputed bars."""
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

        if day_stopped or i < cooldown_idx: continue

        # SCORE
        ls = bar["ls"]; ss = bar["ss"]
        buy = ls >= min_score
        sell = ss >= min_score
        if not buy and not sell: continue
        new_dir = 1 if (buy and (not sell or ls >= ss)) else -1

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
            sl_m = REGIME_PARAMS.get(bar["regime"], DEFAULT_PARAMS)[0]
            sl_dist = min(max(atr_val * sl_m, atr_val * sl_mult), bar["sl_cap"])
            risk_amount = eq * RISK_PCT
            pip_value_per_lot = (sl_dist / pt) * tv
            trade_lot = max(risk_amount / pip_value_per_lot, 0.01) if pip_value_per_lot > 0 else 0.01
            entry = bar["o"] + spread / 2 * d
            pos_sl = entry - sl_dist * d; in_trade = True

    if in_trade and bars:
        pnl = d * (bars[-1]["c"] - entry) / pt * tv * trade_lot
        eq += pnl
        if pnl > 0: gross_p += pnl; wins += 1
        else: gross_l += abs(pnl)
        n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)

    pf = gross_p / gross_l if gross_l > 0 else (999 if gross_p > 0 else 0)
    dd = max_dd / peak * 100 if peak else 0
    wr = wins / n_trades * 100 if n_trades else 0
    return {"trades": n_trades, "wr": round(wr, 1), "pf": round(pf, 2), "dd": round(dd, 1),
            "eq": round(eq, 2), "ret": round((eq - START_EQ) / START_EQ * 100, 1)}


def tune_symbol(symbol, days=365, top_n=5):
    """Full grid search: indicator params × strategy params."""
    t0 = _time.time()
    scfg = M15_SYMBOLS[symbol]
    cache_path = CACHE / scfg["cache"]
    if not cache_path.exists():
        print(f"  {symbol}: NO CACHE"); return symbol, []

    df = pickle.load(open(cache_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    pt = scfg["point"]; tv = scfg["tv"]; spread = scfg["spread"]
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    available_days = (df["time"].max() - df["time"].min()).days
    actual_days = min(days, available_days)

    print(f"\n  {symbol}: {len(df)} bars ({available_days}d available, using {actual_days}d)")

    # Indicator grid (compute indicators once per param set)
    ind_combos = list(product(EMA_S_GRID, EMA_L_GRID, MACD_F_GRID, MACD_SL_GRID, ST_F_GRID))
    strategy_combos = list(product(SL_MULTS, TRAIL_PROFILES.keys(), MIN_SCORES))

    print(f"    {len(ind_combos)} indicator combos × {len(strategy_combos)} strategy combos = {len(ind_combos) * len(strategy_combos)} total")

    all_results = []
    tested_ind = 0

    for ema_s, ema_l, macd_f, macd_sl, st_f in ind_combos:
        if ema_s >= ema_l: continue  # Skip invalid combos
        tested_ind += 1

        # Build indicator config
        icfg = {
            "EMA_S": ema_s, "EMA_L": ema_l, "EMA_T": ema_l * 2,
            "ST_F": st_f, "ST_ATR": max(10, ema_s // 2),
            "MACD_F": macd_f, "MACD_SL": macd_sl, "MACD_SIG": max(7, macd_f // 3),
            "ATR_LEN": max(14, ema_s // 3),
        }

        # Compute indicators with this config
        start_idx = max(int(df[df["time"] >= cutoff].index[0]) if len(df[df["time"] >= cutoff]) > 0 else 0,
                        icfg["EMA_T"] + 50)
        if start_idx >= len(df) - 100: continue

        try:
            ind = _compute_indicators(df, icfg)
        except Exception:
            continue

        bars = precompute_bars(df, ind, scfg, start_idx)
        if len(bars) < 100: continue

        # Run all strategy combos on this indicator set
        for sl_m, trail_name, min_score in strategy_combos:
            trail = TRAIL_PROFILES[trail_name]
            r = run_combo(bars, pt, tv, spread, sl_m, trail, min_score)

            if r["trades"] < 20: continue  # Need minimum trades
            if r["pf"] < 1.0: continue     # Skip losers early

            score = r["pf"] * 0.6 - r["dd"] * 0.02 + min(r["trades"] / 500, 0.3)
            all_results.append({
                "ind": icfg.copy(),
                "sl": sl_m, "trail": trail_name, "min_score": min_score,
                "result": r, "score": score,
            })

        # Progress
        if tested_ind % 50 == 0:
            print(f"    ... {tested_ind}/{len(ind_combos)} indicator combos tested, {len(all_results)} viable configs", flush=True)

    elapsed = _time.time() - t0
    print(f"    Done in {elapsed:.0f}s. {len(all_results)} viable configs found.")

    # Sort by composite score and return top N
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top = all_results[:top_n]

    if top:
        print(f"\n    TOP {min(top_n, len(top))} CONFIGS:")
        print(f"    {'EMA_S':>5} {'EMA_L':>5} {'MACD_F':>6} {'MACD_SL':>7} {'ST_F':>4} {'SL':>4} {'Trail':<6} {'Min':>4} | {'PF':>5} {'WR%':>5} {'Ret%':>7} {'DD%':>5} {'Trades':>6}")
        for cfg in top:
            ic = cfg["ind"]; r = cfg["result"]
            print(f"    {ic['EMA_S']:>5} {ic['EMA_L']:>5} {ic['MACD_F']:>6} {ic['MACD_SL']:>7} {ic['ST_F']:>4.1f} "
                  f"{cfg['sl']:>4.1f} {cfg['trail']:<6} {cfg['min_score']:>4.1f} | "
                  f"{r['pf']:>5.2f} {r['wr']:>4.1f}% {r['ret']:>+6.1f}% {r['dd']:>4.1f}% {r['trades']:>6}")

    return symbol, top


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--top", type=int, default=3, help="Top N configs per symbol")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else sorted(M15_SYMBOLS.keys())

    print("=" * 110)
    print(f"  M15 GRID TUNER — Full indicator + strategy parameter search")
    print(f"  {len(symbols)} symbols | {args.days}d | ~34K combos/symbol")
    print("=" * 110)

    all_top = {}
    for sym in symbols:
        sym_name, top = tune_symbol(sym, args.days, args.top)
        all_top[sym_name] = top

    print(f"\n{'='*110}")
    print(f"  FINAL M15 TUNED PARAMETERS — Copy to config.py")
    print(f"{'='*110}")
    print("\nIND_M15_OVERRIDES = {")
    for sym in sorted(all_top.keys()):
        top = all_top[sym]
        if top:
            best = top[0]
            ic = best["ind"]; r = best["result"]
            print(f'    "{sym}": {{"EMA_S": {ic["EMA_S"]}, "EMA_L": {ic["EMA_L"]}, "EMA_T": {ic["EMA_T"]}, '
                  f'"ST_F": {ic["ST_F"]}, "ST_ATR": {ic["ST_ATR"]}, '
                  f'"MACD_F": {ic["MACD_F"]}, "MACD_SL": {ic["MACD_SL"]}, "MACD_SIG": {ic["MACD_SIG"]}, '
                  f'"ATR_LEN": {ic["ATR_LEN"]}}},  # PF={r["pf"]:.2f} WR={r["wr"]:.1f}% DD={r["dd"]:.1f}% ({r["trades"]}t)')
    print("}")

    print("\nSYMBOL_ATR_SL_OVERRIDE_M15 = {")
    for sym in sorted(all_top.keys()):
        top = all_top[sym]
        if top:
            best = top[0]; r = best["result"]
            print(f'    "{sym}": {best["sl"]},  # PF={r["pf"]:.2f} trail={best["trail"]} min_score={best["min_score"]}')
    print("}")

    print("\nDRAGON_M15_SYMBOL_MIN_SCORE = {")
    for sym in sorted(all_top.keys()):
        top = all_top[sym]
        if top:
            best = top[0]; r = best["result"]
            ms = best["min_score"]
            print(f'    "{sym}": {{"trending": {ms}, "ranging": {ms+0.5}, "volatile": {ms+0.5}, "low_vol": {ms+0.5}}},  # PF={r["pf"]:.2f}')
    print("}")
