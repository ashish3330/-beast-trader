"""
DRAGON 3-SUB BACKTEST — validates the 3-sub-position architecture.
Same signals as dragon_backtest.py but with:
- Sub0: 50% lot @ TP1 (2R) — auto-close at 2R
- Sub1: 30% lot @ TP2 (3R) — auto-close at 3R
- Sub2: 20% lot @ trailing — BE+0.2R lock after TP1, then trail
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
RISK_PCT = 0.003
DAILY_LOSS_LIMIT = 0.01
CONSEC_LOSS_COOLDOWN = 24

# 3-sub splits
SUB_SPLITS = [0.50, 0.30, 0.20]
SUB_TP_R   = [2.0,  3.0,  999.0]  # Sub2 has no fixed TP — trailing exits

# Dragon's 6 live symbols only
DRAGON_SYMBOLS = {
    "XAUUSD":   {"cache": "raw_h1_xauusd.pkl",   "point": 0.01,    "tv": 1.0,     "spread": 0.33,   "cat": "Gold"},
    "XAGUSD":   {"cache": "raw_h1_XAGUSD.pkl",   "point": 0.001,   "tv": 5.0,     "spread": 0.035,  "cat": "Gold"},
    "BTCUSD":   {"cache": "raw_h1_BTCUSD.pkl",   "point": 0.01,    "tv": 0.01,    "spread": 17.0,   "cat": "Crypto"},
    "NAS100.r": {"cache": "raw_h1_NAS100_r.pkl", "point": 0.01,    "tv": 0.01,    "spread": 1.80,   "cat": "Index"},
    "JPN225ft": {"cache": "raw_h1_JPN225ft.pkl", "point": 0.01,    "tv": 0.0063,  "spread": 10.0,   "cat": "Index"},
    "USDJPY":   {"cache": "raw_h1_USDJPY.pkl",   "point": 0.001,   "tv": 0.63,    "spread": 0.018,  "cat": "Forex"},
}

def get_adaptive_min_score(regime):
    return {"trending": 6.0, "ranging": 8.0, "volatile": 7.0, "low_vol": 7.0}.get(regime, 7.0)

def get_regime(ind, bi):
    if bi < 21 or np.isnan(ind["bbw"][bi]): return "unknown"
    bbw = float(ind["bbw"][bi])
    adx = float(ind["adx"][bi]) if not np.isnan(ind["adx"][bi]) else 0
    if bbw < 1.5 and adx < 20: return "ranging"
    if 1.5 <= bbw < 3.0 and adx > 25: return "trending"
    if bbw >= 3.0: return "volatile"
    return "low_vol"


def run(symbol, days=365):
    scfg = DRAGON_SYMBOLS[symbol]
    cache_path = CACHE / scfg["cache"]
    if not cache_path.exists():
        return None
    df = pickle.load(open(cache_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    pt = scfg["point"]; tv = scfg["tv"]; spread = scfg["spread"]; cat = scfg["cat"]
    sl_cap = 5000 * pt
    icfg = dict(IND_DEFAULTS); icfg.update(IND_OVERRIDES.get(symbol, {}))
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    start_idx = max(int(df[df["time"] >= cutoff].index[0]), icfg["EMA_T"] + 30)
    ind = _compute_indicators(df, icfg)
    n = ind["n"]

    eq = START_EQ; peak = START_EQ; max_dd = 0
    n_entries = 0; wins = 0; gross_p = 0; gross_l = 0

    # Sub-position state: list of 3 subs per trade
    # Each sub: {"active": bool, "lot_frac": float, "tp_r": float}
    subs = []
    in_trade = False; d = 0; entry = 0; sl_dist = 0; pos_sl = 0
    trade_lot = 0.0
    tp1_hit = False  # track if Sub0 closed

    consec_losses = 0; cooldown_until = 0
    daily_pnl = 0.0; current_day = None; day_eq_start = START_EQ; day_stopped = False
    r_multiples = []; max_consec_loss = 0; current_streak = 0
    np.random.seed(42)

    for i in range(start_idx, n):
        atr_val = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0
        if atr_val == 0: continue

        bar_time = df["time"].iloc[i]
        bar_hour = bar_time.hour if hasattr(bar_time, "hour") else 12
        if cat != "Crypto" and (bar_hour >= 22 or bar_hour < 6): continue

        bar_date = bar_time.date() if hasattr(bar_time, "date") else None
        if bar_date and bar_date != current_day:
            current_day = bar_date
            day_eq_start = eq
            daily_pnl = 0.0
            day_stopped = False

        # ═══ MANAGE OPEN SUBS ═══
        if in_trade:
            cur = float(ind["c"][i])
            h = float(ind["h"][i])
            l = float(ind["l"][i])

            # Check SL hit (all remaining subs close)
            sl_hit = (d == 1 and l <= pos_sl) or (d == -1 and h >= pos_sl)
            if sl_hit:
                total_pnl = 0
                for s in subs:
                    if not s["active"]: continue
                    pnl = d * (pos_sl - entry) / pt * tv * (trade_lot * s["lot_frac"])
                    pnl -= (spread + SLIP * pt) / pt * tv * (trade_lot * s["lot_frac"])
                    total_pnl += pnl
                    s["active"] = False

                eq += total_pnl; daily_pnl += total_pnl
                r_val = total_pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
                r_multiples.append(r_val)
                if total_pnl > 0:
                    gross_p += total_pnl; wins += 1; consec_losses = 0; current_streak = 0
                else:
                    gross_l += abs(total_pnl); consec_losses += 1; current_streak += 1
                    max_consec_loss = max(max_consec_loss, current_streak)
                    if consec_losses >= 3:
                        cooldown_until = i + CONSEC_LOSS_COOLDOWN; consec_losses = 0

                n_entries += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
                in_trade = False; subs = []; tp1_hit = False
                if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                    day_stopped = True
                continue

            # Check TP hits for each sub
            profit_dist = (cur - entry) * d
            for si_idx, s in enumerate(subs):
                if not s["active"]: continue
                tp_dist = sl_dist * s["tp_r"]
                # Check if TP was hit this bar
                if d == 1 and h >= entry + tp_dist:
                    exit_price = entry + tp_dist
                    pnl = (exit_price - entry) / pt * tv * (trade_lot * s["lot_frac"])
                    pnl -= (spread + SLIP * pt) / pt * tv * (trade_lot * s["lot_frac"])
                    eq += pnl; daily_pnl += pnl; gross_p += pnl
                    s["active"] = False
                    if si_idx == 0: tp1_hit = True
                elif d == -1 and l <= entry - tp_dist:
                    exit_price = entry - tp_dist
                    pnl = (entry - exit_price) / pt * tv * (trade_lot * s["lot_frac"])
                    pnl -= (spread + SLIP * pt) / pt * tv * (trade_lot * s["lot_frac"])
                    eq += pnl; daily_pnl += pnl; gross_p += pnl
                    s["active"] = False
                    if si_idx == 0: tp1_hit = True

            # If TP1 hit, move SL to BE+0.2R for remaining subs
            if tp1_hit and any(s["active"] for s in subs):
                be_sl = entry + 0.2 * sl_dist * d
                if d == 1 and be_sl > pos_sl:
                    pos_sl = be_sl
                elif d == -1 and be_sl < pos_sl:
                    pos_sl = be_sl

            # Check if all subs closed (count as win)
            if not any(s["active"] for s in subs):
                # All TPs hit — full win
                wins += 1; consec_losses = 0; current_streak = 0
                n_entries += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
                r_val = (eq - (eq - daily_pnl)) / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
                r_multiples.append(r_val)
                in_trade = False; subs = []; tp1_hit = False
                continue

            # Trail remaining subs (same logic as single-position)
            profit_r = profit_dist / sl_dist if sl_dist > 0 else 0
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

        if day_stopped: continue
        if i < cooldown_until: continue

        # ═══ SCORE ═══
        bi = i - 1
        if bi < 21: continue
        ls, ss = _score(ind, bi)
        regime = get_regime(ind, bi)
        adaptive_min = get_adaptive_min_score(regime)
        buy = ls >= adaptive_min
        sell = ss >= adaptive_min
        if not buy and not sell: continue
        new_dir = 1 if (buy and (not sell or ls >= ss)) else -1

        # ML filter (50% rejection)
        best_score = max(ls, ss)
        pass_prob = min(1.0, best_score / 10.0)
        if np.random.random() > pass_prob: continue

        # Spread filter: reject if spread > 30% of ATR
        if atr_val > 0 and spread / atr_val > 0.3: continue

        # REVERSAL: close all subs, enter new direction
        if in_trade and new_dir != d:
            total_pnl = 0
            for s in subs:
                if not s["active"]: continue
                pnl = d * (cur - entry) / pt * tv * (trade_lot * s["lot_frac"])
                pnl -= (spread + SLIP * pt) / pt * tv * (trade_lot * s["lot_frac"])
                total_pnl += pnl
                s["active"] = False
            eq += total_pnl; daily_pnl += total_pnl
            r_val = total_pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
            r_multiples.append(r_val)
            if total_pnl > 0: gross_p += total_pnl; wins += 1; consec_losses = 0; current_streak = 0
            else:
                gross_l += abs(total_pnl); consec_losses += 1; current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
                if consec_losses >= 3:
                    cooldown_until = i + CONSEC_LOSS_COOLDOWN; consec_losses = 0
            n_entries += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
            in_trade = False; subs = []; tp1_hit = False
            if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                day_stopped = True; continue

        # ═══ ENTRY: 3 subs ═══
        if not in_trade:
            d = new_dir
            sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
            sl_dist = max(atr_val * sl_m, atr_val * 1.5)
            sl_dist = min(sl_dist, sl_cap)

            risk_amount = eq * RISK_PCT
            pip_value_per_lot = (sl_dist / pt) * tv
            if pip_value_per_lot > 0:
                trade_lot = max(risk_amount / pip_value_per_lot, 0.01)
            else:
                trade_lot = 0.01

            entry_cost = (spread + SLIP * pt)
            entry = float(ind["o"][i]) + entry_cost / 2 * d
            pos_sl = entry - sl_dist * d
            in_trade = True
            tp1_hit = False

            subs = [
                {"active": True, "lot_frac": SUB_SPLITS[0], "tp_r": SUB_TP_R[0]},
                {"active": True, "lot_frac": SUB_SPLITS[1], "tp_r": SUB_TP_R[1]},
                {"active": True, "lot_frac": SUB_SPLITS[2], "tp_r": SUB_TP_R[2]},
            ]

    # Close remaining at end
    if in_trade:
        cur = float(ind["c"][n-1])
        total_pnl = 0
        for s in subs:
            if not s["active"]: continue
            pnl = d * (cur - entry) / pt * tv * (trade_lot * s["lot_frac"])
            total_pnl += pnl
        eq += total_pnl
        if total_pnl > 0: gross_p += total_pnl; wins += 1
        else: gross_l += abs(total_pnl)
        n_entries += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
        r_val = total_pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
        r_multiples.append(r_val)

    pf = gross_p / gross_l if gross_l > 0 else (999 if gross_p > 0 else 0)
    dd = max_dd / peak * 100 if peak else 0
    ret = (eq - START_EQ) / START_EQ * 100
    wr = wins / n_entries * 100 if n_entries else 0
    avg_r = np.mean(r_multiples) if r_multiples else 0
    std_r = np.std(r_multiples) if len(r_multiples) > 1 else 1
    sharpe = (avg_r / std_r) * np.sqrt(252) if std_r > 0 else 0

    return {"sym": symbol, "trades": n_entries, "wr": round(wr, 1), "pf": round(pf, 2),
            "ret": round(ret, 1), "dd": round(dd, 1), "eq": round(eq, 2),
            "sharpe": round(sharpe, 2), "max_consec_loss": max_consec_loss}


if __name__ == "__main__":
    print("=" * 100)
    print("  DRAGON 3-SUB BACKTEST | 50%@2R + 30%@3R + 20%@trail | BE+0.2R lock after TP1")
    print("  $1,000 | 365d | 0.3% risk | Spread filter 30% ATR | ML filter 50%")
    print("=" * 100)
    print(f"\n{'Symbol':<12} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Return%':>9} {'DD%':>7} {'Final$':>10} {'Sharpe':>8} {'MaxStreak':>10}")
    print("-" * 85)

    results = []
    for sym in sorted(DRAGON_SYMBOLS.keys()):
        r = run(sym, 365)
        if r:
            results.append(r)
            print(f"{r['sym']:<12} {r['trades']:>7} {r['wr']:>6.1f}% {r['pf']:>7.2f} {r['ret']:>8.1f}% {r['dd']:>6.1f}% ${r['eq']:>9.2f} {r['sharpe']:>8.2f} {r['max_consec_loss']:>10}")

    print("-" * 85)
    if results:
        gp = sum(max(0, r["eq"] - START_EQ) for r in results if r["eq"] > START_EQ)
        gl = sum(max(0, START_EQ - r["eq"]) for r in results if r["eq"] < START_EQ)
        total_pf = gp / gl if gl > 0 else 999
        avg_ret = np.mean([r["ret"] for r in results])
        avg_dd = np.mean([r["dd"] for r in results])
        print(f"{'PORTFOLIO':<12} {'':>7} {'':>7} {total_pf:>7.2f} {avg_ret:>8.1f}% {avg_dd:>6.1f}%")
    print("=" * 100)
