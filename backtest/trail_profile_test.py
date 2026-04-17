"""
TRAIL PROFILE COMPARISON TEST
Tests 4 trailing SL configurations across 6 Dragon symbols.
Run: python3 -B backtest/trail_profile_test.py
"""
import sys, numpy as np, pandas as pd, pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.momentum_scorer import _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, REGIME_PARAMS, DEFAULT_PARAMS
from backtest.dragon_backtest import (
    ALL_SYMBOLS, get_adaptive_min_score, get_regime,
    SYMBOL_SESSION_OVERRIDE, SYMBOL_ATR_SL_OVERRIDE, SYMBOL_MIN_SCORE_OVERRIDE
)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
START_EQ = 1000.0
SLIP = 0.0
RISK_PCT = 0.003
DAILY_LOSS_LIMIT = 0.01
CONSEC_LOSS_COOLDOWN = 24


def apply_trail(profile_name, profit_r, cur, entry, atr_val, sl_dist, d, pt):
    """Return new_sl based on trail profile, or None if no update."""
    new_sl = None

    if profile_name == "CURRENT":
        if profit_r >= 6.0:   new_sl = cur - 0.7 * atr_val * d
        elif profit_r >= 4.0: new_sl = cur - 1.0 * atr_val * d
        elif profit_r >= 2.5: new_sl = cur - 1.5 * atr_val * d
        elif profit_r >= 1.5:
            new_sl = cur - 2.0 * atr_val * d
            floor = entry + 0.5 * sl_dist * d
            new_sl = max(new_sl, floor) if d == 1 else min(new_sl, floor)
        elif profit_r >= 1.0: new_sl = entry + 0.5 * sl_dist * d
        elif profit_r >= 0.5: new_sl = entry + 2 * pt * d

    elif profile_name == "AGGRESSIVE":
        if profit_r >= 8.0:   new_sl = cur - 0.5 * atr_val * d
        elif profit_r >= 4.0: new_sl = cur - 0.7 * atr_val * d
        elif profit_r >= 2.0: new_sl = cur - 1.0 * atr_val * d
        elif profit_r >= 1.5:
            new_sl = cur - 1.5 * atr_val * d
            floor = entry + 0.5 * sl_dist * d
            new_sl = max(new_sl, floor) if d == 1 else min(new_sl, floor)
        elif profit_r >= 0.8: new_sl = entry + 0.2 * sl_dist * d

    elif profile_name == "WIDE":
        if profit_r >= 6.0:   new_sl = cur - 1.0 * atr_val * d
        elif profit_r >= 4.0: new_sl = cur - 1.5 * atr_val * d
        elif profit_r >= 2.5: new_sl = cur - 2.5 * atr_val * d
        elif profit_r >= 1.5:
            new_sl = cur - 3.0 * atr_val * d
            floor = entry + 0.5 * sl_dist * d
            new_sl = max(new_sl, floor) if d == 1 else min(new_sl, floor)
        elif profit_r >= 1.0: new_sl = entry + 0.5 * sl_dist * d
        elif profit_r >= 0.5: new_sl = entry + 2 * pt * d

    elif profile_name == "NO_LOCK":
        if profit_r >= 6.0:   new_sl = cur - 0.7 * atr_val * d
        elif profit_r >= 4.0: new_sl = cur - 1.0 * atr_val * d
        elif profit_r >= 2.5: new_sl = cur - 1.5 * atr_val * d
        elif profit_r >= 1.5: new_sl = cur - 2.0 * atr_val * d
        elif profit_r >= 1.0: new_sl = cur - 2.5 * atr_val * d
        elif profit_r >= 0.5: new_sl = cur - 3.0 * atr_val * d

    return new_sl


def run_with_profile(symbol, days, use_ml_filter, profile_name):
    scfg = ALL_SYMBOLS[symbol]
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
    n_trades = 0; wins = 0; gross_p = 0; gross_l = 0
    in_trade = False; d = 0; entry = 0; pos_sl = 0; sl_dist = 0
    trade_lot = 0.0
    consec_losses = 0; cooldown_until = 0
    daily_pnl = 0.0; current_day = None; day_eq_start = START_EQ; day_stopped = False
    r_multiples = []; max_consec_loss = 0; current_streak = 0
    np.random.seed(42)

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
            current_day = bar_date; day_eq_start = eq; daily_pnl = 0.0; day_stopped = False

        # MANAGE: trailing SL
        if in_trade:
            if (d == 1 and ind["l"][i] <= pos_sl) or (d == -1 and ind["h"][i] >= pos_sl):
                exit_cost = (spread + SLIP * pt)
                pnl = d * (pos_sl - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
                eq += pnl; daily_pnl += pnl
                r_val = pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
                r_multiples.append(r_val)
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
                    day_stopped = True
                continue

            cur = float(ind["c"][i])
            profit_r = ((cur - entry) * d) / sl_dist if sl_dist > 0 else 0

            # === TRAIL PROFILE ===
            new_sl = apply_trail(profile_name, profit_r, cur, entry, atr_val, sl_dist, d, pt)

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

        if use_ml_filter:
            best_score = max(ls, ss)
            pass_prob = min(1.0, best_score / 10.0)
            if np.random.random() > pass_prob:
                continue

        if in_trade and new_dir != d:
            exit_cost = (spread + SLIP * pt)
            pnl = d * (float(ind["c"][i]) - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
            eq += pnl; daily_pnl += pnl
            r_val = pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
            r_multiples.append(r_val)
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

        if not in_trade:
            d = new_dir
            sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
            sym_sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, 1.5)
            sl_dist = max(atr_val * sl_m, atr_val * sym_sl_mult)
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

    if in_trade:
        pnl = d * (float(ind["c"][n - 1]) - entry) / pt * tv * trade_lot
        eq += pnl
        r_val = pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
        r_multiples.append(r_val)
        if pnl > 0:
            gross_p += pnl; wins += 1
        else:
            gross_l += abs(pnl)
        n_trades += 1; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)

    pf = gross_p / gross_l if gross_l > 0 else (999 if gross_p > 0 else 0)
    dd = max_dd / peak * 100 if peak else 0
    ret = (eq - START_EQ) / START_EQ * 100
    wr = wins / n_trades * 100 if n_trades else 0
    avg_r = np.mean(r_multiples) if r_multiples else 0
    std_r = np.std(r_multiples) if len(r_multiples) > 1 else 1
    sharpe = (avg_r / std_r) * np.sqrt(252) if std_r > 0 else 0
    return {
        "sym": symbol, "trades": n_trades, "wr": round(wr, 1), "pf": round(pf, 2),
        "ret": round(ret, 1), "dd": round(dd, 1), "eq": round(eq, 2),
        "gross_p": round(gross_p, 2), "gross_l": round(gross_l, 2),
        "sharpe": round(sharpe, 2), "avg_r": round(avg_r, 3),
    }


# ===================== MAIN =====================
if __name__ == "__main__":
    DRAGON_SYMS = ["XAUUSD", "XAGUSD", "BTCUSD", "NAS100.r", "JPN225ft", "USDJPY"]
    PROFILES = ["CURRENT", "AGGRESSIVE", "WIDE", "NO_LOCK"]

    PROFILE_DESC = {
        "CURRENT":    ">=6R:0.7ATR  >=4R:1.0ATR  >=2.5R:1.5ATR  >=1.5R:2.0ATR(floor0.5R)  >=1R:lock0.5R  >=0.5R:lockBE",
        "AGGRESSIVE": ">=8R:0.5ATR  >=4R:0.7ATR  >=2R:1.0ATR    >=1.5R:1.5ATR(floor0.5R)  >=0.8R:lock0.2R",
        "WIDE":       ">=6R:1.0ATR  >=4R:1.5ATR  >=2.5R:2.5ATR  >=1.5R:3.0ATR(floor0.5R)  >=1R:lock0.5R  >=0.5R:lockBE",
        "NO_LOCK":    ">=6R:0.7ATR  >=4R:1.0ATR  >=2.5R:1.5ATR  >=1.5R:2.0ATR  >=1R:2.5ATR  >=0.5R:3.0ATR (pure trail)",
    }

    all_results = {}

    for prof in PROFILES:
        print(f"\n{'=' * 105}")
        print(f"  TRAIL PROFILE: {prof}")
        print(f"  {PROFILE_DESC[prof]}")
        print(f"{'=' * 105}")
        print(f"{'Symbol':<12} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Return%':>9} {'DD%':>7} {'Final$':>10} {'Sharpe':>8} {'AvgR':>7}")
        print("-" * 80)

        results = []
        for sym in DRAGON_SYMS:
            ml = sym in ["XAUUSD", "XAGUSD"]
            r = run_with_profile(sym, 365, ml, prof)
            if r:
                results.append(r)
                print(f"{r['sym']:<12} {r['trades']:>7} {r['wr']:>6.1f}% {r['pf']:>7.2f} {r['ret']:>8.1f}% {r['dd']:>6.1f}% ${r['eq']:>9.2f} {r['sharpe']:>8.2f} {r['avg_r']:>+7.3f}")

        if results:
            gp = sum(r["gross_p"] for r in results)
            gl = sum(r["gross_l"] for r in results)
            tot_ret = sum(r["ret"] for r in results) / len(results)
            avg_dd = sum(r["dd"] for r in results) / len(results)
            avg_sharpe = sum(r["sharpe"] for r in results) / len(results)
            port_pf = gp / gl if gl > 0 else 0
            print("-" * 80)
            print(f"{'PORTFOLIO':<12} {'':>7} {'':>7} {port_pf:>7.2f} {tot_ret:>8.1f}% {avg_dd:>6.1f}% {'':>10} {avg_sharpe:>8.2f}")
        all_results[prof] = results

    # ===================== COMPARISON SUMMARY =====================
    print(f"\n\n{'=' * 110}")
    print("  COMPARISON SUMMARY — ALL TRAIL PROFILES (PF / Return%)")
    print(f"{'=' * 110}")
    print(f"{'Profile':<14}", end="")
    for sym in DRAGON_SYMS:
        print(f" {sym:>14}", end="")
    print(f" {'PORT_PF':>10} {'AVG_RET':>10} {'AVG_DD':>10}")
    print("-" * 110)

    for prof in PROFILES:
        results = all_results.get(prof, [])
        if not results:
            continue
        sym_map = {r["sym"]: r for r in results}
        print(f"{prof:<14}", end="")
        for sym in DRAGON_SYMS:
            r = sym_map.get(sym)
            if r:
                print(f" {r['pf']:>5.2f}/{r['ret']:>+6.0f}%", end="")
            else:
                print(f" {'---':>14}", end="")
        gp = sum(r["gross_p"] for r in results)
        gl = sum(r["gross_l"] for r in results)
        port_pf = gp / gl if gl > 0 else 0
        avg_ret = sum(r["ret"] for r in results) / len(results)
        avg_dd = sum(r["dd"] for r in results) / len(results)
        print(f" {port_pf:>10.2f} {avg_ret:>9.1f}% {avg_dd:>9.1f}%")

    # ===================== PER-SYMBOL WINNERS =====================
    print(f"\n{'=' * 110}")
    print("  PER-SYMBOL BEST PROFILE:")
    print(f"{'=' * 110}")
    for sym in DRAGON_SYMS:
        best_prof = None; best_pf = 0
        for prof in PROFILES:
            results = all_results.get(prof, [])
            sym_map = {r["sym"]: r for r in results}
            r = sym_map.get(sym)
            if r and r["pf"] > best_pf:
                best_pf = r["pf"]; best_prof = prof
        if best_prof:
            sym_map = {r["sym"]: r for r in all_results[best_prof]}
            r = sym_map[sym]
            print(f"  {sym:<12} -> {best_prof:<14} PF={r['pf']:.2f}  WR={r['wr']:.1f}%  Ret={r['ret']:+.1f}%  DD={r['dd']:.1f}%  Sharpe={r['sharpe']:.2f}")

    # ===================== PORTFOLIO-LEVEL WINNER =====================
    print(f"\n{'=' * 110}")
    print("  PORTFOLIO-LEVEL WINNER:")
    print(f"{'=' * 110}")
    best_port_prof = None; best_port_pf = 0
    for prof in PROFILES:
        results = all_results.get(prof, [])
        if not results:
            continue
        gp = sum(r["gross_p"] for r in results)
        gl = sum(r["gross_l"] for r in results)
        port_pf = gp / gl if gl > 0 else 0
        avg_ret = sum(r["ret"] for r in results) / len(results)
        avg_dd = sum(r["dd"] for r in results) / len(results)
        tag = " <-- BEST" if port_pf > best_port_pf else ""
        if port_pf > best_port_pf:
            best_port_pf = port_pf; best_port_prof = prof
        print(f"  {prof:<14} Portfolio PF={port_pf:.2f}  AvgRet={avg_ret:+.1f}%  AvgDD={avg_dd:.1f}%{tag}")

    print(f"\n  RECOMMENDATION: {best_port_prof}")
    print(f"{'=' * 110}")
