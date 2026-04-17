"""
Seed trade_journal.db with backtest trades so the LearningEngine has 500+ trades.
Adds a 'source' column to distinguish live vs backtest trades.
Does NOT overwrite existing live trades.
"""
import sys, pickle, sqlite3, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.momentum_scorer import _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, REGIME_PARAMS, DEFAULT_PARAMS
from backtest.dragon_backtest import (
    get_regime, get_adaptive_min_score, ALL_SYMBOLS,
    SYMBOL_ATR_SL_OVERRIDE, SYMBOL_SESSION_OVERRIDE,
    SLIP, RISK_PCT, DAILY_LOSS_LIMIT, CONSEC_LOSS_COOLDOWN, START_EQ,
)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
JOURNAL = Path("/Users/ashish/Documents/beast-trader/data/trade_journal.db")

DAYS = 365


def ensure_source_column(conn):
    """Add source column if missing, mark existing trades as 'live'."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()]
    if "source" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN source TEXT DEFAULT 'live'")
        conn.execute("UPDATE trades SET source='live' WHERE source IS NULL")
        conn.commit()
        print(f"  Added 'source' column — {conn.execute('SELECT COUNT(*) FROM trades').fetchone()[0]} existing trades marked 'live'")


def clear_old_backtest(conn):
    """Remove previous backtest seeds so re-runs are idempotent."""
    deleted = conn.execute("DELETE FROM trades WHERE source='backtest'").rowcount
    conn.commit()
    if deleted:
        print(f"  Cleared {deleted} old backtest trades")


def run_and_collect(symbol):
    """Run dragon backtest and return list of trade dicts."""
    scfg = ALL_SYMBOLS[symbol]
    cache_path = CACHE / scfg["cache"]
    if not cache_path.exists():
        return []

    df = pickle.load(open(cache_path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    pt = scfg["point"]; tv = scfg["tv"]; spread = scfg["spread"]
    cat = scfg["cat"]; sl_cap = 5000 * pt
    icfg = dict(IND_DEFAULTS); icfg.update(IND_OVERRIDES.get(symbol, {}))
    cutoff = df["time"].max() - pd.Timedelta(days=DAYS)
    start_idx = max(int(df[df["time"] >= cutoff].index[0]), icfg["EMA_T"] + 30)
    ind = _compute_indicators(df, icfg)
    n = ind["n"]

    eq = START_EQ; peak = START_EQ
    in_trade = False; d = 0; entry = 0; pos_sl = 0; sl_dist = 0
    trade_lot = 0.0
    consec_losses = 0; cooldown_until = 0
    daily_pnl = 0.0; current_day = None; day_eq_start = START_EQ; day_stopped = False

    np.random.seed(42)

    trades = []
    entry_bar = 0
    entry_time = None
    entry_score = 0.0
    entry_regime = ""

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

        # MANAGE
        if in_trade:
            if (d == 1 and ind["l"][i] <= pos_sl) or (d == -1 and ind["h"][i] >= pos_sl):
                exit_cost = (spread + SLIP * pt)
                pnl = d * (pos_sl - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
                eq += pnl; daily_pnl += pnl
                r_val = pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
                exit_price = pos_sl
                duration = i - entry_bar

                trades.append({
                    "timestamp": str(bar_time),
                    "symbol": symbol, "direction": "LONG" if d == 1 else "SHORT",
                    "entry_price": entry, "exit_price": exit_price,
                    "pnl": round(pnl, 4), "risk_pct": RISK_PCT,
                    "score": entry_score, "regime": entry_regime,
                    "gate": "backtest", "duration_bars": duration,
                    "r_multiple": round(r_val, 4),
                    "session_hour": bar_hour,
                    "day_of_week": bar_time.weekday() if hasattr(bar_time, "weekday") else 0,
                    "exit_reason": "SL" if pnl < 0 else "trailing",
                })

                if pnl > 0:
                    consec_losses = 0
                else:
                    consec_losses += 1
                    if consec_losses >= 3:
                        cooldown_until = i + CONSEC_LOSS_COOLDOWN
                        consec_losses = 0

                peak = max(peak, eq)
                in_trade = False
                if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                    day_stopped = True
                continue

            # Trailing SL update
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

        if day_stopped: continue
        if i < cooldown_until: continue

        # SCORE
        bi = i - 1
        if bi < 21: continue
        ls, ss = _score(ind, bi)
        regime = get_regime(ind, bi)
        adaptive_min = get_adaptive_min_score(regime, symbol=symbol)
        buy = ls >= adaptive_min
        sell = ss >= adaptive_min
        if not buy and not sell: continue
        new_dir = 1 if (buy and (not sell or ls >= ss)) else -1

        # ML filter (50% filtered, same as backtest)
        best_score = max(ls, ss)
        pass_prob = min(1.0, best_score / 10.0)
        if np.random.random() > pass_prob: continue

        # REVERSAL
        if in_trade and new_dir != d:
            exit_cost = (spread + SLIP * pt)
            exit_price = float(ind["c"][i])
            pnl = d * (exit_price - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
            eq += pnl; daily_pnl += pnl
            r_val = pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
            duration = i - entry_bar

            trades.append({
                "timestamp": str(bar_time),
                "symbol": symbol, "direction": "LONG" if d == 1 else "SHORT",
                "entry_price": entry, "exit_price": exit_price,
                "pnl": round(pnl, 4), "risk_pct": RISK_PCT,
                "score": entry_score, "regime": entry_regime,
                "gate": "backtest", "duration_bars": duration,
                "r_multiple": round(r_val, 4),
                "session_hour": bar_hour,
                "day_of_week": bar_time.weekday() if hasattr(bar_time, "weekday") else 0,
                "exit_reason": "reversal",
            })

            if pnl > 0:
                consec_losses = 0
            else:
                consec_losses += 1
                if consec_losses >= 3:
                    cooldown_until = i + CONSEC_LOSS_COOLDOWN
                    consec_losses = 0

            peak = max(peak, eq)
            in_trade = False
            if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                day_stopped = True
                continue

        # ENTRY
        if not in_trade:
            d = new_dir
            sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
            sym_sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, 1.5)
            sl_dist = max(atr_val * sl_m, atr_val * sym_sl_mult)
            sl_dist = min(sl_dist, sl_cap)

            risk_amount = eq * RISK_PCT
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
            entry_bar = i
            entry_time = bar_time
            entry_score = best_score
            entry_regime = regime

    # Close open trade at end
    if in_trade:
        exit_price = float(ind["c"][n - 1])
        pnl = d * (exit_price - entry) / pt * tv * trade_lot
        r_val = pnl / (RISK_PCT * day_eq_start) if day_eq_start > 0 else 0
        bar_time = df["time"].iloc[n - 1]
        trades.append({
            "timestamp": str(bar_time),
            "symbol": symbol, "direction": "LONG" if d == 1 else "SHORT",
            "entry_price": entry, "exit_price": exit_price,
            "pnl": round(pnl, 4), "risk_pct": RISK_PCT,
            "score": entry_score, "regime": entry_regime,
            "gate": "backtest", "duration_bars": n - 1 - entry_bar,
            "r_multiple": round(r_val, 4),
            "session_hour": bar_time.hour if hasattr(bar_time, "hour") else 0,
            "day_of_week": bar_time.weekday() if hasattr(bar_time, "weekday") else 0,
            "exit_reason": "eod_close",
        })

    return trades


def main():
    conn = sqlite3.connect(str(JOURNAL))
    ensure_source_column(conn)
    clear_old_backtest(conn)

    live_count = conn.execute("SELECT COUNT(*) FROM trades WHERE source='live'").fetchone()[0]
    print(f"\n  Live trades preserved: {live_count}")

    total = 0
    print(f"\n{'Symbol':<14} {'Trades':>7} {'Wins':>6} {'PnL':>10}")
    print("-" * 42)

    for symbol in sorted(ALL_SYMBOLS.keys()):
        trades = run_and_collect(symbol)
        if not trades:
            continue

        wins = sum(1 for t in trades if t["pnl"] > 0)
        pnl = sum(t["pnl"] for t in trades)

        for t in trades:
            conn.execute("""
                INSERT INTO trades (timestamp, symbol, direction, entry_price, exit_price,
                    pnl, risk_pct, score, regime, gate, duration_bars, r_multiple,
                    session_hour, day_of_week, exit_reason, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'backtest')
            """, (
                t["timestamp"], t["symbol"], t["direction"],
                t["entry_price"], t["exit_price"], t["pnl"],
                t["risk_pct"], t["score"], t["regime"], t["gate"],
                t["duration_bars"], t["r_multiple"],
                t["session_hour"], t["day_of_week"], t["exit_reason"],
            ))

        conn.commit()
        total += len(trades)
        print(f"{symbol:<14} {len(trades):>7} {wins:>6} {pnl:>+10.2f}")

    print("-" * 42)
    bt_count = conn.execute("SELECT COUNT(*) FROM trades WHERE source='backtest'").fetchone()[0]
    all_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    print(f"\n  Backtest trades added: {bt_count}")
    print(f"  Live trades preserved: {live_count}")
    print(f"  Total in journal:      {all_count}")
    conn.close()
    print("\n  Done. Learning engine now has enough data for stable adaptive risk.")


if __name__ == "__main__":
    main()
