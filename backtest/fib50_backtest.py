#!/usr/bin/env python3 -B
"""
Fib-50 Pullback Continuation backtester — validation gate before flipping
FIB50_TRADE_LIVE = True.

Mirrors backtest/sma_breakout_backtest.py:
  * Imports Fib50Strategy from agent/fib50_strategy.py AS-IS (no detector
    re-implementation).
  * Walks the M15 cache; at each closed bar a FakeState serves a trailing
    window with a synthetic forming bar appended so iloc[-2] of what the
    detector sees is the candidate bar (live-shape mimic).
  * Simulates each fired signal through TP1 / TP2 / SL.

CLI:
    python3 -B backtest/fib50_backtest.py
    python3 -B backtest/fib50_backtest.py --symbol XAUUSD --days 180
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.fib50_strategy import Fib50Strategy  # noqa: E402

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")

SPREAD = {
    "EURUSD": 0.00015, "GBPUSD": 0.00020, "USDJPY": 0.015, "USDCAD": 0.00020,
    "USDCHF": 0.00020, "AUDJPY": 0.020, "EURJPY": 0.020, "GBPJPY": 0.025,
    "XAUUSD": 0.30, "XAGUSD": 0.030, "BTCUSD": 30.0, "ETHUSD": 2.0,
    "NAS100.r": 1.50, "SP500.r": 0.50, "DJ30.r": 2.0, "US2000.r": 0.50,
    "GER40.r": 2.0, "UK100.r": 2.0, "JPN225ft": 10.0, "SPI200.r": 2.0,
    "SWI20.r": 3.0, "XPTUSD.r": 1.0, "USOUSD": 0.05,
}

UNIVERSE = ["XAUUSD", "EURUSD", "BTCUSD", "DJ30.r", "SP500.r", "JPN225ft"]
START_CAPITAL = 5000.0
RISK_PCT = 0.0020                # 0.20%/trade — matches config.FIB50_RISK_PCT
MIN_BARS_FOR_BT = 500
M15_WINDOW = 400                 # 400 M15 bars = enough for swing detection
TIME_STOP_BARS = 96              # M15 bars (24h) — close runner if peak < 0.5R
TIME_STOP_PEAK_R = 0.5


def _load(symbol, base="m15"):
    for v in (f"raw_{base}_{symbol}.pkl",
              f"raw_{base}_{symbol.replace('.', '_')}.pkl",
              f"raw_{base}_{symbol.lower()}.pkl"):
        p = CACHE / v
        if p.exists():
            df = pickle.load(open(p, "rb")).copy()
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time").reset_index(drop=True)
            try:
                df["time"] = df["time"].dt.as_unit("ns")
            except Exception:
                pass
            cols = ["time", "open", "high", "low", "close"]
            if "tick_volume" in df.columns:
                cols.append("tick_volume")
            return df[cols].astype({c: float for c in cols if c != "time"})
    return None


class _FakeState:
    """Truncate cache at cursor + append a synthetic forming bar so iloc[-2]
    of what the detector sees is the cursor bar (live-shape mimic)."""

    def __init__(self, m15):
        self.m15 = m15
        self.cursor = 0

    def set_cursor(self, i):
        self.cursor = i

    def get_candles(self, symbol, tf):
        if tf != 15:
            return None
        c = self.cursor
        lo = max(0, c + 1 - M15_WINDOW)
        base = self.m15.iloc[lo: c + 1]
        if len(base) == 0:
            return None
        last_close = base["close"].iloc[-1]
        last_time = base["time"].iloc[-1]
        row = {"time": last_time + pd.Timedelta(minutes=15),
               "open": last_close, "high": last_close,
               "low": last_close, "close": last_close}
        return pd.concat([base, pd.DataFrame([row])], ignore_index=True)


def _simulate_trade(sig, m15_arr, entry_i, spread):
    """Walk forward from entry_i+1; return blended R outcome (or None).

    Half lot taken at TP1, half runs to TP2 (or breakeven trail post-TP1).
    """
    H, L, C = m15_arr
    n = len(C)
    d = 1 if sig["direction"] == "LONG" else -1
    entry = sig["entry"] + d * spread
    sl = sig["sl"]
    tp1 = sig["tp1"]
    tp2 = sig["tp2"]
    risk = (entry - sl) if d == 1 else (sl - entry)
    if risk <= 0:
        return None

    tp1_hit = False
    runner_open = True
    r_tp1 = 0.0
    r_run = 0.0
    peak_r = 0.0
    bars_held = 0
    runner_sl = sl

    end = min(n, entry_i + 1 + 800)
    for j in range(entry_i + 1, end):
        bars_held += 1
        hi, lo = H[j], L[j]
        cur_r = (hi - entry) / risk if d == 1 else (entry - lo) / risk
        if cur_r > peak_r:
            peak_r = cur_r

        if d == 1:
            if not tp1_hit:
                if lo <= sl:
                    r_tp1 = -1.0
                    r_run = -1.0
                    runner_open = False
                    break
                if hi >= tp1:
                    r_tp1 = (tp1 - entry) / risk
                    tp1_hit = True
                    runner_sl = entry      # BE post-TP1
            else:
                if lo <= runner_sl:
                    r_run = (runner_sl - entry) / risk
                    runner_open = False
                    break
                if hi >= tp2:
                    r_run = (tp2 - entry) / risk
                    runner_open = False
                    break
        else:
            if not tp1_hit:
                if hi >= sl:
                    r_tp1 = -1.0
                    r_run = -1.0
                    runner_open = False
                    break
                if lo <= tp1:
                    r_tp1 = (entry - tp1) / risk
                    tp1_hit = True
                    runner_sl = entry
            else:
                if hi >= runner_sl:
                    r_run = (entry - runner_sl) / risk
                    runner_open = False
                    break
                if lo <= tp2:
                    r_run = (entry - tp2) / risk
                    runner_open = False
                    break

        if bars_held >= TIME_STOP_BARS and peak_r < TIME_STOP_PEAK_R:
            close_px = C[j]
            r_now = (close_px - entry) / risk if d == 1 else (entry - close_px) / risk
            if not tp1_hit:
                r_tp1 = r_now
                r_run = r_now
            else:
                r_run = r_now
            runner_open = False
            break
    else:
        if runner_open:
            close_px = C[end - 1]
            r_now = (close_px - entry) / risk if d == 1 else (entry - close_px) / risk
            if not tp1_hit:
                r_tp1 = r_now
                r_run = r_now
            else:
                r_run = r_now

    return {"R": 0.5 * r_tp1 + 0.5 * r_run, "tp1_hit": tp1_hit, "dir": d,
            "entry_i": entry_i, "bars_held": bars_held, "peak_r": peak_r}


def backtest_symbol(symbol, days=None):
    m15 = _load(symbol, "m15")
    if m15 is None:
        return {"symbol": symbol, "status": "NO_DATA"}, []
    if days is not None and days > 0:
        n_bars_keep = int(days * 24 * 4)
        if n_bars_keep < len(m15):
            m15 = m15.iloc[-n_bars_keep:].reset_index(drop=True)
    n = len(m15)
    if n < MIN_BARS_FOR_BT:
        return {"symbol": symbol, "status": f"INSUFFICIENT_DATA ({n} bars)"}, []

    spread = SPREAD.get(symbol, 0.0002)
    state = _FakeState(m15)
    strat = Fib50Strategy(state)
    arr = (m15["high"].values, m15["low"].values, m15["close"].values)

    trades = []
    open_until = -1
    start_i = max(M15_WINDOW, 120)
    for i in range(start_i, n - 1):
        if i <= open_until:
            continue
        state.set_cursor(i)
        strat._last_bar_t.pop(symbol, None)
        sig = strat.evaluate(symbol)
        if sig is None:
            continue
        tr = _simulate_trade(sig, arr, entry_i=i, spread=spread)
        if tr is None:
            continue
        trades.append(tr)
        open_until = i + tr["bars_held"]

    return _summarize(symbol, trades, n), trades


def _summarize(symbol, trades, n_bars):
    if not trades:
        return {"symbol": symbol, "status": "NO_SIGNALS",
                "trades": 0, "bars": n_bars}
    Rs = np.array([t["R"] for t in trades])
    wins = Rs[Rs > 0]
    losses = Rs[Rs <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    max_cons = cur = 0
    for r in Rs:
        if r <= 0:
            cur += 1
            max_cons = max(max_cons, cur)
        else:
            cur = 0
    equity = [START_CAPITAL]
    risk_d = START_CAPITAL * RISK_PCT
    for r in Rs:
        equity.append(equity[-1] + r * risk_d)
    eq = np.array(equity)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    return {
        "symbol": symbol, "status": "OK", "bars": n_bars,
        "trades": int(len(trades)),
        "longs": int(sum(1 for t in trades if t["dir"] == 1)),
        "shorts": int(sum(1 for t in trades if t["dir"] == -1)),
        "wr": float((Rs > 0).mean()), "pf": float(pf),
        "avg_R": float(Rs.mean()), "total_R": float(Rs.sum()),
        "max_cons_loss": int(max_cons),
        "best_R": float(Rs.max()), "worst_R": float(Rs.min()),
        "tp1_rate": float(np.mean([t["tp1_hit"] for t in trades])),
        "signals_per_100_bars": float(100.0 * len(trades) / n_bars),
        "end_equity": float(eq[-1]), "max_dd_pct": float(abs(dd.min()) * 100.0),
    }


def _print_row(r):
    if r["status"] != "OK":
        print(f"{r['symbol']:<12} {r['status']}")
        return
    print(f"{r['symbol']:<12} {r['bars']:>6} {r['trades']:>5} "
          f"{r['longs']:>3}/{r['shorts']:<3} "
          f"{r['wr']*100:>5.1f}% {r['tp1_rate']*100:>5.0f}% "
          f"{r['pf']:>6.2f} {r['avg_R']:>+6.2f} {r['total_R']:>+8.1f} "
          f"{r['signals_per_100_bars']:>5.2f} "
          f"{r['best_R']:>+5.2f}/{r['worst_R']:>+5.2f} "
          f"{r['max_cons_loss']:>3} ${r['end_equity']:>8.0f} {r['max_dd_pct']:>5.1f}%")


def main():
    parser = argparse.ArgumentParser(description="FIB50 Backtest")
    parser.add_argument("--days", type=int, default=None,
                        help="Trailing window in days (default: full cache).")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol (default: FIB50_WHITELIST).")
    args = parser.parse_args()

    syms = [args.symbol] if args.symbol else UNIVERSE
    days_str = f"last {args.days}d" if args.days else "full cache"

    print("\n" + "=" * 120)
    print("  FIB50 BACKTEST  —  validation gate before FIB50_TRADE_LIVE=True")
    print("=" * 120)
    print(f"  Universe: {syms}  |  Window: {days_str}")
    print(f"  Risk model: ${START_CAPITAL:.0f} start, "
          f"{RISK_PCT*100:.2f}% risk/trade, FIXED-FRACTIONAL")
    print("-" * 120)
    print(f"{'Symbol':<12} {'Bars':>6} {'Trd':>5} {'L/S':>7} {'WR':>6} {'TP1%':>6} "
          f"{'PF':>6} {'avgR':>6} {'totalR':>8} {'/100b':>5} {'best/worst':>11} "
          f"{'mCL':>3} {'EndEq':>9} {'mDD%':>6}")
    print("-" * 120)

    results = []
    all_trades = []
    for sym in syms:
        r, trs = backtest_symbol(sym, days=args.days)
        results.append(r)
        all_trades.extend(trs)
        _print_row(r)

    print("-" * 120)
    ok = [r for r in results if r.get("status") == "OK"]
    if not ok:
        print("NO SYMBOLS PRODUCED RESULTS — cache is too thin / unavailable. "
              "VERDICT: CANNOT VALIDATE.")
        return

    total_trades = sum(r["trades"] for r in ok)
    total_R = sum(r["total_R"] for r in ok)
    all_Rs = np.array([t["R"] for t in all_trades])
    if len(all_Rs) == 0:
        print("AGGREGATE: 0 trades fired across universe.")
        return
    wins = all_Rs[all_Rs > 0]
    losses = all_Rs[all_Rs <= 0]
    agg_pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    agg_wr = float((all_Rs > 0).mean())
    equity = [START_CAPITAL]
    risk_d = START_CAPITAL * RISK_PCT
    for r in all_Rs:
        equity.append(equity[-1] + r * risk_d)
    eq = np.array(equity)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    agg_dd = float(abs(dd.min()) * 100.0)
    agg_end = float(eq[-1])

    print(f"AGGREGATE   trades={total_trades}  total_R={total_R:+.1f}  "
          f"PF={agg_pf:.2f}  WR={agg_wr*100:.1f}%  "
          f"end=${agg_end:.0f}  maxDD={agg_dd:.1f}%")
    print("=" * 120)
    print()
    print("VERDICT")
    print("-" * 120)
    if total_trades < 30:
        print(f"  INSUFFICIENT SAMPLE ({total_trades} trades) — need 100+ for confidence.")
    elif agg_pf >= 1.3 and agg_wr >= 0.40 and agg_dd < 25.0:
        print(f"  VERDICT: DEPLOY  —  PF {agg_pf:.2f} >= 1.3, "
              f"WR {agg_wr*100:.1f}% >= 40%, DD {agg_dd:.1f}% < 25%.")
    elif agg_pf >= 1.0:
        print(f"  VERDICT: MARGINAL — tune params before deploy.  "
              f"PF {agg_pf:.2f}, WR {agg_wr*100:.1f}%, DD {agg_dd:.1f}%.")
    else:
        print(f"  VERDICT: DO NOT DEPLOY — edge not present as configured.  "
              f"PF {agg_pf:.2f} < 1.0, WR {agg_wr*100:.1f}%, DD {agg_dd:.1f}%.")
    print("-" * 120)


if __name__ == "__main__":
    main()
