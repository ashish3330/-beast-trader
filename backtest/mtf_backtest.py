"""
DRAGON MTF BACKTEST -- Multi-Timeframe Intelligence A/B test.

Compares:
  A) Plain H1-only Dragon backtest (baseline)
  B) MTF-enhanced: simulated M15/M5 from H1, using MTFIntelligence.analyze()

Simulated lower TFs:
  - M15: each H1 bar -> 4 M15 bars (interpolated OHLC)
  - M5:  each H1 bar -> 12 M5 bars (interpolated OHLC)
  These are synthetic but directionally correct for testing whether
  the MTF confluence/entry_quality/exit_urgency logic adds value.

MTF entry filter:  entry_quality >= 30 AND score >= adaptive_min
MTF SL:            use smart_sl instead of fixed ATR
MTF exit:          exit_urgency >= 0.7 triggers immediate close

Run: python3 -B backtest/mtf_backtest.py [--days 365]
"""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from signals.momentum_scorer import (
    _compute_indicators, _score, IND_DEFAULTS, IND_OVERRIDES, REGIME_PARAMS, DEFAULT_PARAMS
)
from mtf_intelligence import MTFIntelligence

# ---------------------------------------------------------------------------
#  CONFIG (mirrors dragon_backtest.py)
# ---------------------------------------------------------------------------

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
START_EQ = 1000.0
SLIP = 0.0
RISK_PCT = 0.008
DAILY_LOSS_LIMIT = 0.01
CONSEC_LOSS_COOLDOWN = 24

SYMBOL_SESSION_OVERRIDE = {"JPN225ft": (0, 22)}
SYMBOL_ATR_SL_OVERRIDE = {"BTCUSD": 2.0, "XAGUSD": 1.8, "USDJPY": 1.2}
SYMBOL_MIN_SCORE_OVERRIDE = {
    "BTCUSD":   {"trending": 5.5, "ranging": 8.0, "volatile": 6.5, "low_vol": 7.0},
    "XAGUSD":   {"trending": 5.5, "ranging": 8.0, "volatile": 6.5, "low_vol": 7.0},
    "XAUUSD":   {"trending": 5.5, "ranging": 8.0, "volatile": 7.0, "low_vol": 7.0},
    "USDJPY":   {"trending": 6.5, "ranging": 8.5, "volatile": 7.5, "low_vol": 7.5},
}

# Dragon's 6 live symbols
DRAGON_SYMBOLS = {
    "XAUUSD":   {"cache": "raw_h1_xauusd.pkl",   "point": 0.01,    "tv": 1.0,   "spread": 0.33,  "lot": 0.01, "cat": "Gold"},
    "XAGUSD":   {"cache": "raw_h1_XAGUSD.pkl",   "point": 0.001,   "tv": 5.0,   "spread": 0.035, "lot": 0.01, "cat": "Gold"},
    "BTCUSD":   {"cache": "raw_h1_BTCUSD.pkl",   "point": 0.01,    "tv": 0.01,  "spread": 17.0,  "lot": 0.01, "cat": "Crypto"},
    "NAS100.r": {"cache": "raw_h1_NAS100_r.pkl",  "point": 0.01,    "tv": 0.01,  "spread": 1.80,  "lot": 0.10, "cat": "Index"},
    "JPN225ft": {"cache": "raw_h1_JPN225ft.pkl",  "point": 0.01,    "tv": 0.0063,"spread": 10.0,  "lot": 1.00, "cat": "Index"},
    "USDJPY":   {"cache": "raw_h1_USDJPY.pkl",    "point": 0.001,   "tv": 0.63,  "spread": 0.018, "lot": 0.20, "cat": "Forex"},
}

# ---------------------------------------------------------------------------
#  SIMULATE LOWER TIMEFRAMES FROM H1
# ---------------------------------------------------------------------------

def _simulate_sub_bars(h1_df, n_sub):
    """Synthesize n_sub bars from each H1 bar using linear interpolation.

    For each H1 bar:
      - Divide into n_sub segments
      - If bullish: O -> L -> H -> C path
      - If bearish: O -> H -> L -> C path
      - Distribute tick_volume evenly
    Returns DataFrame with same columns as h1_df.
    """
    rows = []
    for idx in range(len(h1_df)):
        row = h1_df.iloc[idx]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        t = row["time"]
        tv = int(row["tick_volume"]) if "tick_volume" in row.index else 100
        sp = int(row["spread"]) if "spread" in row.index else 0

        bullish = c >= o

        # Create a price path through the H1 bar
        # Path: O -> first_extreme -> second_extreme -> C
        if bullish:
            # O -> dip to L -> rally to H -> settle at C
            path = [o, l, h, c]
        else:
            # O -> rally to H -> drop to L -> settle at C
            path = [o, h, l, c]

        # Interpolate n_sub+1 price points along this path
        # Map path positions: 0, 0.33, 0.67, 1.0 (for 4-point path)
        path_pos = np.linspace(0, 1, len(path))
        sub_pos = np.linspace(0, 1, n_sub + 1)
        prices = np.interp(sub_pos, path_pos, path)

        # Time delta per sub-bar
        td = pd.Timedelta(minutes=60 // n_sub)
        sub_vol = max(1, tv // n_sub)

        for j in range(n_sub):
            sub_o = prices[j]
            sub_c = prices[j + 1]
            sub_h = max(sub_o, sub_c) * (1 + np.random.uniform(0, 0.0002))
            sub_l = min(sub_o, sub_c) * (1 - np.random.uniform(0, 0.0002))
            # Ensure H >= max(O,C) and L <= min(O,C)
            sub_h = max(sub_h, sub_o, sub_c)
            sub_l = min(sub_l, sub_o, sub_c)

            rows.append({
                "time": t + td * j,
                "open": sub_o,
                "high": sub_h,
                "low": sub_l,
                "close": sub_c,
                "tick_volume": sub_vol,
                "spread": sp,
                "real_volume": 0,
            })

    sub_df = pd.DataFrame(rows)
    # Ensure correct dtypes
    for col in ["open", "high", "low", "close"]:
        sub_df[col] = sub_df[col].astype(np.float64)
    sub_df["tick_volume"] = sub_df["tick_volume"].astype(np.uint64)
    sub_df["spread"] = sub_df["spread"].astype(np.int32)
    sub_df["real_volume"] = sub_df["real_volume"].astype(np.uint64)
    return sub_df


# ---------------------------------------------------------------------------
#  FAKE STATE ADAPTER for MTFIntelligence
# ---------------------------------------------------------------------------

class _FakeState:
    """Minimal state adapter so MTFIntelligence.analyze() works in backtest."""

    def __init__(self):
        self._candles = {}  # (symbol, tf_minutes) -> DataFrame

    def set_candles(self, symbol, tf, df):
        self._candles[(symbol, tf)] = df

    def get_candles(self, symbol, tf):
        return self._candles.get((symbol, tf))


# ---------------------------------------------------------------------------
#  HELPERS (shared with dragon_backtest)
# ---------------------------------------------------------------------------

def get_adaptive_min_score(regime, symbol=None):
    if symbol and symbol in SYMBOL_MIN_SCORE_OVERRIDE:
        sym_scores = SYMBOL_MIN_SCORE_OVERRIDE[symbol]
        if regime in sym_scores:
            return sym_scores[regime]
    return {"trending": 6.0, "ranging": 8.0, "volatile": 7.0, "low_vol": 7.0}.get(regime, 7.0)


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


# ---------------------------------------------------------------------------
#  CORE BACKTEST ENGINE
# ---------------------------------------------------------------------------

def run(symbol, days=365, mode="plain"):
    """Run backtest for a single symbol.

    mode:
      "plain"  -- baseline H1-only Dragon backtest (identical to dragon_backtest.py)
      "mtf"    -- MTF-enhanced: uses MTFIntelligence for entry filter, SL, and exit
    """
    scfg = DRAGON_SYMBOLS[symbol]
    cache_path = CACHE / scfg["cache"]
    if not cache_path.exists():
        return None
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

    # ----- MTF setup -----
    mtf_engine = None
    fake_state = None
    h1_df_full = None
    m15_df_full = None
    m5_df_full = None
    m15_time_idx = None
    m5_time_idx = None

    if mode == "mtf":
        fake_state = _FakeState()
        mtf_engine = MTFIntelligence(fake_state)
        # Disable cache (we update candles every bar)
        mtf_engine._CACHE_TTL = 0

        h1_df_full = df.copy()

        # Pre-generate M15 and M5 from H1
        np.random.seed(777)  # deterministic sub-bar noise
        m15_df_full = _simulate_sub_bars(df, 4)   # 4 x M15 per H1
        m5_df_full = _simulate_sub_bars(df, 12)   # 12 x M5 per H1

        # Build time -> index mappings for fast slicing
        # For each H1 bar index i, the M15 sub-bars are at indices i*4 .. i*4+3
        # and M5 sub-bars at i*12 .. i*12+11

    # ----- Trade state -----
    eq = START_EQ; peak = START_EQ; max_dd = 0
    n_trades = 0; wins = 0; gross_p = 0; gross_l = 0
    in_trade = False; d = 0; entry = 0; pos_sl = 0; sl_dist = 0
    trade_lot = 0.0
    consec_losses = 0; cooldown_until = 0
    daily_pnl = 0.0; current_day = None; day_eq_start = START_EQ; day_stopped = False
    r_multiples = []; max_consec_loss = 0; current_streak = 0
    mtf_exits = 0  # count MTF exit_urgency closes

    np.random.seed(42)  # ML filter seed

    for i in range(start_idx, n):
        atr_val = float(ind["at"][i]) if not np.isnan(ind["at"][i]) else 0
        if atr_val == 0:
            continue

        # Session filter
        bar_time = df["time"].iloc[i]
        bar_hour = bar_time.hour if hasattr(bar_time, "hour") else 12
        sess_start, sess_end = SYMBOL_SESSION_OVERRIDE.get(symbol, (6, 22))
        if cat != "Crypto" and (bar_hour >= sess_end or bar_hour < sess_start):
            continue

        # Daily loss reset
        bar_date = bar_time.date() if hasattr(bar_time, "date") else None
        if bar_date and bar_date != current_day:
            current_day = bar_date
            day_eq_start = eq
            daily_pnl = 0.0
            day_stopped = False

        # ----- MTF analysis for this bar -----
        mtf_result = None
        if mode == "mtf" and i >= 80:
            # Feed windowed candles to the fake state
            # H1: last 120 bars up to current
            h1_window = h1_df_full.iloc[max(0, i - 119):i + 1].reset_index(drop=True)
            fake_state.set_candles(symbol, 60, h1_window)

            # M15: last 120*4 = 480 sub-bars
            m15_end = (i + 1) * 4
            m15_start = max(0, m15_end - 480)
            m15_window = m15_df_full.iloc[m15_start:m15_end].reset_index(drop=True)
            fake_state.set_candles(symbol, 15, m15_window)

            # M5: last 120*12 = 1440 sub-bars
            m5_end = (i + 1) * 12
            m5_start = max(0, m5_end - 1440)
            m5_window = m5_df_full.iloc[m5_start:m5_end].reset_index(drop=True)
            fake_state.set_candles(symbol, 5, m5_window)

            # M1 not simulated (too noisy from H1 interpolation) -- MTF handles None gracefully
            fake_state.set_candles(symbol, 1, None)

            try:
                mtf_result = mtf_engine._analyze_impl(symbol)
            except Exception:
                mtf_result = None

        # ----- MTF exit urgency check (while in trade) -----
        if in_trade and mode == "mtf" and mtf_result is not None:
            exit_urg = mtf_result.get("exit_urgency", 0.0)
            if exit_urg >= 0.7:
                # MTF says exit NOW
                exit_price = float(ind["c"][i])
                exit_cost = (spread + SLIP * pt)
                pnl = d * (exit_price - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
                eq += pnl
                daily_pnl += pnl
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
                in_trade = False; mtf_exits += 1
                if day_eq_start > 0 and daily_pnl < -(DAILY_LOSS_LIMIT * day_eq_start):
                    day_stopped = True
                continue

        # MANAGE: trailing SL
        if in_trade:
            if (d == 1 and ind["l"][i] <= pos_sl) or (d == -1 and ind["h"][i] >= pos_sl):
                exit_cost = (spread + SLIP * pt)
                pnl = d * (pos_sl - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
                eq += pnl
                daily_pnl += pnl
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
            new_sl = None
            if profit_r >= 6.0:   new_sl = cur - 0.7 * atr_val * d
            elif profit_r >= 4.0: new_sl = cur - 1.0 * atr_val * d
            elif profit_r >= 2.5: new_sl = cur - 1.5 * atr_val * d
            elif profit_r >= 1.5:
                new_sl = cur - 2.0 * atr_val * d
                floor = entry + 0.5 * sl_dist * d
                if d == 1:   new_sl = max(new_sl, floor)
                else:        new_sl = min(new_sl, floor)
            elif profit_r >= 1.0: new_sl = entry + 0.5 * sl_dist * d
            elif profit_r >= 0.5: new_sl = entry + 2 * pt * d
            if new_sl is not None:
                if d == 1 and new_sl > pos_sl:     pos_sl = new_sl
                elif d == -1 and new_sl < pos_sl:  pos_sl = new_sl

        if day_stopped:
            continue
        if i < cooldown_until:
            continue

        # SCORE
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

        # ----- MTF entry filter -----
        if mode == "mtf" and mtf_result is not None:
            entry_quality = mtf_result.get("entry_quality", 0.0)
            if entry_quality < 30:
                continue  # MTF says: not good enough

            # Also check confluence direction agrees with signal
            mtf_dir = "LONG" if new_dir == 1 else "SHORT"
            h1_dir = mtf_result.get("h1_dir", "FLAT")
            # Require at least H1 not opposing
            if h1_dir != "FLAT" and h1_dir != mtf_dir:
                continue

        # ML filter (50% of signals, same as dragon_backtest)
        best_score = max(ls, ss)
        pass_prob = min(1.0, best_score / 10.0)
        if np.random.random() > pass_prob:
            continue

        # REVERSAL
        if in_trade and new_dir != d:
            exit_cost = (spread + SLIP * pt)
            pnl = d * (float(ind["c"][i]) - entry) / pt * tv * trade_lot - exit_cost / pt * tv * trade_lot
            eq += pnl
            daily_pnl += pnl
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

        # ENTRY
        if not in_trade:
            d = new_dir

            # ----- SL: MTF smart_sl or plain ATR -----
            if mode == "mtf" and mtf_result is not None:
                mtf_sl = mtf_result.get("optimal_sl", 0.0)
                # Use MTF smart SL if it's reasonable, else fall back to ATR
                if mtf_sl > 0 and mtf_sl < sl_cap:
                    sl_dist = mtf_sl
                    # Still enforce ATR floor (1.0x ATR minimum)
                    sl_dist = max(sl_dist, atr_val * 1.0)
                else:
                    # Fallback: same as plain
                    sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
                    sym_sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, 1.5)
                    sl_dist = max(atr_val * sl_m, atr_val * sym_sl_mult)
            else:
                sl_m = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)[0]
                sym_sl_mult = SYMBOL_ATR_SL_OVERRIDE.get(symbol, 1.5)
                sl_dist = max(atr_val * sl_m, atr_val * sym_sl_mult)

            sl_dist = min(sl_dist, sl_cap)

            # Dynamic position sizing
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

    # Close open trade at end
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
        "sym": symbol, "mode": mode, "trades": n_trades,
        "wr": round(wr, 1), "pf": round(pf, 2),
        "ret": round(ret, 1), "dd": round(dd, 1), "eq": round(eq, 2),
        "gross_p": round(gross_p, 2), "gross_l": round(gross_l, 2),
        "max_consec_loss": max_consec_loss, "avg_r": round(avg_r, 3),
        "sharpe": round(sharpe, 2), "mtf_exits": mtf_exits,
    }


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, time as _time

    parser = argparse.ArgumentParser(description="Dragon MTF Backtest A/B")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="Symbols to test (default: all Dragon symbols)")
    args = parser.parse_args()

    symbols = args.symbols or sorted(DRAGON_SYMBOLS.keys())
    # Validate
    symbols = [s for s in symbols if s in DRAGON_SYMBOLS]

    print("=" * 130)
    print(f"  DRAGON MTF BACKTEST: Plain H1 vs MTF-Enhanced | {args.days}d | $1,000 start")
    print(f"  MTF entry: quality>=30 + H1 direction agree | MTF SL: smart_sl | MTF exit: urgency>=0.7")
    print("=" * 130)

    plain_results = []
    mtf_results = []

    for sym in symbols:
        t0 = _time.time()
        print(f"\n  Running {sym}...", end=" ", flush=True)

        r_plain = run(sym, args.days, mode="plain")
        print("plain done...", end=" ", flush=True)

        r_mtf = run(sym, args.days, mode="mtf")
        elapsed = _time.time() - t0
        print(f"mtf done ({elapsed:.1f}s)")

        if r_plain:
            plain_results.append(r_plain)
        if r_mtf:
            mtf_results.append(r_mtf)

    # --------------- REPORT ---------------
    print("\n" + "=" * 130)
    print(f"  {'':60s} PLAIN H1-ONLY")
    print(f"{'Symbol':<12} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Ret%':>9} {'DD%':>7} {'Final$':>10} {'Sharpe':>7}")
    print("-" * 70)
    for r in plain_results:
        print(f"{r['sym']:<12} {r['trades']:>7} {r['wr']:>6.1f}% {r['pf']:>7.2f} {r['ret']:>8.1f}% {r['dd']:>6.1f}% ${r['eq']:>9.2f} {r['sharpe']:>7.2f}")

    gp = sum(r["gross_p"] for r in plain_results)
    gl = sum(r["gross_l"] for r in plain_results)
    print(f"{'TOTAL':<12} {'':>7} {'':>7} {gp/gl if gl else 0:>7.2f} {'':>9} {'':>7} {'':>10} {'':>7}")

    print(f"\n  {'':60s} MTF-ENHANCED")
    print(f"{'Symbol':<12} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Ret%':>9} {'DD%':>7} {'Final$':>10} {'Sharpe':>7} {'MTFExits':>9}")
    print("-" * 85)
    for r in mtf_results:
        print(f"{r['sym']:<12} {r['trades']:>7} {r['wr']:>6.1f}% {r['pf']:>7.2f} {r['ret']:>8.1f}% {r['dd']:>6.1f}% ${r['eq']:>9.2f} {r['sharpe']:>7.2f} {r['mtf_exits']:>9}")

    gp_m = sum(r["gross_p"] for r in mtf_results)
    gl_m = sum(r["gross_l"] for r in mtf_results)
    print(f"{'TOTAL':<12} {'':>7} {'':>7} {gp_m/gl_m if gl_m else 0:>7.2f} {'':>9} {'':>7} {'':>10} {'':>7} {'':>9}")

    # --------------- DELTA COMPARISON ---------------
    print(f"\n{'=' * 130}")
    print(f"  DELTA: MTF-Enhanced vs Plain H1")
    print(f"{'=' * 130}")
    print(f"{'Symbol':<12} {'Trades':>12} {'WR%':>12} {'PF':>12} {'Ret%':>14} {'DD%':>12} {'Sharpe':>12} {'Verdict':>10}")
    print("-" * 100)

    mtf_wins = 0; mtf_losses = 0; mtf_neutral = 0

    for rp in plain_results:
        sym = rp["sym"]
        rm = next((r for r in mtf_results if r["sym"] == sym), None)
        if not rm:
            continue

        dt = rm["trades"] - rp["trades"]
        dwr = rm["wr"] - rp["wr"]
        dpf = rm["pf"] - rp["pf"]
        dret = rm["ret"] - rp["ret"]
        ddd = rm["dd"] - rp["dd"]  # negative is better (less DD)
        dsh = rm["sharpe"] - rp["sharpe"]

        # Verdict: MTF wins if PF improved AND DD not significantly worse
        if dpf > 0.05 and ddd < 5.0:
            verdict = "MTF WIN"
            mtf_wins += 1
        elif dpf < -0.05:
            verdict = "PLAIN WIN"
            mtf_losses += 1
        else:
            verdict = "NEUTRAL"
            mtf_neutral += 1

        dt_s = f"{dt:+d}"
        dwr_s = f"{dwr:+.1f}%"
        dpf_s = f"{dpf:+.2f}"
        dret_s = f"{dret:+.1f}%"
        ddd_s = f"{ddd:+.1f}%"
        dsh_s = f"{dsh:+.2f}"

        print(f"{sym:<12} {dt_s:>12} {dwr_s:>12} {dpf_s:>12} {dret_s:>14} {ddd_s:>12} {dsh_s:>12} {verdict:>10}")

    print("-" * 100)
    print(f"\n  MTF wins: {mtf_wins} | Plain wins: {mtf_losses} | Neutral: {mtf_neutral}")

    # Portfolio-level delta
    if plain_results and mtf_results:
        p_pf = gp / gl if gl else 0
        m_pf = gp_m / gl_m if gl_m else 0
        p_avg_dd = np.mean([r["dd"] for r in plain_results])
        m_avg_dd = np.mean([r["dd"] for r in mtf_results])
        p_avg_sh = np.mean([r["sharpe"] for r in plain_results])
        m_avg_sh = np.mean([r["sharpe"] for r in mtf_results])
        p_trades = sum(r["trades"] for r in plain_results)
        m_trades = sum(r["trades"] for r in mtf_results)

        print(f"\n  PORTFOLIO LEVEL:")
        print(f"    Plain:  PF={p_pf:.2f}  AvgDD={p_avg_dd:.1f}%  AvgSharpe={p_avg_sh:.2f}  Trades={p_trades}")
        print(f"    MTF:    PF={m_pf:.2f}  AvgDD={m_avg_dd:.1f}%  AvgSharpe={m_avg_sh:.2f}  Trades={m_trades}")
        print(f"    Delta:  PF={m_pf - p_pf:+.2f}  AvgDD={m_avg_dd - p_avg_dd:+.1f}%  AvgSharpe={m_avg_sh - p_avg_sh:+.2f}  Trades={m_trades - p_trades:+d}")

    print(f"\n  NOTE: M15/M5 are interpolated from H1 (synthetic). Results are directional,")
    print(f"        not exact. Real MTF with live M5/M15 feeds will differ.")
    print("=" * 130)
