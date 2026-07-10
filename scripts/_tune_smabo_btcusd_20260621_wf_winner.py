#!/usr/bin/env python3 -B
"""
SMABO XAUUSD walk-forward validation — 2026-06-21.

Loads the tuned winner (params={} => DEFAULTS) and runs a strict 5-fold
contiguous walk-forward on the same 365d M15 window used by the tune phase.

PASS rule: 4-of-5 folds with fold-PF >= 1.3 AND fold-total_R > 0.
Also reports a pooled OOS combined PF / WR / DD.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

from agent.sma_breakout import SMABreakoutStrategy  # noqa: E402

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
SYMBOL = "BTCUSD"
DAYS = 365
N_FOLDS = 5
SPREAD_XAU = 30.0
START_CAPITAL = 5000.0
RISK_PCT = 0.0025
MIN_BARS_FOR_BT = 2000
M15_WINDOW = 800
H1_WINDOW = 240
TIME_STOP_BARS = 96
TIME_STOP_PEAK_R = 0.5

OUT_JSON = ROOT / "scripts" / "_tune_smabo_btcusd_20260621_wf_winner.json"

# Winning params from tune phase. Empty dict => DEFAULTS in strategy.
WINNING_PARAMS = {"FAST_SMA":13,"SLOW_SMA":20,"TRAIL_SMA":34,"HTF_LOOKBACK_BARS":30,"MIN_RR":3.0}


def _load_m15(symbol):
    # Try a few capitalisations because cache uses lowercase for some syms.
    candidates = [
        f"raw_m15_{symbol}.pkl",
        f"raw_m15_{symbol.lower()}.pkl",
        f"raw_m15_{symbol.upper()}.pkl",
        f"raw_m15_{symbol.replace('.', '_')}.pkl",
    ]
    for v in candidates:
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
            return df[cols].astype({c: float for c in cols if c != "time"})
    return None


def _resample_h1(m15):
    d = m15.set_index("time")
    out = pd.DataFrame({
        "open":  d["open"].resample("1h").first(),
        "high":  d["high"].resample("1h").max(),
        "low":   d["low"].resample("1h").min(),
        "close": d["close"].resample("1h").last(),
    }).dropna().reset_index()
    return out


class _FakeState:
    def __init__(self, m15, h1):
        self.m15 = m15
        self.h1 = h1
        self.cursor = 0
        self._m15_ns = m15["time"].astype("int64").values
        self._h1_ns = h1["time"].astype("int64").values

    def set_cursor(self, i):
        self.cursor = i

    def get_candles(self, symbol, tf):
        c = self.cursor
        if tf == 15:
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
        if tf == 60:
            if c >= len(self._m15_ns):
                end = len(self.h1)
            else:
                end = int(np.searchsorted(self._h1_ns,
                                          int(self._m15_ns[c]),
                                          side="right"))
            if end == 0:
                return None
            lo = max(0, end - H1_WINDOW)
            return self.h1.iloc[lo: end]
        return None


def _simulate_trade(sig, H, L, C, entry_i, spread):
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
                    runner_sl = entry
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


def _summarize(trades):
    if not trades:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "total_R": 0.0,
                "max_dd_pct": 0.0, "avg_R": 0.0}
    Rs = np.array([t["R"] for t in trades])
    wins = Rs[Rs > 0]
    losses = Rs[Rs <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 999.0
    equity = [START_CAPITAL]
    risk_d = START_CAPITAL * RISK_PCT
    for r in Rs:
        equity.append(equity[-1] + r * risk_d)
    eq = np.array(equity)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    return {
        "trades": int(len(trades)),
        "pf": float(pf),
        "wr": float((Rs > 0).mean()),
        "total_R": float(Rs.sum()),
        "avg_R": float(Rs.mean()),
        "max_dd_pct": float(abs(dd.min()) * 100.0),
    }


# ─── Walk-forward driver ────────────────────────────────────────────────
def _load_full_data():
    m15 = _load_m15(SYMBOL)
    if m15 is None:
        raise RuntimeError(f"No M15 cache for {SYMBOL}")
    n_keep = int(DAYS * 24 * 4)
    if n_keep < len(m15):
        m15 = m15.iloc[-n_keep:].reset_index(drop=True)
    h1 = _resample_h1(m15)
    return m15, h1


def run_fold(m15, h1, params, fold_lo, fold_hi, warmup):
    """Run BT on bar range [fold_lo, fold_hi) — entries only inside this range.

    `warmup` is the number of bars to allow the strategy to build indicator
    history; we still call set_cursor at every bar in [fold_lo, fold_hi) and
    only emit a trade if the strategy returns a signal.
    """
    H = m15["high"].values
    L = m15["low"].values
    C = m15["close"].values
    n = len(m15)
    if n < MIN_BARS_FOR_BT:
        return None

    state = _FakeState(m15, h1)
    strat = SMABreakoutStrategy(state, params=params)

    trades = []
    open_until = -1
    # iterate over the fold (NOT the full history) — but cursor still has
    # the full series available because _FakeState references m15/h1 in full,
    # so M15_WINDOW / H1_WINDOW of warmup is automatic.
    start_i = max(fold_lo, warmup)
    end_i = min(fold_hi, n - 1)
    for i in range(start_i, end_i):
        if i <= open_until:
            continue
        state.set_cursor(i)
        strat._last_bar_t.pop(SYMBOL, None)
        sig = strat.evaluate(SYMBOL)
        if sig is None:
            continue
        tr = _simulate_trade(sig, H, L, C, entry_i=i, spread=SPREAD_XAU)
        if tr is None:
            continue
        trades.append(tr)
        open_until = i + tr["bars_held"]

    summ = _summarize(trades)
    return summ, trades


def main():
    m15, h1 = _load_full_data()
    n = len(m15)
    print(f"Loaded {SYMBOL} M15: {n} bars (~{n/(24*4):.0f} days)")
    print(f"Winning params (empty => DEFAULTS): {WINNING_PARAMS}")

    warmup = max(M15_WINDOW, 60)
    # Fold the BARS available AFTER warmup so every fold has full indicator
    # history. This mirrors what the tune phase did (start_i = max(800, 60)).
    usable_lo = warmup
    usable_hi = n - 1
    total_usable = usable_hi - usable_lo
    fold_size = total_usable // N_FOLDS

    folds = []
    for k in range(N_FOLDS):
        f_lo = usable_lo + k * fold_size
        f_hi = (usable_lo + (k + 1) * fold_size) if k < N_FOLDS - 1 else usable_hi
        folds.append((f_lo, f_hi))

    print("\nFold ranges (M15-bar indices, ~73d each):")
    for k, (lo, hi) in enumerate(folds, 1):
        t_lo = m15["time"].iloc[lo]
        t_hi = m15["time"].iloc[hi - 1]
        days = (t_hi - t_lo).total_seconds() / 86400.0
        print(f"  fold {k}: bars [{lo}..{hi})  {t_lo}  ->  {t_hi}  "
              f"(~{days:.1f}d)")

    print("\n========== WALK-FORWARD RUN ==========")
    fold_summaries = []
    pooled_trades = []
    pass_count = 0
    for k, (lo, hi) in enumerate(folds, 1):
        out = run_fold(m15, h1, WINNING_PARAMS, lo, hi, warmup)
        if out is None:
            print(f"  fold {k}: SKIPPED (data too short)")
            continue
        summ, trades = out
        fold_summaries.append({"fold": k, "lo": lo, "hi": hi, "summary": summ})
        pooled_trades.extend(trades)
        ok = (summ["pf"] >= 1.3 and summ["total_R"] > 0 and summ["trades"] > 0)
        if ok:
            pass_count += 1
        tag = "PASS" if ok else "FAIL"
        print(f"  fold {k}: trades={summ['trades']:>3}  PF={summ['pf']:.2f}  "
              f"totR={summ['total_R']:+7.2f}  WR={summ['wr']*100:5.1f}%  "
              f"DD={summ['max_dd_pct']:5.1f}%   [{tag}]")

    combined = _summarize(pooled_trades)
    print("\n========== COMBINED (pooled OOS) ==========")
    print(f"  trades={combined['trades']}  PF={combined['pf']:.2f}  "
          f"totR={combined['total_R']:+.2f}  WR={combined['wr']*100:.1f}%  "
          f"DD={combined['max_dd_pct']:.1f}%")

    passed = (pass_count >= 4)
    print(f"\nFolds passed: {pass_count}/{len(fold_summaries)}  "
          f"==>  WF_PASSED={passed}")

    out = {
        "symbol": SYMBOL,
        "days": DAYS,
        "n_folds": N_FOLDS,
        "winning_params": WINNING_PARAMS,
        "fold_summaries": fold_summaries,
        "combined": combined,
        "pass_count": pass_count,
        "wf_passed": passed,
        "pass_rule": "4-of-5 folds with PF>=1.3 AND total_R>0",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
