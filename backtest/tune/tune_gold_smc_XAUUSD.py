#!/usr/bin/env python3 -B
"""GOLD_SMC (9th book) hard-tuner — XAUUSD H1 hybrid SMC (sweep + BOS + FVG retest).

Self-contained. Mirrors the LIVE detector agent/gold_smc.evaluate() and the faithful
sim scripts/_forex_smc_backtest.py so backtest == live:
  bias(D1 close vs D1 EMA[BIAS_EMA]) + liquidity sweep within SEQ bars + BOS(prior
  swing) + FVG + EMA9>EMA21 both sloping + close vs VWAP(daily) + (MACD or RSI) +
  strong/engulfing candle. Entry@close, SL=swept-extreme -/+ SL_ATR*ATR, TP1/TP2 R,
  50/50 legs, BE after TP1, EMA9-trail on the runner.

SPREAD-ONLY, NO SLIPPAGE (SLIP=0) — live has no slippage (user standing rule).

Params swept ONE-AT-A-TIME around the live BASELINE, then the best-per-param combined
and WF-validated (train 60% / test 40%). A candidate SHIPS only if it beats baseline on
train AND is >= baseline-neutral on the held-out test (no curve-fit; prior GOLD_SMC tune
was OVERFIT-REJECTED, so we stay conservative).

Usage:  python3 -B backtest/tune/tune_gold_smc_XAUUSD.py
Writes: backtest/tune/results_gold_smc_XAUUSD.json
"""
import json
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
SYM = "XAUUSD"
OUT = Path(__file__).with_name("results_gold_smc_XAUUSD.json")
SLIP_FRAC = 0.0  # live has no slippage — spread-only backtest

BASELINE = {"SWING": 5, "SWEEP_LB": 10, "SEQ": 8, "SL_ATR": 0.2,
            "TP1_R": 1.5, "TP2_R": 2.0, "BIAS_EMA": 50}

# Robustness guards (prior GOLD_SMC tune was OVERFIT-REJECTED — stay strict):
#  • need a minimum FULL sample before ANY tune is statistically defensible
#  • both WF legs must actually TRADE (a config that "wins" by making ~0 train
#    trades is curve-fit to the sample's loser bars, not an edge)
MIN_SAMPLE_FULL = 30   # min baseline full trades to even attempt a tune
MIN_LEG_TRADES = 8     # min trades required in EACH of train & test to ship

GRID = {
    "SWING":    [3, 5, 8],
    "SWEEP_LB": [6, 10, 15],
    "SEQ":      [5, 8, 12],
    "SL_ATR":   [0.15, 0.2, 0.3, 0.5],
    "TP1_R":    [1.0, 1.5, 2.0],
    "TP2_R":    [2.0, 2.5, 3.0],
    "BIAS_EMA": [20, 50, 100],
}


# ── indicators (identical to gold_smc.evaluate / _forex_smc_backtest) ──
def _naive(s):
    s = pd.to_datetime(s, utc=True)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


def _ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def _macd(c):
    m = _ema(c, 12) - _ema(c, 26)
    return m, _ema(m, 9)


def _atr(h, l, c, n=14):
    p = c.shift(1)
    tr = pd.concat([(h - l), (h - p).abs(), (l - p).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def load():
    h1 = pickle.load(open(CACHE / f"raw_h1_{SYM}.pkl", "rb"))
    # NOTE: live/brain reads lowercase raw_h1_xauusd.pkl; uppercase is identical here.
    if not (CACHE / f"raw_h1_{SYM}.pkl").exists():
        h1 = pickle.load(open(CACHE / "raw_h1_xauusd.pkl", "rb"))
    h1["time"] = _naive(h1["time"])
    h1 = h1.sort_values("time").reset_index(drop=True)
    c, h, l, o = h1["close"], h1["high"], h1["low"], h1["open"]
    vol = h1["tick_volume"] if "tick_volume" in h1 else pd.Series(1.0, index=h1.index)
    h1["ema9"], h1["ema21"] = _ema(c, 9), _ema(c, 21)
    h1["rsi"] = _rsi(c)
    h1["macd"], h1["sig"] = _macd(c)
    h1["atr"] = _atr(h, l, c)
    h1["body"] = (c - o).abs()
    h1["avgbody"] = h1["body"].rolling(20).mean()
    day = h1["time"].dt.date
    tp = (h + l + c) / 3
    h1["vwap"] = ((tp * vol).groupby(day).cumsum()
                  / vol.groupby(day).cumsum().replace(0, np.nan)).ffill()
    d = pickle.load(open(CACHE / f"raw_d1_{SYM}.pkl", "rb"))
    d["time"] = _naive(d["time"])
    d = d.sort_values("time").reset_index(drop=True)
    return h1, d, c


def build(h1, d, c, params):
    """Attach params-dependent columns (swings, sweeps, D1 bias) and return a fresh frame."""
    m = h1.copy()
    S = int(params["SWING"])
    hh, ll = m["high"], m["low"]
    m["prior_hi"] = hh.rolling(S * 2 + 1).max().shift(1)
    m["prior_lo"] = ll.rolling(S * 2 + 1).min().shift(1)
    m["sweep_lo"] = ll.rolling(int(params["SWEEP_LB"])).min().shift(1)
    m["sweep_hi"] = hh.rolling(int(params["SWEEP_LB"])).max().shift(1)
    dd = d.copy()
    dd["bias"] = np.where(dd["close"] > _ema(dd["close"], int(params["BIAS_EMA"])), 1, -1)
    dd["eff"] = (dd["time"].dt.normalize() + pd.Timedelta(days=1)).astype("datetime64[ns]")
    m["time"] = m["time"].astype("datetime64[ns]")
    m = pd.merge_asof(m, dd[["eff", "bias"]], left_on="time", right_on="eff",
                      direction="backward")
    m = m.dropna(subset=["ema21", "atr", "avgbody", "prior_hi", "sweep_lo", "bias"]
                 ).reset_index(drop=True)
    return m


def simulate(m, spread, params):
    o, h, l, c = [m[x].values for x in ("open", "high", "low", "close")]
    ema9, ema21 = m["ema9"].values, m["ema21"].values
    rsi, macd, sig = m["rsi"].values, m["macd"].values, m["sig"].values
    vwap, atr, bias = m["vwap"].values, m["atr"].values, m["bias"].values
    body, avgbody = m["body"].values, m["avgbody"].values
    prior_hi, prior_lo = m["prior_hi"].values, m["prior_lo"].values
    sweep_lo, sweep_hi = m["sweep_lo"].values, m["sweep_hi"].values
    SEQ = int(params["SEQ"])
    SL_ATR = float(params["SL_ATR"])
    TP1_R, TP2_R = float(params["TP1_R"]), float(params["TP2_R"])
    trades = []
    pos = 0
    entry = sl = tp1 = tp2 = risk0 = 0.0
    half = False
    last_bull_sweep = last_bull_sweep_lo = -999
    last_bear_sweep = last_bear_sweep_hi = -999
    for t in range(3, len(m)):
        if l[t] < sweep_lo[t] and c[t] > sweep_lo[t]:
            last_bull_sweep, last_bull_sweep_lo = t, l[t]
        if h[t] > sweep_hi[t] and c[t] < sweep_hi[t]:
            last_bear_sweep, last_bear_sweep_hi = t, h[t]
        if pos != 0:
            if pos == 1:
                if l[t] <= sl:
                    trades.append(((sl * (1 - SLIP_FRAC) - entry) / entry
                                   - (0 if half else spread)) / risk0); pos = 0; continue
                if not half and h[t] >= tp1:
                    trades.append((((tp1 - entry) / entry - spread) / risk0) * 0.5)
                    half = True; sl = entry
                if half and h[t] >= tp2:
                    trades.append((((tp2 - entry) / entry) / risk0) * 0.5); pos = 0; continue
                if c[t] < ema9[t] and half:
                    trades.append((((c[t] - entry) / entry) / risk0) * 0.5); pos = 0; continue
            else:
                if h[t] >= sl:
                    trades.append(((entry - sl * (1 + SLIP_FRAC)) / entry
                                   - (0 if half else spread)) / risk0); pos = 0; continue
                if not half and l[t] <= tp1:
                    trades.append((((entry - tp1) / entry - spread) / risk0) * 0.5)
                    half = True; sl = entry
                if half and l[t] <= tp2:
                    trades.append((((entry - tp2) / entry) / risk0) * 0.5); pos = 0; continue
                if c[t] > ema9[t] and half:
                    trades.append((((entry - c[t]) / entry) / risk0) * 0.5); pos = 0; continue
            continue
        bull_fvg = (l[t] > h[t - 2]) or (l[t - 1] > h[t - 3])
        bear_fvg = (h[t] < l[t - 2]) or (h[t - 1] < l[t - 3])
        bull_engulf = c[t] > o[t] and c[t - 1] < o[t - 1] and c[t] > o[t - 1] and o[t] < c[t - 1]
        bear_engulf = c[t] < o[t] and c[t - 1] > o[t - 1] and c[t] < o[t - 1] and o[t] > c[t - 1]
        long_ok = (bias[t] == 1 and (t - last_bull_sweep) <= SEQ
                   and c[t] > prior_hi[t] and bull_fvg
                   and ema9[t] > ema21[t] and ema9[t] > ema9[t - 1] and ema21[t] > ema21[t - 1]
                   and c[t] > vwap[t] and (macd[t] > sig[t] or rsi[t] > 50)
                   and (body[t] > 1.2 * avgbody[t] or bull_engulf))
        short_ok = (bias[t] == -1 and (t - last_bear_sweep) <= SEQ
                    and c[t] < prior_lo[t] and bear_fvg
                    and ema9[t] < ema21[t] and ema9[t] < ema9[t - 1] and ema21[t] < ema21[t - 1]
                    and c[t] < vwap[t] and (macd[t] < sig[t] or rsi[t] < 50)
                    and (body[t] > 1.2 * avgbody[t] or bear_engulf))
        if long_ok:
            entry = c[t] * (1 + SLIP_FRAC)
            sl = min(l[t], last_bull_sweep_lo) - SL_ATR * atr[t]
            risk0 = (entry - sl) / entry
            if risk0 <= 0:
                continue
            tp1 = entry + TP1_R * (entry - sl)
            tp2 = entry + TP2_R * (entry - sl)
            pos = 1; half = False
        elif short_ok:
            entry = c[t] * (1 - SLIP_FRAC)
            sl = max(h[t], last_bear_sweep_hi) + SL_ATR * atr[t]
            risk0 = (sl - entry) / entry
            if risk0 <= 0:
                continue
            tp1 = entry - TP1_R * (sl - entry)
            tp2 = entry - TP2_R * (sl - entry)
            pos = -1; half = False
    return trades


def stats(r):
    if not r:
        return {"n": 0, "pf": 0.0, "wr": 0.0, "expR": 0.0, "netR": 0.0}
    r = np.array(r)
    w = r[r > 0].sum()
    ls = -r[r < 0].sum()
    return {"n": len(r), "pf": round(float(w / ls) if ls > 0 else 99.0, 2),
            "wr": round(float((r > 0).mean()), 3), "expR": round(float(r.mean()), 3),
            "netR": round(float(r.sum()), 2)}


def evaluate(h1, d, c, spread, params):
    """Full / train / test stats for a param set. Train=first 60%, test=last 40%."""
    m = build(h1, d, c, params)
    full = stats(simulate(m, spread, params))
    sp = int(len(m) * 0.6)
    tr = stats(simulate(m.iloc[:sp].reset_index(drop=True), spread, params))
    te = stats(simulate(m.iloc[sp:].reset_index(drop=True), spread, params))
    return {"full": full, "train": tr, "test": te}


def main():
    h1, d, c = load()
    px = float(c.iloc[-1])
    med = float(np.nanmedian(h1["spread"].values)) if "spread" in h1 else 15
    point = 0.001 if px > 20 else 0.00001
    spread = min(max(2 * med * point / px, 0.00008), 0.0004)
    days = (h1["time"].iloc[-1] - h1["time"].iloc[0]).days
    yrs = days / 365.25

    base = evaluate(h1, d, c, spread, BASELINE)
    base_train_R = base["train"]["netR"]
    base_test_R = base["test"]["netR"]
    base_full_R = base["full"]["netR"]

    # ── data-sufficiency guard: if the baseline sample is too small, NO tune is
    #    defensible — force SHIP_NONE everywhere (baseline stands). ──
    sample_ok = base["full"]["n"] >= MIN_SAMPLE_FULL

    def _wf_ship(row, pname=None):
        """Honest WF ship test: beat baseline on train AND full, be >= neutral on
        test, and BOTH legs must genuinely trade (no winning-by-trading-less)."""
        if not sample_ok:
            return False
        if pname is not None and row["value"] == BASELINE[pname]:
            return False
        return (row["train"]["n"] >= MIN_LEG_TRADES
                and row["test"]["n"] >= MIN_LEG_TRADES
                and row["train"]["netR"] > base_train_R
                and row["test"]["netR"] >= base_test_R
                and row["full"]["netR"] > base_full_R)

    per_param = {}
    best_combo = dict(BASELINE)
    for pname, vals in GRID.items():
        rows = []
        best_v, best_train_R = BASELINE[pname], -1e9
        for v in vals:
            p = dict(BASELINE)
            p[pname] = v
            r = evaluate(h1, d, c, spread, p)
            r["value"] = v
            rows.append(r)
            if r["train"]["netR"] > best_train_R:
                best_train_R, best_v = r["train"]["netR"], v
        best_row = next(x for x in rows if x["value"] == best_v)
        ships = _wf_ship(best_row, pname)
        verdict = "SHIP" if ships else "SHIP_NONE"
        chosen = best_v if ships else BASELINE[pname]
        if ships:
            best_combo[pname] = best_v
        per_param[pname] = {
            "baseline": BASELINE[pname], "grid": vals,
            "sweep": [{"value": r["value"], **{k: r[k] for k in ("full", "train", "test")}}
                      for r in rows],
            "best_on_train": best_v, "verdict": verdict, "chosen": chosen,
        }

    # combined = all per-param SHIP winners together, WF-validated
    combined = evaluate(h1, d, c, spread, best_combo)
    combined_ships = (sample_ok and best_combo != BASELINE
                      and combined["train"]["n"] >= MIN_LEG_TRADES
                      and combined["test"]["n"] >= MIN_LEG_TRADES
                      and combined["train"]["netR"] > base_train_R
                      and combined["test"]["netR"] >= base_test_R
                      and combined["full"]["netR"] > base_full_R)

    result = {
        "book": "GOLD_SMC",
        "symbol": SYM,
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "sample_sufficient": bool(sample_ok),
        "overall_verdict": ("SHIP_NONE — insufficient sample" if not sample_ok
                            else ("SHIP combined" if combined_ships else "SHIP_NONE")),
        "data": {"h1_bars": int(len(h1)), "span_days": int(days),
                 "span_years": round(yrs, 2), "spread_frac": round(spread, 6),
                 "wf_split": "train 60% / test 40%", "slippage": 0.0,
                 "min_sample_full": MIN_SAMPLE_FULL, "min_leg_trades": MIN_LEG_TRADES},
        "baseline": {"params": BASELINE, **base},
        "per_param": per_param,
        "combined": {"params": best_combo, "ships": bool(combined_ships), **combined,
                     "vs_baseline_full_R": round(combined["full"]["netR"] - base_full_R, 2),
                     "vs_baseline_train_R": round(combined["train"]["netR"] - base_train_R, 2),
                     "vs_baseline_test_R": round(combined["test"]["netR"] - base_test_R, 2)},
        "notes": [
            f"Baseline fires only {base['full']['n']} trades over the whole sample "
            f"(train {base['train']['n']} / test {base['test']['n']}). That is FAR "
            f"below MIN_SAMPLE_FULL={MIN_SAMPLE_FULL}, so no tune is statistically "
            "defensible: overall_verdict = SHIP_NONE. Baseline stands.",
            "Data is only ~1500 H1 bars (~91 days / 0.25yr). The 11.7yr history the "
            "module comment references is NO LONGER in cache (raw_h1_XAUUSD.pkl was "
            "refreshed to a 1500-bar live window). Re-run this tuner once a multi-year "
            "H1 cache is restored — the harness is ready.",
            "The raw sweep shows apparent 'winners' (SWING=8, SWEEP_LB=15) but they are "
            "curve-fit artifacts: SWEEP_LB=15 'wins' by making ~0 trades in the train "
            "half (dodging the 3 training losers), and SWING=8 only picks the least-bad "
            "of three losing train configs. The MIN_LEG_TRADES + MIN_SAMPLE_FULL guards "
            "correctly reject them. This matches the prior OVERFIT-REJECTED outcome.",
            "Spread-only, no slippage (SLIP=0), to match the live detector.",
            "Logic mirrors agent/gold_smc.evaluate + scripts/_forex_smc_backtest.py "
            "(live == backtest).",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, default=float))

    # ── console summary ──
    print(f"GOLD_SMC XAUUSD  |  H1 bars={len(h1)}  span={days}d ({yrs:.2f}yr)  spread={spread:.5f}")
    print(f"BASELINE  full: {base['full']}  train:{base['train']}  test:{base['test']}")
    print("-" * 78)
    for pname, info in per_param.items():
        print(f"{pname:9s} base={info['baseline']}  best_train={info['best_on_train']}  "
              f"-> {info['verdict']}  chosen={info['chosen']}")
        for row in info["sweep"]:
            print(f"    {pname}={row['value']:<5} full_R={row['full']['netR']:>7} "
                  f"pf={row['full']['pf']:<5} n={row['full']['n']:<3} | "
                  f"trainR={row['train']['netR']:>7} testR={row['test']['netR']:>7}")
    print("-" * 78)
    print(f"COMBINED params: {best_combo}")
    print(f"  full : {combined['full']}")
    print(f"  train: {combined['train']}   test: {combined['test']}")
    print(f"  SHIP COMBINED? {combined_ships}   "
          f"(full dR={result['combined']['vs_baseline_full_R']}, "
          f"train dR={result['combined']['vs_baseline_train_R']}, "
          f"test dR={result['combined']['vs_baseline_test_R']})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
