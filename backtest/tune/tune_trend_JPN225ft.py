#!/usr/bin/env python3 -B
"""Hard-tune EVERY parameter of the live TREND book for JPN225ft.

Book under test (magic +6000, live):
  * D1 3-EMA ensemble signal  (agent/trend_follower.evaluate)
      sig = mean_over_pairs( sign( EMA_fast - EMA_slow ) );
      dir = 0 if |sig| < MIN_ABS_SIGNAL else sign(sig)
  * SL = ATR_STOP * ATR(ATR_PERIOD)   (catastrophic tail guard, D1)
  * TP = TP_ATR * ATR                  (realistic target, D1)
  * H1 intraday exits (checked every H1 bar, ~live 60s cadence):
      - chandelier trail   sl = HH_22 - TRAIL*ATR  (long) / LL_22 + TRAIL*ATR (short)
      - profit-lock        once peak >= ACT*ATR:  sl = entry + LOCK*peak
      - peak-giveback      once peak >= ACT*ATR:  exit if price gives back GIVEBACK*peak
      - daily signal flip  close + reverse when D1 dir reverses
  Objective = total R (1R = the actual ATR_STOP*ATR risk at entry), PF robustness.

Faithful to the VALIDATED reference scripts/_trend_exit_tune_h1.py and the
working reconstruction jpn_losscap_tune.py (42 trades, 3.5yr, WR ~90%).

METHOD
  * Full history from raw_d1/raw_h1_JPN225ft.pkl.
  * Sweep ONE param at a time around the live baseline; then combine the
    WF-validated winners and re-check jointly.
  * WF 60/40 chronological on the trade sequence. SHIP a value only if it
    (a) strictly beats baseline train_R  AND
    (b) is non-regressive on test_R (>= baseline test_R - eps).
  * n ~ 42 (slow, low-turnover book). Small-n: prefer robust/neutral over
    marginal in-sample wins; flag any "win" that rests on < 5 trades of
    difference. SHIP_NONE is the honest default for most params.

Deliverables written to backtest/tune/:
  tune_trend_JPN225ft.py   (this file, self-contained)
  results_trend_JPN225ft.json
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
SYM = "JPN225ft"
POINT = 0.01                 # JPN225ft broker point
WF_TRAIN_FRAC = 0.60
EPS = 1e-9
# --- churn / re-entry artifact guard ---------------------------------------
# The TREND book is designed to HOLD ONE position and trail it. In the sim the
# re-entry `blocked` flag is only set on a giveback exit; a value that makes the
# lock-SL / chandelier-trail fire BEFORE giveback (e.g. LOCK >= 1-GIVEBACK, or a
# looser signal) leaves `blocked` unset, so the same-direction position closes
# and RE-OPENS every H1 bar in a trend — manufacturing hundreds/thousands of
# fake tiny-profit trades. Those totals are a simulation artifact, NOT edge, so
# any value whose trade count inflates past this factor of baseline is
# disqualified from SHIP (and tagged ARTIFACT in the report).
ARTIFACT_N_FACTOR = 1.5
OUT_JSON = Path(__file__).with_name("results_trend_JPN225ft.json")

# ---- LIVE BASELINE (config.py) --------------------------------------------
BASE = dict(
    EMA_PAIRS=[(16, 64), (32, 128), (64, 256)],   # TREND_EMA_PAIRS (default)
    MIN_ABS=0.34,                                  # TREND_MIN_ABS_SIGNAL
    ATR_P=20,                                      # TREND_ATR_PERIOD
    ATR_STOP=3.0,                                  # TREND_ATR_STOP
    TP=6.0,                                        # TREND_TP_ATR
    TRAIL_LOOKBACK=22,                             # TREND_TRAIL_LOOKBACK
    TR=3.0, LK=0.6, GB=0.35, ACT=0.3,              # TREND_EXIT_PER_SYMBOL[JPN225ft]
)

# ---- SWEEP GRIDS -----------------------------------------------------------
# EMA alternatives: default, faster, slower (all 3-speed geometric ensembles).
EMA_ALTS = {
    "default":      [(16, 64), (32, 128), (64, 256)],
    "faster":       [(8, 32), (16, 64), (32, 128)],
    "slower":       [(24, 96), (48, 192), (96, 384)],
}
GRID = {
    "ATR_STOP": [2.0, 2.5, 3.0, 3.5, 4.0],
    "ATR_P":    [14, 20, 28],
    "TP":       [4.0, 5.0, 6.0, 8.0, 10.0, None],   # None => no TP
    "TR":       [2.0, 2.5, 3.0, 3.5],
    "MIN_ABS":  [0.20, 0.34, 0.50, 0.67],
    "EMA":      ["default", "faster", "slower"],     # keyed into EMA_ALTS
    "LK":       [0.5, 0.6, 0.7],
    "GB":       [0.25, 0.35, 0.50],
    "ACT":      [0.2, 0.3, 0.4],
}
TP_NONE = 1e9   # internal sentinel: TP disabled (>= 999 => tp=None in sim)


# ---------------------------------------------------------------------------
def _naive(s):
    s = pd.to_datetime(s)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


def _load_raw():
    d = pickle.load(open(C / f"raw_d1_{SYM}.pkl", "rb"))
    d["time"] = _naive(d["time"]); d = d.sort_values("time").reset_index(drop=True)
    h1 = pickle.load(open(C / f"raw_h1_{SYM}.pkl", "rb"))
    h1["time"] = _naive(h1["time"]); h1 = h1.sort_values("time").reset_index(drop=True)
    return d, h1


def d1_context(d, ema_pairs, min_abs, atr_p, lookback):
    c, h, l = d["close"], d["high"], d["low"]
    sig = pd.Series(0.0, index=d.index)
    for f, s in ema_pairs:
        sig = sig + np.sign(c.ewm(span=f).mean() - c.ewm(span=s).mean())
    sig = (sig / len(ema_pairs)).apply(
        lambda v: 0 if abs(v) < min_abs else (1 if v > 0 else -1))
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_p, adjust=False).mean()
    out = pd.DataFrame({"eff": d["time"].dt.normalize() + pd.Timedelta(days=1),
                        "sig": sig.astype(int), "atr": atr,
                        "hh": h.rolling(lookback).max(),
                        "ll": l.rolling(lookback).min()})
    return out.dropna().reset_index(drop=True)


def build(d, h1, p):
    d1 = d1_context(d, p["EMA_PAIRS"], p["MIN_ABS"], p["ATR_P"], p["TRAIL_LOOKBACK"])
    hh = h1.copy()
    hh["time"] = hh["time"].astype("datetime64[ns]")
    d1["eff"] = d1["eff"].astype("datetime64[ns]")
    m = pd.merge_asof(hh, d1, left_on="time", right_on="eff", direction="backward").dropna(
        subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)
    return m


def simulate(m, p, cost):
    """Return per-trade R list (pnl already normalised by entry sl_dist)."""
    ATR_STOP = p["ATR_STOP"]; TR = p["TR"]; LK = p["LK"]; GB = p["GB"]
    ACT = p["ACT"]; TP = TP_NONE if p["TP"] is None else p["TP"]
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    pos = 0; entry = sl = peak = 0.0; tp = None; blocked = 0
    sl_dist = 0.0
    trades = []

    def close(exit_px):
        return ((exit_px - entry) * pos) / sl_dist - cost / sl_dist

    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blocked and s != blocked:
                blocked = 0
            if s != 0 and s != blocked:
                pos = s; entry = o[t]; peak = 0.0
                sl_dist = ATR_STOP * a
                sl = entry - ATR_STOP * a if s == 1 else entry + ATR_STOP * a
                tp = None if TP >= 999 else ((entry + TP * a) if s == 1 else (entry - TP * a))
            continue
        if s != 0 and s != pos:                       # daily signal flip
            trades.append(close(o[t]))
            pos = s; entry = o[t]; peak = 0.0; blocked = 0
            sl_dist = ATR_STOP * a
            sl = entry - ATR_STOP * a if pos == 1 else entry + ATR_STOP * a
            tp = None if TP >= 999 else ((entry + TP * a) if pos == 1 else (entry - TP * a))
            continue
        if pos == 1:
            sl = max(sl, hh[t] - TR * a)
            if peak >= ACT * a:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else -1e18
            ex = None
            if l[t] <= sl:
                ex = sl
            elif gb > -1e17 and l[t] <= gb:
                ex = gb; blocked = pos
            elif tp is not None and h[t] >= tp:
                ex = tp
            if ex is not None:
                trades.append(close(ex)); pos = 0
            else:
                peak = max(peak, h[t] - entry)
        else:
            sl = min(sl, ll[t] + TR * a)
            if peak >= ACT * a:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else 1e18
            ex = None
            if h[t] >= sl:
                ex = sl
            elif gb < 1e17 and h[t] >= gb:
                ex = gb; blocked = pos
            elif tp is not None and l[t] <= tp:
                ex = tp
            if ex is not None:
                trades.append(close(ex)); pos = 0
            else:
                peak = max(peak, entry - l[t])
    return trades


def pf(rs):
    rs = np.array(rs, dtype=float)
    if rs.size == 0:
        return 0.0
    w = rs[rs > 0].sum(); ls = -rs[rs < 0].sum()
    return float(w / ls) if ls > 0 else 99.0


def wr(rs):
    return float(np.mean([r > 0 for r in rs])) if rs else 0.0


def wf(trades):
    """60/40 chronological split -> (train_R, test_R, n_train, n_test)."""
    n = len(trades)
    split = int(n * WF_TRAIN_FRAC)
    tr, te = trades[:split], trades[split:]
    return sum(tr), sum(te), len(tr), len(te)


def evaluate(d, h1, p, cost):
    m = build(d, h1, p)
    trades = simulate(m, p, cost)
    tr_R, te_R, n_tr, n_te = wf(trades)
    return {
        "n": len(trades), "total_R": round(sum(trades), 4), "PF": round(pf(trades), 3),
        "WR": round(wr(trades), 3),
        "train_R": round(tr_R, 4), "test_R": round(te_R, 4),
        "n_train": n_tr, "n_test": n_te,
    }


def main():
    d, h1 = _load_raw()
    m0 = build(d, h1, {**BASE, "EMA_PAIRS": BASE["EMA_PAIRS"]})
    cost = 2.0 * (float(np.nanmedian(m0["spread"].values)) * POINT)   # round-trip price units
    span = "%s..%s" % (str(m0['time'].iloc[0])[:10], str(m0['time'].iloc[-1])[:10])

    base_p = dict(BASE)
    base = evaluate(d, h1, base_p, cost)
    print("=== JPN225ft TREND full-parameter hard-tune ===")
    print("H1 bars=%d  D1 span=%s  round-trip cost=%.1f price units" % (len(m0), span, cost))
    print("BASELINE  n=%d  total_R=%.3f  PF=%.2f  WR=%.1f%%  | WF train_R=%.3f (n=%d)  test_R=%.3f (n=%d)"
          % (base["n"], base["total_R"], base["PF"], 100 * base["WR"],
             base["train_R"], base["n_train"], base["test_R"], base["n_test"]))
    print()

    base_train, base_test = base["train_R"], base["test_R"]
    base_n = base["n"]
    artifact_n = base_n * ARTIFACT_N_FACTOR
    per_param = []
    ship_overrides = {}   # param -> chosen value (only WF-validated, non-artifact wins)

    def base_value(param):
        if param == "EMA":
            return "default"
        return {"ATR_STOP": BASE["ATR_STOP"], "ATR_P": BASE["ATR_P"], "TP": BASE["TP"],
                "TR": BASE["TR"], "MIN_ABS": BASE["MIN_ABS"], "LK": BASE["LK"],
                "GB": BASE["GB"], "ACT": BASE["ACT"]}[param]

    def apply_value(p, param, v):
        q = dict(p)
        if param == "EMA":
            q["EMA_PAIRS"] = EMA_ALTS[v]
        else:
            q[param] = v
        return q

    for param, values in GRID.items():
        rows = []
        for v in values:
            p = apply_value(base_p, param, v)
            r = evaluate(d, h1, p, cost)
            r["value"] = v
            r["artifact"] = bool(r["n"] > artifact_n)   # re-entry churn guard
            rows.append(r)
        bval = base_value(param)
        base_row = next(r for r in rows if r["value"] == bval)
        # pick best-on-train among NON-artifact values only.
        clean = [r for r in rows if not r["artifact"]]
        best = max(clean, key=lambda r: r["train_R"])   # baseline is always clean
        # SHIP criteria: strictly beats baseline train AND non-regressive on test.
        beats_train = best["train_R"] > base_train + EPS
        keeps_test = best["test_R"] >= base_test - EPS
        is_baseline = (best["value"] == bval)
        # small-n flag: thin folds or a win that rests on very few trades.
        train_edge = best["train_R"] - base_row["train_R"]
        small_n_flag = (best["n"] < 20) or (best["n_test"] < 5) or (best["n_train"] < 5)
        ship = (not is_baseline) and beats_train and keeps_test
        verdict = "SHIP" if ship else "SHIP_NONE"
        n_artifact = sum(r["artifact"] for r in rows)
        if ship:
            ship_overrides[param] = best["value"]

        print("--- %s ---  (baseline value=%s)" % (param, bval))
        for r in rows:
            mark = " <-BEST" if (r is best) else ""
            bmark = " (base)" if r["value"] == bval else ""
            amark = " [ARTIFACT-churn]" if r["artifact"] else ""
            print("  %-8s n=%-4d totR=%8.3f PF=%6.2f WR=%4.0f%% | trainR=%8.3f(n=%d) testR=%8.3f(n=%d)%s%s%s"
                  % (str(r["value"]), r["n"], r["total_R"], r["PF"], 100 * r["WR"],
                     r["train_R"], r["n_train"], r["test_R"], r["n_test"], bmark, mark, amark))
        print("  => best CLEAN value=%s  beats_train=%s  keeps_test=%s  small_n=%s  n_artifact=%d/%d  =>  %s"
              % (str(best["value"]), beats_train, keeps_test, small_n_flag, n_artifact, len(rows), verdict))
        if ship and small_n_flag:
            print("     [FLAG] small-n: win rests on thin data (best n=%d, test n=%d, train_edge=%.3fR) — treat as fragile"
                  % (best["n"], best["n_test"], train_edge))
        print()

        per_param.append({
            "param": param,
            "baseline_value": bval,
            "values": [{k: r[k] for k in ("value", "n", "total_R", "PF", "WR",
                                          "train_R", "test_R", "n_train", "n_test", "artifact")}
                       for r in rows],
            "best_clean_value": best["value"],
            "best_train_R": best["train_R"], "best_test_R": best["test_R"],
            "best_n": best["n"], "best_n_train": best["n_train"], "best_n_test": best["n_test"],
            "beats_train": bool(beats_train), "keeps_test": bool(keeps_test),
            "small_n_flag": bool(small_n_flag),
            "n_artifact_values": int(n_artifact),
            "verdict": verdict,
        })

    # ---- COMBINED: apply all WF-validated overrides jointly -----------------
    combined_p = dict(base_p)
    for param, v in ship_overrides.items():
        combined_p = apply_value(combined_p, param, v)
    combined = evaluate(d, h1, combined_p, cost)
    comb_ship = (combined["train_R"] > base_train + EPS) and (combined["test_R"] >= base_test - EPS)

    print("=== COMBINED (WF-validated overrides only) ===")
    if ship_overrides:
        print("overrides: %s" % ship_overrides)
    else:
        print("overrides: NONE  (every param returned SHIP_NONE)")
    print("BASELINE  total_R=%.3f PF=%.2f | train_R=%.3f test_R=%.3f"
          % (base["total_R"], base["PF"], base_train, base_test))
    print("COMBINED  total_R=%.3f PF=%.2f | train_R=%.3f test_R=%.3f  =>  %s"
          % (combined["total_R"], combined["PF"], combined["train_R"], combined["test_R"],
             "SHIP" if (ship_overrides and comb_ship) else "SHIP_NONE"))

    notes = [
        "n~42 over 3.5yr: slow, high-WR, low-turnover book. Tiny total_R because "
        "exits are tight (most trades = small profit-lock/giveback wins).",
        "1R is defined as the entry-time ATR_STOP*ATR risk, so total_R is risk-normalised "
        "and comparable across ATR_STOP values.",
        "MIN_ABS 0.34 and 0.50 are identical (both require >=2/3 EMA-pairs to agree); "
        "0.20 requires >=1/3; 0.67 requires all 3.",
        "ARTIFACT GUARD: values whose trade count inflates past %.1fx baseline (%d) are "
        "disqualified. The sim's re-entry `blocked` flag is only set on a giveback exit; a "
        "value that makes the lock-SL/chandelier-trail fire BEFORE giveback (LOCK>=1-GIVEBACK, "
        "or a looser signal) closes and RE-OPENS the same-direction position every H1 bar in a "
        "trend, manufacturing hundreds/thousands of fake tiny-profit trades. Those 100-400+R "
        "totals (TR=2.0, MIN_ABS=0.2, EMA slower/faster, LK=0.7, GB=0.5) are churn, not edge."
        % (ARTIFACT_N_FACTOR, base_n),
        "SHIP rule: best CLEAN (non-artifact) value must strictly beat baseline train_R AND be "
        "non-regressive on test_R. small_n_flag marks wins on <20 total trades or <5 in a fold.",
        "ATR_STOP 2.5 CAVEAT: n/PF/WR/exits are IDENTICAL to baseline (42, 2.49, 90%) — the 3.0*ATR "
        "catastrophic stop never bound on any of the 42 trades, so 2.5 is a pure risk-unit RESCALE "
        "(every R x1.2 = 3.0/2.5), i.e. size 1.2x larger for the same signals. It is NOT a new edge "
        "and it removes tail cushion for a future trade that exceeds 2.5*ATR before the trail catches. "
        "Ship only if you accept the sizing/tail trade-off; the honest edge win here is ACT.",
        "ACT 0.4 is the one GENUINE outcome change: delaying profit-lock activation (0.3->0.4*ATR) "
        "lets winners breathe; n 42->51 (below the 1.5x churn guard), PF 2.49->3.12, both folds up "
        "(train +0.23R, test +0.86R). Robust, WF-validated, non-churn.",
        "Deliverables are analysis-only; config.py / brain.py are NOT modified.",
    ]

    result = {
        "symbol": SYM,
        "generated": "2026-07-15",
        "data": {"h1_bars": int(len(m0)), "d1_span": span, "cost_price_units": round(cost, 2),
                 "wf_train_frac": WF_TRAIN_FRAC},
        "baseline": {"params": {k: (v if not isinstance(v, list) else str(v))
                                for k, v in BASE.items()},
                     **base},
        "per_param": per_param,
        "combined": {"overrides": ship_overrides, **combined,
                     "verdict": "SHIP" if (ship_overrides and comb_ship) else "SHIP_NONE"},
        "notes": notes,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
    print("\nwrote %s" % OUT_JSON)


if __name__ == "__main__":
    main()
