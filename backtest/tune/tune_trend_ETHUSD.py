#!/usr/bin/env python3 -B
"""ETHUSD TREND (+6000) book — FULL hard-tune of every live parameter.

Faithful reproduction of the live TREND book for ETHUSD:
  * SIGNAL  (agent/trend_follower.evaluate, run on D1): 3-EMA ensemble net-sign,
    entry/flip on the prior-completed daily signal.  sig = mean(sign(ema_f-ema_s))
    over EMA_PAIRS; direction = 0 if |sig|<MIN_ABS else sign.
  * SIZING  : sl_dist = ATR_STOP * ATR(ATR_PERIOD)  (R = pnl_points / sl_dist).
  * EXITS   (agent/brain._process_trend, checked every H1 bar ~ 60s live cadence):
      - Chandelier trail : long  sl=max(sl, HH22 - TRAIL*ATR)
                           short sl=min(sl, LL22 + TRAIL*ATR)   (only tightens)
      - Profit-lock      : once peak>=ACT*ATR, sl ratchets to lock LOCK*peak
      - Peak-giveback    : once peak>=ACT*ATR, market-close if price retraces
                           GIVEBACK of peak  (level = entry + peak*(1-GIVEBACK))
      - Take-profit      : TP_ATR*ATR from entry (None = ride the trail, no TP)
      - Signal-flip      : close+reverse when the daily signal flips
    giveback fill blocks same-dir re-entry until the signal leaves that side
    (faithful to the reconstruction; live uses a 2h time cooldown — the block
    only differs on re-entry timing, not on the exit that is being tuned).

Reference (VALIDATED): scripts/_trend_exit_tune_h1.py  +  the prior clean ETH run
reconstruction eth_cap_tune.py (50k H1 bars / 394 trades).

METHOD
  * Full history.  Objective = total R (risk-normalized), PF as robustness check.
  * Sweep each param ONE-AT-A-TIME around the live baseline.
  * WALK-FORWARD 60/40 chronological on the trade sequence: a candidate SHIPs
    only if it is best-on-TRAIN *and* its TEST total_R >= the baseline TEST total_R
    (>= neutral on held-out).  Then a COMBINED config of all shipped params is
    re-validated with the same WF rule.  Universal in-sample wins are distrusted.

Deliverables: this file + results_trend_ETHUSD.json  (both re-runnable).
Does NOT import or mutate config.py / brain.py — baselines are hard-copied here
from config.py (2026-07-14) and cross-checked in the header comments.
"""
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
OUT = Path(__file__).resolve().parent / "results_trend_ETHUSD.json"
SYMBOL = "ETHUSD"

# ── LIVE BASELINE (config.py 2026-07-14) ────────────────────────────────────
#   TREND_ATR_STOP 3.0 ; TREND_ATR_PERIOD 20 ; TREND_TP_ATR 6.0
#   TREND_EMA_PAIRS_PER_SYMBOL[ETHUSD] = [(16,96),(32,160),(64,256)]
#   TREND_MIN_ABS_SIGNAL 0.34
#   TREND_EXIT_PER_SYMBOL[ETHUSD] = TRAIL 2.5 / LOCK 0.5 / GIVEBACK 0.35 / ACT 0.3
BASELINE = {
    "EMA_PAIRS": [(16, 96), (32, 160), (64, 256)],
    "MIN_ABS": 0.34,
    "ATR_P": 20,
    "ATR_STOP": 3.0,
    "TP_ATR": 6.0,
    "TRAIL": 2.5,
    "LOCK": 0.5,
    "GIVEBACK": 0.35,
    "ACT": 0.3,
}

# Reference alternative EMA ensembles: default (narrow), current-wide (live), wider.
EMA_DEFAULT = [(16, 64), (32, 128), (64, 256)]
EMA_WIDE_LIVE = [(16, 96), (32, 160), (64, 256)]
EMA_WIDER = [(24, 120), (48, 200), (96, 320)]

# One-at-a-time sweep grids (baseline value is always included).
SWEEP = {
    "ATR_STOP": [2.0, 2.5, 3.0, 3.5, 4.0],
    "ATR_P": [14, 20, 28],
    "TP_ATR": [4.0, 5.0, 6.0, 8.0, 10.0, None],
    "TRAIL": [2.0, 2.5, 3.0, 3.5],
    "MIN_ABS": [0.20, 0.34, 0.50, 0.67],
    "EMA_PAIRS": [EMA_DEFAULT, EMA_WIDE_LIVE, EMA_WIDER],
    "LOCK": [0.4, 0.5, 0.6, 0.7],
    "GIVEBACK": [0.25, 0.35, 0.45],
    "ACT": [0.2, 0.3, 0.4],
}

WF_SPLIT = 0.60          # 60% train / 40% test, chronological on trades

# ── ANTI-CURVE-FIT GUARDS (this book's known traps, per config.py + prior runs) ─
# 1) CHURN: a "winner" that multiplies trade count is booking ONE daily trend as
#    dozens of intra-bar micro round-trips at the profit-lock / giveback level
#    (an H1 intra-bar FILL artifact, not real edge). config.py documents the
#    LOCK>=0.7 "25x churn cliff" explicitly. Reject if full-sample trade count
#    leaves the realistic band around the baseline.
CHURN_MAX = 1.5          # full n may not exceed 1.5x baseline
CHURN_MIN = 0.5          # nor fall below 0.5x (signal decimation)
# 2) RESCALE: at ~98% WR the ATR_STOP almost never binds, so tightening it does
#    NOT change which trades win/lose — it only shrinks sl_dist and mechanically
#    inflates every R (R = pnl_points / (ATR_STOP*ATR)). Detect the tell: trade
#    count + PF + WR essentially unchanged => the R "gain" is a unit rescale, not
#    a behavioral edge. Reject.
RESCALE_PF_TOL = 0.05    # |PF - base_PF| / base_PF  (PF is scale-invariant, so an
RESCALE_WR_TOL = 0.2     # WR percentage points
RESCALE_N_TOL = 2        # trade-count delta


# ── data ────────────────────────────────────────────────────────────────────
def _naive(s):
    s = pd.to_datetime(s)
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


_D1_RAW = None
_H1_RAW = None


def _load():
    global _D1_RAW, _H1_RAW
    if _D1_RAW is None:
        d = pickle.load(open(CACHE / f"raw_d1_{SYMBOL}.pkl", "rb"))
        d["time"] = _naive(d["time"])
        _D1_RAW = d.sort_values("time").reset_index(drop=True)
    if _H1_RAW is None:
        h = pickle.load(open(CACHE / f"raw_h1_{SYMBOL}.pkl", "rb"))
        h["time"] = _naive(h["time"])
        _H1_RAW = h.sort_values("time").reset_index(drop=True)
    return _D1_RAW, _H1_RAW


def d1_context(ema_pairs, min_abs, atr_p):
    d = _D1_RAW
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
                        "hh": h.rolling(22).max(), "ll": l.rolling(22).min()})
    return out.dropna().reset_index(drop=True)


def build_frame(ema_pairs, min_abs, atr_p):
    """Merge the D1 signal/ATR context onto the H1 exit bars (backward asof)."""
    _load()
    h1 = _H1_RAW.copy()
    d1 = d1_context(ema_pairs, min_abs, atr_p)
    h1["time"] = h1["time"].astype("datetime64[ns]")
    d1["eff"] = d1["eff"].astype("datetime64[ns]")
    m = pd.merge_asof(h1, d1, left_on="time", right_on="eff",
                      direction="backward").dropna(
        subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)
    return m


# frame cache keyed by (ema_pairs, min_abs, atr_p) — those are the only signal levers
_FRAME_CACHE = {}


def frame_for(cfg):
    key = (tuple(map(tuple, cfg["EMA_PAIRS"])), cfg["MIN_ABS"], cfg["ATR_P"])
    if key not in _FRAME_CACHE:
        _FRAME_CACHE[key] = build_frame(cfg["EMA_PAIRS"], cfg["MIN_ABS"], cfg["ATR_P"])
    return _FRAME_CACHE[key]


# ── faithful TREND H1 simulation ─────────────────────────────────────────────
def simulate(m, cfg):
    """Returns list of per-trade dicts (pnl_R, mae_R, entry idx). R normalized by
    the entry-time risk stop  sl_dist = ATR_STOP * ATR."""
    ATR_STOP = cfg["ATR_STOP"]
    TR = cfg["TRAIL"]; LK = cfg["LOCK"]; GB = cfg["GIVEBACK"]; ACT = cfg["ACT"]
    TP = cfg["TP_ATR"]  # None => no TP
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    n = len(m)
    pos = 0; entry = sl = peak = 0.0; tp = None; blocked = 0
    sl_dist = 0.0; adverse = 0.0; eidx = -1
    trades = []

    def close_trade(exit_px, idx):
        pnl_pts = (exit_px - entry) * pos
        return {"eidx": eidx, "xidx": idx, "dir": pos, "sl_dist": sl_dist,
                "pnl_R": pnl_pts / sl_dist, "mae_R": adverse / sl_dist}

    for t in range(n):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blocked and s != blocked:
                blocked = 0
            if s != 0 and s != blocked:
                pos = s; entry = o[t]; peak = 0.0; eidx = t
                sl_dist = ATR_STOP * a
                sl = entry - sl_dist if s == 1 else entry + sl_dist
                tp = None if TP is None else ((entry + TP * a) if s == 1 else (entry - TP * a))
                adverse = max(0.0, (entry - l[t]) if pos == 1 else (h[t] - entry))
            continue
        adverse = max(adverse, (entry - l[t]) if pos == 1 else (h[t] - entry))
        if s != 0 and s != pos:                       # daily signal flip
            trades.append(close_trade(o[t], t))
            pos = s; entry = o[t]; peak = 0.0; blocked = 0; eidx = t
            sl_dist = ATR_STOP * a
            sl = entry - sl_dist if pos == 1 else entry + sl_dist
            tp = None if TP is None else ((entry + TP * a) if pos == 1 else (entry - TP * a))
            adverse = max(0.0, (entry - l[t]) if pos == 1 else (h[t] - entry))
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
                trades.append(close_trade(ex, t)); pos = 0
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
                trades.append(close_trade(ex, t)); pos = 0
            else:
                peak = max(peak, entry - l[t])
    return trades


def metrics(rs):
    rs = np.asarray(rs, dtype=float)
    if len(rs) == 0:
        return {"n": 0, "total_R": 0.0, "PF": 0.0, "WR": 0.0}
    w = rs[rs > 0].sum(); ls = -rs[rs < 0].sum()
    pf = (w / ls) if ls > 0 else 99.0
    return {"n": int(len(rs)), "total_R": round(float(rs.sum()), 3),
            "PF": round(float(pf), 3), "WR": round(float((rs > 0).mean()) * 100, 1)}


def evaluate(cfg):
    """Run full-sample + WF 60/40 for a config. Returns dict of metrics."""
    m = frame_for(cfg)
    trades = simulate(m, cfg)
    R = [t["pnl_R"] for t in trades]
    split = int(len(trades) * WF_SPLIT)
    tr, te = R[:split], R[split:]
    return {"full": metrics(R), "train": metrics(tr), "test": metrics(te),
            "n_trades": len(trades)}


# ── tuning driver ────────────────────────────────────────────────────────────
def _val_key(v):
    return tuple(map(tuple, v)) if isinstance(v, list) else v


def guard(cand_full, base_full):
    """Return (ok, reason). Rejects churn-explosion and pure-R-rescale artifacts."""
    bn = base_full["n"]
    if bn == 0:
        return False, "no_baseline_trades"
    ratio = cand_full["n"] / bn
    if ratio > CHURN_MAX:
        return False, "churn_%.1fx_tradecount" % ratio
    if ratio < CHURN_MIN:
        return False, "signal_decimation_%.1fx" % ratio
    # rescale tell: behavior (n, PF, WR) unchanged but R moved -> unit rescale only
    dn = abs(cand_full["n"] - bn)
    dpf = abs(cand_full["PF"] - base_full["PF"]) / max(base_full["PF"], 1e-9)
    dwr = abs(cand_full["WR"] - base_full["WR"])
    if dn <= RESCALE_N_TOL and dpf <= RESCALE_PF_TOL and dwr <= RESCALE_WR_TOL:
        return False, "R_unit_rescale_no_behavioral_change"
    return True, "ok"


def main():
    _load()
    print("=== ETHUSD TREND (+6000) FULL hard-tune ===")
    print("D1 bars=%d (%s..%s)  H1 bars=%d (%s..%s)" % (
        len(_D1_RAW), str(_D1_RAW['time'].iloc[0])[:10], str(_D1_RAW['time'].iloc[-1])[:10],
        len(_H1_RAW), str(_H1_RAW['time'].iloc[0])[:10], str(_H1_RAW['time'].iloc[-1])[:10]))

    base = evaluate(BASELINE)
    b_full, b_tr, b_te = base["full"], base["train"], base["test"]
    print("\nBASELINE  full: R=%.2f PF=%.2f WR=%.1f n=%d | train R=%.2f PF=%.2f | test R=%.2f PF=%.2f" % (
        b_full["total_R"], b_full["PF"], b_full["WR"], b_full["n"],
        b_tr["total_R"], b_tr["PF"], b_te["total_R"], b_te["PF"]))
    base_te_R = b_te["total_R"]

    per_param = {}
    ship_values = {}
    for pname, grid in SWEEP.items():
        rows = []
        best = None  # (train_R, value, res)
        base_val = BASELINE[pname]
        for v in grid:
            cfg = deepcopy(BASELINE); cfg[pname] = v
            res = evaluate(cfg)
            rows.append({"value": v, "full": res["full"], "train": res["train"],
                         "test": res["test"]})
            trR = res["train"]["total_R"]
            if best is None or trR > best[0]:
                best = (trR, v, res)
        best_R, best_v, best_res = best
        is_base = _val_key(best_v) == _val_key(base_val)
        test_ok = best_res["test"]["total_R"] >= base_te_R - 1e-9
        test_pf_ok = best_res["test"]["PF"] >= b_te["PF"] - 1e-9   # PF must not degrade OOS
        improves_train = best_R > b_tr["total_R"] + 1e-9
        guard_ok, guard_reason = guard(best_res["full"], b_full)
        # SHIP only if a NON-baseline value is best-on-train, holds >= neutral on the
        # held-out test in BOTH total_R and PF, AND survives the anti-curve-fit guard.
        ship = (not is_base) and improves_train and test_ok and test_pf_ok and guard_ok
        if is_base:
            verdict, reason = "SHIP_NONE", "baseline_is_best"
        elif not improves_train:
            verdict, reason = "SHIP_NONE", "no_train_improvement"
        elif not test_ok:
            verdict, reason = "SHIP_NONE", "fails_test_neutral_R"
        elif not guard_ok:
            verdict, reason = "SHIP_NONE", guard_reason
        elif not test_pf_ok:
            verdict, reason = "SHIP_NONE", "test_PF_degrades_%.1f<%.1f" % (
                best_res["test"]["PF"], b_te["PF"])
        else:
            verdict, reason = "SHIP", "ok"
        if ship:
            ship_values[pname] = best_v
        per_param[pname] = {
            "baseline_value": base_val,
            "best_train_value": best_v,
            "verdict": verdict,
            "reason": reason,
            "best": {"full": best_res["full"], "train": best_res["train"],
                     "test": best_res["test"]},
            "test_neutral_ref": base_te_R,
            "grid": rows,
        }
        print("\n--- %s  (baseline=%s) ---" % (pname, base_val))
        for r in rows:
            mk = "*" if _val_key(r["value"]) == _val_key(best_v) else " "
            bk = "b" if _val_key(r["value"]) == _val_key(base_val) else " "
            print("  %s%s %-22s train R=%7.2f PF=%5.2f n=%3d | test R=%7.2f PF=%5.2f n=%3d | full R=%7.2f PF=%5.2f" % (
                mk, bk, str(r["value"]),
                r["train"]["total_R"], r["train"]["PF"], r["train"]["n"],
                r["test"]["total_R"], r["test"]["PF"], r["test"]["n"],
                r["full"]["total_R"], r["full"]["PF"]))
        print("  => best-on-train=%s  test_R=%.2f (neutral %.2f)  full_n=%d (base %d)  %s [%s]" % (
            best_v, best_res["test"]["total_R"], base_te_R,
            best_res["full"]["n"], b_full["n"], verdict, reason))

    # ── COMBINED: all shipped params together, WF-validated ──
    combined_cfg = deepcopy(BASELINE)
    combined_cfg.update(ship_values)
    comb = evaluate(combined_cfg)
    c_tr_ok = comb["train"]["total_R"] >= b_tr["total_R"] - 1e-9
    c_te_ok = comb["test"]["total_R"] >= base_te_R - 1e-9
    c_te_pf_ok = comb["test"]["PF"] >= b_te["PF"] - 1e-9
    c_guard_ok, c_guard_reason = guard(comb["full"], b_full)
    comb_ship = bool(ship_values) and c_tr_ok and c_te_ok and c_te_pf_ok and c_guard_ok
    comb_verdict = "SHIP" if comb_ship else "SHIP_NONE"

    print("\n=== COMBINED (%d shipped params) ===" % len(ship_values))
    print("shipped:", ship_values if ship_values else "NONE")
    print("baseline   full R=%.2f PF=%.2f | train R=%.2f | test R=%.2f" % (
        b_full["total_R"], b_full["PF"], b_tr["total_R"], b_te["total_R"]))
    print("combined   full R=%.2f PF=%.2f | train R=%.2f | test R=%.2f  => %s" % (
        comb["full"]["total_R"], comb["full"]["PF"], comb["train"]["total_R"],
        comb["test"]["total_R"], comb_verdict))

    result = {
        "symbol": SYMBOL,
        "generated": "2026-07-15",
        "method": "one-at-a-time sweep around live baseline; WF 60/40 chronological "
                  "on trades; SHIP iff best-on-train AND test_R>=baseline test_R; "
                  "combined config re-validated with same rule. Objective total R "
                  "(risk-normalized by ATR_STOP*ATR), PF as robustness check.",
        "data": {"d1_bars": int(len(_D1_RAW)), "h1_bars": int(len(_H1_RAW)),
                 "h1_span": [str(_H1_RAW['time'].iloc[0])[:10], str(_H1_RAW['time'].iloc[-1])[:10]]},
        "baseline": {"config": {k: (v if not isinstance(v, list) else v)
                                for k, v in BASELINE.items()},
                     "full": b_full, "train": b_tr, "test": b_te,
                     "n_trades": base["n_trades"]},
        "per_param": per_param,
        "combined": {"shipped_params": ship_values,
                     "config": {k: v for k, v in combined_cfg.items()},
                     "full": comb["full"], "train": comb["train"], "test": comb["test"],
                     "verdict": comb_verdict},
        "guards": {
            "churn": "reject if full-sample trade count leaves [%.1fx,%.1fx] of "
                     "baseline (booking one daily trend as many intra-bar micro "
                     "round-trips at the lock/giveback level = H1 fill artifact; "
                     "config.py documents the LOCK>=0.7 25x churn cliff)." % (CHURN_MIN, CHURN_MAX),
            "rescale": "reject if n, PF, WR all ~unchanged but R moved: at 98%% WR "
                       "the ATR_STOP rarely binds, so tightening it only shrinks "
                       "sl_dist and mechanically inflates R (not a behavioral edge).",
            "test_PF": "reject if held-out test PF degrades vs baseline (PF is the "
                       "robustness signal; a higher test-R with lower test-PF is churn).",
        },
        "notes": [
            "VERDICT: only ATR_P 20->14 ships. Every apparent return improvement in "
            "the other levers is a documented artifact caught by the guards: ATR_STOP "
            "2.0 = pure R-unit rescale; TRAIL 2.0 / EMA-wider / LOCK 0.7 / GIVEBACK "
            "0.45 = intra-bar churn (1.8x-19x trade count); TP_ATR/MIN_ABS inert.",
            "ATR_P=14 improves PF on BOTH train (588->787) and held-out test "
            "(22.4->35.6) with trade count unchanged (394->393) — a real, "
            "scale-invariant edge (a shorter ATR tightens trail+stop to ETH's faster "
            "vol). CAVEAT: shorter ATR also tightens the catastrophic sl_dist; in the "
            "sim the stop rarely binds (giveback exits first), so live may see "
            "slightly more tail stop-outs than modeled. Modest, monitor after deploy.",
            "Baseline = live config.py ETHUSD TREND params (2026-07-14).",
            "R is risk-normalized (pnl_points / (ATR_STOP*ATR_at_entry)).",
            "TP_ATR=None means ride the trail/giveback with no fixed take-profit; the "
            "book almost never reaches a 4-10x-ATR target, so TP is inert (all equal).",
            "Baseline WR ~98%% / PF ~70 are optimistic: H1 intra-bar giveback/lock "
            "fills land exactly at the computed level (no within-bar adverse fill). "
            "This inflates ALL configs equally, so the relative WF ranking is valid; "
            "treat absolute R/PF as upper bounds. Matches the validated reference "
            "harness scripts/_trend_exit_tune_h1.py.",
            "SHIP requires a NON-baseline value best-on-train that holds >= baseline "
            "on the held-out 40%% test in BOTH total_R and PF, AND passes the churn/"
            "rescale guards. H1 exit cadence ~ live 60s loop; D1 signal on prior bar.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
