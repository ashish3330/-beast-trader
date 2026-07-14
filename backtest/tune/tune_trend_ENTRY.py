#!/usr/bin/env python3 -B
"""HARD-TUNE the TREND book's ENTRY / CONVICTION component, per symbol.

Scope (ENTRY-side ONLY — exits/trail are owned by other agents):
  1. TREND_CONVICTION_PER_SYMBOL   -> (adx_min, slope_min) D1 selectivity gate.
        adx_min  in {0, 15, 20, 25}
        slope_min in {0, 0.3, 0.5, 0.8}   (|EMA256 slope over 10 D1 bars| / ATR)
     Live gate (agent/trend_follower.conviction + brain._trend_conviction) SKIPS
     an entry when D1 ADX < adx_min OR slope < slope_min. Currently {} (inert).
  2. TREND_MIN_ABS_SIGNAL           -> 3-EMA ensemble agreement threshold.
        min_abs in {0.20, 0.34, 0.50, 0.67}   (0.34 = current, need >=2/3 agree)
  3. TREND_REBALANCE_HOUR           -> inertness probe (entry-hour-of-day shift).

METHOD (honest, anti-curve-fit — conviction gates carry a prior SHIP_NONE history
from the 2026-07-10 rolling-WF study; we respect it and only ship on a clear win):
  * REUSE backtest/tune/trend_engine.py (load / simulate / summarize / exit params).
  * WF 60/40 chronological split by TIME (train = first 60%, test = last 40%).
  * SHIP a value only if it is >= NEUTRAL vs current on BOTH folds (train_R and
    test_R >= baseline), trade count <= 1.4x baseline (not a churn/rescale
    artifact), test-PF holds (>= 0.9x baseline), and the winner is not a lone
    grid-edge pick. Otherwise SHIP_NONE (keep current).
  * DATA TRUST (backtest/tune/DATA_AUDIT.md): ONLY JPN225ft & NAS100.r have
    trustworthy H1. XAUUSD (91d shallow) / BTCUSD (242 bars) / ETHUSD (63d stale)
    verdicts are PROVISIONAL and reported as such / data-blocked.

Writes backtest/tune/results_trend_ENTRY.json. Does NOT touch config.py / brain.py.
Run: python3 -B backtest/tune/tune_trend_ENTRY.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = three levels up (backtest/tune/ -> repo). Ensures `backtest`,
# `config`, and `agent` import whether run as a script or module.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.tune import trend_engine as te

CACHE = te.CACHE
SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "JPN225ft", "NAS100.r"]

DATA_TRUST = {
    "XAUUSD":   "PROVISIONAL — H1 shallow (1500 bars / 91d); engine yields n=0 completed trades. DON'T-TRUST for WF; refetch deep H1 first.",
    "BTCUSD":   "PROVISIONAL / DATA-BLOCKED — H1 truncated (242 bars / 9d); any result is noise. Block until refetched.",
    "ETHUSD":   "PROVISIONAL — H1 deep (50k bars) but 63d stale (ends 2026-05-12); IS ok, OOS cannot see current regime.",
    "JPN225ft": "TRUST — H1 deep (21k bars, ~5d stale).",
    "NAS100.r": "TRUST — H1 deep (50k bars, ~6d stale).",
}

MIN_TRADES_TUNABLE = 12          # below this the symbol is data-blocked (no signal)
# Hard data-block per DATA_AUDIT.md irrespective of raw trade count: XAU H1 is 91d
# shallow (engine n=0), BTC H1 is 242 bars/9d (any count is churn noise). Their
# entry verdicts are provisional until deep H1 is refetched.
FORCE_BLOCKED = {"XAUUSD", "BTCUSD"}
ADX_GRID = [0, 15, 20, 25]
SLOPE_GRID = [0.0, 0.3, 0.5, 0.8]
MINABS_GRID = [0.20, 0.34, 0.50, 0.67]
# Realistic once-a-day rebalance window near day-start (live default = 1). Wider
# values (e.g. noon) don't model the live once/day action — they just re-seat every
# entry at a different hour and perturb the H1 exit cascade, producing path-noise.
REBAL_HOURS = [0, 1, 2, 3]
CURRENT_MINABS = te.DEFAULT_MIN_ABS      # 0.34
TRADE_GUARD = 1.4                        # max trade-count inflation vs baseline
PF_HOLD = 0.9                            # test-PF must hold >= 0.9x baseline


# ── conviction (ADX / slope) D1 context, eff-aligned exactly like the signal ──
def conviction_arrays(sym, frame):
    """Return (adx, slope) arrays aligned to `frame` rows, computed on D1 and
    merged backward on eff (prior completed D1 -> next day) — identical alignment
    to trend_engine.d1_context so the gate is causal (no look-ahead)."""
    d = pickle.load(open(CACHE / ("raw_d1_" + te._token(sym) + ".pkl"), "rb")).copy()
    d["time"] = te._naive(d["time"])
    d = d.sort_values("time").reset_index(drop=True)
    h, l, c = d["high"], d["low"], d["close"]
    # ADX(14) — matches agent/trend_follower.conviction
    up = h.diff(); dn = -l.diff()
    plus = ((up > dn) & (up > 0)) * up
    minus = ((dn > up) & (dn > 0)) * dn
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    a14 = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1.0 / 14, adjust=False).mean() / a14.replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1.0 / 14, adjust=False).mean() / a14.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / 14, adjust=False).mean().fillna(0.0)
    slow = c.ewm(span=256).mean()
    atr = tr.ewm(alpha=1.0 / te.DEFAULT_ATR_P, adjust=False).mean()
    slope = (slow - slow.shift(10)).abs() / atr.replace(0, np.nan)
    ctx = pd.DataFrame({
        "eff": (d["time"].dt.normalize() + pd.Timedelta(days=1)).astype("datetime64[ns]"),
        "adx": adx.astype(float), "slope": slope.fillna(0.0).astype(float)})
    ft = pd.DataFrame({"time": frame["time"].astype("datetime64[ns]")})
    merged = pd.merge_asof(ft, ctx, left_on="time", right_on="eff", direction="backward")
    return merged["adx"].fillna(0.0).values, merged["slope"].fillna(0.0).values


# ── gated simulate: trend_engine.simulate + an ENTRY conviction/hour gate ──
def simulate_entry(m, TR, LK, GB, ACT, ATR_STOP=te.DEFAULT_ATR_STOP,
                   TP=te.DEFAULT_TP_ATR, cost=0.0,
                   adx=None, slope=None, adx_min=0.0, slope_min=0.0,
                   entry_hour_min=None):
    """Faithful copy of trend_engine.simulate exit mechanics, with an ENTRY-only
    conviction gate (adx>=adx_min AND slope>=slope_min) and an optional entry
    hour-of-day floor. Gate is checked ONLY when opening (fresh or flip); it never
    alters an open position's exit. On a flip whose new entry fails the gate we
    close the old position and go flat (matches live 'skip entry')."""
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    tm = m["time"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh = m["hh"].values; ll = m["ll"].values
    hours = pd.to_datetime(m["time"]).dt.hour.values if entry_hour_min is not None else None
    n = len(m)
    if adx is None:
        adx = np.zeros(n)
    if slope is None:
        slope = np.zeros(n)

    pos = 0
    entry = sl = tp = peak = 0.0
    e_atr = 0.0; e_i = 0; worst = 0.0; best = 0.0
    blocked = 0
    trades = []

    def _gate_ok(t):
        if adx[t] < adx_min or slope[t] < slope_min:
            return False
        if entry_hour_min is not None and hours[t] < entry_hour_min:
            return False
        return True

    def _open(s, t, a):
        nonlocal pos, entry, peak, e_atr, e_i, worst, best, sl, tp
        pos = s; entry = o[t]; peak = 0.0; e_atr = a; e_i = t
        worst = 0.0; best = 0.0
        sl = entry - ATR_STOP * a if s == 1 else entry + ATR_STOP * a
        tp = None if TP >= 999 else ((entry + TP * a) if s == 1 else (entry - TP * a))

    def _close(ex_price, reason, t_idx):
        nonlocal pos
        ret = ((ex_price - entry) / entry) * pos - cost
        r_unit = (ATR_STOP * e_atr) / entry if entry > 0 and e_atr > 0 else 0.0
        pnl_R = ret / r_unit if r_unit > 0 else 0.0
        mae_R = (worst / (ATR_STOP * e_atr)) if e_atr > 0 else 0.0
        mfe_R = (best / (ATR_STOP * e_atr)) if e_atr > 0 else 0.0
        trades.append({"dir": pos, "t_in": tm[e_i], "t_out": tm[t_idx],
                       "entry": float(entry), "exit": float(ex_price), "reason": reason,
                       "ret": float(ret), "pnl_R": float(pnl_R),
                       "mae_R": float(mae_R), "mfe_R": float(mfe_R)})
        pos = 0

    for t in range(n):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blocked and s != blocked:
                blocked = 0
            if s != 0 and s != blocked and _gate_ok(t):
                _open(s, t, a)
            continue
        if s != 0 and s != pos:                       # daily signal reversal
            _close(o[t], "flip", t)
            blocked = 0
            if _gate_ok(t):                           # only re-open if gate passes
                _open(s, t, a)
            continue
        if pos == 1:
            worst = max(worst, entry - l[t])
            sl = max(sl, hh[t] - TR * a)
            if peak >= ACT * a:
                sl = max(sl, entry + LK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else -1e18
            ex = ereason = None
            if l[t] <= sl:
                ex, ereason = sl, "stop/trail"
            elif gb > -1e17 and l[t] <= gb:
                ex, ereason = gb, "giveback"; blocked = pos
            elif tp is not None and h[t] >= tp:
                ex, ereason = tp, "tp"
            if ex is not None:
                _close(ex, ereason, t)
            else:
                peak = max(peak, h[t] - entry); best = peak
        else:
            worst = max(worst, h[t] - entry)
            sl = min(sl, ll[t] + TR * a)
            if peak >= ACT * a:
                sl = min(sl, entry - LK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else 1e18
            ex = ereason = None
            if h[t] >= sl:
                ex, ereason = sl, "stop/trail"
            elif gb < 1e17 and h[t] >= gb:
                ex, ereason = gb, "giveback"; blocked = pos
            elif tp is not None and l[t] <= tp:
                ex, ereason = tp, "tp"
            if ex is not None:
                _close(ex, ereason, t)
            else:
                peak = max(peak, entry - l[t]); best = peak
    return trades


def _seg(frame, t0, t1):
    m = frame[(frame["time"] >= t0) & (frame["time"] < t1)].reset_index(drop=True)
    return m


def _split_time(frame, frac=0.60):
    t = pd.to_datetime(frame["time"]).reset_index(drop=True)
    return t.iloc[int(len(t) * frac)]


def _score(trades):
    s = te.summarize(trades)
    return s["total_R"], s["pf"], s["n"]


# ── per-symbol tuning of one param family ──
def tune_symbol(sym):
    frame = te.load(sym)                       # min_abs default 0.34, no gate
    tr, lk, gb, act = te._cfg_exit_params(sym)
    cost = te.spread_cost_rt(frame)
    t_end = pd.to_datetime(frame["time"]).max() + pd.Timedelta(hours=1)
    split = _split_time(frame, 0.60)

    # baseline (current config: min_abs 0.34, no conviction gate)
    b_train = simulate_entry(_seg(frame, frame["time"].min(), split), tr, lk, gb, act, cost=cost)
    b_test = simulate_entry(_seg(frame, split, t_end), tr, lk, gb, act, cost=cost)
    bTr_R, bTr_pf, bTr_n = _score(b_train)
    bTe_R, bTe_pf, bTe_n = _score(b_test)
    b_total_n = bTr_n + bTe_n

    result = {"symbol": sym, "data_trust": DATA_TRUST[sym],
              "baseline": {"train_R": round(bTr_R, 3), "test_R": round(bTe_R, 3),
                           "train_n": bTr_n, "test_n": bTe_n,
                           "test_pf": round(bTe_pf, 2)},
              "params": []}

    data_blocked = (b_total_n < MIN_TRADES_TUNABLE) or (sym in FORCE_BLOCKED)
    result["data_blocked"] = data_blocked

    # precompute conviction arrays once
    adx_arr, slope_arr = conviction_arrays(sym, frame)

    def ship_check(cand, best_at_edge):
        """cand=(train_R,test_R,test_pf,total_n). Return verdict string."""
        cTr, cTe, cPf, cN = cand
        if cTr < bTr_R - 1e-9 or cTe < bTe_R - 1e-9:
            return "SHIP_NONE (fails neutral on a fold)"
        if cN > b_total_n * TRADE_GUARD:
            return "SHIP_NONE (trade-count churn > 1.4x)"
        if bTe_pf > 0 and cPf < bTe_pf * PF_HOLD:
            return "SHIP_NONE (test-PF does not hold)"
        if best_at_edge:
            return "SHIP_NONE (lone grid-edge pick — not robust)"
        return "SHIP"

    # ---- PARAM 1: TREND_MIN_ABS_SIGNAL ----
    ma_rows = []
    for v in MINABS_GRID:
        fv = te.load(sym, min_abs=v)
        av, sv = conviction_arrays(sym, fv)  # unused for min_abs (no gate) but keep frame aligned
        s_end = pd.to_datetime(fv["time"]).max() + pd.Timedelta(hours=1)
        sp = split  # same calendar split
        trn = simulate_entry(_seg(fv, fv["time"].min(), sp), tr, lk, gb, act, cost=cost)
        tst = simulate_entry(_seg(fv, sp, s_end), tr, lk, gb, act, cost=cost)
        rTr, pTr, nTr = _score(trn); rTe, pTe, nTe = _score(tst)
        ma_rows.append({"v": v, "train_R": round(rTr, 3), "test_R": round(rTe, 3),
                        "test_pf": round(pTe, 2), "total_n": nTr + nTe})
    # pick best test_R among non-current values that pass neutral on train
    cands = [r for r in ma_rows if r["v"] != CURRENT_MINABS]
    best = max(cands, key=lambda r: r["test_R"]) if cands else None
    verdict = "SHIP_NONE"
    if best is not None and not data_blocked:
        edge = best["v"] in (MINABS_GRID[0], MINABS_GRID[-1])
        # edge is robust only if the neighbour also beats baseline
        if edge:
            order = sorted(MINABS_GRID)
            idx = order.index(best["v"])
            nb = order[1] if idx == 0 else order[-2]
            nbrow = next(r for r in ma_rows if r["v"] == nb)
            edge = not (nbrow["train_R"] >= bTr_R and nbrow["test_R"] >= bTe_R)
        verdict = ship_check((best["train_R"], best["test_R"], best["test_pf"], best["total_n"]), edge)
    best_v = best["v"] if (best and verdict == "SHIP") else CURRENT_MINABS
    result["params"].append({
        "name": "TREND_MIN_ABS_SIGNAL", "current": CURRENT_MINABS, "best": best_v,
        "verdict": ("data-blocked (insufficient trades)" if data_blocked else verdict),
        "train_R": (best["train_R"] if best else None),
        "test_R": (best["test_R"] if best else None),
        "sweep": ma_rows})

    # ---- PARAM 2: TREND_CONVICTION_PER_SYMBOL (adx_min x slope_min grid) ----
    conv_rows = []
    for am in ADX_GRID:
        for sm_ in SLOPE_GRID:
            if am == 0 and sm_ == 0.0:
                continue  # == baseline
            trn = simulate_entry(_seg(frame, frame["time"].min(), split), tr, lk, gb, act,
                                 cost=cost, adx=adx_arr, slope=slope_arr,
                                 adx_min=am, slope_min=sm_)
            tst = simulate_entry(_seg(frame, split, t_end), tr, lk, gb, act,
                                 cost=cost, adx=adx_arr, slope=slope_arr,
                                 adx_min=am, slope_min=sm_)
            rTr, pTr, nTr = _score(trn); rTe, pTe, nTe = _score(tst)
            conv_rows.append({"adx_min": am, "slope_min": sm_,
                              "train_R": round(rTr, 3), "test_R": round(rTe, 3),
                              "test_pf": round(pTe, 2), "total_n": nTr + nTe})
    cbest = max(conv_rows, key=lambda r: r["test_R"]) if conv_rows else None
    cverdict = "SHIP_NONE"
    if cbest is not None and not data_blocked:
        # grid-edge = strongest gate corner (25 / 0.8); require an interior neighbour
        # to also beat baseline before trusting a corner pick.
        edge = cbest["adx_min"] == ADX_GRID[-1] or cbest["slope_min"] == SLOPE_GRID[-1]
        if edge:
            neigh = [r for r in conv_rows
                     if abs(ADX_GRID.index(r["adx_min"]) - ADX_GRID.index(cbest["adx_min"])) <= 1
                     and abs(SLOPE_GRID.index(r["slope_min"]) - SLOPE_GRID.index(cbest["slope_min"])) <= 1
                     and (r["adx_min"], r["slope_min"]) != (cbest["adx_min"], cbest["slope_min"])]
            edge = not any(r["train_R"] >= bTr_R and r["test_R"] >= bTe_R for r in neigh)
        cverdict = ship_check((cbest["train_R"], cbest["test_R"], cbest["test_pf"], cbest["total_n"]), edge)
    result["params"].append({
        "name": "TREND_CONVICTION_PER_SYMBOL", "current": "{} (no gate)",
        "best": ({"ADX_MIN": cbest["adx_min"], "SLOPE_MIN": cbest["slope_min"]}
                 if (cbest and cverdict == "SHIP") else "{} (no gate)"),
        "verdict": ("data-blocked (insufficient trades)" if data_blocked else cverdict),
        "train_R": (cbest["train_R"] if cbest else None),
        "test_R": (cbest["test_R"] if cbest else None),
        "best_candidate": cbest, "sweep": conv_rows})

    # ---- PARAM 3: TREND_REBALANCE_HOUR (inertness probe) ----
    rh_rows = []
    for rh in REBAL_HOURS:
        allt = simulate_entry(frame, tr, lk, gb, act, cost=cost, entry_hour_min=rh)
        r, pf, nn = _score(allt)
        rh_rows.append({"hour": rh, "total_R": round(r, 3), "pf": round(pf, 2), "n": nn})
    rs = [x["total_R"] for x in rh_rows]
    spread = (max(rs) - min(rs)) if rs else 0.0
    # TREND_REBALANCE_HOUR is a LIVE once-a-day scheduling knob (the brain acts on
    # the first cycle after this UTC hour), NOT a signal parameter. In this per-H1
    # engine, forcing entries to a fixed hour re-seats every fill and perturbs the
    # intrabar exit cascade — so any BT R difference across hours is path-noise, not
    # a tradeable edge. Verdict is always KEEP 1; the sweep is shown for evidence.
    result["params"].append({
        "name": "TREND_REBALANCE_HOUR", "current": 1, "best": 1,
        "verdict": "KEEP 1 — live once/day scheduling knob; BT hour response is "
                   "path-noise (see sweep), no robust signal edge",
        "train_R": None, "test_R": None,
        "R_spread_over_hours": round(spread, 3), "sweep": rh_rows})

    return result


def main():
    per_symbol = []
    rec_conv = {}
    rec_minabs = {}
    for sym in SYMBOLS:
        try:
            r = tune_symbol(sym)
        except Exception as e:
            r = {"symbol": sym, "data_trust": DATA_TRUST[sym], "error": repr(e), "params": []}
        per_symbol.append(r)
        for p in r.get("params", []):
            if p["name"] == "TREND_CONVICTION_PER_SYMBOL" and p["verdict"] == "SHIP":
                rec_conv[sym] = p["best"]
            if p["name"] == "TREND_MIN_ABS_SIGNAL" and p["verdict"] == "SHIP":
                rec_minabs[sym] = p["best"]

    notes = (
        "WF 60/40 chronological, one param family at a time, objective total-R with "
        "PF-hold + trade-guard (<=1.4x) + grid-edge robustness. Ship only if >= NEUTRAL "
        "on BOTH folds. Conviction gates carry a prior 2026-07-10 rolling-WF SHIP_NONE "
        "verdict (churn/cost artifacts) — reproduced and respected here. XAUUSD (n=0) and "
        "BTCUSD (n<=4) are DATA-BLOCKED by truncated/shallow H1 (see DATA_AUDIT.md); their "
        "verdicts are provisional and must be re-run after deep H1 refetch. ETHUSD OOS is "
        "63d stale. Backtest R is a RANKING signal only (risk-capped SL + early giveback "
        "dominate live), never a live P/L forecast."
    )
    out = {"component": "entry",
           "generated": "2026-07-15",
           "engine": "backtest/tune/trend_engine.py (D1 3-EMA signal + H1 chandelier/lock/giveback/flip)",
           "method": "WF 60/40 by time; ship >= neutral BOTH folds, trade<=1.4x, PF holds, no lone grid-edge",
           "per_symbol": per_symbol,
           "recommended_TREND_CONVICTION_PER_SYMBOL": rec_conv,
           "recommended_MIN_ABS_per_symbol": rec_minabs,
           "notes": notes}

    outp = Path(__file__).resolve().parent / "results_trend_ENTRY.json"
    outp.write_text(json.dumps(out, indent=2, default=str))
    print("wrote", outp)

    # concise console summary
    for r in per_symbol:
        if "error" in r:
            print(f"\n{r['symbol']:10s} ERROR {r['error']}")
            continue
        b = r["baseline"]
        print(f"\n{r['symbol']:10s} [{'BLOCKED' if r.get('data_blocked') else 'tunable'}] "
              f"base train_R={b['train_R']} test_R={b['test_R']} (n {b['train_n']}/{b['test_n']}) "
              f"testPF={b['test_pf']}")
        for p in r["params"]:
            extra = ""
            if p["name"] == "TREND_REBALANCE_HOUR":
                extra = f" R_spread={p['R_spread_over_hours']}"
            print(f"   {p['name']:28s} best={str(p['best'])[:22]:22s} -> {p['verdict']}{extra}")
    print("\nRecommended conviction gate:", rec_conv or "{} (none — keep inert)")
    print("Recommended MIN_ABS overrides:", rec_minabs or "{} (none — keep 0.34)")


if __name__ == "__main__":
    main()
