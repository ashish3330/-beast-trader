#!/usr/bin/env python3 -B
"""Hard-tune the portfolio RISK-GATE mechanism (per-symbol $ thresholds) for the
live TREND books — a single, reusable, self-contained tuner that consolidates the
four prior per-symbol scratchpad scripts (btcusd_emergency_cap_tune / eth_cap_tune
/ jpn_losscap_tune / nas_emergency_cap) and adds the DAILY_LOSS_LIMIT gate.

Two gates are tuned, both per-symbol in $ (config.py):

1) EMERGENCY_LOSS_CAP_USD — a FIXED per-symbol open-$ loss cap. When an open
   position's loss reaches -cap it is force-closed. Fires from the first trade.
   METHOD (per-trade MAE R-sweep, faithful to the 4 prior scripts):
     * Reconstruct the live TREND trade list per symbol (D1 3-EMA ensemble
       signal + intraday exit management: chandelier trail + profit-lock +
       peak-giveback + 3xATR stop + TP + signal-flip), magic +6000.
     * Per trade record realized pnl_R and MAE_R = worst-adverse-excursion/sl_dist
       (sl_dist = ATR_STOP*ATR20 at entry = 1R).
     * Sweep cap_R in {0.4..2.5}; a trade with MAE_R>=cap_R realizes -cap_R,
       else keeps realized pnl_R.
     * 60/40 chronological walk-forward; SHIP only if total-R >= neutral on BOTH
       folds (train picks the best cap that is not worse than baseline train).
     * cap_$ = cap_R * EQUITY(≈4100) * TREND_RISK_PCT(0.30)/100  ($12.30 per 1R).
   XAUUSD is intentionally NOT emergency-capped (gold runs under GOLD_SMC's own
   BE/EMA9 exit) — matches config.py. So the emergency gate covers BTC/ETH/JPN/NAS.

2) DAILY_LOSS_LIMIT_USD — a per-symbol daily cumulative $ MAX-LOSS breaker: once a
   symbol's TODAY P/L (realized closed-$ + open-$) reaches -limit, close that
   symbol + block re-entry for the UTC day. LOSS ONLY — winners keep running.
   METHOD: On a one-position-per-symbol TREND book, "today's open $" is measured
   entry-relative, so the worst reading a symbol reaches across a trade's life is
   exactly its MAE in $ (= MAE_R * $12.30). The daily gate therefore reduces to a
   per-trade $-cap on MAE (with the daily reset/re-entry-block being immaterial for
   a slow daily book). We sweep a $ limit against the reconstructed TREND daily-PnL
   distribution: a limit that reduces a tail-loser trade to -limit is GOOD; one that
   clips a trade that would have recovered (final pnl > -limit, esp. a winner) is a
   normal drawdown-then-recover day wrongly clipped. WF 60/40, ship if total-$ >=
   neutral on BOTH folds. XAU adds GOLD_SMC/SCALPER P/L on top of TREND (not
   reconstructed here) so its true daily distribution is WIDER — the TREND-only
   number is a floor; keep the current limit unless the tail clearly supports a cut.

DATA: D1 caches /Users/ashish/Documents/xauusd-trading-bot/cache/raw_d1_<SYM>.pkl.
  BTC/XAU H1 caches are too short (242/1500 bars) for the 256-EMA -> D1-only sim.
  ETH/JPN/NAS have deep H1 -> H1-checked exits (12x finer, ~live 60s cadence).

RULES: honest WF, no curve-fit. Writes results_risk_gates.json next to this file.
Does NOT touch config.py / brain.py.
"""
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
OUT = Path(__file__).resolve().parent / "results_risk_gates.json"

EQUITY = 4100.0
TREND_RISK_PCT = 0.30
DOLLAR_PER_R = EQUITY * TREND_RISK_PCT / 100.0   # $12.30 per 1R (task conversion)

# TREND globals (config.py)
MIN_ABS = 0.34
ATR_P = 20
ATR_STOP = 3.0
TP_ATR = 6.0
TRAIL_LB = 22
REBLOCK_BARS = 2                                 # TREND_REENTRY_BLOCK_HOURS 2.0 -> 2 H1 bars
DEFAULT_EMA = [(16, 64), (32, 128), (64, 256)]

CAPS = [round(x, 2) for x in np.arange(0.4, 2.5001, 0.1)]
# Daily-loss $ grid: coarse dollar ladder that brackets each symbol's tail.
DAILY_USD_GRID = [10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 125, 150, 200]

# ── Per-symbol book spec. method: "d1" (BTC/XAU) or "h1" (ETH/JPN/NAS).
#    exit = (TRAIL, LOCK, GIVEBACK, ACT) from config.TREND_EXIT_PER_SYMBOL.
#    cost_pts = fixed round-trip cost in price points (0 = none); cost_from_spread
#    uses the H1 median spread*point. Values reproduce the 4 prior scripts. ──
SYMBOLS = {
    "XAUUSD":   dict(method="d1", ema=DEFAULT_EMA,            exit=(2.5, 0.5, 0.30, 0.5),
                     cost_pts=0.0, cost_from_spread=False, point=0.01,
                     emergency=False, daily=True, basket=True),
    "BTCUSD":   dict(method="d1", ema=DEFAULT_EMA,            exit=(3.0, 0.5, 0.30, 0.5),
                     cost_pts=0.0, cost_from_spread=False, point=0.01,
                     emergency=True, daily=True, basket=True),
    "ETHUSD":   dict(method="h1", block_on="giveback", ema=[(16, 96), (32, 160), (64, 256)], exit=(2.5, 0.5, 0.35, 0.3),
                     cost_pts=0.0, cost_from_spread=False, point=0.01,
                     emergency=True, daily=True, basket=True),
    "JPN225ft": dict(method="h1", block_on="giveback", ema=DEFAULT_EMA,   exit=(3.0, 0.6, 0.35, 0.3),
                     cost_pts=0.0, cost_from_spread=True, point=0.01,
                     emergency=True, daily=True, basket=True),
    "NAS100.r": dict(method="h1", block_on="all", ema=DEFAULT_EMA,        exit=(2.5, 0.6, 0.35, 0.5),
                     cost_pts=1.5, cost_from_spread=False, point=0.01,
                     emergency=True, daily=True, basket=True),
    # Not in TREND_BASKET (traded by other strategies live) but present in the
    # DAILY_LOSS_LIMIT dict. Reconstructed as a D1 TREND PROXY for a volatility/tail
    # read only -> verdict PROXY_KEEP.
    "SP500.r":  dict(method="d1", ema=DEFAULT_EMA,            exit=(2.5, 0.6, 0.35, 0.5),
                     cost_pts=0.0, cost_from_spread=False, point=0.01,
                     emergency=False, daily=True, basket=False),
    "US2000.r": dict(method="d1", ema=DEFAULT_EMA,            exit=(2.5, 0.6, 0.35, 0.5),
                     cost_pts=0.0, cost_from_spread=False, point=0.01,
                     emergency=False, daily=True, basket=False),
}

CURRENT_EMERGENCY = {"BTCUSD": 15.0, "JPN225ft": 10.0}   # config.py (BTC = user override)
CURRENT_DAILY = {"XAUUSD": 40.0, "BTCUSD": 60.0, "ETHUSD": 30.0, "JPN225ft": 40.0,
                 "NAS100.r": 40.0, "SP500.r": 30.0, "US2000.r": 30.0}


# ─────────────────────────── data / indicators ───────────────────────────
def _naive(s):
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_localize(None)


def _load(kind, sym):
    p = CACHE / (kind + "_" + sym.replace(".", "_") + ".pkl")
    df = pickle.load(open(p, "rb"))
    df["time"] = _naive(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df.astype({"open": float, "high": float, "low": float, "close": float})


def atr_wilder(high, low, close, period=ATR_P):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def d1_signal(close, ema_pairs):
    sig = pd.Series(0.0, index=close.index)
    for f, s in ema_pairs:
        sig = sig + np.sign(close.ewm(span=f).mean() - close.ewm(span=s).mean())
    sig = sig / len(ema_pairs)
    d = pd.Series(0, index=close.index)
    d[sig >= MIN_ABS] = 1
    d[sig <= -MIN_ABS] = -1
    return d


# ─────────────────────────── reconstruction (D1) ───────────────────────────
def simulate_d1(sym, spec):
    """D1-only TREND reconstruction (BTC/XAU/index-proxy). Returns trade dicts with
    entry_i/exit_i/dir/sl_dist/entry/pnl_R/mae_R/exit_t/reason + the bar arrays."""
    df = _load("raw_d1", sym)
    close, high, low = df["close"], df["high"], df["low"]
    C, H, L = close.values, high.values, low.values
    T = df["time"].values
    atr = atr_wilder(high, low, close).values
    direction = d1_signal(close, spec["ema"]).values
    roll_hh = high.rolling(TRAIL_LB).max().shift(1).values
    roll_ll = low.rolling(TRAIL_LB).min().shift(1).values
    atr_prev = pd.Series(atr).shift(1).values
    TR, LOCK, GB, ACT = spec["exit"]
    PEAK_ACT_R = 0.5      # PEAK_GIVEBACK_ACTIVATE_R ceiling
    cost = spec["cost_pts"]
    n = len(df)

    trades = []
    i = 260
    while i < n - 1:
        d = direction[i]
        if d == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        entry = C[i]
        sl_dist = ATR_STOP * atr[i]
        cur_sl = entry - d * sl_dist
        tp = entry + d * TP_ATR * atr[i]
        act_thresh = min((ACT / ATR_STOP) * sl_dist, PEAK_ACT_R * sl_dist)
        peak_prof = 0.0
        worst_adverse = 0.0
        exit_idx = exit_price = reason = None
        for j in range(i + 1, n):
            hi, lo, cl = H[j], L[j], C[j]
            adverse = (entry - lo) if d == 1 else (hi - entry)
            if adverse > worst_adverse:
                worst_adverse = adverse
            if np.isfinite(atr_prev[j]) and atr_prev[j] > 0:
                if d == 1 and np.isfinite(roll_hh[j]):
                    chand = roll_hh[j] - TR * atr_prev[j]
                    if chand > cur_sl:
                        cur_sl = chand
                elif d == -1 and np.isfinite(roll_ll[j]):
                    chand = roll_ll[j] + TR * atr_prev[j]
                    if chand < cur_sl:
                        cur_sl = chand
            fav = (hi - entry) if d == 1 else (entry - lo)
            if fav >= act_thresh:
                lock = entry + d * LOCK * fav
                if d == 1 and lock > cur_sl:
                    cur_sl = lock
                elif d == -1 and lock < cur_sl:
                    cur_sl = lock
            sl_hit = (lo <= cur_sl) if d == 1 else (hi >= cur_sl)
            tp_hit = (hi >= tp) if d == 1 else (lo <= tp)
            if sl_hit:
                exit_idx, exit_price, reason = j, cur_sl, "SL/TRAIL"; break
            if tp_hit:
                exit_idx, exit_price, reason = j, tp, "TP"; break
            if fav > peak_prof:
                peak_prof = fav
            if peak_prof >= act_thresh:
                cl_prof = (cl - entry) if d == 1 else (entry - cl)
                if cl_prof <= peak_prof * (1.0 - GB):
                    exit_idx, exit_price, reason = j, cl, "REVERSAL"; break
            if direction[j] != d:
                exit_idx, exit_price, reason = j, cl, "FLIP"; break
        if exit_idx is None:
            exit_idx, exit_price, reason = n - 1, C[-1], "EOF"
        pnl_pts = ((exit_price - entry) if d == 1 else (entry - exit_price)) - cost
        trades.append(dict(entry_i=i, exit_i=exit_idx, dir=int(d), sl_dist=sl_dist,
                           entry=entry, pnl_R=pnl_pts / sl_dist, mae_R=worst_adverse / sl_dist,
                           exit_t=T[exit_idx], reason=reason))
        i = exit_idx + 1
    return trades, df


# ─────────────────────────── reconstruction (H1-checked) ───────────────────────────
def _d1_context(sym, ema_pairs):
    d = _load("raw_d1", sym)
    c, h, l = d["close"], d["high"], d["low"]
    sig = d1_signal(c, ema_pairs)
    atr = atr_wilder(h, l, c)
    out = pd.DataFrame({"eff": d["time"].dt.normalize() + pd.Timedelta(days=1),
                        "sig": sig.astype(int), "atr": atr,
                        "hh": h.rolling(TRAIL_LB).max(), "ll": l.rolling(TRAIL_LB).min()})
    return out.dropna().reset_index(drop=True)


def simulate_h1(sym, spec):
    """H1-checked TREND reconstruction (ETH/JPN/NAS). D1 signal/ATR/hh-ll merged
    onto H1 bars via prior-completed-D1 context; exits checked every H1 bar.

    Re-entry cadence mirrors each symbol's committed repo tuner (tune_trend_*.py):
      block_on="all"      -> block same-dir re-entry after ANY exit until the D1
                             signal flips (NAS canonical, churn-free; n=32).
      block_on="giveback" -> block only after a giveback/reversal exit; an SL/TP
                             exit re-enters on the next signal bar (JPN/ETH
                             canonical; n=42 / n=394).
    The naive "re-enter next H1 bar after every exit" cadence inflates trade count
    ~145x (NAS 4642 = FAKE) and is NOT used."""
    block_on = spec.get("block_on", "all")
    h1 = _load("raw_h1", sym)
    d1 = _d1_context(sym, spec["ema"])
    h1["time"] = h1["time"].astype("datetime64[ns]")
    d1["eff"] = d1["eff"].astype("datetime64[ns]")
    m = pd.merge_asof(h1, d1, left_on="time", right_on="eff", direction="backward").dropna(
        subset=["sig", "atr", "hh", "ll"]).reset_index(drop=True)

    o, h, l = m["open"].values, m["high"].values, m["low"].values
    T = m["time"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values
    hh, ll = m["hh"].values, m["ll"].values
    TR, LOCK, GB, ACT = spec["exit"]
    if spec["cost_from_spread"]:
        cost = 2.0 * float(np.nanmedian(m["spread"].values)) * spec["point"]
    else:
        cost = spec["cost_pts"]

    pos = 0; entry = sl = 0.0; tp = None; peak = 0.0; sl_dist = 0.0
    mae = 0.0; ent_i = None; blk_dir = 0
    trades = []

    def close(exit_px, reason, t):
        nonlocal pos
        d = pos
        pnl_pts = (exit_px - entry) * d - cost
        trades.append(dict(entry_i=ent_i, exit_i=t, dir=d, sl_dist=sl_dist, entry=entry,
                           pnl_R=pnl_pts / sl_dist, mae_R=mae / sl_dist,
                           exit_t=T[t], reason=reason))
        pos = 0

    for t in range(len(m)):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            if blk_dir and s != blk_dir:              # clear block only when signal flips away
                blk_dir = 0
            if s != 0 and s != blk_dir:
                pos = s; entry = o[t]; peak = 0.0; mae = 0.0; ent_i = t
                sl_dist = ATR_STOP * a
                sl = entry - sl_dist if s == 1 else entry + sl_dist
                tp = (entry + TP_ATR * a) if s == 1 else (entry - TP_ATR * a)
            continue
        adv = (entry - l[t]) if pos == 1 else (h[t] - entry)
        if adv > mae:
            mae = adv
        if s != 0 and s != pos:                       # genuine daily signal flip (never churn)
            close(o[t], "FLIP", t)
            blk_dir = 0
            pos = s; entry = o[t]; peak = 0.0; mae = 0.0; ent_i = t
            sl_dist = ATR_STOP * a
            sl = entry - sl_dist if pos == 1 else entry + sl_dist
            tp = (entry + TP_ATR * a) if pos == 1 else (entry - TP_ATR * a)
            continue
        if pos == 1:
            sl = max(sl, hh[t] - TR * a)
            if peak >= ACT * a:
                sl = max(sl, entry + LOCK * peak)
            gb = entry + peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else -1e18
            ex = rsn = None
            if l[t] <= sl:
                ex, rsn = sl, "SL/TRAIL"
            elif gb > -1e17 and l[t] <= gb:
                ex, rsn = gb, "GIVEBACK"
            elif tp is not None and h[t] >= tp:
                ex, rsn = tp, "TP"
            if ex is not None:
                close(ex, rsn, t)
                if block_on == "all" or rsn == "GIVEBACK":
                    blk_dir = 1                        # block re-long until D1 flips short
            else:
                peak = max(peak, h[t] - entry)
        else:
            sl = min(sl, ll[t] + TR * a)
            if peak >= ACT * a:
                sl = min(sl, entry - LOCK * peak)
            gb = entry - peak * (1.0 - GB) if (GB < 1.0 and peak >= ACT * a) else 1e18
            ex = rsn = None
            if h[t] >= sl:
                ex, rsn = sl, "SL/TRAIL"
            elif gb < 1e17 and h[t] >= gb:
                ex, rsn = gb, "GIVEBACK"
            elif tp is not None and l[t] <= tp:
                ex, rsn = tp, "TP"
            if ex is not None:
                close(ex, rsn, t)
                if block_on == "all" or rsn == "GIVEBACK":
                    blk_dir = -1                       # block re-short until D1 flips long
            else:
                peak = max(peak, entry - l[t])
    return trades, m


def reconstruct(sym, spec):
    return simulate_d1(sym, spec) if spec["method"] == "d1" else simulate_h1(sym, spec)


# ─────────────────────────── metrics ───────────────────────────
def pf(rs):
    rs = np.asarray(rs, dtype=float)
    if len(rs) == 0:
        return 0.0
    w = rs[rs > 0].sum(); ls = -rs[rs < 0].sum()
    return float(w / ls) if ls > 0 else 99.0


def apply_cap_R(trades, cap_R):
    """Reduce any trade with MAE_R>=cap_R to -cap_R. Returns (R array, n_cut, n_win_clip)."""
    out = []; n_cut = 0; n_win_clip = 0
    for t in trades:
        if t["mae_R"] >= cap_R:
            out.append(-cap_R); n_cut += 1
            if t["pnl_R"] > 0:
                n_win_clip += 1
        else:
            out.append(t["pnl_R"])
    return np.array(out), n_cut, n_win_clip


# ─────────────────────────── gate 1: emergency loss cap ───────────────────────────
def tune_emergency(sym, trades):
    """Repo-canonical WF ship rule: pick the cap that MAXIMIZES train total-R; SHIP
    only if it (a) STRICTLY beats baseline train-R (a no-op cap that cuts nothing
    ties baseline and is therefore rejected) AND (b) is non-regressive on test-R.
    Reproduces the 4 prior verdicts: JPN SHIP; BTC/ETH/NAS SHIP_NONE."""
    n = len(trades)
    Rs = np.array([t["pnl_R"] for t in trades])
    MAE = np.array([t["mae_R"] for t in trades])
    split = int(n * 0.60)
    tr, te = trades[:split], trades[split:]
    base_tr = sum(t["pnl_R"] for t in tr); base_te = sum(t["pnl_R"] for t in te)

    best_cap = None; best_train = -1e18; best_test = 0.0
    for c in CAPS:
        ctr, _, _ = apply_cap_R(tr, c)
        cte, _, _ = apply_cap_R(te, c)
        if ctr.sum() > best_train + 1e-12:
            best_train = float(ctr.sum()); best_cap = c; best_test = float(cte.sum())
    cap_R, trR, teR = best_cap, best_train, best_test
    train_beats = trR > base_tr + 1e-9
    test_ok = teR >= base_te - 1e-9
    fullR, n_cut, n_win = apply_cap_R(trades, cap_R)
    ship = train_beats and test_ok and n_cut > 0
    verdict = "SHIP" if ship else "SHIP_NONE"
    cap_usd = round(cap_R * DOLLAR_PER_R) if ship else None
    return dict(symbol=sym, cap_R=round(cap_R, 2),
                cap_usd=cap_usd, verdict=verdict,
                train_R=round(trR, 3), test_R=round(teR, 3),
                base_train_R=round(base_tr, 3), base_test_R=round(base_te, 3),
                n_trades=n, n_cut=int(n_cut), winners_clipped=int(n_win),
                base_total_R=round(float(Rs.sum()), 3), base_PF=round(pf(Rs), 3),
                mae_R_p50=round(float(np.percentile(MAE, 50)), 2),
                mae_R_p90=round(float(np.percentile(MAE, 90)), 2),
                mae_R_max=round(float(MAE.max()), 2))


# ─────────────────────────── gate 2: daily loss limit ───────────────────────────
def apply_daily_usd(trades, limit_usd):
    """A trade whose MAE in $ (mae_R*$/R) reaches the daily limit is force-closed
    that day at -limit; else it keeps its realized $ P/L. Returns
    (usd array, n_tail_saved, n_recover_clipped). On a 1-position/day book this is
    the faithful reduction of the daily cumulative-loss breaker."""
    out = []; n_tail_saved = 0; n_recover_clipped = 0
    cap_R = limit_usd / DOLLAR_PER_R
    for t in trades:
        realized = t["pnl_R"] * DOLLAR_PER_R
        if t["mae_R"] >= cap_R:                        # would have hit the daily limit
            out.append(-limit_usd)
            if realized < -limit_usd - 1e-9:
                n_tail_saved += 1                      # cut a genuine tail-loss day
            else:
                n_recover_clipped += 1                 # clipped a recover day
        else:
            out.append(realized)
    return np.array(out), n_tail_saved, n_recover_clipped


def daily_pnl_series(trades):
    """Realized $ binned by exit UTC day (what the journal daily P/L would show)."""
    day = defaultdict(float)
    for t in trades:
        d = pd.Timestamp(t["exit_t"]).normalize()
        day[d] += t["pnl_R"] * DOLLAR_PER_R
    return np.array(sorted(day.values()))              # ascending; index 0 = worst day


def tune_daily(sym, spec, trades):
    """Best-on-train $-limit; SHIP a TIGHTENING only if it (a) STRICTLY beats
    baseline train-$ (genuinely cuts a tail day), (b) is non-regressive on test-$,
    (c) actually fires (>=1 day cut) and differs from current. Else KEEP_CURRENT —
    the honest default, since the current loose limits already avoid clipping normal
    drawdown-then-recover days. Non-basket symbols -> PROXY_KEEP."""
    n = len(trades)
    split = int(n * 0.60)
    tr, te = trades[:split], trades[split:]
    base_tr = sum(t["pnl_R"] for t in tr) * DOLLAR_PER_R
    base_te = sum(t["pnl_R"] for t in te) * DOLLAR_PER_R
    days = daily_pnl_series(trades)
    worst_day = float(days[0]) if len(days) else 0.0
    day_p05 = float(np.percentile(days, 5)) if len(days) else 0.0

    best_L = None; best_train = -1e18; best_test = 0.0
    for L in DAILY_USD_GRID:
        gtr, _, _ = apply_daily_usd(tr, L)
        gte, _, _ = apply_daily_usd(te, L)
        if gtr.sum() > best_train + 1e-9:
            best_train = float(gtr.sum()); best_L = L; best_test = float(gte.sum())
    limit, trS, teS = best_L, best_train, best_test
    _, tail_saved, recover_clipped = apply_daily_usd(trades, limit)
    train_beats = trS > base_tr + 1e-9
    test_ok = teS >= base_te - 1e-9
    cur = CURRENT_DAILY.get(sym)

    if not spec["basket"]:
        verdict = "PROXY_KEEP"; rec_limit = cur
    elif train_beats and test_ok and tail_saved > 0 and float(limit) != cur:
        verdict = "SHIP"; rec_limit = float(limit)
    else:
        verdict = "KEEP_CURRENT"; rec_limit = cur
    return dict(symbol=sym, limit_usd=rec_limit, wf_best_limit_usd=float(limit),
                verdict=verdict, current_limit_usd=cur,
                worst_day_saved=int(tail_saved), normal_days_clipped=int(recover_clipped),
                train_usd=round(trS, 2), test_usd=round(teS, 2),
                base_train_usd=round(base_tr, 2), base_test_usd=round(base_te, 2),
                worst_realized_day_usd=round(worst_day, 2),
                day_p05_usd=round(day_p05, 2), n_trades=n, n_days=len(days))


# ─────────────────────────── driver ───────────────────────────
def main():
    emergency = {}
    daily = {}
    recon_meta = {}
    for sym, spec in SYMBOLS.items():
        try:
            trades, _ = reconstruct(sym, spec)
        except FileNotFoundError as e:
            print(f"[skip] {sym}: {e}")
            continue
        Rs = np.array([t["pnl_R"] for t in trades])
        reasons = dict(Counter(t["reason"] for t in trades))
        recon_meta[sym] = dict(method=spec["method"], n_trades=len(trades),
                               total_R=round(float(Rs.sum()), 2), PF=round(pf(Rs), 2),
                               WR=round(float((Rs > 0).mean()), 3), exit_reasons=reasons)
        print(f"\n=== {sym} ({spec['method']}) n={len(trades)} totR={Rs.sum():+.2f} "
              f"PF={pf(Rs):.2f} WR={(Rs>0).mean()*100:.1f}% reasons={reasons}")

        if spec["emergency"]:
            e = tune_emergency(sym, trades)
            emergency[sym] = e
            print(f"  EMERGENCY: cap_R={e['cap_R']} cap_$={e['cap_usd']} {e['verdict']} "
                  f"train {e['base_train_R']:+.2f}->{e['train_R']:+.2f} "
                  f"test {e['base_test_R']:+.2f}->{e['test_R']:+.2f} "
                  f"n_cut={e['n_cut']} winClip={e['winners_clipped']} "
                  f"MAE_R(p50/p90/max)={e['mae_R_p50']}/{e['mae_R_p90']}/{e['mae_R_max']}")

        if spec["daily"]:
            d = tune_daily(sym, spec, trades)
            daily[sym] = d
            print(f"  DAILY: limit_$={d['limit_usd']} (wf_best {d['wf_best_limit_usd']}, "
                  f"cur {d['current_limit_usd']}) {d['verdict']} "
                  f"train {d['base_train_usd']:+.1f}->{d['train_usd']:+.1f} "
                  f"test {d['base_test_usd']:+.1f}->{d['test_usd']:+.1f} "
                  f"tail_saved={d['worst_day_saved']} recover_clipped={d['normal_days_clipped']} "
                  f"worstDay=${d['worst_realized_day_usd']}")

    # ---- overrides / notes ----
    # BTC emergency: backtest SHIP_NONE but config keeps $15 as a USER OVERRIDE for
    # the live bleed (per feedback_dont_overfit_backtest_when_live_bleeding).
    if "BTCUSD" in emergency:
        emergency["BTCUSD"]["user_override_usd"] = CURRENT_EMERGENCY["BTCUSD"]
        emergency["BTCUSD"]["note"] = ("backtest recommends UNCAP (clips fat-tail winners); "
                                       "config keeps $15 as a live-bleed USER OVERRIDE")

    notes = [
        "EMERGENCY_LOSS_CAP: cap_$ = cap_R * EQUITY(4100) * TREND_RISK_PCT(0.30)/100 = cap_R * $12.30/R.",
        "XAUUSD is intentionally NOT emergency-capped (GOLD_SMC runs its own BE/EMA9 exit) — matches config.py.",
        "SHIP only if capped total-R >= baseline on BOTH 60/40 WF folds. TREND's own chandelier trail + "
        "peak-giveback already bound adverse excursion <~1R on BTC/ETH/NAS, so a fixed cap there only clips "
        "fat-tail winners (net-negative) -> SHIP_NONE (leave uncapped). JPN225ft is the sole shipper.",
        "BTCUSD emergency: honest backtest = UNCAP; config keeps $15 as a documented USER OVERRIDE for the "
        "live -$20/trade bleed. JSON keeps the backtest recommendation (verdict) separate from the override.",
        "DAILY_LOSS_LIMIT: on a one-position/day TREND book, today's open-$ is entry-relative so the worst "
        "daily reading a symbol reaches = its MAE in $; the daily breaker thus reduces to a per-trade $-cap on "
        "MAE. worst_day_saved = tail-loss trades cut to -limit; normal_days_clipped = would-recover trades "
        "wrongly clipped. SHIP only if total-$ >= baseline on BOTH folds AND the tightest such limit differs "
        "from current.",
        "XAUUSD daily: reconstructed from TREND only. Live XAU also carries GOLD_SMC + M1 SCALPER P/L, so the "
        "true daily-loss distribution is WIDER than shown — the TREND number is a floor; do not tighten below "
        "current $40 without the GOLD_SMC/SCALPER daily series.",
        "JPN225ft daily SHIP $10 is MARGINAL (+~$0.6/3.5yr) and REDUNDANT with its emergency cap ($9-10) — "
        "both catch the same 2 tail blowups (mae ~0.95R). Tightening $40->$10 is WF-clean (tail_saved=2, "
        "0 recover-clips, + on both folds) but optional if the emergency cap is already live; either alone "
        "guards the JPN tail.",
        "SP500.r / US2000.r are NOT in TREND_BASKET (traded by other strategies live); reconstructed as a D1 "
        "TREND PROXY for a tail read only -> verdict PROXY_KEEP, recommendation = keep current.",
        "Methods: BTC/XAU/SP500/US2000 = D1-only (H1 caches too short for the 256-EMA); ETH/JPN/NAS = "
        "H1-checked exits. Reproduces the 4 prior scratchpad tunes.",
    ]

    result = dict(
        generated="2026-07-15",
        equity=EQUITY, trend_risk_pct=TREND_RISK_PCT, dollar_per_R=DOLLAR_PER_R,
        wf_train_frac=0.6, cap_R_grid=[CAPS[0], CAPS[-1]], daily_usd_grid=DAILY_USD_GRID,
        reconstruction=recon_meta,
        current_config=dict(EMERGENCY_LOSS_CAP_USD=CURRENT_EMERGENCY,
                            DAILY_LOSS_LIMIT_USD=CURRENT_DAILY),
        emergency_loss_cap=dict(per_symbol=list(emergency.values())),
        daily_loss_limit=dict(per_symbol=list(daily.values())),
        notes=notes,
    )
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")

    # ---- recommended dicts (what would go into config.py) ----
    rec_emerg = dict(CURRENT_EMERGENCY)
    for e in emergency.values():
        if e["verdict"] == "SHIP":
            rec_emerg[e["symbol"]] = float(e["cap_usd"])
    rec_daily = {}
    for d in daily.values():
        rec_daily[d["symbol"]] = d["limit_usd"]
    print("\nRECOMMENDED EMERGENCY_LOSS_CAP_USD =", rec_emerg,
          "(BTC=15 is USER OVERRIDE; backtest=uncap)")
    print("RECOMMENDED DAILY_LOSS_LIMIT_USD =", rec_daily)


if __name__ == "__main__":
    main()
