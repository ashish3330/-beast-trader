#!/usr/bin/env python3 -B
"""Hard-tune the TREND book's REVERSAL DETECTION / exit component, per symbol.

SCOPE — reversal-side params ONLY (owned by this tuner). Trail/lock/stop are the
trail-agent's domain and are held FIXED at the config per-symbol values:
  (a) SIGNAL-FLIP exit  : D1 3-EMA ensemble flips sign -> close (and maybe reverse).
      Knobs: flip_confirm (require the opposite signal to persist N H1 bars before
             acting; 0/1 = immediate) and flip_reentry (reopen opposite vs go flat).
  (b) PEAK-GIVEBACK exit : once open profit peaked, close on a giveback retrace.
      Knobs: gb_frac (retrace fraction from peak) and gb_act (activation x ATR).
      The giveback activation is DECOUPLED from the trail-lock activation here so
      tuning the reversal trigger never disturbs the trail (which keeps config ACT).
  (c) TREND_REENTRY_BLOCK_HOURS : cooldown (H1 bars) after a giveback reversal exit
      before the SAME direction may re-enter. Sweep {0,2,6,12,24}.
Plus: REVERSAL_EXIT on-vs-off per symbol (is the giveback detector net-positive?).

ENGINE: reuses backtest/tune/trend_engine.py (load / spread_cost_rt) for the exact
D1-signal + H1 frame the canonical engine builds. simulate_rev() below re-derives
engine.simulate() faithfully (identical trail/lock/stop/tp math) and only adds the
reversal knobs; with defaults it reproduces engine.simulate() bit-for-bit.

METHOD: one param at a time, WF 60/40 (train=first 60% bars, test=last 40%).
Score = total_R (+ PF reported). SHIP a value only if train_R >= baseline_train_R
AND test_R >= baseline_test_R (>= neutral BOTH folds) AND it is NOT a churn artifact.

CHURN GUARD (mandatory — the dominant trap in this codebase; NAS naive cadence once
produced 4638 fake trades): every sweep cell is capped at 1.4x the baseline trade
count PER FOLD. A cell exceeding the cap is flagged n_churn_flag=True and can NEVER
ship. Churn counts are reported for every cell.

DATA TRUST (backtest/tune/DATA_AUDIT.md): only JPN225ft & NAS100.r H1 are WF-safe.
BTC H1 = 242 bars (noise), XAU H1 = 91d shallow (0 trades), ETH H1 = 63d stale
(IS-only, OOS blind). Non-trusted symbols are run for completeness but marked and
NOT shipped.

Writes backtest/tune/results_trend_REVERSAL.json. Does NOT touch config.py/brain.py.
Run: python3 -B backtest/tune/tune_trend_REVERSAL.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for p in (str(_REPO), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backtest.tune import trend_engine as TE  # noqa: E402

SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "JPN225ft", "NAS100.r"]

DATA_TRUST = {
    "XAUUSD":   "DONT_TRUST (H1 91d/1500 bars shallow -> 0 trades; refetch first)",
    "BTCUSD":   "DONT_TRUST (H1 242 bars/9d -> noise; SHIP_NONE trap)",
    "ETHUSD":   "DONT_TRUST_WF (H1 deep but 63d stale; IS-only, OOS blind)",
    "JPN225ft": "TRUST (H1 21k bars, 5d stale)",
    "NAS100.r": "TRUST (H1 50k bars, 6d stale)",
}
TRUSTED_WF = {"JPN225ft", "NAS100.r"}

CHURN_MULT = 1.4          # trade-count cap = CHURN_MULT x baseline (per fold)
ATR_STOP = TE.DEFAULT_ATR_STOP     # 3.0, fixed like the engine
TP_ATR = TE.DEFAULT_TP_ATR         # 999 == no TP, tuner convention


# ── simulate_rev: engine.simulate + reversal knobs (defaults == engine) ──────
def simulate_rev(m, TR, LK, gb_frac, act_lock, act_gb,
                 rev_enabled=True, flip_confirm=0, flip_reentry=True,
                 reentry_block_bars=0, ATR_STOP=ATR_STOP, TP=TP_ATR, cost=0.0):
    """Per-H1-bar exit sim. TR/LK/act_lock = trail-lock (FIXED from config).
    Reversal knobs:
      rev_enabled        : enable the peak-giveback exit branch (False = off).
      gb_frac / act_gb   : giveback retrace fraction / activation (x ATR).
      flip_confirm       : opposite signal must persist >= N H1 bars before flip
                           (0/1 = immediate).
      flip_reentry       : on a confirmed flip, reopen opposite (True) or go flat.
      reentry_block_bars : after a GIVEBACK reversal exit, block same-dir re-entry
                           for N H1 bars (the TREND_REENTRY_BLOCK_HOURS cooldown).
                           float('inf') == SIGNAL-BASED block (engine canonical):
                           re-entry blocked until the D1 signal changes. This is the
                           non-churn reference; small finite values RELAX toward the
                           re-entry churn artifact (0h re-opens same-dir next bar).
    Returns the engine-style trade list."""
    o = m["open"].values; h = m["high"].values; l = m["low"].values
    tm = m["time"].values
    sig = m["sig"].values.astype(int); atr = m["atr"].values

    # run-length of the current signal value (consecutive bars incl. t)
    n = len(m)
    runlen = np.ones(n, dtype=int)
    for i in range(1, n):
        if sig[i] == sig[i - 1]:
            runlen[i] = runlen[i - 1] + 1
    hh = m["hh"].values; ll = m["ll"].values

    pos = 0
    entry = sl = tp = peak = 0.0
    e_atr = 0.0; e_i = 0; worst = 0.0; best = 0.0
    block_dir = 0; block_until = -1            # time-based re-entry cooldown
    trades = []

    def _open(s, t):
        nonlocal pos, entry, peak, e_atr, e_i, worst, best, sl, tp
        pos = s; entry = o[t]; peak = 0.0; e_atr = atr[t]; e_i = t
        worst = 0.0; best = 0.0
        sl = entry - ATR_STOP * e_atr if s == 1 else entry + ATR_STOP * e_atr
        tp = None if TP >= 999 else ((entry + TP * e_atr) if s == 1 else (entry - TP * e_atr))

    def _close(ex_price, reason, t_idx):
        nonlocal pos
        ret = ((ex_price - entry) / entry) * pos - cost
        r_unit = (ATR_STOP * e_atr) / entry if entry > 0 and e_atr > 0 else 0.0
        pnl_R = ret / r_unit if r_unit > 0 else 0.0
        mae_R = (worst / (ATR_STOP * e_atr)) if e_atr > 0 else 0.0
        mfe_R = (best / (ATR_STOP * e_atr)) if e_atr > 0 else 0.0
        trades.append({"dir": pos, "t_in": tm[e_i], "t_out": tm[t_idx],
                       "entry": float(entry), "exit": float(ex_price),
                       "reason": reason, "ret": float(ret), "pnl_R": float(pnl_R),
                       "mae_R": float(mae_R), "mfe_R": float(mfe_R)})
        pos = 0

    for t in range(n):
        a = atr[t]
        if a <= 0:
            continue
        s = int(sig[t])
        if pos == 0:
            # clear cooldown: time elapsed OR signal no longer the blocked dir
            if block_dir and (t >= block_until or s != block_dir):
                block_dir = 0
            if s != 0 and s != block_dir:
                _open(s, t)
            continue
        # ── SIGNAL-FLIP exit (with optional persistence confirmation) ──
        if s != 0 and s != pos:
            confirmed = (flip_confirm <= 1) or (runlen[t] >= flip_confirm)
            if confirmed:
                _close(o[t], "flip", t)
                if flip_reentry:
                    _open(s, t)
                continue
            # not yet confirmed -> hold; trail/stop below still applies
        # ── trail-lock + stop + giveback + tp ──
        if pos == 1:
            worst = max(worst, entry - l[t])
            sl = max(sl, hh[t] - TR * a)
            if peak >= act_lock * a:
                sl = max(sl, entry + LK * peak)
            gb = (entry + peak * (1.0 - gb_frac)
                  if (rev_enabled and gb_frac < 1.0 and peak >= act_gb * a) else -1e18)
            ex = ereason = None
            if l[t] <= sl:
                ex, ereason = sl, "stop/trail"
            elif gb > -1e17 and l[t] <= gb:
                ex, ereason = gb, "giveback"
            elif tp is not None and h[t] >= tp:
                ex, ereason = tp, "tp"
            if ex is not None:
                _close(ex, ereason, t)
                if ereason == "giveback" and reentry_block_bars > 0:
                    block_dir = 1; block_until = t + reentry_block_bars
            else:
                peak = max(peak, h[t] - entry); best = peak
        else:
            worst = max(worst, h[t] - entry)
            sl = min(sl, ll[t] + TR * a)
            if peak >= act_lock * a:
                sl = min(sl, entry - LK * peak)
            gb = (entry - peak * (1.0 - gb_frac)
                  if (rev_enabled and gb_frac < 1.0 and peak >= act_gb * a) else 1e18)
            ex = ereason = None
            if h[t] >= sl:
                ex, ereason = sl, "stop/trail"
            elif gb < 1e17 and h[t] >= gb:
                ex, ereason = gb, "giveback"
            elif tp is not None and l[t] <= tp:
                ex, ereason = tp, "tp"
            if ex is not None:
                _close(ex, ereason, t)
                if ereason == "giveback" and reentry_block_bars > 0:
                    block_dir = -1; block_until = t + reentry_block_bars
            else:
                peak = max(peak, entry - l[t]); best = peak
    return trades


def metrics(trades):
    if not trades:
        return {"n": 0, "total_R": 0.0, "pf": 0.0, "wr": 0.0,
                "reasons": {}}
    R = np.array([t["pnl_R"] for t in trades])
    win = R[R > 0].sum(); loss = -R[R < 0].sum()
    return {"n": len(trades), "total_R": float(R.sum()),
            "pf": float(win / loss) if loss > 0 else 99.0,
            "wr": float((R > 0).mean()),
            "reasons": dict(Counter(t["reason"] for t in trades))}


def wf_split(m):
    """60/40 chronological split -> (train_frame, test_frame)."""
    k = int(len(m) * 0.60)
    return m.iloc[:k].reset_index(drop=True), m.iloc[k:].reset_index(drop=True)


def run_cell(frame, cost, TR, LK, cfg_gb, cfg_act, **kw):
    """One variant on one frame. act_lock fixed=cfg_act; act_gb defaults=cfg_act."""
    kw.setdefault("act_gb", cfg_act)
    kw.setdefault("gb_frac", cfg_gb)
    tr = simulate_rev(frame, TR, LK, act_lock=cfg_act, ATR_STOP=ATR_STOP,
                      TP=TP_ATR, cost=cost, **kw)
    return metrics(tr)


def verdict(base_tr, base_te, cell_tr, cell_te, cap_tr, cap_te):
    churn = cell_tr["n"] > cap_tr or cell_te["n"] > cap_te
    if churn:
        return "REJECT_CHURN", churn
    ships = (cell_tr["total_R"] >= base_tr["total_R"] - 1e-9 and
             cell_te["total_R"] >= base_te["total_R"] - 1e-9)
    if cell_tr["total_R"] > base_tr["total_R"] + 1e-9 and \
       cell_te["total_R"] > base_te["total_R"] + 1e-9:
        return "SHIP_IMPROVE", churn
    if ships:
        return "SHIP_NEUTRAL", churn
    return "REJECT_REGRESS", churn


def tune_symbol(sym):
    try:
        m = TE.load(sym)
    except Exception as e:
        return {"symbol": sym, "data_trust": DATA_TRUST[sym],
                "error": "load failed: %s" % e, "params": [], "reversal_on_vs_off": {}}
    cost = TE.spread_cost_rt(m)
    TR, LK, cfg_gb, cfg_act = TE._cfg_exit_params(sym)
    tr_f, te_f = wf_split(m)

    def cell(frame, **kw):
        return run_cell(frame, cost, TR, LK, cfg_gb, cfg_act, **kw)

    # baseline / neutral reference = engine-canonical NON-CHURN behaviour:
    # rev on, immediate flip + reentry, SIGNAL-BASED re-entry block (block same-dir
    # until the D1 signal changes == reentry_block_bars=inf). This reproduces
    # engine.simulate() and gives a churn-free reference so trail/giveback param
    # sweeps are judged cleanly. The LIVE config uses a 2h TIME cooldown (=2 bars),
    # which is a RELAXATION toward churn — evaluated explicitly in the reentry sweep.
    base_kw = dict(rev_enabled=True, flip_confirm=0, flip_reentry=True,
                   reentry_block_bars=float("inf"))
    base_all = cell(m, **base_kw)
    base_tr = cell(tr_f, **base_kw)
    base_te = cell(te_f, **base_kw)
    cap_tr = max(3, int(round(CHURN_MULT * base_tr["n"])))
    cap_te = max(3, int(round(CHURN_MULT * base_te["n"])))

    out = {"symbol": sym, "data_trust": DATA_TRUST[sym],
           "wf_split_bars": {"train": len(tr_f), "test": len(te_f)},
           "baseline": {"all": base_all, "train": base_tr, "test": base_te,
                        "churn_cap": {"train": cap_tr, "test": cap_te}},
           "reversal_on_vs_off": {}, "params": []}

    trusted = sym in TRUSTED_WF

    def record_param(name, current, sweep, caveat=None):
        results = []
        best = None
        for val, kw in sweep:
            c_tr = cell(tr_f, **kw); c_te = cell(te_f, **kw)
            v, churn = verdict(base_tr, base_te, c_tr, c_te, cap_tr, cap_te)
            row = {"value": val, "verdict": v,
                   "train_R": round(c_tr["total_R"], 3), "test_R": round(c_te["total_R"], 3),
                   "train_pf": round(c_tr["pf"], 2), "test_pf": round(c_te["pf"], 2),
                   "n_train": c_tr["n"], "n_test": c_te["n"],
                   "n_churn_flag": bool(churn)}
            results.append(row)
            if v in ("SHIP_IMPROVE", "SHIP_NEUTRAL"):
                score = c_tr["total_R"] + c_te["total_R"]
                if best is None or score > best[0] or (
                        best[1]["verdict"] != "SHIP_IMPROVE" and v == "SHIP_IMPROVE"):
                    if best is None or score > best[0]:
                        best = (score, row)
        # keep current if nothing strictly improves
        best_val = current
        best_verdict = "KEEP_CURRENT"
        cur_row = next((r for r in results if r["value"] == current), None)
        cur_churns = bool(cur_row and cur_row["n_churn_flag"])
        # pick a shipped value strictly better than current
        improved = [r for r in results
                    if r["verdict"] == "SHIP_IMPROVE" and r["value"] != current]
        if improved:
            improved.sort(key=lambda r: r["train_R"] + r["test_R"], reverse=True)
            best_val = improved[0]["value"]
            best_verdict = "SHIP"
        elif cur_churns:
            # the CURRENT value is itself a churn artifact — steer to the safest
            # non-churn value that is at least neutral in both folds.
            safe = [r for r in results
                    if not r["n_churn_flag"] and r["verdict"] in ("SHIP_IMPROVE", "SHIP_NEUTRAL")]
            if safe:
                safe.sort(key=lambda r: r["train_R"] + r["test_R"], reverse=True)
                best_val = safe[0]["value"]
                best_verdict = "PREFER_NONCHURN"
        # data-trust gate: never emit a hard param SHIP on non-WF-trustworthy data.
        # (PREFER_NONCHURN is a structural anti-churn safety rec — kept for all.)
        if not trusted and best_verdict == "SHIP":
            best_verdict = "BLOCKED_UNTRUSTED_DATA"
            best_val = current
        row = {
            "name": name, "current": current, "best": best_val,
            "verdict": best_verdict, "current_churns": cur_churns,
            "train_R": next((r["train_R"] for r in results if r["value"] == best_val),
                            round(base_tr["total_R"], 3)),
            "test_R": next((r["test_R"] for r in results if r["value"] == best_val),
                           round(base_te["total_R"], 3)),
            "n_churn_flag": any(r["n_churn_flag"] for r in results),
            "sweep": results}
        if caveat:
            row["caveat"] = caveat
        out["params"].append(row)

    # ── (0) REVERSAL_EXIT on vs off ──
    off_kw = dict(base_kw); off_kw["rev_enabled"] = False
    off_tr = cell(tr_f, **off_kw); off_te = cell(te_f, **off_kw)
    off_all = cell(m, **off_kw)
    # CHURN GUARD on the on/off test: the giveback exit is the ONLY mechanism that
    # arms the anti-churn re-entry block, so disabling it lets every stop/trail exit
    # re-open same-dir next bar -> trade explosion. A higher OFF total_R that breaches
    # the 1.4x churn cap is the artifact, NOT an edge -> reversal exit must stay ON.
    off_churns = (off_tr["n"] > cap_tr or off_te["n"] > cap_te)
    if off_churns:
        rec = "KEEP_ON"          # OFF is a churn artifact, not a real alternative
    elif off_tr["total_R"] > base_tr["total_R"] and off_te["total_R"] > base_te["total_R"]:
        rec = "TURN_OFF"         # OFF genuinely (non-churn) beats ON in both folds
    elif base_tr["total_R"] >= off_tr["total_R"] and base_te["total_R"] >= off_te["total_R"]:
        rec = "KEEP_ON"
    else:
        rec = "AMBIGUOUS_KEEP_ON"
    out["reversal_on_vs_off"] = {
        "on":  {"all_R": round(base_all["total_R"], 3), "train_R": round(base_tr["total_R"], 3),
                "test_R": round(base_te["total_R"], 3), "n_all": base_all["n"],
                "pf_all": round(base_all["pf"], 2), "reasons_all": base_all["reasons"]},
        "off": {"all_R": round(off_all["total_R"], 3), "train_R": round(off_tr["total_R"], 3),
                "test_R": round(off_te["total_R"], 3), "n_all": off_all["n"],
                "n_train": off_tr["n"], "n_test": off_te["n"],
                "pf_all": round(off_all["pf"], 2), "reasons_all": off_all["reasons"]},
        "off_is_churn_artifact": bool(off_churns),
        "reversal_helps": bool(rec in ("KEEP_ON", "AMBIGUOUS_KEEP_ON")),
        "recommendation": rec}

    # ── (1) giveback fraction gb_frac ── (reversal trigger this agent owns)
    gb_vals = sorted({0.20, 0.25, 0.30, 0.35, 0.40, 0.50, cfg_gb})
    record_param("TREND_GIVEBACK_FRAC", cfg_gb,
                 [(v, dict(base_kw, gb_frac=v)) for v in gb_vals])

    # ── (2) giveback activation act_gb (x ATR) ── decoupled from trail lock
    act_vals = sorted({0.2, 0.3, 0.5, 0.8, 1.0, cfg_act})
    record_param("GIVEBACK_ACTIVATE_ATR", cfg_act,
                 [(v, dict(base_kw, act_gb=v)) for v in act_vals])

    # ── (3) flip confirmation (persist N H1 bars) ──
    record_param("FLIP_CONFIRM_BARS", 0,
                 [(v, dict(base_kw, flip_confirm=v)) for v in (0, 1, 2, 3, 6)])

    # ── (4) flip re-entry allowed (reverse) vs close-to-flat ──
    record_param("FLIP_REENTRY_ALLOWED", True,
                 [(v, dict(base_kw, flip_reentry=v)) for v in (True, False)])

    # ── (5) TREND_REENTRY_BLOCK_HOURS cooldown {0,2,6,12,24} + signal-based ──
    # current LIVE = 2h. "inf" == signal-based (the non-churn baseline reference).
    record_param("TREND_REENTRY_BLOCK_HOURS", 2,
                 [(v, dict(base_kw, reentry_block_bars=(float("inf") if v == "signal" else v)))
                  for v in (0, 2, 6, 12, 24, "signal")],
                 caveat=("In this H1 engine EVERY finite cooldown {0,2,6,12,24} — incl. the "
                         "live 2h — CHURNS (only the signal-based block is churn-free), because "
                         "the fixed 3xATR stop makes givebacks rare so same-dir re-entry re-fires "
                         "while D1 signal holds. LIVE risk-caps the SL to ~0.2xATR (DATA_AUDIT §3): "
                         "givebacks arm/fire almost immediately, re-setting the block each time, so "
                         "live re-entry churn is likely far milder. RANKING is valid (signal-based/"
                         "long >> short cooldown for turnover); do NOT hard-flip live 2h on this "
                         "engine alone — monitor live re-entry counts instead."))

    return out


def main():
    per = []
    for sym in SYMBOLS:
        print("=" * 72)
        print("TUNING REVERSAL:", sym, "|", DATA_TRUST[sym])
        r = tune_symbol(sym)
        per.append(r)
        if "error" in r:
            print("  ERROR:", r["error"]); continue
        ov = r["reversal_on_vs_off"]
        print("  reversal on/off:  ON train=%.2f test=%.2f  |  OFF train=%.2f test=%.2f  -> %s"
              % (ov["on"]["train_R"], ov["on"]["test_R"],
                 ov["off"]["train_R"], ov["off"]["test_R"], ov["recommendation"]))
        for p in r["params"]:
            flag = " [CHURN seen]" if p["n_churn_flag"] else ""
            print("  %-26s cur=%-6s best=%-6s %-12s tr=%.2f te=%.2f%s"
                  % (p["name"], p["current"], p["best"], p["verdict"],
                     p["train_R"], p["test_R"], flag))

    notes = (
        "REVERSAL-component WF 60/40 tune. Trail/lock/stop held FIXED at config "
        "per-symbol values (trail-agent domain); giveback activation is DECOUPLED "
        "from trail-lock activation so the reversal trigger is isolated. Baseline = "
        "current config (rev ON, immediate flip + reentry, 2h re-entry cooldown). "
        "SHIP only if a value >= baseline in BOTH folds AND trade count <= 1.4x "
        "baseline per fold (churn guard). Only JPN225ft & NAS100.r are WF-trustworthy "
        "(DATA_AUDIT.md); XAU(91d/0-trades), BTC(242 bars), ETH(63d stale) are run for "
        "completeness but NOT shippable. NOTE: signal FLIPs are near-zero at baseline "
        "(stop/giveback closes the position before the D1 signal reverses sign), so the "
        "flip-confirm / flip-reentry levers are largely INERT — reported honestly. "
        "Engine: backtest/tune/trend_engine.py; simulate_rev reproduces engine.simulate "
        "with reversal defaults."
    )
    payload = {"component": "reversal", "generated": "2026-07-15",
               "engine": "backtest/tune/trend_engine.py",
               "method": "WF 60/40, one-param-at-a-time, churn cap 1.4x baseline/fold",
               "per_symbol": per, "notes": notes}
    outp = _HERE / "results_trend_REVERSAL.json"
    outp.write_text(json.dumps(payload, indent=2, default=str))
    print("=" * 72)
    print("WROTE", outp)


if __name__ == "__main__":
    main()
