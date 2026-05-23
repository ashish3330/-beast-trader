#!/usr/bin/env python3 -B
"""
Live↔BT parity FINAL audit (2026-05-23) — pre-Monday verification.

Re-runs yesterday's parity replay against the journal AFTER today's 4 BT-
mirror patches (06784f4 + 7361367). Goal: confirm the new safety-layer
modeling closed the gap that yesterday's audit flagged.

Reads:
  - data_backups/20260522_224826/trade_journal.db  (30d journal,
    790 mt5_deal trades — current beast-trader/data/trade_journal.db has
    been reset to 65 trades for the Monday fresh-account run).
  - Cache at /Users/ashish/Documents/xauusd-trading-bot/cache/

Writes:
  - audit_pre_monday/parity_final_raw.json
  - audit_pre_monday/parity_final_detail.csv
"""
from __future__ import annotations

import sys
import json
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

from signals.momentum_scorer import _compute_indicators, IND_DEFAULTS, IND_OVERRIDES
from backtest.v5_backtest import (
    simulate_trail, SL_OVERRIDE, SL_OVERRIDE_REGIME,
    TRAIL_OVERRIDE, TRAIL_OVERRIDE_REGIME,
    ALL_SYMBOLS, DEFAULT_PARAMS, get_regime,
    SCORE_TIER_THRESHOLD, MARGINAL_TRAIL,
)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
JDB = ROOT / "data_backups" / "20260522_224826" / "trade_journal.db"

# ─── classification ───────────────────────────────────────────────────
SAFETY_LAYER_REASONS = {
    "PeakGiveback": "PEAK_GIVEBACK_EARLY",
    "EarlyLossCut": "EARLY_LOSS_CUT_EARLY",
    "EarlyLossCut_T1-MARGINAL-SCALP": "EARLY_LOSS_CUT_EARLY",
    "EarlyLossCut_T2-MARGINAL": "EARLY_LOSS_CUT_EARLY",
    "EarlyLossCut_T2-SWING": "EARLY_LOSS_CUT_EARLY",
    "EarlyLossCut_T3-IMMEDIATE": "EARLY_LOSS_CUT_EARLY",
    "AvgWinLossCap": "AVG_WIN_LOSS_CAP_EARLY",
    "HardDollarCap": "HARD_DOLLAR_CAP",
    "GuardianStaleLoser": "GUARDIAN_STALE",
    "GuardianDayLoss": "EXTERNAL",
    "GuardianHeatReduce": "EXTERNAL",
    "GuardianSharpLoss": "EXTERNAL",
    "EmergencyDD": "EXTERNAL",
    "DailyKillSwitch": "EXTERNAL",
    "DragonReversal": "EXTERNAL",
    "DragonWeekendClose": "EXTERNAL",
}
TRUNC_MAP = {
    "GuardianStaleLos": "GuardianStaleLoser",
    "GuardianDayLos":   "GuardianDayLoss",
    "GuardianHeatRedu": "GuardianHeatReduce",
    "GuardianSharpLos": "GuardianSharpLoss",
    "DragonWeekendClo": "DragonWeekendClose",
    "EarlyLossCut_T1-": "EarlyLossCut_T1-MARGINAL-SCALP",
    "EarlyLossCut_T2-": "EarlyLossCut_T2-MARGINAL",
    "EarlyLossCut_T3-": "EarlyLossCut_T3-IMMEDIATE",
}


def normalize_reason(raw: str) -> str:
    if not raw:
        return ""
    r = raw.strip()
    if r in TRUNC_MAP:
        return TRUNC_MAP[r]
    return r


def divergence_category(live_reason: str) -> str:
    if not live_reason:
        return "OTHER"
    if live_reason.startswith("[sl ") or live_reason.startswith("[tp "):
        return "SL_OR_TP"
    norm = normalize_reason(live_reason)
    return SAFETY_LAYER_REASONS.get(norm, "OTHER")


# ─── cache loader ─────────────────────────────────────────────────────
_CACHE_DF: dict = {}
_CACHE_IND: dict = {}


def get_h1_df(symbol: str):
    if symbol in _CACHE_DF:
        return _CACHE_DF[symbol]
    meta = ALL_SYMBOLS.get(symbol)
    if not meta:
        _CACHE_DF[symbol] = None
        return None
    fn = CACHE / meta["cache"]
    if not fn.exists():
        _CACHE_DF[symbol] = None
        return None
    df = pickle.load(open(fn, "rb"))
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    _CACHE_DF[symbol] = df
    return df


def get_ind(symbol: str, df: pd.DataFrame):
    if symbol in _CACHE_IND:
        return _CACHE_IND[symbol]
    icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(symbol, {})}
    try:
        ind = _compute_indicators(df, icfg)
    except Exception:
        ind = None
    _CACHE_IND[symbol] = ind
    return ind


# ─── replay per-trade ─────────────────────────────────────────────────
def replay_trade(t: dict) -> dict | None:
    symbol = t["symbol"]
    df = get_h1_df(symbol)
    if df is None or len(df) < 100:
        return {"replay_status": "NO_CACHE"}

    entry_ts = pd.Timestamp(t["timestamp"])
    if entry_ts.tz is None:
        entry_ts = entry_ts.tz_localize("UTC")
    try:
        entry_ts = entry_ts.as_unit("ns")
    except Exception:
        pass
    last_bar = df["time"].iloc[-1]
    if entry_ts > last_bar:
        return {"replay_status": "DATA_GAP_BEFORE_EXIT"}

    times = df["time"].values.astype("datetime64[ns]")
    idx = np.searchsorted(times, np.datetime64(entry_ts.to_numpy(), "ns"), side="right") - 1
    if idx < 30:
        return {"replay_status": "WARMUP_NOT_READY"}
    bar_i = int(idx)
    n = len(df)

    ind = get_ind(symbol, df)
    if ind is None:
        return {"replay_status": "INDICATOR_NONE"}

    atr = float(ind["at"][bar_i])
    if not np.isfinite(atr) or atr <= 0:
        return {"replay_status": "BAD_ATR"}

    bbw_val = float(ind["bbw"][bar_i]) if not np.isnan(ind["bbw"][bar_i]) else 0.02
    adx_val = float(ind["adx"][bar_i]) if not np.isnan(ind["adx"][bar_i]) else 20
    regime = get_regime(bbw_val, adx_val)
    sl_mult = (SL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
               or SL_OVERRIDE.get(symbol)
               or DEFAULT_PARAMS["sl_atr_mult"])
    sl_dist = atr * sl_mult

    direction = 1 if t["direction"] == "LONG" else -1
    entry_price = float(t["entry_price"] or 0)
    if entry_price <= 0:
        return {"replay_status": "ZERO_ENTRY_PRICE"}

    trail_steps = (TRAIL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
                   or TRAIL_OVERRIDE.get(symbol)
                   or DEFAULT_PARAMS["trail"])

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)

    start_i = bar_i + 1
    if start_i >= n - 1:
        return {"replay_status": "ENTRY_AT_LAST_BAR"}

    # ── SCORE-TIER DISPATCH ─ mirror BT entry loop (commit 7361367)
    # If raw score available in journal, use it; otherwise treat as
    # unknown and assume swing tier (matches the BT fallback when
    # raw_score==0 is recorded for legacy trades).
    raw_score = float(t.get("score") or 0)
    is_marginal_bt = 0 < raw_score < SCORE_TIER_THRESHOLD
    if is_marginal_bt:
        eff_trail = MARGINAL_TRAIL
    else:
        eff_trail = trail_steps

    try:
        exit_price, exit_bar, exit_reason, peak_r = simulate_trail(
            entry_price, sl_dist, direction, h, l, c,
            start_i, n,
            spread=0.0,
            trail_steps=eff_trail,
            ratchet_1r=DEFAULT_PARAMS.get("ratchet_1r", 0.3),
            ratchet_2r=DEFAULT_PARAMS.get("ratchet_2r", 0.7),
            is_marginal=is_marginal_bt,
        )
    except Exception as e:
        return {"replay_status": f"TRAIL_ERR:{e}"}

    pnl_r = ((exit_price - entry_price) * direction) / sl_dist if sl_dist > 0 else 0
    live_pnl = float(t.get("pnl") or 0)
    live_r = float(t.get("r_multiple") or 0)

    dollar_risk = None
    if abs(live_r) >= 0.05 and abs(live_r) <= 5.0 and abs(live_pnl) > 0.01:
        implied = abs(live_pnl / live_r)
        if 0.30 <= implied <= 30.0:
            dollar_risk = implied
    if dollar_risk is None:
        rp = float(t.get("risk_pct") or 0.4)
        dollar_risk = 1500.0 * (rp / 100.0)

    bt_pnl_dollar = dollar_risk * pnl_r
    div_r = live_r - pnl_r
    div_dollar = live_pnl - bt_pnl_dollar
    div_dollar_clamped = max(-50.0, min(50.0, div_dollar))

    return {
        "replay_status": "OK",
        "bt_exit_price": float(exit_price),
        "bt_exit_reason": exit_reason,
        "bt_pnl_r": float(pnl_r),
        "bt_pnl_dollar": float(bt_pnl_dollar),
        "bt_peak_r": float(peak_r),
        "bt_bars_to_exit": int(exit_bar - bar_i),
        "divergence_dollar": float(div_dollar_clamped),
        "divergence_dollar_raw": float(div_dollar),
        "divergence_r": float(div_r),
        "regime_at_entry": regime,
        "sl_dist": float(sl_dist),
        "dollar_risk": float(dollar_risk),
        "is_marginal_bt": bool(is_marginal_bt),
        "raw_score_used": raw_score,
    }


def main():
    print(f"Reading journal: {JDB}")
    con = sqlite3.connect(JDB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT * FROM trades WHERE source='mt5_deal' ORDER BY timestamp"""
    ).fetchall()
    con.close()
    trades = [dict(r) for r in rows]
    print(f"Loaded {len(trades)} closed live trades.")

    by_category = defaultdict(lambda: {"count": 0, "divergence": 0.0,
                                       "divergence_r": 0.0,
                                       "live_pnl_all": 0.0,
                                       "live_pnl": 0.0,
                                       "bt_pnl": 0.0,
                                       "live_r": 0.0, "bt_r": 0.0,
                                       "replayed": 0})
    by_symbol = defaultdict(lambda: {"count": 0, "divergence": 0.0,
                                     "divergence_r": 0.0,
                                     "live_pnl_all": 0.0,
                                     "live_pnl": 0.0,
                                     "bt_pnl": 0.0,
                                     "live_r": 0.0, "bt_r": 0.0,
                                     "replayed": 0})
    by_tier = defaultdict(lambda: {"count": 0, "divergence_r": 0.0,
                                   "live_r": 0.0, "bt_r": 0.0,
                                   "replayed": 0})
    skip_reasons = defaultdict(int)
    n_replayed = 0
    n_skipped = 0
    total_live_pnl = 0.0
    total_bt_pnl = 0.0
    total_divergence = 0.0
    total_live_r = 0.0
    total_bt_r = 0.0
    total_div_r = 0.0

    detail_rows = []

    for i_t, t in enumerate(trades):
        if i_t % 100 == 0:
            print(f"  [{i_t}/{len(trades)}] processed...", flush=True)
        live_reason = normalize_reason(t.get("exit_reason") or "")
        cat = divergence_category(live_reason)
        sym = t["symbol"]
        live_pnl = float(t.get("pnl") or 0)
        total_live_pnl += live_pnl
        by_category[cat]["count"] += 1
        by_category[cat]["live_pnl_all"] += live_pnl
        by_symbol[sym]["count"] += 1
        by_symbol[sym]["live_pnl_all"] += live_pnl

        r = replay_trade(t)
        if not r or r.get("replay_status") != "OK":
            n_skipped += 1
            skip_reasons[r["replay_status"] if r else "NO_REPLAY"] += 1
            continue
        n_replayed += 1
        bt_pnl = r["bt_pnl_dollar"]
        div = r["divergence_dollar"]
        div_r = r["divergence_r"]
        live_r = float(t.get("r_multiple") or 0)
        bt_r = r["bt_pnl_r"]
        tier_key = "MARGINAL_BT" if r["is_marginal_bt"] else "SWING_BT"
        total_bt_pnl += bt_pnl
        total_divergence += div
        total_live_r += live_r
        total_bt_r += bt_r
        total_div_r += div_r
        by_category[cat]["live_pnl"] += live_pnl
        by_category[cat]["bt_pnl"] += bt_pnl
        by_category[cat]["divergence"] += div
        by_category[cat]["divergence_r"] += div_r
        by_category[cat]["live_r"] += live_r
        by_category[cat]["bt_r"] += bt_r
        by_category[cat]["replayed"] += 1
        by_symbol[sym]["live_pnl"] += live_pnl
        by_symbol[sym]["bt_pnl"] += bt_pnl
        by_symbol[sym]["divergence"] += div
        by_symbol[sym]["divergence_r"] += div_r
        by_symbol[sym]["live_r"] += live_r
        by_symbol[sym]["bt_r"] += bt_r
        by_symbol[sym]["replayed"] += 1
        by_tier[tier_key]["replayed"] += 1
        by_tier[tier_key]["count"] += 1
        by_tier[tier_key]["divergence_r"] += div_r
        by_tier[tier_key]["live_r"] += live_r
        by_tier[tier_key]["bt_r"] += bt_r
        detail_rows.append({
            "id": t["id"], "ts": t["timestamp"], "symbol": sym,
            "direction": t["direction"], "live_reason": live_reason,
            "live_pnl": live_pnl, "live_r": live_r,
            "bt_reason": r["bt_exit_reason"],
            "bt_pnl": bt_pnl, "bt_r": bt_r,
            "divergence": div, "divergence_r": div_r,
            "divergence_dollar_raw": r["divergence_dollar_raw"],
            "category": cat, "bt_peak_r": r["bt_peak_r"],
            "bt_bars": r["bt_bars_to_exit"],
            "is_marginal_bt": r["is_marginal_bt"],
            "raw_score": r["raw_score_used"],
            "regime": r["regime_at_entry"],
        })

    print(f"Replayed: {n_replayed}, Skipped: {n_skipped}")
    print(f"  Skip reasons: {dict(skip_reasons)}")
    print(f"Live total PnL (all):       ${total_live_pnl:.2f}")
    print(f"BT replay PnL (replayed):   ${total_bt_pnl:.2f}")
    print(f"Total $ divergence (clamped): ${total_divergence:.2f}")
    print(f"Live total R (replayed):  {total_live_r:+.2f}R")
    print(f"BT replay R (replayed):   {total_bt_r:+.2f}R")
    print(f"Total R divergence:       {total_div_r:+.2f}R")
    print()
    print("BY TIER (BT dispatch):")
    for k, v in by_tier.items():
        print(f"  {k}: n={v['count']} live_r={v['live_r']:+.2f} bt_r={v['bt_r']:+.2f} div={v['divergence_r']:+.2f}")

    out = {
        "n_trades_total": len(trades),
        "n_replayed": n_replayed,
        "n_skipped": n_skipped,
        "skip_reasons": dict(skip_reasons),
        "total_live_pnl": round(total_live_pnl, 2),
        "total_bt_pnl": round(total_bt_pnl, 2),
        "total_divergence_dollar": round(total_divergence, 2),
        "total_live_r": round(total_live_r, 2),
        "total_bt_r": round(total_bt_r, 2),
        "total_divergence_r": round(total_div_r, 2),
        "by_category": {k: {x: round(v[x], 2) if isinstance(v[x], float) else v[x]
                            for x in v} for k, v in by_category.items()},
        "by_symbol": {k: {x: round(v[x], 2) if isinstance(v[x], float) else v[x]
                          for x in v} for k, v in by_symbol.items()},
        "by_tier": {k: {x: round(v[x], 2) if isinstance(v[x], float) else v[x]
                        for x in v} for k, v in by_tier.items()},
    }
    out_json = ROOT / "audit_pre_monday" / "parity_final_raw.json"
    json.dump(out, open(out_json, "w"), indent=2, default=str)
    print(f"Wrote {out_json}")

    df_det = pd.DataFrame(detail_rows)
    df_det.to_csv(ROOT / "audit_pre_monday" / "parity_final_detail.csv", index=False)
    print(f"Wrote {ROOT / 'audit_pre_monday' / 'parity_final_detail.csv'}")

    return out


if __name__ == "__main__":
    main()
