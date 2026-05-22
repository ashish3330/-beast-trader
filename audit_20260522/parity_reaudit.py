#!/usr/bin/env python3 -B
"""
Live↔BT parity re-audit (2026-05-22) — post-trail-fix + post-safety-layer.

Replays each closed live trade against the current BT trail/exit logic and
quantifies per-category divergence.

Approach (read-only):
  1. Pull all closed mt5_deal trades from journal.
  2. For each trade, find the H1 bar at-or-just-before entry timestamp in
     cache. Compute ATR + SL_dist using same logic as BT (SL_OVERRIDE *
     ATR). Run simulate_trail (BT's exit logic) starting from entry bar.
  3. Compare:
       - exit reason categorization (live vs BT)
       - exit PnL ($ Δ live − BT)
       - exit timing (bars-to-exit)
  4. Group divergence by category and symbol.

Categories of divergence:
  - SAFETY_LAYER_FIRED — live cut by safety layer that BT lacks
        (PeakGiveback, EarlyLossCut*, AvgWinLossCap, HardDollarCap,
         Guardian*, DragonReversal, DailyKillSwitch, EmergencyDD)
  - SL_TIMING — both hit SL but at different bars/prices
  - DATA_GAP — cache doesn't cover the entry timestamp (skipped)
  - REPLAY_OK — exit prices/reasons agree within tolerance

Constraints:
  - APPROXIMATION: BT uses H1 bars. Live trails per-tick (every 0.5s).
    H1 underestimates intra-bar excursion granularity → BT trail-lag is
    capped at one H1 bar.
  - Symbols with stale cache (data ends mid-period) → trades after data
    end are tagged DATA_GAP, not replayed.
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
)

CACHE = Path("/Users/ashish/Documents/xauusd-trading-bot/cache")
JDB = ROOT / "data" / "trade_journal.db"

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
# (truncated MT5 comments map to base names)
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


def is_safety_layer(reason: str) -> bool:
    if not reason:
        return False
    if reason.startswith("[sl ") or reason.startswith("[tp "):
        return False
    base = reason.split(" ")[0]
    return base in SAFETY_LAYER_REASONS or any(
        reason.startswith(k.split(" ")[0]) for k in SAFETY_LAYER_REASONS
    )


def divergence_category(live_reason: str) -> str:
    """Pre-categorize by live exit reason. Refined by replay."""
    if live_reason.startswith("[sl ") or live_reason.startswith("[tp "):
        return "SL_OR_TP"
    norm = normalize_reason(live_reason)
    return SAFETY_LAYER_REASONS.get(norm, "OTHER")


# ─── cache loader ─────────────────────────────────────────────────────
_CACHE_DF: dict = {}
_CACHE_IND: dict = {}


def get_h1_df(symbol: str) -> pd.DataFrame | None:
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
    # Ensure timezone-aware UTC
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
    """Replay one live trade against BT logic.

    Returns dict with keys: bt_exit_price, bt_exit_reason, bt_pnl_dollar,
    bt_peak_r, bt_bars_to_exit, divergence_dollar, replay_status.
    """
    symbol = t["symbol"]
    df = get_h1_df(symbol)
    if df is None or len(df) < 100:
        return {"replay_status": "NO_CACHE"}

    entry_ts = pd.Timestamp(t["timestamp"])
    if entry_ts.tz is None:
        entry_ts = entry_ts.tz_localize("UTC")
    # Normalize to ns precision to match cache (which may use s precision)
    try:
        entry_ts = entry_ts.as_unit("ns")
    except Exception:
        pass
    last_bar = df["time"].iloc[-1]
    if entry_ts > last_bar:
        return {"replay_status": "DATA_GAP_BEFORE_EXIT"}

    # Find bar at-or-just-before entry (build numpy array to handle unit mismatch)
    times = df["time"].values.astype("datetime64[ns]")
    idx = np.searchsorted(times, np.datetime64(entry_ts.to_numpy(), "ns"), side="right") - 1
    if idx < 30:
        return {"replay_status": "WARMUP_NOT_READY"}
    bar_i = int(idx)
    # cap end window — replay up to last available bar
    n = len(df)

    # Compute indicators (cached per-symbol)
    ind = get_ind(symbol, df)
    if ind is None:
        return {"replay_status": "INDICATOR_NONE"}

    atr = float(ind["at"][bar_i])
    if not np.isfinite(atr) or atr <= 0:
        return {"replay_status": "BAD_ATR"}

    # SL — same chain as BT: SL_OVERRIDE_REGIME → SL_OVERRIDE → default
    bbw_val = float(ind["bbw"][bar_i]) if not np.isnan(ind["bbw"][bar_i]) else 0.02
    adx_val = float(ind["adx"][bar_i]) if not np.isnan(ind["adx"][bar_i]) else 20
    regime = get_regime(bbw_val, adx_val)
    sl_mult = (SL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
               or SL_OVERRIDE.get(symbol)
               or DEFAULT_PARAMS["sl_atr_mult"])
    sl_dist = atr * sl_mult

    # Direction
    direction = 1 if t["direction"] == "LONG" else -1

    # Entry price — use ACTUAL live entry price for fair compare (avoid
    # spread/slippage modelling here; we're auditing trail not entry).
    entry_price = float(t["entry_price"] or 0)
    if entry_price <= 0:
        # MT5 deal-history sync occasionally returns 0 for closing-deal
        # entry_price. Skip — can't fairly replay.
        return {"replay_status": "ZERO_ENTRY_PRICE"}

    # Trail steps — same chain as BT
    trail_steps = (TRAIL_OVERRIDE_REGIME.get(symbol, {}).get(regime)
                   or TRAIL_OVERRIDE.get(symbol)
                   or DEFAULT_PARAMS["trail"])

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)

    # Replay starts at next bar (BT convention: signal at bar i → entry at i+1)
    start_i = bar_i + 1
    if start_i >= n - 1:
        return {"replay_status": "ENTRY_AT_LAST_BAR"}

    try:
        exit_price, exit_bar, exit_reason, peak_r = simulate_trail(
            entry_price, sl_dist, direction, h, l, c,
            start_i, n,
            spread=0.0,  # we're using live entry; no spread overlay
            trail_steps=trail_steps,
            ratchet_1r=DEFAULT_PARAMS.get("ratchet_1r", 0.3),
            ratchet_2r=DEFAULT_PARAMS.get("ratchet_2r", 0.7),
        )
    except Exception as e:
        return {"replay_status": f"TRAIL_ERR:{e}"}

    # BT pnl in R-multiples. R-space is the only fair compare because
    # live `pnl` includes broker friction (commission, swap, spread,
    # slippage) while BT replay is gross. Imply dollar_risk for $
    # divergence is unreliable on tiny PnL trades (~$0.5 div by 0.1R
    # yields fake $5 risk vs actual $0.10).
    pnl_r = ((exit_price - entry_price) * direction) / sl_dist if sl_dist > 0 else 0
    live_pnl = float(t.get("pnl") or 0)
    live_r = float(t.get("r_multiple") or 0)

    # Dollar risk: use live r_multiple basis when sane (0.05 ≤ |r| ≤ 5
    # AND implied risk in [$0.30, $30] range — wider tolerance now that
    # we filter zero-entry). Otherwise fall back to nominal $1500 × risk_pct%.
    dollar_risk = None
    if abs(live_r) >= 0.05 and abs(live_r) <= 5.0 and abs(live_pnl) > 0.01:
        implied = abs(live_pnl / live_r)
        if 0.30 <= implied <= 30.0:
            dollar_risk = implied
    if dollar_risk is None:
        rp = float(t.get("risk_pct") or 0.4)
        dollar_risk = 1500.0 * (rp / 100.0)

    bt_pnl_dollar = dollar_risk * pnl_r
    # Cap divergence in $ at ±$50 per trade (clamp obvious outliers from
    # implied-risk noise); also report R-divergence which has no such issue.
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
    }


# ─── main ─────────────────────────────────────────────────────────────
def main():
    con = sqlite3.connect(JDB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT * FROM trades WHERE source='mt5_deal' ORDER BY timestamp"""
    ).fetchall()
    con.close()
    trades = [dict(r) for r in rows]
    print(f"Loaded {len(trades)} closed live trades.")

    # Replay
    by_category = defaultdict(lambda: {"count": 0, "divergence": 0.0,
                                       "divergence_r": 0.0,
                                       "live_pnl_all": 0.0,    # all trades
                                       "live_pnl": 0.0,        # replayed only
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
        div = r["divergence_dollar"]   # clamped $
        div_r = r["divergence_r"]
        live_r = float(t.get("r_multiple") or 0)
        bt_r = r["bt_pnl_r"]
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
        })

    print(f"Replayed: {n_replayed}, Skipped: {n_skipped}")
    print(f"  Skip reasons: {dict(skip_reasons)}")
    print(f"Live total PnL (all):       ${total_live_pnl:.2f}")
    print(f"BT replay PnL (replayed):   ${total_bt_pnl:.2f}")
    print(f"Total $ divergence (clamped): ${total_divergence:.2f}")
    print(f"Live total R (replayed):  {total_live_r:+.2f}R")
    print(f"BT replay R (replayed):   {total_bt_r:+.2f}R")
    print(f"Total R divergence:       {total_div_r:+.2f}R")

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
    }
    out_json = ROOT / "audit_20260522" / "parity_reaudit_raw.json"
    json.dump(out, open(out_json, "w"), indent=2, default=str)
    print(f"Wrote {out_json}")

    # Save detail rows
    df_det = pd.DataFrame(detail_rows)
    df_det.to_csv(ROOT / "audit_20260522" / "parity_reaudit_detail.csv", index=False)

    return out, by_category, by_symbol, detail_rows


if __name__ == "__main__":
    main()
