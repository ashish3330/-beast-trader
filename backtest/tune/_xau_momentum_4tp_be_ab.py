#!/usr/bin/env python3 -B
"""XAUUSD MOMENTUM: 4-TP ladder + BE-after-TP2 vs 3-TP + BE-after-TP1.

Faithful COPY of the live momentum sub-ladder + BE-move exit (executor.open_trade
3-sub path + _move_remaining_to_be), driven by the EXACT momentum entry set that
v5_backtest.backtest_symbol produces for XAUUSD (same signal/gates/ML, DIR_BIAS=1).

Method
------
1. Monkeypatch bt.simulate_trail to CAPTURE every momentum entry (entry price,
   sl_dist, direction, start bar) while v5 runs normally. This yields the proven
   momentum entry set with EXACT sl_dist geometry.
2. Re-simulate the exits with our own sub-ladder model (shared stop, fixed-R TP
   legs, BE+0.2R move after N legs bank) — the SAME entry set for every config
   (controlled A/B; only the exit ladder varies).
3. Model live lot sizing on a $5000 demo: total_lot = risk$/(sl_dist*contract),
   each leg TRUNCATED to lot_step then FLOORED to min_lot (== executor L1317-1318).
   R is measured vs INTENDED risk so forced min-lot over-sizing shows up as
   amplified drawdown (honest — does not normalise the over-risk away).
4. Spread charged per leg round-trip (proportional to leg volume) — churn cost.
5. WF discipline: full sample + first/second half + 3 thirds. A variant only
   "wins" if it holds across segments, not just full-sample.

Run: python3 -B backtest/tune/_xau_momentum_4tp_be_ab.py
"""
import sys
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa
import config  # noqa
import auto_tuned  # noqa
from backtest import v5_backtest as bt
from models.signal_model import SignalModel

SYMBOL = "XAUUSD"
DAYS = 1095

# XAUUSD contract specs (from cache/symbol_meta.json)
POINT = 0.01
CONTRACT = 100.0          # usd per 1.0 price move per 1.0 lot = contract_size*point/point = 100
SPREAD_PRICE = 0.36       # round-trip spread in price
MIN_LOT = 0.01
LOT_STEP = 0.01
VOL_MAX = 100.0
USD_PER_PRICE_PER_LOT = CONTRACT  # $100 per $1 gold move per lot

START_EQUITY = 5000.0
RISK_PCT = 0.5            # live XAUUSD SYMBOL_RISK_PCT_OVERRIDE_AUTO (capped by MAX_RISK 0.5)
HORIZON = 500            # bars, matches simulate_trail cap

# ── Load ML meta-model once (mirrors live + deepest sweep) ──
META_MODEL = SignalModel()
try:
    META_MODEL.load(SYMBOL)
except Exception as e:
    print(f"  [ML] load failed: {e}")


# =============================================================================
# 1. Capture the momentum entry set
# =============================================================================
_CAP = {"entries": [], "h": None, "l": None, "c": None, "n": None}


def _capturing_simulate_trail(entry, sl_dist, direction, highs, lows, closes,
                              start_i, end_i, spread, trail_steps, **kw):
    _CAP["entries"].append((float(entry), float(sl_dist), int(direction), int(start_i)))
    if _CAP["h"] is None:
        _CAP["h"] = highs
        _CAP["l"] = lows
        _CAP["c"] = closes
        _CAP["n"] = int(end_i)
    # return the real trail exit so v5 completes normally (cooldowns etc.)
    return _ORIG_SIM(entry, sl_dist, direction, highs, lows, closes,
                     start_i, end_i, spread, trail_steps, **kw)


_ORIG_SIM = bt.simulate_trail


def capture_entries():
    _CAP["entries"] = []
    _CAP["h"] = _CAP["l"] = _CAP["c"] = None
    bt.simulate_trail = _capturing_simulate_trail
    try:
        bt.backtest_symbol(SYMBOL, DAYS, {"_meta_model": META_MODEL}, verbose=False)
    finally:
        bt.simulate_trail = _ORIG_SIM
    # order chronologically by start bar (already in order, but be safe)
    _CAP["entries"].sort(key=lambda e: e[3])
    return list(_CAP["entries"])


# =============================================================================
# 2. Sub-ladder + BE-move exit simulator (faithful to executor)
# =============================================================================
def simulate_ladder(entry, sl_dist, direction, start_i, h, l, c, n,
                    legs_R, be_after):
    """Return list of leg exit R-multiples (price-space, one per leg) in leg order.

    Shared stop starts at -1R. After `be_after` legs have banked, stop -> +0.2R
    (BE+20%) applied from the NEXT bar (mirrors executor._move_remaining_to_be,
    which fires on the cycle after a TP leg closes). Conservative intrabar order:
    stop checked before TP.
    """
    nlegs = len(legs_R)
    exits = [None] * nlegs
    closed = 0
    stop_r = -1.0
    end = min(n, start_i + HORIZON)
    last_bar = end - 1
    for j in range(start_i, end):
        stop_price = entry + stop_r * sl_dist * direction
        if direction == 1:
            if l[j] <= stop_price:
                for k in range(nlegs):
                    if exits[k] is None:
                        exits[k] = stop_r
                return exits
            hi, lo = h[j], l[j]
        else:
            if h[j] >= stop_price:
                for k in range(nlegs):
                    if exits[k] is None:
                        exits[k] = stop_r
                return exits
            hi, lo = h[j], l[j]
        # TP fills this bar (ascending R order)
        for k in range(nlegs):
            if exits[k] is not None:
                continue
            tp_price = entry + legs_R[k] * sl_dist * direction
            hit = (hi >= tp_price) if direction == 1 else (lo <= tp_price)
            if hit:
                exits[k] = legs_R[k]
                closed += 1
        if closed >= nlegs:
            return exits
        # BE move takes effect next bar
        if closed >= be_after and stop_r < 0.2:
            stop_r = 0.2
    # timeout: remaining legs exit at last close
    close_r = (c[last_bar] - entry) * direction / sl_dist if sl_dist > 0 else 0.0
    for k in range(nlegs):
        if exits[k] is None:
            exits[k] = close_r
    return exits


def leg_lots(total_lot, weights):
    """Live lot split: truncate each leg to step, floor to min_lot (executor L1317-18)."""
    out = []
    for w in weights:
        raw = total_lot * w
        v = int(raw / LOT_STEP) * LOT_STEP           # truncate (int())
        v = round(v, 2)
        v = max(MIN_LOT, min(VOL_MAX, v))            # floor to min lot
        out.append(round(v, 2))
    return out


def trade_net_R(entry, sl_dist, direction, start_i, h, l, c, n,
                weights, legs_R, be_after, risk_usd):
    """Net R (vs INTENDED risk) for one trade under a ladder config, with live
    per-leg min-lot flooring + per-leg round-trip spread."""
    # $ risk per 1.0 lot = sl_dist(price) * USD_PER_PRICE_PER_LOT ($100/$1move/lot)
    total_lot_ideal = risk_usd / (sl_dist * USD_PER_PRICE_PER_LOT) if sl_dist > 0 else 0.0
    lots = leg_lots(total_lot_ideal, weights)
    exit_Rs = simulate_ladder(entry, sl_dist, direction, start_i, h, l, c, n, legs_R, be_after)
    pnl_usd = 0.0
    for k in range(len(weights)):
        # price move for the leg = exitR * sl_dist ; minus round-trip spread
        gross_price = exit_Rs[k] * sl_dist
        net_price = gross_price - SPREAD_PRICE
        pnl_usd += net_price * USD_PER_PRICE_PER_LOT * lots[k]
    return pnl_usd / risk_usd, sum(lots), total_lot_ideal


# =============================================================================
# 3. Metrics + WF segmentation
# =============================================================================
def metrics(net_Rs):
    if not net_Rs:
        return dict(trades=0, expectancy_R=0, total_R=0, PF=0, max_dd_R=0, win_rate=0)
    arr = np.array(net_Rs, dtype=float)
    pos = arr[arr > 0].sum()
    neg = -arr[arr < 0].sum()
    pf = (pos / neg) if neg > 0 else float("inf")
    # drawdown on cumulative R curve
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    dd = (peak - np.concatenate([[0.0], cum])).max()
    return dict(
        trades=int(len(arr)),
        expectancy_R=round(float(arr.mean()), 4),
        total_R=round(float(arr.sum()), 3),
        PF=round(float(pf), 3) if pf != float("inf") else 999.0,
        max_dd_R=round(float(dd), 3),
        win_rate=round(float((arr > 0).mean() * 100), 1),
    )


def segments(net_Rs):
    N = len(net_Rs)
    segs = {"full": net_Rs}
    h = N // 2
    segs["half1"] = net_Rs[:h]
    segs["half2"] = net_Rs[h:]
    t = N // 3
    segs["third1"] = net_Rs[:t]
    segs["third2"] = net_Rs[t:2 * t]
    segs["third3"] = net_Rs[2 * t:]
    # 60/40
    s = int(N * 0.6)
    segs["train60"] = net_Rs[:s]
    segs["test40"] = net_Rs[s:]
    return {k: metrics(v) for k, v in segs.items()}


# =============================================================================
# Config catalogue
# =============================================================================
CONFIGS = {
    "BASELINE_3TP_BE1":  dict(weights=[.50, .30, .20],       legs_R=[2, 3, 5],       be_after=1),
    # one-variable BE test on the CURRENT 3-leg ladder
    "3TP_BE2":           dict(weights=[.50, .30, .20],       legs_R=[2, 3, 5],       be_after=2),
    # 4-leg variants
    "4TP_A_be2":         dict(weights=[.40, .30, .20, .10],  legs_R=[2, 3, 5, 8],    be_after=2),
    "4TP_B_be2":         dict(weights=[.40, .30, .20, .10],  legs_R=[1.5, 3, 5, 8],  be_after=2),
    "4TP_A_be1":         dict(weights=[.40, .30, .20, .10],  legs_R=[2, 3, 5, 8],    be_after=1),
    "4TP_B_be1":         dict(weights=[.40, .30, .20, .10],  legs_R=[1.5, 3, 5, 8],  be_after=1),
    "4TP_D_be2":         dict(weights=[.35, .30, .20, .15],  legs_R=[2, 3, 5, 8],    be_after=2),
    "4TP_E_be2":         dict(weights=[.40, .30, .20, .10],  legs_R=[2, 3, 4, 6],    be_after=2),
    "4TP_F_be2":         dict(weights=[.30, .30, .25, .15],  legs_R=[1.5, 2.5, 4, 6],be_after=2),
}


def run_config(entries, h, l, c, n, cfg, risk_usd):
    net_Rs, lots_actual, lots_ideal = [], [], []
    for (entry, sl_dist, direction, start_i) in entries:
        r, la, li = trade_net_R(entry, sl_dist, direction, start_i, h, l, c, n,
                                cfg["weights"], cfg["legs_R"], cfg["be_after"], risk_usd)
        net_Rs.append(r)
        lots_actual.append(la)
        lots_ideal.append(li)
    return net_Rs, lots_actual, lots_ideal


def main():
    print("=" * 90)
    print(f"  XAUUSD MOMENTUM  4-TP + BE-after-TP2  vs  3-TP + BE-after-TP1")
    print(f"  window={DAYS}d  equity=${START_EQUITY:.0f}  risk={RISK_PCT}%  spread={SPREAD_PRICE}")
    print("=" * 90)

    entries = capture_entries()
    h, l, c, n = _CAP["h"], _CAP["l"], _CAP["c"], _CAP["n"]
    print(f"\n  Captured {len(entries)} momentum entries (DIR_BIAS long-only from config).")
    if not entries:
        print("  NO ENTRIES — abort.")
        return

    risk_usd = START_EQUITY * RISK_PCT / 100.0  # $25 base

    # quick lot-regime diagnostic on baseline weights
    _, la, li = run_config(entries, h, l, c, n, CONFIGS["BASELINE_3TP_BE1"], risk_usd)
    print(f"  Ideal total lot: min={min(li):.4f} med={np.median(li):.4f} max={max(li):.4f}")
    print(f"  3-leg actual lot after min-lot flooring: min={min(la):.2f} med={np.median(la):.2f} max={max(la):.2f}")

    results = {}
    for name, cfg in CONFIGS.items():
        net_Rs, la, li = run_config(entries, h, l, c, n, cfg, risk_usd)
        segm = segments(net_Rs)
        over = np.mean([a / i if i > 0 else 1 for a, i in zip(la, li)])
        results[name] = {"seg": segm, "over_risk_mult": round(float(over), 3),
                         "med_actual_lot": round(float(np.median(la)), 2)}

    # ---- print table ----
    hdr = f"{'config':18s} {'trades':>6s} {'exp_R':>7s} {'tot_R':>8s} {'PF':>6s} {'ddR':>7s} {'WR%':>5s} {'ovRisk':>6s}"
    print("\n[FULL SAMPLE]\n" + hdr)
    for name, r in results.items():
        m = r["seg"]["full"]
        print(f"{name:18s} {m['trades']:6d} {m['expectancy_R']:7.4f} {m['total_R']:8.3f} "
              f"{m['PF']:6.3f} {m['max_dd_R']:7.3f} {m['win_rate']:5.1f} {r['over_risk_mult']:6.3f}")

    # WF per-segment for the key comparisons
    key = ["BASELINE_3TP_BE1", "3TP_BE2", "4TP_A_be2", "4TP_B_be2", "4TP_A_be1"]
    for seg in ["half1", "half2", "third1", "third2", "third3", "train60", "test40"]:
        print(f"\n[{seg}]\n" + hdr)
        for name in key:
            m = results[name]["seg"][seg]
            print(f"{name:18s} {m['trades']:6d} {m['expectancy_R']:7.4f} {m['total_R']:8.3f} "
                  f"{m['PF']:6.3f} {m['max_dd_R']:7.3f} {m['win_rate']:5.1f}")

    # ---- risk sensitivity (min-lot binding) ----
    print("\n[RISK SENSITIVITY — full-sample expectancy_R / total_R / ddR]")
    print(f"{'risk$':>7s} " + " ".join(f"{k:>16s}" for k in ["BASELINE_3TP_BE1", "3TP_BE2", "4TP_A_be2"]))
    for rusd in [15.0, 25.0, 50.0, 100.0]:
        cells = []
        for name in ["BASELINE_3TP_BE1", "3TP_BE2", "4TP_A_be2"]:
            net_Rs, la, li = run_config(entries, h, l, c, n, CONFIGS[name], rusd)
            m = metrics(net_Rs)
            cells.append(f"{m['expectancy_R']:+.3f}/{m['total_R']:+.1f}/{m['max_dd_R']:.1f}")
        print(f"{rusd:7.0f} " + " ".join(f"{x:>16s}" for x in cells))

    # ---- decision logic ----
    base = results["BASELINE_3TP_BE1"]["seg"]
    def wins_over_baseline(name):
        ok = True
        for seg in ["full", "half1", "half2"]:
            a = results[name]["seg"][seg]
            b = base[seg]
            # must not degrade expectancy in either half and improve full
            if a["expectancy_R"] < b["expectancy_R"] - 1e-9:
                ok = False
        full_better = results[name]["seg"]["full"]["expectancy_R"] > base["full"]["expectancy_R"] + 1e-9
        dd_ok = results[name]["seg"]["full"]["max_dd_R"] <= base["full"]["max_dd_R"] * 1.15 + 1e-9
        return ok and full_better and dd_ok

    print("\n[DECISION]")
    # helps_be_after_tp2 : compare 3TP_BE2 vs baseline (one-variable)
    be2 = results["3TP_BE2"]["seg"]
    print(f"  3TP BE-after-TP2 vs baseline: exp_R {be2['full']['expectancy_R']} vs {base['full']['expectancy_R']}, "
          f"ddR {be2['full']['max_dd_R']} vs {base['full']['max_dd_R']}, "
          f"half1 {be2['half1']['expectancy_R']} vs {base['half1']['expectancy_R']}, "
          f"half2 {be2['half2']['expectancy_R']} vs {base['half2']['expectancy_R']}")
    best_4 = None
    best_exp = base["full"]["expectancy_R"]
    for name in CONFIGS:
        if name.startswith("4TP") and results[name]["seg"]["full"]["expectancy_R"] > best_exp:
            best_exp = results[name]["seg"]["full"]["expectancy_R"]
            best_4 = name
    print(f"  best 4TP by full exp_R: {best_4} ({best_exp})")
    for name in CONFIGS:
        if name.startswith("4TP"):
            print(f"    {name}: robust_win={wins_over_baseline(name)}")

    return results, base


if __name__ == "__main__":
    main()
