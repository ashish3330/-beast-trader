#!/usr/bin/env python3 -B
"""
Deepest-momentum coordinate-descent sweep for BTCUSD.

3yr (1095d) H1 window requested; cache is 20d so BT window collapses to that.
Baseline reproduces task spec: trades=32 pnl=513.83 pf=3.97 dd=5.1.

Axes (in priority order):
  SL_ATR_MULT -> SIGNAL_QUALITY -> DIR_BIAS -> TRAIL_PROFILE
  -> PULLBACK_RETRACE -> ELC_TRIGGER_R

Output: backtest/results/deepest_sweep_momentum/BTCUSD.json
"""
import sys, json, time, copy
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from backtest import v5_backtest as bt  # noqa: E402
from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

SYMBOL = "BTCUSD"
DAYS = 1095
OUT_DIR = ROOT / "backtest" / "results" / "deepest_sweep_momentum"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / f"{SYMBOL}.json"

# ── Snapshots of module-level overrides we mutate during sweep ────────────
ORIG_SL = bt.SL_OVERRIDE.get(SYMBOL)
ORIG_SQS = copy.deepcopy(config.SIGNAL_QUALITY_SYMBOL.get(SYMBOL))
ORIG_DIR = bt.DIR_BIAS.get(SYMBOL)
ORIG_TRAIL = bt.TRAIL_OVERRIDE.get(SYMBOL)
ORIG_TOXIC = bt.TOXIC_HOURS_PER_SYMBOL.get(SYMBOL)

# Trail profile constants (BT format: [(R, param, type), ...])
def _live_to_bt(steps):
    return [(r, p, t) for (r, t, p) in steps]

TRAIL_LIVE = {
    "_TIGHT_LOCK":   [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_WIDE_RUNNER":  [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7), (1.5, "lock", 0.5), (0.7, "be", 0.0)],
    "_RANGE_TIGHT":  [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6), (0.3, "be", 0.0)],
    "_TREND_LOOSE":  [(15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5), (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "_AGGR_LOCK":    [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8), (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "_RUNNER_NO_BE": [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5), (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)],
    "_FOREX_LOOSE":  [(6.0, "trail", 0.4), (3.0, "trail", 0.6), (2.0, "lock", 0.8), (1.5, "lock", 0.6), (1.0, "lock", 0.3)],
}
TRAIL_BT = {k: _live_to_bt(v) for k, v in TRAIL_LIVE.items()}

# ── Meta-model ────────────────────────────────────────────────────────────
try:
    from models.signal_model import SignalModel
    _MM = SignalModel(); _MM.load(SYMBOL)
    if not _MM.has_model(SYMBOL):
        _MM = None
except Exception:
    _MM = None

# Coordinate-descent state holds current best per-axis.
state = {
    "sl_mult":   ORIG_SL,                                    # 1.25 (BTCUSD)
    "quality":   ORIG_SQS or {"trending": 60, "ranging": 65, "volatile": 55, "low_vol": 60},
    "dir_bias":  None,                                        # None = BOTH
    "trail":     None,                                        # None = current TRAIL_OVERRIDE
    "retrace":   0.2,
    "wait":      1,
}


def _apply(state, extra_params=None):
    """Apply state to module dicts & build params for backtest call."""
    # SL
    if state["sl_mult"] is None:
        bt.SL_OVERRIDE.pop(SYMBOL, None)
    else:
        bt.SL_OVERRIDE[SYMBOL] = float(state["sl_mult"])
    # quality (read live at call-time)
    if state["quality"] is None:
        config.SIGNAL_QUALITY_SYMBOL.pop(SYMBOL, None)
    else:
        config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = dict(state["quality"])
    params = {
        "pullback_atr_retrace": float(state["retrace"]),
        "pullback_max_wait": int(state["wait"]),
        "audit_fix_gates": True,
    }
    if _MM is not None:
        params["_meta_model"] = _MM
    # dir bias
    if state["dir_bias"] in ("LONG", "SHORT", "BOTH"):
        params["force_direction"] = state["dir_bias"]
    elif state["dir_bias"] == "FLAT":
        # no-trade — use a 0-quality-impossible threshold via filtering out
        # both directions. Emulate via tight quality (we will count this
        # separately rather than pretend it traded).
        params["force_direction"] = "FLAT"  # not recognized → falls through
    # trail
    if state["trail"] is not None:
        params["force_trail"] = TRAIL_BT[state["trail"]]
    if extra_params:
        params.update(extra_params)
    return params


def _restore_module_dicts():
    if ORIG_SL is None:
        bt.SL_OVERRIDE.pop(SYMBOL, None)
    else:
        bt.SL_OVERRIDE[SYMBOL] = ORIG_SL
    if ORIG_SQS is None:
        config.SIGNAL_QUALITY_SYMBOL.pop(SYMBOL, None)
    else:
        config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = ORIG_SQS


def run_bt(state, extra_params=None):
    params = _apply(state, extra_params)
    t0 = time.time()
    try:
        r = backtest_symbol(SYMBOL, days=DAYS, params=params, verbose=False)
    except Exception as e:
        return {"trades": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0,
                "error": str(e), "seconds": round(time.time() - t0, 1)}
    dt = time.time() - t0
    if r is None:
        return {"trades": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0,
                "seconds": round(dt, 1)}
    return {
        "trades": int(r.get("trades", 0)),
        "pnl":    float(r.get("pnl", 0)),
        "pf":     float(r.get("pf", 0)),
        "wr":     float(r.get("wr", 0)),
        "dd":     float(r.get("dd", 0)),
        "seconds": round(dt, 1),
    }


# ── BASELINE ─────────────────────────────────────────────────────────────
print(f"== BASELINE ({SYMBOL}, {DAYS}d) ==")
baseline = run_bt(state)
print(f"  baseline: trades={baseline['trades']} pnl={baseline['pnl']:.2f} "
      f"pf={baseline['pf']:.2f} wr={baseline['wr']:.1f} dd={baseline['dd']:.1f}")
baseline_pnl = baseline["pnl"]
baseline_pf = baseline["pf"]
baseline_trades = baseline["trades"]
baseline_dd = baseline["dd"]
MIN_TRADES = max(15, int(0.7 * baseline_trades))
print(f"  MIN_TRADES gate = {MIN_TRADES}")

# ── helpers ──
session_t0 = time.time()
SESSION_BUDGET_S = 40 * 60   # 40 min hard cap (45 min ceiling minus margin)


def time_left():
    return SESSION_BUDGET_S - (time.time() - session_t0)


def fmt(r):
    if r is None:
        return "None"
    return (f"n={r['trades']:>3} pnl={r['pnl']:>+9.2f} pf={r['pf']:.2f} "
            f"wr={r['wr']:.1f} dd={r['dd']:.1f} ({r['seconds']}s)")


def pick_best_pf(tested, prev_best_value, min_trades=MIN_TRADES):
    """Return (best_value, best_pnl, best_pf) using PF (NOT $)."""
    eligible = [t for t in tested if t["trades"] >= min_trades and "error" not in t]
    if not eligible:
        return prev_best_value, baseline_pnl, baseline_pf
    best = max(eligible, key=lambda t: t["pf"])
    return best["_value"], best["pnl"], best["pf"]


# ────────────────────────────────────────────────────────────────────────
# Axis 1: SL_ATR_MULT
# ────────────────────────────────────────────────────────────────────────
print("\n== AXIS 1: SL_ATR_MULT ==")
SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
tested = []
for v in SL_GRID:
    if time_left() < 60:
        print("  time budget exhausted")
        break
    state["sl_mult"] = v
    r = run_bt(state)
    r["_value"] = str(v)
    tested.append(r)
    print(f"  SL={v}: {fmt(r)}")
best_val, best_pnl, best_pf = pick_best_pf(tested, str(state["sl_mult"]))
state["sl_mult"] = float(best_val)
print(f"  -> winner SL_ATR_MULT={best_val} (pf={best_pf:.2f})")
sl_axis = {
    "axis": "SL_ATR_MULT",
    "tested": [{"value": t["_value"], "trades": t["trades"], "pnl": t["pnl"],
                "pf": t["pf"], "max_dd": t["dd"]} for t in tested],
    "best_value": str(best_val),
    "best_lift_pnl": round(best_pnl - baseline_pnl, 2),
    "best_lift_pf":  round(best_pf - baseline_pf, 3),
}

# ────────────────────────────────────────────────────────────────────────
# Axis 2: SIGNAL_QUALITY
# ────────────────────────────────────────────────────────────────────────
print("\n== AXIS 2: SIGNAL_QUALITY ==")
Q_GRID = [25, 30, 35, 40, 45, 50, 55, 60, 65]
tested = []
for v in Q_GRID:
    if time_left() < 60:
        print("  time budget exhausted")
        break
    qd = {"trending": v, "ranging": v, "volatile": v, "low_vol": v}
    state["quality"] = qd
    r = run_bt(state)
    r["_value"] = str(v)
    tested.append(r)
    print(f"  Q={v}: {fmt(r)}")
best_val, best_pnl, best_pf = pick_best_pf(tested, "60")
state["quality"] = {"trending": int(best_val), "ranging": int(best_val),
                    "volatile": int(best_val), "low_vol": int(best_val)}
print(f"  -> winner SIGNAL_QUALITY={best_val} (pf={best_pf:.2f})")
q_axis = {
    "axis": "SIGNAL_QUALITY",
    "tested": [{"value": t["_value"], "trades": t["trades"], "pnl": t["pnl"],
                "pf": t["pf"], "max_dd": t["dd"]} for t in tested],
    "best_value": str(best_val),
    "best_lift_pnl": round(best_pnl - baseline_pnl, 2),
    "best_lift_pf":  round(best_pf - baseline_pf, 3),
}

# ────────────────────────────────────────────────────────────────────────
# Axis 3: DIR_BIAS
# ────────────────────────────────────────────────────────────────────────
print("\n== AXIS 3: DIR_BIAS ==")
DIR_GRID = ["LONG", "SHORT", "BOTH"]   # FLAT = no trade (skipped)
tested = []
for v in DIR_GRID:
    if time_left() < 60:
        print("  time budget exhausted")
        break
    state["dir_bias"] = v
    r = run_bt(state)
    r["_value"] = v
    tested.append(r)
    print(f"  DIR={v}: {fmt(r)}")
best_val, best_pnl, best_pf = pick_best_pf(tested, "BOTH")
state["dir_bias"] = best_val
print(f"  -> winner DIR_BIAS={best_val} (pf={best_pf:.2f})")
dir_axis = {
    "axis": "DIR_BIAS",
    "tested": [{"value": t["_value"], "trades": t["trades"], "pnl": t["pnl"],
                "pf": t["pf"], "max_dd": t["dd"]} for t in tested],
    "best_value": best_val,
    "best_lift_pnl": round(best_pnl - baseline_pnl, 2),
    "best_lift_pf":  round(best_pf - baseline_pf, 3),
}

# ────────────────────────────────────────────────────────────────────────
# Axis 4: TRAIL_PROFILE
# ────────────────────────────────────────────────────────────────────────
print("\n== AXIS 4: TRAIL_PROFILE ==")
TRAIL_GRID = list(TRAIL_BT.keys())
tested = []
for v in TRAIL_GRID:
    if time_left() < 60:
        print("  time budget exhausted")
        break
    state["trail"] = v
    r = run_bt(state)
    r["_value"] = v
    tested.append(r)
    print(f"  TRAIL={v}: {fmt(r)}")
best_val, best_pnl, best_pf = pick_best_pf(tested, "DEFAULT")
state["trail"] = best_val if best_val in TRAIL_BT else None
print(f"  -> winner TRAIL_PROFILE={best_val} (pf={best_pf:.2f})")
trail_axis = {
    "axis": "TRAIL_PROFILE",
    "tested": [{"value": t["_value"], "trades": t["trades"], "pnl": t["pnl"],
                "pf": t["pf"], "max_dd": t["dd"]} for t in tested],
    "best_value": best_val,
    "best_lift_pnl": round(best_pnl - baseline_pnl, 2),
    "best_lift_pf":  round(best_pf - baseline_pf, 3),
}

# ────────────────────────────────────────────────────────────────────────
# Axis 5: PULLBACK_RETRACE
# ────────────────────────────────────────────────────────────────────────
print("\n== AXIS 5: PULLBACK_RETRACE ==")
PB_GRID = [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
tested = []
for v in PB_GRID:
    if time_left() < 60:
        print("  time budget exhausted")
        break
    state["retrace"] = v
    r = run_bt(state)
    r["_value"] = str(v)
    tested.append(r)
    print(f"  PB={v}: {fmt(r)}")
best_val, best_pnl, best_pf = pick_best_pf(tested, str(state["retrace"]))
state["retrace"] = float(best_val)
print(f"  -> winner PULLBACK_RETRACE={best_val} (pf={best_pf:.2f})")
pb_axis = {
    "axis": "PULLBACK_RETRACE",
    "tested": [{"value": t["_value"], "trades": t["trades"], "pnl": t["pnl"],
                "pf": t["pf"], "max_dd": t["dd"]} for t in tested],
    "best_value": str(best_val),
    "best_lift_pnl": round(best_pnl - baseline_pnl, 2),
    "best_lift_pf":  round(best_pf - baseline_pf, 3),
}

# ────────────────────────────────────────────────────────────────────────
# Axis 6: ELC_TRIGGER_R (no BT support — informational only)
# ────────────────────────────────────────────────────────────────────────
# v5_backtest does not simulate EarlyLossCut (no MOMENTUM_ELC_PER_SYMBOL hook
# in the simulate_trail path). Sweep would be no-ops. Record as untestable.
print("\n== AXIS 6: ELC_TRIGGER_R ==")
print("  ELC not simulated in v5_backtest — skipping (informational axis).")
elc_axis = {
    "axis": "ELC_TRIGGER_R",
    "tested": [],
    "best_value": "current=-1.0",
    "_note": "v5_backtest does not implement early-loss-cut path; axis skipped",
    "best_lift_pnl": 0.0,
    "best_lift_pf":  0.0,
}

# ────────────────────────────────────────────────────────────────────────
# Final combined BT
# ────────────────────────────────────────────────────────────────────────
print("\n== FINAL combined BT ==")
final = run_bt(state)
print(f"  combined: {fmt(final)}")

axis_results = [sl_axis, q_axis, dir_axis, trail_axis, pb_axis, elc_axis]
winner_config = {
    "SL_ATR_MULT":      state["sl_mult"],
    "SIGNAL_QUALITY":   state["quality"],
    "DIR_BIAS":         state["dir_bias"] or "BOTH",
    "TRAIL_PROFILE":    state["trail"] or "DEFAULT",
    "PULLBACK_RETRACE": state["retrace"],
}
winner_pnl = final["pnl"]
winner_pf  = final["pf"]
winner_dd  = final["dd"]
winner_trades = final["trades"]

# Decision logic
decision = "NULL"
reason = ""
threshold_pf = max(1.5, baseline_pf * 1.05)
if (winner_pf > threshold_pf and winner_pnl > baseline_pnl
        and winner_dd <= baseline_dd * 1.30 and winner_trades >= MIN_TRADES):
    decision = "ADVANCE_TO_WF"
    reason = (f"PF {winner_pf:.2f} > threshold {threshold_pf:.2f}, "
              f"PnL ${winner_pnl:.2f} > baseline ${baseline_pnl:.2f}, "
              f"DD {winner_dd:.1f}% <= 1.3*baseline ({baseline_dd*1.3:.1f}%)")
else:
    failed = []
    if winner_pf <= threshold_pf:
        failed.append(f"PF {winner_pf:.2f} <= threshold {threshold_pf:.2f}")
    if winner_pnl <= baseline_pnl:
        failed.append(f"PnL ${winner_pnl:.2f} <= baseline ${baseline_pnl:.2f}")
    if winner_dd > baseline_dd * 1.30:
        failed.append(f"DD {winner_dd:.1f}% > 1.3*baseline ({baseline_dd*1.3:.1f}%)")
    if winner_trades < MIN_TRADES:
        failed.append(f"trades {winner_trades} < MIN_TRADES {MIN_TRADES}")
    reason = "; ".join(failed) or "no improvement"

result = {
    "strategy": "momentum",
    "symbol": SYMBOL,
    "window_days_requested": 1095,
    "window_days_effective_note": "BTCUSD cache holds ~20d H1; BT operates on full cache",
    "baseline": {
        "trades": baseline_trades, "pnl": baseline_pnl, "pf": baseline_pf,
        "max_dd": baseline_dd,
    },
    "axis_results": axis_results,
    "winner_config": winner_config,
    "winner": {
        "trades": winner_trades, "pnl": winner_pnl, "pf": winner_pf,
        "max_dd": winner_dd,
    },
    "decision": decision,
    "decision_reason": reason,
    "elapsed_total_s": round(time.time() - session_t0, 1),
}

OUT_FILE.write_text(json.dumps(result, indent=2))
print(f"\nWROTE {OUT_FILE}")
print(f"decision={decision} reason={reason}")

_restore_module_dicts()
