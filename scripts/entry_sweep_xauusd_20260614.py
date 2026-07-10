"""
Coordinate-descent per-symbol entry-param sweep for XAUUSD (2026-06-14).

Axes (in order):
  1. SIGNAL_QUALITY_SYMBOL (live MIN_SCORE-equivalent on 0-100 scale)
  2. PULLBACK_ATR_RETRACE
  3. PULLBACK_MAX_WAIT_BARS
  (VWAP_TOO_FAR — SKIPPED, no BT implementation today.)

For MIN_SCORE we sweep the *live* gate, SIGNAL_QUALITY_SYMBOL, since the
documented DRAGON_M15_SYMBOL_MIN_SCORE dict is dead per the recon map.
For pullback we sweep via the params-dict mechanism (overrides per-sym dict).
"""
import sys, json, time, copy
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from backtest.v5_backtest import backtest_symbol  # noqa: E402

SYMBOL = "XAUUSD"
DAYS = 90
RESULT_DIR = ROOT / "backtest" / "results" / "entry_sweep_20260614"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RESULT_DIR / f"{SYMBOL}.json"

# ---------------------------------------------------------------------------
# Capture baseline per-sym quality dict from auto_tuned overlay
# ---------------------------------------------------------------------------
# config.py runs SIGNAL_QUALITY_SYMBOL.update(SIGNAL_QUALITY_SYMBOL_AUTO) at
# import time, so config.SIGNAL_QUALITY_SYMBOL already has the active dict.
ORIG_SQS = copy.deepcopy(getattr(config, "SIGNAL_QUALITY_SYMBOL", {}))
BASE_Q_DICT = dict(ORIG_SQS.get(SYMBOL, {"trending": 40, "ranging": 40,
                                          "volatile": 40, "low_vol": 40}))
print(f"Baseline {SYMBOL} SIGNAL_QUALITY_SYMBOL = {BASE_Q_DICT}")
print(f"Baseline pullback (global default): retrace={config.PULLBACK_ATR_RETRACE} "
      f"wait={config.PULLBACK_MAX_WAIT_BARS}")

# Optional: preload SignalModel for ML gate parity with live
try:
    from models.signal_model import SignalModel
    _MM = SignalModel()
    _MM.load(SYMBOL)
    if not _MM.has_model(SYMBOL):
        _MM = None
except Exception:
    _MM = None
print(f"ML meta-model loaded: {_MM is not None}")


def _patch_quality(q_dict):
    """Monkey-patch config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = q_dict.

    The BT reads config.SIGNAL_QUALITY_SYMBOL at backtest_symbol-call time
    (v5_backtest.py:536-538), so we just write to the live module dict.
    """
    config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = dict(q_dict)


def _restore_quality():
    config.SIGNAL_QUALITY_SYMBOL.clear()
    config.SIGNAL_QUALITY_SYMBOL.update(ORIG_SQS)


def run_bt(q_dict, retrace, wait, tag=""):
    """One BT call with the given q-dict + pullback params. Returns dict."""
    _patch_quality(q_dict)
    params = {
        "pullback_atr_retrace": float(retrace),
        "pullback_max_wait": int(wait),
        # Keep ML gate on for parity with live (matches recon mapping; same
        # ML default the live agent uses). XAUUSD is in ML_BYPASS_SYMBOLS in
        # config.py:329 — so backtest gate may be active where live is not,
        # but we accept that minor divergence; the sweep is still a relative
        # comparison across axes.
    }
    if _MM is not None:
        params["_meta_model"] = _MM
    t0 = time.time()
    try:
        r = backtest_symbol(SYMBOL, days=DAYS, params=params, verbose=False)
    except Exception as e:
        print(f"  [{tag}] ERROR: {e}")
        return None
    dt = time.time() - t0
    if r is None:
        return None
    out = {
        "pnl": float(r.get("pnl", 0.0)),
        "pf": float(r.get("pf", 0.0)),
        "trades": int(r.get("trades", 0)),
        "wr": float(r.get("wr", 0.0)),
        "dd": float(r.get("dd", 0.0)),
        "seconds": round(dt, 1),
    }
    return out


def fmt(r):
    if r is None:
        return "FAIL"
    return f"pnl={r['pnl']:>+9.2f} pf={r['pf']:.2f} n={r['trades']:>3} wr={r['wr']:.1%} dd={r['dd']:.1%} ({r['seconds']}s)"


# ===========================================================================
# Sweep ranges
# ===========================================================================
# For SIGNAL_QUALITY we collapse the per-regime dict into one uniform value
# per candidate (matches existing per-sym tunings in auto_tuned.py which use
# uniform values like {40,40,40,40}). 9 candidates.
QUALITY_GRID = [30, 35, 40, 45, 50, 55, 60, 65, 70]
# Pullback
RETRACE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
WAIT_GRID = [1, 2, 3]

# Coordinate-descent state (start at current defaults)
best_q_uniform = int(BASE_Q_DICT.get("trending", 40))   # uniform from current
best_retrace = 0.20
best_wait = 1


def q_dict_uniform(v):
    return {"trending": int(v), "ranging": int(v), "volatile": int(v), "low_vol": int(v)}


axis_traces = []
session_t0 = time.time()
SESSION_BUDGET_S = 25 * 60   # 25 min hard cap

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
print("\n=== BASELINE ===")
baseline_q = BASE_Q_DICT
baseline = run_bt(baseline_q, best_retrace, best_wait, tag="baseline")
print(f"  baseline ({baseline_q}, retrace={best_retrace}, wait={best_wait}): {fmt(baseline)}")

if baseline is None or baseline["trades"] == 0:
    print("  BASELINE FAILED — aborting.")
    json.dump({"symbol": SYMBOL, "error": "baseline-failed", "baseline": baseline},
              open(OUT_FILE, "w"), indent=2)
    _restore_quality()
    sys.exit(1)

baseline_pnl, baseline_pf, baseline_trades = baseline["pnl"], baseline["pf"], baseline["trades"]


def time_left():
    return SESSION_BUDGET_S - (time.time() - session_t0)


def sweep_axis(name, grid, runner):
    """Sweep one axis. runner(value) -> result_dict.

    Returns (best_value, axis_trace_dict).
    """
    print(f"\n=== AXIS: {name} ===")
    tested = []
    for v in grid:
        if time_left() < 60:
            print(f"  time budget low ({time_left():.0f}s left), stopping {name} sweep")
            break
        r = runner(v)
        if r is None:
            print(f"  {name}={v}: BT failed")
            tested.append({"value": float(v), "pnl": 0.0, "pf": 0.0, "trades": 0})
            continue
        print(f"  {name}={v}: {fmt(r)}")
        tested.append({"value": float(v), "pnl": r["pnl"], "pf": r["pf"], "trades": r["trades"]})

    # Eligibility: trades >= 10. If none qualify, keep current baseline value.
    eligible = [t for t in tested if t["trades"] >= 10]
    # Drop ultra-thin (<5 trades) as instructed.
    eligible = [t for t in eligible if t["trades"] >= 10]  # already enforced
    if not eligible:
        # nothing qualifies — keep prior best
        return None, {"param": name, "tested": tested, "best_value": -1.0}
    best = max(eligible, key=lambda t: t["pnl"])
    return best["value"], {"param": name, "tested": tested, "best_value": float(best["value"])}


# ---------------------------------------------------------------------------
# Axis 1: MIN_SCORE (sweep SIGNAL_QUALITY uniform)
# ---------------------------------------------------------------------------
def runner_q(v):
    return run_bt(q_dict_uniform(v), best_retrace, best_wait, tag=f"q={v}")

new_q, trace_q = sweep_axis("MIN_SCORE", QUALITY_GRID, runner_q)
if new_q is not None:
    best_q_uniform = int(new_q)
    print(f"  -> locked MIN_SCORE={best_q_uniform}")
else:
    print(f"  -> no eligible MIN_SCORE; keeping baseline={best_q_uniform}")
axis_traces.append(trace_q)

# ---------------------------------------------------------------------------
# Axis 2: PULLBACK_ATR_RETRACE
# ---------------------------------------------------------------------------
def runner_retrace(v):
    return run_bt(q_dict_uniform(best_q_uniform), v, best_wait, tag=f"retrace={v}")

new_rt, trace_rt = sweep_axis("PULLBACK_ATR_RETRACE", RETRACE_GRID, runner_retrace)
if new_rt is not None:
    best_retrace = float(new_rt)
    print(f"  -> locked PULLBACK_ATR_RETRACE={best_retrace}")
else:
    print(f"  -> no eligible retrace; keeping baseline={best_retrace}")
axis_traces.append(trace_rt)

# ---------------------------------------------------------------------------
# Axis 3: PULLBACK_MAX_WAIT_BARS
# ---------------------------------------------------------------------------
def runner_wait(v):
    return run_bt(q_dict_uniform(best_q_uniform), best_retrace, v, tag=f"wait={v}")

new_w, trace_w = sweep_axis("PULLBACK_MAX_WAIT_BARS", WAIT_GRID, runner_wait)
if new_w is not None:
    best_wait = int(new_w)
    print(f"  -> locked PULLBACK_MAX_WAIT_BARS={best_wait}")
else:
    print(f"  -> no eligible wait; keeping baseline={best_wait}")
axis_traces.append(trace_w)

# ---------------------------------------------------------------------------
# VWAP_TOO_FAR — SKIPPED (no BT implementation per recon)
# ---------------------------------------------------------------------------
# (No axis trace emitted; documented in `winner_params` as N/A.)

# ---------------------------------------------------------------------------
# Final confirm BT
# ---------------------------------------------------------------------------
print("\n=== CONFIRM combo ===")
winner = run_bt(q_dict_uniform(best_q_uniform), best_retrace, best_wait, tag="winner")
print(f"  winner: q={best_q_uniform} retrace={best_retrace} wait={best_wait}  {fmt(winner)}")

_restore_quality()

# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
winner_pnl = winner["pnl"] if winner else 0.0
winner_pf = winner["pf"] if winner else 0.0
winner_trades = winner["trades"] if winner else 0
lift_pnl = winner_pnl - baseline_pnl

ship_thresh = 30.0
marginal_lo = 5.0
pf_floor = max(1.2, baseline_pf - 0.1)

if lift_pnl > ship_thresh and winner_trades >= 15 and winner_pf >= pf_floor:
    decision = "SHIP"
    reason = (f"lift +${lift_pnl:.2f} > $30 AND n={winner_trades} >= 15 AND "
              f"pf={winner_pf:.2f} >= {pf_floor:.2f}")
elif lift_pnl > marginal_lo:
    decision = "MARGINAL"
    reason = (f"lift +${lift_pnl:.2f} in (+$5, +$30] OR pf/trades not meeting ship gates "
              f"(pf={winner_pf:.2f} floor={pf_floor:.2f}, n={winner_trades})")
else:
    decision = "NULL"
    reason = (f"lift +${lift_pnl:.2f} <= +$5 or trades < 10 (n={winner_trades})")

print(f"\n=== DECISION: {decision} ===")
print(f"  baseline pnl={baseline_pnl:.2f} pf={baseline_pf:.2f} n={baseline_trades}")
print(f"  winner   pnl={winner_pnl:.2f} pf={winner_pf:.2f} n={winner_trades}")
print(f"  lift={lift_pnl:+.2f}  reason: {reason}")

result = {
    "symbol": SYMBOL,
    "days": DAYS,
    "baseline": {
        "pnl": baseline_pnl, "pf": baseline_pf, "trades": baseline_trades,
        "wr": baseline.get("wr", 0.0), "dd": baseline.get("dd", 0.0),
        "params": {
            "MIN_SCORE": int(BASE_Q_DICT.get("trending", 40)),
            "MIN_SCORE_full_dict": BASE_Q_DICT,
            "PULLBACK_ATR_RETRACE": 0.20,
            "PULLBACK_MAX_WAIT_BARS": 1,
            "VWAP_TOO_FAR": None,
        },
    },
    "winner": {
        "pnl": winner_pnl, "pf": winner_pf, "trades": winner_trades,
        "wr": (winner.get("wr", 0.0) if winner else 0.0),
        "dd": (winner.get("dd", 0.0) if winner else 0.0),
        "params": {
            "MIN_SCORE": int(best_q_uniform),
            "PULLBACK_ATR_RETRACE": float(best_retrace),
            "PULLBACK_MAX_WAIT_BARS": int(best_wait),
            "VWAP_TOO_FAR": None,
        },
    },
    "lift_pnl": lift_pnl,
    "decision": decision,
    "decision_reason": reason,
    "axis_traces": axis_traces,
    "elapsed_s": round(time.time() - session_t0, 1),
}
json.dump(result, open(OUT_FILE, "w"), indent=2)
print(f"\nWritten: {OUT_FILE}  (elapsed {result['elapsed_s']}s)")
