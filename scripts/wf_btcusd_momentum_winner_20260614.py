"""
5-fold walk-forward STRICT validation for BTCUSD with frozen winner_config:

    SL_ATR_MULT = 1.5
    SIGNAL_QUALITY (trending/ranging/volatile/low_vol) = 30 each
    DIR_BIAS = SHORT
    TRAIL_PROFILE = _AGGR_LOCK
    PULLBACK_RETRACE = 0
    PULLBACK_MAX_WAIT = 1

5 non-overlapping 90-day folds (or 180d if 3yr cache available):
    Fold 1: -540 .. -450 d   (or -900..-720 if 3yr)
    Fold 2: -450 .. -360 d
    Fold 3: -360 .. -270 d
    Fold 4: -270 .. -180 d
    Fold 5: -180 .. -90  d
    Holdout: -90 .. 0 (NOT used here — reserved for live)

SHIP gate (STRICT):
    folds_pf_ok   >= 4 (PF >= 1.5 in 4 of 5 folds)
    AND folds_pnl_ok >= 4
    AND no single fold loses more than -2 x baseline_per_fold_pnl
NULL otherwise.
"""
import sys, json, time, copy, pickle
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import auto_tuned  # noqa: E402
from backtest import v5_backtest as v5  # noqa: E402
from backtest.v5_backtest import (  # noqa: E402
    ALL_SYMBOLS, CACHE, backtest_symbol, _live_to_bt_trail,
)

SYMBOL = "BTCUSD"

WINNER_CONFIG = {
    "SL_ATR_MULT": 1.5,
    "SIGNAL_QUALITY": {
        "trending": 30, "ranging": 30, "volatile": 30, "low_vol": 30,
    },
    "DIR_BIAS": "SHORT",                # only SHORT signals allowed
    "TRAIL_PROFILE": "_AGGR_LOCK",
    "PULLBACK_RETRACE": 0,
    "PULLBACK_MAX_WAIT": 1,
}

# DIR_BIAS code map for bt.DIR_BIAS (1=LONG, -1=SHORT, 0=BOTH)
DIR_MAP = {"LONG": 1, "SHORT": -1, "BOTH": 0}

# -----------------------------------------------------------------------------
# Determine if 3yr cache (>=900d) is available -> 180d folds, else 90d
# -----------------------------------------------------------------------------
meta = ALL_SYMBOLS[SYMBOL]
cache_path = CACHE / meta["cache"]
df_all = pickle.load(open(cache_path, "rb"))
if not pd.api.types.is_datetime64_any_dtype(df_all["time"]):
    df_all["time"] = pd.to_datetime(df_all["time"], unit="s", utc=True)
cache_days = (df_all["time"].max() - df_all["time"].min()).days
print(f"{SYMBOL} cache range: {df_all['time'].min()} -> {df_all['time'].max()}  "
      f"({len(df_all)} bars, ~{cache_days} days)")

if cache_days >= 900 + 90:  # leave at least the -90..0 holdout
    FOLD_SIZE = 180
    FOLDS = [
        {"idx": 1, "start_off": 900, "end_off": 720},
        {"idx": 2, "start_off": 720, "end_off": 540},
        {"idx": 3, "start_off": 540, "end_off": 360},
        {"idx": 4, "start_off": 360, "end_off": 180},
        {"idx": 5, "start_off": 180, "end_off":  90},
    ]
    print(f"Using 180d folds (cache has {cache_days}d).")
else:
    FOLD_SIZE = 90
    FOLDS = [
        {"idx": 1, "start_off": 540, "end_off": 450},
        {"idx": 2, "start_off": 450, "end_off": 360},
        {"idx": 3, "start_off": 360, "end_off": 270},
        {"idx": 4, "start_off": 270, "end_off": 180},
        {"idx": 5, "start_off": 180, "end_off":  90},
    ]
    print(f"Using 90d folds (cache has {cache_days}d, "
          f"need >= {540 + 0} for full 5x90d coverage).")

RES_DIR = ROOT / "backtest" / "results" / "wf_btcusd_momentum_20260614"
RES_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RES_DIR / "wf_BTCUSD_momentum_winner.json"


# -----------------------------------------------------------------------------
# Window monkey-patch (copies wf_xauusd_winner_20260614 pattern)
# -----------------------------------------------------------------------------
_FOLD_START_OFFSET = None
_FOLD_END_OFFSET = None


def _windowed_load_data(symbol, days=90):
    sm_meta = ALL_SYMBOLS[symbol]
    path = CACHE / sm_meta["cache"]
    if not path.exists():
        return None
    df = pickle.load(open(path, "rb"))
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    end_of_data = df["time"].max()
    start = end_of_data - pd.Timedelta(days=_FOLD_START_OFFSET)
    end = end_of_data - pd.Timedelta(days=_FOLD_END_OFFSET)
    return df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)


v5.load_data = _windowed_load_data


def set_fold(start_off, end_off):
    global _FOLD_START_OFFSET, _FOLD_END_OFFSET
    _FOLD_START_OFFSET = float(start_off)
    _FOLD_END_OFFSET = float(end_off)


# -----------------------------------------------------------------------------
# Patches for the frozen winner_config (snapshot & restore)
# -----------------------------------------------------------------------------
ORIG_SQS = copy.deepcopy(getattr(config, "SIGNAL_QUALITY_SYMBOL", {}))
ORIG_SL = v5.SL_OVERRIDE.get(SYMBOL, "__MISSING__")
ORIG_DB = v5.DIR_BIAS.get(SYMBOL, "__MISSING__")
ORIG_TR = v5.TRAIL_OVERRIDE.get(SYMBOL, "__MISSING__") if hasattr(v5, "TRAIL_OVERRIDE") else "__MISSING__"


def apply_winner():
    # Signal quality dict (4 regimes)
    config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = dict(WINNER_CONFIG["SIGNAL_QUALITY"])
    # SL ATR mult
    v5.SL_OVERRIDE[SYMBOL] = float(WINNER_CONFIG["SL_ATR_MULT"])
    # DIR bias
    v5.DIR_BIAS[SYMBOL] = DIR_MAP[WINNER_CONFIG["DIR_BIAS"]]
    # Trail profile -- convert live tuples to bt list via _live_to_bt_trail
    profile_name = WINNER_CONFIG["TRAIL_PROFILE"]
    live_steps = getattr(auto_tuned, profile_name, None)
    if live_steps is not None and hasattr(v5, "TRAIL_OVERRIDE"):
        v5.TRAIL_OVERRIDE[SYMBOL] = _live_to_bt_trail(live_steps)


def restore():
    config.SIGNAL_QUALITY_SYMBOL.clear()
    config.SIGNAL_QUALITY_SYMBOL.update(ORIG_SQS)
    if ORIG_SL == "__MISSING__":
        v5.SL_OVERRIDE.pop(SYMBOL, None)
    else:
        v5.SL_OVERRIDE[SYMBOL] = ORIG_SL
    if ORIG_DB == "__MISSING__":
        v5.DIR_BIAS.pop(SYMBOL, None)
    else:
        v5.DIR_BIAS[SYMBOL] = ORIG_DB
    if hasattr(v5, "TRAIL_OVERRIDE"):
        if ORIG_TR == "__MISSING__":
            v5.TRAIL_OVERRIDE.pop(SYMBOL, None)
        else:
            v5.TRAIL_OVERRIDE[SYMBOL] = ORIG_TR


# Optional ML model preload (mirrors live)
try:
    from models.signal_model import SignalModel
    _MM = SignalModel()
    _MM.load(SYMBOL)
    if not _MM.has_model(SYMBOL):
        _MM = None
except Exception:
    _MM = None
print(f"ML meta-model loaded: {_MM is not None}")


# -----------------------------------------------------------------------------
# Run one fold
# -----------------------------------------------------------------------------
def run_fold(fold):
    set_fold(fold["start_off"], fold["end_off"])
    apply_winner()
    params = {
        "pullback_atr_retrace": float(WINNER_CONFIG["PULLBACK_RETRACE"]),
        "pullback_max_wait": int(WINNER_CONFIG["PULLBACK_MAX_WAIT"]),
    }
    if _MM is not None:
        params["_meta_model"] = _MM
    t0 = time.time()
    # Sanity: does the windowed slice even contain bars?
    end_of_data = df_all["time"].max()
    start_ts = end_of_data - pd.Timedelta(days=fold["start_off"])
    end_ts = end_of_data - pd.Timedelta(days=fold["end_off"])
    bar_count = int(((df_all["time"] >= start_ts) & (df_all["time"] <= end_ts)).sum())
    if bar_count == 0:
        return {
            "fold": fold["idx"], "window": [fold["start_off"], fold["end_off"]],
            "bars": 0,
            "error": "insufficient-cache-data",
            "pnl": 0.0, "pf": 0.0, "trades": 0, "wr": 0.0, "dd": 0.0,
            "seconds": round(time.time() - t0, 1),
        }
    try:
        r = backtest_symbol(SYMBOL, days=FOLD_SIZE, params=params, verbose=False)
    except Exception as e:
        return {
            "fold": fold["idx"], "window": [fold["start_off"], fold["end_off"]],
            "bars": bar_count, "error": f"backtest-exception: {e}",
            "pnl": 0.0, "pf": 0.0, "trades": 0, "wr": 0.0, "dd": 0.0,
            "seconds": round(time.time() - t0, 1),
        }
    if r is None:
        return {
            "fold": fold["idx"], "window": [fold["start_off"], fold["end_off"]],
            "bars": bar_count, "error": "no-result",
            "pnl": 0.0, "pf": 0.0, "trades": 0, "wr": 0.0, "dd": 0.0,
            "seconds": round(time.time() - t0, 1),
        }
    return {
        "fold": fold["idx"],
        "window": [fold["start_off"], fold["end_off"]],
        "bars": bar_count,
        "pnl": float(r.get("pnl", 0.0)),
        "pf": float(r.get("pf", 0.0)),
        "trades": int(r.get("trades", 0)),
        "wr": float(r.get("wr", 0.0)),
        "dd": float(r.get("dd", 0.0)),
        "seconds": round(time.time() - t0, 1),
    }


print(f"\n=== 5-fold WF STRICT for {SYMBOL} momentum ===")
print(f"Winner config: {json.dumps(WINNER_CONFIG)}")
print(f"Fold size: {FOLD_SIZE}d\n")

results = []
session_t0 = time.time()
for f in FOLDS:
    print(f"  Fold {f['idx']} window=[-{f['start_off']}d, -{f['end_off']}d) ...")
    r = run_fold(f)
    results.append(r)
    if r.get("error"):
        print(f"    SKIP/ERROR: {r['error']}  bars={r.get('bars',0)}")
    else:
        print(f"    pnl={r['pnl']:+.2f}  pf={r['pf']:.2f}  n={r['trades']}  "
              f"wr={r['wr']:.1f}%  dd={r['dd']:.1f}%  bars={r['bars']}  ({r['seconds']}s)")

restore()

# -----------------------------------------------------------------------------
# STRICT SHIP gate
# -----------------------------------------------------------------------------
valid = [r for r in results if not r.get("error")]
folds_run = len(valid)
folds_pf_ok = sum(1 for r in valid if r["pf"] >= 1.5)
folds_pnl_ok = sum(1 for r in valid if r["pnl"] > 0)

# Baseline per-fold pnl = mean of positive folds (or None if too few)
positive_pnls = [r["pnl"] for r in valid if r["pnl"] > 0]
if positive_pnls:
    baseline = sum(positive_pnls) / len(positive_pnls)
else:
    baseline = 0.0
worst_loss = min((r["pnl"] for r in valid), default=0.0)

dd_violation = baseline > 0 and worst_loss < -2.0 * baseline

if folds_run < 5:
    decision = "NULL"
    reason = (f"insufficient folds run ({folds_run}/5) — "
              f"BTCUSD cache only {cache_days}d; STRICT 5x{FOLD_SIZE}d requires "
              f"{5*FOLD_SIZE}+ days. SHIP gate cannot be evaluated.")
elif folds_pf_ok < 4 or folds_pnl_ok < 4 or dd_violation:
    decision = "NULL"
    parts = []
    if folds_pf_ok < 4:
        parts.append(f"folds_pf_ok={folds_pf_ok}/5 (need >=4)")
    if folds_pnl_ok < 4:
        parts.append(f"folds_pnl_ok={folds_pnl_ok}/5 (need >=4)")
    if dd_violation:
        parts.append(f"single-fold loss {worst_loss:.2f} exceeds -2x baseline {-2.0*baseline:.2f}")
    reason = "STRICT gate failed: " + "; ".join(parts)
else:
    decision = "SHIP"
    reason = (f"folds_pf_ok={folds_pf_ok}/5 AND folds_pnl_ok={folds_pnl_ok}/5 "
              f"AND no fold worse than -2x baseline ({-2.0*baseline:.2f}).")

print(f"\n=== DECISION: {decision} ===")
print(f"  {reason}")

payload = {
    "symbol": SYMBOL,
    "strategy": "momentum",
    "winner_config": WINNER_CONFIG,
    "fold_size_days": FOLD_SIZE,
    "cache_days": cache_days,
    "folds": results,
    "folds_run": folds_run,
    "folds_pf_ok": folds_pf_ok,
    "folds_pnl_ok": folds_pnl_ok,
    "baseline_per_fold_pnl": baseline,
    "worst_fold_loss": worst_loss,
    "decision": decision,
    "decision_reason": reason,
    "elapsed_s": round(time.time() - session_t0, 1),
}
json.dump(payload, open(OUT_FILE, "w"), indent=2, default=str)
print(f"\nWritten: {OUT_FILE}")
