"""
3-fold walk-forward validation for XAUUSD with the winner params
from entry_sweep_xauusd_20260614:

    MIN_SCORE = 50         (applied as SIGNAL_QUALITY_SYMBOL uniform dict)
    PULLBACK_ATR_RETRACE = 0
    PULLBACK_MAX_WAIT_BARS = 1

Three non-overlapping 60-day folds, offsets from end-of-data:
    Fold 1: -180 -> -120   (oldest)
    Fold 2: -120 ->  -60
    Fold 3:  -60 ->   0    (newest)

Uses the time-window monkey-patch trick from scripts/walk_forward.py
(swaps backtest.v5_backtest.load_data with a windowed slicer).

SHIP if 2/3 or 3/3 folds have pnl > 0. NULL otherwise.
"""
import sys, json, time, copy, pickle
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from backtest import v5_backtest as v5  # noqa: E402
from backtest.v5_backtest import ALL_SYMBOLS, CACHE, backtest_symbol  # noqa: E402

SYMBOL = "XAUUSD"
WINNER_MIN_SCORE = 50
WINNER_RETRACE = 0.0
WINNER_WAIT = 1

FOLDS = [
    {"idx": 1, "start_off": 180, "end_off": 120},
    {"idx": 2, "start_off": 120, "end_off": 60},
    {"idx": 3, "start_off": 60,  "end_off": 0},
]

RES_DIR = ROOT / "backtest" / "results" / "entry_sweep_20260614"
RES_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RES_DIR / f"wf_{SYMBOL}_winner.json"

# -----------------------------------------------------------------------------
# Window monkey-patch (copies scripts/walk_forward.py pattern)
# -----------------------------------------------------------------------------
_FOLD_START_OFFSET = None
_FOLD_END_OFFSET = None


def _windowed_load_data(symbol, days=90):
    meta = ALL_SYMBOLS[symbol]
    path = CACHE / meta["cache"]
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
# Quality monkey-patch (matches entry_sweep_xauusd_20260614 mechanism)
# -----------------------------------------------------------------------------
ORIG_SQS = copy.deepcopy(getattr(config, "SIGNAL_QUALITY_SYMBOL", {}))


def q_uniform(v):
    return {"trending": int(v), "ranging": int(v),
            "volatile": int(v), "low_vol": int(v)}


def patch_quality(q_dict):
    config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = dict(q_dict)


def restore_quality():
    config.SIGNAL_QUALITY_SYMBOL.clear()
    config.SIGNAL_QUALITY_SYMBOL.update(ORIG_SQS)


# Optional ML model preload (parity w/ entry sweep — keeps comparison fair)
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
# Run a single fold
# -----------------------------------------------------------------------------
def run_fold(fold):
    set_fold(fold["start_off"], fold["end_off"])
    patch_quality(q_uniform(WINNER_MIN_SCORE))
    params = {
        "pullback_atr_retrace": float(WINNER_RETRACE),
        "pullback_max_wait": int(WINNER_WAIT),
    }
    if _MM is not None:
        params["_meta_model"] = _MM
    t0 = time.time()
    try:
        # `days` arg is ignored inside our windowed load_data — pass 60 for
        # readability.
        r = backtest_symbol(SYMBOL, days=60, params=params, verbose=False)
    except Exception as e:
        return {"fold": fold["idx"], "window": [fold["start_off"], fold["end_off"]],
                "error": str(e), "seconds": round(time.time() - t0, 1)}
    if r is None:
        return {"fold": fold["idx"], "window": [fold["start_off"], fold["end_off"]],
                "error": "no-result", "seconds": round(time.time() - t0, 1)}
    return {
        "fold": fold["idx"],
        "window": [fold["start_off"], fold["end_off"]],
        "pnl": float(r.get("pnl", 0.0)),
        "pf": float(r.get("pf", 0.0)),
        "trades": int(r.get("trades", 0)),
        "wr": float(r.get("wr", 0.0)),
        "dd": float(r.get("dd", 0.0)),
        "seconds": round(time.time() - t0, 1),
    }


# -----------------------------------------------------------------------------
# Sanity: print data-range info
# -----------------------------------------------------------------------------
meta = ALL_SYMBOLS[SYMBOL]
path = CACHE / meta["cache"]
df_all = pickle.load(open(path, "rb"))
if not pd.api.types.is_datetime64_any_dtype(df_all["time"]):
    df_all["time"] = pd.to_datetime(df_all["time"], unit="s", utc=True)
print(f"XAUUSD cache range: {df_all['time'].min()} -> {df_all['time'].max()}  ({len(df_all)} bars)")

print("\n=== 3-fold WF for XAUUSD ===")
print(f"Winner params: MIN_SCORE={WINNER_MIN_SCORE} "
      f"PULLBACK_ATR_RETRACE={WINNER_RETRACE} PULLBACK_MAX_WAIT_BARS={WINNER_WAIT}\n")

results = []
session_t0 = time.time()
for f in FOLDS:
    print(f"  Fold {f['idx']} window=[-{f['start_off']}d, -{f['end_off']}d) ...")
    r = run_fold(f)
    results.append(r)
    if "error" in r:
        print(f"    ERROR: {r['error']}")
    else:
        print(f"    pnl={r['pnl']:+.2f}  pf={r['pf']:.2f}  n={r['trades']}  "
              f"wr={r['wr']:.1f}%  dd={r['dd']:.1f}%  ({r['seconds']}s)")

restore_quality()

# Decision
positives = [r for r in results if r.get("pnl", 0.0) > 0.0 and "error" not in r]
folds_positive = len(positives)
folds_run = len([r for r in results if "error" not in r])
if folds_positive >= 2:
    wf_decision = "SHIP"
    wf_reason = f"{folds_positive}/3 folds positive (>=2 required)"
else:
    wf_decision = "NULL"
    wf_reason = f"only {folds_positive}/3 folds positive (need >=2)"

print(f"\n=== DECISION: {wf_decision} ===")
print(f"  {wf_reason}")

payload = {
    "symbol": SYMBOL,
    "winner_params": {
        "MIN_SCORE": WINNER_MIN_SCORE,
        "PULLBACK_ATR_RETRACE": WINNER_RETRACE,
        "PULLBACK_MAX_WAIT_BARS": WINNER_WAIT,
    },
    "fold_size_days": 60,
    "folds": results,
    "folds_run": folds_run,
    "folds_positive": folds_positive,
    "wf_decision": wf_decision,
    "wf_reason": wf_reason,
    "elapsed_s": round(time.time() - session_t0, 1),
}
json.dump(payload, open(OUT_FILE, "w"), indent=2, default=str)
print(f"\nWritten: {OUT_FILE}")
