"""
Cross-sym verification: apply ALL 4 entry-sweep winners simultaneously
into the live config dicts (SIGNAL_QUALITY_SYMBOL + PULLBACK_CONFIG_PER_SYMBOL),
then re-BT each symbol with all 4 patches live. Compare per-sym combined PnL
vs reported standalone PnL → cross_talk_delta.

flags_for_drop = any sym where combined < standalone * 0.75 (>25% regression).
"""
import sys, json, time, copy
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

DAYS = 90
OUT = ROOT / "backtest" / "results" / "entry_sweep_20260614" / "combined_verify.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Winners exactly as provided by user
WINNERS = [
    {
        "sym": "XAUUSD",
        "min_score": 50,            # 0-100 quality scale (uniform)
        "min_score_scale": "quality",
        "retrace": 0.0,
        "wait": 1,
        "vwap_too_far": None,
        "standalone_pnl": 172.30,
        # XAUUSD sweep used no cost overlays
        "with_slippage": False, "with_commission": False, "with_swap": False,
    },
    {
        "sym": "SPI200.r",
        "min_score": 5.5,           # raw score → quality = 5.5/12*100 = 45.83
        "min_score_scale": "raw",
        "retrace": 0.0,
        "wait": 1,
        "vwap_too_far": -1,
        "standalone_pnl": 391.76,
        "with_slippage": False, "with_commission": False, "with_swap": False,
    },
    {
        "sym": "JPN225ft",
        "min_score": 35,            # 0-100 quality scale (uniform)
        "min_score_scale": "quality",
        "retrace": 0.5,
        "wait": 1,
        "vwap_too_far": None,
        "standalone_pnl": 449.12,
        "with_slippage": False, "with_commission": False, "with_swap": False,
    },
    {
        "sym": "BTCUSD",
        "min_score": 3.81,          # raw → quality = 3.81/12*100 = 31.75
        "min_score_scale": "raw",
        "retrace": 0.0,
        "wait": 1,
        "vwap_too_far": -1,
        "standalone_pnl": 4954.59,
        # BTCUSD sweep ran with slippage+commission+swap = True
        "with_slippage": True, "with_commission": True, "with_swap": True,
    },
]


def to_quality(min_score, scale):
    if scale == "quality":
        q = float(min_score)
    else:  # raw
        q = float(min_score) / 12.0 * 100.0
    return {"trending": q, "ranging": q, "volatile": q, "low_vol": q}


# ---------------------------------------------------------------------------
# 1) Snapshot current config dicts so we can restore at the end.
# ---------------------------------------------------------------------------
ORIG_SQS = copy.deepcopy(getattr(config, "SIGNAL_QUALITY_SYMBOL", {}))
ORIG_PBC = copy.deepcopy(getattr(config, "PULLBACK_CONFIG_PER_SYMBOL", {}))


# ---------------------------------------------------------------------------
# 2) Patch ALL 4 winners into the live config dicts simultaneously.
#    This is the "cross-sym" state — every symbol's BT will see every other's
#    overrides active in the shared config dicts.
# ---------------------------------------------------------------------------
def apply_all_winners():
    for w in WINNERS:
        sym = w["sym"]
        q_dict = to_quality(w["min_score"], w["min_score_scale"])
        config.SIGNAL_QUALITY_SYMBOL[sym] = dict(q_dict)
        config.PULLBACK_CONFIG_PER_SYMBOL[sym] = {
            "retrace": float(w["retrace"]),
            "wait": int(w["wait"]),
        }


def restore_orig():
    config.SIGNAL_QUALITY_SYMBOL.clear()
    config.SIGNAL_QUALITY_SYMBOL.update(ORIG_SQS)
    config.PULLBACK_CONFIG_PER_SYMBOL.clear()
    config.PULLBACK_CONFIG_PER_SYMBOL.update(ORIG_PBC)


# Optional: preload ML meta-models for all 4 winners (matches live + matches
# each per-sym standalone sweep that loaded its own model).
_MM = None
try:
    from models.signal_model import SignalModel
    _MM = SignalModel()
    for w in WINNERS:
        try:
            _MM.load(w["sym"])
        except Exception:
            pass
except Exception as e:
    print(f"[ML-GATE] disabled — SignalModel unavailable: {e}")
    _MM = None
print(f"ML meta-model loaded: {_MM is not None}")


def run_bt_for_sym(w):
    sym = w["sym"]
    q_dict = to_quality(w["min_score"], w["min_score_scale"])
    # Build params: mirror standalone sweep cost flags + pass min_quality so
    # any internal default lookup matches the SIGNAL_QUALITY_SYMBOL patch.
    params = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": bool(w["with_slippage"]),
        "with_commission": bool(w["with_commission"]),
        "with_swap": bool(w["with_swap"]),
        "min_quality": dict(q_dict),
        "pullback_atr_retrace": float(w["retrace"]),
        "pullback_max_wait": int(w["wait"]),
    }
    if _MM is not None:
        params["_meta_model"] = _MM
    t0 = time.time()
    try:
        r = backtest_symbol(sym, days=DAYS, params=params, verbose=False)
    except Exception as e:
        print(f"  [{sym}] BT ERROR: {e}")
        return None
    dt = time.time() - t0
    if r is None:
        return None
    return {
        "pnl":     float(r.get("pnl", 0.0)),
        "pf":      float(r.get("pf", 0.0)),
        "trades":  int(r.get("trades", 0)),
        "wr":      float(r.get("wr", 0.0)),
        "dd":      float(r.get("dd", 0.0)),
        "seconds": round(dt, 1),
    }


def fmt(r):
    if r is None:
        return "FAIL"
    return (f"pnl={r['pnl']:>+10.2f} pf={r['pf']:>5.2f} n={r['trades']:>3} "
            f"wr={r['wr']:.1%} dd={r['dd']:.1%} ({r['seconds']}s)")


# ===========================================================================
# RUN
# ===========================================================================
print("=" * 70)
print(f"  COMBINED VERIFY — {len(WINNERS)} winners, {DAYS}d")
print("=" * 70)

apply_all_winners()
print("\nPatched config dicts:")
for w in WINNERS:
    print(f"  {w['sym']:12s} SIGNAL_QUALITY_SYMBOL={config.SIGNAL_QUALITY_SYMBOL[w['sym']]}"
          f"  PULLBACK={config.PULLBACK_CONFIG_PER_SYMBOL[w['sym']]}")

per_sym = []
total_pnl = 0.0
total_pf_num = 0.0   # sum of gross wins
total_pf_den = 0.0   # sum of gross losses (abs)
total_trades = 0
session_t0 = time.time()

try:
    for w in WINNERS:
        sym = w["sym"]
        print(f"\n--- {sym} ---")
        r = run_bt_for_sym(w)
        if r is None:
            standalone = w["standalone_pnl"]
            per_sym.append({
                "symbol":        sym,
                "standalone_pnl": float(standalone),
                "combined_pnl":   0.0,
                "cross_talk_delta": -float(standalone),
                "combined_pf":   0.0,
                "trades":        0,
                "error":         "BT failed",
            })
            continue
        print(f"  {fmt(r)}")
        delta = r["pnl"] - w["standalone_pnl"]
        per_sym.append({
            "symbol":        sym,
            "standalone_pnl": float(w["standalone_pnl"]),
            "combined_pnl":   r["pnl"],
            "cross_talk_delta": round(delta, 2),
            "combined_pf":   r["pf"],
            "combined_wr":   r["wr"],
            "combined_dd":   r["dd"],
            "trades":        r["trades"],
            "pct_change":    (round(100.0 * delta / w["standalone_pnl"], 2)
                              if w["standalone_pnl"] else None),
        })
        total_pnl   += r["pnl"]
        total_trades += r["trades"]
        # For combined PF: weighted via per-sym PF * trades is too rough.
        # Best-effort: derive gross wins/losses from PF + PnL.
        # PF = win_$/loss_$;  PnL = win_$ - loss_$
        # → loss_$ = PnL / (PF - 1) when PF > 1; win_$ = PnL + loss_$
        try:
            if r["pf"] > 1.0 and r["pnl"] > 0:
                loss_d = r["pnl"] / (r["pf"] - 1.0)
                win_d  = r["pnl"] + loss_d
                total_pf_num += win_d
                total_pf_den += loss_d
            elif r["pf"] < 1.0 and r["pnl"] < 0:
                loss_d = r["pnl"] / (r["pf"] - 1.0)
                win_d  = r["pnl"] + loss_d
                total_pf_num += max(0.0, win_d)
                total_pf_den += max(0.0, loss_d)
            elif r["pf"] >= 999.0:
                total_pf_num += max(0.0, r["pnl"])
        except Exception:
            pass

finally:
    restore_orig()
    print("\n[restored original config dicts]")

combined_bt_pf = (total_pf_num / total_pf_den) if total_pf_den > 0 else float("inf")

# Flags: drop any sym where combined < standalone * 0.75 (>25% regression)
flags = []
for row in per_sym:
    s = row["standalone_pnl"]
    c = row["combined_pnl"]
    if s > 0 and c < s * 0.75:
        flags.append(row["symbol"])
    elif s <= 0 and c <= s - 25.0:  # negative-standalone safety net
        flags.append(row["symbol"])

print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)
print(f"  Combined total PnL:    ${total_pnl:.2f}")
print(f"  Combined total trades: {total_trades}")
print(f"  Combined total PF:     {combined_bt_pf:.2f}")
print("\n  Per-sym breakdown:")
print(f"  {'sym':12s} {'standalone':>11s} {'combined':>11s} {'delta':>11s} {'pct':>8s} {'pf':>6s} {'n':>5s}")
for row in per_sym:
    s = row["standalone_pnl"]
    c = row["combined_pnl"]
    d = row["cross_talk_delta"]
    pct = row.get("pct_change")
    pct_str = f"{pct:+.1f}%" if pct is not None else "—"
    pf = row.get("combined_pf", 0.0)
    n  = row.get("trades", 0)
    print(f"  {row['symbol']:12s} {s:>11.2f} {c:>11.2f} {d:>+11.2f} {pct_str:>8s} {pf:>6.2f} {n:>5d}")
print(f"\n  flags_for_drop: {flags}")
print(f"  elapsed: {time.time() - session_t0:.1f}s")

result = {
    "days":              DAYS,
    "applied_count":     len(WINNERS),
    "applied_winners":   [w["sym"] for w in WINNERS],
    "combined_bt_pnl":   round(total_pnl, 2),
    "combined_bt_pf":    round(combined_bt_pf, 4) if combined_bt_pf != float("inf") else None,
    "combined_total_trades": total_trades,
    "per_sym_breakdown": per_sym,
    "flags_for_drop":    flags,
    "elapsed_s":         round(time.time() - session_t0, 1),
}
OUT.write_text(json.dumps(result, indent=2))
print(f"\nWritten: {OUT}")
