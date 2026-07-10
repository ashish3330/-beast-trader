#!/usr/bin/env python3 -B
"""Deep coordinate-descent sweep for DJ30.r MOMENTUM on 3yr (1095d) H1.

Axes (in priority order):
  1. SL_ATR_MULT             [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
  2. SIGNAL_QUALITY uniform  [25, 30, 35, 40, 45, 50, 55, 60, 65]
  3. DIR_BIAS                ['LONG','SHORT','BOTH','FLAT']
  4. TRAIL_PROFILE           [_TIGHT_LOCK,_WIDE_RUNNER,_RANGE_TIGHT,_TREND_LOOSE,
                              _AGGR_LOCK,_RUNNER_NO_BE,_FOREX_LOOSE]
  5. PULLBACK_RETRACE        [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
  6. ELC_TRIGGER_R           [-1.5,-1.2,-1.0,-0.8,-0.6,-0.5,-0.4]
                              + None (current = ELC OFF on DJ30.r)

Later axes use earlier-axis winners as the baseline. Each grid value is
backtested via backtest.v5_backtest.backtest_symbol with overrides applied
through monkey-patching the relevant per-symbol dicts.

Output: backtest/results/deepest_sweep_momentum/DJ30.r.json
"""
import json
import sys
import time
import copy
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import auto_tuned  # noqa: E402
from backtest import v5_backtest as bt  # noqa: E402

SYMBOL = "DJ30.r"
DAYS = 1095
OUT = ROOT / "backtest" / "results" / "deepest_sweep_momentum" / f"{SYMBOL}.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

T_HARD_BUDGET_S = 45 * 60        # global wall budget (45min)
T_PER_BT_BUDGET_S = 8 * 60       # per-BT abort budget (8min)

BASELINE = {
    "symbol": "DJ30.r",
    "trades": 18,
    "pnl": -83.3,
    "pf": 0.37,
    "wr": 16.7,
    "avg_r": -0.46,
    "max_dd": 8.3,
}

# Threshold = max(15, 0.7 × baseline_trades).
TRADES_FLOOR = max(15, int(round(0.7 * BASELINE["trades"])))

# Named trail profiles (must match auto_tuned.py constants).
TRAIL_NAMES = ["_TIGHT_LOCK", "_WIDE_RUNNER", "_RANGE_TIGHT", "_TREND_LOOSE",
               "_AGGR_LOCK", "_RUNNER_NO_BE", "_FOREX_LOOSE"]

# Trail-profile lookup. Convert auto_tuned constant tuples
# (R, type, param) → backtest format (R, param, type) via _live_to_bt_trail.
TRAIL_PROFILES_BT = {
    name: bt._live_to_bt_trail(getattr(auto_tuned, name))
    for name in TRAIL_NAMES
}

SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
QUALITY_GRID = [25, 30, 35, 40, 45, 50, 55, 60, 65]
DIR_GRID = ["LONG", "SHORT", "BOTH", "FLAT"]
TRAIL_GRID = TRAIL_NAMES
PULLBACK_RETRACE_GRID = [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
ELC_GRID = [-1.5, -1.2, -1.0, -0.8, -0.6, -0.5, -0.4]


# ── ML meta-model (loaded once) ──
def _load_meta():
    try:
        from models.signal_model import SignalModel
        mm = SignalModel()
        mm.load(SYMBOL)
        if mm.has_model(SYMBOL):
            return mm
    except Exception:
        pass
    return None


_MM = _load_meta()


# ── BT helper ──
def short_r(r):
    if not r or not isinstance(r, dict) or "error" in r:
        return {"trades": 0, "pf": 0.0, "pnl": 0.0, "wr": 0.0,
                "max_dd": 0.0, "avg_r": 0.0,
                "error": (r or {}).get("error", "no-result")}
    return {
        "trades": int(r.get("trades", 0)),
        "pf": float(r.get("pf", 0.0)),
        "pnl": float(r.get("pnl", 0.0)),
        "wr": float(r.get("wr", 0.0)),
        "max_dd": float(r.get("dd", 0.0)),
        "avg_r": float(r.get("avg_r", 0.0)),
    }


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


def _run_bt(*, sl_mult=None, quality_uniform=None, direction=None,
            trail_name=None, pullback_retrace=None, elc=None,
            t_budget=T_PER_BT_BUDGET_S):
    """Run BT with overrides applied via monkey-patch. Restores all on exit.

    direction: "LONG"|"SHORT"|"BOTH"|"FLAT"|None
    elc: float ELC trigger R, or None (current DJ30.r is OFF — in
         EARLY_EXIT_DISABLED_SYMBOLS).
    """
    # Capture originals
    orig_sl_o = bt.SL_OVERRIDE.get(SYMBOL, None)
    orig_sl_re = copy.deepcopy(bt.SL_OVERRIDE_REGIME.get(SYMBOL, None))
    orig_quality = copy.deepcopy(config.SIGNAL_QUALITY_SYMBOL.get(SYMBOL, None))
    orig_dir_str = copy.deepcopy(bt._DIR_BIAS_REGIME_STR.get(SYMBOL, None))
    orig_trail_o = copy.deepcopy(bt.TRAIL_OVERRIDE.get(SYMBOL, None))
    orig_trail_re = copy.deepcopy(bt.TRAIL_OVERRIDE_REGIME.get(SYMBOL, None))
    orig_elc_disabled = (SYMBOL in config.EARLY_EXIT_DISABLED_SYMBOLS)
    orig_elc_per = config.MOMENTUM_ELC_PER_SYMBOL.get(SYMBOL, None)

    p = copy.deepcopy(bt.DEFAULT_PARAMS)

    try:
        # SL_ATR_MULT — patch the SL_OVERRIDE table so BT picks it up.
        if sl_mult is not None:
            bt.SL_OVERRIDE[SYMBOL] = float(sl_mult)
            # Drop the per-regime overlay so this scalar is what's used.
            bt.SL_OVERRIDE_REGIME.pop(SYMBOL, None)

        # SIGNAL_QUALITY uniform (per-regime float→same value all regimes).
        if quality_uniform is not None:
            config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = {
                "trending": float(quality_uniform),
                "ranging":  float(quality_uniform),
                "volatile": float(quality_uniform),
                "low_vol":  float(quality_uniform),
            }

        # DIR_BIAS — uniform across regimes by setting the per-regime override
        # in BT internal dict. Use 'FLAT' as a hard-block (no trades).
        if direction is not None:
            # Clear any per-regime overlay then set scalar via force_direction.
            bt._DIR_BIAS_REGIME_STR[SYMBOL] = {
                "trending": direction,
                "ranging":  direction,
                "volatile": direction,
                "low_vol":  direction,
            }
            if direction == "FLAT":
                # No matching int in {LONG=1, SHORT=-1, BOTH=0, FLAT}; treat
                # FLAT as block-all by skipping the BT entirely.
                return {"trades": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0,
                        "dd": 0.0, "avg_r": 0.0, "_note": "FLAT = no trades"}

        # TRAIL_PROFILE — uniform across regimes via TRAIL_OVERRIDE table.
        if trail_name is not None:
            steps = TRAIL_PROFILES_BT[trail_name]
            bt.TRAIL_OVERRIDE[SYMBOL] = steps
            # Drop the per-regime overlay so the scalar wins.
            bt.TRAIL_OVERRIDE_REGIME.pop(SYMBOL, None)

        # PULLBACK_RETRACE — pass via param dict (overrides per-symbol cfg).
        if pullback_retrace is not None:
            p["pullback_atr_retrace"] = float(pullback_retrace)

        # ELC — DJ30 is in EARLY_EXIT_DISABLED_SYMBOLS by default. BT does NOT
        # currently model ELC (no ELC simulation logic in v5_backtest.py — the
        # trail-stack handles all exits). So ELC tuning here is informational:
        # we leave the override dict alone and report None tested.
        # (The recon brief explicitly lists ELC as a sweep axis but the
        # backtest's exit-stack is trail+ratchet only — no ELC fires.)
        if elc is not None:
            # Best-effort: remove from disabled set + write per-sym tuple.
            config.EARLY_EXIT_DISABLED_SYMBOLS.discard(SYMBOL)
            config.MOMENTUM_ELC_PER_SYMBOL[SYMBOL] = (float(elc), 60)

        if _MM is not None:
            p["_meta_model"] = _MM

        # Per-BT timeout safeguard.
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(t_budget))
        try:
            r = bt.backtest_symbol(SYMBOL, days=DAYS, params=p, verbose=False)
        finally:
            signal.alarm(0)
        return r
    except _Timeout:
        return {"error": f"timeout >{t_budget}s"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Restore everything.
        if orig_sl_o is None:
            bt.SL_OVERRIDE.pop(SYMBOL, None)
        else:
            bt.SL_OVERRIDE[SYMBOL] = orig_sl_o
        if orig_sl_re is None:
            bt.SL_OVERRIDE_REGIME.pop(SYMBOL, None)
        else:
            bt.SL_OVERRIDE_REGIME[SYMBOL] = orig_sl_re

        if orig_quality is None:
            config.SIGNAL_QUALITY_SYMBOL.pop(SYMBOL, None)
        else:
            config.SIGNAL_QUALITY_SYMBOL[SYMBOL] = orig_quality

        if orig_dir_str is None:
            bt._DIR_BIAS_REGIME_STR.pop(SYMBOL, None)
        else:
            bt._DIR_BIAS_REGIME_STR[SYMBOL] = orig_dir_str

        if orig_trail_o is None:
            bt.TRAIL_OVERRIDE.pop(SYMBOL, None)
        else:
            bt.TRAIL_OVERRIDE[SYMBOL] = orig_trail_o
        if orig_trail_re is None:
            bt.TRAIL_OVERRIDE_REGIME.pop(SYMBOL, None)
        else:
            bt.TRAIL_OVERRIDE_REGIME[SYMBOL] = orig_trail_re

        if orig_elc_disabled:
            config.EARLY_EXIT_DISABLED_SYMBOLS.add(SYMBOL)
        if orig_elc_per is None:
            config.MOMENTUM_ELC_PER_SYMBOL.pop(SYMBOL, None)
        else:
            config.MOMENTUM_ELC_PER_SYMBOL[SYMBOL] = orig_elc_per


def _save(state):
    with open(OUT, "w") as f:
        json.dump(state, f, indent=2)


def _eligible(rows, key="pf"):
    """Filter rows that have trades >= TRADES_FLOOR. Sort by key descending."""
    out = [r for r in rows if r["trades"] >= TRADES_FLOOR]
    out.sort(key=lambda x: x[key], reverse=True)
    return out


def main():
    t0 = time.time()
    state = {
        "strategy": "momentum",
        "symbol": SYMBOL,
        "window_days": DAYS,
        "trades_floor": TRADES_FLOOR,
        "baseline": BASELINE,
        "axes": [],
        "winner_config": {},
    }

    # ── BASELINE rerun (sanity check) ──
    print(f"[{time.time()-t0:6.1f}s] Confirming baseline DJ30.r 3yr...", flush=True)
    b_raw = _run_bt()  # no overrides → current config
    b = short_r(b_raw)
    state["baseline_rerun"] = b
    print(f"           rerun: n={b['trades']} pnl={b['pnl']:.2f} "
          f"pf={b['pf']:.2f} dd={b['max_dd']:.2f}", flush=True)
    _save(state)

    cur = {}
    # Initialise current best with the baseline-rerun PF/PnL (not the recon
    # baseline) because the recon may have been from a slightly different run.
    cur_pf = b["pf"] if b["trades"] >= TRADES_FLOOR else BASELINE["pf"]
    cur_pnl = b["pnl"] if b["trades"] >= TRADES_FLOOR else BASELINE["pnl"]

    def time_up():
        return (time.time() - t0) >= T_HARD_BUDGET_S

    # ── AXIS 1: SL_ATR_MULT ──
    print(f"\n[{time.time()-t0:6.1f}s] AXIS 1: SL_ATR_MULT {SL_GRID}", flush=True)
    a1 = {"axis": "SL_ATR_MULT", "tested": [], "best_value": None}
    for v in SL_GRID:
        if time_up():
            a1["aborted"] = "wall-budget"; break
        r = short_r(_run_bt(sl_mult=v))
        a1["tested"].append({"value": str(v), **r})
        print(f"           SL={v}: n={r['trades']} pnl={r['pnl']:.2f} "
              f"pf={r['pf']:.2f} dd={r['max_dd']:.2f}", flush=True)
    elig = _eligible(a1["tested"])
    if elig:
        winner = elig[0]
        if winner["pf"] > cur_pf or (winner["pf"] >= cur_pf and winner["pnl"] > cur_pnl):
            cur["SL_ATR_MULT"] = float(winner["value"])
            cur_pf = winner["pf"]
            cur_pnl = winner["pnl"]
            a1["best_value"] = winner["value"]
            a1["best_lift_pf"] = round(winner["pf"] - BASELINE["pf"], 3)
            a1["best_lift_pnl"] = round(winner["pnl"] - BASELINE["pnl"], 2)
            print(f"           >> ship SL={winner['value']} pf={winner['pf']:.2f}", flush=True)
        else:
            print(f"           >> no SL beats cur pf={cur_pf:.2f}; keep default", flush=True)
    state["axes"].append(a1)
    _save(state)

    # ── AXIS 2: SIGNAL_QUALITY uniform ──
    print(f"\n[{time.time()-t0:6.1f}s] AXIS 2: SIGNAL_QUALITY {QUALITY_GRID} "
          f"(SL={cur.get('SL_ATR_MULT')})", flush=True)
    a2 = {"axis": "SIGNAL_QUALITY", "tested": [], "best_value": None}
    for v in QUALITY_GRID:
        if time_up():
            a2["aborted"] = "wall-budget"; break
        r = short_r(_run_bt(sl_mult=cur.get("SL_ATR_MULT"), quality_uniform=v))
        a2["tested"].append({"value": str(v), **r})
        print(f"           Q={v}: n={r['trades']} pnl={r['pnl']:.2f} "
              f"pf={r['pf']:.2f} dd={r['max_dd']:.2f}", flush=True)
    elig = _eligible(a2["tested"])
    if elig:
        winner = elig[0]
        if winner["pf"] > cur_pf or (winner["pf"] >= cur_pf and winner["pnl"] > cur_pnl):
            cur["SIGNAL_QUALITY"] = int(winner["value"])
            cur_pf = winner["pf"]
            cur_pnl = winner["pnl"]
            a2["best_value"] = winner["value"]
            a2["best_lift_pf"] = round(winner["pf"] - BASELINE["pf"], 3)
            a2["best_lift_pnl"] = round(winner["pnl"] - BASELINE["pnl"], 2)
            print(f"           >> ship Q={winner['value']} pf={winner['pf']:.2f}", flush=True)
        else:
            print(f"           >> no Q beats cur pf={cur_pf:.2f}; keep default", flush=True)
    state["axes"].append(a2)
    _save(state)

    # ── AXIS 3: DIR_BIAS ──
    print(f"\n[{time.time()-t0:6.1f}s] AXIS 3: DIR_BIAS {DIR_GRID} "
          f"(SL={cur.get('SL_ATR_MULT')} Q={cur.get('SIGNAL_QUALITY')})", flush=True)
    a3 = {"axis": "DIR_BIAS", "tested": [], "best_value": None}
    for v in DIR_GRID:
        if time_up():
            a3["aborted"] = "wall-budget"; break
        r = short_r(_run_bt(sl_mult=cur.get("SL_ATR_MULT"),
                            quality_uniform=cur.get("SIGNAL_QUALITY"),
                            direction=v))
        a3["tested"].append({"value": v, **r})
        print(f"           DIR={v}: n={r['trades']} pnl={r['pnl']:.2f} "
              f"pf={r['pf']:.2f} dd={r['max_dd']:.2f}", flush=True)
    elig = _eligible(a3["tested"])
    if elig:
        winner = elig[0]
        if winner["pf"] > cur_pf or (winner["pf"] >= cur_pf and winner["pnl"] > cur_pnl):
            cur["DIR_BIAS"] = winner["value"]
            cur_pf = winner["pf"]
            cur_pnl = winner["pnl"]
            a3["best_value"] = winner["value"]
            a3["best_lift_pf"] = round(winner["pf"] - BASELINE["pf"], 3)
            a3["best_lift_pnl"] = round(winner["pnl"] - BASELINE["pnl"], 2)
            print(f"           >> ship DIR={winner['value']} pf={winner['pf']:.2f}", flush=True)
        else:
            print(f"           >> no DIR beats cur pf={cur_pf:.2f}; keep default", flush=True)
    state["axes"].append(a3)
    _save(state)

    # ── AXIS 4: TRAIL_PROFILE ──
    print(f"\n[{time.time()-t0:6.1f}s] AXIS 4: TRAIL {TRAIL_GRID} (combo "
          f"SL={cur.get('SL_ATR_MULT')} Q={cur.get('SIGNAL_QUALITY')} "
          f"DIR={cur.get('DIR_BIAS')})", flush=True)
    a4 = {"axis": "TRAIL_PROFILE", "tested": [], "best_value": None}
    for v in TRAIL_GRID:
        if time_up():
            a4["aborted"] = "wall-budget"; break
        r = short_r(_run_bt(sl_mult=cur.get("SL_ATR_MULT"),
                            quality_uniform=cur.get("SIGNAL_QUALITY"),
                            direction=cur.get("DIR_BIAS"),
                            trail_name=v))
        a4["tested"].append({"value": v, **r})
        print(f"           TRAIL={v}: n={r['trades']} pnl={r['pnl']:.2f} "
              f"pf={r['pf']:.2f} dd={r['max_dd']:.2f}", flush=True)
    elig = _eligible(a4["tested"])
    if elig:
        winner = elig[0]
        if winner["pf"] > cur_pf or (winner["pf"] >= cur_pf and winner["pnl"] > cur_pnl):
            cur["TRAIL_PROFILE"] = winner["value"]
            cur_pf = winner["pf"]
            cur_pnl = winner["pnl"]
            a4["best_value"] = winner["value"]
            a4["best_lift_pf"] = round(winner["pf"] - BASELINE["pf"], 3)
            a4["best_lift_pnl"] = round(winner["pnl"] - BASELINE["pnl"], 2)
            print(f"           >> ship TRAIL={winner['value']} pf={winner['pf']:.2f}", flush=True)
        else:
            print(f"           >> no TRAIL beats cur pf={cur_pf:.2f}; keep default", flush=True)
    state["axes"].append(a4)
    _save(state)

    # ── AXIS 5: PULLBACK_RETRACE ──
    print(f"\n[{time.time()-t0:6.1f}s] AXIS 5: PB_RETRACE {PULLBACK_RETRACE_GRID} "
          f"(combo so far {cur})", flush=True)
    a5 = {"axis": "PULLBACK_RETRACE", "tested": [], "best_value": None}
    for v in PULLBACK_RETRACE_GRID:
        if time_up():
            a5["aborted"] = "wall-budget"; break
        r = short_r(_run_bt(sl_mult=cur.get("SL_ATR_MULT"),
                            quality_uniform=cur.get("SIGNAL_QUALITY"),
                            direction=cur.get("DIR_BIAS"),
                            trail_name=cur.get("TRAIL_PROFILE"),
                            pullback_retrace=v))
        a5["tested"].append({"value": str(v), **r})
        print(f"           PB={v}: n={r['trades']} pnl={r['pnl']:.2f} "
              f"pf={r['pf']:.2f} dd={r['max_dd']:.2f}", flush=True)
    elig = _eligible(a5["tested"])
    if elig:
        winner = elig[0]
        if winner["pf"] > cur_pf or (winner["pf"] >= cur_pf and winner["pnl"] > cur_pnl):
            cur["PULLBACK_RETRACE"] = float(winner["value"])
            cur_pf = winner["pf"]
            cur_pnl = winner["pnl"]
            a5["best_value"] = winner["value"]
            a5["best_lift_pf"] = round(winner["pf"] - BASELINE["pf"], 3)
            a5["best_lift_pnl"] = round(winner["pnl"] - BASELINE["pnl"], 2)
            print(f"           >> ship PB={winner['value']} pf={winner['pf']:.2f}", flush=True)
        else:
            print(f"           >> no PB beats cur pf={cur_pf:.2f}; keep default", flush=True)
    state["axes"].append(a5)
    _save(state)

    # ── AXIS 6: ELC_TRIGGER_R ──
    # Note: backtest.v5_backtest does NOT model momentum ELC. We still sweep
    # so the result table shows it was tested; expect identical results (the
    # override dict is read by live executor only).
    print(f"\n[{time.time()-t0:6.1f}s] AXIS 6: ELC {ELC_GRID} (informational — "
          f"BT does not model momentum ELC)", flush=True)
    a6 = {"axis": "ELC_TRIGGER_R", "tested": [],
          "best_value": None,
          "note": ("v5_backtest.py has no ELC simulation; results show only "
                   "indirect effects (none expected). Values copied from grid "
                   "for orchestrator schema.")}
    for v in ELC_GRID:
        if time_up():
            a6["aborted"] = "wall-budget"; break
        r = short_r(_run_bt(sl_mult=cur.get("SL_ATR_MULT"),
                            quality_uniform=cur.get("SIGNAL_QUALITY"),
                            direction=cur.get("DIR_BIAS"),
                            trail_name=cur.get("TRAIL_PROFILE"),
                            pullback_retrace=cur.get("PULLBACK_RETRACE"),
                            elc=v))
        a6["tested"].append({"value": str(v), **r})
        print(f"           ELC={v}: n={r['trades']} pnl={r['pnl']:.2f} "
              f"pf={r['pf']:.2f} dd={r['max_dd']:.2f}", flush=True)
    # No selection — informational only.
    state["axes"].append(a6)
    _save(state)

    # ── FINAL combo re-BT ──
    print(f"\n[{time.time()-t0:6.1f}s] FINAL combo: {cur}", flush=True)
    final = short_r(_run_bt(
        sl_mult=cur.get("SL_ATR_MULT"),
        quality_uniform=cur.get("SIGNAL_QUALITY"),
        direction=cur.get("DIR_BIAS"),
        trail_name=cur.get("TRAIL_PROFILE"),
        pullback_retrace=cur.get("PULLBACK_RETRACE"),
    ))
    state["winner_config"] = cur
    state["winner"] = final
    print(f"           WINNER: n={final['trades']} pnl={final['pnl']:.2f} "
          f"pf={final['pf']:.2f} dd={final['max_dd']:.2f}", flush=True)

    # ── DECISION ──
    pf_threshold = max(1.5, BASELINE["pf"] * 1.05)
    dd_ceiling = BASELINE["max_dd"] * 1.30
    advance = (final["pf"] > pf_threshold and
               final["pnl"] > BASELINE["pnl"] and
               final["max_dd"] <= dd_ceiling)
    state["decision"] = "ADVANCE_TO_WF" if advance else "NULL"
    state["decision_reason"] = (
        f"pf={final['pf']:.2f} vs threshold {pf_threshold:.2f}; "
        f"pnl={final['pnl']:.2f} vs baseline {BASELINE['pnl']:.2f}; "
        f"dd={final['max_dd']:.2f} vs ceiling {dd_ceiling:.2f}"
    )
    state["elapsed_s"] = round(time.time() - t0, 1)
    _save(state)
    print(f"\n           DECISION: {state['decision']}", flush=True)
    print(f"           reason : {state['decision_reason']}", flush=True)
    print(f"           elapsed: {state['elapsed_s']}s", flush=True)
    print(f"           written: {OUT}", flush=True)


if __name__ == "__main__":
    main()
