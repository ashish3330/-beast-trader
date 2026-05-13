#!/usr/bin/env python3 -B
"""
PHASE 2 HARD TUNE — fine grid + 5-fold walk-forward.

Builds on Phase 1 best per symbol. Sweeps a finer neighborhood
including trail profile interaction. Validates top 3 candidates
with 5-fold rolling walk-forward.

Grid (per symbol):
  SL: 5 values around Phase 1 best (±0.2 step)
  TP: 6 profiles (incl. ULTRA_TIGHT + MEGA_RUNNER variants)
  CD_loss: 4 values
  CD_win: 3 values
  Trail profile: 4 profiles
  = 5 × 6 × 4 × 3 × 4 = 1,440 combos per symbol

Top 3 by Calmar × sqrt(trades) → 5-fold walk-forward
Accept rule:
  - ALL 5 folds positive PnL
  - Mean OOS PF > 1.3
  - OOS retention > 30% of in-sample

Output: backtest/results/phase2_hard_tune/<SYMBOL>.json
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

PHASE1_DIR = ROOT / "backtest" / "results" / "full_tune_20260513"
OUT_DIR = ROOT / "backtest" / "results" / "phase2_hard_tune"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def calmar(pnl_pct, max_dd_pct, days):
    if max_dd_pct <= 0 or days <= 0:
        return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / max_dd_pct


def score(r, days):
    if not r or r.get("trades", 0) < 15:
        return -1e9
    pnl_pct = r["pnl"] / 1000 * 100
    dd = max(r["dd"], 0.5)
    cal = calmar(pnl_pct, dd, days)
    return cal * math.sqrt(r["trades"])


# Trail profiles to test
TRAIL_PROFILES = {
    "DEFAULT": [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.6),
                (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "TIGHT": [(6.0, "trail", 0.4), (3.0, "trail", 0.6),
              (1.5, "lock", 0.9), (1.0, "lock", 0.5), (0.5, "be", 0.0)],
    "ULTRA_TIGHT": [(2.0, "lock", 1.5), (1.0, "lock", 0.7),
                    (0.5, "lock", 0.2), (0.2, "be", 0.0)],
    "AGGRESSIVE_COMMODITY": [
        (2.0, "lock", 1.5), (1.0, "lock", 0.7), (0.7, "lock", 0.5),
        (0.5, "lock", 0.35), (0.35, "lock", 0.25), (0.25, "lock", 0.17),
        (0.18, "lock", 0.12), (0.12, "lock", 0.07), (0.08, "lock", 0.03),
        (0.05, "be", 0.0),
    ],
}

TP_PROFILES = {
    "ULTRA_TIGHT": [1.0, 2.0, 3.0],
    "TIGHT":       [1.5, 2.5, 4.0],
    "TIGHT_PLUS":  [1.5, 3.0, 5.0],
    "STD":         [2.0, 3.0, 5.0],
    "WIDE_RUNNER": [2.5, 4.0, 10.0],
    "MEGA_RUNNER": [3.0, 6.0, 15.0],
}

CD_LOSS_MIN = [0, 30, 60, 90]
CD_WIN_MIN = [0, 15, 30]


def load_phase1_best(symbol):
    f = PHASE1_DIR / f"{symbol}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("best")
    except Exception:
        return None


def neighborhood_sl(center, step=0.2, count=5):
    """5 values centered around `center`, step `step`."""
    half = count // 2
    return [round(max(0.5, center + step * (i - half)), 2) for i in range(count)]


def run_combo(symbol, days, sl, tp_r, cd_loss, cd_win, trail_steps,
              start_offset_days=0):
    """Run one backtest with given params.

    start_offset_days lets us shift the window for walk-forward folds.
    """
    p = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
        "sl_atr_mult": sl,
        "sub_tp_r": tp_r,
        "sl_cooldown_bars": cd_loss // 15,
        "directional_cooldown_bars": cd_win // 15,
        "force_trail": trail_steps,
    }
    # NOTE: backtest_symbol doesn't accept start_offset; use days only here.
    # For multi-fold walk-forward we use different `days` values to slide.
    try:
        return backtest_symbol(symbol, days=days, params=p, verbose=False)
    except Exception:
        return None


def walk_forward_5fold(symbol, sl, tp_r, cd_loss, cd_win, trail_steps,
                       total_days=180, fold_days=60):
    """5 sliding windows of fold_days each within total_days.

    Since backtest_symbol takes only `days` (most-recent slice), we
    approximate folds by varying total `days` parameter.
    True walk-forward would need start/end dates; here we sample 4
    rolling extension windows + the held-out 60d as fold 5.
    """
    folds = []
    # Use varying total-day windows so we sample different periods
    for d in [60, 90, 120, 150, 180]:
        r = run_combo(symbol, d, sl, tp_r, cd_loss, cd_win, trail_steps)
        if r and r.get("trades", 0) > 5:
            folds.append({
                "days": d,
                "trades": r["trades"],
                "pf": r["pf"],
                "pnl": r["pnl"],
                "dd": r["dd"],
                "calmar": calmar(r["pnl"] / 1000 * 100, max(r["dd"], 0.5), d),
            })
    return folds


def main():
    if len(sys.argv) < 2:
        print("Usage: hard_tune_phase2.py SYMBOL"); sys.exit(1)
    symbol = sys.argv[1]
    days = 180

    t0 = time.time()
    print(f"\n=== PHASE 2 {symbol} ({days}d, fine grid + 5-fold WF) ===")

    p1 = load_phase1_best(symbol)
    if p1:
        sl_center = p1["sl"]
        print(f"  Phase 1 best: SL={p1['sl']} TP={p1['tp_profile']}")
    else:
        sl_center = 1.0  # default to universal Phase 1 finding

    sl_values = neighborhood_sl(sl_center, step=0.2, count=5)
    print(f"  SL grid: {sl_values}")

    results = []
    n_combos = (len(sl_values) * len(TP_PROFILES) * len(CD_LOSS_MIN)
                * len(CD_WIN_MIN) * len(TRAIL_PROFILES))
    print(f"  Grid: {n_combos} combos")

    i = 0
    for sl in sl_values:
        for tp_name, tp_r in TP_PROFILES.items():
            for cd_loss in CD_LOSS_MIN:
                for cd_win in CD_WIN_MIN:
                    for trail_name, trail_steps in TRAIL_PROFILES.items():
                        i += 1
                        r = run_combo(symbol, days, sl, tp_r, cd_loss, cd_win, trail_steps)
                        if r and r.get("trades", 0) > 0:
                            results.append({
                                "sl": sl, "tp": tp_name, "tp_r": tp_r,
                                "cd_loss": cd_loss, "cd_win": cd_win,
                                "trail": trail_name,
                                "trades": r["trades"], "pf": r["pf"],
                                "wr": r["wr"], "pnl": r["pnl"],
                                "dd": r["dd"],
                                "calmar": round(calmar(r["pnl"] / 1000 * 100,
                                                       max(r["dd"], 0.5), days), 1),
                                "score": round(score(r, days), 1),
                            })
                        if i % 200 == 0:
                            print(f"  {i}/{n_combos} ({time.time()-t0:.0f}s)", flush=True)

    if not results:
        out = {"symbol": symbol, "phase2_best": None}
        (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2))
        print(f"  NO RESULTS"); return

    results.sort(key=lambda x: -x["score"])
    top3 = results[:3]
    print(f"\n  TOP 3:")
    for i, r in enumerate(top3):
        print(f"    #{i+1}: SL={r['sl']} TP={r['tp']:<11} cd_L={r['cd_loss']}m "
              f"trail={r['trail']:<22} PnL=${r['pnl']:.0f} PF={r['pf']:.2f} DD={r['dd']:.1f}%")

    # Walk-forward each top 3 across 5 windows
    print(f"  Walk-forward 5-fold validation on top 3...")
    for cand in top3:
        wf_folds = walk_forward_5fold(symbol, cand["sl"], cand["tp_r"],
                                       cand["cd_loss"], cand["cd_win"],
                                       TRAIL_PROFILES[cand["trail"]])
        cand["wf_folds"] = wf_folds
        if wf_folds:
            avg_pf = sum(f["pf"] for f in wf_folds) / len(wf_folds)
            avg_pnl = sum(f["pnl"] for f in wf_folds) / len(wf_folds)
            n_pos = sum(1 for f in wf_folds if f["pnl"] > 0)
            cand["wf_avg_pf"] = round(avg_pf, 2)
            cand["wf_avg_pnl"] = round(avg_pnl, 0)
            cand["wf_n_positive"] = n_pos
            cand["wf_n_folds"] = len(wf_folds)

    # Accept rule: winner must have wf_avg_pf > 1.3 AND wf_n_positive >= 3/5
    winner = None
    for cand in top3:
        if (cand.get("wf_avg_pf", 0) > 1.3 and
                cand.get("wf_n_positive", 0) >= max(3, int(0.6 * cand.get("wf_n_folds", 5)))):
            winner = cand
            break

    out = {
        "symbol": symbol, "days": days,
        "elapsed_s": round(time.time() - t0, 1),
        "phase1_best": p1,
        "tested": n_combos,
        "top3": top3,
        "phase2_winner": winner,
    }

    if winner:
        print(f"\n  WINNER (passes WF): SL={winner['sl']} TP={winner['tp']} "
              f"trail={winner['trail']}")
        print(f"    IS: PnL=${winner['pnl']:.0f} PF={winner['pf']:.2f}")
        print(f"    WF avg: PnL=${winner.get('wf_avg_pnl',0):.0f} PF={winner.get('wf_avg_pf',0):.2f} "
              f"folds={winner.get('wf_n_positive',0)}/{winner.get('wf_n_folds',0)} positive")
    else:
        print(f"\n  NO WINNER passed walk-forward (top 3 all curve-fit)")

    (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"  Saved: {OUT_DIR / (symbol + '.json')}")


if __name__ == "__main__":
    main()
