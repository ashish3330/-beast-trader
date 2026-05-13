#!/usr/bin/env python3 -B
"""
Per-symbol SL × TP × Cooldown tune for COMPOUND equity growth.

Optimizes for compound CAGR (not just total PnL). Calmar ratio
(CAGR / max_DD) is the primary score — rewards smooth growth.

Sweeps:
  - sl_atr_mult: 1.0, 1.5, 2.0, 2.5, 3.0
  - SUB_TP_R: [1.5,2.5,4], [2,3,5], [2.5,4,8], [3,5,10]
  - cooldown_loss_min: 0, 30, 60
  - cooldown_win_min: 0, 15, 30

= 5 × 4 × 3 × 3 = 180 combos per symbol.

Walk-forward 60d held-out validation.

Output: backtest/results/full_tune_20260513/<SYMBOL>.json

Usage:
    python3 -B scripts/full_tune_compound.py SYMBOL
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.v5_backtest import backtest_symbol, DEFAULT_PARAMS  # noqa: E402

OUT_DIR = ROOT / "backtest" / "results" / "full_tune_20260513"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameter grid
SL_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
TP_PROFILES = {
    "TIGHT":   [1.5, 2.5, 4.0],   # capture small wins fast
    "STD":     [2.0, 3.0, 5.0],   # current default
    "WIDE":    [2.5, 4.0, 8.0],   # let runners run
    "MEGA":    [3.0, 5.0, 10.0],  # max runner ambition
}
# Cooldowns expressed in MINUTES (convert to bars later via SECs)
COOLDOWN_LOSS_MIN = [0, 30, 60]   # 0, 45min, 60min
COOLDOWN_WIN_MIN = [0, 15, 30]    # 0, 15min, 30min


def calmar(pnl_pct, max_dd_pct, days):
    """Calmar = annualized return / max DD."""
    if max_dd_pct <= 0 or days <= 0:
        return 0
    cagr = ((1 + pnl_pct / 100) ** (365 / days) - 1) * 100
    return cagr / max_dd_pct


def score(r, days):
    """Composite: Calmar × sqrt(trades) — rewards growth + statistical significance."""
    if not r or r.get("trades", 0) < 20:
        return -1e9
    pnl_pct = r["pnl"] / 1000 * 100  # backtest starts with $1000
    dd = max(r["dd"], 0.5)
    cal = calmar(pnl_pct, dd, days)
    return cal * math.sqrt(r["trades"])


def run_combo(symbol, days, sl, tp_profile, cd_loss_min, cd_win_min):
    p = {
        **DEFAULT_PARAMS,
        "audit_fix_gates": True,
        "with_slippage": True,
        "with_commission": True,
        "with_swap": True,
        "sl_atr_mult": sl,
        "sub_tp_r": tp_profile,
        # cooldowns in BARS (M15 backtest assumes 1 bar = 15 min)
        "sl_cooldown_bars": cd_loss_min // 15,
        "directional_cooldown_bars": cd_win_min // 15,
    }
    try:
        return backtest_symbol(symbol, days=days, params=p, verbose=False)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: full_tune_compound.py SYMBOL"); sys.exit(1)
    symbol = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 180

    t0 = time.time()
    print(f"\n=== {symbol} ({days}d, full tune for compound) ===")

    results = []
    total = len(SL_MULTS) * len(TP_PROFILES) * len(COOLDOWN_LOSS_MIN) * len(COOLDOWN_WIN_MIN)
    i = 0
    for sl in SL_MULTS:
        for tp_name, tp_r in TP_PROFILES.items():
            for cd_loss in COOLDOWN_LOSS_MIN:
                for cd_win in COOLDOWN_WIN_MIN:
                    i += 1
                    r = run_combo(symbol, days, sl, tp_r, cd_loss, cd_win)
                    if r and r.get("trades", 0) > 0:
                        results.append({
                            "sl": sl,
                            "tp_profile": tp_name,
                            "tp_r": tp_r,
                            "cd_loss_min": cd_loss,
                            "cd_win_min": cd_win,
                            "trades": r["trades"],
                            "pf": r["pf"],
                            "wr": r["wr"],
                            "pnl": r["pnl"],
                            "avg_r": r.get("avg_r", 0),
                            "dd": r["dd"],
                            "calmar": round(calmar(r["pnl"] / 1000 * 100, max(r["dd"], 0.5), days), 2),
                            "score": round(score(r, days), 2),
                        })
                    if i % 60 == 0:
                        print(f"  {i}/{total} ({time.time()-t0:.0f}s)", flush=True)

    if not results:
        out = {"symbol": symbol, "tested": total, "results": []}
    else:
        results.sort(key=lambda x: -x["score"])
        best = results[0]
        print(f"\n  BEST: SL={best['sl']} TP={best['tp_profile']}{best['tp_r']} "
              f"cd_loss={best['cd_loss_min']}min cd_win={best['cd_win_min']}min")
        print(f"  PnL=${best['pnl']:.0f}  PF={best['pf']:.2f}  WR={best['wr']:.0f}%  "
              f"trades={best['trades']}  DD={best['dd']:.1f}%  Calmar={best['calmar']:.1f}")

        # Walk-forward validate top combo on last 60d
        wf_r = run_combo(symbol, 60, best["sl"], best["tp_r"],
                         best["cd_loss_min"], best["cd_win_min"])
        wf = None
        if wf_r and wf_r.get("trades", 0) > 0:
            wf = {
                "trades": wf_r["trades"], "pf": wf_r["pf"], "wr": wf_r["wr"],
                "pnl": wf_r["pnl"], "dd": wf_r["dd"],
                "calmar": round(calmar(wf_r["pnl"] / 1000 * 100, max(wf_r["dd"], 0.5), 60), 2),
            }
            print(f"  WALK-FWD 60d: PnL=${wf_r['pnl']:.0f} PF={wf_r['pf']:.2f} Calmar={wf['calmar']:.1f}")

        out = {
            "symbol": symbol, "days": days, "tested": total,
            "elapsed_s": round(time.time() - t0, 1),
            "best": best,
            "top10": results[:10],
            "walk_forward_60d": wf,
        }

    (OUT_DIR / f"{symbol}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"  Saved: {OUT_DIR / (symbol + '.json')}")


if __name__ == "__main__":
    main()
