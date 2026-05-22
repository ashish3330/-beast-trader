#!/usr/bin/env python3 -B
"""
EURUSD per-symbol hard-tune — 7-dimensional optimization, ~100 iterations.

Dimensions (per user spec):
  1. SL_atr_mult     ∈ {0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0}
  2. Trail profile   ∈ {7 named profiles}
  3. Pullback ATR    ∈ {0.4, 0.5, 0.6, 0.8, 1.0, 1.2}
  4. Pullback wait   ∈ {3, 4, 5, 6, 8}
  5. VWAP buffer     ∈ {0.0_disabled, 0.3, 0.5, 0.7, 1.0}
  6. min_quality     ∈ {28, 30, 33, 35, 38, 40, 43}  (applied to all 4 regimes)
  7. Toxic hours     ∈ {(5,6,20), (5,20), (6,20), ()}

Phase A — 80 random samples across the joint space (Latin-hypercube-ish).
Phase B — Local search around top-5 from A (~20 perturbations).
Phase C — Walk-forward [60, 90, 120, 150, 180] on top-3 → ship gate
          Δ≥$50 AND WF≥3/5 positive.

Output: per_symbol_tune_20260522/EURUSD.json + EURUSD.md
READ-ONLY — never touches source files. All overrides via monkey-patch
inside each multiprocessing worker (workers reload config + bt fresh).
"""
import os
import sys
import json
import time
import math
import random
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SYMBOL = "EURUSD"
OUT_DIR = ROOT / "per_symbol_tune_20260522"
OUT_JSON = OUT_DIR / f"{SYMBOL}.json"
OUT_MD = OUT_DIR / f"{SYMBOL}.md"

# ── Dimensions ───────────────────────────────────────────────────────────
SL_GRID = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

# 7 named trail profiles (live format: (R, type, param)). Patched into
# config.SYMBOL_TRAIL_OVERRIDE + SYMBOL_REGIME_TRAIL_OVERRIDE inside worker
# so we don't need force_trail mid-loop.
TRAIL_PROFILES = {
    "TIGHT_LOCK":     [(4.0, "lock", 2.5), (2.0, "lock", 1.2), (1.0, "lock", 0.5),
                       (0.3, "be", 0.0)],
    "WIDE_RUNNER":    [(10.0, "trail", 0.3), (5.0, "trail", 0.5), (2.5, "trail", 0.7),
                       (1.5, "lock", 0.5), (0.7, "be", 0.0)],
    "RANGE_TIGHT":    [(4.0, "trail", 0.5), (2.0, "lock", 1.2), (1.0, "lock", 0.6),
                       (0.3, "be", 0.0)],
    "TREND_LOOSE":    [(15.0, "trail", 0.3), (8.0, "trail", 0.4), (4.0, "trail", 0.5),
                       (2.0, "lock", 1.0), (1.0, "lock", 0.5), (0.3, "be", 0.0)],
    "AGGR_LOCK":      [(8.0, "trail", 0.3), (4.0, "trail", 0.5), (2.0, "trail", 0.8),
                       (1.5, "lock", 0.7), (1.0, "lock", 0.4), (0.5, "be", 0.0)],
    "RUNNER_NO_BE":   [(10.0, "trail", 0.3), (5.0, "trail", 0.4), (2.0, "trail", 0.5),
                       (1.0, "trail", 0.5), (0.7, "lock", 0.4), (0.5, "lock", 0.2)],
    "EURUSD_CURRENT": [(5.0, "trail", 0.3), (3.0, "trail", 0.5), (2.0, "trail", 0.8),
                       (1.5, "lock", 0.7), (1.0, "lock", 0.3),
                       (0.7, "lock", 0.15), (0.4, "be", 0.0)],
}

PULLBACK_ATR_GRID = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
PULLBACK_WAIT_GRID = [3, 4, 5, 6, 8]
VWAP_BUF_GRID = [0.0, 0.3, 0.5, 0.7, 1.0]  # 0.0 → disabled
MIN_QUALITY_GRID = [28, 30, 33, 35, 38, 40, 43]
TOXIC_HOURS_GRID = [
    (5, 6, 20),
    (5, 20),
    (6, 20),
    (),
]

WF_FOLDS = [60, 90, 120, 150, 180]

# Ship gates (per user spec)
MIN_DELTA_PNL = 50.0
MIN_WF_POS_FOLDS = 3
MIN_TRADES = 30  # statistical sanity
MIN_PF = 1.5     # quality sanity

# Optimization sizing
PHASE_A_ITERS = 80
PHASE_B_PERTURBATIONS = 4   # per top-5 winner ⇒ 20 extra runs
PHASE_C_TOPN = 3            # WF on top-3

RANDOM_SEED = 20260522
random.seed(RANDOM_SEED)


# ── Worker (runs inside Pool) ───────────────────────────────────────────
def _run_one(args):
    """Run a single BT with all 7 dims monkey-patched.
    args = (idx, days, params_dict)
    """
    idx, days, params = args
    try:
        import importlib
        import config as cfg
        importlib.reload(cfg)

        # 1) SL_atr_mult
        cfg.SYMBOL_ATR_SL_OVERRIDE = dict(cfg.SYMBOL_ATR_SL_OVERRIDE)
        cfg.SYMBOL_ATR_SL_OVERRIDE[SYMBOL] = float(params["sl_mult"])
        # Clear regime-level SL for this symbol so above takes effect
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME = dict(cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME)
        cfg.SYMBOL_ATR_SL_OVERRIDE_REGIME[SYMBOL] = {}

        # 2) Trail profile (live format)
        trail_live = TRAIL_PROFILES[params["trail_name"]]
        cfg.SYMBOL_TRAIL_OVERRIDE = dict(cfg.SYMBOL_TRAIL_OVERRIDE)
        cfg.SYMBOL_TRAIL_OVERRIDE[SYMBOL] = trail_live
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE = dict(cfg.SYMBOL_REGIME_TRAIL_OVERRIDE)
        cfg.SYMBOL_REGIME_TRAIL_OVERRIDE[SYMBOL] = {
            r: trail_live for r in ("trending", "ranging", "volatile", "low_vol")
        }

        # 3, 4) Pullback ATR + wait (read inside loop via config import)
        cfg.PULLBACK_ATR_RETRACE = float(params["pullback_atr"])
        cfg.PULLBACK_MAX_WAIT_BARS = int(params["pullback_wait"])

        # 6) min_quality (all regimes same)
        mq = int(params["min_quality"])
        cfg.SIGNAL_QUALITY_SYMBOL = dict(cfg.SIGNAL_QUALITY_SYMBOL)
        cfg.SIGNAL_QUALITY_SYMBOL[SYMBOL] = {
            "trending": mq, "ranging": mq, "volatile": mq, "low_vol": mq,
        }

        # 7) Toxic hours — backtest only reads global TOXIC_HOURS_UTC; it does
        # NOT add per-symbol extras. So we union the GLOBAL_TOXIC baseline
        # {1,2,3,4} with the symbol-specific test set. This way, "tox=()"
        # means "no EURUSD extras" (matches live with PER_SYMBOL[EURUSD]={}),
        # and "tox=(5,6,20)" means "EURUSD extras 5+6+20 on top of global".
        cfg.TOXIC_HOURS_UTC = {1, 2, 3, 4} | set(params["toxic_hours"])
        # Clear per-symbol so it doesn't add hours (we've already encoded them)
        cfg.TOXIC_HOURS_PER_SYMBOL = dict(cfg.TOXIC_HOURS_PER_SYMBOL)
        cfg.TOXIC_HOURS_PER_SYMBOL[SYMBOL] = set()

        # 5) VWAP buffer — hardcoded inline in v5_backtest.py.  We runtime-patch
        # by reading the source, substituting the constant, compiling into a
        # fresh module namespace, and calling backtest_symbol from there.
        # This keeps the disk source unchanged (READ-ONLY satisfied).
        vwap_buf = float(params["vwap_buffer"])
        bt_module = _load_patched_bt(vwap_buf)

        r = bt_module.backtest_symbol(SYMBOL, days=days, verbose=False)
        if r is None:
            return {"idx": idx, "params": params, "result": None,
                    "error": "bt returned None"}
        out_r = {
            "trades": int(r.get("trades", 0)),
            "wr": float(r.get("wr", 0) or 0),
            "pf": float(r.get("pf", 0) or 0),
            "pnl": float(r.get("pnl", 0) or 0),
            "dd": float(r.get("dd", 0) or 0),
            "avg_r": float(r.get("avg_r", 0) or 0),
        }
        return {"idx": idx, "params": params, "days": days, "result": out_r}
    except Exception as e:
        return {"idx": idx, "params": params,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


# Cache of patched bt modules per (vwap_buf) value, per-process
_BT_CACHE = {}


def _load_patched_bt(vwap_buf):
    """Return a v5_backtest module variant with VWAP buffer hardcoded
    to `vwap_buf`. Special value 0.0 → VWAP gate disabled (bypass continues).
    Each worker process keeps a small cache keyed on vwap_buf.

    READ-ONLY: never writes to disk; just re-execs source under a unique
    module name.
    """
    # Always re-load when config has been patched in this process — caller
    # is responsible for cfg reload prior. But the bt module reads
    # SL_OVERRIDE, TRAIL_OVERRIDE, etc. AT MODULE LOAD time, so we need
    # to reload bt every call so it picks up our cfg patches.
    import importlib
    import types
    src_path = ROOT / "backtest" / "v5_backtest.py"
    src = src_path.read_text()

    # Substitute the VWAP buffer constant.
    # Original line: `                atr_buf = float(ind["at"][bi]) * 0.5`
    if vwap_buf <= 0.0:
        # Disable filter: replace the inner if/continue block. Cleanest: set
        # `vw = None` so subsequent NaN check skips.
        old = 'vw = ind.get("vwap")'
        new = 'vw = None  # VWAP gate disabled by tuner'
        if old not in src:
            raise RuntimeError("VWAP source pattern not found (disable)")
        src = src.replace(old, new, 1)
    else:
        old = 'atr_buf = float(ind["at"][bi]) * 0.5'
        new = f'atr_buf = float(ind["at"][bi]) * {vwap_buf:.4f}'
        if old not in src:
            raise RuntimeError("VWAP source pattern not found (buf)")
        src = src.replace(old, new, 1)

    # Compile + exec into a fresh module namespace.
    mod_name = f"backtest._v5_bt_vwap_{int(vwap_buf * 1000):04d}"
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(src_path)
    # Ensure relative imports resolve like the original module
    mod.__package__ = "backtest"
    code = compile(src, str(src_path), "exec")
    exec(code, mod.__dict__)
    sys.modules[mod_name] = mod
    return mod


# ── Sampling helpers ────────────────────────────────────────────────────
def _random_params():
    return {
        "sl_mult":      random.choice(SL_GRID),
        "trail_name":   random.choice(list(TRAIL_PROFILES.keys())),
        "pullback_atr": random.choice(PULLBACK_ATR_GRID),
        "pullback_wait": random.choice(PULLBACK_WAIT_GRID),
        "vwap_buffer":  random.choice(VWAP_BUF_GRID),
        "min_quality":  random.choice(MIN_QUALITY_GRID),
        "toxic_hours":  random.choice(TOXIC_HOURS_GRID),
    }


def _params_key(p):
    return (p["sl_mult"], p["trail_name"], p["pullback_atr"], p["pullback_wait"],
            p["vwap_buffer"], p["min_quality"], tuple(p["toxic_hours"]))


def _perturb(p, n_neighbours=4):
    """Yield up to n_neighbours single-dim perturbations of params p."""
    dims = [
        ("sl_mult", SL_GRID),
        ("trail_name", list(TRAIL_PROFILES.keys())),
        ("pullback_atr", PULLBACK_ATR_GRID),
        ("pullback_wait", PULLBACK_WAIT_GRID),
        ("vwap_buffer", VWAP_BUF_GRID),
        ("min_quality", MIN_QUALITY_GRID),
        ("toxic_hours", TOXIC_HOURS_GRID),
    ]
    random.shuffle(dims)
    picks = []
    for name, grid in dims:
        cur = p[name]
        # pick a neighbour value
        try:
            idx = grid.index(cur)
            choices = []
            if idx - 1 >= 0:
                choices.append(grid[idx - 1])
            if idx + 1 < len(grid):
                choices.append(grid[idx + 1])
            if not choices:
                continue
            nv = random.choice(choices)
        except ValueError:
            # non-orderable (trail_name, toxic_hours)
            choices = [v for v in grid if v != cur]
            if not choices:
                continue
            nv = random.choice(choices)
        np = dict(p)
        np[name] = nv
        picks.append(np)
        if len(picks) >= n_neighbours:
            break
    return picks


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = {
        "symbol": SYMBOL,
        "session": "per_symbol_tune_20260522",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "spec": {
            "sl_grid": SL_GRID,
            "trail_profiles": {k: v for k, v in TRAIL_PROFILES.items()},
            "pullback_atr_grid": PULLBACK_ATR_GRID,
            "pullback_wait_grid": PULLBACK_WAIT_GRID,
            "vwap_buf_grid": VWAP_BUF_GRID,
            "min_quality_grid": MIN_QUALITY_GRID,
            "toxic_hours_grid": [list(t) for t in TOXIC_HOURS_GRID],
        },
        "gates": {
            "min_delta_pnl": MIN_DELTA_PNL,
            "min_wf_pos_folds": MIN_WF_POS_FOLDS,
            "min_trades": MIN_TRADES,
            "min_pf": MIN_PF,
        },
        "baseline": None,
        "phase_a": [],
        "phase_b": [],
        "phase_c_wf": [],
        "winner": None,
        "elapsed_s": 0,
    }

    workers = max(2, min(6, os.cpu_count() or 4))
    print(f"[EURUSD tune] workers={workers}  seed={RANDOM_SEED}")

    # ── 0) Baseline (current live config, 180d) ──────────────────────────
    print("\n[0] Baseline (current live config, 180d) ...")
    from importlib import reload
    import config as _cfg
    reload(_cfg)
    # Default baseline = live config; just run v5_backtest as-is
    import backtest.v5_backtest as _bt
    reload(_bt)
    r0 = _bt.backtest_symbol(SYMBOL, days=180, verbose=False)
    base = {
        "trades": int(r0["trades"]),
        "wr": float(r0["wr"]),
        "pf": float(r0["pf"]),
        "pnl": float(r0["pnl"]),
        "dd": float(r0["dd"]),
        "avg_r": float(r0["avg_r"]),
    }
    out["baseline"] = base
    base_pnl = base["pnl"]
    print(f"  baseline: n={base['trades']} pf={base['pf']:.2f} "
          f"pnl=${base['pnl']:+.0f} wr={base['wr']:.1f}% dd={base['dd']:.1f}%")

    # ── A) Random sampling, PHASE_A_ITERS samples ────────────────────────
    print(f"\n[A] Random search × {PHASE_A_ITERS} (180d, parallel)")
    sampled_keys = set()
    jobs = []
    idx = 0
    while len(jobs) < PHASE_A_ITERS:
        p = _random_params()
        k = _params_key(p)
        if k in sampled_keys:
            continue
        sampled_keys.add(k)
        jobs.append((idx, 180, p))
        idx += 1

    t_a = time.time()
    results_a = []
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_one, jobs), 1):
            results_a.append(res)
            if i % 10 == 0 or i == len(jobs):
                ok = sum(1 for r in results_a if r.get("result") is not None)
                err = sum(1 for r in results_a if r.get("error"))
                best = max((r["result"]["pnl"] for r in results_a
                            if r.get("result")), default=0)
                print(f"  {i:3d}/{len(jobs)} ({time.time() - t_a:.0f}s)  "
                      f"ok={ok} err={err}  best_pnl=${best:+.0f}")
    out["phase_a"] = results_a

    # Rank Phase A
    ranked_a = sorted(
        [r for r in results_a if r.get("result") is not None],
        key=lambda r: -r["result"]["pnl"],
    )
    print(f"\n[A] Top 5 by PnL:")
    for r in ranked_a[:5]:
        rr = r["result"]; pp = r["params"]
        print(f"  pnl=${rr['pnl']:+7.0f} pf={rr['pf']:5.2f} n={rr['trades']:3d} "
              f"wr={rr['wr']:5.1f}% dd={rr['dd']:4.1f}%  "
              f"SL={pp['sl_mult']} {pp['trail_name']:14s} "
              f"pb=({pp['pullback_atr']},{pp['pullback_wait']}) "
              f"vwap={pp['vwap_buffer']} mQ={pp['min_quality']} "
              f"tox={pp['toxic_hours']}")

    # ── B) Local perturbations around top-5 ─────────────────────────────
    top5 = ranked_a[:5]
    pert_jobs = []
    pert_keys = set(sampled_keys)
    pidx = len(jobs)
    for r in top5:
        for np in _perturb(r["params"], n_neighbours=PHASE_B_PERTURBATIONS):
            k = _params_key(np)
            if k in pert_keys:
                continue
            pert_keys.add(k)
            pert_jobs.append((pidx, 180, np))
            pidx += 1

    print(f"\n[B] Local perturbation × {len(pert_jobs)}")
    t_b = time.time()
    results_b = []
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_one, pert_jobs), 1):
            results_b.append(res)
            if i % 10 == 0 or i == len(pert_jobs):
                ok = sum(1 for r in results_b if r.get("result") is not None)
                print(f"  {i:3d}/{len(pert_jobs)} ({time.time() - t_b:.0f}s) ok={ok}")
    out["phase_b"] = results_b

    # Combine A + B, rerank
    combined = ranked_a + [r for r in results_b if r.get("result") is not None]
    combined.sort(key=lambda r: -r["result"]["pnl"])

    # Deduplicate by params key
    seen = set()
    unique_combined = []
    for r in combined:
        k = _params_key(r["params"])
        if k in seen:
            continue
        seen.add(k)
        unique_combined.append(r)
    combined = unique_combined

    # Filter for quality (≥ MIN_TRADES) and Δ ≥ MIN_DELTA_PNL
    candidates = [
        r for r in combined
        if r["result"]["trades"] >= MIN_TRADES
        and r["result"]["pf"] >= MIN_PF
        and (r["result"]["pnl"] - base_pnl) >= MIN_DELTA_PNL
    ]
    print(f"\n[B] {len(candidates)} candidates pass 180d filters "
          f"(n≥{MIN_TRADES}, PF≥{MIN_PF}, Δ≥${MIN_DELTA_PNL:.0f})")
    for r in candidates[:8]:
        rr = r["result"]; pp = r["params"]
        print(f"  pnl=${rr['pnl']:+7.0f} pf={rr['pf']:5.2f} n={rr['trades']:3d} "
              f"Δ=${rr['pnl']-base_pnl:+6.0f}  "
              f"SL={pp['sl_mult']} {pp['trail_name']:14s} "
              f"pb=({pp['pullback_atr']},{pp['pullback_wait']}) "
              f"vwap={pp['vwap_buffer']} mQ={pp['min_quality']} "
              f"tox={pp['toxic_hours']}")

    # ── C) Walk-forward on top-N ─────────────────────────────────────────
    top_for_wf = candidates[:PHASE_C_TOPN]
    if not top_for_wf:
        print(f"\n[C] No candidates pass 180d gate — skipping WF.")
        out["winner"] = None
    else:
        print(f"\n[C] Walk-forward on top {len(top_for_wf)} candidates × "
              f"{len(WF_FOLDS)} folds = {len(top_for_wf) * len(WF_FOLDS)} BTs")
        wf_jobs = []
        widx = pidx
        for cand in top_for_wf:
            for d in WF_FOLDS:
                wf_jobs.append((widx, d, cand["params"]))
                widx += 1
        t_c = time.time()
        wf_results = []
        with Pool(workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_one, wf_jobs), 1):
                wf_results.append(res)
                if i % 5 == 0 or i == len(wf_jobs):
                    print(f"  {i:3d}/{len(wf_jobs)} ({time.time() - t_c:.0f}s)")
        out["phase_c_wf"] = wf_results

        # Aggregate WF per candidate
        wf_by_key = {}
        for res in wf_results:
            if res.get("result") is None:
                continue
            k = _params_key(res["params"])
            wf_by_key.setdefault(k, []).append({
                "days": res["days"], "result": res["result"]
            })
        ranked_wf = []
        for cand in top_for_wf:
            k = _params_key(cand["params"])
            folds = wf_by_key.get(k, [])
            folds.sort(key=lambda x: x["days"])
            pos = sum(1 for f in folds if f["result"]["pnl"] > 0)
            n_valid = len(folds)
            pfs = [f["result"]["pf"] for f in folds]
            avg_pf = round(sum(pfs) / max(1, n_valid), 3) if n_valid else 0.0
            delta = cand["result"]["pnl"] - base_pnl
            wf_pass = (pos >= MIN_WF_POS_FOLDS and delta >= MIN_DELTA_PNL)
            ranked_wf.append({
                "params": cand["params"],
                "result_180d": cand["result"],
                "delta_pnl": round(delta, 2),
                "wf": {
                    "avg_pf": avg_pf, "pos_folds": pos, "n_valid": n_valid,
                    "folds": folds,
                },
                "wf_pass": wf_pass,
            })
        ranked_wf.sort(key=lambda x: (-(x["wf_pass"]), -x["delta_pnl"]))

        print(f"\n[C] WF results (avg_pf, pos_folds/5):")
        for w in ranked_wf:
            pp = w["params"]; rr = w["result_180d"]
            print(f"  Δ=${w['delta_pnl']:+7.0f} 180d-pf={rr['pf']:5.2f} "
                  f"WF-pf={w['wf']['avg_pf']:5.2f} pos={w['wf']['pos_folds']}/5 "
                  f"PASS={w['wf_pass']}  "
                  f"SL={pp['sl_mult']} {pp['trail_name']:14s} "
                  f"pb=({pp['pullback_atr']},{pp['pullback_wait']}) "
                  f"vwap={pp['vwap_buffer']} mQ={pp['min_quality']} "
                  f"tox={pp['toxic_hours']}")

        winners = [w for w in ranked_wf if w["wf_pass"]]
        if winners:
            out["winner"] = winners[0]
            print(f"\n[*] SHIP WINNER: Δ=${winners[0]['delta_pnl']:+.0f} "
                  f"WF-pos={winners[0]['wf']['pos_folds']}/5")
        else:
            out["winner"] = None
            print("\n[*] NO SHIP (no candidate cleared Δ≥$50 AND WF≥3/5)")

        # Persist full ranked WF
        out["phase_c_ranked"] = ranked_wf

    out["elapsed_s"] = round(time.time() - t0, 1)

    # ── Save JSON ────────────────────────────────────────────────────────
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[✓] Wrote {OUT_JSON.relative_to(ROOT)}")

    # ── Markdown summary ─────────────────────────────────────────────────
    _write_md(out)
    print(f"[✓] Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"\n[done] elapsed {out['elapsed_s']:.0f}s")


def _write_md(out):
    b = out["baseline"]
    lines = []
    lines.append(f"# EURUSD per-symbol tune ({out['session']})")
    lines.append("")
    lines.append(f"- Started: {out['started_at']}")
    lines.append(f"- Elapsed: {out['elapsed_s']:.0f}s")
    lines.append(f"- Phase A iters: {len(out['phase_a'])}")
    lines.append(f"- Phase B iters: {len(out['phase_b'])}")
    lines.append(f"- Phase C iters: {len(out['phase_c_wf'])}")
    lines.append("")
    lines.append("## Baseline (live config, 180d)")
    lines.append(f"- trades={b['trades']}  pf={b['pf']:.2f}  pnl=${b['pnl']:+.0f}  "
                 f"wr={b['wr']:.1f}%  dd={b['dd']:.1f}%  avg_r={b['avg_r']:.2f}")
    lines.append("")

    # Top 10 Phase A+B
    combined = []
    for r in out["phase_a"] + out["phase_b"]:
        if r.get("result"):
            combined.append(r)
    combined.sort(key=lambda r: -r["result"]["pnl"])
    seen = set()
    uc = []
    for r in combined:
        k = (r["params"]["sl_mult"], r["params"]["trail_name"],
             r["params"]["pullback_atr"], r["params"]["pullback_wait"],
             r["params"]["vwap_buffer"], r["params"]["min_quality"],
             tuple(r["params"]["toxic_hours"]))
        if k in seen:
            continue
        seen.add(k)
        uc.append(r)

    lines.append("## Top 10 Phase A+B by PnL (180d)")
    lines.append("")
    lines.append("| Δ | pnl | pf | n | wr | dd | SL | trail | pb_atr | pb_wait | vwap | mQ | toxic |")
    lines.append("|---|-----|----|---|----|----|----|-------|--------|---------|------|----|-------|")
    for r in uc[:10]:
        rr = r["result"]; pp = r["params"]
        delta = rr["pnl"] - b["pnl"]
        lines.append(
            f"| ${delta:+.0f} | ${rr['pnl']:+.0f} | {rr['pf']:.2f} | {rr['trades']} | "
            f"{rr['wr']:.1f}% | {rr['dd']:.1f}% | {pp['sl_mult']} | {pp['trail_name']} | "
            f"{pp['pullback_atr']} | {pp['pullback_wait']} | {pp['vwap_buffer']} | "
            f"{pp['min_quality']} | {list(pp['toxic_hours'])} |")
    lines.append("")

    # Phase C WF table
    if out.get("phase_c_ranked"):
        lines.append("## Phase C — Walk-forward (top-3)")
        lines.append("")
        lines.append("| Δ | 180d-pf | WF-pf | WF-pos | PASS | SL | trail | pb | vwap | mQ | toxic |")
        lines.append("|---|---------|-------|--------|------|----|-------|----|------|----|-------|")
        for w in out["phase_c_ranked"]:
            pp = w["params"]; rr = w["result_180d"]
            lines.append(
                f"| ${w['delta_pnl']:+.0f} | {rr['pf']:.2f} | "
                f"{w['wf']['avg_pf']:.2f} | {w['wf']['pos_folds']}/5 | "
                f"{'YES' if w['wf_pass'] else 'no'} | "
                f"{pp['sl_mult']} | {pp['trail_name']} | "
                f"{pp['pullback_atr']}/{pp['pullback_wait']} | "
                f"{pp['vwap_buffer']} | {pp['min_quality']} | "
                f"{list(pp['toxic_hours'])} |")
        lines.append("")

    # Winner
    if out.get("winner"):
        w = out["winner"]; pp = w["params"]; rr = w["result_180d"]
        lines.append("## SHIP WINNER")
        lines.append("")
        lines.append(f"- **Δ PnL**: ${w['delta_pnl']:+.0f}")
        lines.append(f"- **WF**: avg_pf={w['wf']['avg_pf']:.2f}, "
                     f"pos={w['wf']['pos_folds']}/5")
        lines.append(f"- **180d**: pf={rr['pf']:.2f}, pnl=${rr['pnl']:+.0f}, "
                     f"n={rr['trades']}, wr={rr['wr']:.1f}%, dd={rr['dd']:.1f}%")
        lines.append("")
        lines.append("```python")
        lines.append(f"# Winning params for EURUSD")
        lines.append(f"SL_atr_mult     = {pp['sl_mult']}")
        lines.append(f"Trail           = '{pp['trail_name']}'")
        lines.append(f"Pullback ATR    = {pp['pullback_atr']}")
        lines.append(f"Pullback wait   = {pp['pullback_wait']}")
        lines.append(f"VWAP buffer     = {pp['vwap_buffer']}  "
                     f"({'disabled' if pp['vwap_buffer']==0.0 else 'ATR×'+str(pp['vwap_buffer'])})")
        lines.append(f"min_quality     = {pp['min_quality']}")
        lines.append(f"Toxic hours     = {list(pp['toxic_hours'])}")
        lines.append("```")
    else:
        lines.append("## SHIP WINNER")
        lines.append("")
        lines.append("**No candidate cleared Δ≥$50 AND WF≥3/5. Hold current config.**")

    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
