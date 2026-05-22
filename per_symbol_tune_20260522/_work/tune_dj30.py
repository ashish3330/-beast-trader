#!/usr/bin/env python3 -B
"""
Per-symbol tune for DJ30.r — 7-dim grid.

Phase A: per-dim scan (one dim at a time, others at default) → top-2 per dim.
Phase B: full cartesian of top-2's (2^7 = 128 combos).
Phase C: walk-forward top-5 from B on 3 folds (60d windows offset 0/60/120).

Ship rule: Δ ≥ +$100 vs baseline AND WF ≥ 3/5 folds positive.
Output: per_symbol_tune_20260522/DJ30.r.json + DJ30.r.md
READ-ONLY w.r.t. repo code. Monkey-patches in process only.
"""
import json, sys, time, types, inspect, math
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import config as _config
import backtest.v5_backtest as _bt_mod
from backtest.v5_backtest import DEFAULT_PARAMS

SYMBOL = "DJ30.r"
OUT_DIR = ROOT / "per_symbol_tune_20260522"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────── Dimension definitions ───────────
SL_GRID         = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
TRAIL_PROFILES = {
    "DEFAULT": [
        (8.0, 0.3, "trail"), (4.0, 0.5, "trail"), (2.0, 0.8, "trail"),
        (1.5, 0.7, "lock"),  (1.0, 0.4, "lock"),  (0.7, 0.2, "lock"),
        (0.5, 0.0, "be"),
    ],
    "TIGHT": [
        (6.0, 0.4, "trail"), (3.0, 0.6, "trail"),
        (1.5, 0.9, "lock"),  (1.0, 0.5, "lock"),  (0.5, 0.0, "be"),
    ],
    "LOOSE": [
        (10.0, 0.2, "trail"), (5.0, 0.4, "trail"), (2.5, 0.6, "trail"),
        (1.5, 0.5, "lock"),   (0.7, 0.0, "be"),
    ],
    "AGGR_RUN": [
        (15.0, 0.3, "trail"), (8.0, 0.5, "trail"),
        (3.0, 0.5, "lock"),   (1.0, 0.0, "be"),
    ],
    "DENSE_LOCK": [
        (5.0, 0.3, "trail"), (3.0, 0.5, "trail"), (2.0, 1.2, "lock"),
        (1.5, 0.8, "lock"), (1.0, 0.5, "lock"), (0.7, 0.3, "lock"),
        (0.5, 0.1, "be"),
    ],
    "FAST_BE": [
        (4.0, 0.4, "trail"), (2.0, 0.7, "lock"),
        (1.0, 0.5, "lock"),  (0.4, 0.0, "be"),
    ],
    "SLOW_RUN": [
        (12.0, 0.2, "trail"), (6.0, 0.4, "trail"), (3.0, 0.7, "trail"),
        (1.5, 0.6, "lock"),   (0.8, 0.0, "be"),
    ],
}
TRAIL_NAMES = list(TRAIL_PROFILES.keys())  # 7 named profiles

PB_ATR_GRID  = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
PB_WAIT_GRID = [3, 4, 5, 6, 8]
VWAP_GRID    = [0.0, 0.3, 0.5, 0.7, 1.0]   # 0.0 = disabled
MQ_GRID      = [28, 30, 33, 35, 38, 40]
TOXIC_GRID = [
    frozenset({7, 18}),
    frozenset({7}),
    frozenset({18}),
    frozenset({7, 9, 18}),
    frozenset(),
]

# ─────────── VWAP source-patch infrastructure ───────────
_BT_SRC = inspect.getsource(_bt_mod)
_VWAP_LINE = 'atr_buf = float(ind["at"][bi]) * 0.5'
assert _VWAP_LINE in _BT_SRC, "VWAP line not found in v5_backtest — script must be updated"

# Cache: vwap_buf → backtest_symbol function
_BT_FN_CACHE = {}

def _make_bt(vwap_buf):
    key = round(float(vwap_buf), 4)
    if key in _BT_FN_CACHE:
        return _BT_FN_CACHE[key]
    if vwap_buf == 0.0:
        replacement = 'atr_buf = float(ind["at"][bi]) * 9999.0'  # effectively disable
    else:
        replacement = f'atr_buf = float(ind["at"][bi]) * {vwap_buf}'
    patched = _BT_SRC.replace(_VWAP_LINE, replacement)
    mod = types.ModuleType(f"_bt_patched_vwap_{key}")
    mod.__file__ = _bt_mod.__file__
    exec(compile(patched, _bt_mod.__file__, 'exec'), mod.__dict__)
    fn = mod.backtest_symbol
    _BT_FN_CACHE[key] = fn
    return fn


# ─────────── Single-trial runner ───────────
def run_trial(sl, trail_name, pb_atr, pb_wait, vwap_buf, mq, toxic, days=180,
              days_offset_end=0):
    """Run one BT trial with given hyperparams. Returns metrics dict or None.

    days_offset_end: skip the last N days (for walk-forward training windows).
    Implemented by patching v5_backtest.load_data is heavy — instead we use
    days parameter; for walk-forward we run 60d windows by passing days=60
    and patching load_data to shift cutoff. We KEEP IT SIMPLE: use `days`
    only for in-sample. Walk-forward implemented separately.
    """
    # Monkey-patch shared state
    _bt_mod.TOXIC_HOURS = set(toxic)
    _config.PULLBACK_ATR_RETRACE = float(pb_atr)
    _config.PULLBACK_MAX_WAIT_BARS = int(pb_wait)

    bt_fn = _make_bt(vwap_buf)

    params = dict(DEFAULT_PARAMS)
    params["sl_atr_mult"] = sl
    params["force_trail"] = TRAIL_PROFILES[trail_name]
    # min_quality is a regime-dict; use mq as a flat threshold for all regimes
    params["min_quality"] = {
        "trending": mq, "ranging": mq, "volatile": mq, "low_vol": mq
    }
    try:
        r = bt_fn(SYMBOL, days=days, params=params, verbose=False)
    except Exception as e:
        return None
    if r is None or r.get("trades", 0) < 5:
        return None
    return {
        "pnl": float(r["pnl"]),
        "pf": float(r["pf"]),
        "wr": float(r["wr"]),
        "trades": int(r["trades"]),
        "dd": float(r["dd"]),
        "avg_r": float(r.get("avg_r", 0)),
    }


# ─────────── Walk-forward over 5 folds (180d split into 5 × 36d) ───────────
# We approximate WF by running 5 different `days` windows that shift forward.
# v5_backtest only supports trimming-from-now via `days`. To get OLDER windows,
# we patch load_data to drop the most recent N days.
import pickle
import pandas as pd
from backtest.v5_backtest import CACHE, ALL_SYMBOLS

_orig_load_data = _bt_mod.load_data

def _wf_load_data_factory(end_offset_days):
    """Build a replacement load_data that returns data UP TO max(time)-offset."""
    def _load(symbol, days=90):
        meta = ALL_SYMBOLS[symbol]
        path = CACHE / meta["cache"]
        if not path.exists():
            return None
        df = pickle.load(open(path, "rb"))
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        end = df["time"].max() - pd.Timedelta(days=end_offset_days)
        if days:
            start = end - pd.Timedelta(days=days)
            df = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
        return df
    return _load


def run_wf_trial(sl, trail_name, pb_atr, pb_wait, vwap_buf, mq, toxic,
                 fold_window_days=36, n_folds=5):
    """Walk-forward: 5 non-overlapping 36d folds covering ~180d."""
    fold_results = []
    for fold_i in range(n_folds):
        # Fold 0 = most recent 36d, fold 1 = days 36-72 ago, etc.
        offset = fold_i * fold_window_days
        # Patch load_data
        _bt_mod.load_data = _wf_load_data_factory(offset)
        try:
            r = run_trial(sl, trail_name, pb_atr, pb_wait, vwap_buf, mq, toxic,
                          days=fold_window_days)
        finally:
            _bt_mod.load_data = _orig_load_data
        fold_results.append(r)
    # Count positive folds
    pos = sum(1 for r in fold_results if r and r["pnl"] > 0)
    total_pnl = sum(r["pnl"] for r in fold_results if r)
    return {"folds": fold_results, "positive_folds": pos, "total_pnl_5fold": total_pnl}


# ─────────── Composite score ───────────
def score(r):
    """Composite quality: PnL × PF / sqrt(max(DD, 1)). Higher = better."""
    if not r or r.get("trades", 0) < 10:
        return -1e9
    pf = min(r["pf"], 20.0)
    dd = max(r["dd"], 1.0)
    return r["pnl"] * pf / math.sqrt(dd)


# ─────────── BASELINE ───────────
def baseline():
    """Baseline = defaults + current per-symbol live toxic hours {7, 18}."""
    return run_trial(
        sl=DEFAULT_PARAMS["sl_atr_mult"],
        trail_name="DEFAULT",
        pb_atr=0.8,
        pb_wait=5,
        vwap_buf=0.5,
        mq=45,  # roughly equivalent to DEFAULT_PARAMS["min_quality"] regime avg
        toxic=frozenset({7, 18}),
        days=180,
    )


# ─────────── Phase A: per-dim scan ───────────
def phase_a():
    """One dim at a time, others at baseline defaults. Top-2 per dim → next phase."""
    DEF = dict(
        sl=1.5, trail_name="DEFAULT", pb_atr=0.8, pb_wait=5, vwap_buf=0.5,
        mq=33, toxic=frozenset({7, 18}),
    )
    log = {}
    iters = 0

    # SL
    rs = []
    for v in SL_GRID:
        kw = dict(DEF); kw["sl"] = v
        r = run_trial(**kw, days=180); iters += 1
        rs.append((v, r, score(r)))
    log["sl"] = rs

    # Trail
    rs = []
    for v in TRAIL_NAMES:
        kw = dict(DEF); kw["trail_name"] = v
        r = run_trial(**kw, days=180); iters += 1
        rs.append((v, r, score(r)))
    log["trail"] = rs

    # PB ATR
    rs = []
    for v in PB_ATR_GRID:
        kw = dict(DEF); kw["pb_atr"] = v
        r = run_trial(**kw, days=180); iters += 1
        rs.append((v, r, score(r)))
    log["pb_atr"] = rs

    # PB wait
    rs = []
    for v in PB_WAIT_GRID:
        kw = dict(DEF); kw["pb_wait"] = v
        r = run_trial(**kw, days=180); iters += 1
        rs.append((v, r, score(r)))
    log["pb_wait"] = rs

    # VWAP buf
    rs = []
    for v in VWAP_GRID:
        kw = dict(DEF); kw["vwap_buf"] = v
        r = run_trial(**kw, days=180); iters += 1
        rs.append((v, r, score(r)))
    log["vwap_buf"] = rs

    # min_quality
    rs = []
    for v in MQ_GRID:
        kw = dict(DEF); kw["mq"] = v
        r = run_trial(**kw, days=180); iters += 1
        rs.append((v, r, score(r)))
    log["mq"] = rs

    # Toxic
    rs = []
    for v in TOXIC_GRID:
        kw = dict(DEF); kw["toxic"] = v
        r = run_trial(**kw, days=180); iters += 1
        rs.append((sorted(v), r, score(r)))
    log["toxic"] = rs

    return log, iters


def top2_from_log(log):
    """For each dim, return the 2 values with the highest score (skip None)."""
    out = {}
    for dim, rs in log.items():
        # rs: list of (val, result_dict, score)
        viable = [t for t in rs if t[1] is not None]
        viable.sort(key=lambda t: -t[2])
        out[dim] = [t[0] for t in viable[:2]] if viable else []
    return out


# ─────────── Phase B: top-2^7 cartesian ───────────
def phase_b(top2):
    """Full cartesian of top-2's across dims. Default = 2^7 = 128 combos."""
    sl_v   = top2.get("sl", [1.5])
    trl_v  = top2.get("trail", ["DEFAULT"])
    pba_v  = top2.get("pb_atr", [0.8])
    pbw_v  = top2.get("pb_wait", [5])
    vw_v   = top2.get("vwap_buf", [0.5])
    mq_v   = top2.get("mq", [33])
    tox_v  = top2.get("toxic", [frozenset({7, 18})])
    # Convert toxic items: top2 stored as list for toxic
    tox_v = [frozenset(x) if not isinstance(x, frozenset) else x for x in tox_v]

    combos = list(product(sl_v, trl_v, pba_v, pbw_v, vw_v, mq_v, tox_v))
    results = []
    for sl, trl, pba, pbw, vw, mq, tox in combos:
        r = run_trial(sl, trl, pba, pbw, vw, mq, tox, days=180)
        results.append({
            "sl": sl, "trail": trl, "pb_atr": pba, "pb_wait": pbw,
            "vwap_buf": vw, "mq": mq, "toxic": sorted(tox),
            "result": r, "score": score(r),
        })
    results.sort(key=lambda d: -d["score"])
    return results, len(combos)


# ─────────── Phase C: walk-forward top-5 ───────────
def phase_c(top5):
    """5-fold walk-forward on top-5 candidates."""
    out = []
    for cand in top5:
        wf = run_wf_trial(
            sl=cand["sl"], trail_name=cand["trail"],
            pb_atr=cand["pb_atr"], pb_wait=cand["pb_wait"],
            vwap_buf=cand["vwap_buf"], mq=cand["mq"],
            toxic=frozenset(cand["toxic"]),
            fold_window_days=36, n_folds=5,
        )
        out.append({"cand": cand, "wf": wf})
    return out


# ─────────── Main pipeline ───────────
def main():
    t0 = time.time()

    print(f"\n=== TUNE {SYMBOL} 180d ===")
    print("Baseline (defaults + per-sym toxic {7,18})...")
    base = baseline()
    if base:
        print(f"  baseline: pnl=${base['pnl']:.2f} pf={base['pf']} wr={base['wr']}% n={base['trades']} dd={base['dd']}%")
    base_pnl = base["pnl"] if base else 0.0
    print(f"  [{time.time()-t0:.1f}s] baseline done")

    print("\n--- Phase A: per-dim scan ---")
    log_a, n_a = phase_a()
    print(f"  Phase A: {n_a} trials in {time.time()-t0:.1f}s")
    for dim, rs in log_a.items():
        viable = [t for t in rs if t[1] is not None]
        viable.sort(key=lambda t: -t[2])
        if not viable:
            print(f"  [{dim}] no viable")
            continue
        msg = f"  [{dim}] top3:"
        for v, r, s in viable[:3]:
            msg += f" {v}=${r['pnl']:.0f}(pf{r['pf']:.2f})"
        print(msg)

    top2 = top2_from_log(log_a)
    print(f"\n  top2 picks: {top2}")

    print(f"\n--- Phase B: cartesian top-2^7 ---")
    res_b, n_b = phase_b(top2)
    print(f"  Phase B: {n_b} combos in {time.time()-t0:.1f}s")
    print("  Top 5 Phase B:")
    for i, r in enumerate(res_b[:5], 1):
        if r["result"]:
            print(f"   #{i} pnl=${r['result']['pnl']:.0f} pf={r['result']['pf']} wr={r['result']['wr']}% "
                  f"n={r['result']['trades']} sl={r['sl']} trl={r['trail']} pba={r['pb_atr']} "
                  f"pbw={r['pb_wait']} vw={r['vwap_buf']} mq={r['mq']} tox={r['toxic']}")

    top5_b = [r for r in res_b if r["result"] is not None][:5]

    print(f"\n--- Phase C: walk-forward top-5 (5 × 36d folds) ---")
    res_c = phase_c(top5_b)
    print(f"  Phase C: {len(res_c) * 5} fold-trials in {time.time()-t0:.1f}s")

    # Pick winner: best in-sample with WF >= 3/5 pos and Δpnl >= +$100
    winner = None
    candidates_ranked = []
    for entry in res_c:
        cand = entry["cand"]
        wf = entry["wf"]
        delta = cand["result"]["pnl"] - base_pnl
        ship = (delta >= 100.0) and (wf["positive_folds"] >= 3)
        candidates_ranked.append({
            "cand": cand, "wf": wf,
            "delta_pnl": delta, "ship_eligible": ship,
        })
        # WF positive folds
    # Winner = top ship-eligible by in-sample score
    eligible = [c for c in candidates_ranked if c["ship_eligible"]]
    if eligible:
        winner = max(eligible, key=lambda c: c["cand"]["score"])
        print(f"\n  WINNER: pnl=${winner['cand']['result']['pnl']:.0f} "
              f"Δ${winner['delta_pnl']:.0f} WF={winner['wf']['positive_folds']}/5")
        print(f"    sl={winner['cand']['sl']} trail={winner['cand']['trail']} "
              f"pb={winner['cand']['pb_atr']}/{winner['cand']['pb_wait']} "
              f"vwap={winner['cand']['vwap_buf']} mq={winner['cand']['mq']} "
              f"toxic={winner['cand']['toxic']}")
    else:
        print("\n  NO ELIGIBLE WINNER (Δ < $100 or WF < 3/5). KEEP DEFAULTS.")

    elapsed = time.time() - t0

    # ── Write JSON output ──
    out = {
        "symbol": SYMBOL,
        "days": 180,
        "elapsed_s": round(elapsed, 1),
        "baseline": base,
        "baseline_config": {
            "sl": 1.5, "trail": "DEFAULT", "pb_atr": 0.8, "pb_wait": 5,
            "vwap_buf": 0.5, "mq": 45, "toxic": [7, 18],
        },
        "phase_a": {
            dim: [
                {"val": (sorted(v) if isinstance(v, frozenset) else v),
                 "result": r, "score": s}
                for v, r, s in rs
            ]
            for dim, rs in log_a.items()
        },
        "phase_a_top2": {dim: [(sorted(x) if isinstance(x, frozenset) else x)
                                for x in vals]
                          for dim, vals in top2.items()},
        "phase_b_top10": [
            {"sl": r["sl"], "trail": r["trail"], "pb_atr": r["pb_atr"],
             "pb_wait": r["pb_wait"], "vwap_buf": r["vwap_buf"],
             "mq": r["mq"], "toxic": list(r["toxic"]),
             "result": r["result"], "score": round(r["score"], 2)}
            for r in res_b[:10]
        ],
        "phase_c": [
            {"cand": {
                "sl": c["cand"]["sl"], "trail": c["cand"]["trail"],
                "pb_atr": c["cand"]["pb_atr"], "pb_wait": c["cand"]["pb_wait"],
                "vwap_buf": c["cand"]["vwap_buf"], "mq": c["cand"]["mq"],
                "toxic": list(c["cand"]["toxic"]),
                "in_sample": c["cand"]["result"],
                "score": round(c["cand"]["score"], 2),
             },
             "delta_pnl_vs_baseline": round(c["delta_pnl"], 2),
             "wf_positive_folds": c["wf"]["positive_folds"],
             "wf_total_pnl": round(c["wf"]["total_pnl_5fold"], 2),
             "wf_folds": c["wf"]["folds"],
             "ship_eligible": c["ship_eligible"]}
            for c in candidates_ranked
        ],
        "winner": ({
            "sl": winner["cand"]["sl"], "trail": winner["cand"]["trail"],
            "pb_atr": winner["cand"]["pb_atr"], "pb_wait": winner["cand"]["pb_wait"],
            "vwap_buf": winner["cand"]["vwap_buf"], "mq": winner["cand"]["mq"],
            "toxic": list(winner["cand"]["toxic"]),
            "in_sample": winner["cand"]["result"],
            "delta_pnl_vs_baseline": round(winner["delta_pnl"], 2),
            "wf_positive_folds": winner["wf"]["positive_folds"],
            "wf_total_pnl": round(winner["wf"]["total_pnl_5fold"], 2),
        } if winner else None),
        "ship": winner is not None,
        "ship_rule": "Delta_PnL >= +$100 AND WF >= 3/5 folds positive",
    }
    out_json = OUT_DIR / f"{SYMBOL}.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))

    # ── MD report ──
    md_lines = [
        f"# {SYMBOL} per-symbol tune (2026-05-22)",
        "",
        f"- Days: 180",
        f"- Elapsed: {elapsed:.1f}s",
        f"- Baseline: ${base['pnl']:.2f} PF {base['pf']} WR {base['wr']}% "
        f"trades {base['trades']} DD {base['dd']}%" if base else "Baseline: N/A",
        "",
        "## Phase A — per-dim top 3",
        "",
    ]
    for dim, rs in log_a.items():
        viable = [t for t in rs if t[1] is not None]
        viable.sort(key=lambda t: -t[2])
        md_lines.append(f"### {dim}")
        for v, r, s in viable[:3]:
            md_lines.append(f"- `{v}` → pnl=${r['pnl']:.0f} PF={r['pf']} "
                            f"WR={r['wr']}% n={r['trades']} DD={r['dd']}% score={s:.0f}")
        md_lines.append("")
    md_lines.append("## Phase B — top 10 cartesian")
    for i, r in enumerate(res_b[:10], 1):
        if not r["result"]:
            continue
        md_lines.append(
            f"{i}. sl={r['sl']} trl={r['trail']} pb={r['pb_atr']}/{r['pb_wait']} "
            f"vw={r['vwap_buf']} mq={r['mq']} tox={r['toxic']} → "
            f"${r['result']['pnl']:.0f} PF {r['result']['pf']} "
            f"WR {r['result']['wr']}% n={r['result']['trades']} "
            f"DD {r['result']['dd']}% score={r['score']:.0f}"
        )
    md_lines.append("")
    md_lines.append("## Phase C — walk-forward (5 × 36d folds)")
    for c in candidates_ranked:
        cd = c["cand"]
        md_lines.append(
            f"- sl={cd['sl']} trl={cd['trail']} pb={cd['pb_atr']}/{cd['pb_wait']} "
            f"vw={cd['vwap_buf']} mq={cd['mq']} tox={cd['toxic']} → "
            f"in-sample ${cd['result']['pnl']:.0f}, Δ${c['delta_pnl']:.0f}, "
            f"WF {c['wf']['positive_folds']}/5, ship={c['ship_eligible']}"
        )
    md_lines.append("")
    md_lines.append("## Ship decision")
    md_lines.append(f"Rule: Δ ≥ +$100 vs baseline AND WF ≥ 3/5 positive folds.")
    if winner:
        wc = winner["cand"]
        md_lines.append(
            f"**SHIP** — sl={wc['sl']} trail={wc['trail']} "
            f"pb={wc['pb_atr']}/{wc['pb_wait']} vwap={wc['vwap_buf']} "
            f"mq={wc['mq']} toxic={wc['toxic']} → "
            f"in-sample ${wc['result']['pnl']:.0f} "
            f"(Δ${winner['delta_pnl']:.0f}), "
            f"WF {winner['wf']['positive_folds']}/5 positive."
        )
    else:
        md_lines.append("**NO SHIP** — keep baseline. No candidate cleared both gates.")

    out_md = OUT_DIR / f"{SYMBOL}.md"
    out_md.write_text("\n".join(md_lines))

    print(f"\n[done in {elapsed:.1f}s]")
    print(f"  JSON: {out_json}")
    print(f"  MD:   {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
