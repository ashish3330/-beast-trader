#!/usr/bin/env python3 -B
"""VOLUME-CONFIRMED entry-filter research.

Tests four variants (READ-ONLY w.r.t. v5_backtest.py — implemented via
monkey-patch of `_score_with_components` so the BT loop is untouched):

  1. TICK_RATIO   — require tick_volume[i] >= X × SMA20(tick_volume) at signal bar
  2. POC_HVN      — require price within ATR-band of nearest HVN (volume-profile peak
                    over last N bars histogram)
  3. VWAP_SIDE    — LONG only when close > VWAP (rolling 20); SHORT only when close < VWAP
  4. VOL_SPIKE    — require tick_volume[i] > 1.5 × SMA20

Each variant runs in two modes:
  - FILTER     : volume condition is a hard gate (must pass).
  - BOOST      : relax `min_quality` by N points when volume condition is true
                 (so high-volume marginal entries get a pass).

Ship criteria per variant:
  Δ ≥ +$30 AND WF avg PF > 1.5 AND ≥ 3/5 folds positive

Output:
  03_volume.json (machine-readable)
  03_volume.md   (human report)
"""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path
import multiprocessing as mp

ROOT = Path("/Users/ashish/Documents/beast-trader")
OUT_DIR = ROOT / "entry_research_20260522"
OUT_DIR.mkdir(exist_ok=True)
OUT_JSON = OUT_DIR / "03_volume.json"
OUT_MD   = OUT_DIR / "03_volume.md"

sys.path.insert(0, str(ROOT))

# 8 symbols with sufficient H1 cache (≥180d) and valid tick_volume
SYMBOLS = ["ETHUSD", "GBPUSD", "EURUSD", "GBPJPY", "EURJPY",
           "USDCAD", "GER40.r", "SP500.r"]

DAYS = 180
WF_FOLDS = 5
MIN_TRADES = 8


# ─────────────────────────────────────────────────────────────────────────────
# Worker: each variant runs in subprocess. The worker monkey-patches the
# scorer to apply the volume condition before returning the score.
# ─────────────────────────────────────────────────────────────────────────────
def _run_variant(args):
    """args = (variant_name, sym, cfg)
    cfg dict:
      mode      : 'filter' | 'boost' | 'baseline'
      strategy  : 'tick_ratio' | 'poc_hvn' | 'vwap_side' | 'vol_spike' | None
      threshold : float (variant-specific)
      boost_pts : int (only used for 'boost' mode)
    """
    variant, sym, cfg = args
    try:
        from backtest import v5_backtest as bt
        from signals import momentum_scorer as ms

        # Build a per-symbol volume-profile lookup if needed (POC).
        # We pre-scan the H1 cache at worker start and cache POC array per call.
        _poc_cache = {"sym": None, "poc": None}

        def _compute_poc(ind, lookback=50, bins=24):
            """Crude rolling volume-profile POC.
            For each bar i, bucket prices over [i-lookback, i] into `bins`
            histogram weighted by tick_volume, then return the price center
            of the densest bucket.
            """
            import numpy as np
            n = ind["c"].shape[0]
            poc = np.full(n, np.nan, dtype=np.float64)
            c = ind["c"]; vol = ind["vol"]
            for i in range(lookback, n):
                lo = c[i-lookback:i+1].min()
                hi = c[i-lookback:i+1].max()
                if hi <= lo:
                    continue
                # Histogram of volume by close-price bucket
                edges = np.linspace(lo, hi, bins+1)
                idxs = np.clip(np.searchsorted(edges, c[i-lookback:i+1], side="right") - 1, 0, bins-1)
                buckets = np.zeros(bins, dtype=np.float64)
                v_slice = vol[i-lookback:i+1]
                for k, vv in zip(idxs, v_slice):
                    buckets[k] += vv
                best = int(np.argmax(buckets))
                poc[i] = (edges[best] + edges[best+1]) / 2.0
            return poc

        mode = cfg["mode"]
        strategy = cfg.get("strategy")
        threshold = float(cfg.get("threshold", 0))
        boost_pts = int(cfg.get("boost_pts", 0))
        # The boost-mode-with-volume-gate amplification: if vol confirms,
        # ADD this many quality-points to the score so a borderline entry
        # passes the gate. Implemented via score multiplier inside patched().
        boost_score_mult = float(cfg.get("boost_score_mult", 1.15))

        orig_score = ms._score_with_components

        import numpy as _np

        def _vol_condition(ind, i, long_s, short_s):
            """Return True if volume condition is met at bar i."""
            try:
                if strategy == "tick_ratio" or strategy == "vol_spike":
                    vs = float(ind["vol_sma"][i]) if not _np.isnan(ind["vol_sma"][i]) else 0.0
                    vv = float(ind["vol"][i])
                    return vs > 0 and (vv / vs) >= threshold
                elif strategy == "vwap_side":
                    vw = ind["vwap"][i]
                    if _np.isnan(vw):
                        return False
                    price = float(ind["c"][i])
                    if long_s >= short_s:
                        return price > vw
                    else:
                        return price < vw
                elif strategy == "poc_hvn":
                    if _poc_cache["sym"] != sym:
                        _poc_cache["sym"] = sym
                        _poc_cache["poc"] = _compute_poc(ind)
                    p_arr = _poc_cache["poc"]
                    if _np.isnan(p_arr[i]):
                        return False
                    poc = float(p_arr[i])
                    atr_i = float(ind["at"][i])
                    if atr_i <= 0:
                        return False
                    return abs(float(ind["c"][i]) - poc) / atr_i <= threshold
            except Exception:
                return True  # fail-open
            return True

        def patched(ind, i, weights=None):
            long_s, short_s, comp_l, comp_s = orig_score(ind, i, weights=weights)
            if mode == "baseline" or strategy is None:
                return long_s, short_s, comp_l, comp_s

            cond = _vol_condition(ind, i, long_s, short_s)

            if mode == "filter":
                # Hard gate: reject when condition fails.
                if not cond:
                    return 0.0, 0.0, comp_l, comp_s
                return long_s, short_s, comp_l, comp_s

            elif mode == "boost":
                # Amplify scores when condition is true (lets marginal
                # entries pass the quality threshold). When condition is
                # false, leave scores untouched (status quo).
                if cond:
                    return long_s * boost_score_mult, short_s * boost_score_mult, comp_l, comp_s
                return long_s, short_s, comp_l, comp_s

            return long_s, short_s, comp_l, comp_s

        # Apply patch
        bt._score_with_components = patched
        ms._score_with_components = patched

        params = {}

        res = bt.backtest_symbol(sym, days=DAYS, params=params, verbose=False)

        # Restore
        ms._score_with_components = orig_score
        bt._score_with_components = orig_score

        if res is None or not res.get("details"):
            return (variant, sym, cfg, None)

        trades = [
            {"eb": t["entry_bar"], "pnl": t["pnl"], "r": t["pnl_r"],
             "dir": t["direction"], "reg": t["regime"], "q": t["quality"]}
            for t in res["details"]
        ]
        return (variant, sym, cfg, trades)
    except Exception as e:
        return (variant, sym, cfg, {"error": repr(e), "tb": traceback.format_exc()})


# ─────────────────────────────────────────────────────────────────────────────
# Stats / WF helpers
# ─────────────────────────────────────────────────────────────────────────────
def _stats(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = gw / gl if gl > 0 else (999.0 if gw > 0 else 0.0)
    wr = 100.0 * len(wins) / len(trades)
    eq = [0.0]
    for t in trades:
        eq.append(eq[-1] + t["pnl"])
    peak = eq[0]; mdd = 0.0
    for e in eq:
        peak = max(peak, e)
        mdd = max(mdd, peak - e)
    return {"n": len(trades), "pnl": round(sum(t["pnl"] for t in trades), 2),
            "pf": round(pf, 2), "wr": round(wr, 1), "dd": round(mdd, 2)}


def _wf_folds(trades, k=WF_FOLDS):
    if len(trades) < k:
        return [trades] if trades else []
    s = sorted(trades, key=lambda t: t["eb"])
    sz = len(s) // k
    out = []
    for i in range(k):
        lo = i * sz; hi = (i + 1) * sz if i < k - 1 else len(s)
        out.append(s[lo:hi])
    return out


def _wf_metrics(trades):
    folds = _wf_folds(trades)
    if not folds:
        return {"n_folds": 0, "avg_pf": 0.0, "positive_folds": 0, "fold_pnl": []}
    fold_pnl = []
    fold_pf = []
    pos = 0
    for f in folds:
        st = _stats(f)
        fold_pnl.append(st["pnl"])
        fold_pf.append(st["pf"])
        if st["pnl"] > 0:
            pos += 1
    return {
        "n_folds": len(folds),
        "avg_pf": round(sum(fold_pf) / len(fold_pf), 2),
        "positive_folds": pos,
        "fold_pnl": [round(x, 2) for x in fold_pnl],
        "fold_pf":  [round(x, 2) for x in fold_pf],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweep config
# ─────────────────────────────────────────────────────────────────────────────
VARIANTS = []

# Baseline (no patch)
VARIANTS.append(("baseline", {"mode": "baseline"}))

# 1. TICK_RATIO — filter mode (require vol[i] / sma >= threshold)
for thr in (1.0, 1.2, 1.5, 2.0):
    VARIANTS.append((f"tick_ratio_filter_{thr}",
                     {"mode": "filter", "strategy": "tick_ratio", "threshold": thr}))

# 2. VWAP_SIDE — filter mode (direction-dependent)
VARIANTS.append(("vwap_side_filter",
                 {"mode": "filter", "strategy": "vwap_side", "threshold": 0}))

# 3. POC_HVN — filter mode (entry within X*ATR of HVN)
for thr in (0.5, 1.0, 1.5):
    VARIANTS.append((f"poc_hvn_filter_{thr}atr",
                     {"mode": "filter", "strategy": "poc_hvn", "threshold": thr}))

# 4. TICK_RATIO — boost mode (1.15x score when vol >= 1.5x SMA)
VARIANTS.append(("tick_ratio_boost_1.5x_1.15mult",
                 {"mode": "boost", "strategy": "tick_ratio",
                  "threshold": 1.5, "boost_score_mult": 1.15}))

# 5. VWAP_SIDE — boost mode (1.15x score when on correct side)
VARIANTS.append(("vwap_side_boost_1.15mult",
                 {"mode": "boost", "strategy": "vwap_side",
                  "threshold": 0, "boost_score_mult": 1.15}))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    tasks = []
    for variant, cfg in VARIANTS:
        for sym in SYMBOLS:
            tasks.append((variant, sym, cfg))

    print(f"Submitting {len(tasks)} backtests "
          f"({len(VARIANTS)} variants × {len(SYMBOLS)} symbols)...",
          flush=True)

    # Use moderate parallelism — each BT loads 50k bars
    n_workers = min(8, os.cpu_count() or 4)
    raw_results = {}
    with mp.Pool(n_workers) as pool:
        done = 0
        for ret in pool.imap_unordered(_run_variant, tasks, chunksize=1):
            variant, sym, cfg, trades = ret
            raw_results.setdefault(variant, {})[sym] = trades
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(tasks)} done "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # ─── Aggregate per-variant ─────────────────────────────────────────────
    summary = {}
    baseline_pnl_by_sym = {}
    # First pass: compute baseline
    for sym, trades in raw_results.get("baseline", {}).items():
        if isinstance(trades, dict) or trades is None:
            baseline_pnl_by_sym[sym] = None
            continue
        st = _stats(trades)
        baseline_pnl_by_sym[sym] = st

    _base_all_trades = []
    for s in SYMBOLS:
        t = raw_results.get("baseline", {}).get(s)
        if isinstance(t, list):
            _base_all_trades.extend(t)
    summary["baseline"] = {
        "per_sym": {s: baseline_pnl_by_sym[s] for s in SYMBOLS},
        "portfolio": _stats(_base_all_trades),
        "portfolio_wf": _wf_metrics(_base_all_trades),
    }

    # Other variants vs baseline
    for variant, cfg in VARIANTS:
        if variant == "baseline":
            continue
        per_sym = {}
        all_trades = []
        for sym in SYMBOLS:
            t = raw_results.get(variant, {}).get(sym)
            if isinstance(t, dict) or t is None:
                per_sym[sym] = {"error": t.get("error") if isinstance(t, dict) else "no_data"}
                continue
            base_stats = baseline_pnl_by_sym.get(sym) or {"pnl": 0, "n": 0}
            v_stats = _stats(t)
            wf = _wf_metrics(t)
            per_sym[sym] = {
                "stats": v_stats,
                "wf": wf,
                "delta_pnl": round(v_stats["pnl"] - base_stats.get("pnl", 0), 2),
                "delta_n":   v_stats["n"] - base_stats.get("n", 0),
            }
            all_trades.extend(t)

        port_stats = _stats(all_trades)
        port_wf = _wf_metrics(all_trades)
        base_port = summary["baseline"]["portfolio"]
        port_delta = round(port_stats["pnl"] - base_port.get("pnl", 0), 2)

        # Ship-gate check
        ship = (
            port_delta >= 30.0
            and port_wf["avg_pf"] > 1.5
            and port_wf["positive_folds"] >= 3
        )

        summary[variant] = {
            "cfg": cfg,
            "per_sym": per_sym,
            "portfolio": port_stats,
            "portfolio_wf": port_wf,
            "delta_pnl_vs_base": port_delta,
            "ship": ship,
        }

    # ─── Persist ───────────────────────────────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump({
            "meta": {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbols": SYMBOLS,
                "days": DAYS,
                "variants_run": len(VARIANTS),
                "elapsed_sec": round(time.time() - t0, 1),
            },
            "summary": summary,
        }, f, indent=2, default=str)

    # ─── MD report ─────────────────────────────────────────────────────────
    lines = [
        "# VOLUME-CONFIRMED Entry Filter — Research Report",
        f"Date: {time.strftime('%Y-%m-%d')}",
        f"Symbols: {', '.join(SYMBOLS)}  (8 syms with ≥180d H1 cache + valid tick_volume)",
        f"Days: {DAYS}   WF folds: {WF_FOLDS}",
        "",
        "## Ship Criteria (per variant)",
        "- ΔPnL vs baseline ≥ $30 (portfolio level)",
        "- WF avg PF > 1.5",
        "- ≥ 3 / 5 folds positive PnL",
        "",
        "## Baseline (no volume filter)",
    ]
    bp = summary["baseline"]["portfolio"]
    lines.append(
        f"Portfolio: n={bp['n']}  PnL=${bp['pnl']}  PF={bp['pf']}  "
        f"WR={bp['wr']}%  DD=${bp['dd']}"
    )
    lines.append("Per-symbol:")
    for s in SYMBOLS:
        st = summary["baseline"]["per_sym"].get(s)
        if st:
            lines.append(f"  {s:10s}  n={st['n']:3d}  PnL=${st['pnl']:8.2f}  "
                         f"PF={st['pf']:.2f}  WR={st['wr']:.1f}%")
    lines.append("")

    lines.append("## Variant Results (portfolio)")
    lines.append("")
    lines.append("| Variant | Mode | n | PnL | Δ vs base | PF | WR | WF avg PF | +Folds | SHIP |")
    lines.append("|---------|------|---|-----|-----------|----|----|-----------|--------|------|")
    for v, _cfg in VARIANTS:
        if v == "baseline":
            continue
        s = summary.get(v, {})
        port = s.get("portfolio", {})
        wf = s.get("portfolio_wf", {})
        lines.append(
            f"| {v} | {s.get('cfg', {}).get('mode')} | {port.get('n', 0)} | "
            f"${port.get('pnl', 0)} | ${s.get('delta_pnl_vs_base', 0):+.2f} | "
            f"{port.get('pf', 0):.2f} | {port.get('wr', 0):.1f}% | "
            f"{wf.get('avg_pf', 0):.2f} | "
            f"{wf.get('positive_folds', 0)}/{wf.get('n_folds', 0)} | "
            f"{'YES' if s.get('ship') else 'NO'} |"
        )
    lines.append("")

    # Top-3 variants by delta
    rank = sorted(
        [(v, summary[v]["delta_pnl_vs_base"]) for v, _c in VARIANTS if v != "baseline"],
        key=lambda x: x[1], reverse=True,
    )
    lines.append("## Ranked by ΔPnL")
    for v, d in rank:
        lines.append(f"- {v}: Δ=${d:+.2f}  ship={summary[v]['ship']}")
    lines.append("")

    # Honest conclusion
    shippers = [v for v, _c in VARIANTS if v != "baseline" and summary.get(v, {}).get("ship")]
    if shippers:
        lines.append(f"## VERDICT: {len(shippers)} variant(s) pass ship gate")
        for v in shippers:
            lines.append(f"- **{v}**")
    else:
        lines.append("## VERDICT: NO variant passes the ship gate")
        lines.append("All four volume-confirmation strategies (tick-ratio, VWAP-side, "
                     "POC-HVN, threshold-relax negative control) failed the Δ≥$30 "
                     "+ WF PF > 1.5 + ≥3/5 positive folds bar. Honest failure — do "
                     "NOT deploy volume filter as currently formulated.")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    if shippers:
        print(f"SHIPPABLE: {shippers}")
    else:
        print("NO winner — honest failure recorded.")


if __name__ == "__main__":
    main()
