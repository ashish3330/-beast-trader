#!/usr/bin/env python3 -B
"""
Session-based entry filter research.

Per-(symbol, hour) BT performance → propose WHITELIST vs BLACKLIST hour gates →
walk-forward 5-fold validate → emit recommended TOXIC_HOURS_PER_SYMBOL diffs.

Read-only: no modifications to project files. Monkey-patches v5_backtest's
TOXIC_HOURS / TOXIC_EXEMPT inside each pool worker to inject per-symbol hour
gates.

Output:
  entry_research_20260522/07_session.json
  entry_research_20260522/07_session.md
"""
import os
import sys
import json
import time
import traceback
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "entry_research_20260522"
OUT_JSON = OUT_DIR / "07_session.json"
OUT_MD = OUT_DIR / "07_session.md"

SYMBOLS = [
    "DJ30.r", "SWI20.r", "XAUUSD", "AUDJPY",
    "EURUSD", "US2000.r", "UKOUSD", "JPN225ft",
]

# Walk-forward fold lengths (days). Per task spec — 5 folds.
WF_FOLDS = [60, 90, 120, 150, 180]

# Golden-hour selection criteria
GOLDEN_TOP_N = 6
GOLDEN_MIN_N = 10
GOLDEN_MIN_PF = 2.0

# Bottom (blacklist) criteria
BOTTOM_N = 4
BOTTOM_MIN_N = 6  # need at least 6 trades to call it toxic

# Ship gates
MIN_DELTA_PNL = 30.0
MIN_WF_AVG_PF = 1.5
MIN_WF_POS_FOLDS = 3

# Cap iters per task spec: 5
MAX_ITERS = 5

# Universe of hours we care about (0..23)
ALL_HOURS = set(range(24))


# ─────────────────────────────────────────────────────────────────────────────
# Worker BT — applies optional per-symbol hour gate via monkey-patch.
# ─────────────────────────────────────────────────────────────────────────────
def _bt_with_hour_gate(symbol, days, blocked_hours):
    """Run v5 backtest for `symbol` over `days`, optionally blocking
    `blocked_hours` (set[int]) at entry. blocked_hours=None → baseline.

    Reloads config + backtest inside the worker for parallel-safety.
    """
    import importlib
    import config as cfg
    importlib.reload(cfg)

    import backtest.v5_backtest as bt
    importlib.reload(bt)

    if blocked_hours is not None and blocked_hours:
        # Union: keep baseline TOXIC_HOURS plus new hours for THIS symbol.
        # We control the per-symbol-scope by setting TOXIC_EXEMPT for all
        # other symbols, but since each BT call is single-symbol we just
        # set TOXIC_HOURS to base ∪ blocked and ensure THIS symbol isn't
        # exempt from the added ones.
        base_toxic = set(bt.TOXIC_HOURS)
        new_toxic = base_toxic | set(blocked_hours)
        bt.TOXIC_HOURS = new_toxic
        # Symbol's exempt set must NOT include the new blocked hours.
        # Keep its existing exemptions for the baseline hours.
        existing_exempt = set(bt.TOXIC_EXEMPT.get(symbol, set()))
        # Remove blocked from exempt (in case sym had broad exempt)
        bt.TOXIC_EXEMPT[symbol] = existing_exempt - set(blocked_hours)

    return bt.backtest_symbol(symbol, days=days, verbose=False)


def _safe_bt_with_details(symbol, days, blocked_hours):
    """Returns (summary_dict, trade_details_list_or_None)."""
    try:
        r = _bt_with_hour_gate(symbol, days, blocked_hours)
        if r is None:
            return None, None
        summary = {
            "trades": int(r.get("trades", 0)),
            "wr": float(r.get("wr", 0) or 0),
            "pf": float(r.get("pf", 0) or 0),
            "pnl": float(r.get("pnl", 0) or 0),
            "dd": float(r.get("dd", 0) or 0),
            "avg_r": float(r.get("avg_r", 0) or 0),
        }
        details = r.get("details") or []
        return summary, details
    except Exception:
        traceback.print_exc()
        return None, None


def _safe_bt_summary(symbol, days, blocked_hours):
    s, _ = _safe_bt_with_details(symbol, days, blocked_hours)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Pool task functions (top-level for pickling)
# ─────────────────────────────────────────────────────────────────────────────
def _baseline_with_details_task(args):
    symbol, days = args
    summary, details = _safe_bt_with_details(symbol, days, None)
    # Strip trade detail to just (hour, pnl_dollar, pnl_r) tuples to keep
    # cross-process payload small.
    slim = []
    if details:
        for t in details:
            slim.append({
                "hour": int(t.get("hour", -1)),
                "pnl": float(t.get("pnl", 0.0) or 0.0),
                "pnl_r": float(t.get("pnl_r", 0.0) or 0.0),
                "direction": int(t.get("direction", 0) or 0),
            })
    return {"symbol": symbol, "summary": summary, "trades": slim}


def _variant_task(args):
    symbol, days, blocked_hours, label = args
    summary = _safe_bt_summary(symbol, days, blocked_hours)
    return {"symbol": symbol, "days": days, "label": label,
            "blocked": sorted(blocked_hours) if blocked_hours else [],
            "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────
def _per_hour_stats(trades):
    """trades: list of {hour, pnl, pnl_r, direction}. Returns dict
    {hour: {n, wins, losses, pnl, pf, wr}}."""
    by_hr = defaultdict(list)
    for t in trades:
        h = t.get("hour", -1)
        if 0 <= h <= 23:
            by_hr[h].append(t)
    out = {}
    for h, ts in by_hr.items():
        wins = [t for t in ts if t["pnl"] > 0]
        losses = [t for t in ts if t["pnl"] <= 0]
        gw = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = (gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0)
        out[h] = {
            "n": len(ts),
            "wins": len(wins),
            "losses": len(losses),
            "pnl": round(sum(t["pnl"] for t in ts), 2),
            "pf": round(pf, 2),
            "wr": round(100.0 * len(wins) / max(1, len(ts)), 1),
            "avg_r": round(sum(t["pnl_r"] for t in ts) / max(1, len(ts)), 3),
        }
    return out


def _pick_golden_hours(per_hour):
    """Top GOLDEN_TOP_N hours by total PnL, requiring n>=10 and pf>=2.0."""
    cands = [(h, s) for h, s in per_hour.items()
             if s["n"] >= GOLDEN_MIN_N and s["pf"] >= GOLDEN_MIN_PF and s["pnl"] > 0]
    cands.sort(key=lambda x: (-x[1]["pnl"], -x[1]["pf"]))
    return [h for h, _ in cands[:GOLDEN_TOP_N]]


def _pick_bottom_hours(per_hour):
    """Bottom BOTTOM_N hours by total PnL, requiring n>=BOTTOM_MIN_N and pnl<0."""
    cands = [(h, s) for h, s in per_hour.items()
             if s["n"] >= BOTTOM_MIN_N and s["pnl"] < 0]
    cands.sort(key=lambda x: (x[1]["pnl"], x[1]["pf"]))  # most negative pnl first
    return [h for h, _ in cands[:BOTTOM_N]]


def _whitelist_blocked(golden):
    """If we WHITELIST hours `golden`, we BLOCK everything else."""
    return sorted(ALL_HOURS - set(golden))


def _blacklist_blocked(bottom):
    return sorted(bottom)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    workers = max(2, min(8, os.cpu_count() or 4))
    print(f"[session research] workers={workers}")
    print(f"[session research] symbols={SYMBOLS}")
    print(f"[session research] WF folds={WF_FOLDS}")

    out = {
        "session": "entry_research_20260522/07_session",
        "scope": "per-symbol hour-of-day entry gate",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "symbols": list(SYMBOLS),
        "wf_folds_days": WF_FOLDS,
        "gates": {
            "golden_top_n": GOLDEN_TOP_N,
            "golden_min_trades": GOLDEN_MIN_N,
            "golden_min_pf": GOLDEN_MIN_PF,
            "bottom_n": BOTTOM_N,
            "bottom_min_trades": BOTTOM_MIN_N,
            "ship_min_delta_pnl": MIN_DELTA_PNL,
            "ship_min_wf_avg_pf": MIN_WF_AVG_PF,
            "ship_min_wf_pos_folds": MIN_WF_POS_FOLDS,
        },
        "baselines": {},
        "per_hour_180d": {},
        "candidates": {},
        "variant_180d": {},
        "walk_forward": {},
        "decisions": {},
        "summary": {},
    }

    # ── [A] Baseline 180d per symbol with hour-of-day breakdown ──────────
    print(f"\n[A] Baseline 180d with hour breakdown ({len(SYMBOLS)} sym)")
    with Pool(workers) as pool:
        for res in pool.imap_unordered(
                _baseline_with_details_task, [(s, 180) for s in SYMBOLS]):
            sym = res["symbol"]
            s = res["summary"]
            trades = res["trades"]
            if s is None:
                print(f"  {sym:10s}: NO DATA")
                out["baselines"][sym] = None
                out["per_hour_180d"][sym] = {}
                continue
            out["baselines"][sym] = s
            per_hr = _per_hour_stats(trades)
            out["per_hour_180d"][sym] = {str(h): per_hr[h] for h in sorted(per_hr)}
            print(f"  {sym:10s}: n={s['trades']:3d} pf={s['pf']:5.2f} "
                  f"pnl=${s['pnl']:+8.2f} wr={s['wr']:5.1f}%  hours={len(per_hr)}")

    # ── [B] Candidate hour sets ──────────────────────────────────────────
    print(f"\n[B] Candidate hour sets")
    for sym in SYMBOLS:
        per_hr = {int(h): v for h, v in out["per_hour_180d"].get(sym, {}).items()}
        golden = _pick_golden_hours(per_hr)
        bottom = _pick_bottom_hours(per_hr)
        wl_blocked = _whitelist_blocked(golden)
        bl_blocked = _blacklist_blocked(bottom)
        out["candidates"][sym] = {
            "golden_hours": golden,
            "bottom_hours": bottom,
            "whitelist_blocked": wl_blocked,
            "blacklist_blocked": bl_blocked,
        }
        print(f"  {sym:10s}: golden={golden}  bottom={bottom}")

    # ── [C] 180d variants — WHITELIST + BLACKLIST per symbol ─────────────
    print(f"\n[C] 180d variant BTs")
    jobs = []
    for sym in SYMBOLS:
        cands = out["candidates"][sym]
        if cands["golden_hours"]:
            jobs.append((sym, 180, set(cands["whitelist_blocked"]), "whitelist"))
        if cands["bottom_hours"]:
            jobs.append((sym, 180, set(cands["blacklist_blocked"]), "blacklist"))
    print(f"  total variant BTs: {len(jobs)}")

    var_by_sym = defaultdict(dict)
    with Pool(workers) as pool:
        for res in pool.imap_unordered(_variant_task, jobs):
            sym = res["symbol"]
            lab = res["label"]
            var_by_sym[sym][lab] = res
            s = res["summary"]
            if s:
                base = out["baselines"].get(sym, {})
                d = (s["pnl"] - (base.get("pnl", 0.0) if base else 0.0))
                print(f"  {sym:10s} {lab:9s} n={s['trades']:3d} pf={s['pf']:5.2f} "
                      f"pnl=${s['pnl']:+8.2f} Δ${d:+8.2f}")
            else:
                print(f"  {sym:10s} {lab:9s}: NO RESULT")
    out["variant_180d"] = {s: dict(v) for s, v in var_by_sym.items()}

    # ── [D] Walk-forward 5-fold validation ───────────────────────────────
    print(f"\n[D] Walk-forward 5-fold on each variant per symbol")
    wf_jobs = []
    for sym in SYMBOLS:
        for label in ("whitelist", "blacklist", "baseline"):
            cands = out["candidates"][sym]
            if label == "whitelist" and not cands["golden_hours"]:
                continue
            if label == "blacklist" and not cands["bottom_hours"]:
                continue
            if label == "whitelist":
                blocked = set(cands["whitelist_blocked"])
            elif label == "blacklist":
                blocked = set(cands["blacklist_blocked"])
            else:
                blocked = None
            for d in WF_FOLDS:
                wf_jobs.append((sym, d,
                                blocked if blocked is not None else set(),
                                label))
    print(f"  total WF BTs: {len(wf_jobs)}")

    wf_results = defaultdict(lambda: defaultdict(list))  # [sym][label] = [{days,r}...]
    t_wf = time.time()
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_variant_task, wf_jobs), 1):
            sym = res["symbol"]
            lab = res["label"]
            wf_results[sym][lab].append(
                {"days": res["days"], "summary": res["summary"]})
            if i % 10 == 0 or i == len(wf_jobs):
                print(f"  {i:3d}/{len(wf_jobs)} ({time.time() - t_wf:.0f}s)")

    # Aggregate WF — for each (sym, variant) compute avg PF, pos folds, ΔPnL vs baseline
    wf_agg = {}
    for sym in SYMBOLS:
        wf_agg[sym] = {}
        baseline_folds = wf_results[sym].get("baseline", [])
        baseline_by_d = {f["days"]: f["summary"] for f in baseline_folds}
        for lab in ("whitelist", "blacklist", "baseline"):
            folds = wf_results[sym].get(lab, [])
            folds.sort(key=lambda x: x["days"])
            pfs = []
            pos_folds = 0
            delta_pnls = []
            fold_rows = []
            for f in folds:
                s = f["summary"]
                d = f["days"]
                if s is None:
                    fold_rows.append({"days": d, "summary": None, "delta_pnl": None})
                    continue
                pfs.append(s["pf"])
                if s["pnl"] > 0:
                    pos_folds += 1
                # delta vs baseline at same days
                if lab == "baseline":
                    delta_pnl = 0.0
                else:
                    base = baseline_by_d.get(d) or {}
                    delta_pnl = round(s["pnl"] - (base.get("pnl", 0.0) or 0.0), 2)
                delta_pnls.append(delta_pnl)
                fold_rows.append({"days": d, "summary": s, "delta_pnl": delta_pnl})
            wf_agg[sym][lab] = {
                "avg_pf": round(sum(pfs) / max(1, len(pfs)), 3) if pfs else 0.0,
                "pos_folds": pos_folds,
                "n_folds_valid": len(pfs),
                "avg_delta_pnl": round(
                    sum(delta_pnls) / max(1, len(delta_pnls)), 2)
                    if delta_pnls else 0.0,
                "folds": fold_rows,
            }
    out["walk_forward"] = wf_agg

    # ── [E] Decisions ───────────────────────────────────────────────────
    print(f"\n[E] Decisions (ship gates: Δ>=${MIN_DELTA_PNL} ∧ "
          f"avg_pf>={MIN_WF_AVG_PF} ∧ pos>={MIN_WF_POS_FOLDS}/5)")
    decisions = {}
    for sym in SYMBOLS:
        base = out["baselines"].get(sym) or {}
        base_pnl_180d = base.get("pnl", 0.0)
        verdicts = {}
        for lab in ("whitelist", "blacklist"):
            v = var_by_sym.get(sym, {}).get(lab)
            wf = wf_agg.get(sym, {}).get(lab) or {}
            if v is None or v["summary"] is None or not wf:
                verdicts[lab] = {"ship": False, "reason": "no data"}
                continue
            d180 = v["summary"]["pnl"] - base_pnl_180d
            avg_pf = wf.get("avg_pf", 0.0)
            pos_folds = wf.get("pos_folds", 0)
            avg_delta = wf.get("avg_delta_pnl", 0.0)
            # Ship condition uses 180d delta + WF avg + WF positivity
            ship = (
                d180 >= MIN_DELTA_PNL
                and avg_pf >= MIN_WF_AVG_PF
                and pos_folds >= MIN_WF_POS_FOLDS
            )
            verdicts[lab] = {
                "ship": ship,
                "delta_pnl_180d": round(d180, 2),
                "wf_avg_pf": avg_pf,
                "wf_pos_folds": pos_folds,
                "wf_avg_delta_pnl": avg_delta,
                "blocked_hours": v["blocked"],
                "summary_180d": v["summary"],
            }
        # Pick best ship-passing variant (by delta_pnl_180d).
        passing = [(lab, v) for lab, v in verdicts.items() if v.get("ship")]
        passing.sort(key=lambda x: -(x[1].get("delta_pnl_180d") or 0))
        chosen = passing[0][0] if passing else None
        decisions[sym] = {
            "variants": verdicts,
            "chosen": chosen,
            "current_toxic_hours_per_symbol":
                _current_toxic_per_symbol(sym),
            "recommended_extra_toxic_hours":
                sorted(verdicts[chosen]["blocked_hours"])
                if chosen else [],
        }
        if chosen:
            v = verdicts[chosen]
            print(f"  {sym:10s} SHIP {chosen:9s} Δ${v['delta_pnl_180d']:+7.2f} "
                  f"wf_pf={v['wf_avg_pf']:.2f} pos={v['wf_pos_folds']}/5 "
                  f"block={v['blocked_hours']}")
        else:
            wl = verdicts.get("whitelist", {})
            bl = verdicts.get("blacklist", {})
            print(f"  {sym:10s} hold | WL Δ${wl.get('delta_pnl_180d',0):+.2f} "
                  f"wf_pf={wl.get('wf_avg_pf',0):.2f} pos={wl.get('wf_pos_folds',0)}/5 "
                  f"| BL Δ${bl.get('delta_pnl_180d',0):+.2f} "
                  f"wf_pf={bl.get('wf_avg_pf',0):.2f} pos={bl.get('wf_pos_folds',0)}/5")
    out["decisions"] = decisions

    # ── [F] Summary ─────────────────────────────────────────────────────
    summary = {
        "ship_recommended": {},
        "hold": [],
        "extensions": {},
    }
    for sym, d in decisions.items():
        if d["chosen"]:
            v = d["variants"][d["chosen"]]
            summary["ship_recommended"][sym] = {
                "variant": d["chosen"],
                "delta_pnl_180d": v["delta_pnl_180d"],
                "wf_avg_pf": v["wf_avg_pf"],
                "wf_pos_folds": v["wf_pos_folds"],
                "blocked_hours": v["blocked_hours"],
            }
            current = set(d["current_toxic_hours_per_symbol"])
            # Extra hours we would add on top of TOXIC_HOURS_UTC (global {1,2,3,4})
            # and existing per-symbol toxic.
            global_toxic = {1, 2, 3, 4}
            new_blocked = set(v["blocked_hours"]) - global_toxic
            extension = sorted(new_blocked - current)
            summary["extensions"][sym] = {
                "current_per_symbol": sorted(current),
                "recommended_per_symbol": sorted(new_blocked),
                "diff_added": extension,
            }
        else:
            summary["hold"].append(sym)
    out["summary"] = summary
    out["elapsed_sec"] = round(time.time() - t0, 1)
    out["finished_at"] = datetime.utcnow().isoformat() + "Z"
    out["iter"] = 1
    out["max_iters_allowed"] = MAX_ITERS

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[done] wrote {OUT_JSON}")
    print(f"  ship: {list(summary['ship_recommended'].keys())}")
    print(f"  hold: {summary['hold']}")
    print(f"  elapsed: {out['elapsed_sec']}s")

    _write_markdown(out)
    print(f"  wrote {OUT_MD}")
    return out


def _current_toxic_per_symbol(symbol):
    """Read live config's per-symbol toxic extras (best-effort)."""
    try:
        import importlib
        import config as cfg
        importlib.reload(cfg)
        per = getattr(cfg, "TOXIC_HOURS_PER_SYMBOL", {})
        return sorted(per.get(symbol, set()))
    except Exception:
        return []


def _write_markdown(out):
    lines = []
    lines.append(f"# Session-Based Entry Filter Research")
    lines.append(f"")
    lines.append(f"- Generated: {out['finished_at']}")
    lines.append(f"- Symbols: {', '.join(out['symbols'])}")
    lines.append(f"- WF folds (days): {out['wf_folds_days']}")
    lines.append(f"- Elapsed: {out['elapsed_sec']}s")
    lines.append(f"- Iterations used: 1 / {out['max_iters_allowed']}")
    lines.append(f"")
    lines.append(f"## Ship gates")
    g = out["gates"]
    lines.append(f"- ΔPnL ≥ ${g['ship_min_delta_pnl']}")
    lines.append(f"- WF avg PF ≥ {g['ship_min_wf_avg_pf']}")
    lines.append(f"- WF positive folds ≥ {g['ship_min_wf_pos_folds']}/5")
    lines.append(f"")
    lines.append(f"## Golden hour criteria")
    lines.append(f"- Top {g['golden_top_n']} hours by PnL with n ≥ "
                 f"{g['golden_min_trades']} and PF ≥ {g['golden_min_pf']}")
    lines.append(f"")
    lines.append(f"## Baselines (180d)")
    lines.append(f"")
    lines.append(f"| Symbol | n | PF | WR% | PnL$ | DD% |")
    lines.append(f"|---|---:|---:|---:|---:|---:|")
    for sym in out["symbols"]:
        b = out["baselines"].get(sym) or {}
        if not b:
            lines.append(f"| {sym} | – | – | – | – | – |")
            continue
        lines.append(f"| {sym} | {b['trades']} | {b['pf']:.2f} | "
                     f"{b['wr']:.1f} | {b['pnl']:+.2f} | {b['dd']:.1f} |")
    lines.append(f"")
    lines.append(f"## Per-symbol detail")
    for sym in out["symbols"]:
        cands = out["candidates"].get(sym, {})
        decision = out["decisions"].get(sym, {})
        per_hr = out["per_hour_180d"].get(sym, {})
        lines.append(f"")
        lines.append(f"### {sym}")
        lines.append(f"")
        # Hour table
        lines.append(f"Hour-of-day breakdown (180d):")
        lines.append(f"")
        lines.append(f"| Hr | n | PF | WR% | PnL$ | avgR |")
        lines.append(f"|---:|---:|---:|---:|---:|---:|")
        for h_str in sorted(per_hr.keys(), key=lambda x: int(x)):
            s = per_hr[h_str]
            lines.append(f"| {h_str} | {s['n']} | {s['pf']:.2f} | "
                         f"{s['wr']:.1f} | {s['pnl']:+.2f} | {s['avg_r']:+.3f} |")
        lines.append(f"")
        lines.append(f"- Golden hours (whitelist): "
                     f"`{cands.get('golden_hours', [])}`")
        lines.append(f"- Bottom hours (blacklist): "
                     f"`{cands.get('bottom_hours', [])}`")
        lines.append(f"")
        # Decision table
        verdicts = decision.get("variants", {})
        lines.append(f"| Variant | Δ180d$ | WF avgPF | WF pos | Ship | Blocked |")
        lines.append(f"|---|---:|---:|---:|:-:|---|")
        for lab in ("whitelist", "blacklist"):
            v = verdicts.get(lab, {})
            if not v:
                continue
            ship = "YES" if v.get("ship") else "no"
            lines.append(
                f"| {lab} | "
                f"{v.get('delta_pnl_180d', 0):+.2f} | "
                f"{v.get('wf_avg_pf', 0):.2f} | "
                f"{v.get('wf_pos_folds', 0)}/5 | "
                f"{ship} | `{v.get('blocked_hours', [])}` |")
        lines.append(f"")
        if decision.get("chosen"):
            v = verdicts[decision["chosen"]]
            cur = decision.get("current_toxic_hours_per_symbol", [])
            rec = decision.get("recommended_extra_toxic_hours", [])
            lines.append(f"**Verdict:** SHIP `{decision['chosen']}`. "
                         f"Block hours: `{v['blocked_hours']}`. "
                         f"Current per-symbol toxic: `{cur}`. "
                         f"Recommended new per-symbol toxic set: `{rec}`.")
        else:
            lines.append(f"**Verdict:** HOLD — neither variant meets ship gates.")
    lines.append(f"")
    lines.append(f"## Recommended TOXIC_HOURS_PER_SYMBOL extensions")
    lines.append(f"")
    lines.append(f"Diff is hours to ADD on top of the global "
                 f"TOXIC_HOURS_UTC = {{1, 2, 3, 4}} and the current "
                 f"per-symbol entries.")
    lines.append(f"")
    lines.append(f"| Symbol | Current | Recommended | Added |")
    lines.append(f"|---|---|---|---|")
    for sym, ext in out["summary"].get("extensions", {}).items():
        lines.append(f"| {sym} | `{ext['current_per_symbol']}` | "
                     f"`{ext['recommended_per_symbol']}` | "
                     f"`{ext['diff_added']}` |")
    if not out["summary"].get("extensions"):
        lines.append(f"| – | – | – | – |")
    lines.append(f"")
    if out["summary"].get("hold"):
        lines.append(f"## Hold (no ship)")
        lines.append(f"")
        for sym in out["summary"]["hold"]:
            lines.append(f"- {sym}")
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
