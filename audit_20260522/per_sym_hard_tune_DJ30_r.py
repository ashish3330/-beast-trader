#!/usr/bin/env python3 -B
"""DJ30.r per-symbol HARD-TUNE — 2026-05-23.

Mirror-aware BT (live config layers). risk_pct=2.0 ($1,219 equity).

Phase A: independent sweep per dim (~50 BTs).
Phase B: top-2 cartesian, cap 200 BTs.
Phase C: 5-fold disjoint WF, top 5 by Phase B PnL.

Ship rule: Δ ≥ $30 AND ≥3/5 WF folds positive AND avg PF ≥ 1.5.

READ-ONLY on source (no auto_tuned.py / config.py rewrites). All injection
done per-worker via process-fork (so module-state isolation is real, no
intra-process leak). multiprocessing.Pool with maxtasksperchild=1 to force
worker recycle between BTs.
"""
import os, sys, json, time, itertools
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path("/Users/ashish/Documents/beast-trader")
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "audit_20260522"
OUT_JSON = OUT_DIR / "per_sym_hard_tune_DJ30.r.json"
OUT_MD   = OUT_DIR / "per_sym_hard_tune_DJ30.r.md"
LOG_FILE = OUT_DIR / "per_sym_hard_tune_DJ30.r.log"

SYMBOL = "DJ30.r"
DAYS = 180
RISK_PCT = 2.0
START_EQUITY = 1219.0

# Ship gates
MIN_DELTA = 30.0
WF_MIN_POS = 3
WF_NUM_FOLDS = 5
WF_FOLD_DAYS = 36  # 180 / 5
MIN_AVG_PF = 1.5

# Time cap — must finish under 2h
TIME_CAP_S = 6900   # 1h55m

# ── Trail profiles (mirror live auto_tuned) ──
def _load_trail_profiles():
    """Load the named trail profiles from auto_tuned (snapshot once)."""
    import auto_tuned as _at
    return {
        "_TIGHT_LOCK":       list(_at._TIGHT_LOCK),
        "_AGGR_LOCK":        list(_at._AGGR_LOCK),
        "_RUNNER_NO_BE":     list(_at._RUNNER_NO_BE),
        "_WIDE_RUNNER":      list(_at._WIDE_RUNNER),
        "_RANGE_TIGHT":      list(_at._RANGE_TIGHT),
        "_TREND_LOOSE":      list(_at._TREND_LOOSE),
        "_WIDE_RUNNER_BE07": list(_at._WIDE_RUNNER_BE07),
    }

TRAIL_PROFILES = _load_trail_profiles()
TRAIL_NAMES = list(TRAIL_PROFILES.keys())

# ── Dim grids ──
GRID_SL          = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
GRID_TRAIL       = TRAIL_NAMES                                    # 7
GRID_MQ          = [22, 25, 28, 30, 33, 35, 38, 40]               # 8
GRID_PB_ATR      = [0.4, 0.6, 0.8, 1.0, 1.2]                      # 5
GRID_PB_WAIT     = [3, 5, 7]                                      # 3
GRID_VWAP_BUF    = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]                 # 6
GRID_PBW_CD      = [1800, 3600, 5400, 7200, 10800]                # 5 (sec → bars)
GRID_LS_CD       = [3600, 7200, 10800, 14400, 18000]              # 5 (sec → bars)

# Direction bias per regime — multiplexed
GRID_DIR_BIAS_REGIME = [
    None,                                              # baseline (use live)
    {'ranging':'LONG'},                                # live default
    {'ranging':'BOTH'},                                # remove the LONG bias
    {'ranging':'LONG','volatile':'LONG'},              # add volatile LONG
    {'ranging':'LONG','trending':'LONG'},              # add trending LONG
    {'ranging':'LONG','volatile':'SHORT'},
    {'ranging':'SHORT'},
    {'trending':'LONG','ranging':'LONG','volatile':'LONG','low_vol':'LONG'},  # all LONG
]

# ── Logging ──
def _log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────
# Worker — single BT with all dims patched. Process recycles each call.
# Returns dict {trades, pf, wr, pnl, dd, avg_r, err?}
# ──────────────────────────────────────────────────────────────────────
def _bt_worker(args):
    """args is a dict-config of overlays."""
    cfg_overlay = args
    try:
        import importlib as _il
        import auto_tuned as _at; _il.reload(_at)
        import config as _cfg; _il.reload(_cfg)
        import backtest.v5_backtest as _bt; _il.reload(_bt)

        sym = SYMBOL

        # 1. SL (clear regime override so flat SL applies)
        if "sl" in cfg_overlay:
            _bt.SL_OVERRIDE[sym] = float(cfg_overlay["sl"])
            _bt.SL_OVERRIDE_REGIME[sym] = {}

        # 2. Trail (clear regime override so flat trail applies)
        if "trail" in cfg_overlay:
            tname = cfg_overlay["trail"]
            _bt.TRAIL_OVERRIDE[sym] = list(TRAIL_PROFILES[tname])
            _bt.TRAIL_OVERRIDE_REGIME[sym] = {}

        # 3. min_quality — all 4 regimes uniform
        if "mq" in cfg_overlay:
            v = int(cfg_overlay["mq"])
            _cfg.SIGNAL_QUALITY_SYMBOL[sym] = {
                'trending': v, 'ranging': v, 'volatile': v, 'low_vol': v
            }

        # 4. pb_atr / pb_wait
        if "pb_atr" in cfg_overlay:
            _cfg.PULLBACK_ATR_RETRACE_PER_SYMBOL[sym] = float(cfg_overlay["pb_atr"])
        if "pb_wait" in cfg_overlay:
            _cfg.PULLBACK_MAX_WAIT_BARS_PER_SYMBOL[sym] = int(cfg_overlay["pb_wait"])

        # 5. VWAP buffer (0.0 = disabled)
        if "vwap_buf" in cfg_overlay:
            _cfg.VWAP_BUFFER_PER_SYMBOL[sym] = float(cfg_overlay["vwap_buf"])

        # 6. POST_BIG_WIN cooldown (seconds → BT reads at function entry, bars)
        if "pbw_cd_secs" in cfg_overlay:
            _cfg.POST_BIG_WIN_COOLDOWN_SECS = int(cfg_overlay["pbw_cd_secs"])

        # 7. Direction bias per regime
        if "dir_bias" in cfg_overlay and cfg_overlay["dir_bias"] is not None:
            _bt._DIR_BIAS_REGIME_STR[sym] = dict(cfg_overlay["dir_bias"])
        elif cfg_overlay.get("clear_dir_bias"):
            _bt._DIR_BIAS_REGIME_STR[sym] = {}

        # 8. LOSS_STREAK cooldown (seconds → bars)
        ls_cd_bars = None
        if "ls_cd_secs" in cfg_overlay:
            ls_cd_bars = max(1, int(round(cfg_overlay["ls_cd_secs"] / 3600.0)))

        # 9. Fold slicing — overlay load_data() to slice WF fold range
        fold = cfg_overlay.get("fold")  # None or (fold_n, num_folds, fold_days)
        if fold is not None:
            import pandas as pd
            fold_n, num, fold_d = fold
            orig_load = _bt.load_data
            def load_data_fold(s, _ignored=None):
                df = orig_load(s, days=None)
                if df is None or df.empty:
                    return df
                end = df["time"].max()
                offset_end = (num - fold_n) * fold_d
                offset_start = offset_end + fold_d
                t_end = end - pd.Timedelta(days=offset_end)
                t_start = end - pd.Timedelta(days=offset_start)
                return df[(df["time"] > t_start) & (df["time"] <= t_end)].reset_index(drop=True)
            _bt.load_data = load_data_fold
            days_arg = None
        else:
            days_arg = DAYS

        params = {
            "risk_pct": RISK_PCT,
            "start_equity": START_EQUITY,
        }
        if ls_cd_bars is not None:
            params["loss_streak_cooldown_bars"] = ls_cd_bars

        r = _bt.backtest_symbol(sym, days=days_arg, params=params, verbose=False)
        if r is None:
            return {"err": "result_none"}
        return {
            "trades": int(r.get("trades", 0)),
            "pf":     float(r.get("pf", 0)),
            "wr":     float(r.get("wr", 0)),
            "pnl":    float(r.get("pnl", 0)),
            "dd":     float(r.get("dd", 0)),
            "avg_r":  float(r.get("avg_r", 0)),
        }
    except Exception as e:
        import traceback
        return {"err": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:400]}"}


def _score(result):
    """Composite score: PnL × (PF ≥ 1.5 multiplier) × (DD < 10 multiplier).
    Lower DD better, higher PF better.
    """
    if not result or "err" in result:
        return -1e9
    pnl = result["pnl"]
    pf  = result["pf"]
    dd  = result["dd"]
    n   = result["trades"]
    if n < 20:
        return pnl - 1000  # penalize too-few-trades configs
    pf_mult = 1.0 if pf >= 1.5 else (pf / 1.5)
    dd_mult = 1.0 if dd <= 8.0 else max(0.3, 1.0 - (dd - 8.0) * 0.1)
    return pnl * pf_mult * dd_mult


def main():
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)
    LOG_FILE.unlink(missing_ok=True)
    _log("════════════════════════════════════════════════════════════════")
    _log(f"DJ30.r HARD-TUNE (mirror-aware BT, risk_pct={RISK_PCT}, equity=${START_EQUITY})")
    _log(f"Days={DAYS}, time cap={TIME_CAP_S}s ({TIME_CAP_S//60}min)")
    _log("════════════════════════════════════════════════════════════════")

    n_workers = min(8, max(2, cpu_count() - 2))
    _log(f"Workers: {n_workers}")

    # ── Baseline (live config + risk_pct=2.0) ──
    baseline = _bt_worker({"clear_dir_bias": False})
    _log(f"BASELINE: trades={baseline.get('trades',0)} pf={baseline.get('pf',0)} pnl=${baseline.get('pnl',0):.2f} dd={baseline.get('dd',0)}%")
    if "err" in baseline or baseline.get("trades", 0) == 0:
        _log("FATAL: baseline failed → aborting.")
        with open(OUT_JSON, "w") as f:
            json.dump({"error": "baseline failed", "baseline": baseline}, f, indent=2)
        return

    base_pnl = baseline["pnl"]

    # ──────────────────────────────────────────────────────────
    # PHASE A — independent sweep per dim
    # ──────────────────────────────────────────────────────────
    phaseA = {}
    a_tasks = []

    # SL only
    for v in GRID_SL:
        a_tasks.append(("sl", v, {"sl": v}))
    # Trail only
    for v in GRID_TRAIL:
        a_tasks.append(("trail", v, {"trail": v}))
    # mq only
    for v in GRID_MQ:
        a_tasks.append(("mq", v, {"mq": v}))
    # pb_atr only
    for v in GRID_PB_ATR:
        a_tasks.append(("pb_atr", v, {"pb_atr": v}))
    # pb_wait only
    for v in GRID_PB_WAIT:
        a_tasks.append(("pb_wait", v, {"pb_wait": v}))
    # vwap_buf only
    for v in GRID_VWAP_BUF:
        a_tasks.append(("vwap_buf", v, {"vwap_buf": v}))
    # PBW cooldown only
    for v in GRID_PBW_CD:
        a_tasks.append(("pbw_cd", v, {"pbw_cd_secs": v}))
    # LS cooldown only
    for v in GRID_LS_CD:
        a_tasks.append(("ls_cd", v, {"ls_cd_secs": v}))
    # Direction bias only
    for i, v in enumerate(GRID_DIR_BIAS_REGIME):
        tag = f"db{i}"
        if v is None:
            cfg = {"clear_dir_bias": False}  # leave live setting
        else:
            cfg = {"dir_bias": v}
        a_tasks.append(("dir_bias", tag, cfg))

    _log(f"PHASE A: {len(a_tasks)} BTs")
    t_a = time.time()
    with Pool(n_workers, maxtasksperchild=1) as pool:
        a_results = pool.map(_bt_worker, [t[2] for t in a_tasks])
    _log(f"PHASE A complete in {time.time()-t_a:.1f}s")

    # Bucket by dim
    for (dim, val, _), res in zip(a_tasks, a_results):
        phaseA.setdefault(dim, []).append({"val": val, "result": res, "score": _score(res)})

    # Pretty print top 3 per dim
    for dim in ("sl", "trail", "mq", "pb_atr", "pb_wait", "vwap_buf", "pbw_cd", "ls_cd", "dir_bias"):
        items = phaseA.get(dim, [])
        items.sort(key=lambda x: -x["score"])
        _log(f"  TOP-3 {dim}:")
        for it in items[:3]:
            r = it["result"]
            if "err" in r:
                _log(f"    {it['val']!s:18s} → ERR {r['err'][:50]}")
            else:
                _log(f"    {it['val']!s:18s} → pnl=${r['pnl']:8.2f} pf={r['pf']:5.2f} wr={r['wr']:.1f}% n={r['trades']:3d} dd={r['dd']:.1f}% score={it['score']:.0f}")

    # Phase A top-2 per dim (pick from sorted)
    def topk(dim, k=2):
        items = sorted(phaseA.get(dim, []), key=lambda x: -x["score"])
        # require non-error & non-zero trades
        valid = [i for i in items if "err" not in i["result"] and i["result"]["trades"] > 0]
        return [it["val"] for it in valid[:k]]

    a_top = {
        "sl":       topk("sl", 2),
        "trail":    topk("trail", 2),
        "mq":       topk("mq", 2),
        "pb_atr":   topk("pb_atr", 2),
        "pb_wait":  topk("pb_wait", 2),
        "vwap_buf": topk("vwap_buf", 2),
        "pbw_cd":   topk("pbw_cd", 2),
        "ls_cd":    topk("ls_cd", 2),
        "dir_bias": topk("dir_bias", 2),
    }
    _log(f"PHASE A top-2 per dim: {a_top}")

    # ──────────────────────────────────────────────────────────
    # PHASE B — top-2 cartesian (cap to 200)
    # ──────────────────────────────────────────────────────────
    # cartesian: 2^9 = 512 max; cap at 200 by truncating dir_bias to 1
    sl_top      = a_top["sl"] or [GRID_SL[0]]
    tr_top      = a_top["trail"] or [GRID_TRAIL[0]]
    mq_top      = a_top["mq"] or [GRID_MQ[0]]
    pba_top     = a_top["pb_atr"] or [GRID_PB_ATR[2]]
    pbw_top     = a_top["pb_wait"] or [GRID_PB_WAIT[1]]
    vw_top      = a_top["vwap_buf"] or [GRID_VWAP_BUF[0]]
    pbwcd_top   = a_top["pbw_cd"] or [GRID_PBW_CD[2]]
    lscd_top    = a_top["ls_cd"] or [GRID_LS_CD[2]]
    db_top      = a_top["dir_bias"] or ["db1"]

    # If 2x9 = 512, drop pbw_cd or ls_cd or dir_bias to 1
    total = (len(sl_top) * len(tr_top) * len(mq_top) * len(pba_top) *
             len(pbw_top) * len(vw_top) * len(pbwcd_top) * len(lscd_top) * len(db_top))
    _log(f"PHASE B cartesian size: {total}")
    if total > 200:
        # Drop dim with the weakest top-1 vs baseline differentiation first.
        # Simple heuristic: shrink dir_bias and ls_cd and pbw_cd to top-1.
        db_top = db_top[:1]
        lscd_top = lscd_top[:1]
        pbwcd_top = pbwcd_top[:1]
        total = (len(sl_top) * len(tr_top) * len(mq_top) * len(pba_top) *
                 len(pbw_top) * len(vw_top) * len(pbwcd_top) * len(lscd_top) * len(db_top))
        _log(f"  shrunk db/ls_cd/pbw_cd to top-1 → cartesian size: {total}")

    # Build configs
    b_configs = []
    for sl in sl_top:
        for tr in tr_top:
            for mq in mq_top:
                for pba in pba_top:
                    for pbw in pbw_top:
                        for vw in vw_top:
                            for pbwcd in pbwcd_top:
                                for lscd in lscd_top:
                                    for db_tag in db_top:
                                        cfg = {
                                            "sl": sl, "trail": tr, "mq": mq,
                                            "pb_atr": pba, "pb_wait": pbw,
                                            "vwap_buf": vw,
                                            "pbw_cd_secs": pbwcd,
                                            "ls_cd_secs": lscd,
                                        }
                                        # Resolve db_tag → dir_bias dict
                                        if db_tag.startswith("db"):
                                            idx = int(db_tag[2:])
                                            db_val = GRID_DIR_BIAS_REGIME[idx]
                                            if db_val is None:
                                                pass  # leave live
                                            else:
                                                cfg["dir_bias"] = db_val
                                        b_configs.append(cfg)
    _log(f"PHASE B: {len(b_configs)} BTs")

    t_b = time.time()
    with Pool(n_workers, maxtasksperchild=1) as pool:
        b_results = pool.map(_bt_worker, b_configs)
    _log(f"PHASE B complete in {time.time()-t_b:.1f}s")

    b_entries = []
    for cfg, res in zip(b_configs, b_results):
        b_entries.append({"config": cfg, "result": res, "score": _score(res)})
    b_entries.sort(key=lambda e: -e["score"])

    _log(f"  PHASE B top-10:")
    for e in b_entries[:10]:
        r = e["result"]
        c = e["config"]
        if "err" in r:
            _log(f"    ERR: {r['err'][:60]}")
            continue
        _log(f"    sl={c['sl']} tr={c['trail']:15s} mq={c['mq']:>3} pb={c['pb_atr']}/{c['pb_wait']} vw={c['vwap_buf']} pbwcd={c['pbw_cd_secs']} lscd={c['ls_cd_secs']} db={c.get('dir_bias','live')!s:30s} → pnl=${r['pnl']:8.2f} pf={r['pf']:5.2f} n={r['trades']:3d} dd={r['dd']:.1f}% score={e['score']:.0f}")

    # ──────────────────────────────────────────────────────────
    # PHASE C — 5-fold disjoint WF on top-5
    # ──────────────────────────────────────────────────────────
    top5 = [e for e in b_entries[:5] if "err" not in e["result"]]
    if not top5:
        _log("PHASE C: no valid Phase B candidates — abort.")
        _save(baseline, phaseA, a_top, b_entries, [], None)
        return

    wf_tasks = []
    for ci, e in enumerate(top5):
        for f in range(1, WF_NUM_FOLDS + 1):
            cfg = dict(e["config"])
            cfg["fold"] = (f, WF_NUM_FOLDS, WF_FOLD_DAYS)
            wf_tasks.append((ci, f, cfg))

    _log(f"PHASE C: {len(wf_tasks)} BTs")
    t_c = time.time()
    with Pool(n_workers, maxtasksperchild=1) as pool:
        wf_results = pool.map(_bt_worker, [t[2] for t in wf_tasks])
    _log(f"PHASE C complete in {time.time()-t_c:.1f}s")

    # Aggregate per candidate
    phaseC = []
    for ci, e in enumerate(top5):
        folds = []
        for (cii, f, _), res in zip(wf_tasks, wf_results):
            if cii == ci:
                folds.append(res)
        pos = sum(1 for r in folds if r and "err" not in r and r["pnl"] > 0)
        avg_pf = sum(r["pf"] for r in folds if r and "err" not in r) / max(1, len([r for r in folds if r and "err" not in r]))
        total_pnl = sum(r["pnl"] for r in folds if r and "err" not in r)
        delta = e["result"]["pnl"] - base_pnl
        ship = (delta >= MIN_DELTA and pos >= WF_MIN_POS and avg_pf >= MIN_AVG_PF)
        phaseC.append({
            "config": e["config"],
            "in_sample": e["result"],
            "in_sample_score": e["score"],
            "delta_vs_baseline": delta,
            "wf_folds": folds,
            "wf_pos_count": pos,
            "wf_avg_pf": avg_pf,
            "wf_total_pnl": total_pnl,
            "ship_eligible": ship,
        })

    # Find winner
    ship_winners = [c for c in phaseC if c["ship_eligible"]]
    if ship_winners:
        # Best by in-sample score among ship-eligible
        winner = max(ship_winners, key=lambda c: c["in_sample_score"])
    else:
        # No ship — return None winner
        winner = None

    _log("PHASE C summary:")
    for c in phaseC:
        cfg = c["config"]
        ship_tag = "SHIP" if c["ship_eligible"] else "HOLD"
        _log(f"  [{ship_tag}] sl={cfg['sl']} tr={cfg['trail']:15s} mq={cfg['mq']:>3} pb={cfg['pb_atr']}/{cfg['pb_wait']} vw={cfg['vwap_buf']} → Δ=${c['delta_vs_baseline']:+8.2f} WF pos={c['wf_pos_count']}/5 avg_pf={c['wf_avg_pf']:.2f}")

    _save(baseline, phaseA, a_top, b_entries, phaseC, winner)
    _log(f"════════════════════════════════════════════════════════════════")
    _log(f"DONE. Elapsed: {time.time()-t0:.1f}s")
    if winner:
        _log(f"WINNER: {winner['config']}")
        _log(f"  Δ=${winner['delta_vs_baseline']:+.2f} WF pos={winner['wf_pos_count']}/5 avg_pf={winner['wf_avg_pf']:.2f}")
    else:
        _log("NO SHIP — keep current live config.")


def _save(baseline, phaseA, a_top, b_entries, phaseC, winner):
    out = {
        "symbol": SYMBOL,
        "session_date": "2026-05-23",
        "days": DAYS,
        "risk_pct": RISK_PCT,
        "start_equity": START_EQUITY,
        "baseline": baseline,
        "phase_a": {
            dim: [
                {"val": e["val"], "result": e["result"], "score": e["score"]}
                for e in sorted(phaseA[dim], key=lambda x: -x["score"])
            ]
            for dim in phaseA
        },
        "phase_a_top2": a_top,
        "phase_b_top10": [
            {"config": e["config"], "result": e["result"], "score": e["score"]}
            for e in b_entries[:10]
        ],
        "phase_c": phaseC,
        "winner": winner,
        "ship_rule": f"Δ ≥ ${MIN_DELTA} AND ≥{WF_MIN_POS}/{WF_NUM_FOLDS} WF folds positive AND avg PF ≥ {MIN_AVG_PF}",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Markdown summary
    base = baseline
    lines = []
    lines.append(f"# DJ30.r per-symbol HARD-TUNE (2026-05-23)")
    lines.append("")
    lines.append(f"- Days: {DAYS}")
    lines.append(f"- Risk %: {RISK_PCT}")
    lines.append(f"- Start equity: ${START_EQUITY}")
    lines.append(f"- BT = mirror-aware (loads live config, auto_tuned overlays)")
    lines.append(f"- **Baseline**: trades={base.get('trades',0)} pf={base.get('pf',0)} wr={base.get('wr',0)}% pnl=${base.get('pnl',0):.2f} dd={base.get('dd',0)}%")
    lines.append("")
    lines.append("## Phase A — top-3 per dim")
    for dim in ("sl", "trail", "mq", "pb_atr", "pb_wait", "vwap_buf", "pbw_cd", "ls_cd", "dir_bias"):
        items = sorted(phaseA.get(dim, []), key=lambda x: -x["score"])
        lines.append(f"### {dim}")
        for it in items[:3]:
            r = it["result"]
            if "err" in r:
                lines.append(f"- `{it['val']}` → ERR")
            else:
                lines.append(f"- `{it['val']}` → pnl=${r['pnl']:.2f} pf={r['pf']} wr={r['wr']}% n={r['trades']} dd={r['dd']}% score={it['score']:.0f}")
    lines.append("")
    lines.append("## Phase A → Top-2 per dim (fed to Phase B)")
    lines.append(f"```\n{json.dumps(a_top, indent=2, default=str)}\n```")
    lines.append("")
    lines.append("## Phase B — top-10 cartesian")
    for i, e in enumerate(b_entries[:10]):
        c = e["config"]
        r = e["result"]
        if "err" in r:
            lines.append(f"{i+1}. ERR")
            continue
        lines.append(f"{i+1}. sl={c['sl']} tr={c['trail']} mq={c['mq']} pb={c['pb_atr']}/{c['pb_wait']} vw={c['vwap_buf']} pbwcd={c.get('pbw_cd_secs','-')} lscd={c.get('ls_cd_secs','-')} db={c.get('dir_bias','live')} → pnl=${r['pnl']:.2f} pf={r['pf']} n={r['trades']} dd={r['dd']}% score={e['score']:.0f}")
    lines.append("")
    lines.append("## Phase C — 5-fold WF on top-5")
    for c in phaseC:
        cfg = c["config"]
        ship_tag = "**SHIP**" if c["ship_eligible"] else "hold"
        lines.append(f"- {ship_tag} sl={cfg['sl']} tr={cfg['trail']} mq={cfg['mq']} pb={cfg['pb_atr']}/{cfg['pb_wait']} vw={cfg['vwap_buf']} → in-sample=${c['in_sample']['pnl']:.2f}, Δ=${c['delta_vs_baseline']:+.2f}, WF {c['wf_pos_count']}/5 avg_pf={c['wf_avg_pf']:.2f}")
    lines.append("")
    lines.append("## Ship decision")
    lines.append(f"Rule: Δ ≥ ${MIN_DELTA} AND ≥{WF_MIN_POS}/{WF_NUM_FOLDS} folds positive AND avg PF ≥ {MIN_AVG_PF}.")
    if winner:
        c = winner["config"]
        lines.append(f"**SHIP** — sl={c['sl']} tr={c['trail']} mq={c['mq']} pb={c['pb_atr']}/{c['pb_wait']} vw={c['vwap_buf']} pbwcd={c.get('pbw_cd_secs','-')} lscd={c.get('ls_cd_secs','-')} db={c.get('dir_bias','live')} → in-sample=${winner['in_sample']['pnl']:.2f} (Δ=${winner['delta_vs_baseline']:+.2f}), WF {winner['wf_pos_count']}/5 positive, avg_pf={winner['wf_avg_pf']:.2f}")
    else:
        lines.append("**HOLD** — no ship-eligible candidate. Keep current live DJ30.r config.")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
