"""
Dual-period robustness check for pullback sweep winners.

A winner chosen on the full window can be a period-specific curve-fit. This
re-runs each CHANGED symbol's winner vs the default (0.2, 1) on two disjoint
sub-periods — recent (last RECENT_DAYS) and older (the prior window) — and
only CONFIRMS the winner if it beats the default PF in BOTH sub-periods.
Otherwise it reverts that symbol to the default.

Reads  backtest/results/pullback_sweep_<tag>.json
Writes backtest/results/pullback_confirmed_<tag>.json  (symbol -> [retrace, wait])

Usage:
  python3 -B scripts/validate_pullback.py --tag 3yr --recent-days 365 --older-days 1095 --workers 5
"""
import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DEFAULT = (0.20, 1)
RES = Path(__file__).resolve().parent.parent / "backtest" / "results"


def _bt(symbol, retrace, wait, days, offset_days=0):
    """Run backtest for a sub-window. offset_days trims the most-recent N days
    so we can isolate the 'older' period (older = [offset_days, offset_days+days])."""
    from backtest.v5_backtest import backtest_symbol, load_data, ALL_SYMBOLS
    import backtest.v5_backtest as bt
    mm = None
    try:
        from models.signal_model import SignalModel
        mm = SignalModel(); mm.load(symbol)
        if not mm.has_model(symbol): mm = None
    except Exception:
        mm = None
    # Monkey-patch load_data to trim the recent `offset_days` for the older window.
    if offset_days > 0:
        import pandas as pd
        _orig = bt.load_data
        def _trimmed(sym, days=days, _o=offset_days, _orig=_orig):
            df = _orig(sym, None)  # full
            if df is None: return None
            cutoff_recent = df["time"].max() - pd.Timedelta(days=_o)
            df = df[df["time"] < cutoff_recent]
            cutoff_old = df["time"].max() - pd.Timedelta(days=days)
            return df[df["time"] >= cutoff_old].reset_index(drop=True)
        bt.load_data = _trimmed
    try:
        p = {"pullback_atr_retrace": retrace, "pullback_max_wait": wait}
        if mm is not None: p["_meta_model"] = mm
        r = backtest_symbol(symbol, days, p, verbose=False)
    finally:
        if offset_days > 0:
            bt.load_data = _orig
    return r


def _check(symbol, winner, recent_days, older_days):
    win = tuple(winner)
    # recent sub-window
    rw = _bt(symbol, win[0], win[1], recent_days, 0)
    rd = _bt(symbol, DEFAULT[0], DEFAULT[1], recent_days, 0)
    # older sub-window (exclude the recent days)
    ow = _bt(symbol, win[0], win[1], older_days, recent_days)
    od = _bt(symbol, DEFAULT[0], DEFAULT[1], older_days, recent_days)
    def pf(r): return float(r["pf"]) if r else 0.0
    def tr(r): return int(r["trades"]) if r else 0
    # Require: winner beats default PF, enough trades, AND a real edge (PF>1)
    # in BOTH sub-periods. The PF>1 floor rejects "improvements" from one
    # losing config to a slightly-less-losing one (e.g. 0.04 vs 0.04).
    recent_ok = bool(pf(rw) >= pf(rd) and tr(rw) >= 10 and pf(rw) > 1.0)
    older_ok = bool(pf(ow) >= pf(od) and tr(ow) >= 10 and pf(ow) > 1.0)
    confirmed = bool(recent_ok and older_ok)
    return symbol, {
        "winner": [float(win[0]), int(win[1])], "confirmed": confirmed,
        "recent": {"win_pf": pf(rw), "def_pf": pf(rd), "win_n": tr(rw), "ok": recent_ok},
        "older":  {"win_pf": pf(ow), "def_pf": pf(od), "win_n": tr(ow), "ok": older_ok},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="3yr")
    ap.add_argument("--recent-days", type=int, default=365)
    ap.add_argument("--older-days", type=int, default=1095)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    sweep = json.load(open(RES / f"pullback_sweep_{args.tag}.json"))["results"]
    changed = {s: r["winner"] for s, r in sweep.items()
               if r and tuple(r["winner"]) != DEFAULT}
    print(f"Validating {len(changed)} changed symbols on recent={args.recent_days}d / "
          f"older window, dual-period consistency required\n")

    confirmed = {}
    details = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_check, s, w, args.recent_days, args.older_days): s
                for s, w in changed.items()}
        for f in as_completed(futs):
            s, d = f.result()
            details[s] = d
            tag = "CONFIRM" if d["confirmed"] else "REVERT "
            print(f"  [{tag}] {s:12s} winner={d['winner']}  "
                  f"recent PF {d['recent']['win_pf']:.2f} vs {d['recent']['def_pf']:.2f} (ok={d['recent']['ok']}) | "
                  f"older PF {d['older']['win_pf']:.2f} vs {d['older']['def_pf']:.2f} (ok={d['older']['ok']})")
            if d["confirmed"]:
                confirmed[s] = d["winner"]

    out = RES / f"pullback_confirmed_{args.tag}.json"
    json.dump({"confirmed": confirmed, "details": details}, open(out, "w"), indent=2)
    print(f"\n  CONFIRMED ({len(confirmed)}): {confirmed}")
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
