"""
Retrain meta-label models with a champion/challenger AUC gate.

Without this gate, every retrain blindly overwrites the production model —
even if the new model is worse. Institutional pattern: keep the champion,
only promote the challenger if it clears a quality bar.

Pipeline per symbol:
  1. Load production AUC from current saved ensemble metadata (the champion)
  2. Train a fresh model on latest data (the challenger)
  3. Compare:
       new test AUC must beat champion by >= MIN_AUC_LIFT (default 0.005)
       AND new test AUC must clear MIN_AUC_FLOOR (default 0.55) — reject
       weak challengers even if they beat a weaker champion
  4. If gate passes: backup champion to models/saved/.archive/, save challenger
     If gate fails: log decision, discard challenger, keep champion

Usage:
  python3 -B scripts/retrain_with_gate.py                  # all live symbols
  python3 -B scripts/retrain_with_gate.py --symbol XAUUSD  # one symbol
  python3 -B scripts/retrain_with_gate.py --dry-run        # train but don't save
  python3 -B scripts/retrain_with_gate.py --min-lift 0.01  # stricter gate

Cron-friendly: exit code 0 if gate ran cleanly (regardless of promotions).
Exit code 2 if any symbol's training itself errored.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

MODEL_DIR = ROOT / "models" / "saved"
ARCHIVE_DIR = MODEL_DIR / ".archive"
LOG_PATH = ROOT / "logs" / "retrain_gate.log"

# Gate parameters — tunable via CLI
DEFAULT_MIN_AUC_LIFT = 0.005       # challenger must beat champion by this much
DEFAULT_MIN_AUC_FLOOR = 0.55       # absolute floor — reject weak challengers
DEFAULT_MIN_TRADES = 200           # need this many sim signals to trust the metric

log = logging.getLogger("dragon.retrain_gate")


def _setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_h = logging.FileHandler(LOG_PATH)
    file_h.setFormatter(fmt)
    cli_h = logging.StreamHandler(sys.stdout)
    cli_h.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(file_h)
    log.addHandler(cli_h)


def _load_champion_metrics(symbol: str) -> dict | None:
    """Read AUC + trade count from saved ensemble metadata."""
    safe = symbol.replace(".", "_")
    path = MODEL_DIR / f"{safe}_meta_lgb_ensemble.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = data.get("metrics") or {}
        return {
            "auc": float(m.get("test_auc", m.get("ensemble_test_auc", 0.0))),
            "trades": int(m.get("n_signals", 0)),
            "timestamp": float(data.get("timestamp", 0.0)),
            "path": str(path),
        }
    except Exception as e:
        log.warning("[%s] champion read failed: %s", symbol, e)
        return None


def _archive_champion(symbol: str) -> Path | None:
    """Move existing model files to .archive/ with timestamp suffix."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace(".", "_")
    ts = time.strftime("%Y%m%d_%H%M%S")
    moved = []
    for suffix in ("_meta_lgb.pkl", "_meta_lgb_ensemble.pkl"):
        src = MODEL_DIR / f"{safe}{suffix}"
        if not src.exists():
            continue
        dst = ARCHIVE_DIR / f"{safe}{suffix.replace('.pkl', f'.{ts}.pkl')}"
        try:
            shutil.copy2(src, dst)
            moved.append(dst.name)
        except Exception as e:
            log.warning("[%s] archive failed: %s", symbol, e)
    return moved or None


def _train_one(symbol: str, mock_mt5):
    """Run a fresh train. Returns metrics dict or {'status':'error'}."""
    from models.signal_model import SignalModel
    sm = SignalModel()
    try:
        # signal_model.train() already saves the new model on success.
        # We move the prior champion BEFORE this call (we already did that
        # in the gate flow if the challenger is going to win), but the
        # cleanest route here is: train into an isolated SignalModel,
        # capture metrics, and only call its `.save()` if the gate passes.
        metrics = sm.train(symbol, mock_mt5, None)
        # SignalModel.train usually persists internally — note the saved file
        # path so the gate can decide whether to keep or roll back.
        return {"sm": sm, "metrics": metrics}
    except Exception as e:
        log.error("[%s] training failed: %s", symbol, e, exc_info=True)
        return {"status": "error", "reason": str(e)}


def _retrain_one_symbol(symbol: str, args, mock_mt5) -> dict:
    log.info("[%s] === RETRAIN GATE START ===", symbol)
    champion = _load_champion_metrics(symbol)
    if champion:
        age_d = (time.time() - champion["timestamp"]) / 86400.0
        log.info("[%s] champion AUC %.4f (n_signals=%d, age=%.1fd)",
                 symbol, champion["auc"], champion["trades"], age_d)
    else:
        log.info("[%s] no champion on disk — challenger auto-promotes if it clears floor",
                 symbol)

    # Train challenger. signal_model.train() persists internally on success,
    # so we must archive champion BEFORE training to preserve fallback.
    archived = _archive_champion(symbol) if champion else None
    if archived:
        log.info("[%s] champion archived to .archive/ (%s)", symbol, archived)

    result = _train_one(symbol, mock_mt5)
    if result.get("status") == "error":
        # Roll back: restore archived champion
        if archived:
            _restore_archive(symbol, archived)
        return {"symbol": symbol, "promoted": False,
                "reason": "training error: " + result.get("reason", "")}

    metrics = result["metrics"] or {}
    new_auc = float(metrics.get("test_auc", 0.0))
    new_trades = int(metrics.get("n_signals", 0))

    # Gate decision
    floor_pass = new_auc >= args.min_auc_floor
    trades_pass = new_trades >= args.min_trades
    if champion:
        lift = new_auc - champion["auc"]
        lift_pass = lift >= args.min_lift
    else:
        lift = 0.0
        lift_pass = True   # auto-promote if no champion on disk

    promoted = floor_pass and trades_pass and lift_pass

    if promoted:
        log.info("[%s] PROMOTE — new AUC %.4f, lift %+.4f (champion %.4f → %.4f)",
                 symbol, new_auc, lift,
                 champion["auc"] if champion else 0.0, new_auc)
    else:
        reasons = []
        if not floor_pass:
            reasons.append(f"AUC {new_auc:.4f} < floor {args.min_auc_floor}")
        if not trades_pass:
            reasons.append(f"n_signals {new_trades} < min {args.min_trades}")
        if champion and not lift_pass:
            reasons.append(f"lift {lift:+.4f} < min {args.min_lift}")
        log.info("[%s] REJECT — keeping champion. Reasons: %s",
                 symbol, "; ".join(reasons))
        # Restore champion (signal_model.train already saved over it)
        if archived:
            _restore_archive(symbol, archived)

    return {
        "symbol": symbol,
        "promoted": promoted,
        "champion_auc": (champion["auc"] if champion else None),
        "challenger_auc": new_auc,
        "lift": lift,
        "n_signals": new_trades,
        "archived": archived,
    }


def _restore_archive(symbol: str, archived_names: list[str]):
    """If gate rejects, copy archived champion back over the new (worse) save."""
    safe = symbol.replace(".", "_")
    for archived_name in archived_names:
        # archived names look like XAUUSD_meta_lgb_ensemble.20260503_010500.pkl
        # restore to MODEL_DIR / XAUUSD_meta_lgb_ensemble.pkl
        if "_meta_lgb_ensemble" in archived_name:
            dst = MODEL_DIR / f"{safe}_meta_lgb_ensemble.pkl"
        else:
            dst = MODEL_DIR / f"{safe}_meta_lgb.pkl"
        src = ARCHIVE_DIR / archived_name
        try:
            shutil.copy2(src, dst)
            log.info("[%s] restored %s ← %s", symbol, dst.name, archived_name)
        except Exception as e:
            log.error("[%s] restore failed: %s", symbol, e)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=None,
                    help="single symbol; default = all live symbols")
    ap.add_argument("--min-lift", type=float, default=DEFAULT_MIN_AUC_LIFT,
                    help=f"min AUC improvement to promote (default {DEFAULT_MIN_AUC_LIFT})")
    ap.add_argument("--min-auc-floor", type=float, default=DEFAULT_MIN_AUC_FLOOR,
                    help=f"absolute AUC floor (default {DEFAULT_MIN_AUC_FLOOR})")
    ap.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES,
                    help=f"min n_signals for valid metric (default {DEFAULT_MIN_TRADES})")
    ap.add_argument("--dry-run", action="store_true",
                    help="train and report but never save (always restores champion)")
    args = ap.parse_args()

    _setup_logging()

    # Build symbol list
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = sorted(config.SYMBOLS.keys())

    log.info("Retrain gate run: %d symbols, min_lift=%.4f, min_floor=%.3f, dry_run=%s",
             len(symbols), args.min_lift, args.min_auc_floor, args.dry_run)

    # Build mock MT5 — reads cache pickles
    sys.path.insert(0, str(ROOT))
    from train_meta_labels import MockMT5
    mock_mt5 = MockMT5()

    summary = []
    n_promoted = 0
    n_rejected = 0
    n_errored = 0

    for sym in symbols:
        try:
            r = _retrain_one_symbol(sym, args, mock_mt5)
            if args.dry_run and r.get("promoted"):
                # Even if gate would promote, restore champion for dry-run safety
                _restore_archive(sym, r.get("archived") or [])
                r["promoted"] = False
                r["dry_run_reverted"] = True
            summary.append(r)
            if r.get("promoted"):
                n_promoted += 1
            elif r.get("reason", "").startswith("training error"):
                n_errored += 1
            else:
                n_rejected += 1
        except Exception as e:
            log.error("[%s] gate run errored: %s", sym, e, exc_info=True)
            n_errored += 1
            summary.append({"symbol": sym, "promoted": False, "reason": str(e)})

    # Final report
    log.info("=" * 60)
    log.info("RETRAIN GATE SUMMARY")
    log.info(f"  Promoted: {n_promoted}    Rejected: {n_rejected}    Errored: {n_errored}")
    for r in summary:
        if r.get("promoted"):
            log.info("  ✓ %s  AUC %.4f → %.4f  (lift %+.4f)",
                     r["symbol"],
                     r.get("champion_auc") or 0.0,
                     r.get("challenger_auc") or 0.0,
                     r.get("lift") or 0.0)
        else:
            log.info("  · %s  rejected: %s",
                     r["symbol"],
                     r.get("reason") or
                     f"AUC {r.get('challenger_auc',0):.4f} (champion {r.get('champion_auc'):.4f if r.get('champion_auc') is not None else 'NA'})")

    # Persist summary JSON for cron monitoring
    out = ROOT / "backtest" / "results" / "retrain_gate_last_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "min_lift": args.min_lift,
            "min_auc_floor": args.min_auc_floor,
            "n_promoted": n_promoted,
            "n_rejected": n_rejected,
            "n_errored": n_errored,
            "results": summary,
        }, f, indent=2, default=str)
    log.info("Wrote %s", out)

    # Exit code: 0 if gate ran cleanly, 2 if any training errored
    sys.exit(2 if n_errored else 0)


if __name__ == "__main__":
    main()
