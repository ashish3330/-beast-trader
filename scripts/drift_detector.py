#!/usr/bin/env python3 -B
"""
Drift detector — flags symbols whose live edge has decayed vs training/baseline.

Why this exists: ML meta-label models and per-symbol score weights are tuned
on historical data. Markets drift; what worked 60 days ago may not work today.
Without surveillance, the agent silently slides from profitable to break-even
to losing as regime shifts. This detector runs daily and flags drift before
the equity curve does.

Two parallel checks per symbol:

  A) Train-vs-live PF  — compares live last-30-days PF to model.train_metrics
     filtered_pf. If live_pf < train_pf * 0.6, the model's edge has eroded.

  B) Recency split    — splits the last 60 days of trades for the symbol into
     older-half and newer-half. If newer_pf < older_pf * 0.7, edge is decaying
     in real time.

Outputs:
  - logs/drift_report.log (rolling)
  - logs/drift_latest.json (latest snapshot for dashboards)

Exit codes:
  0  no drift detected
  1  drift on >=1 symbol (use for launchd → ops alerting)
  2  internal error

Run via launchd com.dragon.drift-check daily.
"""
import json
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import SYMBOLS, DB_PATH  # noqa: E402

LOG_PATH = ROOT / "logs" / "drift_report.log"
JSON_PATH = ROOT / "logs" / "drift_latest.json"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("drift")
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_h)

# Drift thresholds
TRAIN_PF_FLOOR_RATIO = 0.6   # live_pf < train_pf * 0.6 → drift
RECENCY_PF_FLOOR_RATIO = 0.7  # newer_pf < older_pf * 0.7 → drift
MIN_TRADES_FOR_PF = 5         # need ≥5 trades in a window to trust PF (lowered from 8 — early-life detector)
WINDOW_DAYS = 30              # train-vs-live window
RECENCY_DAYS = 60             # recency-split window total


PF_DISPLAY_CAP = 99.99  # hide nonsensical "1000x" PF when there are no losses


def _pf_wr(rows):
    """Given iterable of (pnl,) rows, return (n, wr, pf, total_pnl).
    PF capped at PF_DISPLAY_CAP — windows with zero losses produce divide-by-tiny
    that inflates PF to absurd values, which then breaks ratio comparisons."""
    pnls = [r[0] for r in rows if r[0] is not None]
    if not pnls:
        return 0, 0.0, 0.0, 0.0
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    if gl < 0.01:
        # No real losses — treat as max-edge ceiling so drift comparisons stay sane.
        pf = PF_DISPLAY_CAP if gp > 0 else 0.0
    else:
        pf = min(gp / gl, PF_DISPLAY_CAP)
    return len(pnls), len(wins) / len(pnls), pf, sum(pnls)


def get_train_metrics():
    """Load saved model metrics (train-time filtered_pf etc)."""
    out = {}
    try:
        from models.signal_model import SignalModel
        m = SignalModel()
        for sym in SYMBOLS:
            try:
                m.load(sym)
            except Exception:
                pass
            if m.has_model(sym):
                out[sym] = m._train_metrics.get(sym, {})
    except Exception as e:
        log.warning("Failed to load train metrics: %s", e)
    return out


def detect_drift():
    train_metrics = get_train_metrics()
    journal = DB_PATH.parent / "trade_journal.db"
    if not journal.exists():
        log.error("Trade journal not found at %s", journal)
        return 2, {}

    conn = sqlite3.connect(str(journal), timeout=10.0)
    now = time.time()
    cutoff_30d = now - WINDOW_DAYS * 86400
    cutoff_60d = now - RECENCY_DAYS * 86400

    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds": {
            "train_pf_floor_ratio": TRAIN_PF_FLOOR_RATIO,
            "recency_pf_floor_ratio": RECENCY_PF_FLOOR_RATIO,
            "min_trades_for_pf": MIN_TRADES_FOR_PF,
        },
        "per_symbol": {},
        "drifted": [],
        "ok": [],
    }

    for sym in SYMBOLS:
        # Live last 30d
        rows30 = conn.execute(
            "SELECT pnl FROM trades WHERE symbol=? "
            "AND strftime('%s', timestamp) >= ?",
            (sym, str(int(cutoff_30d)))
        ).fetchall()
        n30, wr30, pf30, pnl30 = _pf_wr(rows30)

        # Recency split — last 60d, split by mid timestamp
        rows60 = conn.execute(
            "SELECT pnl, strftime('%s', timestamp) as ts FROM trades "
            "WHERE symbol=? AND strftime('%s', timestamp) >= ? "
            "ORDER BY ts",
            (sym, str(int(cutoff_60d)))
        ).fetchall()
        if rows60:
            mid_idx = len(rows60) // 2
            older_rows = [(r[0],) for r in rows60[:mid_idx]]
            newer_rows = [(r[0],) for r in rows60[mid_idx:]]
            n_o, wr_o, pf_o, _ = _pf_wr(older_rows)
            n_n, wr_n, pf_n, _ = _pf_wr(newer_rows)
        else:
            n_o = n_n = 0
            pf_o = pf_n = wr_o = wr_n = 0.0

        # Train-time PF (from model)
        train_m = train_metrics.get(sym, {})
        train_pf = float(train_m.get("filtered_pf", 0))
        train_auc = float(train_m.get("test_auc", train_m.get("val_auc", 0)))

        # Drift detection
        train_drift = False
        recency_drift = False
        reasons = []
        if n30 >= MIN_TRADES_FOR_PF and train_pf > 0:
            if pf30 < train_pf * TRAIN_PF_FLOOR_RATIO:
                train_drift = True
                reasons.append(
                    f"live_pf={pf30:.2f} < train_pf={train_pf:.2f} × {TRAIN_PF_FLOOR_RATIO}"
                )
        if (n_o >= MIN_TRADES_FOR_PF and n_n >= MIN_TRADES_FOR_PF
                and pf_o > 0):
            if pf_n < pf_o * RECENCY_PF_FLOOR_RATIO:
                recency_drift = True
                reasons.append(
                    f"newer_pf={pf_n:.2f} < older_pf={pf_o:.2f} × {RECENCY_PF_FLOOR_RATIO}"
                )

        drift = train_drift or recency_drift

        sym_report = {
            "live_30d": {"n": n30, "wr": round(wr30, 3),
                          "pf": round(pf30, 2), "pnl": round(pnl30, 2)},
            "recency": {
                "older_half": {"n": n_o, "wr": round(wr_o, 3), "pf": round(pf_o, 2)},
                "newer_half": {"n": n_n, "wr": round(wr_n, 3), "pf": round(pf_n, 2)},
            },
            "train": {"pf": round(train_pf, 2), "auc": round(train_auc, 3)},
            "train_drift": train_drift,
            "recency_drift": recency_drift,
            "drift": drift,
            "reasons": reasons,
        }
        report["per_symbol"][sym] = sym_report
        if drift:
            report["drifted"].append(sym)
        elif n30 >= MIN_TRADES_FOR_PF:
            report["ok"].append(sym)

    conn.close()

    # Persist
    with open(JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # Log
    log.info("Drift check: drifted=%s ok=%s",
             report["drifted"] or "[]", report["ok"] or "[]")
    print(f"\n{'symbol':10s} {'n30':>4s} {'live_pf':>7s} {'train_pf':>8s} "
          f"{'older':>6s} {'newer':>6s} {'verdict':>10s}")
    print("-" * 70)
    for sym, s in report["per_symbol"].items():
        verdict = "DRIFTED" if s["drift"] else (
            "ok" if s["live_30d"]["n"] >= MIN_TRADES_FOR_PF else "thin"
        )
        print(f"  {sym:10s} {s['live_30d']['n']:4d} "
              f"{s['live_30d']['pf']:7.2f} {s['train']['pf']:8.2f} "
              f"{s['recency']['older_half']['pf']:6.2f} "
              f"{s['recency']['newer_half']['pf']:6.2f} "
              f"  {verdict}")
        if s["drift"]:
            for r in s["reasons"]:
                print(f"      ↳ {r}")

    print(f"\nWrote {JSON_PATH}")
    return (1 if report["drifted"] else 0), report


def main():
    rc, _ = detect_drift()
    sys.exit(rc)


if __name__ == "__main__":
    main()
