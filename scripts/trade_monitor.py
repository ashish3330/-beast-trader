#!/usr/bin/env python3 -B
"""TRADE MONITOR (2026-07-10) — PASSIVE, ISOLATED, READ-ONLY. Flags in real time:
  1. EXEC_FAIL       — a close/order that FAILED (position left UNPROTECTED / entry
                       rejected). Detected from the trader's own explicit log markers.
  2. GAVE_BACK_TO_LOSS — a position that was meaningfully IN PROFIT and then closed
                       at a LOSS (the exact NAS pattern the user caught by hand).

It reads ONLY: logs/dragon_stderr.log (tail), data/live_positions.json (the sync
file), and the trade journal. It makes NO MT5 calls, opens NO orders, and touches
NO trader state — so it CANNOT interfere with trading. Output:
  logs/trade_monitor.log        (human-readable, greppable)
  data/monitor_alerts.json      (dashboard reads this; recent alerts + counts)

Runs every 60s via launchd (com.dragon.trade-monitor).
"""
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "dragon_stderr.log"
SYNC = ROOT / "data" / "live_positions.json"
STATE = ROOT / "data" / "monitor_state.json"
ALERTS_OUT = ROOT / "data" / "monitor_alerts.json"
MON_LOG = ROOT / "logs" / "trade_monitor.log"
JOURNAL = ROOT / "data" / "trade_journal.db"

# Explicit failure markers the trader logs (executor.py / brain.py).
FAIL_MARKERS = ("CLOSE FAILED", "position UNPROTECTED", "open reject after",
                "trail bail", "trail modify retcode")
MIN_PEAK_FRAC = 0.0008   # "meaningfully in profit" = peak >= 0.08% of entry price
MAX_ALERTS = 200         # ring-buffer cap


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _mlog(msg):
    try:
        with open(MON_LOG, "a") as f:
            f.write(f"[{_now()}] {msg}\n")
    except Exception:
        pass


def _load(path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _write_atomic(path, obj):
    try:
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(obj))
        tmp.replace(path)
    except Exception as e:
        _mlog(f"write {path} err: {e}")


def _recent_realized_pnl(symbol):
    """Most recent CLOSED realized P&L for symbol from the journal (best-effort
    match for a just-closed ticket; the journal has no ticket column)."""
    try:
        c = sqlite3.connect(str(JOURNAL), timeout=3.0)
        row = c.execute("SELECT pnl FROM trades WHERE symbol=? ORDER BY id DESC LIMIT 1",
                        (symbol,)).fetchone()
        c.close()
        return float(row[0]) if row else None
    except Exception:
        return None


def main():
    state = _load(STATE, {})
    first_run = "log_pos" not in state
    log_pos = int(state.get("log_pos", 0))
    peaks = dict(state.get("peaks", {}))     # ticket -> {symbol, entry, peak_pts, last_pts}
    alerts = list(state.get("alerts", []))

    # ── 1) NEW log lines → EXEC_FAIL alerts ──
    try:
        size = LOG.stat().st_size
        if first_run:                    # start from END of log — flag only NEW failures
            log_pos = size
        if log_pos > size:               # log rotated/truncated → restart from 0
            log_pos = 0
        with open(LOG, "r", errors="ignore") as f:
            f.seek(log_pos)
            chunk = f.read()
            log_pos = f.tell()
        for line in chunk.splitlines():
            if any(m in line for m in FAIL_MARKERS):
                a = {"ts": _now(), "type": "EXEC_FAIL", "detail": line.strip()[:220]}
                alerts.append(a)
                _mlog("EXEC_FAIL: " + line.strip()[:180])
    except FileNotFoundError:
        pass
    except Exception as e:
        _mlog(f"log-scan err: {e}")

    # ── 2) track peak P&L per open ticket; flag giveback-to-loss on close ──
    sync = _load(SYNC, {"positions": []})
    positions = sync.get("positions", []) or []
    open_now = {}
    for p in positions:
        try:
            tk = str(int(p.get("ticket") or 0))
            if tk == "0":
                continue
            po = float(p.get("price_open") or 0)
            pc = float(p.get("price_cur") or 0)
            typ = int(p.get("type", 0))
            if po <= 0 or pc <= 0:
                continue
            pts = (pc - po) if typ == 0 else (po - pc)   # profit in price points
            open_now[tk] = True
            rec = peaks.get(tk) or {"symbol": p.get("symbol"), "entry": po, "peak_pts": pts}
            rec["symbol"] = p.get("symbol")
            rec["entry"] = po
            rec["peak_pts"] = max(float(rec.get("peak_pts", pts)), pts)
            rec["last_pts"] = pts
            peaks[tk] = rec
        except Exception as e:
            _mlog(f"pos parse err: {e}")

    # tickets we WERE tracking that are gone now = closed
    for tk in [t for t in list(peaks.keys()) if t not in open_now]:
        rec = peaks.pop(tk)
        try:
            sym = rec.get("symbol")
            entry = float(rec.get("entry") or 0)
            peak_pts = float(rec.get("peak_pts", 0))
            last_pts = float(rec.get("last_pts", 0))
            peak_frac = (peak_pts / entry) if entry > 0 else 0.0
            was_in_profit = peak_pts > 0 and peak_frac >= MIN_PEAK_FRAC
            if was_in_profit:
                # Gate on the DIRECTLY-OBSERVED giveback (peaked positive, last seen
                # negative) — robust and journal-independent. Realized P&L from the
                # journal is best-effort CONTEXT only (no ticket column to match on).
                ended_loss = last_pts < 0
                realized = _recent_realized_pnl(sym) if ended_loss else None
                if ended_loss:
                    a = {"ts": _now(), "type": "GAVE_BACK_TO_LOSS", "symbol": sym,
                         "peak_pts": round(peak_pts, 1), "close_pts": round(last_pts, 1),
                         "peak_pct": round(peak_frac * 100, 2), "realized_pnl": realized}
                    alerts.append(a)
                    _mlog(f"GAVE_BACK_TO_LOSS: {sym} peaked +{peak_pts:.0f}pts "
                          f"({peak_frac*100:.2f}%) closed {last_pts:+.0f}pts realized={realized}")
        except Exception as e:
            _mlog(f"close-detect err: {e}")

    alerts = alerts[-MAX_ALERTS:]
    state = {"log_pos": log_pos, "peaks": peaks, "alerts": alerts, "ts": time.time()}
    _write_atomic(STATE, state)

    # dashboard-facing summary (recent 25 + counts over the ring buffer)
    _write_atomic(ALERTS_OUT, {
        "ts": time.time(), "updated": _now(),
        "exec_fail": sum(1 for a in alerts if a.get("type") == "EXEC_FAIL"),
        "giveback": sum(1 for a in alerts if a.get("type") == "GAVE_BACK_TO_LOSS"),
        "open_tracked": len(peaks),
        "recent": alerts[-25:],
    })


if __name__ == "__main__":
    main()
