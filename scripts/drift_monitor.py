#!/usr/bin/env python3 -B
"""
Drift monitor cron — recomputes per-symbol drift state every 5 min.

Runs as com.dragon.drift-monitor launchd job. The brain reads the
resulting symbol_drift_state table cheaply each cycle to apply a
risk multiplier in master_brain.calculate_swing_risk().

Logs are tail-friendly so the watchdog can detect monitor death.
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.drift_detector import update_all  # noqa: E402

LOG_PATH = ROOT / "logs" / "drift_monitor.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("drift_monitor")
_stream_h = logging.StreamHandler()
_stream_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_stream_h)


def main() -> None:
    log.info("drift_monitor tick")
    result = update_all()
    if not result:
        log.info("no symbols evaluated")
        return
    summary = {s: f"{r['state']} (WR={r['wr']*100:.0f}% PF={r['pf']} n={r['n']})"
               for s, r in result.items()}
    log.info("drift snapshot: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
