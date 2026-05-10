#!/usr/bin/env python3 -B
"""Parse v5_backtest.py stdout into structured JSON.

Extracts per-symbol stats and aggregate. Usage:
    python3 -B scripts/parse_backtest_log.py LOG_PATH OUT_JSON [LABEL]
"""
import json
import re
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"^\s+(\S+)\s+\|\s+(\d+) trades \|\s+WR\s+([\d.]+)%\s+\|\s+PF\s+([\d.]+|inf)"
    r"\s+\|\s+PnL \$\s*(-?[\d.]+)\s+\|\s+DD\s+([\d.]+)%"
)
TOTAL_RE = re.compile(r"^\s+TOTAL:\s+(\d+) trades \|\s+PnL \$([\d.\-]+)")


def parse(log_path: Path) -> dict:
    syms = {}
    total = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = LINE_RE.match(line)
        if m:
            sym, n, wr, pf, pnl, dd = m.groups()
            syms[sym] = {
                "trades": int(n),
                "wr": float(wr),
                "pf": float(pf) if pf != "inf" else 99.0,
                "pnl": float(pnl),
                "dd": float(dd),
            }
        m = TOTAL_RE.match(line)
        if m:
            total = {"trades": int(m.group(1)), "pnl": float(m.group(2))}
    return {"symbols": syms, "total": total or {"trades": 0, "pnl": 0.0}}


if __name__ == "__main__":
    log_path = Path(sys.argv[1])
    out_json = Path(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else log_path.stem
    parsed = parse(log_path)
    parsed["label"] = label
    out_json.write_text(json.dumps(parsed, indent=2))
    s = parsed["symbols"]
    profitable = sum(1 for v in s.values() if v["pnl"] > 0)
    print(f"{label}: {parsed['total']['trades']} trades, ${parsed['total']['pnl']:.0f} PnL, "
          f"{profitable}/{len(s)} symbols profitable")
