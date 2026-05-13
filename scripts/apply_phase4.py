#!/usr/bin/env python3 -B
"""Apply Phase 4a direction bias winners to auto_tuned.py.

DIRECTION_BIAS_AUTO format: {symbol: 1 for LONG, -1 for SHORT, 0 for BOTH}
"""
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DIR = ROOT / "backtest" / "results" / "phase4_combined"

# config.py expects DIRECTION_BIAS as Dict[str, str] with "LONG"/"SHORT"
new_bias = {}
for f in sorted(DIR.glob("*.json")):
    d = json.loads(f.read_text())
    db = d.get("direction_bias")
    if db and db["direction"] in ("LONG", "SHORT"):
        new_bias[d["symbol"]] = db["direction"]

print(f"Direction bias winners ({len(new_bias)}): {new_bias}")

bk = ROOT / f"auto_tuned.py.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}-pre-phase4"
shutil.copy(ROOT / "auto_tuned.py", bk)
print(f"Backup: {bk}")

import importlib, auto_tuned  # noqa: E402
importlib.reload(auto_tuned)
src = (ROOT / "auto_tuned.py").read_text()

# Merge with existing DIRECTION_BIAS_AUTO
merged = dict(getattr(auto_tuned, "DIRECTION_BIAS_AUTO", {}))
merged.update(new_bias)

lines = ["DIRECTION_BIAS_AUTO = {"]
for s in sorted(merged): lines.append(f"    {s!r:<22}: {merged[s]!r},")  # !r for string quoting
lines.append("}")
block = "\n".join(lines)

if "DIRECTION_BIAS_AUTO" in src:
    src = re.sub(r"^DIRECTION_BIAS_AUTO = \{.*?^\}$", block, src, count=1,
                 flags=re.DOTALL | re.MULTILINE)
else:
    src += "\n\n" + block + "\n"

(ROOT / "auto_tuned.py").write_text(src)
importlib.reload(auto_tuned)
print(f"\nApplied DIRECTION_BIAS_AUTO: {len(auto_tuned.DIRECTION_BIAS_AUTO)} symbols total")
for s in sorted(new_bias):
    print(f"  {s:<11} → {auto_tuned.DIRECTION_BIAS_AUTO[s]}")
