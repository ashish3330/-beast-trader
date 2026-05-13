#!/usr/bin/env python3 -B
"""
Apply agent-tune synthesized findings to live config.

Backs up auto_tuned.py, then patches:
  - SL_OVERRIDE_AUTO for 14 APPLY symbols (new SL values from tune)
  - TRAIL_OVERRIDE_AUTO for 14 APPLY symbols (new trail profile)
  - Leaves SIGNAL_QUALITY/DIR_BIAS/RISK_CAP/TOXIC_HOURS untouched

Disabling 3 bleeders (CHFJPY/FRA40.r/GBPJPY) requires config.py edit
which this script does NOT do (separate user-approval step).
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYNTH = json.loads((ROOT / "backtest" / "results" / "agent_tune_20260513" / "synthesized.json").read_text())

# Trail profile → live trail_steps format (R, type, param)
TRAIL_PROFILES_LIVE = {
    "DEFAULT": [(8.0, 'trail', 0.3), (4.0, 'trail', 0.5), (2.0, 'trail', 0.8),
                (1.5, 'lock', 0.7), (1.0, 'lock', 0.4), (0.5, 'be', 0.0)],
    "TIGHT":   [(6.0, 'trail', 0.4), (3.0, 'trail', 0.6),
                (1.5, 'lock', 0.9), (1.0, 'lock', 0.5), (0.5, 'be', 0.0)],
    "LOOSE":   [(10.0, 'trail', 0.2), (5.0, 'trail', 0.4), (2.5, 'trail', 0.6),
                (1.5, 'lock', 0.5), (0.7, 'be', 0.0)],
    "AGGR_RUN":[(15.0, 'trail', 0.3), (8.0, 'trail', 0.5),
                (3.0, 'lock', 0.5), (1.0, 'be', 0.0)],
}

apply_syms = SYNTH["summary"]["APPLY"]
print(f"APPLYing tune for {len(apply_syms)} symbols: {apply_syms}\n")

# Read existing auto_tuned.py
src = (ROOT / "auto_tuned.py").read_text()

# Back up
backup = ROOT / f"auto_tuned.py.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
shutil.copy(ROOT / "auto_tuned.py", backup)
print(f"Backed up: {backup}")

# Import existing dicts to merge
sys.path.insert(0, str(ROOT))
import auto_tuned  # noqa: E402
import importlib; importlib.reload(auto_tuned)

new_sl = dict(getattr(auto_tuned, "SL_OVERRIDE_AUTO", {}))
new_trail = dict(getattr(auto_tuned, "TRAIL_OVERRIDE_AUTO", {}))

# Apply per-symbol from synthesized
for sym in apply_syms:
    d = SYNTH["decisions"][sym]
    p = d["params"]
    new_sl[sym] = p["sl"]
    new_trail[sym] = TRAIL_PROFILES_LIVE[p["trail"]]
    print(f"  {sym:<11}  SL → {p['sl']}  trail → {p['trail']}")

# Now write the patched auto_tuned.py — preserve other dicts intact, only
# update SL_OVERRIDE_AUTO and TRAIL_OVERRIDE_AUTO blocks.
import re

# Format SL_OVERRIDE_AUTO
sl_lines = ["SL_OVERRIDE_AUTO = {"]
for sym in sorted(new_sl):
    sl_lines.append(f"    {sym!r:<22}: {new_sl[sym]},")
sl_lines.append("}")
sl_block = "\n".join(sl_lines)

# Format TRAIL_OVERRIDE_AUTO
trail_lines = ["TRAIL_OVERRIDE_AUTO = {"]
for sym in sorted(new_trail):
    steps = new_trail[sym]
    steps_str = "[" + ", ".join(repr(s) for s in steps) + "]"
    trail_lines.append(f"    {sym!r:<22}: {steps_str},")
trail_lines.append("}")
trail_block = "\n".join(trail_lines)

# Replace blocks
src = re.sub(
    r"SL_OVERRIDE_AUTO = \{[^}]*\}",
    sl_block,
    src,
    count=1,
    flags=re.DOTALL,
)
src = re.sub(
    r"TRAIL_OVERRIDE_AUTO = \{[^}]*\}",
    trail_block,
    src,
    count=1,
    flags=re.DOTALL,
)

(ROOT / "auto_tuned.py").write_text(src)
print(f"\nWritten new auto_tuned.py — {len(apply_syms)} symbols updated")

# Verify by re-importing
importlib.reload(auto_tuned)
for sym in apply_syms:
    actual_sl = auto_tuned.SL_OVERRIDE_AUTO.get(sym)
    actual_trail = auto_tuned.TRAIL_OVERRIDE_AUTO.get(sym)
    p = SYNTH["decisions"][sym]["params"]
    if actual_sl != p["sl"]:
        print(f"  WARN: {sym} SL {actual_sl} != expected {p['sl']}")
    else:
        print(f"  OK   {sym}  SL={actual_sl}  trail set")
