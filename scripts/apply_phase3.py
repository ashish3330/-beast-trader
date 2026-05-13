#!/usr/bin/env python3 -B
"""Apply Phase 3a (quality + Kelly) and Phase 3b (component weights) winners.

Writes to auto_tuned.py:
  - SIGNAL_QUALITY_SYMBOL_AUTO (overrides via quality_delta)
  - SYMBOL_RISK_PCT_OVERRIDE_AUTO (NEW — per-symbol risk_pct from Kelly tune)
  - COMPONENT_WEIGHTS_AUTO (3 components per symbol)
"""
import json
import shutil
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

P3A = ROOT / "backtest" / "results" / "phase3a_quality_kelly"
P3B = ROOT / "backtest" / "results" / "phase3b_component_weights"

DEFAULT_QUALITY = {"trending": 40, "ranging": 42, "volatile": 45, "low_vol": 40}

p3a_winners = {}
for f in sorted(P3A.glob("*.json")):
    d = json.loads(f.read_text())
    w = d.get("winner")
    if w:
        p3a_winners[d["symbol"]] = w

p3b_winners = {}
for f in sorted(P3B.glob("*.json")):
    d = json.loads(f.read_text())
    w = d.get("winner")
    if w:
        p3b_winners[d["symbol"]] = w

print(f"Phase 3a winners: {len(p3a_winners)} — {sorted(p3a_winners.keys())}")
print(f"Phase 3b winners: {len(p3b_winners)} — {sorted(p3b_winners.keys())}")

# Backup
bk = ROOT / f"auto_tuned.py.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}-pre-phase3"
shutil.copy(ROOT / "auto_tuned.py", bk)
print(f"\nBackup: {bk}")

import importlib, auto_tuned  # noqa: E402
importlib.reload(auto_tuned)
src = (ROOT / "auto_tuned.py").read_text()

# === Build SIGNAL_QUALITY_SYMBOL_AUTO patches ===
new_quality = dict(getattr(auto_tuned, "SIGNAL_QUALITY_SYMBOL_AUTO", {}))
for sym, w in p3a_winners.items():
    new_quality[sym] = w["quality"]

# === Build SYMBOL_RISK_PCT_OVERRIDE_AUTO (new) ===
new_risk = dict(getattr(auto_tuned, "SYMBOL_RISK_PCT_OVERRIDE_AUTO", {}))
for sym, w in p3a_winners.items():
    new_risk[sym] = w["risk_pct"]

# === Build COMPONENT_WEIGHTS_AUTO ===
new_weights = dict(getattr(auto_tuned, "COMPONENT_WEIGHTS_AUTO", {}))
for sym, w in p3b_winners.items():
    new_weights[sym] = w["weights"]

# Rewrite/add SIGNAL_QUALITY_SYMBOL_AUTO
sq_lines = ["SIGNAL_QUALITY_SYMBOL_AUTO = {"]
for s in sorted(new_quality):
    sq_lines.append(f"    {s!r:<22}: {new_quality[s]},")
sq_lines.append("}")
sq_block = "\n".join(sq_lines)
if "SIGNAL_QUALITY_SYMBOL_AUTO" in src:
    # Match dict that contains NESTED dicts (with {} inside). Match from
    # opening line to the closing `}` at column 0 (line-anchored).
    src = re.sub(r"^SIGNAL_QUALITY_SYMBOL_AUTO = \{.*?^\}$",
                 sq_block, src, count=1, flags=re.DOTALL | re.MULTILINE)
else:
    src += "\n\n" + sq_block + "\n"

# Add SYMBOL_RISK_PCT_OVERRIDE_AUTO (new key)
rp_lines = ["SYMBOL_RISK_PCT_OVERRIDE_AUTO = {"]
for s in sorted(new_risk):
    rp_lines.append(f"    {s!r:<22}: {new_risk[s]},")
rp_lines.append("}")
rp_block = "\n".join(rp_lines)
if "SYMBOL_RISK_PCT_OVERRIDE_AUTO" in src:
    src = re.sub(r"^SYMBOL_RISK_PCT_OVERRIDE_AUTO = \{.*?^\}$",
                 rp_block, src, count=1, flags=re.DOTALL | re.MULTILINE)
else:
    src += "\n\n" + rp_block + "\n"

# Add COMPONENT_WEIGHTS_AUTO
cw_lines = ["COMPONENT_WEIGHTS_AUTO = {"]
for s in sorted(new_weights):
    w = new_weights[s]
    cw_lines.append(f"    {s!r:<22}: {dict(w)},")
cw_lines.append("}")
cw_block = "\n".join(cw_lines)
if "COMPONENT_WEIGHTS_AUTO" in src:
    src = re.sub(r"^COMPONENT_WEIGHTS_AUTO = \{.*?^\}$",
                 cw_block, src, count=1, flags=re.DOTALL | re.MULTILINE)
else:
    src += "\n\n" + cw_block + "\n"

(ROOT / "auto_tuned.py").write_text(src)

# Verify
importlib.reload(auto_tuned)
print(f"\n=== APPLIED to auto_tuned.py ===")
print(f"SIGNAL_QUALITY_SYMBOL_AUTO: {len(auto_tuned.SIGNAL_QUALITY_SYMBOL_AUTO)} symbols")
print(f"SYMBOL_RISK_PCT_OVERRIDE_AUTO: {len(getattr(auto_tuned, 'SYMBOL_RISK_PCT_OVERRIDE_AUTO', {}))} symbols")
print(f"COMPONENT_WEIGHTS_AUTO: {len(getattr(auto_tuned, 'COMPONENT_WEIGHTS_AUTO', {}))} symbols")
