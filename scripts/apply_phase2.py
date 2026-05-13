#!/usr/bin/env python3 -B
"""Apply Phase 2 hard tune winners (passed 5-fold walk-forward).

Reads:  backtest/results/phase2_hard_tune/<SYMBOL>.json
Updates: auto_tuned.py SL_OVERRIDE_AUTO + SUB_TP_R_OVERRIDE_AUTO + COOLDOWN_LOSS_OVERRIDE_AUTO
Only for symbols with phase2_winner != null (strict WF gate passed).
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIR = ROOT / "backtest" / "results" / "phase2_hard_tune"

apply_sl = {}
apply_tp = {}
apply_cd = {}
winners = []

for f in sorted(DIR.glob("*.json")):
    d = json.loads(f.read_text())
    w = d.get("phase2_winner")
    if not w: continue
    sym = d["symbol"]
    winners.append(sym)
    apply_sl[sym] = w["sl"]
    apply_tp[sym] = w["tp_r"]
    if w["cd_loss"] > 0:
        apply_cd[sym] = w["cd_loss"] * 60  # min → sec

print(f"Phase 2 WF-validated winners ({len(winners)}): {winners}")

# Load and patch auto_tuned.py
import importlib, auto_tuned  # noqa: E402
importlib.reload(auto_tuned)

src = (ROOT / "auto_tuned.py").read_text()
bk = ROOT / f"auto_tuned.py.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}-pre-phase2"
shutil.copy(ROOT / "auto_tuned.py", bk)
print(f"Backup: {bk}")

new_sl = dict(getattr(auto_tuned, "SL_OVERRIDE_AUTO", {}))
new_sl.update(apply_sl)

new_tp = dict(getattr(auto_tuned, "SUB_TP_R_OVERRIDE_AUTO", {}))
new_tp.update(apply_tp)

new_cd = dict(getattr(auto_tuned, "COOLDOWN_LOSS_OVERRIDE_AUTO", {}))
new_cd.update(apply_cd)

# Rewrite SL block
sl_lines = ["SL_OVERRIDE_AUTO = {"]
for s in sorted(new_sl): sl_lines.append(f"    {s!r:<22}: {new_sl[s]},")
sl_lines.append("}")
src = re.sub(r"SL_OVERRIDE_AUTO = \{[^}]*\}", "\n".join(sl_lines), src, count=1, flags=re.DOTALL)

# TP block
tp_lines = ["SUB_TP_R_OVERRIDE_AUTO = {"]
for s in sorted(new_tp): tp_lines.append(f"    {s!r:<22}: {new_tp[s]},")
tp_lines.append("}")
src = re.sub(r"SUB_TP_R_OVERRIDE_AUTO = \{[^}]*\}", "\n".join(tp_lines), src, count=1, flags=re.DOTALL)

# CD block
cd_lines = ["COOLDOWN_LOSS_OVERRIDE_AUTO = {"]
for s in sorted(new_cd): cd_lines.append(f"    {s!r:<22}: {new_cd[s]},")
cd_lines.append("}")
src = re.sub(r"COOLDOWN_LOSS_OVERRIDE_AUTO = \{[^}]*\}", "\n".join(cd_lines), src, count=1, flags=re.DOTALL)

(ROOT / "auto_tuned.py").write_text(src)

# Verify
importlib.reload(auto_tuned)
print(f"\nApplied to auto_tuned.py:")
for s in winners:
    print(f"  {s:<11} SL={auto_tuned.SL_OVERRIDE_AUTO[s]} "
          f"TP={auto_tuned.SUB_TP_R_OVERRIDE_AUTO.get(s)} "
          f"cd_loss={auto_tuned.COOLDOWN_LOSS_OVERRIDE_AUTO.get(s, 0)}s")
