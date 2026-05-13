#!/usr/bin/env python3 -B
"""Apply full SL × TP × cooldown tune to live config.

Reads:  backtest/results/full_tune_20260513/<SYMBOL>.json
Writes: updates auto_tuned.py SL_OVERRIDE_AUTO
        updates SUB_TP_R_OVERRIDE_AUTO (new key)
        updates SYMBOL_COOLDOWN_LOSS_OVERRIDE (new key)
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIR = ROOT / "backtest" / "results" / "full_tune_20260513"
LIVE = {"XAUUSD","XAGUSD","BTCUSD","ETHUSD","DJ30.r","GER40.r","HK50.r","JPN225ft","SPI200.r","SWI20.r","US2000.r","COPPER-Cr","GAS-Cr","NG-Cr","UKOUSD","AUDJPY","CADJPY","AUDUSD","EURAUD","EURUSD","USDCAD","USDCHF"}

# Decision rule: APPLY only if WF PnL > 0 AND WF PF > 1.3
sl_overrides = {}
tp_overrides = {}
cd_loss_overrides = {}
cd_win_overrides = {}
disable_syms = []
apply_syms = []

for f in sorted(DIR.glob("*.json")):
    sym = f.stem
    if sym not in LIVE: continue
    d = json.loads(f.read_text())
    if not d.get("best"):
        disable_syms.append(sym); continue
    b = d["best"]
    wf = d.get("walk_forward_60d") or {}
    wf_pnl = wf.get("pnl", 0)
    wf_pf = wf.get("pf", 0)

    if wf_pnl <= 0 or wf_pf < 1.3:
        # Disable if both IS and WF look bad
        if b["pnl"] < 0 or b["pf"] < 1.3:
            disable_syms.append(sym)
            continue
        # Keep current params (don't apply unproven)
        continue

    apply_syms.append(sym)
    sl_overrides[sym] = b["sl"]
    tp_overrides[sym] = b["tp_r"]
    if b["cd_loss_min"] > 0:
        cd_loss_overrides[sym] = b["cd_loss_min"] * 60  # convert to seconds
    if b["cd_win_min"] > 0:
        cd_win_overrides[sym] = b["cd_win_min"] * 60

print(f"APPLY ({len(apply_syms)}): {apply_syms}")
print(f"DISABLE ({len(disable_syms)}): {disable_syms}")
print()

# Update auto_tuned.py SL_OVERRIDE_AUTO
src = (ROOT / "auto_tuned.py").read_text()
bk = ROOT / f"auto_tuned.py.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
shutil.copy(ROOT / "auto_tuned.py", bk)
print(f"Backup: {bk}")

import importlib, auto_tuned
importlib.reload(auto_tuned)
new_sl = dict(getattr(auto_tuned, "SL_OVERRIDE_AUTO", {}))
new_sl.update(sl_overrides)

import re
sl_lines = ["SL_OVERRIDE_AUTO = {"]
for sym in sorted(new_sl): sl_lines.append(f"    {sym!r:<22}: {new_sl[sym]},")
sl_lines.append("}")
src = re.sub(r"SL_OVERRIDE_AUTO = \{[^}]*\}", "\n".join(sl_lines), src, count=1, flags=re.DOTALL)

# Append new SUB_TP_R_OVERRIDE_AUTO at the end if not present
if "SUB_TP_R_OVERRIDE_AUTO" not in src:
    src += "\n\n# Per-symbol SUB_TP_R override (TP ladder)\nSUB_TP_R_OVERRIDE_AUTO = {\n"
    for sym, tp in sorted(tp_overrides.items()): src += f"    {sym!r:<22}: {tp},\n"
    src += "}\n"
else:
    tp_lines = ["SUB_TP_R_OVERRIDE_AUTO = {"]
    for sym, tp in sorted(tp_overrides.items()): tp_lines.append(f"    {sym!r:<22}: {tp},")
    tp_lines.append("}")
    src = re.sub(r"SUB_TP_R_OVERRIDE_AUTO = \{[^}]*\}", "\n".join(tp_lines), src, count=1, flags=re.DOTALL)

if "COOLDOWN_LOSS_OVERRIDE_AUTO" not in src and cd_loss_overrides:
    src += "\n\n# Per-symbol cooldown overrides (seconds)\nCOOLDOWN_LOSS_OVERRIDE_AUTO = {\n"
    for sym, cd in sorted(cd_loss_overrides.items()): src += f"    {sym!r:<22}: {cd},\n"
    src += "}\n"

(ROOT / "auto_tuned.py").write_text(src)
print(f"Updated auto_tuned.py: {len(sl_overrides)} SL, {len(tp_overrides)} TP, {len(cd_loss_overrides)} CD")

# Report disabled symbols (manual config edit needed)
print(f"\nDISABLE (remove from config.py SYMBOLS dict): {disable_syms}")
