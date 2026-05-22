#!/usr/bin/env python3 -B
"""Test reload behavior of mutated dict."""
import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path("/Users/ashish/Documents/beast-trader")))

import config as cfg
print("XAUUSD initial:", cfg.SIGNAL_QUALITY_SYMBOL.get("XAUUSD"))
print("id of dict:", id(cfg.SIGNAL_QUALITY_SYMBOL))

cfg.SIGNAL_QUALITY_SYMBOL["XAUUSD"]["volatile"] = 55
print("After mutation:", cfg.SIGNAL_QUALITY_SYMBOL.get("XAUUSD"))
print("id of dict:", id(cfg.SIGNAL_QUALITY_SYMBOL))

importlib.reload(cfg)
print("After reload:", cfg.SIGNAL_QUALITY_SYMBOL.get("XAUUSD"))
print("id of dict:", id(cfg.SIGNAL_QUALITY_SYMBOL))

# Force re-import of auto_tuned too
import auto_tuned as at
importlib.reload(at)
importlib.reload(cfg)
print("After reload(at)+reload(cfg):", cfg.SIGNAL_QUALITY_SYMBOL.get("XAUUSD"))
