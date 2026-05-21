#!/usr/bin/env python3 -B
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
d3 = json.load(open(ROOT / "iter3_results.json"))
SYMS = ['DJ30.r', 'SWI20.r', 'XAUUSD', 'AUDJPY', 'EURUSD', 'US2000.r', 'UKOUSD', 'JPN225ft']
print(f"{'Symbol':<12} {'base':<8} {'d05_3':<8} {'d07_5':<8} {'d08_5':<8} {'d08_8':<8}")
print(f"{'(DD%)':<12}")
for s in SYMS:
    b = d3['results']['baseline'][s]['dd']
    d05 = d3['results']['deep_05_3bar'][s]['dd']
    d07 = d3['results']['deep_07_5bar'][s]['dd']
    d08_5 = d3['results']['deep_08_5bar'][s]['dd']
    d08_8 = d3['results']['deep_08_8bar'][s]['dd']
    print(f'{s:<12} {b:<8.1f} {d05:<8.1f} {d07:<8.1f} {d08_5:<8.1f} {d08_8:<8.1f}')
