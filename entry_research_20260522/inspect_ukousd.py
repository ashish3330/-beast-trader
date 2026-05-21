#!/usr/bin/env python3 -B
"""Inspect UKOUSD deep_08_5bar to understand the huge edge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from signals.momentum_scorer import _compute_indicators, _score_with_components, IND_DEFAULTS, IND_OVERRIDES, _ema
from backtest.v5_backtest import (
    DEFAULT_PARAMS, ALL_SYMBOLS, SL_OVERRIDE, SL_OVERRIDE_REGIME,
    TRAIL_OVERRIDE, RISK_CAP, RISK_CAP_REGIME,
    DIR_BIAS, TOXIC_HOURS, TOXIC_EXEMPT, SESSION,
    _dir_bias_for_regime, get_regime, load_data,
)

SYM = "UKOUSD"
DAYS = 180
df = load_data(SYM, DAYS)
print(f"UKOUSD: {len(df)} bars, {df['time'].min()} → {df['time'].max()}")

# Indicators
icfg = {**IND_DEFAULTS, **IND_OVERRIDES.get(SYM, {})}
ind = _compute_indicators(df, icfg)
n = len(df)
c = df["close"].values.astype(float)
h = df["high"].values.astype(float)
l = df["low"].values.astype(float)
atrs = ind["at"]

# Look at distribution of "deep pullback hit rate" by ATR depth
sample_size = 0
hit_rates = {0.3: 0, 0.5: 0, 0.7: 0, 0.8: 0, 1.0: 0}
for i in range(100, n - 10):
    if not np.isfinite(atrs[i]) or atrs[i] == 0: continue
    sample_size += 1
    atr = float(atrs[i])
    close_now = float(c[i])
    for depth in hit_rates:
        target = close_now - depth * atr
        # check next 5 bars
        for k in range(1, 6):
            if i + k >= n: break
            if l[i + k] <= target:
                hit_rates[depth] += 1
                break

print(f"\nDeep pullback hit rates over {sample_size} bars (LONG only, within 5 bars):")
for depth, hits in sorted(hit_rates.items()):
    print(f"  {depth} ATR: {hits} ({100.0 * hits / sample_size:.1f}%)")

# Look at actual fill price improvement vs baseline
# We compute: if we waited up to 5 bars and target is 0.8 ATR below close,
# what's the avg fill price improvement vs c[i]?
total_imp = 0; n_filled = 0; n_signals = 0
for i in range(100, n - 10):
    if not np.isfinite(atrs[i]) or atrs[i] == 0: continue
    # only count "would-be signals" — bars with moderate score
    long_s, short_s, _, _ = _score_with_components(ind, i)
    if max(long_s, short_s) < 4.5:  # rough quality threshold proxy
        continue
    n_signals += 1
    atr = float(atrs[i])
    close_now = float(c[i])
    if long_s >= short_s:
        target = close_now - 0.8 * atr
        for k in range(1, 6):
            if i + k >= n: break
            if l[i + k] <= target:
                total_imp += (close_now - target)
                n_filled += 1
                break

print(f"\nSignal-level fill stats (~{n_signals} signals):")
print(f"  Pullback fills: {n_filled} ({100*n_filled/max(n_signals,1):.1f}%)")
if n_filled > 0:
    avg_imp = total_imp / n_filled
    print(f"  Avg price improvement per fill: {avg_imp:.4f} (in price units)")
    # Convert to ATR
    avg_atr = np.nanmean(atrs[100:])
    print(f"  Avg ATR over period: {avg_atr:.4f}")
    print(f"  Improvement in ATR units: {avg_imp / avg_atr:.2f}")
