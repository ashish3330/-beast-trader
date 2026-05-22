#!/usr/bin/env python3 -B
"""Trail-lag distribution + outlier inspection."""
import pandas as pd
from pathlib import Path

ROOT = Path('/Users/ashish/Documents/beast-trader/audit_20260522')
df = pd.read_csv(ROOT / 'parity_reaudit_detail.csv')
sl_agree = df[df['bt_reason'] == 'SL']
print('SL-agree:', len(sl_agree))
print('R-difference (live_r - bt_r) stats:')
print((sl_agree['live_r'] - sl_agree['bt_r']).describe())
print()
print('Distribution buckets:')
diff = sl_agree['live_r'] - sl_agree['bt_r']
print(f'  Within ±0.1R:        {((diff.abs() < 0.1)).sum()}')
print(f'  Within ±0.5R:        {((diff.abs() < 0.5)).sum()}')
print(f'  >+1.0R (live better):  {(diff > 1.0).sum()}')
print(f'  <-1.0R (BT better):    {(diff < -1.0).sum()}')
print()
print('=== SL_AGREE outliers where BT peak was high but live R near -1 (potential lingering trail-lag) ===')
outliers = sl_agree[(sl_agree['bt_peak_r'] > 0.5) & (sl_agree['live_r'] < -0.5)]
print(f'  count: {len(outliers)}, mean div_r: {outliers["divergence_r"].mean():.2f}, sum_div_r: {outliers["divergence_r"].sum():.2f}')
print(outliers[['id','symbol','live_reason','live_r','bt_r','divergence_r','bt_peak_r','bt_bars']].to_string())
print()
print('=== Within-1-bar exits (BT bt_bars<=2) — likely matched live exits ===')
fast = sl_agree[sl_agree['bt_bars'] <= 2]
print(f'  count: {len(fast)}, mean div_r: {fast["divergence_r"].mean():.2f}, sum: {fast["divergence_r"].sum():.2f}')
print()
print('=== Long-running BT trades (bt_bars >= 10) — most likely lagging path ===')
slow = sl_agree[sl_agree['bt_bars'] >= 10]
print(f'  count: {len(slow)}, mean div_r: {slow["divergence_r"].mean():.2f}, sum: {slow["divergence_r"].sum():.2f}')
