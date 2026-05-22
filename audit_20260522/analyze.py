#!/usr/bin/env python3 -B
"""Post-replay analysis: top divergence rows, trail-lag detection, gap quantification."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('/Users/ashish/Documents/beast-trader/audit_20260522')
df = pd.read_csv(ROOT / 'parity_reaudit_detail.csv')

print('=== Replayed trades:', len(df), '===')
print()

# === Refined categorization ===
# When live category is SL_OR_TP and BT also says SL → TRAIL_LAG or AGREE
# When live category is safety-layer (PeakGiveback, EarlyLossCut, etc.) and BT says SL → SAFETY_FIRED
def refine_cat(row):
    live_cat = row['category']
    bt_reason = str(row['bt_reason'])
    if live_cat in ('PEAK_GIVEBACK_EARLY', 'EARLY_LOSS_CUT_EARLY', 'AVG_WIN_LOSS_CAP_EARLY',
                    'HARD_DOLLAR_CAP', 'GUARDIAN_STALE', 'EXTERNAL'):
        return live_cat   # safety layer fired live, BT didn't have it
    if live_cat == 'SL_OR_TP':
        if bt_reason == 'SL':
            return 'SL_AGREE'
        if bt_reason == 'TIMEOUT':
            return 'BT_TIMEOUT_LIVE_SLTP'
        return 'TRAIL_LAG'
    return 'OTHER'

df['cat_refined'] = df.apply(refine_cat, axis=1)

print('=== Refined category divergence (R-basis) ===')
g = df.groupby('cat_refined').agg(
    n=('id', 'count'),
    live_r=('live_r', 'sum'),
    bt_r=('bt_r', 'sum'),
    div_r=('divergence_r', 'sum'),
    live_pnl=('live_pnl', 'sum'),
    bt_pnl=('bt_pnl', 'sum'),
    div_dollar=('divergence', 'sum'),
)
g['div_r_pct'] = (g['div_r'].abs() / g['div_r'].abs().sum() * 100).round(1)
print(g.sort_values('div_r').to_string())
print()

# Now compute per-symbol
print('=== Per-symbol R-divergence (positive=live earned more than BT) ===')
g = df.groupby('symbol').agg(
    n=('id', 'count'),
    live_r=('live_r', 'sum'),
    bt_r=('bt_r', 'sum'),
    div_r=('divergence_r', 'sum'),
    live_pnl=('live_pnl', 'sum'),
    bt_pnl=('bt_pnl', 'sum'),
    div_dollar=('divergence', 'sum'),
).sort_values('div_r')
print(g.to_string())
print()

# === Trail-lag specific: SL_AGREE rows ===
sl_agree = df[df['cat_refined'] == 'SL_AGREE']
print(f'=== SL_AGREE: {len(sl_agree)} trades — both hit SL ===')
print(f'  live R: {sl_agree["live_r"].sum():+.2f}R   BT R: {sl_agree["bt_r"].sum():+.2f}R')
print(f'  divergence R: {sl_agree["divergence_r"].sum():+.2f}R')
print(f'  live $: ${sl_agree["live_pnl"].sum():+.2f}   BT $: ${sl_agree["bt_pnl"].sum():+.2f}')
print(f'  Trail-lag indicator: live_r vs bt_r at same SL exit reason ⇒ different fill prices')
print()

# === Big winners in BT vs live (BT projected $$ but live cut) ===
print('=== TOP 10 trades where BT outperformed live (BT - live > 0) ===')
gainers = df.nsmallest(15, 'divergence_r')[['id','symbol','live_reason','live_r','bt_reason','bt_r','divergence_r','bt_peak_r','bt_bars','cat_refined']]
print(gainers.to_string())
print()

print('=== TOP 10 where live outperformed BT (BT - live < 0 ⇒ live earned more) ===')
gainers = df.nlargest(15, 'divergence_r')[['id','symbol','live_reason','live_r','bt_reason','bt_r','divergence_r','bt_peak_r','bt_bars','cat_refined']]
print(gainers.to_string())
print()

# === Specific: did AvgWinLossCap fire? ===
avg_win_cap = df[df['live_reason'].str.contains('AvgWinLossCap', na=False)]
print(f'=== AvgWinLossCap fires: {len(avg_win_cap)} ===')
if len(avg_win_cap) > 0:
    print(avg_win_cap[['id','symbol','live_reason','live_r','live_pnl','bt_pnl','bt_peak_r']].to_string())
print()

# === Compare to 2026-05-21 baseline ===
# Baseline: $850 of $1452 = 58.5% was TRAIL_LAG
# Today: total $ div = sum of div_dollar (clamped)
total_div_r = df['divergence_r'].sum()
total_div_dollar = df['divergence'].sum()
print(f'=== Today summary ===')
print(f'  Replayed: {len(df)} trades')
print(f'  Total R-divergence: {total_div_r:+.2f}R  (negative ⇒ live underperforms BT)')
print(f'  Total $-divergence (clamped): {total_div_dollar:+.2f}')
print()

# What is TRAIL_LAG share?
trail_lag = df[df['cat_refined'] == 'TRAIL_LAG']
sl_agree = df[df['cat_refined'] == 'SL_AGREE']
# For "trail lag" we want cases where live closed at TP/SL (= broker close, trailing SL hit)
# and BT also closed at SL. The dollar difference = trail-lag tax.
print(f'=== TRAIL_LAG (live closed at SL/TP, BT closed differently) ===')
print(f'  n={len(trail_lag)}, R-div: {trail_lag["divergence_r"].sum():+.2f}R, $-div: {trail_lag["divergence"].sum():+.2f}')
print()

# Save refined CSV
df.to_csv(ROOT / 'parity_reaudit_detail_refined.csv', index=False)

# Aggregate summary for report
summary = {
    'n_replayed': len(df),
    'n_skipped_total': 364,
    'total_div_r': float(total_div_r),
    'total_div_dollar': float(total_div_dollar),
    'category_breakdown': g.to_dict() if hasattr(g, 'to_dict') else None,
}
import json
json.dump({
    'n_replayed': len(df),
    'total_div_r': float(total_div_r),
    'total_div_dollar': float(total_div_dollar),
}, open(ROOT / 'analyze_summary.json', 'w'), indent=2)
