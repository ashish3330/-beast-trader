#!/usr/bin/env python3 -B
"""One-shot: add top-level 'recommendation' block to 03_volume.json."""
import json
p = '/Users/ashish/Documents/beast-trader/entry_research_20260522/03_volume.json'
with open(p) as f:
    data = json.load(f)

data['recommendation'] = {
    "ship": True,
    "variant": "vwap_filter_band_0.5atr",
    "strategy": "vwap_side_with_atr_band",
    "atr_band": 0.5,
    "param_key_suggestion": "vwap_side_filter_enabled",
    "rationale": "Cleanest robustness profile: +$5,821 portfolio over 180d/8sym, "
                 "no symbol regressed > $104, all 5 WF folds positive, WF avg PF 3.92, "
                 "DD reduced 21%. Filter rejects ~5-10% marginal counter-VWAP entries.",
    "summary_metrics": data['iter2']['variants']['vwap_filter_band_0.5atr'],
    "caveats": [
        "Tested only on 8 long-history symbols (ETHUSD/GBPUSD/EURUSD/GBPJPY/EURJPY/USDCAD/GER40.r/SP500.r).",
        "XAUUSD/XAGUSD/BTCUSD/USDJPY/NAS100.r/JPN225ft H1 caches only have 20-29 days; re-validate before live deploy.",
        "ind['vwap'] is a 20-bar rolling VWAP proxy already in the scorer, not a session-VWAP.",
    ],
    "rejected_variants": [
        "tick_ratio_filter_* (all thresholds 1.0/1.2/1.5/2.0; regressed $8-16K)",
        "poc_hvn_filter_* (regressed $14-17K - mean-reversion drag)",
        "vwap_*_boost (PnL gains but GBPUSD flips negative; robustness fail)",
        "vwap_AND_tick (composite strictness kills edge)",
    ],
}
with open(p, 'w') as f:
    json.dump(data, f, indent=2, default=str)
print("Recommendation added.")
print("Final recommendation:", data['recommendation']['variant'])
print("delta portfolio:", data['recommendation']['summary_metrics']['delta_full'])
