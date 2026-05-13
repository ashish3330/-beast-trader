# Auto-synthesized 2026-05-13 from agent_tune_20260513 — DO NOT HAND-EDIT
# Apply: SL overrides + trail profile per symbol

SL_OVERRIDE_AUTO = {
    "AUDJPY": 1.0,
    "AUDUSD": 1.0,
    "COPPER-Cr": 1.0,
    "DJ30.r": 1.0,
    "ETHUSD": 1.0,
    "GAS-Cr": 3.0,
    "GER40.r": 1.0,
    "HK50.r": 1.0,
    "JPN225ft": 1.0,
    "SPI200.r": 3.0,
    "SWI20.r": 1.0,
    "USDCAD": 1.0,
    "XAGUSD": 1.0,
    "XAUUSD": 1.0,
}

TRAIL_PROFILE_AUTO = {
    "AUDJPY": "DEFAULT",  # WF +$65
    "AUDUSD": "AGGR_RUN",  # WF +$727
    "COPPER-Cr": "LOOSE",  # WF +$1209
    "DJ30.r": "DEFAULT",  # WF +$1837
    "ETHUSD": "AGGR_RUN",  # WF +$22
    "GAS-Cr": "DEFAULT",  # WF +$182
    "GER40.r": "LOOSE",  # WF +$383
    "HK50.r": "TIGHT",  # WF +$185
    "JPN225ft": "AGGR_RUN",  # WF +$919
    "SPI200.r": "DEFAULT",  # WF +$190
    "SWI20.r": "LOOSE",  # WF +$214
    "USDCAD": "LOOSE",  # WF +$816
    "XAGUSD": "DEFAULT",  # WF +$36
    "XAUUSD": "LOOSE",  # WF +$432
}

DISABLE_AUTO = ['CHFJPY', 'FRA40.r', 'GBPJPY']