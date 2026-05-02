# Auto-generated SYMBOL_RISK_CAP overrides from tune_180d_pass1.json
# Generated 2026-05-02 — based on observed best.result PF/DD/trades.
#
# Rules applied:
#   dd>=12%                              -> 0.5%
#   dd 8-12% AND pf<2.0                  -> 0.6%
#   dd<5% AND pf>=3.0 AND trades>=150    -> 1.2%
#   dd<5% AND pf>=2.5 AND trades>=100    -> 1.0%
#   otherwise omitted (use default 0.8% from config)
#
# Symbols with negative PnL (EURGBP, EURNZD, NZDJPY, UKOUSD) are skipped — they
# should be excluded from the live set, not risk-tuned.

RISK_CAP_AUTO = {
    # Boosted — winners with low DD
    "XAUUSD":  1.2,   # pf=4.83 dd=3.0 trades=267
    "EURAUD":  1.2,   # pf=3.91 dd=4.0 trades=270
    "BTCUSD":  1.0,   # pf=2.75 dd=4.1 trades=698
    "XAGUSD":  1.0,   # pf=2.79 dd=4.2 trades=201

    # Capped down — elevated DD with mediocre PF
    "BCHUSD":  0.6,   # pf=1.05 dd=8.3 trades=100
    "EURCAD":  0.6,   # pf=1.73 dd=8.1 trades=61
    "GBPCHF":  0.6,   # pf=1.61 dd=8.3 trades=157
    "NZDUSD":  0.6,   # pf=1.08 dd=8.1 trades=138
    "XPDUSD.r": 0.6,  # pf=1.28 dd=8.2 trades=265
}
