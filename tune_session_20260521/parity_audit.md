# Live↔BT Parity Audit — 2026-05-21

Audit window: last 30 days of mt5_deal exit-legs from `data/trade_journal.db`.
Replay: per-(symbol, direction, entry_price) clusters mapped to H1 cache bar; simulate_trail() invoked with live SYMBOL_ATR_SL_OVERRIDE / SYMBOL_TRAIL_OVERRIDE / regime overrides.

## Summary stats (last 30 days)

- MT5 deal-legs scanned: 387
- Logical entries clustered: 324
- Replayable entries (after price+ATR match): 150
- UNREPLAYABLE / dropped: 174 (53.7% of clusters — reasons: bar/price mismatch >0.5%, missing cache, no ATR)
- Entries with |Δ| > $1: 118 (79%)
- Entries with category mismatch (live exit ≠ BT SL/TIMEOUT): 124 (83%)
- Net Δ live − BT: $+115.68
- |Δ| sum (total drift magnitude): $1452.52
- Worst-divergence symbol: NAS100.r (|Δ| $167.07 over 4 entries)

## Divergence breakdown by category

| category | n | net Δ ($) | |Δ| ($) |
|---|---:|---:|---:|
| TRAIL_LAG | 53 | +45.48 | 850.00 |
| SLIPPAGE | 31 | +188.57 | 278.55 |
| OK | 26 | +0.88 | 9.84 |
| EXIT_MISMATCH | 23 | -138.16 | 225.30 |
| EXTERNAL_CLOSE | 17 | +18.91 | 88.83 |

## Top 15 worst-divergence entries

| # | symbol | dir | entry | live_exit | bt_exit | live_pnl | bt_pnl | Δ | category | reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | NAS100.r | L | 28066.8 | 28277 | 28438.7 | +21.02 | +88.25 | -67.23 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 2.36× ATR |
| 2 | XAGUSD | S | 71.783 | 71.647 | 71.9406 | +26.90 | -39.78 | +66.68 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 0.56× ATR |
| 3 | NAS100.r | L | 27943.7 | 27995.8 | 28187.6 | +5.22 | +65.87 | -60.65 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 2.52× ATR |
| 4 | GBPUSD | S | 1.34986 | 1.35025 | 1.34939 | -20.95 | +38.05 | -59.00 | EXIT_MISMATCH | TP_broker: live closed at fixed TP; BT trails open |
| 5 | XAUUSD | L | 4620.24 | 4618.02 | 4604.8 | +21.54 | -23.52 | +45.06 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 0.60× ATR |
| 6 | USDJPY | L | 159.664 | 159.666 | 159.732 | -19.50 | +24.43 | -43.93 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 0.73× ATR |
| 7 | SP500.r | L | 7269.53 | 7277.03 | 7328.84 | +1.50 | +43.40 | -41.90 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 5.01× ATR |
| 8 | SP500.r | L | 7247.89 | 7235.55 | 7211.69 | -8.66 | -49.41 | +40.75 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 1.32× ATR |
| 9 | NAS100.r | L | 27438.5 | 27447.3 | 27276 | +1.75 | -37.02 | +38.77 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 1.58× ATR |
| 10 | JPN225ft | S | 58804.2 | 59018.8 | 59450.5 | -11.55 | -46.57 | +35.02 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 1.67× ATR |
| 11 | USDJPY | L | 159.846 | 159.946 | 160.07 | +22.60 | +57.58 | -34.98 | EXIT_MISMATCH | TP_broker: live closed at fixed TP; BT trails open |
| 12 | GBPJPY | L | 216.521 | 216.256 | 216.181 | -2.44 | -36.80 | +34.36 | SLIPPAGE | SL_broker: broker SL hit; BT also SL but at different price |
| 13 | EURUSD | S | 1.16738 | 1.16796 | 1.16728 | -21.71 | +5.97 | -27.68 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 0.64× ATR |
| 14 | XAGUSD | L | 87.104 | 87.3455 | 88.697 | +8.70 | +35.91 | -27.21 | TRAIL_LAG | SL_broker: live SL trailed differently — exit-price gap 1.53× ATR |
| 15 | UK100.r | S | 10183.9 | 10235.1 | 10246.6 | -3.44 | -30.27 | +26.83 | SLIPPAGE | SL_broker: broker SL hit; BT also SL but at different price |

## Per-symbol divergence summary

| symbol | entries | net Δ | |Δ| | avg Δ/entry | dominant live exit |
|---|---:|---:|---:|---:|---|
| NAS100.r | 4 | -88.69 | 167.07 | -22.17 | SL_broker (3) |
| XAGUSD | 8 | +42.20 | 144.02 | +5.27 | SL_broker (6) |
| SP500.r | 12 | +16.11 | 135.35 | +1.34 | SL_broker (10) |
| USDCAD | 11 | +51.62 | 119.90 | +4.69 | SL_broker (8) |
| XAUUSD | 15 | +43.07 | 118.11 | +2.87 | SL_broker (12) |
| GBPUSD | 5 | -27.77 | 103.77 | -5.55 | SL_broker (3) |
| USDJPY | 5 | -58.35 | 99.47 | -11.67 | SL_broker (3) |
| JPN225ft | 15 | +60.69 | 87.69 | +4.05 | SL_broker (8) |
| EURUSD | 10 | -34.82 | 75.72 | -3.48 | SL_broker (8) |
| GBPJPY | 7 | +73.96 | 73.96 | +10.57 | SL_broker (6) |
| BTCUSD | 21 | -16.65 | 63.51 | -0.79 | SL_broker (13) |
| UK100.r | 4 | +14.00 | 53.68 | +3.50 | SL_broker (3) |
| ETHUSD | 9 | -7.49 | 46.47 | -0.83 | SL_broker (8) |
| GER40.r | 7 | -10.66 | 42.68 | -1.52 | SL_broker (5) |
| EURJPY | 4 | -25.68 | 27.06 | -6.42 | SL_broker (4) |
| EURAUD | 3 | +24.87 | 25.89 | +8.29 | SL_broker (2) |
| AUDJPY | 2 | +24.17 | 24.17 | +12.09 | SL_broker (2) |
| AUDUSD | 3 | +20.08 | 20.08 | +6.69 | SL_broker (2) |
| USDCHF | 3 | +18.13 | 19.95 | +6.04 | SL_broker (2) |
| EURGBP | 2 | -3.11 | 3.97 | -1.55 | SL_broker (2) |

## Suggested fixes (root-cause first)

0. **TRAIL_LAG dominates** — 53 broker-SL entries had exit-price gap > 0.5×ATR (|Δ| sum $850.00, net $+45.48). This means live SL was at a materially different price than BT's trailed SL at the same bar — most likely cause: per-tick trail updates in live (executor.py:1337-1422) tighten SL faster than BT's per-bar simulate_trail. Fix: change simulate_trail to use intra-bar high/low for trail check (currently uses close only at v5_backtest.py:400).
1. **EarlyLossCut over-firing** — 11 entries closed via tiered early-cut, net Δ $-83.29 vs BT-trail. Live cuts at -2R / -2.5R / -3R tiers (executor.py:1422) while BT lets SL hit at the configured ATR distance. Either: (a) widen BT SL multiplier to match the effective early-cut R, or (b) port EarlyLossCut into simulate_trail with the live tier thresholds.
2. **PeakGiveback unmodeled** — 6 winners closed by peak-R giveback (executor.py:1337), net Δ $+1.10. BT's ratchet (0.2R/0.5R floors) plus trail-lock differs. Mirror executor's peak-giveback check inside simulate_trail.
3. **Portfolio guardians not in BT** — 17 entries killed by EmergencyDD / GuardianDayLoss / GuardianStale / HardDollarCap / DailyKillSwitch / GuardianHeat / GuardianSharp (net Δ $+18.91). These are PORTFOLIO-level rules — single-symbol BT cannot model the cross-symbol equity coupling. Acceptable structural gap; flag remains as long as it doesn't dominate the divergence.
4. **Broker SL slippage** — 31 broker-SL exits diverged by net Δ $+188.57. Live SL fills at actual broker price; BT fills at exact SL level. Enable `with_slippage` in backtest cost overlay to compare apples-apples.

---

## Methodology notes

- Cluster: legs sharing identical `(symbol, direction, entry_price)` are treated as a single logical entry (Dragon opens 3 pyramid legs per signal).
- BT entry-bar match: locate H1 bar prior to first-leg timestamp, then refine within ±8 bars to the bar whose close best matches recorded entry_price. Tolerance 0.5%; mismatches → UNREPLAYABLE.
- BT exit: `simulate_trail()` with live SL/trail/regime overrides (SYMBOL_ATR_SL_OVERRIDE_REGIME → SYMBOL_ATR_SL_OVERRIDE → DEFAULT).
- Live PnL: sum of leg `pnl` from journal (post-execution actual). BT PnL$: bt_pnl_r × median($/R from legs) × legs_n.
- BT-only exit reasons modeled: SL, TIMEOUT. Live-only exit families automatically flagged as divergence regardless of dollar Δ.