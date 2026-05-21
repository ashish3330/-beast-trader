# Pullback entry research — 2026-05-22

## TL;DR

Two industry-grade findings, one of which ships:

### Finding 1: Pullback-as-GATE is wrong for this strategy (REJECT)

All five textbook pullback **gating** strategies (Fib retracement, EMA20/EMA50 kiss, Order Block re-entry, BB middle, trendline retest) cut PnL by 95-100% across the top 8 symbols. The current bot is a **momentum-rider**: 100% of trades exit on the trailing-SL (no fixed TPs are ever hit), and average R-multiple is 0.4-0.95 with the occasional big runner. Hard-gating signals to "wait for a pullback" removes the breakout entries that pay the bills. The bot is not, and should not be, a dip-buyer at the signal level.

| Variant       | Top-8 180d Δ |
|---------------|--------------|
| fib_50        | -$54,100     |
| ema20 (gate)  | -$53,425     |
| order_block   | -$53,848     |
| bb_mid        | -$54,361     |
| trendline     | -$53,880     |

### Finding 2: Pullback-as-FILL-DEPTH ships big (deep_08_5bar)

The live brain already places a limit order 0.2 ATR from signal price and waits 1 bar before falling back to direct entry (config: `PULLBACK_ATR_RETRACE=0.2`, `PULLBACK_MAX_WAIT_BARS=1`). Deepening to **0.8 ATR / 5 bars** materially improves every test metric:

| Symbol     | Baseline PnL | deep_08_5bar PnL | Δ        | DD%: base → new |
|------------|-------------:|-----------------:|---------:|-----------------|
| DJ30.r     |     $11,104  |          $16,559 | +$5,455  | 5.6 → 3.1       |
| SWI20.r    |        $738  |             $930 | +$191    | 2.5 → 2.5       |
| XAUUSD     |        $456  |             $737 | +$282    | 2.9 → 2.5       |
| AUDJPY     |      $4,835  |           $6,633 | +$1,798  | 4.6 → 4.0       |
| EURUSD     |      $2,080  |           $3,939 | +$1,859  | 7.8 → 4.1       |
| US2000.r   |     $26,147  |          $40,332 | +$14,185 | 6.4 → 3.8       |
| UKOUSD     |      $6,979  |          $46,362 | +$39,383 | 6.4 → 4.2       |
| JPN225ft   |      $2,022  |           $2,358 | +$337    | 6.0 → 4.0       |
| **Total**  |    $54,361   |         $117,849 | **+$63,489** | uniformly lower |

**OOS confirmation** (10 other live symbols, never used in tuning): +$23,730 / 7/10 wins. Only minor regressions on GBPUSD (-$374) and XAGUSD (-$143).

**Walk-forward** (5-fold, 36-day folds, full 18-symbol portfolio):

| Variant          | F1      | F2      | F3      | F4      | F5      | Sum     |
|------------------|---------|---------|---------|---------|---------|---------|
| baseline         | $10,220 | $5,456  | $1,998  | $4,794  | $4,601  | $27,069 |
| **deep_08_5bar** | $14,379 | $7,550  | $4,103  | $8,575  | $7,113  | $41,720 |

deep_08_5bar wins **every fold**, total summed-folds gain **+$14,651 (+54%)**.

Average per-symbol PF across folds: baseline 3.30 → deep_08_5bar 4.99.

## Mechanism (why it works)

Median ATR retrace probability within 5 H1 bars (UKOUSD profile, ~2.6K bars):
- 0.5 ATR: 67.6%
- 0.7 ATR: 57.6%
- 0.8 ATR: 53.1%

So a limit order 0.8 ATR away from signal close gets filled ~53% of the time. The 47% no-fill cases fall back to direct entry (no worse than baseline). When the limit fills, you enter 0.8 ATR closer to the SL — meaning your R-multiple per trade is materially improved (SL is fixed ATR-distance from entry; better fill = lower SL = bigger TP headroom). Average improvement per fill: 0.84 ATR, exactly the target depth.

The strategy fundamentally exploits intra-bar mean-reversion on the **same** signal — the entry isn't chasing the breakout but giving the market a chance to retrace before committing. When momentum is genuine, the retrace is shallow and we still take the trade; when it's a false breakout, we save the loss (or get a better entry on the eventual move).

## Verdict: SHIP

**Recommended config delta:**

```python
# config.py
PULLBACK_ATR_RETRACE   = 0.8   # was 0.2
PULLBACK_MAX_WAIT_BARS = 5     # was 1
# PULLBACK_ENTRY_ENABLED = True  (already on)
# PULLBACK_REGIMES = {"trending", "volatile"}  (already configured)
```

No code changes needed — the live `brain.py:1942-1967` framework already implements the deferred-limit-fill + fallback semantics. This is a pure tune.

### Caveats / live-watch list

- **Slippage**: Backtest uses spread but no slippage. Real fills on a 0.8 ATR limit order could be 0.05-0.15 ATR worse on volatile bars. Expected impact: ~10-20% haircut on the $63K projected gain. Still a clear win.
- **UKOUSD concentration**: ~62% of the top-8 gain comes from UKOUSD alone (+$39K of $63K). Oil's intra-bar volatility may be more retrace-prone than other markets. The OOS test on 10 other symbols (+$23K) still wins, so this isn't a single-symbol artifact, but watch UKOUSD specifically in the first week.
- **GBPUSD / XAGUSD**: Minor OOS regressions. Consider per-symbol opt-out if they continue to regress in live for 5+ trades.
- **Demo first**: Per project_dragon_demo_week_20260514 — 7-day demo window before real money. Watch for limit-order partial-fill behaviour at the broker.

## Files

- `01_pullback.json` — full result blob with all iterations
- `pullback_bt.py` — inline backtest with gating variants (iter 1)
- `pullback_bt_v2.py` — inline backtest with fill-depth variants (iter 2-4)
- `walkforward.py`, `walkforward_iter3.py`, `walkforward_final.py` — 5-fold WF
- `iter1_results.json`, `iter2_results.json`, `iter3_results.json`, `iter4_results.json`, `walkforward_final.json` — raw data
- `baseline.json` — v5_backtest sanity-check baseline
- `inspect_ukousd.py` — mechanism analysis
- `eligibility.py`, `compare_dd.py` — post-hoc scoring

## Iteration audit

| Iter | Approach                                  | Outcome    |
|------|-------------------------------------------|------------|
| 1    | Hard pullback gates (fib/ema/OB/BB/TL)    | REJECT     |
| 2    | Deeper limit-fill with fallback           | INVESTIGATE|
| 3    | Sensitivity sweep (0.4–1.0 ATR × 3-8 bar) | INVESTIGATE|
| 4    | Realistic fallback (next_open) stress     | EDGE HOLDS |
| 5    | OOS + full-portfolio walk-forward         | SHIP       |

Five iterations used, winner found in iter 3, validated in iter 4-5.
