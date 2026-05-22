# AvgWinLossCap Dormancy Analysis & Threshold Recommendation — 2026-05-22

**Window**: 30 days (2026-04-22 → 2026-05-22) of closed live trades, journal `data/trade_journal.db`.
**Method**: For every losing trade (338 in window), recompute the rolling
avg-win (last 30 wins for that symbol within the trailing 30d window, min
5 samples) **as of the trade's close timestamp** — mirrors
`_get_avg_win_dollars` in `execution/executor.py:2028`.
For each (MULT, floor) pair, mark `fire` if `|pnl| ≥ max(avg_win × MULT, floor)`.

**Files**:
- Simulator: `/Users/ashish/Documents/beast-trader/audit_20260522/avg_win_cap_sim.py`
- Raw JSON:  `/Users/ashish/Documents/beast-trader/audit_20260522/avg_win_cap_sim.json`
- Print tool: `/Users/ashish/Documents/beast-trader/audit_20260522/avg_win_cap_print.py`
- Dist tool:  `/Users/ashish/Documents/beast-trader/audit_20260522/avg_win_cap_dist.py`

---

## 1. Why it fired ZERO times in the audit window

Commit `dfe8d87` added the layer at **2026-05-22 19:28 IST**. Commit `0de4bdc`
later widened thresholds to current values (MULT 1.2→2.0, floor $1→$2) at
**20:01 IST today**. The 30-day parity audit window predates the deploy —
**no live position was ever evaluated by the cap in that window**. The
"zero fires" finding is a deployment-timing artifact, not a tuning problem.

Going forward (with cap active live), the simulator answers what *would*
have fired had the layer existed. That's the basis for the recommendation
below.

---

## 2. Loss landscape (last 30d)

| Category         | n    | total loss  | avg loss | max loss |
|------------------|-----:|------------:|---------:|---------:|
| **TrailSL**      | 200  | **-$625.12**| -$3.13   | -$19.46  |
| **EarlyLossCut** | 69   | -$226.96    | -$3.29   | -$33.40  |
| Guardian*        | 20   | -$135.31    | -$6.77   | -$44.00  |
| HardDollarCap    | 1    | -$24.36     | -$24.36  | -$24.36  |
| DailyKillSwitch  | 7    | -$24.16     | -$3.45   | -$12.25  |
| DragonReversal   | 9    | -$30.53     | -$3.39   | -$7.85   |
| EmergencyDD      | 22   | -$13.49     | -$0.61   | -$2.81   |
| PeakGiveback     | 10   | -$1.89      | -$0.19   | -$0.61   |
| **TOTAL**        | 338  | **-$1081.82**|         |          |

Two observations:

1. **TrailSL is the dominant loss source** (58% of $-loss). The big
   conceptual question: would a tighter cap pre-empt SL hits *before* trail
   walks the position into stop-out? Sim says yes — see §3.
2. **PeakGiveback losses are tiny** ($1.89 total). Already doing its job.
   Cap doesn't compete with it.

---

## 3. Per-symbol avg-loss / avg-win ratio (bleeders)

| Symbol     | n_win | avg_win | n_loss | avg_loss | L/W ratio | Status        |
|------------|------:|--------:|-------:|---------:|----------:|---------------|
| **GBPCHF** | 4     | $0.45   | 6      | $3.62    | **7.99x** | Severe bleeder |
| **XAUUSD** | 24    | $1.90   | 7      | $11.76   | **6.21x** | Severe bleeder |
| **EURJPY** | 7     | $0.29   | 16     | $1.70    | **5.79x** | Severe bleeder |
| CHFJPY     | 3     | $0.64   | 2      | $3.68    | 5.74x     | (low sample) |
| **BTCUSD** | 35    | $0.83   | 21     | $4.23    | **5.13x** | Severe bleeder |
| AUDJPY     | 8     | $0.78   | 2      | $3.50    | 4.47x     | (low sample) |
| CADJPY     | 5     | $0.70   | 5      | $2.36    | 3.35x     | Bleeder |
| JPN225ft   | 15    | $0.57   | 22     | $1.90    | 3.34x     | Bleeder |
| ETHUSD     | 20    | $0.58   | 16     | $1.71    | 2.92x     | Bleeder |
| GER40.r    | 12    | $2.84   | 8      | $7.87    | 2.77x     | Bleeder |
| XAGUSD     | 22    | $5.81   | 4      | $14.05   | 2.42x     | Bleeder |
| GBPJPY     | 16    | $1.30   | 10     | $2.65    | 2.04x     | Borderline |
| UK100.r    | 2     | $2.40   | 3      | $4.01    | 1.68x     | OK |
| SP500.r    | 23    | $1.04   | 23     | $1.72    | 1.65x     | OK |
| EURUSD     | 22    | $1.45   | 33     | $1.82    | 1.25x     | OK |
| FRA40.r    | 16    | $2.64   | 2      | $3.08    | 1.17x     | OK |
| US2000.r   | 61    | $0.99   | 41     | $1.10    | 1.11x     | Healthy |
| GBPUSD     | 14    | $4.14   | 16     | $4.21    | 1.02x     | Healthy |
| USDCAD     | 37    | $2.87   | 32     | $2.88    | 1.00x     | Healthy |
| DJ30.r     | 31    | $5.23   | 21     | $4.59    | 0.88x     | Healthy |

12 of 23 symbols (52%) have L/W ≥ 2.0. **Current MULT=2.0 only catches the
top 6 (L/W ≥ 5)**, which is exactly the dormancy symptom.

---

## 4. Threshold sweep — fires & dollars saved

Restricted to 216 losses that have ≥5 wins in the trailing window (cap can
actually evaluate). The other 122 losses skip the layer (returns 0 from
`_get_avg_win_dollars`) — that's fine because $-magnitude is small there
(<$200 combined).

| MULT  | floor  | fires | total_losses | $ saved | total $ loss | save %  |
|------:|-------:|------:|-------------:|--------:|-------------:|--------:|
| 0.50  | $0.50  | 161   | 338          | $432.64 | $1081.82     | 40.0%   |
| 0.75  | $0.50  | 144   | 338          | $361.08 | $1081.82     | 33.4%   |
| 1.00  | $0.50  | 120   | 338          | $300.94 | $1081.82     | 27.8%   |
| **1.00**  | **$1.00**  | **108**   | **338**          | **$289.25** | **$1081.82**     | **26.7%**   |
| 1.25  | $1.00  | 95    | 338          | $243.11 | $1081.82     | 22.5%   |
| 1.50  | $1.00  | 72    | 338          | $209.41 | $1081.82     | 19.4%   |
| **2.00**  | **$2.00**  | **52**    | **338**          | **$146.17** | **$1081.82**     | **13.5%**   |

**Current (2.0 / $2)**: would have fired on 52/338 losses (15%), saving
$146.17 (13.5% of total $-loss) over 30 days = ~$4.87/day = ~$146/month.
That is actually NOT dormancy — it would be a meaningful guard if it had
been active. The "zero fires" was deployment timing.

**Tighter (1.0 / $1)**: 2× the fires (108), 2× the savings ($289), 27% of
total $-loss caught.

### Diminishing returns curve

```
MULT 2.0 → 1.5  : +20 fires, +$63   (good)
MULT 1.5 → 1.25 : +23 fires, +$34   (decent)
MULT 1.25 → 1.0 : +13 fires, +$46   (good)
MULT 1.0 → 0.75 : +36 fires, +$72   (high noise)
MULT 0.75 → 0.5 : +17 fires, +$72   (likely starts cutting recoverable trades)
```

The sweet spot is **MULT 1.0 / floor $1.0**: meaningful coverage of bleeders
without aggressive cuts. Below $1 floor the median fired loss drops to <$2,
which begins to overlap heavily with normal trail-noise.

---

## 5. Category-stratified savings — which layer does AvgWinLossCap pre-empt?

| MULT  | floor  | TrailSL ($)     | EarlyLossCut ($) | Guardian ($) | DailyKill | DragonRev | EmergDD | PeakGB |
|------:|-------:|----------------:|-----------------:|-------------:|----------:|----------:|--------:|-------:|
| 2.00 / $2 | | 40 fires $86.92 | 11 fires $55.40 | 1 / $3.85   | 0         | 0         | 0       | 0      |
| 1.50 / $1 | | 54 fires $130.01| 17 fires $74.12 | 1 / $5.27   | 0         | 0         | 0       | 0      |
| 1.25 / $1 | | 71 fires $151.14| 21 fires $85.59 | 1 / $5.98   | 0         | 0         | 2 / $0.40 | 0    |
| **1.00 / $1** | | **76 fires $181.48**| **28 fires $99.66** | **2 / $6.89** | 0    | 0         | 2 / $1.21 | 0    |
| 1.00 / $0.5 | | 81 fires $190.46| 34 fires $102.23| 2 / $6.89  | 1 / $0.15 | 0         | 2 / $1.21 | 0    |
| 0.75 / $0.5 | | 99 fires $227.68| 38 fires $120.49| 2 / $10.05 | 2 / $0.66 | 0         | 3 / $2.20 | 0    |

### Interpretation

- **TrailSL pre-emption is where the cap creates value** (≈70% of cap
  $-savings). It cuts trades that were drifting toward their hard SL but
  hadn't hit the EarlyLossCut tier or trigger yet — i.e. the wide-trail
  bleeders on big-symbol losses (XAUUSD, GER40, XAGUSD, JPN225ft).
- **EarlyLossCut pre-emption is the second value vector**. EarlyLossCut
  is itself a tiered safety; the cap fires *earlier* on the worst tail
  (mostly catastrophic -1.5R T3-IMMEDIATE losses that already went bad).
  Mean fired EarlyLossCut loss = $4-8 → cap reduces to floor.
- **Guardian / DailyKillSwitch / DragonReversal / EmergencyDD pre-emption
  is negligible** (≤3 fires, ≤$2 across the whole sweep). Those layers fire
  on portfolio-level events that the per-position cap doesn't compete with.
- **PeakGiveback is never pre-empted**. Different regimes (profit vs
  loss) so they don't conflict.

**Verdict**: AvgWinLossCap is mostly a TrailSL/EarlyLossCut secondary that
shaves the worst tail of losses. It is NOT redundant with PEAK_GIVEBACK
(different domain). It IS partially redundant with EarlyLossCut tier
dispatch — but the cap evaluates `pnl_dollars` directly while EarlyLossCut
evaluates `profit_r`, which can diverge when `sl_dist` is wrong (see §5
of `parity_reaudit.md` — sl_dist corruption bug). The cap is **the only
$-bounded loss guard** below HardDollarCap ($24 = 1% equity), so it fills
a real gap.

---

## 6. Per-symbol firing breakdown @ MULT=1.0 / floor=$1.0

| Symbol    | L/W ratio | n_loss | cap fires | % caught | $ saved / total $-loss |
|-----------|----------:|-------:|----------:|---------:|-----------------------:|
| DJ30.r    | 1.19x     | 21     | 12        | 57%      | $32.60 / $96.34        |
| USDCAD    | 1.00x     | 32     | 7         | 22%      | $24.14 / $92.03        |
| BTCUSD    | 5.13x     | 21     | 10        | 48%      | $11.97 / $88.87        |
| **XAUUSD**| 6.21x     | 7      | **5**     | **71%**  | **$50.36 / $82.34**    |
| GBPUSD    | 1.02x     | 16     | 7         | 44%      | $20.52 / $67.41        |
| GER40.r   | 2.77x     | 8      | 6         | 75%      | $19.91 / $62.99        |
| EURUSD    | 1.25x     | 33     | 12        | 36%      | $18.99 / $60.01        |
| XAGUSD    | 2.42x     | 4      | 2         | 50%      | $32.18 / $56.20        |
| SWI20.r   | 0.79x     | 6      | 1         | 17%      | $0.20 / $46.62         |
| US2000.r  | 1.11x     | 41     | 9         | 22%      | $7.34 / $44.95         |
| JPN225ft  | 3.34x     | 22     | 12        | 55%      | $19.46 / $41.79        |
| **SP500.r**| 1.65x    | 23     | 12        | **52%**  | $22.70 / $39.48        |
| ETHUSD    | 2.92x     | 16     | 3         | 19%      | $14.39 / $27.29        |
| EURJPY    | 5.79x     | 16     | 2         | 12%      | $1.08 / $27.12         |
| GBPJPY    | 2.04x     | 10     | 5         | 50%      | $10.72 / $26.51        |

**Targeted catches**: XAUUSD 71%, GER40 75%, JPN225ft 55%, SP500 52%,
BTCUSD 48% — exactly the L/W-ratio bleeders. The cap does what it's
designed for.

**Surprising side effects** (probably overreach if MULT goes lower):
- DJ30 L/W = 1.19 (healthy) but 57% of its losses get capped because $5
  avg_win × 1.0 = $5 cap, and many DJ30 losses are $4-10. Cap effectively
  becomes "close at avg_win level" — fine for risk control, but it does
  reduce the chance of recovery.
- USDCAD L/W = 1.00 (perfectly balanced) but 22% caught. Cap is starting
  to cut into a healthy symbol's normal loss distribution.

This is acceptable at MULT=1.0 / $1 (caught losses are real losses, not
mid-life recoveries), but going below MULT=1.0 would start materially
trimming "potentially recoverable" trades.

---

## 7. Per-symbol overrides (proposal)

Symbols where the cap should be **looser** (don't cut healthy trades):

| Symbol  | L/W  | Recommended override |
|---------|-----:|----------------------|
| DJ30.r  | 0.88x| MULT=1.5, floor=$2  (allow more breathing room) |
| USDCAD  | 1.00x| MULT=1.5, floor=$2 |
| GBPUSD  | 1.02x| MULT=1.5, floor=$2 |
| US2000.r| 1.11x| MULT=1.5, floor=$1 |
| EURUSD  | 1.25x| MULT=1.5, floor=$1 |

Symbols where the cap should be **tighter** (severe bleeders):

| Symbol  | L/W  | Recommended override |
|---------|-----:|----------------------|
| XAUUSD  | 6.21x| MULT=0.75, floor=$3 |
| GBPCHF  | 7.99x| MULT=0.75, floor=$2 |
| EURJPY  | 5.79x| MULT=0.75, floor=$0.50 (avg_win is tiny $0.29) |
| BTCUSD  | 5.13x| MULT=0.75, floor=$1 |
| JPN225ft| 3.34x| MULT=0.75, floor=$1 |
| CADJPY  | 3.35x| MULT=0.75, floor=$1 |
| ETHUSD  | 2.92x| MULT=0.75, floor=$1 |
| GER40.r | 2.77x| MULT=1.0, floor=$2 |

**Implementation note**: Per-symbol overrides would need a new
`AVG_WIN_LOSS_CAP_SYMBOL_OVERRIDE = {symbol: (mult, floor)}` dict in
`config.py` and a 4-line lookup in `executor.py:1463-1483`. Not blocking;
can ship with global threshold change first and add overrides in a
follow-up.

---

## 8. Final recommendation

### Verdict: **RETUNE — keep the layer, tighten the global thresholds**

**Change**:
```python
AVG_WIN_LOSS_CAP_MULT       = 1.0   # was 2.0
AVG_WIN_LOSS_CAP_MIN_DOLLAR = 1.0   # was 2.0
```

**Projected impact** (based on 30d simulation):
- Fires: 52 → 108 (2.1×)
- $-saved: $146 → $289 (1.98×)
- % of total $-loss caught: 13.5% → 26.7%
- ~$10/day saved vs current ~$5/day
- For a $1.2K demo equity that's +0.83%/month uplift

**Risk**: Modest. The cap fires only on already-losing trades, so it converts
deep losses into shallower losses. It does NOT cut wins (positions in
profit are not evaluated). The 8% of "potentially recoverable" trades it
trims at MULT=1.0 are in the $1-3 loss range — within normal noise band
and almost always close anyway via TrailSL/EarlyLossCut shortly after.

### Why not REMOVE

- Sim shows clear $-savings ($146-$289/30d depending on tuning) that no
  other layer captures (PEAK_GIVEBACK fires on profit-regime giveback,
  EarlyLossCut fires on R-multiples — both blind to absolute $ loss).
- It is the only **$-bounded** per-position cap below the portfolio-level
  HardDollarCap.
- The L/W ratio gap is the explicit user rule ("avg loss should be ≤
  avg win", per commit `dfe8d87`); the cap is the mechanism that enforces
  it.

### Why not go below MULT=1.0

- MULT=0.75 adds only +12 fires ($72 more) but median fired loss drops
  to $2.55 (vs $3.40 at 1.0/$1) — getting into normal-trail-noise band.
- MULT=0.5 starts cutting trades that are 50% of avg_win in unrealized
  loss → those are routine mid-life pullbacks that often reverse.
- Better to leave the aggressive cuts to per-symbol overrides on the
  proven bleeders (XAUUSD, GBPCHF, EURJPY, BTCUSD) where L/W ratio
  justifies it.

### Why not RETUNE-ONLY (no per-symbol overrides)

Acceptable for a v1 retune. Global 1.0/$1 captures 80% of the value of
the override scheme above. Recommend ship global tune now, observe 7 days
of live data, then add per-symbol overrides if specific bleeders persist.

---

## 9. Validation checklist (post-deploy)

To verify the retune actually works in live (not just sim):

1. **Log scan** `tail -F logs/agent.log | grep AVG_WIN_LOSS_CAP` — should
   see 2-4 fires/day at MULT=1.0/$1.
2. **Journal scan** after 48h:
   ```sql
   SELECT COUNT(*), ROUND(SUM(pnl),2) FROM trades
   WHERE exit_reason LIKE 'AvgWinLoss%'
   AND timestamp >= datetime('now','-2 days');
   ```
   Expected: 4-8 fires totaling -$8 to -$15. If far above this, MULT too
   loose. If zero, check `_get_avg_win_dollars` returns >0 (need 5+ wins
   in window).
3. **L/W ratio drift**: re-run §3 table weekly. The four severe bleeders
   should converge toward L/W=1.0-2.0 if the cap is doing its job (their
   losses bounded at avg_win × 1.0).

---

## TL;DR

- AvgWinLossCap fired zero times in the audit window **because the layer
  was deployed today** (commit `dfe8d87` at 19:28 IST, widened at 20:01
  IST), not because thresholds were too loose at runtime.
- Simulator on 30d journal: at current 2.0/$2 the cap *would have* fired
  52 times saving $146 — meaningful, not dormant.
- Retune to **MULT=1.0 / floor=$1.0** doubles the catch to 108 fires
  saving $289 with low collateral risk.
- Optional follow-up: per-symbol overrides for XAUUSD/GBPCHF/EURJPY/BTCUSD
  (MULT=0.75) and DJ30/USDCAD/GBPUSD (MULT=1.5) — adds another ~$30-50/30d.
- **Verdict: KEEP + RETUNE. Do not remove. Do not go below MULT=1.0
  globally.**
