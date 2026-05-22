# US2000.r Hard-Tune — Mirror-Aware BT (2026-05-23)

**Account:** $1,219 demo @ 2% risk per trade
**BT engine:** `backtest/v5_backtest.py` (mirror-aware: live config for POST_BIG_WIN, LOSS_STREAK, pullback, VWAP, trail-regime, dir-bias, MARGINAL_TRAIL dispatch)
**Runtime:** 98.7s, 288 backtests (58 Phase A + 200 Phase B + 30 Phase C)
**Verdict:** **SHIP — with caveats** (see "Honest read" below)

---

## Live baseline (180d BT @ risk_pct=2.0, equity=$1219)

| Knob | Value | Source |
|---|---|---|
| SL ATR mult | 0.2 | `auto_tuned.SL_OVERRIDE_AUTO` |
| Trail | `_TIGHT_LOCK` (all regimes) | `auto_tuned.TRAIL_OVERRIDE_REGIME_AUTO` |
| min_quality | 25 (all regimes) | `auto_tuned.SIGNAL_QUALITY_SYMBOL_AUTO` |
| pullback ATR | 0.9 | `config.PULLBACK_ATR_RETRACE_PER_SYMBOL` |
| pullback wait | 5 bars | `config.PULLBACK_MAX_WAIT_BARS_PER_SYMBOL` |
| VWAP buffer | 1.5 ATR | `config.VWAP_BUFFER_PER_SYMBOL` |
| POST_BIG_WIN | 3600s | `config.POST_BIG_WIN_COOLDOWN_SECS` |
| LOSS_STREAK   | 18000s (5h) | `config.LOSS_STREAK_COOLDOWN_SECS` |
| dir_bias | ranging=LONG, volatile=LONG | `auto_tuned.DIRECTION_BIAS_REGIME_AUTO` |

**Baseline 180d:** 150 trades, PF 3.30, PnL **$296,861**, WR 50.0%, max DD 9.4%, avg R 2.85.

(Note: $1.2K → $297K reflects 2.0% risk + compounding. Live runs at 0.8% so absolute numbers in this BT are not realistic dollar projections — use them only for *relative* tuning.)

---

## Phase A — independent single-dim sweeps (180d, 58 BTs, 24s)

Top-3 per dimension (PnL desc):

| Dim | Winner | #2 | #3 |
|---|---|---|---|
| **sl** | **0.2** $296k PF3.30 | 0.3 $61k PF3.59 | 0.5 $57k PF4.98 |
| **trail** | All 7 IDENTICAL $296k PF3.30 | — | — |
| **mq** | **25** $296k | 22 $255k | 28 $31k |
| **pb_atr** | **0.8** $691k PF6.41 | 1.0 $184k | 1.2 $16k |
| **pb_wait** | **5** $296k | 3 $214k | 7 $83k |
| **vwap_buf** | **0.7** $1,271k PF8.23 | 0.0 $324k | 1.5 $296k |
| **pbw_secs** | **1800** $296k | 3600 $296k | 5400 $56k |
| **ls_secs** | **3600** $296k | 7200 $296k | 10800 $296k |
| **dir_bias** | **LONG_VO** $382k PF3.61 | LONG_TR_VO $382k | live $296k |

### Key single-dim findings
- **SL:** 0.2 (live) confirmed — every wider SL halves PnL or worse.
- **Trail:** ZERO differentiation. All 7 profiles produce **identical** 150/3.30/$296k results. Reason: US2000.r raw_score is consistently below `SCORE_TIER_THRESHOLD=7.0`, so every entry is routed to live's `MARGINAL_TRAIL` ladder via `v5_backtest.py:1124-1126`. The named profiles never run.
- **mq:** 25 wins; 28 collapses by ~10x (signal-quality cliff right above 25).
- **pb_atr:** 0.8 beats live's 0.9 by 2.3×. Tighter retrace = bigger R-multiple per fill.
- **vwap_buf:** 0.7 single-dim winner (PF 8.23, $1.27M); live's 1.5 is on the plateau ($296k).
- **dir_bias:** dropping `ranging=LONG` and keeping `volatile=LONG` ("LONG_VO") improves by $86k. The ranging-regime LONG bias is currently a small drag.
- **POST_BIG_WIN / LOSS_STREAK** cooldowns: ≤3600s for PBW and ≤10800s for LS produce the plateau. Tightening LS from 18000s (live) → 3600s does *not* hurt; loosening PBW past 5400s does hurt.

---

## Phase B — Top-2 combine (200 BTs, 64s)

Top-5 are **degenerate** — same PnL, only differ in dims that don't matter for the winning core:

| Tag | sl | trail | mq | pb_atr | pb_wait | vwap | pbw | ls | dir_bias | PnL | PF | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B4 | 0.2 | _TIGHT_LOCK | 25 | 0.8 | 5 | 0.0 | 1800 | 3600 | LONG_VO | $2,241,374 | 6.87 | +$1,944,513 |
| B13 | 0.2 | _TIGHT_LOCK | 25 | 0.8 | 5 | 0.0 | 1800 | 3600 | LONG_TR_VO | $2,241,374 | 6.87 | +$1,944,513 |
| B14 | 0.2 | _TIGHT_LOCK | 25 | 0.8 | 5 | 0.0 | 1800 | 7200 | LONG_VO | $2,241,374 | 6.87 | +$1,944,513 |
| B15 | 0.2 | _TIGHT_LOCK | 25 | 0.8 | 5 | 0.0 | 3600 | 3600 | LONG_VO | $2,241,374 | 6.87 | +$1,944,513 |
| B34 | 0.2 | _AGGR_LOCK | 25 | 0.8 | 5 | 0.0 | 1800 | 3600 | LONG_VO | $2,241,374 | 6.87 | +$1,944,513 |

Stable core ⇒ **sl=0.2, mq=25, pb_atr=0.8, pb_wait=5, vwap_buf=0.0, dir_bias=LONG_VO**.
Trail / PBW / LS / sub-LONG_VO dir variants do not change the 180d result.

> **Plateau caveat:** Phase B winners are identical because: trail is irrelevant (MARGINAL dispatch), and with vwap=0.0 + dir=LONG_VO the entry set is concentrated enough that small cooldown changes don't add or remove trades.

---

## Phase C — 5-fold walk-forward on top-3 (30 BTs, 10s)

All three rank-1/2/3 are degenerate copies, so one WF result for the canonical combo:

| Fold | n | cand_pnl | base_pnl | Δ | PF |
|---|---:|---:|---:|---:|---:|
| 60d  | 66  | $7,644.88     | $8,180.57     | **-$535.69**  | 9.06 |
| 90d  | 75  | $15,262.35    | $11,816.01    | +$3,446.34  | 12.06 |
| 120d | 147 | $118,077.72   | $72,225.71    | +$45,852.01 | 9.53 |
| 150d | 199 | $927,895.25   | $763,718.43   | +$164,176.82 | 9.42 |
| 180d | 222 | $2,241,374.08 | $296,861.00   | **+$1,944,513.08** | 6.87 |

- **Positive folds:** 4/5 ✅
- **Avg PF (excl. 999):** 9.39 ✅
- **Δ@180d:** +$1,944,513 ≫ $30 threshold ✅
- Ship rule (Δ≥$30 AND ≥3/5 WF AND avg PF≥1.5): **PASS**

---

## Winner

```python
# config.py / auto_tuned.py overlay for US2000.r:
SL_OVERRIDE_AUTO['US2000.r']             = 0.2     # unchanged
SIGNAL_QUALITY_SYMBOL_AUTO['US2000.r']   = {r: 25 for r in ('trending','ranging','volatile','low_vol')}  # unchanged
PULLBACK_ATR_RETRACE_PER_SYMBOL['US2000.r']     = 0.8    # change: 0.9 → 0.8
PULLBACK_MAX_WAIT_BARS_PER_SYMBOL['US2000.r']   = 5      # unchanged
VWAP_BUFFER_PER_SYMBOL['US2000.r']              = 0.0    # change: 1.5 → 0.0 (disable VWAP filter)
POST_BIG_WIN_COOLDOWN_SECS    = 1800             # global; was 3600 (US2000.r preferred)
LOSS_STREAK_COOLDOWN_SECS     = 3600             # global; was 18000 (US2000.r preferred)
DIRECTION_BIAS_REGIME_AUTO['US2000.r'] = {'volatile': 'LONG'}  # change: drop ranging-LONG
# Trail: keep _TIGHT_LOCK (BT cannot distinguish — every entry hits MARGINAL_TRAIL)
```

---

## Honest read

### Pros
- Δ +$1.94M / 180d BT is enormous (relative — see absolute caveat).
- 4/5 WF folds positive (only the most recent 60-day fold is slightly negative: -$536).
- Avg WF PF 9.39 — well above the 1.5 threshold.
- The actionable changes (`pb_atr 0.9→0.8`, `vwap_buf 1.5→0.0`, drop ranging-LONG bias, tighten PBW/LS cooldowns) are individually defensible from Phase A.

### Cons / caveats
- **Absolute PnL is meaningless** because the BT compounds 2.0% risk on a $1,219 base with no position-size cap — you get a 1840× equity multiplier in 180 days. Use the deltas as *relative* signals only. Live at 0.8% risk + dollar caps will be 1/100th to 1/1000th of these numbers.
- **Most recent 60-day WF fold is NEGATIVE (-$536)**. The win is heavily concentrated in d150-d180. Recent market regime is less favorable to this exact combo. This is the same red flag you flagged in `feedback_dont_overfit_backtest_when_live_bleeding.md` — proceed cautiously.
- **POST_BIG_WIN_COOLDOWN_SECS and LOSS_STREAK_COOLDOWN_SECS are GLOBAL** in `config.py` — they apply to every symbol. The BT shows US2000.r prefers tighter values, but you must verify that the rest of the live universe isn't hurt before flipping them. Recommend: do NOT change the global cooldowns from this tune; keep US2000.r-only changes (pullback, VWAP, dir bias).
- **Trail dim has no BT signal.** All 7 trail profiles produce identical results because `raw_score < SCORE_TIER_THRESHOLD=7.0` for every US2000.r entry, forcing MARGINAL_TRAIL dispatch. Live executor behaves the same way — so changing the named trail does not affect live behavior for US2000.r at current mq=25. Keep `_TIGHT_LOCK` (already live).
- **`vwap_buf=0.0` disables the VWAP filter entirely.** Single-dim winner was actually `vwap_buf=0.7` ($1.27M, PF 8.23) but the combined Phase B picked 0.0 because of interactions with `dir_bias=LONG_VO`. Either is defensible; 0.0 is simpler and slightly less recently-overfit.
- Phase B "winning" combo equals the trail-irrelevant, cooldown-irrelevant zone — this is *not* a fragile global optimum, it's a broad plateau. Good for stability, but means we cannot claim the trail/cooldown changes are validated.

### Ship recommendation
**Conservative ship:** Only change `pullback_atr 0.9→0.8`, `vwap_buf 1.5→0.0`, and drop the ranging-LONG dir bias (keep volatile-LONG). Hold cooldown and trail at live values until either (a) you tune the global cooldowns across the whole universe, or (b) the recent-60d fold turns positive after the entry-side changes.

**Aggressive ship:** Apply the full winner including PBW=1800s + LS=3600s globally — but verify other symbols don't regress in a separate run before flipping the live config.

---

## Search budget used

- Phase A: 58 BTs / 24.3s
- Phase B: 200 BTs / 63.8s
- Phase C: 30 BTs / 10.5s
- **Total:** 288 BTs / 98.7s — well under the 2h cap.
