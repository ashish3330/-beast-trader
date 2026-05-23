# Audit 05 — $5K Account Scaling

**Status:** READ-ONLY audit. No source modified.
**Generated:** 2026-05-23
**Scope:** Verify every absolute-dollar threshold and position-sizing constraint before
swapping from $1,219 demo to a fresh $5,000 MT5 demo on Monday.

---

## 1. Every Absolute $-Threshold in the Codebase

Active 5 symbols per `config.py:65–69`: **SP500.r, US2000.r, DJ30.r, USOUSD, XAUUSD**.
`MAX_RISK_PER_TRADE_PCT = 2.0` (2026-05-23, bumped 1.0 → 2.0).
`HARD_DOLLAR_CAP_PCT = 0.040` (2026-05-23, bumped 0.02 → 0.04).
`STARTING_BALANCE = 2500.0` in `config.py:741`.

| Name | Current value | File:line | Used where | Scales? | $5K appropriate |
|---|---|---|---|---|---|
| `POST_BIG_WIN_DOLLAR_THRESHOLD` | **$15.0** | config.py:203 | brain.py:990,1002 — arms 30min same-direction cooldown if `last_pnl ≥ $15` | **NO — absolute $** | **$75** (1.5% of $5K, matches today's 1.23% on $1.2K) |
| `AVG_WIN_LOSS_CAP_MIN_DOLLAR` | $1.0 | config.py:193 | executor.py:1466 — floor for the avg-win loss cap | NO — but feature is **DISABLED** (`AVG_WIN_LOSS_CAP_ENABLED=False`, line 191) | OK to leave (noise floor; feature off) |
| `POST_BIG_WIN_R_THRESHOLD` | 10.0 | config.py:200 | brain.py:1001 | R-multiple, scales naturally | **No change** (R is %-relative) |
| `HARD_DOLLAR_CAP_PCT` | 0.040 (4%) | config.py:268 | executor.py:1428 — `max_loss = equity × PCT` | **YES — % of equity** | **No change** — at $5K = $200/pos, agg-cap = $500 |
| `MAX_RISK_PER_TRADE_PCT` | 2.0 | config.py:288 | executor.py:560 | **YES — % of equity** | No change (already 2%) |
| `MAX_TOTAL_EXPOSURE_PCT` | 25.0 | config.py:291 | executor.py:634 | YES — % | No change |
| `DAILY_LOSS_LIMIT_PCT` | 5.0 (warn) | config.py:326 | guardian.py:68 (1.5× = 7.5% close-all) | YES — % | No change |
| `DD_REDUCE_THRESHOLD` | 8.0 | config.py:328 | (informational) | YES — % | No change |
| `DD_PAUSE_THRESHOLD` / `DD_EMERGENCY_CLOSE` | 12.0 / 12.0 | config.py:329,330 | YES — % | No change |
| `DAILY_HARD_STOP_PCT` | **40.0** | config.py:336 | brain.py:815 | YES — % | **Lower to 15.0** — see Δ |
| `WEEKLY_HARD_STOP_PCT` | **50.0** | config.py:340 | brain.py:830 | YES — % | **Lower to 20.0** — see Δ |
| `MAX_RISK_OVER` (vol_min hard reject) | 3.0× | executor.py:591 | vol_min×SL forced risk multiplier | YES — multiplier of intent | No change |
| `VOL_MIN_ABSOLUTE_CAP_PCT` | 3.0 | executor.py:600 | absolute % cap on vol_min×SL forced risk | YES — % | No change |
| `SCALP_RISK_PCT` | 0.2 | config.py:641 | scalp executor:1070 | YES — % | No change (scalp is off anyway, master `MR_ENABLED = False`) |
| `STARTING_BALANCE` | 2500.0 | config.py:741 | default fallback for `equity` reads when MT5 returns 0 | **NO — absolute $** | **5000.0** — match the new account so fallback equals real balance |
| Guardian sharp-loss cut | 1.5% in <300s | guardian.py:111 | hard-coded literal | YES — % | No change |
| Guardian stale-loser cut | 0.75% over 5400s | guardian.py:127 | hard-coded literal | YES — % | No change |
| Guardian portfolio heat | >4.0% & ≥3 pos | guardian.py:139 | hard-coded literal | YES — % | No change |
| Equity-tier dormancy gates | "≥ $5000" / "≥ $8000" | config.py:54,95 | commented config — manual uncomment | N/A — **policy trigger** | **Tier 2 unlocks Monday** (dormant pool re-enable, see §5) |

**Summary of true absolute-$ thresholds (only 3 found):**
1. `POST_BIG_WIN_DOLLAR_THRESHOLD = 15.0` — **MUST scale**
2. `AVG_WIN_LOSS_CAP_MIN_DOLLAR = 1.0` — feature disabled; leave
3. `STARTING_BALANCE = 2500.0` — fallback constant; align with new account

Everything else uses % of equity or R-multiple and scales automatically.

---

## 2. Why the Daily/Weekly Hard-Stop Should Tighten at $5K

These are already %-based and "scale" mechanically, but the **calibration intent** was a small-account "demo can't really die" buffer:

- `DAILY_HARD_STOP_PCT = 40.0` — at $1.2K that's $480 (account barely survives one bad day).
  At $5K it's **$2,000 = full kill**. Industry standard for a real-money-bound demo is 5–8%/day.
- `WEEKLY_HARD_STOP_PCT = 50.0` — at $5K that's $2,500 (lose half the account before a halt).
  Recommend 15–20%.

**Recommendation: tighten to DAILY 15% / WEEKLY 20%.** Still loose enough to absorb a
multi-symbol gap day (e.g. 5 stops in one session at 2% each = 10% → no kill, just `DD_REDUCE`
at 8% halves risk). Catches accidental bug-loops fast.

---

## 3. Position-Sizing Math at $5K (per active symbol)

**Baseline:** `equity × MAX_RISK_PER_TRADE_PCT% = $5,000 × 2% = $100 intended risk per trade.**

**Inputs taken from `auto_tuned.SL_OVERRIDE_AUTO` (config.py-merged at line 1080) — the active SL multiplier (`× ATR`) per symbol:**

| Symbol | SL mult (× ATR) | Per-sym RISK_CAP | Effective risk % | Intended $ risk | Min-lot risk @ avg ATR (rough) | Within `MAX_RISK_OVER×3` cap? |
|---|---|---|---|---|---|---|
| **SP500.r** | 0.7 × ATR | none → 2.0% | 2.0% | **$100** | ~$15-25 (0.1 lot SP500 @ ATR 5pts, $1/pt) | ✅ well under |
| **US2000.r** | 0.2 × ATR | none → 2.0% | 2.0% | **$100** | ~$8-15 (0.1 lot @ ATR 4pts) | ✅ well under |
| **DJ30.r** | 0.2 × ATR | none → 2.0% | 2.0% | **$100** | ~$10-18 (0.1 lot @ ATR 40pts, $0.1/pt) | ✅ well under |
| **USOUSD** | 2.5 × ATR | none → 2.0% | 2.0% | **$100** | ~$25-40 (1 lot, ATR ~0.5, $0.01/tick) | ✅ under |
| **XAUUSD** | 0.2 × ATR | **1.2** | 1.2% | **$60** | $4-8 was the $1.2K cap problem; at $5K + intended $60 → safely above min-lot | ✅ no force-up |

**Critical change at $5K:** at $1.2K, several "live-earning" symbols (XAUUSD, XAGUSD,
BTCUSD, etc.) tripped the `VOL_MIN_WARN_ONLY_SYMBOLS` whitelist because min-lot×SL exceeded
2-4× intent. **At $5K with 2% risk = $100, ALL active 5 are well above min-lot×SL** —
the warn-only override should no longer fire for the active universe. Whitelist can stay
as-is for dormant symbols.

**SL widening at 0.2× ATR (US2000/DJ30/XAUUSD):** these are very tight SLs. Position size
balloons (e.g. ATR 4pts × $1/pt × 0.2 = $0.8 SL distance, lot = $100 / $0.8 = 125 / lot_value
units → could hit `volume_max` or substantial nominal exposure). Confirm at first trade.
If lots clamp to `volume_max` the `MAX_TOTAL_EXPOSURE_PCT = 25%` cap still applies.

**Per-symbol cap recommendation:** keep XAUUSD at 1.2% (high notional, conservative). No
other changes; the auto_tuned matrix already encodes per-symbol edge.

---

## 4. Drawdown Caps at $5K

| Cap | Type | Value | $5K trigger | Verdict |
|---|---|---|---|---|
| Guardian sharp-loss | % equity, time-bounded | 1.5% in <300s | $75 in 5min | OK |
| Guardian stale-loser | % equity, time-bounded | 0.75% over 1.5h | $37.50 | OK |
| Guardian day-loss close-all | DAILY_LOSS_LIMIT_PCT × 1.5 | 7.5% (5% × 1.5) | $375 | OK |
| Guardian portfolio heat | % unrealized loss | 4% w/ ≥3 pos | $200 | OK |
| `DD_REDUCE_THRESHOLD` (halve risk) | % from peak | 8% | $400 from peak | OK |
| `DD_EMERGENCY_CLOSE` | % from peak | 12% | $600 | OK |
| `HARD_DOLLAR_CAP_PCT` per-pos | % equity unrealized | 4% | $200/position | OK |
| `HARD_DOLLAR_CAP_PCT × 2.5` agg | % equity unrealized aggregate | 10% | $500 portfolio | OK |
| **DAILY_HARD_STOP** | % daily loss | **40%** | $2,000/day (too loose) | **TIGHTEN → 15% ($750)** |
| **WEEKLY_HARD_STOP** | % weekly loss | **50%** | $2,500/week (too loose) | **TIGHTEN → 20% ($1,000)** |

All %-based caps work as-is; just the two hard-stop kill switches need re-calibration for
the new account size (they were widened on 2026-05-13 to absorb a single $-101 force-close
event on the $1K account — that event would be only 2% on $5K, well under any cap).

---

## 5. Realistic Projection at $5K

**30-day live journal (data_backups/20260522_224826):** 245 trades across the active 5
(USOUSD newly added 2026-05-23 — no live history yet). Risk was 1% throughout the window;
current setting is 2% (since 2026-05-23). All $-values must scale **2× for the risk bump,
then 4.17× for the $1,200→$5,000 equity scaling**, total = **8.33×**.

Per-symbol 30d journal vs projected at $5K + 2% risk:

| Sym | n (30d) | live net @1%/$1.2K | avg R | avg $ Win | avg $ Loss | Projected net @ $5K | Projected avg W/L |
|---|---|---|---|---|---|---|---|
| US2000.r | 108 | +$14.85 | -0.02 | $0.98 | -$1.09 | +$124 | $8 / -$9 |
| DJ30.r | 58 | +$91.49 | +0.86 | $5.93 | -$4.53 | +$762 | $49 / -$38 |
| SP500.r | 48 | -$16.30 | -0.06 | $1.01 | -$1.69 | -$136 | $8 / -$14 |
| XAUUSD | 31 | -$36.85 | -0.71 | $1.90 | -$11.76 | -$307 | $16 / -$98 |
| USOUSD | – | (new) | – | – | – | – | – |
| **TOTAL** | **245** | **+$53.19** | – | – | – | **+$443** (~9% / mo) | – |

**Caveats:**
- DJ30 alone delivered +$91/30d (+0.86 avg R). XAUUSD is bleeding (-0.71 avg R despite 77% WR — losses are 6× wins on $). If XAUUSD bleed continues at $5K, it costs ~$300/mo. Recommend monitoring; if WR remains >70% but avg-R stays negative for first week, halve XAUUSD risk or disable.
- SP500.r has poor `avg_R = -0.06` over 48 trades — basically noise around BE.
- USOUSD is new (replaced UKOUSD 2026-05-23) — no live signal yet, BT-only projection.
- Activity rate at active-5 = ~245 trades / 30d = ~8 trades/day → at $5K with $100/trade max loss and the same WR/R profile = roughly **$2-4 / trade net average → ~$15-30 / day → $300-700 / month**.

**Weekly projection (5 days × ~40 trades = ~200 trades/week):**
- Base case (current 30d edge holds): +$100 / week
- Bull case (DJ30/US2000 carry, XAUUSD breaks even): +$300-500 / week
- Bear case (XAUUSD bleed accelerates, SP500 stays neutral): -$200 / week → DD_REDUCE
  at 8% = $400, kicks in fast and halves risk

**This justifies the demo-week-first policy** (`feedback_dragon_demo_week_20260514.md`):
do not move real money until 7 demo-day equity grows + max DD < 10%.

---

## 6. CHANGE LIST — Exact Config Edits Before Monday

All edits live in `/Users/ashish/Documents/beast-trader/config.py`.

### Required (3 edits)

```python
# Line 203 — POST-BIG-WIN trigger must scale to ~1.5% of new equity
- POST_BIG_WIN_DOLLAR_THRESHOLD = 15.0 # 2026-05-22 user: "+15 dollar on any symbol".
+ POST_BIG_WIN_DOLLAR_THRESHOLD = 75.0 # 2026-05-23 $5K scaling: 1.5% of $5K matches the 1.23% on $1.2K. Old $15 fires on routine wins at $5K.

# Line 336 — DAILY hard stop: 40% was a small-account "can't really die" buffer
- DAILY_HARD_STOP_PCT = 40.0
+ DAILY_HARD_STOP_PCT = 15.0    # 2026-05-23 $5K scaling: 40% → 15% = $750/day. Catches bug-loops.

# Line 340 — WEEKLY hard stop: 50% gives up half the account before halt
- WEEKLY_HARD_STOP_PCT = 50.0
+ WEEKLY_HARD_STOP_PCT = 20.0   # 2026-05-23 $5K scaling: 50% → 20% = $1,000/week.
```

### Recommended (1 edit — cleanliness)

```python
# Line 741 — align fallback with new account size
- STARTING_BALANCE = 2500.0
+ STARTING_BALANCE = 5000.0
```

### Equity-tier policy unlocks (manual review — NOT automatic)

`config.py:54-56` says "DORMANT (re-enable at equity ≥ $5000)" lists:
`XAUUSD / EURUSD / GER40.r / UKOUSD / BTCUSD / AUDJPY / XAGUSD`.

XAUUSD is **already active** (replaced UKOUSD on 2026-05-23). The remaining 6 dormant
symbols are eligible to uncomment Monday. **Recommend: do NOT uncomment all 6 immediately.**
Add **one at a time** after 1-2 days of clean demo data on the active 5. Order by 30d-journal edge:

| Sym | 30d journal net | 30d avg R | n | Verdict |
|---|---|---|---|---|
| XAGUSD | +$71.55 | +0.07 | 26 | Strong — first to add (after 1d clean) |
| EURUSD | -$28.10 | -0.01 | 55 | Marginal — wait |
| BTCUSD | -$59.13 | -0.19 | 62 | Bleeding — keep dormant |
| GER40.r | -$28.86 | -0.74 | 20 | Strongly bleeding — keep dormant |
| AUDJPY | -$0.73 | +0.22 | 10 | Tiny sample — wait |
| (UKOUSD) | – | – | – | Just disabled — keep off |

**$8K gate** (COPPER-Cr / GAS-Cr / NG-Cr) stays gated per `feedback_copper_gas_8k_gate.md`.

### NO-CHANGE list (confirmed)

- All other risk %, exposure %, guardian %, kill-switch %, trail R-thresholds, peak-giveback R, EarlyLossCut R, score thresholds, regime trail profiles — all scale-invariant or already $5K-appropriate.
- `MAX_RISK_PER_TRADE_PCT = 2.0` (just bumped; keep for demo week to validate).
- `HARD_DOLLAR_CAP_PCT = 0.040` (just bumped; keep).
- `SYMBOL_RISK_CAP` for XAUUSD = 1.2 (keep — gold has highest notional risk).
- `AVG_WIN_LOSS_CAP_MIN_DOLLAR = 1.0` (feature disabled; no harm leaving).

### Pre-flight verification (run before first trade Monday)

1. Confirm MT5 account ID / password / server is the new $5K demo (`config.py:15-17` or `.env`).
2. Confirm `STARTING_BALANCE` matches.
3. Tail brain log for first 5 trades — verify `risk=$XX.XX` printed in `executor.py:706,1144` is **~$100** (2% of $5K), not ~$24 (old $1.2K residue).
4. Check no `VOL_MIN OVERRIDE` warnings for SP500/US2000/DJ30/USOUSD/XAUUSD — at $5K none of these should need it.
5. Verify `data/agent_equity_state` row updates to ~$5,000 after first tick.

---

## Appendix — Files Touched / Read (for traceability)

- `/Users/ashish/Documents/beast-trader/config.py` (lines 15-340, 741, 1080-1115)
- `/Users/ashish/Documents/beast-trader/auto_tuned.py` (lines 130-230)
- `/Users/ashish/Documents/beast-trader/agent/brain.py` (lines 780-1015)
- `/Users/ashish/Documents/beast-trader/agent/equity_guardian.py` (full)
- `/Users/ashish/Documents/beast-trader/execution/executor.py` (lines 555-650, 1060-1100, 1400-1485)
- `/Users/ashish/Documents/beast-trader/data_backups/20260522_224826/trade_journal.db` (30d window)
- `/Users/ashish/Documents/beast-trader/data/trade_journal.db` (current 3h sample)
