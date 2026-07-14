# TREND-book Data & Harness Integrity Audit
_Agent: DATA/HARNESS INTEGRITY · 2026-07-15 · read-only on live/data (no refetch, no config edits)_

Purpose: make sure the 9 per-symbol tuners (and the live bot) are not standing on
truncated / stale caches, and that the D1-signal + H1-exit reconstruction the
tuners use actually matches live. History: repeated SHIP_NONE tunes were caused
by **truncated caches**, not exhausted params (`feedback_dragon_cache_truncated_top5`).

Cache dir: `/Users/ashish/Documents/xauusd-trading-bot/cache/` · files `raw_<tf>_<SYM>.pkl`, token = `sym.replace('.','_')` (indices `.r`→`_r`), symbols UPPERCASE.

---

## 1. Cache audit — TREND_BASKET × timeframe

TREND_BASKET = `XAUUSD, BTCUSD, ETHUSD, JPN225ft, NAS100.r`
Books: **D1** = TREND signal + ATR + Chandelier windows (LIVE reads this). **H1** = TREND exit tuners + GOLD_SMC (XAU). **M1** = SCALPER (not used by TREND tuning).

| symbol | tf | file | n_bars | first | last | span (d) | stale (d) | status |
|---|---|---|---:|---|---|---:|---:|---|
| XAUUSD | D1 | raw_d1_XAUUSD.pkl | 3000 | 2014-11-26 | 2026-07-14 | 4248 | 1 | **OK** |
| XAUUSD | H1 | raw_h1_XAUUSD.pkl | 1500 | 2026-04-14 | 2026-07-14 | 91 | 0 | OK-fresh / **SHALLOW** (91d) |
| XAUUSD | M1 | raw_m1_XAUUSD.pkl | 500 | 2026-07-13 | 2026-07-14 | 0 | 0 | TRUNCATED (500 bars, scalper-only) |
| BTCUSD | D1 | raw_d1_BTCUSD.pkl | 2707 | 2018-01-02 | 2026-07-14 | 3115 | 1 | **OK** |
| BTCUSD | H1 | raw_h1_BTCUSD.pkl | **242** | 2026-07-04 | 2026-07-14 | 9 | 0 | **TRUNCATED** (9 days — unusable) |
| BTCUSD | M1 | raw_m1_BTCUSD.pkl | 500 | 2026-07-13 | 2026-07-14 | 0 | 0 | TRUNCATED (500 bars, scalper-only) |
| ETHUSD | D1 | raw_d1_ETHUSD.pkl | 3000 | 2017-02-14 | 2026-07-14 | 3437 | 1 | **OK** |
| ETHUSD | H1 | raw_h1_ETHUSD.pkl | 50000 | 2019-11-27 | **2026-05-12** | 2358 | **63** | **STALE** (63d — deep but 2 months behind) |
| ETHUSD | M1 | raw_m1_ETHUSD.pkl | — | — | — | — | — | **MISSING** (scalper-only) |
| JPN225ft | D1 | raw_d1_JPN225ft.pkl | 929 | 2022-12-08 | 2026-07-14 | 1314 | 1 | **OK** (shorter symbol history) |
| JPN225ft | H1 | raw_h1_JPN225ft.pkl | 21123 | 2022-12-08 | 2026-07-09 | 1309 | 5 | **OK** (deep, ~mildly stale) |
| JPN225ft | M1 | raw_m1_JPN225ft.pkl | 500 | 2026-06-15 | 2026-06-16 | 0 | 28 | STALE+TRUNCATED (scalper-only) |
| NAS100.r | D1 | raw_d1_NAS100_r.pkl | 3000 | 2015-04-26 | 2026-07-14 | 4097 | 1 | **OK** |
| NAS100.r | H1 | raw_h1_NAS100_r.pkl | 50000 | 2014-12-18 | 2026-07-08 | 4220 | 6 | **OK** (deep, ~mildly stale) |
| NAS100.r | M1 | raw_m1_NAS100_r.pkl | 500 | 2026-06-15 | 2026-06-16 | 0 | 28 | STALE+TRUNCATED (scalper-only) |

Notes:
- BTC H1 confirmed: **242 bars / 9 days** (matches the known-issue report). The `.bak.pretune` next to it holds only 500 bars ending 2026-05-28 (also stale) — no good local fallback.
- ETH H1 is the deep 50k-bar file but frozen at **2026-05-12**; it misses ~2 months incl. the recent regime. Its `.bak.pretune` cousins for JPN/NAS/SWI are all 500-bar May-29 stubs.
- **Root cause of H1 rot:** `scripts/fetch_h1.py` (the scheduled 15-min H1 job) fetches H1 for **`GOLD_SMC_SYMBOL` only** (= XAUUSD, COUNT=1500). Nothing on a schedule refreshes BTC/ETH/JPN/NAS H1 — they only get topped up by one-off deep-fetch runs (`fetch_h1_deep.py`, last run left JPN/NAS at Jul 8-9). BTC's 242-bar state is a stray partial write on Jul 14 09:30.
- D1 (the LIVE trend signal source) is **healthy for all 5 symbols** — fetched Jul 14/15, deep. The live bot's signal is not at risk; only the H1 *exit tuners* are.

---

## 2. TRUST / DON'T-TRUST verdict per (symbol, tf)

For the current per-symbol **exit** tuning (D1 signal + H1 exits):

| symbol | D1 signal | H1 exit tuning | verdict |
|---|---|---|---|
| **XAUUSD** | TRUST | **DON'T-TRUST for WF** — only 91d / 1500 H1 bars. Fine for a spot-check, far too shallow for rolling walk-forward (a single 256-EMA context barely fits). | Refetch H1 deep first. |
| **BTCUSD** | TRUST | **DO-NOT-TRUST — 242 bars.** Any BTC H1 tune result is noise. This is the SHIP_NONE trap. | **Block BTC tuning until refetched.** |
| **ETHUSD** | TRUST | **DON'T-TRUST as-is** — deep but 63d stale; OOS half ends May-12 so it cannot see current regime. IS results OK, OOS misleading. | Refresh, then trust. |
| **JPN225ft** | TRUST | **TRUST** (21k bars, 5d stale). Smoke-tested clean. | OK to tune now. |
| **NAS100.r** | TRUST | **TRUST** (50k bars, 6d stale). | OK to tune now. |

Bottom line: **only JPN225ft and NAS100.r are safe to walk-forward tune right now.** XAU/BTC/ETH H1 must be refetched first or their tunes will repeat the truncated-cache SHIP_NONE failure. M1/SCALPER caches are all 500-bar/stale but out of scope for the TREND campaign.

---

## 3. BT ↔ LIVE parity (the tuners' reconstruction vs `agent/brain.py::_process_trend`)

The tuners (`scripts/_trend_exit_tune_h1*.py`) and the shared `trend_engine.py`
reconstruct: **D1 3-EMA ensemble signal (prior completed bar, +1-day effective)
→ enter at bar open, SL = `ATR_STOP`(3.0)×ATR, Chandelier trail + profit-lock +
peak-giveback checked per H1 bar, optional ATR-distance TP, flip on daily signal
reversal.** The signal path matches live (`agent/trend_follower.evaluate`, same
EMA pairs via `trend_ema_pairs`, same ATR). The **exit economics do NOT**. Ranked
risks a naive backtest misses:

1. **Risk-capped SL — #1, dominant.** Backtest stop = fixed **3.0×ATR**. LIVE tightens the stop so even a 2×min-lot fill risks ≤ `TREND_MAX_RISK_PCT` (1.0%) → for gold ~**0.2×ATR**, i.e. **~15× tighter**. Consequences a 3×ATR backtest cannot see: (a) live gets stopped out far more often; (b) live scales the lock/giveback **activation to the capped `sl_dist`** (`_act_thresh = min((ACT/3)·sl_dist, 0.5·sl_dist)`), so peak-giveback ARMS almost immediately and live **exits on the first ~30% pullback of a peak that armed at 0.5R of a tiny stop** — whereas the backtest arms giveback only after `peak ≥ ACT×ATR` (a large move) and lets winners ride the wide trail. → Backtest `total_R` / PF is **not a live P/L forecast**; it is only valid for *ranking* exit-param variants on identical data. Documented in `trend_engine.py` header.

2. **No TP in the tuner.** Tuners use `TP=999` (no target). LIVE places a real **`TREND_TP_ATR`=6×ATR** TP. Backtest rides positions the live TP would have banked → inflates both winners and giveback counts.

3. **Re-entry cooldown mismatch.** Backtest `blocked` flag suppresses same-dir re-entry only until the daily signal ≠ blocked (no clock). LIVE adds a **2h time cooldown** (`TREND_REENTRY_BLOCK_HOURS`) after every reversal exit. Different re-entry timing → different trade set after givebacks.

4. **Signal freshness / forming-bar skew.** LIVE `evaluate()` uses `d1.iloc[-1]` — which can be **today's still-forming D1 bar** (fetch_d1 pulls pos-0). The tuner uses the **prior completed** D1 bar (eff = normalize+1day, no look-ahead). So live acts up to a day earlier and on intrabar-updated EMAs. Minor for 256-EMA crossovers, but flip timing can differ by a bar.

5. **Conviction / selectivity gate — currently INERT.** `TREND_CONVICTION_PER_SYMBOL = {}` (all prior gates were removed as non-robust), so live applies no ADX/slope entry gate today → no divergence *now*. Flagged because if a tuner re-introduces one, the backtest must model it or it will over-count entries.

6. **Bridge/ops realities absent from BT (don't affect edge, do affect fills):** fail-closed skips on stale D1 cache (>12h) or stale positions sync; one order-write per 60s cycle (bridge cap); 2-leg split both forced to min-lot. These change *which* fills happen live, not the per-trade R the backtest computes.

**Guidance for tuners:** treat H1-exit backtest numbers as a **relative ranking only**. A variant that wins here can still lose live because the risk-capped SL + early-armed giveback dominate live outcomes. Do not tighten TRAIL/LOCK chasing a backtest spike (the churn-cliff / curve-fit trap already documented in `config.py`).

---

## 4. Remediation — commands the user should run (needs live MT5 bridge up)

All are existing isolated-connection fetchers (own MT5 login, won't fight the live trader). Run from repo root `/Users/ashish/Documents/beast-trader`:

```bash
# 1. Deep H1 for XAU / BTC / JPN / NAS (COUNT=50000). Fixes BTC's 242-bar
#    truncation and XAU's 1500-bar shallowness; refreshes JPN/NAS to current.
python3 -B scripts/fetch_h1_deep.py

# 2. ETHUSD is NOT in fetch_h1_deep's SYMS list — refresh it (63d stale) via
#    the targeted refetcher (50000 H1 bars, backs up the pkl first):
python3 -B scripts/refetch_short_h1.py ETHUSD

# 3. (verify) re-run this engine's smoke test + a quick bar-count check:
python3 -B backtest/tune/trend_engine.py
python3 -B -c "import pickle,pandas as pd;from pathlib import Path;C=Path('/Users/ashish/Documents/xauusd-trading-bot/cache');[print(s, len(pickle.load(open(C/f'raw_h1_{s}.pkl','rb')))) for s in ['XAUUSD','BTCUSD','ETHUSD']]"
```

Optional durability fix (out of scope for this read-only pass, recommend to user):
`scripts/fetch_h1.py` should fetch H1 for the **whole TREND_BASKET**, not just
`GOLD_SMC_SYMBOL`, so BTC/ETH/JPN/NAS H1 never rot again between deep-fetches.

**Do not tune XAUUSD / BTCUSD / ETHUSD H1 exits until step 1-2 complete.**
JPN225ft and NAS100.r can be tuned now on `backtest/tune/trend_engine.py`.

---

## 5. Deliverables written
- `backtest/tune/DATA_AUDIT.md` — this file.
- `backtest/tune/trend_engine.py` — canonical importable TREND reconstruction
  (D1 3-EMA signal + H1 chandelier/lock/giveback/flip exits), faithful to
  `scripts/_trend_exit_tune_h1.py`, returns trades with entry/exit/pnl_R/MAE_R.
  Smoke test (`__main__`, JPN225ft): 20617 H1 bars, 42 trades, total_R 2.84,
  PF 2.69, WR 90%, using config per-symbol exit params (3.0/0.6/0.35/0.3).
