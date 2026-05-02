# Dragon Trader — Session Report 2026-05-02 / 03

**Account:** 25106421 (VantageInternational-Demo) · Balance ~$2,743
**Branch:** `main` · Latest commits: `1d4b6dc → e0022de → 7c8b6e3` (last in this session)
**Live PID:** check `pgrep -f beast-trader/run.py`

---

## 1. What Changed This Session

| Area | Before | After |
|---|---|---|
| Live universe | 7 symbols | **31 symbols** (PF > 2 in 180d backtest) |
| Tuning pipeline | hand-tuned per-symbol | **11-script automated sweep** + synthesizer |
| `auto_tuned.py` source | hand-edited | regenerated from `tune_180d_pass2.json` |
| Bleeder handling | leave or disable | targeted 90d retune → all 4 (USDCAD/ETHUSD/GBPUSD/EURCHF) addressed |
| Backtest baseline | 7-symbol, $2,351 / 180d | 31-symbol, **$13,389 / 90d** ($1k start each) |

### Universe expansion 7 → 31

PF > 2 filter from `validate_full_synth.txt`. **XPTUSD.r excluded** ($49K/180d was a stacked-override data artifact — 90d window confirmed). **EURUSD/GBPUSD/USDJPY/ETHUSD re-enabled** despite memory's "disabled live" rule (user override).

### Tuning pipeline (commit `1d4b6dc`)

11 new scripts under `scripts/`:

| Script | Role | Output |
|---|---|---|
| `tune_universe.py` | Pass-1 coarse grid (96 combos × 60 sym × 180d) | `tune_180d_pass1.json` |
| `tune_pass2_fine.py` | Pass-2 refined (225 combos centered on pass-1 best) | `tune_180d_pass2.json` |
| `tune_direction_bias.py` | Per-sym LONG/SHORT/BOTH sweep | `direction_bias_auto_dict.py` |
| `sweep_trails.py` | Profit ratchet sweep | `trail_overrides_auto_dict.py` |
| `profile_toxic_hours.py` | Hourly PF/PnL profile | `toxic_hours_auto_dict.py` |
| `stress_spread.py` | Spread-shock robustness | `spread_fragile_symbols.json` |
| `rescue_losers.py` | 2nd-chance config for marginal symbols | `rescue_losers_auto_dict.py` |
| `walk_forward.py` | Train/test split overfit detection | `walk_forward.json` |
| `live_vs_backtest.py` | Per-sym divergence audit | `live_vs_backtest.{json,md}` |
| `validate_tuned.py` | Sanity gate before deploy | console |
| `retune_bleeders_90d.py` | **NEW**: targeted 90d retune for symbols that bled 180d-tuned | `retune_bleeders_90d.json` |
| `synthesize_auto_tuned.py` | Merges all auto_dict outputs → one `auto_tuned.py` | `auto_tuned.py` |

`config.py` imports `auto_tuned` at the bottom. To uninstall the tuning entirely: `rm auto_tuned.py`.

---

## 2. Final 90d Backtest — $1,000 start per symbol

Config: latest `auto_tuned.py` + `--rl-trail` + ML gate (where models exist).
Command: `python3 -B backtest/v5_backtest.py --days 90 --all-symbols --rl-trail`

```
SYMBOL          PF        PnL     WR     DD  Trades
----------------------------------------------------
DJ30.r        4.75 $    2384  47.7%   5.6%     174
UKOUSD       10.80 $    2350  88.3%   2.6%     128
US2000.r      2.72 $     908  49.7%   6.8%     155
SP500.r       4.48 $     903  70.3%   4.2%     138
NAS100.r      2.72 $     780  52.4%   6.0%     124
UK100.r       2.27 $     770  51.9%   8.0%     156
COPPER-Cr     2.54 $     665  75.8%   3.8%     211
EURAUD        6.35 $     593  72.8%   1.9%     103
GER40.r       2.71 $     549  60.7%   5.2%     117
AUDUSD        5.18 $     410  61.6%   2.6%      73
EURUSD        2.61 $     394  49.2%   6.2%     130
USDCAD        4.30 $     302  80.0%   4.6%      50
GBPAUD        4.28 $     286  61.8%   4.3%      55
FRA40.r       1.76 $     268  46.2%   5.5%     130
CHFJPY        2.43 $     260  61.4%   4.9%     101
SWI20.r       3.76 $     260  64.4%   2.6%      45
BCHUSD        6.09 $     236  68.3%   2.7%      41
XAGUSD        3.36 $     205  70.6%   2.5%      85
HK50.r        2.65 $     191  54.2%   3.5%      48
CADJPY        1.72 $     188  55.7%   4.2%     140
USDJPY        2.20 $     173  58.5%   4.7%      94
AUDJPY        5.01 $     162  56.1%   1.7%      41
USDCHF        2.21 $     136  54.0%   3.8%      63
XAUUSD        1.27 $      39  77.3%   3.4%      97
EURGBP        1.67 $      33  47.8%   2.4%      23
ETHUSD        1.13 $       9  79.5%   2.3%      83
GBPCHF        1.11 $       9  45.0%   3.8%      40
GBPJPY        1.03 $       5  54.2%   6.5%      48
GBPUSD        0.94 $      -6  40.0%   5.5%      30 ⚠️
EURCHF        0.17 $     -76  15.4%   7.6%      26 ⚠️
NG-Cr           — (cache too short for 90d)
----------------------------------------------------
TOTAL              $   13389              2749   28W / 2L
```

**$13,389 / 90d ≈ $4,463/month** at $1k-per-symbol simulation. With real account ($2,743) and `MAX_POSITIONS=4`, expect ~$300–500/month live. **2 marginal bleeders** (GBPUSD effectively flat at -$6 / 30 trades; EURCHF -$76 due to RL-trail interaction — flag for soak monitoring).

---

## 3. Architecture

### 3.1 Process layout

```
┌─────────────────────────────────────────────────────────────┐
│  Wine MT5 Terminal (broker session)                         │
│      ↑ rpyc bridge on localhost:18813                       │
└─────────────────────────────────────────────────────────────┘
              ↑                              ↑
              │ ticks + orders               │ market data
              │                              │
┌──────────────────────────┐    ┌──────────────────────────┐
│  run.py (main process)   │    │  bridge processes        │
│  ├─ TickStreamer         │    │  ├─ com.dragon.bridge-tick│
│  ├─ AgentBrain (1s loop) │    │  └─ com.dragon.bridge-dash│
│  ├─ Executor             │    │                           │
│  ├─ MasterBrain (gates)  │    │  Managed by launchd       │
│  ├─ RLLearner            │    └──────────────────────────┘
│  ├─ ML meta-models       │
│  └─ SQLite journal       │
└──────────────────────────┘
              │
              ↓ port 7777
   ┌─────────────────────┐
   │  Dashboard (Vue)    │
   └─────────────────────┘
```

### 3.2 Module map

```
beast-trader/
├── run.py                       Main entry point
├── config.py                    Single source of truth: SYMBOLS, risk, gates
│   └─ imports auto_tuned (try/except)
├── auto_tuned.py                GENERATED — 6 layered override dicts
│
├── agent/
│   ├── brain.py                 1s decision cycle, regime + scoring + gates
│   ├── master_brain.py          6 portfolio-level gates (kill-switch, DD, exposure)
│   ├── scalp_brain.py           Scalp signal layer
│   ├── smart_entry.py           Entry timing + level memory
│   ├── exit_intelligence.py     TP1/2/3 ladder + trail decisions
│   ├── mtf_intelligence.py      M1/M5/M15/H1 confluence
│   ├── rl_learner.py            Trail/lock multipliers, exit-rule learning
│   ├── learning_engine.py       Trade outcome → score weight updates
│   ├── pattern_learner.py       Engulfing/pin/HH-HL pattern PnL
│   ├── level_memory.py          S/R level retention across cycles
│   ├── fvg_detector.py          Fair-value-gap detection
│   ├── order_flow.py            Bid/ask volume imbalance
│   ├── trade_intelligence.py    Per-trade post-mortem
│   ├── portfolio_risk.py        Total exposure / corr clusters
│   ├── equity_guardian.py       Hard kill-switches
│   └── calendar_filter.py       Toxic hours / news windows
│
├── signals/
│   ├── momentum_scorer.py       11-component score → 0-10 long/short
│   ├── mean_reversion_scorer.py BB+RSI mean revert
│   ├── scalp_scorer.py          M1/M5 fast scalp
│   └── industry_gates.py        Categorical filters (Forex/Index/Crypto)
│
├── execution/
│   └── executor.py              MT5 send_order, SL/TP modify (action 6),
│                                position sync, magic-number routing
│
├── backtest/
│   ├── v5_backtest.py           Single-process backtest, RL-trail aware
│   ├── cost_model.py            Spread tables (no slippage — backtest correction)
│   ├── scan_all.py              Multi-symbol scanner
│   └── results/                 All tuner outputs + validation runs
│
├── scripts/
│   ├── tune_universe.py         pass1 coarse
│   ├── tune_pass2_fine.py       pass2 fine
│   ├── retune_bleeders_90d.py   targeted 90d retune (NEW)
│   ├── synthesize_auto_tuned.py merger
│   ├── apply_tuned_params.py    legacy direct apply
│   ├── walk_forward.py          OOS validation
│   ├── sweep_trails.py          profit-ratchet sweep
│   ├── tune_direction_bias.py   LONG/SHORT/BOTH sweep
│   ├── profile_toxic_hours.py   hourly profile
│   ├── stress_spread.py         spread shock test
│   ├── rescue_losers.py         marginal-symbol 2nd chance
│   ├── live_vs_backtest.py      divergence audit
│   ├── validate_tuned.py        sanity gate
│   ├── bootstrap_rl_from_backtest.py  seed RL DB from backtest
│   ├── refresh_cache.py         pull fresh H1/M15 from MT5
│   ├── refresh_extended.py      multi-symbol cache refresh
│   ├── recover_no_rates.py      cache repair
│   ├── seed_journal.py          replay backtest into journal
│   ├── drift_detector.py        daily ML AUC erosion alarm
│   ├── watchdog.py              liveness probe (com.dragon.watchdog)
│   └── weekly_retrain.sh        cron-style ML retrain
│
├── dashboard/
│   ├── app.py                   Flask backend on :7777
│   └── vue_app.py               JARVIS UI
│
├── launchd/
│   ├── com.dragon.drift-check.plist
│   └── com.dragon.watchdog.plist
│
├── models/saved/                Per-symbol LightGBM ensemble meta-labels
├── logs/dragon.log              Live agent log
├── trade_journal.db             Trade outcomes (SQLite)
├── rl_learner.db                RL state persistence
└── data/                        Cached H1/M15 pickles
```

### 3.3 Decision flow (per symbol, 1s cycle)

```
1. Session filter           ── crypto 24/7, others 06–22 UTC, toxic hours
2. Indicators (H1)          ── EMA(15/40/80), SuperTrend, MACD, RSI, ATR, BBW
3. Momentum score           ── 11 components × session α × DOW α → 0-10 long/short
4. Regime detect            ── BBW + ADX → trending/ranging/volatile/low_vol
5. Quality gate             ── score / max(L,S) ≥ SIGNAL_QUALITY_SYMBOL[regime]
6. Direction bias gate      ── DIRECTION_BIAS check (LONG/SHORT/BOTH)
7. M15 confluence           ── MTF intelligence agree?
8. ML meta-label gate       ── DRAGON_ML_ENABLED → LightGBM ensemble veto
9. MasterBrain               ── 6 portfolio gates: kill-switch, DD, max-positions,
                                exposure, daily-loss, weekly-loss
10. RL gate                 ── should_skip_entry()
11. Position sizing         ── 1% × risk_cap × ATR-distance-aware
12. Order send              ── 3-position broker-side architecture (TP1/TP2/TP3)
13. Trail update            ── SYMBOL_TRAIL_OVERRIDE + RL adjustments
14. Journal                 ── outcome → RLLearner + LearningEngine
```

### 3.4 Tuning pipeline (the "hard tune")

```
                   tune_universe.py (pass1, 96 combos × 60 sym × 180d)
                          │
                          ↓
                   tune_180d_pass1.json
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
tune_pass2_fine.py   sweep_trails.py    tune_direction_bias.py
        │                 │                 │
        ↓                 ↓                 ↓
tune_180d_pass2.json   trail_*.py      direction_bias_*.py
        │                 │                 │
        └────────┬────────┴────────┬────────┘
                 ↓                 ↓
       profile_toxic_hours.py   rescue_losers.py
                 ↓                 ↓
        toxic_hours_*.py       rescue_*.py
                 │                 │
                 └────────┬────────┘
                          ↓
                 retune_bleeders_90d.py     ← regime-corrects 180d artifacts
                          ↓
                 retune_bleeders_90d.json
                          ↓
                 synthesize_auto_tuned.py   ← merges everything
                          ↓
                    auto_tuned.py            ← imported by config.py
                          ↓
                 v5_backtest.py validate
                          ↓
                  RESTART LIVE
```

---

## 4. Operations Cheat-sheet

### 4.1 Live agent

```bash
# Status
pgrep -f beast-trader/run.py
ps -p $(pgrep -f beast-trader/run.py) -o pid,etime,pcpu,pmem

# Tail log
tail -f /Users/ashish/Documents/beast-trader/logs/dragon.log

# Restart (launchd auto-restarts after SIGTERM)
kill -TERM $(pgrep -f beast-trader/run.py | head -1)

# Force-restart via launchctl (if process is stuck)
launchctl kickstart -k gui/$UID/com.dragon.trader

# Stop entirely (won't auto-restart until next launch)
launchctl unload ~/Library/LaunchAgents/com.dragon.trader.plist

# Start back up
launchctl load ~/Library/LaunchAgents/com.dragon.trader.plist
```

### 4.2 Tuning pipeline (run end-to-end)

```bash
cd /Users/ashish/Documents/beast-trader

# 1. Coarse tune (5 min)
TUNE_DAYS=180 TUNE_PASS=pass1 TUNE_WORKERS=6 python3 -B scripts/tune_universe.py

# 2. Fine tune (10 min) — must run after pass1
TUNE_DAYS=180 TUNE_WORKERS=6 python3 -B scripts/tune_pass2_fine.py

# 3. Per-aspect sweeps (parallel-safe, run independently)
python3 -B scripts/tune_direction_bias.py
python3 -B scripts/sweep_trails.py
python3 -B scripts/profile_toxic_hours.py
python3 -B scripts/rescue_losers.py
python3 -B scripts/walk_forward.py
python3 -B scripts/stress_spread.py

# 4. Targeted 90d retune for any symbols that bled in recent regime
python3 -B scripts/retune_bleeders_90d.py
#   Edit the SYMBOLS list inside the script first

# 5. Merge into auto_tuned.py
python3 -B scripts/synthesize_auto_tuned.py

# 6. Validate
python3 -B backtest/v5_backtest.py --days 90 --all-symbols --rl-trail \
  > backtest/results/validate_90d.txt 2>&1

# 7. Apply: restart live
kill -TERM $(pgrep -f beast-trader/run.py | head -1)
```

### 4.3 Backtest one-offs

```bash
# Single symbol, 90d
python3 -B backtest/v5_backtest.py --days 90 --symbol XAUUSD --rl-trail

# Full universe, 180d
python3 -B backtest/v5_backtest.py --days 180 --all-symbols --rl-trail

# Without ML gate (raw strategy)
python3 -B backtest/v5_backtest.py --days 90 --all-symbols --no-ml-gate

# Live-set only (the 31 in config.SYMBOLS)
python3 -B scripts/backtest_live_set.py
```

### 4.4 Cache + ML

```bash
# Refresh price cache (REQUIRES live agent stopped — only one MT5 session)
launchctl unload ~/Library/LaunchAgents/com.dragon.trader.plist
python3 -B scripts/refresh_cache.py
launchctl load ~/Library/LaunchAgents/com.dragon.trader.plist

# Manually retrain ML meta-labels for a symbol
python3 -B train_meta_labels.py --symbol XAUUSD

# Drift check (daily)
python3 -B scripts/drift_detector.py
```

### 4.5 Dashboard

```bash
# Start
python3 -B dashboard/app.py
# Open http://127.0.0.1:7777
```

### 4.6 Diagnostics

```bash
# Recent live trades (canonical path: data/)
sqlite3 data/trade_journal.db "SELECT symbol, dir, entry_time, pnl, r_multiple FROM trades ORDER BY entry_time DESC LIMIT 20;"

# Per-symbol live stats (last 30d)
python3 -B scripts/live_vs_backtest.py

# RL state
sqlite3 data/rl_learner.db "SELECT symbol, lock_threshold_mult, trail_tightness_mult FROM trail_adjustments;"
```

### 4.7 Persistence + recovery (issue #21)

Nightly DB backup runs at 03:00 local via `launchd/com.dragon.backup.plist`
(`ThrottleInterval=86400`). Snapshots use sqlite's online `.backup` API
(no DB lock) and gzip the result into `~/backups/dragon/<timestamp>/`.

```bash
# Manual backup
python3 -B scripts/backup_dbs.py

# Manual backup to a custom location
python3 -B scripts/backup_dbs.py --target-dir /Volumes/external/dragon-bkp

# Show what retention would prune (no deletion)
python3 -B scripts/backup_dbs.py --skip-backup --retention-dry-run

# Load the backup launchd job
launchctl load ~/Library/LaunchAgents/com.dragon.backup.plist
```

Retention windows: hourly for 7d, daily for 8–30d, weekly (latest per ISO
week) for 31–365d, pruned beyond 12 months. Each snapshot carries a
`manifest.json` with `source_sha256` + `backup_sha256` per DB so integrity
can be verified after restore.

```bash
# Recovery smoke test (DRY-RUN by default — does NOT touch the live agent)
python3 -B scripts/recovery_smoke_test.py

# Real test (DEMO ONLY — SIGTERMs the live agent and waits for launchd
# KeepAlive to relaunch, then re-checks DB state):
python3 -B scripts/recovery_smoke_test.py --actually-test
```

The canonical DB locations are `data/rl_learner.db` and
`data/trade_journal.db`. `run.py` asserts these exist on startup and
refuses to launch if a non-empty stale DB is detected at the repo root
(catches dual-write footguns).

---

## 5. Risk + Safety Rails

| Knob | Value | Purpose |
|---|---|---|
| `MAX_RISK_PER_TRADE_PCT` | 0.4 | Halved after April bleed |
| `MAX_TOTAL_EXPOSURE_PCT` | 4.0 | 4 full positions worst case |
| `MAX_POSITIONS` | 4 | Caps simultaneous trades regardless of universe size |
| `DAILY_HARD_STOP_PCT` | 2.0 | Halts trading on -2% daily |
| `WEEKLY_HARD_STOP_PCT` | 5.0 | Halts trading on -5% weekly |
| `DD_EMERGENCY_CLOSE` | 8.0 | Closes all on 8% account DD |
| `COOLDOWN_BROKER_CLOSE_SECS` | 2700 | 45min between same-symbol entries |

`MAX_POSITIONS=4` is the most important rail: even with 31 symbols flagged for entry, only 4 can trade simultaneously. Universe expansion ≠ risk explosion.

---

## 6. Open Watch List

1. **EURCHF** — backtests at -$76/90d. Retune found +$76 without RL trail; -$76 with RL trail. RL interaction needs investigation. Consider disabling RL for EURCHF or dropping from `config.SYMBOLS`.
2. **GBPUSD** — flat at -$6 in 90d (only 30 trades). Low conviction — soak before deciding.
3. **NG-Cr** — cache file too short for 90d backtest. Run `refresh_cache.py` for it.
4. **5 untuned symbols at universe expansion** (DJ30.r/FRA40.r/HK50.r/SWI20.r) — pass2 v2 covered them. Re-validate after 7d soak.
5. **24h soak watch** — if any symbol shows live PF < 0.5 over 10+ trades, halve its risk or disable.
6. **XPTUSD.r data structure** — pickle has integer index, may affect XPDUSD.r similarly. Investigate later.

---

## 7. File Pointers

- Live config: `config.py` (lines 35–66 = SYMBOLS, 99–108 = risk caps)
- Tuner overrides: `auto_tuned.py` (regenerated, do not hand-edit)
- Latest results: `backtest/results/validate_90d_v4.txt`
- Tuner JSONs: `backtest/results/tune_180d_pass{1,2}.json`, `retune_bleeders_90d.json`
- Live log: `logs/dragon.log`
- Dashboard: http://127.0.0.1:7777

---

*Generated: 2026-05-03 · Session-end commit `7c8b6e3`*
