"""
Bot-only drawdown tracker (2026-07-18).

Builds a synthetic equity curve for the BOT's own books ONLY:

    bot_equity = baseline + Σ(bot realized) + Σ(bot unrealized)

and ratchets an up-only `bot_peak` on it so `bot_dd_pct` reacts to the bot's
own drawdown — NEVER to the user's MANUAL trades (magic 0/2024), which drain
raw account equity but ratchet nothing and used to freeze the whole bot with a
phantom EmergencyDD.

Design (winning spec, 2026-07-18):
  * REALIZED comes from a watermarked `history_deals_get` accumulator filtered by
    the full `_managed_offsets` predicate — the ONLY source that captures every
    bot book uniformly (momentum/scalp/fvg/sr/smabo/fib50/scalper/trend/imr/
    gold_smc/asat) AND structurally excludes manual magic 0/2024. The sparse
    journal/daily_stats SUM was REJECTED because it omits most bot books, which
    would let a real blowup under-report DD (a capital-preservation failure).
  * UNREALIZED reuses the caller's once-per-cycle positions snapshot (zero new
    per-tick Wine-bridge calls). The realized accumulator is throttled off-tick.
  * FAIL CONSERVATIVE: any read failure, stale positions, missing/corrupt state,
    or accumulator gap → healthy=False → caller falls back to RAW dd_pct (the
    over-stops side). A failure must NEVER produce a spuriously-LOW bot_dd that
    disables capital preservation.

Pure/dependency-injected so it is unit-testable without a live MT5 bridge.
"""
from __future__ import annotations

import time as _time
from collections import deque
from datetime import datetime, timezone

_KV_KEY = "bot_equity_state"


def bot_offsets():
    """The set of magic sub-offsets owned by the BOT's books — single source of
    truth is config._STRATEGY_BY_OFFSET (momentum {0,1,2}, scalp, fvg, sr, smabo,
    fib50, scalper, trend, imr, gold_smc, asat). Manual magic 0/2024 resolves to
    a negative offset (magic - base_magic, base ~8100+) → never a member."""
    try:
        from config import _STRATEGY_BY_OFFSET
        return frozenset(int(k) for k in _STRATEGY_BY_OFFSET.keys())
    except Exception:
        # Conservative explicit fallback (kept in sync with config).
        return frozenset({0, 1, 2, 500, 501, 1000, 1001, 2000, 2001, 3000, 3001,
                          4000, 4001, 5000, 5001, 6000, 6001, 7000, 7001,
                          8000, 8001, 9000, 9001})


class BotEquityTracker:
    """Stateful bot-only DD tracker. All external I/O is injected as callables so
    the class can be exercised deterministically in tests.

    Args:
        symbol_cfg: callable(symbol) -> cfg with `.magic` (or None). Manual /
                    unknown symbols return None → not a bot position.
        kv_get / kv_set: durable JSON KV (survives the ~15 daily restarts).
        deals_getter: callable(start_dt, end_dt) -> iterable of deal objects, may
                      raise on bridge outage. Each deal exposes .ticket .magic
                      .symbol .entry(0=entry-leg) .profit .swap .commission .time.
        now: monotonic-ish wall clock (defaults to time.time).
    """

    def __init__(self, *, symbol_cfg, kv_get, kv_set, deals_getter,
                 offsets=None, now=None, log=None,
                 overlap_secs=120.0, accum_throttle_secs=5.0,
                 persist_throttle_secs=60.0, recent_max=2000):
        self._symbol_cfg = symbol_cfg
        self._kv_get = kv_get
        self._kv_set = kv_set
        self._deals_getter = deals_getter
        self._offsets = frozenset(offsets) if offsets is not None else bot_offsets()
        self._now = now or _time.time
        self._log = log
        self._overlap = float(overlap_secs)
        self._accum_throttle = float(accum_throttle_secs)
        self._persist_throttle = float(persist_throttle_secs)
        self._recent_max = int(recent_max)

        # ── persisted state ──
        self.baseline_equity = None      # anchor (any constant; DD is relative)
        self.baseline_ts = None
        self.baseline_date = None
        self.realized_accum = 0.0        # cumulative bot realized since baseline
        self.watermark_ts = None         # latest bot exit-deal time consumed
        self.recent_tickets = deque(maxlen=self._recent_max)
        self._recent_set = set()
        self.bot_peak = 0.0              # up-only ratchet (mandatory restore)
        self.bot_daily_start = None      # bot_equity snapshot at UTC day rollover
        self.bot_weekly_start = None     # bot_equity snapshot at Monday rollover

        # ── runtime (never persisted) ──
        self.healthy = False
        self.last_bot_equity = None
        self.last_bot_dd = 0.0
        self.last_unrealized = 0.0
        self._accum_gap = False          # sticky until a scan succeeds
        self._last_accum_ts = 0.0
        self._last_persist_ts = 0.0
        self._initialized = False

    # ────────────────────────────────────────────────────────────
    #  attribution
    # ────────────────────────────────────────────────────────────
    def is_bot(self, magic, symbol):
        try:
            cfg = self._symbol_cfg(symbol)
        except Exception:
            cfg = None
        if cfg is None:
            return False
        try:
            return (int(magic) - int(cfg.magic)) in self._offsets
        except Exception:
            return False

    # ────────────────────────────────────────────────────────────
    #  persistence
    # ────────────────────────────────────────────────────────────
    def load(self):
        """Restore persisted state. Corrupt/missing → stay uninitialized → the
        caller uses RAW dd_pct until the accumulator re-anchors from baseline_ts.
        Never resets to 0-DD mid-drawdown: bot_peak is restored up-only."""
        try:
            st = self._kv_get(_KV_KEY, None)
        except Exception:
            st = None
        if not st:
            return False
        try:
            self.baseline_equity = float(st["baseline_equity"])
            self.baseline_ts = float(st["baseline_ts"])
            self.baseline_date = st.get("baseline_date")
            self.realized_accum = float(st.get("realized_accum", 0.0))
            self.watermark_ts = float(st.get("watermark_ts", self.baseline_ts))
            rt = [int(x) for x in (st.get("recent_tickets") or [])]
            self.recent_tickets = deque(rt, maxlen=self._recent_max)
            self._recent_set = set(rt)
            self.bot_peak = float(st.get("bot_peak", 0.0))
            bds = st.get("bot_daily_start")
            bws = st.get("bot_weekly_start")
            self.bot_daily_start = float(bds) if bds is not None else None
            self.bot_weekly_start = float(bws) if bws is not None else None
            self._initialized = True
            return True
        except Exception:
            # Corrupt payload → treat as uninitialized (RAW fallback).
            self._initialized = False
            return False

    def _state_dict(self):
        return {
            "baseline_equity": self.baseline_equity,
            "baseline_ts": self.baseline_ts,
            "baseline_date": self.baseline_date,
            "realized_accum": self.realized_accum,
            "watermark_ts": self.watermark_ts,
            "recent_tickets": list(self.recent_tickets),
            "bot_peak": self.bot_peak,
            "bot_daily_start": self.bot_daily_start,
            "bot_weekly_start": self.bot_weekly_start,
            "updated_ts": self._now(),
        }

    def persist(self, force=False):
        now = self._now()
        if not force and (now - self._last_persist_ts) < self._persist_throttle:
            return
        self._last_persist_ts = now
        try:
            self._kv_set(_KV_KEY, self._state_dict())
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────
    #  init
    # ────────────────────────────────────────────────────────────
    def first_init(self, current_equity, raw_peak=None):
        """Anchor a fresh curve. baseline is arbitrary (DD is relative); if the
        account is not flat of bot positions this leaves a one-time cosmetic
        offset that self-corrects at close (realized rises as unrealized falls).

        CAPITAL-PRESERVATION FIX (2026-07-18): seed bot_peak from the account's
        RAW up-only peak when available. A fresh anchor triggered by corrupt/
        missing persisted state MID-DRAWDOWN must NOT reset to 0-DD (that would
        silently drop crash protection). Seeding from raw_peak makes the corrupt-
        state path behave like RAW dd (conservative/over-stops), which is exactly
        the intended fail-safe. On a genuinely flat startup raw_peak≈ce so this is
        a no-op."""
        now = self._now()
        try:
            ce = float(current_equity)
        except Exception:
            ce = 0.0
        try:
            rp = float(raw_peak) if raw_peak is not None else ce
        except Exception:
            rp = ce
        self.baseline_equity = ce
        self.baseline_ts = now
        self.baseline_date = datetime.fromtimestamp(now, tz=timezone.utc).date().isoformat()
        self.realized_accum = 0.0
        self.watermark_ts = now
        self.recent_tickets.clear()
        self._recent_set.clear()
        self.bot_peak = max(ce, rp)
        self.bot_daily_start = ce
        self.bot_weekly_start = ce
        self._accum_gap = False
        self._initialized = True
        self.persist(force=True)

    # ────────────────────────────────────────────────────────────
    #  realized accumulator (throttled, off-tick)
    # ────────────────────────────────────────────────────────────
    def _add_ticket(self, tk):
        if len(self.recent_tickets) == self.recent_tickets.maxlen:
            evicted = self.recent_tickets[0]  # oldest, about to be pushed out
            self._recent_set.discard(evicted)
        self.recent_tickets.append(tk)
        self._recent_set.add(tk)

    def update_realized(self, force=False):
        """Scan broker deals since (watermark - overlap), add bot-attributed
        exit-leg P/L to realized_accum, dedup by ticket. Throttled off-tick.

        Fail-safe: on getter exception we DO NOT advance the watermark (next scan
        widens automatically) and set a sticky gap flag → healthy=False until a
        scan succeeds. Never advances past a gap."""
        if not self._initialized:
            return
        now = self._now()
        if not force and (now - self._last_accum_ts) < self._accum_throttle and not self._accum_gap:
            return
        self._last_accum_ts = now
        base = self.watermark_ts if self.watermark_ts is not None else self.baseline_ts
        if base is None:
            return
        start_ts = base - self._overlap
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(now + 2.0, tz=timezone.utc)
        try:
            deals = self._deals_getter(start_dt, end_dt)
        except Exception:
            # bridge outage — keep watermark, mark gap, fall back to RAW.
            self._accum_gap = True
            return
        if deals is None:
            deals = []
        max_time = base
        for d in deals:
            try:
                if int(getattr(d, "entry", 0)) == 0:
                    continue  # entry-leg fill, not a realized result
                dt = float(getattr(d, "time", 0) or 0.0)
                if dt > max_time:
                    max_time = dt
                tk = int(getattr(d, "ticket", 0))
                if tk in self._recent_set:
                    continue  # already counted (overlap seam)
                if not self.is_bot(getattr(d, "magic", -1), getattr(d, "symbol", "")):
                    continue  # manual / non-bot leg — structurally excluded
                self.realized_accum += (float(getattr(d, "profit", 0.0) or 0.0)
                                        + float(getattr(d, "swap", 0.0) or 0.0)
                                        + float(getattr(d, "commission", 0.0) or 0.0))
                self._add_ticket(tk)
            except Exception:
                continue
        # advance watermark ONLY after a successful scan
        self.watermark_ts = max(base, max_time)
        self._accum_gap = False

    # ────────────────────────────────────────────────────────────
    #  compute
    # ────────────────────────────────────────────────────────────
    def compute(self, positions, positions_ok, account_equity, raw_peak=None):
        """Return (bot_equity, bot_dd_pct, healthy).

        positions      : caller's reused snapshot (list of position objects).
        positions_ok   : False if the snapshot fetch failed OR is stale → unhealthy.
        account_equity : used ONLY to anchor the baseline on first init.
        raw_peak       : the account's RAW up-only peak_equity; seeds bot_peak on a
                         fresh anchor so corrupt-state-mid-drawdown can't reset to 0-DD.

        Peak is ratcheted up-only and ONLY when healthy (never ratchet on a value
        computed from missing/stale unrealized)."""
        if not self._initialized:
            try:
                ce = float(account_equity)
            except Exception:
                ce = 0.0
            if ce > 0:
                self.first_init(ce, raw_peak=raw_peak)
            # Fresh anchor (startup OR corrupt/missing state mid-run): do NOT trust a
            # 0-DD reading this cycle. Force RAW fallback (healthy=False) until a
            # proven persisted curve exists. bot_peak was seeded from raw_peak above,
            # so the NEXT cycle already reflects any real drawdown conservatively.
            self.healthy = False
            return (None, 0.0, False)

        healthy = True
        if not positions_ok:
            healthy = False
        if self._accum_gap:
            healthy = False
        if self.baseline_equity is None:
            healthy = False

        unreal = 0.0
        if positions_ok and positions is not None:
            for p in positions:
                try:
                    if self.is_bot(getattr(p, "magic", -1), getattr(p, "symbol", "")):
                        unreal += (float(getattr(p, "profit", 0.0) or 0.0)
                                   + float(getattr(p, "swap", 0.0) or 0.0))
                except Exception:
                    continue
        self.last_unrealized = unreal

        bot_equity = float(self.baseline_equity) + float(self.realized_accum) + float(unreal)
        self.last_bot_equity = bot_equity

        if healthy and bot_equity > self.bot_peak:
            self.bot_peak = bot_equity

        dd = 0.0
        if self.bot_peak > 0:
            dd = max(0.0, (self.bot_peak - bot_equity) / self.bot_peak * 100.0)
        self.last_bot_dd = dd
        self.healthy = healthy
        return (bot_equity, dd, healthy)

    def bot_dd_or_fallback(self, raw_dd):
        """(dd_pct, used_bot). Returns bot_dd iff healthy, else RAW dd_pct."""
        if self.healthy and self.last_bot_equity is not None:
            return (float(self.last_bot_dd), True)
        try:
            return (float(raw_dd), False)
        except Exception:
            return (0.0, False)

    # ────────────────────────────────────────────────────────────
    #  daily / weekly rollovers (bot-equity snapshots for loss gates)
    # ────────────────────────────────────────────────────────────
    def roll_daily(self):
        if self.last_bot_equity is not None:
            self.bot_daily_start = float(self.last_bot_equity)
            self.persist(force=True)

    def roll_weekly(self):
        if self.last_bot_equity is not None:
            self.bot_weekly_start = float(self.last_bot_equity)
            self.persist(force=True)

    def bot_loss_pct(self, start_attr):
        """Positive loss % of a persisted bot-equity start snapshot, or None if
        unavailable/unhealthy. Fail-safe: None → caller keeps RAW gate."""
        if not self.healthy or self.last_bot_equity is None:
            return None
        start = getattr(self, start_attr, None)
        if start is None or start <= 0:
            return None
        loss = start - self.last_bot_equity
        if loss <= 0:
            return 0.0
        return loss / start * 100.0
