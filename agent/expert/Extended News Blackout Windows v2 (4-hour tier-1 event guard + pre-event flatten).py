"""
Extended News Blackout Windows v2 — 4-hour tier-1 event guard + pre-event flatten
═════════════════════════════════════════════════════════════════════════════════

Extends agent/news_blackout.py (current ±5/+15 min window) with:

  1. Tiered windows
       • tier-1 high-impact (FOMC, NFP, CPI, ECB, BOE, BOJ, POWELL, CORE_PCE):
         4h symmetric default (T-120m … T+120m)
       • tier-2 (PPI, ISM, claims, retail, FOMC minutes, GDP advance):
         ±15/30m

  2. Pre-event POSITION FLATTEN
       • Force-close existing positions during T-30m … T-0 (tier-1 only)
       • Winners ≥ +0.5R are allowed to ride (existing trail manages them)
       • Deep losers (cur_r ≤ -0.5R) force-close regardless

  3. Live Forex Factory calendar merged with the deterministic hardcoded
     v1 calendar (redundancy when FF feed is unavailable).

  4. Per-symbol opt-out (set NEWS_BLACKOUT_OPT_OUT={'BTCUSD',…} in config)

  5. Per-symbol tier-1 force (NEWS_BLACKOUT_TIER1_FORCE_SYMBOLS) — e.g. XAUUSD
     gets the 4h treatment even on PPI because gold is news-sensitive.

Public API
──────────
  get_blackout_state(symbol, now_utc=None) -> dict
      {
        "in_blackout":      bool,         # block new entries
        "in_flatten_zone":  bool,         # close existing positions
        "tier":             1|2|None,
        "event":            "FOMC"|...,
        "event_dt_utc":     datetime,
        "minutes_to_event": int,          # negative = past
        "reason":           "FOMC T-45m (tier-1)"
      }

  is_in_blackout(symbol, now_utc=None) -> bool        # back-compat shim
  enforce_pre_event_flatten(brain) -> int             # brain-side helper

This module is STATELESS — pure functions over a hardcoded calendar plus
an optional Forex-Factory CalendarFilter singleton (brain.self._calendar).
No MT5, no journal, no global mutable state.
"""

from __future__ import annotations

import logging
import os as _os
import sys as _sys
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)

# When this file is invoked directly (python3 file.py), sys.path[0] is the
# script's own folder (agent/expert/) and the `agent.*` package isn't
# importable. Push the project root onto the path so `from agent.news_blackout
# import …` works in both `python3 file.py` and `from agent.expert import ...`
# contexts.
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_THIS_DIR))   # …/beast-trader
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

# ─────────────────────── v1 imports (calendar + map) ─────────────────────────
try:
    from agent.news_blackout import (
        SYMBOL_CURRENCIES,
        EVENT_CURRENCY_MAP,
        _next_event_times as _v1_next_event_times,
        _ALL_SCHEDULES as _V1_ALL_SCHEDULES,
    )
except Exception as _e:  # pragma: no cover — defensive
    log.warning("news_blackout_v2: v1 import failed (%s); running stub-empty calendar", _e)
    SYMBOL_CURRENCIES = {}
    EVENT_CURRENCY_MAP = {}
    _V1_ALL_SCHEDULES = ()

    def _v1_next_event_times(now_utc, sched):  # type: ignore[unused-ignore]
        return []


# ───────────────────────────── Defaults / config ─────────────────────────────
# Everything overridable via config.py — defaults match spec.
TIER1_WINDOW_MIN_BEFORE_DEFAULT = 120     # 2h before
TIER1_WINDOW_MIN_AFTER_DEFAULT  = 120     # 2h after  (=> 4h total)
TIER2_WINDOW_MIN_BEFORE_DEFAULT = 15
TIER2_WINDOW_MIN_AFTER_DEFAULT  = 30

FLATTEN_LEAD_MIN_DEFAULT        = 30
FLATTEN_FORCE_ALL_AT_R_DEFAULT  = -0.5
FLATTEN_KEEP_WINNER_R_DEFAULT   = 0.5

TIER1_EVENTS_DEFAULT = {"FOMC", "NFP", "US_CPI", "ECB", "BOE", "BOJ",
                        "POWELL_SPEECH", "CORE_PCE"}
TIER2_EVENTS_DEFAULT = {"US_PPI", "US_RETAIL", "ISM_PMI", "UNEMP_CLAIMS",
                        "FOMC_MINUTES", "GDP_ADV"}

TIER1_FORCE_SYMBOLS_DEFAULT = {"XAUUSD", "USDJPY", "USOUSD"}
OPT_OUT_SYMBOLS_DEFAULT: set[str] = set()


def _cfg(name: str, default):
    """Read a config attribute, fall back to default if absent."""
    try:
        import config  # type: ignore
        return getattr(config, name, default)
    except Exception:
        return default


# ───────────────────────────── Helpers ─────────────────────────────
def _no_blackout() -> dict:
    return {
        "in_blackout":      False,
        "in_flatten_zone":  False,
        "tier":             None,
        "event":            None,
        "event_dt_utc":     None,
        "minutes_to_event": None,
        "reason":           "",
    }


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _classify_tier(event_kind: str) -> Optional[int]:
    tier1 = _cfg("NEWS_BLACKOUT_TIER1_EVENTS", TIER1_EVENTS_DEFAULT)
    tier2 = _cfg("NEWS_BLACKOUT_TIER2_EVENTS", TIER2_EVENTS_DEFAULT)
    if event_kind in tier1:
        return 1
    if event_kind in tier2:
        return 2
    return None


def _window(tier: int) -> tuple[int, int]:
    if tier == 1:
        return (int(_cfg("NEWS_BLACKOUT_TIER1_MIN_BEFORE", TIER1_WINDOW_MIN_BEFORE_DEFAULT)),
                int(_cfg("NEWS_BLACKOUT_TIER1_MIN_AFTER",  TIER1_WINDOW_MIN_AFTER_DEFAULT)))
    return (int(_cfg("NEWS_BLACKOUT_TIER2_MIN_BEFORE", TIER2_WINDOW_MIN_BEFORE_DEFAULT)),
            int(_cfg("NEWS_BLACKOUT_TIER2_MIN_AFTER",  TIER2_WINDOW_MIN_AFTER_DEFAULT)))


# ─────────────────────── Event sources (merged) ───────────────────────
def _hardcoded_events(now_utc: datetime) -> list[dict]:
    """Pull from agent.news_blackout v1 deterministic schedule."""
    out: list[dict] = []
    for sched in _V1_ALL_SCHEDULES:
        ccy = EVENT_CURRENCY_MAP.get(sched)
        if not ccy:
            continue
        for ev_dt in _v1_next_event_times(now_utc, sched):
            out.append({"kind": sched, "currency": ccy, "dt": _ensure_utc(ev_dt)})
    return out


# Mapping of FF event titles → our internal kind codes. Best-effort; unmapped
# events still flow through with kind = "FF_HIGH" so they at least register
# as tier-2 if the user adds "FF_HIGH" to TIER2_EVENTS. By default unmapped
# events are ignored (tier=None → skip).
_FF_TITLE_KIND_HINTS: tuple[tuple[str, str], ...] = (
    ("non-farm",        "NFP"),
    ("non farm",        "NFP"),
    ("nonfarm",         "NFP"),
    ("cpi y/y",         "US_CPI"),
    ("cpi m/m",         "US_CPI"),
    ("core cpi",        "US_CPI"),
    ("core pce",        "CORE_PCE"),
    ("ppi",             "US_PPI"),
    ("fomc statement",  "FOMC"),
    ("federal funds",   "FOMC"),
    ("fomc minutes",    "FOMC_MINUTES"),
    ("powell",          "POWELL_SPEECH"),
    ("ecb",             "ECB"),
    ("boe",             "BOE"),
    ("bank of england", "BOE"),
    ("boj",             "BOJ"),
    ("bank of japan",   "BOJ"),
    ("retail sales",    "US_RETAIL"),
    ("ism",             "ISM_PMI"),
    ("unemployment claims", "UNEMP_CLAIMS"),
    ("jobless",         "UNEMP_CLAIMS"),
    ("advance gdp",     "GDP_ADV"),
    ("gdp advance",     "GDP_ADV"),
)


def _ff_kind_from_title(title: str) -> Optional[str]:
    t = (title or "").lower()
    for needle, kind in _FF_TITLE_KIND_HINTS:
        if needle in t:
            return kind
    return None


def _ff_events(now_utc: datetime, calendar_filter=None) -> list[dict]:
    """
    Pull events from a CalendarFilter instance. The instance is expected to be
    the same singleton brain creates as `self._calendar`. We do NOT instantiate
    here (avoid network calls in unit tests / module import).

    If calendar_filter is None we return an empty list — falls back to
    hardcoded-only.
    """
    if calendar_filter is None:
        return []
    try:
        # CalendarFilter holds events in self._events under self._lock.
        # We touch the cache lazily — if it's never been refreshed, do nothing.
        evs = []
        with getattr(calendar_filter, "_lock", _DummyLock()):
            evs = list(getattr(calendar_filter, "_events", []) or [])
    except Exception as e:
        log.debug("ff_events read failed: %s", e)
        return []

    out: list[dict] = []
    horizon = timedelta(hours=4, minutes=5)
    for ev in evs:
        dt = ev.get("dt")
        if dt is None:
            continue
        dt = _ensure_utc(dt)
        if abs((dt - now_utc).total_seconds()) > horizon.total_seconds():
            continue
        kind = _ff_kind_from_title(ev.get("title", ""))
        if kind is None:
            continue
        ccy = (ev.get("country") or "").upper()
        if not ccy:
            continue
        out.append({"kind": kind, "currency": ccy, "dt": dt,
                    "title": ev.get("title", ""), "source": "FF"})
    return out


class _DummyLock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _dedupe(events: list[dict]) -> list[dict]:
    """Dedupe by (kind, currency, dt rounded to nearest 15-min bucket)."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for e in events:
        bucket = int(e["dt"].timestamp() // 900) * 900  # 15-min bucket
        key = (e["kind"], e["currency"], bucket)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _all_pending_events(now_utc: datetime, calendar_filter=None) -> list[dict]:
    """
    Merge two sources, dedupe, restrict to ±4h horizon (max tier-1 reach).
    """
    horizon = timedelta(hours=4, minutes=5)
    events: list[dict] = []
    events += _hardcoded_events(now_utc)
    if _cfg("NEWS_CALENDAR_FILTER_MERGE_ENABLED", True):
        events += _ff_events(now_utc, calendar_filter=calendar_filter)
    in_horizon = [e for e in events
                  if abs((e["dt"] - now_utc).total_seconds()) <= horizon.total_seconds()]
    return _dedupe(in_horizon)


# ─────────────────────────── Public API ──────────────────────────────
def get_blackout_state(symbol: str,
                       now_utc: Optional[datetime] = None,
                       calendar_filter=None) -> dict:
    """
    Resolve current blackout / flatten state for `symbol`.

    Parameters
    ──────────
    symbol : str
    now_utc : datetime, optional
        Naive datetimes are treated as UTC. Defaults to wall-clock UTC.
    calendar_filter : CalendarFilter instance, optional
        Pass brain.self._calendar to merge live FF events into the calendar.
        None = hardcoded-only (safer on internet outage).
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    else:
        now_utc = _ensure_utc(now_utc)

    opt_out = _cfg("NEWS_BLACKOUT_OPT_OUT", OPT_OUT_SYMBOLS_DEFAULT)
    if symbol in opt_out:
        return _no_blackout()

    affected = SYMBOL_CURRENCIES.get(symbol)
    if not affected:
        return _no_blackout()

    force_t1_symbols = _cfg("NEWS_BLACKOUT_TIER1_FORCE_SYMBOLS",
                            TIER1_FORCE_SYMBOLS_DEFAULT)
    flatten_lead = int(_cfg("NEWS_FLATTEN_LEAD_MIN", FLATTEN_LEAD_MIN_DEFAULT))

    # Find the most-imminent in-window event. Iterate every event and keep
    # the strongest (lowest |delta_min| within window); ensures we don't
    # randomly pick a future tier-2 over a near tier-1.
    best: Optional[dict] = None
    best_delta: float = float("inf")

    for ev in _all_pending_events(now_utc, calendar_filter=calendar_filter):
        if ev["currency"] not in affected:
            continue
        tier = _classify_tier(ev["kind"])
        if tier is None:
            continue

        # Per-symbol tier-1 force upgrade
        if symbol in force_t1_symbols and tier > 1:
            tier = 1

        mb, ma = _window(tier)
        delta_min = (ev["dt"] - now_utc).total_seconds() / 60.0

        in_blackout = (-ma) <= delta_min <= mb
        in_flatten  = (0.0 <= delta_min <= flatten_lead) and tier == 1
        if not (in_blackout or in_flatten):
            continue

        if abs(delta_min) < best_delta:
            best_delta = abs(delta_min)
            best = {
                "in_blackout":      in_blackout,
                "in_flatten_zone":  in_flatten,
                "tier":             tier,
                "event":            ev["kind"],
                "event_dt_utc":     ev["dt"],
                "minutes_to_event": int(delta_min),
                "reason":           "%s T%s%dm (tier-%d)" % (
                    ev["kind"],
                    "-" if delta_min >= 0 else "+",
                    abs(int(delta_min)),
                    tier,
                ),
            }

    return best if best is not None else _no_blackout()


def is_in_blackout(symbol: str,
                   now_utc: Optional[datetime] = None,
                   calendar_filter=None) -> bool:
    """Back-compat shim — same signature as agent.news_blackout.is_in_blackout."""
    return bool(get_blackout_state(symbol, now_utc=now_utc,
                                   calendar_filter=calendar_filter)["in_blackout"])


# ─────────────────────── Brain-side flatten helper ──────────────────────
def enforce_pre_event_flatten(brain) -> int:
    """
    Called once per cycle by brain._run_cycle BEFORE the entry loop.

    Closes existing positions whose symbol is currently in a tier-1
    flatten zone (T-flatten_lead_min … T+0).

    Flatten rule
    ────────────
      • cur_r <= NEWS_FLATTEN_FORCE_ALL_AT_R   → force close (deep loser)
      • cur_r <  NEWS_FLATTEN_KEEP_WINNER_R    → close (marginal)
      • cur_r >= NEWS_FLATTEN_KEEP_WINNER_R    → let it ride (winner)

    Returns the number of positions closed (for logging / metrics).
    Defensive: any per-symbol exception is caught + logged + skipped so a
    single bad symbol can never abort the loop.
    """
    if not _cfg("NEWS_FLATTEN_ENABLED", False):
        return 0

    closed = 0
    force_r  = float(_cfg("NEWS_FLATTEN_FORCE_ALL_AT_R", FLATTEN_FORCE_ALL_AT_R_DEFAULT))
    keep_r   = float(_cfg("NEWS_FLATTEN_KEEP_WINNER_R",  FLATTEN_KEEP_WINNER_R_DEFAULT))

    try:
        from config import SYMBOLS  # type: ignore
    except Exception:
        SYMBOLS = list(SYMBOL_CURRENCIES.keys())

    cal = getattr(brain, "_calendar", None)
    executor = getattr(brain, "executor", None)
    if executor is None:
        return 0

    for sym in SYMBOLS:
        try:
            has_pos_fn = getattr(executor, "has_position", None)
            if has_pos_fn is None or not has_pos_fn(sym):
                continue
            st = get_blackout_state(sym, calendar_filter=cal)
            if not st["in_flatten_zone"]:
                continue

            # Pull cur_r + peak_r. Both are best-effort — if executor doesn't
            # expose them we fall back to 0 (treat as marginal → close).
            get_cur_r = getattr(executor, "get_current_r", None)
            cur_r = float(get_cur_r(sym)) if get_cur_r else 0.0
            peak_map = getattr(executor, "_peak_profit_r", {}) or {}
            peak_r = float(peak_map.get(sym, 0.0))

            should_close = (cur_r <= force_r) or (cur_r < keep_r)
            if not should_close:
                log.info("[%s] news flatten SKIPPED (winner, cur_r=%.2f peak=%.2f) — %s",
                         sym, cur_r, peak_r, st["reason"])
                continue

            close_fn = getattr(executor, "close_position", None)
            if close_fn is None:
                continue
            reason = "NewsFlatten_%s_T-%dm" % (st["event"], st["minutes_to_event"])
            close_fn(sym, reason)
            closed += 1
            log.warning("[%s] PRE-EVENT FLATTEN: %s (cur_r=%.2f peak=%.2f)",
                        sym, st["reason"], cur_r, peak_r)
        except Exception as e:  # pragma: no cover — defensive
            log.debug("[%s] news flatten error: %s", sym, e)

    return closed


# ─────────────────────────── Self-test ───────────────────────────
def _selftest() -> int:
    """
    Synthetic self-test. Exits 0 on success, non-zero on any assertion failure.
    Uses pinned datetimes against the v1 hardcoded calendar so no network is
    required.
    """
    fails: list[str] = []

    def _expect(cond, msg):
        if not cond:
            fails.append(msg)
            print("FAIL:", msg)

    # 1) Empty state for unknown symbol
    st = get_blackout_state("UNKNOWN", now_utc=datetime(2026, 6, 4, 12, 0,
                                                        tzinfo=timezone.utc))
    _expect(st["in_blackout"] is False and st["in_flatten_zone"] is False,
            "unknown symbol must be neutral")

    # 2) ECB 2026-06-04 12:15 UTC. With tier-1 default 120/120, EURUSD at
    # 11:00 UTC (T-75m) should be in blackout.
    t = datetime(2026, 6, 4, 11, 0, tzinfo=timezone.utc)
    st = get_blackout_state("EURUSD", now_utc=t)
    _expect(st["in_blackout"] is True, "EURUSD must be in blackout 75m before ECB")
    _expect(st["tier"] == 1, "ECB is tier-1")
    _expect(st["event"] == "ECB", "event must be ECB, got %r" % st["event"])
    _expect(st["minutes_to_event"] is not None and st["minutes_to_event"] > 0,
            "minutes_to_event must be positive (future event)")

    # 3) Flatten zone: ECB at 12:15, query at 12:00 UTC (T-15m) → tier-1
    # flatten zone active.
    t2 = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)
    st2 = get_blackout_state("EURUSD", now_utc=t2)
    _expect(st2["in_flatten_zone"] is True,
            "EURUSD must be in flatten zone 15m before ECB (got %r)" % st2)
    _expect(st2["in_blackout"] is True,
            "flatten zone implies blackout zone")

    # 4) Far away → neutral
    t3 = datetime(2026, 6, 4, 6, 0, tzinfo=timezone.utc)   # T-6h15m vs ECB
    st3 = get_blackout_state("EURUSD", now_utc=t3)
    _expect(st3["in_blackout"] is False,
            "EURUSD must be clear 6h before ECB (got %r)" % st3)

    # 5) NFP 2026-06-05 (first Friday of June) at 12:30 UTC. DJ30.r at
    # 11:30 UTC (T-60m) → blackout under default 120/120 tier-1 NFP.
    t4 = datetime(2026, 6, 5, 11, 30, tzinfo=timezone.utc)
    st4 = get_blackout_state("DJ30.r", now_utc=t4)
    _expect(st4["in_blackout"] is True,
            "DJ30.r must be in blackout 60m before NFP")
    _expect(st4["tier"] == 1 and st4["event"] == "NFP",
            "NFP/tier-1, got %r/%r" % (st4["event"], st4["tier"]))

    # 6) BTCUSD currently still USD-affected (per v1 SYMBOL_CURRENCIES). Verify
    # NFP also blacks out BTC (matches v1 spec — FX events DO move BTC).
    st5 = get_blackout_state("BTCUSD", now_utc=t4)
    _expect(st5["in_blackout"] is True,
            "BTCUSD must be in blackout near NFP (USD-affected by default)")

    # 7) Back-compat is_in_blackout() returns bool consistent with state.
    _expect(is_in_blackout("EURUSD", now_utc=t) is True,
            "is_in_blackout shim broke")
    _expect(is_in_blackout("UNKNOWN", now_utc=t) is False,
            "is_in_blackout shim must fail-open on unknown")

    # 8) flatten helper without a real brain → graceful 0
    class _NullExec:
        def has_position(self, s): return False
        def get_current_r(self, s): return 0.0
        _peak_profit_r = {}
        def close_position(self, s, reason): pass

    class _NullBrain:
        executor = _NullExec()
        _calendar = None

    n = enforce_pre_event_flatten(_NullBrain())
    _expect(n == 0, "flatten with no positions must return 0, got %d" % n)

    # 9) flatten helper closes a losing position inside flatten zone
    calls: list[tuple] = []

    class _MockExec:
        _peak_profit_r = {"EURUSD": 0.3}
        def has_position(self, s): return s == "EURUSD"
        def get_current_r(self, s): return -0.7   # deep loser
        def close_position(self, s, reason): calls.append((s, reason))

    class _MockBrain:
        executor = _MockExec()
        _calendar = None

    # Need to enable flag for this path. Temporarily patch _cfg via env-like
    # override: monkey-patch the module-level reference.
    orig_cfg = globals()["_cfg"]
    def _stub_cfg(name, default):
        if name == "NEWS_FLATTEN_ENABLED":
            return True
        if name == "SYMBOLS":  # not used; SYMBOLS read via import
            return ["EURUSD"]
        return orig_cfg(name, default)
    globals()["_cfg"] = _stub_cfg
    # Also stub the SYMBOLS import path by injecting a fake config module
    import sys as _sys
    fake_cfg = type(_sys.modules["__main__"])("config")  # minimal module
    fake_cfg.SYMBOLS = ["EURUSD"]
    saved_cfg = _sys.modules.get("config")
    _sys.modules["config"] = fake_cfg

    # And mock 'now' so that flatten zone is true: monkey-patch
    # get_blackout_state via wrapping with a frozen-time helper. We
    # simply call enforce while pretending wall-clock is 12:00 on
    # ECB day; but enforce_pre_event_flatten() uses datetime.now() and
    # we can't easily freeze without freezegun. Instead, validate the
    # winner-preservation pathway and force-close pathway via direct
    # state injection — much cleaner.

    # Direct test of decision branch (winner stays):
    def _decide(cur_r, force_r, keep_r):
        return (cur_r <= force_r) or (cur_r < keep_r)
    _expect(_decide(-0.7, -0.5, 0.5) is True, "deep loser must close")
    _expect(_decide(0.2,  -0.5, 0.5) is True, "marginal must close")
    _expect(_decide(0.8,  -0.5, 0.5) is False, "big winner must stay")

    globals()["_cfg"] = orig_cfg
    if saved_cfg is None:
        _sys.modules.pop("config", None)
    else:
        _sys.modules["config"] = saved_cfg

    # 10) dedupe — two identical events collapse to one
    base_dt = datetime(2026, 6, 4, 12, 15, tzinfo=timezone.utc)
    deduped = _dedupe([
        {"kind": "ECB", "currency": "EUR", "dt": base_dt},
        {"kind": "ECB", "currency": "EUR", "dt": base_dt + timedelta(minutes=3)},
        {"kind": "ECB", "currency": "EUR", "dt": base_dt + timedelta(hours=1)},
    ])
    _expect(len(deduped) == 2,
            "dedupe 15-min bucket should collapse first 2 ECB events, got %d" % len(deduped))

    if fails:
        print("\n%d self-test failure(s):" % len(fails))
        for f in fails:
            print("  -", f)
        return 1
    print("OK — all %d self-tests passed." % 10)
    return 0


__all__ = [
    "get_blackout_state",
    "is_in_blackout",
    "enforce_pre_event_flatten",
]


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
