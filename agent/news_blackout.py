"""
news_blackout.py — Tier-1 economic event blackout windows.

Provides `is_in_blackout(symbol, now_utc=None) -> bool` that returns True if the
current UTC time falls within ±N minutes of a tier-1 economic event whose
currency intersects the symbol's affected-currency set.

Self-contained: no external API calls; schedule is hardcoded and deterministic.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Window configuration (minutes before/after the scheduled event time)
# -----------------------------------------------------------------------------
try:
    from config import NEWS_BLACKOUT_MIN_BEFORE, NEWS_BLACKOUT_MIN_AFTER  # type: ignore
except ImportError:
    NEWS_BLACKOUT_MIN_BEFORE = 5
    NEWS_BLACKOUT_MIN_AFTER = 15

# -----------------------------------------------------------------------------
# Symbol -> affected currencies
# -----------------------------------------------------------------------------
SYMBOL_CURRENCIES: dict[str, set[str]] = {
    # USD-affected
    "DJ30.r": {"USD"}, "SP500.r": {"USD"}, "NAS100.r": {"USD"}, "US2000.r": {"USD"},
    "XAUUSD": {"USD"}, "XAGUSD": {"USD"}, "USOUSD": {"USD"}, "BTCUSD": {"USD"}, "ETHUSD": {"USD"},
    "EURUSD": {"USD", "EUR"}, "GBPUSD": {"USD", "GBP"},
    "USDJPY": {"USD", "JPY"}, "USDCAD": {"USD", "CAD"}, "USDCHF": {"USD", "CHF"},
    "AUDUSD": {"USD", "AUD"}, "NZDUSD": {"USD", "NZD"},
    # JPY pairs
    "AUDJPY": {"JPY"}, "CHFJPY": {"JPY"}, "GBPJPY": {"GBP", "JPY"},
    "CADJPY": {"CAD", "JPY"}, "EURJPY": {"EUR", "JPY"},
    "JPN225ft": {"JPY"},
    # EU
    "GER40.r": {"EUR"}, "EURAUD": {"EUR", "AUD"}, "EURGBP": {"EUR", "GBP"},
    # UK
    "UK100.r": {"GBP"},
    # Asia
    "SPI200.r": {"AUD"}, "HK50.r": {"USD"}, "SWI20.r": {"CHF"},
}

EVENT_CURRENCY_MAP: dict[str, str] = {
    "NFP": "USD", "US_CPI": "USD", "US_PPI": "USD", "FOMC": "USD",
    "ECB": "EUR", "BOE": "GBP", "BOJ": "JPY",
}

# -----------------------------------------------------------------------------
# Fixed-date schedules (placeholder 2026 calendar)
# -----------------------------------------------------------------------------
FOMC_DATES_2026: list[date] = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 10),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 11, 4), date(2026, 12, 16),
]
FOMC_TIME_UTC = time(18, 0)

ECB_DATES_2026: list[date] = [
    date(2026, 1, 22), date(2026, 3, 12), date(2026, 4, 16), date(2026, 6, 4),
    date(2026, 7, 23), date(2026, 9, 10), date(2026, 10, 29), date(2026, 12, 17),
]
ECB_DECISION_TIME_UTC = time(12, 15)
ECB_PRESSER_TIME_UTC = time(12, 45)

BOJ_DATES_2026: list[date] = [
    date(2026, 1, 23), date(2026, 3, 19), date(2026, 4, 28), date(2026, 6, 19),
    date(2026, 7, 31), date(2026, 9, 18), date(2026, 10, 30), date(2026, 12, 18),
]
# BOJ release timing is uncertain (03:00-05:00 UTC); use a window-centre of 04:00.
BOJ_TIME_UTC = time(4, 0)

BOE_DATES_2026: list[date] = [
    date(2026, 2, 5), date(2026, 3, 19), date(2026, 5, 7), date(2026, 6, 18),
    date(2026, 8, 6), date(2026, 9, 17), date(2026, 11, 5), date(2026, 12, 17),
]
BOE_TIME_UTC = time(11, 0)

# Monthly US data release time (08:30 ET == 12:30 UTC, ignoring DST nuance)
US_DATA_TIME_UTC = time(12, 30)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _nth_weekday_of_month(year: int, month: int, weekday_int: int, n: int) -> date:
    """Return the date of the Nth occurrence of `weekday_int` (Mon=0..Sun=6) in
    the given month. Raises ValueError if month doesn't contain that occurrence."""
    cal = calendar.Calendar()
    matches = [d for d in cal.itermonthdates(year, month)
               if d.month == month and d.weekday() == weekday_int]
    if n < 1 or n > len(matches):
        raise ValueError(f"Month {year}-{month:02d} has no {n}th weekday {weekday_int}")
    return matches[n - 1]


def _first_friday_of_month(year: int, month: int) -> date:
    return _nth_weekday_of_month(year, month, weekday_int=calendar.FRIDAY, n=1)


def _combine_utc(d: date, t: time) -> datetime:
    return datetime.combine(d, t).replace(tzinfo=timezone.utc)


def _next_event_times(now_utc: datetime, schedule_type: str) -> list[datetime]:
    """Return scheduled event datetimes for `schedule_type` within ±1 day of now."""
    out: list[datetime] = []
    candidate_dates: list[tuple[date, time]] = []

    # Generate candidate (date, time) pairs from yesterday through tomorrow.
    today = now_utc.date()
    window_dates = [today - timedelta(days=1), today, today + timedelta(days=1)]

    if schedule_type == "NFP":
        for d in window_dates:
            try:
                ff = _first_friday_of_month(d.year, d.month)
            except ValueError:
                continue
            if ff == d:
                candidate_dates.append((ff, US_DATA_TIME_UTC))
    elif schedule_type == "US_CPI":
        for d in window_dates:
            try:
                second_tue = _nth_weekday_of_month(d.year, d.month, calendar.TUESDAY, 2)
            except ValueError:
                continue
            if second_tue == d:
                candidate_dates.append((second_tue, US_DATA_TIME_UTC))
    elif schedule_type == "US_PPI":
        for d in window_dates:
            try:
                second_wed = _nth_weekday_of_month(d.year, d.month, calendar.WEDNESDAY, 2)
            except ValueError:
                continue
            if second_wed == d:
                candidate_dates.append((second_wed, US_DATA_TIME_UTC))
    elif schedule_type == "FOMC":
        for d in window_dates:
            if d in FOMC_DATES_2026:
                candidate_dates.append((d, FOMC_TIME_UTC))
    elif schedule_type == "ECB":
        for d in window_dates:
            if d in ECB_DATES_2026:
                candidate_dates.append((d, ECB_DECISION_TIME_UTC))
                candidate_dates.append((d, ECB_PRESSER_TIME_UTC))
    elif schedule_type == "BOJ":
        for d in window_dates:
            if d in BOJ_DATES_2026:
                candidate_dates.append((d, BOJ_TIME_UTC))
    elif schedule_type == "BOE":
        for d in window_dates:
            if d in BOE_DATES_2026:
                candidate_dates.append((d, BOE_TIME_UTC))
    else:
        return []

    for d, t in candidate_dates:
        out.append(_combine_utc(d, t))
    return out


def _in_window(now_utc: datetime, event_dt: datetime) -> bool:
    start = event_dt - timedelta(minutes=NEWS_BLACKOUT_MIN_BEFORE)
    end = event_dt + timedelta(minutes=NEWS_BLACKOUT_MIN_AFTER)
    return start <= now_utc <= end


_ALL_SCHEDULES: tuple[str, ...] = ("NFP", "US_CPI", "US_PPI", "FOMC", "ECB", "BOJ", "BOE")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def is_in_blackout(symbol: str, now_utc: datetime | None = None) -> bool:
    """Return True if `now_utc` is within the configured window around a tier-1
    event whose currency affects `symbol`. Fail-open: unknown symbols return False.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    affected = SYMBOL_CURRENCIES.get(symbol)
    if not affected:
        return False

    for sched in _ALL_SCHEDULES:
        ev_ccy = EVENT_CURRENCY_MAP.get(sched)
        if not ev_ccy or ev_ccy not in affected:
            continue
        for ev_dt in _next_event_times(now_utc, sched):
            if _in_window(now_utc, ev_dt):
                log.debug("blackout HIT symbol=%s event=%s at %s (now=%s)",
                          symbol, sched, ev_dt.isoformat(), now_utc.isoformat())
                return True
    return False


# -----------------------------------------------------------------------------
# 2026-06-18 Tier 1 #5: Next-tier-1-event lookup for pre-flatten banner.
# -----------------------------------------------------------------------------
def next_tier1_event(now_utc: datetime | None = None,
                     lookahead_minutes: float = 120.0) -> dict | None:
    """Return the soonest tier-1 event in the next `lookahead_minutes`, or None.

    Used by the news_pre_flatten banner (item #5). Fail-open: any error
    returns None so the banner is just silent — never blocks trading.

    Returns
    -------
    dict | None
        {"event": str, "time_utc": datetime, "minutes_until": float}
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    try:
        soonest: tuple[float, datetime, str] | None = None
        for sched in _ALL_SCHEDULES:
            for ev_dt in _next_event_times(now_utc, sched):
                delta_min = (ev_dt - now_utc).total_seconds() / 60.0
                if delta_min < 0:
                    continue
                if delta_min > lookahead_minutes:
                    continue
                if soonest is None or delta_min < soonest[0]:
                    soonest = (delta_min, ev_dt, sched)
        if soonest is None:
            return None
        return {
            "event": soonest[2],
            "time_utc": soonest[1],
            "minutes_until": float(soonest[0]),
        }
    except Exception as e:  # pragma: no cover - defensive
        log.debug("next_tier1_event failed (fail-open): %s", e)
        return None


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime, timezone
    test_cases = [
        ("EURUSD", datetime(2026, 6, 4, 12, 20, tzinfo=timezone.utc)),  # near ECB
        ("DJ30.r", datetime(2026, 6, 5, 12, 30, tzinfo=timezone.utc)),  # NFP day (first Fri June)
        ("BTCUSD", datetime(2026, 6, 6, 8, 0, tzinfo=timezone.utc)),    # quiet
    ]
    for sym, t in test_cases:
        print(f"{sym} @ {t}: blackout={is_in_blackout(sym, t)}")
