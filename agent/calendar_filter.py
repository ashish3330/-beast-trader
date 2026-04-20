"""
Economic Calendar Filter — Forex Factory high-impact event filter.

Fetches high-impact events from Forex Factory (free JSON feed, no API key).
Caches for 4 hours. Warns before/after major news to avoid entries during
volatile event windows.

Usage:
    cal = CalendarFilter()
    skip, reason = cal.should_skip_entry("XAUUSD")
    if skip:
        print(f"Skip entry: {reason}")
"""
import time
import logging
import threading
from datetime import datetime, timezone, timedelta

log = logging.getLogger("dragon.calendar")

# ── Currency → symbol mapping ──
# Which symbols are affected by each currency's news events
CURRENCY_SYMBOL_MAP = {
    "USD": ["XAUUSD", "XAGUSD", "USDJPY", "USDCHF", "USDCAD", "NAS100.r", "BTCUSD"],
    "JPY": ["USDJPY", "EURJPY", "JPN225ft"],
    "EUR": ["EURJPY"],
    # GBP: no GBP symbols in Dragon
}

# ── Config ──
CACHE_TTL_S = 4 * 3600          # 4 hours
EVENT_WINDOW_MIN = 30           # skip entries within ±30 min of high-impact event
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FETCH_TIMEOUT_S = 10


class CalendarFilter:
    """Filters entries near high-impact economic events."""

    def __init__(self):
        self._events = []           # list of parsed event dicts
        self._cache_ts = 0.0        # last fetch timestamp
        self._lock = threading.Lock()
        self._fetch_error_count = 0
        log.info("CalendarFilter initialized (window=±%d min, cache=%dh)",
                 EVENT_WINDOW_MIN, CACHE_TTL_S // 3600)

    # ═══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════

    def should_skip_entry(self, symbol: str) -> tuple:
        """
        Check if a high-impact event is within ±30 minutes for this symbol.

        Returns:
            (bool, str): (should_skip, reason)
            - (True, "NFP in 15 min (USD)") if skip
            - (False, "") if safe to trade
        """
        try:
            self._maybe_refresh()
            now = datetime.now(timezone.utc)
            window = timedelta(minutes=EVENT_WINDOW_MIN)

            with self._lock:
                events = list(self._events)

            for ev in events:
                ev_time = ev.get("dt")
                if ev_time is None:
                    continue

                # Check if event is within ±window
                delta = ev_time - now
                abs_delta = abs(delta.total_seconds())
                if abs_delta > EVENT_WINDOW_MIN * 60:
                    continue

                # Check if this event's currency affects the symbol
                currency = ev.get("country", "")
                affected = CURRENCY_SYMBOL_MAP.get(currency, [])
                if symbol not in affected:
                    continue

                # Event is relevant and within window
                mins = int(delta.total_seconds() / 60)
                title = ev.get("title", "Unknown Event")
                if mins > 0:
                    reason = "%s in %d min (%s)" % (title, mins, currency)
                elif mins < 0:
                    reason = "%s was %d min ago (%s)" % (title, abs(mins), currency)
                else:
                    reason = "%s happening NOW (%s)" % (title, currency)

                log.info("[%s] NEWS SKIP: %s", symbol, reason)
                return (True, reason)

            return (False, "")

        except Exception as e:
            log.warning("CalendarFilter error: %s — allowing trade", e)
            return (False, "")

    # ═══════════════════════════════════════════════════════════════
    #  CACHE + FETCH
    # ═══════════════════════════════════════════════════════════════

    def _maybe_refresh(self):
        """Refresh cache if stale (>4 hours old)."""
        now = time.time()
        if now - self._cache_ts < CACHE_TTL_S:
            return

        try:
            self._fetch_events()
            self._cache_ts = now
            self._fetch_error_count = 0
        except Exception as e:
            self._fetch_error_count += 1
            log.warning("Calendar fetch failed (%d): %s", self._fetch_error_count, e)
            # On failure, extend cache TTL by 30 min so we don't hammer the API
            if self._cache_ts > 0:
                self._cache_ts = now - CACHE_TTL_S + 1800

    def _fetch_events(self):
        """Fetch and parse Forex Factory calendar JSON."""
        import urllib.request
        import json

        log.info("Fetching economic calendar from Forex Factory...")
        req = urllib.request.Request(
            FF_URL,
            headers={"User-Agent": "DragonTrader/2.0"},
        )
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if not isinstance(data, list):
            log.warning("Calendar data is not a list, got %s", type(data))
            return

        high_impact = []
        for item in data:
            impact = str(item.get("impact", "")).strip()
            if impact != "High":
                continue

            # Parse datetime
            dt = self._parse_event_time(item)
            if dt is None:
                continue

            high_impact.append({
                "title": str(item.get("title", "Unknown")),
                "country": str(item.get("country", "")),
                "dt": dt,
                "impact": impact,
                "forecast": item.get("forecast", ""),
                "previous": item.get("previous", ""),
            })

        with self._lock:
            self._events = high_impact

        log.info("Calendar loaded: %d high-impact events this week", len(high_impact))
        for ev in high_impact:
            log.debug("  %s | %s | %s", ev["dt"].strftime("%a %H:%M UTC"),
                       ev["country"], ev["title"])

    def _parse_event_time(self, item: dict):
        """
        Parse FF event date into UTC datetime.
        FF JSON returns ISO 8601: "2026-04-20T08:30:00-04:00"
        """
        date_str = str(item.get("date", "")).strip()
        if not date_str:
            return None

        try:
            dt = datetime.fromisoformat(date_str)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError) as e:
            log.debug("Failed to parse event time '%s': %s", date_str, e)
            return None

    # ═══════════════════════════════════════════════════════════════
    #  STATUS (for dashboard)
    # ═══════════════════════════════════════════════════════════════

    def get_status(self) -> dict:
        """Return status dict for dashboard display."""
        with self._lock:
            events = list(self._events)

        now = datetime.now(timezone.utc)
        upcoming = []
        for ev in events:
            dt = ev.get("dt")
            if dt and dt > now - timedelta(minutes=EVENT_WINDOW_MIN):
                delta_min = int((dt - now).total_seconds() / 60)
                upcoming.append({
                    "title": ev["title"],
                    "country": ev["country"],
                    "time_utc": dt.strftime("%H:%M"),
                    "delta_min": delta_min,
                })

        return {
            "cached": self._cache_ts > 0,
            "events_loaded": len(events),
            "upcoming_high_impact": upcoming[:10],
            "fetch_errors": self._fetch_error_count,
        }
