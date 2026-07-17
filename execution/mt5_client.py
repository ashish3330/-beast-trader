"""
ResilientMT5Client — self-healing rpyc/mt5linux wrapper.

Drop-in replacement for `mt5linux.MetaTrader5(host, port)`. Catches transport
errors (EOFError, ConnectionError, OSError, rpyc EOFError) on every call and
transparently reconnects with exponential backoff + jitter. Exposes the same
attribute API via __getattr__, so callsites do not change.

Failure modes handled:
  - rpyc stream closed (MT5 quit, bridge crashed, network blip)
  - rpyc syncreq timeout
  - login/initialize failure after reconnect

Circuit breaker:
  - After CB_TRIP_FAILURES consecutive transport failures within CB_WINDOW_S,
    method calls fast-fail with MT5Unavailable for CB_OPEN_S seconds. Brain
    cycle uses this to skip gracefully instead of blocking.

Telemetry:
  - Every reconnect logs {cause, downtime_ms, attempts}.
  - on_reconnect callback fires for dashboard/journal.
"""
import logging
import random
import threading
import time
from typing import Any, Callable, Optional

log = logging.getLogger("dragon.mt5_client")

# Transport-class errors that should trigger reconnect, not crash.
_TRANSPORT_ERRORS = (EOFError, ConnectionError, BrokenPipeError, OSError)
try:
    import rpyc.core.async_  # noqa
    _TRANSPORT_ERRORS = _TRANSPORT_ERRORS + (rpyc.core.async_.AsyncResultTimeout,)
except Exception:
    pass
try:
    from rpyc.core.protocol import PingError  # type: ignore
    _TRANSPORT_ERRORS = _TRANSPORT_ERRORS + (PingError,)
except Exception:
    pass


class MT5Unavailable(RuntimeError):
    """Raised when the circuit breaker is open — MT5 is degraded."""


class ResilientMT5Client:
    """Self-healing MT5 client.

    Reconnects transparently on transport errors. Retries up to RECONNECT_MAX
    attempts per call with exponential backoff (BACKOFF_BASE * 2^n + jitter,
    capped at BACKOFF_CAP). Trips a circuit breaker after CB_TRIP_FAILURES
    consecutive failures so the brain can skip cycles instead of stalling.
    """

    BACKOFF_BASE = 0.25
    BACKOFF_CAP = 8.0
    RECONNECT_MAX = 5

    CB_TRIP_FAILURES = 3
    CB_OPEN_S = 30.0          # how long the breaker stays open

    # 2026-07-17: trade-server WEDGE detector. Failure mode (2026-07-12): reads
    # keep succeeding but order_send returns None for hours — the MT5 trade
    # server dropped while the data path stayed alive. Previously a None result
    # was treated as SUCCESS (reset _consecutive_fails) so the breaker/reconnect
    # never tripped. Methods listed here are WRITE calls: a None result does NOT
    # reset the fail counter and is counted in its own consecutive-None streak.
    # At NONE_WRITE_TRIP consecutive Nones we force a session teardown so the
    # next call rebuilds initialize()+login() (the external MT5 relaunch is
    # handled by watchdog tier-2 / mt5-keeper via the trade_wedge flag).
    WRITE_METHODS = {"order_send"}
    NONE_WRITE_TRIP = 3

    CALL_TIMEOUT_S = 15.0     # max wall time for a single method call (incl. reconnects)
    # 2026-06-04: per-method timeout override. order_send / order_check on a
    # slow broker can take 60-90s legitimately (saw 70513ms on CHFJPY close
    # 2026-06-04 02:32). 15s wall-clock kept aborting valid closes that the
    # broker eventually executed — silent stale-position bug. Read by
    # _invoke() to pick the deadline.
    LONG_CALL_METHODS = {"order_send", "order_check"}
    LONG_CALL_TIMEOUT_S = 90.0   # generous: trade execution on slow broker

    def __init__(
        self,
        host: str,
        port: int,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        terminal_path: Optional[str] = None,
        on_reconnect: Optional[Callable[[dict], None]] = None,
    ):
        self._host = host
        self._port = port
        self._login = login
        self._password = password
        self._server = server
        self._terminal_path = terminal_path
        self._on_reconnect = on_reconnect

        self._lock = threading.RLock()
        self._inner = None              # underlying mt5linux.MetaTrader5
        self._consecutive_fails = 0
        self._consecutive_none_writes = 0  # order_send-None streak (trade wedge)
        self._cb_open_until = 0.0       # epoch seconds when breaker re-closes
        self._last_drop_ts = 0.0
        self._reconnect_count = 0

        # ── READ-CACHE (2026-07-12): the single rpyc bridge saturates when the
        # always-on loops hammer the same symbol (BTC positions_get ~10x/sym/cycle,
        # symbol_info, ticks) → calls time out → None → entries fail 100%. Cache the
        # hot reads to COLLAPSE redundant calls, and SERVE-LAST-GOOD on a failed read
        # so transient bridge failures are invisible to callers. Static specs cache
        # long; ticks/positions cache sub-second. positions cache is invalidated on
        # every order_send (we changed the book). This is the real bridge-robustness
        # fix (the disk-quote fallback was the band-aid).
        self._rcache = {}               # (method, key) -> (value, expiry_ts)
        self._cache_lock = threading.Lock()
        self._TTL = {"positions_get": 0.8, "symbol_info": 60.0, "symbol_info_tick": 0.4}

        self._connect()

    # ── public introspection ──────────────────────────────────────────────
    @property
    def is_healthy(self) -> bool:
        return self._inner is not None and time.time() >= self._cb_open_until

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def consecutive_none_writes(self) -> int:
        """Consecutive order_send calls that returned None (trade-wedge streak)."""
        return self._consecutive_none_writes

    # ── core: lazy method resolution ──────────────────────────────────────
    def __getattr__(self, name):
        # __getattr__ is only called when normal attribute lookup fails,
        # so this routes calls like `client.positions_get(...)` through
        # the resilient wrapper without intercepting our own attrs.
        if name.startswith("_"):
            raise AttributeError(name)

        def _call(*args, **kwargs):
            return self._invoke(name, args, kwargs)
        _call.__name__ = name
        return _call

    # ── CACHED HOT READS (defined methods shadow __getattr__) ──────────────
    def _cached_read(self, name, key, ttl, args, kwargs):
        now = time.time()
        ck = (name, key)
        with self._cache_lock:
            hit = self._rcache.get(ck)
            if hit and hit[1] > now:
                return hit[0]                       # fresh cache hit — no bridge call
        try:
            val = self._invoke(name, args, kwargs)
        except Exception:
            val = None
        if val is not None:
            with self._cache_lock:
                self._rcache[ck] = (val, now + ttl)
            return val
        # FAILED read → serve the last good value (may be slightly stale, but far
        # better than None, which breaks orders/loops). Specs never change; a
        # sub-second-stale tick/positions is harmless for the read-heavy loops.
        with self._cache_lock:
            hit = self._rcache.get(ck)
        return hit[0] if hit else None

    def positions_get(self, *args, **kwargs):
        key = kwargs.get("symbol") or (args[0] if args else "__ALL__")
        return self._cached_read("positions_get", key, self._TTL["positions_get"], args, kwargs)

    def symbol_info(self, symbol, *args, **kwargs):
        return self._cached_read("symbol_info", symbol, self._TTL["symbol_info"], (symbol,) + args, kwargs)

    def symbol_info_tick(self, symbol, *args, **kwargs):
        return self._cached_read("symbol_info_tick", symbol, self._TTL["symbol_info_tick"], (symbol,) + args, kwargs)

    def order_send(self, *args, **kwargs):
        r = self._invoke("order_send", args, kwargs)
        with self._cache_lock:                      # book changed → drop positions cache
            self._rcache = {k: v for k, v in self._rcache.items() if k[0] != "positions_get"}
        return r

    # ── connection management ─────────────────────────────────────────────
    # rpyc sync_request_timeout for the underlying MetaTrader5 client.
    # History: default 300s let a killed bridge hang the trader 10min (chaos
    # test 2026-05-11) → tightened to 10s. But the wine bridge runs at ~137%
    # CPU and slow rpyc calls under load (positions_get, copy_rates) routinely
    # exceeded 10s → "result expired" → forced reconnect → 633/7d reconnects
    # (2026-05-30 audit). Raised to 30s: false-positive timeouts collapse, and
    # a genuine hung bridge is still caught quickly by the mt5-keeper backstop
    # (~75s recovery SLO). Don't go back to 300s — that's the hang risk.
    RPYC_SYNC_TIMEOUT_S = 30

    def _connect(self) -> bool:
        """Build a fresh mt5linux client. Returns True on success."""
        from mt5linux import MetaTrader5
        try:
            inner = MetaTrader5(host=self._host, port=self._port,
                                 timeout=self.RPYC_SYNC_TIMEOUT_S)
            # initialize() + login() are required for a usable session.
            if self._terminal_path:
                ok = inner.initialize(path=self._terminal_path)
            else:
                ok = inner.initialize()
            if not ok:
                log.error("MT5 initialize() failed during reconnect")
                return False
            if self._login is not None:
                if not inner.login(self._login, password=self._password, server=self._server):
                    err = None
                    try:
                        err = inner.last_error()
                    except Exception:
                        pass
                    log.error("MT5 login failed during reconnect: %s", err)
                    return False
            self._inner = inner
            return True
        except Exception as e:
            log.warning("MT5 connect attempt failed: %s", e)
            return False

    def _reconnect_with_backoff(self, cause: str) -> bool:
        """Tear down + rebuild with exponential backoff. Returns True on success."""
        self._last_drop_ts = self._last_drop_ts or time.time()
        self._inner = None
        for attempt in range(1, self.RECONNECT_MAX + 1):
            delay = min(self.BACKOFF_CAP, self.BACKOFF_BASE * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)  # jitter
            time.sleep(delay)
            log.info("MT5 reconnect attempt %d/%d (cause=%s)", attempt, self.RECONNECT_MAX, cause)
            if self._connect():
                # guard: _last_drop_ts is reset to 0 after a reconnect; without a
                # re-stamp, time.time()-0 prints a ~56yr epoch duration. 0 if unknown.
                downtime_ms = (int((time.time() - self._last_drop_ts) * 1000)
                               if self._last_drop_ts else 0)
                self._reconnect_count += 1
                evt = {
                    "ts": time.time(),
                    "cause": cause,
                    "downtime_ms": downtime_ms,
                    "attempts": attempt,
                }
                log.warning("MT5 RECONNECTED after %dms (cause=%s, attempts=%d, total_reconnects=%d)",
                            downtime_ms, cause, attempt, self._reconnect_count)
                self._last_drop_ts = 0.0
                if self._on_reconnect:
                    try:
                        self._on_reconnect(evt)
                    except Exception as e:
                        log.warning("on_reconnect callback failed: %s", e)
                return True
        return False

    # ── circuit breaker ───────────────────────────────────────────────────
    def _trip_breaker(self):
        self._cb_open_until = time.time() + self.CB_OPEN_S
        log.error("MT5 circuit breaker OPEN for %.0fs after %d consecutive failures",
                  self.CB_OPEN_S, self._consecutive_fails)

    # ── invocation core ───────────────────────────────────────────────────
    def _invoke(self, method_name: str, args: tuple, kwargs: dict) -> Any:
        # Fast-fail when breaker is open.
        now = time.time()
        if now < self._cb_open_until:
            raise MT5Unavailable(
                f"MT5 circuit breaker open for {self._cb_open_until - now:.1f}s more"
            )

        # 2026-06-04: long-call methods (order_send) get a generous deadline
        # so a slow broker doesn't abort a valid trade execution.
        _timeout = (self.LONG_CALL_TIMEOUT_S if method_name in self.LONG_CALL_METHODS
                    else self.CALL_TIMEOUT_S)
        deadline = now + _timeout
        last_err: Optional[BaseException] = None

        # First attempt — direct call.
        for round_idx in range(2):  # round 0: original, round 1: post-reconnect
            if self._inner is None:
                if not self._reconnect_with_backoff(cause="not_connected"):
                    self._consecutive_fails += 1
                    if self._consecutive_fails >= self.CB_TRIP_FAILURES:
                        self._trip_breaker()
                    raise MT5Unavailable("MT5 reconnect exhausted")

            try:
                with self._lock:
                    fn = getattr(self._inner, method_name)
                    result = fn(*args, **kwargs)
                # 2026-07-17 WEDGE FIX: a None result from a WRITE method
                # (order_send) is NOT success — it's the reads-OK/writes-None
                # trade-server wedge. Do NOT reset the fail counter for it,
                # and count the consecutive-None streak separately.
                if method_name in self.WRITE_METHODS and result is None:
                    self._consecutive_none_writes += 1
                    log.error(
                        "MT5.%s returned None (consecutive_none_writes=%d/%d) "
                        "— possible trade-server wedge",
                        method_name, self._consecutive_none_writes,
                        self.NONE_WRITE_TRIP,
                    )
                    if self._consecutive_none_writes >= self.NONE_WRITE_TRIP:
                        log.error(
                            "MT5 TRADE-PATH WEDGE suspected after %d consecutive "
                            "None writes — forcing session teardown/reconnect",
                            self._consecutive_none_writes,
                        )
                        # Force a full rebuild (initialize+login) on next call.
                        # Reset the streak so we don't reconnect-storm; if the
                        # wedge persists it re-trips after NONE_WRITE_TRIP more.
                        self._inner = None
                        self._consecutive_none_writes = 0
                    return result
                # genuine success — clear failure counters
                self._consecutive_fails = 0
                if method_name in self.WRITE_METHODS:
                    self._consecutive_none_writes = 0
                return result
            except _TRANSPORT_ERRORS as e:
                last_err = e
                cause = type(e).__name__
                log.warning("MT5.%s transport error: %s — will reconnect", method_name, e)
                if time.time() >= deadline:
                    break
                if not self._reconnect_with_backoff(cause=cause):
                    break
                # loop continues for the post-reconnect retry
            except Exception as e:
                # Non-transport exception — let caller handle (real domain errors).
                self._consecutive_fails = 0
                raise

        # All retries exhausted.
        self._consecutive_fails += 1
        if self._consecutive_fails >= self.CB_TRIP_FAILURES:
            self._trip_breaker()
        raise MT5Unavailable(
            f"MT5.{method_name} failed after reconnect attempts: {last_err}"
        )
