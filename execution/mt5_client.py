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

    CALL_TIMEOUT_S = 15.0     # max wall time for a single method call (incl. reconnects)

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
        self._cb_open_until = 0.0       # epoch seconds when breaker re-closes
        self._last_drop_ts = 0.0
        self._reconnect_count = 0

        self._connect()

    # ── public introspection ──────────────────────────────────────────────
    @property
    def is_healthy(self) -> bool:
        return self._inner is not None and time.time() >= self._cb_open_until

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

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

    # ── connection management ─────────────────────────────────────────────
    # rpyc sync_request_timeout for the underlying MetaTrader5 client. Without
    # this the default is 300s — a clean `kill -9` of the bridge leaves the
    # rpyc socket in half-open state and the trader hung for ~10min before
    # the watchdog noticed (chaos test 2026-05-11). 10s gives a transient hiccup
    # plenty of time to recover while keeping the failure mode bounded.
    RPYC_SYNC_TIMEOUT_S = 10

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
                downtime_ms = int((time.time() - self._last_drop_ts) * 1000)
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

        deadline = now + self.CALL_TIMEOUT_S
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
                # success — clear failure counter
                self._consecutive_fails = 0
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
