"""
Unit tests for ResilientMT5Client.

Strategy: replace `mt5linux.MetaTrader5` with a fake that we can program to
fail on demand, then verify the facade reconnects, surfaces success after
recovery, opens its circuit breaker on persistent failure, and re-closes
it after the cooldown.

Run: cd beast-trader && python3 -B -m pytest tests/test_mt5_client.py -v
"""
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from execution.mt5_client import ResilientMT5Client, MT5Unavailable


class FakeMT5:
    """Programmable fake mt5linux.MetaTrader5 client."""

    def __init__(self, host=None, port=None):
        self.host = host
        self.port = port
        self.calls = []
        self._failures_remaining = 0      # account_info() raises EOFError this many times
        self._fail_class = EOFError("stream has been closed")
        self._init_returns = True
        self._login_returns = True

    # ── mt5linux API surface used by ResilientMT5Client._connect() ──
    def initialize(self, path=None):
        self.calls.append(("initialize", path))
        return self._init_returns

    def login(self, login, password=None, server=None):
        self.calls.append(("login", login, password, server))
        return self._login_returns

    def last_error(self):
        return (0, "ok")

    # ── methods invoked through the facade ──
    def account_info(self):
        self.calls.append(("account_info",))
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise self._fail_class
        return types.SimpleNamespace(name="test", balance=1000.0, equity=1000.0)

    def positions_get(self, symbol=None):
        self.calls.append(("positions_get", symbol))
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise self._fail_class
        return ()


@pytest.fixture
def fake_factory():
    """A factory that returns one FakeMT5 per call (so reconnect = new instance)."""
    instances = []

    def make(host=None, port=None):
        f = FakeMT5(host=host, port=port)
        instances.append(f)
        return f

    return make, instances


def test_clean_call_no_reconnect(fake_factory):
    make, instances = fake_factory
    with patch("mt5linux.MetaTrader5", side_effect=make):
        c = ResilientMT5Client(host="x", port=1, login=1, password="p", server="s")
        info = c.account_info()
    assert info.balance == 1000.0
    assert c.reconnect_count == 0
    # One initial connect → one FakeMT5 instance
    assert len(instances) == 1


def test_transient_drop_reconnects(fake_factory):
    make, instances = fake_factory
    with patch("mt5linux.MetaTrader5", side_effect=make):
        c = ResilientMT5Client(host="x", port=1, login=1, password="p", server="s")
        # Make the first instance fail twice on the next call.
        instances[0]._failures_remaining = 2
        # Speed up backoff for the test
        c.BACKOFF_BASE = 0.01
        c.BACKOFF_CAP = 0.05
        info = c.account_info()
    assert info.balance == 1000.0
    # Should have reconnected at least once
    assert c.reconnect_count >= 1


def test_circuit_breaker_opens_on_persistent_failure(fake_factory):
    make, instances = fake_factory
    with patch("mt5linux.MetaTrader5", side_effect=make):
        c = ResilientMT5Client(host="x", port=1, login=1, password="p", server="s")
        c.BACKOFF_BASE = 0.001
        c.BACKOFF_CAP = 0.002
        c.RECONNECT_MAX = 2
        c.CB_TRIP_FAILURES = 2
        c.CB_OPEN_S = 0.5

        # The instance constructed in __init__ also needs to be unhealthy,
        # otherwise the first call succeeds against it before reconnect is needed.
        instances[0]._failures_remaining = 999

        # Make every new FakeMT5 instance fail forever after construct.
        original_make = make

        def make_failing(host=None, port=None):
            f = original_make(host=host, port=port)
            f._failures_remaining = 999
            return f

        with patch("mt5linux.MetaTrader5", side_effect=make_failing):
            # First call exhausts retries → consecutive_fails increments
            with pytest.raises(MT5Unavailable):
                c.account_info()
            # Second call: CB still under threshold? Force enough fails.
            with pytest.raises(MT5Unavailable):
                c.account_info()

            # Now CB should be open: subsequent calls fast-fail
            t0 = time.time()
            with pytest.raises(MT5Unavailable) as excinfo:
                c.account_info()
            # Fast-fail means no reconnect attempts → near-instant
            assert (time.time() - t0) < 0.1
            assert "circuit breaker" in str(excinfo.value).lower()


def test_circuit_breaker_recloses_after_cooldown(fake_factory):
    make, instances = fake_factory
    with patch("mt5linux.MetaTrader5", side_effect=make):
        c = ResilientMT5Client(host="x", port=1, login=1, password="p", server="s")
        c.BACKOFF_BASE = 0.001
        c.RECONNECT_MAX = 1
        c.CB_TRIP_FAILURES = 1
        c.CB_OPEN_S = 0.2
        # Trip the breaker
        c._trip_breaker()
        c._consecutive_fails = c.CB_TRIP_FAILURES
        with pytest.raises(MT5Unavailable):
            c.account_info()
        time.sleep(0.25)  # past cooldown
        # Should now retry; instances[0] is healthy → success
        info = c.account_info()
        assert info.balance == 1000.0


def test_non_transport_exception_bubbles_up(fake_factory):
    make, instances = fake_factory
    with patch("mt5linux.MetaTrader5", side_effect=make):
        c = ResilientMT5Client(host="x", port=1, login=1, password="p", server="s")
        # Real domain error — facade must NOT eat it.
        instances[0]._fail_class = ValueError("bad symbol")
        instances[0]._failures_remaining = 1
        with pytest.raises(ValueError):
            c.account_info()
        # No reconnect should have been attempted for ValueError.
        assert c.reconnect_count == 0
