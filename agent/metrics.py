"""
agent/metrics.py — Prometheus exporter for Dragon Trader.

Opt-in via env METRICS_ENABLED=1 (default off). Port: METRICS_PORT (default 9090).

Metric inventory:
    dragon_decision_loop_seconds        Histogram
    dragon_positions_open               Gauge      [symbol]
    dragon_balance_usd                  Gauge
    dragon_equity_usd                   Gauge
    dragon_drawdown_pct                 Gauge
    dragon_trade_count_total            Counter    [symbol, side, outcome]
    dragon_pnl_usd_total                Counter    [symbol]
    dragon_mt5_connected                Gauge
    dragon_ml_auc                       Gauge      [symbol]

The MetricsExporter is observability-only. start() is a no-op when prometheus_client
is not installed or METRICS_ENABLED is unset, so the trading loop never blocks
on telemetry.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("dragon.metrics")

# Soft-import: do not crash if prometheus_client missing (e.g. fresh checkout).
try:
    from prometheus_client import (
        CollectorRegistry, Counter, Gauge, Histogram, start_http_server,
    )
    _PROM_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    _PROM_AVAILABLE = False
    log.info("prometheus_client unavailable (%s) — metrics disabled", _e)


# Histogram buckets tuned to typical decision-loop wall time (seconds).
_LOOP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class _NullMetric:
    """No-op metric used when prometheus is disabled. Supports labels()/inc/set/observe."""
    def labels(self, *_a, **_kw):
        return self
    def inc(self, *_a, **_kw):
        return None
    def set(self, *_a, **_kw):
        return None
    def observe(self, *_a, **_kw):
        return None
    def time(self):
        class _Ctx:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *exc):
                return False
        return _Ctx()


class MetricsExporter:
    """Holds metric handles and starts the Prometheus HTTP server.

    Usage:
        m = MetricsExporter()         # reads env, registers metrics
        m.start()                     # serves /metrics on METRICS_PORT (or no-op)
        with m.decision_loop_timer(): ...
        m.set_balance(1000.0)
    """

    def __init__(
        self,
        port: Optional[int] = None,
        enabled: Optional[bool] = None,
    ):
        env_enabled = os.environ.get("METRICS_ENABLED", "0").strip() in ("1", "true", "TRUE", "yes")
        self.enabled = bool(enabled) if enabled is not None else env_enabled
        try:
            self.port = int(port if port is not None else os.environ.get("METRICS_PORT", "9090"))
        except ValueError:
            self.port = 9090

        self.registry = None
        self._started = False

        if not (self.enabled and _PROM_AVAILABLE):
            # All metric handles become no-ops.
            self.decision_loop_seconds = _NullMetric()
            self.positions_open = _NullMetric()
            self.balance_usd = _NullMetric()
            self.equity_usd = _NullMetric()
            self.drawdown_pct = _NullMetric()
            self.trade_count_total = _NullMetric()
            self.pnl_usd_total = _NullMetric()
            self.mt5_connected = _NullMetric()
            self.ml_auc = _NullMetric()
            return

        self.registry = CollectorRegistry()
        self.decision_loop_seconds = Histogram(
            "dragon_decision_loop_seconds",
            "Wall time of each decision loop iteration (seconds).",
            buckets=_LOOP_BUCKETS,
            registry=self.registry,
        )
        self.positions_open = Gauge(
            "dragon_positions_open",
            "Number of open positions (or sub-positions) by symbol.",
            ["symbol"],
            registry=self.registry,
        )
        self.balance_usd = Gauge(
            "dragon_balance_usd",
            "Account balance in USD.",
            registry=self.registry,
        )
        self.equity_usd = Gauge(
            "dragon_equity_usd",
            "Account equity in USD (balance + floating PnL).",
            registry=self.registry,
        )
        self.drawdown_pct = Gauge(
            "dragon_drawdown_pct",
            "Current drawdown from peak equity, percent.",
            registry=self.registry,
        )
        self.trade_count_total = Counter(
            "dragon_trade_count_total",
            "Total executed trades.",
            ["symbol", "side", "outcome"],
            registry=self.registry,
        )
        self.pnl_usd_total = Counter(
            "dragon_pnl_usd_total",
            "Cumulative realized PnL in USD by symbol (negative PnL not counted; "
            "use a Gauge sum query for net).",
            ["symbol"],
            registry=self.registry,
        )
        self.mt5_connected = Gauge(
            "dragon_mt5_connected",
            "1 if MT5 bridge connection is healthy, 0 otherwise.",
            registry=self.registry,
        )
        self.ml_auc = Gauge(
            "dragon_ml_auc",
            "Current ML meta-label AUC by symbol.",
            ["symbol"],
            registry=self.registry,
        )

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if not (self.enabled and _PROM_AVAILABLE):
            log.info("metrics exporter not started (enabled=%s prom=%s)",
                     self.enabled, _PROM_AVAILABLE)
            return
        if self._started:
            return
        try:
            start_http_server(self.port, registry=self.registry)
            self._started = True
            log.info("Prometheus metrics on :%d/metrics", self.port)
        except OSError as e:
            log.warning("metrics server bind failed on port %d: %s", self.port, e)
        except Exception as e:  # noqa: BLE001
            log.warning("metrics server start error: %s", e)

    # ── convenience helpers ──────────────────────────────────────────────
    def decision_loop_timer(self):
        """Context manager wrapping decision loop timing.

        Usage:
            with metrics.decision_loop_timer():
                run_one_loop()
        """
        return self.decision_loop_seconds.time()

    def set_positions_open(self, symbol: str, count: int) -> None:
        self.positions_open.labels(symbol=symbol).set(float(count))

    def set_balance(self, balance: float) -> None:
        self.balance_usd.set(float(balance))

    def set_equity(self, equity: float) -> None:
        self.equity_usd.set(float(equity))

    def set_drawdown_pct(self, pct: float) -> None:
        self.drawdown_pct.set(float(pct))

    def record_trade(self, symbol: str, side: str, outcome: str, pnl: float = 0.0) -> None:
        """outcome should be one of: WIN, LOSS, BE, UNKNOWN."""
        self.trade_count_total.labels(
            symbol=symbol, side=side, outcome=outcome,
        ).inc()
        if pnl > 0:
            self.pnl_usd_total.labels(symbol=symbol).inc(float(pnl))

    def set_mt5_connected(self, connected: bool) -> None:
        self.mt5_connected.set(1.0 if connected else 0.0)

    def set_ml_auc(self, symbol: str, auc: float) -> None:
        self.ml_auc.labels(symbol=symbol).set(float(auc))


# Module-level singleton helper (mirrors alerting.get_default_alerter).
_default: Optional[MetricsExporter] = None


def get_default_metrics() -> MetricsExporter:
    global _default
    if _default is None:
        _default = MetricsExporter()
    return _default
