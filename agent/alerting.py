"""
agent/alerting.py — Pluggable alerting framework for Dragon Trader.

Goals:
- Observability-only. NEVER blocks any trading logic. Every public method swallows
  backend errors and returns immediately. (Per memory: "Never skip trades — warn only".)
- Backends are opt-in via env vars:
    TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID  -> TelegramBackend
    SLACK_WEBHOOK_URL                       -> SlackBackend
    LogBackend is always on, tagged with the [ALERT] prefix.
- Async-safe: all backend dispatch happens on a daemon worker thread, the public
  methods only enqueue a structured event. The decision loop never waits on I/O.

Public API:
    a = Alerter()
    a.position_open(symbol, side, size, entry_price)
    a.position_close(symbol, side, pnl, r_multiple, exit_reason)
    a.dd_breach(level, pct)
    a.connection_lost(component, duration_secs)
    a.drift_detected(symbol, auc_before, auc_now)
    a.error(component, message, context_dict=None)

Smoke test:
    python3 -B -c "from agent.alerting import Alerter; a = Alerter(); \\
        a.position_open('XAUUSD','LONG',0.01,2300.0)"
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib import request as urlrequest
from urllib.error import URLError

log = logging.getLogger("dragon.alerting")

# ──────────────────────────────────────────────────────────────────────────────
# Event model
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AlertEvent:
    """Structured alert payload. `kind` selects the formatter; `fields` carries data."""

    kind: str                       # e.g. "position_open", "dd_breach", "error"
    fields: Dict[str, Any] = field(default_factory=dict)
    severity: str = "INFO"          # INFO | WARN | ERROR
    ts: float = field(default_factory=time.time)

    def title(self) -> str:
        return f"[{self.severity}] {self.kind}"

    def text(self) -> str:
        """One-line human-readable summary."""
        f = self.fields
        k = self.kind
        if k == "position_open":
            return (f"OPEN {f.get('symbol')} {f.get('side')} "
                    f"size={f.get('size')} entry={f.get('entry_price')}")
        if k == "position_close":
            return (f"CLOSE {f.get('symbol')} {f.get('side')} "
                    f"pnl={f.get('pnl')} R={f.get('r_multiple')} "
                    f"reason={f.get('exit_reason')}")
        if k == "dd_breach":
            return f"DD BREACH level={f.get('level')} dd={f.get('pct')}%"
        if k == "connection_lost":
            return (f"CONNECTION LOST {f.get('component')} "
                    f"duration={f.get('duration_secs')}s")
        if k == "drift_detected":
            return (f"DRIFT {f.get('symbol')} "
                    f"AUC {f.get('auc_before')} -> {f.get('auc_now')}")
        if k == "error":
            ctx = f.get('context') or {}
            ctx_s = (" " + json.dumps(ctx, default=str)) if ctx else ""
            return f"ERROR [{f.get('component')}] {f.get('message')}{ctx_s}"
        # Fallback
        return f"{k} {json.dumps(f, default=str)}"


# ──────────────────────────────────────────────────────────────────────────────
# Backends
# ──────────────────────────────────────────────────────────────────────────────


class _Backend:
    name = "base"

    def send(self, ev: AlertEvent) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LogBackend(_Backend):
    """Always-on backend that just logs. Tagged with [ALERT]."""
    name = "log"

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._log = logger or log

    def send(self, ev: AlertEvent) -> None:
        line = f"[ALERT] {ev.title()} :: {ev.text()}"
        sev = ev.severity.upper()
        if sev == "ERROR":
            self._log.error(line)
        elif sev == "WARN":
            self._log.warning(line)
        else:
            self._log.info(line)


class TelegramBackend(_Backend):
    """Telegram bot sendMessage. Activated when both env vars are present."""
    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 5.0):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send(self, ev: AlertEvent) -> None:
        payload = json.dumps({
            "chat_id": self.chat_id,
            "text": f"{ev.title()}\n{ev.text()}",
            "disable_web_page_preview": True,
        }).encode("utf-8")
        req = urlrequest.Request(
            self._url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                _ = resp.read()
        except URLError as e:
            log.warning("[ALERT] telegram send failed: %s", e)
        except Exception as e:  # noqa: BLE001 - never propagate
            log.warning("[ALERT] telegram send error: %s", e)


class SlackBackend(_Backend):
    """Slack incoming webhook. Activated when SLACK_WEBHOOK_URL is present."""
    name = "slack"

    def __init__(self, webhook_url: str, timeout: float = 5.0):
        self.webhook_url = webhook_url
        self.timeout = timeout

    def send(self, ev: AlertEvent) -> None:
        payload = json.dumps({
            "text": f"*{ev.title()}*\n{ev.text()}",
        }).encode("utf-8")
        req = urlrequest.Request(
            self.webhook_url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                _ = resp.read()
        except URLError as e:
            log.warning("[ALERT] slack send failed: %s", e)
        except Exception as e:  # noqa: BLE001 - never propagate
            log.warning("[ALERT] slack send error: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# Alerter
# ──────────────────────────────────────────────────────────────────────────────


class Alerter:
    """Public facade. Always returns immediately; dispatches on a worker thread.

    All public methods catch every exception so a misconfigured backend can
    never block or crash the decision loop.
    """

    def __init__(
        self,
        backends: Optional[List[_Backend]] = None,
        max_queue: int = 1024,
    ):
        self._q: "queue.Queue[Optional[AlertEvent]]" = queue.Queue(maxsize=max_queue)
        self._backends: List[_Backend] = list(backends) if backends else self._discover_backends()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="alerter", daemon=True,
        )
        self._worker.start()
        names = ", ".join(b.name for b in self._backends) or "<none>"
        log.info("[ALERT] Alerter started backends=%s", names)

    # ── Backend discovery ────────────────────────────────────────────────
    @staticmethod
    def _discover_backends() -> List[_Backend]:
        out: List[_Backend] = [LogBackend()]  # always-on
        tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if tg_token and tg_chat:
            out.append(TelegramBackend(tg_token, tg_chat))
        slack_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
        if slack_url:
            out.append(SlackBackend(slack_url))
        return out

    # ── Worker loop ──────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ev = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if ev is None:
                break
            for b in self._backends:
                try:
                    b.send(ev)
                except Exception as e:  # noqa: BLE001
                    log.warning("[ALERT] backend %s raised: %s", b.name, e)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=timeout)

    # ── Internal enqueue ────────────────────────────────────────────────
    def _emit(self, kind: str, severity: str, **fields: Any) -> None:
        try:
            ev = AlertEvent(kind=kind, severity=severity, fields=fields)
            try:
                self._q.put_nowait(ev)
            except queue.Full:
                # Drop oldest, retry once. Trading must never block on alerts.
                try:
                    _ = self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(ev)
                except queue.Full:
                    log.warning("[ALERT] queue full, dropped %s", kind)
        except Exception as e:  # noqa: BLE001 - never propagate
            log.warning("[ALERT] _emit failed: %s", e)

    # ── Public API ──────────────────────────────────────────────────────
    def position_open(self, symbol: str, side: str, size: float, entry_price: float) -> None:
        self._emit(
            "position_open", "INFO",
            symbol=symbol, side=side, size=float(size),
            entry_price=float(entry_price),
        )

    def position_close(
        self, symbol: str, side: str, pnl: float,
        r_multiple: float, exit_reason: str,
    ) -> None:
        sev = "WARN" if float(pnl) < 0 else "INFO"
        self._emit(
            "position_close", sev,
            symbol=symbol, side=side, pnl=float(pnl),
            r_multiple=float(r_multiple), exit_reason=str(exit_reason),
        )

    def dd_breach(self, level: str, pct: float) -> None:
        self._emit("dd_breach", "WARN", level=str(level), pct=float(pct))

    def connection_lost(self, component: str, duration_secs: float) -> None:
        self._emit(
            "connection_lost", "ERROR",
            component=str(component), duration_secs=float(duration_secs),
        )

    def drift_detected(self, symbol: str, auc_before: float, auc_now: float) -> None:
        self._emit(
            "drift_detected", "WARN",
            symbol=str(symbol),
            auc_before=float(auc_before), auc_now=float(auc_now),
        )

    def error(
        self, component: str, message: str,
        context_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(
            "error", "ERROR",
            component=str(component), message=str(message),
            context=dict(context_dict) if context_dict else {},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level fallback singleton (so callers can `from .alerting import alerter`
# without forcing every site to plumb the instance through). The wired-up
# instance lives on AgentBrain._alerter; this is purely a safety net.
# ──────────────────────────────────────────────────────────────────────────────

_default: Optional[Alerter] = None
_default_lock = threading.Lock()


def get_default_alerter() -> Alerter:
    global _default
    with _default_lock:
        if _default is None:
            _default = Alerter()
        return _default
