"""
agent.expert.orchestrator — single ExpertGate that sequences all 11
expert components into one gate-slot in brain.py (between Gate 3f
ICT sweep and Gate 4 position management).

Component sequence (each can short-circuit with REJECT):

    (a) news_blackout_ext   — kill-switch first (tier-1 4h window)
    (b) regime_classifier   — D1 ADX session-stamped regime context
    (c) d1_structure        — D1 HH/HL + BOS/CHoCH bias
    (d) session_setups      — Asian/London/NY × setup-type allow map
    (e) order_block         — ICT/SMC order-block detector
    (f) wyckoff_spring      — Wyckoff Spring/Upthrust detector
    (g) tick_volume_gate    — participation filter (Gate 3g)
    (h) conviction_tier     — A+/B+/B (B = SKIP); returns size_mult
    (i) dynamic_sltp        — ASAT (preferred) → DynamicExitPlanner

Inputs are passed in via a context dict (the brain assembles it once
before invoking the gate) so each sub-component receives only what it
needs without the orchestrator knowing brain internals.

Public API
==========
    ExpertGate(brain).evaluate(ctx) -> dict
        Returns a verdict dict shaped::

            {
              "verdict":   "APPROVED" | "REJECT",
              "reason":    str,             # short tag, e.g. "RANGE_DAY_SKIP"
              "component": str | None,      # which component decided (None if approved)
              "size_mult": float,           # 1.0 default, set by conviction_tier
              "sl":        float | None,    # absolute SL price (if dynamic_sltp ran)
              "tp1":       float | None,    # absolute TP1 price
              "tp2":       float | None,    # absolute TP2 price
              "runner":    float | None,    # absolute runner TP price (DE only)
              "tp_source": str | None,      # provenance of TP2
              "session":   str | None,      # from SCSL
              "setup_type": str | None,     # from SCSL (or derived)
              "regime":    str | None,      # from range_day_classifier
              "d1_bias":   str | None,      # from d1_structure
              "ob_active": bool,            # order_block detected
              "wyckoff_active": bool,       # wyckoff detected
              "telemetry": dict,            # per-component sub-dicts
            }

The orchestrator NEVER raises. Any sub-component exception fails OPEN
on that component (records reason in telemetry but keeps going). The
overall verdict only flips to REJECT when a component explicitly
returns a reject-shaped result AND its own enable flag is True.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

log = logging.getLogger("dragon.expert.orchestrator")


# ── Defensive imports from sibling expert modules. Each block falls back
#    to None on failure so a half-built deploy doesn't crash the brain.
try:
    from agent.expert import (
        get_blackout_state,            # news_blackout_v2
        rdc_stamp_regime,              # range_day_classifier
        rdc_evaluate,
        rdc_classify_signal,
        d1_structure_evaluate,         # D1 swing structure
        evaluate_session_setup,        # SCSL
        detect_order_block,            # Order Block
        evaluate_tv_gate,              # Tick volume gate
        classify_conviction,           # Conviction A+/B+/B
        compute_asat_levels,           # ASAT TP/SL
        compute_exit_plan,             # DynamicExitPlanner
        classify_day_type,             # Dalton day-type classifier
        apply_day_type_routing,        # Dalton routing helper
    )
except Exception:  # pragma: no cover — defensive
    get_blackout_state = None
    rdc_stamp_regime = None
    rdc_evaluate = None
    rdc_classify_signal = None
    d1_structure_evaluate = None
    evaluate_session_setup = None
    detect_order_block = None
    evaluate_tv_gate = None
    classify_conviction = None
    compute_asat_levels = None
    compute_exit_plan = None
    classify_day_type = None
    apply_day_type_routing = None

# Wyckoff lives in agent/wyckoff_spring.py (re-exported via agent.expert).
try:
    from agent.expert import WyckoffSpringUpthrustStrategy
except Exception:  # pragma: no cover
    WyckoffSpringUpthrustStrategy = None


# ─────────────────────────────────────────────────────────────────────────
#  Verdict helpers
# ─────────────────────────────────────────────────────────────────────────
def _approved(size_mult: float = 1.0, **kwargs) -> Dict[str, Any]:
    base = {
        "verdict": "APPROVED",
        "reason": "ok",
        "component": None,
        "size_mult": float(size_mult),
        "sl": None,
        "tp1": None,
        "tp2": None,
        "runner": None,
        "tp_source": None,
        "session": None,
        "setup_type": None,
        "regime": None,
        "d1_bias": None,
        "day_type": None,
        "score_delta": 0.0,
        "ob_active": False,
        "wyckoff_active": False,
        "telemetry": {},
    }
    base.update(kwargs)
    return base


def _reject(component: str, reason: str, **kwargs) -> Dict[str, Any]:
    base = _approved()
    base["verdict"] = "REJECT"
    base["reason"] = reason
    base["component"] = component
    base["size_mult"] = 0.0
    base.update(kwargs)
    return base


def _safe_cfg(name: str, default: Any) -> Any:
    """Read a config flag with a safe default; never raise."""
    try:
        import config as _cfg  # type: ignore
        return getattr(_cfg, name, default)
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────
#  ExpertGate
# ─────────────────────────────────────────────────────────────────────────
class ExpertGate:
    """Single orchestrator that runs the 11 expert sub-components in
    documented order.  Stateless across symbols (each evaluate() call is
    self-contained) — per-component state (RDC daily cache, Wyckoff
    cooldowns, OB params) lives on the components themselves."""

    GATE_ORDER = (
        "news_blackout_ext",
        "regime_classifier",
        "day_type_routing",
        "d1_structure",
        "session_setups",
        "order_block",
        "wyckoff_spring",
        "tick_volume_gate",
        "conviction_tier",
        "dynamic_sltp",
    )

    def __init__(self, brain=None):
        """``brain`` is an optional reference to the AgentBrain — used for
        instances of long-lived helpers (calendar_filter, RDC cache,
        wyckoff strategy). All access is via getattr so the orchestrator
        also runs standalone in tests."""
        self._brain = brain
        # Per-symbol day cache for RDC (mirror what brain would otherwise
        # hold; stay local so the gate is self-contained).
        self._rdc_cache: Dict[str, Dict[str, Any]] = {}
        # Lazy Wyckoff strategy instance (one per brain lifetime).
        self._wyckoff = None
        # Lazy DynamicExitPlanner instance.
        self._dep = None

    # ──────────────────────────────────────────────────────────────────
    #  Component runners
    # ──────────────────────────────────────────────────────────────────
    def _run_news_blackout(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not _safe_cfg("NEWS_BLACKOUT_V2_ENABLED", False):
            return None
        if get_blackout_state is None:
            return None
        try:
            cal = getattr(self._brain, "_calendar", None)
            st = get_blackout_state(ctx["symbol"], calendar_filter=cal)
            if st and st.get("in_blackout"):
                return _reject(
                    "news_blackout_ext",
                    "NEWS_BLACKOUT_T%s_%s" % (st.get("tier", "?"), st.get("event", "?")),
                    telemetry={"news_blackout_ext": st},
                )
            return {"telemetry": {"news_blackout_ext": st or {}}}
        except Exception as e:
            log.debug("news_blackout_ext failed: %s", e)
            return None

    def _run_regime(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not _safe_cfg("RANGE_DAY_CLASSIFIER_ENABLED", False):
            return None
        if rdc_stamp_regime is None or rdc_evaluate is None:
            return None
        try:
            today = datetime.now(timezone.utc).date()
            rec = rdc_stamp_regime(
                ctx["symbol"], today, ctx.get("h1_df"), self._rdc_cache
            )
            # Map our signal source to the RDC class. Default = MOMENTUM.
            sig_class = ctx.get("signal_class") or "MOMENTUM"
            if rdc_classify_signal is not None:
                try:
                    sig_class = rdc_classify_signal(ctx.get("signal_source") or "momentum")
                except Exception:
                    pass
            verdict = rdc_evaluate(rec, sig_class, ctx.get("direction") or "")
            tel = {"regime_classifier": {"record": rec, "verdict": verdict}}
            if verdict.get("action") == "SKIP":
                return _reject(
                    "regime_classifier",
                    verdict.get("reason", "D1_REGIME_SKIP"),
                    regime=(rec or {}).get("regime"),
                    telemetry=tel,
                )
            # ALLOW — extract risk_mult tilt (RANGE_DAY downsizes trend setups).
            risk_mult = float(verdict.get("risk_mult", 1.0) or 1.0)
            return {
                "size_mult_tilt": risk_mult,
                "regime": (rec or {}).get("regime"),
                "telemetry": tel,
            }
        except Exception as e:
            log.debug("regime_classifier failed: %s", e)
            return None

    def _run_day_type_routing(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dalton day-type classifier + routing. Default OFF. Produces
        score_delta (caller may compose into raw_score downstream) and
        size_mult_tilt; NEVER rejects (honours [[feedback_no_skip_trades]]).
        Fail-open on any exception."""
        if not _safe_cfg("DAY_TYPE_ROUTING_ENABLED", False):
            return None
        if classify_day_type is None or apply_day_type_routing is None:
            return None
        try:
            h1 = ctx.get("h1_df")
            if h1 is None:
                return None
            # Pull last N H1 bars of "today" — caller passes a pandas-like
            # frame with high/low/close/open columns. We stay duck-typed so
            # the helper accepts numpy arrays or plain lists too.
            try:
                highs = list(h1["high"].values[-24:])  # type: ignore[index]
                lows = list(h1["low"].values[-24:])    # type: ignore[index]
                closes = list(h1["close"].values[-24:])  # type: ignore[index]
                opens = list(h1["open"].values[-24:])    # type: ignore[index]
            except Exception:
                # Fall back: assume already array-like sequences.
                highs = list(getattr(h1, "high", []) or [])[-24:]
                lows = list(getattr(h1, "low", []) or [])[-24:]
                closes = list(getattr(h1, "close", []) or [])[-24:]
                opens = list(getattr(h1, "open", []) or [])[-24:]

            if not highs or not lows:
                return None

            atr = float(ctx.get("atr", 0.0) or 0.0)
            ib_bars = int(_safe_cfg("DAY_TYPE_IB_BARS", 2))
            verdict = classify_day_type(
                highs, lows, closes, opens, atr14=atr,
                initial_balance_bars=ib_bars,
            )

            routing = apply_day_type_routing(
                verdict,
                ctx.get("signal_source"),
                ctx.get("direction"),
                raw_score=float(ctx.get("raw_score", 0.0) or 0.0),
                size_mult=1.0,
                score_boost=float(_safe_cfg("DAY_TYPE_SCORE_BOOST", 1.0)),
                score_penalty=float(_safe_cfg("DAY_TYPE_SCORE_PENALTY", 1.0)),
                normal_sr_boost=float(_safe_cfg("DAY_TYPE_NORMAL_SR_MULT", 1.2)),
                normal_momentum_drag=float(_safe_cfg("DAY_TYPE_NORMAL_MOMENTUM_MULT", 0.8)),
                double_dist_drag=float(_safe_cfg("DAY_TYPE_DOUBLE_DIST_MULT", 0.7)),
            )
            tel = {"day_type_routing": {"verdict": verdict, "routing": routing}}
            return {
                "day_type": verdict.get("day_type"),
                "size_mult_tilt": float(routing.get("size_mult_tilt", 1.0) or 1.0),
                "score_delta": float(routing.get("score_delta", 0.0) or 0.0),
                "telemetry": tel,
            }
        except Exception as e:
            log.debug("day_type_routing failed: %s", e)
            return None

    def _run_d1_structure(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not _safe_cfg("D1_STRUCTURE_BIAS_ENABLED", False):
            return None
        if d1_structure_evaluate is None:
            return None
        try:
            v = d1_structure_evaluate(
                ctx["symbol"], ctx.get("h1_df"), ctx.get("direction") or ""
            )
            tel = {"d1_structure": v}
            hard_block = set(_safe_cfg("D1_STRUCTURE_HARD_BLOCK_SYMBOLS", set()) or set())
            verdict = v.get("verdict", "OK")
            if verdict == "REJECT" and ctx["symbol"] in hard_block:
                return _reject(
                    "d1_structure",
                    "D1_STRUCTURE_AGAINST_BIAS",
                    d1_bias=v.get("bias"),
                    telemetry=tel,
                )
            return {"d1_bias": v.get("bias"), "telemetry": tel}
        except Exception as e:
            log.debug("d1_structure failed: %s", e)
            return None

    def _run_session_setups(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not _safe_cfg("SESSION_SETUP_ENABLED", False):
            return None
        if evaluate_session_setup is None:
            return None
        try:
            hour_utc = int(datetime.now(timezone.utc).hour)
            v = evaluate_session_setup(
                ctx["symbol"],
                hour_utc,
                ctx.get("h1_df"),
                ctx.get("ind"),
                ctx.get("bi") or 0,
                ctx.get("direction") or "",
                float(ctx.get("signal_quality", 0.0)),
                ctx.get("regime"),
                float(ctx.get("atr", 0.0)),
                float(ctx.get("min_quality", 0.0)),
            )
            tel = {"session_setups": v}
            log_only = bool(v.get("log_only", True))
            if not v.get("allowed", True) and not log_only:
                return _reject(
                    "session_setups",
                    "SCSL_" + str(v.get("reason", "REJECT")),
                    session=v.get("session"),
                    setup_type=v.get("setup_type"),
                    telemetry=tel,
                )
            return {
                "session": v.get("session"),
                "setup_type": v.get("setup_type"),
                "telemetry": tel,
            }
        except Exception as e:
            log.debug("session_setups failed: %s", e)
            return None

    def _run_order_block(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not _safe_cfg("OB_ENABLED", False):
            return None
        # Only acts as a GATE when OB_AS_GATE=True; otherwise OB runs as
        # an independent book elsewhere and the orchestrator only records
        # whether a block is present.
        as_gate = bool(_safe_cfg("OB_AS_GATE", True))
        if detect_order_block is None:
            return None
        try:
            sig = detect_order_block(
                ctx.get("h1_df"),
                ctx.get("m15_df"),
                ctx.get("direction") or "",
                d1=ctx.get("d1_df"),
                spread=float(ctx.get("spread", 0.0)),
            )
            tel = {"order_block": {"detected": sig is not None, "signal": sig}}
            if sig is None and as_gate:
                blacklist = set(_safe_cfg("OB_SYMBOL_BLACKLIST", set()) or set())
                whitelist = set(_safe_cfg("OB_WHITELIST", set()) or set())
                # Skip blacklisted symbols / honor whitelist when set.
                if ctx["symbol"] in blacklist:
                    return {"ob_active": False, "telemetry": tel}
                if whitelist and ctx["symbol"] not in whitelist:
                    return {"ob_active": False, "telemetry": tel}
                return _reject(
                    "order_block",
                    "OB_NO_BLOCK",
                    ob_active=False,
                    telemetry=tel,
                )
            return {"ob_active": sig is not None, "telemetry": tel}
        except Exception as e:
            log.debug("order_block failed: %s", e)
            return None

    def _run_wyckoff(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Wyckoff is an independent strategy/book, not a momentum gate.
        # We just RECORD whether the detector currently fires so the
        # journal/dashboard can slice on it. Never blocks the entry.
        if not _safe_cfg("WYCKOFF_ENABLED", False):
            return None
        if WyckoffSpringUpthrustStrategy is None:
            return None
        try:
            state = getattr(self._brain, "state", None)
            if state is None:
                return None
            if self._wyckoff is None:
                try:
                    self._wyckoff = WyckoffSpringUpthrustStrategy(state)
                except Exception:
                    self._wyckoff = False  # sentinel — don't retry
            if not self._wyckoff:
                return None
            sig = self._wyckoff.evaluate(ctx["symbol"])
            return {
                "wyckoff_active": sig is not None,
                "telemetry": {"wyckoff_spring": {"signal": sig}},
            }
        except Exception as e:
            log.debug("wyckoff_spring failed: %s", e)
            return None

    def _run_tv_gate(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not _safe_cfg("TV_VOLUME_GATE_ENABLED", False):
            return None
        if evaluate_tv_gate is None:
            return None
        try:
            comp_dir = ctx.get("comp_long") if ctx.get("direction") == "LONG" else ctx.get("comp_short")
            per_sym = _safe_cfg("TV_VOLUME_PER_SYMBOL", {}) or {}
            v = evaluate_tv_gate(
                ctx.get("direction"),
                ctx.get("regime"),
                comp_dir,
                ctx.get("h1_df"),
                lookback=int(_safe_cfg("TV_VOLUME_LOOKBACK_BARS", 20)),
                thr_breakout=float(_safe_cfg("TV_VOLUME_BREAKOUT_MIN_RATIO", 1.30)),
                thr_revert=float(_safe_cfg("TV_VOLUME_REVERT_MIN_RATIO", 0.70)),
                per_sym_overrides=per_sym,
                symbol=ctx.get("symbol"),
            )
            tel = {"tick_volume_gate": v}
            warn_only = set(_safe_cfg("TV_VOLUME_WARN_ONLY_SYMBOLS", set()) or set())
            if v.get("verdict") == "REJECT" and ctx["symbol"] not in warn_only:
                return _reject(
                    "tick_volume_gate",
                    v.get("reason", "TV_REJECT"),
                    telemetry=tel,
                )
            return {"telemetry": tel}
        except Exception as e:
            log.debug("tick_volume_gate failed: %s", e)
            return None

    def _run_conviction(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 2026-06-18 Tier 1 #7: shadow mode. CONVICTION_TIER_SHADOW_ENABLED
        # makes the classifier run + journal the tier but force size_mult=1.0
        # and NEVER reject — so we accumulate 14d of "would-have" data before
        # flipping CONVICTION_TIERING_ENABLED for real.
        shadow_on = _safe_cfg("CONVICTION_TIER_SHADOW_ENABLED", False)
        shadow_only = _safe_cfg("CONVICTION_TIER_SHADOW_ONLY", True)
        live_on = _safe_cfg("CONVICTION_TIERING_ENABLED", False)
        if not (shadow_on or live_on):
            return None
        if classify_conviction is None:
            return None
        try:
            v = classify_conviction(
                symbol=ctx["symbol"],
                direction=ctx.get("direction") or "",
                raw_score=float(ctx.get("raw_score", 0.0)),
                signal_quality=float(ctx.get("signal_quality", 0.0)),
                comp_long=ctx.get("comp_long"),
                comp_short=ctx.get("comp_short"),
                mtf_aligned=int(ctx.get("mtf_aligned", 0) or 0),
                m15_dir=ctx.get("m15_dir"),
                h1_df=ctx.get("h1_df"),
                ind=ctx.get("ind"),
                bi=ctx.get("bi"),
                order_flow_intel=getattr(self._brain, "_order_flow", None),
                regime=ctx.get("regime"),
            )
            tel = {"conviction_tier": v}
            tier = v.get("tier", "B+")
            size_mult = float(v.get("size_mult", 1.0) or 0.0)

            # ── Shadow path: log/journal only, never reject, never resize ──
            if shadow_on and (shadow_only or not live_on):
                tel["conviction_tier_shadow"] = True
                tel["would_block"] = (tier in ("B", "FAIL") or size_mult <= 0.0)
                # Always pass-through with neutral size_mult.
                return {"size_mult_tilt": 1.0, "telemetry": tel}

            # ── Live path: original logic ──
            if size_mult <= 0.0 or tier in ("B", "FAIL"):
                return _reject(
                    "conviction_tier",
                    "TIER_" + str(tier) + "_SKIP",
                    telemetry=tel,
                )
            return {"size_mult_tilt": size_mult, "telemetry": tel}
        except Exception as e:
            log.debug("conviction_tier failed: %s", e)
            return None

    def _run_dynamic_sltp(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """ASAT preferred (structure-aware TP, asymmetric SL); else
        DynamicExitPlanner (ATR + structure + regime). Returns absolute
        prices that override the executor's default SL/TP/runner."""
        out: Dict[str, Any] = {}
        # ── ASAT first ───────────────────────────────────────────────
        if _safe_cfg("ASAT_ENABLED", False) and compute_asat_levels is not None:
            try:
                wl = set(_safe_cfg("ASAT_SYMBOL_WHITELIST", set()) or set())
                if not wl or ctx["symbol"] in wl:
                    lvls = compute_asat_levels(
                        ctx["symbol"],
                        ctx.get("direction") or "",
                        float(ctx.get("entry_px", 0.0) or 0.0),
                        float(ctx.get("atr", 0.0) or 0.0),
                        ctx.get("h1_df"),
                        ctx.get("m15_df"),
                        float(ctx.get("sl_mult_base", 2.0) or 2.0),
                    )
                    if lvls is not None:
                        out.update({
                            "sl":  lvls.get("sl"),
                            "tp1": lvls.get("tp1"),
                            "tp2": lvls.get("tp2"),
                            "tp_source": lvls.get("tp2_source"),
                            "telemetry": {"asat": lvls},
                        })
                        return out
            except Exception as e:
                log.debug("ASAT failed: %s", e)

        # ── DynamicExitPlanner fallback ─────────────────────────────
        if _safe_cfg("DYNAMIC_EXIT_ENABLED", False) and compute_exit_plan is not None:
            try:
                wl = set(_safe_cfg("DYNAMIC_EXIT_SYMBOLS", set()) or set())
                if not wl or ctx["symbol"] in wl:
                    entry = float(ctx.get("entry_px", 0.0) or 0.0)
                    direction = (ctx.get("direction") or "").upper()
                    if entry > 0 and direction in ("LONG", "SHORT"):
                        plan = compute_exit_plan(
                            ctx["symbol"],
                            direction,
                            entry,
                            float(ctx.get("atr", 0.0) or 0.0),
                            ctx.get("regime") or "unknown",
                            ctx.get("h1_df"),
                            ctx.get("m15_df"),
                            spread=float(ctx.get("spread", 0.0) or 0.0),
                        )
                        if plan is not None:
                            sign = +1.0 if direction == "LONG" else -1.0
                            sl  = entry - sign * float(plan.sl_dist)
                            tp1 = entry + sign * float(plan.tp1_dist)
                            tp2 = entry + sign * float(plan.tp2_dist)
                            run_ = entry + sign * float(plan.runner_dist)
                            out.update({
                                "sl": sl, "tp1": tp1, "tp2": tp2,
                                "runner": run_,
                                "tp_source": plan.tp2_source,
                                "telemetry": {"dynamic_exit": plan.to_dict()},
                            })
                            return out
            except Exception as e:
                log.debug("DynamicExitPlanner failed: %s", e)

        return None

    # ──────────────────────────────────────────────────────────────────
    #  Main entry point
    # ──────────────────────────────────────────────────────────────────
    def evaluate(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Run all components in GATE_ORDER, short-circuit on REJECT,
        otherwise return an APPROVED verdict with accumulated payload.

        ``ctx`` is a plain dict the caller assembles. Keys consumed:

            symbol, direction, h1_df, m15_df, d1_df, ind, bi, atr,
            entry_px, spread, raw_score, signal_quality, mtf_aligned,
            m15_dir, regime, min_quality, comp_long, comp_short,
            signal_source, signal_class, sl_mult_base

        Missing keys default to safe values; the gate is fail-open.
        """
        # Master flag short-circuit — when the orchestrator is wholly
        # disabled, return immediate APPROVED so brain.py can fall back
        # to its existing pipeline.
        if not _safe_cfg("EXPERT_MODE_ENABLED", True):
            return _approved()

        result = _approved()
        size_mult = 1.0

        runners = {
            "news_blackout_ext":  self._run_news_blackout,
            "regime_classifier":  self._run_regime,
            "day_type_routing":   self._run_day_type_routing,
            "d1_structure":       self._run_d1_structure,
            "session_setups":     self._run_session_setups,
            "order_block":        self._run_order_block,
            "wyckoff_spring":     self._run_wyckoff,
            "tick_volume_gate":   self._run_tv_gate,
            "conviction_tier":    self._run_conviction,
            "dynamic_sltp":       self._run_dynamic_sltp,
        }

        for comp in self.GATE_ORDER:
            try:
                sub = runners[comp](ctx)
            except Exception as e:  # pragma: no cover — defensive
                log.debug("ExpertGate component %s crashed: %s", comp, e)
                sub = None

            if sub is None:
                continue

            # REJECT short-circuit
            if sub.get("verdict") == "REJECT":
                # Merge any accumulated telemetry so the journal sees
                # which components ran successfully before the REJECT.
                sub_tel = sub.get("telemetry") or {}
                merged_tel = dict(result.get("telemetry") or {})
                merged_tel.update(sub_tel)
                sub["telemetry"] = merged_tel
                return sub

            # APPROVED contribution — accumulate fields into result.
            # size_mult is multiplicative across components (RDC tilt
            # × conviction multiplier).
            if "size_mult_tilt" in sub:
                try:
                    size_mult *= float(sub["size_mult_tilt"])
                except Exception:
                    pass

            for k in ("regime", "d1_bias", "day_type", "session", "setup_type",
                      "sl", "tp1", "tp2", "runner", "tp_source"):
                if sub.get(k) is not None:
                    result[k] = sub[k]
            # day_type_routing exports a score_delta the caller may compose
            # into raw_score downstream. Accumulate so multiple components
            # (future) can each contribute.
            if "score_delta" in sub:
                try:
                    result["score_delta"] = float(result.get("score_delta", 0.0) or 0.0) \
                        + float(sub["score_delta"])
                except Exception:
                    pass
            if sub.get("ob_active"):
                result["ob_active"] = True
            if sub.get("wyckoff_active"):
                result["wyckoff_active"] = True

            sub_tel = sub.get("telemetry") or {}
            if sub_tel:
                merged_tel = dict(result.get("telemetry") or {})
                merged_tel.update(sub_tel)
                result["telemetry"] = merged_tel

        result["size_mult"] = float(size_mult)
        return result
