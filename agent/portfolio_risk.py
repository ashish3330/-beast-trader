"""
Dragon Trader — Portfolio-Level Risk Model with Cross-Asset Hedging.

Provides:
1. Portfolio Heat Map — total risk by direction and currency/asset class
2. Dynamic Correlation Matrix — rolling 50-bar H1 correlation between symbol returns
3. Portfolio VaR (Value at Risk) — historical simulation, 95% confidence
4. Hedging Suggestions — directional and asset-class rebalancing advice

Wired into MasterBrain as gate: rejects if portfolio risk too concentrated.
"""
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import SYMBOLS, MAX_POSITIONS

log = logging.getLogger("dragon.portfolio_risk")

# ═══ ASSET CLASS & CURRENCY EXPOSURE MAPPING ═══
# Each symbol maps to its underlying exposures for heat map grouping.
# "long_currencies" = what you're long when you BUY the symbol.
# "short_currencies" = what you're short when you BUY the symbol.
SYMBOL_EXPOSURE = {
    "XAUUSD": {
        "asset_class": "Gold",
        "long_currencies": ["XAU"],
        "short_currencies": ["USD"],
    },
    "XAGUSD": {
        "asset_class": "Gold",
        "long_currencies": ["XAG"],
        "short_currencies": ["USD"],
    },
    "BTCUSD": {
        "asset_class": "Crypto",
        "long_currencies": ["BTC"],
        "short_currencies": ["USD"],
    },
    "NAS100.r": {
        "asset_class": "Index",
        "long_currencies": ["NAS100"],
        "short_currencies": ["USD"],
    },
    "JPN225ft": {
        "asset_class": "Index",
        "long_currencies": ["JPN225"],
        "short_currencies": ["JPY"],
    },
    "USDJPY": {
        "asset_class": "Forex",
        "long_currencies": ["USD"],
        "short_currencies": ["JPY"],
    },
}

# Concentration thresholds
DIRECTION_CONCENTRATION_THRESHOLD = 0.70   # >70% one direction = flag
ASSET_CLASS_CONCENTRATION_THRESHOLD = 0.70  # >70% one asset class = flag
CORRELATION_RISK_THRESHOLD = 0.70           # reduce risk if corr > 0.7
CORRELATION_RISK_REDUCTION = 0.60           # multiply risk by 0.6 for correlated positions
VAR_BLOCK_THRESHOLD_PCT = 5.0              # block new entries if VaR > 5% equity
ROLLING_CORR_BARS = 50                     # H1 bars for correlation matrix
VAR_LOOKBACK_DAYS = 50                     # days for historical VaR simulation
VAR_CONFIDENCE = 0.95                      # 95th percentile


class PortfolioRiskModel:
    """Portfolio-level risk assessment with cross-asset hedging intelligence."""

    def __init__(self, state, executor):
        self.state = state
        self.executor = executor
        self._lock = threading.RLock()

        # Cached correlation matrix — updated hourly
        self._corr_matrix: Optional[np.ndarray] = None
        self._corr_symbols: List[str] = []
        self._corr_last_update: float = 0.0
        self._corr_update_interval: float = 3600.0  # 1 hour

        # Cached VaR — updated hourly
        self._var_pct: float = 0.0
        self._var_last_update: float = 0.0

        # Last full assessment for dashboard
        self._last_assessment: dict = {}

    # ──────────────────────────────────────────────
    #  1. PORTFOLIO HEAT MAP
    # ──────────────────────────────────────────────

    def compute_heat_map(self) -> dict:
        """Sum all open positions' dollar risk by direction and asset class/currency.

        Returns
        -------
        dict
            {
                "long_risk_usd": float,
                "short_risk_usd": float,
                "total_risk_usd": float,
                "direction_pct": {"LONG": float, "SHORT": float},
                "asset_class_risk": {"Gold": float, "Crypto": float, ...},
                "asset_class_pct": {"Gold": float, ...},
                "currency_exposure": {"USD": float, "JPY": float, ...},
                "flags": ["70%+ LONG", "70%+ Gold", ...],
                "positions_detail": [...]
            }
        """
        positions = self.executor.get_positions_info()
        equity = float(self.state.get_agent_state().get("equity", 1000))

        long_risk = 0.0
        short_risk = 0.0
        asset_class_risk: Dict[str, float] = {}
        currency_net: Dict[str, float] = {}  # positive = net long, negative = net short
        details = []

        for pos in positions:
            symbol = pos["symbol"]
            direction = "LONG" if pos["type"] == "BUY" else "SHORT"
            # Dollar risk = |entry - SL| * volume * tick_value / tick_size
            # Approximate from PnL exposure: use SL distance * volume
            risk_usd = self._estimate_position_risk_usd(pos, equity)

            if direction == "LONG":
                long_risk += risk_usd
            else:
                short_risk += risk_usd

            # Asset class grouping
            exposure = SYMBOL_EXPOSURE.get(symbol, {})
            ac = exposure.get("asset_class", "Unknown")
            asset_class_risk[ac] = asset_class_risk.get(ac, 0.0) + risk_usd

            # Currency exposure (flip if SHORT)
            long_ccy = exposure.get("long_currencies", [])
            short_ccy = exposure.get("short_currencies", [])
            if direction == "LONG":
                for c in long_ccy:
                    currency_net[c] = currency_net.get(c, 0.0) + risk_usd
                for c in short_ccy:
                    currency_net[c] = currency_net.get(c, 0.0) - risk_usd
            else:
                # SHORT the pair = short the base, long the quote
                for c in long_ccy:
                    currency_net[c] = currency_net.get(c, 0.0) - risk_usd
                for c in short_ccy:
                    currency_net[c] = currency_net.get(c, 0.0) + risk_usd

            details.append({
                "symbol": symbol,
                "direction": direction,
                "risk_usd": round(risk_usd, 2),
                "asset_class": ac,
                "mode": pos.get("mode", "swing"),
            })

        total_risk = long_risk + short_risk
        flags = []

        # Direction concentration
        if total_risk > 0:
            long_pct = long_risk / total_risk
            short_pct = short_risk / total_risk
            if long_pct > DIRECTION_CONCENTRATION_THRESHOLD:
                flags.append(f"{long_pct:.0%} LONG — USD-short concentrated")
            if short_pct > DIRECTION_CONCENTRATION_THRESHOLD:
                flags.append(f"{short_pct:.0%} SHORT — USD-long concentrated")
        else:
            long_pct = 0.0
            short_pct = 0.0

        # Asset class concentration
        asset_class_pct = {}
        if total_risk > 0:
            for ac, risk in asset_class_risk.items():
                pct = risk / total_risk
                asset_class_pct[ac] = round(pct, 3)
                if pct > ASSET_CLASS_CONCENTRATION_THRESHOLD:
                    flags.append(f"{pct:.0%} in {ac} — asset class concentrated")

        return {
            "long_risk_usd": round(long_risk, 2),
            "short_risk_usd": round(short_risk, 2),
            "total_risk_usd": round(total_risk, 2),
            "total_risk_pct": round(total_risk / equity * 100, 2) if equity > 0 else 0.0,
            "direction_pct": {
                "LONG": round(long_pct, 3),
                "SHORT": round(short_pct, 3),
            },
            "asset_class_risk": {k: round(v, 2) for k, v in asset_class_risk.items()},
            "asset_class_pct": asset_class_pct,
            "currency_exposure": {k: round(v, 2) for k, v in sorted(
                currency_net.items(), key=lambda x: abs(x[1]), reverse=True)},
            "flags": flags,
            "positions_detail": details,
        }

    # ──────────────────────────────────────────────
    #  2. DYNAMIC CORRELATION MATRIX
    # ──────────────────────────────────────────────

    def update_correlation_matrix(self, force: bool = False):
        """Compute rolling 50-bar correlation from H1 close returns.

        Called hourly (or on demand with force=True).
        """
        now = time.time()
        if not force and (now - self._corr_last_update) < self._corr_update_interval:
            return  # still fresh

        symbols = list(SYMBOLS.keys())
        returns_dict: Dict[str, np.ndarray] = {}

        for sym in symbols:
            df = self.state.get_candles(sym, 60)  # H1
            if df is None or len(df) < ROLLING_CORR_BARS + 1:
                continue
            closes = df["close"].values.astype(float)[-(ROLLING_CORR_BARS + 1):]
            rets = np.diff(np.log(closes))  # log returns
            if len(rets) == ROLLING_CORR_BARS:
                returns_dict[sym] = rets

        if len(returns_dict) < 2:
            return

        sym_list = sorted(returns_dict.keys())
        n = len(sym_list)
        ret_matrix = np.column_stack([returns_dict[s] for s in sym_list])

        # Correlation matrix via numpy
        corr = np.corrcoef(ret_matrix, rowvar=False)

        with self._lock:
            self._corr_matrix = corr
            self._corr_symbols = sym_list
            self._corr_last_update = now

        log.info("Correlation matrix updated: %d symbols, %d bars", n, ROLLING_CORR_BARS)

    def get_correlation(self, sym_a: str, sym_b: str) -> Optional[float]:
        """Get pairwise correlation between two symbols."""
        with self._lock:
            if self._corr_matrix is None:
                return None
            syms = self._corr_symbols
            if sym_a not in syms or sym_b not in syms:
                return None
            i = syms.index(sym_a)
            j = syms.index(sym_b)
            return float(self._corr_matrix[i, j])

    def get_correlation_matrix_dict(self) -> dict:
        """Return correlation matrix as nested dict for dashboard."""
        with self._lock:
            if self._corr_matrix is None:
                return {}
            syms = self._corr_symbols
            result = {}
            for i, s1 in enumerate(syms):
                result[s1] = {}
                for j, s2 in enumerate(syms):
                    result[s1][s2] = round(float(self._corr_matrix[i, j]), 3)
            return result

    def check_correlation_risk(self, symbol: str, direction: str) -> Tuple[bool, float, str]:
        """Check if new position is highly correlated with existing open positions.

        Returns
        -------
        tuple
            (has_corr_risk: bool, risk_multiplier: float, reason: str)
            If has_corr_risk=True, multiply risk_pct by risk_multiplier.
        """
        open_positions = self.executor.get_positions_info()
        if not open_positions:
            return False, 1.0, ""

        open_symbols = set()
        open_dirs = {}
        for pos in open_positions:
            sym = pos["symbol"]
            d = "LONG" if pos["type"] == "BUY" else "SHORT"
            open_symbols.add(sym)
            open_dirs[sym] = d

        max_corr = 0.0
        max_corr_sym = ""
        same_dir_corr = False

        for open_sym in open_symbols:
            if open_sym == symbol:
                continue
            corr = self.get_correlation(symbol, open_sym)
            if corr is None:
                continue
            abs_corr = abs(corr)
            if abs_corr > max_corr:
                max_corr = abs_corr
                max_corr_sym = open_sym
                # Same effective direction if: (corr > 0 and same dir) or (corr < 0 and opposite dir)
                open_d = open_dirs.get(open_sym, "")
                if corr > 0 and direction == open_d:
                    same_dir_corr = True
                elif corr < 0 and direction != open_d:
                    same_dir_corr = True
                else:
                    same_dir_corr = False

        if max_corr > CORRELATION_RISK_THRESHOLD and same_dir_corr:
            reason = (f"corr({symbol},{max_corr_sym})={max_corr:.2f} "
                      f"same effective direction — risk x{CORRELATION_RISK_REDUCTION}")
            log.warning("PORTFOLIO CORR RISK: %s", reason)
            return True, CORRELATION_RISK_REDUCTION, reason

        return False, 1.0, ""

    # ──────────────────────────────────────────────
    #  3. PORTFOLIO VaR (VALUE AT RISK)
    # ──────────────────────────────────────────────

    def compute_var(self) -> float:
        """Estimate max 1-day portfolio loss at 95% confidence.

        Uses historical simulation on last 50 days of H1 returns,
        weighted by current position sizes.

        Returns
        -------
        float
            VaR as percentage of equity (positive = potential loss).
        """
        positions = self.executor.get_positions_info()
        if not positions:
            self._var_pct = 0.0
            return 0.0

        equity = float(self.state.get_agent_state().get("equity", 1000))
        if equity <= 0:
            return 0.0

        # Group positions by symbol, sum risk-weighted exposure
        symbol_exposure: Dict[str, float] = {}  # symbol -> net $ risk (+ = long, - = short)
        for pos in positions:
            sym = pos["symbol"]
            risk = self._estimate_position_risk_usd(pos, equity)
            if pos["type"] == "SELL":
                risk = -risk
            symbol_exposure[sym] = symbol_exposure.get(sym, 0.0) + risk

        if not symbol_exposure:
            self._var_pct = 0.0
            return 0.0

        # Collect daily returns for each symbol (from H1, aggregate 24 bars = 1 day)
        daily_returns: Dict[str, np.ndarray] = {}
        min_days = VAR_LOOKBACK_DAYS

        for sym in symbol_exposure:
            df = self.state.get_candles(sym, 60)  # H1
            if df is None or len(df) < 25:
                continue
            closes = df["close"].values.astype(float)
            # Aggregate H1 into daily: take every 24th close (approximate)
            # Use stride of ~24 bars or just compute overlapping daily returns
            # More robust: rolling 24-bar returns
            n = len(closes)
            if n < 25:
                continue
            # Daily return = close[i] / close[i-24] - 1 (overlapping)
            stride = min(24, n - 1)
            d_rets = closes[stride:] / closes[:-stride] - 1.0
            if len(d_rets) < 10:
                continue
            daily_returns[sym] = d_rets[-min_days:]

        if not daily_returns:
            self._var_pct = 0.0
            return 0.0

        # Align lengths
        min_len = min(len(r) for r in daily_returns.values())
        if min_len < 10:
            self._var_pct = 0.0
            return 0.0

        # Compute portfolio daily P&L scenarios
        n_scenarios = min_len
        portfolio_pnl = np.zeros(n_scenarios)

        for sym, exposure in symbol_exposure.items():
            if sym not in daily_returns:
                continue
            rets = daily_returns[sym][-n_scenarios:]
            # P&L = exposure * return (exposure is signed: + long, - short)
            portfolio_pnl += exposure * rets

        # VaR = loss at (1 - confidence) percentile
        # Negative tail: sort ascending, take the 5th percentile
        var_absolute = -np.percentile(portfolio_pnl, (1 - VAR_CONFIDENCE) * 100)
        var_pct = (var_absolute / equity) * 100.0 if equity > 0 else 0.0

        # Floor at 0 (VaR is a loss measure, can't be negative in this context)
        var_pct = max(0.0, var_pct)

        with self._lock:
            self._var_pct = round(var_pct, 3)
            self._var_last_update = time.time()

        log.info("Portfolio VaR(95%%): %.2f%% of equity ($%.2f)", var_pct, var_absolute)
        return var_pct

    def get_var_pct(self) -> float:
        """Return last computed VaR percentage."""
        with self._lock:
            return self._var_pct

    def is_var_breached(self) -> bool:
        """True if portfolio VaR exceeds block threshold."""
        return self._var_pct > VAR_BLOCK_THRESHOLD_PCT

    # ──────────────────────────────────────────────
    #  4. HEDGING SUGGESTIONS
    # ──────────────────────────────────────────────

    def get_hedging_suggestions(self, heat_map: Optional[dict] = None) -> dict:
        """Analyze portfolio imbalance and suggest hedging bias for next entry.

        Returns
        -------
        dict
            {
                "preferred_direction": "LONG" | "SHORT" | None,
                "avoid_asset_classes": ["Gold", ...],
                "prefer_asset_classes": ["Forex", ...],
                "avoid_currencies": ["USD", ...],
                "reasons": [str, ...]
            }
        """
        if heat_map is None:
            heat_map = self.compute_heat_map()

        suggestions = {
            "preferred_direction": None,
            "avoid_asset_classes": [],
            "prefer_asset_classes": [],
            "avoid_currencies": [],
            "reasons": [],
        }

        dir_pct = heat_map.get("direction_pct", {})
        ac_pct = heat_map.get("asset_class_pct", {})
        ccy_exposure = heat_map.get("currency_exposure", {})

        # Direction bias: if heavily long, suggest short bias (and vice versa)
        long_pct = dir_pct.get("LONG", 0)
        short_pct = dir_pct.get("SHORT", 0)

        if long_pct > DIRECTION_CONCENTRATION_THRESHOLD:
            suggestions["preferred_direction"] = "SHORT"
            suggestions["reasons"].append(
                f"Portfolio {long_pct:.0%} LONG — prefer SHORT for next entry to hedge")
        elif short_pct > DIRECTION_CONCENTRATION_THRESHOLD:
            suggestions["preferred_direction"] = "LONG"
            suggestions["reasons"].append(
                f"Portfolio {short_pct:.0%} SHORT — prefer LONG for next entry to hedge")

        # Asset class: avoid concentrated classes, prefer uncorrelated ones
        all_classes = {"Gold", "Crypto", "Index", "Forex"}
        concentrated = set()
        for ac, pct in ac_pct.items():
            if pct > ASSET_CLASS_CONCENTRATION_THRESHOLD:
                concentrated.add(ac)
                suggestions["avoid_asset_classes"].append(ac)
                suggestions["reasons"].append(
                    f"{ac} at {pct:.0%} of risk — avoid more {ac} exposure")

        if concentrated:
            uncorrelated = all_classes - concentrated
            # Map asset classes to symbols for preference
            for ac in uncorrelated:
                suggestions["prefer_asset_classes"].append(ac)

        # Currency: flag heavily exposed currencies
        for ccy, exposure in ccy_exposure.items():
            # Normalize exposure to equity
            equity = float(self.state.get_agent_state().get("equity", 1000))
            if equity > 0 and abs(exposure) / equity > 0.03:  # >3% equity in one currency
                suggestions["avoid_currencies"].append(ccy)
                side = "long" if exposure > 0 else "short"
                suggestions["reasons"].append(
                    f"Net {side} ${abs(exposure):.0f} in {ccy} — avoid adding same-direction {ccy} risk")

        return suggestions

    # ──────────────────────────────────────────────
    #  MASTER GATE — called by MasterBrain
    # ──────────────────────────────────────────────

    def evaluate_portfolio_risk(self, symbol: str, direction: str) -> dict:
        """Full portfolio risk gate for MasterBrain.

        Returns
        -------
        dict
            {
                "approved": bool,
                "risk_multiplier": float,  # multiply MasterBrain risk_pct by this
                "reason": str,
                "var_pct": float,
                "heat_map": dict,
                "hedging": dict,
            }
        """
        result = {
            "approved": True,
            "risk_multiplier": 1.0,
            "reason": "portfolio risk OK",
            "var_pct": 0.0,
            "heat_map": {},
            "hedging": {},
        }

        # --- Gate 1: VaR check (block if > 5% equity) ---
        var_pct = self.compute_var()
        result["var_pct"] = var_pct
        if var_pct > VAR_BLOCK_THRESHOLD_PCT:
            result["approved"] = False
            result["reason"] = f"VaR {var_pct:.1f}% > {VAR_BLOCK_THRESHOLD_PCT}% — portfolio risk too high"
            log.warning("PORTFOLIO GATE REJECT %s %s: %s", symbol, direction, result["reason"])
            return result

        # --- Gate 2: Heat map concentration ---
        heat_map = self.compute_heat_map()
        result["heat_map"] = heat_map

        flags = heat_map.get("flags", [])
        if flags:
            # Concentrated but don't block — reduce risk instead
            result["risk_multiplier"] *= 0.7
            result["reason"] = f"concentration flags: {'; '.join(flags)} — risk x0.7"
            log.warning("PORTFOLIO CONCENTRATION: %s", result["reason"])

        # --- Gate 3: Dynamic correlation ---
        self.update_correlation_matrix()
        has_corr_risk, corr_mult, corr_reason = self.check_correlation_risk(symbol, direction)
        if has_corr_risk:
            result["risk_multiplier"] *= corr_mult
            if result["reason"] == "portfolio risk OK":
                result["reason"] = corr_reason
            else:
                result["reason"] += f" | {corr_reason}"

        # --- Hedging suggestions ---
        hedging = self.get_hedging_suggestions(heat_map)
        result["hedging"] = hedging

        # If hedging strongly suggests opposite direction AND we have concentration flags,
        # add a further risk penalty (but never block — user rule: never skip trades)
        if hedging["preferred_direction"] and hedging["preferred_direction"] != direction:
            if flags:
                result["risk_multiplier"] *= 0.8
                note = f"against hedge suggestion ({hedging['preferred_direction']}) — risk x0.8"
                result["reason"] += f" | {note}"

        # Floor the multiplier at 0.3 (never reduce below 30% of intended risk)
        result["risk_multiplier"] = max(0.3, round(result["risk_multiplier"], 3))

        if result["risk_multiplier"] < 1.0:
            log.info("PORTFOLIO RISK ADJUST %s %s: mult=%.2f reason=%s",
                     symbol, direction, result["risk_multiplier"], result["reason"])

        return result

    # ──────────────────────────────────────────────
    #  PERIODIC UPDATE (call from main loop)
    # ──────────────────────────────────────────────

    def periodic_update(self):
        """Run updates: correlation matrix + VaR hourly, heat map every 60s for dashboard."""
        now = time.time()
        # Full recalc (corr + VaR) hourly; heat map + hedging every 60s
        if now - self._var_last_update < 60:
            return
        try:
            self.update_correlation_matrix()  # internal hourly guard
            self.compute_var()

            heat_map = self.compute_heat_map()
            hedging = self.get_hedging_suggestions(heat_map)

            assessment = {
                "heat_map": heat_map,
                "correlation_matrix": self.get_correlation_matrix_dict(),
                "var_pct": self._var_pct,
                "var_breached": self.is_var_breached(),
                "hedging_suggestions": hedging,
                "last_update": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            }

            with self._lock:
                self._last_assessment = assessment

            # Push to shared state for dashboard
            self.state.update_agent("portfolio_risk", assessment)

            log.info("Portfolio risk updated: VaR=%.2f%% flags=%s",
                     self._var_pct, heat_map.get("flags", []))
        except Exception as e:
            log.error("Portfolio risk update failed: %s", e, exc_info=True)

    def get_status(self) -> dict:
        """Return last assessment for dashboard / API."""
        with self._lock:
            return dict(self._last_assessment)

    # ──────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────

    def _estimate_position_risk_usd(self, pos: dict, equity: float) -> float:
        """Estimate dollar risk for a single position from its SL distance.

        Uses |price_open - sl| * volume as proxy. For accurate conversion we'd
        need tick_value/tick_size from the broker, but executor already computes
        risk at entry. This gives a reasonable relative measure.
        """
        entry = pos.get("price_open", 0)
        sl = pos.get("sl", 0)
        volume = pos.get("volume", 0)

        if entry <= 0 or sl <= 0 or volume <= 0:
            # Fallback: assume 1% of equity per position
            return equity * 0.01

        sl_dist = abs(entry - sl)

        # For symbols quoted in USD (XAUUSD, XAGUSD, BTCUSD, NAS100.r):
        # risk ~= sl_dist * volume * contract_multiplier
        # For USDJPY: risk = sl_dist * volume * 100000 / price (approx)
        # For JPN225ft: depends on contract spec
        # Use a practical approximation based on symbol
        symbol = pos.get("symbol", "")
        cfg = SYMBOLS.get(symbol)
        if cfg is None:
            return equity * 0.01

        # Approximate tick value from typical contract specs
        # This is a rough estimate; actual risk was computed at entry by executor
        if symbol in ("XAUUSD",):
            risk_usd = sl_dist * volume * 100  # 1 lot = 100 oz, $1/oz move = $100
        elif symbol in ("XAGUSD",):
            risk_usd = sl_dist * volume * 5000  # 1 lot = 5000 oz
        elif symbol in ("BTCUSD",):
            risk_usd = sl_dist * volume * 1  # 1 lot = 1 BTC typically
        elif symbol in ("NAS100.r",):
            risk_usd = sl_dist * volume * 1  # CFD, $1/point/lot
        elif symbol in ("JPN225ft",):
            # JPY-denominated, approximate conversion
            risk_usd = sl_dist * volume * 1  # CFD, ~$1/point at current rates
        elif symbol in ("USDJPY",):
            # Forex: 1 lot = 100,000 units, PnL in JPY, convert at ~150
            risk_usd = sl_dist * volume * 100000 / max(entry, 100)
        else:
            risk_usd = sl_dist * volume

        return max(0.01, risk_usd)
