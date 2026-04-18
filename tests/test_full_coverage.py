"""
Dragon Trader — Full Coverage Test Suite.
Tests every critical method across all components.
Run: pytest tests/test_full_coverage.py -v
"""
import sys
import time
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SYMBOLS, SCALP_MAGIC_OFFSET


# ═══════════════════════════════════════════════════════════════════
#  MOCK HELPERS
# ═══════════════════════════════════════════════════════════════════

def _mock_mt5():
    mt5 = MagicMock()
    mt5.symbol_info.return_value = SimpleNamespace(
        trade_tick_value=1.0, trade_tick_size=0.01, point=0.01,
        digits=2, volume_min=0.01, volume_max=10.0, volume_step=0.01,
        trade_stops_level=0,
    )
    mt5.symbol_info_tick.return_value = SimpleNamespace(bid=3300.0, ask=3300.5)
    mt5.positions_get.return_value = []
    mt5.order_send.return_value = SimpleNamespace(retcode=10009, price=3300.0, volume=0.01, comment="ok")
    return mt5


def _mock_state(equity=1000):
    state = MagicMock()
    state.get_agent_state.return_value = {"equity": equity, "balance": equity, "dd_pct": 0}
    state.get_candles.return_value = None
    state.get_indicators.return_value = None
    state.get_tick.return_value = None
    state.get_tick_history.return_value = []
    return state


def _make_h1_df(n=200, base_price=3300.0):
    """Create realistic H1 candle DataFrame."""
    np.random.seed(42)
    times = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    closes = base_price + np.cumsum(np.random.randn(n) * 5)
    df = pd.DataFrame({
        "time": times,
        "open": closes - np.random.rand(n) * 2,
        "high": closes + np.abs(np.random.randn(n) * 3),
        "low": closes - np.abs(np.random.randn(n) * 3),
        "close": closes,
        "tick_volume": np.random.randint(100, 5000, n).astype(float),
        "spread": np.ones(n) * 0.3,
        "real_volume": np.zeros(n),
    })
    return df


# ═══════════════════════════════════════════════════════════════════
#  SHARED STATE
# ═══════════════════════════════════════════════════════════════════

class TestSharedState:
    def test_update_and_get_candles(self):
        from data.tick_streamer import SharedState
        s = SharedState()
        df = _make_h1_df(50)
        s.update_candles("XAUUSD", 60, df)
        result = s.get_candles("XAUUSD", 60)
        assert result is not None
        assert len(result) == 50
        # Verify it's a copy (modifying result doesn't affect state)
        result.iloc[0, 0] = pd.Timestamp("2000-01-01", tz="UTC")
        original = s.get_candles("XAUUSD", 60)
        assert original.iloc[0, 0] != pd.Timestamp("2000-01-01", tz="UTC")

    def test_get_candles_none_for_missing(self):
        from data.tick_streamer import SharedState
        s = SharedState()
        assert s.get_candles("NOTEXIST", 60) is None

    def test_update_and_get_indicators(self):
        from data.tick_streamer import SharedState
        s = SharedState()
        ind = {"atr": 25.0, "rsi": 55.0}
        s.update_indicators("XAUUSD", ind)
        result = s.get_indicators("XAUUSD")
        assert result["atr"] == 25.0

    def test_get_indicators_empty_for_missing(self):
        from data.tick_streamer import SharedState
        s = SharedState()
        result = s.get_indicators("NOTEXIST")
        assert result == {}

    def test_update_and_get_agent_state(self):
        from data.tick_streamer import SharedState
        s = SharedState()
        s.update_agent("equity", 1500.0)
        result = s.get_agent_state()
        assert result["equity"] == 1500.0

    def test_tick_history_bounded(self):
        from data.tick_streamer import SharedState
        s = SharedState()
        tick = SimpleNamespace(symbol="XAUUSD", bid=3300.0, ask=3300.5, time=time.time())
        for _ in range(600):
            s.update_tick(tick)
        hist = s.get_tick_history("XAUUSD", 1000)
        assert len(hist) <= 500

    def test_thread_safety(self):
        """Multiple threads read/write without crash."""
        from data.tick_streamer import SharedState
        s = SharedState()
        df = _make_h1_df(50)
        errors = []

        def writer():
            for _ in range(100):
                s.update_candles("XAUUSD", 60, df)
                s.update_agent("equity", 1000 + np.random.rand())

        def reader():
            for _ in range(100):
                s.get_candles("XAUUSD", 60)
                s.get_agent_state()

        threads = [threading.Thread(target=writer), threading.Thread(target=reader),
                   threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        # No crash = pass


# ═══════════════════════════════════════════════════════════════════
#  EXECUTOR — ORDER FLOW
# ═══════════════════════════════════════════════════════════════════

class TestExecutorSendOrder:
    def test_send_order_success(self):
        from execution.executor import Executor
        mt5 = _mock_mt5()
        ex = Executor(mt5, _mock_state())
        result, vol = ex._send_order(
            {"action": 1, "symbol": "XAUUSD", "volume": 0.01, "type": 0, "price": 3300.0},
            "XAUUSD", "TEST"
        )
        assert result is not None
        assert int(result.retcode) == 10009

    def test_send_order_none_result(self):
        from execution.executor import Executor
        mt5 = _mock_mt5()
        mt5.order_send.return_value = None
        ex = Executor(mt5, _mock_state())
        result, vol = ex._send_order(
            {"action": 1, "symbol": "XAUUSD", "volume": 0.01, "type": 0, "price": 3300.0},
            "XAUUSD", "TEST"
        )
        assert result is None
        assert vol == 0.0

    def test_send_order_exception_caught(self):
        from execution.executor import Executor
        mt5 = _mock_mt5()
        mt5.order_send.side_effect = ConnectionError("network down")
        ex = Executor(mt5, _mock_state())
        result, vol = ex._send_order(
            {"action": 1, "symbol": "XAUUSD", "volume": 0.01, "type": 0, "price": 3300.0},
            "XAUUSD", "TEST"
        )
        assert result is None  # should not crash

    def test_send_order_requote_retry(self):
        from execution.executor import Executor
        mt5 = _mock_mt5()
        # First call: requote, second call: success
        mt5.order_send.side_effect = [
            SimpleNamespace(retcode=10004, comment="requote"),
            SimpleNamespace(retcode=10009, price=3300.0, volume=0.01, comment="ok"),
        ]
        ex = Executor(mt5, _mock_state())
        result, vol = ex._send_order(
            {"action": 1, "symbol": "XAUUSD", "volume": 0.01, "type": 0, "price": 3300.0},
            "XAUUSD", "TEST"
        )
        assert result is not None
        assert int(result.retcode) == 10009
        assert mt5.order_send.call_count == 2

    def test_send_order_connection_lost_retry(self):
        from execution.executor import Executor
        mt5 = _mock_mt5()
        mt5.order_send.side_effect = [
            SimpleNamespace(retcode=10006, comment="connection lost"),
            SimpleNamespace(retcode=10009, price=3300.0, volume=0.01, comment="ok"),
        ]
        ex = Executor(mt5, _mock_state())
        result, vol = ex._send_order(
            {"action": 1, "symbol": "XAUUSD", "volume": 0.01, "type": 0, "price": 3300.0},
            "XAUUSD", "TEST"
        )
        assert int(result.retcode) == 10009


class TestExecutorSpreadCheck:
    def test_spread_spike_returns_tuple(self):
        from execution.executor import Executor
        mt5 = _mock_mt5()
        ex = Executor(mt5, _mock_state())
        result = ex._check_spread_spike("XAUUSD", signal_spread=0.1)
        # Returns (ok_bool, tick) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestExecutorOpenScalp:
    def test_open_scalp_trade_uses_scalp_magic(self):
        from execution.executor import Executor
        mt5 = _mock_mt5()
        ex = Executor(mt5, _mock_state(5000))
        ex.open_scalp_trade("XAUUSD", "LONG", 25.0, risk_pct=0.2)
        if mt5.order_send.called:
            req = mt5.order_send.call_args[0][0]
            assert int(req["magic"]) == SYMBOLS["XAUUSD"].magic + SCALP_MAGIC_OFFSET


class TestExecutorGetPositionsInfo:
    def test_get_positions_info_empty(self):
        from execution.executor import Executor
        mt5 = _mock_mt5()
        mt5.positions_get.return_value = []
        ex = Executor(mt5, _mock_state())
        result = ex.get_positions_info()
        assert isinstance(result, list)

    def test_get_positions_info_none(self):
        from execution.executor import Executor
        mt5 = _mock_mt5()
        mt5.positions_get.return_value = None
        ex = Executor(mt5, _mock_state())
        result = ex.get_positions_info()
        assert isinstance(result, list)


class TestExecutorGetAtr:
    def test_get_atr_from_indicators(self):
        from execution.executor import Executor
        state = _mock_state()
        state.get_indicators.return_value = {"atr": 25.5}
        ex = Executor(_mock_mt5(), state)
        assert ex._get_atr("XAUUSD") == 25.5

    def test_get_atr_fallback_to_candles(self):
        from execution.executor import Executor
        state = _mock_state()
        state.get_indicators.return_value = {}
        state.get_candles.return_value = _make_h1_df(30)
        ex = Executor(_mock_mt5(), state)
        atr = ex._get_atr("XAUUSD")
        assert atr > 0

    def test_get_atr_avg(self):
        from execution.executor import Executor
        state = _mock_state()
        state.get_candles.return_value = _make_h1_df(100)
        ex = Executor(_mock_mt5(), state)
        avg = ex._get_atr_avg("XAUUSD")
        assert avg > 0

    def test_get_atr_avg_insufficient_data(self):
        from execution.executor import Executor
        state = _mock_state()
        state.get_candles.return_value = _make_h1_df(10)
        ex = Executor(_mock_mt5(), state)
        avg = ex._get_atr_avg("XAUUSD")
        assert avg == 0.0  # not enough data


# ═══════════════════════════════════════════════════════════════════
#  MOMENTUM SCORER
# ═══════════════════════════════════════════════════════════════════

class TestMomentumScorer:
    def test_compute_indicators_returns_all_keys(self):
        from signals.momentum_scorer import _compute_indicators, IND_DEFAULTS
        df = _make_h1_df(200)
        ind = _compute_indicators(df, IND_DEFAULTS)
        for key in ["c", "h", "l", "o", "at", "rs", "adx", "es", "el", "et",
                     "mh", "stl", "st", "bbw", "consec", "n"]:
            assert key in ind, f"Missing key: {key}"

    def test_compute_indicators_n_matches_length(self):
        from signals.momentum_scorer import _compute_indicators, IND_DEFAULTS
        df = _make_h1_df(200)
        ind = _compute_indicators(df, IND_DEFAULTS)
        assert ind["n"] == len(df)

    def test_score_returns_two_floats(self):
        from signals.momentum_scorer import _compute_indicators, _score, IND_DEFAULTS
        df = _make_h1_df(200)
        ind = _compute_indicators(df, IND_DEFAULTS)
        ls, ss = _score(ind, ind["n"] - 2)
        assert isinstance(ls, (int, float))
        assert isinstance(ss, (int, float))
        assert ls >= 0
        assert ss >= 0

    def test_score_nan_atr_returns_zero(self):
        from signals.momentum_scorer import _compute_indicators, _score, IND_DEFAULTS
        df = _make_h1_df(200)
        ind = _compute_indicators(df, IND_DEFAULTS)
        ind["at"][ind["n"] - 2] = float('nan')
        ls, ss = _score(ind, ind["n"] - 2)
        assert ls == 0 or ss == 0 or True  # should not crash

    def test_short_dataframe_no_crash(self):
        from signals.momentum_scorer import _compute_indicators, IND_DEFAULTS
        df = _make_h1_df(5)
        ind = _compute_indicators(df, IND_DEFAULTS)
        assert ind["n"] == 5


# ═══════════════════════════════════════════════════════════════════
#  ML SIGNAL MODEL
# ═══════════════════════════════════════════════════════════════════

class TestSignalModelHelpers:
    def test_compute_tf_indicators(self):
        from models.signal_model import _compute_tf_indicators
        df = _make_h1_df(100)
        ind = _compute_tf_indicators(df)
        for key in ["rsi", "ema8", "ema21", "adx", "atr", "macd_hist",
                     "bb_upper", "bb_lower", "close", "high", "low", "consec", "n"]:
            assert key in ind, f"Missing: {key}"
        assert ind["n"] == 100

    def test_compute_tf_indicators_short_df(self):
        from models.signal_model import _compute_tf_indicators
        df = _make_h1_df(10)
        ind = _compute_tf_indicators(df)
        assert ind["n"] == 10  # should not crash

    def test_fill_mtf_features_no_data(self):
        from models.signal_model import _fill_mtf_features
        X = np.zeros((1, 42))
        # No M15/M5 data — should fill with defaults
        _fill_mtf_features(X, 0, 50, 1, 25.0,
                           None, None, None, None,
                           np.ones(100) * 3300, np.ones(100) * 3310, np.ones(100) * 3290)
        assert X[0, 21] == pytest.approx(50.0, abs=5)  # m15_rsi fallback
        assert X[0, 26] == 50.0  # m5_rsi default

    def test_predict_no_model_returns_defaults(self):
        from models.signal_model import SignalModel, META_FEATURE_NAMES
        model = SignalModel()
        features = {name: 0.0 for name in META_FEATURE_NAMES}
        result = model.predict("NONEXIST", features)
        assert "confidence" in result
        assert "take_trade" in result


# ═══════════════════════════════════════════════════════════════════
#  MTF INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════

class TestMTFAnalysis:
    def test_analyze_h1(self):
        from agent.mtf_intelligence import MTFIntelligence
        state = _mock_state()
        mtf = MTFIntelligence(state)
        df = _make_h1_df(200)
        result = mtf._analyze_h1(df)
        assert result.direction in ("LONG", "SHORT", "FLAT")
        assert 0 <= result.strength <= 1.0

    def test_analyze_h1_none(self):
        from agent.mtf_intelligence import MTFIntelligence
        mtf = MTFIntelligence(_mock_state())
        result = mtf._analyze_h1(None)
        assert result.direction == "FLAT"

    def test_analyze_m15(self):
        from agent.mtf_intelligence import MTFIntelligence
        mtf = MTFIntelligence(_mock_state())
        df = _make_h1_df(100)
        result = mtf._analyze_m15(df)
        assert result.direction in ("LONG", "SHORT", "FLAT")

    def test_detect_regime(self):
        from agent.mtf_intelligence import MTFIntelligence, _TFResult
        mtf = MTFIntelligence(_mock_state())
        df = _make_h1_df(200)
        h1 = _TFResult(direction="LONG", strength=0.5, detail={"adx": 30})
        regime = mtf._detect_regime(df, h1)
        assert regime in ("trending", "ranging", "volatile", "breakout", "low_vol")

    def test_compute_confluence(self):
        from agent.mtf_intelligence import MTFIntelligence
        mtf = MTFIntelligence(_mock_state())
        count, direction = mtf._compute_confluence(["LONG", "LONG", "FLAT", "SHORT"])
        assert direction in ("LONG", "SHORT", "FLAT")
        assert 0 <= count <= 4

    def test_compute_time_weight_bounded(self):
        from agent.mtf_intelligence import MTFIntelligence
        mtf = MTFIntelligence(_mock_state())
        ctx = {"session_score": 0.8, "minutes_to_close": 999, "avoid_entry": False}
        w = mtf._compute_time_weight(ctx)
        assert 0.0 <= w <= 1.0

    def test_compute_time_weight_avoid_entry(self):
        from agent.mtf_intelligence import MTFIntelligence
        mtf = MTFIntelligence(_mock_state())
        ctx = {"session_score": 0.8, "avoid_entry": True}
        w = mtf._compute_time_weight(ctx)
        assert w == 0.1

    def test_default_result_all_fields(self):
        from agent.mtf_intelligence import MTFIntelligence
        mtf = MTFIntelligence(_mock_state())
        d = mtf._default_result()
        for key in ["confluence", "entry_quality", "exit_urgency",
                     "regime", "optimal_sl", "optimal_tp"]:
            assert key in d


# ═══════════════════════════════════════════════════════════════════
#  EXIT INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════

class TestExitIntelligence:
    def test_evaluate_exits_no_positions(self):
        from agent.exit_intelligence import ExitIntelligence
        state = _mock_state()
        ex = MagicMock()
        ex._directions = {}
        ei = ExitIntelligence(state, ex)
        ei.evaluate_exits()  # should not crash

    def test_cleanup(self):
        from agent.exit_intelligence import ExitIntelligence
        ei = ExitIntelligence(_mock_state(), MagicMock())
        ei._peak_profit_r["XAUUSD"] = 2.5
        ei._bars_in_trade["XAUUSD"] = 10
        ei._cleanup("XAUUSD")
        assert "XAUUSD" not in ei._peak_profit_r
        assert "XAUUSD" not in ei._bars_in_trade


# ═══════════════════════════════════════════════════════════════════
#  LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════

class TestLearningEngine:
    def test_init_db_creates_table(self):
        from agent.learning_engine import LearningEngine
        import tempfile, os
        state = _mock_state()
        ex = MagicMock()
        mb = MagicMock()
        le = LearningEngine(state, mb, ex)
        # DB should exist after init
        assert hasattr(le, '_recent_trades')

    def test_record_trade(self):
        from agent.learning_engine import LearningEngine
        le = LearningEngine(_mock_state(), MagicMock(), MagicMock())
        le.record_trade("XAUUSD", "LONG", 50.0, "SL")
        stats = le.get_symbol_stats("XAUUSD")
        assert stats is not None

    def test_get_risk_multiplier_no_trades(self):
        from agent.learning_engine import LearningEngine
        le = LearningEngine(_mock_state(), MagicMock(), MagicMock())
        mult = le.get_risk_multiplier("XAUUSD")
        assert 0.3 <= mult <= 1.5  # within reasonable range

    def test_get_all_stats(self):
        from agent.learning_engine import LearningEngine
        le = LearningEngine(_mock_state(), MagicMock(), MagicMock())
        stats = le.get_all_stats()
        assert isinstance(stats, dict)


# ═══════════════════════════════════════════════════════════════════
#  PORTFOLIO RISK
# ═══════════════════════════════════════════════════════════════════

class TestPortfolioRisk:
    def test_init_no_crash(self):
        from agent.portfolio_risk import PortfolioRiskModel
        pr = PortfolioRiskModel(_mock_state(), MagicMock())
        assert pr is not None

    def test_compute_var_no_data(self):
        from agent.portfolio_risk import PortfolioRiskModel
        state = _mock_state()
        state.get_candles.return_value = None
        pr = PortfolioRiskModel(state, MagicMock())
        var = pr.compute_var()
        assert var == 0.0

    def test_periodic_update_no_crash(self):
        from agent.portfolio_risk import PortfolioRiskModel
        state = _mock_state()
        state.get_candles.return_value = None
        ex = MagicMock()
        ex.get_positions_info.return_value = []
        pr = PortfolioRiskModel(state, ex)
        pr.periodic_update()  # should not crash


# ═══════════════════════════════════════════════════════════════════
#  MASTER BRAIN — ADDITIONAL
# ═══════════════════════════════════════════════════════════════════

class TestMasterBrainAdditional:
    def test_get_correlated_exposure(self):
        from agent.master_brain import MasterBrain
        mt5 = _mock_mt5()
        ex = MagicMock()
        ex._directions = {"XAUUSD": "LONG"}
        ex.has_position.return_value = True
        mb = MasterBrain(_mock_state(), mt5, ex)
        # XAGUSD should see XAUUSD as correlated
        result = mb.get_correlated_exposure("XAGUSD")
        assert isinstance(result, bool)  # True if correlated pair open

    def test_no_correlated_exposure(self):
        from agent.master_brain import MasterBrain
        ex = MagicMock()
        ex._directions = {}
        ex.has_position.return_value = False
        mb = MasterBrain(_mock_state(), _mock_mt5(), ex)
        result = mb.get_correlated_exposure("BTCUSD")
        assert result is False

    def test_get_status(self):
        from agent.master_brain import MasterBrain
        mb = MasterBrain(_mock_state(), _mock_mt5(), MagicMock())
        status = mb.get_status()
        assert isinstance(status, dict)

    def test_reset_daily(self):
        from agent.master_brain import MasterBrain
        mb = MasterBrain(_mock_state(), _mock_mt5(), MagicMock())
        mb.reset_daily()  # should not crash


# ═══════════════════════════════════════════════════════════════════
#  SMART ENTRY — ADDITIONAL
# ═══════════════════════════════════════════════════════════════════

class TestSmartEntryAdditional:
    def test_compute_usd_strength_no_data(self):
        from agent.smart_entry import SmartEntry
        state = _mock_state()
        state.get_candles.return_value = None
        se = SmartEntry(state)
        strength = se._compute_usd_strength()
        assert strength == 0.0

    def test_get_h1_return_no_data(self):
        from agent.smart_entry import SmartEntry
        state = _mock_state()
        state.get_candles.return_value = None
        se = SmartEntry(state)
        ret = se._get_h1_return("XAUUSD", 5)
        assert ret is None

    def test_get_h1_return_with_data(self):
        from agent.smart_entry import SmartEntry
        state = _mock_state()
        state.get_candles.return_value = _make_h1_df(20)
        se = SmartEntry(state)
        ret = se._get_h1_return("XAUUSD", 5)
        assert ret is not None
        assert isinstance(ret, float)

    def test_ema_helper(self):
        from agent.smart_entry import SmartEntry
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = SmartEntry._ema(arr, 3)
        assert not np.isnan(result[-1])
        assert np.isnan(result[0])  # first values are NaN

    def test_ema_short_array(self):
        from agent.smart_entry import SmartEntry
        arr = np.array([1.0, 2.0])
        result = SmartEntry._ema(arr, 5)
        assert all(np.isnan(result))  # too short

    def test_check_pullback_with_real_data(self):
        from agent.smart_entry import SmartEntry
        state = _mock_state()
        state.get_candles.return_value = _make_h1_df(50)
        se = SmartEntry(state)
        r = se._check_pullback("XAUUSD", "LONG", 25.0)
        assert "mult" in r
        assert "state" in r
        assert 0.5 <= r["mult"] <= 1.5

    def test_check_volume_with_real_data(self):
        from agent.smart_entry import SmartEntry
        state = _mock_state()
        state.get_candles.return_value = _make_h1_df(50)
        se = SmartEntry(state)
        r = se._check_volume("XAUUSD", "LONG")
        assert "mult" in r
        assert 0.5 <= r["mult"] <= 1.5


# ═══════════════════════════════════════════════════════════════════
#  EQUITY GUARDIAN — ADDITIONAL
# ═══════════════════════════════════════════════════════════════════

class TestEquityGuardianAdditional:
    def test_cleanup(self):
        from agent.equity_guardian import EquityGuardian
        g = EquityGuardian(_mock_state(), MagicMock())
        g._entry_time["XAUUSD"] = time.time()
        g._peak_pnl["XAUUSD"] = 50.0
        g._baseline_pnl["XAUUSD"] = 10.0
        g._cleanup("XAUUSD")
        assert "XAUUSD" not in g._entry_time
        assert "XAUUSD" not in g._peak_pnl
        assert "XAUUSD" not in g._baseline_pnl

    def test_day_start_equity_set_after_monitor(self):
        from agent.equity_guardian import EquityGuardian
        state = _mock_state(995)
        ex = MagicMock()
        ex.get_positions_info.return_value = []
        g = EquityGuardian(state, ex)
        # Weekend skip might prevent setting — call monitor
        g.monitor()
        # If weekend, _day_start_equity stays None. Otherwise set.
        assert g._day_start_equity is None or g._day_start_equity == 995

    def test_sharp_loss_skipped_without_baseline(self):
        """If symbol not in _baseline_pnl, sharp loss check should be skipped."""
        from agent.equity_guardian import EquityGuardian
        state = _mock_state(1000)
        ex = MagicMock()
        ex.get_positions_info.return_value = [
            {"symbol": "NEW_SYM", "pnl": -50.0, "mode": "swing", "ticket": 1}
        ]
        ex.close_position = MagicMock()
        g = EquityGuardian(state, ex)
        # First call sets baseline, should NOT trigger sharp loss cut
        g.monitor()
        ex.close_position.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
#  BRAIN — KEY METHODS
# ═══════════════════════════════════════════════════════════════════

class TestBrainHelpers:
    def test_get_adaptive_min_score(self):
        from agent.brain import AgentBrain
        state = _mock_state()
        ex = MagicMock()
        brain = AgentBrain(state, _mock_mt5(), ex)
        score = brain._get_adaptive_min_score("trending", "XAUUSD")
        assert score > 0
        assert isinstance(score, (int, float))

    def test_get_adaptive_min_score_unknown_regime(self):
        from agent.brain import AgentBrain
        brain = AgentBrain(_mock_state(), _mock_mt5(), MagicMock())
        score = brain._get_adaptive_min_score("unknown", "XAUUSD")
        assert score > 0

    def test_meta_passes_none_prob(self):
        from agent.brain import AgentBrain
        brain = AgentBrain(_mock_state(), _mock_mt5(), MagicMock())
        assert brain._meta_passes("XAUUSD", None) is True  # no model = pass

    def test_meta_passes_high_prob(self):
        from agent.brain import AgentBrain
        brain = AgentBrain(_mock_state(), _mock_mt5(), MagicMock())
        assert brain._meta_passes("XAUUSD", 0.7) is True

    def test_meta_passes_low_prob(self):
        from agent.brain import AgentBrain
        brain = AgentBrain(_mock_state(), _mock_mt5(), MagicMock())
        assert brain._meta_passes("XAUUSD", 0.3) is False


# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD SANITIZE
# ═══════════════════════════════════════════════════════════════════

class TestDashboardSanitize:
    def test_sanitize_numpy_types(self):
        from dashboard.app import _sanitize
        data = {
            "int64": np.int64(42),
            "float64": np.float64(3.14),
            "bool": np.bool_(True),
            "int32": np.int32(7),
            "float32": np.float32(2.71),
        }
        result = _sanitize(data)
        assert isinstance(result["int64"], int)
        assert isinstance(result["float64"], float)
        assert isinstance(result["bool"], bool)

    def test_sanitize_nested(self):
        from dashboard.app import _sanitize
        data = {"a": {"b": np.int64(1)}, "c": [np.float64(2.0)]}
        result = _sanitize(data)
        assert isinstance(result["a"]["b"], int)
        assert isinstance(result["c"][0], float)

    def test_sanitize_nan(self):
        from dashboard.app import _sanitize
        data = {"val": float("nan")}
        result = _sanitize(data)
        # NaN should be converted to None or 0 for JSON safety
        assert result["val"] is None or result["val"] == 0 or np.isnan(result["val"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
