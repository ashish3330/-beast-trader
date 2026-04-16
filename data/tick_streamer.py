"""
Beast Trader — Tick Streamer.
Streams bid/ask from MT5 every 500ms, builds real-time OHLC candles,
calculates indicators, stores tick history in SQLite.
Thread-safe shared state for agent + dashboard.
"""
import time
import threading
import sqlite3
import logging
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from mt5linux import MetaTrader5

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    SYMBOLS, TICK_INTERVAL_MS, CANDLE_WINDOW, TIMEFRAMES, DB_PATH,
)

log = logging.getLogger("beast.streamer")


class TickData:
    """Single tick snapshot."""
    __slots__ = ("symbol", "bid", "ask", "time", "volume")

    def __init__(self, symbol, bid, ask, t, volume=0):
        self.symbol = symbol
        self.bid = float(bid)
        self.ask = float(ask)
        self.time = t
        self.volume = int(volume)


class SharedState:
    """Thread-safe shared state between streamer, agent, and dashboard."""

    def __init__(self):
        self._lock = threading.RLock()
        # Latest tick per symbol
        self.ticks = {}                    # symbol -> TickData
        self.tick_history = defaultdict(list)  # symbol -> list of TickData (last 500)
        # OHLC candles per symbol per timeframe
        self.candles = {}                  # (symbol, tf) -> pd.DataFrame
        # Real-time indicators per symbol
        self.indicators = {}               # symbol -> dict
        # Building candle accumulators
        self._candle_acc = {}              # (symbol, tf) -> {open, high, low, close, volume, start_time}
        # Agent state (written by brain, read by dashboard)
        self.agent_state = {
            "running": False,
            "cycle": 0,
            "equity": 0.0,
            "balance": 0.0,
            "profit": 0.0,
            "dd_pct": 0.0,
            "peak_equity": 0.0,
            "daily_loss": 0.0,
            "positions": [],
            "model_confidence": {},        # symbol -> {direction, prob_up, prob_down, confidence}
            "trade_log": [],
            "feature_importance": {},
        }

    def update_tick(self, tick: TickData):
        with self._lock:
            self.ticks[tick.symbol] = tick
            hist = self.tick_history[tick.symbol]
            hist.append(tick)
            if len(hist) > 500:
                self.tick_history[tick.symbol] = hist[-500:]

    def get_tick(self, symbol) -> TickData:
        with self._lock:
            return self.ticks.get(symbol)

    def get_tick_history(self, symbol, count=100):
        with self._lock:
            return list(self.tick_history.get(symbol, []))[-count:]

    def update_candles(self, symbol, tf, df):
        with self._lock:
            self.candles[(symbol, tf)] = df

    def get_candles(self, symbol, tf) -> pd.DataFrame:
        with self._lock:
            df = self.candles.get((symbol, tf))
            if df is not None:
                return df.copy()
            return None

    def update_indicators(self, symbol, ind_dict):
        with self._lock:
            self.indicators[symbol] = ind_dict

    def get_indicators(self, symbol):
        with self._lock:
            return dict(self.indicators.get(symbol, {}))

    def update_agent(self, key, value):
        with self._lock:
            self.agent_state[key] = value

    def get_agent_state(self):
        with self._lock:
            return dict(self.agent_state)


class TickStreamer:
    """Streams tick data from MT5, builds candles, calculates indicators."""

    def __init__(self, state: SharedState):
        self.state = state
        self.mt5 = None
        self.running = False
        self._thread = None
        self._db_conn = None
        self._init_db()

    def _init_db(self):
        """Initialize SQLite for tick storage."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._db_lock = threading.Lock()
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                spread REAL NOT NULL,
                volume INTEGER DEFAULT 0,
                timestamp REAL NOT NULL,
                dt TEXT NOT NULL
            )
        """)
        self._db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticks_sym_ts ON ticks(symbol, timestamp)
        """)
        self._db_conn.commit()

    def connect(self) -> bool:
        """Connect to MT5 via mt5linux bridge."""
        try:
            self.mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
            if not self.mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
                log.error("MT5 initialize failed")
                return False
            if not self.mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                log.error("MT5 login failed: %s", self.mt5.last_error())
                return False
            info = self.mt5.account_info()
            log.info("MT5 connected: %s | Balance: $%.2f", info.name, info.balance)
            self.state.update_agent("balance", float(info.balance))
            self.state.update_agent("equity", float(info.equity))
            self.state.update_agent("peak_equity", float(info.equity))

            # Update symbol configs with real tick_value/digits
            for sym, cfg in SYMBOLS.items():
                si = self.mt5.symbol_info(sym)
                if si:
                    cfg.digits = si.digits
                    cfg.tick_value = float(si.trade_tick_value) if si.trade_tick_value else 1.0
                    cfg.volume_min = float(si.volume_min) if si.volume_min else 0.01
                    cfg.volume_max = float(si.volume_max) if si.volume_max else 10.0
                    cfg.volume_step = float(si.volume_step) if si.volume_step else 0.01
                    log.info("[%s] digits=%d tick_value=%.4f vol_min=%.2f",
                             sym, cfg.digits, cfg.tick_value, cfg.volume_min)

            # Load initial candles from MT5
            self._load_initial_candles()
            return True
        except Exception as e:
            log.error("MT5 connect error: %s", e)
            return False

    def _load_initial_candles(self):
        """Load historical candles from MT5 for all symbols/timeframes."""
        tf_map = {1: 1, 5: 5, 15: 15, 60: 16385}
        for sym in SYMBOLS:
            for tf in TIMEFRAMES:
                try:
                    mt5_tf = tf_map.get(tf, tf)
                    rates = self.mt5.copy_rates_from_pos(sym, mt5_tf, 0, CANDLE_WINDOW)
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                        self.state.update_candles(sym, tf, df)
                        log.info("[%s] Loaded %d M%d candles", sym, len(df), tf)
                except Exception as e:
                    log.warning("[%s] Failed loading M%d candles: %s", sym, tf, e)

            # Calculate initial indicators
            self._calculate_indicators(sym)

    def start(self):
        """Start tick streaming in background thread."""
        self.running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True, name="TickStreamer")
        self._thread.start()
        log.info("Tick streamer started (interval=%dms)", TICK_INTERVAL_MS)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.mt5:
            self.mt5.shutdown()
        if self._db_conn:
            self._db_conn.close()
        log.info("Tick streamer stopped")

    def _stream_loop(self):
        """Main tick streaming loop."""
        interval = TICK_INTERVAL_MS / 1000.0
        tick_batch = []

        while self.running:
            loop_start = time.time()
            try:
                for sym in SYMBOLS:
                    tick = self.mt5.symbol_info_tick(sym)
                    if tick is None:
                        continue

                    now = time.time()
                    td = TickData(
                        symbol=sym,
                        bid=tick.bid,
                        ask=tick.ask,
                        t=now,
                        volume=int(tick.volume) if hasattr(tick, 'volume') else 0,
                    )
                    self.state.update_tick(td)

                    # Accumulate into candle builders
                    self._accumulate_tick(td)

                    # Queue for DB insert
                    tick_batch.append((
                        sym, float(tick.bid), float(tick.ask),
                        float(tick.ask - tick.bid), int(td.volume),
                        now, datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                    ))

                # Batch insert ticks to DB every 10 seconds
                if len(tick_batch) >= 80:  # ~4 symbols * 2 ticks/sec * 10s
                    self._flush_ticks(tick_batch)
                    tick_batch = []

                # Refresh candles from MT5 every 5 seconds
                if int(loop_start) % 5 == 0:
                    self._refresh_candles()

                # Update account info
                try:
                    info = self.mt5.account_info()
                    if info:
                        self.state.update_agent("equity", float(info.equity))
                        self.state.update_agent("balance", float(info.balance))
                        self.state.update_agent("profit", float(info.profit))
                        peak = self.state.get_agent_state().get("peak_equity", float(info.equity))
                        if float(info.equity) > peak:
                            self.state.update_agent("peak_equity", float(info.equity))
                            peak = float(info.equity)
                        dd_pct = ((peak - float(info.equity)) / peak * 100) if peak > 0 else 0
                        self.state.update_agent("dd_pct", round(dd_pct, 2))
                except Exception:
                    pass

            except Exception as e:
                log.error("Tick stream error: %s", e)
                time.sleep(2)
                try:
                    self.connect()
                except Exception:
                    pass

            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _accumulate_tick(self, tick: TickData):
        """Accumulate tick into candle builders for each timeframe."""
        mid = (tick.bid + tick.ask) / 2.0
        for tf in TIMEFRAMES:
            key = (tick.symbol, tf)
            tf_seconds = tf * 60
            candle_start = int(tick.time // tf_seconds) * tf_seconds

            acc = self.state._candle_acc.get(key)
            if acc is None or acc["start_time"] != candle_start:
                # New candle period — save old one if exists
                if acc is not None:
                    self._finalize_candle(tick.symbol, tf, acc)
                self.state._candle_acc[key] = {
                    "start_time": candle_start,
                    "open": mid,
                    "high": mid,
                    "low": mid,
                    "close": mid,
                    "tick_volume": 1,
                }
            else:
                acc["high"] = max(acc["high"], mid)
                acc["low"] = min(acc["low"], mid)
                acc["close"] = mid
                acc["tick_volume"] += 1

    def _finalize_candle(self, symbol, tf, acc):
        """Append completed candle to the dataframe."""
        df = self.state.get_candles(symbol, tf)
        new_row = pd.DataFrame([{
            "time": pd.Timestamp(acc["start_time"], unit="s", tz="UTC"),
            "open": acc["open"],
            "high": acc["high"],
            "low": acc["low"],
            "close": acc["close"],
            "tick_volume": acc["tick_volume"],
            "spread": 0,
            "real_volume": 0,
        }])
        if df is not None and len(df) > 0:
            df = pd.concat([df, new_row], ignore_index=True)
            if len(df) > CANDLE_WINDOW:
                df = df.iloc[-CANDLE_WINDOW:]
        else:
            df = new_row
        self.state.update_candles(symbol, tf, df)

    def _refresh_candles(self):
        """Refresh candles from MT5 and recalculate indicators."""
        tf_map = {1: 1, 5: 5, 15: 15, 60: 16385}
        for sym in SYMBOLS:
            try:
                for tf in TIMEFRAMES:
                    mt5_tf = tf_map.get(tf, tf)
                    rates = self.mt5.copy_rates_from_pos(sym, mt5_tf, 0, CANDLE_WINDOW)
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                        self.state.update_candles(sym, tf, df)
                self._calculate_indicators(sym)
            except Exception as e:
                log.warning("[%s] Refresh failed: %s", sym, e)

    def _calculate_indicators(self, symbol):
        """Calculate real-time indicators from candle data."""
        # Use M15 as primary timeframe for indicators
        df = self.state.get_candles(symbol, 15)
        if df is None or len(df) < 50:
            # Fall back to M5
            df = self.state.get_candles(symbol, 5)
            if df is None or len(df) < 50:
                return

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        ind = {}

        # EMA 20, 50, 200
        ind["ema20"] = float(self._ema(close, 20)[-1])
        ind["ema50"] = float(self._ema(close, 50)[-1])
        if len(close) >= 200:
            ind["ema200"] = float(self._ema(close, 200)[-1])
        else:
            ind["ema200"] = float(self._ema(close, len(close))[-1])

        # ATR(14)
        atr = self._atr(high, low, close, 14)
        ind["atr"] = float(atr[-1]) if len(atr) > 0 else 0.0

        # RSI(14)
        rsi = self._rsi(close, 14)
        ind["rsi"] = float(rsi[-1]) if len(rsi) > 0 else 50.0

        # MACD
        macd_line, signal_line, histogram = self._macd(close)
        ind["macd"] = float(macd_line[-1])
        ind["macd_signal"] = float(signal_line[-1])
        ind["macd_hist"] = float(histogram[-1])

        # SuperTrend
        st, st_dir = self._supertrend(high, low, close, 10, 3.0)
        ind["supertrend"] = float(st[-1])
        ind["supertrend_dir"] = int(st_dir[-1])

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = self._bollinger(close, 20, 2.0)
        ind["bb_upper"] = float(bb_upper[-1])
        ind["bb_mid"] = float(bb_mid[-1])
        ind["bb_lower"] = float(bb_lower[-1])

        # VWAP approximation (using typical price * volume)
        if "tick_volume" in df.columns:
            vol = df["tick_volume"].values.astype(float)
            typical = (high + low + close) / 3.0
            cum_vol = np.cumsum(vol[-60:])
            cum_tpv = np.cumsum((typical * vol)[-60:])
            if cum_vol[-1] > 0:
                ind["vwap"] = float(cum_tpv[-1] / cum_vol[-1])
            else:
                ind["vwap"] = float(close[-1])
        else:
            ind["vwap"] = float(close[-1])

        # ADX
        adx = self._adx(high, low, close, 14)
        ind["adx"] = float(adx[-1]) if len(adx) > 0 else 0.0

        self.state.update_indicators(symbol, ind)

    # ═══ INDICATOR MATH ═══

    @staticmethod
    def _ema(data, period):
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(data, dtype=float)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _atr(high, low, close, period):
        n = len(close)
        if n < 2:
            return np.array([0.0])
        tr = np.zeros(n - 1)
        for i in range(1, n):
            tr[i - 1] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        if len(tr) < period:
            return np.array([np.mean(tr)]) if len(tr) > 0 else np.array([0.0])
        atr = np.zeros(len(tr))
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr[period - 1:]

    @staticmethod
    def _rsi(close, period):
        n = len(close)
        if n < period + 1:
            return np.array([50.0])
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        rsi_vals = []
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi_vals.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_vals.append(100.0 - 100.0 / (1.0 + rs))
        return np.array(rsi_vals) if rsi_vals else np.array([50.0])

    @staticmethod
    def _macd(close, fast=12, slow=26, signal=9):
        ema_fast = TickStreamer._ema(close, fast)
        ema_slow = TickStreamer._ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TickStreamer._ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _supertrend(high, low, close, period=10, multiplier=3.0):
        n = len(close)
        atr = TickStreamer._atr(high, low, close, period)
        if len(atr) == 0:
            return np.full(n, close[-1]), np.ones(n, dtype=int)
        # Pad ATR to match close length
        pad = n - len(atr)
        atr_full = np.concatenate([np.full(pad, atr[0]), atr])

        hl2 = (high + low) / 2.0
        upper = hl2 + multiplier * atr_full
        lower = hl2 - multiplier * atr_full
        st = np.zeros(n)
        direction = np.ones(n, dtype=int)  # 1 = bullish, -1 = bearish

        st[0] = upper[0]
        direction[0] = 1

        for i in range(1, n):
            if close[i - 1] <= st[i - 1]:
                # Was bearish
                st[i] = upper[i]
                if close[i] > upper[i]:
                    st[i] = lower[i]
                    direction[i] = 1
                else:
                    direction[i] = -1
            else:
                # Was bullish
                st[i] = lower[i]
                if close[i] < lower[i]:
                    st[i] = upper[i]
                    direction[i] = -1
                else:
                    direction[i] = 1
                    if lower[i] > st[i - 1]:
                        st[i] = lower[i]
                    else:
                        st[i] = st[i - 1]

        return st, direction

    @staticmethod
    def _bollinger(close, period=20, std_dev=2.0):
        n = len(close)
        mid = np.full(n, close[-1])
        upper = np.full(n, close[-1])
        lower = np.full(n, close[-1])
        for i in range(period - 1, n):
            window = close[i - period + 1:i + 1]
            m = np.mean(window)
            s = np.std(window)
            mid[i] = m
            upper[i] = m + std_dev * s
            lower[i] = m - std_dev * s
        return upper, mid, lower

    @staticmethod
    def _adx(high, low, close, period=14):
        n = len(close)
        if n < period + 1:
            return np.array([25.0])

        up_move = np.diff(high)
        down_move = -np.diff(low)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr = TickStreamer._atr(high, low, close, period)
        if len(atr) == 0:
            return np.array([25.0])

        # Smooth DM
        plus_di = np.zeros(len(plus_dm))
        minus_di = np.zeros(len(minus_dm))
        plus_di[period - 1] = np.mean(plus_dm[:period])
        minus_di[period - 1] = np.mean(minus_dm[:period])

        for i in range(period, len(plus_dm)):
            plus_di[i] = (plus_di[i - 1] * (period - 1) + plus_dm[i]) / period
            minus_di[i] = (minus_di[i - 1] * (period - 1) + minus_dm[i]) / period

        # ADX
        pad = len(plus_di) - len(atr)
        atr_aligned = np.concatenate([np.full(max(0, pad), atr[0]), atr])[:len(plus_di)]

        with np.errstate(divide='ignore', invalid='ignore'):
            pdi = np.where(atr_aligned > 0, 100 * plus_di / atr_aligned, 0)
            mdi = np.where(atr_aligned > 0, 100 * minus_di / atr_aligned, 0)
            dx = np.where((pdi + mdi) > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0)

        adx = np.zeros_like(dx)
        start = min(2 * period - 1, len(dx) - 1)
        if start >= period:
            adx[start] = np.mean(dx[start - period + 1:start + 1])
            for i in range(start + 1, len(dx)):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
        else:
            adx[-1] = np.mean(dx[-period:]) if len(dx) >= period else 25.0

        return adx[start:] if start < len(adx) else np.array([25.0])

    def _flush_ticks(self, batch):
        """Batch insert ticks into SQLite."""
        if not batch:
            return
        with self._db_lock:
            try:
                self._db_conn.executemany(
                    "INSERT INTO ticks (symbol, bid, ask, spread, volume, timestamp, dt) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                self._db_conn.commit()
            except Exception as e:
                log.warning("DB flush error: %s", e)

    def get_training_ticks(self, symbol, limit=100000):
        """Read ticks from DB for ML training."""
        with self._db_lock:
            df = pd.read_sql_query(
                "SELECT * FROM ticks WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
                self._db_conn, params=(symbol, limit),
            )
        return df.sort_values("timestamp").reset_index(drop=True) if len(df) > 0 else df
