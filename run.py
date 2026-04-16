#!/usr/bin/env python3
"""
Beast Trader — Entry Point.
Starts tick streamer, ML model, agent brain, and dashboard.

Usage: python run.py [--train]
  --train   Train models before starting agent (takes a few minutes)
"""
import sys
import signal
import logging
import threading
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "logs" / "beast.log", mode="a"),
    ],
)
log = logging.getLogger("beast")

# Ensure logs dir exists
(Path(__file__).parent / "logs").mkdir(exist_ok=True)

from config import SYMBOLS, MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, TRADING_MODE
from data.tick_streamer import TickStreamer, SharedState
from data.feature_engine import FeatureEngine
from models.signal_model import SignalModel
from execution.executor import Executor
from agent.brain import AgentBrain
from agent.scalp_brain import ScalpBrain
from dashboard.app import init_dashboard, run_dashboard


def main():
    parser = argparse.ArgumentParser(description="Beast Trader — Tick-Level ML Trading Agent")
    parser.add_argument("--train", action="store_true", help="Train models before starting")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  B.E.A.S.T — ML Trading Agent v1.0")
    log.info("  Account: %d | Server: %s", MT5_LOGIN, MT5_SERVER)
    log.info("  Bridge: %s:%d", MT5_HOST, MT5_PORT)
    log.info("  Mode: %s", TRADING_MODE)
    log.info("  Symbols: %s", ", ".join(SYMBOLS.keys()))
    log.info("=" * 60)

    # ═══ 1. SHARED STATE ═══
    state = SharedState()

    # ═══ 2. TICK STREAMER (connects to MT5) ═══
    streamer = TickStreamer(state)
    if not streamer.connect():
        log.error("Failed to connect to MT5. Is the bridge running on port %d?", MT5_PORT)
        sys.exit(1)

    # ═══ 3. FEATURE ENGINE ═══
    feature_engine = FeatureEngine(state)

    # ═══ 4. ML MODEL ═══
    model = SignalModel()

    if args.train:
        log.info("Training models (this may take a few minutes)...")
        model.train_all(streamer.mt5, feature_engine)
        log.info("Training complete.")
    else:
        model.load_all()
        # If no models found, train them
        missing = [sym for sym in SYMBOLS if not model.has_model(sym)]
        if missing:
            log.info("No saved models for %s — training now...", ", ".join(missing))
            for sym in missing:
                try:
                    model.train(sym, streamer.mt5, feature_engine)
                except Exception as e:
                    log.error("[%s] Training failed: %s", sym, e)

    # ═══ 5. EXECUTOR + VOL MODEL ═══
    executor = Executor(streamer.mt5, state)
    # Wire vol model for dynamic SL
    try:
        from models.vol_model import VolModel
        vol_model = VolModel()
        vol_model.load()
        executor._vol_model = vol_model
        log.info("Volatility model loaded for dynamic SL")
    except Exception as e:
        log.warning("Vol model not loaded: %s", e)

    # ═══ 6. AGENT BRAIN (swing) ═══
    brain = None
    if TRADING_MODE in ("swing", "hybrid"):
        brain = AgentBrain(state, streamer.mt5, executor, meta_model=model)

    # ═══ 6b. SCALP BRAIN (M5 scalper) ═══
    scalp_brain = None
    if TRADING_MODE in ("scalp", "hybrid"):
        scalp_brain = ScalpBrain(state, streamer.mt5, executor)

    # ═══ 7. DASHBOARD ═══
    init_dashboard(state, executor)

    # ═══ SIGNAL HANDLER ═══
    shutdown_event = threading.Event()

    def handle_shutdown(signum, frame):
        log.info("Shutdown signal received...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # ═══ START ALL COMPONENTS ═══
    try:
        # Start tick streamer
        streamer.start()
        log.info("Tick streamer started")

        # Wait for initial data
        import time
        log.info("Waiting 3s for initial tick data...")
        time.sleep(3)

        # Start agent brain(s)
        if brain:
            brain.start()
            log.info("Swing brain started (mode=%s)", TRADING_MODE)

        if scalp_brain:
            scalp_brain.start()
            log.info("Scalp brain started (mode=%s)", TRADING_MODE)

        # Start dashboard (in background thread)
        dash_thread = threading.Thread(target=run_dashboard, daemon=True, name="Dashboard")
        dash_thread.start()
        log.info("Dashboard: http://localhost:8888")

        log.info("")
        log.info("All systems online. Press Ctrl+C to stop.")
        log.info("")

        # Main thread waits for shutdown
        shutdown_event.wait()

    except Exception as e:
        log.error("Fatal error: %s", e)
    finally:
        log.info("Shutting down...")
        if scalp_brain:
            scalp_brain.stop()
        if brain:
            brain.stop()
        streamer.stop()
        log.info("Beast Trader stopped.")


if __name__ == "__main__":
    main()
