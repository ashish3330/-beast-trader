#!/usr/bin/env python3
"""
Dragon Trader — Entry Point.
Starts tick streamer, ML model, agent brain, master brain, and dashboard.

Usage: python run.py [--train]
  --train   Train models before starting agent (takes a few minutes)
"""
import sys
import signal
import logging
import threading
import argparse
from pathlib import Path

# Ensure logs dir exists
(Path(__file__).parent / "logs").mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "logs" / "dragon.log", mode="a"),
    ],
)
log = logging.getLogger("dragon")

from config import SYMBOLS, MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, TRADING_MODE
from data.tick_streamer import TickStreamer, SharedState
from data.feature_engine import FeatureEngine
from models.signal_model import SignalModel
from execution.executor import Executor
from agent.brain import AgentBrain
from agent.scalp_brain import ScalpBrain
from agent.master_brain import MasterBrain
from agent.exit_intelligence import ExitIntelligence
from agent.learning_engine import LearningEngine
from agent.mtf_intelligence import MTFIntelligence
from agent.portfolio_risk import PortfolioRiskModel
from agent.equity_guardian import EquityGuardian
from agent.smart_entry import SmartEntry
from dashboard.app import init_dashboard, run_dashboard


def main():
    parser = argparse.ArgumentParser(description="Dragon Trader — Ultra-Conservative ML Trading Agent")
    parser.add_argument("--train", action="store_true", help="Train models before starting")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  D.R.A.G.O.N — Ultra-Conservative ML Trading Agent v2.0")
    log.info("  Account: %d | Server: %s", MT5_LOGIN, MT5_SERVER)
    log.info("  Bridge: %s:%d", MT5_HOST, MT5_PORT)
    log.info("  Mode: %s", TRADING_MODE)
    log.info("  Symbols: %s", ", ".join(SYMBOLS.keys()))
    log.info("=" * 60)

    # === 1. SHARED STATE ===
    state = SharedState()

    # === 2. TICK STREAMER (connects to MT5) ===
    streamer = TickStreamer(state)
    if not streamer.connect():
        log.error("Failed to connect to MT5. Is the bridge running on port %d?", MT5_PORT)
        sys.exit(1)

    # === 3. FEATURE ENGINE ===
    feature_engine = FeatureEngine(state)

    # === 4. ML MODEL ===
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

    # === 5. EXECUTOR + VOL MODEL ===
    executor = Executor(streamer.mt5, state)
    # Wire vol model for dynamic SL
    try:
        from models.vol_model import VolatilityModel
        vol_model = VolatilityModel()
        vol_model.load_all()
        executor._vol_model = vol_model
        log.info("Volatility model loaded for dynamic SL")
    except Exception as e:
        log.warning("Vol model not loaded: %s", e)

    # === 6. MASTER BRAIN + EXIT INTELLIGENCE + LEARNING ENGINE ===
    master_brain = MasterBrain(state, streamer.mt5, executor, meta_model=model)
    exit_intel = ExitIntelligence(state, executor)
    learner = LearningEngine(state, master_brain, executor)
    master_brain.learning_engine = learner  # wire adaptive risk
    learner.set_meta_model(model)           # wire for auto-retrain
    learner.set_mt5(streamer.mt5)           # wire MT5 for data fetch
    mtf_intel = MTFIntelligence(state)
    master_brain.mtf_intelligence = mtf_intel  # wire MTF into MasterBrain decisions
    portfolio_risk = PortfolioRiskModel(state, executor)
    master_brain.portfolio_risk = portfolio_risk  # wire portfolio risk gate
    guardian = EquityGuardian(state, executor)
    smart_entry = SmartEntry(state)
    log.info("MasterBrain, ExitIntelligence, LearningEngine, MTFIntelligence, PortfolioRisk, EquityGuardian, SmartEntry initialized")

    # === 7. AGENT BRAIN (swing) ===
    brain = None
    if TRADING_MODE in ("swing", "hybrid"):
        brain = AgentBrain(state, streamer.mt5, executor, meta_model=model,
                           master_brain=master_brain, exit_intelligence=exit_intel,
                           mtf_intelligence=mtf_intel,
                           learning_engine=learner,
                           equity_guardian=guardian,
                           smart_entry=smart_entry)

    # === 7b. SCALP BRAIN (M5 scalper) ===
    scalp_brain = None
    if TRADING_MODE in ("scalp", "hybrid"):
        scalp_brain = ScalpBrain(state, streamer.mt5, executor, master_brain=master_brain)

    # === 8. DASHBOARD ===
    init_dashboard(state, executor)

    # === SIGNAL HANDLER ===
    shutdown_event = threading.Event()

    def handle_shutdown(signum, frame):
        log.info("Shutdown signal received...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # === START ALL COMPONENTS ===
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

        # Start learning engine
        learner.start()
        log.info("Learning Engine started")

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
        learner.stop()
        if scalp_brain:
            scalp_brain.stop()
        if brain:
            brain.stop()
        streamer.stop()
        log.info("Dragon Trader stopped.")


if __name__ == "__main__":
    main()
