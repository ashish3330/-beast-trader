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
import logging.handlers
import threading
import argparse
from pathlib import Path

# Ensure logs dir exists
(Path(__file__).parent / "logs").mkdir(exist_ok=True)

# ── Logging setup with rotation ──
# 50 MB per file, keep 10 rotations (= 500 MB max history). Override-able via env
# DRAGON_LOG_BYTES / DRAGON_LOG_BACKUPS for ops convenience.
import os as _os
_LOG_PATH = Path(__file__).parent / "logs" / "dragon.log"
_LOG_BYTES = int(_os.environ.get("DRAGON_LOG_BYTES", str(50 * 1024 * 1024)))
_LOG_BACKUPS = int(_os.environ.get("DRAGON_LOG_BACKUPS", "10"))
_fmt = logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
_stream = logging.StreamHandler()
_stream.setFormatter(_fmt)
_rotating = logging.handlers.RotatingFileHandler(
    _LOG_PATH, maxBytes=_LOG_BYTES, backupCount=_LOG_BACKUPS,
)
_rotating.setFormatter(_fmt)
_root = logging.getLogger()
_root.setLevel(logging.INFO)
# Avoid duplicate handlers on reimport / dev-reload.
for _h in list(_root.handlers):
    _root.removeHandler(_h)
_root.addHandler(_stream)
_root.addHandler(_rotating)
log = logging.getLogger("dragon")

from config import SYMBOLS, MT5_HOST, MT5_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, TRADING_MODE, DB_PATH


def _assert_canonical_db_paths() -> None:
    """Bail out early if the canonical DB paths are missing or a stale legacy
    DB at the repo root has somehow grown bytes — that means something is
    writing to the wrong path and silently fragmenting state.

    Canonical paths (anchored on config.DB_PATH.parent = data/):
        - data/rl_learner.db       (RL learning state — CRITICAL)
        - data/trade_journal.db    (trade history — CRITICAL)
    """
    canonical_dir = DB_PATH.parent
    rl_canon = canonical_dir / "rl_learner.db"
    journal_canon = canonical_dir / "trade_journal.db"
    if not rl_canon.exists():
        log.error("Canonical RL DB missing: %s — refusing to start.", rl_canon)
        sys.exit(2)
    if not journal_canon.exists():
        log.error("Canonical trade journal missing: %s — refusing to start.", journal_canon)
        sys.exit(2)

    repo_root = Path(__file__).resolve().parent
    for legacy_name in ("rl_learner.db", "trade_journal.db"):
        legacy = repo_root / legacy_name
        if legacy.exists() and legacy.stat().st_size > 0:
            log.error(
                "Stale legacy DB at %s is non-empty (%d bytes). Something is "
                "writing to the wrong path. Canonical is %s. Refusing to start.",
                legacy, legacy.stat().st_size, canonical_dir / legacy_name,
            )
            sys.exit(2)


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
from agent.calendar_filter import CalendarFilter
from agent.trade_intelligence import TradeIntelligence
from agent.alerting import Alerter
from agent.metrics import MetricsExporter
try:
    from agent.rl_learner import RLLearner
except ImportError:
    RLLearner = None
try:
    from agent.pattern_learner import PatternLearner
except ImportError:
    PatternLearner = None
try:
    from agent.order_flow import OrderFlowIntel
except ImportError:
    OrderFlowIntel = None
try:
    from agent.level_memory import LevelMemory
except ImportError:
    LevelMemory = None
try:
    from agent.fvg_detector import FVGDetector
except ImportError:
    FVGDetector = None
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

    # Canonical DB-path enforcement (issue #21): catches dual-DB-path footgun
    # before any process starts writing to the wrong location.
    _assert_canonical_db_paths()

    # === 0. PRE-FLIGHT (24/7 hardening) ===
    # Logs warnings only — never blocks startup. Durability stack heals
    # transient issues, but loud warnings at boot make degraded conditions
    # visible instead of silent.
    try:
        from scripts.preflight import run_preflight
        run_preflight(parent_log=log, mt5_port=MT5_PORT, db_path=DB_PATH)
    except Exception as e:
        log.warning("preflight check itself failed: %s", e)

    # === 0. OBSERVABILITY (alerter + metrics) ===
    # Both are opt-in / log-only-by-default and never block trading logic.
    alerter = Alerter()
    metrics = MetricsExporter()
    metrics.start()

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
    # Wire observability handles (no behaviour change — observability only).
    executor._alerter = alerter
    executor._metrics = metrics
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
    calendar = CalendarFilter()
    trade_intel = TradeIntelligence(state, learner)
    learner._trade_intel = trade_intel  # wire for SL re-entry tracking

    # === 6b. RL LEARNING MODULES ===
    rl_learner = RLLearner(state) if RLLearner else None
    pattern_learner = PatternLearner(state) if PatternLearner else None
    order_flow = OrderFlowIntel(state) if OrderFlowIntel else None
    level_memory = LevelMemory() if LevelMemory else None
    fvg_detector = FVGDetector(state) if FVGDetector else None

    # Wire RL + level memory into learning engine so deal sync feeds all modules
    learner._rl_learner = rl_learner
    learner._level_memory = level_memory
    log.info("MasterBrain, ExitIntelligence, LearningEngine, MTFIntelligence, PortfolioRisk, EquityGuardian, SmartEntry, CalendarFilter, TradeIntelligence initialized")
    rl_modules = [m for m in ["RLLearner",
                               "PatternLearner" if pattern_learner else None,
                               "OrderFlowIntel" if order_flow else None,
                               "LevelMemory" if level_memory else None,
                               "FVGDetector" if fvg_detector else None] if m]
    log.info("RL modules initialized: %s", ", ".join(rl_modules))

    # === 7. AGENT BRAIN (swing) ===
    brain = None
    if TRADING_MODE in ("swing", "hybrid"):
        brain = AgentBrain(state, streamer.mt5, executor, meta_model=model,
                           master_brain=master_brain, exit_intelligence=exit_intel,
                           mtf_intelligence=mtf_intel,
                           learning_engine=learner,
                           equity_guardian=guardian,
                           smart_entry=smart_entry,
                           calendar_filter=calendar,
                           trade_intelligence=trade_intel,
                           rl_learner=rl_learner,
                           pattern_learner=pattern_learner,
                           order_flow=order_flow,
                           level_memory=level_memory,
                           fvg_detector=fvg_detector,
                           alerter=alerter,
                           metrics=metrics)

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
        try:
            alerter.stop()
        except Exception as e:
            log.debug("alerter stop: %s", e)
        log.info("Dragon Trader stopped.")


if __name__ == "__main__":
    main()
