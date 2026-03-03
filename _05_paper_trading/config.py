"""
Paper Trading Configuration — loads credentials from .env, defines paths and thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Betfair API Credentials ──
BETFAIR_USERNAME = os.getenv("BETFAIR_USERNAME", "")
BETFAIR_PASSWORD = os.getenv("BETFAIR_PASSWORD", "")
BETFAIR_APP_KEY = os.getenv("BETFAIR_APP_KEY", "")
BETFAIR_CERT_PATH = os.getenv("BETFAIR_CERT_PATH", str(PROJECT_ROOT / "certs" / "client-2048.crt"))
BETFAIR_KEY_PATH = os.getenv("BETFAIR_KEY_PATH", str(PROJECT_ROOT / "certs" / "client-2048.key"))

# ── Directories ──
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "paper_trading_logs"

# ── Time definitions (mirrors parameters.py) ──
# Each t_def maps to: (tm3, tm2, tm1, t0, tp1) in seconds before scheduled start
TIME_DEFS = {
    0: {"tm3": 600, "tm2": 180, "tm1": 75, "t0": 60, "tp1": 10},
    1: {"tm3": 600, "tm2": 240, "tm1": 135, "t0": 120, "tp1": 10},
    2: {"tm3": 300, "tm2": 120, "tm1": 35, "t0": 30, "tp1": 10},
    3: {"tm3": 300, "tm2": 120, "tm1": 25, "t0": 20, "tp1": 5},
}

# ── Trading thresholds ──
EDGE_THRESHOLD = 0.03       # Minimum edge to place a paper bet
MIN_BACK_ODDS = 1.01        # Minimum odds to consider
MAX_BACK_ODDS = 50.0        # Maximum odds to consider (backtest avg=5; model extrapolates poorly beyond ~30:1)
MIN_MARKET_PROB = 0.02      # Minimum market implied probability (filters illiquid runners with unreliable features)
COMMISSION_RATE = 0.075      # Betfair commission (7.5% for AU greyhounds)
STAKE_SIZE = 0.01            # $0.01 bets for paper trading (data collection)

# ── Kill switch ──
MAX_DAILY_LOSS = 20.0        # Stop trading if daily loss exceeds this (in AUD)

# ── Scheduling ──
DECISION_SECONDS_BEFORE_START = 20  # When to compute features and decide
STREAM_SUBSCRIBE_MINUTES_BEFORE = 15  # When to start streaming a market (15min = ~858s pre-off with ~42s offset, 258s margin beyond tm3=600s)
MARKET_DISCOVERY_INTERVAL_SECONDS = 120  # How often to poll for new markets
MARKET_CLEANUP_MINUTES_AFTER = 30  # Cleanup finished markets after this

# ── Ensemble weights (from _10_ultimate_ensemble.py optimization) ──
# Backtest optimal: V1=0.20, V2=0.80 (was 0.30/0.70 which inflated prob sums)
V1_WEIGHT = 0.20
V2_WEIGHT = 0.80
V1_TOP_N = 7   # Top N models per t_def for V1
V2_TOP_N = 15  # Top N models per t_def for V2

# ── Model Variants (multi-variant paper trading) ──
# Run variants simultaneously to compare live performance.
# v1_key defaults to "v1" (shared). Set explicitly for variants needing different V1 models.
MODEL_VARIANTS = {
    "surface": {"v1_key": "v1_surface", "v2_key": "v2_surface", "n_features": 85},
    "n5": {"v1_key": "v1", "v2_key": "v2_n5", "n_features": 5},
}
PRIMARY_VARIANT = "surface"  # Surface (85 feat) as primary, n5 as comparison

# ── Market filters ──
EVENT_TYPE_ID = "4339"  # Greyhound racing
MARKET_COUNTRIES = ["AU"]
MARKET_TYPE = "WIN"

# ── Email Notifications (via claude-email-mcp Gmail API) ──
EMAIL_TO = "antoinedidisheim@gmail.com"

# ── Backtest Baselines (from OOS cross-t super-ensemble at edge>3%) ──
BACKTEST_BASELINES = {
    "surface": {
        "bets_per_day": 5.7,       # 1705 bets / 298 racing days
        "avg_edge": 0.0682,
        "avg_odds": 4.13,
        "avg_model_prob": 0.4189,
        "avg_market_prob": 0.3507,
        "fill_rate": 0.97,
        "roi_pct": 26.3,
        "win_rate": 0.4188,
    },
    "n5": {
        "bets_per_day": 4.1,       # 1173 bets / 288 racing days
        "avg_edge": 0.0743,
        "avg_odds": 4.65,
        "avg_model_prob": 0.4006,
        "avg_market_prob": 0.3263,
        "fill_rate": 0.97,
        "roi_pct": 47.8,
        "win_rate": 0.4228,
    },
}

# ── Anomaly Detection ──
ANOMALY_MIN_BETS_FOR_CHECK = 10    # Need this many bets before anomaly checks
ANOMALY_EDGE_Z_THRESHOLD = 2.5    # Z-score threshold for win rate deviation
ANOMALY_BET_RATE_TOLERANCE = 3.0  # Flag if bet rate > 3x backtest
ANOMALY_PAUSE_ON_CRITICAL = True   # Pause trading on critical anomalies
ANOMALY_CHECK_INTERVAL = 600       # Check every 10 minutes (seconds)

# ── Betfair tick table (from _13_fill_simulator.py) ──
BETFAIR_TICKS = [
    (1.01, 2.0, 0.01),
    (2.0, 3.0, 0.02),
    (3.0, 4.0, 0.05),
    (4.0, 6.0, 0.10),
    (6.0, 10.0, 0.20),
    (10.0, 20.0, 0.50),
    (20.0, 30.0, 1.0),
    (30.0, 50.0, 2.0),
    (50.0, 100.0, 5.0),
    (100.0, 1000.0, 10.0),
]

# ── Feature Validation ──
VALIDATION_BASELINES_DIR = Path(__file__).resolve().parent / "baselines"
VALIDATION_MIN_RACES = 20          # Minimum races before validation is meaningful
VALIDATION_NAN_TOLERANCE = 0.10    # abs(live_nan_rate - hist_nan_rate) threshold
VALIDATION_MEAN_Z_THRESHOLD = 3.0  # z-score threshold for mean shift
VALIDATION_STD_RATIO_LOW = 0.3     # min acceptable live_std / hist_std
VALIDATION_STD_RATIO_HIGH = 3.0    # max acceptable live_std / hist_std
VALIDATION_RANGE_VIOLATION_MAX = 0.25  # max fraction of values outside [p5, p95]
VALIDATION_TIMEOUT_HOURS = 2       # Max hours to run in --validate mode
