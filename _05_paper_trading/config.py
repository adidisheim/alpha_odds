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
MAX_BACK_ODDS = 1000.0      # Maximum odds to consider
COMMISSION_RATE = 0.075      # Betfair commission (7.5% for AU greyhounds)
STAKE_SIZE = 25.0            # Notional stake per bet in AUD

# ── Kill switch ──
MAX_DAILY_LOSS = 500.0       # Stop trading if daily loss exceeds this (in AUD, at STAKE_SIZE)

# ── Scheduling ──
DECISION_SECONDS_BEFORE_START = 20  # When to compute features and decide
STREAM_SUBSCRIBE_MINUTES_BEFORE = 11  # When to start streaming a market
MARKET_DISCOVERY_INTERVAL_SECONDS = 120  # How often to poll for new markets
MARKET_CLEANUP_MINUTES_AFTER = 30  # Cleanup finished markets after this

# ── Ensemble weights (from _10_ultimate_ensemble.py optimization) ──
V1_WEIGHT = 0.20
V2_WEIGHT = 0.80
V1_TOP_N = 7   # Top N models per t_def for V1
V2_TOP_N = 15  # Top N models per t_def for V2

# ── Market filters ──
EVENT_TYPE_ID = "4339"  # Greyhound racing
MARKET_COUNTRIES = ["AU"]
MARKET_TYPE = "WIN"

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
