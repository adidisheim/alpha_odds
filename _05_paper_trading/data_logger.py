"""
Data Logger — persists tick data, features, decisions, bets, and daily summaries.

Directory structure:
  paper_trading_logs/
    ticks/YYYY-MM-DD/          Raw tick parquets per market
    features/YYYY-MM-DD.parquet  All features computed
    decisions/YYYY-MM-DD.parquet All model predictions
    bets/YYYY-MM-DD.parquet      Paper bet outcomes
    daily/YYYY-MM-DD.json        Daily summary
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import LOG_DIR

logger = logging.getLogger(__name__)


class DataLogger:
    """Handles all paper trading data persistence."""

    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir or LOG_DIR)
        self._ensure_dirs()
        self._tick_buffers = {}      # market_id -> list of tick dicts
        self._feature_buffer = []    # list of feature row dicts
        self._decision_buffer = []   # list of prediction row dicts
        self._bet_buffer = []        # list of bet dicts

    def _ensure_dirs(self):
        """Create log directory structure."""
        for subdir in ["ticks", "features", "decisions", "bets", "daily"]:
            (self.log_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _today_str(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def log_tick(self, market_id, tick_dict):
        """Buffer a single tick for later flushing."""
        if market_id not in self._tick_buffers:
            self._tick_buffers[market_id] = []
        self._tick_buffers[market_id].append(tick_dict)

    def log_ticks_from_cache(self, market_cache):
        """
        Log all ticks from a MarketCache.

        Saves ticks in the same schema as win_*.parquet from the historical pipeline.
        """
        df = market_cache.to_dataframe()
        if df.empty:
            return

        today = self._today_str()
        tick_dir = self.log_dir / "ticks" / today
        tick_dir.mkdir(parents=True, exist_ok=True)

        safe_name = market_cache.file_name.replace("/", "_").replace(".", "_")
        path = tick_dir / f"{safe_name}.parquet"
        df.to_parquet(path)
        logger.debug(f"Saved {len(df)} ticks to {path}")

    def log_features(self, features_dict, market_id):
        """
        Buffer computed features for a market.

        Args:
            features_dict: dict from FeatureComputer.compute(), keyed by t_def
            market_id: for identification
        """
        for t_def, fr in features_dict.items():
            if "raw_features" in fr:
                raw = fr["raw_features"].copy()
                raw["t_def"] = t_def
                raw["market_id"] = market_id
                self._feature_buffer.extend(raw.to_dict(orient="records"))

    def log_decisions(self, predictions_df, market_id):
        """
        Buffer model predictions for a market.

        Args:
            predictions_df: DataFrame from CrossTEnsemble.predict()
            market_id: for identification
        """
        if predictions_df.empty:
            return
        preds = predictions_df.copy()
        preds["market_id"] = market_id
        preds["timestamp"] = datetime.now(timezone.utc)
        self._decision_buffer.extend(preds.to_dict(orient="records"))

    def log_bets(self, bets):
        """Buffer paper bet outcomes."""
        for bet in bets:
            self._bet_buffer.append(bet.to_dict())

    def log_daily_summary(self, summary_dict):
        """Save daily summary as JSON."""
        today = self._today_str()
        path = self.log_dir / "daily" / f"{today}.json"
        with open(path, "w") as f:
            json.dump(summary_dict, f, indent=2, default=str)
        logger.info(f"Daily summary saved to {path}")

    def flush(self):
        """Flush all buffered data to disk."""
        today = self._today_str()

        # Flush features
        if self._feature_buffer:
            path = self.log_dir / "features" / f"{today}.parquet"
            df = pd.DataFrame(self._feature_buffer)
            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_parquet(path)
            self._feature_buffer.clear()
            logger.debug(f"Flushed {len(df)} feature rows to {path}")

        # Flush decisions
        if self._decision_buffer:
            path = self.log_dir / "decisions" / f"{today}.parquet"
            df = pd.DataFrame(self._decision_buffer)
            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_parquet(path)
            self._decision_buffer.clear()
            logger.debug(f"Flushed {len(df)} decision rows to {path}")

        # Flush bets
        if self._bet_buffer:
            path = self.log_dir / "bets" / f"{today}.parquet"
            df = pd.DataFrame(self._bet_buffer)
            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_parquet(path)
            self._bet_buffer.clear()
            logger.debug(f"Flushed {len(df)} bet rows to {path}")

    def flush_market_ticks(self, market_id):
        """Flush ticks for a specific market."""
        if market_id not in self._tick_buffers:
            return

        today = self._today_str()
        tick_dir = self.log_dir / "ticks" / today
        tick_dir.mkdir(parents=True, exist_ok=True)

        ticks = self._tick_buffers.pop(market_id)
        if ticks:
            df = pd.DataFrame(ticks)
            safe_name = market_id.replace(".", "_")
            path = tick_dir / f"{safe_name}.parquet"
            df.to_parquet(path)
            logger.debug(f"Flushed {len(df)} ticks for {market_id} to {path}")
