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

import numpy as np
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
        for subdir in [
            "ticks", "features", "decisions", "bets", "daily",
            "features_normalized", "features_v2_enriched", "model_details",
        ]:
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

    def log_decisions(self, predictions_df, market_id, venue="", num_runners=0):
        """
        Buffer model predictions for a market.

        Args:
            predictions_df: DataFrame from CrossTEnsemble.predict()
            market_id: for identification
            venue: market venue name
            num_runners: number of active runners in the market
        """
        if predictions_df.empty:
            return
        preds = predictions_df.copy()
        preds["market_id"] = market_id
        preds["timestamp"] = datetime.now(timezone.utc)
        preds["venue"] = venue
        preds["num_runners"] = num_runners
        preds["model_prob_sum"] = preds["model_prob"].sum() if "model_prob" in preds.columns else np.nan
        preds["winner_id"] = np.nan
        self._decision_buffer.extend(preds.to_dict(orient="records"))

    def log_bets(self, bets):
        """Buffer paper bet outcomes."""
        for bet in bets:
            self._bet_buffer.append(bet.to_dict())

    def log_features_normalized(self, feature_results, market_id):
        """
        Save normalized V1 and V2 features to features_normalized/.

        Args:
            feature_results: dict from FeatureComputer.compute(), keyed by t_def
            market_id: for identification
        """
        rows = []
        for t_def, fr in feature_results.items():
            for version, key in [("v1", "features_v1"), ("v2", "features_v2")]:
                if key not in fr:
                    continue
                df = fr[key].copy()
                df["t_def"] = t_def
                df["version"] = version
                df["market_id"] = market_id
                rows.append(df)

        if not rows:
            return

        combined = pd.concat(rows, ignore_index=True)
        today = self._today_str()
        path = self.log_dir / "features_normalized" / f"{today}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, combined], ignore_index=True)
        combined.to_parquet(path)
        logger.debug(f"Saved {len(combined)} normalized feature rows to {path}")

    def log_features_v2_enriched(self, feature_results, market_id):
        """
        Save V2 features with cross-runner features (pre-normalization) to features_v2_enriched/.

        Args:
            feature_results: dict from FeatureComputer.compute(), keyed by t_def
            market_id: for identification
        """
        rows = []
        for t_def, fr in feature_results.items():
            if "features_v2_pre_norm" not in fr:
                continue
            df = fr["features_v2_pre_norm"].copy()
            df["t_def"] = t_def
            df["market_id"] = market_id
            rows.append(df)

        if not rows:
            return

        combined = pd.concat(rows, ignore_index=True)
        today = self._today_str()
        path = self.log_dir / "features_v2_enriched" / f"{today}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, combined], ignore_index=True)
        combined.to_parquet(path)
        logger.debug(f"Saved {len(combined)} V2 enriched feature rows to {path}")

    def log_model_details(self, details_list, market_id, runner_ids):
        """
        Save per-config individual model predictions to model_details/.

        Args:
            details_list: list of dicts with keys: config, version, t_def, predictions
            market_id: for identification
            runner_ids: array of runner IDs (matching prediction order)
        """
        if not details_list:
            return

        rows = []
        timestamp = datetime.now(timezone.utc)
        for detail in details_list:
            preds = detail["predictions"]
            for i, pred in enumerate(preds):
                rows.append({
                    "market_id": market_id,
                    "timestamp": timestamp,
                    "runner_id": runner_ids[i] if i < len(runner_ids) else None,
                    "config_name": detail["config"],
                    "version": detail["version"],
                    "t_def": detail["t_def"],
                    "prediction": pred,
                })

        df = pd.DataFrame(rows)
        today = self._today_str()
        path = self.log_dir / "model_details" / f"{today}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path)
        logger.debug(f"Saved {len(df)} model detail rows to {path}")

    def update_decisions_winner(self, market_id, winner_id):
        """
        After settlement, update the decisions parquet to add winner_id for that market.

        Args:
            market_id: the settled market
            winner_id: the winning runner's selection_id
        """
        today = self._today_str()
        path = self.log_dir / "decisions" / f"{today}.parquet"
        if not path.exists():
            logger.debug(f"No decisions file to update for {market_id}")
            return

        try:
            df = pd.read_parquet(path)
            mask = df["market_id"] == market_id
            if mask.any():
                df.loc[mask, "winner_id"] = winner_id
                df.to_parquet(path)
                logger.debug(f"Updated winner_id={winner_id} for {mask.sum()} rows in {market_id}")
        except Exception as e:
            logger.warning(f"Failed to update decisions winner for {market_id}: {e}")

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
