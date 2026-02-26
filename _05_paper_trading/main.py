"""
Paper Trading Orchestrator — main event loop.

Flow:
  1. Every 2 min: poll Betfair API for upcoming AU greyhound WIN markets
  2. Subscribe to streaming for markets ~11 min before start
  3. Accumulate ticks in MarketCache
  4. At scheduled_start - 20s: compute features → run ensemble → evaluate edge → log
  5. After settlement: record outcomes, compute P&L
  6. Cleanup finished markets after 30 min

Usage:
    python main.py [--dry-run]
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import betfairlightweight

from betfair_client import BetfairClient
from config import (
    DECISION_SECONDS_BEFORE_START,
    LOG_DIR,
    MARKET_CLEANUP_MINUTES_AFTER,
    MARKET_DISCOVERY_INTERVAL_SECONDS,
    MODELS_DIR,
    STREAM_SUBSCRIBE_MINUTES_BEFORE,
)
from data_logger import DataLogger
from feature_engine import FeatureComputer, SavedNormalizerParams
from market_cache import MarketCache
from model_engine import CrossTEnsemble
from paper_trader import PaperTrader

logger = logging.getLogger(__name__)


def load_normalization_params(models_dir):
    """Load saved normalization parameters for all t_defs."""
    norm_params = {}
    for t_def in range(4):
        norm_params[t_def] = {}

        # V1 normalization
        v1_path = models_dir / "normalization" / f"feature_normalization_params_v1_t{t_def}.parquet"
        if v1_path.exists():
            df = pd.read_parquet(v1_path)
            norm_params[t_def]["v1"] = SavedNormalizerParams(df)
            logger.info(f"Loaded V1 normalization params for t{t_def}: {len(df)} features")

        # V2 normalization
        v2_path = models_dir / "normalization" / f"feature_normalization_params_v2_t{t_def}.parquet"
        if v2_path.exists():
            df = pd.read_parquet(v2_path)
            norm_params[t_def]["v2"] = SavedNormalizerParams(df)
            logger.info(f"Loaded V2 normalization params for t{t_def}: {len(df)} features")

    return norm_params


def setup_logging(dry_run=False):
    """Configure logging to both file and console."""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prefix = "dry_run_" if dry_run else ""
    log_file = log_dir / f"{prefix}{today}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def derive_file_name(catalogue):
    """Derive a file_name from market catalogue (mirrors BZ2 file naming)."""
    return f"{catalogue.market_id}.bz2"


def get_local_dow(market_start_time):
    """Get day of week in AEST/AEDT (approximate: UTC+10)."""
    aest = market_start_time + timedelta(hours=10)
    return aest.weekday()


class PaperTradingEngine:
    """Main engine that orchestrates the paper trading loop."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.client = BetfairClient()
        self.ensemble = CrossTEnsemble()
        self.paper_trader = PaperTrader()
        self.data_logger = DataLogger()
        self.feature_computer = None  # Initialized after loading norm params

        # Market tracking
        self.market_caches = {}  # market_id -> MarketCache
        self.known_markets = set()  # market_ids we've already seen
        self.subscribed_markets = set()  # market_ids currently streaming

        # Streaming
        self._stream = None
        self._stream_thread = None
        self._output_queue = None
        self._listener = None

        # Timing
        self._last_discovery = 0.0
        self._last_flush = 0.0
        self._running = True

    def setup(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("Paper Trading System Starting")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("=" * 60)

        # Load models
        logger.info("Loading model artifacts...")
        self.ensemble.load()

        # Load normalization params
        models_dir = Path(MODELS_DIR)
        norm_params = load_normalization_params(models_dir)
        self.feature_computer = FeatureComputer(norm_params)

        # Login to Betfair and set up streaming
        if not self.dry_run:
            logger.info("Logging in to Betfair...")
            self.client.login()
            self._setup_streaming()

        logger.info("Setup complete. Entering main loop.")

    def _setup_streaming(self):
        """Initialize the streaming connection with betfairlightweight."""
        import queue
        import threading
        from betfairlightweight.filters import (
            streaming_market_data_filter,
            streaming_market_filter,
        )

        self._output_queue = queue.Queue()
        self._listener = betfairlightweight.StreamListener(
            output_queue=self._output_queue,
            max_latency=2.0,
        )
        self._stream = self.client.create_streaming_connection(self._listener)

        # Start stream in background thread
        self._stream_thread = threading.Thread(
            target=self._stream.start, daemon=True, name="BetfairStream"
        )
        self._stream_thread.start()
        logger.info("Streaming thread started")

    def _process_stream_updates(self):
        """Drain the streaming queue and route updates to MarketCaches."""
        if self._output_queue is None:
            return

        processed = 0
        while not self._output_queue.empty():
            try:
                market_books = self._output_queue.get_nowait()
            except Exception:
                break

            for market_book in market_books:
                market_id = market_book.market_id
                if market_id not in self.market_caches:
                    continue

                cache = self.market_caches[market_id]
                cache.process_market_change(market_book)

                # Update fill tracking for active bets
                if hasattr(market_book, "runners") and market_book.runners:
                    now = datetime.now(timezone.utc)
                    for runner in market_book.runners:
                        rid = runner.selection_id
                        best_lay = np.nan
                        tick_price = np.nan
                        tick_qty = 0.0

                        if hasattr(runner, "ex") and runner.ex:
                            if runner.ex.available_to_lay:
                                best_lay = runner.ex.available_to_lay[0]["price"]
                            if runner.ex.traded_volume:
                                last_trade = runner.ex.traded_volume[-1]
                                tick_price = last_trade["price"]
                                tick_qty = last_trade["size"]

                        self.paper_trader.process_tick(
                            market_id, rid, tick_price, tick_qty, best_lay, now
                        )
                processed += 1

        if processed > 0:
            logger.debug(f"Processed {processed} streaming market updates")

    def run(self):
        """Main event loop."""
        self.setup()

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        while self._running:
            try:
                now = datetime.now(timezone.utc)

                # Process streaming updates (non-blocking drain of queue)
                if not self.dry_run:
                    self._process_stream_updates()

                # Keep-alive
                if not self.dry_run:
                    self.client.keep_alive()

                # Discover new markets
                if time.time() - self._last_discovery > MARKET_DISCOVERY_INTERVAL_SECONDS:
                    self._discover_markets()
                    self._last_discovery = time.time()

                # Check for decision points
                self._check_decision_points(now)

                # Check for settlements
                self._check_settlements()

                # Cleanup old markets
                self._cleanup_old_markets(now)

                # Periodic flush
                if time.time() - self._last_flush > 300:  # Every 5 min
                    self.data_logger.flush()
                    self._last_flush = time.time()

                # Sleep briefly to avoid busy-waiting
                time.sleep(0.5)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(10)

        self._shutdown()

    def _discover_markets(self):
        """Poll for upcoming markets and set up caches."""
        if self.dry_run:
            return

        try:
            catalogues = self.client.list_upcoming_markets(
                minutes_ahead=STREAM_SUBSCRIBE_MINUTES_BEFORE + 5
            )
        except Exception as e:
            logger.error(f"Market discovery failed: {e}")
            return

        for cat in catalogues:
            if cat.market_id in self.known_markets:
                continue

            self.known_markets.add(cat.market_id)
            file_name = derive_file_name(cat)
            runner_ids = [r.selection_id for r in cat.runners]
            num_runners = len(runner_ids)
            market_start = cat.market_start_time

            # Get market base rate from description
            market_base_rate = 8.0  # Default AU greyhound commission
            if hasattr(cat, "description") and cat.description:
                if hasattr(cat.description, "market_base_rate"):
                    market_base_rate = cat.description.market_base_rate

            venue = ""
            if hasattr(cat, "event") and cat.event:
                venue = getattr(cat.event, "venue", "")

            local_dow = get_local_dow(market_start)

            cache = MarketCache(
                market_id=cat.market_id,
                market_start_time=market_start,
                file_name=file_name,
                runner_ids=runner_ids,
                num_active_runners=num_runners,
                market_base_rate=market_base_rate,
                venue=venue,
                local_dow=local_dow,
            )
            self.market_caches[cat.market_id] = cache

            logger.info(
                f"New market: {cat.market_id} ({venue}) "
                f"{num_runners} runners, starts {market_start.strftime('%H:%M:%S')}"
            )

        # Subscribe to markets that need streaming
        self._update_subscriptions()

    def _update_subscriptions(self):
        """Subscribe to markets that are within the streaming window."""
        now = datetime.now(timezone.utc)
        subscribe_cutoff = now + timedelta(minutes=STREAM_SUBSCRIBE_MINUTES_BEFORE)

        new_subs = []
        for market_id, cache in self.market_caches.items():
            if market_id in self.subscribed_markets:
                continue
            if cache.is_settled:
                continue

            start_time = cache.market_start_time
            if isinstance(start_time, datetime):
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                if start_time <= subscribe_cutoff:
                    new_subs.append(market_id)

        if new_subs and not self.dry_run and self._stream is not None:
            try:
                from betfairlightweight.filters import (
                    streaming_market_data_filter,
                    streaming_market_filter,
                )
                # Resubscribe with the full set of active market IDs
                all_active = list(self.subscribed_markets | set(new_subs))
                market_data_filter = streaming_market_data_filter(
                    fields=["EX_ALL_OFFERS", "EX_TRADED", "EX_TRADED_VOL", "EX_MARKET_DEF"],
                    ladder_levels=10,
                )
                self._stream.subscribe_to_markets(
                    market_filter=streaming_market_filter(market_ids=all_active),
                    market_data_filter=market_data_filter,
                    initial_clk=self._listener.initial_clk,
                    clk=self._listener.clk,
                )
                self.subscribed_markets.update(new_subs)
                logger.info(
                    f"Subscribed to {len(new_subs)} new markets "
                    f"({len(all_active)} total active)"
                )
            except Exception as e:
                logger.error(f"Subscription failed: {e}", exc_info=True)

    def _check_decision_points(self, now):
        """Check if any markets have reached their decision point."""
        for market_id, cache in self.market_caches.items():
            if cache.decision_made or cache.is_settled:
                continue

            start_time = cache.market_start_time
            if not isinstance(start_time, datetime):
                continue

            # Ensure start_time is timezone-aware
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)

            decision_time = start_time - timedelta(seconds=DECISION_SECONDS_BEFORE_START)

            if now >= decision_time:
                self._make_decision(market_id, cache)
                cache.decision_made = True

    def _make_decision(self, market_id, cache):
        """Compute features, run ensemble, evaluate edges, and place paper bets."""
        logger.info(f"Decision point for market {market_id} ({cache.venue})")

        try:
            # Compute features for all 4 t_defs
            feature_results = self.feature_computer.compute(cache)

            if not feature_results:
                logger.warning(f"No features computed for {market_id}")
                return

            # Log features
            self.data_logger.log_features(feature_results, market_id)

            # Run ensemble
            predictions = self.ensemble.predict(feature_results)

            if predictions.empty:
                logger.warning(f"No predictions for {market_id}")
                return

            # Log decisions
            self.data_logger.log_decisions(predictions, market_id)

            # Evaluate edges and place paper bets
            new_bets = self.paper_trader.evaluate_and_bet(
                predictions, market_id, cache.file_name
            )

            if new_bets:
                logger.info(f"Placed {len(new_bets)} paper bets on {market_id}")
                self.data_logger.log_bets(new_bets)
            else:
                logger.info(f"No qualifying bets for {market_id}")

            # Log ticks for this market
            self.data_logger.log_ticks_from_cache(cache)

        except Exception as e:
            logger.error(f"Decision failed for {market_id}: {e}", exc_info=True)

    def _check_settlements(self):
        """Check for settled markets and record outcomes."""
        for market_id, cache in list(self.market_caches.items()):
            if not cache.is_settled:
                continue
            if cache.winner_id is None:
                # Try to get result from API
                if not self.dry_run:
                    try:
                        result = self.client.get_market_result(market_id)
                        if result and hasattr(result, "runners"):
                            for r in result.runners:
                                if r.status == "WINNER":
                                    cache.winner_id = r.selection_id
                    except Exception as e:
                        logger.warning(f"Could not get result for {market_id}: {e}")

            if cache.winner_id is not None:
                # Settle bets
                active_for_market = [
                    b for b in self.paper_trader.active_bets
                    if b.market_id == market_id
                ]
                if active_for_market:
                    self.paper_trader.settle_market(market_id, cache.winner_id)

                    # Update bet logs
                    settled = [
                        b for b in self.paper_trader.settled_bets
                        if b.market_id == market_id
                    ]
                    self.data_logger.log_bets(settled)

    def _cleanup_old_markets(self, now):
        """Remove markets that finished more than MARKET_CLEANUP_MINUTES_AFTER ago."""
        to_remove = []
        cutoff = now - timedelta(minutes=MARKET_CLEANUP_MINUTES_AFTER)

        for market_id, cache in self.market_caches.items():
            start_time = cache.market_start_time
            if isinstance(start_time, datetime):
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                if start_time < cutoff and cache.is_settled:
                    to_remove.append(market_id)

        for market_id in to_remove:
            del self.market_caches[market_id]
            self.subscribed_markets.discard(market_id)
            logger.debug(f"Cleaned up market {market_id}")

    def _shutdown(self):
        """Graceful shutdown: stop stream, flush logs, save daily summary."""
        logger.info("Shutting down...")

        # Stop streaming
        if self._stream is not None:
            try:
                self._stream.stop()
                logger.info("Streaming stopped")
            except Exception as e:
                logger.warning(f"Error stopping stream: {e}")

        # Flush all data
        self.data_logger.flush()

        # Save daily summary
        summary = self.paper_trader.get_daily_summary()
        self.data_logger.log_daily_summary(summary)

        logger.info(f"Daily summary: {summary}")
        logger.info("Shutdown complete.")

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="Paper Trading System")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without connecting to Betfair (for testing)"
    )
    args = parser.parse_args()

    setup_logging(dry_run=args.dry_run)

    engine = PaperTradingEngine(dry_run=args.dry_run)
    engine.run()


if __name__ == "__main__":
    main()
