"""
Betfair API Client — authentication, market discovery, and streaming.

Uses betfairlightweight for API access and streaming.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.filters import (
    market_filter,
    streaming_market_filter,
    streaming_market_data_filter,
)

from config import (
    BETFAIR_USERNAME,
    BETFAIR_PASSWORD,
    BETFAIR_APP_KEY,
    BETFAIR_CERT_PATH,
    BETFAIR_KEY_PATH,
    EVENT_TYPE_ID,
    MARKET_COUNTRIES,
    MARKET_TYPE,
)

logger = logging.getLogger(__name__)


class BetfairClient:
    """Handles Betfair API auth, keep-alive, market catalogue, and streaming."""

    def __init__(self):
        # betfairlightweight expects `certs` to be the directory containing
        # client-2048.crt and client-2048.key files.
        # If BETFAIR_CERT_PATH points to a file, use its parent directory.
        import os
        certs_path = BETFAIR_CERT_PATH
        if os.path.isfile(certs_path):
            certs_path = os.path.dirname(certs_path)

        self.api = betfairlightweight.APIClient(
            username=BETFAIR_USERNAME,
            password=BETFAIR_PASSWORD,
            app_key=BETFAIR_APP_KEY,
            certs=certs_path,
        )
        self._last_keepalive = 0.0
        self._stream = None

    def login(self):
        """Certificate-based login (non-interactive)."""
        if BETFAIR_CERT_PATH:
            self.api.login()
        else:
            self.api.login_interactive()
        logger.info("Logged in to Betfair API")
        self._last_keepalive = time.time()

    def keep_alive(self):
        """Send keep-alive if >15 min since last one (tokens expire after 20 min)."""
        if time.time() - self._last_keepalive > 900:
            self.api.keep_alive()
            self._last_keepalive = time.time()
            logger.debug("Keep-alive sent")

    def list_upcoming_markets(self, minutes_ahead=20):
        """
        Find AU greyhound WIN markets starting within the next `minutes_ahead` minutes.

        Returns list of MarketCatalogue objects with runner info.
        """
        now = datetime.now(timezone.utc)
        time_from = now
        time_to = now + timedelta(minutes=minutes_ahead)

        mf = market_filter(
            event_type_ids=[EVENT_TYPE_ID],
            market_countries=MARKET_COUNTRIES,
            market_type_codes=[MARKET_TYPE],
            market_start_time={
                "from": time_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": time_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )

        catalogues = self.api.betting.list_market_catalogue(
            filter=mf,
            market_projection=[
                "RUNNER_DESCRIPTION",
                "RUNNER_METADATA",
                "MARKET_START_TIME",
                "EVENT",
                "COMPETITION",
                "MARKET_DESCRIPTION",
            ],
            max_results=100,
            sort="FIRST_TO_START",
        )

        logger.info(f"Found {len(catalogues)} upcoming markets (next {minutes_ahead} min)")
        return catalogues

    def create_streaming_connection(self, listener):
        """
        Create a streaming connection using the provided listener.

        Returns the stream object. Call stream.start() and stream.subscribe_to_markets()
        after creation.
        """
        self._stream = self.api.streaming.create_stream(listener=listener)
        return self._stream

    def subscribe_to_markets(self, market_ids):
        """
        Subscribe to market data stream for given market IDs.

        NOTE: In the current architecture, streaming is managed by PaperTradingEngine
        directly via _update_subscriptions(). This method is kept for backward compat.
        """
        logger.info(f"subscribe_to_markets called with {len(market_ids)} markets")

    def get_market_result(self, market_id):
        """
        Get settled market result to determine winners.

        Returns list of RunnerResult with status (WINNER/LOSER/etc).
        """
        try:
            result = self.api.betting.list_market_book(
                market_ids=[market_id],
                price_projection={"priceData": ["SP_TRADED"]},
            )
            if result:
                return result[0]
        except Exception as e:
            logger.warning(f"Failed to get result for {market_id}: {e}")
        return None
