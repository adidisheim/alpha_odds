"""
Paper Trader — edge evaluation, paper bet tracking, fill simulation, and P&L computation.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    EDGE_THRESHOLD,
    MIN_BACK_ODDS,
    MAX_BACK_ODDS,
    COMMISSION_RATE,
    STAKE_SIZE,
    MAX_DAILY_LOSS,
    BETFAIR_TICKS,
)

logger = logging.getLogger(__name__)


def get_tick_increment(price):
    """Return the Betfair tick increment for a given price level."""
    for low, high, inc in BETFAIR_TICKS:
        if low <= price < high:
            return inc
    return 10.0


def adjust_ticks(price, n_ticks):
    """Move price by n_ticks on the Betfair tick ladder."""
    for _ in range(abs(n_ticks)):
        inc = get_tick_increment(price)
        if n_ticks > 0:
            price = round(price + inc, 2)
        else:
            price = round(price - inc, 2)
    return max(1.01, price)


class PaperBet:
    """Represents a single paper bet."""

    def __init__(self, market_id, file_name, runner_id, model_prob, market_prob,
                 edge, back_odds, limit_price, stake, timestamp):
        self.market_id = market_id
        self.file_name = file_name
        self.runner_id = runner_id
        self.model_prob = model_prob
        self.market_prob = market_prob
        self.edge = edge
        self.back_odds = back_odds
        self.limit_price = limit_price
        self.stake = stake
        self.timestamp = timestamp

        # Fill tracking
        self.conservative_fill = False
        self.moderate_fill = False
        self.fill_price = np.nan
        self.fill_time = None

        # Settlement
        self.is_settled = False
        self.winner = None
        self.pnl = np.nan

    def check_fill(self, tick_price, tick_qty, best_lay, timestamp):
        """
        Check if this bet would have filled from a new tick.

        Conservative: trade at prc >= limit_price
        Moderate: conservative OR best_lay <= limit_price
        """
        if self.conservative_fill:
            return  # Already filled

        # Conservative fill: a trade occurred at our price or better
        if not pd.isna(tick_price) and tick_price > 0 and tick_price >= self.limit_price:
            self.conservative_fill = True
            self.moderate_fill = True
            self.fill_price = tick_price
            self.fill_time = timestamp

        # Moderate fill: spread crossed our price
        if not self.moderate_fill and not pd.isna(best_lay) and best_lay > 1.0:
            if best_lay <= self.limit_price:
                self.moderate_fill = True
                if pd.isna(self.fill_price):
                    self.fill_price = self.limit_price
                    self.fill_time = timestamp

    def settle(self, winner_id):
        """Settle the bet based on the race winner."""
        self.is_settled = True
        self.winner = (self.runner_id == winner_id)

        if self.conservative_fill:
            # P&L: win → (odds-1)*(1-commission)*stake, lose → -stake
            if self.winner:
                self.pnl = (self.fill_price - 1) * (1 - COMMISSION_RATE) * self.stake
            else:
                self.pnl = -self.stake
        else:
            self.pnl = 0.0  # Didn't fill, no P&L

    def to_dict(self):
        """Convert to dict for logging."""
        return {
            "timestamp": self.timestamp,
            "market_id": self.market_id,
            "file_name": self.file_name,
            "runner_id": self.runner_id,
            "model_prob": self.model_prob,
            "market_prob": self.market_prob,
            "edge": self.edge,
            "back_odds": self.back_odds,
            "limit_price": self.limit_price,
            "stake": self.stake,
            "conservative_fill": self.conservative_fill,
            "moderate_fill": self.moderate_fill,
            "fill_price": self.fill_price,
            "fill_time": self.fill_time,
            "is_settled": self.is_settled,
            "winner": self.winner,
            "pnl": self.pnl,
        }


class PaperTrader:
    """
    Evaluates edges from model predictions and manages paper bets.

    Tracks:
    - Active (unsettled) bets
    - Daily P&L
    - Kill switch state
    """

    def __init__(self):
        self.active_bets = []      # Bets awaiting settlement
        self.settled_bets = []     # Completed bets
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.kill_switch_active = False
        self._today = datetime.now(timezone.utc).date()

    def evaluate_and_bet(self, predictions_df, market_id, file_name):
        """
        Evaluate model predictions and place paper bets where edge > threshold.

        Args:
            predictions_df: DataFrame from CrossTEnsemble.predict() with columns:
                file_name, id, model_prob, market_prob, back_odds, edge
            market_id: Betfair market ID
            file_name: file_name used for this market

        Returns:
            list of PaperBet objects created
        """
        self._check_daily_reset()

        if self.kill_switch_active:
            logger.warning("Kill switch active — not placing bets")
            return []

        new_bets = []
        now = datetime.now(timezone.utc)

        for _, row in predictions_df.iterrows():
            edge = row["edge"]
            back_odds = row["back_odds"]

            # Check thresholds
            if edge <= EDGE_THRESHOLD:
                continue
            if back_odds <= MIN_BACK_ODDS or back_odds >= MAX_BACK_ODDS:
                continue
            if pd.isna(row["model_prob"]) or pd.isna(row["market_prob"]):
                continue

            bet = PaperBet(
                market_id=market_id,
                file_name=file_name,
                runner_id=row["id"],
                model_prob=row["model_prob"],
                market_prob=row["market_prob"],
                edge=edge,
                back_odds=back_odds,
                limit_price=back_odds,  # Place at best available back
                stake=STAKE_SIZE,
                timestamp=now,
            )
            self.active_bets.append(bet)
            new_bets.append(bet)

            logger.info(
                f"Paper bet: {file_name} runner={row['id']} "
                f"edge={edge:.3f} odds={back_odds:.2f} model_p={row['model_prob']:.3f}"
            )

        return new_bets

    def process_tick(self, market_id, runner_id, tick_price, tick_qty, best_lay, timestamp):
        """Update fill status for active bets based on new tick data."""
        for bet in self.active_bets:
            if bet.market_id == market_id and bet.runner_id == runner_id:
                bet.check_fill(tick_price, tick_qty, best_lay, timestamp)

    def settle_market(self, market_id, winner_id):
        """
        Settle all bets for a market.

        Args:
            market_id: Betfair market ID
            winner_id: runner ID of the winner
        """
        to_settle = [b for b in self.active_bets if b.market_id == market_id]
        for bet in to_settle:
            bet.settle(winner_id)
            self.daily_pnl += bet.pnl
            self.total_pnl += bet.pnl
            self.settled_bets.append(bet)
            self.active_bets.remove(bet)

            status = "WIN" if bet.winner else "LOSE"
            fill_str = "FILLED" if bet.conservative_fill else "NO_FILL"
            logger.info(
                f"Settled: {bet.file_name} runner={bet.runner_id} "
                f"{status} {fill_str} pnl=${bet.pnl:.2f}"
            )

        # Check kill switch
        if self.daily_pnl < -MAX_DAILY_LOSS:
            self.kill_switch_active = True
            logger.warning(
                f"KILL SWITCH ACTIVATED: daily loss ${self.daily_pnl:.2f} "
                f"exceeds limit ${MAX_DAILY_LOSS:.2f}"
            )

    def get_daily_summary(self):
        """Get summary statistics for today's trading."""
        today_bets = [b for b in self.settled_bets
                      if b.timestamp.date() == self._today]

        filled_bets = [b for b in today_bets if b.conservative_fill]
        winning_bets = [b for b in filled_bets if b.winner]

        n_evaluated = len(today_bets) + len(self.active_bets)
        n_bets = len(today_bets)
        n_filled = len(filled_bets)
        n_wins = len(winning_bets)
        n_active = len(self.active_bets)

        pnl_values = [b.pnl for b in filled_bets if not pd.isna(b.pnl)]
        avg_edge = np.mean([b.edge for b in today_bets]) if today_bets else 0
        avg_odds = np.mean([b.back_odds for b in today_bets]) if today_bets else 0

        return {
            "date": str(self._today),
            "markets_evaluated": n_evaluated,
            "bets_placed": n_bets,
            "bets_filled": n_filled,
            "bets_won": n_wins,
            "bets_active": n_active,
            "win_rate": n_wins / n_filled if n_filled > 0 else 0,
            "fill_rate": n_filled / n_bets if n_bets > 0 else 0,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "avg_edge": avg_edge,
            "avg_odds": avg_odds,
            "kill_switch": self.kill_switch_active,
            "roi_pct": (sum(pnl_values) / (n_filled * STAKE_SIZE) * 100) if n_filled > 0 else 0,
        }

    def _check_daily_reset(self):
        """Reset daily counters at midnight UTC."""
        today = datetime.now(timezone.utc).date()
        if today != self._today:
            logger.info(
                f"New day: {today}. Previous day P&L: ${self.daily_pnl:.2f}"
            )
            self._today = today
            self.daily_pnl = 0.0
            self.kill_switch_active = False
