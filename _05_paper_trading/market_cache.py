"""
Market Cache — maintains incremental order book state from streaming deltas.

Mirrors the data schema produced by process_runner_order_book() in the historical pipeline.
Converts streaming updates (atb/atl/trd) into a tick DataFrame suitable for feature computation.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RunnerBook:
    """
    Maintains the order book for a single runner.

    back_book / lay_book: {price: qty} dicts representing current depth.
    ticks: list of (time, event_type, updates) for full replay.
    """

    def __init__(self, runner_id, market_id):
        self.runner_id = runner_id
        self.market_id = market_id
        self.back_book = {}  # price -> qty (available to back)
        self.lay_book = {}   # price -> qty (available to lay)
        self.trades = []     # (time, price, qty)
        self.snapshots = []  # (time, best_back, best_lay, best_back_cum_qty, best_lay_cum_qty,
                             #  total_back_qty, total_lay_qty, best_back_q100, best_lay_q100,
                             #  best_back_q1000, best_lay_q1000, prc, qty)

    def update_back(self, updates, timestamp):
        """Apply available-to-back updates: list of [price, qty] pairs."""
        for price, qty in updates:
            if qty == 0:
                self.back_book.pop(price, None)
            else:
                self.back_book[price] = qty
        self._record_snapshot(timestamp, trade_prc=np.nan, trade_qty=0.0)

    def update_lay(self, updates, timestamp):
        """Apply available-to-lay updates: list of [price, qty] pairs."""
        for price, qty in updates:
            if qty == 0:
                self.lay_book.pop(price, None)
            else:
                self.lay_book[price] = qty
        self._record_snapshot(timestamp, trade_prc=np.nan, trade_qty=0.0)

    def update_trades(self, updates, timestamp):
        """Apply traded volume updates: list of [price, qty] pairs."""
        for price, qty in updates:
            if qty > 0:
                self.trades.append((timestamp, price, qty))
                self._record_snapshot(timestamp, trade_prc=price, trade_qty=qty)

    def _get_best_back(self):
        """Best back price (highest price available to back)."""
        if not self.back_book:
            return np.nan, 0.0
        best_price = max(self.back_book.keys())
        return best_price, self.back_book[best_price]

    def _get_best_lay(self):
        """Best lay price (lowest price available to lay)."""
        if not self.lay_book:
            return np.nan, 0.0
        best_price = min(self.lay_book.keys())
        return best_price, self.lay_book[best_price]

    def _get_execution_price(self, book, order_type, q_value):
        """
        Compute execution price for a given order size.

        Mirrors get_best_value() from process_races.py.
        For backs: walk prices from highest to lowest.
        For lays: walk prices from lowest to highest.
        """
        if not book or q_value == 0:
            if order_type == "atb":
                return self._get_best_back()
            else:
                return self._get_best_lay()

        if order_type == "atb":
            sorted_prices = sorted(book.keys(), reverse=True)
        else:
            sorted_prices = sorted(book.keys())

        remaining = q_value
        notional = 0.0
        total_qty = 0.0

        for price in sorted_prices:
            qty_at_level = book[price]
            used = min(qty_at_level, remaining)
            notional += used * price
            total_qty += used
            remaining -= used
            if remaining <= 0:
                break

        if total_qty < q_value:
            return np.nan, np.nan

        return notional / total_qty, total_qty

    def _record_snapshot(self, timestamp, trade_prc, trade_qty):
        """Record current order book state as a snapshot."""
        best_back, best_back_cum_qty = self._get_best_back()
        best_lay, best_lay_cum_qty = self._get_best_lay()
        total_back_qty = sum(self.back_book.values())
        total_lay_qty = sum(self.lay_book.values())

        best_back_q100, _ = self._get_execution_price(self.back_book, "atb", 100)
        best_lay_q100, _ = self._get_execution_price(self.lay_book, "atl", 100)
        best_back_q1000, _ = self._get_execution_price(self.back_book, "atb", 1000)
        best_lay_q1000, _ = self._get_execution_price(self.lay_book, "atl", 1000)

        # Handle missing q values: lays→1001, backs→1.0 (matches historical pipeline)
        if np.isnan(best_lay_q100) or best_lay_q100 == 0:
            best_lay_q100 = 1001.0
        if np.isnan(best_back_q100) or best_back_q100 == 0:
            best_back_q100 = 1.0
        if np.isnan(best_lay_q1000) or best_lay_q1000 == 0:
            best_lay_q1000 = 1001.0
        if np.isnan(best_back_q1000) or best_back_q1000 == 0:
            best_back_q1000 = 1.0

        self.snapshots.append({
            "time": timestamp,
            "best_back": best_back,
            "best_lay": best_lay,
            "best_back_cum_qty": best_back_cum_qty,
            "best_lay_cum_qty": best_lay_cum_qty,
            "total_back_qty": total_back_qty,
            "total_lay_qty": total_lay_qty,
            "best_back_q_100": best_back_q100,
            "best_lay_q_100": best_lay_q100,
            "best_back_q_1000": best_back_q1000,
            "best_lay_q_1000": best_lay_q1000,
            "prc": trade_prc,
            "qty": trade_qty,
        })


class MarketCache:
    """
    Maintains order book state for all runners in a market.

    Processes streaming updates and produces DataFrames matching
    the historical pipeline's schema.
    """

    def __init__(self, market_id, market_start_time, file_name, runner_ids,
                 num_active_runners, market_base_rate, venue="", local_dow=0):
        self.market_id = market_id
        self.market_start_time = market_start_time
        self.file_name = file_name
        self.runner_ids = runner_ids
        self.num_active_runners = num_active_runners
        self.market_base_rate = market_base_rate
        self.venue = venue
        self.local_dow = local_dow
        self.runners = {rid: RunnerBook(rid, market_id) for rid in runner_ids}
        self.is_settled = False
        self.winner_id = None
        self._decision_made = False

    def process_market_change(self, market_book):
        """
        Process a MarketBook update from betfairlightweight streaming.

        betfairlightweight's StreamListener produces MarketBook objects where:
          - market_book.market_definition contains status and runner definitions
          - market_book.runners is a list of RunnerBook objects with:
            - runner.selection_id (int)
            - runner.ex.available_to_back: [{'price': 5.0, 'size': 150.0}, ...]
            - runner.ex.available_to_lay: [{'price': 5.2, 'size': 100.0}, ...]
            - runner.ex.traded_volume: [{'price': 5.0, 'size': 500.0}, ...]
            - runner.status: 'ACTIVE', 'WINNER', 'LOSER', 'REMOVED'
        """
        timestamp = datetime.now(timezone.utc)

        # Handle market definition changes (status, runners)
        if hasattr(market_book, "market_definition") and market_book.market_definition:
            md = market_book.market_definition
            if hasattr(md, "status") and md.status in ("CLOSED", "COMPLETE"):
                self.is_settled = True

        # Check market status directly
        if hasattr(market_book, "status") and market_book.status in ("CLOSED",):
            self.is_settled = True

        # Process runner updates
        if hasattr(market_book, "runners") and market_book.runners:
            for rc in market_book.runners:
                rid = rc.selection_id
                if rid not in self.runners:
                    self.runners[rid] = RunnerBook(rid, self.market_id)
                runner_book = self.runners[rid]

                # Check for winner
                if hasattr(rc, "status") and rc.status == "WINNER":
                    self.winner_id = rid
                    self.is_settled = True
                    logger.info(f"Market {self.market_id}: winner = {self.winner_id}")

                # betfairlightweight gives full snapshots of the book state.
                # We rebuild the book from the snapshot and record it.
                if hasattr(rc, "ex") and rc.ex:
                    ex = rc.ex

                    # Rebuild back book from available_to_back
                    if ex.available_to_back:
                        runner_book.back_book.clear()
                        for level in ex.available_to_back:
                            price = level["price"]
                            size = level["size"]
                            if size > 0:
                                runner_book.back_book[price] = size

                    # Rebuild lay book from available_to_lay
                    if ex.available_to_lay:
                        runner_book.lay_book.clear()
                        for level in ex.available_to_lay:
                            price = level["price"]
                            size = level["size"]
                            if size > 0:
                                runner_book.lay_book[price] = size

                    # Process traded volume — detect new trades
                    trade_prc = np.nan
                    trade_qty = 0.0
                    if ex.traded_volume:
                        # Compute total traded to detect incremental trades
                        new_total = sum(tv["size"] for tv in ex.traded_volume)
                        old_total = sum(
                            qty for _, _, qty in runner_book.trades
                        ) if runner_book.trades else 0.0
                        if new_total > old_total + 0.01:
                            # New trade detected — use last traded price
                            if hasattr(rc, "last_price_traded") and rc.last_price_traded:
                                trade_prc = rc.last_price_traded
                                trade_qty = new_total - old_total
                                runner_book.trades.append(
                                    (timestamp, trade_prc, trade_qty)
                                )

                    # Record snapshot
                    runner_book._record_snapshot(
                        timestamp, trade_prc=trade_prc, trade_qty=trade_qty
                    )

    def to_dataframe(self):
        """
        Convert accumulated snapshots to a DataFrame matching the historical pipeline's schema.

        Returns DataFrame with columns:
            time, file_name, id, best_back, best_lay, best_back_cum_qty, best_lay_cum_qty,
            total_back_qty, total_lay_qty, best_back_q_100, best_lay_q_100,
            best_back_q_1000, best_lay_q_1000, prc, qty, order_type
        """
        rows = []
        for rid, runner in self.runners.items():
            for snap in runner.snapshots:
                row = snap.copy()
                row["file_name"] = self.file_name
                row["id"] = rid
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values(["file_name", "id", "time"])

        # Fill NaN qty/prc with 0 (matches historical: df[['qty','prc']] = df[['qty','prc']].fillna(0.0))
        df[["qty", "prc"]] = df[["qty", "prc"]].fillna(0.0)

        # Infer order type (same logic as process_races.py infer_order_type)
        def infer_order_type(row):
            prc = row["prc"]
            if pd.isna(prc) or prc == 0:
                return np.nan
            if prc >= row["best_lay"]:
                return "lay"
            if prc <= row["best_back"]:
                return "back"
            return "cross_matching"

        df["order_type"] = df.apply(infer_order_type, axis=1)

        return df

    def get_runner_position_map(self):
        """Return {runner_id: position} from the original runner list order."""
        return {rid: i + 1 for i, rid in enumerate(self.runner_ids)}

    @property
    def decision_made(self):
        return self._decision_made

    @decision_made.setter
    def decision_made(self, value):
        self._decision_made = value
