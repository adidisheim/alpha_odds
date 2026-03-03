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
        # Trade detection state
        self._last_traded_vol = {}   # price -> cumulative qty (per-price tracking)
        self._initial_volume_set = False  # First snapshot is baseline, not a trade
        self._cumulative_trade_qty = 0.0  # Sum of incremental trade quantities

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

        # Use small epsilon for float comparison — accumulated float arithmetic
        # can leave total_qty at 99.999999... instead of exactly 100.0
        if total_qty < q_value - 0.01:
            return np.nan, np.nan

        return notional / total_qty, total_qty

    def _record_snapshot(self, timestamp, trade_prc, trade_qty,
                         back_book=None, lay_book=None):
        """Record order book state as a snapshot.

        Args:
            back_book/lay_book: if provided, use these for book features instead
                of self.back_book/lay_book. This allows trade snapshots to use the
                PRE-trade book state (matching the historical pipeline where atb
                updates are processed before trd records).
        """
        bb = back_book if back_book is not None else self.back_book
        lb = lay_book if lay_book is not None else self.lay_book

        # Best prices from the specified book
        if bb:
            best_back = max(bb.keys())
            best_back_cum_qty = bb[best_back]
        else:
            best_back = np.nan
            best_back_cum_qty = 0.0

        if lb:
            best_lay = min(lb.keys())
            best_lay_cum_qty = lb[best_lay]
        else:
            best_lay = np.nan
            best_lay_cum_qty = 0.0

        total_back_qty = sum(bb.values())
        total_lay_qty = sum(lb.values())

        # Freeze the book as a sorted list for reproducible computation
        lb_frozen = sorted(lb.items()) if lb else []
        bb_frozen = sorted(bb.items(), reverse=True) if bb else []

        best_back_q100, _ = self._get_execution_price(bb, "atb", 100)
        best_lay_q100, _ = self._get_execution_price(lb, "atl", 100)
        best_back_q1000, _ = self._get_execution_price(bb, "atb", 1000)
        best_lay_q1000, _ = self._get_execution_price(lb, "atl", 1000)

        # Consistency check: if total_lay_qty >= 100, q100 should not be NaN
        # (uses epsilon for float comparison since _get_execution_price now does too)
        if total_lay_qty >= 99.99 and (np.isnan(best_lay_q100) or best_lay_q100 == 0):
            # Recompute directly from the frozen book
            remaining = 100.0
            notional = 0.0
            t_qty = 0.0
            for price, size in lb_frozen:
                used = min(size, remaining)
                notional += used * price
                t_qty += used
                remaining -= used
                if remaining <= 0:
                    break
            if t_qty >= 99.99:
                best_lay_q100 = notional / t_qty

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
        self.off_time = None  # When market goes SUSPENDED (race starts)
        self._last_status = None  # Track status transitions for debugging
        self._last_inplay = None  # Track in_play transitions

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
        # Use Betfair's publish time for accurate temporal alignment.
        # The historical pipeline uses the stream's 'pt' field; we must match
        # that so time_delta and forward-fill produce the same feature values.
        # datetime.now() is wrong here because streaming updates queue up and
        # get processed in bursts, collapsing real-world time differences.
        if hasattr(market_book, "publish_time") and market_book.publish_time is not None:
            timestamp = market_book.publish_time
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # Detect market status from MarketBook (populated from marketDefinition by betfairlightweight)
        status = getattr(market_book, "status", None)
        inplay = getattr(market_book, "inplay", None)

        # Log status transitions for debugging
        if status and status != self._last_status:
            logger.info(f"Market {self.market_id}: status {self._last_status} -> {status}")
            self._last_status = status
        if inplay is not None and inplay != self._last_inplay:
            logger.info(f"Market {self.market_id}: inplay {self._last_inplay} -> {inplay}")
            self._last_inplay = inplay

        # Detect off_time: SUSPENDED or in_play transition = race start
        if self.off_time is None:
            if status == "SUSPENDED":
                self.off_time = timestamp
                logger.info(f"Market {self.market_id}: SUSPENDED (off) at {timestamp}")
            elif inplay is True:
                # Fallback: in greyhound races, SUSPENDED can be <50ms and missed
                # by the 50ms conflation window. inplay=True is a reliable indicator.
                self.off_time = timestamp
                logger.info(f"Market {self.market_id}: IN_PLAY (off fallback) at {timestamp}")

        # Detect settlement
        if status in ("CLOSED", "COMPLETE"):
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

                    # Save PRE-update book state for trade snapshots.
                    # betfairlightweight's available_to_back is the POST-trade state
                    # (liquidity already consumed). The historical pipeline processes
                    # atb updates (which add liquidity) BEFORE recording trades, so
                    # trade rows see pre-depletion book state. Using the previous
                    # book for trade snapshots matches this behavior.
                    pre_back = runner_book.back_book.copy()
                    pre_lay = runner_book.lay_book.copy()

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
                    # Create one snapshot per price-level delta (matches historical
                    # pipeline which creates one row per [prc,qty] pair)
                    had_trade_snapshot = False
                    if ex.traded_volume:
                        new_vol = {tv["price"]: tv["size"] for tv in ex.traded_volume}

                        if not runner_book._initial_volume_set:
                            # First volume snapshot: set baseline, don't record as trade.
                            runner_book._last_traded_vol = new_vol.copy()
                            runner_book._initial_volume_set = True
                        else:
                            old_vol = runner_book._last_traded_vol
                            # Collect per-price deltas
                            price_deltas = []
                            for price, qty in new_vol.items():
                                prev = old_vol.get(price, 0.0)
                                if qty > prev + 0.001:
                                    price_deltas.append((price, qty - prev))

                            # Record one snapshot per price-level delta,
                            # using PRE-update book for book features
                            for trade_prc, trade_qty in price_deltas:
                                runner_book.trades.append(
                                    (timestamp, trade_prc, trade_qty)
                                )
                                runner_book._cumulative_trade_qty += trade_qty
                                runner_book._record_snapshot(
                                    timestamp,
                                    trade_prc=trade_prc,
                                    trade_qty=trade_qty,
                                    back_book=pre_back,
                                    lay_book=pre_lay,
                                )
                                had_trade_snapshot = True

                            runner_book._last_traded_vol = new_vol.copy()

                    # If no trade snapshots were recorded this message,
                    # record a book-only snapshot using CURRENT (post-update) book
                    if not had_trade_snapshot:
                        runner_book._record_snapshot(
                            timestamp, trade_prc=np.nan, trade_qty=0.0
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

    def get_trade_diagnostics(self):
        """Return per-runner trade diagnostics for debugging.

        Returns dict with:
            runners: list of {runner_id, n_snapshots, n_trades, cumulative_qty,
                              initial_vol_set, price_levels_tracked}
            summary: {total_runners, runners_with_trades, avg_trades_per_runner}
        """
        diag = []
        for rid, runner in self.runners.items():
            diag.append({
                "runner_id": rid,
                "n_snapshots": len(runner.snapshots),
                "n_trades": len(runner.trades),
                "cumulative_qty": runner._cumulative_trade_qty,
                "initial_vol_set": runner._initial_volume_set,
                "price_levels_tracked": len(runner._last_traded_vol),
            })
        runners_with_trades = sum(1 for d in diag if d["n_trades"] > 0)
        avg_trades = (
            sum(d["n_trades"] for d in diag) / len(diag) if diag else 0
        )
        return {
            "runners": diag,
            "summary": {
                "total_runners": len(diag),
                "runners_with_trades": runners_with_trades,
                "avg_trades_per_runner": round(avg_trades, 1),
            },
        }

    def get_book_depth_diagnostics(self):
        """Return per-runner book depth stats to verify full depth after removing ladder_levels cap.

        Returns dict with:
            runners: list of {runner_id, back_levels, lay_levels, total_back_qty, total_lay_qty}
            summary: {avg_back_levels, avg_lay_levels, max_back_levels, max_lay_levels}
        """
        diag = []
        for rid, runner in self.runners.items():
            diag.append({
                "runner_id": rid,
                "back_levels": len(runner.back_book),
                "lay_levels": len(runner.lay_book),
                "total_back_qty": sum(runner.back_book.values()),
                "total_lay_qty": sum(runner.lay_book.values()),
            })
        if not diag:
            return {"runners": [], "summary": {}}
        avg_back = sum(d["back_levels"] for d in diag) / len(diag)
        avg_lay = sum(d["lay_levels"] for d in diag) / len(diag)
        max_back = max(d["back_levels"] for d in diag)
        max_lay = max(d["lay_levels"] for d in diag)
        return {
            "runners": diag,
            "summary": {
                "avg_back_levels": round(avg_back, 1),
                "avg_lay_levels": round(avg_lay, 1),
                "max_back_levels": max_back,
                "max_lay_levels": max_lay,
            },
        }

    def force_snapshot_all(self, timestamp):
        """
        Record the current book state for all runners at the given timestamp.

        The streaming API only delivers MCMs when book state changes. For thin
        markets with few orders, updates can be minutes apart. This creates
        sparse tick data that causes stale forward-fills when computing features.

        The historical pipeline processes raw BZ2 data which records every single
        delta (often hundreds per runner per 10 minutes). By forcing periodic
        snapshots (e.g., every 5 seconds), we match the historical pipeline's
        density and ensure forward-fill picks up recent book states.
        """
        for rid, runner in self.runners.items():
            if runner.back_book or runner.lay_book:
                runner._record_snapshot(timestamp, trade_prc=np.nan, trade_qty=0.0)

    @property
    def decision_made(self):
        return self._decision_made

    @decision_made.setter
    def decision_made(self, value):
        self._decision_made = value
