"""
Signal Tracker — monitors live trading signals and compares to backtest baselines.

Tracks:
  - Edge, model_prob, market_prob, odds distributions
  - Bet rate, win rate, fill rate
  - V1/V2 component predictions
  - Per-market prediction counts

Detects anomalies and applies auto-fixes:
  - Edge too high → reload normalization params
  - Bet rate too high → temporarily increase edge threshold
  - Model probs extreme → reload models
  - Other deviations → alert + optional pause
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import (
    BACKTEST_BASELINES,
    ANOMALY_MIN_BETS_FOR_CHECK,
    ANOMALY_EDGE_Z_THRESHOLD,
    ANOMALY_BET_RATE_TOLERANCE,
    ANOMALY_PAUSE_ON_CRITICAL,
    EDGE_THRESHOLD,
    LOG_DIR,
)

logger = logging.getLogger(__name__)


class SignalTracker:
    """Tracks live signal distributions and compares to backtest baselines."""

    def __init__(self, variant_name=None):
        # Use variant-specific baselines if available, else fall back to "all"
        if isinstance(BACKTEST_BASELINES, dict) and variant_name and variant_name in BACKTEST_BASELINES:
            self.baselines = BACKTEST_BASELINES[variant_name]
        elif isinstance(BACKTEST_BASELINES, dict) and "all" in BACKTEST_BASELINES:
            self.baselines = BACKTEST_BASELINES["all"]
        else:
            self.baselines = BACKTEST_BASELINES

        # ── Accumulators for qualifying bets ──
        self.edges = []
        self.model_probs = []
        self.market_probs = []
        self.odds = []

        # ── Settlement tracking ──
        self.bet_count = 0
        self.fill_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.pnl_values = []

        # ── All predictions (including non-bets) ──
        self.markets_evaluated = 0
        self.predictions_per_market = []
        self.edges_all = []  # All edges, not just qualifying

        # ── V1/V2 component tracking ──
        self.v1_cross_values = []
        self.v2_cross_values = []

        # ── Anomaly state ──
        self.anomaly_pause = False
        self.anomaly_history = []
        self.edge_threshold_bump = 0.0  # Auto-fix: temporary increase to edge threshold

        # ── Timing ──
        self.start_time = datetime.now(timezone.utc)

    @property
    def effective_edge_threshold(self):
        """Current edge threshold including any auto-fix bump."""
        return EDGE_THRESHOLD + self.edge_threshold_bump

    # ── Recording methods ──

    def record_predictions(self, predictions_df):
        """Record all predictions from a market evaluation (bets + non-bets)."""
        self.markets_evaluated += 1
        self.predictions_per_market.append(len(predictions_df))

        for _, row in predictions_df.iterrows():
            if "edge" in row and not np.isnan(row["edge"]):
                self.edges_all.append(row["edge"])
            if "v1_cross" in row and not np.isnan(row.get("v1_cross", np.nan)):
                self.v1_cross_values.append(row["v1_cross"])
            if "v2_cross" in row and not np.isnan(row.get("v2_cross", np.nan)):
                self.v2_cross_values.append(row["v2_cross"])

    def record_bet(self, bet):
        """Record a placed bet."""
        self.bet_count += 1
        self.edges.append(bet.edge)
        self.model_probs.append(bet.model_prob)
        self.market_probs.append(bet.market_prob)
        self.odds.append(bet.back_odds)

    def record_settlement(self, bet):
        """Record a settled bet."""
        if bet.conservative_fill:
            self.fill_count += 1
            self.pnl_values.append(bet.pnl)
            if bet.winner:
                self.win_count += 1
            else:
                self.loss_count += 1

    def get_daily_record(self):
        """Short record for email notifications."""
        return {
            "bets_settled": self.fill_count,
            "wins": self.win_count,
            "losses": self.loss_count,
            "fills": self.fill_count,
        }

    # ── Statistics ──

    def get_live_stats(self):
        """Return current live signal statistics."""
        hours = max(
            (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600,
            0.1,
        )

        stats = {
            "hours_running": round(hours, 1),
            "markets_evaluated": self.markets_evaluated,
            "bets_placed": self.bet_count,
            "bets_filled": self.fill_count,
            "wins": self.win_count,
            "losses": self.loss_count,
            "bets_per_hour": round(self.bet_count / hours, 2),
        }

        if self.edges:
            stats["avg_edge"] = round(float(np.mean(self.edges)), 4)
            stats["std_edge"] = round(float(np.std(self.edges)), 4)
            stats["median_edge"] = round(float(np.median(self.edges)), 4)
            stats["max_edge"] = round(float(np.max(self.edges)), 4)

        if self.model_probs:
            stats["avg_model_prob"] = round(float(np.mean(self.model_probs)), 4)
            stats["std_model_prob"] = round(float(np.std(self.model_probs)), 4)

        if self.market_probs:
            stats["avg_market_prob"] = round(float(np.mean(self.market_probs)), 4)

        if self.odds:
            stats["avg_odds"] = round(float(np.mean(self.odds)), 2)
            stats["median_odds"] = round(float(np.median(self.odds)), 2)

        if self.fill_count > 0:
            stats["win_rate"] = round(self.win_count / self.fill_count, 3)
            stats["fill_rate"] = round(self.fill_count / max(self.bet_count, 1), 3)
            total_wagered = self.fill_count * (
                self.pnl_values[0] if self.pnl_values else 1
            )
            if total_wagered != 0:
                stats["roi_pct"] = round(sum(self.pnl_values) / abs(total_wagered) * 100, 1)

        if self.v1_cross_values:
            stats["avg_v1_cross"] = round(float(np.mean(self.v1_cross_values)), 4)
        if self.v2_cross_values:
            stats["avg_v2_cross"] = round(float(np.mean(self.v2_cross_values)), 4)

        if self.edges_all:
            stats["avg_edge_all_runners"] = round(float(np.mean(self.edges_all)), 4)
            stats["pct_positive_edge"] = round(
                sum(1 for e in self.edges_all if e > 0) / len(self.edges_all), 3
            )

        if self.edge_threshold_bump > 0:
            stats["edge_threshold_bump"] = self.edge_threshold_bump

        return stats

    def get_comparison_report(self):
        """Compare live stats to backtest baselines. Returns a structured report."""
        live = self.get_live_stats()
        bl = self.baselines
        deviations = {}

        def _dev(key, live_val, bl_val):
            pct = (live_val - bl_val) / max(abs(bl_val), 0.001) * 100
            deviations[key] = {
                "live": round(live_val, 4),
                "backtest": round(bl_val, 4),
                "pct_deviation": round(pct, 1),
            }

        if "avg_edge" in live:
            _dev("avg_edge", live["avg_edge"], bl["avg_edge"])

        if "avg_odds" in live:
            _dev("avg_odds", live["avg_odds"], bl["avg_odds"])

        if "avg_model_prob" in live:
            _dev("avg_model_prob", live["avg_model_prob"], bl["avg_model_prob"])

        if "avg_market_prob" in live:
            _dev("avg_market_prob", live["avg_market_prob"], bl["avg_market_prob"])

        # Extrapolate bet rate to daily (~13 hours of racing)
        if "bets_per_hour" in live and live["hours_running"] > 1:
            estimated_daily = live["bets_per_hour"] * 13
            _dev("bets_per_day", estimated_daily, bl["bets_per_day"])

        if "win_rate" in live and self.fill_count >= ANOMALY_MIN_BETS_FOR_CHECK:
            _dev("win_rate", live["win_rate"], bl["win_rate"])

        if "fill_rate" in live and self.bet_count >= ANOMALY_MIN_BETS_FOR_CHECK:
            _dev("fill_rate", live["fill_rate"], bl["fill_rate"])

        return {"live": live, "backtest": bl, "deviations": deviations}

    # ── Anomaly Detection & Auto-Fix ──

    def check_anomalies(self, engine=None):
        """
        Check for anomalies in live signals vs backtest expectations.

        Args:
            engine: PaperTradingEngine reference, used for auto-fix (reload models, etc.)

        Returns:
            dict with severity, findings, fixes_applied, and action_taken — or None if clean.
        """
        if self.bet_count < ANOMALY_MIN_BETS_FOR_CHECK:
            return None

        bl = self.baselines
        findings = []
        fixes = []
        severity = "WARNING"

        # 1. Edge distribution — too high means possible feature bug
        if self.edges:
            avg_edge = float(np.mean(self.edges))
            if avg_edge > bl["avg_edge"] * 2.5:
                findings.append(
                    f"Avg edge {avg_edge:.1%} is {avg_edge / bl['avg_edge']:.1f}x "
                    f"backtest ({bl['avg_edge']:.1%}). Possible feature computation bug."
                )
                severity = "CRITICAL"
                # Auto-fix: try reloading normalization params
                if engine and hasattr(engine, "_reload_normalization"):
                    engine._reload_normalization()
                    fixes.append("Reloaded normalization parameters from disk.")
            elif avg_edge < bl["avg_edge"] * 0.3:
                findings.append(
                    f"Avg edge {avg_edge:.1%} is only "
                    f"{avg_edge / bl['avg_edge']:.0%} of backtest ({bl['avg_edge']:.1%}). "
                    f"Model may be degraded or market conditions shifted."
                )

        # 2. Bet rate — too many bets = filtering broken
        hours = max(
            (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600, 0.5
        )
        bets_per_hour = self.bet_count / hours
        expected_per_hour = bl["bets_per_day"] / 13
        if bets_per_hour > expected_per_hour * ANOMALY_BET_RATE_TOLERANCE:
            findings.append(
                f"Bet rate {bets_per_hour:.1f}/hr is "
                f"{bets_per_hour / expected_per_hour:.1f}x expected "
                f"({expected_per_hour:.1f}/hr). Edge threshold may not be filtering."
            )
            severity = "CRITICAL"
            # Log only — no auto-fix, we want clean data at original threshold
            fixes.append("No auto-fix applied (data collection mode).")

        # 3. Model probability — extreme values = model broken
        if self.model_probs:
            avg_mp = float(np.mean(self.model_probs))
            if avg_mp > 0.6:
                findings.append(
                    f"Avg model probability {avg_mp:.1%} is abnormally high. "
                    f"Model may be outputting near-1 probabilities."
                )
                severity = "CRITICAL"
                if engine and hasattr(engine, "_reload_models"):
                    engine._reload_models()
                    fixes.append("Reloaded model artifacts from disk.")
            elif avg_mp < 0.05:
                findings.append(
                    f"Avg model probability {avg_mp:.1%} is abnormally low. "
                    f"Model may be outputting near-0 probabilities."
                )
                severity = "CRITICAL"
                if engine and hasattr(engine, "_reload_models"):
                    engine._reload_models()
                    fixes.append("Reloaded model artifacts from disk.")

        # 4. Odds distribution — sanity check
        if self.odds:
            avg_odds = float(np.mean(self.odds))
            if avg_odds < 1.5:
                findings.append(
                    f"Avg odds {avg_odds:.2f} is very low — "
                    f"betting mostly on heavy favorites."
                )
            elif avg_odds > bl["avg_odds"] * 3:
                findings.append(
                    f"Avg odds {avg_odds:.1f} is "
                    f"{avg_odds / bl['avg_odds']:.1f}x backtest ({bl['avg_odds']:.1f}). "
                    f"Betting on longshots."
                )

        # 5. Win rate — z-test for proportions (need enough settled bets)
        if self.fill_count >= 20:
            win_rate = self.win_count / self.fill_count
            p0 = bl["win_rate"]
            se = (p0 * (1 - p0) / self.fill_count) ** 0.5
            z = abs(win_rate - p0) / max(se, 0.001)
            if z > ANOMALY_EDGE_Z_THRESHOLD:
                findings.append(
                    f"Win rate {win_rate:.1%} deviates from backtest {p0:.1%} "
                    f"(z={z:.1f}). Model performance may have shifted."
                )

        # 6. Fill rate — too low means liquidity changed
        if self.bet_count >= 15:
            fill_rate = self.fill_count / self.bet_count
            if fill_rate < bl["fill_rate"] * 0.7:
                findings.append(
                    f"Fill rate {fill_rate:.1%} is below 70% of backtest "
                    f"({bl['fill_rate']:.1%}). Liquidity may have changed."
                )

        # 7. V1 vs V2 divergence
        if len(self.v1_cross_values) >= 20 and len(self.v2_cross_values) >= 20:
            v1_mean = float(np.mean(self.v1_cross_values))
            v2_mean = float(np.mean(self.v2_cross_values))
            divergence = abs(v1_mean - v2_mean)
            if divergence > 0.15:
                findings.append(
                    f"V1 ({v1_mean:.3f}) and V2 ({v2_mean:.3f}) are diverging "
                    f"(gap={divergence:.3f}). Ensemble components may be inconsistent."
                )

        if not findings:
            return None

        # Determine action
        if severity == "CRITICAL" and ANOMALY_PAUSE_ON_CRITICAL and not fixes:
            self.anomaly_pause = True
            action = "Trading PAUSED. Manual review required — restart bot to resume."
        elif severity == "CRITICAL" and fixes:
            action = (
                "Auto-fixes applied. Monitoring continues. "
                "Will pause if anomaly persists on next check."
            )
        else:
            action = "Continuing with warning. Monitor closely."

        report = {
            "severity": severity,
            "findings": findings,
            "fixes_applied": fixes,
            "action_taken": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stats": self.get_live_stats(),
        }

        self.anomaly_history.append(report)
        logger.warning(f"ANOMALY [{severity}]: {findings}")
        if fixes:
            logger.info(f"Auto-fixes applied: {fixes}")

        return report

    # ── Persistence ──

    def save_state(self):
        """Save tracker state to disk for post-hoc analysis."""
        path = Path(LOG_DIR) / "signal_tracker"
        path.mkdir(parents=True, exist_ok=True)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state = {
            "date": today,
            "comparison": self.get_comparison_report(),
            "anomaly_history": self.anomaly_history,
        }

        filepath = path / f"{today}.json"
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Signal tracker state saved to {filepath}")
