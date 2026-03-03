"""
Feature Validator — compares live feature distributions against historical baselines.

Loads pre-computed baseline statistics (from extract_baselines.py) and runs
diagnostic checks on live features to detect distribution drift.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    VALIDATION_BASELINES_DIR,
    VALIDATION_MEAN_Z_THRESHOLD,
    VALIDATION_NAN_TOLERANCE,
    VALIDATION_RANGE_VIOLATION_MAX,
    VALIDATION_STD_RATIO_HIGH,
    VALIDATION_STD_RATIO_LOW,
)

logger = logging.getLogger(__name__)

# Feature groups for structured reporting
FEATURE_GROUPS = {
    "order_book_levels": lambda c: any(p in c for p in [
        "best_back_m", "best_lay_m", "best_back_q_", "best_lay_q_",
        "best_back_cum_qty_m", "best_lay_cum_qty_m",
        "total_back_qty_m", "total_lay_qty_m",
    ]) and "mom" not in c and "std" not in c,
    "momentum": lambda c: "mom_3_1" in c or "mom_2_1" in c,
    "volatility": lambda c: "std_3_1" in c or "std_2_1" in c,
    "trade_activity": lambda c: any(p in c for p in [
        "count_3_1", "count_2_1", "qty_mean", "qty_std", "prc_mean", "prc_std",
        "order_is_back",
    ]),
    "fraction": lambda c: "_frac" in c,
    "cross_runner": lambda c: c in {
        "prob_rank", "prob_vs_favorite", "prob_share", "race_herfindahl",
        "n_close_runners", "spread_m0", "spread_rank", "total_qty_m0",
        "volume_rank", "avg_mom_3_1", "momentum_rank", "race_overround",
        "is_favorite", "prob_deviation", "bl_imbalance_rank",
    },
    "fixed_effects": lambda c: c in {
        "local_dow", "marketBaseRate", "numberOfActiveRunners", "runner_position",
    },
}


def classify_feature(col):
    """Classify a feature column into a group."""
    for group, matcher in FEATURE_GROUPS.items():
        if matcher(col):
            return group
    return "other"


class CheckResult:
    """Result of a single validation check."""
    def __init__(self, feature, check_name, passed, live_value, hist_value, threshold, detail=""):
        self.feature = feature
        self.check_name = check_name
        self.passed = passed
        self.live_value = live_value
        self.hist_value = hist_value
        self.threshold = threshold
        self.detail = detail

    def to_dict(self):
        return {
            "feature": self.feature,
            "check": self.check_name,
            "passed": self.passed,
            "live": self.live_value,
            "historical": self.hist_value,
            "threshold": self.threshold,
            "detail": self.detail,
        }


class FeatureValidator:
    """Compares live feature distributions against historical baselines."""

    def __init__(self, baselines_dir=None):
        self.baselines_dir = Path(baselines_dir or VALIDATION_BASELINES_DIR)
        self.overall = None
        self.by_rank = None
        self._loaded = False

    def load_baselines(self):
        """Load baseline parquets."""
        overall_path = self.baselines_dir / "feature_baselines_overall.parquet"
        by_rank_path = self.baselines_dir / "feature_baselines_by_rank.parquet"

        if not overall_path.exists():
            logger.error(f"Baseline file not found: {overall_path}")
            return False

        self.overall = pd.read_parquet(overall_path)
        if by_rank_path.exists():
            self.by_rank = pd.read_parquet(by_rank_path)
        else:
            logger.warning(f"By-rank baselines not found: {by_rank_path}")

        self._loaded = True
        logger.info(
            f"Loaded baselines: {len(self.overall)} overall stats"
            + (f", {len(self.by_rank)} by-rank stats" if self.by_rank is not None else "")
        )
        return True

    def validate(self, live_features_df, t_def):
        """
        Run all validation checks on live features for a given t_def.

        Args:
            live_features_df: DataFrame with one row per (file_name, id),
                              same schema as raw_features from FeatureComputer.
            t_def: int (0-3)

        Returns:
            dict with keys: passed (bool), checks (list of CheckResult dicts),
                  summary (dict of group -> {n_checks, n_passed, n_failed}),
                  overall_pass_rate (float)
        """
        if not self._loaded:
            if not self.load_baselines():
                return {"passed": False, "error": "Baselines not loaded"}

        baseline = self.overall[self.overall["t_def"] == t_def]
        if baseline.empty:
            return {"passed": False, "error": f"No baseline for t_def={t_def}"}

        # Index baseline by feature name for fast lookup
        bl = baseline.set_index("feature")

        checks = []
        meta_cols = {"file_name", "id", "key", "market_id", "t_def", "timestamp",
                     "market_prob", "marketTime_local", "venue", "marketType",
                     "win", "winner", "_year", "_rank_group"}

        feature_cols = [c for c in live_features_df.columns
                        if c not in meta_cols and c in bl.index]

        for col in feature_cols:
            hist = bl.loc[col]
            live_series = live_features_df[col]

            # Check 1: NaN rate
            live_nan = live_series.isna().mean()
            hist_nan = hist["nan_rate"]
            nan_diff = abs(live_nan - hist_nan)
            checks.append(CheckResult(
                feature=col,
                check_name="nan_rate",
                passed=nan_diff < VALIDATION_NAN_TOLERANCE,
                live_value=round(live_nan, 4),
                hist_value=round(hist_nan, 4),
                threshold=VALIDATION_NAN_TOLERANCE,
                detail=f"diff={nan_diff:.4f}",
            ))

            # Check 2: Mean shift (z-score)
            live_valid = live_series.dropna()
            if len(live_valid) > 0 and pd.notna(hist["std"]) and hist["std"] > 0:
                live_mean = live_valid.mean()
                z_score = abs(live_mean - hist["mean"]) / hist["std"]
                checks.append(CheckResult(
                    feature=col,
                    check_name="mean_shift",
                    passed=z_score < VALIDATION_MEAN_Z_THRESHOLD,
                    live_value=round(float(live_mean), 6),
                    hist_value=round(float(hist["mean"]), 6),
                    threshold=VALIDATION_MEAN_Z_THRESHOLD,
                    detail=f"z={z_score:.2f}",
                ))

            # Check 3: Std ratio
            if len(live_valid) > 1 and pd.notna(hist["std"]) and hist["std"] > 0:
                live_std = live_valid.std()
                ratio = live_std / hist["std"] if hist["std"] > 0 else float("inf")
                checks.append(CheckResult(
                    feature=col,
                    check_name="std_ratio",
                    passed=VALIDATION_STD_RATIO_LOW < ratio < VALIDATION_STD_RATIO_HIGH,
                    live_value=round(float(live_std), 6),
                    hist_value=round(float(hist["std"]), 6),
                    threshold=f"[{VALIDATION_STD_RATIO_LOW}, {VALIDATION_STD_RATIO_HIGH}]",
                    detail=f"ratio={ratio:.2f}",
                ))

            # Check 4: Range violations (fraction outside [p5, p95])
            if len(live_valid) > 0 and pd.notna(hist["p5"]) and pd.notna(hist["p95"]):
                outside = ((live_valid < hist["p5"]) | (live_valid > hist["p95"])).mean()
                checks.append(CheckResult(
                    feature=col,
                    check_name="range_violation",
                    passed=outside < VALIDATION_RANGE_VIOLATION_MAX,
                    live_value=round(float(outside), 4),
                    hist_value=f"[{hist['p5']:.4f}, {hist['p95']:.4f}]",
                    threshold=VALIDATION_RANGE_VIOLATION_MAX,
                    detail=f"{outside:.1%} outside [p5,p95]",
                ))

        # Build summary by group
        summary = {}
        for check in checks:
            group = classify_feature(check.feature)
            if group not in summary:
                summary[group] = {"n_checks": 0, "n_passed": 0, "n_failed": 0, "failures": []}
            summary[group]["n_checks"] += 1
            if check.passed:
                summary[group]["n_passed"] += 1
            else:
                summary[group]["n_failed"] += 1
                summary[group]["failures"].append(check.to_dict())

        total_checks = len(checks)
        total_passed = sum(1 for c in checks if c.passed)
        overall_pass_rate = total_passed / total_checks if total_checks > 0 else 0.0

        # Overall pass: all groups have >= 80% pass rate
        group_pass_rates = {}
        for group, s in summary.items():
            rate = s["n_passed"] / s["n_checks"] if s["n_checks"] > 0 else 1.0
            group_pass_rates[group] = rate

        all_passed = all(rate >= 0.80 for rate in group_pass_rates.values())

        return {
            "passed": all_passed,
            "t_def": t_def,
            "n_features_checked": len(feature_cols),
            "n_checks": total_checks,
            "n_passed": total_passed,
            "overall_pass_rate": round(overall_pass_rate, 4),
            "group_pass_rates": {k: round(v, 4) for k, v in group_pass_rates.items()},
            "summary": {k: {kk: vv for kk, vv in v.items() if kk != "failures"}
                        for k, v in summary.items()},
            "failures_by_group": {k: v["failures"] for k, v in summary.items() if v["failures"]},
        }

    def validate_all_t_defs(self, live_features_by_tdef):
        """
        Validate live features for all t_defs.

        Args:
            live_features_by_tdef: dict of t_def -> DataFrame

        Returns:
            dict with per-t_def results and overall pass/fail
        """
        results = {}
        all_passed = True
        for t_def, df in live_features_by_tdef.items():
            result = self.validate(df, t_def)
            results[t_def] = result
            if not result.get("passed", False):
                all_passed = False

        return {
            "overall_passed": all_passed,
            "t_defs": results,
        }

    def print_report(self, validation_result):
        """Print a human-readable validation report."""
        if "error" in validation_result:
            print(f"ERROR: {validation_result['error']}")
            return

        if "t_defs" in validation_result:
            # Multi-t_def result
            status = "PASS" if validation_result["overall_passed"] else "FAIL"
            print(f"\n{'='*60}")
            print(f"Feature Validation Report: {status}")
            print(f"{'='*60}")
            for t_def, result in sorted(validation_result["t_defs"].items()):
                self._print_single_report(result)
        else:
            # Single t_def result
            self._print_single_report(validation_result)

    def _print_single_report(self, result):
        """Print report for a single t_def."""
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            return

        status = "PASS" if result["passed"] else "FAIL"
        t_def = result.get("t_def", "?")
        print(f"\n--- t_def={t_def}: {status} ---")
        print(f"  Features checked: {result['n_features_checked']}")
        print(f"  Total checks: {result['n_checks']} (passed: {result['n_passed']})")
        print(f"  Overall pass rate: {result['overall_pass_rate']:.1%}")
        print()

        # Group summary
        print(f"  {'Group':<20} {'Checks':>7} {'Passed':>7} {'Rate':>7}")
        print(f"  {'-'*45}")
        for group, stats in sorted(result["summary"].items()):
            rate = result["group_pass_rates"].get(group, 0)
            marker = " *" if rate < 0.80 else ""
            print(f"  {group:<20} {stats['n_checks']:>7} {stats['n_passed']:>7} {rate:>6.1%}{marker}")

        # Top failures
        failures = result.get("failures_by_group", {})
        if failures:
            print(f"\n  Top failures:")
            count = 0
            for group, group_failures in sorted(failures.items()):
                for f in group_failures[:3]:  # Show top 3 per group
                    print(f"    [{group}] {f['feature']}: {f['check']} "
                          f"(live={f['live']}, hist={f['historical']}, {f['detail']})")
                    count += 1
                    if count >= 15:
                        remaining = sum(len(v) for v in failures.values()) - count
                        if remaining > 0:
                            print(f"    ... and {remaining} more failures")
                        return
