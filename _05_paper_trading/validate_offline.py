"""
Offline Feature Validator — validates already-collected feature logs against historical baselines.

Usage:
    python validate_offline.py [--date 2026-02-28] [--baselines-dir baselines/]

Loads feature parquets from paper_trading_logs/features/, runs FeatureValidator
per t_def, prints console report and saves JSON to paper_trading_logs/validation/.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LOG_DIR, VALIDATION_BASELINES_DIR
from feature_validator import FeatureValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_feature_logs(log_dir, date_str):
    """Load raw feature logs for a given date.

    Returns dict of t_def -> DataFrame.
    """
    feature_path = Path(log_dir) / "features" / f"{date_str}.parquet"
    if not feature_path.exists():
        logger.error(f"Feature log not found: {feature_path}")
        return {}

    df = pd.read_parquet(feature_path)
    logger.info(f"Loaded {len(df)} feature rows from {feature_path}")

    if "t_def" not in df.columns:
        logger.error("Feature log missing 't_def' column")
        return {}

    # Split by t_def
    result = {}
    for t_def in sorted(df["t_def"].unique()):
        subset = df[df["t_def"] == t_def].copy()
        n_markets = subset["file_name"].nunique() if "file_name" in subset.columns else "?"
        n_runners = len(subset)
        logger.info(f"  t_def={t_def}: {n_runners} runners across {n_markets} markets")
        result[int(t_def)] = subset

    return result


def main():
    parser = argparse.ArgumentParser(description="Offline Feature Validator")
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date to validate (YYYY-MM-DD), default: today",
    )
    parser.add_argument(
        "--baselines-dir",
        default=str(VALIDATION_BASELINES_DIR),
        help="Path to baselines directory",
    )
    parser.add_argument(
        "--log-dir",
        default=str(LOG_DIR),
        help="Path to paper_trading_logs directory",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save validation results as JSON (default: True)",
    )
    args = parser.parse_args()

    print(f"\nFeature Validation: date={args.date}")
    print(f"  Baselines: {args.baselines_dir}")
    print(f"  Log dir:   {args.log_dir}")

    # Load feature logs
    features_by_tdef = load_feature_logs(args.log_dir, args.date)
    if not features_by_tdef:
        print("\nNo feature data to validate. Exiting.")
        sys.exit(1)

    # Load validator
    validator = FeatureValidator(baselines_dir=args.baselines_dir)
    if not validator.load_baselines():
        print("\nFailed to load baselines. Exiting.")
        sys.exit(1)

    # Run validation
    result = validator.validate_all_t_defs(features_by_tdef)

    # Print report
    validator.print_report(result)

    # Save JSON
    if args.save_json:
        val_dir = Path(args.log_dir) / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        json_path = val_dir / f"{args.date}.json"

        # Convert to JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            return str(obj)

        with open(json_path, "w") as f:
            json.dump(make_serializable(result), f, indent=2)
        print(f"\nResults saved to {json_path}")

    # Exit code
    if result["overall_passed"]:
        print("\nVALIDATION PASSED")
        sys.exit(0)
    else:
        print("\nVALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
