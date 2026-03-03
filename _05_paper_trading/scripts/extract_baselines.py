"""
Extract Historical Feature Baselines — run on Spartan.

Loads merged feature parquets for each t_def, filters to 2025 OOS,
computes per-feature distribution statistics, and saves baseline parquets.

Usage (on Spartan via srun):
    python extract_baselines.py

Output:
    res/baselines/feature_baselines_overall.parquet
    res/baselines/feature_baselines_by_rank.parquet
"""

import os
import sys
import numpy as np
import pandas as pd

# Detect environment
HOSTNAME = os.uname().nodename
if "spartan" in HOSTNAME.lower() or HOSTNAME.startswith("sp"):
    RES_DIR = "/data/projects/punim2039/alpha_odds/res/"
    SAVE_DIR = "/data/projects/punim2039/alpha_odds/res/baselines/"
else:
    RES_DIR = "./res/"
    SAVE_DIR = "./res/baselines/"

os.makedirs(SAVE_DIR, exist_ok=True)

# Feature groups for reporting
FEATURE_GROUPS = {
    "order_book": ["best_back", "best_lay", "best_back_cum_qty", "best_lay_cum_qty",
                   "total_back_qty", "total_lay_qty", "best_back_q_100", "best_lay_q_100",
                   "best_back_q_1000", "best_lay_q_1000", "tot_bl_imbalance", "best_bl_imbalance"],
    "momentum": ["mom_3_1", "mom_2_1"],
    "volatility": ["std_3_1", "std_2_1"],
    "trade_activity": ["count_3_1", "count_2_1", "mean_3_1", "mean_2_1", "std_3_1", "std_2_1",
                       "order_is_back"],
    "fraction": ["_frac"],
    "cross_runner": ["prob_rank", "prob_vs_favorite", "prob_share", "race_herfindahl",
                     "n_close_runners", "spread_m0", "spread_rank", "total_qty_m0",
                     "volume_rank", "avg_mom_3_1", "momentum_rank", "race_overround",
                     "is_favorite", "prob_deviation", "bl_imbalance_rank"],
    "fixed_effects": ["local_dow", "marketBaseRate", "numberOfActiveRunners", "runner_position"],
}


def classify_feature(col):
    """Classify a feature column into a group."""
    # Cross-runner features (exact match)
    if col in FEATURE_GROUPS["cross_runner"]:
        return "cross_runner"
    if col in FEATURE_GROUPS["fixed_effects"]:
        return "fixed_effects"
    if "_frac" in col:
        return "fraction"
    # Trade activity features
    for pattern in ["count_3_1", "count_2_1", "qty_mean", "qty_std", "prc_mean", "prc_std",
                    "order_is_back"]:
        if pattern in col:
            return "trade_activity"
    if "mom_3_1" in col or "mom_2_1" in col:
        return "momentum"
    if "std_3_1" in col or "std_2_1" in col:
        return "volatility"
    for pattern in FEATURE_GROUPS["order_book"]:
        if pattern in col:
            return "order_book"
    return "other"


def add_derived_features(df):
    """Add fraction and cross-runner features to match live pipeline."""
    # Fraction features
    if "total_back_qty_m1" in df.columns and "total_lay_qty_m1" in df.columns:
        df["total_qty_m1"] = df[["total_back_qty_m1", "total_lay_qty_m1"]].sum(axis=1)
    if "total_back_qty_m3" in df.columns and "total_lay_qty_m3" in df.columns:
        df["total_qty_m3"] = df[["total_back_qty_m3", "total_lay_qty_m3"]].sum(axis=1)

    col_todo = [c for c in ["total_qty_m1", "total_back_qty_m1", "total_lay_qty_m1",
                             "total_qty_m3", "total_back_qty_m3", "total_lay_qty_m3"]
                if c in df.columns]
    for col in col_todo:
        c = col + "_frac"
        race_total = df.groupby("file_name")[col].transform("sum")
        df[c] = df[col] / race_total.replace(0, np.nan)

    # Fraction momentum
    for col in [c for c in df.columns if c.endswith("_m1_frac")]:
        m3_col = col.replace("_m1", "_m3")
        if m3_col in df.columns:
            mom_col = col.replace("_m1", "_mom_3_1")
            df[mom_col] = df[col] - df[m3_col]

    # Cross-runner features
    if "best_back_m0" in df.columns and "best_lay_m0" in df.columns:
        g = df.groupby("file_name")
        df["market_prob"] = df[["best_back_m0", "best_lay_m0"]].mean(axis=1)
        df["prob_rank"] = g["market_prob"].rank(method="min", ascending=False)
        df["prob_vs_favorite"] = df["market_prob"] / g["market_prob"].transform("max")
        df["prob_share"] = df["market_prob"] / g["market_prob"].transform("sum")
        df["_psq"] = df["prob_share"] ** 2
        df["race_herfindahl"] = df.groupby("file_name")["_psq"].transform("sum")
        df.drop(columns=["_psq"], inplace=True)
        race_std = g["market_prob"].transform("std").fillna(0)
        df["n_close_runners"] = (race_std < 0.05).astype(int) * (g["market_prob"].transform("count") - 1)
        df["spread_m0"] = (df["best_back_m0"] - df["best_lay_m0"]).abs()
        df["spread_rank"] = g["spread_m0"].rank(method="min", ascending=True)
        df["total_qty_m0"] = df["total_back_qty_m0"] + df["total_lay_qty_m0"]
        df["volume_rank"] = g["total_qty_m0"].rank(method="min", ascending=False)
        if "best_back_mom_3_1" in df.columns and "best_lay_mom_3_1" in df.columns:
            df["avg_mom_3_1"] = df[["best_back_mom_3_1", "best_lay_mom_3_1"]].mean(axis=1)
            df["momentum_rank"] = g["avg_mom_3_1"].rank(method="min", ascending=False)
        df["race_overround"] = g["market_prob"].transform("sum")
        df["is_favorite"] = (df["prob_rank"] == 1).astype(int)
        df["prob_deviation"] = df["market_prob"] - g["market_prob"].transform("mean")
        if "best_bl_imbalance_m0" in df.columns:
            df["bl_imbalance_m0"] = df["best_bl_imbalance_m0"]
            df["bl_imbalance_rank"] = g["bl_imbalance_m0"].rank(method="min", ascending=False)

    return df


def compute_stats(series):
    """Compute distribution statistics for a feature series."""
    valid = series.dropna()
    n_total = len(series)
    n_valid = len(valid)
    return {
        "nan_rate": 1.0 - n_valid / n_total if n_total > 0 else 1.0,
        "mean": valid.mean() if n_valid > 0 else np.nan,
        "std": valid.std() if n_valid > 1 else np.nan,
        "median": valid.median() if n_valid > 0 else np.nan,
        "p5": valid.quantile(0.05) if n_valid > 0 else np.nan,
        "p25": valid.quantile(0.25) if n_valid > 0 else np.nan,
        "p75": valid.quantile(0.75) if n_valid > 0 else np.nan,
        "p95": valid.quantile(0.95) if n_valid > 0 else np.nan,
        "n_total": n_total,
        "n_valid": n_valid,
    }


def main():
    overall_rows = []
    by_rank_rows = []

    for t_def in range(4):
        merged_path = os.path.join(RES_DIR, f"features_t{t_def}", "greyhound_au_features_merged.parquet")
        print(f"\nLoading t_def={t_def}: {merged_path}", flush=True)

        if not os.path.exists(merged_path):
            print(f"  WARNING: {merged_path} not found, skipping", flush=True)
            continue

        df = pd.read_parquet(merged_path)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns", flush=True)

        # Filter to 2025 OOS
        if "file_name" in df.columns:
            # Extract year from file_name (format: "1.XXXXXXXXX.bz2")
            # Need to check a different way — use marketTime_local if available
            if "marketTime_local" in df.columns:
                df["_year"] = pd.to_datetime(df["marketTime_local"], errors="coerce").dt.year
            else:
                # Fall back: try to infer from the data
                print("  No marketTime_local, using all data as baseline", flush=True)
                df["_year"] = 2025  # Use all data

            n_before = len(df)
            df = df[df["_year"] == 2025].copy()
            print(f"  Filtered to 2025: {len(df)} rows (from {n_before})", flush=True)

            if len(df) == 0:
                print("  No 2025 data, skipping", flush=True)
                continue

        # Add derived features
        df = add_derived_features(df)

        # Compute prob_rank for grouping
        if "prob_rank" not in df.columns:
            if "best_back_m0" in df.columns and "best_lay_m0" in df.columns:
                df["market_prob"] = df[["best_back_m0", "best_lay_m0"]].mean(axis=1)
                df["prob_rank"] = df.groupby("file_name")["market_prob"].rank(
                    method="min", ascending=False
                )

        # Identify feature columns (exclude meta columns)
        meta_cols = {"file_name", "id", "key", "_year", "market_prob", "marketTime_local",
                     "venue", "marketType", "win", "winner"}
        feature_cols = [c for c in df.columns if c not in meta_cols]

        print(f"  Computing baselines for {len(feature_cols)} features...", flush=True)

        # Overall stats
        for col in feature_cols:
            stats = compute_stats(df[col])
            stats["t_def"] = t_def
            stats["feature"] = col
            stats["group"] = classify_feature(col)
            overall_rows.append(stats)

        # By prob_rank stats
        if "prob_rank" in df.columns:
            # Bin into 1,2,3,...,7,8+ groups (drop NaN ranks)
            df["_rank_group"] = df["prob_rank"].clip(upper=8)
            df.loc[df["_rank_group"].notna(), "_rank_group"] = df.loc[df["_rank_group"].notna(), "_rank_group"].astype(int)
            for rank_val in sorted(df["_rank_group"].unique()):
                subset = df[df["_rank_group"] == rank_val]
                for col in feature_cols:
                    stats = compute_stats(subset[col])
                    stats["t_def"] = t_def
                    stats["feature"] = col
                    stats["group"] = classify_feature(col)
                    stats["prob_rank"] = rank_val
                    by_rank_rows.append(stats)

        print(f"  Done: {len(overall_rows)} overall stats, {len(by_rank_rows)} by-rank stats", flush=True)

    # Save
    overall_df = pd.DataFrame(overall_rows)
    by_rank_df = pd.DataFrame(by_rank_rows)

    overall_path = os.path.join(SAVE_DIR, "feature_baselines_overall.parquet")
    by_rank_path = os.path.join(SAVE_DIR, "feature_baselines_by_rank.parquet")

    overall_df.to_parquet(overall_path)
    by_rank_df.to_parquet(by_rank_path)

    print(f"\nSaved overall baselines: {overall_path} ({len(overall_df)} rows)")
    print(f"Saved by-rank baselines: {by_rank_path} ({len(by_rank_df)} rows)")

    # Print summary
    print("\n=== Baseline Summary ===")
    for t_def in overall_df["t_def"].unique():
        subset = overall_df[overall_df["t_def"] == t_def]
        print(f"\nt_def={t_def}: {len(subset)} features")
        for group in subset["group"].unique():
            g = subset[subset["group"] == group]
            avg_nan = g["nan_rate"].mean()
            print(f"  {group}: {len(g)} features, avg NaN rate = {avg_nan:.1%}")


if __name__ == "__main__":
    main()
