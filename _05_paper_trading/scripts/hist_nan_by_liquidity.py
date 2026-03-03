"""
Compute historical trade NaN rates by liquidity bin — for comparison with live data.
Run on Spartan via srun.
"""
import os
import numpy as np
import pandas as pd

RES_DIR = "/data/projects/punim2039/alpha_odds/res/"

trade_cols = ['qty_count_3_1', 'qty_mean_3_1', 'prc_mean_3_1',
              'qty_count_2_1', 'qty_mean_2_1', 'prc_mean_2_1']

for t_def in range(4):
    path = os.path.join(RES_DIR, f"features_t{t_def}", "greyhound_au_features_merged.parquet")
    df = pd.read_parquet(path)

    # Filter to 2025
    if "marketTime_local" in df.columns:
        df["_year"] = pd.to_datetime(df["marketTime_local"], errors="coerce").dt.year
        df = df[df["_year"] == 2025].copy()

    # Liquidity per runner
    df['liquidity'] = df['total_back_qty_m0'] + df['total_lay_qty_m0']

    # Trade NaN rate per runner
    avail = [c for c in trade_cols if c in df.columns]
    df['trade_nan'] = df[avail].isna().mean(axis=1)

    # Race-level liquidity
    race_liq = df.groupby('file_name')['liquidity'].sum().rename('race_liquidity')
    df = df.merge(race_liq, on='file_name', how='left')

    # Per-runner liquidity quintiles
    df['runner_liq_bin'] = pd.qcut(df['liquidity'], q=5, labels=['Q1_low','Q2','Q3','Q4','Q5_high'], duplicates='drop')

    # Per-race liquidity quintiles
    race_liq_vals = df.groupby('file_name')['race_liquidity'].first()
    race_bins = pd.qcut(race_liq_vals, q=5, labels=['Q1_low','Q2','Q3','Q4','Q5_high'], duplicates='drop')
    race_bin_map = race_bins.to_dict()
    df['race_liq_bin'] = df['file_name'].map(race_bin_map)

    print(f"\n=== t_def={t_def} (2025 OOS, {len(df)} runners, {df['file_name'].nunique()} races) ===")

    # By runner liquidity
    print("\nPer-runner trade NaN rate by runner liquidity quintile:")
    g1 = df.groupby('runner_liq_bin').agg(
        n=('trade_nan','count'),
        avg_liq=('liquidity','mean'),
        trade_nan=('trade_nan','mean'),
    )
    for idx, row in g1.iterrows():
        print(f"  {idx:<10} n={row['n']:>6.0f}  avg_liq={row['avg_liq']:>9.0f}  trade_NaN={row['trade_nan']*100:>5.1f}%")

    # By race liquidity
    print("\nPer-runner trade NaN rate by race liquidity quintile:")
    g2 = df.groupby('race_liq_bin').agg(
        n=('trade_nan','count'),
        avg_race_liq=('race_liquidity','mean'),
        trade_nan=('trade_nan','mean'),
    )
    for idx, row in g2.iterrows():
        print(f"  {idx:<10} n={row['n']:>6.0f}  avg_race_liq={row['avg_race_liq']:>9.0f}  trade_NaN={row['trade_nan']*100:>5.1f}%")

    # Overall
    print(f"\n  Overall trade NaN = {df['trade_nan'].mean()*100:.1f}%")

    # Also: per-race NaN rate distribution
    race_nan = df.groupby('file_name')['trade_nan'].mean()
    print(f"  Per-race NaN rate: mean={race_nan.mean()*100:.1f}%, median={race_nan.median()*100:.1f}%, p75={race_nan.quantile(0.75)*100:.1f}%, p95={race_nan.quantile(0.95)*100:.1f}%")
