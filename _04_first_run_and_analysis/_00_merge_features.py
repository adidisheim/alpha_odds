"""
Merge and validate feature parquet parts into a single file per t_definition.

Usage:
    python _00_merge_features.py <t_definition>

    t_definition: 0, 1, 2, or 3

Reads: res/features_t{t_definition}/greyhound_au_features_part_{0-9}.parquet
Writes: res/features_t{t_definition}/greyhound_au_features_merged.parquet

Validation checks:
    - All 10 parts exist
    - Column counts are consistent
    - No _x/_y duplicate columns
    - Row counts reported
    - Win rate sanity check (should be ~12.5% for 8-runner races)
    - No fully-null columns
"""

import pandas as pd
import numpy as np
import os
import sys
from parameters import Constant
from utils_locals.parser import parse


if __name__ == '__main__':
    args = parse()
    t_definition = args.a

    load_dir = f'{Constant.RES_DIR}/features_t{t_definition}'
    print(f"=== Merging features for t_definition={t_definition} ===", flush=True)
    print(f"Load dir: {load_dir}", flush=True)

    # ── Step 1: Check all parts exist ──
    missing = []
    for i in range(10):
        path = f'{load_dir}/greyhound_au_features_part_{i}.parquet'
        if not os.path.exists(path):
            missing.append(i)

    if missing:
        print(f"FATAL: Missing parts: {missing}", flush=True)
        print("Feature engineering is not complete. Exiting.", flush=True)
        sys.exit(1)
    print("All 10 parts found.", flush=True)

    # ── Step 2: Load and validate each part ──
    parts = []
    col_counts = {}
    for i in range(10):
        path = f'{load_dir}/greyhound_au_features_part_{i}.parquet'
        df_part = pd.read_parquet(path)
        col_counts[i] = len(df_part.columns)
        parts.append(df_part)
        print(f"  Part {i}: {df_part.shape[0]:>8,} rows, {df_part.shape[1]:>3} cols", flush=True)

    # Check column consistency
    unique_col_counts = set(col_counts.values())
    if len(unique_col_counts) > 1:
        print(f"WARNING: Inconsistent column counts across parts: {col_counts}", flush=True)
    else:
        print(f"Column count consistent: {unique_col_counts.pop()} cols in all parts.", flush=True)

    # ── Step 3: Merge ──
    df = pd.concat(parts, ignore_index=True)
    print(f"\nMerged shape: {df.shape[0]:,} rows × {df.shape[1]} cols", flush=True)

    # ── Step 4: Validate merged data ──
    # Check for _x/_y columns (sign of duplicate merges)
    xy_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    if xy_cols:
        print(f"FATAL: Found _x/_y duplicate columns: {xy_cols}", flush=True)
        sys.exit(1)
    print("No _x/_y duplicate columns.", flush=True)

    # Check for fully null columns
    null_cols = [c for c in df.columns if df[c].isna().all()]
    if null_cols:
        print(f"WARNING: Fully null columns: {null_cols}", flush=True)

    # Win rate sanity check
    df['win'] = (df['id'] == -1).astype(int)
    win_rate = df['win'].mean()
    n_races = df['file_name'].nunique()
    n_runners = len(df)
    avg_runners_per_race = n_runners / n_races if n_races > 0 else 0
    print(f"\nRaces: {n_races:,}", flush=True)
    print(f"Runners: {n_runners:,}", flush=True)
    print(f"Avg runners/race: {avg_runners_per_race:.1f}", flush=True)
    print(f"Win rate: {win_rate:.4f} (expected ~{1/avg_runners_per_race:.4f} for {avg_runners_per_race:.0f}-runner races)", flush=True)

    # Year distribution
    if 'marketTime_local' in df.columns:
        years = pd.to_datetime(df['marketTime_local']).dt.year.value_counts().sort_index()
        print(f"\nYear distribution:", flush=True)
        for year, count in years.items():
            print(f"  {year}: {count:>8,} rows", flush=True)

    # Column list
    print(f"\nAll columns ({len(df.columns)}):", flush=True)
    for c in sorted(df.columns):
        print(f"  {c}", flush=True)

    # ── Step 5: Save merged file ──
    save_path = f'{load_dir}/greyhound_au_features_merged.parquet'
    df.drop(columns=['win']).to_parquet(save_path)
    print(f"\nSaved merged features to: {save_path}", flush=True)
    print(f"File size: {os.path.getsize(save_path) / 1024**3:.2f} GB", flush=True)
    print("Done!", flush=True)
