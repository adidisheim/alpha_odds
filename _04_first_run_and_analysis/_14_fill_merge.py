"""
Merge fill simulation results from parallel tasks.

Usage: python3 _14_fill_merge.py

Reads: res/fill_simulation/fill_part_{0..29}.parquet
Writes: res/fill_simulation/fill_results_merged.parquet

Validates all 30 parts exist, checks row counts, and reports summary stats.
"""

import pandas as pd
import numpy as np
import os
import sys
from parameters import Constant

N_TASKS = 30


if __name__ == '__main__':
    load_dir = f'{Constant.RES_DIR}/fill_simulation/'
    print(f"=== Merging Fill Simulation Results ===", flush=True)
    print(f"Load dir: {load_dir}", flush=True)

    # ── Step 1: Check all parts exist ──
    missing = []
    for i in range(N_TASKS):
        path = f'{load_dir}/fill_part_{i}.parquet'
        if not os.path.exists(path):
            missing.append(i)

    if missing:
        print(f"FATAL: Missing parts: {missing}", flush=True)
        print("Fill simulation is not complete. Exiting.", flush=True)
        sys.exit(1)
    print(f"All {N_TASKS} parts found.", flush=True)

    # ── Step 2: Load and report each part ──
    parts = []
    for i in range(N_TASKS):
        path = f'{load_dir}/fill_part_{i}.parquet'
        df_part = pd.read_parquet(path)
        parts.append(df_part)
        print(f"  Part {i:>2}: {df_part.shape[0]:>8,} rows, {df_part.shape[1]:>3} cols", flush=True)

    # ── Step 3: Merge ──
    df = pd.concat(parts, ignore_index=True)
    print(f"\nMerged shape: {df.shape[0]:,} rows x {df.shape[1]} cols", flush=True)

    # ── Step 4: Validate ──
    print(f"\nColumns: {list(df.columns)}", flush=True)

    n_unique_keys = df['key'].nunique()
    n_unique_markets = df['file_name'].nunique()
    print(f"Unique runners (keys): {n_unique_keys:,}", flush=True)
    print(f"Unique markets: {n_unique_markets:,}", flush=True)

    # Check per-scenario counts
    print(f"\nScenario breakdown:", flush=True)
    for window in df['window'].unique():
        for variant in df['price_variant'].unique():
            sub = df[(df['window'] == window) & (df['price_variant'] == variant)]
            fill_rate_c = sub['conservative_fill'].mean()
            fill_rate_m = sub['moderate_fill'].mean()
            print(f"  {window} / {variant}: {len(sub):,} rows, "
                  f"conservative fill={fill_rate_c:.1%}, moderate fill={fill_rate_m:.1%}", flush=True)

    # Edge distribution
    print(f"\nEdge distribution (across all scenarios):", flush=True)
    base = df[(df['window'] == 't0_60s') & (df['price_variant'] == 'best_back')]
    for thresh in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        n = (base['edge'] > thresh).sum()
        print(f"  edge > {thresh:.0%}: {n:,} runners", flush=True)

    # ── Step 5: Save merged file ──
    save_path = f'{load_dir}/fill_results_merged.parquet'
    df.to_parquet(save_path)
    file_size = os.path.getsize(save_path) / 1024**2
    print(f"\nSaved merged results to: {save_path}", flush=True)
    print(f"File size: {file_size:.1f} MB", flush=True)
    print("Done!", flush=True)
