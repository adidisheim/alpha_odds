"""
Merge all inplay_{year}_{month}_{day}.parquet files into a single
in_play_times_all.parquet lookup table. Run after all _04_extract_in_play_times jobs complete.

Usage:
    python _05_merge_in_play_times.py
"""

import pandas as pd
import os
from parameters import Constant

if __name__ == '__main__':
    base_path = Constant.DATA_DIR + 'p/greyhound_au/'
    inplay_files = sorted([f for f in os.listdir(base_path) if f.startswith('inplay_') and f.endswith('.parquet')])

    print(f'Found {len(inplay_files)} in-play files', flush=True)

    if len(inplay_files) == 0:
        print('No in-play files found!', flush=True)
        exit(1)

    dfs = []
    for f in inplay_files:
        df = pd.read_parquet(base_path + f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    # Drop duplicates (same file_name may appear if day partitions overlap somehow)
    merged = merged.drop_duplicates(subset=['file_name'], keep='first')

    n_total = len(merged)
    n_inplay = merged['in_play_time'].notna().sum()
    n_nat = merged['in_play_time'].isna().sum()

    print(f'Total markets: {n_total}', flush=True)
    print(f'With in-play time: {n_inplay} ({n_inplay/n_total*100:.1f}%)', flush=True)
    print(f'Without in-play time (NaT): {n_nat} ({n_nat/n_total*100:.1f}%)', flush=True)

    out_path = base_path + 'in_play_times_all.parquet'
    merged.to_parquet(out_path, index=False)
    print(f'Saved merged file: {out_path}', flush=True)

    # Also compare against win_ files to see coverage
    win_files = sorted([f for f in os.listdir(base_path) if f.startswith('win_') and f.endswith('.parquet')])
    if win_files:
        sample = pd.read_parquet(base_path + win_files[0])
        win_fnames = set(sample['file_name'].unique())
        inplay_fnames = set(merged.loc[merged['in_play_time'].notna(), 'file_name'])
        coverage = len(win_fnames & inplay_fnames) / len(win_fnames) * 100 if win_fnames else 0
        print(f'\nSample coverage check ({win_files[0]}):')
        print(f'  Win file markets: {len(win_fnames)}')
        print(f'  With in-play match: {len(win_fnames & inplay_fnames)} ({coverage:.1f}%)')

    # Summary stats on in-play times
    valid = merged.loc[merged['in_play_time'].notna()].copy()
    print(f'\nIn-play time range: {valid["in_play_time"].min()} to {valid["in_play_time"].max()}')

    print('\nAll done!', flush=True)
