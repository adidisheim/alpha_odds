"""
Compute dog track record features from historical data.
Merges dog names with existing features, then computes per-dog
expanding/rolling historical features (strictly no lookahead).

Usage: python _06_02_compute_dog_features.py <t_definition>
  arg a = t_definition (0, 1, 2, or 3)

Outputs:
  res/dog_features/features_with_dog_t{t_def}.parquet
"""

import pandas as pd
import numpy as np
import os
from utils_locals.parser import parse
from parameters import Constant


def compute_dog_track_features(df):
    """
    Compute per-dog historical features. Strictly no lookahead:
    for each race, only uses data from prior races of the same dog.

    Assumes df is sorted by (dog_name, market_time_dt, file_name).
    """
    # Create win indicator from id convention (id == -1 means winner)
    df['won'] = (df['id'] == -1).astype(np.int8)

    # Sort chronologically per dog
    df = df.sort_values(['dog_name', 'market_time_dt', 'file_name']).copy()

    g = df.groupby('dog_name')

    # 1. Number of prior races (0-indexed: first race = 0 prior races)
    df['dog_n_races'] = g.cumcount()

    # 2. Expanding win rate (shifted to exclude current race)
    df['dog_cum_wins'] = g['won'].transform(lambda x: x.cumsum().shift(1).fillna(0))
    df['dog_win_rate'] = np.where(
        df['dog_n_races'] > 0,
        df['dog_cum_wins'] / df['dog_n_races'],
        np.nan
    )

    # 3. Recent form: win rate in last 5 races
    df['dog_win_rate_last5'] = g['won'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # 4. Recent form: win rate in last 10 races
    df['dog_win_rate_last10'] = g['won'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    # 5. Average market implied probability from prior races
    if 'best_back_m0' in df.columns:
        df['dog_avg_market_prob'] = g['best_back_m0'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        # 6. Overperformance: win_rate - avg market prob
        df['dog_overperformance'] = df['dog_win_rate'] - df['dog_avg_market_prob']
    else:
        print('WARNING: best_back_m0 not found in features, skipping market prob features')

    # 7. Days since last race
    race_dates = df['market_time_dt'].dt.normalize()  # strip time for day comparison
    prev_date = g['market_time_dt'].shift(1)
    df['dog_days_since_last'] = (df['market_time_dt'] - prev_date).dt.total_seconds() / 86400.0

    # 8. Average runner position from prior races
    df['dog_avg_position'] = g['runner_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 9. Venue-specific features
    gv = df.groupby(['dog_name', 'venue'])
    df['dog_venue_n_races'] = gv.cumcount()
    df['dog_venue_cum_wins'] = gv['won'].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    df['dog_venue_win_rate'] = np.where(
        df['dog_venue_n_races'] > 0,
        df['dog_venue_cum_wins'] / df['dog_venue_n_races'],
        np.nan
    )

    # 10. Streak: consecutive wins/losses coming into this race
    def compute_streak(series):
        """Compute current streak length (positive = wins, negative = losses)."""
        shifted = series.shift(1)
        streaks = []
        current_streak = 0
        for val in shifted:
            if pd.isna(val):
                streaks.append(np.nan)
                continue
            if val == 1:
                current_streak = max(1, current_streak + 1)
            else:
                current_streak = min(-1, current_streak - 1)
            streaks.append(current_streak)
        return pd.Series(streaks, index=series.index)

    df['dog_streak'] = g['won'].transform(compute_streak)

    # Clean up intermediate columns
    df.drop(columns=['dog_cum_wins', 'dog_venue_cum_wins', 'won'], inplace=True)

    return df


if __name__ == '__main__':
    args = parse()
    t_def = args.a

    save_dir = Constant.RES_DIR + 'dog_features/'
    os.makedirs(save_dir, exist_ok=True)

    # Load dog names mapping
    dog_names_path = save_dir + 'dog_names_mapping.parquet'
    print(f'Loading dog names from {dog_names_path}', flush=True)
    dog_names = pd.read_parquet(dog_names_path)

    # Keep only the columns we need for merging
    dog_merge = dog_names[['file_name', 'position_num', 'dog_name', 'market_time_dt']].copy()
    # Remove duplicates (same dog in same market)
    dog_merge = dog_merge.drop_duplicates(subset=['file_name', 'position_num'])

    # Load features (prefer merged file, fallback to parts)
    features_dir = f'{Constant.RES_DIR}features_t{t_def}/'
    merged_path = features_dir + 'greyhound_au_features_merged.parquet'
    if os.path.exists(merged_path):
        print(f'Loading merged features from {merged_path}', flush=True)
        df = pd.read_parquet(merged_path)
    else:
        print(f'Merged file not found, loading parts from {features_dir}', flush=True)
        df = pd.DataFrame()
        for i in range(10):
            part_path = features_dir + f'greyhound_au_features_part_{i}.parquet'
            try:
                df = pd.concat([df, pd.read_parquet(part_path)], ignore_index=False)
            except Exception as e:
                print(f'Non-fatal ERROR: Could not read part {i}: {e}', flush=True)
    print(f'Features shape: {df.shape}', flush=True)

    # runner_position is float after groupby().mean() and may have NaN; handle gracefully
    df['runner_position_int'] = df['runner_position'].round().fillna(-1).astype(int)

    # Merge dog names into features on (file_name, runner_position)
    print('Merging dog names with features...', flush=True)
    n_before = len(df)
    df = df.merge(
        dog_merge,
        left_on=['file_name', 'runner_position_int'],
        right_on=['file_name', 'position_num'],
        how='left'
    )
    n_after = len(df)
    assert n_after == n_before, f'Merge changed row count: {n_before} -> {n_after}'

    matched = df['dog_name'].notna().sum()
    print(f'Dog name match rate: {matched}/{n_before} ({100*matched/n_before:.1f}%)')

    # Drop rows with no dog name (can't compute track features for unknown dogs)
    df_with_dogs = df.loc[df['dog_name'].notna()].copy()
    print(f'Rows with dog names: {len(df_with_dogs):,}')
    print(f'Unique dogs: {df_with_dogs["dog_name"].nunique():,}')

    # Compute dog track record features
    print('Computing dog track record features...', flush=True)
    df_with_dogs = compute_dog_track_features(df_with_dogs)

    # List the new dog features
    dog_feature_cols = [c for c in df_with_dogs.columns if c.startswith('dog_')]
    print(f'\nDog features created ({len(dog_feature_cols)}):')
    for c in dog_feature_cols:
        non_null = df_with_dogs[c].notna().sum()
        print(f'  {c}: {non_null:,} non-null ({100*non_null/len(df_with_dogs):.1f}%)')

    # Summary: how many dogs have enough history for useful features?
    has_history = df_with_dogs['dog_n_races'] >= 5
    print(f'\nRunners with 5+ prior races: {has_history.sum():,} ({100*has_history.mean():.1f}%)')
    has_10 = df_with_dogs['dog_n_races'] >= 10
    print(f'Runners with 10+ prior races: {has_10.sum():,} ({100*has_10.mean():.1f}%)')

    # Save
    save_path = save_dir + f'features_with_dog_t{t_def}.parquet'
    df_with_dogs.to_parquet(save_path, index=False)
    print(f'\nSaved to {save_path}')
    print(f'Shape: {df_with_dogs.shape}')
    print('Done!', flush=True)
