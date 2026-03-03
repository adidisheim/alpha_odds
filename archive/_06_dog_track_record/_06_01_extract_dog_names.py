"""
Extract dog names from mdef parquet files.
Creates a flat mapping: (file_name, position_num) -> dog_name + metadata.

The mdef parquets already contain runner info (id, name, status) in a 'runners'
column (numpy array of dicts). We just need to explode and parse.

Usage: python _06_01_extract_dog_names.py
No arguments needed — reads all mdef_*.parquet files.
"""

import pandas as pd
import numpy as np
import os
import tqdm
from parameters import Constant


if __name__ == '__main__':
    data_dir = Constant.DATA_DIR + 'p/greyhound_au/'
    save_dir = Constant.RES_DIR + 'dog_features/'
    os.makedirs(save_dir, exist_ok=True)

    mdef_files = sorted([f for f in os.listdir(data_dir) if f.startswith('mdef_')])
    print(f'Found {len(mdef_files)} mdef files')

    all_records = []
    n_failed = 0

    for f in tqdm.tqdm(mdef_files, desc='Extracting dog names'):
        try:
            mdef = pd.read_parquet(data_dir + f)
        except Exception as e:
            print(f'WARNING: Could not read {f}: {e}')
            n_failed += 1
            continue

        for _, row in mdef.iterrows():
            file_name = row.get('file_name', '')
            venue = row.get('venue', '')
            market_time = row.get('marketTime', '')
            runners = row.get('runners', None)

            if runners is None:
                continue

            for runner in runners:
                name = runner.get('name', '')
                # Parse "{position}. {dog_name}" format
                parts = name.split('. ', 1)
                if len(parts) == 2:
                    try:
                        position_num = int(parts[0])
                    except ValueError:
                        position_num = -1
                    dog_name = parts[1].strip()
                else:
                    position_num = -1
                    dog_name = name.strip()

                all_records.append({
                    'file_name': file_name,
                    'selection_id': runner.get('id', -1),
                    'position_num': position_num,
                    'dog_name': dog_name,
                    'venue': venue,
                    'market_time': market_time,
                })

    df = pd.DataFrame(all_records)

    # Parse market_time to datetime for sorting
    df['market_time_dt'] = pd.to_datetime(df['market_time'], errors='coerce', utc=True)

    # Summary stats
    print(f'\n=== DOG NAME EXTRACTION SUMMARY ===')
    print(f'Total records: {len(df):,}')
    print(f'Unique markets (file_name): {df["file_name"].nunique():,}')
    print(f'Unique dogs: {df["dog_name"].nunique():,}')
    print(f'Unique venues: {df["venue"].nunique():,}')
    print(f'Date range: {df["market_time_dt"].min()} to {df["market_time_dt"].max()}')
    print(f'Failed files: {n_failed}')

    # Dog race count distribution
    races_per_dog = df.groupby('dog_name')['file_name'].nunique()
    print(f'\nRaces per dog distribution:')
    print(f'  Mean: {races_per_dog.mean():.1f}')
    print(f'  Median: {races_per_dog.median():.0f}')
    print(f'  Dogs with 1 race: {(races_per_dog == 1).sum():,}')
    print(f'  Dogs with 5+ races: {(races_per_dog >= 5).sum():,}')
    print(f'  Dogs with 10+ races: {(races_per_dog >= 10).sum():,}')
    print(f'  Dogs with 20+ races: {(races_per_dog >= 20).sum():,}')
    print(f'  Max races: {races_per_dog.max()}')

    save_path = save_dir + 'dog_names_mapping.parquet'
    df.to_parquet(save_path, index=False)
    print(f'\nSaved to {save_path}')
    print('Done!', flush=True)
