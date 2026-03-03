"""
Extract "off" timestamps from raw BZ2 Betfair streaming files.

For greyhound markets, there is no in-play trading. Markets go OPEN -> SUSPENDED -> CLOSED.
The SUSPENDED transition is when the race starts (the "off"). We use this as the reference
point for time_delta in feature engineering (replacing the old max(time) approach).

We keep the column named 'in_play_time' for compatibility with downstream code (feature
engineering, live trading), but the actual event captured is the first SUSPENDED transition.

Usage:
    python _04_extract_in_play_times.py <grid_id> <run_on_spartan=1>
    grid_id: 0-323 (same grid as _02_process_all_files_para.py)
"""

import bz2
import json
import numpy as np
import pandas as pd
import os
from utils_locals.parser import parse
from parameters import Constant

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]
YEARS = range(2017, 2026)
DAY_PARA_NUMBER = [1, 2, 3]


def extract_off_time(file_path):
    """
    Parse a BZ2 file and find the timestamp of the first SUSPENDED transition.
    For greyhound markets this is when the race starts (the "off").
    Returns (file_name, off_time) or (file_name, NaT) if never suspended.
    """
    try:
        with bz2.open(file_path, mode='rt') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Error reading {file_path}: {e}", flush=True)
        return None

    file_name = os.path.basename(file_path)
    prev_status = None

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        pt = record.get('pt', None)
        mc_list = record.get('mc', [])
        if not mc_list:
            continue

        mc = mc_list[0]
        if 'marketDefinition' in mc:
            mdef = mc['marketDefinition']
            status = mdef.get('status', prev_status)

            # First SUSPENDED transition = race start ("the off")
            if status == 'SUSPENDED' and prev_status != 'SUSPENDED':
                off_time = pd.to_datetime(pt, unit='ms', utc=True)
                return (file_name, off_time)

            prev_status = status

    # Never went SUSPENDED (cancelled/void market)
    return (file_name, pd.NaT)


if __name__ == '__main__':
    args = parse()

    grid = [(m, y, d) for y in YEARS for m in MONTHS for d in DAY_PARA_NUMBER]
    print(f'Nb of jobs in the grid: {len(grid)}', flush=True)

    run_locally = args.b == 0
    if run_locally:
        grid = [('Oct', 2025, 1), ('Nov', 2025, 1)]

    month = grid[args.a][0]
    year = grid[args.a][1]
    day_para_number = grid[args.a][2]

    if run_locally:
        start_path = '../data/raw/PRO/'
    else:
        start_path = '/data/projects/punim2039/alpha_odds/untar/greyhound_au/PRO/'

    dest_path = Constant.DATA_DIR + 'p/greyhound_au/'
    os.makedirs(dest_path, exist_ok=True)

    year_month_path = start_path + str(year) + '/' + month
    if not os.path.exists(year_month_path):
        print(f'Path does not exist: {year_month_path}, skipping.', flush=True)
        exit(0)

    day_list = np.sort([x for x in os.listdir(year_month_path) if '.' not in x])
    if day_para_number == 1:
        day_list = [x for x in day_list if int(x) <= 10]
    elif day_para_number == 2:
        day_list = [x for x in day_list if (int(x) > 10) & (int(x) <= 20)]
    elif day_para_number == 3:
        day_list = [x for x in day_list if int(x) > 20]

    print(f'Processing {year}/{month}, day_para={day_para_number}, days: {day_list}', flush=True)

    for day in day_list:
        out_file = dest_path + f'inplay_{year}_{month}_{day}.parquet'
        if os.path.exists(out_file):
            print(f'Already exists: {out_file}, skipping.', flush=True)
            continue

        results = []
        day_path = year_month_path + '/' + day
        event_list = np.sort([x for x in os.listdir(day_path) if '.' not in x])

        for event in event_list:
            event_path = day_path + '/' + event
            market_files = np.sort([x for x in os.listdir(event_path) if 'bz2' in x])
            for market in market_files:
                file_path = event_path + '/' + market
                result = extract_off_time(file_path)
                if result is not None:
                    results.append(result)

        if results:
            # Column named 'in_play_time' for downstream compatibility
            df = pd.DataFrame(results, columns=['file_name', 'in_play_time'])
            df.to_parquet(out_file)
            n_valid = df['in_play_time'].notna().sum()
            print(f'Saved {out_file}: {len(df)} markets, {n_valid} with off time ({n_valid/len(df)*100:.1f}%)', flush=True)
        else:
            print(f'No markets found for {year}/{month}/{day}', flush=True)

    print('All done!', flush=True)
