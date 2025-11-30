import bz2
import json
import pathlib
from utils_locals.process_races import *
import pandas as pd
import tqdm
import os
from utils_locals.parser import parse
from parameters import Constant
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
]

YEARS = range(2017, 2024)

if __name__ == '__main__':
    args = parse()

    grid = [(m, y) for y in YEARS for m in MONTHS]
    print('Nb of jobs in the grid:', len(grid))
    run_locally = args.b ==0
    if run_locally: # for debugging
        grid = [('Oct', 2025), ('Nov', 2025)]

    month = grid[args.a][0]
    year = grid[args.a][1]


    MIN_NB_ROWS = 25
    if run_locally:
        start_path = 'data/raw/PRO/'
    else:
        start_path = '/data/projects/punim2039/alpha_odds/greyhound_au/PRO/'

    dest_path = Constant.DATA_DIR+'p/greyhound_au/'
    os.makedirs(dest_path, exist_ok=True)

    list_of_valid_market_type = ['WIN', 'PLACE']




    day_list = np.sort([x for x in os.listdir(start_path+str(year)+'/'+month) if '.' not in x])
    day_list = ['13']
    month = 'Jan'
    year = '2017'
    for day in day_list:
        df_win = pd.DataFrame()
        df_place = pd.DataFrame()
        df_mdef = pd.DataFrame()
        print("List of days:", day_list)
        event_list = np.sort([x for x in os.listdir(start_path+str(year)+'/'+month+'/'+day) if '.' not in x])
        event_list = [event_list[1]]
        for event in tqdm.tqdm(event_list, desc=f'Processing {year}/{month}/{day}'):
            market_lists = np.sort([x for x in os.listdir(start_path+str(year)+'/'+month+'/'+day+'/'+event) if 'bz2' in x])
            for market in market_lists:
                file_path = start_path+str(year)+'/'+month+'/'+day+'/'+event+'/'+market
                df_full = load_bz2_json(file_path)
                # process time
                df_full["pt"] = pd.to_datetime(df_full["pt"], unit="ms")
                ### extract all mc from non-market-definition entries
                runners, first_mdef = process_market_file(df_full)
                if (first_mdef['marketType'] in list_of_valid_market_type) &(df_full.shape[0]>MIN_NB_ROWS):
                    df_all_bl = create_df_with_all_atl_and_atb(df_full)
                    df = pd.DataFrame()
                    for runner_id in runners['id'].tolist():
                        temp = process_runner_order_book(df_all_bl, runners, runner_id, q_low=0, q_grid=[100, 1000])
                        df = pd.concat([df, temp], ignore_index=False)
                        df['file_name']= market
                        first_mdef['file_name']= market
                        if first_mdef['marketType'] == 'WIN':
                            df_win = pd.concat([df_win, df], ignore_index=False)
                        if first_mdef['marketType'] == 'PLACE':
                            df_place = pd.concat([df_place, df], ignore_index=False)
                        df_mdef = pd.concat([df_mdef, first_mdef], ignore_index=False,axis=1)
            print(df_win.memory_usage(deep=True).sum() / 1024**3)
            print(temp.memory_usage(deep=True).sum() / 1024**3)
    # df_win.to_parquet(dest_path+f'win_{year}_{month}.parquet')
    # df_place.to_parquet(dest_path+f'place_{year}_{month}.parquet')
    # df_mdef.to_parquet(dest_path+f'mdef_{year}_{month}.parquet')
    print('Saved to:', dest_path+f'win_{year}_{month}.parquet')
    print('Saved to:', dest_path+f'place_{year}_{month}.parquet')
    print('Saved to:', dest_path+f'mdef_{year}_{month}.parquet')
