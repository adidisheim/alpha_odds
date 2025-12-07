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

YEARS = range(2017, 2026)
DAY_PARA_NUMBER = [1,2,3]

if __name__ == '__main__':
    args = parse()

    grid = [(m, y, d) for y in YEARS for m in MONTHS for d in DAY_PARA_NUMBER]
    print('Nb of jobs in the grid:', len(grid))
    run_locally = args.b ==0
    if run_locally: # for debugging
        grid = [('Oct', 2025,1), ('Nov', 2025,1)]
    month = grid[args.a][0]
    year = grid[args.a][1]
    day_para_number = grid[args.a][2]
    MIN_NB_ROWS = 25
    if run_locally:
        start_path = '../data/raw/PRO/'
    else:
        start_path = '/data/projects/punim2039/alpha_odds/untar/greyhound_au/PRO/'

    dest_path = Constant.DATA_DIR+'p/greyhound_au/'
    os.makedirs(dest_path, exist_ok=True)

    # list_of_valid_market_type = ['WIN', 'PLACE'] # for now only do WIN
    list_of_valid_market_type = ['WIN']

    day_list = np.sort([x for x in os.listdir(start_path+str(year)+'/'+month) if '.' not in x])
    if day_para_number == 1:
        day_list = [x for x in day_list if int(x)<=10]
    if day_para_number == 2:
        day_list = [x for x in day_list if (int(x)>10) & (int(x)<=20)]
    if day_para_number == 3:
        day_list = [x for x in day_list if int(x)>20]
    print(f'After appltying day_para_number filter ({day_para_number}), processing days:', day_list, flush=True)

    for day in day_list:
        if not os.path.exists(dest_path + f'mdef_{year}_{month}_{day}.parquet'):
            df_win = pd.DataFrame()
            df_place = pd.DataFrame()
            df_mdef = pd.DataFrame()
            event_list = np.sort([x for x in os.listdir(start_path+str(year)+'/'+month+'/'+day) if '.' not in x])
            for event in tqdm.tqdm(event_list, desc=f'Processing {year}/{month}/{day}'):
                market_lists = np.sort([x for x in os.listdir(start_path+str(year)+'/'+month+'/'+day+'/'+event) if 'bz2' in x])
                for market in market_lists:
                    file_path = start_path+str(year)+'/'+month+'/'+day+'/'+event+'/'+market
                    df_full = load_bz2_json(file_path)
                    # process time
                    df_full["pt"] = pd.to_datetime(df_full["pt"], unit="ms")
                    ### extract all mc from non-market-definition entries
                    runners, first_mdef = process_market_file(df_full)
                    runners["name_num"] = runners["name"].str.extract(r"^(\d+)\.")
                    if (first_mdef['marketType'] in list_of_valid_market_type) &(df_full.shape[0]>MIN_NB_ROWS):
                        first_mdef['file_name'] = market
                        df_mdef = pd.concat([df_mdef, first_mdef], ignore_index=False, axis=1)
                        df_all_bl = create_df_with_all_atl_and_atb(df_full)
                        df = pd.DataFrame()
                        for runner_id in runners['id'].tolist():
                            temp = process_runner_order_book(df_all_bl, runners, runner_id, q_low=0, q_grid=[100, 200, 1000])
                            temp['runner_position'] = runners.loc[runners['id'] ==runner_id, 'name_num'].iloc[0]
                            df = pd.concat([df, temp], ignore_index=False)
                        df['file_name']= market
                        if first_mdef['marketType'] == 'WIN':
                            df_win = pd.concat([df_win, df], ignore_index=False)
                        if first_mdef['marketType'] == 'PLACE':
                            df_place = pd.concat([df_place, df], ignore_index=False)

            df_win.columns = [str(x) for x in df_win.columns]
            df_place.columns = [str(x) for x in df_place.columns]
            df_mdef.columns = [str(x) for x in df_mdef.columns]
            if df_win.shape[0]>0:
                df_win.to_parquet(dest_path+f'win_{year}_{month}_{day}.parquet')
                print('Saved to:', dest_path+f'win_{year}_{month}_{day}.parquet',flush=True)
            else:
                print('Win df is empty, not saved', flush=True)
            if df_place.shape[0]>0:
                df_place.to_parquet(dest_path+f'place_{year}_{month}_{day}.parquet')
                print('Saved to:', dest_path+f'place_{year}_{month}_{day}.parquet',flush=True)
            else:
                print('Place df is empty, not saved', flush=True)
            if df_mdef.shape[0]>0:
                df_mdef.T.reset_index(drop=True).to_parquet(dest_path + f'mdef_{year}_{month}_{day}.parquet')
                print('Saved to:', dest_path+f'mdef_{year}_{month}_{day}.parquet',flush=True)
            else:
                print('Mdef df is empty, not saved', flush=True)
    print('All done!', flush=True)