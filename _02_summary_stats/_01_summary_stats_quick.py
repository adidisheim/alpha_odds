import bz2
import json
import pathlib
from utils_locals.process_races import *
import pandas as pd
import tqdm
import os
from utils_locals.parser import parse
from parameters import Constant
import pyarrow.dataset as ds
import pandas as pd

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

YEARS = range(2017, 2025)

if __name__ == '__main__':
    args = parse()
    year = YEARS[args.a]
    dest_path = Constant.DATA_DIR+'p/greyhound_au/'
    os.makedirs(dest_path, exist_ok=True)

    list_of_valid_market_type = ['WIN', 'PLACE']

    # merge all the win file
    k = 0
    for type_files in ['win_']:
        df_total = pd.DataFrame()
        win_files = [x for x in os.listdir(dest_path) if type_files in x]
        # year = '2023'
        win_files_in_year = [x for x in win_files if f'_{year}_' in x]
        for file in tqdm.tqdm(win_files_in_year):
            df = pd.read_parquet(dest_path+file)
            df = df.reset_index()
            df['time_delta'] = df.groupby('file_name')['time'].transform('max') - df['time']
            df['time_delta_sec'] = df['time_delta'].dt.total_seconds().round()

            df['time_delta_5s'] = (df['time_delta_sec'] + 4) // 5 * 5



            df['mid_price'] = (df['best_back'] + df['best_lay']) / 2
            df['mid_prb_win'] = 1 / df['mid_price']

            # finding the top k dogs per volume
            temp = df.groupby(['file_name', 'id'])['qty'].sum()
            temp = temp.reset_index().sort_values(['file_name', 'qty'], ascending=[True, False])
            temp["vol_rank"] = temp.groupby("file_name").cumcount() + 1
            temp = temp.drop(columns=['qty'])
            df = df.merge(temp, on=['file_name', 'id'], how='left')

            ind = (df['time_delta_sec'] <= 60*5)
            df['won'] = df['id'] == -1
            temp = df.loc[ind, :].groupby(['file_name', 'id','time_delta_sec'])[['won', 'mid_prb_win','vol_rank']].mean()
            temp2 = df.loc[ind, :].groupby(['file_name', 'id','time_delta_sec'])[['qty']].sum()
            temp = temp.merge(temp2,left_index=True, right_index=True,how='outer')
            temp['date'] = pd.to_datetime(df["time"].iloc[0:1]).dt.date.iloc[0]
            df_total = pd.concat([df_total, temp], ignore_index=False)
            print('File size in gb, after merging:', df_total.memory_usage(deep=True).sum()/1e9)
            k+=1
            if k % 30==0:
                print('Saving intermediate file...')
                df_total.to_parquet(Constant.RES_DIR+f'{year}_ss.parquet')
        df_total.to_parquet(Constant.RES_DIR+f'{year}_ss.parquet')



