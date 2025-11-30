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

YEARS = range(2017, 2024)

if __name__ == '__main__':
    args = parse()

    grid = [(m, y) for y in YEARS for m in MONTHS]
    dest_path = Constant.DATA_DIR+'p/greyhound_au/'
    os.makedirs(dest_path, exist_ok=True)

    list_of_valid_market_type = ['WIN', 'PLACE']

    # merge all the win file
    for type_files in ['win_', 'place_']:
        df_total = pd.DataFrame()
        win_files = [x for x in os.listdir(dest_path) if type_files in x]
        year = '2023'
        win_files_in_year = [x for x in win_files if f'_{year}_' in x]
        for file in tqdm.tqdm(win_files_in_year[:10]):
            print(file)
            temp = pd.read_parquet(dest_path+file)
            df_total = pd.concat([df_total, temp], ignore_index=False)
            print('File size in gb, after merging:', df_total.memory_usage(deep=True).sum()/1e9)
        df_total.to_parquet(Constant.RES_DIR+f'{type_files}greyhound_merged.parquet')


