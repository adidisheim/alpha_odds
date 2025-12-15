import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
import psutil
import os
from parameters import Params, LassoModelParams, RandomForestModelParams, XGBoostModelParams
from utils_locals.parser import parse

import socket
from parameters import Constant




if __name__ == '__main__':

    # for t_definition in [0,1]:
    for t_definition in [2,3]:
        load_dir = f'{Constant.RES_DIR}/features_t{t_definition}'
        df = pd.DataFrame()
        to_load = range(10)
        print('start loading from', load_dir, flush=True)
        print('Range', to_load)
        for i in to_load:
            try:
                df = pd.concat([df, pd.read_parquet(f'{load_dir}/greyhound_au_features_part_{i}.parquet')], ignore_index=False)
                print('load from', f'{load_dir}/greyhound_au_features_part_{i}.parquet')
            except:
                print('Non-fatal ERROR: Could not read part ', i, flush=True)

        process = psutil.Process(os.getpid())
        ram_gb = process.memory_info().rss / 1024 ** 3
        print(f"Python process RAM usage: {ram_gb:.2f} GB")

        df['total_qty_m1'] = df[['total_back_qty_m1', 'total_lay_qty_m1']].sum(axis=1)
        df['total_qty_m3'] = df[['total_back_qty_m3', 'total_lay_qty_m3']].sum(axis=1)
        # detect the top3 runners by total liquidity at m1
        topX = 5 #  put it t o the max to have a df that saves with everything
        df = df.reset_index()
        ind = df.groupby('file_name')['total_qty_m1'].rank(method='min', ascending=False) <= topX
        fixed_effect_columns = ['local_dow', 'marketBaseRate', 'numberOfActiveRunners']
        df = df[ind].copy()


        ind_oos = df['marketTime_local'].dt.year == 2025

        suffix_enter = "_m0"
        suffix_leave = "_p1"
        col_save = ['best_back' + suffix_enter, 'best_lay' + suffix_enter] + ['best_back_q_100' + suffix_enter, 'best_lay_q_100' + suffix_enter]
        col_save += ['best_back' + suffix_leave, 'best_lay' + suffix_leave] + ['best_back_q_100' + suffix_leave, 'best_lay_q_100' + suffix_leave]
        id_cols = ['file_name','id']

        # we do it with the merge to get back the un-normalized odds.
        df_save = df.loc[ind_oos,id_cols+fixed_effect_columns+col_save].copy()
        df_save.to_parquet(Constant.RES_DIR+f'oos_df_t{t_definition}.parquet')
        print('Saved to', Constant.RES_DIR+f'oos_df_t{t_definition}.parquet')

