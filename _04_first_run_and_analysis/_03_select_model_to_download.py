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




POSSIBLE_Y_VARIABLE = [
    "delta_avg_odds",
    "delta_back_then_lay_odds",
    "delta_lay_then_back_odds",
    "delta_start_limit_back_then_lay_odds",
    "delta_start_limit_lay_then_back_odds",

    "delta_avg_odds_q_100",
    "delta_back_then_lay_odds_q_100",
    "delta_lay_then_back_odds_q_100",
    "delta_start_limit_back_then_lay_odds_q_100",
    "delta_start_limit_lay_then_back_odds_q_100",

    "win"

    # "delta_avg_odds_q_1000",
    # "delta_back_then_lay_odds_q_1000",
    # "delta_lay_then_back_odds_q_1000",
    # "delta_start_limit_back_then_lay_odds_q_1000",
    # "delta_start_limit_lay_then_back_odds_q_1000"
]


if __name__ == '__main__':
    args = parse()
    par = Params()
    shared_grid = [
        ['grid','start_ins_year',[2000]],
        ['grid','y_var', POSSIBLE_Y_VARIABLE],
        ['grid','topk_restriction', [1,2,3,4]],
        ['grid','t_definition', [0,1]]
    ]
    save_dir = Constant.RES_DIR+'model_to_download/'
    os.makedirs(save_dir, exist_ok=True)

    for t_definition in [0,1]:
        df_to_merge = pd.read_parquet(Constant.RES_DIR+f'oos_df_t{t_definition}.parquet')
        for topk_restrcition in [1,2,3,4]:
            for y_var in POSSIBLE_Y_VARIABLE:
                par.model = XGBoostModelParams()

                grid = [
                    ['model', 'n_estimators', [50, 100, 1000]],
                    ['model', 'max_depth', [3, 6, 10]],
                    ['model', 'learning_rate', [0.01, 0.1, 0.2]],
                ]
                res = {}
                for i in range(27):
                    par.update_param_grid(grid, i)
                    par.grid.t_definition = t_definition
                    par.grid.topk_restriction = topk_restrcition
                    par.grid.y_var = y_var
                    df = pd.read_parquet(par.get_model_grid_dir(old_style=True) + 'save_df.parquet')
                    if par.grid.y_var in ['win']:
                        p = df['prediction'].clip(1e-8, 1 - 1e-8)
                        y = df[par.grid.y_var]
                        entropy = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
                        res[i] = entropy
                    else:
                        mse = ((df['prediction'] - df[par.grid.y_var]) ** 2).mean()
                        res[i] = mse
                i_best = pd.Series(res).idxmin()

                par.update_param_grid(grid, i_best)
                par.grid.t_definition = t_definition
                par.grid.topk_restriction = topk_restrcition
                par.grid.y_var = y_var
                df = pd.read_parquet(par.get_model_grid_dir(old_style=True) + 'save_df.parquet')
                df = df.merge(df_to_merge, on=['file_name','id'], how='left')
                corr = df[['prediction',par.grid.y_var]].corr()
                print('For, t_definition = ', t_definition, 'topk_restriction = ', topk_restrcition, 'y_var = ', y_var)
                print('Correlation = ', corr.iloc[0,1],flush=True)
                df_importance = pd.read_parquet(par.get_model_grid_dir(old_style=True) + 'xgboost_feature_importances.parquet')
                save_name = f'tdef{t_definition}topK{topk_restrcition}yvar{y_var}'
                df.to_parquet(save_dir+ save_name+'_df.parquet')
                df_importance.to_parquet(save_dir+ save_name+'_importances.parquet')


