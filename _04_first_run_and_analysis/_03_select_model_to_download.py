import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
import psutil
import os
from parameters import Params, LassoModelParams, RandomForestModelParams, XGBoostModelParams
from utils_locals.parser import parse

import socket
from parameters import Constant, POSSIBLE_Y_VARIABLE, SHARED_GRID, SpreadTopKCriterion
from xgboost import XGBRegressor, XGBRFClassifier



if __name__ == '__main__':
    args = parse()
    par = Params()
    save_dir = Constant.RES_DIR+'model_to_download/'
    os.makedirs(save_dir, exist_ok=True)

    # for t_definition in [0,1]:
    for spread_restriction in [-1, 0.1, 0.05]:
        for spread_to_k_criterion in [SpreadTopKCriterion.VANILLA, SpreadTopKCriterion.Q100]:
            for t_definition in [1, 2, 3]:
                df_to_merge = pd.read_parquet(Constant.RES_DIR+f'oos_df_t{t_definition}.parquet')
                for topk_restriction in [1,2,3]:  # Miles: fixed variable name typo
                    for y_var in POSSIBLE_Y_VARIABLE:
                        par.model = XGBoostModelParams()

                        grid = [
                            ['model', 'n_estimators', [50, 100, 1000]],
                            ['model', 'max_depth', [3, 6, 10]],
                            ['model', 'learning_rate', [0.01, 0.1, 0.2]],
                        ]
                        res = {}
                        for i in range(27):
                            par.update_param_grid(grid, i,verbose=False)
                            par.grid.t_definition = t_definition
                            par.grid.topk_restriction = topk_restriction
                            par.grid.y_var = y_var
                            par.grid.spread_restriction = spread_restriction
                            par.grid.spread_top_k_criterion = spread_to_k_criterion
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
                        par.grid.topk_restriction = topk_restriction
                        par.grid.y_var = y_var

                        # save the model itself for Andrew.
                        model_path = par.get_model_grid_dir(old_style=True) + 'xgboost_model.json'
                        save_name = f'tdef{t_definition}topK{topk_restriction}yvar{y_var}'
                        out_path = save_dir + save_name + '_model.json'

                        if par.grid.y_var in ['win']:
                            m = XGBRFClassifier()
                        else:
                            m = XGBRegressor()

                        m.load_model(model_path)
                        m.save_model(out_path)

                        df = pd.read_parquet(par.get_model_grid_dir(old_style=True) + 'save_df.parquet')
                        df = df.merge(df_to_merge, on=['file_name','id'], how='left')

                        temp = df.dropna(subset=['prediction',par.grid.y_var])
                        y = temp[par.grid.y_var].to_numpy()
                        yhat = temp['prediction'].to_numpy()

                        # Normal R^2
                        sse = np.mean((y - yhat) ** 2)
                        var_y = np.var(y)
                        r2 = 1.0 - sse / var_y if var_y > 0 else np.nan

                        # Demeaned R^2 (demean both series, then compute R^2 on demeaned)
                        y_dm = y - np.mean(y)
                        yhat_dm = yhat - np.mean(yhat)
                        sse_dm = np.mean((y_dm - yhat_dm) ** 2)
                        var_y_dm = np.var(y_dm)
                        r2_dm = 1.0 - sse_dm / var_y_dm if var_y_dm > 0 else np.nan

                        print(
                            f"For "
                            f"t_definition={t_definition}, "
                            f"topK={topk_restriction}, "
                            f"y_var={y_var}, "
                            f"spread={spread_restriction}, "
                            f"spread_criterion={spread_to_k_criterion.name}"
                        )
                        print(f"R2={r2}, Demeaned R2={r2_dm}", flush=True)
                        df_importance = pd.read_parquet(par.get_model_grid_dir(old_style=True) + 'xgboost_feature_importances.parquet')
                        # save_name = f'tdef{t_definition}topK{topk_restriction}yvar{y_var}'
                        save_name = (
                            f"tdef{t_definition}"
                            f"_topK{topk_restriction}"
                            f"_yvar{y_var}"
                            f"_spread{spread_restriction}"
                            f"_spreadcrit{spread_to_k_criterion.name}"
                        )
                        df.to_parquet(save_dir+ save_name+'_df.parquet')
                        df_importance.to_parquet(save_dir+ save_name+'_importances.parquet')


