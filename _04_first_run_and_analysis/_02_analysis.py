import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import psutil
import os
from parameters import Params, LassoModelParams, RandomForestModelParams, XGBoostModelParams
from utils_locals.parser import parse
import matplotlib.pyplot as plt
from parameters import Constant
import enum
class ExecutionType(enum.Enum):
    PURE_MARKET = 0
    PURE_LIMIT = 1
    START_LIMIT_END_MARKET = 2
    STAY_IN = 3

if __name__ == '__main__':
    args = parse()
    par = Params()
    args.b = 2
    execution_type = ExecutionType.STAY_IN
    shared_grid = [
        ['grid','start_ins_year',[2000,2021]]
    ]

    if args.b == 0: # lasso model
        print("Lasso model selected", flush=True)
        par.model = LassoModelParams()
        grid = [
            ['model','alpha',np.logspace(-12, 1, 6)],
        ]
        grid = shared_grid + grid
    elif args.b ==1: # random forest
        print("Random Forest model selected", flush=True)
        par.model = RandomForestModelParams()
        grid = [
            ['model','n_estimators',[50,100,1000]],
            ['model','max_depth',[5,10,None]],
        ]
        grid = shared_grid + grid
    elif args.b ==2: # xgboost
        print("XGBoost model selected", flush=True)
        par.model = XGBoostModelParams()
        grid = [
            ['model','n_estimators',[50,100,1000]],
            ['model','max_depth',[3,6,10]],
            ['model','learning_rate',[0.01,0.1,0.2]],
        ]
        grid = shared_grid + grid
    else:
        raise ValueError(f"Unsupported model type: {args.b}")

    res = pd.DataFrame()
    for i in range(54):
        par.update_param_grid(grid, i)
        df = pd.read_parquet(par.get_model_grid_dir()+'save_df.parquet')
        c = df.corr().loc['delta_avg_odds','prediction']
        print('index:', i, 'corr:', np.round(c,2), flush=True)
        r = pd.Series({
            'index': i,
            'corr': c,
            'start_year': par.grid.start_ins_year
        })
        res = pd.concat([res, r.to_frame().T], ignore_index=True)

    if args.b == 0:
        print("Lasso model selected", flush=True)
        i = 3
        par.update_param_grid(grid, i)
        df = pd.read_parquet(par.get_model_grid_dir() + 'save_df.parquet')
        lc = pd.read_parquet(par.get_model_grid_dir() + 'lasso_coefficients.parquet')
        lc['abs_coeff'] = lc['coefficient'].abs()
        lc = lc.sort_values(by='abs_coeff', ascending=False)
    elif args.b == 1:
        print("Random Forest model selected", flush=True)
        breakpoint()
        par.update_param_grid(grid, i)
        df = pd.read_parquet(par.get_model_grid_dir() + 'save_df.parquet')
    elif args.b == 2:
        i = 19
        print("XGBoost model selected", flush=True)
        par.update_param_grid(grid, i)
        df = pd.read_parquet(par.get_model_grid_dir() + 'save_df.parquet')
    else:
        raise ValueError(f"Unsupported model type: {args.b}")

    temp_translation = pd.read_parquet('temp_translation.parquet')
    temp_translation['prediction'] = df['prediction'].values
    temp_translation['delta_avg_odds'] = df['delta_avg_odds'].values
    df = temp_translation.copy()
    # in one line shuffle the prediction column
    # df['prediction'] = df['prediction'].sample(frac=1, random_state=42).values
    # df['prediction'] = df['delta_avg_odds'] # CHEATING HERE
    qty_str = '_q_100'
    qty_str = ''
    ind =  (df[f"best_back{qty_str}_m0"] -df[f"best_lay{qty_str}_m0"])< 0.025
    # df = df.loc[ind,:]

    # in previous code we store implied probas (1/odds), so we need to invert them back
    for suffix in ['p1', 'm0']:
        df[f'best_back_{suffix}'] = 1/df[f'best_back_{suffix}']
        df[f'best_lay_{suffix}'] = 1/df[f'best_lay_{suffix}']
        df[f'best_back_q_100_{suffix}'] = 1/df[f'best_back_q_100_{suffix}']
        df[f'best_lay_q_100_{suffix}'] = 1/df[f'best_lay_q_100_{suffix}']

    max_lay = 5
    ind = (df['best_lay_m0']<max_lay)
    print(ind.mean())
    df.loc[ind,['delta_avg_odds','prediction']].corr()
    df = df[ind].copy()

    # for suffix in ['p1', 'm0']:
    #     df[f'best_back_{suffix}'] = 1/df[f'best_back_{suffix}']
    #     df[f'best_lay_{suffix}'] = 1/df[f'best_lay_{suffix}']
    #     df[f'best_back_q_100_{suffix}'] = 1/df[f'best_back_q_100_{suffix}']
    #     df[f'best_lay_q_100_{suffix}'] = 1/df[f'best_lay_q_100_{suffix}']
    # df["profit_back_then_lay"] = (df[f"best_back{qty_str}_m0"]) - (df[f"best_lay{qty_str}_p1"])
    # df["profit_lay_then_back"] = (df[f"best_lay{qty_str}_m0"]) - (df[f"best_back{qty_str}_p1"])
    b0 = df[f"best_back{qty_str}_m0"]
    l0 = df[f"best_lay{qty_str}_m0"]
    b1 = df[f"best_back{qty_str}_p1"]
    l1 = df[f"best_lay{qty_str}_p1"]

    if execution_type == ExecutionType.PURE_MARKET:
        # you cross the spread on both legs
        back_entry = b0
        lay_exit = l1
        lay_entry = l0
        back_exit = b1

    elif execution_type == ExecutionType.PURE_LIMIT:
        # you *receive* the spread on both legs (pick a model you like)
        # e.g. assume you get hit at the opposite quote
        back_entry = l0  # improved back vs b0 (you get better odds to back)
        lay_exit = b1  # worse lay vs l1 (you give worse odds when you lay)
        lay_entry = b0  # better lay entry vs l0, etc.
        back_exit = l1

    elif execution_type == ExecutionType.START_LIMIT_END_MARKET:
        # entry as maker, exit as taker
        back_entry = l0
        lay_exit = l1
        lay_entry = b0
        back_exit = b1

    # else:
    #     raise ValueError(f"Unsupported execution type: {execution_type}")
    #
    # df["s_back_lay"] = back_entry / lay_exit
    # df["t_lay_back"] = lay_entry / back_exit
    #
    # df["profit_back_then_lay"] = df["s_back_lay"] - 1
    # df["profit_lay_then_back"] = 1 - df["t_lay_back"]

    if execution_type == ExecutionType.STAY_IN:
        df["profit_back_then_lay"] = ((l0)*(df['id']==-1)) -1
        df["profit_lay_then_back"] = 1- (b0*(df['id']==-1))


    for profit_col in ["profit_back_then_lay", "profit_lay_then_back"]:
        ind = df[profit_col] > 0
        df.loc[ind,profit_col] =df.loc[ind,profit_col] * (1-df.loc[ind,'marketBaseRate']/100)


    df[["profit_back_then_lay", "profit_lay_then_back"]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

    back_first_id = df['prediction'] > 0
    lay_first_id = df['prediction'] < 0
    back_first_profit = df.loc[back_first_id, "profit_back_then_lay"].mean()
    lay_first_profit = df.loc[lay_first_id, "profit_lay_then_back"].mean()

    res = pd.DataFrame()

    agg_func = 'mean'
    # agg_func = lambda x: x.mean() / x.std()
    # agg_func = lambda x: x.quantile(0.75)
    for q in np.arange(0.03, 0.98,0.01):
        tresh = df['prediction'].quantile(q)
        profit_bl = df.loc[df['prediction']<=tresh, "profit_back_then_lay"]
        profit_lb = df.loc[df['prediction']>=tresh, "profit_lay_then_back"]
        r = pd.Series({
            'quantile': q,
            'tresh': tresh,
            'profit_bl': profit_bl.aggregate(agg_func),

            'profit_lb': profit_lb.aggregate(agg_func),
            # 'profit': profit.mean()/profit.std()
        })
        res = pd.concat([res, r.to_frame().T], ignore_index=True)


    avg_back = df["profit_back_then_lay"].aggregate(agg_func)
    avg_lay = df["profit_lay_then_back"].aggregate(agg_func)

    plt.figure(figsize=(8, 5))
    plt.plot(res["quantile"], res["profit_bl"], color="#55A868",linestyle ='--', label="Strategy profit Back→Lay")
    plt.plot(res["quantile"], res["profit_lb"], color="red",linestyle ='--', label="Strategy profit Lay→Back")
    plt.axhline(avg_back, color="#55A868", linewidth=1.5, label="Avg back→lay")
    plt.axhline(avg_lay, color="red", linewidth=1.5, label="Avg lay→back")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Quantile")
    plt.ylabel("Profit")
    plt.title(execution_type.name)
    plt.legend()
    plt.tight_layout()
    plt.show()

    df['unique_bet'] = df['file_name'] + '_' + df['id'].astype(str)
    # df['']
    import didipack
    df

