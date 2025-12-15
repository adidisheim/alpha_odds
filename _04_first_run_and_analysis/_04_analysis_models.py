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
import enum
import matplotlib.pyplot as plt

class ExecutionType(enum.Enum):
    PURE_MARKET = 0
    PURE_LIMIT = 1
    START_LIMIT_END_MARKET = 2
    STAY_IN = 3



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


]

def transform_prb_to_odds(df):
    # in previous code we store implied probas (1/odds), so we need to invert them back
    for suffix in ['p1', 'm0']:
        df[f'best_back_{suffix}'] = 1/df[f'best_back_{suffix}']
        df[f'best_lay_{suffix}'] = 1/df[f'best_lay_{suffix}']
        df[f'best_back_q_100_{suffix}'] = 1/df[f'best_back_q_100_{suffix}']
        df[f'best_lay_q_100_{suffix}'] = 1/df[f'best_lay_q_100_{suffix}']
    return df
# i have tryied ton size bets up using kelly crieterio. I think this is impormant as we should bet largwer when there is more edge identified
def calculate_kelly_fraction(profit_rate, min_profit_threshold=0.0, max_kelly=1.0):
    """
    Calculate Kelly fraction from profit rate.

    Formula: Kelly = profit_rate / (1 + profit_rate) for positive edges
    This is a simplified version appropriate for small profit rates.

    Parameters:
    - profit_rate: Expected profit rate (fractional, e.g., 0.05 = 5% profit)
    - min_profit_threshold: Minimum profit rate to consider (default: 0.0)
    - max_kelly: Maximum Kelly fraction to cap at (default: 1.0, i.e., 100% of bankroll)

    Returns:
    - kelly_fraction: Position size as fraction of bankroll (clipped between 0 and max_kelly)
    """
    # Only calculate Kelly for positive expected profits
    kelly = np.where(
        profit_rate > min_profit_threshold,
        profit_rate / (1 + profit_rate),
        0.0
    )
    # Cap at maximum Kelly fraction
    return np.clip(kelly, 0.0, max_kelly)

def apply_kelly_sizing(df, use_kelly=True, kelly_fraction=0.25, min_profit_threshold=0.0, max_kelly=1.0):
    """
    Apply Kelly criterion sizing to profit calculations.

    Parameters:
    - df: DataFrame with profit_back_then_lay and profit_lay_then_back columns
    - use_kelly: If False, return binary sizing (current behavior)
    - kelly_fraction: Fraction of full Kelly to use (e.g., 0.25 = quarter Kelly)
    - min_profit_threshold: Minimum profit to consider for sizing
    - max_kelly: Maximum Kelly fraction cap

    Returns:
    - df: DataFrame with additional columns for sized profits
    """
    if not use_kelly:
        # Binary sizing: 1 if profitable, 0 otherwise
        df['position_size_bl'] = (df['profit_back_then_lay'] > 0).astype(float)
        df['position_size_lb'] = (df['profit_lay_then_back'] > 0).astype(float)
    else:
        # Calculate Kelly fractions
        df['kelly_bl'] = calculate_kelly_fraction(
            df['profit_back_then_lay'],
            min_profit_threshold,
            max_kelly
        )
        df['kelly_lb'] = calculate_kelly_fraction(
            df['profit_lay_then_back'],
            min_profit_threshold,
            max_kelly
        )

        # Apply fractional Kelly
        df['position_size_bl'] = df['kelly_bl'] * kelly_fraction
        df['position_size_lb'] = df['kelly_lb'] * kelly_fraction

    # Calculate sized profits (profit rate × position size)
    df['profit_back_then_lay_sized'] = df['profit_back_then_lay'] * df['position_size_bl']
    df['profit_lay_then_back_sized'] = df['profit_lay_then_back'] * df['position_size_lb']

    return df


'''
       profit_back_then_lay  profit_lay_then_back
count          81055.000000          81055.000000
mean              -0.056196             -0.243703
std                0.844983             11.271021
min               -0.998990           -989.099010
10%               -0.187500             -0.214286
25%               -0.107143             -0.111111
50%               -0.045455             -0.045113
75%                0.000000              0.000000
90%                0.059240              0.059701
max              237.095238              0.995800
'''


if __name__ == '__main__':
    args = parse()
    par = Params()
    shared_grid = [
        ['grid','start_ins_year',[2000]],
        ['grid','y_var', POSSIBLE_Y_VARIABLE],
        ['grid','topk_restriction', [1,2,3,4]],
        ['grid','t_definition', [0,1]]
    ]
    load_dir = Constant.RES_DIR+'model_to_download/'
    t_definition = 0
    qty_str = ''
    # qty_str = '_q_100'
    topk_restriction = 1  # Miles: fixed variable name typo
    execution_type = ExecutionType.START_LIMIT_END_MARKET
    # y_var = f'delta_avg_odds{qty_str}'
    # y_var = f'delta_back_then_lay_odds{qty_str}'
    # y_var = f'delta_lay_then_back_odds{qty_str}'
    # y_var = f'delta_start_limit_back_then_lay_odds{qty_str}'
    y_var = f'delta_start_limit_lay_then_back_odds{qty_str}'
    # y_var = 'win'
    save_name = f'tdef{t_definition}topK{topk_restriction}yvar{y_var}'
    df = pd.read_parquet(load_dir+ save_name+'_df.parquet')
    df_importance = pd.read_parquet(load_dir+ save_name+'_importances.parquet')

    df[['prediction',y_var]].corr()

    df = transform_prb_to_odds(df)

    df[[f"best_back_m0",'best_back_q_100_m0']]
    df[[f"best_lay_m0",'best_lay_q_100_m0']]

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

    elif execution_type == ExecutionType.STAY_IN:
        # you cross the spread on both legs
        back_entry = b0
        lay_exit = l1
        lay_entry = l0
        back_exit = b1

    else:
        raise ValueError(f"Unsupported execution type: {execution_type}")

    df["s_back_lay"] = back_entry / lay_exit
    df["t_lay_back"] = lay_entry / back_exit

    df["profit_back_then_lay"] = df["s_back_lay"] - 1
    df["profit_lay_then_back"] = 1 - df["t_lay_back"]

    if execution_type == ExecutionType.STAY_IN:
        df["profit_back_then_lay"] = (back_entry*(df['id']==-1)) - 1
        df["profit_lay_then_back"] = 1 - (lay_entry*(df['id']==-1))

    # Apply Kelly bet sizing
    use_kelly = True  # Parameter to enable/disable Kelly sizing
    kelly_fraction = 0.25  # kelly is super aggresive half or quater kelyl is common
    min_profit_threshold = 0.0  # Only for positive expected profit
    max_kelly = 1.0  # Cap at full Kelly

    df = apply_kelly_sizing(
        df,
        use_kelly=use_kelly,
        kelly_fraction=kelly_fraction,
        min_profit_threshold=min_profit_threshold,
        max_kelly=max_kelly
    )

    # Choose which profit columns to use for evaluation
    profit_col_bl = 'profit_back_then_lay_sized' if use_kelly else 'profit_back_then_lay'
    profit_col_lb = 'profit_lay_then_back_sized' if use_kelly else 'profit_lay_then_back'

    # add the market comission
    # for profit_col in ["profit_back_then_lay", "profit_lay_then_back"]:
    #     ind = df[profit_col] > 0
    #     if 'marketBas'
    #     df.loc[ind,profit_col] =df.loc[ind,profit_col] * (1-df.loc[ind,'marketBaseRate']/100)

    ind = df['profit_back_then_lay']>10
    df.loc[ind,:]
    df.loc[ind,["profit_back_then_lay", "profit_lay_then_back"]]
    desc = df[["profit_back_then_lay", "profit_lay_then_back"]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(desc)

    back_first_id = df['prediction'] > 0
    lay_first_id = df['prediction'] < 0
    back_first_profit = df.loc[back_first_id, profit_col_bl].mean()
    lay_first_profit = df.loc[lay_first_id, profit_col_lb].mean()

    res = pd.DataFrame()
    agg_func = 'mean'
    # agg_func = lambda x: x.mean() / x.std()
    # agg_func = lambda x: x.quantile(0.75)
    for q in np.arange(0.03, 0.98,0.01):
        tresh = df['prediction'].quantile(q)
        profit_bl = df.loc[df['prediction']<=tresh, profit_col_bl]
        profit_lb = df.loc[df['prediction']>=tresh, profit_col_lb]
        r = pd.Series({
            'quantile': q,
            'tresh': tresh,
            'profit_bl': profit_bl.aggregate(agg_func),

            'profit_lb': profit_lb.aggregate(agg_func),
            # 'profit': profit.mean()/profit.std()
        })
        res = pd.concat([res, r.to_frame().T], ignore_index=True)


    avg_back = df[profit_col_bl].aggregate(agg_func)
    avg_lay = df[profit_col_lb].aggregate(agg_func)

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




