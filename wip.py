import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import psutil
import os
from parameters import Params, LassoModelParams, RandomForestModelParams, XGBoostModelParams
from utils_locals.parser import parse

import socket
from parameters import Constant


import didipack

if __name__ == '__main__':

    # df = pd.read_pickle('/data/projects/punim2039/refinitiv_processed/en/news_link_ticker/news_2023.p')
    df = pd.read_parquet('res/model_to_download/tdef0topK1yvardelta_avg_odds_df.parquet')
    df_top_4 = pd.read_parquet('res/model_to_download/tdef0topK4yvardelta_avg_odds_df.parquet')

    df_top_4




