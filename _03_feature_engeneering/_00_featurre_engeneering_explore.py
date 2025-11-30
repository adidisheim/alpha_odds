import bz2
import json
import pathlib

import numpy as np

from utils_locals.process_races import *
import pandas as pd
import tqdm
import os
from utils_locals.parser import parse
from parameters import Constant
import pyarrow.dataset as ds
import pandas as pd
import didipack
from utils_locals.feature_tools import *


YEARS = range(2017, 2025)

if __name__ == '__main__':
    args = parse()
    year = YEARS[args.a]
    dest_path = Constant.DATA_DIR+'p/greyhound_au/'
    os.makedirs(dest_path, exist_ok=True)

    df = pd.read_parquet('data/p/greyhound_au/win_2021_Sep_22.parquet')
    mdef = pd.read_parquet('data/p/greyhound_au/mdef_2021_Sep_22.parquet')
    id_cols = ['file_name', 'id']
    df['runner_position'] = pd.to_numeric(df['runner_position'])
    runner_position = df.groupby(id_cols)['runner_position'].count()

    # mdef = mdef.loc[mdef['marketType'] == 'WIN', :].copy()
    mdef["marketTime_local"] = mdef.apply(lambda r: pd.to_datetime(r.marketTime, utc=True).tz_convert(r.timezone), axis=1)
    mdef["marketTime_local"] = pd.to_datetime(mdef["marketTime_local"].apply(lambda x: str(x).split(' ')[0]), errors='coerce')
    mdef['local_dow'] = mdef['marketTime_local'].dt.dayofweek

    # ffe columns
    fixed_effect_columns = ['file_name','local_dow','marketBaseRate','marketType','numberOfActiveRunners','venue']

    df[['qty','prc']] =df[['qty','prc']].fillna(0.0)
    df = df.reset_index()
    df['time_delta'] = df.groupby('file_name')['time'].transform('max') - df['time']

    # define the exact time values for the snapshot
    tm3 = pd.Timedelta("0 days 00:15:00")
    tm2 = pd.Timedelta("0 days 00:05:00")
    tm1 = pd.Timedelta("0 days 00:04:00")
    t0 = pd.Timedelta("0 days 00:03:00")
    t1 = pd.Timedelta("0 days 00:01:00")

    # add the runner poistion
    df['runner_position'] = pd.to_numeric(df['runner_position'])
    runner_position = df.groupby(id_cols)[['runner_position']].mean()
    df = df.drop(columns=['runner_position'])

    df['tot_bl_imbalance'] = df['total_lay_qty'] - df['total_back_qty']
    df['best_bl_imbalance'] = df['best_lay_cum_qty'] - df['best_back_cum_qty']
    df['order_is_back'] = df['order_type'] == 'back'

    to_keep_columns = ['best_lay', 'best_back', 'best_lay_cum_qty', 'best_back_cum_qty', 'total_lay_qty', 'total_back_qty', 'best_lay_q_100', 'best_back_q_100', 'best_lay_q_1000', 'best_back_q_1000']
    to_keep_columns += ['tot_bl_imbalance', 'best_bl_imbalance']
    qty_columns = ['qty','prc','order_is_back']
    # t0 col to keep
    df, dt0 = add_time_snapshot(df, t0)
    df, dt1 = add_time_snapshot(df, t1)
    df, dtm3 = add_time_snapshot(df, tm3)
    df, dtm2 = add_time_snapshot(df, tm2)
    df, dtm1 = add_time_snapshot(df, tm1)

    # computing momentums
    m_input_0 = dt0.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()
    m_input_1 = dt1.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()
    m_input_3 = dtm3.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()
    m_input_2 = dtm2.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()

    mom_3_1 = (m_input_1 - m_input_3).add_suffix('_mom_3_1').reset_index().groupby(id_cols).mean()
    mom_2_1 = (m_input_1 - m_input_2).add_suffix('_mom_2_1').reset_index().groupby(id_cols).mean()

    # computing variance and trade 3_1 (then 2_1)
    temp = df.loc[df['time_delta'].between(tm1, tm3),:].copy()
    std_3_1 = temp.groupby(id_cols)[to_keep_columns].std().reset_index().groupby(id_cols).mean().add_suffix('_std_3_1')
    temp = temp.loc[temp['qty']>0,:]
    qty_count_3_1 = temp.groupby(id_cols)['qty'].count().reset_index().groupby(id_cols).mean().add_suffix('_count_3_1')
    qty_mean_3_1 = temp.groupby(id_cols)[qty_columns].mean().reset_index().groupby(id_cols).mean().add_suffix('_mean_3_1')
    qty_std_3_1 = temp.groupby(id_cols)[qty_columns].std().reset_index().groupby(id_cols).mean().add_suffix('_std_3_1')
    order_type_3_1 = temp.groupby(id_cols)[['order_is_back']].mean().add_suffix('_order_is_back_3_1').reset_index().groupby(id_cols).mean()

    # same but for 2_1
    temp = df.loc[df['time_delta'].between(tm1, tm2),:].copy()
    std_2_1 = temp.groupby(id_cols)[to_keep_columns].std().reset_index().groupby(id_cols).mean().add_suffix('_std_2_1')
    temp = temp.loc[temp['qty'] > 0, :]
    qty_count_2_1 = temp.groupby(id_cols)['qty'].count().reset_index().groupby(id_cols).mean().add_suffix('_count_2_1')
    qty_mean_2_1 = temp.groupby(id_cols)[qty_columns].mean().reset_index().groupby(id_cols).mean().add_suffix('_mean_2_1')
    qty_std_2_1 = temp.groupby(id_cols)[qty_columns].std().reset_index().groupby(id_cols).mean().add_suffix('_std_2_1')
    order_type_2_1 = temp.groupby(id_cols)[['order_is_back']].mean().add_suffix('_order_is_back_2_1').reset_index().groupby(id_cols).mean()


    # ensure everything is indexed the same way
    m_input_0 = m_input_0.add_suffix('_m0')
    m_input_1 = m_input_1.add_suffix('_m1')
    m_input_2 = m_input_2.add_suffix('_m2')
    m_input_3 = m_input_3.add_suffix('_m3')

    # straight merge chain
    final_df = (
        m_input_0
        .merge(m_input_1, on=id_cols, how='outer')
        .merge(m_input_2, on=id_cols, how='outer')
        .merge(m_input_3, on=id_cols, how='outer')
        .merge(mom_3_1, on=id_cols, how='outer')
        .merge(mom_2_1, on=id_cols, how='outer')
        .merge(std_3_1, on=id_cols, how='outer')
        .merge(std_2_1, on=id_cols, how='outer')
        .merge(qty_count_3_1, on=id_cols, how='outer')
        .merge(qty_mean_3_1, on=id_cols, how='outer')
        .merge(qty_mean_3_1, on=id_cols, how='outer')
        .merge(qty_std_3_1, on=id_cols, how='outer')
        .merge(qty_count_2_1, on=id_cols, how='outer')
        .merge(qty_mean_2_1, on=id_cols, how='outer')
        .merge(qty_mean_2_1, on=id_cols, how='outer')
        .merge(qty_std_2_1, on=id_cols, how='outer')
        .merge(order_type_3_1, on=id_cols, how='outer')
        .merge(order_type_2_1, on=id_cols, how='outer')
        .merge(runner_position, on=id_cols, how='outer')
    )
    # solve any remaining duplicates by averaging
    final_df = final_df.reset_index().groupby(id_cols).mean()
    final_df.reset_index().merge(mdef[fixed_effect_columns],how='left',on='file_name')

