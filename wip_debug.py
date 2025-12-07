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
from utils_locals.feature_tools import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def compute_features(paths, t_definition = 0):
    df = pd.read_parquet(paths)
    mdef = pd.read_parquet(paths.replace('win_', 'mdef_'))
    id_cols = ['file_name', 'id']
    df['runner_position'] = pd.to_numeric(df['runner_position'])

    # mdef = mdef.loc[mdef['marketType'] == 'WIN', :].copy()
    mdef["marketTime_local"] = mdef.apply(lambda r: pd.to_datetime(r.marketTime, utc=True).tz_convert(r.timezone), axis=1)
    mdef["marketTime_local"] = pd.to_datetime(mdef["marketTime_local"].apply(lambda x: str(x).split(' ')[0]), errors='coerce')
    mdef['local_dow'] = mdef['marketTime_local'].dt.dayofweek

    # ffe columns
    fixed_effect_columns = ['file_name', 'local_dow', 'marketBaseRate', 'marketType', 'numberOfActiveRunners', 'venue','marketTime_local']

    df[['qty', 'prc']] = df[['qty', 'prc']].fillna(0.0)
    df = df.reset_index()
    if df.columns[0] == 'index':
        df = df.rename(columns={'index':'time'})
    df['time_delta'] = df.groupby('file_name')['time'].transform('max') - df['time']

    # define the exact time values for the snapshot
    if t_definition == 0:
        tm3 = pd.Timedelta("0 days 00:10:00")
        tm2 = pd.Timedelta("0 days 00:03:00")
        tm1 = pd.Timedelta("0 days 00:01:15")
        t0 = pd.Timedelta("0 days 00:01:00")
        tp1 = pd.Timedelta("0 days 00:00:10")
    elif t_definition == 1:
        tm3 = pd.Timedelta("0 days 00:10:00")
        tm2 = pd.Timedelta("0 days 00:04:00")
        tm1 = pd.Timedelta("0 days 00:02:15")
        t0 = pd.Timedelta("0 days 00:02:00")
        tp1 = pd.Timedelta("0 days 00:00:10")
    else:
        raise ValueError('t_definition not defined')

    # add the runner poistion
    df['runner_position'] = pd.to_numeric(df['runner_position'])
    runner_position = df.groupby(id_cols)[['runner_position']].mean()
    df = df.drop(columns=['runner_position'])

    df['tot_bl_imbalance'] = df['total_lay_qty'] - df['total_back_qty']
    df['best_bl_imbalance'] = df['best_lay_cum_qty'] - df['best_back_cum_qty']
    df['order_is_back'] = df['order_type'] == 'back'

    to_keep_columns = ['best_lay', 'best_back', 'best_lay_cum_qty', 'best_back_cum_qty', 'total_lay_qty', 'total_back_qty', 'best_lay_q_100', 'best_back_q_100', 'best_lay_q_1000', 'best_back_q_1000']
    to_keep_columns += ['tot_bl_imbalance', 'best_bl_imbalance']
    qty_columns = ['qty', 'prc', 'order_is_back']

    # transform all lay/back odds to implied prob
    col_with_back_lay = ['best_lay', 'best_back', 'best_lay_q_100', 'best_back_q_100', 'best_lay_q_1000', 'best_back_q_1000']
    for col in col_with_back_lay:
        df[col] = 1 / df[col]

    # t0 col to keep
    df, dtp1 = add_time_snapshot(df, tp1)
    df, dt0 = add_time_snapshot(df, t0)
    df, dtm3 = add_time_snapshot(df, tm3)
    df, dtm2 = add_time_snapshot(df, tm2)
    df, dtm1 = add_time_snapshot(df, tm1)

    # computing momentums
    p_input_1 = dtp1.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()
    m_input_0 = dt0.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()
    m_input_1 = dtm1.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()
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
    p_input_1 = p_input_1.add_suffix('_p1')
    m_input_0 = m_input_0.add_suffix('_m0')
    m_input_1 = m_input_1.add_suffix('_m1')
    m_input_2 = m_input_2.add_suffix('_m2')
    m_input_3 = m_input_3.add_suffix('_m3')

    # straight merge chain
    final_df = (
        m_input_0
        .merge(m_input_1, on=id_cols, how='outer')
        .merge(p_input_1, on=id_cols, how='outer')
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
    final_df = final_df.reset_index().merge(mdef[fixed_effect_columns], how='left', on='file_name')
    return final_df

if __name__ == '__main__':
    args = parse()
    t_definition = args.b
    start_path = Constant.DATA_DIR+'p/greyhound_au/'
    file_to_run = np.sort([x for x in os.listdir(start_path) if 'win_' in x])
    save_dir = Constant.RES_DIR+f'features_t{t_definition}/'
    os.makedirs(save_dir, exist_ok=True)
    file_to_run = np.array_split(file_to_run, 10)[args.a]

    df = pd.DataFrame()
    k = 0
    # for file in file_to_run:
    for file in ['win_2017_Oct_15.parquet']:
    # for file in ['win_2017_Oct_16.parquet']:
    # for file in ['win_2017_Oct_16.parquet']:
    # for file in ['win_2018_May_9.parquet']:
        print(f'Processing file {file}', flush=True)
        temp = compute_features(start_path+file,t_definition)
        if '1.247509320.bz2' in temp['file_name']:
            breakpoint()
        # temp.loc[temp['best_lay_q_100_m0']>100,['best_lay_q_100_m0']]
        # temp.loc[temp['file_name']=='1.246471259.bz2',['best_lay_q_100_m0']]
        df = pd.concat([df, temp], ignore_index=False)
        k+=1
        if k % 10==0:
            print('Saving intermediate file...',flush=True)
            df.to_parquet(save_dir+f'greyhound_au_features_part_{args.a}.parquet')
            print('saved to:', save_dir+f'greyhound_au_features_part_{args.a}.parquet', flush=True)
    df.to_parquet(save_dir+f'greyhound_au_features_part_{args.a}.parquet')


    df['in_out_lay'] = (df['best_lay_m0'] - df['best_back_m1'])/(df['best_lay_m0']-1)
    df['in_out_back'] = df['best_back_m0']  - df['best_lay_m1']







