import pandas as pd

from utils_locals.loader import load_and_merge_ss
from utils_locals.parser import parse
import numpy as np
#
import socket


def plot_temp_df(df):
    fig, ax = plt.subplots(figsize=(10, 6))


    cols = df.columns

    for i, col in enumerate(cols):

        ax.plot(df.index, df[col], label=str(col), linewidth=1.8)

    ax.set_xlabel("time_delta_sec")
    ax.set_ylabel("value")
    ax.set_title("temp dataframe")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = parse()
    if socket.gethostname() == 'UML-FNQ2JDW1GV':
        from matplotlib import pyplot as plt
        # temp = pd.read_pickle('ss_qty_per_time.p')
        temp = pd.read_pickle('ss_mid_diff_per_time.p')
    else:
        df = load_and_merge_ss()
        # temp = df.groupby(['time_delta_sec','vol_rank'])['qty'].mean().unstack()
        # temp.to_pickle('ss_qty_per_time.p')
        df = df.merge(df.loc[df['time_delta_sec']==0,['file_name','id','mid_prb_win']].rename(columns={'mid_prb_win':'last_prb'}),how='left')
        df['delta_prb_abs'] = (df['mid_prb_win']-df['last_prb']).abs()
        temp = df.groupby(['time_delta_sec','vol_rank'])['delta_prb_abs'].quantile(0.75).unstack()
        temp.to_pickle('ss_mid_diff_per_time.p')
    temp.index/=60
    # temp = temp.loc[temp.index<=10,:]
    plot_temp_df(temp)

    # df['date'] = pd.to_datetime(df['date'])
    # df['year'] = df['date'].dt.year
    # df['mid_prb_win_rounded'] = ((df['mid_prb_win'] *100)/ 2.5).round() * 2.5
    # df.groupby('vol_rank')['won'].mean()
    #
    # temp = df.groupby('mid_prb_win_rounded')['won'].aggregate(['mean', 'count'])
    # temp['mean'] = temp['mean']*100
    # temp = temp.loc[temp['count']>10000]
    # plt.scatter(temp.index, temp['mean'])
    # plt.plot([0,temp['mean'].max()],[0,temp['mean'].max()], color='red')
    # plt.xlabel('Mid implied prob (%)')
    # plt.ylabel('Empirical freq won (%)')
    # plt.tight_layout()
    # plt.show()
    #

