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
import didipack
from matplotlib import pyplot as plt
from utils_locals.loader import load_and_merge_ss

if __name__ == '__main__':
    args = parse()
    df = load_and_merge_ss()

    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['mid_prb_win_rounded'] = ((df['mid_prb_win'] *100)/ 2.5).round() * 2.5
    df.groupby('vol_rank')['won'].mean()

    temp = df.groupby('mid_prb_win_rounded')['won'].aggregate(['mean', 'count'])
    temp['mean'] = temp['mean']*100
    temp = temp.loc[temp['count']>10000]
    plt.scatter(temp.index, temp['mean'])
    plt.plot([0,temp['mean'].max()],[0,temp['mean'].max()], color='red')
    plt.xlabel('Mid implied prob (%)')
    plt.ylabel('Empirical freq won (%)')
    plt.tight_layout()
    plt.show()


