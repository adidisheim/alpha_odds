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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def plot_value_matrix(df, figsize=(8, 6), vmin=None, vmax=None):
    """
    Plot a matrix-like DataFrame with:
      - color coded squares
      - numbers in each square (rounded to 2 decimals)
      - distinct hatched pattern for NaN values
    """

    # Blue to green colormap
    colors = ["#4C72B0", "#55A868"]
    cmap = LinearSegmentedColormap.from_list("blue_green", colors, N=256)

    data = df.values
    n_rows, n_cols = data.shape

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    fig, ax = plt.subplots(figsize=figsize)

    # Main heatmap (NaNs ignored)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Overlay hatching (NaNs)
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(data[i, j]):
                # Draw a hatched rectangle for NaN
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="red",
                        linewidth=1.5
                    )
                )

    # Tick labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add numbers
    for i in range(n_rows):
        for j in range(n_cols):
            value = data[i, j]
            if not np.isnan(value):
                frac = (value - vmin) / (vmax - vmin + 1e-12)
                color = "white" if frac > 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value")

    ax.set_title("Matrix plot")
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = parse()
    df = load_and_merge_ss()

    df = df.reset_index()
    print(df['file_name'].nunique(),' unique races')
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


    df['mid_odds'] = 1 / df['mid_prb_win']

    # skewness strat
    rate = 0.08
    df['payout_back'] = (df['won']*df['mid_odds'])-1
    df['payout_back_net'] = (((df['mid_odds']-1)*(1-rate)+1)*df['won'])-1
    df['payout_lay'] = (1 - df['payout_back'])-1
    df['payout_lay_net'] = np.where(
        df['won'] == 1,
        1 - df['mid_odds'],  # lose O-1 if it wins
        1 - rate  # win 1 minus commission if it loses
    )
    df.groupby('vol_rank')[['payout_back','payout_back_net']].mean()
    df.groupby(['year','vol_rank'])['payout_back_net'].mean()
    df.groupby('vol_rank')[['payout_lay','payout_lay_net']].mean()


    min_nb = 1000
    ind = (df['mid_prb_win_rounded']<=40) & (df['year']>=2023)
    nb = df.loc[ind,:].groupby(['mid_prb_win_rounded','vol_rank'])['won'].count().unstack()
    won = df.loc[ind,:].groupby(['mid_prb_win_rounded','vol_rank'])['won'].mean().unstack()
    payout_lay_net = df.loc[ind,:].groupby(['mid_prb_win_rounded','vol_rank'])['payout_lay_net'].mean().unstack()
    payout_back_net = df.loc[ind,:].groupby(['mid_prb_win_rounded','vol_rank'])['payout_back_net'].mean().unstack()
    payout_lay_net = payout_lay_net.where(nb >= min_nb, np.nan)
    payout_back_net = payout_back_net.where(nb >= min_nb, np.nan)
    won = won.where(nb >= min_nb, np.nan)

    plot_value_matrix(won)
    plot_value_matrix(payout_lay_net)
    plot_value_matrix(payout_back_net)


