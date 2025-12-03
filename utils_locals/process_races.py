import bz2
import json
import numpy as np
import pandas as pd


def process_order(ord, time,index):
    try:
        if ('atb' in ord.keys()) | ('atl' in ord.keys())| ('trd' in ord.keys()):
        # if ('atb' in ord.keys()) | ('atl' in ord.keys()):
            ord_type = list(ord.keys())[0]
            temp = pd.DataFrame(ord[ord_type],columns =['prc','qty'])
            temp['type'] = ord_type
            temp['time'] = time
            temp['index'] = index
            temp['runner_id'] = ord['id']
        else:
            temp = pd.DataFrame()
    except:
        raise 'Error processing order'
    return temp

def process_event_list(event_list, time,index):
    event_df  = pd.DataFrame()
    for ord in event_list:
        event_df = pd.concat([event_df, process_order(ord, time,index)], ignore_index=True)
    return event_df

def get_half_order_book(df_all_bl, dog_id, type_trade):
    df_all_bl=df_all_bl.drop_duplicates()
    df = df_all_bl[(df_all_bl['runner_id'] == dog_id) & (df_all_bl['type'] == type_trade)].copy()
    try:
        df = df.groupby(['time','prc'])['qty'].sum().reset_index()
        df = df.pivot(columns='prc', values='qty', index=['time'])
    except:
        breakpoint()
    df = df.sort_index(ascending=True)
    df = df.ffill().fillna(0)
    return df



def get_best_value(df, order_type, q=0.0):
    """
    For a given order book half (df: time x price with qty),
    return:
      - best price at which an order of size q can be executed
      - cumulative quantity available up to that price (including all better levels)

    For q == 0, this reduces to the top-of-book price and its quantity.
    """
    if order_type not in ("atl", "atb"):
        raise ValueError("order_type must be 'atl' or 'atb'")
    if order_type == 'atb':
        col = pd.Series(df.columns).sort_values(ascending=False).values
        df = df[col]
    df = df.fillna(0)
    cum_qty = df.cumsum(axis=1)
    qty_times_price= df.mul(df.columns, axis=1)
    avg_price = qty_times_price.cumsum(axis=1)/cum_qty
    mask = cum_qty.gt(q).cumsum(axis=1)
    mask = mask==1

    execution_price = avg_price.where(mask).stack().reset_index(level=1, drop=True)
    max_qty = cum_qty.where(mask).stack().reset_index(level=1, drop=True)
    return execution_price.reindex(df.index), max_qty.reindex(df.index)


def get_best_value_old(df, order_type, q=0.0):
    """
    For a given order book half (df: time x price with qty),
    return:
      - best price at which an order of size q can be executed
      - cumulative quantity available up to that price (including all better levels)

    For q == 0, this reduces to the top-of-book price and its quantity.
    """
    if order_type not in ("atl", "atb"):
        raise ValueError("order_type must be 'atl' or 'atb'")

    prices = df.columns.to_numpy(float)
    data = df.to_numpy()  # shape (T, N)

    if order_type == "atb":
        # for backs, best is highest price so we work from right to left
        prices = prices[::-1]
        data = data[:, ::-1]

    T, N = data.shape

    best_price = np.full(T, np.nan, dtype=float)
    best_qty = np.full(T, np.nan, dtype=float)

    if q == 0.0:
        # top-of-book: first level with strictly positive qty
        for i in range(T):
            row = data[i]
            pos_mask = row > 0
            if not np.any(pos_mask):
                continue
            j = np.argmax(pos_mask)
            best_price[i] = prices[j]
            best_qty[i] = row[j]
    else:
        # find first level where cumulative qty exceeds q
        cum = np.cumsum(data, axis=1)  # along price levels
        for i in range(T):
            total = cum[i, -1]
            if total < q:
                # not enough liquidity to fill size q
                continue
            row_cum = cum[i]
            hit_mask = row_cum > q
            if not np.any(hit_mask):
                continue
            j = np.argmax(hit_mask)
            best_price[i] = prices[j]
            best_qty[i] = row_cum[j]

    # restore original time index
    best_price = pd.Series(best_price, index=df.index, name="best_price")
    best_qty = pd.Series(best_qty, index=df.index, name="cum_qty")


    # rows with no volume at any price should return NaN for both
    total_vol = (data > 0).sum(axis=1)
    no_vol_mask = total_vol == 0
    best_price[no_vol_mask] = np.nan
    best_qty[no_vol_mask] = np.nan
    return best_price, best_qty

def infer_order_type(row):
    prc = row["prc"]
    if pd.isna(prc):
        return np.nan
    if prc >= row["best_lay"]:
        return "lay"
    if prc <= row["best_back"]:
        return "back"
    return "cross_matching"

def load_bz2_json(file_path)-> pd.DataFrame:
    with bz2.open(file_path, mode='rt') as f:
        lines = f.readlines()
    # Parse each line as JSON
    data = [json.loads(line) for line in lines]
    return pd.DataFrame(data)

def process_market_file(df_full):
    # find market_defitnion
    ind_market_def = df_full['mc'].apply(lambda x: 'marketDefinition' in x[0].keys())
    m_def = df_full.loc[ind_market_def, 'mc'].apply(lambda x: x[0]['marketDefinition'])
    first_mdef = pd.Series(m_def.iloc[0])
    runners = pd.DataFrame(m_def.iloc[-1]['runners']).copy()

    # latter we use -1 as id to indicate winner so we need to assert no dog has id -1
    assert (runners['id'] == -1).sum() == 0, 'runner with id -1 found, conflicts with winner id convention.'
    return runners, first_mdef

def create_df_with_all_atl_and_atb(df_full):
    ind_market_def = df_full['mc'].apply(lambda x: 'marketDefinition' in x[0].keys())
    mc = df_full.loc[~ind_market_def, 'mc']
    mc_time = df_full.loc[~ind_market_def, 'pt']
    # extract the first element
    mc = mc.apply(lambda x: x[0])

    # assert that our cleaning assumptions holds
    assert mc.apply(lambda x: x['id']).nunique()==1, 'multiple market ids found in the same file??'
    assert mc.apply(lambda x: x['con']).unique() == np.array([True]), 'con!=[True] in this set of mc'
    assert mc.apply(lambda x: x['img']).unique() == np.array([False]), 'img!=[false] in this set of mc'

    # extract the rcs and format as a df for processing with apply (potentiall pandarallel)
    rc = mc.apply(lambda x: x['rc'])
    rc = pd.DataFrame(rc).rename(columns={'mc': 'rc'})
    rc['time'] = mc_time
    rc = rc.reset_index()

    lists_of_events_df = rc.apply(lambda x: process_event_list(x['rc'], x['time'],x['index']),axis=1)
    df_all_bl = pd.concat(lists_of_events_df.tolist(), ignore_index=True)
    return df_all_bl

def process_runner_order_book(df_all_bl, runners, runner_id, q_low=0, q_grid=None) -> pd.DataFrame:
    if q_grid is None:
        q_grid = [100, 1000]
    lay = get_half_order_book(df_all_bl, runner_id, 'atl')
    back = get_half_order_book(df_all_bl, runner_id, 'atb')

    best_lay, qty_lay = get_best_value(lay, "atl", q_low)
    best_back, qty_back = get_best_value(back, "atb", q_low)

    lb_df = (
        pd.DataFrame(
            {
                "best_lay": best_lay,
                "best_back": best_back,
                "best_lay_cum_qty": qty_lay,
                "best_back_cum_qty": qty_back,
                "total_lay_qty": lay.sum(axis=1),
                "total_back_qty": back.sum(axis=1),
            }
        )
        .ffill().dropna() # we dropna now so we drop the initial times with absolutely no liquidity
    )

    # # add to the lb_df the best prices and cum qty for each q in the q_grid
    for q in q_grid:
        best_lay_high, qty_lay_high = get_best_value(lay, "atl", q)
        best_back_high, qty_back_high = get_best_value(back, "atb", q)
        lb_df[f'best_lay_q_{q}'] = best_lay_high
        lb_df[f'best_back_q_{q}'] = best_back_high
        lb_df[[f'best_lay_q_{q}', f'best_back_q_{q}']] = lb_df[[f'best_lay_q_{q}', f'best_back_q_{q}']].ffill()
        # same for the cumulative qty.
        # lb_df[f'best_lay_cum_qty_q_{q}'] = qty_lay_high
        # lb_df[f'best_back_cum_qty_q_{q}'] = qty_back_high
        # lb_df[[f'best_lay_cum_qty_q_{q}', f'best_back_cum_qty_q_{q}']] = lb_df[[f'best_lay_cum_qty_q_{q}', f'best_back_cum_qty_q_{q}']].ffill()



    # extracting the trade for that  runner
    trd = df_all_bl.loc[(df_all_bl['runner_id'] == runner_id) & (df_all_bl['type'] == 'trd'), :]
    trd = trd.groupby(['time','prc'])['qty'].sum().reset_index()
    # drop trd with zero qty (seems to happen sometimes)
    ind = trd['qty']>0
    trd = trd.loc[ind,:]

    # we have a few trd with the same time we just sume the qty and avg the price
    if trd['time'].duplicated().any():
        a = trd.groupby(['time'])['qty'].sum().reset_index()
        b = trd.groupby(['time'])['prc'].mean().reset_index()
        trd = a.merge(b, on='time')
        del a,b
        if trd['time'].duplicated().any():
            breakpoint()
    trd = trd.set_index('time')
    temp = lb_df.merge(trd, left_index=True, right_index=True, how='outer').sort_index()

    # after merging we need to ffill best lay and back prices for all variations
    col_list = [x for x in temp.columns if x.startswith('best_lay') or x.startswith('best_back')]
    temp[col_list] = temp[col_list].ffill()
    # now we can infer the order type by closness to best lay/back
    if temp.shape[0]>0: # if the particular runner has no trades at all temp can be empty
        temp['order_type'] = temp.apply(infer_order_type, axis=1)
        # check if the runner is the winner
        runner_won = runners.loc[runners['id'] == runner_id, 'status'].iloc[0] == 'WINNER'
        temp['id'] = -1 if runner_won else runner_id

    return temp



if __name__ == '__main__':
    # example usage
    file_path = 'data/raw/PRO/2025/Oct/1/34791542/1.248460630.bz2'
    df_full = load_bz2_json(file_path)
    # process time
    df_full["pt"] = pd.to_datetime(df_full["pt"], unit="ms")
    ### extract all mc from non-market-definition entries
    runners, first_mdef = process_market_file(df_full)
    df_all_bl = create_df_with_all_atl_and_atb(df_full)

    df = pd.DataFrame()
    for runner_id in runners['id'].tolist():
        temp = process_runner_order_book(df_all_bl, runners, runner_id, q_low=0, q_grid=[100, 1000])
        df = pd.concat([df, temp], ignore_index=False)
