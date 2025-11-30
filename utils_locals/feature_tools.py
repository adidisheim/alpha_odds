import numpy as np
import pandas as pd

def add_time_snapshot(df,snap_time):
    temp = df[['file_name','id']].drop_duplicates()
    for c in df.columns:
        if c not in ['file_name','id','time_delta']:
            temp[c] = np.nan
    temp['time_delta'] = snap_time
    # df = pd.concat([temp, df], axis=0, ignore_index=True)
    df = pd.concat([temp.astype(df.dtypes), df], ignore_index=True) # (to fix the stupide warning of pandas)
    df = df.sort_values(['file_name','id','time_delta'], ascending=[True, True, False])
    df = df.ffill()
    return df, df.loc[df['time_delta'] == snap_time, :].drop_duplicates().copy()