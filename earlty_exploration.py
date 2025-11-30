import bz2
import json

import pandas as pd


def diff_market_definitions(mdefs):
    """
    mdefs: list of marketDefinition dicts in chronological order
    """
    def dict_diff(d1, d2, path=""):
        """
        Recursively find differences between d1 and d2.
        Returns a list of (path, old, new).
        """
        diffs = []

        # keys in either dict
        for key in sorted(set(d1.keys()) | set(d2.keys())):
            p = f"{path}.{key}" if path else key

            if key not in d1:
                diffs.append((p, None, d2[key]))
                continue
            if key not in d2:
                diffs.append((p, d1[key], None))
                continue

            v1, v2 = d1[key], d2[key]

            # nested dict
            if isinstance(v1, dict) and isinstance(v2, dict):
                diffs.extend(dict_diff(v1, v2, p))
                continue

            # nested list (runners list)
            if isinstance(v1, list) and isinstance(v2, list):
                # compare by index
                max_len = max(len(v1), len(v2))
                for i in range(max_len):
                    if i >= len(v1):
                        diffs.append((f"{p}[{i}]", None, v2[i]))
                        continue
                    if i >= len(v2):
                        diffs.append((f"{p}[{i}]", v1[i], None))
                        continue
                    # if elements are dicts, recurse
                    if isinstance(v1[i], dict) and isinstance(v2[i], dict):
                        diffs.extend(dict_diff(v1[i], v2[i], f"{p}[{i}]"))
                    else:
                        if v1[i] != v2[i]:
                            diffs.append((f"{p}[{i}]", v1[i], v2[i]))
                continue

            # primitive values
            if v1 != v2:
                diffs.append((p, v1, v2))

        return diffs

    for i in range(1, len(mdefs)):
        print(f"\nChanges from #{i-1} → #{i}")
        diffs = dict_diff(mdefs[i-1], mdefs[i])
        for p, old, new in diffs:
            print(f"{p}: {old} → {new}")

if __name__ == '__main__':
    file_path = 'data/raw/PRO/2025/Oct/1/34791542/1.248460630.bz2'
    # file_path = 'data/raw/PRO/2025/Oct/1/34791542/1.248460682.bz2'
    # file_path = 'data/raw/PRO/2025/Oct/1/34791542/1.248460675.bz2'

    with bz2.open(file_path, mode='rt') as f:
        lines = f.readlines()

    # Parse each line as JSON
    data = [json.loads(line) for line in lines]

    # simple df
    df_full = pd.DataFrame(data)

    # study time
    df_full["pt"] = pd.to_datetime(df_full["pt"], unit="ms")
    # check that it's running in order
    df_full["delta_pt_ms"] = df_full["pt"].diff().dt.total_seconds() * 1000
    df_full['delta_pt_ms'].describe()
    df_full['op'].unique() # it's all MCM to check in future code with assert

    # check when and why market defitnion is in mc
    ind = df_full['mc'].apply(lambda x: 'marketDefinition' in x[0].keys())
    print('percentage of entry with marketDefinition in mc:', ind.mean(),' total count:', ind.sum())
    all_m_def = df_full.loc[ind, 'mc'].apply(lambda x: x[0]['marketDefinition'])
    k = 0
    # study the changes in market defitnions
    diff_market_definitions(all_m_def.tolist())

    # save runners list for latter comparison
    runners = all_m_def.iloc[0]['runners']


    ### study the non market defintion order
    ind = ~df_full['mc'].apply(lambda x: 'marketDefinition' in x[0].keys())
    mc = df_full.loc[ind,'mc']
    # seems to always be a list of 1
    mc.apply(len).unique()
    # extract the first element
    mc = mc.apply(lambda x: x[0])

    # contains only id, rc, con, img
    mc.apply(lambda x: x.keys()).explode().unique()

    # all the same market id
    mc.apply(lambda x: x['id']).nunique()

    # in sample observer, "con" is always true (and not even defined so fuck it)
    mc.apply(lambda x: x['con']).unique()

    # and img is always false (notes said can be disregarded as part of the dataset
    mc.apply(lambda x: x['img']).unique()

    # --> it's all about rc .
    rc = mc.apply(lambda x: x['rc'])

    # length varies a lot.
    rc.apply(len)
    # concatenate all the lists into one to study the uniqe type of events
    rc_concat = [x for sub in rc for x in sub]
    # possible type of unique dictionary: atb, spn, atl, spl, trd, spb (multipole combinations are possible on any given day)
    pd.Series([list(x.keys())[0] for x in rc_concat]).unique()
    # defintions of those key stuff from the pdf:
    '''
    atb: Available to Back - PriceVol tuple delta of price changes (0 vol is remove) 
    atl: Available to Lay - PriceVol tuple delta of price changes (0 vol is remove)
    
    I think th
    spl: Starting price Lay - PriceVol tuple delta of spl price changes (0 vol is remove) 
    spb: Starting Price Back - PriceVol tuple delta of price changes (0 vol is remove)
    -- check hypothesis: it's atb/atl but just first value for a given odds. 
    
    spn: Starting price Near - The near starting price spn (or null if un-changed)
    trd: Traded PriceVol tuple delta of price trd changes (0 vol is remove) 
    
    todo/ask what's the difference between starting and available
    '''
    # checking number of each event.
    nb_order_types = pd.Series([list(x.keys())[0] for x in rc_concat]).value_counts()
    print(nb_order_types)
    print(nb_order_types/nb_order_types.sum())

    ## find all the trd
    rc_series = pd.Series(rc_concat)
    ind = rc_series.apply(lambda x: 'trd' in x.keys())
    trd = rc_series.loc[ind]
    # total volume is sometimes zero?
    tv = trd.apply(lambda x: x['tv'])
    print((tv==0).mean())    # 2% of the time total volume is zero, why?
    # total volume per race
    print("total traded", tv.sum())
    # outside of this mystery, it seems fairly straightforward market order lists.
    trd.loc[tv>0].iloc[-1]

    ## move to atb.
    atb = rc_series.loc[rc_series.apply(lambda x: 'atb' in x.keys())]
    # again do I understand correctly that if zero volume it's a removal? Meaning whatever was there before isn ow gone entierly?
    atb.iloc[50]

    # sam to atl
    atl = rc_series.loc[rc_series.apply(lambda x: 'atl' in x.keys())]
    atl.iloc[50]



