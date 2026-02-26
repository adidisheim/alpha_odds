"""Resume artifact saving: finish t2/t3 isotonic + manifest + V1 norm."""
import json, os, pickle, socket
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

RES_DIR = '/data/projects/punim2039/alpha_odds/res/'
SAVE_DIR = os.path.join(RES_DIR, 'paper_trading_artifacts/')
os.makedirs(SAVE_DIR, exist_ok=True)
V1_TOP_N = 7
V2_TOP_N = 15
CROSS_COLS = ['prob_rank','prob_vs_favorite','prob_share','race_herfindahl','n_close_runners','spread_m0','spread_rank','total_qty_m0','volume_rank','avg_mom_3_1','momentum_rank','race_overround','is_favorite','prob_deviation','bl_imbalance_rank']

def custom_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def get_configs_ranked(model_dir, t_def):
    base_dir = os.path.join(RES_DIR, f'{model_dir}/t{t_def}')
    if not os.path.exists(base_dir): return []
    configs = {}
    for c in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, c, 'save_df.parquet')
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                ll = custom_log_loss(df['win'].values, df['model_prob'].clip(0.001,0.999).values)
                configs[c] = ll
            except: pass
    return sorted(configs.keys(), key=lambda c: configs[c])

def load_features_and_prep(t_def):
    load_dir = os.path.join(RES_DIR, f'features_t{t_def}')
    merged = os.path.join(load_dir, 'greyhound_au_features_merged.parquet')
    if os.path.exists(merged):
        df = pd.read_parquet(merged)
    else:
        df = pd.DataFrame()
        for i in range(10):
            try: df = pd.concat([df, pd.read_parquet(os.path.join(load_dir, f'greyhound_au_features_part_{i}.parquet'))], ignore_index=False)
            except: pass
    df['win'] = (df['id'] == -1).astype(int)
    df = df.dropna(subset=['best_back_m0','best_lay_m0'])
    g = df.groupby('file_name')
    df['market_prob'] = df[['best_back_m0','best_lay_m0']].mean(axis=1)
    df['prob_rank'] = g['market_prob'].rank(method='min', ascending=False)
    df['prob_vs_favorite'] = df['market_prob'] / g['market_prob'].transform('max')
    df['prob_share'] = df['market_prob'] / g['market_prob'].transform('sum')
    df['_psq'] = df['prob_share']**2
    df['race_herfindahl'] = df.groupby('file_name')['_psq'].transform('sum')
    df.drop(columns=['_psq'], inplace=True)
    rs = g['market_prob'].transform('std').fillna(0)
    df['n_close_runners'] = (rs<0.05).astype(int)*(g['market_prob'].transform('count')-1)
    df['spread_m0'] = (df['best_back_m0']-df['best_lay_m0']).abs()
    df['spread_rank'] = g['spread_m0'].rank(method='min', ascending=True)
    df['total_qty_m0'] = df['total_back_qty_m0']+df['total_lay_qty_m0']
    df['volume_rank'] = g['total_qty_m0'].rank(method='min', ascending=False)
    df['avg_mom_3_1'] = df[['best_back_mom_3_1','best_lay_mom_3_1']].mean(axis=1)
    df['momentum_rank'] = g['avg_mom_3_1'].rank(method='min', ascending=False)
    df['race_overround'] = g['market_prob'].transform('sum')
    df['is_favorite'] = (df['prob_rank']==1).astype(int)
    df['prob_deviation'] = df['market_prob']-g['market_prob'].transform('mean')
    df['bl_imbalance_m0'] = df['best_bl_imbalance_m0']
    df['bl_imbalance_rank'] = g['bl_imbalance_m0'].rank(method='min', ascending=False)
    df['total_qty_m1'] = df[['total_back_qty_m1','total_lay_qty_m1']].sum(axis=1)
    df['total_qty_m3'] = df[['total_back_qty_m3','total_lay_qty_m3']].sum(axis=1)
    for col in ['total_qty_m1','total_back_qty_m1','total_lay_qty_m1','total_qty_m3','total_back_qty_m3','total_lay_qty_m3']:
        df[col+'_frac'] = df[col]/df.groupby('file_name')[col].transform('sum')
    for col in [x for x in df.columns if x.endswith('_m1_frac')]:
        df[col.replace('_m1','_mom_3_1')] = df[col]-df[col.replace('_m1','_m3')]
    return df

def get_predictors(df):
    sfx = ["_count_2_1","_count_3_1","_mean_2_1","_mean_3_1","_m0","_mom_2_1","_mom_3_1","_order_is_back_2_1","_order_is_back_3_1","_std_2_1","_std_3_1"]
    p = [c for c in df.columns if c.endswith(tuple(sfx))]
    p += [c for c in df.columns if '_mom_3_1' in c and '_frac' in c.replace('_mom_3_1','')]
    p += [c for c in df.columns if c.endswith('_frac')]
    if 'runner_position' in df.columns: p.append('runner_position')
    p += ['local_dow','marketBaseRate','numberOfActiveRunners']
    p += CROSS_COLS
    seen = set(); return [c for c in p if not (c in seen or seen.add(c))]

def extract_norm(normalizer):
    data = []
    gm = {}
    for c in getattr(normalizer,'mom_cols',[]): gm[c]='momentum'
    for c in getattr(normalizer,'std_cols',[]): gm[c]='std_dev'
    for c in getattr(normalizer,'count_cols',[]): gm[c]='count'
    for c in getattr(normalizer,'order_dir_mean_cols',[]): gm[c]='order_dir'
    for c in getattr(normalizer,'frac_cols',[]): gm[c]='fraction'
    for c in getattr(normalizer,'other_z_cols',[]): gm[c]='misc_z'
    for col in normalizer.predictors_col:
        data.append({'feature':col,'group':gm.get(col,'other'),'fill_median':normalizer.medians.get(col,np.nan),'z_mean':normalizer.z_means.get(col,np.nan),'z_std':normalizer.z_stds.get(col,np.nan),'is_log1p':col in normalizer.log1p_cols,'is_high_missing':col in normalizer.high_missing_cols})
    return pd.DataFrame(data)

print('=== Resume: finishing artifacts ===', flush=True)
manifest = {'v1':{},'v2':{}}

for t_def in range(4):
    print(f'\n--- t_def = {t_def} ---', flush=True)
    v1_ranked = get_configs_ranked('win_model', t_def)
    v2_ranked = get_configs_ranked('win_model_v2', t_def)
    manifest['v1'][f't{t_def}'] = v1_ranked[:V1_TOP_N]
    manifest['v2'][f't{t_def}'] = v2_ranked[:V2_TOP_N]
    print(f'V1: top {min(V1_TOP_N,len(v1_ranked))}/{len(v1_ranked)}, V2: top {min(V2_TOP_N,len(v2_ranked))}/{len(v2_ranked)}', flush=True)

    need_iso = any(not os.path.exists(os.path.join(RES_DIR,f'win_model_v2/t{t_def}/{c}/isotonic_calibrator.pkl')) for c in v2_ranked[:V2_TOP_N])
    need_v2n = not os.path.exists(os.path.join(SAVE_DIR,f'normalization/feature_normalization_params_v2_t{t_def}.parquet'))
    need_v1n = not os.path.exists(os.path.join(SAVE_DIR,f'normalization/feature_normalization_params_v1_t{t_def}.parquet'))

    if not (need_iso or need_v2n or need_v1n):
        print(f'All artifacts exist for t{t_def}, skipping', flush=True)
        continue

    print(f'Loading features (iso={need_iso}, v2n={need_v2n}, v1n={need_v1n})...', flush=True)
    df = load_features_and_prep(t_def)
    predictors_col = get_predictors(df)
    print(f'{len(df)} rows, {len(predictors_col)} predictors', flush=True)

    if need_iso or need_v2n:
        from _03_win_probability_model_v2 import FeatureNormalizer as FNv2
        ind_train = ~df['marketTime_local'].dt.year.isin([2025,2024])
        ind_val = df['marketTime_local'].dt.year == 2024
        norm = FNv2(predictors_col.copy())
        norm.normalize_ins(df.loc[ind_train,:])
        df_val_n = norm.normalize_oos(df.loc[ind_val,:]).dropna(subset=['win'])
        pcols = norm.predictors_col
        for c in pcols:
            if c not in df_val_n.columns: df_val_n[c] = 0.0
        X_val = df_val_n[pcols].fillna(0.0)
        y_val = df_val_n['win'].values
        print(f'Val set: {len(df_val_n)} rows', flush=True)

        for i, cn in enumerate(v2_ranked[:V2_TOP_N]):
            iso_path = os.path.join(RES_DIR,f'win_model_v2/t{t_def}/{cn}/isotonic_calibrator.pkl')
            if os.path.exists(iso_path):
                print(f'  {cn}: exists', flush=True)
                continue
            xgb_p = os.path.join(RES_DIR,f'win_model_v2/t{t_def}/{cn}/xgboost_model.json')
            lgb_p = os.path.join(RES_DIR,f'win_model_v2/t{t_def}/{cn}/lightgbm_model.txt')
            if not os.path.exists(xgb_p):
                print(f'  {cn}: no xgb model', flush=True)
                continue
            try:
                xm = XGBClassifier(); xm.load_model(xgb_p)
                xp = xm.predict_proba(X_val)[:,1]
                if os.path.exists(lgb_p) and HAS_LGBM:
                    lm = lgb.Booster(model_file=lgb_p)
                    lp = lm.predict(X_val)
                    ep = 0.5*xp + 0.5*lp
                else: ep = xp
                iso = IsotonicRegression(y_min=0.001,y_max=0.999,out_of_bounds='clip')
                iso.fit(ep, y_val)
                with open(iso_path,'wb') as f: pickle.dump(iso,f,protocol=4)
                print(f'  {cn}: saved ({i+1}/{min(V2_TOP_N,len(v2_ranked))})', flush=True)
            except Exception as e:
                print(f'  {cn}: ERROR {e}', flush=True)

        if need_v2n:
            nm = extract_norm(norm)
            nd = os.path.join(SAVE_DIR,'normalization'); os.makedirs(nd, exist_ok=True)
            nm.to_parquet(os.path.join(nd, f'feature_normalization_params_v2_t{t_def}.parquet'))
            print(f'Saved V2 norm for t{t_def}', flush=True)

    if need_v1n:
        from _03_win_probability_model_v2 import FeatureNormalizer as FNv2
        v1_preds = [c for c in predictors_col if c not in CROSS_COLS]
        ind_train = df['marketTime_local'].dt.year != 2025
        norm = FNv2(v1_preds.copy())
        norm.normalize_ins(df.loc[ind_train,:])
        nm = extract_norm(norm)
        nd = os.path.join(SAVE_DIR,'normalization'); os.makedirs(nd, exist_ok=True)
        nm.to_parquet(os.path.join(nd, f'feature_normalization_params_v1_t{t_def}.parquet'))
        print(f'Saved V1 norm for t{t_def}', flush=True)

mp = os.path.join(SAVE_DIR, 'manifest.json')
with open(mp,'w') as f: json.dump(manifest, f, indent=2)
print(f'\nManifest saved!', flush=True)
for v in ['v1','v2']:
    for tk, cs in manifest[v].items():
        print(f'  {v} {tk}: {len(cs)} configs', flush=True)
print('Done!', flush=True)
