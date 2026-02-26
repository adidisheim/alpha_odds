"""
Save Model Artifacts — run on Spartan to extract and save all artifacts needed for paper trading.

For each V2 config: re-load features, re-fit IsotonicRegression on 2024 val set, save as pickle.
Generate manifest.json listing top-7 V1 / top-15 V2 configs per t_def (ranked by log-loss).
Save FeatureNormalizer params per t_def.

Usage:
    Run on Spartan via srun:
    srun --partition=interactive --time=01:00:00 --cpus-per-task=4 --mem=32G \
        bash -c 'source load_module.sh && python3 save_model_artifacts.py'
"""

import json
import os
import pickle
import socket

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss as sklearn_log_loss
from xgboost import XGBClassifier

# Detect environment
if socket.gethostname() == 'UML-FNQ2JDW1GV':
    RES_DIR = './res/'
else:
    RES_DIR = '/data/projects/punim2039/alpha_odds/res/'

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available", flush=True)

V1_TOP_N = 7
V2_TOP_N = 15
SAVE_DIR = os.path.join(RES_DIR, 'paper_trading_artifacts/')


def extract_normalizer_metadata(normalizer):
    """
    Extract normalization metadata from a fitted FeatureNormalizer.

    Works with both V1 and V2 FeatureNormalizer classes (which lack get_feature_metadata()).
    Produces the same DataFrame schema that SavedNormalizerParams expects.
    """
    data = []
    mom_cols = getattr(normalizer, 'mom_cols', [])
    std_cols = getattr(normalizer, 'std_cols', [])
    count_cols = getattr(normalizer, 'count_cols', [])
    order_dir_cols = getattr(normalizer, 'order_dir_mean_cols', [])
    frac_cols = getattr(normalizer, 'frac_cols', [])
    other_z = getattr(normalizer, 'other_z_cols', [])

    group_map = {}
    for c in mom_cols: group_map[c] = 'momentum'
    for c in std_cols: group_map[c] = 'std_dev'
    for c in count_cols: group_map[c] = 'count'
    for c in order_dir_cols: group_map[c] = 'order_dir'
    for c in frac_cols: group_map[c] = 'fraction'
    for c in other_z: group_map[c] = 'misc_z'

    for col in normalizer.predictors_col:
        data.append({
            'feature': col,
            'group': group_map.get(col, 'other'),
            'fill_median': normalizer.medians.get(col, np.nan),
            'z_mean': normalizer.z_means.get(col, np.nan),
            'z_std': normalizer.z_stds.get(col, np.nan),
            'is_log1p': col in normalizer.log1p_cols,
            'is_high_missing': col in normalizer.high_missing_cols,
        })

    return pd.DataFrame(data)
os.makedirs(SAVE_DIR, exist_ok=True)


def custom_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def get_v1_configs_ranked(t_def):
    """Get V1 model configs ranked by OOS log-loss (best first)."""
    base_dir = os.path.join(RES_DIR, f'win_model/t{t_def}')
    if not os.path.exists(base_dir):
        print(f"V1 dir not found: {base_dir}", flush=True)
        return []

    configs = {}
    for c in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, c, 'save_df.parquet')
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                ll = custom_log_loss(df['win'].values, df['model_prob'].clip(0.001, 0.999).values)
                configs[c] = ll
            except Exception as e:
                print(f"  Error loading V1 {c}: {e}", flush=True)

    ranked = sorted(configs.keys(), key=lambda c: configs[c])
    print(f"V1 t{t_def}: {len(ranked)} configs, best LL={configs[ranked[0]]:.6f}" if ranked else f"V1 t{t_def}: no configs", flush=True)
    return ranked


def get_v2_configs_ranked(t_def):
    """Get V2 model configs ranked by OOS log-loss (best first)."""
    base_dir = os.path.join(RES_DIR, f'win_model_v2/t{t_def}')
    if not os.path.exists(base_dir):
        print(f"V2 dir not found: {base_dir}", flush=True)
        return []

    configs = {}
    for c in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, c, 'save_df.parquet')
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                ll = custom_log_loss(df['win'].values, df['model_prob'].clip(0.001, 0.999).values)
                configs[c] = ll
            except Exception as e:
                print(f"  Error loading V2 {c}: {e}", flush=True)

    ranked = sorted(configs.keys(), key=lambda c: configs[c])
    print(f"V2 t{t_def}: {len(ranked)} configs, best LL={configs[ranked[0]]:.6f}" if ranked else f"V2 t{t_def}: no configs", flush=True)
    return ranked


def refit_and_save_isotonic(t_def, config_name):
    """
    Refit isotonic calibrator for a V2 config using 2024 validation data.

    This mirrors the training code in _03_win_probability_model_v2.py:
      1. Load features
      2. Load XGBoost + LightGBM models
      3. Predict on 2024 val set
      4. Fit IsotonicRegression
      5. Save as pickle
    """
    model_dir = os.path.join(RES_DIR, f'win_model_v2/t{t_def}/{config_name}')
    xgb_path = os.path.join(model_dir, 'xgboost_model.json')
    lgb_path = os.path.join(model_dir, 'lightgbm_model.txt')

    if not os.path.exists(xgb_path):
        print(f"  XGBoost model not found: {xgb_path}", flush=True)
        return False

    # Load models
    xgb_model = XGBClassifier()
    xgb_model.load_model(xgb_path)

    lgb_model = None
    if os.path.exists(lgb_path) and HAS_LGBM:
        lgb_model = lgb.Booster(model_file=lgb_path)

    # Load save_df to get the predictor columns and feature names
    save_df = pd.read_parquet(os.path.join(model_dir, 'save_df.parquet'))

    # Load features for this t_def
    load_dir = os.path.join(RES_DIR, f'features_t{t_def}')
    merged_path = os.path.join(load_dir, 'greyhound_au_features_merged.parquet')
    if os.path.exists(merged_path):
        df = pd.read_parquet(merged_path)
    else:
        df = pd.DataFrame()
        for i in range(10):
            try:
                df = pd.concat([df, pd.read_parquet(os.path.join(load_dir, f'greyhound_au_features_part_{i}.parquet'))], ignore_index=False)
            except Exception:
                pass

    df['win'] = (df['id'] == -1).astype(int)
    df = df.dropna(subset=['best_back_m0', 'best_lay_m0'])

    # Add cross-runner features (same as V2 training)
    g = df.groupby('file_name')
    df['market_prob'] = df[['best_back_m0', 'best_lay_m0']].mean(axis=1)
    df['prob_rank'] = g['market_prob'].rank(method='min', ascending=False)
    df['prob_vs_favorite'] = df['market_prob'] / g['market_prob'].transform('max')
    df['prob_share'] = df['market_prob'] / g['market_prob'].transform('sum')
    df['_prob_share_sq'] = df['prob_share'] ** 2
    df['race_herfindahl'] = df.groupby('file_name')['_prob_share_sq'].transform('sum')
    df.drop(columns=['_prob_share_sq'], inplace=True)
    race_std = g['market_prob'].transform('std').fillna(0)
    df['n_close_runners'] = (race_std < 0.05).astype(int) * (g['market_prob'].transform('count') - 1)
    df['spread_m0'] = (df['best_back_m0'] - df['best_lay_m0']).abs()
    df['spread_rank'] = g['spread_m0'].rank(method='min', ascending=True)
    df['total_qty_m0'] = df['total_back_qty_m0'] + df['total_lay_qty_m0']
    df['volume_rank'] = g['total_qty_m0'].rank(method='min', ascending=False)
    df['avg_mom_3_1'] = df[['best_back_mom_3_1', 'best_lay_mom_3_1']].mean(axis=1)
    df['momentum_rank'] = g['avg_mom_3_1'].rank(method='min', ascending=False)
    df['race_overround'] = g['market_prob'].transform('sum')
    df['is_favorite'] = (df['prob_rank'] == 1).astype(int)
    df['prob_deviation'] = df['market_prob'] - g['market_prob'].transform('mean')
    df['bl_imbalance_m0'] = df['best_bl_imbalance_m0']
    df['bl_imbalance_rank'] = g['bl_imbalance_m0'].rank(method='min', ascending=False)

    # Fraction features
    df['total_qty_m1'] = df[['total_back_qty_m1', 'total_lay_qty_m1']].sum(axis=1)
    df['total_qty_m3'] = df[['total_back_qty_m3', 'total_lay_qty_m3']].sum(axis=1)
    col_todo = ['total_qty_m1', 'total_back_qty_m1', 'total_lay_qty_m1', 'total_qty_m3', 'total_back_qty_m3', 'total_lay_qty_m3']
    for col in col_todo:
        df[col + '_frac'] = df[col] / df.groupby('file_name')[col].transform('sum')
    for col in [x for x in df.columns if x.endswith('_m1_frac')]:
        df[col.replace('_m1', '_mom_3_1')] = df[col] - df[col.replace('_m1', '_m3')]

    # Build predictor list (same as V2 training)
    suffix_available_at_t0 = [
        "_count_2_1", "_count_3_1", "_mean_2_1", "_mean_3_1",
        "_m0", "_mom_2_1", "_mom_3_1",
        "_order_is_back_2_1", "_order_is_back_3_1",
        "_std_2_1", "_std_3_1",
    ]
    predictors_col = [c for c in df.columns if c.endswith(tuple(suffix_available_at_t0))]
    col_frac = [c for c in df.columns if c.endswith('_frac')]
    col_frac_mom = [c for c in df.columns if '_mom_3_1' in c and '_frac' in c.replace('_mom_3_1', '')]
    predictors_col = predictors_col + col_frac_mom + col_frac
    if 'runner_position' in df.columns:
        predictors_col.append('runner_position')
    predictors_col = predictors_col + ['local_dow', 'marketBaseRate', 'numberOfActiveRunners']
    cross_runner_cols = [
        'prob_rank', 'prob_vs_favorite', 'prob_share', 'race_herfindahl',
        'n_close_runners', 'spread_m0', 'spread_rank', 'total_qty_m0',
        'volume_rank', 'avg_mom_3_1', 'momentum_rank', 'race_overround',
        'is_favorite', 'prob_deviation', 'bl_imbalance_rank',
    ]
    predictors_col = predictors_col + cross_runner_cols

    # Deduplicate predictors
    seen = set()
    deduped = []
    for c in predictors_col:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    predictors_col = deduped

    # Split into train/val (same as V2 training: 2024 as val)
    val_year = 2024
    ind_val = df['marketTime_local'].dt.year == val_year
    ind_train = df['marketTime_local'].dt.year.isin([y for y in df['marketTime_local'].dt.year.unique() if y not in [2025, 2024]])

    # Normalize (fit on train, apply to val)
    from _03_win_probability_model_v2 import FeatureNormalizer
    normalizer = FeatureNormalizer(predictors_col.copy())
    df_train = normalizer.normalize_ins(df.loc[ind_train, :])
    df_val = normalizer.normalize_oos(df.loc[ind_val, :])
    predictors_col = normalizer.predictors_col

    df_val = df_val.dropna(subset=['win'])

    # Ensure all predictor columns exist
    for c in predictors_col:
        if c not in df_val.columns:
            df_val[c] = 0.0

    X_val = df_val[predictors_col].fillna(0.0)
    y_val = df_val['win'].values

    # Predict on val set
    xgb_prob = xgb_model.predict_proba(X_val)[:, 1]

    if lgb_model is not None:
        lgb_prob = lgb_model.predict(X_val)
        ensemble_prob = 0.5 * xgb_prob + 0.5 * lgb_prob
    else:
        ensemble_prob = xgb_prob

    # Fit isotonic regression
    iso_reg = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso_reg.fit(ensemble_prob, y_val)

    # Save
    iso_path = os.path.join(model_dir, 'isotonic_calibrator.pkl')
    with open(iso_path, 'wb') as f:
        pickle.dump(iso_reg, f, protocol=4)

    # Also save the normalizer params for this t_def
    norm_meta = extract_normalizer_metadata(normalizer)
    norm_dir = os.path.join(SAVE_DIR, 'normalization')
    os.makedirs(norm_dir, exist_ok=True)
    norm_meta.to_parquet(os.path.join(norm_dir, f'feature_normalization_params_v2_t{t_def}.parquet'))

    print(f"  Saved isotonic for {config_name} and V2 norm params for t{t_def}", flush=True)
    return True


def save_v1_normalization(t_def):
    """Save V1 normalization params for a t_def by re-fitting on train data."""
    load_dir = os.path.join(RES_DIR, f'features_t{t_def}')
    merged_path = os.path.join(load_dir, 'greyhound_au_features_merged.parquet')
    if os.path.exists(merged_path):
        df = pd.read_parquet(merged_path)
    else:
        df = pd.DataFrame()
        for i in range(10):
            try:
                df = pd.concat([df, pd.read_parquet(os.path.join(load_dir, f'greyhound_au_features_part_{i}.parquet'))], ignore_index=False)
            except Exception:
                pass

    df['win'] = (df['id'] == -1).astype(int)
    df = df.dropna(subset=['best_back_m0', 'best_lay_m0'])
    df['market_prob'] = df[['best_back_m0', 'best_lay_m0']].mean(axis=1)

    # Fraction features
    df['total_qty_m1'] = df[['total_back_qty_m1', 'total_lay_qty_m1']].sum(axis=1)
    df['total_qty_m3'] = df[['total_back_qty_m3', 'total_lay_qty_m3']].sum(axis=1)
    col_todo = ['total_qty_m1', 'total_back_qty_m1', 'total_lay_qty_m1', 'total_qty_m3', 'total_back_qty_m3', 'total_lay_qty_m3']
    for col in col_todo:
        df[col + '_frac'] = df[col] / df.groupby('file_name')[col].transform('sum')
    for col in [x for x in df.columns if x.endswith('_m1_frac')]:
        df[col.replace('_m1', '_mom_3_1')] = df[col] - df[col.replace('_m1', '_m3')]

    suffix_available_at_t0 = [
        "_count_2_1", "_count_3_1", "_mean_2_1", "_mean_3_1",
        "_m0", "_mom_2_1", "_mom_3_1",
        "_order_is_back_2_1", "_order_is_back_3_1",
        "_std_2_1", "_std_3_1",
    ]
    predictors_col = [c for c in df.columns if c.endswith(tuple(suffix_available_at_t0))]
    col_frac = [c for c in df.columns if c.endswith('_frac')]
    col_frac_mom = [c for c in df.columns if '_mom_3_1' in c and '_frac' in c.replace('_mom_3_1', '')]
    predictors_col = predictors_col + col_frac_mom + col_frac
    if 'runner_position' in df.columns:
        predictors_col.append('runner_position')
    predictors_col = predictors_col + ['local_dow', 'marketBaseRate', 'numberOfActiveRunners']

    # Use the V1 FeatureNormalizer from _02_win_probability_model.py
    from _02_win_probability_model import FeatureNormalizer
    ind_train = df['marketTime_local'].dt.year != 2025
    normalizer = FeatureNormalizer(predictors_col.copy())
    normalizer.normalize_ins(df.loc[ind_train, :])

    norm_meta = extract_normalizer_metadata(normalizer)
    norm_dir = os.path.join(SAVE_DIR, 'normalization')
    os.makedirs(norm_dir, exist_ok=True)
    norm_meta.to_parquet(os.path.join(norm_dir, f'feature_normalization_params_v1_t{t_def}.parquet'))
    print(f"Saved V1 normalization params for t{t_def}", flush=True)


if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("Saving Paper Trading Artifacts", flush=True)
    print("=" * 60, flush=True)

    manifest = {"v1": {}, "v2": {}}

    for t_def in range(4):
        print(f"\n--- t_def = {t_def} ---", flush=True)

        # V1: rank configs and save normalization
        v1_ranked = get_v1_configs_ranked(t_def)
        manifest["v1"][f"t{t_def}"] = v1_ranked[:V1_TOP_N]

        print(f"Saving V1 normalization for t{t_def}...", flush=True)
        try:
            save_v1_normalization(t_def)
        except Exception as e:
            print(f"  WARNING: V1 norm failed for t{t_def}: {e}", flush=True)

        # V2: rank configs and save isotonic calibrators + normalization
        v2_ranked = get_v2_configs_ranked(t_def)
        manifest["v2"][f"t{t_def}"] = v2_ranked[:V2_TOP_N]

        # Refit and save isotonic for top V2 configs
        # (Only need to do this once per t_def since norm params are shared)
        for i, config_name in enumerate(v2_ranked[:V2_TOP_N]):
            print(f"  Refitting isotonic for V2 {config_name} ({i+1}/{min(V2_TOP_N, len(v2_ranked))})...", flush=True)
            try:
                refit_and_save_isotonic(t_def, config_name)
            except Exception as e:
                print(f"  WARNING: Isotonic refit failed for {config_name}: {e}", flush=True)

    # Save manifest
    manifest_path = os.path.join(SAVE_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}", flush=True)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("Summary:", flush=True)
    for version in ["v1", "v2"]:
        for t_key, configs in manifest[version].items():
            print(f"  {version} {t_key}: {len(configs)} configs", flush=True)
    print("Done!", flush=True)
