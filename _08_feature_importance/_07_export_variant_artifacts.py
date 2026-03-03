"""
Export model artifacts for feature-restricted V2 variants (n5, n30).

For each (n_feat, t_def): retrains the top-15 V2 models (ranked by log-loss from
existing full_retrain results), saves XGBoost, LightGBM, isotonic artifacts.

Array job: task_id = 0..7  (2 feature counts × 4 t_defs)
  feat_idx = task_id // 4   (0=n5, 1=n30)
  t_def    = task_id % 4

Each task trains 15 models sequentially (~75 min total).

Output directory: res/paper_trading_artifacts/v2_n{N}/t{T}/{config}/
  - xgboost_model.json
  - lightgbm_model.txt
  - isotonic_calibrator.pkl

Also saves normalization params:
  res/paper_trading_artifacts/normalization/feature_normalization_params_v2_n{N}_t{T}.parquet

Usage:
    python _07_export_variant_artifacts.py <task_id>
"""

import json
import os
import pickle
import itertools

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from parameters import Constant
from utils_locals.parser import parse

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, using XGBoost only.", flush=True)


FEATURE_COUNTS = [5, 30]
T_DEFS = [0, 1, 2, 3]
TOP_N = 15

GRID_PARAMS = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
}


def get_grid_combinations():
    keys = list(GRID_PARAMS.keys())
    vals = [GRID_PARAMS[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def custom_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def get_top_configs(n_feat, t_def, top_n=TOP_N):
    """Get top-N configs ranked by log-loss from existing full_retrain metrics."""
    retrain_dir = f'{Constant.RES_DIR}/feature_importance_study/full_retrain/n{n_feat}/t{t_def}'
    if not os.path.isdir(retrain_dir):
        print(f"  No retrain dir: {retrain_dir}", flush=True)
        return []

    configs = {}
    for config_name in sorted(os.listdir(retrain_dir)):
        metrics_path = os.path.join(retrain_dir, config_name, 'metrics.json')
        if not os.path.exists(metrics_path):
            continue
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            ll = metrics.get('Calibrated_logloss', metrics.get('XGBoost_logloss', 999))
            configs[config_name] = ll
        except Exception:
            pass

    ranked = sorted(configs.keys(), key=lambda c: configs[c])
    print(f"  Found {len(ranked)} configs, best LL={configs[ranked[0]]:.6f}", flush=True)
    return ranked[:top_n]


def add_cross_runner_features(df):
    """Exact copy from _05_full_retrain.py / V2 pipeline."""
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
    cross_runner_cols = [
        'prob_rank', 'prob_vs_favorite', 'prob_share', 'race_herfindahl',
        'n_close_runners', 'spread_m0', 'spread_rank', 'total_qty_m0',
        'volume_rank', 'avg_mom_3_1', 'momentum_rank', 'race_overround',
        'is_favorite', 'prob_deviation', 'bl_imbalance_rank',
    ]
    return df, cross_runner_cols


class FeatureNormalizer:
    """Exact copy from _05_full_retrain.py."""
    def __init__(self, predictors_col):
        seen = set()
        deduped = []
        for c in predictors_col:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        self.predictors_col = deduped
        self.high_missing_cols = []
        self.medians = {}
        self.z_means = {}
        self.z_stds = {}
        self.log1p_cols = set()
        self.mom_cols = []
        self.std_cols = []
        self.count_cols = []
        self.order_dir_mean_cols = []
        self.frac_cols = []
        self.other_z_cols = ['marketBaseRate', 'numberOfActiveRunners', 'local_dow']
        self.fitted = False

    def _detect_groups(self):
        cols = self.predictors_col
        self.mom_cols = [c for c in cols if "_mom_" in c]
        self.std_cols = [c for c in cols if "_std_" in c]
        self.count_cols = [c for c in cols if c.startswith("qty_count_")]
        self.order_dir_mean_cols = [c for c in cols if c.startswith("order_is_back_order_is_back_") and c not in self.std_cols]
        self.frac_cols = [c for c in cols if c.endswith("_frac")]
        self.other_z_cols = self.other_z_cols + [c for c in cols if c.endswith("_m0")]

    @staticmethod
    def _zscore_col(series, mean, std):
        if std == 0 or np.isnan(std):
            std = 1.0
        return (series - mean) / std

    def normalize_ins(self, df):
        df = df.copy()
        miss = df[self.predictors_col].isna().mean()
        miss_dict = miss.to_dict()
        self.high_missing_cols = [c for c in self.predictors_col if miss_dict.get(c, 0.0) > 0.5]
        for c in self.high_missing_cols:
            if c not in df.columns: continue
            ind_col = f"{c}_missing"
            df[ind_col] = df[c].isna().astype("int8")
            if ind_col not in self.predictors_col:
                self.predictors_col.append(ind_col)
            df[c] = df[c].fillna(0)
        for c in self.predictors_col:
            if c not in df.columns: continue
            if df[c].isna().any():
                med = df[c].median()
                self.medians[c] = med
                df[c] = df[c].fillna(med)
            else:
                self.medians[c] = df[c].median()
        self._detect_groups()
        z_cols = set(self.mom_cols + self.std_cols + self.count_cols + self.order_dir_mean_cols + self.frac_cols + self.other_z_cols)
        self.log1p_cols = set(self.std_cols)
        for c in self.log1p_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c].clip(lower=0))
        for c in z_cols:
            if c not in df.columns: continue
            mean_c = df[c].mean()
            std_c = df[c].std(ddof=0)
            self.z_means[c] = mean_c
            self.z_stds[c] = std_c
            df[c] = self._zscore_col(df[c], mean_c, std_c)
        self.fitted = True
        return df

    def normalize_oos(self, df):
        if not self.fitted:
            raise RuntimeError("Fit first.")
        df = df.copy()
        for c in self.predictors_col:
            if c not in df.columns:
                df[c] = np.nan
        for c in self.high_missing_cols:
            if c in df.columns:
                ind_col = f"{c}_missing"
                df[ind_col] = df[c].isna().astype("int8")
                df[c] = df[c].fillna(0)
        for c in self.predictors_col:
            if c not in df.columns: continue
            if df[c].isna().any():
                med = self.medians.get(c, 0.0)
                df[c] = df[c].fillna(med)
        for c in self.log1p_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c].clip(lower=0))
        for c in self.z_means.keys():
            if c not in df.columns: continue
            df[c] = self._zscore_col(df[c], self.z_means[c], self.z_stds[c])
        return df


def extract_normalizer_metadata(normalizer):
    """Extract normalization metadata for SavedNormalizerParams."""
    data = []
    group_map = {}
    for c in getattr(normalizer, 'mom_cols', []): group_map[c] = 'momentum'
    for c in getattr(normalizer, 'std_cols', []): group_map[c] = 'std_dev'
    for c in getattr(normalizer, 'count_cols', []): group_map[c] = 'count'
    for c in getattr(normalizer, 'order_dir_mean_cols', []): group_map[c] = 'order_dir'
    for c in getattr(normalizer, 'frac_cols', []): group_map[c] = 'fraction'
    for c in getattr(normalizer, 'other_z_cols', []): group_map[c] = 'misc_z'

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


if __name__ == '__main__':
    args = parse()
    task_id = args.a

    feat_idx = task_id // 4
    t_definition = task_id % 4

    n_features_target = FEATURE_COUNTS[feat_idx]
    v2_key = f"v2_n{n_features_target}"

    print(f"=== Export Variant Artifacts: n{n_features_target}, t_def={t_definition} ===", flush=True)
    print(f"Task {task_id}: feat_idx={feat_idx}, t_def={t_definition}", flush=True)
    print(f"LightGBM: {HAS_LGBM}", flush=True)

    # ── Find top-15 configs from existing retrain results ──
    top_configs = get_top_configs(n_features_target, t_definition, TOP_N)
    if not top_configs:
        print("ERROR: No configs found. Exiting.", flush=True)
        exit(1)
    print(f"Will retrain {len(top_configs)} configs: {top_configs}", flush=True)

    # ── Load feature ranking ──
    ranking_path = f'{Constant.RES_DIR}/feature_importance_study/feature_ranking.parquet'
    ranking = pd.read_parquet(ranking_path)
    all_ranked_features = ranking['feature'].tolist()

    # ── Load features ──
    load_dir = f'{Constant.RES_DIR}/features_t{t_definition}'
    merged_path = f'{load_dir}/greyhound_au_features_merged.parquet'
    if os.path.exists(merged_path):
        df = pd.read_parquet(merged_path)
    else:
        df = pd.DataFrame()
        for i in range(10):
            try:
                df = pd.concat([df, pd.read_parquet(f'{load_dir}/greyhound_au_features_part_{i}.parquet')], ignore_index=False)
            except Exception as e:
                print(f'  Part {i}: {e}', flush=True)

    print(f"Data shape: {df.shape}", flush=True)
    df['win'] = (df['id'] == -1).astype(int)
    df = df.dropna(subset=['best_back_m0', 'best_lay_m0'])

    # ── Cross-runner features ──
    df, cross_runner_cols = add_cross_runner_features(df)

    # ── Build full predictor list (same as V2) ──
    suffix_available_at_t0 = [
        "_count_2_1", "_count_3_1", "_mean_2_1", "_mean_3_1",
        "_m0", "_mom_2_1", "_mom_3_1",
        "_order_is_back_2_1", "_order_is_back_3_1",
        "_std_2_1", "_std_3_1"
    ]
    predictors_col = [c for c in df.columns if c.endswith(tuple(suffix_available_at_t0))]

    # Fraction features
    df['total_qty_m1'] = df[['total_back_qty_m1', 'total_lay_qty_m1']].sum(axis=1)
    df['total_qty_m3'] = df[['total_back_qty_m3', 'total_lay_qty_m3']].sum(axis=1)
    col_todo = ['total_qty_m1', 'total_back_qty_m1', 'total_lay_qty_m1',
                'total_qty_m3', 'total_back_qty_m3', 'total_lay_qty_m3']
    col_frac = []
    for col in col_todo:
        c = col + '_frac'
        df[c] = df[col] / df.groupby('file_name')[col].transform('sum')
        col_frac.append(c)
    col_frac_mom = []
    for col in [x for x in col_frac if x.endswith('_m1_frac')]:
        c = col.replace('_m1', '_mom_3_1')
        df[c] = df[col] - df[col.replace('_m1', '_m3')]
        col_frac_mom.append(c)
    predictors_col = predictors_col + col_frac_mom + col_frac

    if 'runner_position' in df.columns:
        predictors_col.append('runner_position')

    fixed_effect_columns = ['local_dow', 'marketBaseRate', 'numberOfActiveRunners']
    predictors_col = predictors_col + fixed_effect_columns + cross_runner_cols

    # Deduplicate
    seen = set()
    deduped = []
    for c in predictors_col:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    predictors_col = deduped

    # ── Train/Val split ──
    oos_year = 2025
    val_year = 2024
    train_years = [int(x) for x in df['marketTime_local'].dt.year.unique() if x not in [oos_year, val_year]]

    ind_train = df['marketTime_local'].dt.year.isin(train_years)
    ind_val = df['marketTime_local'].dt.year == val_year

    print(f"Train: {ind_train.sum()}, Val: {ind_val.sum()}", flush=True)

    # ── Normalize ALL features first (same as V2) ──
    # This ensures models are trained on the same normalization the live bot uses.
    all_predictors = list(predictors_col)  # full 105-feature list
    normalizer = FeatureNormalizer(all_predictors)
    df_train = normalizer.normalize_ins(df.loc[ind_train, :])
    df_val = normalizer.normalize_oos(df.loc[ind_val, :])
    all_predictors = normalizer.predictors_col  # may include _missing indicators

    df_train = df_train.dropna(subset=['win'])
    df_val = df_val.dropna(subset=['win'])

    print(f"Normalized with {len(all_predictors)} features (same as V2)", flush=True)

    # ── NOW restrict to top-N features for training ──
    ranked_available = [f for f in all_ranked_features if f in all_predictors]
    n_to_keep = min(n_features_target, len(ranked_available))
    selected_features = ranked_available[:n_to_keep]
    for essential in ['best_back_m0', 'best_lay_m0']:
        if essential not in selected_features and essential in all_predictors:
            selected_features.append(essential)
    predictors_col = selected_features
    print(f"Training with {len(predictors_col)} features (subset of normalized)", flush=True)

    save_base = f'{Constant.RES_DIR}/paper_trading_artifacts'

    # ── Parse grid combinations ──
    grid_combs = get_grid_combinations()
    hp_lookup = {}
    for combo in grid_combs:
        hp_str = f"ne{combo['n_estimators']}_md{combo['max_depth']}_lr{combo['learning_rate']}"
        hp_lookup[hp_str] = combo

    # ── Train top-15 models ──
    trained_configs = []
    for i, config_name in enumerate(top_configs):
        print(f"\n--- Training model {i+1}/{len(top_configs)}: {config_name} ---", flush=True)

        hp = hp_lookup.get(config_name)
        if hp is None:
            print(f"  WARNING: Could not find HP for {config_name}, skipping", flush=True)
            continue

        # XGBoost
        xgb_model = XGBClassifier(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            learning_rate=hp['learning_rate'],
            random_state=12345,
            n_jobs=-1,
            verbosity=0,
            eval_metric='logloss',
            early_stopping_rounds=50,
        )
        xgb_model.fit(
            df_train[predictors_col], df_train['win'],
            eval_set=[(df_val[predictors_col], df_val['win'])],
            verbose=False,
        )
        print(f"  XGB best iter: {xgb_model.best_iteration}/{hp['n_estimators']}", flush=True)

        # LightGBM
        lgb_model = None
        if HAS_LGBM:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=hp['n_estimators'],
                max_depth=hp['max_depth'],
                learning_rate=hp['learning_rate'],
                random_state=12345,
                n_jobs=-1,
                verbose=-1,
                num_leaves=min(2**hp['max_depth'] - 1, 127),
                min_child_samples=50,
            )
            lgb_model.fit(
                df_train[predictors_col], df_train['win'],
                eval_set=[(df_val[predictors_col], df_val['win'])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            print(f"  LGB best iter: {lgb_model.best_iteration_}/{hp['n_estimators']}", flush=True)

        # Isotonic calibration on val set
        xgb_val_prob = xgb_model.predict_proba(df_val[predictors_col])[:, 1]
        if lgb_model is not None:
            lgb_val_prob = lgb_model.predict_proba(df_val[predictors_col])[:, 1]
            ens_val_prob = 0.5 * xgb_val_prob + 0.5 * lgb_val_prob
        else:
            ens_val_prob = xgb_val_prob

        iso_reg = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        iso_reg.fit(ens_val_prob, df_val['win'].values)

        # Save artifacts
        artifact_dir = os.path.join(save_base, v2_key, f't{t_definition}', config_name)
        os.makedirs(artifact_dir, exist_ok=True)

        xgb_model.save_model(os.path.join(artifact_dir, 'xgboost_model.json'))

        if lgb_model is not None:
            lgb_model.booster_.save_model(os.path.join(artifact_dir, 'lightgbm_model.txt'))

        with open(os.path.join(artifact_dir, 'isotonic_calibrator.pkl'), 'wb') as f:
            pickle.dump(iso_reg, f, protocol=4)

        trained_configs.append(config_name)
        print(f"  Saved to {artifact_dir}", flush=True)

    print(f"\n=== Done: trained {len(trained_configs)}/{len(top_configs)} models ===", flush=True)
    print(f"Configs: {trained_configs}", flush=True)

    # Save this task's config list for manifest building
    task_info = {
        'v2_key': v2_key,
        't_def': int(t_definition),
        'n_features': int(n_features_target),
        'configs': trained_configs,
    }
    info_path = os.path.join(save_base, f'export_task_{task_id}.json')
    with open(info_path, 'w') as f:
        json.dump(task_info, f, indent=2)
    print(f"Task info saved to {info_path}", flush=True)
