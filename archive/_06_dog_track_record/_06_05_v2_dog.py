"""
V2 Win Probability Model WITH dog track record features.
Identical to _03_win_probability_model_v2.py except:
  - Loads augmented features from res/dog_features/
  - Adds dog track record columns to predictors
  - Saves results to res/dog_features/win_model_v2/

Usage: python _06_05_v2_dog.py <task_id>
  task_id encodes (grid_comb_id, t_definition):
    grid_comb_id = task_id // 4
    t_definition = task_id % 4
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import psutil
import os
import socket
import itertools
import json

from parameters import Constant
from utils_locals.parser import parse

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, using XGBoost only.", flush=True)


# ──────────────────────────────────────────────
# Dog feature columns
# ──────────────────────────────────────────────
DOG_FEATURE_COLS = [
    'dog_n_races', 'dog_win_rate', 'dog_win_rate_last5', 'dog_win_rate_last10',
    'dog_avg_market_prob', 'dog_overperformance',
    'dog_days_since_last', 'dog_avg_position',
    'dog_venue_n_races', 'dog_venue_win_rate', 'dog_streak',
]


# ──────────────────────────────────────────────
# Hyperparameter grid (same as V2)
# ──────────────────────────────────────────────
GRID_PARAMS = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
}

def get_grid_combinations():
    keys = list(GRID_PARAMS.keys())
    vals = [GRID_PARAMS[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


# ──────────────────────────────────────────────
# Cross-runner features (same as V2)
# ──────────────────────────────────────────────
def add_cross_runner_features(df):
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


# ──────────────────────────────────────────────
# Feature normalizer (same as V2)
# ──────────────────────────────────────────────
class FeatureNormalizer:
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


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    args = parse()

    # Decode task_id → (grid_comb_id, t_definition)
    task_id = args.a
    grid_combs = get_grid_combinations()
    t_definition = task_id % 4
    grid_idx = task_id // 4

    if grid_idx >= len(grid_combs):
        print(f"Grid index {grid_idx} out of range (max {len(grid_combs)-1}). Exiting.", flush=True)
        exit(0)

    hp = grid_combs[grid_idx]
    print(f"=== V2 Dog Feature Model ===", flush=True)
    print(f"Task {task_id}: grid_idx={grid_idx}, t_def={t_definition}", flush=True)
    print(f"Hyperparams: {hp}", flush=True)
    print(f"LightGBM available: {HAS_LGBM}", flush=True)

    # ── Load augmented features ──
    dog_dir = Constant.RES_DIR + 'dog_features/'
    features_path = dog_dir + f'features_with_dog_t{t_definition}.parquet'
    print(f"Loading features from {features_path}", flush=True)
    df = pd.read_parquet(features_path)

    process = psutil.Process(os.getpid())
    print(f"RAM after load: {process.memory_info().rss / 1024**3:.2f} GB", flush=True)
    print(f"Data shape: {df.shape}", flush=True)

    # ── Define target ──
    df['win'] = (df['id'] == -1).astype(int)

    # ── Drop rows with missing order book ──
    n_before = len(df)
    df = df.dropna(subset=['best_back_m0', 'best_lay_m0'])
    print(f"Dropped {n_before - len(df)} rows with NaN best_back/lay_m0", flush=True)

    # ── Add cross-runner features ──
    df, cross_runner_cols = add_cross_runner_features(df)

    # ── Define predictors ──
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
    col_todo = ['total_qty_m1','total_back_qty_m1','total_lay_qty_m1','total_qty_m3','total_back_qty_m3','total_lay_qty_m3']
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
    predictors_col = predictors_col + fixed_effect_columns
    predictors_col = predictors_col + cross_runner_cols

    # ── ADD DOG FEATURES ──
    dog_cols_available = [c for c in DOG_FEATURE_COLS if c in df.columns]
    predictors_col = predictors_col + dog_cols_available
    print(f"Added {len(dog_cols_available)} dog features", flush=True)

    # ── Train/Val/Test split ──
    oos_year = 2025
    val_year = 2024
    train_years = df['marketTime_local'].dt.year.unique()
    train_years = [int(x) for x in train_years if x not in [oos_year, val_year]]

    ind_train = df['marketTime_local'].dt.year.isin(train_years)
    ind_val = df['marketTime_local'].dt.year == val_year
    ind_oos = df['marketTime_local'].dt.year == oos_year

    print(f"Train: {ind_train.sum()}, Val: {ind_val.sum()}, OOS: {ind_oos.sum()}", flush=True)

    # ── Preserve original prices ──
    original_cols = ['best_back_m0', 'best_lay_m0', 'best_back_q_100_m0', 'best_lay_q_100_m0', 'marketBaseRate']
    original_cols = [c for c in original_cols if c in df.columns]
    oos_originals = df.loc[ind_oos, original_cols].copy()

    # ── Normalize ──
    normalizer = FeatureNormalizer(predictors_col)
    df_train = normalizer.normalize_ins(df.loc[ind_train, :])
    df_val = normalizer.normalize_oos(df.loc[ind_val, :])
    df_oos = normalizer.normalize_oos(df.loc[ind_oos, :])
    predictors_col = normalizer.predictors_col

    df_train = df_train.dropna(subset=['win'])
    df_val = df_val.dropna(subset=['win'])
    df_oos = df_oos.dropna(subset=['win'])

    print(f"Features: {len(predictors_col)}", flush=True)

    # ── XGBoost ──
    print(f"Training XGBClassifier...", flush=True)
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
    print(f"XGBoost best iteration: {xgb_model.best_iteration}", flush=True)

    df_oos = df_oos.copy()
    df_oos['xgb_prob'] = xgb_model.predict_proba(df_oos[predictors_col])[:, 1]
    df_val = df_val.copy()
    df_val['xgb_prob'] = xgb_model.predict_proba(df_val[predictors_col])[:, 1]

    # ── LightGBM ──
    if HAS_LGBM:
        print(f"Training LightGBM...", flush=True)
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
        print(f"LightGBM best iteration: {lgb_model.best_iteration_}", flush=True)

        df_oos['lgb_prob'] = lgb_model.predict_proba(df_oos[predictors_col])[:, 1]
        df_val['lgb_prob'] = lgb_model.predict_proba(df_val[predictors_col])[:, 1]
        df_oos['ensemble_prob'] = 0.5 * df_oos['xgb_prob'] + 0.5 * df_oos['lgb_prob']
        df_val['ensemble_prob'] = 0.5 * df_val['xgb_prob'] + 0.5 * df_val['lgb_prob']
    else:
        df_oos['ensemble_prob'] = df_oos['xgb_prob']
        df_val['ensemble_prob'] = df_val['xgb_prob']

    # ── Isotonic calibration ──
    iso_reg = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso_reg.fit(df_val['ensemble_prob'].values, df_val['win'].values)
    df_oos['calibrated_prob'] = iso_reg.predict(df_oos['ensemble_prob'].values)
    df_oos['model_prob'] = df_oos['calibrated_prob']
    df_oos['edge'] = df_oos['model_prob'] - df_oos['market_prob']

    # ── Evaluate ──
    ll = log_loss(df_oos['win'], df_oos['model_prob'])
    ll_market = log_loss(df_oos['win'], df_oos['market_prob'].clip(0.001, 0.999))
    print(f"Calibrated log-loss: {ll:.6f}, Market log-loss: {ll_market:.6f}", flush=True)

    # ── Save ──
    hp_str = f"ne{hp['n_estimators']}_md{hp['max_depth']}_lr{hp['learning_rate']}"
    save_dir = f'{dog_dir}win_model_v2/t{t_definition}/{hp_str}/'
    os.makedirs(save_dir, exist_ok=True)

    df_oos['orig_best_back_m0'] = oos_originals['best_back_m0'].values
    df_oos['orig_best_lay_m0'] = oos_originals['best_lay_m0'].values
    df_oos['orig_marketBaseRate'] = oos_originals['marketBaseRate'].values

    save_cols = ['file_name', 'id', 'win', 'model_prob', 'market_prob', 'edge',
                 'xgb_prob', 'calibrated_prob',
                 'orig_best_back_m0', 'orig_best_lay_m0',
                 'marketBaseRate', 'numberOfActiveRunners', 'local_dow', 'runner_position',
                 'marketTime_local']
    if HAS_LGBM:
        save_cols.insert(7, 'lgb_prob')
        save_cols.insert(8, 'ensemble_prob')
    save_cols = [c for c in save_cols if c in df_oos.columns]
    df_oos[save_cols].to_parquet(save_dir + 'save_df.parquet')

    importances = pd.DataFrame({
        'feature': predictors_col,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    importances.to_parquet(save_dir + 'feature_importances.parquet')

    metrics = {
        'hp': hp, 't_definition': t_definition,
        'Calibrated_logloss': float(ll),
        'Market_logloss': float(ll_market),
        'has_lgbm': HAS_LGBM,
    }
    with open(save_dir + 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {save_dir}", flush=True)
    print("Done!", flush=True)
