"""
V1 Win Probability Model WITH dog track record features.
Identical to _02_win_probability_model.py except:
  - Loads augmented features from res/dog_features/
  - Adds dog track record columns to predictors
  - Saves results to res/dog_features/win_model/

Usage: python _06_04_v1_dog.py <task_id>
  task_id encodes (grid_comb_id, t_definition):
    grid_comb_id = task_id // 4
    t_definition = task_id % 4
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss
import psutil
import os
import socket
import itertools

from parameters import Constant
from utils_locals.parser import parse


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
# Hyperparameter grid (same as V1)
# ──────────────────────────────────────────────
GRID_PARAMS = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1],
}

def get_grid_combinations():
    keys = list(GRID_PARAMS.keys())
    vals = [GRID_PARAMS[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


# ──────────────────────────────────────────────
# Feature normalizer (with V2 bug fix for duplicates)
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
        miss_dict = miss.to_dict()  # V2 fix: avoids Series ambiguity with duplicate index
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
    print(f"=== V1 Dog Feature Model ===", flush=True)
    print(f"Task {task_id}: grid_idx={grid_idx}, t_def={t_definition}", flush=True)
    print(f"Hyperparams: {hp}", flush=True)

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

    # ── Market implied probability ──
    df['market_prob'] = df[['best_back_m0', 'best_lay_m0']].mean(axis=1)
    n_before = len(df)
    df = df.dropna(subset=['market_prob', 'best_back_m0', 'best_lay_m0'])
    print(f"Dropped {n_before - len(df)} rows with NaN market_prob", flush=True)

    # ── Define predictors (same as V1 + dog features) ──
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

    # ── ADD DOG FEATURES ──
    dog_cols_available = [c for c in DOG_FEATURE_COLS if c in df.columns]
    predictors_col = predictors_col + dog_cols_available
    print(f"Added {len(dog_cols_available)} dog features: {dog_cols_available}", flush=True)

    # ── Train/test split ──
    oos_year = 2025
    ins_years = df['marketTime_local'].dt.year.unique()
    ins_years = [int(x) for x in ins_years if x != oos_year]
    ind_ins = df['marketTime_local'].dt.year.isin(ins_years)
    ind_oos = df['marketTime_local'].dt.year == oos_year

    print(f"In-sample: {ind_ins.sum()} rows, OOS: {ind_oos.sum()} rows", flush=True)

    # ── Preserve original prices ──
    original_cols = ['best_back_m0', 'best_lay_m0', 'best_back_q_100_m0', 'best_lay_q_100_m0', 'marketBaseRate']
    original_cols = [c for c in original_cols if c in df.columns]
    oos_originals = df.loc[ind_oos, ['file_name', 'id'] + original_cols].copy()

    # ── Normalize ──
    normalizer = FeatureNormalizer(predictors_col)
    df_ins = normalizer.normalize_ins(df.loc[ind_ins, :])
    df_oos = normalizer.normalize_oos(df.loc[ind_oos, :])
    predictors_col = normalizer.predictors_col

    df_ins = df_ins.dropna(subset=['win'])
    df_oos = df_oos.dropna(subset=['win'])

    # ── Train ──
    print(f"Training XGBClassifier with {hp}, {len(predictors_col)} features", flush=True)
    model = XGBClassifier(
        n_estimators=hp['n_estimators'],
        max_depth=hp['max_depth'],
        learning_rate=hp['learning_rate'],
        random_state=12345,
        n_jobs=-1,
        verbosity=0,
        eval_metric='logloss',
        use_label_encoder=False,
    )
    model.fit(df_ins[predictors_col], df_ins['win'])

    # ── Predict ──
    df_oos = df_oos.copy()
    df_oos['model_prob'] = model.predict_proba(df_oos[predictors_col])[:, 1]
    df_oos['edge'] = df_oos['model_prob'] - df_oos['market_prob']

    # ── Evaluate ──
    ll = log_loss(df_oos['win'], df_oos['model_prob'])
    ll_market = log_loss(df_oos['win'], df_oos['market_prob'].clip(0.001, 0.999))
    print(f"Model log-loss: {ll:.6f}, Market log-loss: {ll_market:.6f}", flush=True)

    # ── Save ──
    hp_str = f"ne{hp['n_estimators']}_md{hp['max_depth']}_lr{hp['learning_rate']}"
    save_dir = f'{dog_dir}win_model/t{t_definition}/{hp_str}/'
    os.makedirs(save_dir, exist_ok=True)

    # Restore original prices
    df_oos = df_oos.reset_index(drop=True)
    oos_originals = oos_originals.reset_index(drop=True)
    for c in original_cols:
        df_oos[f'orig_{c}'] = oos_originals[c].values

    save_cols = ['file_name', 'id', 'win', 'model_prob', 'market_prob', 'edge',
                 'orig_best_back_m0', 'orig_best_lay_m0',
                 'marketBaseRate', 'numberOfActiveRunners', 'local_dow', 'runner_position',
                 'marketTime_local']
    save_cols = [c for c in save_cols if c in df_oos.columns]
    df_oos[save_cols].to_parquet(save_dir + 'save_df.parquet')

    importances = pd.DataFrame({
        'feature': predictors_col,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importances.to_parquet(save_dir + 'feature_importances.parquet')

    print(f"Saved to {save_dir}", flush=True)
    print("Done!", flush=True)
