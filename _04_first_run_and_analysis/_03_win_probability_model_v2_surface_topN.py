"""
Win Probability Model V2 — Surface features only, top-N feature selection.

Removes q_100/q_1000 depth features, then selects top N features by importance
from a quick initial XGBoost fit on the training data.

Usage:
    python _03_win_probability_model_v2_surface_topN.py <grid_comb_id> <t_definition> <n_features>

    grid_comb_id: index into the hyperparameter grid (0-based)
    t_definition: which time snapshot features to use (0, 1, 2, or 3)
    n_features: number of top features to keep (5, 10, or 30)

Output saved to: res/win_model_v2_surface_n{n_features}/t{t_definition}/{hp_str}/
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
# Hyperparameter grid
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
# Cross-runner features
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
# Feature normalizer
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
    t_definition = args.b
    n_features = int(args.c)
    grid_combs = get_grid_combinations()
    if args.a >= len(grid_combs):
        print(f"Grid index {args.a} out of range (max {len(grid_combs)-1}). Exiting.", flush=True)
        exit(0)

    hp = grid_combs[args.a]
    print(f"=== Win Probability Model V2 — Surface Top-{n_features} ===", flush=True)
    print(f"Grid combo {args.a}/{len(grid_combs)}: {hp}", flush=True)
    print(f"t_definition: {t_definition}, n_features: {n_features}", flush=True)
    print(f"LightGBM available: {HAS_LGBM}", flush=True)

    # ── Load features ──
    load_dir = f'{Constant.RES_DIR}/features_t{t_definition}'
    merged_path = f'{load_dir}/greyhound_au_features_merged.parquet'
    if os.path.exists(merged_path):
        print(f"Loading merged features from {merged_path}", flush=True)
        df = pd.read_parquet(merged_path)
    else:
        print(f"Merged file not found, loading parts...", flush=True)
        df = pd.DataFrame()
        for i in range(10):
            try:
                df = pd.concat([df, pd.read_parquet(f'{load_dir}/greyhound_au_features_part_{i}.parquet')], ignore_index=False)
            except Exception as e:
                print(f'Non-fatal ERROR: Could not read part {i}: {e}', flush=True)

    if socket.gethostname() == 'UML-FNQ2JDW1GV':
        df = df.dropna(subset='file_name').sample(frac=1, random_state=42).head(20000)

    process = psutil.Process(os.getpid())
    print(f"RAM after load: {process.memory_info().rss / 1024**3:.2f} GB", flush=True)
    print(f"Data shape: {df.shape}", flush=True)

    df['win'] = (df['id'] == -1).astype(int)

    n_before = len(df)
    df = df.dropna(subset=['best_back_m0', 'best_lay_m0'])
    print(f"Dropped {n_before - len(df)} rows with NaN best_back/lay_m0 ({(n_before - len(df))/n_before*100:.2f}%)", flush=True)

    print("Adding cross-runner features...", flush=True)
    df, cross_runner_cols = add_cross_runner_features(df)
    print(f"Added {len(cross_runner_cols)} cross-runner features.", flush=True)

    # ── Define all surface predictors ──
    suffix_available_at_t0 = [
        "_count_2_1", "_count_3_1", "_mean_2_1", "_mean_3_1",
        "_m0", "_mom_2_1", "_mom_3_1",
        "_order_is_back_2_1", "_order_is_back_3_1",
        "_std_2_1", "_std_3_1"
    ]
    all_predictors = [c for c in df.columns if c.endswith(tuple(suffix_available_at_t0))]

    # Remove q_100 and q_1000 depth features
    n_before_filter = len(all_predictors)
    all_predictors = [c for c in all_predictors if 'q_100' not in c and 'q_1000' not in c]
    print(f"Surface filter: removed {n_before_filter - len(all_predictors)} depth features, keeping {len(all_predictors)}", flush=True)

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
    all_predictors = all_predictors + col_frac_mom + col_frac

    if 'runner_position' in df.columns:
        all_predictors.append('runner_position')

    fixed_effect_columns = ['local_dow', 'marketBaseRate', 'numberOfActiveRunners']
    all_predictors = all_predictors + fixed_effect_columns
    all_predictors = all_predictors + cross_runner_cols

    # ── Train/Val/Test split ──
    oos_year = 2025
    val_year = 2024
    train_years = df['marketTime_local'].dt.year.unique()
    train_years = [int(x) for x in train_years if x not in [oos_year, val_year]]

    ind_train = df['marketTime_local'].dt.year.isin(train_years)
    ind_val = df['marketTime_local'].dt.year == val_year
    ind_oos = df['marketTime_local'].dt.year == oos_year

    # ── Preserve original prices ──
    original_cols = ['best_back_m0', 'best_lay_m0', 'marketBaseRate']
    original_cols = [c for c in original_cols if c in df.columns]
    oos_originals = df.loc[ind_oos, original_cols].copy()

    # ── Normalize ALL surface features first (needed for ranking) ──
    normalizer_all = FeatureNormalizer(list(all_predictors))
    df_train_all = normalizer_all.normalize_ins(df.loc[ind_train, :])
    df_val_all = normalizer_all.normalize_oos(df.loc[ind_val, :])
    df_oos_all = normalizer_all.normalize_oos(df.loc[ind_oos, :])
    all_predictors_normalized = normalizer_all.predictors_col

    df_train_all = df_train_all.dropna(subset=['win'])
    df_val_all = df_val_all.dropna(subset=['win'])
    df_oos_all = df_oos_all.dropna(subset=['win'])

    print(f"\nAll surface features after normalization: {len(all_predictors_normalized)}", flush=True)

    # ══════════════════════════════════════════════
    # Quick ranking model to determine top-N features
    # ══════════════════════════════════════════════
    print(f"\n=== Fitting ranking model (XGB ne=500 md=4 lr=0.05) ===", flush=True)
    ranking_model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        random_state=12345, n_jobs=-1, verbosity=0,
        eval_metric='logloss', early_stopping_rounds=50,
    )
    ranking_model.fit(
        df_train_all[all_predictors_normalized], df_train_all['win'],
        eval_set=[(df_val_all[all_predictors_normalized], df_val_all['win'])],
        verbose=False,
    )
    importances_ranking = pd.DataFrame({
        'feature': all_predictors_normalized,
        'importance': ranking_model.feature_importances_,
    }).sort_values('importance', ascending=False)

    # Select top-N features
    top_features = importances_ranking.head(n_features)['feature'].tolist()
    print(f"\nTop {n_features} features:", flush=True)
    for i, row in importances_ranking.head(n_features).iterrows():
        print(f"  {row['importance']:.4f}  {row['feature']}", flush=True)

    # Use only top-N features from here on
    predictors_col = top_features

    # Re-normalize with only the selected features (cleaner z-scores)
    normalizer = FeatureNormalizer(list(predictors_col))
    df_train = normalizer.normalize_ins(df.loc[ind_train, :])
    df_val = normalizer.normalize_oos(df.loc[ind_val, :])
    df_oos = normalizer.normalize_oos(df.loc[ind_oos, :])
    predictors_col = normalizer.predictors_col

    df_train = df_train.dropna(subset=['win'])
    df_val = df_val.dropna(subset=['win'])
    df_oos = df_oos.dropna(subset=['win'])

    print(f"\nFinal features: {len(predictors_col)}", flush=True)
    print(f"Train: {len(df_train):,}, Val: {len(df_val):,}, OOS: {len(df_oos):,}", flush=True)

    # ══════════════════════════════════════════════
    # Model 1: XGBoost
    # ══════════════════════════════════════════════
    print(f"\n=== Training XGBClassifier with {hp} ===", flush=True)
    xgb_model = XGBClassifier(
        n_estimators=hp['n_estimators'],
        max_depth=hp['max_depth'],
        learning_rate=hp['learning_rate'],
        random_state=12345, n_jobs=-1, verbosity=0,
        eval_metric='logloss', early_stopping_rounds=50,
    )
    xgb_model.fit(
        df_train[predictors_col], df_train['win'],
        eval_set=[(df_val[predictors_col], df_val['win'])],
        verbose=False,
    )
    xgb_best_iter = xgb_model.best_iteration
    print(f"XGBoost best iteration: {xgb_best_iter} / {hp['n_estimators']}", flush=True)

    df_oos = df_oos.copy()
    df_oos['xgb_prob'] = xgb_model.predict_proba(df_oos[predictors_col])[:, 1]
    df_val = df_val.copy()
    df_val['xgb_prob'] = xgb_model.predict_proba(df_val[predictors_col])[:, 1]

    # ══════════════════════════════════════════════
    # Model 2: LightGBM
    # ══════════════════════════════════════════════
    if HAS_LGBM:
        print(f"\n=== Training LightGBM ===", flush=True)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            learning_rate=hp['learning_rate'],
            random_state=12345, n_jobs=-1, verbose=-1,
            num_leaves=min(2**hp['max_depth'] - 1, 127),
            min_child_samples=50,
        )
        lgb_model.fit(
            df_train[predictors_col], df_train['win'],
            eval_set=[(df_val[predictors_col], df_val['win'])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        lgb_best_iter = lgb_model.best_iteration_
        print(f"LightGBM best iteration: {lgb_best_iter} / {hp['n_estimators']}", flush=True)

        df_oos['lgb_prob'] = lgb_model.predict_proba(df_oos[predictors_col])[:, 1]
        df_val['lgb_prob'] = lgb_model.predict_proba(df_val[predictors_col])[:, 1]

        df_oos['ensemble_prob'] = 0.5 * df_oos['xgb_prob'] + 0.5 * df_oos['lgb_prob']
        df_val['ensemble_prob'] = 0.5 * df_val['xgb_prob'] + 0.5 * df_val['lgb_prob']
    else:
        df_oos['ensemble_prob'] = df_oos['xgb_prob']
        df_val['ensemble_prob'] = df_val['xgb_prob']

    # ══════════════════════════════════════════════
    # Calibration
    # ══════════════════════════════════════════════
    print(f"\n=== Calibrating with isotonic regression ===", flush=True)
    iso_reg = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso_reg.fit(df_val['ensemble_prob'].values, df_val['win'].values)
    df_oos['calibrated_prob'] = iso_reg.predict(df_oos['ensemble_prob'].values)
    df_oos['model_prob'] = df_oos['calibrated_prob']
    df_oos['edge'] = df_oos['model_prob'] - df_oos['market_prob']

    # ══════════════════════════════════════════════
    # Evaluation
    # ══════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print(f"=== OOS Evaluation (t_def={t_definition}, n_feat={n_features}, hp={hp}) ===", flush=True)
    print(f"{'='*60}", flush=True)

    models_to_eval = {
        'Market': df_oos['market_prob'].clip(0.001, 0.999),
        'XGBoost': df_oos['xgb_prob'],
        'Calibrated': df_oos['calibrated_prob'],
    }
    if HAS_LGBM:
        models_to_eval['LightGBM'] = df_oos['lgb_prob']
        models_to_eval['Ensemble'] = df_oos['ensemble_prob']

    for name, probs in models_to_eval.items():
        ll = log_loss(df_oos['win'], probs)
        bs = brier_score_loss(df_oos['win'], probs)
        print(f"  {name:12s}  Log-loss: {ll:.6f}  Brier: {bs:.6f}", flush=True)

    # ── Value betting simulation ──
    df_oos['orig_best_back_m0'] = oos_originals['best_back_m0'].values
    df_oos['orig_best_lay_m0'] = oos_originals['best_lay_m0'].values
    df_oos['orig_marketBaseRate'] = oos_originals['marketBaseRate'].values

    commission_rate = df_oos['orig_marketBaseRate'].median() / 100

    bet_results = []
    for edge_threshold in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]:
        bets = df_oos[df_oos['edge'] > edge_threshold].copy()
        if len(bets) == 0:
            continue
        bets['back_odds'] = 1 / bets['orig_best_back_m0']
        bets = bets[(bets['back_odds'] > 1.01) & (bets['back_odds'] < 1000)]
        if len(bets) == 0:
            continue
        bets['pnl'] = bets['win'] * (bets['back_odds'] - 1) * (1 - commission_rate) - (1 - bets['win'])
        n_bets = len(bets)
        total_pnl = bets['pnl'].sum()
        avg_pnl = bets['pnl'].mean()
        win_rate = bets['win'].mean()
        avg_odds = bets['back_odds'].mean()
        pnl_std = bets['pnl'].std()
        sharpe = avg_pnl / pnl_std * np.sqrt(n_bets) if pnl_std > 0 else 0

        print(f"\nEdge>{edge_threshold}: {n_bets} bets, WR={win_rate:.4f}, ROI={avg_pnl*100:+.2f}%, z={sharpe:.2f}", flush=True)

        bet_results.append({
            'edge_threshold': edge_threshold,
            'n_bets': n_bets, 'win_rate': win_rate, 'avg_odds': avg_odds,
            'total_pnl': total_pnl, 'avg_pnl': avg_pnl, 'roi_pct': avg_pnl * 100,
            'sharpe': sharpe,
        })

    # ══════════════════════════════════════════════
    # Feature importance
    # ══════════════════════════════════════════════
    importances = pd.DataFrame({
        'feature': predictors_col,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\n=== Feature Importances ===", flush=True)
    print(importances.to_string(index=False), flush=True)

    # ══════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════
    hp_str = f"ne{hp['n_estimators']}_md{hp['max_depth']}_lr{hp['learning_rate']}"
    save_dir = f'{Constant.RES_DIR}/win_model_v2_surface_n{n_features}/t{t_definition}/{hp_str}/'
    os.makedirs(save_dir, exist_ok=True)

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

    if bet_results:
        pd.DataFrame(bet_results).to_parquet(save_dir + 'bet_results.parquet')
    importances.to_parquet(save_dir + 'feature_importances.parquet')

    xgb_model.save_model(save_dir + 'xgboost_model.json')
    if HAS_LGBM:
        lgb_model.booster_.save_model(save_dir + 'lightgbm_model.txt')

    # Save the feature ranking and selected features
    importances_ranking.to_parquet(save_dir + 'feature_ranking_all_surface.parquet')
    with open(save_dir + 'selected_features.json', 'w') as f:
        json.dump({'n_features': n_features, 'features': top_features}, f, indent=2)

    metrics = {
        'hp': hp,
        't_definition': int(t_definition),
        'n_features_requested': int(n_features),
        'n_features_actual': int(len(predictors_col)),
        'xgb_best_iter': int(xgb_best_iter),
        'has_lgbm': bool(HAS_LGBM),
    }
    for name, probs in models_to_eval.items():
        metrics[f'{name}_logloss'] = float(log_loss(df_oos['win'], probs))
        metrics[f'{name}_brier'] = float(brier_score_loss(df_oos['win'], probs))
    with open(save_dir + 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll saved to {save_dir}", flush=True)
    print("Done!", flush=True)
