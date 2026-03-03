"""
Phase 2: Ablation study — retrain V2-style models with feature subsets.

Takes the ranked feature list from Phase 1 and retrains with restricted feature sets.
Uses a single good HP config (best V2 by log-loss) for all ablations.

Feature counts to test: top-5, top-10, top-15, top-20, top-30, top-50, top-80, all
Array job: task_id = 0..31 (8 feature counts x 4 t_defs)

Usage:
    python _02_ablation_study.py <task_id>

Output:
    res/feature_importance_study/ablation_{task_id}.parquet
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import psutil
import os
import json
from math import erf, sqrt

from parameters import Constant
from utils_locals.parser import parse

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, using XGBoost only.", flush=True)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
FEATURE_COUNTS = [5, 10, 15, 20, 30, 50, 80, None]  # None = all features
N_FEATURE_COUNTS = len(FEATURE_COUNTS)
T_DEFS = [0, 1, 2, 3]
N_T_DEFS = len(T_DEFS)

# Best HP from V2 grid search (will be auto-detected from saved metrics)
DEFAULT_HP = {'n_estimators': 2000, 'max_depth': 6, 'learning_rate': 0.05}


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def find_best_hp(t_def):
    """Find the best-performing V2 HP config for this t_def by log-loss."""
    base_dir = f'{Constant.RES_DIR}/win_model_v2/t{t_def}'
    if not os.path.isdir(base_dir):
        return DEFAULT_HP

    best_ll = 999
    best_hp = None
    for config in os.listdir(base_dir):
        metrics_path = os.path.join(base_dir, config, 'metrics.json')
        if not os.path.exists(metrics_path):
            continue
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            ll = metrics.get('Calibrated_logloss', metrics.get('XGBoost_logloss', 999))
            if ll < best_ll:
                best_ll = ll
                best_hp = metrics.get('hp', None)
        except Exception:
            continue

    if best_hp is not None:
        return best_hp
    return DEFAULT_HP


# ──────────────────────────────────────────────
# Cross-runner features (copied from V2 pipeline)
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
# Feature normalizer (copied from V2 pipeline)
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
    task_id = args.a

    # Decode task_id -> (feature_count_idx, t_def)
    feat_idx = task_id % N_FEATURE_COUNTS
    t_def_idx = task_id // N_FEATURE_COUNTS
    n_features_target = FEATURE_COUNTS[feat_idx]
    t_definition = T_DEFS[t_def_idx]

    feat_label = str(n_features_target) if n_features_target else 'all'
    print(f"=== Ablation Study ===", flush=True)
    print(f"Task {task_id}: top-{feat_label} features, t_def={t_definition}", flush=True)

    # ── Load feature ranking from Phase 1 ──
    ranking_path = f'{Constant.RES_DIR}/feature_importance_study/feature_ranking.parquet'
    if not os.path.exists(ranking_path):
        print(f"ERROR: Feature ranking not found at {ranking_path}", flush=True)
        print("Run Phase 1 (_01_collect_importances.py) first!", flush=True)
        exit(1)
    ranking = pd.read_parquet(ranking_path)
    all_ranked_features = ranking['feature'].tolist()
    print(f"Loaded ranking with {len(all_ranked_features)} features.", flush=True)

    # ── Find best HP for this t_def ──
    hp = find_best_hp(t_definition)
    print(f"Using HP: {hp}", flush=True)

    # ── Load features ──
    load_dir = f'{Constant.RES_DIR}/features_t{t_definition}'
    merged_path = f'{load_dir}/greyhound_au_features_merged.parquet'
    if os.path.exists(merged_path):
        print(f"Loading merged features from {merged_path}", flush=True)
        df = pd.read_parquet(merged_path)
    else:
        print(f"Loading feature parts...", flush=True)
        df = pd.DataFrame()
        for i in range(10):
            try:
                df = pd.concat([df, pd.read_parquet(f'{load_dir}/greyhound_au_features_part_{i}.parquet')], ignore_index=False)
            except Exception as e:
                print(f'  Part {i}: {e}', flush=True)

    process = psutil.Process(os.getpid())
    print(f"RAM after load: {process.memory_info().rss / 1024**3:.2f} GB", flush=True)
    print(f"Data shape: {df.shape}", flush=True)

    # ── Define target ──
    df['win'] = (df['id'] == -1).astype(int)

    # ── Drop NaN order book ──
    n_before = len(df)
    df = df.dropna(subset=['best_back_m0', 'best_lay_m0'])
    print(f"Dropped {n_before - len(df)} rows with NaN best_back/lay_m0", flush=True)

    # ── Add cross-runner features ──
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
    predictors_col = predictors_col + fixed_effect_columns
    predictors_col = predictors_col + cross_runner_cols

    # Deduplicate
    seen = set()
    deduped = []
    for c in predictors_col:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    predictors_col = deduped

    full_n_features = len(predictors_col)
    print(f"Full feature set: {full_n_features} features", flush=True)

    # ── Feature subsetting ──
    # Only keep features that are in the ranking AND in our predictors_col
    # Ranked features may include _missing indicators added during normalization;
    # we handle those by letting the normalizer add them later
    if n_features_target is not None:
        # Take top-N from ranking that exist in our predictor list
        ranked_available = [f for f in all_ranked_features if f in predictors_col]
        # Also check if ranked features have _missing variants we need
        n_to_keep = min(n_features_target, len(ranked_available))
        selected_features = ranked_available[:n_to_keep]

        # Always include market_prob dependencies (best_back_m0, best_lay_m0)
        # since we need them for edge calculation
        for essential in ['best_back_m0', 'best_lay_m0']:
            if essential not in selected_features and essential in predictors_col:
                selected_features.append(essential)

        predictors_col = selected_features
        print(f"Restricted to top-{n_features_target}: {len(predictors_col)} features available", flush=True)
    else:
        print(f"Using all {full_n_features} features", flush=True)

    # ── Train/Val/Test split ──
    oos_year = 2025
    val_year = 2024
    train_years = df['marketTime_local'].dt.year.unique()
    train_years = [int(x) for x in train_years if x not in [oos_year, val_year]]

    ind_train = df['marketTime_local'].dt.year.isin(train_years)
    ind_val = df['marketTime_local'].dt.year == val_year
    ind_oos = df['marketTime_local'].dt.year == oos_year

    print(f"Train: {ind_train.sum()} rows, Val: {ind_val.sum()}, OOS: {ind_oos.sum()}", flush=True)

    # ── Preserve original prices before normalization ──
    original_cols = ['best_back_m0', 'best_lay_m0', 'marketBaseRate']
    original_cols = [c for c in original_cols if c in df.columns]
    oos_originals = df.loc[ind_oos, original_cols].copy()

    # ── Normalize ──
    normalizer = FeatureNormalizer(predictors_col)
    df_train = normalizer.normalize_ins(df.loc[ind_train, :])
    df_val = normalizer.normalize_oos(df.loc[ind_val, :])
    df_oos = normalizer.normalize_oos(df.loc[ind_oos, :])
    predictors_col = normalizer.predictors_col  # may have _missing cols added

    df_train = df_train.dropna(subset=['win'])
    df_val = df_val.dropna(subset=['win'])
    df_oos = df_oos.dropna(subset=['win'])

    actual_n_features = len(predictors_col)
    print(f"Final feature count (after normalization): {actual_n_features}", flush=True)

    # ══════════════════════════════════════════════
    # Train XGBoost
    # ══════════════════════════════════════════════
    print(f"\n=== Training XGBoost ({hp}) ===", flush=True)
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
    xgb_best_iter = xgb_model.best_iteration
    print(f"XGBoost best iteration: {xgb_best_iter}", flush=True)

    df_oos = df_oos.copy()
    df_oos['xgb_prob'] = xgb_model.predict_proba(df_oos[predictors_col])[:, 1]
    df_val = df_val.copy()
    df_val['xgb_prob'] = xgb_model.predict_proba(df_val[predictors_col])[:, 1]

    # ══════════════════════════════════════════════
    # Train LightGBM
    # ══════════════════════════════════════════════
    if HAS_LGBM:
        print(f"\n=== Training LightGBM ===", flush=True)
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

    # ══════════════════════════════════════════════
    # Isotonic calibration
    # ══════════════════════════════════════════════
    iso_reg = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso_reg.fit(df_val['ensemble_prob'].values, df_val['win'].values)
    df_oos['calibrated_prob'] = iso_reg.predict(df_oos['ensemble_prob'].values)
    df_oos['model_prob'] = df_oos['calibrated_prob']

    # ── Edge (market_prob already set by add_cross_runner_features before normalization) ──
    df_oos['edge'] = df_oos['model_prob'] - df_oos['market_prob']

    # ══════════════════════════════════════════════
    # Evaluation metrics
    # ══════════════════════════════════════════════
    market_ll = log_loss(df_oos['win'], df_oos['market_prob'].clip(0.001, 0.999))
    xgb_ll = log_loss(df_oos['win'], df_oos['xgb_prob'])
    cal_ll = log_loss(df_oos['win'], df_oos['calibrated_prob'])
    market_brier = brier_score_loss(df_oos['win'], df_oos['market_prob'].clip(0.001, 0.999))
    cal_brier = brier_score_loss(df_oos['win'], df_oos['calibrated_prob'])

    print(f"\n{'='*60}", flush=True)
    print(f"OOS Results: top-{feat_label}, t_def={t_definition}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Market LL:     {market_ll:.6f}", flush=True)
    print(f"  XGBoost LL:    {xgb_ll:.6f}", flush=True)
    print(f"  Calibrated LL: {cal_ll:.6f}", flush=True)
    print(f"  Market Brier:  {market_brier:.6f}", flush=True)
    print(f"  Cal Brier:     {cal_brier:.6f}", flush=True)

    # ══════════════════════════════════════════════
    # Value betting backtest
    # ══════════════════════════════════════════════
    df_oos['orig_best_back_m0'] = oos_originals['best_back_m0'].values
    df_oos['orig_best_lay_m0'] = oos_originals['best_lay_m0'].values if 'best_lay_m0' in oos_originals.columns else np.nan
    df_oos['orig_marketBaseRate'] = oos_originals['marketBaseRate'].values if 'marketBaseRate' in oos_originals.columns else 8.0

    commission_rate = df_oos['orig_marketBaseRate'].median() / 100
    if np.isnan(commission_rate) or commission_rate <= 0:
        commission_rate = 0.075

    bet_results = []
    for edge_threshold in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        bets = df_oos[df_oos['edge'] > edge_threshold].copy()
        if len(bets) == 0:
            bet_results.append({
                'edge_threshold': edge_threshold,
                'n_bets': 0, 'win_rate': np.nan, 'avg_odds': np.nan,
                'total_pnl': 0, 'avg_pnl': np.nan, 'roi_pct': np.nan,
                'sharpe': np.nan,
            })
            continue

        bets['back_odds'] = 1 / bets['orig_best_back_m0']
        bets = bets[(bets['back_odds'] > 1.01) & (bets['back_odds'] < 1000)]
        if len(bets) == 0:
            bet_results.append({
                'edge_threshold': edge_threshold,
                'n_bets': 0, 'win_rate': np.nan, 'avg_odds': np.nan,
                'total_pnl': 0, 'avg_pnl': np.nan, 'roi_pct': np.nan,
                'sharpe': np.nan,
            })
            continue

        bets['pnl'] = bets['win'] * (bets['back_odds'] - 1) * (1 - commission_rate) - (1 - bets['win'])
        n_bets = len(bets)
        total_pnl = bets['pnl'].sum()
        avg_pnl = bets['pnl'].mean()
        win_rate = bets['win'].mean()
        avg_odds = bets['back_odds'].mean()
        pnl_std = bets['pnl'].std()
        sharpe = avg_pnl / pnl_std * np.sqrt(n_bets) if pnl_std > 0 else 0

        # Monthly Sharpe (annualized)
        bets['month'] = pd.to_datetime(bets['marketTime_local']).dt.to_period('M')
        monthly = bets.groupby('month')['pnl'].sum()
        sharpe_monthly = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0

        # z-stat and p-value
        z = avg_pnl / pnl_std * np.sqrt(n_bets) if pnl_std > 0 else 0
        p_value = 1 - norm_cdf(z)

        print(f"\n  Edge>{edge_threshold}: {n_bets} bets, ROI={avg_pnl*100:+.1f}%, "
              f"Sharpe={sharpe_monthly:.1f}, z={z:.2f}, p={p_value:.4f}, "
              f"$25/bet=${total_pnl*25:,.0f}", flush=True)

        bet_results.append({
            'edge_threshold': edge_threshold,
            'n_bets': n_bets, 'win_rate': win_rate, 'avg_odds': avg_odds,
            'total_pnl': total_pnl, 'avg_pnl': avg_pnl, 'roi_pct': avg_pnl * 100,
            'sharpe': sharpe, 'sharpe_annual': sharpe_monthly,
            'z_stat': z, 'p_value': p_value,
            'profit_25': total_pnl * 25,
        })

    # ══════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════
    out_dir = f'{Constant.RES_DIR}/feature_importance_study'
    os.makedirs(out_dir, exist_ok=True)

    # Build summary row for each edge threshold
    rows = []
    for br in bet_results:
        row = {
            'task_id': int(task_id),
            'n_features_target': int(n_features_target) if n_features_target else full_n_features,
            'n_features_actual': actual_n_features,
            'feat_label': feat_label,
            't_definition': int(t_definition),
            'hp': str(hp),
            'has_lgbm': bool(HAS_LGBM),
            'xgb_best_iter': int(xgb_best_iter),
            'n_oos': int(ind_oos.sum()),
            'market_logloss': float(market_ll),
            'calibrated_logloss': float(cal_ll),
            'xgb_logloss': float(xgb_ll),
            'market_brier': float(market_brier),
            'calibrated_brier': float(cal_brier),
        }
        row.update(br)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    save_path = os.path.join(out_dir, f'ablation_{task_id}.parquet')
    result_df.to_parquet(save_path, index=False)

    # Save per-runner OOS predictions for cross-t ensemble assembly
    pred_cols = ['file_name', 'id', 'win', 'model_prob', 'market_prob',
                 'orig_best_back_m0', 'orig_best_lay_m0', 'marketTime_local']
    pred_cols = [c for c in pred_cols if c in df_oos.columns]
    pred_path = os.path.join(out_dir, f'predictions_{task_id}.parquet')
    df_oos[pred_cols].to_parquet(pred_path, index=False)
    print(f"Saved {len(df_oos)} per-runner predictions to {pred_path}", flush=True)
    print(f"\nSaved results to {save_path}", flush=True)

    # Also save the feature list used
    feat_list_path = os.path.join(out_dir, f'features_used_{task_id}.txt')
    with open(feat_list_path, 'w') as f:
        for feat in predictors_col:
            f.write(feat + '\n')

    print(f"Features used saved to {feat_list_path}", flush=True)
    print(f"\nRAM: {process.memory_info().rss / 1024**3:.2f} GB", flush=True)
    print("Done!", flush=True)
