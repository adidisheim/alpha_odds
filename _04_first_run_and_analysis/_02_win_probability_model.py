"""
Win Probability Model for Value Betting Strategy.

Trains an XGBoost classifier to predict win probabilities for each runner.
Evaluates calibration and compares model probabilities to market implied probabilities.

Usage:
    python _02_win_probability_model.py <grid_comb_id> <t_definition>

    grid_comb_id: index into the hyperparameter grid (0-based)
    t_definition: which time snapshot features to use (0, 1, 2, or 3)

Output saved to: res/win_model/t{t_definition}/{grid_hash}/
    - save_df.parquet: OOS predictions with model_prob, market_prob, edge
    - calibration.parquet: calibration table (predicted vs actual win rate by decile)
    - feature_importances.parquet: XGBoost feature importances
    - xgboost_model.json: trained model
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss
import psutil
import os
import socket
import itertools

from parameters import Constant, SpreadTopKCriterion
from utils_locals.parser import parse


# ──────────────────────────────────────────────
# Hyperparameter grid
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
# Feature normalizer (reused from _01_run.py)
# ──────────────────────────────────────────────
class FeatureNormalizer:
    def __init__(self, predictors_col):
        self.predictors_col = list(predictors_col)
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
        self.high_missing_cols = [c for c in self.predictors_col if miss.get(c, 0.0) > 0.5]
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
    grid_combs = get_grid_combinations()
    if args.a >= len(grid_combs):
        print(f"Grid index {args.a} out of range (max {len(grid_combs)-1}). Exiting.", flush=True)
        exit(0)

    hp = grid_combs[args.a]
    print(f"Grid combo {args.a}/{len(grid_combs)}: {hp}", flush=True)
    print(f"t_definition: {t_definition}", flush=True)

    # ── Load features (prefer merged file, fallback to parts) ──
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
        df = df.dropna(subset='file_name').sample(frac=1).head(10000)

    process = psutil.Process(os.getpid())
    print(f"RAM after load: {process.memory_info().rss / 1024**3:.2f} GB", flush=True)
    print(f"Data shape: {df.shape}", flush=True)

    # ── Define target ──
    df['win'] = (df['id'] == -1).astype(int)

    # ── Market implied probability (mid-price at decision time) ──
    # Note: best_back_m0 and best_lay_m0 are already in implied prob space (1/odds)
    df['market_prob'] = df[['best_back_m0', 'best_lay_m0']].mean(axis=1)
    # Drop rows with missing market probability (can't evaluate without market benchmark)
    n_before = len(df)
    df = df.dropna(subset=['market_prob', 'best_back_m0', 'best_lay_m0'])
    print(f"Dropped {n_before - len(df)} rows with NaN market_prob ({(n_before - len(df))/n_before*100:.2f}%)", flush=True)

    # ── Define predictors (same as _01_run.py) ──
    suffix_available_at_t0 = [
        "_count_2_1", "_count_3_1", "_mean_2_1", "_mean_3_1",
        "_m0", "_mom_2_1", "_mom_3_1",
        "_order_is_back_2_1", "_order_is_back_3_1",
        "_std_2_1", "_std_3_1"
    ]
    predictors_col = [c for c in df.columns if c.endswith(tuple(suffix_available_at_t0))]

    # Add fraction features
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

    # Add runner_position as feature (box/trap number matters in greyhounds)
    if 'runner_position' in df.columns:
        predictors_col.append('runner_position')

    fixed_effect_columns = ['local_dow', 'marketBaseRate', 'numberOfActiveRunners']
    predictors_col = predictors_col + fixed_effect_columns

    # ── Train/test split ──
    oos_year = 2025
    ins_years = df['marketTime_local'].dt.year.unique()
    ins_years = [int(x) for x in ins_years if x != oos_year]
    ind_ins = df['marketTime_local'].dt.year.isin(ins_years)
    ind_oos = df['marketTime_local'].dt.year == oos_year

    # No topK or spread restriction for win model — we want ALL runners
    print(f"In-sample: {ind_ins.sum()} rows, OOS: {ind_oos.sum()} rows", flush=True)
    print(f"In-sample win rate: {df.loc[ind_ins, 'win'].mean():.4f}", flush=True)
    print(f"OOS win rate: {df.loc[ind_oos, 'win'].mean():.4f}", flush=True)

    # ── Preserve original prices for betting simulation (before normalization z-scores them) ──
    original_cols = ['best_back_m0', 'best_lay_m0', 'best_back_q_100_m0', 'best_lay_q_100_m0', 'marketBaseRate']
    original_cols = [c for c in original_cols if c in df.columns]
    oos_originals = df.loc[ind_oos, ['file_name', 'id'] + original_cols].copy()

    # ── Normalize ──
    normalizer = FeatureNormalizer(predictors_col)
    df_ins = normalizer.normalize_ins(df.loc[ind_ins, :])
    df_oos = normalizer.normalize_oos(df.loc[ind_oos, :])
    predictors_col = normalizer.predictors_col  # may have added _missing cols

    df_ins = df_ins.dropna(subset=['win'])
    df_oos = df_oos.dropna(subset=['win'])

    # ── Train XGBoost classifier ──
    print(f"Training XGBClassifier with {hp}", flush=True)
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
    print("Training complete.", flush=True)

    # ── Predict probabilities ──
    df_oos = df_oos.copy()
    df_oos['model_prob'] = model.predict_proba(df_oos[predictors_col])[:, 1]

    # ── Edge = model_prob - market_prob ──
    df_oos['edge'] = df_oos['model_prob'] - df_oos['market_prob']

    # ── Evaluate ──
    ll = log_loss(df_oos['win'], df_oos['model_prob'])
    ll_market = log_loss(df_oos['win'], df_oos['market_prob'].clip(0.001, 0.999))
    brier = brier_score_loss(df_oos['win'], df_oos['model_prob'])
    brier_market = brier_score_loss(df_oos['win'], df_oos['market_prob'].clip(0.001, 0.999))

    print(f"\n=== OOS Evaluation ===", flush=True)
    print(f"Model log-loss:  {ll:.6f}", flush=True)
    print(f"Market log-loss: {ll_market:.6f}", flush=True)
    print(f"Model Brier:     {brier:.6f}", flush=True)
    print(f"Market Brier:    {brier_market:.6f}", flush=True)
    print(f"Model beats market? Log-loss: {ll < ll_market}, Brier: {brier < brier_market}", flush=True)

    # ── Calibration by decile ──
    df_oos['prob_decile'] = pd.qcut(df_oos['model_prob'], 10, labels=False, duplicates='drop')
    calibration = df_oos.groupby('prob_decile').agg(
        mean_predicted=('model_prob', 'mean'),
        mean_actual=('win', 'mean'),
        mean_market=('market_prob', 'mean'),
        mean_edge=('edge', 'mean'),
        count=('win', 'count'),
    ).reset_index()
    print(f"\n=== Calibration Table ===", flush=True)
    print(calibration.to_string(index=False), flush=True)

    # ── Value betting simulation (simple) ──
    # Restore original (un-normalized) prices for odds computation
    df_oos = df_oos.reset_index(drop=True)
    oos_originals = oos_originals.reset_index(drop=True)
    for c in original_cols:
        df_oos[f'orig_{c}'] = oos_originals[c].values

    # Restore original marketBaseRate for commission (z-scored version would be wrong)
    df_oos['orig_marketBaseRate'] = oos_originals['marketBaseRate'].values
    commission_rate = df_oos['orig_marketBaseRate'].median() / 100  # typically 0.07 or 0.08
    print(f"\nCommission rate: {commission_rate:.4f} (original marketBaseRate median: {df_oos['orig_marketBaseRate'].median():.2f})", flush=True)
    for edge_threshold in [0.01, 0.02, 0.05, 0.10]:
        bets = df_oos[df_oos['edge'] > edge_threshold].copy()
        if len(bets) == 0:
            print(f"\nEdge threshold {edge_threshold}: no qualifying bets", flush=True)
            continue
        # Simulate: back at original best_back_m0 (in implied prob), so odds = 1/implied_prob
        bets['back_odds'] = 1 / bets['orig_best_back_m0']
        # Filter out unreasonable odds (< 1.01 or > 1000)
        bets = bets[(bets['back_odds'] > 1.01) & (bets['back_odds'] < 1000)]
        if len(bets) == 0:
            print(f"\nEdge threshold {edge_threshold}: no valid bets after odds filter", flush=True)
            continue
        # P&L per $1 bet: win → (odds-1)*(1-commission), lose → -1
        bets['pnl'] = bets['win'] * (bets['back_odds'] - 1) * (1 - commission_rate) - (1 - bets['win'])
        n_bets = len(bets)
        total_pnl = bets['pnl'].sum()
        avg_pnl = bets['pnl'].mean()
        win_rate = bets['win'].mean()
        avg_odds = bets['back_odds'].mean()
        print(f"\nEdge threshold {edge_threshold}:", flush=True)
        print(f"  Bets: {n_bets}, Win rate: {win_rate:.4f}, Avg odds: {avg_odds:.1f}", flush=True)
        print(f"  Total P&L per $1/bet: ${total_pnl:.2f}, Avg P&L: ${avg_pnl:.4f}", flush=True)
        print(f"  ROI: {avg_pnl*100:.2f}%", flush=True)

    # ── Save results ──
    hp_str = f"ne{hp['n_estimators']}_md{hp['max_depth']}_lr{hp['learning_rate']}"
    save_dir = f'{Constant.RES_DIR}/win_model/t{t_definition}/{hp_str}/'
    os.makedirs(save_dir, exist_ok=True)

    save_cols = ['file_name', 'id', 'win', 'model_prob', 'market_prob', 'edge',
                 'orig_best_back_m0', 'orig_best_lay_m0', 'orig_best_back_q_100_m0', 'orig_best_lay_q_100_m0',
                 'marketBaseRate', 'numberOfActiveRunners', 'local_dow', 'runner_position',
                 'marketTime_local']
    save_cols = [c for c in save_cols if c in df_oos.columns]
    df_oos[save_cols].to_parquet(save_dir + 'save_df.parquet')
    calibration.to_parquet(save_dir + 'calibration.parquet')

    importances = pd.DataFrame({
        'feature': predictors_col,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importances.to_parquet(save_dir + 'feature_importances.parquet')
    model.save_model(save_dir + 'xgboost_model.json')

    print(f"\nAll saved to {save_dir}", flush=True)
    print("Done!", flush=True)
