"""
A/B test: compare XGBoost model with and without dog track record features.

Usage: python _06_03_test_dog_features.py <t_definition>
  arg a = t_definition (0, 1, 2, or 3)

Loads the augmented features from _06_02, trains two XGBoost classifiers
(baseline vs +dog features), and compares OOS log-loss.
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier
from utils_locals.parser import parse
from parameters import Constant


# Dog feature columns to test
DOG_FEATURE_COLS = [
    'dog_n_races',
    'dog_win_rate',
    'dog_win_rate_last5',
    'dog_win_rate_last10',
    'dog_avg_market_prob',
    'dog_overperformance',
    'dog_days_since_last',
    'dog_avg_position',
    'dog_venue_n_races',
    'dog_venue_win_rate',
    'dog_streak',
]

# Suffix-based feature selection (same as _01_run.py)
SUFFIX_AVAILABLE_AT_T0 = [
    "_count_2_1", "_count_3_1",
    "_mean_2_1", "_mean_3_1",
    "_m0",
    "_mom_2_1", "_mom_3_1",
    "_order_is_back_2_1", "_order_is_back_3_1",
    "_std_2_1", "_std_3_1",
]

FIXED_EFFECT_COLS = ['local_dow', 'marketBaseRate', 'numberOfActiveRunners']

# XGBoost hyperparameters (reasonable defaults for quick test)
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=12345,
    n_jobs=-1,
    verbosity=0,
    eval_metric='logloss',
    use_label_encoder=False,
)


def build_features(df, include_dog=False):
    """Build predictor columns, add fraction features, return (df, predictor_list)."""
    df = df.copy()

    # Base predictors by suffix
    predictors = [c for c in df.columns if c.endswith(tuple(SUFFIX_AVAILABLE_AT_T0))]

    # Fraction features (same as _01_run.py)
    df['total_qty_m1'] = df[['total_back_qty_m1', 'total_lay_qty_m1']].sum(axis=1)
    df['total_qty_m3'] = df[['total_back_qty_m3', 'total_lay_qty_m3']].sum(axis=1)

    frac_cols = []
    for col in ['total_qty_m1', 'total_back_qty_m1', 'total_lay_qty_m1',
                'total_qty_m3', 'total_back_qty_m3', 'total_lay_qty_m3']:
        c = col + '_frac'
        df[c] = df[col] / df.groupby('file_name')[col].transform('sum')
        frac_cols.append(c)

    frac_mom = []
    for col in [x for x in frac_cols if x.endswith('_m1_frac')]:
        c = col.replace('_m1', '_mom_3_1')
        df[c] = df[col] - df[col.replace('_m1', '_m3')]
        frac_mom.append(c)

    predictors = predictors + frac_cols + frac_mom + FIXED_EFFECT_COLS

    # Add dog features if requested
    if include_dog:
        dog_cols_available = [c for c in DOG_FEATURE_COLS if c in df.columns]
        predictors = predictors + dog_cols_available

    # Deduplicate
    predictors = list(dict.fromkeys(predictors))

    # Remove any predictors not actually in df
    predictors = [c for c in predictors if c in df.columns]

    return df, predictors


def train_and_evaluate(df, predictors, label='Model'):
    """Train XGBoost on IS, evaluate on OOS. Returns (oos_logloss, oos_auc, model)."""
    df = df.copy()

    # Target
    df['win'] = (df['id'] == -1).astype(int)

    # Top-k filtering (keep top 3 by spread)
    df = df.reset_index(drop=True)
    df['spread_for_top_k'] = (df['best_back_m1'] - df['best_lay_m1']).abs()
    ind_top_k = df.groupby('file_name')['spread_for_top_k'].rank(
        method='min', ascending=False
    ) <= 3

    # Spread restriction
    ind_spread = df['spread_for_top_k'] <= 0.1

    # Train/test split
    ind_ins = (df['marketTime_local'].dt.year < 2025) & ind_top_k & ind_spread
    ind_oos = (df['marketTime_local'].dt.year == 2025) & ind_top_k

    df_ins = df.loc[ind_ins].copy()
    df_oos = df.loc[ind_oos].copy()

    # Fill NaN in predictors with median (fit on IS)
    medians = df_ins[predictors].median()
    df_ins[predictors] = df_ins[predictors].fillna(medians)
    df_oos[predictors] = df_oos[predictors].fillna(medians)

    print(f'\n  [{label}]')
    print(f'  IS rows: {len(df_ins):,}  |  OOS rows: {len(df_oos):,}')
    print(f'  IS win rate: {df_ins["win"].mean():.4f}  |  OOS win rate: {df_oos["win"].mean():.4f}')
    print(f'  Features: {len(predictors)}')

    # Train
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(df_ins[predictors], df_ins['win'])

    # Predict probabilities
    oos_proba = model.predict_proba(df_oos[predictors])[:, 1]

    # Evaluate
    oos_ll = log_loss(df_oos['win'], oos_proba)
    oos_auc = roc_auc_score(df_oos['win'], oos_proba)

    # Market benchmark: use best_back_m0 as implied probability
    if 'best_back_m0' in df_oos.columns:
        market_proba = df_oos['best_back_m0'].clip(0.001, 0.999)
        market_ll = log_loss(df_oos['win'], market_proba)
        print(f'  Market log-loss: {market_ll:.6f}')

    print(f'  Model  log-loss: {oos_ll:.6f}')
    print(f'  Model  AUC:      {oos_auc:.6f}')

    return oos_ll, oos_auc, model, df_oos


if __name__ == '__main__':
    args = parse()
    t_def = args.a

    save_dir = Constant.RES_DIR + 'dog_features/'
    features_path = save_dir + f'features_with_dog_t{t_def}.parquet'

    print('=' * 60)
    print('DOG TRACK RECORD FEATURE — A/B TEST')
    print('=' * 60)
    print(f'Time definition: {t_def}')
    print(f'Loading features from: {features_path}')

    df = pd.read_parquet(features_path)
    print(f'Total rows: {len(df):,}')
    print(f'Unique dogs: {df["dog_name"].nunique():,}')

    # How many OOS runners have enough dog history?
    oos_mask = df['marketTime_local'].dt.year == 2025
    oos_dogs = df.loc[oos_mask]
    for thresh in [0, 5, 10, 20]:
        n = (oos_dogs['dog_n_races'] >= thresh).sum()
        print(f'  OOS runners with {thresh}+ prior races: {n:,} ({100*n/len(oos_dogs):.1f}%)')

    # =====================
    # BASELINE: no dog features
    # =====================
    print('\n' + '=' * 60)
    print('BASELINE MODEL (no dog features)')
    print('=' * 60)
    df_base, preds_base = build_features(df, include_dog=False)
    ll_base, auc_base, model_base, oos_base = train_and_evaluate(df_base, preds_base, 'Baseline')

    # =====================
    # TREATMENT: with dog features
    # =====================
    print('\n' + '=' * 60)
    print('TREATMENT MODEL (with dog features)')
    print('=' * 60)
    df_dog, preds_dog = build_features(df, include_dog=True)
    ll_dog, auc_dog, model_dog, oos_dog = train_and_evaluate(df_dog, preds_dog, '+Dog Features')

    # =====================
    # COMPARISON
    # =====================
    print('\n' + '=' * 60)
    print('COMPARISON')
    print('=' * 60)
    ll_diff = ll_base - ll_dog
    ll_pct = 100 * ll_diff / ll_base
    auc_diff = auc_dog - auc_base

    print(f'Baseline log-loss:  {ll_base:.6f}')
    print(f'+Dog log-loss:      {ll_dog:.6f}')
    print(f'Improvement:        {ll_diff:.6f} ({ll_pct:+.3f}%)')
    print(f'')
    print(f'Baseline AUC:       {auc_base:.6f}')
    print(f'+Dog AUC:           {auc_dog:.6f}')
    print(f'AUC improvement:    {auc_diff:+.6f}')

    if ll_dog < ll_base:
        print(f'\n>>> DOG FEATURES HELP: log-loss improved by {ll_pct:.3f}%')
    else:
        print(f'\n>>> DOG FEATURES DO NOT HELP: log-loss worsened by {-ll_pct:.3f}%')

    # =====================
    # FEATURE IMPORTANCE (dog features in treatment model)
    # =====================
    print('\n' + '=' * 60)
    print('DOG FEATURE IMPORTANCES (in treatment model)')
    print('=' * 60)
    importances = pd.Series(model_dog.feature_importances_, index=preds_dog)
    dog_imp = importances[[c for c in DOG_FEATURE_COLS if c in importances.index]]
    dog_imp = dog_imp.sort_values(ascending=False)
    total_imp = importances.sum()
    print(f'Dog features share of total importance: {dog_imp.sum()/total_imp:.2%}')
    for feat, imp in dog_imp.items():
        print(f'  {feat:30s}: {imp:.4f} ({imp/total_imp:.2%})')

    # Top 20 overall features in treatment model
    print('\nTop 20 features overall (treatment model):')
    top20 = importances.sort_values(ascending=False).head(20)
    for feat, imp in top20.items():
        marker = ' ***' if feat.startswith('dog_') else ''
        print(f'  {feat:40s}: {imp:.4f}{marker}')

    # =====================
    # SAVE RESULTS
    # =====================
    results = {
        't_definition': t_def,
        'baseline_logloss': ll_base,
        'dog_logloss': ll_dog,
        'logloss_improvement_pct': ll_pct,
        'baseline_auc': auc_base,
        'dog_auc': auc_dog,
        'auc_improvement': auc_diff,
        'n_baseline_features': len(preds_base),
        'n_dog_features': len(preds_dog),
        'n_oos_rows': len(oos_dog),
    }
    results_df = pd.DataFrame([results])
    results_path = save_dir + f'ab_test_results_t{t_def}.parquet'
    results_df.to_parquet(results_path, index=False)

    # Save feature importances
    imp_df = pd.DataFrame({
        'feature': preds_dog,
        'importance': model_dog.feature_importances_,
        'is_dog_feature': [c.startswith('dog_') for c in preds_dog]
    }).sort_values('importance', ascending=False)
    imp_path = save_dir + f'feature_importances_t{t_def}.parquet'
    imp_df.to_parquet(imp_path, index=False)

    print(f'\nResults saved to {results_path}')
    print(f'Importances saved to {imp_path}')
    print('\nAll done!', flush=True)
