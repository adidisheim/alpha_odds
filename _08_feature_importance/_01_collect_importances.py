"""
Phase 1: Aggregate feature importances from all 108 V2 model directories.

Walks res/win_model_v2/t{0,1,2,3}/*/ and loads feature_importances.parquet + metrics.json.
Computes raw average, log-loss-weighted average, and rank stability.
Saves to res/feature_importance_study/feature_ranking.parquet

Usage:
    python _01_collect_importances.py
"""

import os
import json
import numpy as np
import pandas as pd

from parameters import Constant


def main():
    base_dir = f'{Constant.RES_DIR}/win_model_v2'
    out_dir = f'{Constant.RES_DIR}/feature_importance_study'
    os.makedirs(out_dir, exist_ok=True)

    records = []  # (t_def, config, importance_df, logloss)

    for t_def in [0, 1, 2, 3]:
        t_dir = os.path.join(base_dir, f't{t_def}')
        if not os.path.isdir(t_dir):
            print(f"  Skipping t{t_def}: directory not found", flush=True)
            continue
        for config in sorted(os.listdir(t_dir)):
            config_dir = os.path.join(t_dir, config)
            imp_path = os.path.join(config_dir, 'feature_importances.parquet')
            metrics_path = os.path.join(config_dir, 'metrics.json')

            if not os.path.exists(imp_path):
                continue

            try:
                imp_df = pd.read_parquet(imp_path)
            except Exception as e:
                print(f"  WARNING: Could not read {imp_path}: {e}", flush=True)
                continue

            # Load metrics for log-loss weighting
            logloss = None
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    # Try Calibrated first (the final model), then XGBoost
                    logloss = metrics.get('Calibrated_logloss',
                                          metrics.get('XGBoost_logloss', None))
                except Exception:
                    pass

            records.append({
                't_def': t_def,
                'config': config,
                'imp_df': imp_df,
                'logloss': logloss,
            })

    print(f"Loaded {len(records)} model importance files.", flush=True)
    if not records:
        print("ERROR: No feature importance files found!", flush=True)
        return

    # ── 1. Raw average importance across all models ──
    # Normalize each model's importances to sum to 1, then average
    all_imp = []
    for r in records:
        imp = r['imp_df'][['feature', 'importance']].copy()
        total = imp['importance'].sum()
        if total > 0:
            imp['importance'] = imp['importance'] / total
        imp = imp.set_index('feature')['importance']
        all_imp.append(imp)

    combined = pd.DataFrame(all_imp).fillna(0)
    raw_avg = combined.mean(axis=0).rename('raw_avg_importance')

    # ── 2. Log-loss-weighted importance ──
    # Better models (lower LL) get higher weight
    weighted_imp = []
    weights = []
    for i, r in enumerate(records):
        if r['logloss'] is not None and r['logloss'] > 0:
            # Weight = 1/LL (lower LL = better model = higher weight)
            w = 1.0 / r['logloss']
        else:
            w = 1.0  # fallback: equal weight
        imp = all_imp[i]
        weighted_imp.append(imp * w)
        weights.append(w)

    total_weight = sum(weights)
    weighted_combined = pd.DataFrame(weighted_imp).fillna(0)
    ll_weighted_avg = weighted_combined.sum(axis=0) / total_weight
    ll_weighted_avg = ll_weighted_avg.rename('ll_weighted_importance')

    # ── 3. Rank stability ──
    # For each model, rank features. Then compute mean rank and how often in top-10/20/30.
    all_ranks = []
    for imp_series in all_imp:
        ranks = imp_series.rank(ascending=False, method='min')
        all_ranks.append(ranks)

    ranks_df = pd.DataFrame(all_ranks).fillna(len(raw_avg))  # unranked features get worst rank
    mean_rank = ranks_df.mean(axis=0).rename('mean_rank')
    pct_top10 = (ranks_df <= 10).mean(axis=0).rename('pct_in_top10')
    pct_top20 = (ranks_df <= 20).mean(axis=0).rename('pct_in_top20')
    pct_top30 = (ranks_df <= 30).mean(axis=0).rename('pct_in_top30')

    # ── 4. Combine into final ranking ──
    ranking = pd.DataFrame({
        'raw_avg_importance': raw_avg,
        'll_weighted_importance': ll_weighted_avg,
        'mean_rank': mean_rank,
        'pct_in_top10': pct_top10,
        'pct_in_top20': pct_top20,
        'pct_in_top30': pct_top30,
    })
    ranking.index.name = 'feature'
    ranking = ranking.reset_index()
    ranking = ranking.sort_values('ll_weighted_importance', ascending=False).reset_index(drop=True)
    ranking['overall_rank'] = ranking.index + 1

    # ── 5. Categorize features ──
    def categorize(feat):
        if feat in ['prob_rank', 'prob_vs_favorite', 'prob_share', 'race_herfindahl',
                     'n_close_runners', 'spread_m0', 'spread_rank', 'total_qty_m0',
                     'volume_rank', 'avg_mom_3_1', 'momentum_rank', 'race_overround',
                     'is_favorite', 'prob_deviation', 'bl_imbalance_rank']:
            return 'cross_runner'
        if feat in ['local_dow', 'marketBaseRate', 'numberOfActiveRunners', 'runner_position']:
            return 'fixed_effect'
        if '_mom_' in feat:
            return 'momentum'
        if '_std_' in feat:
            return 'volatility'
        if '_count_' in feat or '_mean_' in feat:
            return 'trade_activity'
        if '_frac' in feat:
            return 'fraction'
        if 'order_is_back' in feat:
            return 'order_direction'
        if feat.endswith('_m0'):
            return 'order_book_snapshot'
        if '_missing' in feat:
            return 'missing_indicator'
        return 'other'

    ranking['category'] = ranking['feature'].apply(categorize)

    # ── Save ──
    ranking.to_parquet(os.path.join(out_dir, 'feature_ranking.parquet'), index=False)
    print(f"\nSaved feature ranking ({len(ranking)} features) to {out_dir}/feature_ranking.parquet", flush=True)

    # ── Print summary ──
    print(f"\n{'='*80}")
    print("TOP 30 FEATURES (by LL-weighted importance)")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'Feature':<45} {'Importance':>10} {'MeanRank':>8} {'Top10%':>6} {'Category':<20}")
    print("-" * 100)
    for _, row in ranking.head(30).iterrows():
        print(f"{row['overall_rank']:>4} {row['feature']:<45} "
              f"{row['ll_weighted_importance']:>10.6f} "
              f"{row['mean_rank']:>8.1f} "
              f"{row['pct_in_top10']:>6.1%} "
              f"{row['category']:<20}", flush=True)

    # ── Category summary ──
    print(f"\n{'='*80}")
    print("FEATURE CATEGORY SUMMARY")
    print(f"{'='*80}")
    cat_summary = ranking.groupby('category').agg(
        n_features=('feature', 'count'),
        total_importance=('ll_weighted_importance', 'sum'),
        best_rank=('overall_rank', 'min'),
    ).sort_values('total_importance', ascending=False)
    print(cat_summary.to_string(), flush=True)

    print(f"\nTotal features: {len(ranking)}", flush=True)
    print(f"Total importance sum: {ranking['ll_weighted_importance'].sum():.4f}", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
