"""
Phase 4: Build cross-t super-ensemble for each feature count and backtest.

For each feature count (5, 10, 15, 20, 30, 50, 80, all):
  1. Load per-runner predictions from all 4 t_defs
  2. Align by (file_name, id) key — inner join
  3. Average across t_defs to get V2_cross_t
  4. Blend with original V1_cross_t (20/80) from the full super-ensemble
  5. Run value betting backtest at standard edge thresholds

This replicates the _10_ultimate_ensemble.py approach for fair comparison.

Usage:
    python _04_cross_t_ensemble.py
"""

import os
import numpy as np
import pandas as pd
from math import erf, sqrt

from parameters import Constant


FEATURE_COUNTS = [5, 10, 15, 20, 30, 50, 80, None]
N_FEATURE_COUNTS = len(FEATURE_COUNTS)
T_DEFS = [0, 1, 2, 3]


def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def load_v1_cross_t():
    """Load the original V1 cross-t ensemble (unchanged regardless of feature count)."""
    v1_base_dir = f'{Constant.RES_DIR}/win_model'
    v1_probs_by_t = {}

    for t in T_DEFS:
        t_dir = os.path.join(v1_base_dir, f't{t}')
        if not os.path.isdir(t_dir):
            continue
        dfs = {}
        for config in sorted(os.listdir(t_dir)):
            path = os.path.join(t_dir, config, 'save_df.parquet')
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                if 'win' not in df.columns or 'model_prob' not in df.columns:
                    continue
                dfs[config] = df
            except Exception:
                continue

        if not dfs:
            continue

        # V1 top-7 by log-loss
        sorted_configs = sorted(dfs.keys(), key=lambda c: log_loss(
            dfs[c]['win'].values, np.clip(dfs[c]['model_prob'].values, 0.001, 0.999)))
        n = min(7, len(sorted_configs))

        base = dfs[sorted_configs[0]].copy()
        base['key'] = base['file_name'].astype(str) + '_' + base['id'].astype(str)
        ens_prob = np.mean([dfs[c]['model_prob'].values for c in sorted_configs[:n]], axis=0)
        base['v1_prob'] = ens_prob
        v1_probs_by_t[t] = base[['key', 'v1_prob', 'win']].copy()
        ll = log_loss(base['win'].values, ens_prob)
        print(f"  V1 t{t}: LL={ll:.6f} ({len(base):,} rows, top-{n} models)", flush=True)

    if not v1_probs_by_t:
        print("  WARNING: No V1 models found, will use V2-only ensemble", flush=True)
        return None

    # Align across t_defs and average
    aligned = v1_probs_by_t[T_DEFS[0]].rename(columns={'v1_prob': f'v1_t{T_DEFS[0]}'})
    for t in T_DEFS[1:]:
        if t not in v1_probs_by_t:
            continue
        tmp = v1_probs_by_t[t].rename(columns={'v1_prob': f'v1_t{t}'})
        aligned = aligned.merge(tmp[['key', f'v1_t{t}']], on='key', how='inner')

    v1_cols = [c for c in aligned.columns if c.startswith('v1_t')]
    aligned['v1_cross_t'] = aligned[v1_cols].mean(axis=1)
    ll = log_loss(aligned['win'].values, aligned['v1_cross_t'].values)
    print(f"  V1 cross-t ({len(v1_cols)} t_defs): LL={ll:.6f} ({len(aligned):,} rows)", flush=True)

    return aligned[['key', 'v1_cross_t', 'win']]


def main():
    study_dir = f'{Constant.RES_DIR}/feature_importance_study'

    # ── Load V1 cross-t (stays constant) ──
    print("Loading V1 cross-t ensemble...", flush=True)
    v1_df = load_v1_cross_t()

    # ── For each feature count, build V2 cross-t and blend ──
    all_results = []

    for feat_idx, n_feat in enumerate(FEATURE_COUNTS):
        feat_label = str(n_feat) if n_feat else 'all'
        print(f"\n{'='*80}", flush=True)
        print(f"Feature count: {feat_label}", flush=True)
        print(f"{'='*80}", flush=True)

        # Load predictions for all 4 t_defs
        v2_by_t = {}
        for t_idx, t_def in enumerate(T_DEFS):
            task_id = feat_idx + t_idx * N_FEATURE_COUNTS
            pred_path = os.path.join(study_dir, f'predictions_{task_id}.parquet')
            if not os.path.exists(pred_path):
                print(f"  WARNING: Missing predictions_{task_id}.parquet (t_def={t_def})", flush=True)
                continue
            df = pd.read_parquet(pred_path)
            df['key'] = df['file_name'].astype(str) + '_' + df['id'].astype(str)
            v2_by_t[t_def] = df
            ll = log_loss(df['win'].values, df['model_prob'].clip(0.001, 0.999).values)
            print(f"  V2 t{t_def}: LL={ll:.6f} ({len(df):,} rows)", flush=True)

        if len(v2_by_t) < 2:
            print(f"  ERROR: Need at least 2 t_defs, got {len(v2_by_t)}. Skipping.", flush=True)
            continue

        # Align V2 across t_defs
        first_t = list(v2_by_t.keys())[0]
        v2_aligned = v2_by_t[first_t][['key', 'win', 'market_prob',
                                        'orig_best_back_m0', 'marketTime_local']].copy()
        v2_aligned[f'v2_t{first_t}'] = v2_by_t[first_t]['model_prob'].values

        for t_def in list(v2_by_t.keys())[1:]:
            tmp = v2_by_t[t_def][['key', 'model_prob']].copy()
            tmp = tmp.rename(columns={'model_prob': f'v2_t{t_def}'})
            v2_aligned = v2_aligned.merge(tmp, on='key', how='inner')

        v2_cols = [c for c in v2_aligned.columns if c.startswith('v2_t')]
        v2_aligned['v2_cross_t'] = v2_aligned[v2_cols].mean(axis=1)
        v2_ll = log_loss(v2_aligned['win'].values, v2_aligned['v2_cross_t'].values)
        print(f"  V2 cross-t ({len(v2_cols)} t_defs): LL={v2_ll:.6f} ({len(v2_aligned):,} rows)", flush=True)

        # Blend with V1 cross-t (20/80)
        if v1_df is not None:
            bt = v2_aligned.merge(v1_df[['key', 'v1_cross_t']], on='key', how='inner')
            bt['model_prob'] = 0.2 * bt['v1_cross_t'] + 0.8 * bt['v2_cross_t']
            ensemble_label = f"V1(20%)+V2(80%) cross-t, {feat_label} features"
        else:
            bt = v2_aligned.copy()
            bt['model_prob'] = bt['v2_cross_t']
            ensemble_label = f"V2-only cross-t, {feat_label} features"

        super_ll = log_loss(bt['win'].values, bt['model_prob'].values)
        market_ll = log_loss(bt['win'].values, bt['market_prob'].clip(0.001, 0.999).values)
        print(f"  Super-ensemble: LL={super_ll:.6f} (market: {market_ll:.6f})", flush=True)
        print(f"  Rows in final ensemble: {len(bt):,}", flush=True)

        # ── Backtest ──
        bt['edge'] = bt['model_prob'] - bt['market_prob']
        bt['back_odds'] = 1 / bt['orig_best_back_m0']
        bt = bt[(bt['back_odds'] > 1.01) & (bt['back_odds'] < 1000)]

        commission = 0.075

        for et in [0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.07, 0.10]:
            bets = bt[bt['edge'] > et].copy()
            if len(bets) < 10:
                continue

            bets['pnl'] = bets['win'] * (bets['back_odds'] - 1) * (1 - commission) - (1 - bets['win'])
            n = len(bets)
            wr = bets['win'].mean()
            roi = bets['pnl'].mean() * 100
            pnl_std = bets['pnl'].std()
            z = bets['pnl'].mean() / pnl_std * np.sqrt(n) if pnl_std > 0 else 0
            p = 1 - norm_cdf(z)
            profit_25 = bets['pnl'].sum() * 25

            # Monthly Sharpe
            bets['month'] = pd.to_datetime(bets['marketTime_local']).dt.to_period('M')
            monthly = bets.groupby('month')['pnl'].sum()
            sh_m = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
            pm = int((monthly > 0).sum())

            print(f"  Edge>{et:.1%}: {n:,} bets, ROI={roi:+.1f}%, "
                  f"Sharpe={sh_m:.1f}, z={z:.2f}, p={p:.6f}, "
                  f"${profit_25:,.0f} @$25", flush=True)

            actual_n = n_feat if n_feat else 105
            all_results.append({
                'n_features': actual_n,
                'feat_label': feat_label,
                'edge_threshold': et,
                'n_bets': n,
                'win_rate': wr,
                'roi_pct': roi,
                'sharpe_annual': sh_m,
                'z_stat': z,
                'p_value': p,
                'profit_25': profit_25,
                'profitable_months': pm,
                'total_months': len(monthly),
                'super_logloss': super_ll,
                'v2_cross_logloss': v2_ll,
                'market_logloss': market_ll,
                'n_oos_aligned': len(bt),
            })

    # ── Save results ──
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(os.path.join(study_dir, 'cross_t_ensemble_results.parquet'), index=False)

    # ── Print comparison table at key edge thresholds ──
    print(f"\n{'='*100}", flush=True)
    print("CROSS-T SUPER-ENSEMBLE COMPARISON", flush=True)
    print(f"{'='*100}", flush=True)

    for et in [0.02, 0.025, 0.03, 0.05, 0.07]:
        df_et = results_df[results_df['edge_threshold'] == et]
        if len(df_et) == 0:
            continue
        print(f"\n--- Edge > {et:.1%} ---", flush=True)
        print(f"{'Features':>8} | {'Bets':>6} | {'ROI%':>7} | {'Sharpe':>7} | {'z':>6} | {'p-value':>8} | {'$25/bet':>10} | {'LL':>10}", flush=True)
        print("-" * 80, flush=True)
        for _, row in df_et.sort_values('n_features').iterrows():
            print(f"{row['feat_label']:>8} | {int(row['n_bets']):>6} | {row['roi_pct']:>+7.1f} | "
                  f"{row['sharpe_annual']:>7.1f} | {row['z_stat']:>6.2f} | {row['p_value']:>8.4f} | "
                  f"${row['profit_25']:>9,.0f} | {row['super_logloss']:>10.6f}", flush=True)

    print(f"\nResults saved to {study_dir}/cross_t_ensemble_results.parquet", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
