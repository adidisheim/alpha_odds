"""
Build full cross-t super-ensembles for each feature count and compare.

Exact replica of _10_ultimate_ensemble.py logic:
  - V2: top-15 by log-loss per t_def, averaged cross-t
  - V1: top-7 by log-loss per t_def, averaged cross-t (unchanged, full features)
  - Super: V1(20%) + V2(80%)

Feature counts: 5, 10, 20, 30, plus "all" (original V2 from win_model_v2/)

Usage:
    python _06_full_ensemble.py
"""

import os
import numpy as np
import pandas as pd
from math import erf, sqrt

from parameters import Constant


FEATURE_COUNTS = [5, 10, 20, 30]
T_DEFS = [0, 1, 2, 3]


def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def load_ensemble_from_dir(base_dir, t_def, top_n, prob_col='model_prob'):
    """Load models for a given t_def and return top-N ensemble probs + base df."""
    t_dir = os.path.join(base_dir, f't{t_def}')
    if not os.path.isdir(t_dir):
        return None, None
    dfs = {}
    for c in sorted(os.listdir(t_dir)):
        path = os.path.join(t_dir, c, 'save_df.parquet')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
            if 'win' not in df.columns or prob_col not in df.columns:
                continue
            dfs[c] = df
        except Exception:
            pass
    if not dfs:
        return None, None

    sorted_configs = sorted(dfs.keys(), key=lambda c: log_loss(
        dfs[c]['win'].values, dfs[c][prob_col].clip(0.001, 0.999).values))
    n = min(top_n, len(sorted_configs))
    base = dfs[sorted_configs[0]].copy()
    ens_prob = np.mean([dfs[c][prob_col].values for c in sorted_configs[:n]], axis=0)
    return base, ens_prob, n, len(dfs)


def build_cross_t(base_dir, top_n, label, prob_col='model_prob'):
    """Build cross-t ensemble from a model directory."""
    probs_by_t = {}
    for t in T_DEFS:
        result = load_ensemble_from_dir(base_dir, t, top_n, prob_col)
        if result[0] is None:
            continue
        base, ens_prob, n_used, n_total = result
        base['key'] = base['file_name'].astype(str) + '_' + base['id'].astype(str)
        ll = log_loss(base['win'].values, ens_prob)
        probs_by_t[t] = (base, ens_prob)
        print(f"    {label} t{t}: LL={ll:.6f} (top-{n_used}/{n_total}, {len(base):,} rows)", flush=True)

    if len(probs_by_t) < 2:
        return None

    # Align across t_defs
    first_t = list(probs_by_t.keys())[0]
    base_df, first_prob = probs_by_t[first_t]
    aligned = base_df[['key', 'win', 'market_prob', 'orig_best_back_m0',
                        'orig_best_lay_m0', 'marketTime_local']].copy()
    aligned[f'{label}_t{first_t}'] = first_prob

    for t in list(probs_by_t.keys())[1:]:
        df_t, prob_t = probs_by_t[t]
        df_t['key'] = df_t['file_name'].astype(str) + '_' + df_t['id'].astype(str)
        col = f'{label}_t{t}'
        tmp = pd.DataFrame({'key': df_t['key'].values, col: prob_t})
        aligned = aligned.merge(tmp, on='key', how='inner')

    cols = [c for c in aligned.columns if c.startswith(f'{label}_t')]
    aligned[f'{label}_cross_t'] = aligned[cols].mean(axis=1)
    ll = log_loss(aligned['win'].values, aligned[f'{label}_cross_t'].values)
    print(f"    {label} cross-t ({len(cols)} t_defs): LL={ll:.6f} ({len(aligned):,} rows)", flush=True)

    return aligned


def run_backtest(bt, label, commission=0.075):
    """Run value betting backtest and return results."""
    bt = bt.copy()
    bt['edge'] = bt['model_prob'] - bt['market_prob']
    bt['back_odds'] = 1 / bt['orig_best_back_m0']
    bt = bt[(bt['back_odds'] > 1.01) & (bt['back_odds'] < 1000)]

    results = []
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

        bets['month'] = pd.to_datetime(bets['marketTime_local']).dt.to_period('M')
        monthly = bets.groupby('month')['pnl'].sum()
        sh_a = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
        pm = int((monthly > 0).sum())

        results.append({
            'edge_threshold': et, 'n_bets': n, 'win_rate': wr,
            'roi_pct': roi, 'sharpe_annual': sh_a, 'z_stat': z,
            'p_value': p, 'profit_25': profit_25,
            'profitable_months': pm, 'total_months': len(monthly),
        })
    return results


def main():
    study_dir = f'{Constant.RES_DIR}/feature_importance_study'

    # ══════════════════════════════════════════════
    # V1 cross-t (unchanged, full features)
    # ══════════════════════════════════════════════
    print("=" * 80, flush=True)
    print("Loading V1 cross-t (full features, top-7 per t_def)...", flush=True)
    v1_aligned = build_cross_t(f'{Constant.RES_DIR}/win_model', top_n=7, label='V1')

    # ══════════════════════════════════════════════
    # Original V2 (all features) — the baseline
    # ══════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("Loading original V2 (all 105 features, top-15 per t_def)...", flush=True)
    v2_all_aligned = build_cross_t(f'{Constant.RES_DIR}/win_model_v2', top_n=15, label='V2_all')

    # ══════════════════════════════════════════════
    # Retrained V2 with restricted features
    # ══════════════════════════════════════════════
    all_results = []

    for n_feat in FEATURE_COUNTS:
        print(f"\n{'='*80}", flush=True)
        print(f"V2 with top-{n_feat} features (top-15 per t_def)...", flush=True)

        retrain_dir = os.path.join(study_dir, 'full_retrain', f'n{n_feat}')
        v2_feat_aligned = build_cross_t(retrain_dir, top_n=15, label=f'V2_n{n_feat}')

        if v2_feat_aligned is None:
            print(f"  SKIPPED: not enough models for n={n_feat}", flush=True)
            continue

        # Blend with V1
        if v1_aligned is not None:
            bt = v2_feat_aligned.merge(
                v1_aligned[['key', 'V1_cross_t']], on='key', how='inner')
            bt['model_prob'] = 0.2 * bt['V1_cross_t'] + 0.8 * bt[f'V2_n{n_feat}_cross_t']
        else:
            bt = v2_feat_aligned.copy()
            bt['model_prob'] = bt[f'V2_n{n_feat}_cross_t']

        super_ll = log_loss(bt['win'].values, bt['model_prob'].values)
        market_ll = log_loss(bt['win'].values, bt['market_prob'].clip(0.001, 0.999).values)
        print(f"  Super-ensemble: LL={super_ll:.6f} (market: {market_ll:.6f}, {len(bt):,} rows)", flush=True)

        results = run_backtest(bt, f'n{n_feat}')
        for r in results:
            r['n_features'] = n_feat
            r['feat_label'] = str(n_feat)
            r['super_logloss'] = super_ll
            r['market_logloss'] = market_ll
            r['n_oos'] = len(bt)
            all_results.append(r)

            if r['edge_threshold'] in [0.03, 0.05]:
                print(f"  Edge>{r['edge_threshold']:.0%}: {r['n_bets']:,} bets, "
                      f"ROI={r['roi_pct']:+.1f}%, Sharpe={r['sharpe_annual']:.1f}, "
                      f"z={r['z_stat']:.2f}, p={r['p_value']:.6f}, "
                      f"${r['profit_25']:,.0f} @$25", flush=True)

    # ══════════════════════════════════════════════
    # Original all-features super-ensemble
    # ══════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("Original V2 (all 105 features) super-ensemble...", flush=True)

    if v2_all_aligned is not None and v1_aligned is not None:
        bt_all = v2_all_aligned.merge(
            v1_aligned[['key', 'V1_cross_t']], on='key', how='inner')
        bt_all['model_prob'] = 0.2 * bt_all['V1_cross_t'] + 0.8 * bt_all['V2_all_cross_t']

        super_ll = log_loss(bt_all['win'].values, bt_all['model_prob'].values)
        market_ll = log_loss(bt_all['win'].values, bt_all['market_prob'].clip(0.001, 0.999).values)
        print(f"  Super-ensemble: LL={super_ll:.6f} (market: {market_ll:.6f}, {len(bt_all):,} rows)", flush=True)

        results = run_backtest(bt_all, 'all')
        for r in results:
            r['n_features'] = 105
            r['feat_label'] = 'all'
            r['super_logloss'] = super_ll
            r['market_logloss'] = market_ll
            r['n_oos'] = len(bt_all)
            all_results.append(r)

            if r['edge_threshold'] in [0.03, 0.05]:
                print(f"  Edge>{r['edge_threshold']:.0%}: {r['n_bets']:,} bets, "
                      f"ROI={r['roi_pct']:+.1f}%, Sharpe={r['sharpe_annual']:.1f}, "
                      f"z={r['z_stat']:.2f}, p={r['p_value']:.6f}, "
                      f"${r['profit_25']:,.0f} @$25", flush=True)

    # ══════════════════════════════════════════════
    # Save and print comparison
    # ══════════════════════════════════════════════
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(os.path.join(study_dir, 'full_ensemble_results.parquet'), index=False)

    print(f"\n{'='*100}", flush=True)
    print("FULL SUPER-ENSEMBLE COMPARISON (V1 20% + V2 80%, top-15 per t_def, cross-t)", flush=True)
    print(f"{'='*100}", flush=True)

    for et in [0.02, 0.025, 0.03, 0.05, 0.07]:
        df_et = results_df[results_df['edge_threshold'] == et]
        if len(df_et) == 0:
            continue
        print(f"\n--- Edge > {et:.1%} ---", flush=True)
        print(f"{'Features':>8} | {'Bets':>6} | {'ROI%':>7} | {'Sharpe':>7} | "
              f"{'z':>6} | {'p-value':>8} | {'$25/bet':>10} | {'LL':>10} | "
              f"{'ProfMo':>6}", flush=True)
        print("-" * 90, flush=True)
        for _, row in df_et.sort_values('n_features').iterrows():
            print(f"{row['feat_label']:>8} | {int(row['n_bets']):>6} | "
                  f"{row['roi_pct']:>+7.1f} | {row['sharpe_annual']:>7.1f} | "
                  f"{row['z_stat']:>6.2f} | {row['p_value']:>8.6f} | "
                  f"${row['profit_25']:>9,.0f} | {row['super_logloss']:>10.6f} | "
                  f"{int(row['profitable_months'])}/{int(row['total_months'])}", flush=True)

    print(f"\nResults saved to {study_dir}/full_ensemble_results.parquet", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
