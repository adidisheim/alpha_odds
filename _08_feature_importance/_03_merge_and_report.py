"""
Phase 3: Merge ablation results and produce comparison report.

Merges all 32 ablation parquets, finds the performance elbow, and lists minimal feature sets.

Usage:
    python _03_merge_and_report.py

Output:
    res/feature_importance_study/ablation_merged.parquet
    res/feature_importance_study/ablation_report.txt
"""

import os
import numpy as np
import pandas as pd

from parameters import Constant


def main():
    study_dir = f'{Constant.RES_DIR}/feature_importance_study'

    # ── 1. Merge all ablation results ──
    parts = []
    for task_id in range(32):
        path = os.path.join(study_dir, f'ablation_{task_id}.parquet')
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
        else:
            print(f"WARNING: Missing ablation_{task_id}.parquet", flush=True)

    if not parts:
        print("ERROR: No ablation results found!", flush=True)
        return

    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(os.path.join(study_dir, 'ablation_merged.parquet'), index=False)
    print(f"Merged {len(parts)} ablation files -> {len(df)} rows", flush=True)

    # ── 2. Load feature ranking for reference ──
    ranking_path = os.path.join(study_dir, 'feature_ranking.parquet')
    ranking = pd.read_parquet(ranking_path) if os.path.exists(ranking_path) else None

    # ── 3. Build comparison tables ──
    lines = []
    lines.append("=" * 100)
    lines.append("FEATURE IMPORTANCE ABLATION STUDY — RESULTS REPORT")
    lines.append("=" * 100)
    lines.append("")

    # Summary table at edge threshold = 3% (the main operating point)
    lines.append("=" * 100)
    lines.append("TABLE 1: Performance by Feature Count (edge > 3%)")
    lines.append("=" * 100)
    lines.append("")

    et_main = 0.03
    df_3 = df[df['edge_threshold'] == et_main].copy()

    if len(df_3) > 0:
        header = f"{'n_feat':>6} | {'t_def':>5} | {'Cal_LL':>10} | {'Mkt_LL':>10} | {'LL_gain':>8} | {'Brier':>8} | {'Bets':>6} | {'ROI%':>7} | {'Sharpe':>7} | {'z':>6} | {'p':>8} | {'$25/bet':>10}"
        lines.append(header)
        lines.append("-" * len(header))

        for _, row in df_3.sort_values(['n_features_target', 't_definition']).iterrows():
            ll_gain = row.get('market_logloss', 0) - row.get('calibrated_logloss', 0)
            lines.append(
                f"{int(row['n_features_target']):>6} | "
                f"{int(row['t_definition']):>5} | "
                f"{row.get('calibrated_logloss', np.nan):>10.6f} | "
                f"{row.get('market_logloss', np.nan):>10.6f} | "
                f"{ll_gain:>8.6f} | "
                f"{row.get('calibrated_brier', np.nan):>8.6f} | "
                f"{int(row.get('n_bets', 0)):>6} | "
                f"{row.get('roi_pct', np.nan):>+7.1f} | "
                f"{row.get('sharpe_annual', row.get('sharpe', np.nan)):>7.1f} | "
                f"{row.get('z_stat', np.nan):>6.2f} | "
                f"{row.get('p_value', np.nan):>8.4f} | "
                f"${row.get('profit_25', 0):>9,.0f}"
            )
        lines.append("")

    # ── 4. Average across t_defs for each feature count ──
    lines.append("=" * 100)
    lines.append("TABLE 2: Average Performance Across All t_defs (edge > 3%)")
    lines.append("=" * 100)
    lines.append("")

    if len(df_3) > 0:
        avg = df_3.groupby('n_features_target').agg(
            cal_ll_mean=('calibrated_logloss', 'mean'),
            cal_ll_std=('calibrated_logloss', 'std'),
            mkt_ll_mean=('market_logloss', 'mean'),
            brier_mean=('calibrated_brier', 'mean'),
            n_bets_mean=('n_bets', 'mean'),
            roi_mean=('roi_pct', 'mean'),
            roi_std=('roi_pct', 'std'),
            sharpe_mean=('sharpe', 'mean'),
            profit_mean=('profit_25', 'mean'),
        ).reset_index().sort_values('n_features_target')

        header2 = f"{'n_feat':>6} | {'Avg_LL':>10} | {'LL_std':>8} | {'LL_gain':>8} | {'Brier':>8} | {'Avg_bets':>8} | {'ROI%':>7} | {'ROI_std':>7} | {'Sharpe':>7} | {'Avg_$25':>10}"
        lines.append(header2)
        lines.append("-" * len(header2))

        for _, row in avg.iterrows():
            ll_gain = row['mkt_ll_mean'] - row['cal_ll_mean']
            lines.append(
                f"{int(row['n_features_target']):>6} | "
                f"{row['cal_ll_mean']:>10.6f} | "
                f"{row['cal_ll_std']:>8.6f} | "
                f"{ll_gain:>8.6f} | "
                f"{row['brier_mean']:>8.6f} | "
                f"{row['n_bets_mean']:>8.0f} | "
                f"{row['roi_mean']:>+7.1f} | "
                f"{row['roi_std']:>7.1f} | "
                f"{row['sharpe_mean']:>7.1f} | "
                f"${row['profit_mean']:>9,.0f}"
            )
        lines.append("")

    # ── 5. Find the elbow ──
    lines.append("=" * 100)
    lines.append("ELBOW DETECTION: Smallest N within 1% of full model")
    lines.append("=" * 100)
    lines.append("")

    if len(df_3) > 0:
        # Use calibrated log-loss averaged across t_defs as the metric
        avg_ll = df_3.groupby('n_features_target')['calibrated_logloss'].mean()
        full_ll = avg_ll.max()  # max n_features = all features
        best_ll = avg_ll.min()

        # Find highest n_features (the "all" baseline)
        max_n = avg_ll.index.max()
        full_model_ll = avg_ll.loc[max_n]

        lines.append(f"Full model ({max_n} features) avg LL: {full_model_ll:.6f}")
        lines.append(f"1% threshold: {full_model_ll * 1.01:.6f}")
        lines.append("")

        threshold = full_model_ll * 1.01  # within 1% of full model
        candidates = avg_ll[avg_ll <= threshold].index
        if len(candidates) > 0:
            elbow_n = candidates.min()
            lines.append(f">>> ELBOW: {elbow_n} features (LL={avg_ll.loc[elbow_n]:.6f})")
            lines.append(f"    vs full model LL={full_model_ll:.6f} ({(avg_ll.loc[elbow_n] / full_model_ll - 1) * 100:+.3f}%)")
        else:
            elbow_n = max_n
            lines.append(f">>> No subset within 1% threshold — full model required.")
        lines.append("")

        # Also find elbow by ROI
        avg_roi = df_3.groupby('n_features_target')['roi_pct'].mean()
        full_roi = avg_roi.loc[max_n] if max_n in avg_roi.index else 0
        lines.append(f"Full model avg ROI@3%: {full_roi:+.1f}%")
        roi_threshold = full_roi * 0.90  # within 90% of full ROI
        roi_candidates = avg_roi[avg_roi >= roi_threshold].index
        if len(roi_candidates) > 0:
            roi_elbow = roi_candidates.min()
            lines.append(f">>> ROI elbow: {roi_elbow} features (ROI={avg_roi.loc[roi_elbow]:+.1f}%)")
        lines.append("")

    # ── 6. List features in the minimal set ──
    lines.append("=" * 100)
    lines.append("MINIMAL FEATURE SET")
    lines.append("=" * 100)
    lines.append("")

    if ranking is not None and 'elbow_n' in dir():
        minimal = ranking.head(elbow_n)
        lines.append(f"Top {elbow_n} features:")
        lines.append("")
        for _, row in minimal.iterrows():
            lines.append(f"  {row['overall_rank']:>3}. {row['feature']:<45} "
                         f"imp={row['ll_weighted_importance']:.6f}  "
                         f"cat={row['category']}")
        lines.append("")
        cat_counts = minimal['category'].value_counts()
        lines.append("Category breakdown:")
        for cat, cnt in cat_counts.items():
            lines.append(f"  {cat}: {cnt}")

    # ── 7. Multi-threshold comparison ──
    lines.append("")
    lines.append("=" * 100)
    lines.append("TABLE 3: Full Results Across All Edge Thresholds (averaged across t_defs)")
    lines.append("=" * 100)
    lines.append("")

    for et in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        df_et = df[df['edge_threshold'] == et]
        if len(df_et) == 0:
            continue
        lines.append(f"--- Edge > {et:.0%} ---")
        avg_et = df_et.groupby('n_features_target').agg(
            n_bets=('n_bets', 'mean'),
            roi=('roi_pct', 'mean'),
            profit=('profit_25', 'mean'),
        ).reset_index().sort_values('n_features_target')
        for _, row in avg_et.iterrows():
            lines.append(f"  {int(row['n_features_target']):>4} feat: "
                         f"{row['n_bets']:>5.0f} bets, "
                         f"ROI={row['roi']:>+6.1f}%, "
                         f"$25/bet=${row['profit']:>8,.0f}")
        lines.append("")

    # ── Save report ──
    report = '\n'.join(lines)
    report_path = os.path.join(study_dir, 'ablation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    print(report, flush=True)
    print(f"\nReport saved to {report_path}", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
