"""
Edge Threshold Optimization Analysis
=====================================
Investigates whether the current edge > 3% betting criterion is optimal.

Uses the local ensemble predictions file (no Spartan needed).
Analyzes ROI, Sharpe, z-stat, total profit, monthly consistency across
a fine grid of edge thresholds.
"""

import pandas as pd
import numpy as np
import os
from math import sqrt, erf

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def analyze_threshold(df, edge_min, bet_size=25.0, commission=0.075):
    """Compute strategy metrics for a given edge threshold."""
    sel = df[df['edge'] >= edge_min].copy()
    n_bets = len(sel)
    if n_bets == 0:
        return {
            'edge_threshold': edge_min, 'n_bets': 0, 'roi': np.nan,
            'total_profit': 0, 'sharpe': np.nan, 'z_stat': np.nan,
            'p_value': np.nan, 'win_rate': np.nan, 'avg_odds': np.nan,
            'avg_edge': np.nan, 'max_drawdown_pct': np.nan,
            'profit_factor': np.nan, 'months_positive': np.nan,
            'months_total': np.nan, 'monthly_consistency': np.nan,
            'days_positive': 0, 'days_total': 0, 'daily_consistency': np.nan,
        }

    # PnL per bet: win pays (odds-1)*bet*(1-commission), lose pays -bet
    sel = sel.copy()
    sel['pnl'] = np.where(
        sel['win'] == 1,
        (sel['back_odds'] - 1) * bet_size * (1 - commission),
        -bet_size
    )

    total_profit = sel['pnl'].sum()
    total_wagered = n_bets * bet_size
    roi = total_profit / total_wagered

    # Sharpe (annualized from daily)
    sel['date'] = pd.to_datetime(sel['marketTime_local'])
    daily_pnl = sel.groupby(sel['date'].dt.date)['pnl'].sum()
    n_days = len(daily_pnl)
    if n_days > 1 and daily_pnl.std() > 0:
        sharpe = daily_pnl.mean() / daily_pnl.std() * sqrt(252)
    else:
        sharpe = np.nan

    # z-stat for ROI > 0
    pnl_arr = sel['pnl'].values
    if pnl_arr.std() > 0:
        z_stat = pnl_arr.mean() / (pnl_arr.std() / sqrt(n_bets))
        p_value = 1 - norm_cdf(z_stat)
    else:
        z_stat, p_value = np.nan, np.nan

    # Max drawdown
    cum_pnl = sel['pnl'].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()
    max_dd_pct = max_dd / bet_size if total_profit != 0 else np.nan

    # Profit factor
    gross_win = sel.loc[sel['pnl'] > 0, 'pnl'].sum()
    gross_loss = -sel.loc[sel['pnl'] < 0, 'pnl'].sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    # Monthly consistency
    monthly_pnl = sel.groupby(sel['date'].dt.to_period('M'))['pnl'].sum()
    months_total = len(monthly_pnl)
    months_positive = (monthly_pnl > 0).sum()
    monthly_consistency = months_positive / months_total if months_total > 0 else np.nan

    # Daily consistency
    days_total = len(daily_pnl)
    days_positive = (daily_pnl > 0).sum()
    daily_consistency = days_positive / days_total if days_total > 0 else np.nan

    return {
        'edge_threshold': edge_min,
        'n_bets': n_bets,
        'roi': roi,
        'total_profit': total_profit,
        'sharpe': sharpe,
        'z_stat': z_stat,
        'p_value': p_value,
        'win_rate': sel['win'].mean(),
        'avg_odds': sel['back_odds'].mean(),
        'avg_edge': sel['edge'].mean(),
        'max_drawdown_dollar': max_dd,
        'max_drawdown_pct': max_dd_pct,
        'profit_factor': profit_factor,
        'months_positive': int(months_positive),
        'months_total': int(months_total),
        'monthly_consistency': monthly_consistency,
        'days_positive': int(days_positive),
        'days_total': int(days_total),
        'daily_consistency': daily_consistency,
    }


def find_optimal_regions(results_df):
    """Identify the optimal thresholds by different criteria."""
    valid = results_df[results_df['n_bets'] >= 50].copy()
    if valid.empty:
        return {}

    optima = {}

    # Max total profit
    idx = valid['total_profit'].idxmax()
    optima['max_profit'] = valid.loc[idx, 'edge_threshold']

    # Max Sharpe
    idx = valid['sharpe'].idxmax()
    optima['max_sharpe'] = valid.loc[idx, 'edge_threshold']

    # Max ROI (with min 100 bets)
    valid100 = valid[valid['n_bets'] >= 100]
    if not valid100.empty:
        idx = valid100['roi'].idxmax()
        optima['max_roi_100bets'] = valid100.loc[idx, 'edge_threshold']

    # Best z-stat
    idx = valid['z_stat'].idxmax()
    optima['max_z_stat'] = valid.loc[idx, 'edge_threshold']

    # Best monthly consistency (with min 200 bets for meaningful months)
    valid200 = valid[valid['n_bets'] >= 200]
    if not valid200.empty:
        idx = valid200['monthly_consistency'].idxmax()
        optima['max_monthly_consistency_200bets'] = valid200.loc[idx, 'edge_threshold']

    # Composite score: normalize and weight
    for col in ['total_profit', 'sharpe', 'z_stat', 'roi']:
        mn, mx = valid[col].min(), valid[col].max()
        if mx > mn:
            valid[f'{col}_norm'] = (valid[col] - mn) / (mx - mn)
        else:
            valid[f'{col}_norm'] = 0.5

    valid['composite'] = (
        0.30 * valid['total_profit_norm'] +
        0.25 * valid['sharpe_norm'] +
        0.25 * valid['z_stat_norm'] +
        0.20 * valid['roi_norm']
    )
    idx = valid['composite'].idxmax()
    optima['composite_best'] = valid.loc[idx, 'edge_threshold']

    return optima


def monthly_breakdown(df, thresholds, bet_size=25.0, commission=0.075):
    """Show monthly PnL for a few key thresholds."""
    records = []
    for thr in thresholds:
        sel = df[df['edge'] >= thr].copy()
        if len(sel) == 0:
            continue
        sel['pnl'] = np.where(
            sel['win'] == 1,
            (sel['back_odds'] - 1) * bet_size * (1 - commission),
            -bet_size
        )
        sel['date'] = pd.to_datetime(sel['marketTime_local'])
        monthly = sel.groupby(sel['date'].dt.to_period('M')).agg(
            n_bets=('pnl', 'count'),
            profit=('pnl', 'sum'),
            win_rate=('win', 'mean'),
        )
        monthly['roi'] = monthly['profit'] / (monthly['n_bets'] * bet_size)
        monthly['threshold'] = thr
        monthly = monthly.reset_index()
        records.append(monthly)
    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


if __name__ == '__main__':
    # Load predictions — handle both local and Spartan paths
    import socket
    hostname = socket.gethostname()
    if hostname == 'UML-FNQ2JDW1GV':
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = '/data/projects/punim2039/alpha_odds'
    pred_path = os.path.join(base_dir, 'res', 'analysis', 'ultimate_cross_t_ensemble_predictions.parquet')
    print(f"Loading predictions from: {pred_path}")
    df = pd.read_parquet(pred_path)
    print(f"Loaded {len(df):,} rows")
    print(f"Edge range: [{df['edge'].min():.4f}, {df['edge'].max():.4f}]")
    print(f"Edge > 0: {(df['edge'] > 0).sum():,} ({(df['edge'] > 0).mean()*100:.1f}%)")
    print()

    # ── Fine grid of thresholds ──
    thresholds = sorted(set(
        list(np.arange(0.000, 0.010, 0.001)) +   # 0.0% to 1.0% by 0.1%
        list(np.arange(0.010, 0.030, 0.002)) +    # 1.0% to 3.0% by 0.2%
        list(np.arange(0.030, 0.060, 0.005)) +    # 3.0% to 6.0% by 0.5%
        list(np.arange(0.060, 0.101, 0.010))       # 6.0% to 10.0% by 1.0%
    ))

    print("=" * 100)
    print("EDGE THRESHOLD SWEEP")
    print("=" * 100)
    print(f"{'Threshold':>10} {'N_bets':>8} {'ROI':>8} {'Profit':>10} {'Sharpe':>8} "
          f"{'z-stat':>8} {'p-value':>10} {'WinRate':>8} {'AvgOdds':>8} {'AvgEdge':>8} "
          f"{'ProfFact':>8} {'Mo+/Mo':>8} {'Day+/Day':>10}")
    print("-" * 112)

    results = []
    for thr in thresholds:
        r = analyze_threshold(df, thr)
        results.append(r)
        if r['n_bets'] > 0:
            print(f"{r['edge_threshold']:>10.3%} {r['n_bets']:>8,} {r['roi']:>8.1%} "
                  f"${r['total_profit']:>9,.0f} {r['sharpe']:>8.2f} "
                  f"{r['z_stat']:>8.2f} {r['p_value']:>10.6f} "
                  f"{r['win_rate']:>8.1%} {r['avg_odds']:>8.2f} {r['avg_edge']:>8.3%} "
                  f"{r['profit_factor']:>8.2f} {r['months_positive']:>3}/{r['months_total']:<3}"
                  f"  {r['days_positive']:>4}/{r['days_total']:<4} ({r['daily_consistency']:>5.1%})")

    results_df = pd.DataFrame(results)

    # ── Optimal thresholds ──
    print()
    print("=" * 100)
    print("OPTIMAL THRESHOLDS")
    print("=" * 100)
    optima = find_optimal_regions(results_df)
    for criterion, thr in optima.items():
        row = results_df[results_df['edge_threshold'] == thr].iloc[0]
        print(f"  {criterion:<40}: edge>{thr:.1%}  →  {row['n_bets']:,} bets, "
              f"ROI={row['roi']:.1%}, Profit=${row['total_profit']:,.0f}, "
              f"Sharpe={row['sharpe']:.2f}, z={row['z_stat']:.2f}")

    # ── Compare current 3% vs best alternatives ──
    print()
    print("=" * 100)
    print("CURRENT (3%) vs KEY ALTERNATIVES")
    print("=" * 100)
    comparison_thresholds = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
    for thr in comparison_thresholds:
        row = results_df.loc[(results_df['edge_threshold'] - thr).abs().idxmin()]
        marker = " <<<" if abs(row['edge_threshold'] - 0.03) < 0.001 else ""
        print(f"  edge>{row['edge_threshold']:.1%}: {row['n_bets']:>6,} bets, "
              f"ROI={row['roi']:>7.1%}, ${row['total_profit']:>8,.0f}, "
              f"Sharpe={row['sharpe']:>6.2f}, z={row['z_stat']:>5.2f}, "
              f"p={row['p_value']:.6f}, "
              f"Mo+={row['months_positive']:.0f}/{row['months_total']:.0f}, "
              f"Day+={row['days_positive']:.0f}/{row['days_total']:.0f} ({row['daily_consistency']:.1%}){marker}")

    # ── Monthly breakdown for key thresholds ──
    print()
    print("=" * 100)
    print("MONTHLY BREAKDOWN (key thresholds)")
    print("=" * 100)
    monthly_thrs = [0.01, 0.02, 0.03, 0.04, 0.05]
    monthly_df = monthly_breakdown(df, monthly_thrs)
    if not monthly_df.empty:
        for thr in monthly_thrs:
            sub = monthly_df[monthly_df['threshold'] == thr]
            if sub.empty:
                continue
            print(f"\n  --- Edge > {thr:.0%} ---")
            for _, row in sub.iterrows():
                bar = "+" if row['profit'] > 0 else "-"
                print(f"    {row['date']}:  {row['n_bets']:>5} bets, "
                      f"${row['profit']:>8,.0f} ({row['roi']:>7.1%}), "
                      f"winrate={row['win_rate']:.1%}  {bar}")

    # ── Diminishing returns analysis ──
    print()
    print("=" * 100)
    print("MARGINAL ANALYSIS: What happens to bets between thresholds?")
    print("=" * 100)
    print("(Shows the quality of bets IN each edge band, not cumulative)")
    print()
    bands = [(0.00, 0.01), (0.01, 0.015), (0.015, 0.02), (0.02, 0.025),
             (0.025, 0.03), (0.03, 0.035), (0.035, 0.04), (0.04, 0.05), (0.05, 0.10)]
    print(f"  {'Band':>14} {'N_bets':>8} {'WinRate':>8} {'ROI':>8} {'Profit':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for lo, hi in bands:
        sel = df[(df['edge'] >= lo) & (df['edge'] < hi)].copy()
        n = len(sel)
        if n == 0:
            continue
        sel['pnl'] = np.where(
            sel['win'] == 1,
            (sel['back_odds'] - 1) * 25.0 * (1 - 0.075),
            -25.0
        )
        profit = sel['pnl'].sum()
        roi = profit / (n * 25.0)
        wr = sel['win'].mean()
        print(f"  {lo:.1%}-{hi:.1%}:  {n:>8,} {wr:>8.1%} {roi:>8.1%} ${profit:>9,.0f}")

    # ── Save results ──
    save_dir = os.path.join(base_dir, 'res', 'edge_analysis')
    os.makedirs(save_dir, exist_ok=True)
    results_df.to_parquet(os.path.join(save_dir, 'edge_threshold_sweep.parquet'), index=False)
    if not monthly_df.empty:
        monthly_df.to_parquet(os.path.join(save_dir, 'monthly_breakdown.parquet'), index=False)
    print()
    print(f"Results saved to {save_dir}/")
    print("Done!")
