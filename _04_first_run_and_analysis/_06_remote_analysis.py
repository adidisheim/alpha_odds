"""
Remote analysis script - runs on Spartan to analyze all model results.
Computes ensembles, backtests, and produces summary statistics.
"""
import pandas as pd
import numpy as np
import os
import json
from math import erf, sqrt

def norm_cdf(x):
    """Standard normal CDF without scipy."""
    return 0.5 * (1 + erf(x / sqrt(2)))

os.chdir("/data/projects/punim2039/alpha_odds/res")

def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ============================================================
# PART 1: V1 TOP-7 ENSEMBLE BACKTEST (t0)
# ============================================================
top_configs = [
    "ne1000_md6_lr0.01", "ne100_md6_lr0.1", "ne500_md3_lr0.05",
    "ne500_md6_lr0.01", "ne100_md6_lr0.05", "ne1000_md3_lr0.05",
    "ne500_md6_lr0.05"
]

model_dfs = {}
for config in top_configs:
    path = f"win_model/t0/{config}/save_df.parquet"
    model_dfs[config] = pd.read_parquet(path)

base = model_dfs[top_configs[0]].copy()
ensemble_prob = np.mean([model_dfs[c]["model_prob"].values for c in top_configs], axis=0)
base["model_prob"] = ensemble_prob
base["edge"] = base["model_prob"] - base["market_prob"]

commission_rate = 0.075  # 7.5% standard Betfair commission for AU greyhounds

print("=" * 80)
print("=== V1 TOP-7 ENSEMBLE BACKTEST (t0) ===")
print("=" * 80)
print(f"Total OOS observations: {len(base):,}")
print(f"Ensemble LL: {log_loss(base['win'].values, base['model_prob'].values):.6f}")
print(f"Market LL:   {log_loss(base['win'].values, base['market_prob'].clip(0.001, 0.999).values):.6f}")
print(f"Commission rate: {commission_rate:.4f}")

base['back_odds'] = 1 / base['orig_best_back_m0']
base = base[(base['back_odds'] > 1.01) & (base['back_odds'] < 1000)]

results_v1 = []
for edge_threshold in [0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.07, 0.10]:
    bets = base[base['edge'] > edge_threshold].copy()
    if len(bets) == 0:
        continue

    bets['pnl'] = bets['win'] * (bets['back_odds'] - 1) * (1 - commission_rate) - (1 - bets['win'])

    n_bets = len(bets)
    total_pnl = bets['pnl'].sum()
    avg_pnl = bets['pnl'].mean()
    win_rate = bets['win'].mean()
    avg_odds = bets['back_odds'].mean()
    roi = avg_pnl * 100

    pnl_std = bets['pnl'].std()
    z_stat = avg_pnl / pnl_std * np.sqrt(n_bets) if pnl_std > 0 else 0
    p_value = 1 - norm_cdf(z_stat)

    bets['month'] = pd.to_datetime(bets['marketTime_local']).dt.to_period('M')
    monthly = bets.groupby('month')['pnl'].sum()
    cum_pnl = monthly.cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()
    monthly_returns = monthly.values
    sharpe_monthly = monthly_returns.mean() / monthly_returns.std() if monthly_returns.std() > 0 else 0
    sharpe_annual = sharpe_monthly * np.sqrt(12)

    total_staked_25 = n_bets * 25
    total_profit_25 = total_pnl * 25
    monthly_profit_25 = total_profit_25 / len(monthly)
    profitable_months = int((monthly > 0).sum())

    print(f"\nEdge > {edge_threshold:.1%}:")
    print(f"  Bets: {n_bets:,}, Win rate: {win_rate:.1%}, Avg odds: {avg_odds:.1f}")
    print(f"  ROI: {roi:+.2f}%, Sharpe: {sharpe_annual:.2f}, z={z_stat:.3f}, p={p_value:.6f}")
    print(f"  Max monthly DD: ${max_dd*25:.0f}, Profitable months: {profitable_months}/{len(monthly)}")
    print(f"  $25/bet: Staked ${total_staked_25:,.0f}, Profit ${total_profit_25:,.0f}, ${monthly_profit_25:,.0f}/month")

    results_v1.append({
        'edge': edge_threshold, 'n_bets': n_bets, 'win_rate': round(win_rate, 4),
        'avg_odds': round(avg_odds, 1), 'roi_pct': round(roi, 2), 'sharpe': round(sharpe_annual, 2),
        'z_stat': round(z_stat, 3), 'p_value': round(p_value, 6), 'max_dd_25': round(max_dd*25, 0),
        'profit_25': round(total_profit_25, 0), 'monthly_profit_25': round(monthly_profit_25, 0),
        'profitable_months': f'{profitable_months}/{len(monthly)}'
    })

# ============================================================
# PART 2: Monthly breakdown for best edge threshold
# ============================================================
print("\n" + "=" * 80)
print("=== MONTHLY BREAKDOWN (Edge > 2.5%) ===")
print("=" * 80)

bets = base[base['edge'] > 0.025].copy()
bets['pnl'] = bets['win'] * (bets['back_odds'] - 1) * (1 - commission_rate) - (1 - bets['win'])
bets['month'] = pd.to_datetime(bets['marketTime_local']).dt.to_period('M')

monthly = bets.groupby('month').agg(
    n_bets=('pnl', 'count'),
    win_rate=('win', 'mean'),
    total_pnl=('pnl', 'sum'),
    avg_odds=('back_odds', 'mean'),
    avg_edge=('edge', 'mean'),
).reset_index()
monthly['roi_pct'] = (monthly['total_pnl'] / monthly['n_bets']) * 100
monthly['cum_pnl'] = monthly['total_pnl'].cumsum()
monthly['profit_25'] = monthly['total_pnl'] * 25
monthly['cum_profit_25'] = monthly['cum_pnl'] * 25

print(monthly.to_string(index=False))

# ============================================================
# PART 3: Edge source analysis - where does performance come from?
# ============================================================
print("\n" + "=" * 80)
print("=== EDGE SOURCE ANALYSIS ===")
print("=" * 80)

# By odds range
bets_all = base.copy()
bets_all['pnl'] = bets_all['win'] * (bets_all['back_odds'] - 1) * (1 - commission_rate) - (1 - bets_all['win'])
bets_all['odds_range'] = pd.cut(bets_all['back_odds'], bins=[1, 3, 5, 8, 15, 30, 1000], labels=['1-3', '3-5', '5-8', '8-15', '15-30', '30+'])

odds_analysis = bets_all.groupby('odds_range').agg(
    count=('pnl', 'count'),
    win_rate=('win', 'mean'),
    model_win_rate=('model_prob', 'mean'),
    market_win_rate=('market_prob', 'mean'),
    avg_edge=('edge', 'mean'),
    roi=('pnl', 'mean'),
).reset_index()
odds_analysis['roi'] = odds_analysis['roi'] * 100
print("\nPerformance by odds range:")
print(odds_analysis.to_string(index=False))

# By time of day
bets_all['hour'] = pd.to_datetime(bets_all['marketTime_local']).dt.hour
time_analysis = bets_all[bets_all['edge'] > 0.02].groupby('hour').agg(
    count=('pnl', 'count'),
    win_rate=('win', 'mean'),
    roi=('pnl', 'mean'),
).reset_index()
time_analysis['roi'] = time_analysis['roi'] * 100
print("\nPerformance by hour (edge > 2%):")
print(time_analysis.to_string(index=False))

# ============================================================
# PART 4: LAY side opportunity
# ============================================================
print("\n" + "=" * 80)
print("=== LAY SIDE ANALYSIS ===")
print("=" * 80)

# For lay: we want market_prob > model_prob (overpriced runners)
base['lay_edge'] = base['market_prob'] - base['model_prob']
base['lay_odds'] = 1 / base['orig_best_lay_m0']
base_lay = base[(base['lay_odds'] > 1.01) & (base['lay_odds'] < 1000)].copy()

for edge_threshold in [0.01, 0.02, 0.03, 0.05]:
    lay_bets = base_lay[base_lay['lay_edge'] > edge_threshold].copy()
    if len(lay_bets) == 0:
        continue

    # Lay P&L: win when horse loses, lose when horse wins
    # Lay: profit = stake * commission_adjusted, loss = stake * (odds - 1)
    lay_bets['pnl'] = (1 - lay_bets['win']) * (1 - commission_rate) - lay_bets['win'] * (lay_bets['lay_odds'] - 1)

    n_bets = len(lay_bets)
    roi = lay_bets['pnl'].mean() * 100
    win_rate = (1 - lay_bets['win']).mean()  # lay wins when horse loses
    z_stat = lay_bets['pnl'].mean() / lay_bets['pnl'].std() * np.sqrt(n_bets) if lay_bets['pnl'].std() > 0 else 0
    p_value = 1 - norm_cdf(z_stat)

    print(f"\nLay edge > {edge_threshold:.1%}:")
    print(f"  Bets: {n_bets:,}, Win rate: {win_rate:.1%}, Avg lay odds: {lay_bets['lay_odds'].mean():.1f}")
    print(f"  ROI: {roi:+.2f}%, z={z_stat:.3f}, p={p_value:.6f}")
    print(f"  $25/bet profit: ${lay_bets['pnl'].sum()*25:,.0f}")

# ============================================================
# PART 5: Summary comparison table
# ============================================================
print("\n" + "=" * 80)
print("=== FINAL SUMMARY TABLE ===")
print("=" * 80)
df_res = pd.DataFrame(results_v1)
print(df_res.to_string(index=False))

# Save results
save_dir = "/data/projects/punim2039/alpha_odds/res/analysis/"
os.makedirs(save_dir, exist_ok=True)
df_res.to_parquet(save_dir + "v1_backtest_summary.parquet")
print(f"\nResults saved to {save_dir}")
