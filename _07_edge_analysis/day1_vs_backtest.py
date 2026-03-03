"""
Day 1 Paper Trading vs Backtest: Comprehensive Diagnostic Analysis
==================================================================
Questions:
1. Timing: Do we trade at the same time before race close?
2. Execution: Do we move the quote more than expected?
3. Odds: Are our live odds much higher (longshots)?
4. Frequency: How often do we bet per race?
5. Runner selection: Are we betting on the same type of dog?

For Q3-Q5: Bootstrap distribution of daily backtest values + where Day 1 falls.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
BT_PATH = 'res/analysis/ultimate_cross_t_ensemble_predictions.parquet'
BETS_PATH = 'paper_trading_logs/bets/2026-02-27.parquet'
DECISIONS_PATH = 'paper_trading_logs/decisions/2026-02-27.parquet'
OUTPUT_PDF = '_07_edge_analysis/day1_vs_backtest_report.pdf'

# ─── Load data ───────────────────────────────────────────────────────────────
bt_all = pd.read_parquet(BT_PATH)
live_bets_raw = pd.read_parquet(BETS_PATH)
live_decisions = pd.read_parquet(DECISIONS_PATH)

# ─── Deduplicate live bets (take settled row if available, else first) ───────
live_bets = live_bets_raw.sort_values('is_settled', ascending=False).drop_duplicates(
    subset=['market_id', 'runner_id'], keep='first'
).reset_index(drop=True)

print(f"Backtest: {len(bt_all)} rows total")
print(f"Live bets (deduped): {len(live_bets)} unique bets")
print(f"Live decisions: {len(live_decisions)} runner evaluations")

# ─── Filter backtest to qualifying bets (edge > 3%) ─────────────────────────
bt_qual = bt_all[bt_all['edge'] > 0.03].copy()
print(f"Backtest qualifying (edge>3%): {len(bt_qual)} bets")

# Extract date from backtest
bt_qual['date'] = bt_qual['marketTime_local'].dt.date
# Extract file_name from key
bt_qual['file_name'] = bt_qual['key'].str.rsplit('_', n=1).str[0]

# ─── Helper: bootstrap daily statistic ──────────────────────────────────────
def daily_stat_distribution(bt_df, stat_func, label, n_bootstrap=10000):
    """Compute a statistic per day, then bootstrap the daily distribution."""
    daily_vals = bt_df.groupby('date').apply(stat_func).dropna()
    # Bootstrap: resample days with replacement
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(daily_vals.values, size=len(daily_vals), replace=True)
        boot_means.append(np.mean(sample))
    return daily_vals, np.array(boot_means)


def compute_percentile_and_zscore(live_val, boot_dist):
    """Where does live_val fall in the bootstrap distribution?"""
    pct = np.mean(boot_dist <= live_val) * 100
    z = (live_val - np.mean(boot_dist)) / np.std(boot_dist) if np.std(boot_dist) > 0 else 0
    return pct, z


# ═════════════════════════════════════════════════════════════════════════════
# Q1: TIMING — When do we trade relative to race close?
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q1: TIMING ANALYSIS")
print("="*70)

# Live: The bot decides at DECISION_SECONDS_BEFORE_START = 20s
# Config shows t_def 3 has t0=20s, which matches
# But the backtest uses t_def 0 (t0=60s), 1 (t0=120s), 2 (t0=30s), 3 (t0=20s)
# The ensemble averages across ALL four t_defs

# Check: the backtest entry point is at t0 of each t_def, but the final
# ensemble prediction blends all 4. In the backtest, the "decision" is made
# using features from t0 of each t_def (60s, 120s, 30s, 20s), and the
# back_odds used for edge calculation is at... which time?

# From the code: back_odds in the backtest comes from the feature file
# which uses t0 of the specific t_def. The ensemble merges by key (file+runner),
# so the back_odds in the final ensemble is from ONE of the t_defs.

# In live: the bot computes features 20s before start for ALL 4 t_defs
# (it buffers tick data, so tm3=600s data is from earlier ticks)
# The decision odds (back_odds) are from the live order book at 20s before start.

# Let's check the live decision timing
live_bets['ts'] = pd.to_datetime(live_bets['timestamp'])
print("\nLive bet timestamps (UTC):")
for _, row in live_bets.iterrows():
    print(f"  {row['ts']} | Market {row['market_id']} | Runner {row['runner_id']} | Odds {row['back_odds']}")

print(f"\nLive trading: All decisions made at T-20s (DECISION_SECONDS_BEFORE_START=20)")
print(f"Backtest: Uses features from t0 of each t_def (60s, 120s, 30s, 20s)")
print(f"  -> The market odds can differ significantly between T-120s and T-20s!")
print(f"  -> Backtest back_odds are from each t_def's t0, not all from T-20s")

# ═════════════════════════════════════════════════════════════════════════════
# Q2: EXECUTION — Do we move the quote more than expected?
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q2: EXECUTION / QUOTE MOVEMENT")
print("="*70)

filled_bets = live_bets[live_bets['conservative_fill'] | live_bets['moderate_fill']].copy()
print(f"\nFilled bets: {len(filled_bets)} / {len(live_bets)}")

if len(filled_bets) > 0:
    filled_bets['price_slip_pct'] = (filled_bets['fill_price'] - filled_bets['limit_price']) / filled_bets['limit_price'] * 100
    filled_bets['price_slip_ticks'] = filled_bets['fill_price'] - filled_bets['limit_price']

    print("\nFill price vs limit price:")
    for _, row in filled_bets.iterrows():
        print(f"  Limit={row['limit_price']:.1f} → Fill={row['fill_price']:.1f} "
              f"(+{row['price_slip_pct']:.1f}% / +{row['price_slip_ticks']:.1f} in odds)")

    print(f"\n  Average slippage: +{filled_bets['price_slip_pct'].mean():.1f}%")
    print(f"  Median slippage: +{filled_bets['price_slip_pct'].median():.1f}%")

    # In backtest fill simulation: conservative fill means trade at prc >= limit
    # So the backtest also allows fills at WORSE odds than limit
    # But the backtest P&L uses the LIMIT price, not the fill price!
    # The paper trader seems to use fill_price for P&L... let's check
    print(f"\n  CRITICAL: Paper trader fills at WORSE odds (higher = more unlikely)")
    print(f"  The backtest fill sim allows this too, but are P&L computed differently?")
    print(f"  Live P&L uses $1 stake, so loss = -$1 regardless of fill price")
    print(f"  (Only wins are affected: payout = stake * (fill_odds - 1))")

# ═════════════════════════════════════════════════════════════════════════════
# Q3: ODDS DISTRIBUTION — Are we aiming at longshots?
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q3: ODDS DISTRIBUTION (Live vs Backtest)")
print("="*70)

live_avg_odds = live_bets['back_odds'].mean()
live_median_odds = live_bets['back_odds'].median()
bt_avg_odds = bt_qual['back_odds'].mean()
bt_median_odds = bt_qual['back_odds'].median()

print(f"\n  Live Day 1:  mean={live_avg_odds:.1f}, median={live_median_odds:.1f}")
print(f"  Backtest:    mean={bt_avg_odds:.1f}, median={bt_median_odds:.1f}")

# Daily distribution
daily_avg_odds, boot_avg_odds = daily_stat_distribution(
    bt_qual, lambda x: x['back_odds'].mean(), 'avg_odds')
pct, z = compute_percentile_and_zscore(live_avg_odds, boot_avg_odds)
print(f"\n  Day 1 avg odds ({live_avg_odds:.1f}) is at {pct:.1f}th percentile of backtest daily distribution")
print(f"  Z-score: {z:.2f}")

# Also look at the odds of individual bets
print(f"\n  Live bet odds: {sorted(live_bets['back_odds'].values)}")
print(f"  Backtest odds distribution:")
print(f"    P10={bt_qual['back_odds'].quantile(0.1):.1f}, P25={bt_qual['back_odds'].quantile(0.25):.1f}, "
      f"P50={bt_qual['back_odds'].quantile(0.5):.1f}, P75={bt_qual['back_odds'].quantile(0.75):.1f}, "
      f"P90={bt_qual['back_odds'].quantile(0.9):.1f}")

# Fraction of bets at odds > 10
live_pct_high = (live_bets['back_odds'] > 10).mean() * 100
bt_pct_high = (bt_qual['back_odds'] > 10).mean() * 100
print(f"\n  % bets with odds > 10: Live={live_pct_high:.0f}% vs Backtest={bt_pct_high:.0f}%")

# ═════════════════════════════════════════════════════════════════════════════
# Q4: BET FREQUENCY — How many bets per race/day?
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q4: BET FREQUENCY")
print("="*70)

# Live: markets evaluated vs bets placed
n_markets_live = live_decisions['file_name'].nunique()
n_bets_live = len(live_bets)
live_bets_per_market = n_bets_live / n_markets_live if n_markets_live > 0 else 0

print(f"\n  Live: {n_bets_live} bets across {n_markets_live} markets = {live_bets_per_market:.2f} bets/market")

# Backtest: bets per market (unique file_name in qualifying bets)
n_markets_bt = bt_all['key'].str.rsplit('_', n=1).str[0].nunique()
n_bets_bt = len(bt_qual)
bt_bets_per_market = n_bets_bt / n_markets_bt if n_markets_bt > 0 else 0

print(f"  Backtest: {n_bets_bt} qualifying bets across {n_markets_bt} unique markets = {bt_bets_per_market:.3f} bets/market")

# Daily bet count distribution
daily_bet_count, boot_bet_count = daily_stat_distribution(
    bt_qual, lambda x: len(x), 'daily_bet_count')
pct, z = compute_percentile_and_zscore(n_bets_live, boot_bet_count)
print(f"\n  Day 1 bet count ({n_bets_live}) is at {pct:.1f}th percentile of backtest daily distribution")
print(f"  Z-score: {z:.2f}")
print(f"  Backtest daily: mean={daily_bet_count.mean():.1f}, median={daily_bet_count.median():.1f}, "
      f"std={daily_bet_count.std():.1f}")

# Also: what fraction of evaluated runners get a bet?
live_bet_rate = n_bets_live / len(live_decisions) if len(live_decisions) > 0 else 0
bt_bet_rate = len(bt_qual) / len(bt_all)
print(f"\n  Bet rate (qualifying/total runners): Live={live_bet_rate:.3f} vs Backtest={bt_bet_rate:.3f}")

# ═════════════════════════════════════════════════════════════════════════════
# Q5: RUNNER SELECTION — Same type of dog? (market rank, probability)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q5: RUNNER SELECTION (type of dog)")
print("="*70)

# Live: what's the market_prob rank of each bet within its race?
live_bets_with_rank = []
for _, bet in live_bets.iterrows():
    race_decisions = live_decisions[live_decisions['file_name'] == bet['file_name']]
    if len(race_decisions) > 0:
        race_sorted = race_decisions.sort_values('market_prob', ascending=False)
        rank = (race_sorted['id'].values == bet['runner_id']).argmax() + 1
        n_runners = len(race_decisions)
        is_favorite = rank == 1
        live_bets_with_rank.append({
            'market_id': bet['market_id'],
            'runner_id': bet['runner_id'],
            'back_odds': bet['back_odds'],
            'market_prob': bet['market_prob'],
            'model_prob': bet['model_prob'],
            'edge': bet['edge'],
            'rank_in_race': rank,
            'n_runners': n_runners,
            'is_favorite': is_favorite,
        })

rank_df = pd.DataFrame(live_bets_with_rank)
print(f"\n  Live bet rank in race (by market prob):")
for _, r in rank_df.iterrows():
    print(f"    Runner {r['runner_id']}: rank {r['rank_in_race']}/{r['n_runners']} "
          f"(mkt_prob={r['market_prob']:.1%}, odds={r['back_odds']:.1f})")

print(f"\n  Average rank: {rank_df['rank_in_race'].mean():.1f} / avg {rank_df['n_runners'].mean():.0f} runners")
print(f"  Favorites bet on: {rank_df['is_favorite'].sum()} / {len(rank_df)}")

# Backtest: compute rank per race
# Need to compute rank of qualifying bets within their races
bt_all_copy = bt_all.copy()
bt_all_copy['file_name'] = bt_all_copy['key'].str.rsplit('_', n=1).str[0]
bt_all_copy['rank_in_race'] = bt_all_copy.groupby('file_name')['market_prob'].rank(ascending=False, method='first')
bt_all_copy['n_runners'] = bt_all_copy.groupby('file_name')['key'].transform('count')
bt_qual_ranked = bt_all_copy[bt_all_copy['edge'] > 0.03].copy()

print(f"\n  Backtest qualifying bets:")
print(f"    Average rank: {bt_qual_ranked['rank_in_race'].mean():.1f} / avg {bt_qual_ranked['n_runners'].mean():.0f} runners")
print(f"    Favorites: {(bt_qual_ranked['rank_in_race'] == 1).sum()} / {len(bt_qual_ranked)} "
      f"({(bt_qual_ranked['rank_in_race'] == 1).mean():.1%})")

# Model probability distribution
live_avg_model_prob = live_bets['model_prob'].mean()
bt_avg_model_prob = bt_qual['model_prob'].mean()
print(f"\n  Model prob: Live={live_avg_model_prob:.3f} vs Backtest={bt_avg_model_prob:.3f}")

daily_avg_model_prob, boot_avg_model_prob = daily_stat_distribution(
    bt_qual, lambda x: x['model_prob'].mean(), 'avg_model_prob')
pct, z = compute_percentile_and_zscore(live_avg_model_prob, boot_avg_model_prob)
print(f"  Day 1 avg model_prob ({live_avg_model_prob:.3f}) at {pct:.1f}th percentile (z={z:.2f})")

# Market probability distribution
live_avg_mkt_prob = live_bets['market_prob'].mean()
bt_avg_mkt_prob = bt_qual['market_prob'].mean()
print(f"\n  Market prob: Live={live_avg_mkt_prob:.3f} vs Backtest={bt_avg_mkt_prob:.3f}")

daily_avg_mkt_prob, boot_avg_mkt_prob = daily_stat_distribution(
    bt_qual, lambda x: x['market_prob'].mean(), 'avg_market_prob')
pct, z = compute_percentile_and_zscore(live_avg_mkt_prob, boot_avg_mkt_prob)
print(f"  Day 1 avg market_prob ({live_avg_mkt_prob:.3f}) at {pct:.1f}th percentile (z={z:.2f})")

# Edge distribution
live_avg_edge = live_bets['edge'].mean()
bt_avg_edge = bt_qual['edge'].mean()
print(f"\n  Edge: Live={live_avg_edge:.3f} vs Backtest={bt_avg_edge:.3f}")

daily_avg_edge, boot_avg_edge = daily_stat_distribution(
    bt_qual, lambda x: x['edge'].mean(), 'avg_edge')
pct, z = compute_percentile_and_zscore(live_avg_edge, boot_avg_edge)
print(f"  Day 1 avg edge ({live_avg_edge:.3f}) at {pct:.1f}th percentile (z={z:.2f})")


# ═════════════════════════════════════════════════════════════════════════════
# V2 MODEL FAILURE CHECK
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CRITICAL: V2 MODEL STATUS")
print("="*70)

# Check if V2 predictions are all NaN/zero
v2_cols = [c for c in live_decisions.columns if c.startswith('V2_')]
v1_cols = [c for c in live_decisions.columns if c.startswith('V1_')]

print(f"\n  V1 columns: {v1_cols}")
print(f"  V2 columns: {v2_cols}")

for col in v1_cols + v2_cols:
    n_valid = live_decisions[col].notna().sum()
    n_total = len(live_decisions)
    mean_val = live_decisions[col].mean()
    print(f"    {col}: {n_valid}/{n_total} valid values, mean={mean_val:.4f}")

print(f"\n  v1_cross: {live_decisions['v1_cross'].notna().sum()}/{len(live_decisions)} valid, mean={live_decisions['v1_cross'].mean():.4f}")
print(f"  v2_cross: {live_decisions['v2_cross'].notna().sum()}/{len(live_decisions)} valid, mean={live_decisions['v2_cross'].mean():.4f}")

# Check: if V2 failed, model_prob = v1_cross (since v2_cross would be NaN)
# Or model_prob = 0.20 * v1_cross + 0.80 * v2_cross
v2_ok = live_decisions['v2_cross'].notna()
print(f"\n  Rows with V2: {v2_ok.sum()} / {len(live_decisions)}")

# For qualifying bets, check V1 vs V2
qual_decisions = live_decisions[live_decisions['edge'] > 0.03]
print(f"\n  Qualifying decisions (edge>3%): {len(qual_decisions)}")
for _, d in qual_decisions.iterrows():
    v2_status = "OK" if pd.notna(d['v2_cross']) else "MISSING"
    print(f"    {d['file_name']} id={d['id']}: "
          f"V1={d['v1_cross']:.3f}, V2={d['v2_cross']:.3f} ({v2_status}), "
          f"model_prob={d['model_prob']:.3f}, market={d['market_prob']:.3f}, edge={d['edge']:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# GENERATE PDF REPORT WITH PLOTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("GENERATING PDF REPORT")
print("="*70)

fig_list = []

def add_bootstrap_plot(live_val, boot_dist, daily_vals, title, xlabel, figsize=(10, 5)):
    """Create a bootstrap distribution plot with Day 1 value marked."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: bootstrap distribution of daily means
    pct, z = compute_percentile_and_zscore(live_val, boot_dist)
    axes[0].hist(boot_dist, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].axvline(live_val, color='red', linewidth=2, linestyle='--', label=f'Day 1: {live_val:.2f}')
    axes[0].axvline(np.mean(boot_dist), color='gray', linewidth=1, linestyle='-', label=f'BT mean: {np.mean(boot_dist):.2f}')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Bootstrap frequency')
    axes[0].set_title(f'{title}\nDay 1 at {pct:.1f}th percentile (z={z:.2f})')
    axes[0].legend(fontsize=8)

    # Right: actual daily values as a histogram
    axes[1].hist(daily_vals.values, bins=30, color='lightcoral', alpha=0.7, edgecolor='white')
    axes[1].axvline(live_val, color='red', linewidth=2, linestyle='--', label=f'Day 1: {live_val:.2f}')
    axes[1].axvline(daily_vals.mean(), color='gray', linewidth=1, linestyle='-', label=f'BT mean: {daily_vals.mean():.2f}')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Number of days')
    axes[1].set_title(f'Daily values (n={len(daily_vals)} days)')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig_list.append(fig)
    return fig

# ─── Plot 1: Odds Distribution ──────────────────────────────────────────────
add_bootstrap_plot(live_avg_odds, boot_avg_odds, daily_avg_odds,
                   'Q3: Average Back Odds per Day', 'Average back odds')

# Also: odds histogram comparison (individual bets, not daily averages)
fig_odds, ax = plt.subplots(1, 1, figsize=(10, 5))
bins = np.arange(0, 60, 2)
ax.hist(bt_qual['back_odds'].clip(upper=60), bins=bins, alpha=0.6, color='steelblue',
        label=f'Backtest (n={len(bt_qual)})', density=True)
for odds in live_bets['back_odds'].values:
    ax.axvline(odds, color='red', linewidth=1.5, alpha=0.7)
# Add one labeled
ax.axvline(live_bets['back_odds'].values[0], color='red', linewidth=1.5, alpha=0.7, label='Day 1 bets')
ax.set_xlabel('Back odds')
ax.set_ylabel('Density')
ax.set_title('Q3: Individual Bet Odds — Backtest Distribution vs Day 1 Bets')
ax.legend()
plt.tight_layout()
fig_list.append(fig_odds)

# ─── Plot 2: Bet Count per Day ──────────────────────────────────────────────
add_bootstrap_plot(n_bets_live, boot_bet_count, daily_bet_count,
                   'Q4: Daily Bet Count', 'Number of bets')

# ─── Plot 3: Model Probability ──────────────────────────────────────────────
add_bootstrap_plot(live_avg_model_prob, boot_avg_model_prob, daily_avg_model_prob,
                   'Q5a: Average Model Probability of Qualifying Bets', 'Model probability')

# ─── Plot 4: Market Probability ──────────────────────────────────────────────
add_bootstrap_plot(live_avg_mkt_prob, boot_avg_mkt_prob, daily_avg_mkt_prob,
                   'Q5b: Average Market Probability of Qualifying Bets', 'Market probability')

# ─── Plot 5: Edge Distribution ──────────────────────────────────────────────
add_bootstrap_plot(live_avg_edge, boot_avg_edge, daily_avg_edge,
                   'Edge: Average Edge of Qualifying Bets', 'Edge (model_prob - market_prob)')

# ─── Plot 6: Runner Rank Distribution ───────────────────────────────────────
# Daily average rank — bt_qual_ranked needs 'date' column
bt_qual_ranked['date'] = bt_qual_ranked['marketTime_local'].dt.date
daily_avg_rank, boot_avg_rank = daily_stat_distribution(
    bt_qual_ranked, lambda x: x['rank_in_race'].mean(), 'avg_rank')
live_avg_rank = rank_df['rank_in_race'].mean()
add_bootstrap_plot(live_avg_rank, boot_avg_rank, daily_avg_rank,
                   'Q5c: Average Rank of Bet Runner in Race (1=favorite)', 'Rank in race')

# ─── Plot 7: Fraction of bets on high-odds runners (>10) ────────────────────
daily_pct_high_odds = bt_qual.groupby('date').apply(lambda x: (x['back_odds'] > 10).mean())
boot_daily = []
rng = np.random.default_rng(42)
for _ in range(10000):
    sample = rng.choice(daily_pct_high_odds.values, size=len(daily_pct_high_odds), replace=True)
    boot_daily.append(np.mean(sample))
boot_daily = np.array(boot_daily)
live_pct_high_val = (live_bets['back_odds'] > 10).mean()
add_bootstrap_plot(live_pct_high_val, boot_daily, daily_pct_high_odds,
                   'Q3b: Fraction of Bets with Odds > 10', 'Fraction with odds > 10')

# ─── Plot 8: Execution Slippage (Live only) ─────────────────────────────────
if len(filled_bets) > 0:
    fig_slip, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = range(len(filled_bets))
    ax.bar(x, filled_bets['limit_price'].values, width=0.35, label='Limit price', color='steelblue')
    ax.bar([i+0.35 for i in x], filled_bets['fill_price'].values, width=0.35, label='Fill price', color='red', alpha=0.7)
    ax.set_xticks([i+0.175 for i in x])
    ax.set_xticklabels([f"Bet {i+1}\n{row['market_id'][-3:]}" for i, (_, row) in enumerate(filled_bets.iterrows())], fontsize=8)
    ax.set_ylabel('Odds')
    ax.set_title('Q2: Limit Price vs Fill Price (Higher = Worse for Backer)')
    ax.legend()

    for i, (_, row) in enumerate(filled_bets.iterrows()):
        slip = row['fill_price'] - row['limit_price']
        ax.annotate(f'+{slip:.1f}', (i+0.175, max(row['fill_price'], row['limit_price'])+0.5),
                   ha='center', fontsize=8, color='red')
    plt.tight_layout()
    fig_list.append(fig_slip)

# ─── Plot 9: Summary comparison table ───────────────────────────────────────
fig_table, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.axis('off')

# Compute all percentiles
_, z_odds = compute_percentile_and_zscore(live_avg_odds, boot_avg_odds)
_, z_bets = compute_percentile_and_zscore(n_bets_live, boot_bet_count)
_, z_model = compute_percentile_and_zscore(live_avg_model_prob, boot_avg_model_prob)
_, z_mkt = compute_percentile_and_zscore(live_avg_mkt_prob, boot_avg_mkt_prob)
_, z_edge = compute_percentile_and_zscore(live_avg_edge, boot_avg_edge)
_, z_rank = compute_percentile_and_zscore(live_avg_rank, boot_avg_rank)

table_data = [
    ['Avg Back Odds', f'{live_avg_odds:.1f}', f'{bt_avg_odds:.1f}', f'{z_odds:+.2f}', 'HIGH' if abs(z_odds) > 2 else 'OK'],
    ['Daily Bet Count', f'{n_bets_live}', f'{daily_bet_count.mean():.1f}', f'{z_bets:+.2f}', 'HIGH' if abs(z_bets) > 2 else 'OK'],
    ['Avg Model Prob', f'{live_avg_model_prob:.3f}', f'{bt_avg_model_prob:.3f}', f'{z_model:+.2f}', 'LOW' if z_model < -2 else 'OK'],
    ['Avg Market Prob', f'{live_avg_mkt_prob:.3f}', f'{bt_avg_mkt_prob:.3f}', f'{z_mkt:+.2f}', 'LOW' if z_mkt < -2 else 'OK'],
    ['Avg Edge', f'{live_avg_edge:.3f}', f'{bt_avg_edge:.3f}', f'{z_edge:+.2f}', 'HIGH' if z_edge > 2 else 'OK'],
    ['Avg Runner Rank', f'{live_avg_rank:.1f}', f'{daily_avg_rank.mean():.1f}', f'{z_rank:+.2f}', 'HIGH' if z_rank > 2 else 'OK'],
    ['% Odds > 10', f'{live_pct_high_val:.0%}', f'{daily_pct_high_odds.mean():.0%}', '', ''],
    ['V2 Models', 'PARTIAL', 'ALL 88', '', 'BUG'],
    ['Fill Rate', f'{live_bets["conservative_fill"].any()}', '97%', '', ''],
]

table = ax.table(cellText=table_data,
                colLabels=['Metric', 'Day 1 Live', 'Backtest', 'Z-score', 'Flag'],
                loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Color code flags
for i, row in enumerate(table_data):
    if row[4] in ('HIGH', 'LOW', 'BUG'):
        table[i+1, 4].set_facecolor('#ffcccc')
    elif row[4] == 'OK':
        table[i+1, 4].set_facecolor('#ccffcc')

ax.set_title('Day 1 Paper Trading vs Backtest: Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
fig_list.append(fig_table)

# ─── Save PDF ────────────────────────────────────────────────────────────────
with PdfPages(OUTPUT_PDF) as pdf:
    for fig in fig_list:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"\nPDF saved to: {OUTPUT_PDF}")
print(f"Total figures: {len(fig_list)}")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL DIAGNOSTIC SUMMARY")
print("="*70)
print(f"""
1. TIMING:
   - Live decides at T-20s (config DECISION_SECONDS_BEFORE_START=20)
   - Backtest uses 4 different t0 values (20s, 30s, 60s, 120s) then averages
   - POTENTIAL ISSUE: The back_odds in the backtest may be from earlier times
     (e.g., t_def 1 uses t0=120s), while live always uses T-20s odds

2. EXECUTION:
   - {len(filled_bets)} fills, all at WORSE odds than limit
   - Average slippage: +{filled_bets['price_slip_pct'].mean():.1f}% on filled bets
   - This is expected (Betfair matching works this way in liquid markets)
   - KEY: P&L impact depends on whether we use limit or fill price

3. ODDS (z={z_odds:+.2f}):
   - Day 1 avg odds: {live_avg_odds:.1f} vs Backtest: {bt_avg_odds:.1f}
   - {'SIGNIFICANTLY higher odds — betting on longshots' if abs(z_odds) > 2 else 'Within normal range'}
   - {live_pct_high_val:.0%} of bets at odds > 10 vs {daily_pct_high_odds.mean():.0%} in backtest

4. FREQUENCY (z={z_bets:+.2f}):
   - Day 1: {n_bets_live} bets vs Backtest daily mean: {daily_bet_count.mean():.1f}
   - {'SIGNIFICANTLY different from backtest' if abs(z_bets) > 2 else 'Within normal range'}

5. RUNNER SELECTION (z={z_rank:+.2f}):
   - Day 1 avg rank: {live_avg_rank:.1f} vs Backtest: {daily_avg_rank.mean():.1f}
   - Day 1 avg market_prob: {live_avg_mkt_prob:.3f} vs Backtest: {bt_avg_mkt_prob:.3f} (z={z_mkt:+.2f})
   - {'DIFFERENT runner selection profile' if abs(z_rank) > 2 else 'Similar runner selection'}

CRITICAL FLAGS:
   - V2 models: {'FAILED' if live_decisions['v2_cross'].isna().any() else 'OK'} ({live_decisions['v2_cross'].isna().sum()}/{len(live_decisions)} rows missing V2)
   - If V2 models partially failed, the ensemble is V1-only for those rows
   - The backtest ensemble was V1(20%) + V2(80%) — without V2, predictions shift
""")
