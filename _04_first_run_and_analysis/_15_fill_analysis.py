"""
Fill-Adjusted Backtest Report — Analyze fill rate simulation results and generate PDF.

Usage: python3 _15_fill_analysis.py

Reads: res/fill_simulation/fill_results_merged.parquet
Writes: res/fill_simulation/fill_analysis_report.pdf

Generates a multi-page PDF with:
  1. Title + headline fill-adjusted metrics
  2. Fill rate summary by odds bucket and scenario
  3. Side-by-side P&L comparison (100% fill vs conservative vs moderate)
  4. Charts: fill rate by odds, cumulative P&L, fill vs spread, time-to-fill
  5. Odds bucket breakdown: does longshot profit survive realistic fills?
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from math import erf, sqrt

os.chdir("/data/projects/punim2039/alpha_odds/res")

# ── Constants ──
FLAT_BET = 25
COMMISSION = 0.075
EDGE_THRESH = 0.03  # primary analysis threshold

ODDS_BINS = [1.0, 3.0, 5.0, 8.0, 15.0, 30.0, 1000.0]
ODDS_LABELS = ['1-3', '3-5', '5-8', '8-15', '15-30', '30+']

# Colors
C_TITLE = '#1a1a2e'
C_SUBTITLE = '#16213e'
C_TEXT = '#333'
C_MUTED = '#888'
C_BLUE = '#3282b8'
C_GREEN = '#27ae60'
C_RED = '#e94560'
C_ORANGE = '#f39c12'
C_PURPLE = '#8e44ad'


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def backtest_stats(df, label=''):
    """Compute standard backtest metrics for a DataFrame with 'win', 'back_odds', 'pnl' cols."""
    if len(df) == 0:
        return {}
    n = len(df)
    wr = df['win'].mean()
    roi = df['pnl'].mean() * 100
    total_pnl = df['pnl'].sum()
    z = df['pnl'].mean() / df['pnl'].std() * np.sqrt(n) if df['pnl'].std() > 0 else 0
    p = 1 - norm_cdf(z)

    profit_flat = total_pnl * FLAT_BET
    df_m = df.copy()
    df_m['month'] = pd.to_datetime(df_m['marketTime_local']).dt.to_period('M')
    monthly = df_m.groupby('month')['pnl'].sum()
    pm = int((monthly > 0).sum())
    n_months = len(monthly)
    sh_m = monthly.mean() / monthly.std() if monthly.std() > 0 else 0
    sh_a = sh_m * np.sqrt(12)

    return {
        'label': label, 'n': n, 'wr': wr, 'roi': roi, 'total_pnl': total_pnl,
        'z': z, 'p': p, 'sharpe': sh_a, 'profit_flat': profit_flat,
        'pm': pm, 'n_months': n_months, 'monthly_pnl': monthly,
    }


# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════
print("Loading fill simulation results...", flush=True)
fill = pd.read_parquet('fill_simulation/fill_results_merged.parquet')
print(f"Loaded {len(fill):,} rows", flush=True)

# Also load the full ensemble predictions for 100% fill baseline
ens = pd.read_parquet('analysis/ultimate_cross_t_ensemble_predictions.parquet')
ens['pnl'] = ens['win'] * (ens['back_odds'] - 1) * (1 - COMMISSION) - (1 - ens['win'])

print(f"Ensemble: {len(ens):,} rows", flush=True)

# Merge marketTime_local from ensemble into fill data
ens_time = ens[['key', 'marketTime_local']].drop_duplicates(subset='key')
fill = fill.merge(ens_time, on='key', how='left')

# ── Primary scenario: t0_60s window, best_back price variant ──
primary = fill[(fill['window'] == 't0_60s') & (fill['price_variant'] == 'best_back')].copy()
print(f"Primary scenario (t0_60s, best_back): {len(primary):,} rows", flush=True)

# Add odds bucket
primary['odds_bucket'] = pd.cut(primary['back_odds'], bins=ODDS_BINS, labels=ODDS_LABELS, right=False)
ens['odds_bucket'] = pd.cut(ens['back_odds'], bins=ODDS_BINS, labels=ODDS_LABELS, right=False)

# Compute P&L columns for the primary scenario
primary['pnl_100pct'] = primary['win'] * (primary['back_odds'] - 1) * (1 - COMMISSION) - (1 - primary['win'])
primary['pnl_conservative'] = np.where(
    primary['conservative_fill'],
    primary['win'] * (primary['limit_price'] - 1) * (1 - COMMISSION) - (1 - primary['win']),
    0.0
)
primary['pnl_moderate'] = np.where(
    primary['moderate_fill'],
    primary['win'] * (primary['limit_price'] - 1) * (1 - COMMISSION) - (1 - primary['win']),
    0.0
)


# ══════════════════════════════════════════════
# COMPUTE SUMMARY TABLES
# ══════════════════════════════════════════════
def scenario_summary(df, edge_thresh):
    """Compute fill rates and P&L for a given edge threshold."""
    bets = df[df['edge'] > edge_thresh].copy()
    n = len(bets)
    if n == 0:
        return None

    fill_c = bets['conservative_fill'].mean()
    fill_m = bets['moderate_fill'].mean()

    # 100% fill
    pnl_100 = bets['pnl_100pct']
    roi_100 = pnl_100.mean() * 100
    z_100 = pnl_100.mean() / pnl_100.std() * np.sqrt(n) if pnl_100.std() > 0 else 0

    # Conservative fill
    n_c = int(bets['conservative_fill'].sum())
    filled_c = bets[bets['conservative_fill']]
    pnl_c = bets['pnl_conservative']
    roi_c = pnl_c.sum() / n * 100 if n > 0 else 0  # ROI over all attempted bets
    z_c = pnl_c.mean() / pnl_c.std() * np.sqrt(n) if pnl_c.std() > 0 else 0

    # Moderate fill
    n_m = int(bets['moderate_fill'].sum())
    pnl_m = bets['pnl_moderate']
    roi_m = pnl_m.sum() / n * 100 if n > 0 else 0
    z_m = pnl_m.mean() / pnl_m.std() * np.sqrt(n) if pnl_m.std() > 0 else 0

    return {
        'n': n, 'fill_c': fill_c, 'fill_m': fill_m,
        'n_c': n_c, 'n_m': n_m,
        'roi_100': roi_100, 'roi_c': roi_c, 'roi_m': roi_m,
        'z_100': z_100, 'z_c': z_c, 'z_m': z_m,
        'profit_100': pnl_100.sum() * FLAT_BET,
        'profit_c': pnl_c.sum() * FLAT_BET,
        'profit_m': pnl_m.sum() * FLAT_BET,
    }


# Summary for primary edge threshold
primary_summary = scenario_summary(primary, EDGE_THRESH)

# Summary by odds bucket
bucket_summaries = {}
for label in ODDS_LABELS:
    sub = primary[primary['odds_bucket'] == label]
    s = scenario_summary(sub, EDGE_THRESH)
    if s is not None:
        bucket_summaries[label] = s

# ── All scenarios comparison ──
all_scenarios = {}
for window in ['t0_60s', 't0_20s']:
    for variant in ['best_back', 'minus_1_tick', 'plus_1_tick']:
        sub = fill[(fill['window'] == window) & (fill['price_variant'] == variant)].copy()
        bets = sub[sub['edge'] > EDGE_THRESH]
        if len(bets) > 0:
            all_scenarios[(window, variant)] = {
                'n': len(bets),
                'fill_c': bets['conservative_fill'].mean(),
                'fill_m': bets['moderate_fill'].mean(),
            }

print("\n=== Fill Rate Summary (edge > 3%) ===", flush=True)
if primary_summary:
    print(f"Bets: {primary_summary['n']:,}", flush=True)
    print(f"Conservative fill: {primary_summary['fill_c']:.1%} ({primary_summary['n_c']:,} filled)", flush=True)
    print(f"Moderate fill: {primary_summary['fill_m']:.1%} ({primary_summary['n_m']:,} filled)", flush=True)
    print(f"100% fill ROI: {primary_summary['roi_100']:+.1f}%, z={primary_summary['z_100']:.2f}", flush=True)
    print(f"Conservative ROI: {primary_summary['roi_c']:+.1f}%, z={primary_summary['z_c']:.2f}", flush=True)
    print(f"Moderate ROI: {primary_summary['roi_m']:+.1f}%, z={primary_summary['z_m']:.2f}", flush=True)


# ══════════════════════════════════════════════
# GENERATE PDF REPORT
# ══════════════════════════════════════════════
output_path = 'fill_simulation/fill_analysis_report.pdf'
print(f"\nGenerating report to {output_path}...", flush=True)

with PdfPages(output_path) as pdf:

    # ════════════════════════════════════════════
    # PAGE 1: Title & Key Fill-Adjusted Metrics
    # ════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    ax.text(0.5, 0.93, 'Fill Rate Simulation', fontsize=40, fontweight='bold',
            ha='center', va='top', color=C_TITLE)
    ax.text(0.5, 0.86, 'Limit Order Fill Analysis — Cross-t Super-Ensemble',
            fontsize=18, ha='center', va='top', color=C_SUBTITLE)
    ax.text(0.5, 0.82, 'OOS 2025 | Replay of tick-by-tick order book data',
            fontsize=12, ha='center', va='top', color=C_MUTED)

    if primary_summary:
        ps = primary_summary

        # 100% fill baseline
        ax.text(0.5, 0.73, '100% Fill Baseline (current assumption)',
                fontsize=13, fontweight='bold', ha='center', color=C_TEXT)
        metrics_100 = [
            (f"{ps['n']:,}", 'Bets'), (f"{ps['roi_100']:+.0f}%", 'ROI'),
            (f"${ps['profit_100']:,.0f}", f'P&L ({FLAT_BET} AUD/bet)'),
        ]
        for i, (val, label) in enumerate(metrics_100):
            x = 0.20 + i * 0.25
            bbox = dict(boxstyle='round,pad=0.4', facecolor='#e8f4f8', edgecolor='#bbb', linewidth=0.5)
            ax.text(x, 0.66, val, fontsize=20, fontweight='bold', ha='center', color=C_BLUE, bbox=bbox)
            ax.text(x, 0.61, label, fontsize=9, ha='center', color=C_MUTED)

        # Conservative fill
        ax.text(0.5, 0.53, f'Conservative Fill ({ps["fill_c"]:.0%} fill rate)',
                fontsize=13, fontweight='bold', ha='center',
                color=C_GREEN if ps['roi_c'] > 0 else C_RED)
        metrics_c = [
            (f"{ps['n_c']:,}", 'Filled'), (f"{ps['roi_c']:+.0f}%", 'ROI'),
            (f"${ps['profit_c']:,.0f}", f'P&L ({FLAT_BET} AUD/bet)'),
        ]
        for i, (val, label) in enumerate(metrics_c):
            x = 0.20 + i * 0.25
            color = C_GREEN if ps['profit_c'] > 0 else C_RED
            bbox = dict(boxstyle='round,pad=0.4', facecolor='#e8f8e8', edgecolor='#bbb', linewidth=0.5)
            ax.text(x, 0.46, val, fontsize=20, fontweight='bold', ha='center', color=color, bbox=bbox)
            ax.text(x, 0.41, label, fontsize=9, ha='center', color=C_MUTED)

        # Moderate fill
        ax.text(0.5, 0.33, f'Moderate Fill ({ps["fill_m"]:.0%} fill rate)',
                fontsize=13, fontweight='bold', ha='center',
                color=C_GREEN if ps['roi_m'] > 0 else C_RED)
        metrics_m = [
            (f"{ps['n_m']:,}", 'Filled'), (f"{ps['roi_m']:+.0f}%", 'ROI'),
            (f"${ps['profit_m']:,.0f}", f'P&L ({FLAT_BET} AUD/bet)'),
        ]
        for i, (val, label) in enumerate(metrics_m):
            x = 0.20 + i * 0.25
            color = C_GREEN if ps['profit_m'] > 0 else C_RED
            bbox = dict(boxstyle='round,pad=0.4', facecolor='#f8f0e8', edgecolor='#bbb', linewidth=0.5)
            ax.text(x, 0.26, val, fontsize=20, fontweight='bold', ha='center', color=color, bbox=bbox)
            ax.text(x, 0.21, label, fontsize=9, ha='center', color=C_MUTED)

        # Footnote
        ax.text(0.5, 0.12, f'Edge threshold: {EDGE_THRESH:.0%} | Commission: {COMMISSION:.1%} | '
                f'Window: 60s before close | Price: best back at decision time',
                fontsize=9, ha='center', color='#999')
        ax.text(0.5, 0.08, 'Conservative: trade at limit price or better  |  '
                'Moderate: trade fill OR best_lay crosses limit price',
                fontsize=8, ha='center', color='#aaa')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 2: Fill Rate by Odds Bucket
    # ════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Fill Rate Analysis by Odds Bucket (edge > 3%, t0=60s, best_back)',
                 fontsize=14, fontweight='bold', y=0.98, color=C_TITLE)

    bets = primary[primary['edge'] > EDGE_THRESH].copy()

    # Chart 1: Fill rate by odds bucket (bar chart)
    ax = axes[0, 0]
    if len(bucket_summaries) > 0:
        labels = list(bucket_summaries.keys())
        fill_c = [bucket_summaries[l]['fill_c'] * 100 for l in labels]
        fill_m = [bucket_summaries[l]['fill_m'] * 100 for l in labels]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, fill_c, w, label='Conservative', color=C_BLUE, alpha=0.8)
        ax.bar(x + w/2, fill_m, w, label='Moderate', color=C_GREEN, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Fill Rate (%)')
        ax.set_title('Fill Rate by Odds Bucket', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        for i, (c, m) in enumerate(zip(fill_c, fill_m)):
            ax.text(i - w/2, c + 1, f'{c:.0f}%', ha='center', fontsize=7, color=C_BLUE)
            ax.text(i + w/2, m + 1, f'{m:.0f}%', ha='center', fontsize=7, color=C_GREEN)

    # Chart 2: Number of bets per bucket
    ax = axes[0, 1]
    if len(bucket_summaries) > 0:
        n_bets = [bucket_summaries[l]['n'] for l in labels]
        n_filled = [bucket_summaries[l]['n_c'] for l in labels]
        ax.bar(x - w/2, n_bets, w, label='Attempted', color='#ccc', edgecolor='#999')
        ax.bar(x + w/2, n_filled, w, label='Filled (conservative)', color=C_BLUE, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Number of Bets')
        ax.set_title('Bet Count by Odds Bucket', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        for i, (nb, nf) in enumerate(zip(n_bets, n_filled)):
            ax.text(i - w/2, nb + max(n_bets)*0.01, str(nb), ha='center', fontsize=7)
            ax.text(i + w/2, nf + max(n_bets)*0.01, str(nf), ha='center', fontsize=7, color=C_BLUE)

    # Chart 3: Fill rate vs spread at order time
    ax = axes[1, 0]
    valid = bets[bets['spread_at_order'].notna() & (bets['spread_at_order'] < 50)].copy()
    if len(valid) > 0:
        spread_bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        spread_labels = ['0-0.5', '0.5-1', '1-2', '2-5', '5-10', '10+']
        valid['spread_bin'] = pd.cut(valid['spread_at_order'], bins=spread_bins, labels=spread_labels, right=False)
        grouped = valid.groupby('spread_bin', observed=True).agg(
            fill_c=('conservative_fill', 'mean'),
            fill_m=('moderate_fill', 'mean'),
            count=('conservative_fill', 'count')
        )
        if len(grouped) > 0:
            gx = np.arange(len(grouped))
            ax.bar(gx - w/2, grouped['fill_c'] * 100, w, label='Conservative', color=C_BLUE, alpha=0.8)
            ax.bar(gx + w/2, grouped['fill_m'] * 100, w, label='Moderate', color=C_GREEN, alpha=0.8)
            ax.set_xticks(gx)
            ax.set_xticklabels(grouped.index.tolist(), fontsize=8)
            ax.set_xlabel('Spread at Order Time (odds)')
            ax.set_ylabel('Fill Rate (%)')
            ax.set_title('Fill Rate vs Spread', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.set_ylim(0, 105)

    # Chart 4: Time-to-fill distribution (conservative fills only)
    ax = axes[1, 1]
    ttf = bets.loc[bets['conservative_fill'] & bets['time_to_fill_s'].notna(), 'time_to_fill_s']
    if len(ttf) > 0:
        ax.hist(ttf, bins=30, color=C_BLUE, alpha=0.7, edgecolor='white')
        ax.axvline(ttf.median(), color=C_RED, linestyle='--', label=f'Median: {ttf.median():.1f}s')
        ax.set_xlabel('Time to Fill (seconds)')
        ax.set_ylabel('Count')
        ax.set_title('Time-to-Fill Distribution', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 3: P&L Comparison — Cumulative P&L
    # ════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('P&L Comparison: 100% Fill vs Fill-Adjusted',
                 fontsize=14, fontweight='bold', y=0.98, color=C_TITLE)

    bets_sorted = bets.sort_values('marketTime_local').reset_index(drop=True)

    # Chart 1: Cumulative P&L comparison
    ax = axes[0, 0]
    if len(bets_sorted) > 0:
        cum_100 = bets_sorted['pnl_100pct'].cumsum() * FLAT_BET
        cum_c = bets_sorted['pnl_conservative'].cumsum() * FLAT_BET
        cum_m = bets_sorted['pnl_moderate'].cumsum() * FLAT_BET
        ax.plot(cum_100, label='100% Fill', color=C_BLUE, linewidth=1.5)
        ax.plot(cum_c, label='Conservative', color=C_RED, linewidth=1.5)
        ax.plot(cum_m, label='Moderate', color=C_GREEN, linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
        ax.set_xlabel('Bet Number')
        ax.set_ylabel(f'Cumulative P&L ({FLAT_BET} AUD/bet)')
        ax.set_title('Cumulative P&L Comparison', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    # Chart 2: Monthly P&L comparison
    ax = axes[0, 1]
    if len(bets_sorted) > 0:
        bets_sorted['month'] = pd.to_datetime(bets_sorted['marketTime_local']).dt.to_period('M')
        monthly = bets_sorted.groupby('month').agg(
            pnl_100=('pnl_100pct', 'sum'),
            pnl_c=('pnl_conservative', 'sum'),
            pnl_m=('pnl_moderate', 'sum'),
        ) * FLAT_BET
        mx = np.arange(len(monthly))
        w3 = 0.25
        ax.bar(mx - w3, monthly['pnl_100'], w3, label='100% Fill', color=C_BLUE, alpha=0.7)
        ax.bar(mx, monthly['pnl_c'], w3, label='Conservative', color=C_RED, alpha=0.7)
        ax.bar(mx + w3, monthly['pnl_m'], w3, label='Moderate', color=C_GREEN, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(mx)
        ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, fontsize=7)
        ax.set_ylabel(f'Monthly P&L ({FLAT_BET} AUD/bet)')
        ax.set_title('Monthly P&L Comparison', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)

    # Chart 3: ROI by odds bucket — 100% vs conservative
    ax = axes[1, 0]
    if len(bucket_summaries) > 0:
        labels_b = list(bucket_summaries.keys())
        roi_100 = [bucket_summaries[l]['roi_100'] for l in labels_b]
        roi_c = [bucket_summaries[l]['roi_c'] for l in labels_b]
        roi_m = [bucket_summaries[l]['roi_m'] for l in labels_b]
        bx = np.arange(len(labels_b))
        w3 = 0.25
        ax.bar(bx - w3, roi_100, w3, label='100% Fill', color=C_BLUE, alpha=0.7)
        ax.bar(bx, roi_c, w3, label='Conservative', color=C_RED, alpha=0.7)
        ax.bar(bx + w3, roi_m, w3, label='Moderate', color=C_GREEN, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(bx)
        ax.set_xticklabels(labels_b, fontsize=8)
        ax.set_ylabel('ROI (%)')
        ax.set_title('ROI by Odds Bucket', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)

    # Chart 4: Profit by odds bucket ($)
    ax = axes[1, 1]
    if len(bucket_summaries) > 0:
        p_100 = [bucket_summaries[l]['profit_100'] for l in labels_b]
        p_c = [bucket_summaries[l]['profit_c'] for l in labels_b]
        p_m = [bucket_summaries[l]['profit_m'] for l in labels_b]
        ax.bar(bx - w3, p_100, w3, label='100% Fill', color=C_BLUE, alpha=0.7)
        ax.bar(bx, p_c, w3, label='Conservative', color=C_RED, alpha=0.7)
        ax.bar(bx + w3, p_m, w3, label='Moderate', color=C_GREEN, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(bx)
        ax.set_xticklabels(labels_b, fontsize=8)
        ax.set_ylabel(f'Profit ({FLAT_BET} AUD/bet)')
        ax.set_title('Profit by Odds Bucket', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 4: Detailed Tables
    # ════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    fig.suptitle('Detailed Fill-Adjusted Backtest Results',
                 fontsize=16, fontweight='bold', y=0.97, color=C_TITLE)

    y = 0.88
    lh = 0.023

    # Table 1: Side-by-side comparison across edge thresholds
    ax.text(0.04, y, 'P&L Comparison Across Edge Thresholds (t0=60s, best_back)',
            fontsize=12, fontweight='bold', color=C_SUBTITLE)
    y -= lh * 1.5

    header = f"{'Edge':>6}  {'Bets':>6}  {'Fill%':>6} {'ROI_100':>8} {'ROI_c':>8} {'ROI_m':>8}  {'z_100':>6} {'z_c':>6} {'z_m':>6}  {'$_100':>8} {'$_c':>8} {'$_m':>8}"
    ax.text(0.04, y, header, fontsize=8, fontweight='bold', family='monospace', color=C_TEXT)
    y -= lh

    for et in [0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.07, 0.10]:
        s = scenario_summary(primary, et)
        if s is None:
            continue
        line = (f"{et:>5.0%}  {s['n']:>6,}  {s['fill_c']:>5.0%} "
                f"{s['roi_100']:>+7.1f}% {s['roi_c']:>+7.1f}% {s['roi_m']:>+7.1f}%  "
                f"{s['z_100']:>5.2f} {s['z_c']:>5.2f} {s['z_m']:>5.2f}  "
                f"${s['profit_100']:>7,.0f} ${s['profit_c']:>7,.0f} ${s['profit_m']:>7,.0f}")
        bold = et == EDGE_THRESH
        ax.text(0.04, y, line, fontsize=7.5, family='monospace',
                fontweight='bold' if bold else 'normal',
                color=C_SUBTITLE if bold else C_TEXT)
        y -= lh
    y -= lh

    # Table 2: Odds bucket breakdown for primary threshold
    ax.text(0.04, y, f'Odds Bucket Breakdown (edge > {EDGE_THRESH:.0%})',
            fontsize=12, fontweight='bold', color=C_SUBTITLE)
    y -= lh * 1.5

    header2 = f"{'Bucket':>8}  {'Bets':>6}  {'Fill_c%':>7} {'Fill_m%':>7}  {'ROI_100':>8} {'ROI_c':>8} {'ROI_m':>8}  {'$_100':>8} {'$_c':>8} {'$_m':>8}"
    ax.text(0.04, y, header2, fontsize=8, fontweight='bold', family='monospace', color=C_TEXT)
    y -= lh

    for label in ODDS_LABELS:
        if label not in bucket_summaries:
            continue
        s = bucket_summaries[label]
        line = (f"{label:>8}  {s['n']:>6,}  {s['fill_c']:>6.0%} {s['fill_m']:>7.0%}  "
                f"{s['roi_100']:>+7.1f}% {s['roi_c']:>+7.1f}% {s['roi_m']:>+7.1f}%  "
                f"${s['profit_100']:>7,.0f} ${s['profit_c']:>7,.0f} ${s['profit_m']:>7,.0f}")
        ax.text(0.04, y, line, fontsize=7.5, family='monospace', color=C_TEXT)
        y -= lh
    y -= lh

    # Table 3: All scenario comparison
    ax.text(0.04, y, f'Fill Rates Across All Scenarios (edge > {EDGE_THRESH:.0%})',
            fontsize=12, fontweight='bold', color=C_SUBTITLE)
    y -= lh * 1.5

    header3 = f"{'Window':>10}  {'Price Variant':>14}  {'Bets':>6}  {'Fill_c':>7} {'Fill_m':>7}"
    ax.text(0.04, y, header3, fontsize=8, fontweight='bold', family='monospace', color=C_TEXT)
    y -= lh

    for (window, variant), s in sorted(all_scenarios.items()):
        line = f"{window:>10}  {variant:>14}  {s['n']:>6,}  {s['fill_c']:>6.0%} {s['fill_m']:>7.0%}"
        bold = (window == 't0_60s' and variant == 'best_back')
        ax.text(0.04, y, line, fontsize=7.5, family='monospace',
                fontweight='bold' if bold else 'normal', color=C_TEXT)
        y -= lh

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 5: Does Longshot Profit Survive?
    # ════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    fig.suptitle('Key Question: Does Longshot Profit Survive Realistic Fills?',
                 fontsize=16, fontweight='bold', y=0.97, color=C_TITLE)

    y = 0.88

    # Identify the longshot buckets (15-30, 30+)
    longshot_labels = ['15-30', '30+']
    short_labels = ['1-3', '3-5', '5-8']

    ax.text(0.04, y, 'Longshot Analysis (odds 15-30 and 30+)', fontsize=13, fontweight='bold', color=C_SUBTITLE)
    y -= lh * 1.5

    for label in longshot_labels:
        if label not in bucket_summaries:
            continue
        s = bucket_summaries[label]
        ax.text(0.06, y, f'Odds {label}:', fontsize=11, fontweight='bold', color=C_TEXT)
        y -= lh
        ax.text(0.08, y, f'Bets: {s["n"]:,}  |  Conservative fill: {s["fill_c"]:.0%}  |  Moderate fill: {s["fill_m"]:.0%}',
                fontsize=9, color=C_TEXT)
        y -= lh
        ax.text(0.08, y, f'100% fill ROI: {s["roi_100"]:+.1f}%  |  Conservative ROI: {s["roi_c"]:+.1f}%  |  Moderate ROI: {s["roi_m"]:+.1f}%',
                fontsize=9, color=C_GREEN if s["roi_c"] > 0 else C_RED)
        y -= lh
        ax.text(0.08, y, f'100% fill P&L: \\${s["profit_100"]:,.0f}  |  Conservative P&L: \\${s["profit_c"]:,.0f}  |  Moderate P&L: \\${s["profit_m"]:,.0f}',
                fontsize=9, color=C_TEXT)
        y -= lh * 1.5

    ax.text(0.04, y, 'Short-Price Analysis (odds 1-8)', fontsize=13, fontweight='bold', color=C_SUBTITLE)
    y -= lh * 1.5

    for label in short_labels:
        if label not in bucket_summaries:
            continue
        s = bucket_summaries[label]
        ax.text(0.06, y, f'Odds {label}:', fontsize=11, fontweight='bold', color=C_TEXT)
        y -= lh
        ax.text(0.08, y, f'Bets: {s["n"]:,}  |  Conservative fill: {s["fill_c"]:.0%}  |  Moderate fill: {s["fill_m"]:.0%}',
                fontsize=9, color=C_TEXT)
        y -= lh
        ax.text(0.08, y, f'100% fill ROI: {s["roi_100"]:+.1f}%  |  Conservative ROI: {s["roi_c"]:+.1f}%  |  Moderate ROI: {s["roi_m"]:+.1f}%',
                fontsize=9, color=C_GREEN if s["roi_c"] > 0 else C_RED)
        y -= lh * 1.5

    # Summary verdict
    y -= lh
    total_100 = sum(bucket_summaries.get(l, {}).get('profit_100', 0) for l in longshot_labels)
    total_c = sum(bucket_summaries.get(l, {}).get('profit_c', 0) for l in longshot_labels)
    total_m = sum(bucket_summaries.get(l, {}).get('profit_m', 0) for l in longshot_labels)
    pct_retained = total_c / total_100 * 100 if total_100 != 0 else 0

    ax.text(0.04, y, 'Verdict:', fontsize=14, fontweight='bold', color=C_TITLE)
    y -= lh * 1.3
    if total_c > 0:
        ax.text(0.06, y, 'Longshot profit SURVIVES conservative fills: '
                + f'\\${total_c:,.0f} ({pct_retained:.0f}% of \\${total_100:,.0f} retained)',
                fontsize=11, fontweight='bold', color=C_GREEN)
    elif total_m > 0:
        ax.text(0.06, y, f'Longshot profit partially survives with moderate fills: \\${total_m:,.0f}',
                fontsize=11, fontweight='bold', color=C_ORANGE)
    else:
        ax.text(0.06, y, 'Longshot profit does NOT survive realistic fills',
                fontsize=11, fontweight='bold', color=C_RED)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 6: 20s Window Comparison
    # ════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Window Comparison: 60s vs 20s Before Close',
                 fontsize=14, fontweight='bold', y=0.98, color=C_TITLE)

    # Load 20s scenario
    primary_20s = fill[(fill['window'] == 't0_20s') & (fill['price_variant'] == 'best_back')].copy()
    primary_20s['odds_bucket'] = pd.cut(primary_20s['back_odds'], bins=ODDS_BINS, labels=ODDS_LABELS, right=False)
    primary_20s['pnl_100pct'] = primary_20s['win'] * (primary_20s['back_odds'] - 1) * (1 - COMMISSION) - (1 - primary_20s['win'])
    primary_20s['pnl_conservative'] = np.where(
        primary_20s['conservative_fill'],
        primary_20s['win'] * (primary_20s['limit_price'] - 1) * (1 - COMMISSION) - (1 - primary_20s['win']),
        0.0
    )

    bets_60s = primary[primary['edge'] > EDGE_THRESH]
    bets_20s = primary_20s[primary_20s['edge'] > EDGE_THRESH]

    # Chart 1: Fill rate comparison by odds bucket
    ax = axes[0, 0]
    buckets_60 = bets_60s.groupby('odds_bucket', observed=True)['conservative_fill'].mean() * 100
    buckets_20 = bets_20s.groupby('odds_bucket', observed=True)['conservative_fill'].mean() * 100
    common = sorted(set(buckets_60.index) & set(buckets_20.index), key=lambda x: ODDS_LABELS.index(x) if x in ODDS_LABELS else 99)
    if common:
        cx = np.arange(len(common))
        ax.bar(cx - 0.2, [buckets_60.get(l, 0) for l in common], 0.35, label='60s window', color=C_BLUE, alpha=0.8)
        ax.bar(cx + 0.2, [buckets_20.get(l, 0) for l in common], 0.35, label='20s window', color=C_ORANGE, alpha=0.8)
        ax.set_xticks(cx)
        ax.set_xticklabels(common, fontsize=8)
        ax.set_ylabel('Conservative Fill Rate (%)')
        ax.set_title('Fill Rate: 60s vs 20s Window', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)

    # Chart 2: Cumulative P&L comparison
    ax = axes[0, 1]
    if len(bets_60s) > 0 and len(bets_20s) > 0:
        s60 = bets_60s.sort_values('marketTime_local').reset_index(drop=True)
        s20 = bets_20s.sort_values('marketTime_local').reset_index(drop=True)
        ax.plot(s60['pnl_conservative'].cumsum() * FLAT_BET, label='60s conservative', color=C_BLUE, linewidth=1.5)
        ax.plot(s20['pnl_conservative'].cumsum() * FLAT_BET, label='20s conservative', color=C_ORANGE, linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Bet Number')
        ax.set_ylabel(f'Cumulative P&L ({FLAT_BET} AUD/bet)')
        ax.set_title('Cumulative P&L: 60s vs 20s Window', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    # Chart 3: Price variant comparison (60s window)
    ax = axes[1, 0]
    variants = {}
    for variant in ['minus_1_tick', 'best_back', 'plus_1_tick']:
        sub = fill[(fill['window'] == 't0_60s') & (fill['price_variant'] == variant)]
        sub_bets = sub[sub['edge'] > EDGE_THRESH]
        if len(sub_bets) > 0:
            variants[variant] = sub_bets['conservative_fill'].mean() * 100
    if variants:
        vx = np.arange(len(variants))
        colors = [C_RED, C_BLUE, C_GREEN]
        ax.bar(vx, list(variants.values()), color=colors[:len(variants)], alpha=0.8)
        ax.set_xticks(vx)
        ax.set_xticklabels(list(variants.keys()), fontsize=8)
        ax.set_ylabel('Conservative Fill Rate (%)')
        ax.set_title('Fill Rate by Price Variant (60s)', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 105)
        for i, v in enumerate(variants.values()):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=8)

    # Chart 4: Volume analysis
    ax = axes[1, 1]
    if len(bets_60s) > 0:
        vol = bets_60s[bets_60s['volume_at_price'] > 0]['volume_at_price']
        if len(vol) > 0:
            ax.hist(vol.clip(upper=vol.quantile(0.95)), bins=30, color=C_PURPLE, alpha=0.7, edgecolor='white')
            ax.axvline(vol.median(), color=C_RED, linestyle='--', label=f'Median: ${vol.median():.0f}')
            ax.set_xlabel('Volume Traded at Limit Price ($)')
            ax.set_ylabel('Count')
            ax.set_title('Volume at Limit Price (Conservative Fills)', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

print(f"\nReport saved to {output_path}", flush=True)
print("Done!", flush=True)
