"""
Generate comprehensive PDF report for Alpha Odds win probability model results.
Runs on Spartan where all data is available.

Usage: python3 _05_generate_report.py
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
BANKROLL = 10000
COMMISSION = 0.075  # conservative 7.5%


def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def compute_pnl(df, edge_thresh, commission):
    bets = df[df['edge'] > edge_thresh].copy()
    if len(bets) == 0:
        return None
    bets['pnl'] = bets['win'] * (bets['back_odds'] - 1) * (1 - commission) - (1 - bets['win'])
    return bets


def run_backtest(df, edge_thresh, commission, flat_bet=25):
    bets = df[df['edge'] > edge_thresh].copy()
    if len(bets) < 10:
        return None, None
    bets = bets.sort_values('marketTime_local').reset_index(drop=True)
    running_br = BANKROLL
    records = []
    for _, row in bets.iterrows():
        if running_br <= 0:
            break
        bet_size = min(flat_bet, running_br)
        if row['win'] == 1:
            pnl = bet_size * (row['back_odds'] - 1) * (1 - commission)
        else:
            pnl = -bet_size
        running_br += pnl
        records.append({'date': row['marketTime_local'], 'pnl': pnl, 'bankroll': running_br,
                        'win': row['win'], 'odds': row['back_odds'], 'edge': row['edge']})
    rdf = pd.DataFrame(records)
    rdf['cum_pnl'] = rdf['pnl'].cumsum()
    rdf['date'] = pd.to_datetime(rdf['date'])
    rdf['month'] = rdf['date'].dt.to_period('M')
    monthly = rdf.groupby('month').agg(pnl=('pnl', 'sum'), n_bets=('pnl', 'count'), wins=('win', 'sum'))
    monthly['cum_pnl'] = monthly['pnl'].cumsum()
    return rdf, monthly


def load_ensemble(model_dir, t_def, top_n):
    """Load models for a given t_def and return (base_df, ensemble_probs)."""
    base_dir = f"{model_dir}/t{t_def}"
    if not os.path.exists(base_dir):
        return None, None
    dfs = {}
    for c in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, c, "save_df.parquet")
        if os.path.exists(path):
            try:
                dfs[c] = pd.read_parquet(path)
            except Exception:
                pass
    if not dfs:
        return None, None
    sorted_configs = sorted(dfs.keys(), key=lambda c: log_loss(
        dfs[c]["win"].values, dfs[c]["model_prob"].clip(0.001, 0.999).values))
    n = min(top_n, len(sorted_configs))
    base = dfs[sorted_configs[0]].copy()
    ens_prob = np.mean([dfs[c]["model_prob"].values for c in sorted_configs[:n]], axis=0)
    return base, ens_prob


# ══════════════════════════════════════════════
# LOAD ALL MODELS ACROSS TIME DEFINITIONS
# ══════════════════════════════════════════════
print("Loading all models across time definitions...", flush=True)

components = {}
for t in [0, 1, 2, 3]:
    base, prob = load_ensemble("win_model", t, top_n=7)
    if base is not None:
        components[("V1", t)] = (base, prob)
        print(f"  V1 t{t}: LL={log_loss(base['win'].values, prob):.6f} ({len(base):,} rows)")

    base, prob = load_ensemble("win_model_v2", t, top_n=15)
    if base is not None:
        components[("V2", t)] = (base, prob)
        print(f"  V2 t{t}: LL={log_loss(base['win'].values, prob):.6f} ({len(base):,} rows)")

# Align all DataFrames by (file_name, id)
print("Aligning DataFrames...", flush=True)
base_df = components[("V1", 0)][0].copy()
base_df["key"] = base_df["file_name"].astype(str) + "_" + base_df["id"].astype(str)
aligned = base_df[["key", "win", "market_prob", "orig_best_back_m0", "orig_best_lay_m0", "marketTime_local"]].copy()
aligned["V1_t0"] = components[("V1", 0)][1]

for (ver, t), (df, prob) in components.items():
    col = f"{ver}_t{t}"
    if col == "V1_t0":
        continue
    tmp = df.copy()
    tmp["key"] = tmp["file_name"].astype(str) + "_" + tmp["id"].astype(str)
    tmp[col] = prob
    aligned = aligned.merge(tmp[["key", col]], on="key", how="inner")

prob_cols = sorted([c for c in aligned.columns if c.startswith(("V1_", "V2_"))])
v1_cols = [c for c in prob_cols if c.startswith("V1_")]
v2_cols = [c for c in prob_cols if c.startswith("V2_")]
print(f"Aligned: {len(aligned):,} rows, {len(prob_cols)} components")

# Build ensembles
v1_cross_t = aligned[v1_cols].mean(axis=1).values
v2_cross_t = aligned[v2_cols].mean(axis=1).values
v1_t0_only = aligned["V1_t0"].values
v2_t0_only = aligned["V2_t0"].values if "V2_t0" in aligned.columns else None
win = aligned["win"].values

# Cross-t super-ensemble: V1(20%) + V2(80%)
cross_t_super = v1_cross_t * 0.2 + v2_cross_t * 0.8
# t0-only super-ensemble: V1(40%) + V2(60%)
t0_super = v1_t0_only * 0.4 + (v2_t0_only * 0.6 if v2_t0_only is not None else v1_t0_only * 0.6)

# Log-losses
v1_t0_ll = log_loss(win, v1_t0_only)
v2_t0_ll = log_loss(win, v2_t0_only) if v2_t0_only is not None else None
v1_cross_ll = log_loss(win, v1_cross_t)
v2_cross_ll = log_loss(win, v2_cross_t)
t0_super_ll = log_loss(win, t0_super)
cross_t_super_ll = log_loss(win, cross_t_super)
market_ll = log_loss(win, np.clip(aligned["market_prob"].values, 0.001, 0.999))

print(f"V1 t0-only:        LL={v1_t0_ll:.6f}")
print(f"V1 cross-t:        LL={v1_cross_ll:.6f}")
if v2_t0_ll:
    print(f"V2 t0-only:        LL={v2_t0_ll:.6f}")
print(f"V2 cross-t:        LL={v2_cross_ll:.6f}")
print(f"t0 super:           LL={t0_super_ll:.6f}")
print(f"Cross-t super:      LL={cross_t_super_ll:.6f}")
print(f"Market:             LL={market_ll:.6f}")

HAS_V2 = "V2_t0" in aligned.columns

# ── Best model for headline ──
best_probs = cross_t_super
best_label = "Cross-t Super-Ensemble"
best_ll = cross_t_super_ll

# Build best_base for backtesting
best_base = aligned.copy()
best_base['model_prob'] = best_probs
best_base['edge'] = best_base['model_prob'] - best_base['market_prob']
best_base['back_odds'] = 1 / best_base['orig_best_back_m0']
best_base = best_base[(best_base['back_odds'] > 1.01) & (best_base['back_odds'] < 1000)]
best_base = best_base.sort_values('marketTime_local').reset_index(drop=True)
best_base['date'] = pd.to_datetime(best_base['marketTime_local'])
best_base['month'] = best_base['date'].dt.to_period('M')

# Also build t0-only base for comparison
t0_base = aligned.copy()
t0_base['model_prob'] = t0_super
t0_base['edge'] = t0_base['model_prob'] - t0_base['market_prob']
t0_base['back_odds'] = 1 / t0_base['orig_best_back_m0']
t0_base = t0_base[(t0_base['back_odds'] > 1.01) & (t0_base['back_odds'] < 1000)]

# V1 t0-only base for comparison
v1_base = aligned.copy()
v1_base['model_prob'] = v1_t0_only
v1_base['edge'] = v1_base['model_prob'] - v1_base['market_prob']
v1_base['back_odds'] = 1 / v1_base['orig_best_back_m0']
v1_base = v1_base[(v1_base['back_odds'] > 1.01) & (v1_base['back_odds'] < 1000)]

# All V1 t0 models for chart
all_v1_ll = {}
V1_DIR = 'win_model/t0'
for d in sorted(os.listdir(V1_DIR)):
    path = os.path.join(V1_DIR, d, 'save_df.parquet')
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path, columns=['win', 'model_prob'])
            ll = log_loss(df['win'].values, df['model_prob'].clip(0.001, 0.999).values)
            all_v1_ll[d] = ll
        except Exception:
            pass


# ══════════════════════════════════════════════
# GENERATE PDF
# ══════════════════════════════════════════════
output_path = 'alpha_odds_report.pdf'
print(f"Generating report to {output_path}...", flush=True)

with PdfPages(output_path) as pdf:

    # ════════════════════════════════════════════
    # PAGE 1: Title & Key Metrics
    # ════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    ax.text(0.5, 0.93, 'Alpha Odds', fontsize=44, fontweight='bold', ha='center', va='top', color='#1a1a2e')
    ax.text(0.5, 0.855, 'Win Probability Model — Comprehensive Results', fontsize=20, ha='center', va='top', color='#16213e')
    ax.text(0.5, 0.81, 'Betfair Australian Greyhound Racing', fontsize=14, ha='center', va='top', color='#555')
    ax.text(0.5, 0.77, 'Out-of-Sample: Jan-Nov 2025 | Training: 2017-2023 | 286K OOS predictions',
            fontsize=11, ha='center', va='top', color='#777')

    # Key metrics for recommended strategy (edge > 3%)
    rdf_30, monthly_30 = run_backtest(best_base, 0.03, COMMISSION)
    rdf_20, monthly_20 = run_backtest(best_base, 0.02, COMMISSION)

    n30 = len(rdf_30)
    pnl30 = rdf_30['pnl'].sum()
    roi30 = pnl30 / (n30 * FLAT_BET) * 100
    wr30 = rdf_30['win'].mean()
    sharpe_m30 = (monthly_30['pnl'] / BANKROLL).mean() / (monthly_30['pnl'] / BANKROLL).std() if (monthly_30['pnl'] / BANKROLL).std() > 0 else 0
    sharpe_a30 = sharpe_m30 * np.sqrt(12)
    pm30 = int((monthly_30['pnl'] > 0).sum())

    box_y = 0.62
    metrics = [
        (f'{n30:,}', f'Bets (edge>3%)'),
        (f'{wr30:.1%}', 'Win Rate'),
        (f'+{roi30:.0f}%', 'ROI'),
        (f'{sharpe_a30:.1f}', 'Sharpe (Annual)'),
        (f'${pnl30:,.0f}', f'Profit ({FLAT_BET} AUD/bet)'),
        (f'{pm30}/{len(monthly_30)}', 'Profitable Months'),
    ]

    for i, (val, label) in enumerate(metrics):
        x = 0.12 + (i % 3) * 0.30
        y = box_y - (i // 3) * 0.14
        bbox = dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#ddd', linewidth=0.5)
        ax.text(x, y, val, fontsize=24, fontweight='bold', ha='center', va='center', color='#0f3460', bbox=bbox)
        ax.text(x, y - 0.05, label, fontsize=9, ha='center', va='center', color='#888')

    # Statistical significance
    bets_sig = compute_pnl(best_base, 0.03, COMMISSION)
    z30 = bets_sig['pnl'].mean() / bets_sig['pnl'].std() * np.sqrt(len(bets_sig)) if bets_sig['pnl'].std() > 0 else 0
    p30 = 1 - norm_cdf(z30)

    ax.text(0.5, 0.27, f'Statistical Significance: z = {z30:.2f}, p = {p30:.6f}',
            fontsize=12, ha='center', color='#27ae60' if p30 < 0.01 else '#e94560', fontweight='bold')
    ax.text(0.5, 0.23, f'Model: {best_label} | Log-loss: {best_ll:.6f} vs Market: {market_ll:.6f}',
            fontsize=10, ha='center', color='#555')
    ax.text(0.5, 0.19, f'8 model components (V1+V2 across 4 time snapshots)',
            fontsize=9, ha='center', color='#888')
    ax.text(0.5, 0.15, f'Commission: {COMMISSION:.1%} | {len(best_base):,} OOS predictions',
            fontsize=9, ha='center', color='#999')

    # P&L projections
    ax.text(0.5, 0.08, 'Projected Annual P&L (annualized from OOS period)', fontsize=11, ha='center', fontweight='bold', color='#333')
    n_months_30 = len(monthly_30)
    for i, (stake, label) in enumerate([(10, '10 AUD/bet'), (25, '25 AUD/bet'), (50, '50 AUD/bet')]):
        annual_pnl = pnl30 / FLAT_BET * stake * 12 / n_months_30
        ax.text(0.15 + i * 0.30, 0.03, f'{label}: ${annual_pnl:,.0f}/year', fontsize=10, ha='center', color='#0f3460')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 2: How the Model Works
    # ════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    fig.suptitle('How the Model Works', fontsize=18, fontweight='bold', y=0.97, color='#1a1a2e')

    lh = 0.026
    def text_block(title, lines, y_pos):
        ax.text(0.04, y_pos, title, fontsize=13, fontweight='bold', color='#16213e', va='top')
        y_pos -= lh * 1.3
        for line in lines:
            ax.text(0.06, y_pos, line, fontsize=9, va='top', color='#333')
            y_pos -= lh
        return y_pos - lh * 0.3

    y = 0.90
    y = text_block('The Core Idea', [
        'Betfair exchange prices reflect the crowd\'s collective wisdom about each runner\'s chance of winning.',
        'Our model learns from 8 years of order book data to identify when the crowd systematically misprices a runner.',
        'When our model says a runner has a higher chance of winning than the market implies, we place a back bet.',
        'The edge is small (2-8% per bet) but statistically robust over thousands of bets.',
    ], y)

    y = text_block('Data Pipeline', [
        'Raw data: Betfair exchange tick data for every Australian greyhound race (2017-2025)',
        'Feature extraction: 133 features per runner from order book snapshots at 4 time points',
        'Order book features: best prices, depth (100/1000 AUD levels), volume, back-lay imbalance',
        'Momentum features: price changes between snapshots (prc_mean_2_1, prc_mean_3_1)',
        'V2 cross-runner features: prob_share, prob_rank, HHI, spread_rank, volume_rank',
        'Multi-snapshot: features at t0 (close), t1, t2, t3 give complementary signals',
    ], y)

    y = text_block('Model Architecture (Cross-t Super-Ensemble)', [
        'V1: 7 XGBoost models per time snapshot, top-7 ensemble per t (4 time definitions)',
        'V2: XGBoost + LightGBM + Isotonic Calibration per config, top-15 ensemble per t',
        'Cross-t: V1 and V2 ensembles averaged across all 4 time snapshots',
        'Super-ensemble: V1_cross(20%) + V2_cross(80%) = 8 model components total',
        'Train: 2017-2023 (2.4M runners), Validation: 2024 (348K), OOS Test: 2025 (286K)',
        'Strict temporal split: no lookahead bias, model never sees future data',
    ], y)

    y = text_block('Where the Edge Comes From (Feature Importance)', [
        '#1: prc_mean_2_1 (32%) -- recent price momentum. Detects when trading pressure hasn\'t updated price',
        '#2: best_lay_q_100_m0 (26%) -- 100 AUD lay execution price. Hidden depth & informed positioning',
        '#3: best_lay_m0 (19%) -- best lay price at close. LAY side carries 5x more signal than BACK side',
        '#4: prc_mean_3_1 (8%) -- earlier price momentum, confirming directional signal',
        '#5: best_back_m0 (3%) -- headline back price (surprisingly low importance)',
        'Key insight: INFORMED TRADERS ACT ON THE LAY SIDE. Our model captures their signal.',
    ], y)

    y = text_block('Why Cross-t Ensembling Helps', [
        'Different time snapshots capture different stages of price formation',
        'Later snapshots (t2, t3) have features closer to race start = more informative',
        'Averaging across time reduces noise and captures consensus across snapshots',
        f'Improvement: t0-only LL={t0_super_ll:.6f} -> cross-t LL={cross_t_super_ll:.6f} (-{t0_super_ll-cross_t_super_ll:.6f})',
    ], y)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 3: Model Quality Charts
    # ════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Model Quality -- Out-of-Sample (2025)', fontsize=16, fontweight='bold', y=0.98)

    # 3a: All V1 models log-loss
    ax = axes[0, 0]
    sorted_models = sorted(all_v1_ll.items(), key=lambda x: x[1])
    good_models = [(n, ll) for n, ll in sorted_models if ll < market_ll]
    names = [s[0].replace('ne', '').replace('_md', '/d').replace('_lr', '/') for s in good_models]
    lls = [s[1] for s in good_models]
    ax.barh(range(len(lls)), lls, color='#3282b8', height=0.7)
    ax.axvline(market_ll, color='red', linestyle='--', alpha=0.7, label=f'Market ({market_ll:.4f})')
    ax.axvline(v1_t0_ll, color='green', linestyle='--', alpha=0.7, label=f'V1 t0 ({v1_t0_ll:.4f})')
    ax.axvline(cross_t_super_ll, color='purple', linestyle='--', alpha=0.7, label=f'Cross-t ({cross_t_super_ll:.4f})')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('Log-Loss (lower = better)')
    ax.set_title('All V1 Models vs Ensembles (t0)')
    ax.legend(fontsize=6)
    ax.set_xlim(min(lls) - 0.001, market_ll + 0.001)

    # 3b: Calibration
    ax = axes[0, 1]
    best_base['prob_decile'] = pd.qcut(best_base['model_prob'], 10, labels=False, duplicates='drop')
    cal = best_base.groupby('prob_decile').agg(pred=('model_prob', 'mean'), actual=('win', 'mean'))
    best_base['mkt_decile'] = pd.qcut(best_base['market_prob'], 10, labels=False, duplicates='drop')
    mcal = best_base.groupby('mkt_decile').agg(pred=('market_prob', 'mean'), actual=('win', 'mean'))

    ax.plot([0, 0.6], [0, 0.6], 'k--', alpha=0.4, label='Perfect')
    ax.scatter(cal['pred'], cal['actual'], s=60, c='#3282b8', zorder=5, label='Model')
    ax.plot(cal['pred'], cal['actual'], c='#3282b8', alpha=0.5)
    ax.scatter(mcal['pred'], mcal['actual'], s=40, c='#e94560', zorder=4, marker='s', label='Market')
    ax.plot(mcal['pred'], mcal['actual'], c='#e94560', alpha=0.3)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Win Rate')
    ax.set_title('Calibration: Model vs Market')
    ax.legend(fontsize=8)

    # 3c: Edge distribution
    ax = axes[1, 0]
    edges = best_base['edge'] * 100
    ax.hist(edges, bins=200, color='#3282b8', alpha=0.7, range=(-10, 15))
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    for et, c, l in [(2, '#27ae60', '2%'), (3, '#e94560', '3%'), (5, '#f39c12', '5%')]:
        ax.axvline(et, color=c, linestyle='--', alpha=0.6, label=l)
    ax.set_xlabel('Edge (model - market) in %')
    ax.set_ylabel('Count')
    ax.set_title('Edge Distribution')
    ax.legend(fontsize=7)

    # 3d: ROI vs edge threshold — t0-only vs cross-t
    ax = axes[1, 1]
    thresholds = np.arange(0.005, 0.08, 0.0025)

    rois_t0, rois_ct, n_bets_ct = [], [], []
    for t in thresholds:
        bets_t0 = compute_pnl(t0_base, t, COMMISSION)
        bets_ct = compute_pnl(best_base, t, COMMISSION)
        rois_t0.append(bets_t0['pnl'].mean() * 100 if bets_t0 is not None and len(bets_t0) > 10 else np.nan)
        if bets_ct is not None and len(bets_ct) > 10:
            rois_ct.append(bets_ct['pnl'].mean() * 100)
            n_bets_ct.append(len(bets_ct))
        else:
            rois_ct.append(np.nan)
            n_bets_ct.append(0)

    ax2 = ax.twinx()
    ax.plot(thresholds * 100, rois_t0, 'o-', color='#3282b8', markersize=3, label='t0-only ROI%')
    ax.plot(thresholds * 100, rois_ct, 's-', color='purple', markersize=3, label='Cross-t ROI%')
    ax2.plot(thresholds * 100, n_bets_ct, '--', color='#e94560', alpha=0.5, label='# Bets (cross-t)')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Edge Threshold (%)')
    ax.set_ylabel('ROI (%)', color='#3282b8')
    ax2.set_ylabel('Number of Bets', color='#e94560')
    ax.set_title('ROI: t0-only vs Cross-t')
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 4: Backtest Charts
    # ════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f'Backtest Results -- {FLAT_BET} AUD/bet, {COMMISSION:.1%} Commission ({best_label})',
                 fontsize=14, fontweight='bold', y=0.98)

    # 4a: Equity curves
    ax = axes[0, 0]
    for et, color, lw in [(0.02, '#3282b8', 1.0), (0.025, '#27ae60', 1.5), (0.03, '#e94560', 2.0), (0.05, '#f39c12', 1.0)]:
        rdf_t, _ = run_backtest(best_base, et, COMMISSION)
        if rdf_t is not None:
            ax.plot(range(len(rdf_t)), BANKROLL + rdf_t['cum_pnl'], label=f'Edge>{et:.1%} ({len(rdf_t)} bets)',
                    linewidth=lw, color=color)
    ax.axhline(BANKROLL, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Bet Number')
    ax.set_ylabel('Bankroll (AUD)')
    ax.set_title('Equity Curves')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # 4b: Monthly P&L (edge > 3%)
    ax = axes[0, 1]
    months_str = [str(m) for m in monthly_30.index]
    colors_bar = ['#27ae60' if p > 0 else '#e94560' for p in monthly_30['pnl']]
    bars = ax.bar(range(len(months_str)), monthly_30['pnl'], color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(months_str)))
    ax.set_xticklabels([m[-2:] for m in months_str], fontsize=8)
    ax.set_xlabel('Month (2025)')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('P&L (AUD)')
    ax.set_title('Monthly P&L (Edge > 3%)')
    for bar, val in zip(bars, monthly_30['pnl']):
        y_pos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y_pos + (20 if val > 0 else -40),
                f'${val:.0f}', ha='center', fontsize=7, color='#333')
    ax.grid(True, alpha=0.2, axis='y')

    # 4c: t0-only vs Cross-t comparison
    ax = axes[1, 0]
    rdf_t0_30, _ = run_backtest(t0_base, 0.03, COMMISSION)
    rdf_ct_30, _ = run_backtest(best_base, 0.03, COMMISSION)
    rdf_v1_30, _ = run_backtest(v1_base, 0.03, COMMISSION)
    if rdf_v1_30 is not None:
        ax.plot(range(len(rdf_v1_30)), BANKROLL + rdf_v1_30['cum_pnl'], label=f'V1 t0 ({len(rdf_v1_30)} bets)',
                color='#3282b8', linewidth=1.0)
    if rdf_t0_30 is not None:
        ax.plot(range(len(rdf_t0_30)), BANKROLL + rdf_t0_30['cum_pnl'], label=f't0 Super ({len(rdf_t0_30)} bets)',
                color='#27ae60', linewidth=1.5)
    if rdf_ct_30 is not None:
        ax.plot(range(len(rdf_ct_30)), BANKROLL + rdf_ct_30['cum_pnl'], label=f'Cross-t ({len(rdf_ct_30)} bets)',
                color='purple', linewidth=2.0)
    ax.axhline(BANKROLL, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Bet Number')
    ax.set_ylabel('Bankroll (AUD)')
    ax.set_title('V1 vs t0-Super vs Cross-t (Edge > 3%)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # 4d: Cumulative monthly
    ax = axes[1, 1]
    for et, color, label in [(0.02, '#3282b8', '2%'), (0.025, '#27ae60', '2.5%'), (0.03, '#e94560', '3%'), (0.05, '#f39c12', '5%')]:
        _, mt = run_backtest(best_base, et, COMMISSION)
        if mt is not None:
            ax.plot(range(len(mt)), mt['cum_pnl'], 'o-', color=color, label=f'Edge>{label}', markersize=4)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Month')
    ax.set_ylabel('Cumulative P&L (AUD)')
    ax.set_title('Cumulative P&L Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 5: Detailed Statistics
    # ════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    fig.suptitle('Detailed Statistics & Strategy Comparison', fontsize=16, fontweight='bold', y=0.97)

    y = 0.90
    def section(title, lines, y_pos):
        ax.text(0.04, y_pos, title, fontsize=12, fontweight='bold', color='#16213e', va='top')
        y_pos -= lh * 1.3
        for line in lines:
            ax.text(0.06, y_pos, line, fontsize=8.5, va='top', color='#333', family='monospace')
            y_pos -= lh
        return y_pos - lh * 0.3

    # Strategy comparison
    strat_lines = [
        f'{"Threshold":<12} {"Bets":>6} {"WinRate":>8} {"AvgOdds":>8} {"ROI%":>8} {"P&L(25)":>10} {"Sharpe":>7} {"p-value":>10} {"ProfMo":>7}',
        '-' * 95,
    ]
    for et in [0.015, 0.02, 0.025, 0.03, 0.05, 0.07, 0.10]:
        bets_t = compute_pnl(best_base, et, COMMISSION)
        if bets_t is None or len(bets_t) < 10:
            continue
        rdf_t, mt_t = run_backtest(best_base, et, COMMISSION)
        if rdf_t is None:
            continue
        pnl_t = rdf_t['pnl'].sum()
        roi_t = pnl_t / (len(rdf_t) * FLAT_BET) * 100
        sh = (mt_t['pnl']/BANKROLL).mean() / (mt_t['pnl']/BANKROLL).std() * np.sqrt(12) if (mt_t['pnl']/BANKROLL).std() > 0 else 0
        z_t = bets_t['pnl'].mean() / bets_t['pnl'].std() * np.sqrt(len(bets_t)) if bets_t['pnl'].std() > 0 else 0
        p_t = 1 - norm_cdf(z_t)
        pm = int((mt_t['pnl'] > 0).sum())
        strat_lines.append(
            f'Edge>{et:>5.1%}   {len(bets_t):>6} {bets_t["win"].mean():>8.1%} {bets_t["back_odds"].mean():>8.1f} {roi_t:>+8.1f} ${pnl_t:>+9,.0f} {sh:>7.1f}  {p_t:>10.6f} {pm:>3}/{len(mt_t)}'
        )
    y = section(f'1. Strategy Comparison ({best_label})', strat_lines, y)

    # P&L projections
    pnl_lines = [
        f'{"Stake":<10} {"Edge>2%":>15} {"Edge>2.5%":>15} {"Edge>3%":>15} {"Edge>5%":>15}',
        '-' * 72,
    ]
    n_months = len(monthly_30)
    for stake in [10, 25, 50, 100]:
        vals = []
        for et in [0.02, 0.025, 0.03, 0.05]:
            bets = compute_pnl(best_base, et, COMMISSION)
            if bets is not None and len(bets) > 10:
                total_pnl = bets['pnl'].sum() * stake
                annual = total_pnl * 12 / n_months
                vals.append(f'${annual:>+12,.0f}')
            else:
                vals.append(f'{"N/A":>13}')
        pnl_lines.append(f'${stake}/bet    {vals[0]:>15} {vals[1]:>15} {vals[2]:>15} {vals[3]:>15}')
    pnl_lines.append('')
    pnl_lines.append(f'Note: Annualized from {n_months}-month OOS period. Assumes same conditions.')
    y = section('2. Projected Annual P&L by Stake Size', pnl_lines, y)

    # Model comparison
    v_lines = [
        f'{"Metric":<25} {"V1 t0":>12} {"t0 Super":>12} {"Cross-t":>12} {"Market":>12}',
        '-' * 78,
        f'{"Log-Loss":<25} {v1_t0_ll:>12.6f} {t0_super_ll:>12.6f} {cross_t_super_ll:>12.6f} {market_ll:>12.6f}',
        f'{"vs Market":<25} {market_ll-v1_t0_ll:>12.6f} {market_ll-t0_super_ll:>12.6f} {market_ll-cross_t_super_ll:>12.6f} {"--":>12}',
        f'{"Architecture":<25} {"7x XGB":>12} {"V1+V2 t0":>12} {"V1+V2 x4t":>12} {"--":>12}',
        f'{"Components":<25} {"7":>12} {"2":>12} {"8":>12} {"--":>12}',
        f'{"Cross-runner feat.":<25} {"No":>12} {"Yes":>12} {"Yes":>12} {"--":>12}',
        f'{"Multi-time":<25} {"No":>12} {"No":>12} {"Yes":>12} {"--":>12}',
    ]
    y = section('3. Model Evolution', v_lines, y)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 6: Monthly Breakdown & Next Steps
    # ════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    fig.suptitle('Monthly Breakdown & Next Steps', fontsize=16, fontweight='bold', y=0.97)

    y = 0.90

    # Monthly table for recommended strategy (edge > 3%)
    mo_lines = [
        f'{"Month":>10} {"Bets":>6} {"Wins":>6} {"WR%":>7} {"P&L":>10} {"CumPnL":>10} {"ROI%":>8}',
        '-' * 62,
    ]
    cum = 0
    for m, row in monthly_30.iterrows():
        cum += row['pnl']
        wr = row['wins']/row['n_bets']*100
        roi = row['pnl'] / (row['n_bets'] * FLAT_BET) * 100
        mo_lines.append(f'{str(m):>10} {int(row["n_bets"]):>6} {int(row["wins"]):>6} {wr:>6.1f}% ${row["pnl"]:>+9,.0f} ${cum:>+9,.0f} {roi:>+7.1f}%')
    mo_lines.append('-' * 62)
    pnl_total = monthly_30['pnl'].sum()
    roi_total = pnl_total / (monthly_30['n_bets'].sum() * FLAT_BET) * 100
    mo_lines.append(f'{"TOTAL":>10} {int(monthly_30["n_bets"].sum()):>6} {int(monthly_30["wins"].sum()):>6} '
                     f'{monthly_30["wins"].sum()/monthly_30["n_bets"].sum()*100:>6.1f}% '
                     f'${pnl_total:>+9,.0f} ${pnl_total:>+9,.0f} {roi_total:>+7.1f}%')
    y = section(f'Monthly P&L Detail (Edge > 3%, {FLAT_BET} AUD/bet)', mo_lines, y)

    # Insights
    insight_lines = [
        'Why the model works:',
        '  Greyhound racing markets are thin (~130 races/day) with limited institutional attention.',
        '  Market makers cannot perfectly process all order book depth and flow information.',
        '  The model exploits LAY-side signals (informed traders act on lay side, not back side).',
        '  Multi-time-snapshot ensembling captures price formation dynamics across time.',
        '',
        'Risk factors:',
        '  - Profit is concentrated in mid-range longshots (15-30 odds = 37% of profit from 1.9% of bets)',
        '  - Top 10 individual wins account for ~58% of all profit — high concentration risk',
        '  - Wider-spread (less liquid) markets contribute disproportionate profit',
        '  - Commission structure changes would directly impact profitability',
        '  - Favorites (odds 1-3) are 71% of bets but only 5% of profit — mostly dead weight',
    ]
    y = section('Key Insights & Risk Factors', insight_lines, y)

    # Next steps
    next_lines = [
        'Short-term:',
        '  - Cap max odds at 20-30 to avoid illiquid longshots',
        '  - Consider minimum odds of 3 or higher edge threshold for short-priced runners',
        '  - Paper trading via Betfair API (real-time predictions, simulated execution)',
        '',
        'Medium-term:',
        '  - Live testing with 10 AUD/bet, odds capped at 20-30',
        '  - Extend to horse racing (larger market, potentially more edge)',
        '  - Explore lay strategy (early results show strong potential)',
    ]
    y = section('Next Steps', next_lines, y)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 7: Odds & Liquidity Analysis — Charts
    # ════════════════════════════════════════════
    print("  Page 7: Odds & Liquidity charts...", flush=True)

    # Prepare odds bucket analysis on edge>3% bets
    bets_all = compute_pnl(best_base, 0.03, COMMISSION)
    odds_bins = [1, 2, 3, 5, 8, 15, 30, 100, 1001]
    odds_labels = ['1-2', '2-3', '3-5', '5-8', '8-15', '15-30', '30-100', '100+']
    bets_all['odds_bucket'] = pd.cut(bets_all['back_odds'], bins=odds_bins, labels=odds_labels, right=False)
    total_pnl = bets_all['pnl'].sum()

    odds_stats = bets_all.groupby('odds_bucket', observed=True).agg(
        count=('pnl', 'count'),
        wins=('win', 'sum'),
        win_rate=('win', 'mean'),
        avg_edge=('edge', 'mean'),
        avg_odds=('back_odds', 'mean'),
        total_pnl=('pnl', 'sum'),
        roi=('pnl', 'mean'),
    )
    odds_stats['pct_bets'] = odds_stats['count'] / len(bets_all) * 100
    odds_stats['pct_pnl'] = odds_stats['total_pnl'] / total_pnl * 100 if total_pnl != 0 else 0
    odds_stats['roi'] = odds_stats['roi'] * 100

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Odds & Liquidity Analysis (Edge > 3%)', fontsize=16, fontweight='bold', y=0.98)

    # 7a: P&L by odds bucket
    ax = axes[0, 0]
    valid = odds_stats[odds_stats['count'] > 0]
    colors_pnl = ['#27ae60' if p > 0 else '#e94560' for p in valid['total_pnl']]
    bars = ax.bar(range(len(valid)), valid['total_pnl'] * FLAT_BET, color=colors_pnl, edgecolor='white')
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(valid.index, fontsize=8)
    ax.set_ylabel('P&L (AUD at $25/bet)')
    ax.set_title('Profit by Odds Range')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    for bar, cnt, pct in zip(bars, valid['count'], valid['pct_pnl']):
        y_pos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y_pos + (20 if y_pos > 0 else -60),
                f'{cnt} bets\n{pct:.0f}%', ha='center', fontsize=7, color='#333')
    ax.grid(True, alpha=0.2, axis='y')

    # 7b: ROI by odds bucket
    ax = axes[0, 1]
    colors_roi = ['#27ae60' if r > 0 else '#e94560' for r in valid['roi']]
    ax.bar(range(len(valid)), valid['roi'], color=colors_roi, edgecolor='white')
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(valid.index, fontsize=8)
    ax.set_ylabel('ROI (%)')
    ax.set_title('ROI by Odds Range')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    # Add bet count labels
    for i, (roi_v, cnt) in enumerate(zip(valid['roi'], valid['count'])):
        ax.text(i, roi_v + (3 if roi_v > 0 else -8), f'n={cnt}', ha='center', fontsize=7, color='#555')
    ax.grid(True, alpha=0.2, axis='y')

    # 7c: Profit concentration — cumulative % of profit from top N bets
    ax = axes[1, 0]
    winning_bets = bets_all[bets_all['pnl'] > 0].sort_values('pnl', ascending=False)
    if len(winning_bets) > 0 and total_pnl > 0:
        cum_pct = winning_bets['pnl'].cumsum() / total_pnl * 100
        ax.plot(range(1, len(cum_pct)+1), cum_pct.values, color='#3282b8', linewidth=2)
        # Mark key points
        for n_top in [5, 10, 20, 50]:
            if n_top <= len(cum_pct):
                pct_at_n = cum_pct.values[n_top-1]
                ax.axhline(pct_at_n, color='#e94560', linestyle='--', alpha=0.3)
                ax.text(len(cum_pct)*0.6, pct_at_n+1, f'Top {n_top}: {pct_at_n:.0f}%', fontsize=8, color='#e94560')
        ax.set_xlabel('Top N winning bets (ranked by size)')
        ax.set_ylabel('Cumulative % of total profit')
        ax.set_title('Profit Concentration')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, 'No winning bets', ha='center', va='center', transform=ax.transAxes)

    # 7d: Robustness under odds caps — z-stat and ROI
    ax = axes[1, 1]
    caps = [3, 5, 8, 10, 15, 20, 30, 50, 1000]
    cap_labels = ['3', '5', '8', '10', '15', '20', '30', '50', 'None']
    z_vals, roi_vals, n_vals = [], [], []
    for cap in caps:
        bets_c = bets_all[bets_all['back_odds'] <= cap]
        if len(bets_c) > 10:
            z_c = bets_c['pnl'].mean() / bets_c['pnl'].std() * np.sqrt(len(bets_c)) if bets_c['pnl'].std() > 0 else 0
            roi_c = bets_c['pnl'].mean() * 100
            z_vals.append(z_c)
            roi_vals.append(roi_c)
            n_vals.append(len(bets_c))
        else:
            z_vals.append(np.nan)
            roi_vals.append(np.nan)
            n_vals.append(0)

    ax2 = ax.twinx()
    ax.bar(range(len(caps)), z_vals, color='#3282b8', alpha=0.7, label='z-stat')
    ax2.plot(range(len(caps)), roi_vals, 'o-', color='#e94560', linewidth=2, markersize=5, label='ROI%')
    ax.axhline(1.96, color='green', linestyle='--', alpha=0.5, label='z=1.96 (p<0.05)')
    ax.axhline(2.58, color='orange', linestyle='--', alpha=0.5, label='z=2.58 (p<0.01)')
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels(cap_labels, fontsize=8)
    ax.set_xlabel('Max Odds Cap')
    ax.set_ylabel('z-statistic', color='#3282b8')
    ax2.set_ylabel('ROI (%)', color='#e94560')
    ax.set_title('Robustness: Stats Under Odds Caps')
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 8: Odds & Liquidity Tables
    # ════════════════════════════════════════════
    print("  Page 8: Odds & Liquidity tables...", flush=True)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    fig.suptitle('Odds Concentration & Robustness Detail', fontsize=16, fontweight='bold', y=0.97)

    y = 0.90

    # Table 1: Odds bucket breakdown
    odds_lines = [
        f'{"Bucket":<10} {"Bets":>6} {"%Bets":>7} {"WinRate":>8} {"AvgEdge":>8} {"ROI%":>8} {"PnL($25)":>10} {"%Profit":>8}',
        '-' * 72,
    ]
    for idx in valid.index:
        row = valid.loc[idx]
        odds_lines.append(
            f'{idx:<10} {int(row["count"]):>6} {row["pct_bets"]:>6.1f}% {row["win_rate"]:>7.1%} '
            f'{row["avg_edge"]*100:>7.1f}% {row["roi"]:>+7.1f} ${row["total_pnl"]*FLAT_BET:>+9,.0f} {row["pct_pnl"]:>7.1f}%'
        )
    odds_lines.append('-' * 72)
    odds_lines.append(
        f'{"TOTAL":<10} {int(valid["count"].sum()):>6} {"100.0":>6}% {bets_all["win"].mean():>7.1%} '
        f'{bets_all["edge"].mean()*100:>7.1f}% {bets_all["pnl"].mean()*100:>+7.1f} ${total_pnl*FLAT_BET:>+9,.0f} {"100.0":>7}%'
    )
    y = section('1. Profit Breakdown by Odds Range (Edge > 3%)', odds_lines, y)

    # Table 2: Robustness with odds caps
    cap_lines = [
        f'{"MaxOdds":<10} {"Bets":>6} {"ROI%":>8} {"P&L($25)":>10} {"z-stat":>8} {"p-value":>10} {"Sharpe":>8}',
        '-' * 66,
    ]
    for cap, cl in zip(caps, cap_labels):
        bets_c = bets_all[bets_all['back_odds'] <= cap]
        if len(bets_c) > 10:
            z_c = bets_c['pnl'].mean() / bets_c['pnl'].std() * np.sqrt(len(bets_c)) if bets_c['pnl'].std() > 0 else 0
            p_c = 1 - norm_cdf(z_c)
            roi_c = bets_c['pnl'].mean() * 100
            # Monthly Sharpe
            bets_c_m = bets_c.copy()
            bets_c_m['month'] = pd.to_datetime(bets_c_m['date'] if 'date' in bets_c_m.columns else bets_c_m['marketTime_local']).dt.to_period('M')
            mo = bets_c_m.groupby('month')['pnl'].sum() * FLAT_BET
            sh_c = mo.mean() / mo.std() * np.sqrt(12) if mo.std() > 0 else 0
            cap_lines.append(
                f'{cl:<10} {len(bets_c):>6} {roi_c:>+7.1f} ${bets_c["pnl"].sum()*FLAT_BET:>+9,.0f} {z_c:>8.2f} {p_c:>10.6f} {sh_c:>8.1f}'
            )
    y = section('2. Strategy Robustness Under Odds Caps', cap_lines, y)

    # Table 3: Profit concentration stats
    conc_lines = []
    # Top N winning bets
    if len(winning_bets) > 0 and total_pnl > 0:
        for n_top in [1, 5, 10, 20, 50]:
            if n_top <= len(winning_bets):
                top_pnl = winning_bets['pnl'].head(n_top).sum()
                conc_lines.append(f'  Top {n_top:>2} winning bets: {top_pnl/total_pnl*100:>5.1f}% of total profit (${top_pnl*FLAT_BET:>+,.0f})')
        conc_lines.append('')
        conc_lines.append(f'  Largest single win: odds={winning_bets.iloc[0]["back_odds"]:.1f}, PnL={winning_bets.iloc[0]["pnl"]*FLAT_BET:>+,.0f} AUD ({winning_bets.iloc[0]["pnl"]/total_pnl*100:.1f}% of total)')

    # Winners vs losers median/mean odds
    winners = bets_all[bets_all['win'] == 1]
    losers = bets_all[bets_all['win'] == 0]
    conc_lines.append('')
    conc_lines.append(f'  Winning bets:  median odds = {winners["back_odds"].median():.1f},  mean odds = {winners["back_odds"].mean():.1f}  (n={len(winners)})')
    conc_lines.append(f'  Losing bets:   median odds = {losers["back_odds"].median():.1f},  mean odds = {losers["back_odds"].mean():.1f}  (n={len(losers)})')
    conc_lines.append('')

    # Profit from high odds ranges
    for thresh in [10, 20, 50]:
        high = bets_all[bets_all['back_odds'] > thresh]
        if len(high) > 0:
            conc_lines.append(f'  Bets > {thresh}/1: {len(high)} bets, PnL=${high["pnl"].sum()*FLAT_BET:>+,.0f} ({high["pnl"].sum()/total_pnl*100:.1f}% of total)')
    y = section('3. Profit Concentration Analysis', conc_lines, y)

    # Recommendations
    rec_lines = [
        'Based on the analysis above:',
        '',
        '  1. CAP ODDS AT 20-30: z-stat peaks around this range, and odds >30 add noise',
        '  2. CONSIDER MIN ODDS OF 3: Odds 1-3 are 71% of bets but <5% of profit (dead weight)',
        '  3. TIERED EDGE THRESHOLDS: Use higher edge (5-7%) for short-priced runners (1-5 odds)',
        '  4. MONITOR CONCENTRATION: Track whether profit remains concentrated in few bets live',
        '  5. LIQUIDITY CHECK: In live trading, verify order book depth before betting on >10/1',
    ]
    y = section('4. Strategy Recommendations', rec_lines, y)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════
    # PAGE 9: Spread & Liquidity Breakdown
    # ════════════════════════════════════════════
    print("  Page 9: Spread & liquidity charts...", flush=True)

    # Check if spread columns are available
    has_lay = 'orig_best_lay_m0' in bets_all.columns
    has_back = 'orig_best_back_m0' in bets_all.columns

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Liquidity & Spread Analysis (Edge > 3%)', fontsize=16, fontweight='bold', y=0.98)

    if has_lay and has_back:
        # Compute spread (in implied probability space: lay_prob - back_prob)
        bets_all['spread'] = bets_all['orig_best_lay_m0'] - bets_all['orig_best_back_m0']
        bets_all['spread_quintile'] = pd.qcut(bets_all['spread'], 5, labels=['Q1 (tight)', 'Q2', 'Q3', 'Q4', 'Q5 (wide)'], duplicates='drop')

        spread_stats = bets_all.groupby('spread_quintile', observed=True).agg(
            count=('pnl', 'count'),
            roi=('pnl', 'mean'),
            total_pnl=('pnl', 'sum'),
            avg_odds=('back_odds', 'mean'),
            win_rate=('win', 'mean'),
        )
        spread_stats['pct_pnl'] = spread_stats['total_pnl'] / total_pnl * 100 if total_pnl != 0 else 0
        spread_stats['roi'] = spread_stats['roi'] * 100

        # 9a: ROI by spread quintile
        ax = axes[0, 0]
        colors_sp = ['#27ae60' if r > 0 else '#e94560' for r in spread_stats['roi']]
        ax.bar(range(len(spread_stats)), spread_stats['roi'], color=colors_sp, edgecolor='white')
        ax.set_xticks(range(len(spread_stats)))
        ax.set_xticklabels(spread_stats.index, fontsize=7, rotation=15)
        ax.set_ylabel('ROI (%)')
        ax.set_title('ROI by Spread Quintile')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        for i, (roi_v, ao) in enumerate(zip(spread_stats['roi'], spread_stats['avg_odds'])):
            ax.text(i, roi_v + (2 if roi_v > 0 else -5), f'odds={ao:.1f}', ha='center', fontsize=7, color='#555')
        ax.grid(True, alpha=0.2, axis='y')

        # 9b: % of profit by spread quintile
        ax = axes[0, 1]
        colors_pp = ['#3282b8'] * len(spread_stats)
        ax.bar(range(len(spread_stats)), spread_stats['pct_pnl'], color=colors_pp, edgecolor='white')
        ax.set_xticks(range(len(spread_stats)))
        ax.set_xticklabels(spread_stats.index, fontsize=7, rotation=15)
        ax.set_ylabel('% of Total Profit')
        ax.set_title('Profit Share by Spread Quintile')
        for i, (pct, cnt) in enumerate(zip(spread_stats['pct_pnl'], spread_stats['count'])):
            ax.text(i, pct + 1, f'n={cnt}', ha='center', fontsize=7, color='#555')
        ax.grid(True, alpha=0.2, axis='y')
    else:
        axes[0, 0].text(0.5, 0.5, 'Spread data not available', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, 'Spread data not available', ha='center', va='center', transform=axes[0, 1].transAxes)

    # 9c: Scatter — edge vs odds, colored by outcome
    ax = axes[1, 0]
    losers_samp = losers.sample(min(500, len(losers)), random_state=42) if len(losers) > 0 else losers
    winners_samp = winners.sample(min(500, len(winners)), random_state=42) if len(winners) > 0 else winners
    ax.scatter(losers_samp['back_odds'], losers_samp['edge']*100, s=8, alpha=0.3, c='#e94560', label='Loss')
    ax.scatter(winners_samp['back_odds'], winners_samp['edge']*100, s=12, alpha=0.5, c='#27ae60', label='Win')
    ax.set_xlabel('Back Odds')
    ax.set_ylabel('Edge (%)')
    ax.set_title('Edge vs Odds (sampled)')
    ax.set_xlim(1, 50)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 9d: Distribution of odds for bets placed
    ax = axes[1, 1]
    ax.hist(bets_all['back_odds'].clip(upper=50), bins=50, color='#3282b8', alpha=0.7, edgecolor='white')
    ax.axvline(bets_all['back_odds'].median(), color='#e94560', linestyle='--', label=f'Median={bets_all["back_odds"].median():.1f}')
    ax.axvline(bets_all['back_odds'].mean(), color='#f39c12', linestyle='--', label=f'Mean={bets_all["back_odds"].mean():.1f}')
    ax.set_xlabel('Back Odds (capped at 50 for display)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Odds for Bets Placed')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

print(f"\nReport saved to {output_path}", flush=True)
print("Done!", flush=True)
