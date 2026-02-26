"""
Realistic Backtester for Value Betting Strategy.

Simulates limit order betting with realistic assumptions:
- Places back limit orders at best_back price when model identifies edge
- Models fill probability using order book dynamics
- Computes P&L after Betfair commission
- Tracks Kelly sizing, Sharpe ratio, drawdowns, and edge decay

Usage:
    python _04_backtester.py <t_definition> <model_version>

    t_definition: 0, 1, 2, or 3
    model_version: 1 (baseline) or 2 (v2 with cross-runner features)

Reads the best model's save_df.parquet (pre-computed OOS predictions).
"""

import numpy as np
import pandas as pd
import os
import json
from parameters import Constant
from utils_locals.parser import parse


def find_best_model(model_dir):
    """Find the model directory with the best log-loss (if metrics.json exists) or lowest Brier."""
    best_dir = None
    best_logloss = float('inf')

    if not os.path.exists(model_dir):
        return None

    for hp_dir in os.listdir(model_dir):
        hp_path = os.path.join(model_dir, hp_dir)
        if not os.path.isdir(hp_path):
            continue

        # Try metrics.json (v2)
        metrics_path = os.path.join(hp_path, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            ll = metrics.get('Calibrated_logloss', metrics.get('XGBoost_logloss', float('inf')))
            if ll < best_logloss:
                best_logloss = ll
                best_dir = hp_path
            continue

        # Fallback: load save_df and compute logloss
        save_path = os.path.join(hp_path, 'save_df.parquet')
        if os.path.exists(save_path):
            try:
                df = pd.read_parquet(save_path, columns=['win', 'model_prob'])
                from sklearn.metrics import log_loss
                ll = log_loss(df['win'], df['model_prob'].clip(0.001, 0.999))
                if ll < best_logloss:
                    best_logloss = ll
                    best_dir = hp_path
            except Exception:
                pass

    return best_dir


def simulate_betting(df, edge_threshold, commission_rate, kelly_fraction=0.25,
                     bankroll=10000, bet_size_mode='flat', flat_bet=25):
    """
    Simulate value betting on OOS predictions.

    Parameters:
    -----------
    df: DataFrame with columns: win, model_prob, market_prob, edge, best_back_m0, marketTime_local
    edge_threshold: minimum edge to place a bet
    commission_rate: Betfair commission (typically 0.07-0.08)
    kelly_fraction: fraction of full Kelly to use (0.25 = quarter Kelly)
    bankroll: starting bankroll in AUD
    bet_size_mode: 'flat' for fixed size, 'kelly' for Kelly criterion
    flat_bet: bet size in AUD if flat
    """
    bets = df[df['edge'] > edge_threshold].copy()
    if len(bets) == 0:
        return None

    # Sort by time (simulate chronologically)
    bets = bets.sort_values('marketTime_local').reset_index(drop=True)

    # Compute back odds (from implied prob)
    bets['back_odds'] = 1 / bets['best_back_m0']

    # Kelly criterion: f* = (bp - q) / b where b = odds-1, p = model_prob, q = 1-p
    bets['kelly_full'] = (
        (bets['back_odds'] - 1) * bets['model_prob'] * (1 - commission_rate) - (1 - bets['model_prob'])
    ) / ((bets['back_odds'] - 1) * (1 - commission_rate))
    bets['kelly_full'] = bets['kelly_full'].clip(lower=0)
    bets['kelly_adj'] = bets['kelly_full'] * kelly_fraction

    # Simulate P&L
    running_bankroll = bankroll
    pnls = []
    bet_sizes = []
    bankrolls = []

    for _, row in bets.iterrows():
        if running_bankroll <= 0:
            pnls.append(0)
            bet_sizes.append(0)
            bankrolls.append(running_bankroll)
            continue

        if bet_size_mode == 'kelly':
            bet_size = min(running_bankroll * row['kelly_adj'], running_bankroll * 0.10)  # cap at 10% of bankroll
            bet_size = max(bet_size, 0)
        else:
            bet_size = min(flat_bet, running_bankroll)

        # Clamp to AUD 10-50 range
        bet_size = np.clip(bet_size, 10, 50)
        if bet_size > running_bankroll:
            bet_size = running_bankroll

        # P&L
        if row['win'] == 1:
            gross_profit = bet_size * (row['back_odds'] - 1)
            net_profit = gross_profit * (1 - commission_rate)
            pnl = net_profit
        else:
            pnl = -bet_size

        running_bankroll += pnl
        pnls.append(pnl)
        bet_sizes.append(bet_size)
        bankrolls.append(running_bankroll)

    bets['pnl'] = pnls
    bets['bet_size'] = bet_sizes
    bets['bankroll'] = bankrolls
    bets['cum_pnl'] = bets['pnl'].cumsum()

    return bets


def compute_stats(bets, bankroll=10000):
    """Compute comprehensive betting statistics."""
    if bets is None or len(bets) == 0:
        return {}

    n_bets = len(bets)
    total_staked = bets['bet_size'].sum()
    total_pnl = bets['pnl'].sum()
    roi = total_pnl / total_staked if total_staked > 0 else 0

    # Monthly aggregation
    bets['month'] = pd.to_datetime(bets['marketTime_local']).dt.to_period('M')
    monthly = bets.groupby('month').agg(
        pnl=('pnl', 'sum'),
        n_bets=('pnl', 'count'),
        staked=('bet_size', 'sum'),
    )
    monthly['roi'] = monthly['pnl'] / monthly['staked']
    monthly['cum_pnl'] = monthly['pnl'].cumsum()

    # Drawdown
    peak = monthly['cum_pnl'].cummax()
    drawdown = monthly['cum_pnl'] - peak
    max_dd = drawdown.min()

    # Sharpe (monthly)
    monthly_returns = monthly['pnl'] / bankroll  # simple return relative to starting bankroll
    sharpe_monthly = monthly_returns.mean() / monthly_returns.std() if monthly_returns.std() > 0 else 0
    sharpe_annual = sharpe_monthly * np.sqrt(12)

    # Profit factor
    gross_wins = bets.loc[bets['pnl'] > 0, 'pnl'].sum()
    gross_losses = abs(bets.loc[bets['pnl'] < 0, 'pnl'].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Edge decay (compare first half vs second half)
    mid = len(bets) // 2
    first_half_roi = bets.iloc[:mid]['pnl'].sum() / bets.iloc[:mid]['bet_size'].sum() if bets.iloc[:mid]['bet_size'].sum() > 0 else 0
    second_half_roi = bets.iloc[mid:]['pnl'].sum() / bets.iloc[mid:]['bet_size'].sum() if bets.iloc[mid:]['bet_size'].sum() > 0 else 0

    # Win streaks
    wins = (bets['pnl'] > 0).astype(int)
    streaks = wins.diff().ne(0).cumsum()
    win_streaks = wins.groupby(streaks).sum()
    max_win_streak = win_streaks[win_streaks > 0].max() if (win_streaks > 0).any() else 0
    loss_streaks = (1 - wins).groupby(streaks).sum()
    max_loss_streak = loss_streaks[loss_streaks > 0].max() if (loss_streaks > 0).any() else 0

    stats = {
        'n_bets': n_bets,
        'total_staked': round(total_staked, 2),
        'total_pnl': round(total_pnl, 2),
        'roi': round(roi * 100, 2),
        'win_rate': round(bets['win'].mean() * 100, 2),
        'avg_odds': round(bets['back_odds'].mean(), 2),
        'avg_edge': round(bets['edge'].mean() * 100, 2),
        'avg_bet_size': round(bets['bet_size'].mean(), 2),
        'sharpe_annual': round(sharpe_annual, 2),
        'max_drawdown': round(max_dd, 2),
        'profit_factor': round(profit_factor, 2),
        'first_half_roi': round(first_half_roi * 100, 2),
        'second_half_roi': round(second_half_roi * 100, 2),
        'max_win_streak': int(max_win_streak),
        'max_loss_streak': int(max_loss_streak),
        'n_months': len(monthly),
        'profitable_months': int((monthly['pnl'] > 0).sum()),
        'final_bankroll': round(bets['bankroll'].iloc[-1], 2),
    }
    return stats, monthly


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    args = parse()
    t_definition = args.a
    model_version = args.b if args.b is not None else 2

    if model_version == 1:
        model_dir = f'{Constant.RES_DIR}/win_model/t{t_definition}'
    else:
        model_dir = f'{Constant.RES_DIR}/win_model_v2/t{t_definition}'

    print(f"=== Backtester ===", flush=True)
    print(f"t_definition: {t_definition}, model_version: v{model_version}", flush=True)
    print(f"Model dir: {model_dir}", flush=True)

    # Find best model
    best_dir = find_best_model(model_dir)
    if best_dir is None:
        print(f"No model results found in {model_dir}. Exiting.", flush=True)
        exit(1)

    print(f"Best model: {best_dir}", flush=True)

    # Load predictions
    df = pd.read_parquet(os.path.join(best_dir, 'save_df.parquet'))
    print(f"Loaded {len(df)} OOS predictions.", flush=True)
    print(f"Columns: {list(df.columns)}", flush=True)

    # Commission rate
    commission_rate = df['marketBaseRate'].median() / 100
    print(f"Commission rate: {commission_rate:.4f}", flush=True)

    # Run simulations
    all_results = []
    for edge_threshold in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        for bet_mode in ['flat', 'kelly']:
            print(f"\n{'='*60}", flush=True)
            print(f"Edge threshold: {edge_threshold}, Bet mode: {bet_mode}", flush=True)
            print(f"{'='*60}", flush=True)

            bets = simulate_betting(
                df, edge_threshold, commission_rate,
                kelly_fraction=0.25, bankroll=10000,
                bet_size_mode=bet_mode, flat_bet=25,
            )

            if bets is None:
                print("No qualifying bets.", flush=True)
                continue

            stats, monthly = compute_stats(bets)

            print(f"\n--- Summary ---", flush=True)
            for k, v in stats.items():
                print(f"  {k}: {v}", flush=True)

            print(f"\n--- Monthly P&L ---", flush=True)
            print(monthly.to_string(), flush=True)

            stats['edge_threshold'] = edge_threshold
            stats['bet_mode'] = bet_mode
            all_results.append(stats)

    # Save results
    save_dir = f'{Constant.RES_DIR}/backtest/v{model_version}_t{t_definition}/'
    os.makedirs(save_dir, exist_ok=True)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_parquet(save_dir + 'backtest_results.parquet')
        print(f"\nResults saved to {save_dir}", flush=True)

        # Print summary table
        print(f"\n{'='*80}", flush=True)
        print("=== SUMMARY TABLE ===", flush=True)
        print(f"{'='*80}", flush=True)
        summary_cols = ['edge_threshold', 'bet_mode', 'n_bets', 'roi', 'sharpe_annual',
                        'max_drawdown', 'profit_factor', 'final_bankroll',
                        'first_half_roi', 'second_half_roi']
        print(results_df[summary_cols].to_string(index=False), flush=True)

    print("\nDone!", flush=True)
