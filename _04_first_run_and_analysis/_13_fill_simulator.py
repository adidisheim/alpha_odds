"""
Fill Rate Simulator — Replay tick-by-tick order book data to simulate limit order fills.

Usage: python3 _13_fill_simulator.py <task_id>

Splits 305 OOS day-files into 30 chunks. For each runner with positive edge in the
ensemble predictions, replays tick data after the model's decision point to check
if a back limit order would have filled.

Outputs: res/fill_simulation/fill_part_{task_id}.parquet
"""

import pandas as pd
import numpy as np
import os
from parameters import Constant
from utils_locals.parser import parse

# ── Betfair tick table ──
BETFAIR_TICKS = [
    (1.01, 2.0, 0.01),
    (2.0, 3.0, 0.02),
    (3.0, 4.0, 0.05),
    (4.0, 6.0, 0.10),
    (6.0, 10.0, 0.20),
    (10.0, 20.0, 0.50),
    (20.0, 30.0, 1.0),
    (30.0, 50.0, 2.0),
    (50.0, 100.0, 5.0),
    (100.0, 1000.0, 10.0),
]


def get_tick_increment(price):
    """Return the Betfair tick increment for a given price level."""
    for low, high, inc in BETFAIR_TICKS:
        if low <= price < high:
            return inc
    return 10.0


def adjust_ticks(price, n_ticks):
    """Move price by n_ticks on the Betfair tick ladder (positive = increase odds)."""
    for _ in range(abs(n_ticks)):
        inc = get_tick_increment(price)
        if n_ticks > 0:
            price = round(price + inc, 2)
        else:
            price = round(price - inc, 2)
    return max(1.01, price)


# ── Order placement windows (time_delta before market close) ──
T0_WINDOWS = {
    't0_60s': pd.Timedelta(seconds=60),   # t_definition=0 decision point
    't0_20s': pd.Timedelta(seconds=20),   # t_definition=3 decision point
}


def simulate_fill(runner_ticks, limit_price, t0_delta):
    """
    Simulate whether a back limit order at limit_price would fill.

    runner_ticks: DataFrame with time_delta, best_back, best_lay, prc, qty columns.
                  Sorted by time_delta descending (earliest wall-clock first).
    limit_price: odds at which we place our back order
    t0_delta: Timedelta for the decision point (time before close)

    Returns dict with fill simulation results.
    """
    # Ticks after our order placement (time_delta < t0_delta = closer to close)
    post_order = runner_ticks[runner_ticks['time_delta'] < t0_delta]

    if len(post_order) == 0:
        return {
            'conservative_fill': False,
            'moderate_fill': False,
            'fill_price': np.nan,
            'time_to_fill_s': np.nan,
            'volume_at_price': 0.0,
            'spread_at_order': np.nan,
            'n_ticks_post': 0,
        }

    # Conservative: a trade occurred at prc >= limit_price (someone matched at our odds or higher)
    has_trade = post_order['prc'] > 0
    trades_at_price = post_order[has_trade & (post_order['prc'] >= limit_price)]
    conservative_fill = len(trades_at_price) > 0

    # Moderate: conservative OR best_lay dropped to <= limit_price (spread crossed our price)
    lay_cross = post_order[(post_order['best_lay'] <= limit_price) & (post_order['best_lay'] > 1.0)]
    moderate_fill = conservative_fill or (len(lay_cross) > 0)

    # Fill details
    fill_price = np.nan
    time_to_fill_s = np.nan

    if conservative_fill:
        # First fill tick (largest time_delta in post_order that qualifies = earliest after t0)
        first_fill = trades_at_price.iloc[0]
        fill_price = first_fill['prc']
        time_to_fill_s = (t0_delta - first_fill['time_delta']).total_seconds()
    elif moderate_fill:
        first_cross = lay_cross.iloc[0]
        fill_price = limit_price  # assume fill at our limit when spread crosses
        time_to_fill_s = (t0_delta - first_cross['time_delta']).total_seconds()

    # Volume traded at our exact price after t0
    vol_at_price = post_order.loc[has_trade & (post_order['prc'] == limit_price), 'qty'].sum()

    # Spread at order placement time
    at_or_before_t0 = runner_ticks[runner_ticks['time_delta'] >= t0_delta]
    if len(at_or_before_t0) > 0:
        state = at_or_before_t0.iloc[-1]  # most recent tick at or before t0
        spread = state['best_lay'] - state['best_back']
    else:
        spread = np.nan

    return {
        'conservative_fill': conservative_fill,
        'moderate_fill': moderate_fill,
        'fill_price': fill_price,
        'time_to_fill_s': time_to_fill_s,
        'volume_at_price': vol_at_price,
        'spread_at_order': spread,
        'n_ticks_post': len(post_order),
    }


if __name__ == '__main__':
    args = parse()
    task_id = args.a
    N_TASKS = 30

    print(f"=== Fill Rate Simulator: Task {task_id}/{N_TASKS-1} ===", flush=True)

    # ── Load ensemble predictions ──
    ens_path = f'{Constant.RES_DIR}/analysis/ultimate_cross_t_ensemble_predictions.parquet'
    ens = pd.read_parquet(ens_path)
    print(f"Loaded {len(ens):,} ensemble predictions", flush=True)

    # Parse keys into file_name and runner_id
    key_parts = ens['key'].str.rsplit('_', n=1, expand=True)
    ens['file_name'] = key_parts[0]
    ens['runner_id'] = key_parts[1].astype(float).astype(int)

    # Filter to positive edge (only runners we'd actually bet on)
    ens = ens[ens['edge'] > 0].copy()
    print(f"After edge>0 filter: {len(ens):,} runners", flush=True)

    # Build lookup: file_name -> set of runner_ids
    bet_lookup = ens.groupby('file_name')['runner_id'].apply(set).to_dict()
    bet_file_names = set(bet_lookup.keys())

    # Build fast key->row lookup
    ens_indexed = ens.set_index('key')

    # ── List 2025 day-files and split into chunks ──
    data_dir = f'{Constant.DATA_DIR}/p/greyhound_au/'
    all_files = sorted([f for f in os.listdir(data_dir) if f.startswith('win_2025')])
    print(f"Total 2025 day-files: {len(all_files)}", flush=True)

    chunks = np.array_split(all_files, N_TASKS)
    my_files = chunks[task_id]
    print(f"This task: {len(my_files)} files ({my_files[0]} .. {my_files[-1]})", flush=True)

    # ── Process each day-file ──
    results = []

    for fi, day_file in enumerate(my_files):
        path = os.path.join(data_dir, day_file)
        tick_df = pd.read_parquet(path)
        tick_df = tick_df.reset_index()
        if tick_df.columns[0] == 'index':
            tick_df = tick_df.rename(columns={'index': 'time'})

        # Find overlapping file_names between tick data and our bets
        tick_file_names = set(tick_df['file_name'].unique())
        relevant = tick_file_names & bet_file_names

        if not relevant:
            if (fi + 1) % 5 == 0:
                print(f"  [{fi+1}/{len(my_files)}] {day_file}: 0 relevant markets", flush=True)
            continue

        # Filter to relevant markets and compute time_delta
        tick_df = tick_df[tick_df['file_name'].isin(relevant)].copy()
        tick_df['time_delta'] = tick_df.groupby('file_name')['time'].transform('max') - tick_df['time']

        n_bets_this_file = 0

        for market_fn in relevant:
            market_ticks = tick_df[tick_df['file_name'] == market_fn]
            runner_ids = bet_lookup[market_fn]

            for rid in runner_ids:
                runner_ticks = market_ticks[market_ticks['id'] == rid].sort_values(
                    'time_delta', ascending=False
                )

                if len(runner_ticks) == 0:
                    continue

                # Look up bet info
                bet_key = f"{market_fn}_{float(rid)}"
                if bet_key not in ens_indexed.index:
                    continue
                bet_row = ens_indexed.loc[bet_key]

                for window_name, t0_delta in T0_WINDOWS.items():
                    # Find order book state at t0 via forward-fill
                    at_or_before_t0 = runner_ticks[runner_ticks['time_delta'] >= t0_delta]
                    if len(at_or_before_t0) == 0:
                        continue

                    # Most recent tick at or before t0 (smallest time_delta >= t0)
                    state_at_t0 = at_or_before_t0.iloc[-1]
                    base_price = state_at_t0['best_back']

                    if pd.isna(base_price) or base_price <= 1.0 or base_price >= 1000:
                        continue

                    # Simulate 3 price variants
                    for price_variant, price_adj in [('best_back', 0), ('minus_1_tick', -1), ('plus_1_tick', 1)]:
                        adj_price = adjust_ticks(base_price, price_adj) if price_adj != 0 else base_price
                        fill_result = simulate_fill(runner_ticks, adj_price, t0_delta)

                        results.append({
                            'key': bet_key,
                            'file_name': market_fn,
                            'runner_id': rid,
                            'win': int(bet_row['win']),
                            'model_prob': bet_row['model_prob'],
                            'market_prob': bet_row['market_prob'],
                            'edge': bet_row['edge'],
                            'back_odds': bet_row['back_odds'],
                            'window': window_name,
                            'price_variant': price_variant,
                            'limit_price': adj_price,
                            'tick_best_back_at_t0': base_price,
                            'tick_best_lay_at_t0': state_at_t0['best_lay'],
                            **fill_result,
                        })
                        n_bets_this_file += 1

        if (fi + 1) % 5 == 0 or fi == len(my_files) - 1:
            print(f"  [{fi+1}/{len(my_files)}] {day_file}: {len(relevant)} markets, "
                  f"{n_bets_this_file} scenarios, {len(results):,} total", flush=True)

    # ── Save results ──
    save_dir = f'{Constant.RES_DIR}/fill_simulation/'
    os.makedirs(save_dir, exist_ok=True)

    if results:
        df_out = pd.DataFrame(results)
        save_path = f'{save_dir}/fill_part_{task_id}.parquet'
        df_out.to_parquet(save_path)
        print(f"\nSaved {len(df_out):,} results to {save_path}", flush=True)
    else:
        print(f"\nNo results for task {task_id}", flush=True)

    print("Done!", flush=True)
