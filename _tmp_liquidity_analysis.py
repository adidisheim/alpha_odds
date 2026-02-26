"""
Comprehensive OOS analysis: Are profitable bets concentrated on illiquid longshots?
Breaks down by odds bucket, implied probability, liquidity proxy, profit distribution.
"""
import pandas as pd
import numpy as np
import os
from math import erf, sqrt

os.chdir("/data/projects/punim2039/alpha_odds/res")

def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

# ============================================================
# STEP 1: Load or rebuild the cross-t super-ensemble
# ============================================================
saved_path = "analysis/ultimate_cross_t_ensemble_predictions.parquet"
need_rebuild = True

if os.path.exists(saved_path):
    bt_saved = pd.read_parquet(saved_path)
    print(f"Found saved predictions: {len(bt_saved):,} rows")
    print(f"Columns: {list(bt_saved.columns)}")
    # Check if it has what we need
    if all(c in bt_saved.columns for c in ['win', 'model_prob', 'market_prob', 'edge', 'back_odds']):
        need_rebuild = False
        bt = bt_saved.copy()
        print("Using saved predictions.")
    else:
        print("Saved file missing columns, rebuilding...")

if need_rebuild:
    print("Rebuilding cross-t super-ensemble from scratch...")

    def load_ensemble(model_dir, t_def, top_n):
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

    all_components = {}
    for t in [0, 1, 2, 3]:
        base, prob = load_ensemble("win_model", t, top_n=7)
        if base is not None:
            all_components[("V1", t)] = (base, prob)
            print(f"  V1 t{t}: {len(base):,} rows")
        base, prob = load_ensemble("win_model_v2", t, top_n=15)
        if base is not None:
            all_components[("V2", t)] = (base, prob)
            print(f"  V2 t{t}: {len(base):,} rows")

    base_df = all_components[("V1", 0)][0].copy()
    base_df["key"] = base_df["file_name"].astype(str) + "_" + base_df["id"].astype(str)
    aligned = base_df[["key", "win", "market_prob", "orig_best_back_m0", "orig_best_lay_m0", "marketTime_local"]].copy()
    aligned["V1_t0"] = all_components[("V1", 0)][1]

    for (ver, t), (df, prob) in all_components.items():
        col = f"{ver}_t{t}"
        if col == "V1_t0":
            continue
        tmp = df.copy()
        tmp["key"] = tmp["file_name"].astype(str) + "_" + tmp["id"].astype(str)
        tmp[col] = prob
        aligned = aligned.merge(tmp[["key", col]], on="key", how="inner")

    v1_cols = [c for c in aligned.columns if c.startswith("V1_")]
    v2_cols = [c for c in aligned.columns if c.startswith("V2_")]
    v1_cross_t = aligned[v1_cols].mean(axis=1).values
    v2_cross_t = aligned[v2_cols].mean(axis=1).values

    # Use 20/80 V1/V2 as per memory (best_w=0.80)
    best_probs = v1_cross_t * 0.20 + v2_cross_t * 0.80

    bt = aligned.copy()
    bt["model_prob"] = best_probs
    bt["edge"] = bt["model_prob"] - bt["market_prob"]
    bt["back_odds"] = 1 / bt["orig_best_back_m0"]
    bt = bt[(bt["back_odds"] > 1.01) & (bt["back_odds"] < 1000)]
    print(f"Built ensemble: {len(bt):,} rows")

# ============================================================
# STEP 2: Ensure all needed columns exist
# ============================================================
# We need: back_odds, market_prob, model_prob, edge, win, back-lay spread
# If back_odds not in saved, compute from market_prob
if 'back_odds' not in bt.columns and 'market_prob' in bt.columns:
    bt['back_odds'] = 1.0 / bt['market_prob']

# We also want spread info. Load from win_model base if available.
if 'orig_best_lay_m0' not in bt.columns:
    # Try to get it from the V1 t0 base
    try:
        def load_ensemble_base(model_dir, t_def, top_n):
            base_dir = f"{model_dir}/t{t_def}"
            if not os.path.exists(base_dir):
                return None
            dfs = {}
            for c in sorted(os.listdir(base_dir)):
                path = os.path.join(base_dir, c, "save_df.parquet")
                if os.path.exists(path):
                    try:
                        dfs[c] = pd.read_parquet(path)
                    except Exception:
                        pass
            if not dfs:
                return None
            sorted_configs = sorted(dfs.keys(), key=lambda c: log_loss(
                dfs[c]["win"].values, dfs[c]["model_prob"].clip(0.001, 0.999).values))
            return dfs[sorted_configs[0]].copy()

        base_for_spread = load_ensemble_base("win_model", 0, 7)
        if base_for_spread is not None:
            base_for_spread["key"] = base_for_spread["file_name"].astype(str) + "_" + base_for_spread["id"].astype(str)
            spread_cols = [c for c in base_for_spread.columns if 'lay' in c.lower() or 'back' in c.lower() or 'q_' in c.lower()]
            merge_cols = ["key"] + [c for c in spread_cols if c in base_for_spread.columns]
            bt = bt.merge(base_for_spread[merge_cols].drop_duplicates("key"), on="key", how="left")
            print(f"Merged spread/liquidity columns: {spread_cols}")
    except Exception as e:
        print(f"Could not load spread data: {e}")

commission = 0.075

# Compute PnL for ALL rows (not just edge > 3%)
bt["pnl"] = bt["win"] * (bt["back_odds"] - 1) * (1 - commission) - (1 - bt["win"])

# Filter to edge > 3% for the main analysis
bets = bt[bt["edge"] > 0.03].copy()
bets["pnl"] = bets["win"] * (bets["back_odds"] - 1) * (1 - commission) - (1 - bets["win"])

print(f"\n{'#'*80}")
print(f"# OOS LIQUIDITY & ODDS CONCENTRATION ANALYSIS")
print(f"# Cross-t Super-Ensemble | Edge > 3% | Commission {commission:.1%}")
print(f"# Total universe: {len(bt):,} | Bets (edge>3%): {len(bets):,}")
print(f"{'#'*80}")

# ============================================================
# A) BY ODDS BUCKET
# ============================================================
print(f"\n{'='*80}")
print("SECTION A: BREAKDOWN BY ODDS BUCKET")
print(f"{'='*80}")

odds_bins = [1, 2, 3, 5, 8, 15, 30, 100, 1001]
odds_labels = ['[1-2]', '[2-3]', '[3-5]', '[5-8]', '[8-15]', '[15-30]', '[30-100]', '[100+]']
bets['odds_bucket'] = pd.cut(bets['back_odds'], bins=odds_bins, labels=odds_labels, right=False)

total_pnl = bets['pnl'].sum()
print(f"\nTotal PnL (unit stakes): {total_pnl:+.2f}")
print(f"Total PnL ($25/bet): ${total_pnl * 25:+,.0f}")
print(f"Overall ROI: {bets['pnl'].mean() * 100:+.1f}%")
print(f"Overall win rate: {bets['win'].mean():.1%}")

header = f"{'Bucket':>10} | {'Count':>6} | {'%Bets':>5} | {'WR':>6} | {'Avg Edge':>8} | {'ROI':>7} | {'Tot PnL':>8} | {'%PnL':>6} | {'Avg Odds':>9} | {'Med Odds':>9}"
print(f"\n{header}")
print("-" * len(header))

for bucket in odds_labels:
    sub = bets[bets['odds_bucket'] == bucket]
    if len(sub) == 0:
        print(f"{bucket:>10} | {'0':>6} | {'0.0%':>5} |")
        continue
    n = len(sub)
    pct_bets = n / len(bets) * 100
    wr = sub['win'].mean()
    avg_edge = sub['edge'].mean()
    roi = sub['pnl'].mean() * 100
    tot_pnl_bucket = sub['pnl'].sum()
    pct_pnl = tot_pnl_bucket / total_pnl * 100 if total_pnl != 0 else 0
    avg_odds = sub['back_odds'].mean()
    med_odds = sub['back_odds'].median()
    print(f"{bucket:>10} | {n:>6,} | {pct_bets:>4.1f}% | {wr:>5.1%} | {avg_edge:>7.3f} | {roi:>+6.1f}% | {tot_pnl_bucket:>+8.1f} | {pct_pnl:>+5.1f}% | {avg_odds:>9.1f} | {med_odds:>9.1f}")

# ============================================================
# B) BY IMPLIED PROBABILITY DECILE
# ============================================================
print(f"\n{'='*80}")
print("SECTION B: BREAKDOWN BY IMPLIED PROBABILITY (MARKET_PROB) DECILE")
print(f"{'='*80}")

bets['mkt_prob_decile'] = pd.qcut(bets['market_prob'], 10, labels=False, duplicates='drop')

header2 = f"{'Decile':>7} | {'MktProb Range':>18} | {'Count':>6} | {'WR':>6} | {'Avg Edge':>8} | {'ROI':>7} | {'Tot PnL':>8} | {'%PnL':>6}"
print(f"\n{header2}")
print("-" * len(header2))

for d in sorted(bets['mkt_prob_decile'].dropna().unique()):
    sub = bets[bets['mkt_prob_decile'] == d]
    if len(sub) == 0:
        continue
    lo = sub['market_prob'].min()
    hi = sub['market_prob'].max()
    n = len(sub)
    wr = sub['win'].mean()
    avg_edge = sub['edge'].mean()
    roi = sub['pnl'].mean() * 100
    tot_pnl_d = sub['pnl'].sum()
    pct_pnl = tot_pnl_d / total_pnl * 100 if total_pnl != 0 else 0
    print(f"     {int(d):>2} | [{lo:.3f}, {hi:.3f}] | {n:>6,} | {wr:>5.1%} | {avg_edge:>7.3f} | {roi:>+6.1f}% | {tot_pnl_d:>+8.1f} | {pct_pnl:>+5.1f}%")

# ============================================================
# C) BY LIQUIDITY PROXY (SPREAD)
# ============================================================
print(f"\n{'='*80}")
print("SECTION C: BREAKDOWN BY LIQUIDITY PROXY")
print(f"{'='*80}")

# Compute spread = best_lay_m0 - best_back_m0 (in implied prob space)
# orig_best_back_m0 is implied prob (1/odds), orig_best_lay_m0 is implied prob (1/odds)
# Higher implied prob spread = tighter market (because lay price < back price in decimal odds)
# Actually in implied prob: lay_prob > back_prob means wider spread
# Let's compute in odds space: lay_odds - back_odds (lay > back)
# Or better: relative spread in probability space

spread_col = None
for c in ['orig_best_lay_m0', 'best_lay_m0']:
    if c in bt.columns:
        spread_col = c
        break

back_col = None
for c in ['orig_best_back_m0', 'best_back_m0']:
    if c in bt.columns:
        back_col = c
        break

if spread_col and back_col:
    # In implied prob space: lay_prob - back_prob (lay_prob > back_prob means wide spread)
    # Actually: best_back_m0 is 1/back_odds, best_lay_m0 is 1/lay_odds
    # lay_odds < back_odds, so 1/lay_odds > 1/back_odds
    # spread in prob space = lay_implied - back_implied
    bets['spread_prob'] = bets[spread_col] - bets[back_col] if spread_col in bets.columns else np.nan

    if 'spread_prob' not in bets.columns or bets['spread_prob'].isna().all():
        # Try from bt columns
        if spread_col in bt.columns and back_col in bt.columns:
            bt['spread_prob'] = bt[spread_col] - bt[back_col]
            bets = bt[bt["edge"] > 0.03].copy()
            bets["pnl"] = bets["win"] * (bets["back_odds"] - 1) * (1 - commission) - (1 - bets["win"])
            bets['odds_bucket'] = pd.cut(bets['back_odds'], bins=odds_bins, labels=odds_labels, right=False)

    if 'spread_prob' in bets.columns and not bets['spread_prob'].isna().all():
        # Also compute spread in odds terms
        bets['lay_odds'] = 1.0 / bets[spread_col].clip(lower=0.001)
        bets['spread_odds_pct'] = (bets['lay_odds'] - bets['back_odds']) / bets['back_odds'] * 100

        print(f"\nSpread computed as: lay_implied_prob - back_implied_prob")
        print(f"  Mean spread (prob): {bets['spread_prob'].mean():.4f}")
        print(f"  Median spread (prob): {bets['spread_prob'].median():.4f}")
        print(f"  Mean spread (% of odds): {bets['spread_odds_pct'].mean():.1f}%")

        # Spread quintiles
        bets['spread_quintile'] = pd.qcut(bets['spread_prob'], 5, labels=False, duplicates='drop')

        header3 = f"{'Quintile':>9} | {'Spread Range':>20} | {'Count':>6} | {'WR':>6} | {'Avg Edge':>8} | {'ROI':>7} | {'Tot PnL':>8} | {'%PnL':>6} | {'Avg Odds':>9}"
        print(f"\n{header3}")
        print("-" * len(header3))

        for q in sorted(bets['spread_quintile'].dropna().unique()):
            sub = bets[bets['spread_quintile'] == q]
            lo = sub['spread_prob'].min()
            hi = sub['spread_prob'].max()
            n = len(sub)
            wr = sub['win'].mean()
            avg_edge = sub['edge'].mean()
            roi = sub['pnl'].mean() * 100
            tot_pnl_q = sub['pnl'].sum()
            pct_pnl = tot_pnl_q / total_pnl * 100 if total_pnl != 0 else 0
            avg_odds = sub['back_odds'].mean()
            label = "tight" if q == 0 else ("wide" if q == max(bets['spread_quintile'].dropna().unique()) else "")
            print(f"  Q{int(q)} {label:>4} | [{lo:.4f}, {hi:.4f}] | {n:>6,} | {wr:>5.1%} | {avg_edge:>7.3f} | {roi:>+6.1f}% | {tot_pnl_q:>+8.1f} | {pct_pnl:>+5.1f}% | {avg_odds:>9.1f}")
    else:
        print("No spread data available for liquidity analysis.")
else:
    print(f"No spread columns found. Available: {[c for c in bt.columns if 'back' in c.lower() or 'lay' in c.lower()]}")

# Check for q_100 style columns
q_cols = [c for c in bt.columns if 'q_100' in c.lower() or 'q_200' in c.lower()]
if q_cols:
    print(f"\nQuantity columns available: {q_cols}")
    # Use q_100 execution price gap as another liquidity proxy
    for qc in ['best_back_q_100_m0', 'orig_best_back_q_100_m0']:
        if qc in bets.columns:
            back_q100_col = qc
            lay_q100_col = qc.replace('back', 'lay')
            if lay_q100_col in bets.columns:
                bets['spread_q100'] = bets[lay_q100_col] - bets[back_q100_col]
                print(f"\nQ100 spread (execution price for $100 order):")
                print(f"  Mean: {bets['spread_q100'].mean():.4f}")
                print(f"  Median: {bets['spread_q100'].median():.4f}")

                bets['q100_quintile'] = pd.qcut(bets['spread_q100'], 5, labels=False, duplicates='drop')

                print(f"\n  Q100 Spread Quintile Breakdown:")
                for q in sorted(bets['q100_quintile'].dropna().unique()):
                    sub = bets[bets['q100_quintile'] == q]
                    n = len(sub)
                    roi = sub['pnl'].mean() * 100
                    tot_pnl_q = sub['pnl'].sum()
                    pct_pnl = tot_pnl_q / total_pnl * 100 if total_pnl != 0 else 0
                    avg_odds = sub['back_odds'].mean()
                    print(f"    Q{int(q)}: n={n:,}, ROI={roi:+.1f}%, PnL={tot_pnl_q:+.1f} ({pct_pnl:+.1f}%), AvgOdds={avg_odds:.1f}")
            break

# ============================================================
# D) PROFIT DISTRIBUTION STATS
# ============================================================
print(f"\n{'='*80}")
print("SECTION D: PROFIT DISTRIBUTION & CONCENTRATION")
print(f"{'='*80}")

total_profit = bets['pnl'].sum()
total_profit_25 = total_profit * 25

# Longshot concentration
for threshold in [10, 20, 30, 50, 100]:
    longshots = bets[bets['back_odds'] > threshold]
    if len(longshots) > 0:
        ls_pnl = longshots['pnl'].sum()
        ls_pct = ls_pnl / total_profit * 100 if total_profit != 0 else 0
        ls_wr = longshots['win'].mean()
        ls_n = len(longshots)
        ls_wins = int(longshots['win'].sum())
        print(f"\nBets with odds > {threshold}/1:")
        print(f"  Count: {ls_n:,} ({ls_n/len(bets)*100:.1f}% of all bets)")
        print(f"  Wins: {ls_wins}")
        print(f"  Win Rate: {ls_wr:.2%}")
        print(f"  Total PnL: {ls_pnl:+.2f} units ({ls_pct:+.1f}% of total)")
        print(f"  ROI: {longshots['pnl'].mean()*100:+.1f}%")

# Excluding longshots
print(f"\n--- EXCLUDING LONGSHOTS ---")
for threshold in [20, 30, 50]:
    non_ls = bets[bets['back_odds'] <= threshold]
    if len(non_ls) > 0:
        pnl_nls = non_ls['pnl'].sum()
        roi_nls = non_ls['pnl'].mean() * 100
        z_nls = non_ls['pnl'].mean() / non_ls['pnl'].std() * np.sqrt(len(non_ls)) if non_ls['pnl'].std() > 0 else 0
        p_nls = 1 - norm_cdf(z_nls)
        wr_nls = non_ls['win'].mean()
        print(f"\n  Odds <= {threshold}: {len(non_ls):,} bets, WR={wr_nls:.1%}, ROI={roi_nls:+.1f}%, PnL={pnl_nls:+.1f}, z={z_nls:.2f}, p={p_nls:.4f}")

# Median odds comparison
winners = bets[bets['win'] == 1]
losers = bets[bets['win'] == 0]
print(f"\n--- ODDS COMPARISON: WINNERS vs LOSERS ---")
print(f"  Winners ({len(winners):,}):  Mean odds={winners['back_odds'].mean():.1f}, Median={winners['back_odds'].median():.1f}, IQR=[{winners['back_odds'].quantile(0.25):.1f}, {winners['back_odds'].quantile(0.75):.1f}]")
print(f"  Losers  ({len(losers):,}):  Mean odds={losers['back_odds'].mean():.1f}, Median={losers['back_odds'].median():.1f}, IQR=[{losers['back_odds'].quantile(0.25):.1f}, {losers['back_odds'].quantile(0.75):.1f}]")

# Biggest single bet contribution
bets_sorted = bets.sort_values('pnl', ascending=False)
biggest_win = bets_sorted.iloc[0]
biggest_loss = bets_sorted.iloc[-1]
print(f"\n--- SINGLE BET EXTREMES ---")
print(f"  Biggest win:  odds={biggest_win['back_odds']:.1f}, PnL={biggest_win['pnl']:+.2f} units ({biggest_win['pnl']/total_profit*100:.1f}% of total)")
print(f"  Biggest loss: odds={biggest_loss['back_odds']:.1f}, PnL={biggest_loss['pnl']:+.2f} units ({biggest_loss['pnl']/total_profit*100:.1f}% of total)")

# Top-5 wins contribution
top5 = bets_sorted.head(5)
top5_pnl = top5['pnl'].sum()
print(f"\n  Top 5 winning bets: PnL={top5_pnl:+.2f} ({top5_pnl/total_profit*100:.1f}% of total)")
top10 = bets_sorted.head(10)
top10_pnl = top10['pnl'].sum()
print(f"  Top 10 winning bets: PnL={top10_pnl:+.2f} ({top10_pnl/total_profit*100:.1f}% of total)")

# Cumulative PnL contribution by odds
print(f"\n--- CUMULATIVE PROFIT CONTRIBUTION ---")
bets_sorted_odds = bets.sort_values('back_odds')
bets_sorted_odds['cum_pnl'] = bets_sorted_odds['pnl'].cumsum()
n_total = len(bets_sorted_odds)
for pct in [10, 25, 50, 75, 90]:
    idx = int(n_total * pct / 100)
    cum = bets_sorted_odds.iloc[:idx]['pnl'].sum()
    max_odds_at_pct = bets_sorted_odds.iloc[idx-1]['back_odds']
    print(f"  Lowest {pct}% of odds (odds <= {max_odds_at_pct:.1f}): {cum:+.1f} PnL ({cum/total_profit*100:+.1f}% of total)")

# ============================================================
# E) TOP 20 MOST PROFITABLE INDIVIDUAL BETS
# ============================================================
print(f"\n{'='*80}")
print("SECTION E: TOP 20 MOST PROFITABLE INDIVIDUAL BETS")
print(f"{'='*80}")

top20 = bets.sort_values('pnl', ascending=False).head(20)
header5 = f"{'#':>3} | {'Odds':>8} | {'Model P':>7} | {'Mkt P':>7} | {'Edge':>7} | {'Win':>4} | {'PnL':>8} | {'%Tot':>6}"
if 'marketTime_local' in top20.columns:
    header5 += f" | {'Date':>12}"
print(f"\n{header5}")
print("-" * len(header5))

for i, (_, row) in enumerate(top20.iterrows(), 1):
    odds = row['back_odds']
    mp = row['model_prob']
    mktp = row['market_prob']
    edge_v = row['edge']
    w = int(row['win'])
    pnl_v = row['pnl']
    pct_tot = pnl_v / total_profit * 100 if total_profit != 0 else 0
    line = f"{i:>3} | {odds:>8.1f} | {mp:>6.3f} | {mktp:>6.3f} | {edge_v:>6.3f} | {w:>4} | {pnl_v:>+8.2f} | {pct_tot:>+5.1f}%"
    if 'marketTime_local' in top20.columns:
        try:
            date_str = str(row['marketTime_local'])[:10]
        except:
            date_str = "N/A"
        line += f" | {date_str:>12}"
    print(line)

# ============================================================
# F) BOTTOM 20 BIGGEST LOSING BETS
# ============================================================
print(f"\n{'='*80}")
print("SECTION F: TOP 20 BIGGEST LOSING BETS")
print(f"{'='*80}")

bottom20 = bets.sort_values('pnl', ascending=True).head(20)
header6 = f"{'#':>3} | {'Odds':>8} | {'Model P':>7} | {'Mkt P':>7} | {'Edge':>7} | {'Win':>4} | {'PnL':>8}"
print(f"\n{header6}")
print("-" * len(header6))

for i, (_, row) in enumerate(bottom20.iterrows(), 1):
    line = f"{i:>3} | {row['back_odds']:>8.1f} | {row['model_prob']:>6.3f} | {row['market_prob']:>6.3f} | {row['edge']:>6.3f} | {int(row['win']):>4} | {row['pnl']:>+8.2f}"
    print(line)

# ============================================================
# G) ROBUSTNESS: What if we cap max odds?
# ============================================================
print(f"\n{'='*80}")
print("SECTION G: ROBUSTNESS - PERFORMANCE WITH ODDS CAPS")
print(f"{'='*80}")

print(f"\n{'Max Odds Cap':>13} | {'Bets':>6} | {'WR':>6} | {'ROI':>7} | {'PnL':>8} | {'z-stat':>7} | {'p-val':>8} | {'Sharpe_a':>8}")
print("-" * 90)

for cap in [5, 8, 10, 15, 20, 30, 50, 100, 500, 1000]:
    sub = bets[bets['back_odds'] <= cap]
    if len(sub) < 10:
        continue
    n = len(sub)
    wr = sub['win'].mean()
    roi = sub['pnl'].mean() * 100
    pnl_total = sub['pnl'].sum()
    z = sub['pnl'].mean() / sub['pnl'].std() * np.sqrt(n) if sub['pnl'].std() > 0 else 0
    p = 1 - norm_cdf(z)

    # Monthly Sharpe
    if 'marketTime_local' in sub.columns:
        try:
            sub_copy = sub.copy()
            sub_copy['month'] = pd.to_datetime(sub_copy['marketTime_local']).dt.to_period("M")
            monthly = sub_copy.groupby('month')['pnl'].sum()
            sh_m = monthly.mean() / monthly.std() if monthly.std() > 0 else 0
            sh_a = sh_m * np.sqrt(12)
        except:
            sh_a = 0
    else:
        sh_a = 0

    print(f"{cap:>13} | {n:>6,} | {wr:>5.1%} | {roi:>+6.1f}% | {pnl_total:>+8.1f} | {z:>+7.2f} | {p:>8.5f} | {sh_a:>8.1f}")

# ============================================================
# H) MONTHLY BREAKDOWN BY ODDS RANGE
# ============================================================
if 'marketTime_local' in bets.columns:
    print(f"\n{'='*80}")
    print("SECTION H: MONTHLY PnL - SHORT ODDS (<15) vs LONGSHOTS (>=15)")
    print(f"{'='*80}")

    bets_copy = bets.copy()
    bets_copy['month'] = pd.to_datetime(bets_copy['marketTime_local']).dt.to_period("M")

    short = bets_copy[bets_copy['back_odds'] < 15]
    long = bets_copy[bets_copy['back_odds'] >= 15]

    months = sorted(bets_copy['month'].unique())

    print(f"\n{'Month':>10} | {'Short<15 n':>10} | {'Short PnL':>10} | {'Long>=15 n':>10} | {'Long PnL':>10} | {'Total PnL':>10}")
    print("-" * 75)

    for m in months:
        s = short[short['month'] == m]
        l = long[long['month'] == m]
        s_pnl = s['pnl'].sum() if len(s) > 0 else 0
        l_pnl = l['pnl'].sum() if len(l) > 0 else 0
        t_pnl = s_pnl + l_pnl
        print(f"{str(m):>10} | {len(s):>10,} | {s_pnl:>+10.1f} | {len(l):>10,} | {l_pnl:>+10.1f} | {t_pnl:>+10.1f}")

    # Totals
    print("-" * 75)
    s_tot = short['pnl'].sum()
    l_tot = long['pnl'].sum()
    print(f"{'TOTAL':>10} | {len(short):>10,} | {s_tot:>+10.1f} | {len(long):>10,} | {l_tot:>+10.1f} | {s_tot+l_tot:>+10.1f}")
    print(f"{'ROI':>10} | {'':>10} | {short['pnl'].mean()*100:>+9.1f}% | {'':>10} | {long['pnl'].mean()*100:>+9.1f}% | {bets['pnl'].mean()*100:>+9.1f}%")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
