"""
Analyze the profile of bets from the cross-t super-ensemble (edge > 3%).
Loads save_df (for depth/features) and ultimate ensemble predictions, merges them.
"""
import pandas as pd
import numpy as np

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading save_df...")
save_df = pd.read_parquet("/data/projects/punim2039/alpha_odds/res/win_model/t0/ne1000_md6_lr0.01/save_df.parquet")
print(f"  save_df shape: {save_df.shape}")

print("Loading ultimate ensemble predictions...")
ens = pd.read_parquet("/data/projects/punim2039/alpha_odds/res/analysis/ultimate_cross_t_ensemble_predictions.parquet")
print(f"  ensemble shape: {ens.shape}")

# ─── Create key in save_df and merge ─────────────────────────────────────────
save_df["key"] = save_df["file_name"] + "_" + save_df["id"].astype(str)
print(f"\n  save_df key examples: {save_df['key'].head(3).tolist()}")
print(f"  ensemble key examples: {ens['key'].head(3).tolist()}")

# Merge: get depth columns from save_df, use ensemble for model_prob/edge
# Suffix the save_df columns to avoid clash with ensemble columns
merged = ens.merge(
    save_df[["key", "orig_best_back_m0", "orig_best_lay_m0", "orig_best_back_q_100_m0",
             "orig_best_lay_q_100_m0", "marketBaseRate", "numberOfActiveRunners", "runner_position"]],
    on="key", how="left"
)
print(f"\nMerged shape: {merged.shape}")
print(f"  Merge success rate: {merged['orig_best_back_m0'].notna().mean():.4f}")

# ─── Filter to edge > 3% bets ───────────────────────────────────────────────
bets = merged[merged["edge"] > 0.03].copy()
print(f"\n{'='*80}")
print(f"EDGE > 3% BETS: {len(bets):,} out of {len(merged):,} total ({100*len(bets)/len(merged):.1f}%)")
print(f"{'='*80}")

# ─── Helper: compute back odds from implied probability ──────────────────────
# orig_best_back_m0 is stored as implied probability (1/odds)
bets["back_odds_from_save"] = 1.0 / bets["orig_best_back_m0"]
bets["lay_odds"] = 1.0 / bets["orig_best_lay_m0"]

# ─── Compute market rank per race ────────────────────────────────────────────
# Extract race identifier from key (file_name part)
bets["race_file"] = bets["key"].str.rsplit("_", n=1).str[0]

# Rank by market_prob within each race (higher prob = lower rank number = more favored)
# We need to do this on the full dataset to get accurate ranks
merged["race_file"] = merged["key"].str.rsplit("_", n=1).str[0]
merged["market_rank"] = merged.groupby("race_file")["market_prob"].rank(ascending=False, method="min").astype(int)
bets = bets.merge(merged[["key", "market_rank"]], on="key", how="left")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. WHAT RUNNERS ARE WE BETTING ON?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("1. WHAT RUNNERS ARE WE BETTING ON?")
print(f"{'='*80}")

# Distribution of market implied probability
print("\n--- Distribution of market_prob (implied win probability from market) ---")
print(bets["market_prob"].describe())
print(f"\nPercentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    val = bets["market_prob"].quantile(p/100)
    odds = 1/val if val > 0 else float('inf')
    print(f"  {p}th: market_prob={val:.4f}  (implied odds={odds:.1f})")

# Distribution of back odds
print("\n--- Distribution of back_odds ---")
print(bets["back_odds"].describe())
print(f"\nPercentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    val = bets["back_odds"].quantile(p/100)
    print(f"  {p}th: {val:.1f}")

# Fraction by market rank
print("\n--- Fraction of bets by market rank ---")
rank_counts = bets["market_rank"].value_counts().sort_index()
rank_total = len(bets)
print(f"{'Rank':<8} {'Count':>8} {'Fraction':>10} {'Cum%':>8}")
cum = 0
for rank, count in rank_counts.items():
    frac = count / rank_total
    cum += frac
    label = {1: "(favorite)", 2: "(2nd)", 3: "(3rd)"}.get(rank, "")
    print(f"  {rank:<6} {count:>8,} {frac:>10.1%} {cum:>8.1%}  {label}")

# Win rate by market rank
print("\n--- Win rate by market rank (for edge>3% bets) ---")
rank_stats = bets.groupby("market_rank").agg(
    count=("win", "count"),
    wins=("win", "sum"),
    avg_market_prob=("market_prob", "mean"),
    avg_model_prob=("model_prob", "mean"),
    avg_edge=("edge", "mean"),
    avg_odds=("back_odds", "mean"),
).reset_index()
rank_stats["win_rate"] = rank_stats["wins"] / rank_stats["count"]
print(f"{'Rank':<6} {'Count':>7} {'Wins':>6} {'WinRate':>8} {'MktProb':>8} {'ModelP':>8} {'AvgEdge':>8} {'AvgOdds':>8}")
for _, r in rank_stats.iterrows():
    print(f"  {int(r['market_rank']):<4} {int(r['count']):>7,} {int(r['wins']):>6,} {r['win_rate']:>8.1%} "
          f"{r['avg_market_prob']:>8.4f} {r['avg_model_prob']:>8.4f} {r['avg_edge']:>8.4f} {r['avg_odds']:>8.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. WHEN DO WE BET WINNERS VS LOSERS?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("2. WHEN DO WE BET WINNERS VS LOSERS?")
print(f"{'='*80}")

n_winners = bets["win"].sum()
n_losers = len(bets) - n_winners
print(f"\nOf {len(bets):,} edge>3% bets:")
print(f"  Winners: {n_winners:,} ({100*n_winners/len(bets):.1f}%)")
print(f"  Losers:  {n_losers:,} ({100*n_losers/len(bets):.1f}%)")

# When we bet on a winner: what were the odds and market prob?
winners = bets[bets["win"] == 1]
losers = bets[bets["win"] == 0]

print(f"\n--- Profile of WINNING bets (n={len(winners)}) ---")
print(f"  Avg back odds:    {winners['back_odds'].mean():.2f}")
print(f"  Median back odds: {winners['back_odds'].median():.2f}")
print(f"  Avg market_prob:  {winners['market_prob'].mean():.4f} (implied odds {1/winners['market_prob'].mean():.1f})")
print(f"  Avg model_prob:   {winners['model_prob'].mean():.4f}")
print(f"  Avg edge:         {winners['edge'].mean():.4f}")

print(f"\n--- Profile of LOSING bets (n={len(losers)}) ---")
print(f"  Avg back odds:    {losers['back_odds'].mean():.2f}")
print(f"  Median back odds: {losers['back_odds'].median():.2f}")
print(f"  Avg market_prob:  {losers['market_prob'].mean():.4f} (implied odds {1/losers['market_prob'].mean():.1f})")
print(f"  Avg model_prob:   {losers['model_prob'].mean():.4f}")
print(f"  Avg edge:         {losers['edge'].mean():.4f}")

# Distribution of winning bets by odds bucket
print(f"\n--- Winning bets by odds bucket ---")
odds_bins = [0, 2, 3, 5, 10, 50, 1000]
odds_labels = ["1-2", "2-3", "3-5", "5-10", "10-50", "50+"]
winners["odds_bucket"] = pd.cut(winners["back_odds"], bins=odds_bins, labels=odds_labels)
win_by_bucket = winners.groupby("odds_bucket", observed=False).agg(count=("win", "count")).reset_index()
print(f"{'Bucket':<10} {'Winners':>8} {'Fraction':>10}")
for _, r in win_by_bucket.iterrows():
    print(f"  {r['odds_bucket']:<8} {int(r['count']):>8,} {r['count']/len(winners):>10.1%}")

# Where do we make money? By rank
print(f"\n--- Do we mainly profit from favorites or longshots? ---")
rank_pnl = bets.groupby("market_rank").apply(
    lambda g: pd.Series({
        "count": len(g),
        "wins": g["win"].sum(),
        "pnl": ((g["back_odds"] - 1) * g["win"] - (1 - g["win"])).sum(),
        "avg_odds": g["back_odds"].mean(),
    })
).reset_index()
rank_pnl["roi"] = rank_pnl["pnl"] / rank_pnl["count"]
rank_pnl["win_rate"] = rank_pnl["wins"] / rank_pnl["count"]
print(f"{'Rank':<6} {'Count':>7} {'Wins':>6} {'WinRate':>8} {'P&L':>10} {'ROI':>8} {'AvgOdds':>8}")
for _, r in rank_pnl.iterrows():
    print(f"  {int(r['market_rank']):<4} {int(r['count']):>7,} {int(r['wins']):>6,} {r['win_rate']:>8.1%} "
          f"{r['pnl']:>10.1f} {r['roi']:>8.1%} {r['avg_odds']:>8.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. P&L BREAKDOWN BY RUNNER TYPE (ODDS BUCKET)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("3. P&L BREAKDOWN BY ODDS BUCKET")
print(f"{'='*80}")

bets["odds_bucket"] = pd.cut(bets["back_odds"], bins=odds_bins, labels=odds_labels)

# P&L per bet: win pays (odds-1), loss pays -1 (unit stakes)
bets["pnl_unit"] = (bets["back_odds"] - 1) * bets["win"] - (1 - bets["win"])

# Also compute commission-adjusted P&L
# Note: marketBaseRate in save_df is normalized (z-scored), not raw commission rate
# Betfair greyhound commission is typically 7-8% of net profit per market
# We approximate as per-bet for now (actual commission is per-market-net)
COMMISSION_RATE = 0.07  # 7% flat assumption
bets["pnl_after_commission"] = np.where(
    bets["pnl_unit"] > 0,
    bets["pnl_unit"] * (1 - COMMISSION_RATE),
    bets["pnl_unit"]
)

bucket_stats = bets.groupby("odds_bucket", observed=False).agg(
    count=("win", "count"),
    wins=("win", "sum"),
    avg_edge=("edge", "mean"),
    avg_model_prob=("model_prob", "mean"),
    avg_market_prob=("market_prob", "mean"),
    avg_odds=("back_odds", "mean"),
    total_pnl=("pnl_unit", "sum"),
    total_pnl_after_comm=("pnl_after_commission", "sum"),
).reset_index()
bucket_stats["win_rate"] = bucket_stats["wins"] / bucket_stats["count"]
bucket_stats["roi"] = bucket_stats["total_pnl"] / bucket_stats["count"]
bucket_stats["roi_after_comm"] = bucket_stats["total_pnl_after_comm"] / bucket_stats["count"]

print(f"\n{'Bucket':<8} {'Count':>7} {'Wins':>6} {'WinRate':>8} {'AvgEdge':>8} {'AvgOdds':>8} "
      f"{'TotalP&L':>10} {'ROI':>8} {'P&L(comm)':>10} {'ROI(comm)':>10}")
for _, r in bucket_stats.iterrows():
    print(f"  {r['odds_bucket']:<6} {int(r['count']):>7,} {int(r['wins']):>6,} {r['win_rate']:>8.1%} "
          f"{r['avg_edge']:>8.4f} {r['avg_odds']:>8.1f} {r['total_pnl']:>10.1f} {r['roi']:>8.1%} "
          f"{r['total_pnl_after_comm']:>10.1f} {r['roi_after_comm']:>10.1%}")

print(f"\n--- Totals ---")
print(f"  Total bets:              {len(bets):,}")
print(f"  Total wins:              {int(bets['win'].sum()):,}")
print(f"  Overall win rate:        {bets['win'].mean():.1%}")
print(f"  Total P&L (unit stakes): {bets['pnl_unit'].sum():.1f}")
print(f"  Overall ROI:             {bets['pnl_unit'].sum()/len(bets):.1%}")
print(f"  Total P&L after comm:    {bets['pnl_after_commission'].sum():.1f}")
print(f"  ROI after commission:    {bets['pnl_after_commission'].sum()/len(bets):.1%}")

# Cumulative P&L fraction by bucket
print(f"\n--- Where does profit come from? (cumulative) ---")
total_pnl = bets["pnl_unit"].sum()
cum_pnl = 0
print(f"{'Bucket':<8} {'BucketP&L':>10} {'%ofTotal':>10} {'Cumul%':>10}")
for _, r in bucket_stats.iterrows():
    if total_pnl != 0:
        pct = r["total_pnl"] / total_pnl
        cum_pnl += r["total_pnl"]
        cum_pct = cum_pnl / total_pnl
    else:
        pct = 0
        cum_pct = 0
    print(f"  {r['odds_bucket']:<6} {r['total_pnl']:>10.1f} {pct:>10.1%} {cum_pct:>10.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DEPTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("4. DEPTH / SLIPPAGE ANALYSIS")
print(f"{'='*80}")

# orig_best_back_m0 = implied prob at best back (1/best_back_odds)
# orig_best_back_q_100_m0 = implied prob for $100 execution (weighted avg)
# Slippage = execution price is worse (lower odds = higher implied prob)

has_depth = bets["orig_best_back_q_100_m0"].notna()
depth_bets = bets[has_depth].copy()
print(f"\nBets with depth data: {has_depth.sum():,} / {len(bets):,}")

# Best back odds vs $100 execution odds
depth_bets["best_back_odds"] = 1.0 / depth_bets["orig_best_back_m0"]
depth_bets["exec_odds_100"] = 1.0 / depth_bets["orig_best_back_q_100_m0"]

# Slippage: how much worse is the $100 execution vs best price
depth_bets["slippage_pct"] = (depth_bets["best_back_odds"] - depth_bets["exec_odds_100"]) / depth_bets["best_back_odds"]

print(f"\n--- Slippage: best back vs $100 execution ---")
print(f"  Mean slippage:   {depth_bets['slippage_pct'].mean():.2%}")
print(f"  Median slippage: {depth_bets['slippage_pct'].median():.2%}")
print(f"  Slippage percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    val = depth_bets["slippage_pct"].quantile(p/100)
    print(f"    {p}th: {val:.2%}")

print(f"\n  Fraction with zero slippage (full $100 at best price): "
      f"{(depth_bets['slippage_pct'] <= 0.001).mean():.1%}")
print(f"  Fraction with <1% slippage: {(depth_bets['slippage_pct'] < 0.01).mean():.1%}")
print(f"  Fraction with <5% slippage: {(depth_bets['slippage_pct'] < 0.05).mean():.1%}")

# Estimate depth at best price
# If exec_odds_100 == best_back_odds, then >= $100 available at best price
# If exec_odds_100 < best_back_odds, there's less at best price
# Rough estimate: if slippage is X% for $100, then ~($100 * (1-X)) is at best price
# More precisely: depth_at_best ≈ $100 * (exec_price / best_price) ... this is rough
# Actually we can't compute exact depth without level-by-level data
# But we can infer: if slippage is 0, there's >= $100 at best price

print(f"\n--- Depth at best price (estimates) ---")
print(f"  Fraction with >= $100 at best back:  {(depth_bets['slippage_pct'] <= 0.001).mean():.1%}")

# For $25 at best price: harder to estimate without more data
# If slippage for $100 is < 75%, then roughly >= $25 is at the best price
# (very rough approximation)
print(f"  Fraction with >= ~$25 at best back (slippage<75% for $100): "
      f"{(depth_bets['slippage_pct'] < 0.75).mean():.1%}")

# Slippage by odds bucket
print(f"\n--- Slippage by odds bucket ---")
depth_bets["odds_bucket"] = pd.cut(depth_bets["best_back_odds"], bins=odds_bins, labels=odds_labels)
slip_by_bucket = depth_bets.groupby("odds_bucket", observed=False).agg(
    count=("slippage_pct", "count"),
    mean_slip=("slippage_pct", "mean"),
    median_slip=("slippage_pct", "median"),
    best_odds_avg=("best_back_odds", "mean"),
    exec_odds_avg=("exec_odds_100", "mean"),
    frac_zero_slip=("slippage_pct", lambda x: (x <= 0.001).mean()),
).reset_index()
print(f"{'Bucket':<8} {'Count':>7} {'MeanSlip':>10} {'MedSlip':>10} {'AvgBest':>8} {'AvgExec':>8} {'ZeroSlip':>10}")
for _, r in slip_by_bucket.iterrows():
    print(f"  {r['odds_bucket']:<6} {int(r['count']):>7,} {r['mean_slip']:>10.2%} {r['median_slip']:>10.2%} "
          f"{r['best_odds_avg']:>8.1f} {r['exec_odds_avg']:>8.1f} {r['frac_zero_slip']:>10.1%}")

# ─── Bonus: Effective edge after slippage ────────────────────────────────────
print(f"\n--- Edge erosion from slippage ---")
# If we have to execute at worse odds, our actual edge is reduced
# edge_raw = model_prob - market_prob
# But execution is at exec_odds_100, so effective market_prob is 1/exec_odds_100
depth_bets["effective_market_prob"] = depth_bets["orig_best_back_q_100_m0"]  # already implied prob
depth_bets["effective_edge"] = depth_bets["model_prob"] - depth_bets["effective_market_prob"]

print(f"  Mean raw edge:       {depth_bets['edge'].mean():.4f}")
print(f"  Mean effective edge: {depth_bets['effective_edge'].mean():.4f}")
print(f"  Mean edge erosion:   {(depth_bets['edge'] - depth_bets['effective_edge']).mean():.4f}")
print(f"  Fraction still >3% after slippage: {(depth_bets['effective_edge'] > 0.03).mean():.1%}")
print(f"  Fraction still >0% after slippage: {(depth_bets['effective_edge'] > 0).mean():.1%}")

# ─── Bonus: P&L if we used q_100 execution prices ───────────────────────────
print(f"\n--- P&L comparison: best price vs $100 execution ---")
depth_bets["pnl_best"] = (depth_bets["best_back_odds"] - 1) * depth_bets["win"] - (1 - depth_bets["win"])
depth_bets["pnl_exec100"] = (depth_bets["exec_odds_100"] - 1) * depth_bets["win"] - (1 - depth_bets["win"])
print(f"  P&L at best back:     {depth_bets['pnl_best'].sum():.1f}  (ROI: {depth_bets['pnl_best'].sum()/len(depth_bets):.1%})")
print(f"  P&L at $100 exec:     {depth_bets['pnl_exec100'].sum():.1f}  (ROI: {depth_bets['pnl_exec100'].sum()/len(depth_bets):.1%})")
print(f"  P&L difference:       {depth_bets['pnl_best'].sum() - depth_bets['pnl_exec100'].sum():.1f}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
