"""
Analyze order book depth and slippage for value betting strategy.
Questions:
1. For edge > 3% bets, what's the average slippage at q_100 and q_1000?
2. What fraction of bets have >= $25 available at best back price?
3. How does slippage affect ROI?
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ORDER BOOK DEPTH & SLIPPAGE ANALYSIS")
print("=" * 80)

# ── Load ensemble predictions ──
print("\nLoading ensemble predictions...")
ens = pd.read_parquet(
    "/data/projects/punim2039/alpha_odds/res/analysis/ultimate_cross_t_ensemble_predictions.parquet",
    engine="pyarrow"
)
print(f"  Ensemble rows: {len(ens):,}")

# ── Load save_df (has orig_best_back_m0, orig_best_back_q_100_m0) ──
print("Loading save_df (t0 model with q_100 data)...")
sdf = pd.read_parquet(
    "/data/projects/punim2039/alpha_odds/res/win_model/t0/ne1000_md6_lr0.01/save_df.parquet",
    engine="pyarrow"
)
sdf["key"] = sdf["file_name"].astype(str) + "_" + sdf["id"].astype(str)

# ── Load features to get q_1000 and cum_qty columns ──
print("Loading feature files for q_1000 and depth data...")
feat_cols = ["file_name", "id", "best_back_q_1000_m0", "best_back_cum_qty_m0",
             "total_back_qty_m0", "best_lay_q_1000_m0", "best_back_m0"]
feat_parts = []
for i in range(10):
    fpath = f"/data/projects/punim2039/alpha_odds/res/features_t0/greyhound_au_features_part_{i}.parquet"
    try:
        part = pd.read_parquet(fpath, columns=feat_cols, engine="pyarrow")
        feat_parts.append(part)
    except Exception as e:
        print(f"  Warning: Could not load part {i}: {e}")
features = pd.concat(feat_parts, ignore_index=True)
features["key"] = features["file_name"].astype(str) + "_" + features["id"].astype(str)
# Deduplicate (in case of overlaps)
features = features.drop_duplicates(subset="key", keep="first")
print(f"  Feature rows: {len(features):,}")

# ── Merge everything ──
print("Merging datasets...")
df = ens.merge(
    sdf[["key", "orig_best_back_m0", "orig_best_back_q_100_m0",
         "orig_best_lay_m0", "orig_best_lay_q_100_m0"]],
    on="key", how="left"
)
df = df.merge(
    features[["key", "best_back_q_1000_m0", "best_back_cum_qty_m0",
              "total_back_qty_m0", "best_lay_q_1000_m0"]],
    on="key", how="left"
)
print(f"  Final merged rows: {len(df):,}")
print(f"  Rows with q_1000 data: {df['best_back_q_1000_m0'].notna().sum():,}")
print(f"  Rows with cum_qty data: {df['best_back_cum_qty_m0'].notna().sum():,}")

# ── Convert implied probabilities to odds ──
# All stored as implied prob = 1/odds. So odds = 1/prob.
# Special: missing backs are 1.0 (1/1.0 = odds of 1), missing lays are 1/1001
df["back_odds_best"] = 1.0 / df["orig_best_back_m0"]
df["back_odds_q100"] = 1.0 / df["orig_best_back_q_100_m0"]
df["back_odds_q1000"] = 1.0 / df["best_back_q_1000_m0"]

# Handle edge cases (div by zero, etc.)
df.loc[df["orig_best_back_m0"] <= 0, "back_odds_best"] = np.nan
df.loc[df["orig_best_back_q_100_m0"] <= 0, "back_odds_q100"] = np.nan
df.loc[df["best_back_q_1000_m0"] <= 0, "back_odds_q1000"] = np.nan

# ── Slippage metrics ──
# Slippage = (q_X price - best price) / best price as percentage
# In odds: lower odds = worse for backer. In prob: higher prob = worse for backer.
# Slippage in odds: (best_odds - q100_odds) / best_odds
df["slippage_q100_pct"] = (df["back_odds_best"] - df["back_odds_q100"]) / df["back_odds_best"] * 100
df["slippage_q1000_pct"] = (df["back_odds_best"] - df["back_odds_q1000"]) / df["back_odds_best"] * 100

# ── Define strategy universe ──
edge_thresholds = [0.01, 0.02, 0.03, 0.05]

print("\n" + "=" * 80)
print("SECTION 1: SLIPPAGE BY EDGE THRESHOLD")
print("=" * 80)

for thresh in edge_thresholds:
    mask = df["edge"] > thresh
    sub = df[mask].copy()
    n = len(sub)
    if n == 0:
        print(f"\n  Edge > {thresh*100:.0f}%: No bets")
        continue

    print(f"\n{'─' * 60}")
    print(f"  Edge > {thresh*100:.0f}% : {n:,} bets ({n/len(df)*100:.1f}% of universe)")
    print(f"{'─' * 60}")

    # Odds stats
    valid_best = sub["back_odds_best"].dropna()
    valid_q100 = sub["back_odds_q100"].dropna()
    valid_q1000 = sub["back_odds_q1000"].dropna()

    print(f"\n  Average back odds (best):  {valid_best.mean():.2f}  (median: {valid_best.median():.2f})")
    if len(valid_q100) > 0:
        print(f"  Average back odds (q100):  {valid_q100.mean():.2f}  (median: {valid_q100.median():.2f})")
    if len(valid_q1000) > 0:
        print(f"  Average back odds (q1000): {valid_q1000.mean():.2f}  (median: {valid_q1000.median():.2f})")

    # Slippage stats
    slip100 = sub["slippage_q100_pct"].dropna()
    slip1000 = sub["slippage_q1000_pct"].dropna()

    if len(slip100) > 0:
        print(f"\n  Slippage to fill $100 order:")
        print(f"    Mean: {slip100.mean():.2f}%   Median: {slip100.median():.2f}%")
        print(f"    25th: {slip100.quantile(0.25):.2f}%   75th: {slip100.quantile(0.75):.2f}%")
        print(f"    90th: {slip100.quantile(0.90):.2f}%   95th: {slip100.quantile(0.95):.2f}%")

    if len(slip1000) > 0:
        print(f"\n  Slippage to fill $1000 order:")
        print(f"    Mean: {slip1000.mean():.2f}%   Median: {slip1000.median():.2f}%")
        print(f"    25th: {slip1000.quantile(0.25):.2f}%   75th: {slip1000.quantile(0.75):.2f}%")
        print(f"    90th: {slip1000.quantile(0.90):.2f}%   95th: {slip1000.quantile(0.95):.2f}%")


print("\n" + "=" * 80)
print("SECTION 2: DEPTH AT BEST PRICE — CAN WE FILL $25?")
print("=" * 80)

# best_back_cum_qty_m0 = total AUD available at the best back price
# If cum_qty >= 25, we can fill a $25 order at best price
print(f"\n  Rows with depth data: {df['best_back_cum_qty_m0'].notna().sum():,}")

depth = df["best_back_cum_qty_m0"]
valid_depth = depth.dropna()
print(f"\n  Overall depth at best back price (AUD):")
print(f"    Mean:   ${valid_depth.mean():.2f}")
print(f"    Median: ${valid_depth.median():.2f}")
print(f"    25th:   ${valid_depth.quantile(0.25):.2f}")
print(f"    75th:   ${valid_depth.quantile(0.75):.2f}")
print(f"    Min:    ${valid_depth.min():.2f}")
print(f"    Max:    ${valid_depth.max():.2f}")

for amount in [10, 25, 50, 100, 250]:
    frac_all = (valid_depth >= amount).mean() * 100
    print(f"\n  Fraction with >= ${amount} at best back price (all): {frac_all:.1f}%")

    for thresh in [0.03, 0.05]:
        mask = (df["edge"] > thresh) & df["best_back_cum_qty_m0"].notna()
        sub_depth = df.loc[mask, "best_back_cum_qty_m0"]
        if len(sub_depth) > 0:
            frac = (sub_depth >= amount).mean() * 100
            print(f"    ... among edge > {thresh*100:.0f}%: {frac:.1f}%  (n={len(sub_depth):,})")


print("\n" + "=" * 80)
print("SECTION 3: SLIPPAGE IMPACT ON ROI")
print("=" * 80)

# ROI calculation: back a runner at given odds.
# If they win: profit = stake * (odds - 1) - commission * stake * (odds - 1)
# If they lose: profit = -stake
# Commission applies to NET PROFIT only (not gross).
# net_roi = (1 - commission) * win_prob * (odds - 1) - (1 - win_prob)
# where win_prob = model_prob (our estimated true probability)

# For each scenario, compute expected ROI per bet using model_prob as true probability
# Also compute REALIZED ROI using actual win/loss outcomes

commission = 0.07  # 7% Betfair commission on net profit

for thresh in [0.01, 0.02, 0.03, 0.05]:
    mask = df["edge"] > thresh
    sub = df[mask].copy()
    n = len(sub)
    if n == 0:
        continue

    print(f"\n{'─' * 60}")
    print(f"  Edge > {thresh*100:.0f}% : {n:,} bets")
    print(f"{'─' * 60}")

    # --- Scenario A: Best back price (best case, tiny order) ---
    odds_a = sub["back_odds_best"]
    wins_a = sub["win"]
    profit_a = np.where(wins_a == 1, (1 - commission) * (odds_a - 1), -1.0)
    roi_a = profit_a.mean() * 100
    win_rate = wins_a.mean()

    print(f"\n  Win rate: {win_rate*100:.2f}%")
    print(f"  Avg model_prob: {sub['model_prob'].mean()*100:.2f}%")
    print(f"  Avg market_prob: {sub['market_prob'].mean()*100:.2f}%")
    print(f"  Avg edge: {sub['edge'].mean()*100:.2f}%")

    print(f"\n  Scenario A — Best back price (thin, best case):")
    print(f"    Avg odds: {odds_a.mean():.2f}")
    print(f"    Realized ROI per $1 staked: {roi_a:.2f}%")
    print(f"    Total P&L on $1/bet: ${profit_a.sum():.2f} over {n:,} bets")

    # --- Scenario B: q_100 price (realistic for ~$25 orders) ---
    mask_q100 = sub["back_odds_q100"].notna()
    if mask_q100.sum() > 0:
        sub_q100 = sub[mask_q100]
        odds_b = sub_q100["back_odds_q100"]
        wins_b = sub_q100["win"]
        profit_b = np.where(wins_b == 1, (1 - commission) * (odds_b - 1), -1.0)
        roi_b = profit_b.mean() * 100

        print(f"\n  Scenario B — q_100 price (execution price for $100 order):")
        print(f"    Avg odds: {odds_b.mean():.2f}")
        print(f"    Realized ROI per $1 staked: {roi_b:.2f}%")
        print(f"    ROI degradation vs best: {roi_a - roi_b:.2f} pp")
        print(f"    Total P&L on $1/bet: ${profit_b.sum():.2f} over {mask_q100.sum():,} bets")

    # --- Scenario C: q_1000 price (worst case, big order) ---
    mask_q1000 = sub["back_odds_q1000"].notna()
    if mask_q1000.sum() > 0:
        sub_q1000 = sub[mask_q1000]
        odds_c = sub_q1000["back_odds_q1000"]
        wins_c = sub_q1000["win"]
        profit_c = np.where(wins_c == 1, (1 - commission) * (odds_c - 1), -1.0)
        roi_c = profit_c.mean() * 100

        print(f"\n  Scenario C — q_1000 price (execution price for $1000 order):")
        print(f"    Avg odds: {odds_c.mean():.2f}")
        print(f"    Realized ROI per $1 staked: {roi_c:.2f}%")
        print(f"    ROI degradation vs best: {roi_a - roi_c:.2f} pp")
        print(f"    Total P&L on $1/bet: ${profit_c.sum():.2f} over {mask_q1000.sum():,} bets")


print("\n" + "=" * 80)
print("SECTION 4: SLIPPAGE IMPACT — CONDITIONAL ON DEPTH SUFFICIENCY")
print("=" * 80)

# For the 3% edge threshold, show ROI broken down by depth availability
thresh = 0.03
mask = (df["edge"] > thresh) & df["best_back_cum_qty_m0"].notna()
sub = df[mask].copy()
print(f"\n  Edge > 3%, with depth data: {len(sub):,} bets")

for min_depth in [0, 10, 25, 50, 100]:
    depth_mask = sub["best_back_cum_qty_m0"] >= min_depth
    ss = sub[depth_mask]
    if len(ss) == 0:
        continue

    odds = ss["back_odds_best"]
    wins = ss["win"]
    profit = np.where(wins == 1, (1 - commission) * (odds - 1), -1.0)
    roi = profit.mean() * 100

    odds_q100 = ss["back_odds_q100"]
    profit_q100 = np.where(wins == 1, (1 - commission) * (odds_q100 - 1), -1.0)
    roi_q100 = profit_q100.mean() * 100

    print(f"\n  Depth >= ${min_depth}: {len(ss):,} bets ({len(ss)/len(sub)*100:.1f}%)")
    print(f"    ROI at best price: {roi:.2f}%")
    print(f"    ROI at q100 price: {roi_q100:.2f}%")
    print(f"    Avg back odds (best): {odds.mean():.2f}")
    print(f"    Avg edge: {ss['edge'].mean()*100:.2f}%")
    print(f"    Win rate: {wins.mean()*100:.2f}%")


print("\n" + "=" * 80)
print("SECTION 5: SLIPPAGE BY ODDS BUCKET")
print("=" * 80)

thresh = 0.03
mask = df["edge"] > thresh
sub = df[mask].copy()

# Create odds buckets
bins = [1, 2, 3, 5, 8, 15, 30, 1001]
labels = ["1-2", "2-3", "3-5", "5-8", "8-15", "15-30", "30+"]
sub["odds_bucket"] = pd.cut(sub["back_odds_best"], bins=bins, labels=labels)

print(f"\n  Edge > 3%: Slippage and depth by odds bucket\n")
print(f"  {'Bucket':<10} {'Count':>8} {'Avg Depth':>12} {'Slip q100':>12} {'Slip q1000':>12} {'ROI best':>10} {'ROI q100':>10}")
print(f"  {'─'*10} {'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

for bucket in labels:
    bmask = sub["odds_bucket"] == bucket
    bs = sub[bmask]
    if len(bs) == 0:
        continue

    avg_depth = bs["best_back_cum_qty_m0"].mean()
    slip100 = bs["slippage_q100_pct"].mean()
    slip1000 = bs["slippage_q1000_pct"].mean()

    odds_best = bs["back_odds_best"]
    wins = bs["win"]
    profit_best = np.where(wins == 1, (1 - commission) * (odds_best - 1), -1.0)
    roi_best = profit_best.mean() * 100

    odds_q100 = bs["back_odds_q100"]
    profit_q100 = np.where(wins == 1, (1 - commission) * (odds_q100 - 1), -1.0)
    roi_q100 = profit_q100.mean() * 100

    depth_str = f"${avg_depth:.1f}" if not np.isnan(avg_depth) else "N/A"
    slip100_str = f"{slip100:.1f}%" if not np.isnan(slip100) else "N/A"
    slip1000_str = f"{slip1000:.1f}%" if not np.isnan(slip1000) else "N/A"

    print(f"  {bucket:<10} {len(bs):>8,} {depth_str:>12} {slip100_str:>12} {slip1000_str:>12} {roi_best:>9.1f}% {roi_q100:>9.1f}%")


print("\n" + "=" * 80)
print("SECTION 6: PRACTICAL RECOMMENDATION FOR $25 BETS")
print("=" * 80)

thresh = 0.03
mask = df["edge"] > thresh
sub = df[mask].copy()

# For $25 bet: the execution price is somewhere between best and q_100
# Linear interpolation: q_25 ~ best + (25/100) * (q_100 - best)  [very approximate]
# In odds space:
sub["back_odds_q25_approx"] = sub["back_odds_best"] - 0.25 * (sub["back_odds_best"] - sub["back_odds_q100"])

valid_mask = sub["back_odds_q25_approx"].notna() & (sub["back_odds_q25_approx"] > 0)
sub_valid = sub[valid_mask]

odds_q25 = sub_valid["back_odds_q25_approx"]
wins_q25 = sub_valid["win"]
profit_q25 = np.where(wins_q25 == 1, (1 - commission) * (odds_q25 - 1), -1.0)
roi_q25 = profit_q25.mean() * 100

print(f"\n  For $25 bets (edge > 3%), {len(sub_valid):,} bets:")
print(f"    Approx execution odds (interpolated): {odds_q25.mean():.2f}")
print(f"    ROI at best price:     {sub_valid['back_odds_best'].apply(lambda o: (1-commission)*(o-1) if sub_valid.loc[sub_valid['back_odds_best']==o,'win'].iloc[0]==1 else -1).mean()*100:.2f}%")

# Simpler computation
profit_best_v = np.where(sub_valid["win"] == 1, (1 - commission) * (sub_valid["back_odds_best"] - 1), -1.0)
roi_best_v = profit_best_v.mean() * 100

print(f"    ROI at best price:       {roi_best_v:.2f}%")
print(f"    ROI at q25 price:        {roi_q25:.2f}%")
print(f"    ROI at q100 price:       {(np.where(sub_valid['win']==1, (1-commission)*(sub_valid['back_odds_q100']-1), -1.0)).mean()*100:.2f}%")
print(f"    ROI degradation (best → q25): {roi_best_v - roi_q25:.2f} pp")

# What if we also filter for sufficient depth?
depth_mask = sub_valid["best_back_cum_qty_m0"] >= 25
sub_deep = sub_valid[depth_mask]
if len(sub_deep) > 0:
    profit_deep = np.where(sub_deep["win"] == 1, (1 - commission) * (sub_deep["back_odds_best"] - 1), -1.0)
    roi_deep = profit_deep.mean() * 100
    print(f"\n  With >= $25 depth at best price ({len(sub_deep):,} bets, {len(sub_deep)/len(sub_valid)*100:.1f}% of total):")
    print(f"    ROI at best price: {roi_deep:.2f}%")
    print(f"    These bets can likely be filled at best price with $25 stake.")

# How about $25 with a limit order?
print(f"\n  LIMIT ORDER STRATEGY:")
print(f"    If placing limit orders at best back price for $25:")
print(f"    - Bets where depth >= $25: you get filled immediately at best price")
print(f"    - Bets where depth < $25: limit order waits, may or may not fill")
print(f"    This is a natural way to manage slippage for small bet sizes.")


print("\n" + "=" * 80)
print("SECTION 7: SUMMARY TABLE")
print("=" * 80)

print(f"\n  {'Metric':<45} {'Value':>15}")
print(f"  {'─'*45} {'─'*15}")

thresh = 0.03
mask = df["edge"] > thresh
sub = df[mask]
n_bets = len(sub)
avg_edge = sub["edge"].mean() * 100

# Using best price
profit_best = np.where(sub["win"] == 1, (1 - commission) * (sub["back_odds_best"] - 1), -1.0)
roi_best = profit_best.mean() * 100

# Using q100 price
valid_q100 = sub["back_odds_q100"].notna()
sub_q100 = sub[valid_q100]
profit_q100 = np.where(sub_q100["win"] == 1, (1 - commission) * (sub_q100["back_odds_q100"] - 1), -1.0)
roi_q100 = profit_q100.mean() * 100

depth_data = sub["best_back_cum_qty_m0"].dropna()
frac_25 = (depth_data >= 25).mean() * 100

avg_slip_100 = sub["slippage_q100_pct"].dropna().mean()
avg_slip_1000 = sub["slippage_q1000_pct"].dropna().mean()

print(f"  {'Total bets (edge > 3%)':<45} {n_bets:>15,}")
print(f"  {'Average edge':<45} {avg_edge:>14.2f}%")
print(f"  {'Win rate':<45} {sub['win'].mean()*100:>14.2f}%")
print(f"  {'Avg back odds (best)':<45} {sub['back_odds_best'].mean():>15.2f}")
print(f"  {'Avg back odds (q100)':<45} {sub['back_odds_q100'].dropna().mean():>15.2f}")
print(f"  {'Avg slippage for $100 order':<45} {avg_slip_100:>14.2f}%")
print(f"  {'Avg slippage for $1000 order':<45} {avg_slip_1000:>14.2f}%")
print(f"  {'Median depth at best back (AUD)':<45} ${depth_data.median():>13.2f}")
print(f"  {'Fraction with >= $25 at best price':<45} {frac_25:>14.1f}%")
print(f"  {'ROI at best price':<45} {roi_best:>14.2f}%")
print(f"  {'ROI at q100 price':<45} {roi_q100:>14.2f}%")
print(f"  {'ROI degradation (best → q100)':<45} {roi_best - roi_q100:>14.2f}pp")

# Daily P&L estimate
avg_daily_bets = n_bets / 365  # rough estimate
daily_pnl_best = profit_best.sum() / 365 * 25  # $25 per bet
daily_pnl_q100 = profit_q100.sum() / 365 * 25

print(f"\n  {'─'*60}")
print(f"  DAILY P&L ESTIMATES (at $25/bet, edge > 3%)")
print(f"  {'─'*60}")
print(f"  {'Avg bets per day':<45} {avg_daily_bets:>15.1f}")
print(f"  {'Daily P&L at best price':<45} ${daily_pnl_best:>13.2f}")
print(f"  {'Daily P&L at q100 price':<45} ${daily_pnl_q100:>13.2f}")
print(f"  {'Annual P&L at best price':<45} ${daily_pnl_best*365:>12.2f}")
print(f"  {'Annual P&L at q100 price':<45} ${daily_pnl_q100*365:>12.2f}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
