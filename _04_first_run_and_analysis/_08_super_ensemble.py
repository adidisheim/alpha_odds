"""Build V1+V2 super-ensemble and run definitive backtest."""
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

# Load V2 models
v2_dfs = {}
for config in sorted(os.listdir("win_model_v2/t0")):
    path = f"win_model_v2/t0/{config}/save_df.parquet"
    if os.path.exists(path):
        v2_dfs[config] = pd.read_parquet(path)

v2_sorted = sorted(v2_dfs.keys(), key=lambda c: log_loss(
    v2_dfs[c]["win"].values, v2_dfs[c]["model_prob"].clip(0.001, 0.999).values))

# Load V1 models
v1_configs = ["ne1000_md6_lr0.01", "ne100_md6_lr0.1", "ne500_md3_lr0.05", "ne500_md6_lr0.01",
              "ne100_md6_lr0.05", "ne1000_md3_lr0.05", "ne500_md6_lr0.05"]
v1_dfs = {}
for c in v1_configs:
    path = f"win_model/t0/{c}/save_df.parquet"
    if os.path.exists(path):
        v1_dfs[c] = pd.read_parquet(path)

# Build individual ensembles
v1_ens = np.mean([v1_dfs[c]["model_prob"].values for c in v1_configs if c in v1_dfs], axis=0)
v2_ens_10 = np.mean([v2_dfs[c]["model_prob"].values for c in v2_sorted[:10]], axis=0)
v2_ens_15 = np.mean([v2_dfs[c]["model_prob"].values for c in v2_sorted[:15]], axis=0)

base = v1_dfs[v1_configs[0]]
win = base["win"].values

v1_ll = log_loss(win, v1_ens)
v2_10_ll = log_loss(win, v2_ens_10)
v2_15_ll = log_loss(win, v2_ens_15)

print(f"V1 Top-7:  {v1_ll:.6f}")
print(f"V2 Top-10: {v2_10_ll:.6f}")
print(f"V2 Top-15: {v2_15_ll:.6f}")

# Super-ensemble: V1 + V2
print("\n=== SUPER-ENSEMBLE WEIGHTS ===")
for w2 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    combined = v1_ens * (1 - w2) + v2_ens_15 * w2
    ll = log_loss(win, combined)
    print(f"V1({1-w2:.0%}) + V2({w2:.0%}): {ll:.6f}")

# Equal weight
super_ens = (v1_ens + v2_ens_15) / 2
super_ll = log_loss(win, super_ens)
print(f"\nSuper-ensemble (50/50): {super_ll:.6f}")

# Pick the best
best_probs = v2_ens_15
best_label = "V2 Top-15"
best_ll_val = v2_15_ll
if super_ll < v2_15_ll:
    best_probs = super_ens
    best_label = "Super-Ensemble"
    best_ll_val = super_ll

base_bt = base.copy()
base_bt["model_prob"] = best_probs
base_bt["edge"] = base_bt["model_prob"] - base_bt["market_prob"]
base_bt["back_odds"] = 1 / base_bt["orig_best_back_m0"]
base_bt = base_bt[(base_bt["back_odds"] > 1.01) & (base_bt["back_odds"] < 1000)]

commission = 0.075
print(f"\n{'='*80}")
print(f"=== {best_label} BACKTEST ({commission:.1%} commission) ===")
print(f"{'='*80}")
print(f"Best model LL: {best_ll_val:.6f}")
print(f"Total OOS: {len(base_bt):,}")

for et in [0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.07, 0.10]:
    bets = base_bt[base_bt["edge"] > et].copy()
    if len(bets) < 10:
        continue
    bets["pnl"] = bets["win"] * (bets["back_odds"] - 1) * (1 - commission) - (1 - bets["win"])
    n = len(bets)
    wr = bets["win"].mean()
    roi = bets["pnl"].mean() * 100
    z = bets["pnl"].mean() / bets["pnl"].std() * np.sqrt(n) if bets["pnl"].std() > 0 else 0
    p = 1 - norm_cdf(z)
    bets["month"] = pd.to_datetime(bets["marketTime_local"]).dt.to_period("M")
    monthly = bets.groupby("month")["pnl"].sum()
    pm = int((monthly > 0).sum())
    sh_m = monthly.mean() / monthly.std() if monthly.std() > 0 else 0
    sh_a = sh_m * np.sqrt(12)
    profit_25 = bets["pnl"].sum() * 25
    monthly_25 = profit_25 / len(monthly)

    print(f"\nEdge>{et:.1%}:")
    print(f"  Bets: {n:,}, WR: {wr:.1%}, ROI: {roi:+.1f}%")
    print(f"  Sharpe: {sh_a:.1f}, z={z:.2f}, p={p:.6f}")
    print(f"  Profitable months: {pm}/{len(monthly)}")
    print(f"  $25/bet: ${profit_25:,.0f} total, ${monthly_25:,.0f}/month")

# Monthly breakdown for recommended strategy
print(f"\n{'='*80}")
print(f"=== MONTHLY BREAKDOWN (Edge > 2.5%) ===")
print(f"{'='*80}")
bets_25 = base_bt[base_bt["edge"] > 0.025].copy()
bets_25["pnl"] = bets_25["win"] * (bets_25["back_odds"] - 1) * (1 - commission) - (1 - bets_25["win"])
bets_25["month"] = pd.to_datetime(bets_25["marketTime_local"]).dt.to_period("M")
monthly_25 = bets_25.groupby("month").agg(
    n_bets=("pnl", "count"),
    win_rate=("win", "mean"),
    total_pnl=("pnl", "sum"),
    avg_odds=("back_odds", "mean"),
).reset_index()
monthly_25["roi_pct"] = monthly_25["total_pnl"] / monthly_25["n_bets"] * 100
monthly_25["cum_pnl"] = monthly_25["total_pnl"].cumsum()
monthly_25["profit_25"] = monthly_25["total_pnl"] * 25
print(monthly_25.to_string(index=False))

# Save super-ensemble predictions
save_dir = "/data/projects/punim2039/alpha_odds/res/analysis/"
os.makedirs(save_dir, exist_ok=True)
base_bt[["file_name", "id", "win", "model_prob", "market_prob", "edge",
         "back_odds", "orig_best_back_m0", "marketTime_local"]].to_parquet(
    save_dir + "super_ensemble_predictions_t0.parquet"
)
print(f"\nSuper-ensemble saved to {save_dir}")
print("\nDone!")
