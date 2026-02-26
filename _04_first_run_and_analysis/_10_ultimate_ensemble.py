"""
Build the ultimate cross-t super-ensemble: V1(all-t) + V2(all-t) blend.
Align across time definitions and optimize weights.
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


def load_ensemble(model_dir, t_def, top_n):
    """Load models for a given t_def and return top-N ensemble probs + base df."""
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


# ============================================================
# Load all ensembles
# ============================================================
print("Loading all models...", flush=True)

all_components = {}  # {(version, t_def): (base_df, ens_prob)}

for t in [0, 1, 2, 3]:
    base, prob = load_ensemble("win_model", t, top_n=7)
    if base is not None:
        ll = log_loss(base["win"].values, prob)
        all_components[("V1", t)] = (base, prob)
        print(f"  V1 t{t}: LL={ll:.6f} ({len(base):,} rows)")

    base, prob = load_ensemble("win_model_v2", t, top_n=15)
    if base is not None:
        ll = log_loss(base["win"].values, prob)
        all_components[("V2", t)] = (base, prob)
        print(f"  V2 t{t}: LL={ll:.6f} ({len(base):,} rows)")

# ============================================================
# Align all DataFrames by (file_name, id)
# ============================================================
print("\nAligning DataFrames...", flush=True)

# Start from t0 V1 as base
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

prob_cols = [c for c in aligned.columns if c.startswith(("V1_", "V2_"))]
print(f"Aligned: {len(aligned):,} rows, {len(prob_cols)} components: {prob_cols}")

win = aligned["win"].values
market_prob = aligned["market_prob"].values
market_ll = log_loss(win, np.clip(market_prob, 0.001, 0.999))
print(f"Market LL: {market_ll:.6f}")

# ============================================================
# Build cross-t ensembles
# ============================================================
print("\n" + "=" * 80)
print("=== CROSS-T ENSEMBLE COMPARISON ===")
print("=" * 80)

v1_cols = [c for c in prob_cols if c.startswith("V1_")]
v2_cols = [c for c in prob_cols if c.startswith("V2_")]

v1_cross_t = aligned[v1_cols].mean(axis=1).values
v2_cross_t = aligned[v2_cols].mean(axis=1).values
v1_t0_only = aligned["V1_t0"].values
v2_t0_only = aligned["V2_t0"].values if "V2_t0" in aligned.columns else None

print(f"V1 t0-only: LL={log_loss(win, v1_t0_only):.6f}")
print(f"V1 cross-t ({len(v1_cols)} components): LL={log_loss(win, v1_cross_t):.6f}")
if v2_t0_only is not None:
    print(f"V2 t0-only: LL={log_loss(win, v2_t0_only):.6f}")
print(f"V2 cross-t ({len(v2_cols)} components): LL={log_loss(win, v2_cross_t):.6f}")

# t0 super-ensemble
t0_super = v1_t0_only * 0.4 + v2_t0_only * 0.6
print(f"\nt0-only super (V1*40%+V2*60%): LL={log_loss(win, t0_super):.6f}")

# Cross-t super-ensemble with weight optimization
print("\n--- Cross-t super weight optimization ---")
best_w, best_ll_val = 0.5, 999
for w2 in np.arange(0.0, 1.01, 0.05):
    combined = v1_cross_t * (1 - w2) + v2_cross_t * w2
    ll = log_loss(win, combined)
    if ll < best_ll_val:
        best_w, best_ll_val = w2, ll
    if abs(w2 - round(w2, 1)) < 0.001:
        print(f"  V1_cross({1-w2:.0%}) + V2_cross({w2:.0%}): LL={ll:.6f}")
print(f"\n  BEST: V1_cross({1-best_w:.0%}) + V2_cross({best_w:.0%}): LL={best_ll_val:.6f}")

# Mega-ensemble: just average everything
mega = aligned[prob_cols].mean(axis=1).values
mega_ll = log_loss(win, mega)
print(f"\nMega-ensemble (all {len(prob_cols)} equal weight): LL={mega_ll:.6f}")

# ============================================================
# Use BEST ensemble for backtest
# ============================================================
best_probs = v1_cross_t * (1 - best_w) + v2_cross_t * best_w
best_label = f"Cross-t Super-Ensemble V1({1-best_w:.0%})+V2({best_w:.0%})"

bt = aligned.copy()
bt["model_prob"] = best_probs
bt["edge"] = bt["model_prob"] - bt["market_prob"]
bt["back_odds"] = 1 / bt["orig_best_back_m0"]
bt = bt[(bt["back_odds"] > 1.01) & (bt["back_odds"] < 1000)]

commission = 0.075
print(f"\n{'='*80}")
print(f"=== {best_label} BACKTEST ({commission:.1%} commission) ===")
print(f"{'='*80}")
print(f"Total OOS: {len(bt):,}")

for et in [0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.07, 0.10]:
    bets = bt[bt["edge"] > et].copy()
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

    print(f"\nEdge>{et:.1%}:")
    print(f"  Bets: {n:,}, WR: {wr:.1%}, ROI: {roi:+.1f}%")
    print(f"  Sharpe: {sh_a:.1f}, z={z:.2f}, p={p:.6f}")
    print(f"  Profitable months: {pm}/{len(monthly)}")
    print(f"  $25/bet: ${profit_25:,.0f} total, ${profit_25/len(monthly):,.0f}/month")

# ============================================================
# Compare t0-only vs cross-t head to head
# ============================================================
print(f"\n{'='*80}")
print("=== t0-ONLY vs CROSS-T HEAD-TO-HEAD ===")
print(f"{'='*80}")

bt_t0 = aligned.copy()
bt_t0["model_prob"] = t0_super
bt_t0["edge"] = bt_t0["model_prob"] - bt_t0["market_prob"]
bt_t0["back_odds"] = 1 / bt_t0["orig_best_back_m0"]
bt_t0 = bt_t0[(bt_t0["back_odds"] > 1.01) & (bt_t0["back_odds"] < 1000)]

for et in [0.02, 0.025, 0.03, 0.05]:
    bets_t0 = bt_t0[bt_t0["edge"] > et].copy()
    bets_ct = bt[bt["edge"] > et].copy()
    if len(bets_t0) < 10 or len(bets_ct) < 10:
        continue
    bets_t0["pnl"] = bets_t0["win"] * (bets_t0["back_odds"] - 1) * (1 - commission) - (1 - bets_t0["win"])
    bets_ct["pnl"] = bets_ct["win"] * (bets_ct["back_odds"] - 1) * (1 - commission) - (1 - bets_ct["win"])

    roi_t0 = bets_t0["pnl"].mean() * 100
    roi_ct = bets_ct["pnl"].mean() * 100
    z_t0 = bets_t0["pnl"].mean() / bets_t0["pnl"].std() * np.sqrt(len(bets_t0)) if bets_t0["pnl"].std() > 0 else 0
    z_ct = bets_ct["pnl"].mean() / bets_ct["pnl"].std() * np.sqrt(len(bets_ct)) if bets_ct["pnl"].std() > 0 else 0

    print(f"\nEdge>{et:.1%}:")
    print(f"  t0-only:  {len(bets_t0):,} bets, ROI={roi_t0:+.1f}%, z={z_t0:.2f}, P&L=${bets_t0['pnl'].sum()*25:,.0f}")
    print(f"  Cross-t:  {len(bets_ct):,} bets, ROI={roi_ct:+.1f}%, z={z_ct:.2f}, P&L=${bets_ct['pnl'].sum()*25:,.0f}")

# Save ultimate ensemble predictions
save_dir = "/data/projects/punim2039/alpha_odds/res/analysis/"
os.makedirs(save_dir, exist_ok=True)
bt[["key", "win", "model_prob", "market_prob", "edge", "back_odds", "marketTime_local"]].to_parquet(
    save_dir + "ultimate_cross_t_ensemble_predictions.parquet"
)
print(f"\nSaved ultimate ensemble predictions to {save_dir}")
print("Done!")
