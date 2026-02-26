"""
V2 Model Analysis - runs on Spartan.
Compares v2 (cross-runner + LightGBM ensemble) with v1 baseline.
Builds v2 ensembles and runs comprehensive backtests.
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
# PART 1: Load all available V2 models
# ============================================================
print("=" * 80)
print("=== V2 MODEL COMPARISON ===")
print("=" * 80)

v2_results = []
v2_dfs = {}
for t_def in [0, 1, 2, 3]:
    base_dir = f"win_model_v2/t{t_def}"
    if not os.path.exists(base_dir):
        continue
    for config_dir in sorted(os.listdir(base_dir)):
        save_path = os.path.join(base_dir, config_dir, "save_df.parquet")
        if not os.path.exists(save_path):
            continue
        try:
            df = pd.read_parquet(save_path)
            # Use model_prob (which is the ensemble of xgb+lgbm+calibrated)
            model_ll = log_loss(df["win"].values, df["model_prob"].clip(0.001, 0.999).values)
            market_ll = log_loss(df["win"].values, df["market_prob"].clip(0.001, 0.999).values)
            v2_results.append({
                "t_def": t_def,
                "config": config_dir,
                "ensemble_ll": round(model_ll, 6),
                "market_ll": round(market_ll, 6),
                "improvement": round(market_ll - model_ll, 6),
                "n_obs": len(df),
            })
            v2_dfs[(t_def, config_dir)] = df
        except Exception as e:
            print(f"Error loading {config_dir}: {e}")

if not v2_results:
    print("No V2 results available yet. Exiting.")
    exit(0)

df_v2 = pd.DataFrame(v2_results)
print(df_v2.sort_values("ensemble_ll").to_string(index=False))

# ============================================================
# PART 2: Build V2 multi-model ensemble
# ============================================================
print("\n" + "=" * 80)
print("=== V2 MULTI-MODEL ENSEMBLE ===")
print("=" * 80)

# Take top configs for t0
t0_configs = df_v2[df_v2["t_def"] == 0].sort_values("ensemble_ll")
if len(t0_configs) > 0:
    print(f"\nAvailable t0 configs: {len(t0_configs)}")

    # Build ensembles of increasing size
    sorted_configs = t0_configs["config"].tolist()
    base = v2_dfs[(0, sorted_configs[0])].copy()

    for n in range(2, min(len(sorted_configs) + 1, 8)):
        configs_to_use = sorted_configs[:n]
        probs = np.mean([v2_dfs[(0, c)]["model_prob"].values for c in configs_to_use], axis=0)
        ll = log_loss(base["win"].values, probs)
        print(f"Top-{n} V2 ensemble: LL={ll:.6f} (configs: {', '.join(configs_to_use)})")

    # Use the best ensemble
    best_n = min(len(sorted_configs), 5)
    best_configs = sorted_configs[:best_n]
    ensemble_prob = np.mean([v2_dfs[(0, c)]["model_prob"].values for c in best_configs], axis=0)

    base["model_prob"] = ensemble_prob
    base["edge"] = base["model_prob"] - base["market_prob"]

    print(f"\nUsing top-{best_n} V2 ensemble")
    print(f"Ensemble LL: {log_loss(base['win'].values, ensemble_prob):.6f}")

    # ============================================================
    # PART 3: V2 Ensemble Backtest
    # ============================================================
    print("\n" + "=" * 80)
    print(f"=== V2 TOP-{best_n} ENSEMBLE BACKTEST ===")
    print("=" * 80)

    # Commission from original values
    if "orig_marketBaseRate" in base.columns:
        commission_rate = base["orig_marketBaseRate"].median() / 100
    else:
        commission_rate = 0.08  # 8% default
    print(f"Commission rate: {commission_rate:.4f}")

    base["back_odds"] = 1 / base["orig_best_back_m0"]
    base = base[(base["back_odds"] > 1.01) & (base["back_odds"] < 1000)]

    print(f"Total OOS: {len(base):,}")

    for edge_threshold in [0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.07, 0.10, 0.15]:
        bets = base[base["edge"] > edge_threshold].copy()
        if len(bets) == 0:
            continue

        bets["pnl"] = bets["win"] * (bets["back_odds"] - 1) * (1 - commission_rate) - (1 - bets["win"])

        n_bets = len(bets)
        total_pnl = bets["pnl"].sum()
        avg_pnl = bets["pnl"].mean()
        win_rate = bets["win"].mean()
        avg_odds = bets["back_odds"].mean()
        roi = avg_pnl * 100

        pnl_std = bets["pnl"].std()
        z_stat = avg_pnl / pnl_std * np.sqrt(n_bets) if pnl_std > 0 else 0
        p_value = 1 - norm_cdf(z_stat)

        bets["month"] = pd.to_datetime(bets["marketTime_local"]).dt.to_period("M")
        monthly = bets.groupby("month")["pnl"].sum()
        cum_pnl = monthly.cumsum()
        max_dd = (cum_pnl - cum_pnl.cummax()).min()
        monthly_returns = monthly.values
        sharpe_monthly = monthly_returns.mean() / monthly_returns.std() if monthly_returns.std() > 0 else 0
        sharpe_annual = sharpe_monthly * np.sqrt(12)

        profitable_months = int((monthly > 0).sum())
        total_profit_25 = total_pnl * 25
        monthly_profit_25 = total_profit_25 / len(monthly)

        print(f"\nEdge > {edge_threshold:.1%}:")
        print(f"  Bets: {n_bets:,}, Win rate: {win_rate:.1%}, Avg odds: {avg_odds:.1f}")
        print(f"  ROI: {roi:+.2f}%, Sharpe: {sharpe_annual:.2f}, z={z_stat:.3f}, p={p_value:.6f}")
        print(f"  Max monthly DD: ${max_dd*25:.0f}, Profitable months: {profitable_months}/{len(monthly)}")
        print(f"  $25/bet: Profit ${total_profit_25:,.0f}, ${monthly_profit_25:,.0f}/month")

    # ============================================================
    # PART 4: Compare V1 vs V2
    # ============================================================
    print("\n" + "=" * 80)
    print("=== V1 vs V2 COMPARISON ===")
    print("=" * 80)

    # Load V1 best models
    v1_top_configs = ["ne1000_md6_lr0.01", "ne100_md6_lr0.1", "ne500_md3_lr0.05",
                      "ne500_md6_lr0.01", "ne100_md6_lr0.05", "ne1000_md3_lr0.05",
                      "ne500_md6_lr0.05"]

    v1_dfs = {}
    for config in v1_top_configs:
        path = f"win_model/t0/{config}/save_df.parquet"
        if os.path.exists(path):
            v1_dfs[config] = pd.read_parquet(path)

    if v1_dfs:
        v1_base = v1_dfs[v1_top_configs[0]].copy()
        v1_ensemble = np.mean([v1_dfs[c]["model_prob"].values for c in v1_top_configs if c in v1_dfs], axis=0)
        v1_ll = log_loss(v1_base["win"].values, v1_ensemble)
        v2_ll = log_loss(base["win"].values, ensemble_prob)

        print(f"V1 Top-7 Ensemble LL: {v1_ll:.6f}")
        print(f"V2 Top-{best_n} Ensemble LL: {v2_ll:.6f}")
        print(f"Improvement: {v1_ll - v2_ll:.6f} ({(v1_ll - v2_ll)/v1_ll * 100:.3f}%)")

        # Cross-model ensemble: V1 + V2
        # Align observations
        v1_probs = v1_ensemble
        v2_probs = ensemble_prob
        combined_probs = (v1_probs + v2_probs) / 2
        combined_ll = log_loss(v1_base["win"].values, combined_probs)
        print(f"\nV1+V2 Combined Ensemble LL: {combined_ll:.6f}")

    # ============================================================
    # PART 5: V2 Feature importance summary
    # ============================================================
    print("\n" + "=" * 80)
    print("=== V2 CROSS-RUNNER FEATURE USAGE ===")
    print("=" * 80)

    cross_runner_features = [
        "prob_share", "prob_rank", "prob_deviation", "prob_vs_favorite",
        "spread_rank", "volume_rank", "race_herfindahl"
    ]

    for config in sorted_configs[:3]:
        fi_path = f"win_model_v2/t0/{config}/feature_importances.parquet"
        if os.path.exists(fi_path):
            fi = pd.read_parquet(fi_path)
            total_imp = fi["importance"].sum()
            cr_imp = fi[fi["feature"].isin(cross_runner_features)]["importance"].sum()
            print(f"\n{config}:")
            print(f"  Cross-runner feature share: {cr_imp/total_imp*100:.1f}%")
            top10 = fi.head(10)
            for _, row in top10.iterrows():
                marker = " *CR*" if row["feature"] in cross_runner_features else ""
                print(f"  {row['feature']:35s} {row['importance']:.4f}{marker}")

print("\nDone!")
