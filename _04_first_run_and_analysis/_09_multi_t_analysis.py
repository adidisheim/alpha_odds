"""
Multi-time-definition analysis.
Check if ensembling across t0/t1/t2/t3 improves over t0-only.
Also analyse V2 results for all available time definitions.
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
# PART 1: V1 multi-t analysis
# ============================================================
print("=" * 80)
print("=== V1 MULTI-T ANALYSIS ===")
print("=" * 80)

v1_ensembles = {}
for t_def in [0, 1, 2, 3]:
    base_dir = f"win_model/t{t_def}"
    if not os.path.exists(base_dir):
        continue
    configs = sorted(os.listdir(base_dir))
    dfs = {}
    for c in configs:
        path = os.path.join(base_dir, c, "save_df.parquet")
        if os.path.exists(path):
            try:
                dfs[c] = pd.read_parquet(path)
            except Exception:
                pass
    if not dfs:
        continue

    # Sort by log-loss
    sorted_configs = sorted(dfs.keys(), key=lambda c: log_loss(
        dfs[c]["win"].values, dfs[c]["model_prob"].clip(0.001, 0.999).values))

    base = dfs[sorted_configs[0]]
    # Top-7 ensemble
    top_n = min(7, len(sorted_configs))
    ens_prob = np.mean([dfs[c]["model_prob"].values for c in sorted_configs[:top_n]], axis=0)
    ll = log_loss(base["win"].values, ens_prob)
    v1_ensembles[t_def] = {"prob": ens_prob, "base": base, "ll": ll, "n_configs": len(dfs)}
    print(f"V1 t{t_def}: Top-{top_n} ensemble LL={ll:.6f} ({len(dfs)} configs available)")

# Cross-t V1 ensemble (align by file_name + id)
if len(v1_ensembles) > 1:
    print("\nCross-t V1 ensembles:")
    # Build aligned DataFrame
    base_t0 = v1_ensembles[0]["base"].copy()
    base_t0["key"] = base_t0["file_name"].astype(str) + "_" + base_t0["id"].astype(str)
    base_t0["v1_t0_prob"] = v1_ensembles[0]["prob"]
    aligned = base_t0[["key", "win", "v1_t0_prob"]].copy()
    for t in [1, 2, 3]:
        if t not in v1_ensembles:
            continue
        tdf = v1_ensembles[t]["base"].copy()
        tdf["key"] = tdf["file_name"].astype(str) + "_" + tdf["id"].astype(str)
        tdf[f"v1_t{t}_prob"] = v1_ensembles[t]["prob"]
        aligned = aligned.merge(tdf[["key", f"v1_t{t}_prob"]], on="key", how="inner")

    print(f"  Aligned rows: {len(aligned):,}")
    win = aligned["win"].values
    for combo in [(0, 1), (0, 1, 2), (0, 1, 2, 3)]:
        cols = [f"v1_t{t}_prob" for t in combo if f"v1_t{t}_prob" in aligned.columns]
        if len(cols) < 2:
            continue
        combined = aligned[cols].mean(axis=1).values
        ll = log_loss(win, combined)
        t_str = "+".join([f"t{t}" for t in combo if f"v1_t{t}_prob" in aligned.columns])
        print(f"  V1 {t_str}: LL={ll:.6f}")

# ============================================================
# PART 2: V2 multi-t analysis
# ============================================================
print("\n" + "=" * 80)
print("=== V2 MULTI-T ANALYSIS ===")
print("=" * 80)

v2_ensembles = {}
for t_def in [0, 1, 2, 3]:
    base_dir = f"win_model_v2/t{t_def}"
    if not os.path.exists(base_dir):
        continue
    configs = sorted(os.listdir(base_dir))
    dfs = {}
    for c in configs:
        path = os.path.join(base_dir, c, "save_df.parquet")
        if os.path.exists(path):
            try:
                dfs[c] = pd.read_parquet(path)
            except Exception:
                pass
    if not dfs:
        continue

    sorted_configs = sorted(dfs.keys(), key=lambda c: log_loss(
        dfs[c]["win"].values, dfs[c]["model_prob"].clip(0.001, 0.999).values))

    base = dfs[sorted_configs[0]]
    top_n = min(15, len(sorted_configs))
    ens_prob = np.mean([dfs[c]["model_prob"].values for c in sorted_configs[:top_n]], axis=0)
    ll = log_loss(base["win"].values, ens_prob)
    v2_ensembles[t_def] = {"prob": ens_prob, "base": base, "ll": ll, "n_configs": len(dfs)}
    print(f"V2 t{t_def}: Top-{top_n} ensemble LL={ll:.6f} ({len(dfs)} configs available)")

# Cross-t V2 ensemble (align by file_name + id)
if len(v2_ensembles) > 1:
    print("\nCross-t V2 ensembles:")
    base_t0 = v2_ensembles[0]["base"].copy()
    base_t0["key"] = base_t0["file_name"].astype(str) + "_" + base_t0["id"].astype(str)
    base_t0["v2_t0_prob"] = v2_ensembles[0]["prob"]
    aligned_v2 = base_t0[["key", "win", "v2_t0_prob"]].copy()
    for t in [1, 2, 3]:
        if t not in v2_ensembles:
            continue
        tdf = v2_ensembles[t]["base"].copy()
        tdf["key"] = tdf["file_name"].astype(str) + "_" + tdf["id"].astype(str)
        tdf[f"v2_t{t}_prob"] = v2_ensembles[t]["prob"]
        aligned_v2 = aligned_v2.merge(tdf[["key", f"v2_t{t}_prob"]], on="key", how="inner")

    print(f"  Aligned rows: {len(aligned_v2):,}")
    win_v2 = aligned_v2["win"].values
    for combo in [(0, 1), (0, 1, 2), (0, 1, 2, 3)]:
        cols = [f"v2_t{t}_prob" for t in combo if f"v2_t{t}_prob" in aligned_v2.columns]
        if len(cols) < 2:
            continue
        combined = aligned_v2[cols].mean(axis=1).values
        ll = log_loss(win_v2, combined)
        t_str = "+".join([f"t{t}" for t in combo if f"v2_t{t}_prob" in aligned_v2.columns])
        print(f"  V2 {t_str}: LL={ll:.6f}")

# ============================================================
# PART 3: Best possible super-ensemble
# ============================================================
print("\n" + "=" * 80)
print("=== BEST SUPER-ENSEMBLE SEARCH ===")
print("=" * 80)

if 0 in v1_ensembles and 0 in v2_ensembles:
    base_t0 = v1_ensembles[0]["base"]
    win = base_t0["win"].values
    v1_t0 = v1_ensembles[0]["prob"]
    v2_t0 = v2_ensembles[0]["prob"]

    # V1_t0 + V2_t0 weight optimization (same length, no alignment needed)
    print("\nWeight optimization for V1_t0 + V2_t0:")
    best_w, best_ll_val = 0.5, 999
    for w2 in np.arange(0.0, 1.01, 0.05):
        combined = v1_t0 * (1 - w2) + v2_t0 * w2
        ll = log_loss(win, combined)
        if ll < best_ll_val:
            best_w, best_ll_val = w2, ll
        if abs(w2 - round(w2, 1)) < 0.001:
            print(f"  V1({1-w2:.0%}) + V2({w2:.0%}): LL={ll:.6f}")
    print(f"\n  BEST: V1({1-best_w:.0%}) + V2({best_w:.0%}): LL={best_ll_val:.6f}")

    # Per-t V1+V2 blend comparison
    print("\nPer-t super-ensemble (V1+V2 at optimal weight):")
    for t in sorted(set(v1_ensembles.keys()) & set(v2_ensembles.keys())):
        if v1_ensembles[t]["prob"].shape != v2_ensembles[t]["prob"].shape:
            print(f"  t{t}: SKIPPED (shape mismatch {v1_ensembles[t]['prob'].shape} vs {v2_ensembles[t]['prob'].shape})")
            continue
        combined = v1_ensembles[t]["prob"] * (1 - best_w) + v2_ensembles[t]["prob"] * best_w
        ll = log_loss(v1_ensembles[t]["base"]["win"].values, combined)
        print(f"  t{t}: V1({1-best_w:.0%})+V2({best_w:.0%}) LL={ll:.6f} (V1={v1_ensembles[t]['ll']:.6f}, V2={v2_ensembles[t]['ll']:.6f})")

    # ============================================================
    # PART 4: Backtest the best super-ensemble
    # ============================================================
    print("\n" + "=" * 80)
    print("=== BEST SUPER-ENSEMBLE BACKTEST ===")
    print("=" * 80)

    best_probs = v1_t0 * (1 - best_w) + v2_t0 * best_w
    bt = base_t0.copy()
    bt["model_prob"] = best_probs
    bt["edge"] = bt["model_prob"] - bt["market_prob"]
    bt["back_odds"] = 1 / bt["orig_best_back_m0"]
    bt = bt[(bt["back_odds"] > 1.01) & (bt["back_odds"] < 1000)]

    commission = 0.075
    print(f"Commission: {commission:.1%}")
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
# PART 5: Odds-range analysis
# ============================================================
print("\n" + "=" * 80)
print("=== PERFORMANCE BY ODDS RANGE ===")
print("=" * 80)

if 0 in v1_ensembles and 0 in v2_ensembles:
    bt_analysis = bt.copy()
    bt_analysis["pnl"] = bt_analysis["win"] * (bt_analysis["back_odds"] - 1) * (1 - commission) - (1 - bt_analysis["win"])
    bt_analysis["odds_range"] = pd.cut(bt_analysis["back_odds"],
                                        bins=[1, 3, 5, 8, 15, 30, 1000],
                                        labels=["1-3", "3-5", "5-8", "8-15", "15-30", "30+"])

    for edge_min in [0, 0.025]:
        subset = bt_analysis[bt_analysis["edge"] > edge_min]
        print(f"\nEdge > {edge_min:.1%}:")
        odds_stats = subset.groupby("odds_range", observed=True).agg(
            count=("pnl", "count"),
            win_rate=("win", "mean"),
            avg_edge=("edge", "mean"),
            roi=("pnl", "mean"),
            total_pnl=("pnl", "sum"),
        ).reset_index()
        odds_stats["roi"] = odds_stats["roi"] * 100
        odds_stats["profit_25"] = odds_stats["total_pnl"] * 25
        print(odds_stats.to_string(index=False))

print("\nDone!")
