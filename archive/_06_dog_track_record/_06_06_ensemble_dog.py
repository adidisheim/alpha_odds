"""
Cross-t super-ensemble for dog feature models.
Identical to _10_ultimate_ensemble.py except:
  - Loads from res/dog_features/win_model[_v2]/
  - Also loads ORIGINAL (no-dog) ensemble for head-to-head comparison
  - Saves to res/dog_features/analysis/

Usage: python _06_06_ensemble_dog.py
No arguments.
"""
import pandas as pd
import numpy as np
import os
from math import erf, sqrt


def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def load_ensemble(base_path, t_def, top_n):
    """Load models for a given t_def and return top-N ensemble probs + base df."""
    base_dir = f"{base_path}/t{t_def}"
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
    print(f"    Top-{n} configs: {sorted_configs[:n]}")
    return base, ens_prob


# ============================================================
# MAIN
# ============================================================
res_dir = "/data/projects/punim2039/alpha_odds/res"
dog_dir = f"{res_dir}/dog_features"

print("=" * 80)
print("=== DOG FEATURE CROSS-T SUPER-ENSEMBLE ===")
print("=" * 80)

# ── Load DOG FEATURE models ──
print("\nLoading DOG FEATURE models...", flush=True)
dog_components = {}

for t in [0, 1, 2, 3]:
    base, prob = load_ensemble(f"{dog_dir}/win_model", t, top_n=7)
    if base is not None:
        ll = log_loss(base["win"].values, prob)
        dog_components[("V1", t)] = (base, prob)
        print(f"  Dog V1 t{t}: LL={ll:.6f} ({len(base):,} rows)")

    base, prob = load_ensemble(f"{dog_dir}/win_model_v2", t, top_n=15)
    if base is not None:
        ll = log_loss(base["win"].values, prob)
        dog_components[("V2", t)] = (base, prob)
        print(f"  Dog V2 t{t}: LL={ll:.6f} ({len(base):,} rows)")

if not dog_components:
    print("ERROR: No dog feature models found. Exiting.")
    exit(1)

# ── Load ORIGINAL models for comparison ──
print("\nLoading ORIGINAL (no-dog) models...", flush=True)
orig_components = {}

for t in [0, 1, 2, 3]:
    base, prob = load_ensemble(f"{res_dir}/win_model", t, top_n=7)
    if base is not None:
        ll = log_loss(base["win"].values, prob)
        orig_components[("V1", t)] = (base, prob)
        print(f"  Orig V1 t{t}: LL={ll:.6f} ({len(base):,} rows)")

    base, prob = load_ensemble(f"{res_dir}/win_model_v2", t, top_n=15)
    if base is not None:
        ll = log_loss(base["win"].values, prob)
        orig_components[("V2", t)] = (base, prob)
        print(f"  Orig V2 t{t}: LL={ll:.6f} ({len(base):,} rows)")


def build_cross_t_ensemble(components, label):
    """Build cross-t super-ensemble from a set of components."""
    # Find a base component to start alignment
    first_key = list(components.keys())[0]
    base_df = components[first_key][0].copy()
    base_df["key"] = base_df["file_name"].astype(str) + "_" + base_df["id"].astype(str)
    first_col = f"{first_key[0]}_t{first_key[1]}"
    aligned = base_df[["key", "win", "market_prob", "orig_best_back_m0", "orig_best_lay_m0", "marketTime_local"]].copy()
    aligned[first_col] = components[first_key][1]

    for (ver, t), (df, prob) in components.items():
        col = f"{ver}_t{t}"
        if col == first_col:
            continue
        tmp = df.copy()
        tmp["key"] = tmp["file_name"].astype(str) + "_" + tmp["id"].astype(str)
        tmp[col] = prob
        aligned = aligned.merge(tmp[["key", col]], on="key", how="inner")

    prob_cols = [c for c in aligned.columns if c.startswith(("V1_", "V2_"))]
    print(f"\n  [{label}] Aligned: {len(aligned):,} rows, {len(prob_cols)} components: {prob_cols}")

    win = aligned["win"].values
    market_prob = aligned["market_prob"].values
    market_ll = log_loss(win, np.clip(market_prob, 0.001, 0.999))
    print(f"  Market LL: {market_ll:.6f}")

    v1_cols = [c for c in prob_cols if c.startswith("V1_")]
    v2_cols = [c for c in prob_cols if c.startswith("V2_")]

    v1_cross_t = aligned[v1_cols].mean(axis=1).values if v1_cols else None
    v2_cross_t = aligned[v2_cols].mean(axis=1).values if v2_cols else None

    if v1_cross_t is not None:
        print(f"  V1 cross-t ({len(v1_cols)}): LL={log_loss(win, v1_cross_t):.6f}")
    if v2_cross_t is not None:
        print(f"  V2 cross-t ({len(v2_cols)}): LL={log_loss(win, v2_cross_t):.6f}")

    # Optimize V1/V2 blend
    best_w, best_ll = 0.5, 999
    if v1_cross_t is not None and v2_cross_t is not None:
        for w2 in np.arange(0.0, 1.01, 0.05):
            combined = v1_cross_t * (1 - w2) + v2_cross_t * w2
            ll = log_loss(win, combined)
            if ll < best_ll:
                best_w, best_ll = w2, ll
        best_probs = v1_cross_t * (1 - best_w) + v2_cross_t * best_w
        print(f"  BEST: V1({1-best_w:.0%})+V2({best_w:.0%}): LL={best_ll:.6f}")
    elif v2_cross_t is not None:
        best_probs = v2_cross_t
        best_ll = log_loss(win, best_probs)
        best_w = 1.0
    else:
        best_probs = v1_cross_t
        best_ll = log_loss(win, best_probs)
        best_w = 0.0

    return aligned, best_probs, best_w, best_ll


# ============================================================
# Build ensembles for both
# ============================================================
print("\n" + "=" * 80)
print("BUILDING DOG FEATURE ENSEMBLE")
print("=" * 80)
dog_aligned, dog_probs, dog_w, dog_ll = build_cross_t_ensemble(dog_components, "Dog Features")

print("\n" + "=" * 80)
print("BUILDING ORIGINAL ENSEMBLE")
print("=" * 80)
orig_aligned, orig_probs, orig_w, orig_ll = build_cross_t_ensemble(orig_components, "Original")

# ============================================================
# HEAD-TO-HEAD COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("=" * 80)
print("       HEAD-TO-HEAD: ORIGINAL vs DOG FEATURES")
print("=" * 80)
print("=" * 80)

print(f"\nOriginal ensemble: V1({1-orig_w:.0%})+V2({orig_w:.0%}), LL={orig_ll:.6f}")
print(f"Dog feat ensemble: V1({1-dog_w:.0%})+V2({dog_w:.0%}), LL={dog_ll:.6f}")
ll_diff = orig_ll - dog_ll
ll_pct = 100 * ll_diff / orig_ll
if dog_ll < orig_ll:
    print(f"\n>>> DOG FEATURES HELP: log-loss improved by {ll_diff:.6f} ({ll_pct:+.3f}%)")
else:
    print(f"\n>>> DOG FEATURES DO NOT HELP: log-loss worsened by {-ll_diff:.6f} ({-ll_pct:+.3f}%)")

# ── Backtest comparison ──
commission = 0.075

print(f"\n{'='*80}")
print(f"=== BACKTEST COMPARISON ({commission:.1%} commission) ===")
print(f"{'='*80}")

for label, aligned, probs in [("ORIGINAL", orig_aligned, orig_probs), ("DOG FEATURES", dog_aligned, dog_probs)]:
    bt = aligned.copy()
    bt["model_prob"] = probs
    bt["edge"] = bt["model_prob"] - bt["market_prob"]
    bt["back_odds"] = 1 / bt["orig_best_back_m0"]
    bt = bt[(bt["back_odds"] > 1.01) & (bt["back_odds"] < 1000)]

    print(f"\n--- {label} ---")
    for et in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        bets = bt[bt["edge"] > et].copy()
        if len(bets) < 10:
            continue
        bets["pnl"] = bets["win"] * (bets["back_odds"] - 1) * (1 - commission) - (1 - bets["win"])
        n = len(bets)
        roi = bets["pnl"].mean() * 100
        z = bets["pnl"].mean() / bets["pnl"].std() * np.sqrt(n) if bets["pnl"].std() > 0 else 0
        p = 1 - norm_cdf(z)
        profit_25 = bets["pnl"].sum() * 25
        print(f"  Edge>{et:.0%}: {n:,} bets, ROI={roi:+.1f}%, z={z:.2f}, p={p:.6f}, $25/bet=${profit_25:,.0f}")

# ── Save ──
save_dir = f"{dog_dir}/analysis/"
os.makedirs(save_dir, exist_ok=True)

dog_aligned_save = dog_aligned.copy()
dog_aligned_save["model_prob"] = dog_probs
dog_aligned_save["edge"] = dog_aligned_save["model_prob"] - dog_aligned_save["market_prob"]
dog_aligned_save[["key", "win", "model_prob", "market_prob", "edge", "marketTime_local"]].to_parquet(
    save_dir + "dog_cross_t_ensemble_predictions.parquet"
)

# Save comparison summary
comparison = pd.DataFrame([{
    'model': 'original',
    'logloss': orig_ll,
    'v2_weight': orig_w,
}, {
    'model': 'dog_features',
    'logloss': dog_ll,
    'v2_weight': dog_w,
}])
comparison.to_parquet(save_dir + "ensemble_comparison.parquet")

print(f"\nSaved to {save_dir}")
print("Done!", flush=True)
