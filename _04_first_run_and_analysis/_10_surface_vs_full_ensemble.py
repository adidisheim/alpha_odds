"""
Compare surface-only (no q_100/q_1000) vs full model cross-t super-ensemble.
Run after all _03_win_probability_model_v2_surface.py jobs complete.

Usage: python _10_surface_vs_full_ensemble.py
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
    print(f"  {model_dir} t{t_def}: {n}/{len(dfs)} configs, LL={log_loss(base['win'].values, ens_prob):.6f}")
    return base, ens_prob


def build_cross_t_super(v1_dir, v2_dir, label):
    """Build a cross-t super-ensemble and return aligned DataFrame with model_prob."""
    components = {}
    for t in [0, 1, 2, 3]:
        base, prob = load_ensemble(v1_dir, t, top_n=7)
        if base is not None:
            components[("V1", t)] = (base, prob)
        base, prob = load_ensemble(v2_dir, t, top_n=15)
        if base is not None:
            components[("V2", t)] = (base, prob)

    if not components:
        return None

    # Align by key
    first_key = list(components.keys())[0]
    base_df = components[first_key][0].copy()
    base_df["key"] = base_df["file_name"].astype(str) + "_" + base_df["id"].astype(str)

    save_cols = ["key", "win", "market_prob", "orig_best_back_m0", "orig_best_lay_m0", "marketTime_local"]
    save_cols = [c for c in save_cols if c in base_df.columns]
    aligned = base_df[save_cols].copy()

    first_col = f"{first_key[0]}_t{first_key[1]}"
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
    v1_cols = [c for c in prob_cols if c.startswith("V1_")]
    v2_cols = [c for c in prob_cols if c.startswith("V2_")]

    if not v1_cols or not v2_cols:
        print(f"  Warning: missing V1 or V2 for {label}")
        aligned["model_prob"] = aligned[prob_cols].mean(axis=1)
        return aligned

    v1_cross = aligned[v1_cols].mean(axis=1).values
    v2_cross = aligned[v2_cols].mean(axis=1).values

    # Optimize V1/V2 weight
    best_w, best_ll = 0.5, 999
    win = aligned["win"].values
    for w2 in np.arange(0.0, 1.01, 0.05):
        combined = v1_cross * (1 - w2) + v2_cross * w2
        ll = log_loss(win, combined)
        if ll < best_ll:
            best_w, best_ll = w2, ll

    aligned["model_prob"] = v1_cross * (1 - best_w) + v2_cross * best_w
    print(f"\n  {label}: {len(aligned):,} rows, {len(prob_cols)} components")
    print(f"  Best weight: V1({1-best_w:.0%}) + V2({best_w:.0%}), LL={best_ll:.6f}")
    return aligned


def run_backtest(df, label, commission=0.075):
    """Run backtest and print results for various edge thresholds."""
    bt = df.copy()
    bt["edge"] = bt["model_prob"] - bt["market_prob"]
    bt["back_odds"] = 1 / bt["orig_best_back_m0"]
    bt = bt[(bt["back_odds"] > 1.01) & (bt["back_odds"] < 1000)]

    results = []
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
        profit_25 = bets["pnl"].sum() * 25
        avg_odds = bets["back_odds"].mean()

        bets["month"] = pd.to_datetime(bets["marketTime_local"]).dt.to_period("M")
        monthly = bets.groupby("month")["pnl"].sum()
        pm = int((monthly > 0).sum())
        sh_m = monthly.mean() / monthly.std() if monthly.std() > 0 else 0
        sh_a = sh_m * np.sqrt(12)

        results.append({
            "edge_thresh": et, "n_bets": n, "win_rate": wr, "avg_odds": avg_odds,
            "roi_pct": roi, "sharpe": sh_a, "z": z, "p": p,
            "profit_25": profit_25, "profit_months": pm, "total_months": len(monthly),
        })

    return pd.DataFrame(results)


# ============================================================
# Build all ensembles
# ============================================================
print("=" * 80)
print("ALL VARIANTS COMPARISON")
print("=" * 80)

variants = {}

print("\n--- Full model (105 features, V1 + V2) ---")
variants["Full (105)"] = build_cross_t_super("win_model", "win_model_v2", "Full (105 feat)")

print("\n--- Surface model (no q_100/q_1000, V1_full + V2_surface) ---")
variants["Surface (85)"] = build_cross_t_super("win_model", "win_model_v2_surface", "Surface (85 feat)")

# Pure surface: BOTH V1 and V2 retrained without q_100/q_1000
if os.path.exists("win_model_surface"):
    print("\n--- Pure Surface (V1_surface + V2_surface) ---")
    variants["Pure Surf"] = build_cross_t_super("win_model_surface", "win_model_v2_surface", "Pure Surface")
else:
    print("\n--- Pure Surface: win_model_surface/ NOT FOUND (skipping) ---")

for nf in [5, 10, 30]:
    v2_dir = f"win_model_v2_surface_n{nf}"
    if os.path.exists(v2_dir):
        print(f"\n--- Surface top-{nf} (V1 + V2_surface_n{nf}) ---")
        variants[f"Surf n{nf}"] = build_cross_t_super("win_model", v2_dir, f"Surface n{nf}")
    else:
        print(f"\n--- Surface top-{nf}: NOT FOUND (skipping) ---")

# Remove None entries
variants = {k: v for k, v in variants.items() if v is not None}

if len(variants) < 2:
    print("ERROR: Need at least 2 variants to compare")
    exit(1)

# ============================================================
# Compare log-loss
# ============================================================
ref_key = list(variants.keys())[0]
mkt_ll = log_loss(variants[ref_key]["win"].values,
                  np.clip(variants[ref_key]["market_prob"].values, 0.001, 0.999))

print(f"\n{'='*80}")
print(f"LOG-LOSS COMPARISON")
print(f"{'='*80}")
print(f"  {'Market':<20s}  LL={mkt_ll:.6f}")
for name, df in variants.items():
    ll = log_loss(df["win"].values, df["model_prob"].values)
    print(f"  {name:<20s}  LL={ll:.6f}  (gain: {mkt_ll - ll:.6f})")

# ============================================================
# Backtest comparison
# ============================================================
print(f"\n{'='*80}")
print(f"BACKTEST COMPARISON (7.5% commission)")
print(f"{'='*80}")

all_bt = {}
for name, df in variants.items():
    all_bt[name] = run_backtest(df, name)

# Print header
col_names = list(variants.keys())
header = f"{'Edge':>6s}"
for cn in col_names:
    header += f" | {cn:^25s}"
print(f"\n{header}")

sub_header = f"{'':>6s}"
for cn in col_names:
    sub_header += f" | {'Bets':>5s} {'ROI%':>6s} {'z':>5s} {'$25':>7s}"
print(sub_header)
print("-" * (7 + 28 * len(col_names)))

# Get all edge thresholds from the first variant
ref_bt = all_bt[col_names[0]]
for _, ref_row in ref_bt.iterrows():
    et = ref_row["edge_thresh"]
    line = f"{et:>5.1%}"
    for cn in col_names:
        bt_df = all_bt[cn]
        row = bt_df[bt_df["edge_thresh"] == et]
        if len(row) == 0:
            line += f" | {'---':>5s} {'---':>6s} {'---':>5s} {'---':>7s}"
        else:
            r = row.iloc[0]
            line += f" | {r['n_bets']:>5.0f} {r['roi_pct']:>+5.1f}% {r['z']:>5.2f} {r['profit_25']:>+7,.0f}"
    print(line)

# ============================================================
# Save results
# ============================================================
save_dir = "/data/projects/punim2039/alpha_odds/res/analysis/"
os.makedirs(save_dir, exist_ok=True)

# Save all variant backtest results in one parquet
all_results = []
for name, bt_df in all_bt.items():
    bt_df = bt_df.copy()
    bt_df["variant"] = name
    all_results.append(bt_df)
pd.concat(all_results, ignore_index=True).to_parquet(save_dir + "surface_all_variants_comparison.parquet")

# Save each variant's predictions
for name, df in variants.items():
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    cols_to_save = ["key", "win", "model_prob", "market_prob", "orig_best_back_m0", "orig_best_lay_m0", "marketTime_local"]
    cols_to_save = [c for c in cols_to_save if c in df.columns]
    df[cols_to_save].to_parquet(save_dir + f"{safe_name}_predictions.parquet")

print(f"\nSaved all results to {save_dir}")
print("Done!")
