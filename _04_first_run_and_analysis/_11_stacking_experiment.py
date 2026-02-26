"""
Stacking Meta-Model Experiment.

Instead of simple weighted blending (V1_cross*20% + V2_cross*80%), train a
second-level model that learns optimal weighting of the 8 base components
(V1 t0-t3, V2 t0-t3).

Temporal split within the 2025 OOS data:
  - Meta-train: Jan-Jul 2025  (7 months)
  - Meta-val:   Aug 2025      (1 month, for hyperparameter sanity)
  - Meta-test:  Sep-Oct 2025  (2 months, final evaluation)

All 8 base models were trained on 2017-2024, so 2025 predictions are fully
out-of-sample from the base models' perspective. The stacking layer is trained
only on the first portion of 2025 and tested on the last portion.

Meta-models:
  1. Logistic Regression (simple, low-variance stacking baseline)
  2. Small XGBoost (max_depth=2, n_estimators=50, conservative)

Comparison baselines:
  - Simple average of all 8 (mega-ensemble)
  - V1_cross(20%) + V2_cross(80%) weighted blend
  - Market implied probability

Usage on Spartan:
    srun --partition=interactive --time=00:30:00 --mem=32G --cpus-per-task=4 \
        bash -c 'source load_module.sh && python3 _11_stacking_experiment.py'
"""

import pandas as pd
import numpy as np
import os
from math import erf, sqrt

os.chdir("/data/projects/punim2039/alpha_odds/res")


# ============================================================
# Utility functions
# ============================================================

def log_loss(y_true, y_pred):
    """Binary log-loss (cross-entropy)."""
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def norm_cdf(x):
    """Normal CDF without scipy (using math.erf)."""
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


def backtest_strategy(df, prob_col, edge_thresholds, commission=0.075, label=""):
    """Run value-betting backtest on a DataFrame. Returns summary string."""
    bt = df.copy()
    bt["model_prob"] = prob_col if isinstance(prob_col, str) else prob_col
    if isinstance(prob_col, str):
        bt["model_prob"] = bt[prob_col]
    bt["edge"] = bt["model_prob"] - bt["market_prob"]
    bt["back_odds"] = 1 / bt["orig_best_back_m0"]
    bt = bt[(bt["back_odds"] > 1.01) & (bt["back_odds"] < 1000)]

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  {label} BACKTEST ({commission:.1%} commission)")
    lines.append(f"{'='*70}")
    lines.append(f"  Total rows: {len(bt):,}")

    results = []
    for et in edge_thresholds:
        bets = bt[bt["edge"] > et].copy()
        if len(bets) < 10:
            lines.append(f"  Edge>{et:.1%}: <10 bets, skipped")
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

        lines.append(f"  Edge>{et:.1%}: {n:,} bets, WR={wr:.1%}, ROI={roi:+.1f}%, "
                      f"z={z:.2f}, p={p:.4f}, Sharpe(ann)={sh_a:.1f}, "
                      f"${25}/bet=${profit_25:,.0f}")
        results.append({
            "edge_threshold": et, "n_bets": n, "win_rate": wr, "roi": roi,
            "z_score": z, "p_value": p, "sharpe_annual": sh_a, "profit_25": profit_25
        })

    return "\n".join(lines), results


# ============================================================
# STEP 1: Load all 8 model components
# ============================================================
print("=" * 70)
print("STACKING META-MODEL EXPERIMENT")
print("=" * 70)
print("\nStep 1: Loading all 8 model components...", flush=True)

all_components = {}

for t in [0, 1, 2, 3]:
    base, prob = load_ensemble("win_model", t, top_n=7)
    if base is not None:
        ll = log_loss(base["win"].values, prob)
        all_components[("V1", t)] = (base, prob)
        print(f"  V1 t{t}: LL={ll:.6f} ({len(base):,} rows)", flush=True)
    else:
        print(f"  V1 t{t}: NOT FOUND", flush=True)

    base, prob = load_ensemble("win_model_v2", t, top_n=15)
    if base is not None:
        ll = log_loss(base["win"].values, prob)
        all_components[("V2", t)] = (base, prob)
        print(f"  V2 t{t}: LL={ll:.6f} ({len(base):,} rows)", flush=True)
    else:
        print(f"  V2 t{t}: NOT FOUND", flush=True)

print(f"\nLoaded {len(all_components)} components", flush=True)

if len(all_components) < 2:
    print("ERROR: Need at least 2 components for stacking. Exiting.", flush=True)
    exit(1)


# ============================================================
# STEP 2: Align all DataFrames by (file_name, id)
# ============================================================
print("\nStep 2: Aligning DataFrames by (file_name, id)...", flush=True)

# Start from V1 t0 as base
base_key = ("V1", 0)
if base_key not in all_components:
    # fallback to first available
    base_key = list(all_components.keys())[0]

base_df = all_components[base_key][0].copy()
base_df["key"] = base_df["file_name"].astype(str) + "_" + base_df["id"].astype(str)

aligned = base_df[["key", "win", "market_prob", "orig_best_back_m0", "marketTime_local"]].copy()
col_name = f"{base_key[0]}_t{base_key[1]}"
aligned[col_name] = all_components[base_key][1]

for (ver, t), (df, prob) in all_components.items():
    col = f"{ver}_t{t}"
    if col == col_name:
        continue
    tmp = df.copy()
    tmp["key"] = tmp["file_name"].astype(str) + "_" + tmp["id"].astype(str)
    tmp[col] = prob
    aligned = aligned.merge(tmp[["key", col]], on="key", how="inner")

prob_cols = sorted([c for c in aligned.columns if c.startswith(("V1_", "V2_"))])
v1_cols = sorted([c for c in prob_cols if c.startswith("V1_")])
v2_cols = sorted([c for c in prob_cols if c.startswith("V2_")])

print(f"  Aligned: {len(aligned):,} rows", flush=True)
print(f"  Components: {prob_cols}", flush=True)
print(f"  V1 cols: {v1_cols}", flush=True)
print(f"  V2 cols: {v2_cols}", flush=True)


# ============================================================
# STEP 3: Temporal split within 2025
# ============================================================
print("\nStep 3: Temporal split...", flush=True)

aligned["year"] = pd.to_datetime(aligned["marketTime_local"]).dt.year
aligned["month"] = pd.to_datetime(aligned["marketTime_local"]).dt.month

# Meta-train: Jan-Jul 2025
# Meta-val:   Aug 2025
# Meta-test:  Sep-Oct 2025
ind_train = (aligned["year"] == 2025) & (aligned["month"] <= 7)
ind_val   = (aligned["year"] == 2025) & (aligned["month"] == 8)
ind_test  = (aligned["year"] == 2025) & (aligned["month"].isin([9, 10]))

print(f"  Meta-train (Jan-Jul): {ind_train.sum():,} rows", flush=True)
print(f"  Meta-val   (Aug):     {ind_val.sum():,} rows", flush=True)
print(f"  Meta-test  (Sep-Oct): {ind_test.sum():,} rows", flush=True)

df_train = aligned[ind_train].copy()
df_val   = aligned[ind_val].copy()
df_test  = aligned[ind_test].copy()


# ============================================================
# STEP 4: Build meta-features
# ============================================================
print("\nStep 4: Building meta-features...", flush=True)

def build_meta_features(df, prob_cols, v1_cols, v2_cols):
    """Create meta-feature matrix for stacking."""
    X = df[prob_cols].copy()

    # Market probability
    X["market_prob"] = df["market_prob"].values

    # Back odds (1 / implied prob)
    X["back_odds"] = (1 / df["orig_best_back_m0"]).values
    X["back_odds"] = X["back_odds"].clip(1.01, 1000)

    # Derived features
    if len(v1_cols) > 0:
        X["v1_avg"] = df[v1_cols].mean(axis=1).values
    if len(v2_cols) > 0:
        X["v2_avg"] = df[v2_cols].mean(axis=1).values
    if len(v1_cols) > 0 and len(v2_cols) > 0:
        X["v1_v2_disagreement"] = (X["v1_avg"] - X["v2_avg"]).abs()
    X["max_component"] = df[prob_cols].max(axis=1).values
    X["min_component"] = df[prob_cols].min(axis=1).values
    X["range_component"] = X["max_component"] - X["min_component"]

    # Replace any inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X


meta_cols_list = []  # will be set after first call

X_train = build_meta_features(df_train, prob_cols, v1_cols, v2_cols)
X_val   = build_meta_features(df_val,   prob_cols, v1_cols, v2_cols)
X_test  = build_meta_features(df_test,  prob_cols, v1_cols, v2_cols)

meta_feature_names = list(X_train.columns)
print(f"  Meta-features ({len(meta_feature_names)}): {meta_feature_names}", flush=True)

y_train = df_train["win"].values
y_val   = df_val["win"].values
y_test  = df_test["win"].values


# ============================================================
# STEP 5: Build baseline predictions for comparison
# ============================================================
print("\nStep 5: Computing baselines...", flush=True)

# Baseline 1: Simple average of all 8 (mega-ensemble)
mega_test = df_test[prob_cols].mean(axis=1).values
mega_val  = df_val[prob_cols].mean(axis=1).values

# Baseline 2: V1_cross(20%) + V2_cross(80%) weighted blend
if len(v1_cols) > 0 and len(v2_cols) > 0:
    v1_cross_test = df_test[v1_cols].mean(axis=1).values
    v2_cross_test = df_test[v2_cols].mean(axis=1).values
    weighted_test = 0.20 * v1_cross_test + 0.80 * v2_cross_test

    v1_cross_val = df_val[v1_cols].mean(axis=1).values
    v2_cross_val = df_val[v2_cols].mean(axis=1).values
    weighted_val = 0.20 * v1_cross_val + 0.80 * v2_cross_val
else:
    weighted_test = mega_test
    weighted_val = mega_val

# Baseline 3: Market probability
market_test = df_test["market_prob"].values
market_val  = df_val["market_prob"].values

print(f"  Market LL (val):    {log_loss(y_val, np.clip(market_val, 0.001, 0.999)):.6f}", flush=True)
print(f"  Mega-ens LL (val):  {log_loss(y_val, mega_val):.6f}", flush=True)
print(f"  Weighted LL (val):  {log_loss(y_val, weighted_val):.6f}", flush=True)


# ============================================================
# STEP 6: Train Logistic Regression meta-model
# ============================================================
print("\nStep 6: Training Logistic Regression meta-model...", flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale meta-features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)

lr_pred_val  = lr_model.predict_proba(X_val_scaled)[:, 1]
lr_pred_test = lr_model.predict_proba(X_test_scaled)[:, 1]

print(f"  LR val LL:  {log_loss(y_val, lr_pred_val):.6f}", flush=True)
print(f"  LR test LL: {log_loss(y_test, lr_pred_test):.6f}", flush=True)

# Print LR coefficients for interpretability
print("\n  Logistic Regression coefficients:", flush=True)
for name, coef in sorted(zip(meta_feature_names, lr_model.coef_[0]),
                          key=lambda x: abs(x[1]), reverse=True):
    print(f"    {name:30s}: {coef:+.4f}", flush=True)
print(f"    {'intercept':30s}: {lr_model.intercept_[0]:+.4f}", flush=True)


# ============================================================
# STEP 7: Train small XGBoost meta-model
# ============================================================
print("\nStep 7: Training XGBoost meta-model (conservative)...", flush=True)

from xgboost import XGBClassifier

xgb_meta = XGBClassifier(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    eval_metric='logloss',
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,   # L1 regularization
    reg_lambda=5.0,  # L2 regularization
)
xgb_meta.fit(X_train, y_train)

xgb_pred_val  = xgb_meta.predict_proba(X_val)[:, 1]
xgb_pred_test = xgb_meta.predict_proba(X_test)[:, 1]

print(f"  XGB val LL:  {log_loss(y_val, xgb_pred_val):.6f}", flush=True)
print(f"  XGB test LL: {log_loss(y_test, xgb_pred_test):.6f}", flush=True)

# Print XGB feature importances
print("\n  XGBoost meta-model feature importances:", flush=True)
xgb_importances = xgb_meta.feature_importances_
for name, imp in sorted(zip(meta_feature_names, xgb_importances),
                         key=lambda x: x[1], reverse=True):
    if imp > 0.01:
        print(f"    {name:30s}: {imp:.4f}", flush=True)


# ============================================================
# STEP 8: Full comparison table
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: FULL COMPARISON — LOG-LOSS")
print("=" * 70)

# Validation set
print("\n--- VALIDATION SET (Aug 2025) ---")
val_results = {
    "Market":               log_loss(y_val, np.clip(market_val, 0.001, 0.999)),
    "Mega-ensemble (avg)":  log_loss(y_val, mega_val),
    "Weighted (V1*20+V2*80)": log_loss(y_val, weighted_val),
    "Stacking: LogReg":     log_loss(y_val, lr_pred_val),
    "Stacking: XGBoost":    log_loss(y_val, xgb_pred_val),
}
for name, ll in sorted(val_results.items(), key=lambda x: x[1]):
    print(f"  {name:30s}: {ll:.6f}", flush=True)

# Test set
print("\n--- TEST SET (Sep-Oct 2025) ---")
test_results = {
    "Market":               log_loss(y_test, np.clip(market_test, 0.001, 0.999)),
    "Mega-ensemble (avg)":  log_loss(y_test, mega_test),
    "Weighted (V1*20+V2*80)": log_loss(y_test, weighted_test),
    "Stacking: LogReg":     log_loss(y_test, lr_pred_test),
    "Stacking: XGBoost":    log_loss(y_test, xgb_pred_test),
}
for name, ll in sorted(test_results.items(), key=lambda x: x[1]):
    print(f"  {name:30s}: {ll:.6f}", flush=True)

# Also compute on full 2025 OOS for reference (train+val+test combined)
print("\n--- FULL 2025 OOS (all months, for reference) ---")
y_all = aligned["win"].values
for name, probs in [
    ("Market", np.clip(aligned["market_prob"].values, 0.001, 0.999)),
    ("Mega-ensemble", aligned[prob_cols].mean(axis=1).values),
]:
    print(f"  {name:30s}: {log_loss(y_all, probs):.6f}", flush=True)


# ============================================================
# STEP 9: Backtest on Test Set (Sep-Oct 2025)
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: BACKTEST COMPARISON ON TEST SET (Sep-Oct 2025)")
print("=" * 70)

edge_thresholds = [0.02, 0.025, 0.03, 0.05]

# Mega-ensemble backtest
bt_text, _ = backtest_strategy(df_test, mega_test, edge_thresholds, label="Mega-Ensemble (avg all 8)")
print(bt_text, flush=True)

# Weighted blend backtest
bt_text, _ = backtest_strategy(df_test, weighted_test, edge_thresholds, label="Weighted V1*20%+V2*80%")
print(bt_text, flush=True)

# LR stacking backtest
bt_text, _ = backtest_strategy(df_test, lr_pred_test, edge_thresholds, label="Stacking: LogReg")
print(bt_text, flush=True)

# XGB stacking backtest
bt_text, _ = backtest_strategy(df_test, xgb_pred_test, edge_thresholds, label="Stacking: XGBoost")
print(bt_text, flush=True)


# ============================================================
# STEP 10: Also backtest on FULL 2025 using train-based models
# ============================================================
# For a fairer full-2025 comparison: use LR/XGB trained on Jan-Jul only
# and apply to Aug-Oct, but show mega/weighted on all 2025
print("\n" + "=" * 70)
print("STEP 10: FULL 2025 BACKTEST (Mega & Weighted on all; Stacking on Aug-Oct)")
print("=" * 70)

# Full 2025 mega and weighted
full_mega = aligned[prob_cols].mean(axis=1).values
bt_text, _ = backtest_strategy(aligned, full_mega, edge_thresholds, label="Mega-Ensemble (all 2025)")
print(bt_text, flush=True)

if len(v1_cols) > 0 and len(v2_cols) > 0:
    v1_cross_all = aligned[v1_cols].mean(axis=1).values
    v2_cross_all = aligned[v2_cols].mean(axis=1).values
    full_weighted = 0.20 * v1_cross_all + 0.80 * v2_cross_all
    bt_text, _ = backtest_strategy(aligned, full_weighted, edge_thresholds, label="Weighted V1*20%+V2*80% (all 2025)")
    print(bt_text, flush=True)

# Stacking on val+test (Aug-Oct 2025 — unseen by meta-models)
df_oos_meta = pd.concat([df_val, df_test])
X_oos_meta = pd.concat([X_val, X_test])
X_oos_meta_scaled = scaler.transform(X_oos_meta)

lr_oos_pred = lr_model.predict_proba(X_oos_meta_scaled)[:, 1]
xgb_oos_pred = xgb_meta.predict_proba(X_oos_meta)[:, 1]

bt_text, _ = backtest_strategy(df_oos_meta, lr_oos_pred, edge_thresholds, label="Stacking LR (Aug-Oct 2025)")
print(bt_text, flush=True)

bt_text, _ = backtest_strategy(df_oos_meta, xgb_oos_pred, edge_thresholds, label="Stacking XGB (Aug-Oct 2025)")
print(bt_text, flush=True)


# ============================================================
# STEP 11: Save stacking predictions
# ============================================================
print("\nStep 11: Saving results...", flush=True)

save_dir = "/data/projects/punim2039/alpha_odds/res/analysis/"
os.makedirs(save_dir, exist_ok=True)

# Save test set predictions for all methods
df_save = df_test[["key", "win", "market_prob", "orig_best_back_m0", "marketTime_local"]].copy()
df_save["mega_ensemble"] = mega_test
df_save["weighted_blend"] = weighted_test
df_save["stacking_lr"] = lr_pred_test
df_save["stacking_xgb"] = xgb_pred_test
df_save.to_parquet(save_dir + "stacking_experiment_predictions.parquet")
print(f"  Saved predictions to {save_dir}stacking_experiment_predictions.parquet", flush=True)

# Save LR coefficients
lr_coefs = pd.DataFrame({
    "feature": meta_feature_names,
    "coefficient": lr_model.coef_[0]
})
lr_coefs.loc[len(lr_coefs)] = ["intercept", lr_model.intercept_[0]]
lr_coefs.to_parquet(save_dir + "stacking_lr_coefficients.parquet")
print(f"  Saved LR coefficients", flush=True)

# Save XGB importances
xgb_imp_df = pd.DataFrame({
    "feature": meta_feature_names,
    "importance": xgb_meta.feature_importances_
}).sort_values("importance", ascending=False)
xgb_imp_df.to_parquet(save_dir + "stacking_xgb_importances.parquet")
print(f"  Saved XGB importances", flush=True)


# ============================================================
# STEP 12: Build results summary and send email
# ============================================================
print("\nStep 12: Sending email notification...", flush=True)

results_summary = []
results_summary.append("STACKING META-MODEL EXPERIMENT RESULTS")
results_summary.append("=" * 50)
results_summary.append("")
results_summary.append(f"Components loaded: {len(all_components)}")
results_summary.append(f"Aligned rows: {len(aligned):,}")
results_summary.append(f"Meta-train (Jan-Jul): {ind_train.sum():,}")
results_summary.append(f"Meta-val   (Aug):     {ind_val.sum():,}")
results_summary.append(f"Meta-test  (Sep-Oct): {ind_test.sum():,}")
results_summary.append("")
results_summary.append("LOG-LOSS ON TEST SET (Sep-Oct 2025):")
for name, ll in sorted(test_results.items(), key=lambda x: x[1]):
    results_summary.append(f"  {name:30s}: {ll:.6f}")
results_summary.append("")
results_summary.append("LOG-LOSS ON VALIDATION SET (Aug 2025):")
for name, ll in sorted(val_results.items(), key=lambda x: x[1]):
    results_summary.append(f"  {name:30s}: {ll:.6f}")
results_summary.append("")

# Best test LL
best_method = min(test_results, key=test_results.get)
best_ll = test_results[best_method]
baseline_ll = test_results.get("Weighted (V1*20+V2*80)", test_results.get("Mega-ensemble (avg)"))
improvement = (baseline_ll - best_ll) / baseline_ll * 100 if baseline_ll else 0

results_summary.append(f"BEST METHOD: {best_method} (LL={best_ll:.6f})")
results_summary.append(f"vs Weighted blend: {improvement:+.2f}% improvement")
results_summary.append("")

# LR coefficients
results_summary.append("LOGISTIC REGRESSION COEFFICIENTS:")
for name, coef in sorted(zip(meta_feature_names, lr_model.coef_[0]),
                          key=lambda x: abs(x[1]), reverse=True):
    results_summary.append(f"  {name:30s}: {coef:+.4f}")
results_summary.append(f"  {'intercept':30s}: {lr_model.intercept_[0]:+.4f}")

results_summary = "\n".join(results_summary)

import smtplib
from email.mime.text import MIMEText
try:
    msg = MIMEText(results_summary)
    msg['Subject'] = 'Alpha Odds: Stacking experiment complete'
    msg['From'] = 'adidishe@spartan.hpc.unimelb.edu.au'
    msg['To'] = 'antoine.didisheim@unimelb.edu.au'
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()
    print("  Email sent successfully!", flush=True)
except Exception as e:
    print(f"  Email failed: {e}", flush=True)

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
print(results_summary)
