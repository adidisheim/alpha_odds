"""
Dense Ensemble Experiment: squeeze more signal from existing trained models.

Instead of using fixed top-N per version/t_def, systematically test:
  1. Different ensemble sizes per t_def (top-5, 10, 15, 20, all 27)
  2. Different weighting schemes (equal, inverse-LL, softmax-inverse-LL)
  3. Cross-t combinations (average per-t ensembles vs mega-pool of everything)
  4. V1 vs V2 vs mixed pools

Loads ALL 216 model save_df.parquet files (27 configs x 4 t_defs x 2 versions).
Runs entirely from saved OOS predictions -- no retraining needed.

Usage on Spartan:
    srun --partition=interactive --time=00:30:00 --mem=32G bash -c \
        'source load_module.sh && python3 _12_dense_ensemble_experiment.py'
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from math import erf, sqrt
from collections import defaultdict

# ============================================================
# Setup
# ============================================================
RES_DIR = "/data/projects/punim2039/alpha_odds/res"
os.chdir(RES_DIR)

start_time = time.time()
results_lines = []  # collect output for email


def log(msg):
    print(msg, flush=True)
    results_lines.append(msg)


def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


# ============================================================
# Phase 1: Load ALL models
# ============================================================
log("=" * 80)
log("DENSE ENSEMBLE EXPERIMENT")
log("=" * 80)
log(f"\nPhase 1: Loading all model predictions...")

# Structure: models[(version, t_def, config)] = {df, model_prob, ll}
models = {}
model_counts = defaultdict(int)

for version, model_dir in [("V1", "win_model"), ("V2", "win_model_v2")]:
    for t in [0, 1, 2, 3]:
        base_dir = f"{model_dir}/t{t}"
        if not os.path.exists(base_dir):
            log(f"  SKIP {version} t{t} (dir not found)")
            continue
        for config in sorted(os.listdir(base_dir)):
            path = os.path.join(base_dir, config, "save_df.parquet")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                probs = df["model_prob"].values
                ll = log_loss(df["win"].values, np.clip(probs, 0.001, 0.999))
                models[(version, t, config)] = {
                    "df": df,
                    "probs": probs,
                    "ll": ll,
                }
                model_counts[(version, t)] += 1
            except Exception as e:
                log(f"  ERROR loading {version} t{t} {config}: {e}")

log(f"\nLoaded {len(models)} models total.")
for (ver, t), cnt in sorted(model_counts.items()):
    log(f"  {ver} t{t}: {cnt} models")


# ============================================================
# Phase 2: Align all DataFrames by (file_name, id)
# ============================================================
log(f"\nPhase 2: Aligning all DataFrames...")

# Use V1 t0 first config as the reference frame
ref_key = ("V1", 0, sorted([k[2] for k in models if k[:2] == ("V1", 0)])[0])
ref_df = models[ref_key]["df"].copy()
ref_df["key"] = ref_df["file_name"].astype(str) + "_" + ref_df["id"].astype(str)

# Build aligned matrix: each column is one model's probabilities
aligned = ref_df[["key", "win", "market_prob", "orig_best_back_m0", "orig_best_lay_m0", "marketTime_local"]].copy()

# Track which models survive alignment
aligned_models = {}
n_skipped = 0

for (ver, t, config), info in models.items():
    col_name = f"{ver}_t{t}_{config}"
    tmp = info["df"].copy()
    tmp["key"] = tmp["file_name"].astype(str) + "_" + tmp["id"].astype(str)
    tmp[col_name] = info["probs"]

    merged = aligned[["key"]].merge(tmp[["key", col_name]], on="key", how="inner")
    if len(merged) < 0.9 * len(aligned):
        n_skipped += 1
        continue

    # Inner join to keep only common rows
    aligned = aligned.merge(tmp[["key", col_name]], on="key", how="inner")
    aligned_models[(ver, t, config)] = col_name

log(f"Aligned: {len(aligned):,} rows, {len(aligned_models)} models ({n_skipped} skipped due to <90% overlap)")

win = aligned["win"].values
market_prob = aligned["market_prob"].values
market_ll = log_loss(win, np.clip(market_prob, 0.001, 0.999))
log(f"Market LL: {market_ll:.6f}")


# ============================================================
# Phase 3: Individual model ranking
# ============================================================
log(f"\nPhase 3: Individual model log-loss ranking...")

model_lls = {}
for (ver, t, config), col_name in aligned_models.items():
    probs = aligned[col_name].values
    ll = log_loss(win, np.clip(probs, 0.001, 0.999))
    model_lls[(ver, t, config)] = ll

# Print top 10 and bottom 10
sorted_models = sorted(model_lls.items(), key=lambda x: x[1])
log("\nTop 10 individual models:")
for (ver, t, config), ll in sorted_models[:10]:
    log(f"  {ver} t{t} {config:30s} LL={ll:.6f}")
log("\nBottom 10 individual models:")
for (ver, t, config), ll in sorted_models[-10:]:
    log(f"  {ver} t{t} {config:30s} LL={ll:.6f}")


# ============================================================
# Helper: build ensemble from a set of model keys
# ============================================================
def build_ensemble_equal(keys):
    """Equal-weight average of model probabilities."""
    cols = [aligned_models[k] for k in keys if k in aligned_models]
    if not cols:
        return None
    return aligned[cols].mean(axis=1).values


def build_ensemble_inverse_ll(keys):
    """Inverse-LL weighted average (better models get more weight)."""
    valid = [(k, model_lls[k]) for k in keys if k in aligned_models]
    if not valid:
        return None
    # Weight = 1/LL, normalized
    weights = np.array([1.0 / ll for _, ll in valid])
    weights /= weights.sum()
    cols = [aligned_models[k] for k, _ in valid]
    probs_matrix = aligned[cols].values
    return (probs_matrix * weights[None, :]).sum(axis=1)


def build_ensemble_softmax_ll(keys, temperature=10.0):
    """Softmax-inverse-LL weighted average (sharper than inverse)."""
    valid = [(k, model_lls[k]) for k in keys if k in aligned_models]
    if not valid:
        return None
    # Higher temperature = sharper (more weight on best)
    neg_lls = np.array([-ll for _, ll in valid])
    weights = np.exp(temperature * (neg_lls - neg_lls.max()))
    weights /= weights.sum()
    cols = [aligned_models[k] for k, _ in valid]
    probs_matrix = aligned[cols].values
    return (probs_matrix * weights[None, :]).sum(axis=1)


# ============================================================
# Phase 4: Per-t_def ensemble experiments
# ============================================================
log(f"\n{'='*80}")
log("Phase 4: Per-t_def ensemble experiments")
log(f"{'='*80}")

# Group models by (version, t_def)
groups = defaultdict(list)
for (ver, t, config) in aligned_models:
    groups[(ver, t)].append((ver, t, config))

# Sort each group by LL (best first)
for key in groups:
    groups[key] = sorted(groups[key], key=lambda k: model_lls[k])

# Results table
ensemble_results = []

for (ver, t) in sorted(groups.keys()):
    keys = groups[(ver, t)]
    n_available = len(keys)
    log(f"\n--- {ver} t{t} ({n_available} models) ---")

    for top_n in [3, 5, 7, 10, 15, 20, n_available]:
        if top_n > n_available:
            continue
        label_n = f"all({n_available})" if top_n == n_available else f"top-{top_n}"
        selected = keys[:top_n]

        # Equal weight
        ens_eq = build_ensemble_equal(selected)
        if ens_eq is not None:
            ll_eq = log_loss(win, ens_eq)
            ensemble_results.append({
                "version": ver, "t_def": t, "top_n": top_n,
                "weighting": "equal", "ll": ll_eq, "n_models": top_n
            })
            if top_n in [5, 7, 10, 15, n_available]:
                log(f"  {label_n:>10s} equal:      LL={ll_eq:.6f}")

        # Inverse-LL weight
        ens_inv = build_ensemble_inverse_ll(selected)
        if ens_inv is not None:
            ll_inv = log_loss(win, ens_inv)
            ensemble_results.append({
                "version": ver, "t_def": t, "top_n": top_n,
                "weighting": "inv_ll", "ll": ll_inv, "n_models": top_n
            })
            if top_n in [5, 7, 10, 15, n_available]:
                log(f"  {label_n:>10s} inv-LL:     LL={ll_inv:.6f}")

        # Softmax-LL weight
        ens_sm = build_ensemble_softmax_ll(selected, temperature=10.0)
        if ens_sm is not None:
            ll_sm = log_loss(win, ens_sm)
            ensemble_results.append({
                "version": ver, "t_def": t, "top_n": top_n,
                "weighting": "softmax", "ll": ll_sm, "n_models": top_n
            })
            if top_n in [5, 7, 10, 15, n_available]:
                log(f"  {label_n:>10s} softmax:    LL={ll_sm:.6f}")

# ============================================================
# Phase 5: Cross-t ensembles with different densities
# ============================================================
log(f"\n{'='*80}")
log("Phase 5: Cross-t ensembles (per-version)")
log(f"{'='*80}")

cross_t_results = []

for ver in ["V1", "V2"]:
    log(f"\n--- {ver} cross-t ---")
    for top_n_per_t in [3, 5, 7, 10, 15, 20, 27]:
        for weighting in ["equal", "inv_ll", "softmax"]:
            # Build per-t ensembles first, then average across t
            t_ensembles = {}
            for t in [0, 1, 2, 3]:
                if (ver, t) not in groups:
                    continue
                keys = groups[(ver, t)]
                n = min(top_n_per_t, len(keys))
                selected = keys[:n]
                if weighting == "equal":
                    ens = build_ensemble_equal(selected)
                elif weighting == "inv_ll":
                    ens = build_ensemble_inverse_ll(selected)
                elif weighting == "softmax":
                    ens = build_ensemble_softmax_ll(selected)
                else:
                    ens = None
                if ens is not None:
                    t_ensembles[t] = ens

            if len(t_ensembles) < 2:
                continue

            # Average across time definitions
            cross_t_probs = np.mean(list(t_ensembles.values()), axis=0)
            ll = log_loss(win, cross_t_probs)
            cross_t_results.append({
                "version": ver, "top_n_per_t": top_n_per_t,
                "weighting": weighting, "n_t_defs": len(t_ensembles),
                "ll": ll
            })
            label = f"all" if top_n_per_t >= 27 else f"top-{top_n_per_t}"
            if weighting == "equal" or (weighting == "inv_ll" and top_n_per_t in [7, 15, 27]):
                log(f"  {label:>6s}/t x {len(t_ensembles)}t  {weighting:>7s}: LL={ll:.6f}")


# ============================================================
# Phase 6: Cross-version cross-t blends
# ============================================================
log(f"\n{'='*80}")
log("Phase 6: V1+V2 cross-t super-ensembles")
log(f"{'='*80}")

super_results = []

# For each density, build V1 cross-t and V2 cross-t, then blend
for top_n_per_t in [5, 7, 10, 15, 27]:
    for weighting in ["equal", "inv_ll"]:
        # Build V1 and V2 cross-t
        vx_ensembles = {}
        for ver in ["V1", "V2"]:
            t_ensembles = {}
            for t in [0, 1, 2, 3]:
                if (ver, t) not in groups:
                    continue
                keys = groups[(ver, t)]
                n = min(top_n_per_t, len(keys))
                selected = keys[:n]
                if weighting == "equal":
                    ens = build_ensemble_equal(selected)
                elif weighting == "inv_ll":
                    ens = build_ensemble_inverse_ll(selected)
                else:
                    ens = None
                if ens is not None:
                    t_ensembles[t] = ens
            if t_ensembles:
                vx_ensembles[ver] = np.mean(list(t_ensembles.values()), axis=0)

        if len(vx_ensembles) != 2:
            continue

        # Optimize V1/V2 blend weight
        best_w2, best_ll = 0.5, 999
        for w2 in np.arange(0.0, 1.01, 0.05):
            blend = vx_ensembles["V1"] * (1 - w2) + vx_ensembles["V2"] * w2
            ll = log_loss(win, blend)
            if ll < best_ll:
                best_w2, best_ll = w2, ll

        label = f"all" if top_n_per_t >= 27 else f"top-{top_n_per_t}"
        log(f"  {label:>6s}/t {weighting:>7s}  best V2w={best_w2:.0%}: LL={best_ll:.6f}")
        super_results.append({
            "top_n_per_t": top_n_per_t, "weighting": weighting,
            "best_v2_weight": best_w2, "ll": best_ll
        })


# ============================================================
# Phase 7: Mega-pool -- throw ALL models into one big average
# ============================================================
log(f"\n{'='*80}")
log("Phase 7: Mega-pool experiments (ignore version/t boundaries)")
log(f"{'='*80}")

# All models ranked by LL globally
all_keys_sorted = sorted(aligned_models.keys(), key=lambda k: model_lls[k])

mega_results = []
for top_n in [10, 20, 30, 50, 75, 100, 150, len(all_keys_sorted)]:
    if top_n > len(all_keys_sorted):
        continue
    selected = all_keys_sorted[:top_n]

    ens_eq = build_ensemble_equal(selected)
    ens_inv = build_ensemble_inverse_ll(selected)
    ens_sm = build_ensemble_softmax_ll(selected, temperature=10.0)

    ll_eq = log_loss(win, ens_eq) if ens_eq is not None else None
    ll_inv = log_loss(win, ens_inv) if ens_inv is not None else None
    ll_sm = log_loss(win, ens_sm) if ens_sm is not None else None

    label = f"all({len(all_keys_sorted)})" if top_n == len(all_keys_sorted) else f"top-{top_n}"
    log(f"  {label:>12s}  equal={ll_eq:.6f}  inv-LL={ll_inv:.6f}  softmax={ll_sm:.6f}")

    for wt, ll in [("equal", ll_eq), ("inv_ll", ll_inv), ("softmax", ll_sm)]:
        if ll is not None:
            mega_results.append({"top_n": top_n, "weighting": wt, "ll": ll})

    # Composition: how many from each (ver, t)?
    comp = defaultdict(int)
    for (ver, t, config) in selected:
        comp[(ver, t)] += 1
    comp_str = ", ".join(f"{k[0]}_t{k[1]}:{v}" for k, v in sorted(comp.items()))
    log(f"    Composition: {comp_str}")


# ============================================================
# Phase 8: Compare to the original _10_ultimate_ensemble.py baseline
# ============================================================
log(f"\n{'='*80}")
log("Phase 8: Comparison with original ensemble baselines")
log(f"{'='*80}")

# Original V1: top-7 equal weight per t, then average across t
v1_orig_configs = [
    "ne1000_md6_lr0.01", "ne100_md6_lr0.1", "ne500_md3_lr0.05", "ne500_md6_lr0.01",
    "ne100_md6_lr0.05", "ne1000_md3_lr0.05", "ne500_md6_lr0.05"
]
# Original V2: top-15 by LL at t0
orig_v1_t0 = build_ensemble_equal([("V1", 0, c) for c in v1_orig_configs if ("V1", 0, c) in aligned_models])
if orig_v1_t0 is not None:
    log(f"  Original V1 top-7 (t0 only):  LL={log_loss(win, orig_v1_t0):.6f}")

orig_v2_t0_keys = sorted(
    [k for k in aligned_models if k[0] == "V2" and k[1] == 0],
    key=lambda k: model_lls[k]
)[:15]
orig_v2_t0 = build_ensemble_equal(orig_v2_t0_keys)
if orig_v2_t0 is not None:
    log(f"  Original V2 top-15 (t0 only): LL={log_loss(win, orig_v2_t0):.6f}")

# Original cross-t: V1 top-7, V2 top-15, per t, then blend 40/60
v1_cross_t_orig = []
v2_cross_t_orig = []
for t in [0, 1, 2, 3]:
    # V1: top 7 by LL for each t
    v1_t_keys = sorted(
        [k for k in aligned_models if k[0] == "V1" and k[1] == t],
        key=lambda k: model_lls[k]
    )[:7]
    ens = build_ensemble_equal(v1_t_keys)
    if ens is not None:
        v1_cross_t_orig.append(ens)

    # V2: top 15 by LL for each t
    v2_t_keys = sorted(
        [k for k in aligned_models if k[0] == "V2" and k[1] == t],
        key=lambda k: model_lls[k]
    )[:15]
    ens = build_ensemble_equal(v2_t_keys)
    if ens is not None:
        v2_cross_t_orig.append(ens)

if v1_cross_t_orig:
    v1_ct = np.mean(v1_cross_t_orig, axis=0)
    log(f"  Original V1 cross-t (top-7):   LL={log_loss(win, v1_ct):.6f}")
if v2_cross_t_orig:
    v2_ct = np.mean(v2_cross_t_orig, axis=0)
    log(f"  Original V2 cross-t (top-15):  LL={log_loss(win, v2_ct):.6f}")
if v1_cross_t_orig and v2_cross_t_orig:
    orig_super = v1_ct * 0.4 + v2_ct * 0.6
    log(f"  Original super (40/60):        LL={log_loss(win, orig_super):.6f}")

# Now find the best from all our experiments
log(f"\n--- Best from each phase ---")
# Best per-t
if ensemble_results:
    best_per_t = min(ensemble_results, key=lambda x: x["ll"])
    log(f"  Best per-t single: {best_per_t['version']} t{best_per_t['t_def']} "
        f"top-{best_per_t['top_n']} {best_per_t['weighting']}: LL={best_per_t['ll']:.6f}")

if cross_t_results:
    best_cross_t = min(cross_t_results, key=lambda x: x["ll"])
    log(f"  Best cross-t:      {best_cross_t['version']} top-{best_cross_t['top_n_per_t']}/t "
        f"{best_cross_t['weighting']}: LL={best_cross_t['ll']:.6f}")

if super_results:
    best_super = min(super_results, key=lambda x: x["ll"])
    log(f"  Best V1+V2 super:  top-{best_super['top_n_per_t']}/t "
        f"{best_super['weighting']} V2w={best_super['best_v2_weight']:.0%}: LL={best_super['ll']:.6f}")

if mega_results:
    best_mega = min(mega_results, key=lambda x: x["ll"])
    log(f"  Best mega-pool:    top-{best_mega['top_n']} "
        f"{best_mega['weighting']}: LL={best_mega['ll']:.6f}")


# ============================================================
# Phase 9: Backtest the BEST ensemble found
# ============================================================
log(f"\n{'='*80}")
log("Phase 9: Backtest comparison")
log(f"{'='*80}")

# Build the best ensemble for backtesting
# -- Option A: best super result
if super_results:
    best_sr = min(super_results, key=lambda x: x["ll"])
    top_n = best_sr["top_n_per_t"]
    wt = best_sr["weighting"]
    w2 = best_sr["best_v2_weight"]

    vx_e = {}
    for ver in ["V1", "V2"]:
        t_e = {}
        for t in [0, 1, 2, 3]:
            if (ver, t) not in groups:
                continue
            keys = groups[(ver, t)]
            n = min(top_n, len(keys))
            selected = keys[:n]
            if wt == "equal":
                ens = build_ensemble_equal(selected)
            else:
                ens = build_ensemble_inverse_ll(selected)
            if ens is not None:
                t_e[t] = ens
        if t_e:
            vx_e[ver] = np.mean(list(t_e.values()), axis=0)
    best_new_probs = vx_e["V1"] * (1 - w2) + vx_e["V2"] * w2
    best_new_label = f"Best-Super top-{top_n}/t {wt} V2w={w2:.0%}"
else:
    best_new_probs = None
    best_new_label = "N/A"

# -- Original baseline for comparison
if v1_cross_t_orig and v2_cross_t_orig:
    orig_probs = orig_super
    orig_label = "Original Super (V1-top7 x V2-top15, 40/60)"
else:
    orig_probs = None
    orig_label = "N/A"

commission = 0.075
ensembles_to_bt = []
if orig_probs is not None:
    ensembles_to_bt.append((orig_label, orig_probs))
if best_new_probs is not None:
    ensembles_to_bt.append((best_new_label, best_new_probs))

# Also add mega-pool best
if mega_results:
    best_m = min(mega_results, key=lambda x: x["ll"])
    top_n_m = best_m["top_n"]
    wt_m = best_m["weighting"]
    selected_m = all_keys_sorted[:top_n_m]
    if wt_m == "equal":
        mega_probs = build_ensemble_equal(selected_m)
    elif wt_m == "inv_ll":
        mega_probs = build_ensemble_inverse_ll(selected_m)
    else:
        mega_probs = build_ensemble_softmax_ll(selected_m, temperature=10.0)
    if mega_probs is not None:
        ensembles_to_bt.append((f"Mega-pool top-{top_n_m} {wt_m}", mega_probs))

for label, probs in ensembles_to_bt:
    log(f"\n--- {label} ---")
    ll = log_loss(win, probs)
    log(f"  LL: {ll:.6f}")

    bt = aligned.copy()
    bt["model_prob"] = probs
    bt["edge"] = bt["model_prob"] - bt["market_prob"]
    bt["back_odds"] = 1 / bt["orig_best_back_m0"]
    bt = bt[(bt["back_odds"] > 1.01) & (bt["back_odds"] < 1000)]

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

        log(f"  Edge>{et:.1%}: {n:,} bets, WR={wr:.1%}, ROI={roi:+.1f}%, "
            f"z={z:.2f}, p={p:.4f}, Sharpe={sh_a:.1f}, "
            f"ProfMonths={pm}/{len(monthly)}, $25/bet=${profit_25:,.0f}")


# ============================================================
# Phase 10: Diminishing returns analysis
# ============================================================
log(f"\n{'='*80}")
log("Phase 10: Diminishing returns -- LL vs ensemble size")
log(f"{'='*80}")
log("(Best cross-t super-ensemble as we add more models per t_def)")

for weighting in ["equal", "inv_ll"]:
    log(f"\n  Weighting: {weighting}")
    for top_n_per_t in [1, 2, 3, 5, 7, 10, 15, 20, 27]:
        vx_e = {}
        for ver in ["V1", "V2"]:
            t_e = {}
            for t in [0, 1, 2, 3]:
                if (ver, t) not in groups:
                    continue
                keys = groups[(ver, t)]
                n = min(top_n_per_t, len(keys))
                selected = keys[:n]
                if weighting == "equal":
                    ens = build_ensemble_equal(selected)
                else:
                    ens = build_ensemble_inverse_ll(selected)
                if ens is not None:
                    t_e[t] = ens
            if t_e:
                vx_e[ver] = np.mean(list(t_e.values()), axis=0)

        if len(vx_e) != 2:
            continue

        # Optimize blend
        best_w2, best_ll = 0.5, 999
        for w2 in np.arange(0.0, 1.01, 0.05):
            blend = vx_e["V1"] * (1 - w2) + vx_e["V2"] * w2
            ll = log_loss(win, blend)
            if ll < best_ll:
                best_w2, best_ll = w2, ll

        n_total = min(top_n_per_t, 27) * 4 * 2
        log(f"    {top_n_per_t:>2d}/t ({n_total:>3d} total): LL={best_ll:.6f}  V2w={best_w2:.0%}")


# ============================================================
# Summary
# ============================================================
elapsed = time.time() - start_time
log(f"\n{'='*80}")
log("SUMMARY")
log(f"{'='*80}")
log(f"Total models loaded: {len(models)}")
log(f"Aligned rows: {len(aligned):,}")
log(f"Market baseline LL: {market_ll:.6f}")
if ensemble_results:
    best_per_t = min(ensemble_results, key=lambda x: x["ll"])
    log(f"Best single per-t: {best_per_t['version']} t{best_per_t['t_def']} "
        f"top-{best_per_t['top_n']} {best_per_t['weighting']}: LL={best_per_t['ll']:.6f}")
if cross_t_results:
    best_cross_t = min(cross_t_results, key=lambda x: x["ll"])
    log(f"Best cross-t:      {best_cross_t['version']} top-{best_cross_t['top_n_per_t']}/t "
        f"{best_cross_t['weighting']}: LL={best_cross_t['ll']:.6f}")
if super_results:
    best_super = min(super_results, key=lambda x: x["ll"])
    log(f"Best V1+V2 super:  top-{best_super['top_n_per_t']}/t "
        f"{best_super['weighting']} V2w={best_super['best_v2_weight']:.0%}: LL={best_super['ll']:.6f}")
if mega_results:
    best_mega = min(mega_results, key=lambda x: x["ll"])
    log(f"Best mega-pool:    top-{best_mega['top_n']} "
        f"{best_mega['weighting']}: LL={best_mega['ll']:.6f}")
if v1_cross_t_orig and v2_cross_t_orig:
    log(f"Original baseline: LL={log_loss(win, orig_super):.6f}")
log(f"\nElapsed: {elapsed:.1f}s")


# ============================================================
# Save detailed results
# ============================================================
save_dir = "/data/projects/punim2039/alpha_odds/res/analysis/"
os.makedirs(save_dir, exist_ok=True)

if ensemble_results:
    pd.DataFrame(ensemble_results).to_parquet(save_dir + "dense_ensemble_per_t_results.parquet")
if cross_t_results:
    pd.DataFrame(cross_t_results).to_parquet(save_dir + "dense_ensemble_cross_t_results.parquet")
if super_results:
    pd.DataFrame(super_results).to_parquet(save_dir + "dense_ensemble_super_results.parquet")
if mega_results:
    pd.DataFrame(mega_results).to_parquet(save_dir + "dense_ensemble_mega_results.parquet")
log(f"\nDetailed results saved to {save_dir}")


# ============================================================
# Email results
# ============================================================
results_text = "\n".join(results_lines)

import smtplib
from email.mime.text import MIMEText
try:
    msg = MIMEText(results_text)
    msg['Subject'] = 'Alpha Odds: Dense ensemble experiment complete'
    msg['From'] = 'adidishe@spartan.hpc.unimelb.edu.au'
    msg['To'] = 'antoine.didisheim@unimelb.edu.au'
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()
    log("Email sent successfully.")
except Exception as e:
    log(f"Email failed: {e}")

log("\nDone!")
