"""
Export surface model artifacts for paper trading.

Identifies top-7 V1_surface and top-15 V2_surface configs per t_def by log-loss,
copies model files to an export directory, and generates a manifest fragment.

Usage (on Spartan):
    python _12_export_surface_artifacts.py

Output:
    /home/adidishe/alpha_odds/surface_export/v1_surface/t{0-3}/{config}/xgboost_model.json
    /home/adidishe/alpha_odds/surface_export/v2_surface/t{0-3}/{config}/xgboost_model.json + lgb + iso
    /home/adidishe/alpha_odds/surface_export/manifest_surface.json
"""
import os
import json
import shutil
import numpy as np
import pandas as pd

RES = "/data/projects/punim2039/alpha_odds/res/"
EXPORT = "/home/adidishe/alpha_odds/surface_export/"


def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def rank_configs(model_dir, t_def, top_n):
    """Load save_df.parquet for each config, rank by log-loss, return top-N config names."""
    base = os.path.join(model_dir, f"t{t_def}")
    if not os.path.exists(base):
        return []
    configs = {}
    for c in sorted(os.listdir(base)):
        path = os.path.join(base, c, "save_df.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                ll = log_loss(df["win"].values, np.clip(df["model_prob"].values, 0.001, 0.999))
                configs[c] = ll
            except Exception as e:
                print(f"  Skip {c}: {e}")
    ranked = sorted(configs, key=configs.get)
    top = ranked[:top_n]
    print(f"  t{t_def}: {len(configs)} configs, top-{top_n} LL range: "
          f"{configs[top[0]]:.6f} - {configs[top[-1]]:.6f}")
    return top


def export_models(src_dir, dst_subdir, t_def, config_names):
    """Copy model artifacts from src to export directory."""
    for config_name in config_names:
        src = os.path.join(src_dir, f"t{t_def}", config_name)
        dst = os.path.join(EXPORT, dst_subdir, f"t{t_def}", config_name)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if f.endswith((".json", ".txt", ".pkl")):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))


manifest = {"v1_surface": {}, "v2_surface": {}}
os.makedirs(EXPORT, exist_ok=True)

print("=== V1 Surface (top 7 per t_def) ===")
for t in range(4):
    top = rank_configs(RES + "win_model_surface", t, 7)
    manifest["v1_surface"][f"t{t}"] = top
    export_models(RES + "win_model_surface", "v1_surface", t, top)

print("\n=== V2 Surface (top 15 per t_def) ===")
for t in range(4):
    top = rank_configs(RES + "win_model_v2_surface", t, 15)
    manifest["v2_surface"][f"t{t}"] = top
    export_models(RES + "win_model_v2_surface", "v2_surface", t, top)

# Also export normalization params
norm_src = os.path.join(RES, "win_model_v2_surface")
for t in range(4):
    # Get normalization from the top V2_surface config
    top_config = manifest["v2_surface"][f"t{t}"][0]
    norm_file = os.path.join(norm_src, f"t{t}", top_config, "feature_normalization_params.parquet")
    if os.path.exists(norm_file):
        dst = os.path.join(EXPORT, "normalization_surface", f"t{t}")
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(norm_file, os.path.join(dst, "feature_normalization_params.parquet"))
        print(f"  Copied normalization for t{t}")

manifest_path = os.path.join(EXPORT, "manifest_surface.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest saved to {manifest_path}")
print(json.dumps(manifest, indent=2))
print("\nDone! Export directory:", EXPORT)
