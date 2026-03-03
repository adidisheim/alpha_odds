"""
Build manifest.json for multi-variant paper trading models.

Merges export task results into the existing manifest, adding v2_n5 and v2_n30 entries.

Run via srun after _07_export_variant_artifacts.py completes:
    srun --partition=interactive --time=00:05:00 --mem=4G \
        bash -c 'source load_module.sh && python3 _08_build_variant_manifest.py'
"""

import json
import os

from parameters import Constant


if __name__ == '__main__':
    save_base = f'{Constant.RES_DIR}/paper_trading_artifacts'

    # Load existing manifest
    manifest_path = os.path.join(save_base, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Loaded existing manifest with keys: {list(manifest.keys())}", flush=True)
    else:
        print("ERROR: No existing manifest.json found", flush=True)
        exit(1)

    # Collect export task results
    for task_id in range(8):
        info_path = os.path.join(save_base, f'export_task_{task_id}.json')
        if not os.path.exists(info_path):
            print(f"  WARNING: Missing export_task_{task_id}.json", flush=True)
            continue

        with open(info_path) as f:
            task_info = json.load(f)

        v2_key = task_info['v2_key']
        t_def = task_info['t_def']
        configs = task_info['configs']

        if v2_key not in manifest:
            manifest[v2_key] = {}

        manifest[v2_key][f't{t_def}'] = configs
        print(f"  {v2_key} t{t_def}: {len(configs)} configs", flush=True)

    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nUpdated manifest saved to {manifest_path}", flush=True)

    # Summary
    print("\n=== Manifest Summary ===", flush=True)
    for key in sorted(manifest.keys()):
        for t_key in sorted(manifest[key].keys()):
            print(f"  {key} {t_key}: {len(manifest[key][t_key])} configs", flush=True)

    # Verify artifact files exist
    print("\n=== Verifying Artifacts ===", flush=True)
    for v2_key in ['v2_n5', 'v2_n30']:
        if v2_key not in manifest:
            print(f"  {v2_key}: NOT IN MANIFEST", flush=True)
            continue
        for t_key, configs in manifest[v2_key].items():
            for config_name in configs:
                artifact_dir = os.path.join(save_base, v2_key, t_key, config_name)
                xgb = os.path.exists(os.path.join(artifact_dir, 'xgboost_model.json'))
                lgb_ok = os.path.exists(os.path.join(artifact_dir, 'lightgbm_model.txt'))
                iso = os.path.exists(os.path.join(artifact_dir, 'isotonic_calibrator.pkl'))
                if not (xgb and iso):
                    print(f"  MISSING: {v2_key}/{t_key}/{config_name} xgb={xgb} lgb={lgb_ok} iso={iso}", flush=True)

    print("\nDone!", flush=True)
