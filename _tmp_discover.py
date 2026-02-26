import os
import pandas as pd

res_dir = '/data/projects/punim2039/alpha_odds/res/'

print("=" * 80)
print("FILES matching oos/save_df/ultimate/super/ensemble:")
print("=" * 80)
for f in sorted(os.listdir(res_dir)):
    if any(x in f.lower() for x in ['oos', 'save_df', 'ultimate', 'super', 'ensemble', 'cross']):
        full = os.path.join(res_dir, f)
        sz = os.path.getsize(full) / 1e6
        print(f"  {f}  ({sz:.1f} MB)")

print("\n" + "=" * 80)
print("model_to_download contents:")
print("=" * 80)
dl_dir = os.path.join(res_dir, 'model_to_download')
if os.path.exists(dl_dir):
    for f in sorted(os.listdir(dl_dir)):
        full = os.path.join(dl_dir, f)
        sz = os.path.getsize(full) / 1e6
        print(f"  {f}  ({sz:.1f} MB)")

# Check for features directories
print("\n" + "=" * 80)
print("Feature directories:")
print("=" * 80)
for f in sorted(os.listdir(res_dir)):
    if 'feature' in f.lower():
        full = os.path.join(res_dir, f)
        if os.path.isdir(full):
            contents = os.listdir(full)
            print(f"  {f}/  ({len(contents)} files)")
        else:
            sz = os.path.getsize(full) / 1e6
            print(f"  {f}  ({sz:.1f} MB)")

# Try to load the OOS file and inspect columns
print("\n" + "=" * 80)
print("Inspecting OOS parquet columns:")
print("=" * 80)
for candidate in ['oos_df_t0-t3.parquet', 'oos_df.parquet', 'oos_df_t0.parquet']:
    full = os.path.join(res_dir, candidate)
    if os.path.exists(full):
        df = pd.read_parquet(full)
        print(f"\n{candidate}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 3 rows sample:")
        print(df.head(3).to_string())
        print(f"\nDtypes:\n{df.dtypes}")
        break

# Also check save_df files in model_to_download
print("\n" + "=" * 80)
print("Inspecting save_df files in model_to_download:")
print("=" * 80)
if os.path.exists(dl_dir):
    save_dfs = [f for f in os.listdir(dl_dir) if 'save_df' in f.lower() or '_df' in f.lower()]
    for f in save_dfs[:3]:
        full = os.path.join(dl_dir, f)
        df = pd.read_parquet(full)
        print(f"\n{f}: {df.shape[0]} rows x {df.shape[1]} cols")
        cols = list(df.columns)
        print(f"Columns ({len(cols)}): {cols[:30]}")
        if len(cols) > 30:
            print(f"  ... and {len(cols)-30} more")
        # Check for key columns
        for key in ['win', 'back', 'lay', 'prob', 'odds', 'pred', 'q_100', 'q_200', 'spread', 'volume']:
            matching = [c for c in cols if key in c.lower()]
            if matching:
                print(f"  Cols matching '{key}': {matching}")
