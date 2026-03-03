import pandas as pd
import numpy as np

d = pd.read_parquet("/data/projects/punim2039/alpha_odds/data/p/greyhound_au/win_2025_Jan_1.parquet").reset_index()
print(f"Shape: {d.shape}, races: {d['file_name'].nunique()}")

stats = d.groupby("file_name")["time"].agg(["min", "max", "count"]).reset_index()

mdef = pd.read_parquet("/data/projects/punim2039/alpha_odds/data/p/greyhound_au/mdef_2025_Jan_1.parquet")
mdef["marketTime"] = pd.to_datetime(mdef["marketTime"], utc=True)
mg = stats.merge(mdef[["file_name", "marketTime"]].drop_duplicates(), on="file_name")
# Ensure both are tz-aware
mg["max"] = pd.to_datetime(mg["max"], utc=True)
mg["offset_s"] = (mg["max"] - mg["marketTime"]).dt.total_seconds()

print(f"\nTicks per race: mean={stats['count'].mean():.0f}, median={stats['count'].median():.0f}")
print(f"\nOFFSET = max(time) - scheduledStart (seconds):")
print(mg["offset_s"].describe().to_string())

print("\nSample races:")
for _, r in mg.head(10).iterrows():
    print(f"  {r['file_name']}: offset={r['offset_s']:.0f}s, ticks={r['count']}")
