"""Quick sanity check: compare off_time (SUSPENDED) to scheduledStart."""
import pandas as pd
import os

base = '/data/projects/punim2039/alpha_odds/data/p/greyhound_au/'

inplay = pd.read_parquet(base + 'in_play_times_all.parquet')
print(f"Total markets in in_play_times_all: {len(inplay)}")
print(f"NaT: {inplay['in_play_time'].isna().sum()}")

mdef_files = sorted([f for f in os.listdir(base) if f.startswith('mdef_') and f.endswith('.parquet')])
print(f"Loading {len(mdef_files)} mdef files...")

mdefs = []
for f in mdef_files[:100]:
    try:
        m = pd.read_parquet(base + f)
        m['marketTime'] = pd.to_datetime(m['marketTime'], utc=True)
        mdefs.append(m[['file_name', 'marketTime']].drop_duplicates())
    except:
        pass
mdef_all = pd.concat(mdefs, ignore_index=True)

mg = inplay.merge(mdef_all, on='file_name')
mg['offset_s'] = (mg['in_play_time'] - mg['marketTime']).dt.total_seconds()

print(f"\nSample size: {len(mg)} markets")
print(f"\nFull distribution:")
print(mg['offset_s'].describe())

filt = mg[(mg['offset_s'] > 0) & (mg['offset_s'] < 300)]
print(f"\nFiltered (0s < offset < 300s): {len(filt)}/{len(mg)} ({len(filt)/len(mg)*100:.1f}%)")
print(filt['offset_s'].describe())

print("\nPercentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = mg['offset_s'].quantile(p/100)
    print(f"  P{p} = {val:.1f}s")
