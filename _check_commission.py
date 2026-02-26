import pandas as pd
import numpy as np

save_df = pd.read_parquet("/data/projects/punim2039/alpha_odds/res/win_model/t0/ne1000_md6_lr0.01/save_df.parquet")
ens = pd.read_parquet("/data/projects/punim2039/alpha_odds/res/analysis/ultimate_cross_t_ensemble_predictions.parquet")

save_df["key"] = save_df["file_name"] + "_" + save_df["id"].astype(str)
merged = ens.merge(
    save_df[["key", "marketBaseRate"]],
    on="key", how="left"
)

bets = merged[merged["edge"] > 0.03].copy()
print("marketBaseRate stats for edge>3% bets:")
print(bets["marketBaseRate"].describe())
print()
print("Value counts (sample):")
print(bets["marketBaseRate"].value_counts().head(10))
print()
print("NaN count:", bets["marketBaseRate"].isna().sum())

# The issue: marketBaseRate might be stored as e.g. 5.0 meaning 5%, not 0.05
# Let's check
print("\nSample values:", bets["marketBaseRate"].head(20).tolist())
