import pandas as pd

df = pd.read_parquet("/data/projects/punim2039/alpha_odds/res/win_model/t0/ne1000_md6_lr0.01/save_df.parquet")
print("=== save_df ===")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print()
for c in ["win", "model_prob", "market_prob", "id", "file_name", "orig_best_back_m0", "orig_best_lay_m0",
           "orig_best_back_q_100_m0", "orig_best_back_q_1000_m0", "marketTime_local"]:
    if c in df.columns:
        print(f"  {c}: {df[c].head(3).tolist()}  dtype={df[c].dtype}")
    else:
        print(f"  {c}: NOT FOUND")

print()
ens = pd.read_parquet("/data/projects/punim2039/alpha_odds/res/analysis/ultimate_cross_t_ensemble_predictions.parquet")
print("=== ultimate ensemble ===")
print("Shape:", ens.shape)
print("Columns:", list(ens.columns))
print()
print("First 3 rows:")
print(ens.head(3).to_string())
print()
print("key examples:", ens["key"].head(3).tolist() if "key" in ens.columns else "NO KEY COL")
