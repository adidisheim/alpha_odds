"""Extract backtest baselines for full and surface models at edge > 3%."""
import pandas as pd
import numpy as np

RES = "/data/projects/punim2039/alpha_odds/res/analysis/"
COMMISSION = 0.075

def extract_baselines(path, label):
    df = pd.read_parquet(path)
    df["edge"] = df["model_prob"] - df["market_prob"]
    df["back_odds"] = 1.0 / df["orig_best_back_m0"]
    bets = df[(df["edge"] > 0.03) & (df["back_odds"] > 1.01) & (df["back_odds"] < 1000)].copy()
    bets["date"] = pd.to_datetime(bets["marketTime_local"]).dt.date
    n_days = bets["date"].nunique()
    bets["pnl"] = bets["win"] * (bets["back_odds"] - 1) * (1 - COMMISSION) - (1 - bets["win"])

    print(f"\n=== {label} BASELINES (edge > 3%) ===")
    print(f"  n_bets:          {len(bets)}")
    print(f"  n_days:          {n_days}")
    print(f"  bets_per_day:    {len(bets) / n_days:.1f}")
    e = bets["edge"]
    print(f"  avg_edge:        {e.mean():.4f}")
    o = bets["back_odds"]
    print(f"  avg_odds:        {o.mean():.2f}")
    mp = bets["model_prob"]
    print(f"  avg_model_prob:  {mp.mean():.4f}")
    mkp = bets["market_prob"]
    print(f"  avg_market_prob: {mkp.mean():.4f}")
    w = bets["win"]
    print(f"  win_rate:        {w.mean():.4f}")
    pnl = bets["pnl"]
    print(f"  roi_pct:         {pnl.mean() * 100:.1f}")
    print(f"  total_profit_25: ${pnl.sum() * 25:.0f}")

extract_baselines(RES + "full_105_predictions.parquet", "FULL (105 features)")
extract_baselines(RES + "pure_surf_predictions.parquet", "PURE SURFACE (no q_100/q_1000)")
