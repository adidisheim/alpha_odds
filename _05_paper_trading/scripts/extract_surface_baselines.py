"""Extract backtest baselines for surface_85 and surf_n5 at edge > 3%."""
import pandas as pd
import numpy as np

RES = "/data/projects/punim2039/alpha_odds/res/analysis/"
COMMISSION = 0.075

for name, path in [("surface_85", "surface_85_predictions.parquet"), ("surf_n5", "surf_n5_predictions.parquet")]:
    df = pd.read_parquet(RES + path)
    df["edge"] = df["model_prob"] - df["market_prob"]
    df["back_odds"] = 1.0 / df["orig_best_back_m0"]
    bets = df[(df["edge"] > 0.03) & (df["back_odds"] > 1.01) & (df["back_odds"] < 1000)].copy()
    bets["date"] = pd.to_datetime(bets["marketTime_local"]).dt.date
    n_days = bets["date"].nunique()
    bets["pnl"] = bets["win"] * (bets["back_odds"] - 1) * (1 - COMMISSION) - (1 - bets["win"])
    e = bets["edge"]
    o = bets["back_odds"]
    mp = bets["model_prob"]
    mkp = bets["market_prob"]
    w = bets["win"]
    pnl = bets["pnl"]
    print(f"\n=== {name} BASELINES (edge>3%) ===")
    print(f"  n_bets:          {len(bets)}")
    print(f"  n_days:          {n_days}")
    print(f"  bets_per_day:    {len(bets)/n_days:.1f}")
    print(f"  avg_edge:        {e.mean():.4f}")
    print(f"  avg_odds:        {o.mean():.2f}")
    print(f"  avg_model_prob:  {mp.mean():.4f}")
    print(f"  avg_market_prob: {mkp.mean():.4f}")
    print(f"  win_rate:        {w.mean():.4f}")
    print(f"  roi_pct:         {pnl.mean()*100:.1f}")
