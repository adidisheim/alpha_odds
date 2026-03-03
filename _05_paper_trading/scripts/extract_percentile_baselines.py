"""
Extract percentile baselines from backtest ensemble predictions.

Computes empirical distributions for signal-level metrics (bet, race, daily)
and optionally feature-level metrics (if feature parquets are available).

Signal baselines → baselines/percentile_baselines.json
Feature baselines → baselines/feature_bet_percentiles.parquet
                    baselines/feature_daily_mean_percentiles.parquet
                    baselines/feature_daily_std_percentiles.parquet

Run locally (signals only):
    python _05_paper_trading/scripts/extract_percentile_baselines.py

Run on Spartan (signals + features):
    srun --partition=interactive --time=00:15:00 --mem=32G bash -c \
      'source load_module.sh && python3 extract_percentile_baselines.py'
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import socket


# ── Paths ──
def _is_spartan():
    return "spartan" in socket.gethostname().lower() or Path("/data/projects/punim2039").exists()


def _predictions_path():
    if _is_spartan():
        return Path("/data/projects/punim2039/alpha_odds/res/analysis/ultimate_cross_t_ensemble_predictions.parquet")
    return Path("res/analysis/ultimate_cross_t_ensemble_predictions.parquet")


def _features_path(t_def):
    if _is_spartan():
        return Path(f"/data/projects/punim2039/alpha_odds/res/features_t{t_def}/greyhound_au_features_merged.parquet")
    return Path(f"res/features_t{t_def}/greyhound_au_features_merged.parquet")


def _output_dir():
    # On Spartan scripts run from flat dir; locally from project root
    for d in [Path("_05_paper_trading/baselines"), Path("baselines")]:
        if d.parent.exists():
            d.mkdir(exist_ok=True)
            return d
    Path("baselines").mkdir(exist_ok=True)
    return Path("baselines")


# ── Percentile computation ──
PERCENTILES = [0.5, 2.5, 97.5, 99.5]


def pct_stats(values, name=""):
    """Compute percentile statistics for a series."""
    values = pd.Series(values).dropna()
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "p0.5": None, "p2.5": None, "p97.5": None, "p99.5": None}
    return {
        "n": int(n),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p0.5": float(np.percentile(values, 0.5)),
        "p2.5": float(np.percentile(values, 2.5)),
        "p97.5": float(np.percentile(values, 97.5)),
        "p99.5": float(np.percentile(values, 99.5)),
    }


# ── Signal baselines ──
def extract_signal_baselines(df, edge_threshold=0.03, commission=0.075, stake=25.0):
    """Extract bet/race/daily percentile baselines from predictions."""
    # Apply same filters as paper trader
    bets = df[
        (df["edge"] > edge_threshold)
        & (df["back_odds"] >= 1.01)
        & (df["back_odds"] <= 50.0)
        & (df["market_prob"] >= 0.02)
    ].copy()

    bets["file_name"] = bets["key"].str.rsplit("_", n=1).str[0]
    bets["date"] = bets["marketTime_local"].dt.date

    df["file_name"] = df["key"].str.rsplit("_", n=1).str[0]
    df["date"] = df["marketTime_local"].dt.date

    baselines = {}

    # ── Bet-level ──
    baselines["bet_level"] = {
        "edge": pct_stats(bets["edge"]),
        "back_odds": pct_stats(bets["back_odds"]),
        "model_prob": pct_stats(bets["model_prob"]),
        "market_prob": pct_stats(bets["market_prob"]),
    }

    # ── Race-level ──
    rg = bets.groupby("file_name")
    race = pd.DataFrame({
        "n_bets": rg.size(),
        "sum_model_prob": rg["model_prob"].sum(),
        "mean_edge": rg["edge"].mean(),
        "mean_back_odds": rg["back_odds"].mean(),
        "mean_model_prob": rg["model_prob"].mean(),
        "mean_market_prob": rg["market_prob"].mean(),
    })
    baselines["race_level"] = {c: pct_stats(race[c]) for c in race.columns}

    # ── Daily-level ──
    bets["pnl"] = np.where(
        bets["win"] == 1,
        (bets["back_odds"] - 1) * (1 - commission) * stake,
        -stake,
    )
    dg = bets.groupby("date")
    daily = pd.DataFrame({
        "n_bets": dg.size(),
        "win_rate": dg["win"].mean(),
        "n_races_with_bets": dg["file_name"].nunique(),
        "total_pnl": dg["pnl"].sum(),
        "mean_edge": dg["edge"].mean(),
        "std_edge": dg["edge"].std(),
        "mean_back_odds": dg["back_odds"].mean(),
        "mean_model_prob": dg["model_prob"].mean(),
        "mean_market_prob": dg["market_prob"].mean(),
    })
    daily["roi_pct"] = (daily["total_pnl"] / (daily["n_bets"] * stake)) * 100
    daily["pnl_per_bet"] = daily["total_pnl"] / daily["n_bets"]

    # Fraction of all races that produce a bet
    total_races_per_day = df.groupby("date")["file_name"].nunique()
    daily["total_races"] = total_races_per_day
    daily["frac_races_with_bets"] = daily["n_races_with_bets"] / daily["total_races"]

    baselines["daily_level"] = {c: pct_stats(daily[c]) for c in daily.columns}

    # ── Metadata ──
    baselines["metadata"] = {
        "edge_threshold": edge_threshold,
        "commission": commission,
        "stake": stake,
        "odds_filter": [1.01, 50.0],
        "min_market_prob": 0.02,
        "n_total_runners": int(len(df)),
        "n_qualifying_bets": int(len(bets)),
        "n_races_with_bets": int(bets["file_name"].nunique()),
        "n_days": int(bets["date"].nunique()),
        "date_range": [str(bets["date"].min()), str(bets["date"].max())],
    }

    return baselines


# ── Feature baselines (requires feature parquets on Spartan) ──
def extract_feature_baselines(predictions_df, edge_threshold=0.03):
    """Extract per-feature percentile baselines for bet runners across all t_defs."""
    bets = predictions_df[
        (predictions_df["edge"] > edge_threshold)
        & (predictions_df["back_odds"] >= 1.01)
        & (predictions_df["back_odds"] <= 50.0)
        & (predictions_df["market_prob"] >= 0.02)
    ].copy()
    bet_keys = set(bets["key"].values)
    bets["date"] = bets["marketTime_local"].dt.date
    key_to_date = dict(zip(bets["key"], bets["date"]))

    all_bet_rows = []
    all_daily_mean_rows = []
    all_daily_std_rows = []

    for t_def in range(4):
        fpath = _features_path(t_def)
        if not fpath.exists():
            print(f"  [skip] Feature file not found: {fpath}")
            continue

        print(f"  Loading features t_def={t_def} from {fpath}")
        feat_df = pd.read_parquet(fpath)

        # Build key to match predictions
        feat_df["key"] = feat_df["file_name"].astype(str) + "_" + feat_df["id"].astype(str)

        # Filter to bet runners
        mask = feat_df["key"].isin(bet_keys)
        bet_feat = feat_df[mask].copy()
        print(f"    Matched {len(bet_feat)} / {len(bet_keys)} bet runners")

        if len(bet_feat) == 0:
            continue

        bet_feat["date"] = bet_feat["key"].map(key_to_date)

        # Identify feature columns (numeric, not metadata)
        meta_cols = {"file_name", "id", "key", "win", "date", "marketTime_local",
                     "time", "time_delta", "position", "prob_rank"}
        feature_cols = [c for c in bet_feat.columns
                        if bet_feat[c].dtype in ("float64", "float32", "int64")
                        and c not in meta_cols]

        # Bet-level percentiles per feature
        for col in feature_cols:
            s = pct_stats(bet_feat[col])
            s["feature"] = col
            s["t_def"] = t_def
            all_bet_rows.append(s)

        # Daily mean percentiles per feature
        daily_means = bet_feat.groupby("date")[feature_cols].mean()
        for col in feature_cols:
            s = pct_stats(daily_means[col])
            s["feature"] = col
            s["t_def"] = t_def
            all_daily_mean_rows.append(s)

        # Daily std percentiles per feature
        daily_stds = bet_feat.groupby("date")[feature_cols].std()
        for col in feature_cols:
            s = pct_stats(daily_stds[col])
            s["feature"] = col
            s["t_def"] = t_def
            all_daily_std_rows.append(s)

    result = {}
    if all_bet_rows:
        result["bet"] = pd.DataFrame(all_bet_rows)
    if all_daily_mean_rows:
        result["daily_mean"] = pd.DataFrame(all_daily_mean_rows)
    if all_daily_std_rows:
        result["daily_std"] = pd.DataFrame(all_daily_std_rows)

    return result


# ── Print summary ──
def print_summary(baselines):
    for level_name, level_data in baselines.items():
        if level_name == "metadata":
            print(f"\n{'='*80}")
            print("METADATA")
            for k, v in level_data.items():
                print(f"  {k}: {v}")
            continue
        print(f"\n{'='*80}")
        print(f"  {level_name.upper()}")
        print(f"{'='*80}")
        for metric, s in level_data.items():
            if s.get("n", 0) == 0:
                continue
            print(f"  {metric:35s}  n={s['n']:6d}  mean={s['mean']:9.4f}  "
                  f"std={s['std']:9.4f}  [{s['min']:9.4f}, {s['max']:9.4f}]  "
                  f"p0.5={s['p0.5']:9.4f}  p2.5={s['p2.5']:9.4f}  "
                  f"p97.5={s['p97.5']:9.4f}  p99.5={s['p99.5']:9.4f}")


# ── Main ──
def main():
    pred_path = _predictions_path()
    print(f"Loading predictions from {pred_path}")
    df = pd.read_parquet(pred_path)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    # ── Signal baselines ──
    print("\n── Extracting signal baselines ──")
    baselines = extract_signal_baselines(df)
    print_summary(baselines)

    out_dir = _output_dir()
    sig_path = out_dir / "percentile_baselines.json"
    with open(sig_path, "w") as f:
        json.dump(baselines, f, indent=2, default=str)
    print(f"\nSignal baselines saved to {sig_path}")

    # ── Feature baselines (Spartan only) ──
    print("\n── Extracting feature baselines ──")
    feat_baselines = extract_feature_baselines(df)

    if feat_baselines.get("bet") is not None:
        p = out_dir / "feature_bet_percentiles.parquet"
        feat_baselines["bet"].to_parquet(p, index=False)
        print(f"  Feature bet percentiles: {p}  ({len(feat_baselines['bet'])} rows)")

    if feat_baselines.get("daily_mean") is not None:
        p = out_dir / "feature_daily_mean_percentiles.parquet"
        feat_baselines["daily_mean"].to_parquet(p, index=False)
        print(f"  Feature daily means: {p}  ({len(feat_baselines['daily_mean'])} rows)")

    if feat_baselines.get("daily_std") is not None:
        p = out_dir / "feature_daily_std_percentiles.parquet"
        feat_baselines["daily_std"].to_parquet(p, index=False)
        print(f"  Feature daily stds: {p}  ({len(feat_baselines['daily_std'])} rows)")

    if not feat_baselines:
        print("  [skip] No feature files found — run on Spartan for feature baselines")

    print("\nDone.")


if __name__ == "__main__":
    main()
