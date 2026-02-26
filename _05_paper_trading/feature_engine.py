"""
Feature Engine — computes features from live tick data, mirroring the historical pipeline exactly.

Replicates:
  - _01_feature_engineering_para.py::compute_features()
  - feature_tools.py::add_time_snapshot()
  - _03_win_probability_model_v2.py::add_cross_runner_features()
  - FeatureNormalizer from _03_win_probability_model_v2.py
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config import TIME_DEFS

logger = logging.getLogger(__name__)


def add_time_snapshot(df, snap_time):
    """
    Forward-fill snapshot at a specific time_delta.

    Exact replica of utils_locals/feature_tools.py::add_time_snapshot().
    Inserts a row at snap_time per (file_name, id) with NaN values,
    then sorts by time_delta descending and forward-fills.
    """
    temp = df[["file_name", "id"]].drop_duplicates()
    for c in df.columns:
        if c not in ["file_name", "id", "time_delta"]:
            temp[c] = np.nan
    temp["time_delta"] = snap_time
    df = pd.concat([temp.astype(df.dtypes), df], ignore_index=True)
    df = df.sort_values(["file_name", "id", "time_delta"], ascending=[True, True, False])
    cols_to_fill = [c for c in df.columns if c not in ["file_name", "id", "time_delta"]]
    df[cols_to_fill] = df.groupby(["file_name", "id"])[cols_to_fill].ffill()
    return df, df.loc[df["time_delta"] == snap_time, :].drop_duplicates().copy()


def compute_features_for_t_def(df, t_definition, runner_positions, mdef_info):
    """
    Compute features for a single time definition.

    Mirrors _01_feature_engineering_para.py::compute_features() exactly.

    Args:
        df: DataFrame with columns: time, file_name, id, best_back, best_lay,
            best_back_cum_qty, best_lay_cum_qty, total_back_qty, total_lay_qty,
            best_back_q_100, best_lay_q_100, best_back_q_1000, best_lay_q_1000,
            prc, qty, order_type
        t_definition: int 0-3
        runner_positions: dict {runner_id: position}
        mdef_info: dict with keys: file_name, local_dow, marketBaseRate, numberOfActiveRunners

    Returns:
        DataFrame with one row per (file_name, id) with all features.
    """
    id_cols = ["file_name", "id"]

    # Add runner_position column
    df["runner_position"] = df["id"].map(runner_positions).astype(float)
    runner_position = df.groupby(id_cols)[["runner_position"]].mean()
    df = df.drop(columns=["runner_position"])

    # Compute derived columns
    df["tot_bl_imbalance"] = df["total_lay_qty"] - df["total_back_qty"]
    df["best_bl_imbalance"] = df["best_lay_cum_qty"] - df["best_back_cum_qty"]
    df["order_is_back"] = (df["order_type"] == "back").astype(float)

    to_keep_columns = [
        "best_lay", "best_back", "best_lay_cum_qty", "best_back_cum_qty",
        "total_lay_qty", "total_back_qty", "best_lay_q_100", "best_back_q_100",
        "best_lay_q_1000", "best_back_q_1000",
        "tot_bl_imbalance", "best_bl_imbalance",
    ]
    qty_columns = ["qty", "prc", "order_is_back"]

    # Transform odds to implied probabilities
    col_with_back_lay = [
        "best_lay", "best_back", "best_lay_q_100", "best_back_q_100",
        "best_lay_q_1000", "best_back_q_1000",
    ]
    for col in col_with_back_lay:
        df[col] = 1.0 / df[col]

    # Get time deltas from config
    td = TIME_DEFS[t_definition]
    tm3 = pd.Timedelta(seconds=td["tm3"])
    tm2 = pd.Timedelta(seconds=td["tm2"])
    tm1 = pd.Timedelta(seconds=td["tm1"])
    t0 = pd.Timedelta(seconds=td["t0"])
    tp1 = pd.Timedelta(seconds=td["tp1"])

    # Add time snapshots (same order as historical pipeline)
    df, dtp1 = add_time_snapshot(df, tp1)
    df, dt0 = add_time_snapshot(df, t0)
    df, dtm3 = add_time_snapshot(df, tm3)
    df, dtm2 = add_time_snapshot(df, tm2)
    df, dtm1 = add_time_snapshot(df, tm1)

    # Computing momentums
    p_input_1 = dtp1.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()
    m_input_0 = dt0.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()
    m_input_1 = dtm1.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()
    m_input_2 = dtm2.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()
    m_input_3 = dtm3.set_index(id_cols)[to_keep_columns].fillna(0.0).drop_duplicates()

    mom_3_1 = (m_input_1 - m_input_3).add_suffix("_mom_3_1").reset_index().groupby(id_cols).mean()
    mom_2_1 = (m_input_1 - m_input_2).add_suffix("_mom_2_1").reset_index().groupby(id_cols).mean()

    # Computing variance and trade stats for window [tm1, tm3]
    temp = df.loc[df["time_delta"].between(tm1, tm3), :].copy()
    std_3_1 = temp.groupby(id_cols)[to_keep_columns].std().reset_index().groupby(id_cols).mean().add_suffix("_std_3_1")
    temp_trades = temp.loc[temp["qty"] > 0, :]
    qty_count_3_1 = temp_trades.groupby(id_cols)["qty"].count().reset_index().groupby(id_cols).mean().add_suffix("_count_3_1")
    qty_mean_3_1 = temp_trades.groupby(id_cols)[qty_columns].mean().reset_index().groupby(id_cols).mean().add_suffix("_mean_3_1")
    qty_std_3_1 = temp_trades.groupby(id_cols)[qty_columns].std().reset_index().groupby(id_cols).mean().add_suffix("_std_3_1")
    order_type_3_1 = temp_trades.groupby(id_cols)[["order_is_back"]].mean().add_suffix("_order_is_back_3_1").reset_index().groupby(id_cols).mean()

    # Same for window [tm1, tm2]
    temp = df.loc[df["time_delta"].between(tm1, tm2), :].copy()
    std_2_1 = temp.groupby(id_cols)[to_keep_columns].std().reset_index().groupby(id_cols).mean().add_suffix("_std_2_1")
    temp_trades = temp.loc[temp["qty"] > 0, :]
    qty_count_2_1 = temp_trades.groupby(id_cols)["qty"].count().reset_index().groupby(id_cols).mean().add_suffix("_count_2_1")
    qty_mean_2_1 = temp_trades.groupby(id_cols)[qty_columns].mean().reset_index().groupby(id_cols).mean().add_suffix("_mean_2_1")
    qty_std_2_1 = temp_trades.groupby(id_cols)[qty_columns].std().reset_index().groupby(id_cols).mean().add_suffix("_std_2_1")
    order_type_2_1 = temp_trades.groupby(id_cols)[["order_is_back"]].mean().add_suffix("_order_is_back_2_1").reset_index().groupby(id_cols).mean()

    # Add suffixes for time snapshots
    p_input_1 = p_input_1.add_suffix("_p1")
    m_input_0 = m_input_0.add_suffix("_m0")
    m_input_1 = m_input_1.add_suffix("_m1")
    m_input_2 = m_input_2.add_suffix("_m2")
    m_input_3 = m_input_3.add_suffix("_m3")

    # Merge chain (same as historical pipeline)
    final_df = (
        m_input_0
        .merge(m_input_1, on=id_cols, how="outer")
        .merge(p_input_1, on=id_cols, how="outer")
        .merge(m_input_2, on=id_cols, how="outer")
        .merge(m_input_3, on=id_cols, how="outer")
        .merge(mom_3_1, on=id_cols, how="outer")
        .merge(mom_2_1, on=id_cols, how="outer")
        .merge(std_3_1, on=id_cols, how="outer")
        .merge(std_2_1, on=id_cols, how="outer")
        .merge(qty_count_3_1, on=id_cols, how="outer")
        .merge(qty_mean_3_1, on=id_cols, how="outer")
        .merge(qty_std_3_1, on=id_cols, how="outer")
        .merge(qty_count_2_1, on=id_cols, how="outer")
        .merge(qty_mean_2_1, on=id_cols, how="outer")
        .merge(qty_std_2_1, on=id_cols, how="outer")
        .merge(order_type_3_1, on=id_cols, how="outer")
        .merge(order_type_2_1, on=id_cols, how="outer")
        .merge(runner_position, on=id_cols, how="outer")
    )

    # Resolve any duplicates by averaging
    final_df = final_df.reset_index().groupby(id_cols).mean()
    final_df = final_df.reset_index()

    # Add fixed effects
    final_df["local_dow"] = mdef_info["local_dow"]
    final_df["marketBaseRate"] = mdef_info["marketBaseRate"]
    final_df["numberOfActiveRunners"] = mdef_info["numberOfActiveRunners"]

    return final_df


def add_fraction_features(df):
    """
    Add fraction and fraction-momentum features.

    Mirrors the fraction feature code from _02_win_probability_model.py / _03_win_probability_model_v2.py.
    """
    df["total_qty_m1"] = df[["total_back_qty_m1", "total_lay_qty_m1"]].sum(axis=1)
    df["total_qty_m3"] = df[["total_back_qty_m3", "total_lay_qty_m3"]].sum(axis=1)

    col_todo = [
        "total_qty_m1", "total_back_qty_m1", "total_lay_qty_m1",
        "total_qty_m3", "total_back_qty_m3", "total_lay_qty_m3",
    ]
    col_frac = []
    for col in col_todo:
        c = col + "_frac"
        race_total = df.groupby("file_name")[col].transform("sum")
        df[c] = df[col] / race_total.replace(0, np.nan)
        col_frac.append(c)

    col_frac_mom = []
    for col in [x for x in col_frac if x.endswith("_m1_frac")]:
        c = col.replace("_m1", "_mom_3_1")
        df[c] = df[col] - df[col.replace("_m1", "_m3")]
        col_frac_mom.append(c)

    return df, col_frac, col_frac_mom


def add_cross_runner_features(df):
    """
    Add cross-runner features that capture how each runner compares to others in the same race.

    Exact replica of _03_win_probability_model_v2.py::add_cross_runner_features().
    """
    g = df.groupby("file_name")

    # Market implied probability (mid of back/lay at m0, already in implied prob space)
    df["market_prob"] = df[["best_back_m0", "best_lay_m0"]].mean(axis=1)

    # Probability rank within race (1 = favorite = highest prob)
    df["prob_rank"] = g["market_prob"].rank(method="min", ascending=False)

    # Probability relative to the favorite
    df["prob_vs_favorite"] = df["market_prob"] / g["market_prob"].transform("max")

    # Probability share
    df["prob_share"] = df["market_prob"] / g["market_prob"].transform("sum")

    # Herfindahl index
    df["_prob_share_sq"] = df["prob_share"] ** 2
    df["race_herfindahl"] = df.groupby("file_name")["_prob_share_sq"].transform("sum")
    df.drop(columns=["_prob_share_sq"], inplace=True)

    # Number of close runners
    race_std = g["market_prob"].transform("std").fillna(0)
    df["n_close_runners"] = (race_std < 0.05).astype(int) * (g["market_prob"].transform("count") - 1)

    # Spread
    df["spread_m0"] = (df["best_back_m0"] - df["best_lay_m0"]).abs()
    df["spread_rank"] = g["spread_m0"].rank(method="min", ascending=True)

    # Volume rank
    df["total_qty_m0"] = df["total_back_qty_m0"] + df["total_lay_qty_m0"]
    df["volume_rank"] = g["total_qty_m0"].rank(method="min", ascending=False)

    # Momentum rank
    df["avg_mom_3_1"] = df[["best_back_mom_3_1", "best_lay_mom_3_1"]].mean(axis=1)
    df["momentum_rank"] = g["avg_mom_3_1"].rank(method="min", ascending=False)

    # Overround
    df["race_overround"] = g["market_prob"].transform("sum")

    # Is favorite
    df["is_favorite"] = (df["prob_rank"] == 1).astype(int)

    # Price deviation
    df["prob_deviation"] = df["market_prob"] - g["market_prob"].transform("mean")

    # Imbalance
    df["bl_imbalance_m0"] = df["best_bl_imbalance_m0"]
    df["bl_imbalance_rank"] = g["bl_imbalance_m0"].rank(method="min", ascending=False)

    cross_runner_cols = [
        "prob_rank", "prob_vs_favorite", "prob_share", "race_herfindahl",
        "n_close_runners", "spread_m0", "spread_rank", "total_qty_m0",
        "volume_rank", "avg_mom_3_1", "momentum_rank", "race_overround",
        "is_favorite", "prob_deviation", "bl_imbalance_rank",
    ]
    return df, cross_runner_cols


def get_v1_predictors(df_columns):
    """Get V1 predictor column list (same as _02_win_probability_model.py)."""
    suffix_available_at_t0 = [
        "_count_2_1", "_count_3_1", "_mean_2_1", "_mean_3_1",
        "_m0", "_mom_2_1", "_mom_3_1",
        "_order_is_back_2_1", "_order_is_back_3_1",
        "_std_2_1", "_std_3_1",
    ]
    predictors = [c for c in df_columns if c.endswith(tuple(suffix_available_at_t0))]
    return predictors


def get_v2_predictors(df_columns, col_frac, col_frac_mom, cross_runner_cols):
    """Get V2 predictor column list (same as _03_win_probability_model_v2.py)."""
    suffix_available_at_t0 = [
        "_count_2_1", "_count_3_1", "_mean_2_1", "_mean_3_1",
        "_m0", "_mom_2_1", "_mom_3_1",
        "_order_is_back_2_1", "_order_is_back_3_1",
        "_std_2_1", "_std_3_1",
    ]
    predictors = [c for c in df_columns if c.endswith(tuple(suffix_available_at_t0))]
    predictors = predictors + col_frac_mom + col_frac

    if "runner_position" in df_columns:
        predictors.append("runner_position")

    fixed_effect_columns = ["local_dow", "marketBaseRate", "numberOfActiveRunners"]
    predictors = predictors + fixed_effect_columns
    predictors = predictors + cross_runner_cols
    return predictors


class FeatureComputer:
    """
    Computes features for a market at decision time.

    Produces features for all 4 time definitions from the same tick data,
    ready for both V1 and V2 model inference.
    """

    def __init__(self, normalization_params):
        """
        Args:
            normalization_params: dict of t_def -> {
                'v1': FeatureNormalizerParams,  # from saved parquet
                'v2': FeatureNormalizerParams,
            }
        """
        self.norm_params = normalization_params

    def compute(self, market_cache):
        """
        Compute features for all runners in a market across all 4 time definitions.

        Args:
            market_cache: MarketCache instance with accumulated tick data.

        Returns:
            dict of t_def -> {
                'features_v1': DataFrame (one row per runner, normalized for V1),
                'features_v2': DataFrame (one row per runner, normalized for V2),
                'raw_features': DataFrame (one row per runner, unnormalized),
                'market_prob': Series (market implied prob per runner),
                'back_odds': Series (best back odds per runner),
            }
        """
        tick_df = market_cache.to_dataframe()
        if tick_df.empty:
            logger.warning(f"No tick data for market {market_cache.market_id}")
            return {}

        # Compute time_delta = scheduled_start - time (mirrors historical: max(time) - time)
        # In live, we use scheduled start time so time_delta = scheduled_start - tick_time
        scheduled_start = pd.Timestamp(market_cache.market_start_time, tz="UTC")
        tick_df["time_delta"] = scheduled_start - tick_df["time"]

        # Clamp negative time_deltas (ticks after scheduled start)
        tick_df["time_delta"] = tick_df["time_delta"].clip(lower=pd.Timedelta(0))

        runner_positions = market_cache.get_runner_position_map()
        mdef_info = {
            "file_name": market_cache.file_name,
            "local_dow": market_cache.local_dow,
            "marketBaseRate": market_cache.market_base_rate,
            "numberOfActiveRunners": market_cache.num_active_runners,
        }

        results = {}
        for t_def in range(4):
            try:
                # Need a fresh copy for each t_def (add_time_snapshot modifies df in place)
                df_copy = tick_df.copy()
                features = compute_features_for_t_def(
                    df_copy, t_def, runner_positions, mdef_info
                )

                if features.empty:
                    continue

                # Add fraction features
                features, col_frac, col_frac_mom = add_fraction_features(features)

                # Store raw features before normalization (for V1 and logging)
                raw_features = features.copy()

                # For V2: add cross-runner features
                features_v2 = features.copy()
                features_v2, cross_runner_cols = add_cross_runner_features(features_v2)

                # Get predictor lists
                v1_predictors = get_v1_predictors(features.columns)
                v1_predictors = v1_predictors + col_frac_mom + col_frac
                if "runner_position" in features.columns:
                    v1_predictors.append("runner_position")
                v1_predictors = v1_predictors + ["local_dow", "marketBaseRate", "numberOfActiveRunners"]

                v2_predictors = get_v2_predictors(
                    features_v2.columns, col_frac, col_frac_mom, cross_runner_cols
                )

                # Normalize using saved params
                if t_def in self.norm_params:
                    norm = self.norm_params[t_def]
                    if "v1" in norm:
                        features_v1_norm = norm["v1"].normalize(features.copy(), v1_predictors)
                    else:
                        features_v1_norm = features.copy()

                    if "v2" in norm:
                        features_v2_norm = norm["v2"].normalize(features_v2.copy(), v2_predictors)
                    else:
                        features_v2_norm = features_v2.copy()
                else:
                    features_v1_norm = features.copy()
                    features_v2_norm = features_v2.copy()

                # Market prob and back odds from raw (un-normalized) features
                market_prob = raw_features[["best_back_m0", "best_lay_m0"]].mean(axis=1)
                back_odds = 1.0 / raw_features["best_back_m0"]

                results[t_def] = {
                    "features_v1": features_v1_norm,
                    "features_v2": features_v2_norm,
                    "v1_predictors": v1_predictors,
                    "v2_predictors": v2_predictors,
                    "raw_features": raw_features,
                    "market_prob": market_prob,
                    "back_odds": back_odds,
                    "runner_ids": raw_features["id"].values,
                    "file_name": raw_features["file_name"].values,
                }

            except Exception as e:
                logger.error(f"Feature computation failed for t_def={t_def}: {e}", exc_info=True)

        return results


class SavedNormalizerParams:
    """
    Applies normalization using parameters saved from training.

    Loaded from feature_normalization_params.parquet which contains:
    feature, group, fill_median, z_mean, z_std, is_log1p, is_high_missing
    """

    def __init__(self, params_df):
        """
        Args:
            params_df: DataFrame from feature_normalization_params.parquet
        """
        self.params = params_df.set_index("feature")
        self.high_missing_cols = params_df.loc[
            params_df["is_high_missing"] == True, "feature"
        ].tolist()
        self.log1p_cols = params_df.loc[
            params_df["is_log1p"] == True, "feature"
        ].tolist()

    def normalize(self, df, predictor_cols):
        """
        Apply saved normalization parameters to a DataFrame.

        Mirrors FeatureNormalizer.normalize_oos() exactly.
        """
        df = df.copy()

        # Ensure all predictor cols exist
        for c in predictor_cols:
            if c not in df.columns:
                df[c] = np.nan

        # Handle high-missing columns
        for c in self.high_missing_cols:
            if c in df.columns:
                ind_col = f"{c}_missing"
                df[ind_col] = df[c].isna().astype("int8")
                df[c] = df[c].fillna(0)

        # Fill remaining NaNs with median
        for c in predictor_cols:
            if c not in df.columns:
                continue
            if df[c].isna().any():
                if c in self.params.index:
                    med = self.params.loc[c, "fill_median"]
                    if pd.isna(med):
                        med = 0.0
                else:
                    med = 0.0
                df[c] = df[c].fillna(med)

        # Apply log1p
        for c in self.log1p_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c].clip(lower=0))

        # Apply z-score
        for c in predictor_cols:
            if c not in self.params.index:
                continue
            row = self.params.loc[c]
            z_mean = row.get("z_mean", np.nan)
            z_std = row.get("z_std", np.nan)
            if pd.notna(z_mean) and pd.notna(z_std):
                if z_std == 0:
                    z_std = 1.0
                df[c] = (df[c] - z_mean) / z_std

        return df
